"""HTTP entry point for asynchronous Fade Agent analysis jobs.

The API deliberately wraps the existing synchronous Agent pipeline. The web
client creates a job, polls its status, and fetches the unchanged schema 2.0
package after the worker finishes.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import Agent
from .transcript_api import is_youtube_input


OUTPUT_DIR = Path(os.getenv("FADE_OUTPUT_DIR", "fade-output"))
MAX_WORKERS = max(1, int(os.getenv("FADE_MAX_WORKERS", "1")))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()

STAGES = (
    (10, "Fetching subtitles"),
    (20, "Reconstructing complete sentences"),
    (40, "Translating sentences"),
    (70, "Analyzing language concepts"),
    (90, "Generating quiz questions"),
    (95, "Checking"),
)


class AnalysisRequest(BaseModel):
    youtube_url: str = Field(min_length=1, max_length=2048)
    source_language: str = Field(min_length=2, max_length=20)
    target_language: str = Field(min_length=2, max_length=20)


def _job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    status = job["status"]
    if status == "completed":
        progress, stage = 100, "Ready to learn"
    elif status == "failed":
        progress, stage = job.get("progress", 0), "Analysis failed"
    else:
        elapsed = max(0.0, time.monotonic() - job["started_at"])
        estimated = min(95, int(8 + elapsed / 240 * 87))
        progress = max(job.get("progress", 0), estimated)
        stage = next((label for threshold, label in reversed(STAGES) if progress >= threshold), STAGES[0][1])
    snapshot = {
        "analysis_id": job["id"],
        "status": status,
        "progress": progress,
        "stage": stage,
        "result_url": f"/api/agent/analyses/{job['id']}/result" if status == "completed" else None,
        "error": job.get("error"),
    }
    return snapshot


def _run_job(job_id: str, request: AnalysisRequest) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["started_at"] = time.monotonic()
    try:
        output = Agent(OUTPUT_DIR).run(
            request.youtube_url,
            request.source_language,
            request.target_language,
            language=request.source_language,
        )
        with jobs_lock:
            jobs[job_id].update(status="completed", output_path=str(output), progress=100)
    except Exception as error:  # the API must report a bounded, key-free error
        message = str(error).replace(os.getenv("GEMINI_API_KEY", ""), "[redacted]")[:500]
        with jobs_lock:
            jobs[job_id].update(status="failed", error=message, progress=0)


def _read_package(path: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        package: dict[str, Any] = {}
        for name in ("manifest", "subtitles", "vocabulary", "culture", "quizzes"):
            package[name] = json.loads(archive.read(f"{name}.json").decode("utf-8"))
        return package


app = FastAPI(title="Fade Agent API", version="1.0")
allowed_origins = [origin.strip() for origin in os.getenv("FADE_ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("FADE_AGENT_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Agent API authentication is not configured")
    scheme, separator, supplied = (authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyses", status_code=202, dependencies=[Depends(require_api_key)])
def create_analysis(request: AnalysisRequest) -> dict[str, str]:
    if not is_youtube_input(request.youtube_url.strip()):
        raise HTTPException(status_code=400, detail="youtube_url must be a valid YouTube URL or video id")
    if request.source_language.lower() == request.target_language.lower():
        raise HTTPException(status_code=400, detail="source_language and target_language must differ")
    job_id = secrets.token_hex(8)
    job = {"id": job_id, "status": "queued", "started_at": time.monotonic(), "progress": 0}
    with jobs_lock:
        jobs[job_id] = job
    executor.submit(_run_job, job_id, request)
    return {"analysis_id": job_id}


@app.get("/analyses/{analysis_id}", dependencies=[Depends(require_api_key)])
def get_analysis(analysis_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(analysis_id)
        if not job:
            raise HTTPException(status_code=404, detail="analysis not found")
        return _job_snapshot(job)


@app.get("/analyses/{analysis_id}/result", dependencies=[Depends(require_api_key)])
def get_result(analysis_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(analysis_id)
        if not job:
            raise HTTPException(status_code=404, detail="analysis not found")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="analysis is not complete")
        output_path = job.get("output_path")
    if not output_path or not Path(output_path).is_file():
        raise HTTPException(status_code=500, detail="analysis output is unavailable")
    return _read_package(output_path)
