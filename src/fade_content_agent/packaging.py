"""Deterministic ZIP writer for the six-file learning package contract."""
from __future__ import annotations
import json
import zipfile
from pathlib import Path

def write_package_zip(output_dir: str|Path, video_id: str, run_id: str, package: dict, review_report: str) -> Path:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    path=out/f'{video_id}-{run_id}.zip'
    files={name:package[key] for key,name in (
        ('manifest','manifest.json'),('subtitles','subtitles.json'),('vocabulary','vocabulary.json'),
        ('culture','culture.json'),('quizzes','quizzes.json'))}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items(): archive.writestr(name,json.dumps(data,ensure_ascii=False)+'\n')
        archive.writestr('review-report.md',review_report)
    return path
