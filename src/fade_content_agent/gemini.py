"""Gemini JSON client with bounded retries and usage callbacks."""
from __future__ import annotations
import json, os, random, re, time
from pathlib import Path
from typing import Any
from .languages import language_code

class GeminiError(RuntimeError): pass

class Gemini:
    def __init__(self, model: str|None=None, api_key: str|None=None, event=None):
        self.model=model or os.getenv('GEMINI_MODEL','gemini-2.5-flash')
        key=api_key or os.getenv('GEMINI_API_KEY')
        if not key: raise GeminiError('GEMINI_API_KEY is required; configure it in the environment or .env file.')
        try:
            from google import genai
            from google.genai import types
        except ImportError as e: raise GeminiError('Install dependencies with: python -m pip install -e .') from e
        self.client=genai.Client(api_key=key); self.types=types; self.event=event
        self.min_interval=max(0.0,float(os.getenv('GEMINI_MIN_REQUEST_INTERVAL_SECONDS','4.5'))); self.last_request_started=0.0

    def generate(self, stage: str, prompt: str, schema: dict[str,Any], retries: int=3) -> dict:
        for attempt in range(1,retries+1):
            wait=self.min_interval-(time.monotonic()-self.last_request_started)
            if wait>0: time.sleep(wait)
            started=time.monotonic(); self.last_request_started=started
            try:
                response=self.client.models.generate_content(model=self.model, contents=prompt,
                    config=self.types.GenerateContentConfig(response_mime_type='application/json',response_schema=schema))
                data=json.loads(response.text or '{}')
                usage=getattr(response,'usage_metadata',None)
                if self.event: self.event(stage,'completed',model=self.model,attempt=attempt,duration_seconds=round(time.monotonic()-started,3),usage={
                    'input_tokens':getattr(usage,'prompt_token_count',None),'output_tokens':getattr(usage,'candidates_token_count',None),'total_tokens':getattr(usage,'total_token_count',None)})
                return data
            except Exception as e:
                category=type(e).__name__; message=str(e).replace(os.getenv('GEMINI_API_KEY','<none>'), '[REDACTED]')[:500]
                delay=min(2**(attempt-1)+random.random(),30)
                if '429' in message or 'RESOURCE_EXHAUSTED' in message or 'quota' in message.lower():
                    retry_hint=re.search(r'retry in ([0-9.]+)s',message,re.I)
                    delay=max(delay,float(retry_hint.group(1)) if retry_hint else 60.0)
                if self.event: self.event(stage,'retrying' if attempt<retries else 'failed',model=self.model,attempt=attempt,duration_seconds=round(time.monotonic()-started,3),error_category=category,error=message,retry_delay=round(delay,2) if attempt<retries else None)
                if attempt==retries: raise GeminiError(f'{stage} failed after {retries} attempts: {category}: {message}') from e
                time.sleep(delay)
        raise AssertionError('unreachable')

def load_prompt(name: str, source_language: str|None=None) -> str:
    prompt_dir=Path(__file__).parent/'prompts'
    prompt=(prompt_dir/f'{name}.md').read_text(encoding='utf-8')
    code=language_code(source_language)
    language_prompt=prompt_dir/f'{name}_{code}.md' if code else None
    if language_prompt and language_prompt.exists():
        prompt+='\n\n'+language_prompt.read_text(encoding='utf-8')
    return prompt
