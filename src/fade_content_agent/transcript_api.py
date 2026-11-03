"""TranscriptAPI client and response validation for YouTube captions."""
from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .subtitles import cues_from_transcript

TRANSCRIPT_ENDPOINT = 'https://transcriptapi.com/api/v2/youtube/transcript'


class TranscriptAPIError(RuntimeError):
    """A request or response error from TranscriptAPI."""


def is_youtube_input(value: str) -> bool:
    return value.startswith(('https://www.youtube.com/', 'http://www.youtube.com/',
                              'https://youtube.com/', 'http://youtube.com/',
                              'https://youtu.be/', 'http://youtu.be/',
                              'https://m.youtube.com/', 'http://m.youtube.com/')) or (
        len(value) == 11 and all(char.isalnum() or char in '_-' for char in value)
    )


def fetch_transcript(video_url: str, language: str | None = None, api_key: str | None = None) -> dict:
    key = api_key or os.getenv('TRANSCRIPT_API_KEY')
    if not key:
        raise TranscriptAPIError('TRANSCRIPT_API_KEY is required to fetch YouTube subtitles.')
    params = {'video_url': video_url, 'format': 'json', 'include_timestamp': 'true', 'send_metadata': 'true'}
    if language:
        params['language'] = language
    request = Request(
        f'{TRANSCRIPT_ENDPOINT}?{urlencode(params)}',
        headers={'Authorization': f'Bearer {key}', 'Accept': 'application/json'},
    )
    try:
        with urlopen(request, timeout=30) as response:
            raw_response = response.read()
            payload = json.loads(raw_response.decode('utf-8'))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode('utf-8')).get('detail', str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise TranscriptAPIError(f'TranscriptAPI request failed ({exc.code}): {detail}') from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TranscriptAPIError(f'TranscriptAPI request failed: {exc}') from exc

    if not isinstance(payload, dict):
        raise TranscriptAPIError('TranscriptAPI returned an invalid response object.')
    transcript = payload.get('transcript')
    if not isinstance(transcript, list) or not transcript:
        raise TranscriptAPIError('TranscriptAPI returned an empty or invalid transcript.')
    language_code = payload.get('language')
    if not isinstance(language_code, str) or not language_code.strip():
        raise TranscriptAPIError('TranscriptAPI did not return reliable subtitle language information.')
    if payload.get('is_translated') is True or payload.get('translated') is True:
        raise TranscriptAPIError('TranscriptAPI returned translated subtitles; original-language subtitles are required.')
    try:
        cues = cues_from_transcript(transcript)
    except ValueError as exc:
        raise TranscriptAPIError(str(exc)) from exc
    try:
        duration = float(payload['length_seconds'])
    except (KeyError, TypeError, ValueError) as exc:
        raise TranscriptAPIError('TranscriptAPI did not return reliable video duration metadata (length_seconds).') from exc
    if duration <= 0:
        raise TranscriptAPIError('TranscriptAPI returned invalid video duration metadata.')
    video_id = payload.get('video_id')
    if not isinstance(video_id, str) or not video_id:
        raise TranscriptAPIError('TranscriptAPI did not return a YouTube Video ID.')
    return {
        'cues': cues,
        'language': language_code.strip(),
        'video_id': video_id,
        'duration': duration,
        'title': (payload.get('metadata') or {}).get('title') if isinstance(payload.get('metadata'), dict) else None,
        'metadata': payload.get('metadata') or {},
        'source': 'transcriptapi',
        'requested_language': language,
        'raw_response': raw_response,
    }
