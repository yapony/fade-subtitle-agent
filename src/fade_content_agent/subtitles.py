"""Deterministic SRT/VTT parsing and cue batching."""
from __future__ import annotations
import re
from dataclasses import dataclass, replace
from pathlib import Path

@dataclass(frozen=True)
class Cue:
    id: str
    sequence: int
    start: float
    end: float
    text: str
    duration: float | None = None

    def __post_init__(self):
        if self.duration is None:
            object.__setattr__(self,'duration',self.end-self.start)

_TIME = re.compile(r"(?P<h>\d{2,}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")

def parse_time(value: str) -> float:
    m = _TIME.search(value)
    if not m: raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    return int(m['h'])*3600 + int(m['m'])*60 + int(m['s']) + int(m['ms'])/1000

def parse_subtitles(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding='utf-8-sig').replace('\r\n','\n').replace('\r','\n')
    blocks = re.split(r'\n\s*\n', raw.strip())
    cues=[]
    for block in blocks:
        lines=block.splitlines()
        if not lines or lines[0].strip().upper().startswith('WEBVTT') or lines[0].strip().startswith(('NOTE','STYLE','REGION')): continue
        ti=next((i for i,line in enumerate(lines) if '-->' in line),None)
        if ti is None: continue
        left,right=lines[ti].split('-->',1)
        start,end=parse_time(left),parse_time(right)
        if end < start: raise ValueError(f"Cue ends before it starts: {block[:60]!r}")
        text='\n'.join(x.strip() for x in lines[ti+1:] if x.strip())
        text=re.sub(r'<[^>]+>','',text)
        if text:
            seq=len(cues)+1; cid=lines[0].strip() if ti>0 and lines[0].strip().isdigit() else str(seq)
            cues.append(Cue(cid,seq,start,end,text))
    if not cues: raise ValueError('No subtitle cues found (expected SRT or VTT)')
    return cues

def cues_from_transcript(items: list[dict]) -> list[Cue]:
    """Normalize timestamped transcript items without changing source text or timing."""
    cues=[]
    for index,item in enumerate(items,1):
        try:
            start=float(item['start']); duration=float(item['duration']); text=item['text']
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError(f'Invalid transcript cue at position {index}: {exc}') from exc
        if not isinstance(text,str) or not text:
            raise ValueError(f'Empty transcript cue at position {index}')
        if start<0 or duration<0:
            raise ValueError(f'Invalid transcript timing at position {index}')
        cues.append(Cue(str(index),index,start,start+duration,text,duration))
    if not cues: raise ValueError('TranscriptAPI returned an empty transcript')
    return cues

def clip_cues_to_duration(cues: list[Cue], duration: float) -> tuple[list[Cue], list[dict]]:
    """Clip every cue that ends past the video duration."""
    if not cues:
        raise ValueError('Cannot normalize an empty subtitle cue list')
    adjusted_cues=[]
    adjustments=[]
    for cue in cues:
        if cue.end <= duration:
            adjusted_cues.append(cue)
            continue
        if cue.start >= duration:
            raise ValueError(f'Subtitle cue {cue.id} starts at or after video duration')
        adjusted_cues.append(replace(cue,end=duration,duration=duration-cue.start))
        adjustments.append({
            'cue_id': cue.id,
            'original_end': cue.end,
            'adjusted_end': duration,
        })
    return (adjusted_cues if adjustments else cues), adjustments

def batches(cues: list[Cue], size: int=50, overlap: int=3):
    if size<1 or overlap<0 or overlap>=size: raise ValueError('Require size > overlap >= 0')
    for start in range(0,len(cues),size-overlap):
        end=min(start+size,len(cues))
        yield cues[start:end], set(c.id for c in cues[start:min(start+size-overlap,len(cues))])
