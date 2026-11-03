"""Conservative cue-to-sentence translation alignment diagnostics."""
from __future__ import annotations
import hashlib
import re

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?",re.UNICODE)
_FUNCTION = set("a an the and or but to of in on at for from by with as is am are was were be been being it its this that these those do does did don't doesn't didn't he she they we you i me my your his their our not no yes el la los las un una y o de del en con por para es soy está estábamos pero que se lo te me nos mi tu su al".split())
ALIGNMENT_TYPES = {'direct', 'paraphrase', 'split'}
MIN_ALIGNMENT_CONFIDENCE = 0.7

def cue_translation_warnings(cues: list[dict], sentences: list[dict]) -> list[dict]:
    """Flag empty translations and cue translations whose content is absent from
    their complete-sentence translations. These are review warnings, not proofs.
    """
    sentence_by_id={sentence['id']:sentence for sentence in sentences}
    warnings=[]
    for cue in cues:
        translation=str(cue.get('translation','')).strip()
        if not translation:
            warnings.append({'reason':'empty_cue_translation','cue_id':cue['id']}); continue
        context=' '.join(sentence_by_id[sid].get('translation','') for sid in cue.get('sentence_ids',[]) if sid in sentence_by_id)
        if not context: continue
        translated_tokens=[token.casefold() for token in _TOKEN.findall(translation)]
        context_tokens={token.casefold() for token in _TOKEN.findall(context)}
        unmatched=[token for token in translated_tokens if token not in context_tokens and token not in _FUNCTION]
        no_context_overlap=bool(translated_tokens) and not any(token in context_tokens for token in translated_tokens)
        if unmatched or no_context_overlap:
            warnings.append({'reason':'cue_translation_context_mismatch','cue_id':cue['id'],
                             'unmatched_tokens':sorted(set(unmatched)),
                             'translation':translation,'sentence_translation':context})
    return warnings


def normalize_translation_alignments(result: dict, cues: list[dict], occurrences: list[dict], *,
                                     min_confidence: float = MIN_ALIGNMENT_CONFIDENCE) -> tuple[list[dict], list[dict]]:
    """Resolve model-proposed target text to exact spans in finalized translations.

    Gemini supplies semantic correspondence and target text only. Python finds
    that text in the final cue translation, calculates Unicode code-point
    offsets, assigns stable IDs, and rejects unresolved or ambiguous matches.
    """
    cue_by_id = {str(cue['id']): cue for cue in cues}
    occurrence_by_id = {str(item['occurrence_id']): item for item in occurrences}
    warnings: list[dict] = []
    alignments: list[dict] = []
    seen: set[tuple[str, str, int, int]] = set()
    supplied: set[str] = set()
    items = result.get('alignments', []) if isinstance(result, dict) else []
    if not isinstance(items, list):
        return [], [{'reason': 'invalid_alignment_response', 'detail': 'alignments must be an array'}]

    model_items_by_cue: dict[str, list[dict]] = {}
    model_position = {id(item): index for index, item in enumerate(items) if isinstance(item, dict)}
    for item in items:
        if isinstance(item, dict):
            model_items_by_cue.setdefault(str(item.get('cue_id', '')), []).append(item)

    def find_spans(translation: str, text: str) -> list[tuple[int, int, str]]:
        exact = []
        start = translation.find(text)
        while text and start >= 0:
            exact.append((start, start + len(text), translation[start:start + len(text)]))
            start = translation.find(text, start + 1)
        if exact or not text:
            return exact
        # Case differences should not make an otherwise exact target unusable;
        # offsets and stored text still come from the final translation.
        return [(match.start(), match.end(), match.group(0))
                for match in re.finditer(re.escape(text), translation, re.IGNORECASE)]

    used_by_cue: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        if not isinstance(item, dict):
            warnings.append({'reason': 'invalid_alignment_item'})
            continue
        occurrence_id = str(item.get('occurrence_id', '')).strip()
        if not occurrence_id:
            warnings.append({'reason': 'alignment_missing_occurrence_id'})
            continue
        supplied.add(occurrence_id)
        occurrence = occurrence_by_id.get(occurrence_id)
        if occurrence is None:
            warnings.append({'reason': 'alignment_unknown_occurrence', 'occurrence_id': occurrence_id})
            continue
        cue_id = str(item.get('cue_id', occurrence['cue_id']))
        if cue_id != str(occurrence['cue_id']):
            warnings.append({'reason': 'alignment_cue_mismatch', 'occurrence_id': occurrence_id,
                             'cue_id': cue_id, 'expected_cue_id': str(occurrence['cue_id'])})
            continue
        cue = cue_by_id.get(cue_id)
        if cue is None:
            warnings.append({'reason': 'alignment_unknown_cue', 'occurrence_id': occurrence_id, 'cue_id': cue_id})
            continue
        try:
            confidence = float(item.get('confidence'))
        except (TypeError, ValueError):
            confidence = -1.0
        if not 0.0 <= confidence <= 1.0:
            warnings.append({'reason': 'alignment_invalid_confidence', 'occurrence_id': occurrence_id,
                             'confidence': item.get('confidence')})
            continue
        alignment_type = str(item.get('alignment_type', 'direct')).strip().lower()
        if alignment_type not in ALIGNMENT_TYPES:
            warnings.append({'reason': 'alignment_invalid_type', 'occurrence_id': occurrence_id,
                             'alignment_type': alignment_type})
            continue
        spans = item.get('spans', [])
        if not isinstance(spans, list):
            warnings.append({'reason': 'alignment_spans_not_array', 'occurrence_id': occurrence_id})
            continue
        requires_cross_cue = bool(item.get('requires_cross_cue', False))
        if requires_cross_cue:
            warnings.append({'reason': 'alignment_requires_cross_cue', 'occurrence_id': occurrence_id,
                             'cue_id': cue_id})
            continue
        if not spans:
            warnings.append({'reason': 'alignment_missing_translation_span', 'occurrence_id': occurrence_id,
                             'cue_id': cue_id, 'alignment_type': alignment_type})
            continue
        if confidence < min_confidence:
            warnings.append({'reason': 'alignment_low_confidence', 'occurrence_id': occurrence_id,
                             'cue_id': cue_id, 'confidence': confidence})
            continue
        translation = str(cue.get('translation', ''))
        used = used_by_cue.setdefault(cue_id, [])
        for span in spans:
            if not isinstance(span, dict):
                warnings.append({'reason': 'invalid_translation_span', 'occurrence_id': occurrence_id})
                continue
            text = str(span.get('text', '')).strip()
            if not text:
                warnings.append({'reason': 'empty_translation_span_text', 'occurrence_id': occurrence_id})
                continue
            same_text_items = [candidate for candidate in model_items_by_cue.get(cue_id, [])
                               if candidate.get('spans') and any(
                                   str(value.get('text', '')).strip() == text
                                   for value in candidate.get('spans', []) if isinstance(value, dict))]
            candidates = find_spans(translation, text)
            available = [candidate for candidate in candidates
                         if not any(candidate[0] < prior_end and prior_start < candidate[1]
                                    for prior_start, prior_end in used)]
            if not available:
                warnings.append({'reason': 'translation_text_not_found', 'occurrence_id': occurrence_id,
                                 'cue_id': cue_id, 'text': text})
                continue
            pending_same = sum(1 for candidate in same_text_items
                               if model_position.get(id(candidate), -1) >= model_position.get(id(item), -1))
            if len(available) > 1 and pending_same >= len(available) and pending_same > 1:
                # Repeated target text is assigned left-to-right in the model's
                # occurrence order, which is deterministic and auditable.
                chosen = available[0]
            elif len(available) > 1:
                warnings.append({'reason': 'ambiguous_translation_text', 'occurrence_id': occurrence_id,
                                 'cue_id': cue_id, 'text': text,
                                 'candidate_count': len(available)})
                continue
            else:
                chosen = available[0]
            start, end, matched_text = chosen
            key = (occurrence_id, cue_id, start, end)
            if key in seen:
                warnings.append({'reason': 'duplicate_translation_alignment', 'occurrence_id': occurrence_id,
                                 'cue_id': cue_id, 'start_char': start, 'end_char': end})
                continue
            seen.add(key)
            seed = f'{occurrence_id}|{cue_id}|{start}|{end}'
            alignments.append({
                'alignment_id': 'ta_' + hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16],
                'occurrence_id': occurrence_id,
                'cue_id': cue_id,
                'text': matched_text,
                'start_char': start,
                'end_char': end,
                'alignment_type': alignment_type,
                'confidence': round(confidence, 4),
            })
            used.append((start, end))

    for occurrence_id in sorted(set(occurrence_by_id) - supplied):
        occurrence = occurrence_by_id[occurrence_id]
        warnings.append({'reason': 'alignment_missing_model_item', 'occurrence_id': occurrence_id,
                         'cue_id': str(occurrence['cue_id'])})
    alignments.sort(key=lambda item: (str(item['cue_id']), item['start_char'], item['end_char'], item['alignment_id']))
    non_overlapping: list[dict] = []
    for alignment in alignments:
        prior = next((item for item in reversed(non_overlapping)
                      if item['cue_id'] == alignment['cue_id']), None)
        if prior and alignment['start_char'] < prior['end_char']:
            warnings.append({'reason': 'overlapping_translation_alignment',
                             'occurrence_id': alignment['occurrence_id'],
                             'cue_id': alignment['cue_id'],
                             'alignment_id': alignment['alignment_id'],
                             'overlaps_alignment_id': prior['alignment_id']})
            continue
        non_overlapping.append(alignment)
    return non_overlapping, warnings
