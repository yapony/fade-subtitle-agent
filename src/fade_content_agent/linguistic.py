"""Concept identity and deterministic cue-local occurrence mapping."""
from __future__ import annotations
import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from typing import Any

CATEGORIES = {
    'noun', 'verb', 'adjective', 'adverb', 'fixed_expression',
    'idiom', 'collocation', 'grammar_pattern', 'other',
}
WORD_CATEGORIES = {'noun', 'verb', 'adjective', 'adverb'}
SELECTION_PRIORITIES = {'high', 'medium', 'low'}
MAX_WORD_OCCURRENCES_PER_SENTENCE = 2
MIN_ANNOTATED_SENTENCE_RATIO = 0.30
MAX_ANNOTATED_SENTENCE_RATIO = 0.60
_PRIORITY_RANK = {'high': 3, 'medium': 2, 'low': 1}
DIFFICULTY_RUBRIC = {
    1: 'Basic vocabulary and structures; approximate A1.',
    2: 'Elementary everyday language; approximate A2.',
    3: 'Intermediate vocabulary and structures; approximate B1–B2.',
    4: 'Advanced or specialized language; approximate C1–C2.',
}

def normalize(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())

def concept_key(category: str, lemma: str, meaning_key: str) -> str:
    return '|'.join((normalize(category), normalize(lemma), normalize(meaning_key)))

def concept_id_for(key: str) -> str:
    return 'lc_' + hashlib.sha256(normalize(key).encode('utf-8')).hexdigest()[:16]

def normalize_concept(raw: dict) -> tuple[dict | None, dict | None]:
    category=normalize(raw.get('category','')).replace(' ','_')
    lemma=' '.join(unicodedata.normalize('NFKC',str(raw.get('lemma',''))).split())
    meaning=' '.join(unicodedata.normalize('NFKC',str(raw.get('meaning_key',''))).split())
    difficulty=raw.get('difficulty')
    selection_priority=str(raw.get('selection_priority','medium')).strip().casefold()
    selection_reason=' '.join(str(raw.get('selection_reason','')).split())
    if category not in CATEGORIES:
        return None,{'reason':'invalid_category','category':category,'lemma':lemma}
    if not lemma or not meaning:
        return None,{'reason':'missing_canonical_meaning','category':category,'lemma':lemma}
    if not isinstance(difficulty,int) or isinstance(difficulty,bool) or difficulty not in DIFFICULTY_RUBRIC:
        return None,{'reason':'invalid_difficulty','category':category,'lemma':lemma,'difficulty':difficulty}
    if selection_priority not in SELECTION_PRIORITIES:
        return None,{'reason':'invalid_selection_priority','category':category,'lemma':lemma,'selection_priority':selection_priority}
    key=concept_key(category,lemma,meaning)
    concept={'concept_id':concept_id_for(key),'lemma':lemma,'meaning_key':meaning,'category':category,
             'frequency':0,'difficulty':difficulty,'translation':str(raw.get('translation','')).strip(),
             'explanation':str(raw.get('explanation','')).strip(),'selection_priority':selection_priority,
             'selection_reason':selection_reason,'occurrence_ids':[],'surface_forms':[]}
    if not concept['translation'] or not concept['explanation']:
        return None,{'reason':'missing_concept_metadata','category':category,'lemma':lemma}
    return concept,None

def merge_concept(catalog: dict[str,dict], raw: dict) -> tuple[str | None,dict | None]:
    """Merge a batch concept by deterministic identity; retain the first definition."""
    concept,warning=normalize_concept(raw)
    if warning: return None,warning
    cid=concept['concept_id']
    if cid not in catalog:
        catalog[cid]=concept
        return cid,None
    prior=catalog[cid]
    differing=[field for field in ('category','difficulty','translation','explanation','selection_priority') if normalize(str(prior[field]))!=normalize(str(concept[field]))]
    if differing: return cid,{'reason':'concept_metadata_conflict','concept_id':cid,'fields':differing}
    return cid,None

def normalize_occurrence_mappings(items: list[dict], concept_ids: set[str], sentence_ids: set[str]) -> tuple[list[dict],list[dict]]:
    """Validate one concept/form list per exact sentence context from model output."""
    mappings=[]; warnings=[]
    for index,item in enumerate(items):
        cid=str(item.get('concept_id','')); sid=str(item.get('sentence_id',''))
        if cid not in concept_ids:
            warnings.append({'reason':'unknown_concept_reference','concept_id':cid,'mapping_index':index}); continue
        if sid not in sentence_ids:
            warnings.append({'reason':'out_of_batch_sentence_reference','concept_id':cid,'sentence_id':sid,'mapping_index':index}); continue
        forms=list(dict.fromkeys(str(form).strip() for form in item.get('surface_forms',[]) if str(form).strip()))
        if not forms:
            warnings.append({'reason':'incomplete_occurrence_mapping','concept_id':cid,'sentence_id':sid}); continue
        mappings.append({'concept_id':cid,'surface_forms':forms,'sentence_ids':[sid]})
    return mappings,warnings

def _phrase_pattern(surface: str) -> re.Pattern[str] | None:
    tokens=unicodedata.normalize('NFC',surface).casefold().split()
    if not tokens: return None
    body=r'[\s\-‐‑–—]+'.join(re.escape(token) for token in tokens)
    return re.compile(r'(?<!\w)(' + body + r')(?!\w)', re.IGNORECASE | re.UNICODE)

_NON_LINGUISTIC = re.compile(r'^\s*[\[(]\s*(?:aplausos?|applause|risas?|laugh(?:ter|s)?|music|m[uú]sica|suspiros?|sighs?|inaudible|inaudible speech|silencio|silence|ruido|noise|sonido|sound effects?)[^\])]*[\])]\s*$',re.IGNORECASE)

def is_non_linguistic_cue(text: str) -> bool:
    """Return true only for common bracketed stage/sound cues, never dialogue."""
    return bool(_NON_LINGUISTIC.fullmatch(str(text or '')))

def _cue_text(cue: Any) -> str:
    return str(cue.text if hasattr(cue,'text') else cue.get('source_text',''))

def _cue_sequence(cue: Any) -> int:
    return int(cue.sequence if hasattr(cue,'sequence') else cue['sequence'])

def _surface_variants(surface: str, category: str, source_language: str) -> list[str]:
    """Conservatively add Spanish plural/gender spelling variants for nominals."""
    variants=[surface]
    language=normalize(source_language).split('-')[0].split('_')[0]
    if language!='es' or category not in {'noun','adjective'}: return variants
    tokens=surface.split()
    if not tokens: return variants
    word=tokens[-1]
    stems=[word]
    if category=='adjective' and word.casefold().endswith('o') and len(word)>1:
        stems.append(word[:-1]+'a')
    for stem in stems:
        lower=stem.casefold()
        if lower.endswith('z'):
            plural=stem[:-1]+'ces'
        elif lower.endswith(('a','e','i','o','u','á','é','í','ó','ú')):
            plural=stem+'s'
        elif lower.endswith(('s','x')):
            plural=stem
        else:
            plural=stem+'es'
        if plural.casefold()!=stem.casefold():
            variants.append(' '.join(tokens[:-1]+[plural]))
    return list(dict.fromkeys(variants))

def _sentence_stream(cue_ids: list[str], cue_map: dict[str, Any]):
    text=[]; positions=[]
    for cue_index,cue_id in enumerate(cue_ids):
        cue=cue_map[cue_id]
        source=_cue_text(cue)
        if cue_index:
            text.append(' '); positions.append(None)
        if is_non_linguistic_cue(source):
            # Treat a bracketed sound cue as a separator. It has no linguistic
            # characters to annotate, but an expression may resume after it.
            continue
        offset=0
        while offset<len(source):
            end=offset+1
            while end<len(source) and unicodedata.combining(source[end]): end+=1
            folded=unicodedata.normalize('NFC',source[offset:end]).casefold()
            for normalized_char in folded:
                text.append(normalized_char); positions.append((cue_id,offset,end))
            offset=end
    return ''.join(text),positions

def _cue_runs(linked: list[str], cue_map: dict[str,Any], cue_sequence: dict[str,int]) -> list[list[str]]:
    runs=[]
    for cue_id in sorted(linked,key=lambda cid:cue_sequence[cid]):
        if not runs:
            runs.append([cue_id]); continue
        previous=runs[-1][-1]; gap=cue_sequence[cue_id]-cue_sequence[previous]-1
        if gap<=0:
            runs[-1].append(cue_id); continue
        intervening=[cue for cue in cue_map.values() if cue_sequence[previous]<_cue_sequence(cue)<cue_sequence[cue_id]]
        if len(intervening)==gap and all(is_non_linguistic_cue(_cue_text(cue)) for cue in intervening):
            runs[-1].append(cue_id)
        else:
            runs.append([cue_id])
    return runs

def map_occurrences(concepts: dict[str,dict], mappings: list[dict], cues: list[Any], sentences: list[dict], source_language: str=''):
    """Map model-confirmed concept/sentence links onto exact source cue substrings.

    A cross-cue expression yields one occurrence per contributing cue. Those cue-local
    occurrences share `instance_id`, making concept frequency count the expression once.
    """
    cue_map={c.id:c for c in cues} if cues and hasattr(cues[0],'id') else {c['id']:c for c in cues}
    sentence_map={s['id']:s for s in sentences}
    sentence_to_concepts=defaultdict(set); instances=defaultdict(set); occurrence_map={}
    warnings=[]; warning_keys=set()
    def warn(item):
        key=(item.get('reason'),item.get('concept_id'),item.get('sentence_id'),item.get('surface_form'))
        if key not in warning_keys: warning_keys.add(key); warnings.append(item)
    cue_sequence={cid:_cue_sequence(c) for cid,c in cue_map.items()}
    mapping_pairs=defaultdict(set)
    for mapping in mappings:
        for sentence_id in mapping.get('sentence_ids',[]):
            mapping_pairs[(mapping.get('concept_id'),sentence_id)].update(str(form).strip() for form in mapping.get('surface_forms',[]) if str(form).strip())
    for (cid,sentence_id),form_set in mapping_pairs.items():
        if cid not in concepts:
            warn({'reason':'unknown_concept','concept_id':cid}); continue
        category=concepts[cid].get('category','')
        lemma=normalize(concepts[cid].get('lemma',''))
        forms=sorted(form_set,key=str.casefold)
        sentence=sentence_map.get(sentence_id)
        if sentence is None:
            warn({'reason':'unknown_sentence','concept_id':cid,'sentence_id':sentence_id}); continue
        linked=[cue_id for cue_id in sentence.get('cue_ids',[]) if cue_id in cue_map]
        if not linked:
            warn({'reason':'sentence_has_no_known_cues','concept_id':cid,'sentence_id':sentence_id}); continue
        cue_runs=_cue_runs(linked,cue_map,cue_sequence)
        candidates=[]; found_forms=set()
        for surface in forms:
            for variant in _surface_variants(surface,category,source_language):
                pattern=_phrase_pattern(variant)
                if pattern is None: continue
                for cue_run in cue_runs:
                    stream,positions=_sentence_stream(cue_run,cue_map)
                    hits=list(pattern.finditer(stream))
                    if hits: found_forms.add(surface)
                    candidates.extend((surface,hit,positions) for hit in hits)
        if not found_forms:
            warn({'reason':'unmatched_occurrence','concept_id':cid,'sentence_id':sentence_id,'surface_forms':forms})
            continue
        unmatched_forms=sorted(set(forms)-found_forms,key=str.casefold)
        if unmatched_forms:
            warn({'reason':'alternative_surface_mismatch','concept_id':cid,'sentence_id':sentence_id,'surface_forms':unmatched_forms,'matched_surface_forms':sorted(found_forms,key=str.casefold)})
        # Prefer the canonical lemma, then the most complete model-supplied form.
        candidates.sort(key=lambda row:(normalize(row[0])!=lemma,-len(row[0]),normalize(row[0]),row[1].start()))
        accepted=[]
        for surface,hit,positions in candidates:
            mapped=[p for p in positions[hit.start(1):hit.end(1)] if p is not None]
            if not mapped: continue
            grouped=defaultdict(list)
            for cue_id,start,end in mapped: grouped[cue_id].append((start,end))
            segments=[]
            for cue_id,spans in grouped.items():
                start,end=min(a for a,_ in spans),max(b for _,b in spans)
                source=_cue_text(cue_map[cue_id])
                if source[start:end]: segments.append((cue_id,start,end,source[start:end]))
            if not segments: continue
            overlapping=[]
            for cue_id,start,end,exact_text in segments:
                for prior_cue,prior_start,prior_end in accepted:
                    if prior_cue==cue_id and start<prior_end and prior_start<end and (start,end)!=(prior_start,prior_end):
                        overlapping.append((cue_id,start,end,prior_start,prior_end))
                for (prior_cid,prior_cue,prior_start,prior_end),prior in occurrence_map.items():
                    if prior_cid==cid and prior_cue==cue_id and start<prior_end and prior_start<end and (start,end)!=(prior_start,prior_end):
                        overlapping.append((cue_id,start,end,prior_start,prior_end))
            if overlapping:
                warn({'reason':'redundant_overlap_same_concept','concept_id':cid,'sentence_id':sentence_id,'surface_form':surface,
                      'overlaps':[{'cue_id':cue,'start_char':start,'end_char':end,'kept_start_char':prior_start,'kept_end_char':prior_end} for cue,start,end,prior_start,prior_end in overlapping]})
                continue
            accepted.extend((cue_id,start,end) for cue_id,start,end,_ in segments)
            instance_seed=cid+'|'+'|'.join(f'{c}:{a}:{b}' for c,a,b,_ in segments)
            instance_id='li_' + hashlib.sha256(instance_seed.encode('utf-8')).hexdigest()[:16]
            instances[cid].add(instance_id); sentence_to_concepts[sentence_id].add(cid)
            for cue_id,start,end,exact_text in segments:
                seed=f'{cid}|{cue_id}|{start}|{end}'
                oid='lo_' + hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]
                key=(cid,cue_id,start,end)
                if key in occurrence_map:
                    occurrence_map[key]['sentence_ids']=sorted(set(occurrence_map[key]['sentence_ids'])|{sentence_id})
                    continue
                occurrence_map[key]={'occurrence_id':oid,'instance_id':instance_id,'concept_id':cid,'cue_id':cue_id,'text':exact_text,'start_char':start,'end_char':end,'sentence_ids':[sentence_id]}
    occurrences=sorted(occurrence_map.values(),key=lambda x:(next((c.sequence if hasattr(c,'sequence') else c['sequence'] for c in cues if (c.id if hasattr(c,'id') else c['id'])==x['cue_id']),0),x['start_char'],x['end_char'],x['concept_id']))
    # Surface collisions can be genuine polysemy; retain both concept links and report for review.
    by_span=defaultdict(set)
    for occurrence in occurrences:
        by_span[(occurrence['cue_id'],occurrence['start_char'],occurrence['end_char'])].add(occurrence['concept_id'])
    for (cue_id,start,end),ids in by_span.items():
        if len(ids)>1: warn({'reason':'same_span_multiple_concepts','cue_id':cue_id,'start_char':start,'end_char':end,'concept_ids':sorted(ids)})
    return occurrences,warnings,{cid:len(values) for cid,values in instances.items()},{sid:sorted(ids) for sid,ids in sentence_to_concepts.items()}

def cue_occurrence_ids(cues: list[Any], occurrences: list[dict]) -> dict[str,list[str]]:
    out={c.id if hasattr(c,'id') else c['id']:[] for c in cues}
    for occurrence in occurrences: out[occurrence['cue_id']].append(occurrence['occurrence_id'])
    return out

def exclude_zero_occurrence_concepts(concepts: dict[str,dict]) -> tuple[dict[str,dict],list[dict]]:
    """Return only concepts with located occurrences and explicit exclusion warnings."""
    published={}; warnings=[]
    for cid,concept in concepts.items():
        if concept.get('frequency',0)>0 and concept.get('occurrence_ids'):
            published[cid]=concept
        else:
            warnings.append({'reason':'zero_occurrence_concept_not_published','concept_id':cid,
                             'lemma':concept.get('lemma',''),'category':concept.get('category','')})
    return published,warnings


def limit_word_occurrences(
    occurrences: list[dict],
    concepts: dict[str,dict],
    sentences: list[dict],
    max_per_sentence: int = MAX_WORD_OCCURRENCES_PER_SENTENCE,
) -> tuple[list[dict],list[dict]]:
    """Keep at most the configured number of word spans per semantic sentence.

    AI still proposes all useful concepts. This deterministic pass only limits
    published word occurrences; expressions and grammar patterns are untouched.
    A concept may have several occurrences in a sentence, and each occurrence
    competes for one of that sentence's word slots.
    """
    if not isinstance(max_per_sentence,int) or isinstance(max_per_sentence,bool) or max_per_sentence < 0:
        raise ValueError('max_per_sentence must be a non-negative integer')
    sentence_order={sentence['id']:index for index,sentence in enumerate(sentences)}
    expression_spans=defaultdict(list)
    for occurrence in occurrences:
        concept=concepts.get(occurrence.get('concept_id'))
        if concept and concept.get('category') not in WORD_CATEGORIES:
            expression_spans[occurrence.get('cue_id')].append((occurrence.get('start_char',-1),occurrence.get('end_char',-1)))
    word_candidates=[]
    for occurrence in occurrences:
        concept=concepts.get(occurrence.get('concept_id'))
        if not concept or concept.get('category') not in WORD_CATEGORIES:
            continue
        overlaps_expression=any(
            start < occurrence.get('end_char',-1) and occurrence.get('start_char',-1) < end
            for start,end in expression_spans.get(occurrence.get('cue_id'),[])
        )
        word_candidates.append({
            'occurrence': occurrence,
            'priority': concept.get('selection_priority','medium') if concept.get('selection_priority','medium') in SELECTION_PRIORITIES else 'medium',
            'overlaps_expression': overlaps_expression,
            'category': concept.get('category',''),
            'sentence_ids': [sid for sid in occurrence.get('sentence_ids',[]) if sid in sentence_order],
        })

    selected_ids=set()
    warnings=[]
    priority_rank={'high':0,'medium':1,'low':2}
    counts_by_sentence=defaultdict(int)
    used_categories_by_sentence=defaultdict(set)
    remaining={item['occurrence']['occurrence_id']:item for item in word_candidates if item['sentence_ids']}
    while remaining:
        feasible=[item for item in remaining.values() if all(counts_by_sentence[sid] < max_per_sentence for sid in item['sentence_ids'])]
        if not feasible: break
        best=min(feasible,key=lambda item:(
            priority_rank[item['priority']],
            item['overlaps_expression'],
            item['category'] in set().union(*(used_categories_by_sentence[sid] for sid in item['sentence_ids'])),
            min(sentence_order[sid] for sid in item['sentence_ids']),
            item['occurrence'].get('start_char',10**9),
            item['occurrence'].get('concept_id',''),
        ))
        occurrence=best['occurrence']; occurrence_id=occurrence['occurrence_id']
        selected_ids.add(occurrence_id)
        for sentence_id in best['sentence_ids']:
            counts_by_sentence[sentence_id]+=1
            used_categories_by_sentence[sentence_id].add(best['category'])
        remaining.pop(occurrence_id)
    for occurrence_id,item in remaining.items():
        occurrence=item['occurrence']
        blocked_sentence=next((sid for sid in item['sentence_ids'] if counts_by_sentence[sid] >= max_per_sentence), item['sentence_ids'][0])
        warnings.append({
            'reason':'single_word_sentence_cap',
            'sentence_id':blocked_sentence,
            'concept_id':occurrence.get('concept_id'),
            'occurrence_id':occurrence_id,
            'max_per_sentence':max_per_sentence,
        })
    filtered=[]
    for occurrence in occurrences:
        concept=concepts.get(occurrence.get('concept_id'))
        if concept and concept.get('category') in WORD_CATEGORIES and occurrence.get('occurrence_id') not in selected_ids:
            continue
        filtered.append(occurrence)
    return filtered,warnings


def annotated_sentence_ids(occurrences: list[dict], sentences: list[dict]) -> set[str]:
    known={sentence['id'] for sentence in sentences}
    return {sid for occurrence in occurrences for sid in occurrence.get('sentence_ids',[]) if sid in known}


def limit_annotated_sentence_coverage(
    occurrences: list[dict],
    concepts: dict[str,dict],
    sentences: list[dict],
    max_ratio: float = MAX_ANNOTATED_SENTENCE_RATIO,
) -> tuple[list[dict],list[dict],dict]:
    """Cap the share of semantic sentences with located annotations by priority."""
    if not isinstance(max_ratio,(int,float)) or isinstance(max_ratio,bool) or not 0 <= max_ratio <= 1:
        raise ValueError('max_ratio must be between 0 and 1')
    sentence_order={sentence['id']:index for index,sentence in enumerate(sentences)}
    covered=annotated_sentence_ids(occurrences,sentences)
    limit=math.floor(max_ratio*len(sentences))
    if len(covered)<=limit:
        return occurrences,[],{'sentence_count':len(sentences),'annotated_sentence_count':len(covered),'max_annotated_sentence_count':limit}

    priorities=defaultdict(list); sentence_concepts=defaultdict(set)
    for occurrence in occurrences:
        cid=occurrence.get('concept_id'); concept=concepts.get(cid,{})
        priority=concept.get('selection_priority','medium')
        rank=_PRIORITY_RANK.get(priority,2)
        for sid in occurrence.get('sentence_ids',[]):
            if sid in covered:
                priorities[sid].append(rank)
                sentence_concepts[sid].add(cid)
    ranked=sorted(covered,key=lambda sid:(-max(priorities[sid],default=2),-sum(priorities[sid]),sentence_order[sid]))
    kept=set(ranked[:limit])
    warnings=[{'reason':'annotation_coverage_cap','sentence_id':sid,
               'concept_ids':sorted(sentence_concepts[sid]),'max_ratio':max_ratio}
              for sid in ranked[limit:]]
    filtered=[]
    for occurrence in occurrences:
        retained_ids=sorted(set(occurrence.get('sentence_ids',[]))&kept)
        if retained_ids:
            filtered.append({**occurrence,'sentence_ids':retained_ids})
    return filtered,warnings,{'sentence_count':len(sentences),'annotated_sentence_count':len(kept),'max_annotated_sentence_count':limit}
