"""Cross-reference and structural validation for schema-v2 learning packages."""
from __future__ import annotations
import re
from .linguistic import CATEGORIES, WORD_CATEGORIES, SELECTION_PRIORITIES, MAX_WORD_OCCURRENCES_PER_SENTENCE, DIFFICULTY_RUBRIC, concept_id_for, concept_key, is_non_linguistic_cue
from .schema_validation import validate_schemas
from .checkpoints import CHECKPOINT_INTERVAL_SECONDS, CHECKPOINT_MIN_DURATION_SECONDS, final_question_count

class ValidationError(ValueError): pass

def validate(package: dict, duration: float) -> list[str]:
    errors=validate_schemas(package); sub=package['subtitles']; cues=sub['display_cues']; sentences=sub['sentences']
    concepts=package['vocabulary']['concepts']; occurrences=sub['linguistic_occurrences']
    json_files=('manifest','subtitles','vocabulary','culture','quizzes')
    for name in json_files:
        if package[name].get('schema_version')!='2.0.0': errors.append(f"wrong schema version: {name}")
    cue_by_id={c['id']:c for c in cues}; sentence_by_id={s['id']:s for s in sentences}; concept_by_id={c['concept_id']:c for c in concepts}
    manifest=package['manifest']; video=manifest.get('video',{})
    if video.get('duration_seconds')!=duration: errors.append('manifest duration does not match validation duration')
    required_files={'manifest.json','subtitles.json','vocabulary.json','culture.json','quizzes.json','review-report.md'}
    if set(manifest.get('files',[]))!=required_files: errors.append('manifest file list does not match package contents')
    if len(cue_by_id)!=len(cues): errors.append('display cue IDs are not unique')
    if len(sentence_by_id)!=len(sentences): errors.append('semantic sentence IDs are not unique')
    if len(concept_by_id)!=len(concepts): errors.append('linguistic concept IDs are not unique')
    if [c['sequence'] for c in cues] != list(range(1,len(cues)+1)): errors.append('cue sequence is incomplete or unordered')
    for cue in cues:
        if not (0<=cue['start']<=cue['end']<=duration): errors.append(f"invalid cue timing: {cue['id']}")
        if any(sid not in sentence_by_id for sid in cue.get('sentence_ids',[])): errors.append(f"unknown sentence reference in cue {cue['id']}")
    for warning in sub.get('translation_warnings',[]):
        if warning.get('cue_id') not in cue_by_id: errors.append(f"translation warning references unknown cue: {warning.get('cue_id')}")
    for sentence in sentences:
        if not sentence.get('cue_ids') or any(cid not in cue_by_id for cid in sentence['cue_ids']): errors.append(f"invalid cue mapping: {sentence['id']}")
        if len(sentence.get('concept_ids',[]))!=len(set(sentence.get('concept_ids',[]))): errors.append(f"duplicate concept reference in sentence {sentence['id']}")
        if any(cid not in concept_by_id for cid in sentence.get('concept_ids',[])): errors.append(f"unknown concept reference in sentence {sentence['id']}")
        for cue_id in sentence.get('cue_ids',[]):
            if cue_id in cue_by_id and sentence['id'] not in cue_by_id[cue_id].get('sentence_ids',[]): errors.append(f"cue/sentence mapping not bidirectional: {sentence['id']} -> {cue_id}")
    for concept in concepts:
        cid=concept['concept_id']
        expected=concept_id_for(concept_key(concept.get('category',''),concept.get('lemma',''),concept.get('meaning_key','')))
        if cid!=expected: errors.append(f"concept ID is inconsistent with canonical identity: {cid}")
        if concept.get('category') not in CATEGORIES: errors.append(f"invalid category: {cid}")
        difficulty=concept.get('difficulty')
        if not isinstance(difficulty,int) or isinstance(difficulty,bool) or difficulty not in DIFFICULTY_RUBRIC: errors.append(f"invalid difficulty: {cid}")
        selection_priority=concept.get('selection_priority','medium')
        if selection_priority not in SELECTION_PRIORITIES: errors.append(f"invalid selection priority: {cid}")
        frequency=concept.get('frequency')
        if not isinstance(frequency,int) or isinstance(frequency,bool) or frequency<0: errors.append(f"invalid frequency: {cid}")
        if not concept.get('lemma') or not concept.get('translation') or not concept.get('explanation'): errors.append(f"incomplete concept metadata: {cid}")
        refs=concept.get('occurrence_ids',[])
        if len(refs)!=len(set(refs)): errors.append(f"duplicate concept occurrence references: {cid}")
        if not refs: errors.append(f"published concept has no located occurrences: {cid}")
    occurrence_by_id={o['occurrence_id']:o for o in occurrences}
    if len(occurrence_by_id)!=len(occurrences): errors.append('linguistic occurrence IDs are not unique')
    seen=set(); spans_by_concept_cue={}; instances= {}
    for occurrence in occurrences:
        oid=occurrence['occurrence_id']; cid=occurrence['concept_id']; cue_id=occurrence['cue_id']
        if cid not in concept_by_id: errors.append(f"unknown concept reference in occurrence {oid}"); continue
        if cue_id not in cue_by_id: errors.append(f"unknown cue reference in occurrence {oid}"); continue
        cue=cue_by_id[cue_id]; start,end=occurrence.get('start_char',-1),occurrence.get('end_char',-1)
        if not isinstance(start,int) or isinstance(start,bool) or not isinstance(end,int) or isinstance(end,bool) or start<0 or end<=start or end>len(cue['source_text']) or cue['source_text'][start:end]!=occurrence.get('text'):
            errors.append(f"invalid Unicode code-point offset/text: {oid}")
        signature=(cid,cue_id,start,end)
        if signature in seen: errors.append(f"duplicate occurrence: {oid}")
        seen.add(signature); spans_by_concept_cue.setdefault((cid,cue_id),[]).append((start,end,oid))
        instance_id=occurrence.get('instance_id')
        if not instance_id: errors.append(f"missing occurrence instance ID: {oid}")
        else: instances.setdefault(cid,{})[instance_id]=True
        sentence_refs=occurrence.get('sentence_ids',[])
        if not sentence_refs or len(sentence_refs)!=len(set(sentence_refs)) or any(sid not in sentence_by_id for sid in sentence_refs): errors.append(f"invalid sentence references in occurrence {oid}")
        for sid in sentence_refs:
            if sid in sentence_by_id and cue_id not in sentence_by_id[sid].get('cue_ids',[]): errors.append(f"occurrence sentence/cue mismatch: {oid}")
        if oid not in concept_by_id[cid].get('occurrence_ids',[]): errors.append(f"concept missing occurrence reference: {oid}")
    for key,spans in spans_by_concept_cue.items():
        spans.sort()
        for left,right in zip(spans,spans[1:]):
            if right[0]<left[1]: errors.append(f"overlapping duplicate annotations for concept {key[0]} in cue {key[1]}")
    for concept in concepts:
        cid=concept['concept_id']; linked=[o['occurrence_id'] for o in occurrences if o['concept_id']==cid]
        if set(linked)!=set(concept.get('occurrence_ids',[])): errors.append(f"concept occurrence references are incomplete: {cid}")
        if concept.get('frequency')!=len(instances.get(cid,{})): errors.append(f"concept frequency does not match located instances: {cid}")
    instance_groups={}
    for occurrence in occurrences:
        if occurrence.get('instance_id'): instance_groups.setdefault(occurrence['instance_id'],[]).append(occurrence)
    cue_sequence={c['id']:c['sequence'] for c in cues}
    cue_at_sequence={c['sequence']:c for c in cues}
    for instance_id,parts in instance_groups.items():
        if len({p['concept_id'] for p in parts})!=1: errors.append(f"cross-cue instance maps to multiple concepts: {instance_id}")
        if len({p['cue_id'] for p in parts})!=len(parts): errors.append(f"duplicate cue segment in occurrence instance: {instance_id}")
        if len(parts)>1:
            ordered=sorted(parts,key=lambda p:cue_sequence.get(p['cue_id'],0))
            seq=[cue_sequence.get(p['cue_id'],0) for p in ordered]
            for a,b in zip(seq,seq[1:]):
                if b<=a or any(n not in cue_at_sequence or not is_non_linguistic_cue(cue_at_sequence[n]['source_text']) for n in range(a+1,b)):
                    errors.append(f"cross-cue expression crosses dialogue or unknown cue gap: {instance_id}"); break
            shared=set(ordered[0].get('sentence_ids',[]))
            for part in ordered[1:]: shared &= set(part.get('sentence_ids',[]))
            if not shared: errors.append(f"cross-cue expression has no shared semantic sentence: {instance_id}")
    for cue in cues:
        refs=cue.get('linguistic_occurrence_ids',[])
        if len(refs)!=len(set(refs)): errors.append(f"duplicate occurrence references in cue {cue['id']}")
        for oid in refs:
            if oid not in occurrence_by_id or occurrence_by_id[oid]['cue_id']!=cue['id']: errors.append(f"invalid cue occurrence reference in {cue['id']}")
        actual={o['occurrence_id'] for o in occurrences if o['cue_id']==cue['id']}
        if set(refs)!=actual: errors.append(f"cue occurrence mapping is incomplete: {cue['id']}")
    word_occurrences_by_sentence={}
    for occurrence in occurrences:
        concept=concept_by_id.get(occurrence.get('concept_id'))
        if not concept or concept.get('category') not in WORD_CATEGORIES: continue
        for sentence_id in occurrence.get('sentence_ids',[]):
            word_occurrences_by_sentence.setdefault(sentence_id,[]).append(occurrence['occurrence_id'])
    for sentence_id,occurrence_ids in word_occurrences_by_sentence.items():
        if len(set(occurrence_ids))>MAX_WORD_OCCURRENCES_PER_SENTENCE:
            errors.append(f"too many word occurrences in semantic sentence {sentence_id}: {len(set(occurrence_ids))} > {MAX_WORD_OCCURRENCES_PER_SENTENCE}")
    translation_alignments=sub.get('translation_alignments',[])
    alignment_by_id={item.get('alignment_id'):item for item in translation_alignments}
    if len(alignment_by_id)!=len(translation_alignments): errors.append('translation alignment IDs are not unique')
    seen_alignment_spans=set(); alignment_spans_by_cue={}
    for alignment in translation_alignments:
        alignment_id=alignment.get('alignment_id','<missing-id>'); occurrence_id=alignment.get('occurrence_id'); cue_id=alignment.get('cue_id')
        if occurrence_id not in occurrence_by_id: errors.append(f"unknown source occurrence reference in translation alignment {alignment_id}"); continue
        occurrence=occurrence_by_id[occurrence_id]
        if cue_id!=occurrence['cue_id']: errors.append(f"translation alignment cue mismatch: {alignment_id}")
        if cue_id not in cue_by_id: errors.append(f"unknown cue reference in translation alignment {alignment_id}"); continue
        cue=cue_by_id[cue_id]; start,end=alignment.get('start_char',-1),alignment.get('end_char',-1)
        if (not isinstance(start,int) or isinstance(start,bool) or not isinstance(end,int) or isinstance(end,bool)
                or start<0 or end<=start or end>len(cue.get('translation',''))
                or cue.get('translation','')[start:end]!=alignment.get('text')):
            errors.append(f"invalid Unicode code-point offset/text in translation alignment: {alignment_id}")
        confidence=alignment.get('confidence')
        if not isinstance(confidence,(int,float)) or isinstance(confidence,bool) or not 0<=confidence<=1:
            errors.append(f"invalid alignment confidence: {alignment_id}")
        signature=(occurrence_id,cue_id,start,end)
        if signature in seen_alignment_spans: errors.append(f"duplicate translation alignment span: {alignment_id}")
        seen_alignment_spans.add(signature)
        alignment_spans_by_cue.setdefault(cue_id,[]).append((start,end,alignment_id))
    for cue_id, spans in alignment_spans_by_cue.items():
        spans.sort()
        for left,right in zip(spans,spans[1:]):
            if right[0] < left[1]: errors.append(f"overlapping translation alignment spans in cue {cue_id}: {left[2]}, {right[2]}")
    for warning in sub.get('translation_alignment_warnings',[]):
        if warning.get('cue_id') is not None and warning.get('cue_id') not in cue_by_id:
            errors.append(f"translation alignment warning references unknown cue: {warning.get('cue_id')}")
        if warning.get('occurrence_id') is not None and warning.get('occurrence_id') not in occurrence_by_id:
            errors.append(f"translation alignment warning references unknown occurrence: {warning.get('occurrence_id')}")
    for cue in cues:
        refs=cue.get('translation_alignment_ids',[])
        if len(refs)!=len(set(refs)): errors.append(f"duplicate translation alignment references in cue {cue['id']}")
        for alignment_id in refs:
            if alignment_id not in alignment_by_id or alignment_by_id[alignment_id].get('cue_id')!=cue['id']:
                errors.append(f"invalid cue translation alignment reference in {cue['id']}")
        actual={item['alignment_id'] for item in translation_alignments if item.get('cue_id')==cue['id']}
        if set(refs)!=actual: errors.append(f"cue translation alignment mapping is incomplete: {cue['id']}")
    culture=package['culture']['items']; culture_ids={x['id'] for x in culture}
    if len(culture_ids)!=len(culture): errors.append('cultural annotation IDs are not unique')
    for item in culture:
        if not item.get('sentence_ids') or any(sid not in sentence_by_id for sid in item['sentence_ids']): errors.append(f"invalid culture sentence references: {item.get('id','<missing-id>')}")
        trigger_time=item.get('trigger_time')
        valid_trigger_time=(isinstance(trigger_time,(int,float)) and not isinstance(trigger_time,bool)
                            and 0<=trigger_time<=duration)
        if not valid_trigger_time: errors.append(f"invalid culture trigger time: {item.get('id','<missing-id>')}")
        associated_starts=[cue_by_id[cue_id]['start'] for sid in item.get('sentence_ids',[]) if sid in sentence_by_id for cue_id in sentence_by_id[sid].get('cue_ids',[]) if cue_id in cue_by_id]
        if associated_starts and (not valid_trigger_time or abs(trigger_time-min(associated_starts))>1e-6):
            errors.append(f"culture trigger_time does not match associated subtitle time: {item.get('id','<missing-id>')}")
        confidence=item.get('confidence')
        if not isinstance(confidence,(int,float)) or isinstance(confidence,bool) or not 0<=confidence<=1: errors.append(f"invalid culture confidence: {item.get('id','<missing-id>')}")
        if 'needs_review' in item and not isinstance(item['needs_review'],bool): errors.append(f"invalid culture review flag: {item.get('id','<missing-id>')}")
        note=item.get('verification_note','')
        if item.get('needs_review') and (not isinstance(note,str) or not note.strip()): errors.append(f"cultural review flag has no verification note: {item.get('id','<missing-id>')}")
    quizzes=package['quizzes']; checkpoints=quizzes['checkpoints']; last=-1
    for cp in checkpoints:
        if cp['trigger_time']<last or cp['trigger_time']>duration: errors.append(f"invalid checkpoint order/time: {cp['id']}")
        last=cp['trigger_time']
    expected_targets=list(range(CHECKPOINT_INTERVAL_SECONDS,int(duration),CHECKPOINT_INTERVAL_SECONDS)) if duration>=CHECKPOINT_MIN_DURATION_SECONDS else []
    if [cp['target_time'] for cp in checkpoints]!=expected_targets: errors.append('checkpoint schedule does not match the 10-minute in-video schedule')
    if any(cp.get('is_final') for cp in checkpoints): errors.append('in-video checkpoint cannot be marked final')
    if any(len(cp.get('question_ids',[]))!=10 for cp in checkpoints): errors.append('each in-video checkpoint must have exactly 10 questions')
    final_quiz=quizzes.get('final_quiz',{}); final_questions=final_quiz.get('questions',[])
    expected_final_count=final_question_count(duration)
    final_trigger_time=final_quiz.get('trigger_time')
    if (not isinstance(final_trigger_time,(int,float)) or isinstance(final_trigger_time,bool)
            or abs(final_trigger_time-duration)>1e-6):
        errors.append('final quiz trigger_time must equal video duration')
    if len(final_questions)!=expected_final_count: errors.append(f'final quiz must have {expected_final_count} questions')
    if len(final_quiz.get('question_ids',[]))!=len(final_questions) or set(final_quiz.get('question_ids',[]))!={q.get('id') for q in final_questions}: errors.append('final quiz question references are incomplete')
    questions=quizzes['questions']; checkpoint_ids={c['id'] for c in checkpoints}
    qids=[q.get('id') for q in questions]+[q.get('id') for q in final_questions]
    if len(set(qids))!=len(qids): errors.append('quiz question IDs are not unique')
    for warning in quizzes.get('quiz_warnings',[]):
        if warning.get('question_id') not in qids: errors.append(f"quiz warning references unknown question: {warning.get('question_id')}")
    preceding=0.0; lower_bounds={}
    for cp in checkpoints: lower_bounds[cp['id']]=preceding; preceding=cp['trigger_time']
    all_questions=[(q,'checkpoint') for q in questions]+[(q,'final') for q in final_questions]
    for q,quiz_kind in all_questions:
        qid=q.get('id','<missing-id>'); checkpoint_id=q.get('checkpoint_id') if quiz_kind=='checkpoint' else None
        if quiz_kind=='checkpoint' and checkpoint_id not in checkpoint_ids: errors.append(f"unknown checkpoint: {qid}")
        if q.get('question_type') not in {'content_comprehension','vocabulary_recognition','expression_recognition','grammar_understanding'}: errors.append(f"invalid question type: {qid}")
        options=q.get('options'); answer=q.get('correct_answer_index')
        if not isinstance(options,list) or len(options)!=4 or not isinstance(answer,int) or isinstance(answer,bool) or not 0<=answer<len(options): errors.append(f"invalid answer index/options: {qid}")
        elif any(not isinstance(option,str) or not option.strip() for option in options) or len({option.strip().casefold() for option in options if isinstance(option,str)})!=len(options): errors.append(f"empty or duplicate quiz options: {qid}")
        elif len({bool(re.search(r'[.!?…][\"”’)]*$',option.strip())) for option in options})>1: errors.append(f"quiz options mix sentence and fragment formats: {qid}")
        sentence_refs=q.get('sentence_ids',[])
        if not isinstance(sentence_refs,list) or any(sid not in sentence_by_id for sid in sentence_refs): errors.append(f"unknown quiz evidence sentence: {qid}")
        evidence=q.get('evidence','')
        if not isinstance(evidence,str) or not evidence.strip() or not any(evidence in sentence_by_id[sid].get('text','') for sid in sentence_refs if sid in sentence_by_id): errors.append(f"quiz evidence is not an exact substring of a cited sentence: {qid}")
        evidence_sentence_id=q.get('evidence_sentence_id')
        if evidence_sentence_id not in sentence_refs or evidence_sentence_id not in sentence_by_id or evidence!=sentence_by_id.get(evidence_sentence_id,{}).get('text'):
            errors.append(f"quiz evidence must be copied from its evidence_sentence_id: {qid}")
        concept_refs=q.get('concept_ids',[])
        if not isinstance(concept_refs,list) or len(concept_refs)!=len(set(concept_refs)) or any(cid not in concept_by_id for cid in concept_refs): errors.append(f"unknown/duplicate quiz concept: {qid}")
        else:
            evidence_concepts={cid for sid in sentence_refs if sid in sentence_by_id for cid in sentence_by_id[sid].get('concept_ids',[])}
            if any(cid not in evidence_concepts for cid in concept_refs): errors.append(f"quiz concept is not linked to its evidence sentences: {qid}")
        if quiz_kind=='checkpoint' and checkpoint_id in checkpoint_ids:
            cp=next(c for c in checkpoints if c['id']==checkpoint_id)
            for sid in sentence_refs:
                if sid in sentence_by_id:
                    times=sentence_by_id[sid].get('cue_end_times',[]); end=max(times) if times else float('inf')
                    if end>cp['trigger_time'] or end<=lower_bounds[checkpoint_id]: errors.append(f"quiz uses future/out-of-interval sentence: {qid}")
                    if not sentence_by_id[sid].get('text','').rstrip().endswith(('.', '?', '!', '…')): errors.append(f"quiz uses incomplete sentence: {qid}")
        elif quiz_kind=='final':
            for sid in sentence_refs:
                if sid in sentence_by_id:
                    times=sentence_by_id[sid].get('cue_end_times',[]); end=max(times) if times else float('inf')
                    if end>duration: errors.append(f"final quiz uses sentence beyond video duration: {qid}")
                    if not sentence_by_id[sid].get('text','').rstrip().endswith(('.', '?', '!', '…')): errors.append(f"final quiz uses incomplete sentence: {qid}")
    for cp in checkpoints:
        if any(qid not in qids for qid in cp['question_ids']): errors.append(f"unknown question reference: {cp['id']}")
        actual={q['id'] for q in questions if q.get('checkpoint_id')==cp['id']}
        if set(cp['question_ids'])!=actual: errors.append(f"checkpoint question references are inconsistent: {cp['id']}")
    if errors: raise ValidationError('\n'.join(errors))
    return errors
