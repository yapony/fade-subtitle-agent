"""Orchestrates AI and deterministic stages, persisted state, logs and ZIP export."""
from __future__ import annotations
import json, re, time, uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass
from .subtitles import parse_subtitles, batches, clip_cues_to_duration
from .gemini import Gemini, load_prompt
from .checkpoints import make_checkpoints, final_question_count
from .validation import validate
from .linguistic import CATEGORIES, WORD_CATEGORIES, DIFFICULTY_RUBRIC, MAX_WORD_OCCURRENCES_PER_SENTENCE, MIN_ANNOTATED_SENTENCE_RATIO, MAX_ANNOTATED_SENTENCE_RATIO, normalize, merge_concept, map_occurrences, cue_occurrence_ids, exclude_zero_occurrence_concepts, normalize_occurrence_mappings, limit_word_occurrences, annotated_sentence_ids, limit_annotated_sentence_coverage
from .packaging import write_package_zip
from .translations import cue_translation_warnings, normalize_translation_alignments
from .transcript_api import fetch_transcript, is_youtube_input

def _schema(fields): return {'type':'OBJECT','properties':{k:v for k,v in fields.items()},'required':list(fields)}
def _json(obj): return json.dumps(obj,ensure_ascii=False)

QUESTION_TYPES=('content_comprehension','vocabulary_recognition','expression_recognition','grammar_understanding')

def _quiz_item_schema():
    return {'type':'OBJECT','properties':{
        'question_type':{'type':'STRING','enum':list(QUESTION_TYPES)},'category':{'type':'STRING'},'prompt':{'type':'STRING'},
        'options':{'type':'ARRAY','items':{'type':'STRING'}},'correct_answer_index':{'type':'INTEGER'},
        'explanation':{'type':'STRING'},'sentence_ids':{'type':'ARRAY','items':{'type':'STRING'}},
        'concept_lemma':{'type':'STRING'},'concept_meaning':{'type':'STRING'},
        'evidence_sentence_id':{'type':'STRING'}},
        'required':['question_type','category','prompt','options','correct_answer_index','explanation','sentence_ids','concept_lemma','concept_meaning','evidence_sentence_id']}

def _concept_matches_hint(concept: dict, lemma_hint: str, meaning_hint: str) -> bool:
    lemma_hint=normalize(lemma_hint); meaning_hint=normalize(meaning_hint)
    lemma_match=not lemma_hint or lemma_hint==normalize(concept.get('lemma',''))
    meaning_values={normalize(concept.get('meaning_key','')),normalize(concept.get('translation',''))}
    meaning_match=not meaning_hint or meaning_hint in meaning_values
    return lemma_match and meaning_match

def _generate_quiz(ai, eligible, concepts, quiz_id, count, id_start, kind, time_range, quiz_warnings=None):
    quiz_warnings=quiz_warnings if quiz_warnings is not None else []
    evidence_ids={sentence['id'] for sentence in eligible}
    concepts_in_scope={concept_id for sentence in eligible for concept_id in sentence.get('concept_ids',[])}
    relevant_concepts=[{key:concepts[cid].get(key,'') for key in ('lemma','meaning_key','category','difficulty','translation','explanation')}
                       for cid in sorted(concepts_in_scope) if cid in concepts]
    ratio={QUESTION_TYPES[0]:count*4//10,QUESTION_TYPES[1]:count//5,QUESTION_TYPES[2]:count//5,QUESTION_TYPES[3]:count//5}
    schema=_schema({'items':{'type':'ARRAY','items':_quiz_item_schema()}})
    sentence_by_id={sentence['id']:sentence for sentence in eligible}
    evidence_for_ai=[{key:value for key,value in sentence.items() if key!='concept_ids'} for sentence in eligible]
    base=(load_prompt('quiz')+f"\nQuiz kind: {kind}. Quiz ID: {quiz_id}. Required question count: {count}. "
          f"Default question type counts: {_json(ratio)}. If this evidence lacks suitable vocabulary, expressions, or grammar, "
          "redistribute those questions among suitable types; never invent language concepts. "
          f"Evidence time range: {time_range}. Completed evidence only: {_json(evidence_for_ai)}\nRelevant linguistic concepts (without IDs): {_json(relevant_concepts)}")
    bad=[]; valid=[]; requested=count
    for attempt in range(1,4):
        correction=(f"\nRepair only these {requested} invalid or missing question slots. Do not rewrite valid questions. "
                    f"Invalid question details, including the original item, cited sentence text, and reason: {_json(bad)}. "
                    f"Return exactly {requested} replacement questions. The model must return evidence_sentence_id, not an evidence quote." if attempt>1 else '')
        result=ai.generate('quiz' if attempt==1 else 'quiz_repair',base+correction,schema)
        items=result.get('items',[]); next_bad=[]
        if len(items)!=requested:
            next_bad.append({'problem':f'expected exactly {requested} questions, received {len(items)}', 'original_item': None})
        for index,q in enumerate(items):
            required=('question_type','category','prompt','options','correct_answer_index','explanation','sentence_ids','concept_lemma','concept_meaning','evidence_sentence_id')
            missing=[field for field in required if field not in q]
            options=q.get('options'); correct=q.get('correct_answer_index')
            if missing or q.get('question_type') not in QUESTION_TYPES or not isinstance(options,list) or len(options)!=4 or not isinstance(correct,int) or isinstance(correct,bool) or not 0<=correct<len(options):
                next_bad.append({'item':index,'original_item':q,'missing':missing,'problem':'required fields, question type, or four-option answer invalid'}); continue
            if not isinstance(q.get('sentence_ids'),list) or not q['sentence_ids'] or any(sid not in evidence_ids for sid in q['sentence_ids']):
                next_bad.append({'item':index,'original_item':q,'problem':'sentence_ids must cite only supplied evidence'}); continue
            evidence_sentence_id=q.get('evidence_sentence_id')
            if evidence_sentence_id not in q['sentence_ids'] or evidence_sentence_id not in sentence_by_id:
                next_bad.append({'item':index,'original_item':q,'cited_sentences':[{'id':sid,'text':sentence_by_id[sid]['text']} for sid in q['sentence_ids'] if sid in sentence_by_id], 'problem':'evidence_sentence_id must be one of the cited supplied sentence IDs'}); continue
            if any(not isinstance(option,str) or not option.strip() for option in options) or len({option.strip().casefold() for option in options if isinstance(option,str)})!=4:
                next_bad.append({'item':index,'original_item':q,'problem':'options must be nonempty and unique'}); continue
            if len({bool(re.search(r'[.!?…][\"”’)]*$',option.strip())) for option in options})>1:
                next_bad.append({'item':index,'original_item':q,'problem':'options must use a consistent sentence/fragment format'}); continue
            if not isinstance(q.get('concept_lemma'),str) or not isinstance(q.get('concept_meaning'),str):
                next_bad.append({'item':index,'original_item':q,'problem':'concept_lemma and concept_meaning must be strings'}); continue
            cited_concepts={cid for sid in q['sentence_ids'] for cid in sentence_by_id[sid].get('concept_ids',[]) if cid in concepts}
            resolved_concept_ids=[]
            if q['question_type']!='content_comprehension':
                candidates=[] if not (normalize(q['concept_lemma']) or normalize(q['concept_meaning'])) else [
                    cid for cid in sorted(cited_concepts) if _concept_matches_hint(concepts[cid],q['concept_lemma'],q['concept_meaning'])]
                if len(candidates)==1:
                    resolved_concept_ids=candidates
                else:
                    question_id=f'q{id_start+len(valid):05d}'
                    quiz_warnings.append({'reason':'quiz_concept_resolution_failed','question_id':question_id,'question_type':q['question_type'],'sentence_ids':q['sentence_ids'],'concept_lemma':q['concept_lemma'],'concept_meaning':q['concept_meaning'],'candidate_concept_ids':candidates,'detail':'no unique concept matched the AI hint; human review required'})
            q['evidence']=sentence_by_id[evidence_sentence_id]['text']
            q['concept_ids']=resolved_concept_ids
            q['id']=f'q{id_start+len(valid):05d}'
            valid.append(q)
        if not next_bad and len(valid)==count:
            return valid
        bad=next_bad; requested=len(bad)
        if not requested: requested=count-len(valid)
    raise ValueError(f"Gemini returned invalid {kind} quiz {quiz_id} after 3 repair attempts: {_json(bad)}")

class RunLog:
    def __init__(self, root: Path, run_id: str):
        self.run_id=run_id; self.started=time.monotonic(); self.root=root/'runs'/run_id/'logs'; self.root.mkdir(parents=True,exist_ok=True); self.events=self.root/'events.jsonl'; self.requests=[]
        if self.events.exists():
            for line in self.events.read_text(encoding='utf-8').splitlines():
                try: self.requests.append(json.loads(line))
                except json.JSONDecodeError: pass
    def event(self,stage,status,**kw):
        row={'timestamp':datetime.now(timezone.utc).isoformat(),'run_id':self.run_id,'stage':stage,'status':status,**kw}
        with self.events.open('a',encoding='utf-8') as f: f.write(_json(row)+'\n'); f.flush()
        self.requests.append(row)
    def finish(self, status, output=None):
        models=defaultdict(lambda:{'requests':0,'successful_attempts':0,'failed_attempts':0,'retry_count':0,'input_tokens':0,'output_tokens':0,'total_tokens':0})
        stages=defaultdict(lambda:{'requests':0,'successful_attempts':0,'failed_attempts':0,'retry_count':0,'input_tokens':0,'output_tokens':0,'total_tokens':0})
        failed=retry=0
        for e in self.requests:
            if 'model' not in e: continue
            buckets=(models[e['model']],stages[e['stage']])
            if e['status']=='started':
                for x in buckets: x['requests']+=1
            if e['status']=='completed':
                for x in buckets:
                    x['successful_attempts']+=1
                    usage=e.get('usage',{})
                    for k in ('input_tokens','output_tokens','total_tokens'):
                        if usage.get(k) is not None: x[k]+=usage[k]
            if e['status']=='failed':
                failed+=1
                for x in buckets: x['failed_attempts']+=1
            if e['status']=='retrying':
                retry+=1
                for x in buckets: x['retry_count']+=1
        summary={'run_id':self.run_id,'elapsed_seconds':round(time.monotonic()-self.started,2),'models':dict(models),'stages':dict(stages),'failed_attempts':failed,'retry_count':retry,'cost_estimate':None}
        annotation_events=[e for e in self.requests if e.get('stage')=='linguistic_annotation' and e.get('status')=='completed']
        annotation_summary=annotation_events[-1] if annotation_events else {}
        translation_events=[e for e in self.requests if e.get('stage')=='cue_translation_validation' and e.get('status')=='completed']
        translation_summary=translation_events[-1] if translation_events else {}
        alignment_events=[e for e in self.requests if e.get('stage')=='translation_alignment' and e.get('status')=='completed']
        alignment_summary=alignment_events[-1] if alignment_events else {}
        summary['annotation_summary']={k:annotation_summary[k] for k in ('concept_count','occurrence_count','category_distribution','difficulty_distribution','word_difficulty_distribution','word_annotated_sentence_count','word_occurrence_count','word_sentence_cap','word_cap_warning_count','annotated_sentence_count','semantic_sentence_count','annotation_coverage_ratio','annotation_coverage_target','annotation_coverage_cap','coverage_review_attempted','coverage_cap_warning_count','coverage_below_target','repeated_concepts','unmatched_count','alternative_form_mismatch_count','zero_occurrence_count','ambiguous_count') if k in annotation_summary}
        if translation_summary: summary['annotation_summary']['translation_warning_count']=translation_summary.get('translation_warning_count',0)
        if alignment_summary:
            summary['annotation_summary']['translation_alignment_count']=alignment_summary.get('alignment_count',0)
            summary['annotation_summary']['translation_alignment_warning_count']=alignment_summary.get('alignment_warning_count',0)
        (self.root/'usage_summary.json').write_text(_json(summary)+'\n',encoding='utf-8')
        annotation_md=''
        if annotation_summary:
            annotation_md=(f"\n## Linguistic annotations\n\n- Published concepts: {annotation_summary.get('concept_count',0)}\n- Cue-local occurrences: {annotation_summary.get('occurrence_count',0)}\n- Repeated concepts: {annotation_summary.get('repeated_concepts',0)}\n- By category: `{_json(annotation_summary.get('category_distribution',{}))}`\n- By difficulty: `{_json(annotation_summary.get('difficulty_distribution',{}))}`\n- Genuinely unmatched contexts: {annotation_summary.get('unmatched_count',0)}\n- Alternative form mismatches: {annotation_summary.get('alternative_form_mismatch_count',0)}\n- Zero-occurrence concepts not published: {annotation_summary.get('zero_occurrence_count',0)}\n- Cue translation warnings: {annotation_summary.get('translation_warning_count',0)}\n- Translation alignment spans: {alignment_summary.get('alignment_count',0)}\n- Translation alignment warnings: {alignment_summary.get('alignment_warning_count',0)}\n- Ambiguity warnings: {annotation_summary.get('ambiguous_count',0)}\n")
        if annotation_summary:
            annotation_md += (f"- Word difficulty distribution: `{_json(annotation_summary.get('word_difficulty_distribution',{}))}`\n"
                              f"- Semantic sentences with word annotations: {annotation_summary.get('word_annotated_sentence_count',0)}\n"
                              f"- Word occurrences: {annotation_summary.get('word_occurrence_count',0)}\n"
                              f"- Word occurrence cap per semantic sentence: {annotation_summary.get('word_sentence_cap',MAX_WORD_OCCURRENCES_PER_SENTENCE)}\n"
                              f"- Word candidates excluded by sentence cap: {annotation_summary.get('word_cap_warning_count',0)}\n")
            annotation_md += (f"- Annotated semantic sentence coverage: {annotation_summary.get('annotated_sentence_count',0)}/{annotation_summary.get('semantic_sentence_count',0)} ({annotation_summary.get('annotation_coverage_ratio',0):.1%})\n"
                              f"- Coverage goal / cap: {annotation_summary.get('annotation_coverage_target',MIN_ANNOTATED_SENTENCE_RATIO):.0%} / {annotation_summary.get('annotation_coverage_cap',MAX_ANNOTATED_SENTENCE_RATIO):.0%}\n"
                              f"- Supplemental coverage review: {'attempted' if annotation_summary.get('coverage_review_attempted') else 'not needed'}\n"
                              f"- Sentences excluded by coverage cap: {annotation_summary.get('coverage_cap_warning_count',0)}\n"
                              f"- Below soft coverage goal: {'yes' if annotation_summary.get('coverage_below_target') else 'no'}\n")
        (self.root/'run-report.md').write_text(f"# Run report\n\n- Run: `{self.run_id}`\n- Status: {status}\n- Elapsed: {summary['elapsed_seconds']} seconds\n- Gemini failed attempts: {failed}\n- Retries: {retry}\n- Output: {output or 'not created'}\n"+annotation_md,encoding='utf-8')

class Agent:
    def __init__(self, output_dir: str|Path, batch_size=50, overlap=3, client=None):
        self.out=Path(output_dir); self.out.mkdir(parents=True,exist_ok=True); self.batch_size=batch_size; self.overlap=overlap; self.client=client
    def run(self, subtitle_file: str|Path, source_language: str|None, target_language: str, video_id: str|None=None, title=None, context=None, glossary=None, resume=True, run_id=None, language: str|None=None):
        load_dotenv()
        input_value=str(subtitle_file)
        if is_youtube_input(input_value):
            transcript=fetch_transcript(input_value,language)
            cues=transcript['cues']; source_language=transcript['language']; video_id=transcript['video_id']
            duration_s=transcript['duration']; title=title or transcript['title']
            input_metadata={key:transcript[key] for key in ('source','language','requested_language','metadata')}
            input_metadata['youtube_video_id']=video_id
        else:
            cues=parse_subtitles(subtitle_file)
            if not source_language:
                raise ValueError('--source-language is required for SRT/VTT input')
            if not video_id:
                raise ValueError('--video-id is required for SRT/VTT input')
            duration_s=max(cue.end for cue in cues)
            input_metadata={'source':'subtitle_file','language':source_language,'duration_source':'subtitle_end'}
        if duration_s<=0: raise ValueError('Video duration must be positive')
        cues,timing_adjustments=clip_cues_to_duration(cues,duration_s)
        if timing_adjustments:
            input_metadata['timing_adjustment']={
                'type':'cue_ends_clipped_to_video_duration',
                'cues':timing_adjustments,
            }
        run_id=run_id or uuid.uuid4().hex[:12]; work=self.out/'runs'/run_id; work.mkdir(parents=True,exist_ok=True)
        if input_metadata['source']=='transcriptapi':
            source_dir=work/'input'; source_dir.mkdir(parents=True,exist_ok=True)
            (source_dir/'transcriptapi-response.json').write_bytes(transcript['raw_response'])
        log=RunLog(self.out,run_id); ai=self.client or Gemini(event=log.event)
        state=work/'state.json'; cache=work/'intermediate'; cache.mkdir(exist_ok=True)
        config={'source_file':input_value if is_youtube_input(input_value) else str(Path(subtitle_file).resolve()),'source_language':source_language,'target_language':target_language,'video_id':video_id,'duration':duration_s,'input_metadata':input_metadata,'output_schema_version':'2.0.0'}
        progress={'run_id':run_id,'status':'running',**config,'last_completed_stage':None,'artifacts':{}}
        if resume and state.exists():
            old=json.loads(state.read_text(encoding='utf-8'))
            if old.get('output_schema_version')!=config['output_schema_version']:
                raise ValueError('This saved run uses an incompatible output schema. Start a fresh run without --resume.')
            if any(old.get(k)!=v for k,v in config.items()): raise ValueError('Resume configuration does not match the saved run state')
            done=old.get('artifacts',{}).get('zip')
            if old.get('status')=='completed' and done and Path(done).exists(): return Path(done)
            progress.update(old); progress['status']='running'
        state.write_text(_json(progress),encoding='utf-8')
        log.event('import','completed',cue_count=len(cues),timing_adjustment=input_metadata.get('timing_adjustment'))
        try:
            all_sent=[]; all_culture=[]; concepts={}; concept_mappings=[]; annotation_warnings=[]; cue_by_id={c.id:c for c in cues}; serial=0
            sentence_cache=cache/'sentences.json'
            if resume and sentence_cache.exists():
                all_sent=json.loads(sentence_cache.read_text(encoding='utf-8')); serial=len(all_sent)
            else:
                for bi,(group,primary) in enumerate(batches(cues,self.batch_size,self.overlap),1):
                    bid=f'b{bi:04d}'; group_data=[{'id':c.id,'sequence':c.sequence,'start':c.start,'end':c.end,'text':c.text} for c in group]
                    log.event('batch','started',batch_id=bid,cue_count=len(group))
                    res=ai.generate('reconstruct',load_prompt('reconstruct')+f"\nSource language: {source_language}\nContext: {context or ''}\nCues:\n"+_json(group_data),_schema({'items':{'type':'ARRAY','items':{'type':'OBJECT','properties':{'text':{'type':'STRING'},'cue_ids':{'type':'ARRAY','items':{'type':'STRING'}}},'required':['text','cue_ids']}}}))
                    for row in res['items']:
                        ids=[str(x) for x in row.get('cue_ids',[]) if str(x) in cue_by_id]
                        if not ids or not set(ids)&primary: continue
                        key=(row.get('text','').strip(),tuple(ids))
                        if any((s['text'],tuple(s['cue_ids']))==key for s in all_sent): continue
                        serial+=1; all_sent.append({'id':f's{serial:05d}','text':row['text'].strip(),'cue_ids':ids,'cue_end_times':[cue_by_id[x].end for x in ids],'translation':'','concept_ids':[]})
                    log.event('batch','completed',batch_id=bid,sentences_completed=len(all_sent))
            if not all_sent: raise ValueError('Gemini did not reconstruct any semantic sentences')
            sentence_cache.write_text(_json(all_sent),encoding='utf-8')
            progress['last_completed_stage']='reconstruct'; state.write_text(_json(progress),encoding='utf-8')
            linguistic_cache=cache/'linguistic-v2.json'
            if resume and linguistic_cache.exists():
                saved=json.loads(linguistic_cache.read_text(encoding='utf-8'))
                all_sent=saved['sentences']; concepts=saved['concepts']; concept_mappings=saved['mappings']; all_culture=saved['culture']; annotation_warnings=saved.get('warnings',[])
                for concept in concepts.values():
                    concept.setdefault('selection_priority','medium')
                    concept.setdefault('selection_reason','')
            else:
                for group,_ in batches([type('X',(),{'id':s['id']}) for s in all_sent],self.batch_size,0):
                    batch_ids={x.id for x in group}; sent=[s for s in all_sent if s['id'] in batch_ids]
                    trans=ai.generate('translate',load_prompt('translate')+f"\nLanguages: {source_language} to {target_language}\nGlossary: {_json(glossary or {})}\nSentences: {_json(sent)}",_schema({'items':{'type':'ARRAY','items':{'type':'OBJECT','properties':{'sentence_id':{'type':'STRING'},'translation':{'type':'STRING'}},'required':['sentence_id','translation']}}}))
                    tm={x['sentence_id']:x['translation'] for x in trans['items']}
                    for s in sent:
                        if s['id'] not in tm: raise ValueError(f"Missing translation for {s['id']}")
                        s['translation']=tm[s['id']]
                    known=[{k:c[k] for k in ('concept_id','lemma','meaning_key','category','difficulty','translation','explanation')} | {'selection_priority':c.get('selection_priority','medium'),'selection_reason':c.get('selection_reason','')} for c in concepts.values()]
                    concept_schema=_schema({'concepts':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                        'lemma':{'type':'STRING'},'meaning_key':{'type':'STRING'},'category':{'type':'STRING','enum':sorted(CATEGORIES)},
                        'difficulty':{'type':'INTEGER'},'translation':{'type':'STRING'},'explanation':{'type':'STRING'},
                        'selection_priority':{'type':'STRING','enum':['high','medium','low']},'selection_reason':{'type':'STRING'}},
                        'required':['lemma','meaning_key','category','difficulty','translation','explanation','selection_priority','selection_reason']}},
                        'culture':{'type':'ARRAY','items':{'type':'OBJECT','properties':{'topic':{'type':'STRING'},'explanation':{'type':'STRING'},'sentence_ids':{'type':'ARRAY','items':{'type':'STRING'}},'confidence':{'type':'NUMBER'},'needs_review':{'type':'BOOLEAN'},'verification_note':{'type':'STRING'}},'required':['topic','explanation','sentence_ids','confidence','needs_review','verification_note']}}})
                    prompt=load_prompt('analyze',source_language)+f"\nLanguages: {source_language} to {target_language}\nContext: {context or ''}\nKnown concepts already defined: {_json(known)}\nCurrent semantic sentences: {_json(sent)}"
                    analyzed=ai.generate('concept_analysis',prompt,concept_schema)
                    for raw in analyzed.get('concepts',[]):
                        _,problem=merge_concept(concepts,raw)
                        if problem: annotation_warnings.append(problem)
                    for culture in analyzed.get('culture',[]):
                        culture.setdefault('id',f'culture{len(all_culture)+1:04d}'); culture.setdefault('confidence',0.7)
                        culture.setdefault('needs_review',culture['confidence']<0.7); culture.setdefault('verification_note','')
                        if culture['needs_review'] and not culture['verification_note']:
                            culture['verification_note']='AI uncertainty; verify this cultural explanation against a reliable source.'
                        all_culture.append(culture)
                # Second pass maps all concepts over every batch; only occurrence links/forms are generated here.
                occurrence_schema=_schema({'occurrences':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                    'concept_id':{'type':'STRING'},'surface_forms':{'type':'ARRAY','items':{'type':'STRING'}},
                    'sentence_id':{'type':'STRING'}},'required':['concept_id','surface_forms','sentence_id']}},
                    'unmatched':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                        'concept_id':{'type':'STRING'},'sentence_id':{'type':'STRING'},'surface_form':{'type':'STRING'},'reason':{'type':'STRING'}},
                        'required':['concept_id','sentence_id','surface_form','reason']}}})
                concept_list=list(concepts.values())
                for group,_ in batches([type('X',(),{'id':s['id']}) for s in all_sent],self.batch_size,0):
                    batch_ids={x.id for x in group}; sent=[s for s in all_sent if s['id'] in batch_ids]
                    evidence=[{**s,'cue_texts':[{'cue_id':cid,'text':cue_by_id[cid].text} for cid in s['cue_ids'] if cid in cue_by_id]} for s in sent]
                    occurrence_prompt=load_prompt('occurrences',source_language)+f"\nConcepts: {_json(concept_list)}\nSentences and original cue texts: {_json(evidence)}"
                    mapped=ai.generate('occurrence_mapping',occurrence_prompt,occurrence_schema)
                    normalized,normalization_warnings=normalize_occurrence_mappings(mapped.get('occurrences',[]),set(concepts),batch_ids)
                    concept_mappings.extend(normalized); annotation_warnings.extend(normalization_warnings)
                    for item in mapped.get('unmatched',[]):
                        annotation_warnings.append({**item,'reason':'model_unmatched_occurrence','model_reason':item.get('reason')})
                linguistic_cache.write_text(_json({'sentences':all_sent,'concepts':concepts,'mappings':concept_mappings,'culture':all_culture,'warnings':annotation_warnings}),encoding='utf-8')
            for item in all_culture:
                if 'needs_review' not in item: item['needs_review']=True
                if item['needs_review'] and not item.get('verification_note'):
                    item['verification_note']='Legacy or unverified cultural note; verify against a reliable source.'
            occurrences,locator_warnings,_,_=map_occurrences(concepts,concept_mappings,cues,all_sent,source_language=source_language)
            occurrences,word_cap_warnings=limit_word_occurrences(occurrences,concepts,all_sent,MAX_WORD_OCCURRENCES_PER_SENTENCE)
            coverage_review_warnings=[]
            coverage_before_review=annotated_sentence_ids(occurrences,all_sent)
            coverage_review_attempted=False
            if all_sent and len(coverage_before_review)/len(all_sent)<MIN_ANNOTATED_SENTENCE_RATIO:
                coverage_review_attempted=True
                uncovered=[sentence for sentence in all_sent if sentence['id'] not in coverage_before_review]
                coverage_concept_schema=_schema({'concepts':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                    'lemma':{'type':'STRING'},'meaning_key':{'type':'STRING'},'category':{'type':'STRING','enum':sorted(CATEGORIES)},
                    'difficulty':{'type':'INTEGER'},'translation':{'type':'STRING'},'explanation':{'type':'STRING'},
                    'selection_priority':{'type':'STRING','enum':['high','medium','low']},'selection_reason':{'type':'STRING'}},
                    'required':['lemma','meaning_key','category','difficulty','translation','explanation','selection_priority','selection_reason']}}})
                known=[{key:value for key,value in concept.items() if key in ('concept_id','lemma','meaning_key','category','difficulty','translation','explanation','selection_priority','selection_reason')} for concept in concepts.values()]
                for start in range(0,len(uncovered),self.batch_size):
                    group=uncovered[start:start+self.batch_size]
                    evidence=[{**sentence,'cue_texts':[{'cue_id':cid,'text':cue_by_id[cid].text} for cid in sentence.get('cue_ids',[]) if cid in cue_by_id]} for sentence in group]
                    prompt=(load_prompt('coverage_review',source_language)+f"\nLanguages: {source_language} to {target_language}\nContext: {context or ''}"
                            f"\nKnown concepts: {_json(known)}\nUnannotated semantic sentences and cue texts: {_json(evidence)}")
                    supplemental=ai.generate('concept_coverage_review',prompt,coverage_concept_schema)
                    for raw in supplemental.get('concepts',[]):
                        _,problem=merge_concept(concepts,raw)
                        if problem: coverage_review_warnings.append(problem)
                occurrence_schema=_schema({'occurrences':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                    'concept_id':{'type':'STRING'},'surface_forms':{'type':'ARRAY','items':{'type':'STRING'}},'sentence_id':{'type':'STRING'}},
                    'required':['concept_id','surface_forms','sentence_id']}},
                    'unmatched':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                        'concept_id':{'type':'STRING'},'sentence_id':{'type':'STRING'},'surface_form':{'type':'STRING'},'reason':{'type':'STRING'}},
                        'required':['concept_id','sentence_id','surface_form','reason']}}})
                for start in range(0,len(uncovered),self.batch_size):
                    group=uncovered[start:start+self.batch_size]
                    batch_ids={sentence['id'] for sentence in group}
                    evidence=[{**sentence,'cue_texts':[{'cue_id':cid,'text':cue_by_id[cid].text} for cid in sentence.get('cue_ids',[]) if cid in cue_by_id]} for sentence in group]
                    prompt=load_prompt('occurrences',source_language)+f"\nConcepts: {_json(list(concepts.values()))}\nSentences and original cue texts: {_json(evidence)}"
                    mapped=ai.generate('coverage_occurrence_mapping',prompt,occurrence_schema)
                    normalized,normalization_warnings=normalize_occurrence_mappings(mapped.get('occurrences',[]),set(concepts),batch_ids)
                    concept_mappings.extend(normalized); coverage_review_warnings.extend(normalization_warnings)
                    for item in mapped.get('unmatched',[]):
                        coverage_review_warnings.append({**item,'reason':'model_unmatched_occurrence','model_reason':item.get('reason')})
                occurrences,locator_warnings,_,_=map_occurrences(concepts,concept_mappings,cues,all_sent,source_language=source_language)
                occurrences,word_cap_warnings=limit_word_occurrences(occurrences,concepts,all_sent,MAX_WORD_OCCURRENCES_PER_SENTENCE)
                linguistic_cache.write_text(_json({'sentences':all_sent,'concepts':concepts,'mappings':concept_mappings,'culture':all_culture,'warnings':annotation_warnings+coverage_review_warnings}),encoding='utf-8')

            occurrences,coverage_cap_warnings,coverage_stats=limit_annotated_sentence_coverage(
                occurrences,concepts,all_sent,MAX_ANNOTATED_SENTENCE_RATIO)
            annotation_warnings.extend(locator_warnings)
            annotation_warnings.extend(word_cap_warnings)
            annotation_warnings.extend(coverage_review_warnings)
            annotation_warnings.extend(coverage_cap_warnings)
            annotation_ratio=(coverage_stats['annotated_sentence_count']/coverage_stats['sentence_count']) if coverage_stats['sentence_count'] else 0
            coverage_ratio_below_target=annotation_ratio < MIN_ANNOTATED_SENTENCE_RATIO
            if coverage_ratio_below_target:
                annotation_warnings.append({'reason':'annotation_coverage_below_soft_target','annotated_sentence_count':coverage_stats['annotated_sentence_count'],
                                            'sentence_count':coverage_stats['sentence_count'],'coverage_ratio':round(annotation_ratio,4),
                                            'target_ratio':MIN_ANNOTATED_SENTENCE_RATIO,'coverage_review_attempted':coverage_review_attempted})
            frequency_sets=defaultdict(set); sentence_concept_sets=defaultdict(set)
            for occurrence in occurrences:
                concept_id=occurrence['concept_id']
                frequency_sets[concept_id].add(occurrence.get('instance_id',occurrence['occurrence_id']))
                for sentence_id in occurrence.get('sentence_ids',[]): sentence_concept_sets[sentence_id].add(concept_id)
            frequencies={concept_id:len(instance_ids) for concept_id,instance_ids in frequency_sets.items()}
            sentence_concepts={sentence_id:sorted(concept_ids) for sentence_id,concept_ids in sentence_concept_sets.items()}
            for concept_id,concept in concepts.items():
                linked=[o for o in occurrences if o['concept_id']==concept_id]
                concept['frequency']=frequencies.get(concept_id,0)
                concept['occurrence_ids']=sorted(o['occurrence_id'] for o in linked)
                concept['surface_forms']=sorted({o['text'] for o in linked},key=str.casefold)
            concepts,zero_warnings=exclude_zero_occurrence_concepts(concepts)
            zero_occurrence_ids=[w['concept_id'] for w in zero_warnings]
            annotation_warnings.extend(zero_warnings)
            sentence_concepts={sid:sorted(ids&concepts.keys()) for sid,ids in sentence_concepts.items()}
            for sentence in all_sent: sentence['concept_ids']=sentence_concepts.get(sentence['id'],[])
            cue_refs=cue_occurrence_ids(cues,occurrences)
            cue_rows=[]
            for c in cues:
                refs=[s['id'] for s in all_sent if c.id in s['cue_ids']]
                cue_rows.append({'id':c.id,'sequence':c.sequence,'start':c.start,'duration':c.duration,'end':c.end,'source_text':c.text,'translation':'','sentence_ids':refs,'linguistic_occurrence_ids':cue_refs[c.id],'translation_alignment_ids':[]})
            categories=dict(Counter(c['category'] for c in concepts.values()))
            difficulties={str(k):sum(c['difficulty']==k for c in concepts.values()) for k in DIFFICULTY_RUBRIC}
            word_difficulties={str(k):sum(c['difficulty']==k and c['category'] in WORD_CATEGORIES for c in concepts.values()) for k in DIFFICULTY_RUBRIC}
            word_occurrences=[o for o in occurrences if concepts.get(o['concept_id'],{}).get('category') in WORD_CATEGORIES]
            word_sentence_ids={sid for occurrence in word_occurrences for sid in occurrence.get('sentence_ids',[])}
            ambiguous=sum(w.get('reason')=='same_span_multiple_concepts' for w in annotation_warnings)
            unmatched=sum(w.get('reason') in {'unmatched_occurrence','model_unmatched_occurrence','incomplete_occurrence_mapping'} for w in annotation_warnings)
            alternative_mismatches=sum(w.get('reason')=='alternative_surface_mismatch' for w in annotation_warnings)
            log.event('linguistic_annotation','completed',concept_count=len(concepts),occurrence_count=len(occurrences),category_distribution=categories,difficulty_distribution=difficulties,word_difficulty_distribution=word_difficulties,word_annotated_sentence_count=len(word_sentence_ids),word_occurrence_count=len(word_occurrences),word_sentence_cap=MAX_WORD_OCCURRENCES_PER_SENTENCE,word_cap_warning_count=len(word_cap_warnings),annotated_sentence_count=coverage_stats['annotated_sentence_count'],semantic_sentence_count=coverage_stats['sentence_count'],annotation_coverage_ratio=round(annotation_ratio,4),annotation_coverage_target=MIN_ANNOTATED_SENTENCE_RATIO,annotation_coverage_cap=MAX_ANNOTATED_SENTENCE_RATIO,coverage_review_attempted=coverage_review_attempted,coverage_cap_warning_count=len(coverage_cap_warnings),coverage_below_target=coverage_ratio_below_target,repeated_concepts=sum(c['frequency']>1 for c in concepts.values()),unmatched_count=unmatched,alternative_form_mismatch_count=alternative_mismatches,zero_occurrence_count=len(zero_occurrence_ids),ambiguous_count=ambiguous)
            # Translate cue-level lines independently to preserve cue-time alignment.
            # Smaller cue batches reduce omissions and truncated structured output.
            cue_translation_schema=_schema({'items':{'type':'ARRAY','items':{'type':'OBJECT','properties':{'cue_id':{'type':'STRING'},'translation':{'type':'STRING'}},'required':['cue_id','translation']}}})
            cue_translation_inputs={}
            for group,_ in batches(cues,min(self.batch_size,20),0):
                requested={c.id:c for c in group}
                cue_inputs=[]
                for c in group:
                    context_sentences=[{'sentence_id':s['id'],'text':s['text'],'translation':s['translation']} for s in all_sent if c.id in s['cue_ids']]
                    neighbors=[{'cue_id':n.id,'text':n.text} for n in cues if abs(n.sequence-c.sequence)==1]
                    cue_inputs.append({'cue_id':c.id,'text':c.text,'complete_sentence_context':context_sentences,'neighboring_cues':neighbors})
                    cue_translation_inputs[c.id]=cue_inputs[-1]
                prompt=load_prompt('translate_cues')+f"\nLanguages: {source_language} to {target_language}\nGlossary: {_json(glossary or {})}\nCue inputs and complete-sentence context: {_json(cue_inputs)}"
                result=ai.generate('translate_cues',prompt,cue_translation_schema)
                lookup={str(x['cue_id']):str(x['translation']).strip() for x in result.get('items',[]) if str(x.get('cue_id')) in requested and str(x.get('translation','')).strip()}
                # Repair incomplete batches by asking only for missing cues, with bounded attempts.
                for repair_attempt in range(1,4):
                    missing=[requested[cid] for cid in requested if cid not in lookup]
                    if not missing: break
                    missing_ids={c.id for c in missing}
                    missing_inputs=[item for item in cue_inputs if item['cue_id'] in missing_ids]
                    repair_prompt=load_prompt('translate_cues')+f"\nLanguages: {source_language} to {target_language}\nGlossary: {_json(glossary or {})}\nThe previous response omitted these cue IDs. Return one translation for every ID below, exactly as written: {_json(missing_inputs)}"
                    repaired=ai.generate('translate_cues_repair',repair_prompt,cue_translation_schema)
                    for item in repaired.get('items',[]):
                        cid=str(item.get('cue_id','')); translation=str(item.get('translation','')).strip()
                        if cid in requested and translation: lookup[cid]=translation
                missing=sorted(set(requested)-set(lookup))
                if missing: raise ValueError(f"Gemini omitted display cue translations after 3 repair attempts; cue IDs: {', '.join(missing[:20])}")
                for row in cue_rows:
                    if row['id'] in lookup: row['translation']=lookup[row['id']]
            translation_warnings=cue_translation_warnings(cue_rows,all_sent)
            if translation_warnings:
                warning_by_cue={item['cue_id']:item for item in translation_warnings}
                flagged=[cue_by_id[cid] for cid in warning_by_cue if cid in cue_by_id]
                for group,_ in batches(flagged,min(self.batch_size,20),0):
                    repair_items=[]
                    for cue in group:
                        cue_row=next(row for row in cue_rows if row['id']==cue.id)
                        repair_items.append({**cue_translation_inputs[cue.id],
                            'current_translation':cue_row['translation'],
                            'diagnostic':warning_by_cue[cue.id]})
                    repair_prompt=(load_prompt('translate_cues')+
                        f"\nLanguages: {source_language} to {target_language}\nGlossary: {_json(glossary or {})}\n"
                        "Review only these flagged cue translations against their complete sentence context. Correct a translation only if it adds unsupported meaning, is clearly unnatural/incomplete, or conflicts with the sentence translation. Keep the exact cue IDs and translate only each cue's source text. Return one replacement per cue: "+_json(repair_items))
                    repaired=ai.generate('translate_cues_alignment_repair',repair_prompt,cue_translation_schema)
                    for item in repaired.get('items',[]):
                        cue_id=str(item.get('cue_id','')); translated=str(item.get('translation','')).strip()
                        if cue_id in warning_by_cue and translated:
                            next(row for row in cue_rows if row['id']==cue_id)['translation']=translated
                translation_warnings=cue_translation_warnings(cue_rows,all_sent)
            log.event('cue_translation_validation','completed',translation_warning_count=len(translation_warnings),warning_cue_ids=sorted({w['cue_id'] for w in translation_warnings}))
            # Align each source occurrence to its cue-level translation after all
            # translation repairs. The model proposes semantic target text;
            # Python locates exact offsets and owns stable alignment IDs.
            alignment_schema=_schema({'alignments':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                'occurrence_id':{'type':'STRING'},'cue_id':{'type':'STRING'},
                'spans':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                    'text':{'type':'STRING'}},
                    'required':['text']}},
                'alignment_type':{'type':'STRING','enum':['direct','paraphrase','split']},
                'confidence':{'type':'NUMBER'},'requires_cross_cue':{'type':'BOOLEAN'}},
                'required':['occurrence_id','cue_id','spans','alignment_type','confidence','requires_cross_cue']}}})
            translation_alignments=[]; translation_alignment_warnings=[]
            occurrence_by_cue=defaultdict(list)
            for occurrence in occurrences:
                occurrence_by_cue[occurrence['cue_id']].append(occurrence)
            for group,_ in batches(cues,min(self.batch_size,20),0):
                inputs=[]; batch_occurrences=[]
                for cue in group:
                    cue_occurrences=occurrence_by_cue.get(cue.id,[])
                    if not cue_occurrences: continue
                    cue_row=next(row for row in cue_rows if row['id']==cue.id)
                    sentence_context=[{'sentence_id':s['id'],'source_text':s['text'],'translation':s['translation']}
                                      for s in all_sent if cue.id in s['cue_ids']]
                    inputs.append({'cue_id':cue.id,'source_text':cue.text,'translation':cue_row['translation'],
                                   'sentence_context':sentence_context,'occurrences':[
                                       {'occurrence_id':item['occurrence_id'],'text':item['text'],
                                        'start_char':item['start_char'],'end_char':item['end_char'],
                                        'concept_id':item['concept_id'],
                                        'concept_lemma':concepts.get(item['concept_id'],{}).get('lemma',''),
                                        'concept_translation':concepts.get(item['concept_id'],{}).get('translation',''),
                                        'category':concepts.get(item['concept_id'],{}).get('category',''),
                                        'difficulty':concepts.get(item['concept_id'],{}).get('difficulty')}
                                       for item in cue_occurrences]})
                    batch_occurrences.extend(cue_occurrences)
                if not inputs: continue
                prompt=(load_prompt('translation_alignment')+
                        f"\nLanguages: {source_language} to {target_language}\n"
                        f"Cue inputs and source occurrences: {_json(inputs)}")
                result=ai.generate('translation_alignment',prompt,alignment_schema)
                normalized,alignment_warnings=normalize_translation_alignments(result,inputs and [
                    {'id':item['cue_id'],'translation':item['translation']} for item in inputs],batch_occurrences)
                translation_alignments.extend(normalized)
                translation_alignment_warnings.extend(alignment_warnings)
            alignment_ids_by_cue=defaultdict(list)
            for alignment in translation_alignments:
                alignment_ids_by_cue[alignment['cue_id']].append(alignment['alignment_id'])
            for row in cue_rows:
                row['translation_alignment_ids']=sorted(alignment_ids_by_cue.get(row['id'],[]))
            log.event('translation_alignment','completed',alignment_count=len(translation_alignments),
                      aligned_occurrence_count=len({x['occurrence_id'] for x in translation_alignments}),
                      alignment_warning_count=len(translation_alignment_warnings))
            progress['last_completed_stage']='translate_analyze'; state.write_text(_json(progress),encoding='utf-8')
            # Checkpoint questions use only their completed interval; the final quiz uses the full episode.
            checkpoints=make_checkpoints(duration_s,all_sent); questions=[]; quiz_warnings=[]; previous=0.0
            for cp in checkpoints:
                eligible=[s for s in all_sent if max(s['cue_end_times'])>previous and max(s['cue_end_times'])<=cp['trigger_time'] and s['text'].rstrip().endswith(('.', '?','!','…'))]
                valid_questions=_generate_quiz(ai,eligible,concepts,cp['id'],10,len(questions)+1,'checkpoint',f"({previous}, {cp['trigger_time']}] seconds",quiz_warnings)
                for question in valid_questions:
                    question['checkpoint_id']=cp['id']
                    questions.append(question); cp['question_ids'].append(question['id'])
                previous=cp['trigger_time']
            final_evidence=[s for s in all_sent if s.get('cue_end_times') and max(s['cue_end_times'])<=duration_s and s['text'].rstrip().endswith(('.', '?','!','…'))]
            final_id='final'
            final_items=_generate_quiz(ai,final_evidence,concepts,final_id,final_question_count(duration_s),len(questions)+1,'final',f"(0, {duration_s}] seconds",quiz_warnings)
            final_quiz={'id':final_id,'trigger_time':duration_s,'question_ids':[question['id'] for question in final_items],'questions':final_items}
            culture_items=[]
            for x in all_culture:
                valid_sentences=[s for s in all_sent if s['id'] in x['sentence_ids']]
                valid_sentence_ids=[s['id'] for s in valid_sentences]
                unknown_sentence_ids=sorted(set(x['sentence_ids'])-set(valid_sentence_ids))
                if unknown_sentence_ids:
                    annotation_warnings.append({'reason':'culture_item_unknown_sentence_references','culture_id':x['id'],'sentence_ids':unknown_sentence_ids})
                valid_cues=[cue_by_id[cid] for sentence in valid_sentences for cid in sentence['cue_ids'] if cid in cue_by_id]
                if not valid_sentences or not valid_cues:
                    annotation_warnings.append({'reason':'culture_item_missing_context','culture_id':x['id']}); continue
                culture_items.append({'id':x['id'],'sentence_ids':valid_sentence_ids,'trigger_time':min(c.start for c in valid_cues),'topic':x.get('topic',''),'explanation':x.get('explanation',''),'confidence':x['confidence'],'needs_review':bool(x.get('needs_review',True)),'verification_note':x.get('verification_note','')})
            package={'manifest':{'schema_version':'2.0.0','content_id':video_id,'title':title or video_id,'video':{'platform':'youtube' if input_metadata['source']=='transcriptapi' else 'unknown','id':video_id,'duration_seconds':duration_s},'source_language':source_language,'target_language':target_language,'assessment_interval_seconds':600,'files':['manifest.json','subtitles.json','vocabulary.json','culture.json','quizzes.json','review-report.md'],'subtitle_source':input_metadata},
                'subtitles':{'schema_version':'2.0.0','source':input_metadata,'language':source_language,'display_cues':cue_rows,'sentences':all_sent,'linguistic_occurrences':occurrences,'translation_alignments':translation_alignments,'annotation_warnings':annotation_warnings,'translation_warnings':translation_warnings,'translation_alignment_warnings':translation_alignment_warnings},'vocabulary':{'schema_version':'2.0.0','concepts':list(concepts.values())},'culture':{'schema_version':'2.0.0','items':culture_items},'quizzes':{'schema_version':'2.0.0','checkpoints':checkpoints,'questions':questions,'final_quiz':final_quiz,'quiz_warnings':quiz_warnings}}
            validate(package,duration_s)
            progress['last_completed_stage']='structural_validation'; state.write_text(_json(progress),encoding='utf-8')
            review=ai.generate('quality_review',load_prompt('review')+f"\nPackage content for review: {_json(package)}",_schema({'issues':{'type':'ARRAY','items':{'type':'OBJECT','properties':{'severity':{'type':'STRING'},'path':{'type':'STRING'},'issue':{'type':'STRING'}},'required':['severity','path','issue']}}}))
            warning_lines=[f"- `{w.get('reason','warning')}`: concept `{w.get('concept_id','unknown')}`"+(f", lemma `{w['lemma']}`" if w.get('lemma') else '')+(f", sentence `{w['sentence_id']}`" if w.get('sentence_id') else '')+(f", surface `{w['surface_form']}`" if w.get('surface_form') else '')+(f", forms `{', '.join(w['surface_forms'])}`" if w.get('surface_forms') else '') for w in annotation_warnings]
            coverage_report=(f"## Annotation coverage\n\n- Annotated semantic sentences: {coverage_stats['annotated_sentence_count']}/{coverage_stats['sentence_count']} ({annotation_ratio:.1%})\n"
                             f"- Soft goal: {MIN_ANNOTATED_SENTENCE_RATIO:.0%}; upper cap: {MAX_ANNOTATED_SENTENCE_RATIO:.0%}\n"
                             f"- Supplemental review attempted: {'yes' if coverage_review_attempted else 'no'}\n"
                             f"- Sentences removed by upper cap: {len(coverage_cap_warnings)}\n"
                             f"- Below soft goal: {'yes' if coverage_ratio_below_target else 'no'}\n\n")
            translation_warning_lines=[f"- Cue `{w['cue_id']}`: `{w['reason']}`; review {w.get('translation','')!r} against {w.get('sentence_translation','')!r}." for w in translation_warnings]
            translation_alignment_warning_lines=[f"- `{w.get('reason','warning')}`: occurrence `{w.get('occurrence_id','unknown')}`"+(f", cue `{w['cue_id']}`" if w.get('cue_id') else '') for w in translation_alignment_warnings]
            quiz_warning_lines=[f"- `{w.get('reason','warning')}`: question `{w.get('question_id','unknown')}`; hint `{w.get('concept_lemma','') or w.get('concept_meaning','')}`; candidates `{', '.join(w.get('candidate_concept_ids',[])) or 'none'}`; {w.get('detail','Human review required.')}" for w in quiz_warnings]
            cultural_review=[f"- `{item['id']}`: {item['verification_note'] or 'Cultural claim needs human verification.'}" for item in culture_items if item['needs_review']]
            report=('# Review report\n\nSchema version: `2.0.0`.\n\nStructural validation: **passed**.\n\n'
                f"Linguistic annotations: {len(concepts)} published concepts, {len(occurrences)} cue-local occurrences, {sum(c['frequency']>1 for c in concepts.values())} repeated concepts. {len(zero_occurrence_ids)} zero-occurrence concepts were excluded from the vocabulary.\n\n"
                +coverage_report+'## Annotation warnings\n'+('\n'.join(warning_lines) if warning_lines else '- None.')+'\n\n'
                '## Cue translation warnings\n'+('\n'.join(translation_warning_lines) if translation_warning_lines else '- None.')+'\n\n'
                '## Translation alignment warnings\n'+('\n'.join(translation_alignment_warning_lines) if translation_alignment_warning_lines else '- None.')+'\n\n'
                '## Quiz warnings\n'+('\n'.join(quiz_warning_lines) if quiz_warning_lines else '- None.')+'\n\n'
                '## Cultural notes requiring review\n'+('\n'.join(cultural_review) if cultural_review else '- None.')+'\n\n'
                '## Semantic review\n'+('Issues found.' if review['issues'] else 'No issues reported.')+'\n\n## Semantic issues\n'+
                ('\n'.join(f"- **{x['severity']}** `{x['path']}`: {x['issue']}" for x in review['issues']) if review['issues'] else '- None reported.')+'\n')
            outdir=self.out/'outputs'; zip_path=write_package_zip(outdir,video_id,run_id,package,report)
            progress.update(status='completed',last_completed_stage='export',artifacts={'zip':str(zip_path)}); state.write_text(_json(progress),encoding='utf-8')
            log.event('export','completed',output_path=str(zip_path)); log.finish('completed',str(zip_path)); return zip_path
        except Exception as e:
            progress.update(status='failed',error=f'{type(e).__name__}: {str(e)[:500]}'); state.write_text(_json(progress),encoding='utf-8'); log.event('pipeline','failed',error_category=type(e).__name__,error=str(e)[:500]); log.finish('failed'); raise
