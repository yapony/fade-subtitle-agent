import unittest
from fade_content_agent.linguistic import concept_id_for, concept_key
from fade_content_agent.validation import validate, ValidationError

class ValidationTests(unittest.TestCase):
    def package(self):
        cid=concept_id_for(concept_key('noun','hola','greeting'))
        occurrence={'occurrence_id':'lo1','instance_id':'li1','concept_id':cid,'cue_id':'1','text':'Hola','start_char':0,'end_char':4,'sentence_ids':['s1']}
        concept={'concept_id':cid,'lemma':'hola','meaning_key':'greeting','category':'noun','frequency':1,'difficulty':1,'translation':'hello','explanation':'greeting','occurrence_ids':['lo1'],'surface_forms':['Hola']}
        cue={'id':'1','sequence':1,'start':0,'end':2,'source_text':'Hola','translation':'Hello','sentence_ids':['s1'],'linguistic_occurrence_ids':['lo1']}
        sentence={'id':'s1','cue_ids':['1'],'cue_end_times':[2],'text':'Hola.','translation':'Hello.','concept_ids':[cid]}
        question={'id':'q1','checkpoint_id':'cp1','question_type':'vocabulary_recognition','category':'vocabulary','prompt':'Meaning?','options':['hello','goodbye','table','house'],'correct_answer_index':0,'explanation':'hola means hello','sentence_ids':['s1'],'concept_ids':[cid],'evidence':'Hola.','evidence_sentence_id':'s1'}
        checkpoint_questions=[{**question,'id':f'q{i:05d}'} for i in range(1,11)]
        final_questions=[{**question,'id':f'q{i:05d}'} for i in range(11,31)]; [q.pop('checkpoint_id') for q in final_questions]
        return {'manifest':{'schema_version':'2.0.0','content_id':'v','title':'Test','video':{'id':'v','duration_seconds':1200},'source_language':'es','target_language':'en','assessment_interval_seconds':600,'files':['manifest.json','subtitles.json','vocabulary.json','culture.json','quizzes.json','review-report.md']},'subtitles':{'schema_version':'2.0.0','display_cues':[cue],'sentences':[sentence],'linguistic_occurrences':[occurrence],'annotation_warnings':[],'translation_warnings':[]},'vocabulary':{'schema_version':'2.0.0','concepts':[concept]},'culture':{'schema_version':'2.0.0','items':[]},'quizzes':{'schema_version':'2.0.0','checkpoints':[{'id':'cp1','target_time':600,'trigger_time':600,'reason':'forced_interval','interrupts_sentence':True,'is_final':False,'question_ids':[q['id'] for q in checkpoint_questions]}],'questions':checkpoint_questions,'final_quiz':{'id':'final','trigger_time':1200,'question_ids':[q['id'] for q in final_questions],'questions':final_questions}}}
    def test_valid_concept_occurrence_and_quiz_links(self): self.assertEqual(validate(self.package(),1200),[])
    def test_quiz_question_types_are_fixed_enum(self):
        p=self.package(); p['quizzes']['final_quiz']['questions'][0]['question_type']='vocabulary'
        with self.assertRaisesRegex(ValidationError,'question_type|invalid question type'):
            validate(p,1200)
    def test_final_quiz_question_count_uses_duration_formula(self):
        p=self.package(); p['quizzes']['final_quiz']['questions'].pop()
        with self.assertRaisesRegex(ValidationError,'final quiz must have 20 questions'):
            validate(p,1200)
    def test_final_quiz_appears_at_video_end(self):
        p=self.package(); self.assertEqual(validate(p,1200),[])
        p['quizzes']['final_quiz']['trigger_time']=1199
        with self.assertRaisesRegex(ValidationError,'final quiz trigger_time'):
            validate(p,1200)
    def test_declared_json_schemas_are_applied(self):
        p=self.package(); del p['manifest']['title']
        with self.assertRaisesRegex(ValidationError,'schema manifest.schema.json'):
            validate(p,1200)
    def test_invalid_unicode_offset(self):
        p=self.package(); p['subtitles']['linguistic_occurrences'][0]['end_char']=5
        with self.assertRaisesRegex(ValidationError,'offset'): validate(p,1200)
    def test_translation_alignment_links_back_to_source_and_validates_target_offset(self):
        p=self.package()
        p['subtitles']['display_cues'][0]['translation_alignment_ids']=['ta1']
        p['subtitles']['translation_alignments']=[{'alignment_id':'ta1','occurrence_id':'lo1','cue_id':'1','text':'Hello','start_char':0,'end_char':5,'alignment_type':'direct','confidence':0.95}]
        self.assertEqual(validate(p,1200),[])
        p['subtitles']['translation_alignments'][0]['end_char']=4
        with self.assertRaisesRegex(ValidationError,'translation alignment'):
            validate(p,1200)

    def test_overlapping_translation_alignments_are_rejected(self):
        p=self.package()
        p['subtitles']['display_cues'][0]['translation_alignment_ids']=['ta1','ta2']
        p['subtitles']['translation_alignments']=[
            {'alignment_id':'ta1','occurrence_id':'lo1','cue_id':'1','text':'Hello','start_char':0,'end_char':5,'alignment_type':'direct','confidence':0.95},
            {'alignment_id':'ta2','occurrence_id':'lo1','cue_id':'1','text':'ello','start_char':1,'end_char':5,'alignment_type':'direct','confidence':0.95},
        ]
        with self.assertRaisesRegex(ValidationError,'overlapping translation alignment'):
            validate(p,1200)
    def test_invalid_concept_reference(self):
        p=self.package(); p['subtitles']['linguistic_occurrences'][0]['concept_id']='missing'
        with self.assertRaisesRegex(ValidationError,'unknown concept'): validate(p,1200)
    def test_quiz_concept_must_be_linked_to_evidence_sentence(self):
        p=self.package(); p['quizzes']['questions'][0]['concept_ids']=['missing']
        with self.assertRaisesRegex(ValidationError,'unknown/duplicate quiz concept'): validate(p,1200)
    def test_quiz_concept_must_belong_to_its_cited_sentence(self):
        p=self.package(); other=concept_id_for(concept_key('verb','ser','to be'))
        p['vocabulary']['concepts'].append({'concept_id':other,'lemma':'ser','meaning_key':'to be','category':'verb','frequency':0,'difficulty':1,'translation':'be','explanation':'verb','occurrence_ids':[]})
        p['quizzes']['questions'][0]['concept_ids']=[other]
        with self.assertRaisesRegex(ValidationError,'not linked to its evidence sentences'): validate(p,1200)
    def test_zero_occurrence_concepts_cannot_be_published(self):
        p=self.package(); zero=concept_id_for(concept_key('noun','mesa','table'))
        p['vocabulary']['concepts'].append({'concept_id':zero,'lemma':'mesa','meaning_key':'table','category':'noun','frequency':0,'difficulty':1,'translation':'table','explanation':'table','occurrence_ids':[],'surface_forms':[]})
        with self.assertRaisesRegex(ValidationError,'no located occurrences'): validate(p,1200)
    def test_quiz_evidence_must_be_exactly_cited(self):
        p=self.package(); p['quizzes']['questions'][0]['evidence']='hello'
        with self.assertRaisesRegex(ValidationError,'exact substring'): validate(p,1200)
    def test_quiz_evidence_must_match_selected_evidence_sentence(self):
        p=self.package(); p['quizzes']['questions'][0]['evidence_sentence_id']='s1'; p['quizzes']['questions'][0]['evidence']='hello'
        with self.assertRaisesRegex(ValidationError,'copied from its evidence_sentence_id'): validate(p,1200)
    def test_quiz_option_sentence_fragment_mismatch_is_rejected(self):
        p=self.package(); p['quizzes']['questions'][0]['options']=['Hello.','goodbye','table','house']
        with self.assertRaisesRegex(ValidationError,'mix sentence and fragment'): validate(p,1200)
    def test_uncertain_cultural_note_requires_a_review_reason(self):
        p=self.package(); p['culture']['items']=[{'id':'culture1','sentence_ids':['s1'],'trigger_time':0,'topic':'custom','explanation':'Historical note','confidence':0.4,'needs_review':True,'verification_note':''}]
        with self.assertRaisesRegex(ValidationError,'no verification note'): validate(p,1200)
    def test_culture_trigger_time_matches_associated_subtitle(self):
        p=self.package(); p['culture']['items']=[{'id':'culture1','sentence_ids':['s1'],'trigger_time':1,'topic':'custom','explanation':'Note','confidence':0.8,'needs_review':False,'verification_note':''}]
        with self.assertRaisesRegex(ValidationError,'culture trigger_time does not match'):
            validate(p,1200)
    def test_non_numeric_culture_trigger_time_is_reported_as_validation_error(self):
        p=self.package(); p['culture']['items']=[{'id':'culture1','sentence_ids':['s1'],'trigger_time':'1','topic':'custom','explanation':'Note','confidence':0.8,'needs_review':False,'verification_note':''}]
        with self.assertRaisesRegex(ValidationError,'invalid culture trigger time'):
            validate(p,1200)
    def test_word_occurrence_cap_is_enforced_per_semantic_sentence(self):
        p=self.package()
        p['subtitles']['display_cues'][0]['source_text']='Hola hola hola'
        p['subtitles']['linguistic_occurrences'] += [
            {'occurrence_id':'lo2','instance_id':'li2','concept_id':p['vocabulary']['concepts'][0]['concept_id'],'cue_id':'1','text':'hola','start_char':5,'end_char':9,'sentence_ids':['s1']},
            {'occurrence_id':'lo3','instance_id':'li3','concept_id':p['vocabulary']['concepts'][0]['concept_id'],'cue_id':'1','text':'hola','start_char':10,'end_char':14,'sentence_ids':['s1']},
        ]
        p['subtitles']['display_cues'][0]['linguistic_occurrence_ids'] += ['lo2','lo3']
        p['vocabulary']['concepts'][0]['occurrence_ids'] += ['lo2','lo3']
        p['vocabulary']['concepts'][0]['surface_forms']=['Hola','hola']
        p['vocabulary']['concepts'][0]['frequency']=3
        with self.assertRaisesRegex(ValidationError,'too many word occurrences'):
            validate(p,1200)
    def test_cross_cue_expression_segments_validate_as_one_instance(self):
        p=self.package(); cid=p['vocabulary']['concepts'][0]['concept_id']
        p['subtitles']['display_cues']=[
            {'id':'1','sequence':1,'start':0,'end':1,'source_text':'por','translation':'why','sentence_ids':['s1'],'linguistic_occurrence_ids':['loA']},
            {'id':'2','sequence':2,'start':1,'end':2,'source_text':'[Aplausos]','translation':'[Applause]','sentence_ids':[],'linguistic_occurrence_ids':[]},
            {'id':'3','sequence':3,'start':2,'end':3,'source_text':'qué','translation':'why','sentence_ids':['s1'],'linguistic_occurrence_ids':['loB']}]
        p['subtitles']['sentences'][0]['cue_ids']=['1','3']
        p['subtitles']['linguistic_occurrences']=[
            {'occurrence_id':'loA','instance_id':'liA','concept_id':cid,'cue_id':'1','text':'por','start_char':0,'end_char':3,'sentence_ids':['s1']},
            {'occurrence_id':'loB','instance_id':'liA','concept_id':cid,'cue_id':'3','text':'qué','start_char':0,'end_char':3,'sentence_ids':['s1']}]
        p['vocabulary']['concepts'][0]['occurrence_ids']=['loA','loB']; p['vocabulary']['concepts'][0]['frequency']=1
        p['quizzes']['questions'][0]['concept_ids']=[cid]
        self.assertEqual(validate(p,1200),[])
    def test_cross_cue_expression_cannot_bridge_dialogue_gap(self):
        p=self.package(); p['subtitles']['display_cues']=[
            {'id':'1','sequence':1,'start':0,'end':1,'source_text':'por','translation':'why','sentence_ids':['s1'],'linguistic_occurrence_ids':['loA']},
            {'id':'2','sequence':2,'start':1,'end':2,'source_text':'Hola','translation':'hello','sentence_ids':[],'linguistic_occurrence_ids':[]},
            {'id':'3','sequence':3,'start':2,'end':3,'source_text':'qué','translation':'why','sentence_ids':['s1'],'linguistic_occurrence_ids':['loB']}]
        p['subtitles']['sentences'][0]['cue_ids']=['1','3']
        p['subtitles']['linguistic_occurrences']=[
            {'occurrence_id':'loA','instance_id':'liA','concept_id':p['vocabulary']['concepts'][0]['concept_id'],'cue_id':'1','text':'por','start_char':0,'end_char':3,'sentence_ids':['s1']},
            {'occurrence_id':'loB','instance_id':'liA','concept_id':p['vocabulary']['concepts'][0]['concept_id'],'cue_id':'3','text':'qué','start_char':0,'end_char':3,'sentence_ids':['s1']}]
        p['vocabulary']['concepts'][0]['occurrence_ids']=['loA','loB']; p['vocabulary']['concepts'][0]['frequency']=1
        p['subtitles']['display_cues'][0]['source_text']='por'; p['subtitles']['display_cues'][1]['sentence_ids']=[]
        p['quizzes']['questions'][0]['evidence']='Hola.'
        with self.assertRaisesRegex(ValidationError,'crosses dialogue or unknown cue gap'): validate(p,1200)
    def test_exact_duplicate_occurrence_is_rejected(self):
        p=self.package(); p['subtitles']['linguistic_occurrences'].append(dict(p['subtitles']['linguistic_occurrences'][0]))
        with self.assertRaisesRegex(ValidationError,'occurrence IDs are not unique|duplicate occurrence'):
            validate(p,1200)
    def test_frequency_must_match_occurrence_instances(self):
        p=self.package(); p['vocabulary']['concepts'][0]['frequency']=2
        with self.assertRaisesRegex(ValidationError,'frequency'): validate(p,1200)
    def test_invalid_category_and_difficulty(self):
        p=self.package(); p['vocabulary']['concepts'][0]['category']='preposition'; p['vocabulary']['concepts'][0]['difficulty']=5
        with self.assertRaisesRegex(ValidationError,'invalid category'): validate(p,1200)
