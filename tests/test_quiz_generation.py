import unittest

from fade_content_agent.pipeline import _generate_quiz


class MockQuizAI:
    def __init__(self):
        self.calls = []

    def generate(self, stage, prompt, schema):
        self.calls.append((stage, prompt, schema))
        return {'items': [
            {
                'question_type': 'content_comprehension', 'category': 'comprehension',
                'prompt': f'Question {index}?',
                'options': ['One', 'Two', 'Three', 'Four'], 'correct_answer_index': 0,
                'explanation': 'Supported by the quoted line.', 'sentence_ids': ['s2'],
                'concept_lemma': '', 'concept_meaning': '', 'evidence_sentence_id': 's2',
            }
            for index in range(10)
        ]}


class QuizGenerationTests(unittest.TestCase):
    def test_checkpoint_generation_is_limited_to_supplied_interval_evidence(self):
        ai = MockQuizAI()
        eligible = [{'id': 's2', 'text': 'She arrived.', 'cue_end_times': [650], 'concept_ids': []}]
        result = _generate_quiz(ai, eligible, {}, 'cp0001', 10, 1, 'checkpoint', '(600, 700] seconds')
        self.assertEqual(len(result), 10)
        self.assertTrue(all(q['sentence_ids'] == ['s2'] for q in result))
        self.assertTrue(all(q['evidence'] == 'She arrived.' for q in result))
        self.assertIn('(600, 700]', ai.calls[0][1])
        self.assertEqual(ai.calls[0][2]['properties']['items']['items']['properties']['question_type']['enum'], [
            'content_comprehension', 'vocabulary_recognition', 'expression_recognition', 'grammar_understanding'
        ])

    def test_final_generation_accepts_full_episode_evidence_and_variable_count(self):
        ai = MockQuizAI()
        eligible = [
            {'id': 's1', 'text': 'She arrived.', 'cue_end_times': [10], 'concept_ids': []},
            {'id': 's2', 'text': 'She arrived.', 'cue_end_times': [1400], 'concept_ids': []},
        ]
        result = _generate_quiz(ai, eligible, {}, 'final', 10, 11, 'final', '(0, 1500] seconds')
        self.assertEqual(len(result), 10)
        self.assertTrue(all(q['id'].startswith('q') for q in result))
        self.assertIn('(0, 1500]', ai.calls[0][1])

    def test_repair_prompt_contains_failed_item_and_only_replaces_it(self):
        class RepairAI(MockQuizAI):
            def __init__(self):
                super().__init__(); self.calls_count = 0
            def generate(self, stage, prompt, schema):
                self.calls_count += 1
                self.calls.append((stage, prompt, schema))
                if self.calls_count == 1:
                    items = [{
                        'question_type': 'content_comprehension', 'category': 'comprehension',
                        'prompt': f'Question {index}?', 'options': ['One', 'Two', 'Three', 'Four'],
                        'correct_answer_index': 0, 'explanation': 'Supported.',
                        'sentence_ids': ['s2'], 'concept_lemma': '', 'concept_meaning': '', 'evidence_sentence_id': 's2'}
                        for index in range(10)]
                    items[3]['evidence_sentence_id'] = 'missing'
                    return {'items': items}
                return {'items': [{
                    'question_type': 'content_comprehension', 'category': 'comprehension',
                    'prompt': 'Repaired?', 'options': ['One', 'Two', 'Three', 'Four'],
                    'correct_answer_index': 0, 'explanation': 'Supported.',
                    'sentence_ids': ['s2'], 'concept_lemma': '', 'concept_meaning': '', 'evidence_sentence_id': 's2'}]}
        ai = RepairAI()
        result = _generate_quiz(ai, [{'id': 's2', 'text': 'She arrived.', 'cue_end_times': [650], 'concept_ids': []}], {}, 'cp0001', 10, 1, 'checkpoint', '(600, 700] seconds')
        self.assertEqual(len(result), 10)
        self.assertEqual(len(ai.calls), 2)
        self.assertIn('evidence_sentence_id must be one of the cited supplied sentence IDs', ai.calls[1][1])
        self.assertIn('missing', ai.calls[1][1])

    def test_python_resolves_unique_concept_from_text_hint(self):
        class HintAI:
            def generate(self, stage, prompt, schema):
                return {'items':[{
                    'question_type':'vocabulary_recognition','category':'vocabulary','prompt':'Meaning?','options':['hello','bye','table','house'],
                    'correct_answer_index':0,'explanation':'The cited word means hello.','sentence_ids':['s1'],
                    'concept_lemma':'hola','concept_meaning':'greeting','evidence_sentence_id':'s1'}]}
        concepts={'c1':{'lemma':'hola','meaning_key':'greeting','translation':'hello','category':'noun','difficulty':1}}
        warnings=[]
        result=_generate_quiz(HintAI(),[{'id':'s1','text':'Hola.','cue_end_times':[1],'concept_ids':['c1']}],concepts,'cp1',1,1,'checkpoint','(0, 1]',warnings)
        self.assertEqual(result[0]['concept_ids'],['c1'])
        self.assertEqual(warnings,[])

    def test_unresolved_concept_hint_is_kept_with_review_warning(self):
        class HintAI:
            def generate(self, stage, prompt, schema):
                return {'items':[{
                    'question_type':'grammar_understanding','category':'grammar','prompt':'What is this?','options':['One','Two','Three','Four'],
                    'correct_answer_index':0,'explanation':'Review the grammar point.','sentence_ids':['s1'],
                    'concept_lemma':'tener que','concept_meaning':'must','evidence_sentence_id':'s1'}]}
        concepts={'c1':{'lemma':'tener','meaning_key':'to have','translation':'have','category':'verb','difficulty':1},
                  'c2':{'lemma':'tener que','meaning_key':'have to','translation':'must','category':'grammar_pattern','difficulty':2}}
        warnings=[]
        result=_generate_quiz(HintAI(),[{'id':'s1','text':'Tenemos que estudiar.','cue_end_times':[1],'concept_ids':['c1']}],concepts,'final',1,1,'final','(0, 1]',warnings)
        self.assertEqual(result[0]['concept_ids'],[])
        self.assertEqual(warnings[0]['reason'],'quiz_concept_resolution_failed')
        self.assertEqual(warnings[0]['question_id'],'q00001')

