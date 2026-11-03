import json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

from fade_content_agent import schema_validation

class SchemaContractTests(unittest.TestCase):
    def read(self,name): return json.loads((Path(__file__).parents[1]/'schemas'/f'{name}.schema.json').read_text())

    def test_installed_package_loads_schema_files_beside_module(self):
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory) / 'site-packages' / 'fade_content_agent'
            packaged_schemas = package_dir / 'schemas'
            packaged_schemas.mkdir(parents=True)
            source_schemas = Path(__file__).parents[1] / 'schemas'
            for schema in source_schemas.glob('*.schema.json'):
                shutil.copy(schema, packaged_schemas / schema.name)

            with patch.object(schema_validation, '__file__', str(package_dir / 'schema_validation.py')):
                errors = schema_validation.validate_schemas({})

        self.assertFalse(any(error.startswith('cannot load ') for error in errors))

    def test_schema_version_two_and_concept_contract(self):
        subtitles=self.read('subtitles'); vocabulary=self.read('vocabulary'); quizzes=self.read('quizzes'); culture=self.read('culture')
        self.assertEqual(subtitles['properties']['schema_version']['const'],'2.0.0')
        self.assertIn('linguistic_occurrences',subtitles['properties'])
        self.assertIn('concepts',vocabulary['properties'])
        concept=vocabulary['properties']['concepts']['items']['properties']
        self.assertEqual(concept['difficulty']['enum'],[1,2,3,4])
        self.assertIn('meaning_key',concept)
        question=quizzes['$defs']['question']['properties']
        self.assertIn('concept_ids',question); self.assertNotIn('annotation_ids',question)
        self.assertEqual(question['question_type']['enum'],['content_comprehension','vocabulary_recognition','expression_recognition','grammar_understanding'])
        self.assertEqual(quizzes['$defs']['question'].get('trigger_time'),None)
        self.assertIn('final_quiz',quizzes['properties'])
        self.assertIn('quiz_warnings',quizzes['properties'])
        self.assertIn('trigger_time',quizzes['properties']['final_quiz']['properties'])
        self.assertNotIn('trigger_time',quizzes['$defs']['question']['properties'])
        self.assertEqual(quizzes['$defs']['question']['properties']['options']['minItems'],4)
        self.assertIn('translation_warnings',subtitles['properties'])
        self.assertIn('translation_alignments',subtitles['properties'])
        self.assertIn('translation_alignment_warnings',subtitles['properties'])
        culture_item=culture['properties']['items']['items']['properties']
        self.assertIn('needs_review',culture_item); self.assertIn('verification_note',culture_item)
        vocabulary_item=vocabulary['properties']['concepts']['items']['properties']
        self.assertEqual(vocabulary_item['selection_priority']['enum'],['high','medium','low'])
