import unittest
from fade_content_agent.gemini import language_code, load_prompt


class LanguagePromptTests(unittest.TestCase):
    def test_language_codes_and_names_are_normalized(self):
        for value in ('es','es-MX','Spanish','español','西班牙语'):
            self.assertEqual(language_code(value),'es')
        for value in ('fr','fr-FR','fra','French','français','法语'):
            self.assertEqual(language_code(value),'fr')
        for value in ('ja','ja-JP','Japanese','日本語'):
            self.assertIsNone(language_code(value))

    def test_french_prompt_uses_french_specific_analysis_without_spanish_examples(self):
        prompt=load_prompt('analyze','fr-FR')
        self.assertIn('French-specific analysis',prompt)
        self.assertIn('subjunctive triggers',prompt)
        self.assertNotIn('gustar with its indirect object',prompt)

    def test_spanish_prompt_retains_spanish_specific_guidance(self):
        prompt=load_prompt('analyze','es-MX')
        self.assertIn('gustar with its indirect object',prompt)
        self.assertNotIn('French-specific analysis',prompt)

    def test_french_occurrence_and_coverage_prompts_are_selected(self):
        occurrence_prompt=load_prompt('occurrences','French')
        coverage_prompt=load_prompt('coverage_review','fr')
        self.assertIn('Preserve accents, apostrophes, elisions',occurrence_prompt)
        self.assertIn('French-specific supplemental review',coverage_prompt)
