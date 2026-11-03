import unittest

from fade_content_agent.translations import normalize_translation_alignments


class TranslationAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cues = [{'id': '1', 'translation': 'They share an apartment.'}]
        self.occurrences = [{
            'occurrence_id': 'lo1', 'cue_id': '1', 'text': 'comparten un piso',
            'start_char': 0, 'end_char': 18,
        }]

    def test_python_locates_target_text_and_assigns_stable_id(self):
        result = {'alignments': [{
            'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'paraphrase',
            'confidence': 0.94, 'requires_cross_cue': False,
            'spans': [{'text': 'share an apartment', 'start_char': 999, 'end_char': 1000}],
        }]}
        alignments, warnings = normalize_translation_alignments(result, self.cues, self.occurrences)
        self.assertEqual(warnings, [])
        self.assertEqual(alignments[0]['alignment_id'], 'ta_fc6269ba1f5af16f')
        self.assertEqual((alignments[0]['start_char'], alignments[0]['end_char']), (5, 23))
        self.assertEqual(alignments[0]['text'], 'share an apartment')

    def test_supports_one_to_many_target_texts(self):
        result = {'alignments': [{
            'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'split',
            'confidence': 0.9, 'requires_cross_cue': False,
            'spans': [{'text': 'share'}, {'text': 'an apartment'}],
        }]}
        alignments, warnings = normalize_translation_alignments(result, self.cues, self.occurrences)
        self.assertEqual(len(alignments), 2)
        self.assertEqual([(x['start_char'], x['end_char']) for x in alignments], [(5, 10), (11, 23)])
        self.assertEqual(warnings, [])

    def test_repeated_target_text_uses_model_occurrence_order(self):
        cues = [{'id': '1', 'translation': 'They saw the bank and the bank was closed.'}]
        occurrences = [
            {'occurrence_id': 'lo1', 'cue_id': '1', 'text': 'banco', 'start_char': 0, 'end_char': 5},
            {'occurrence_id': 'lo2', 'cue_id': '1', 'text': 'banco', 'start_char': 10, 'end_char': 15},
        ]
        result = {'alignments': [
            {'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'direct', 'confidence': 0.9,
             'requires_cross_cue': False, 'spans': [{'text': 'bank'}]},
            {'occurrence_id': 'lo2', 'cue_id': '1', 'alignment_type': 'direct', 'confidence': 0.9,
             'requires_cross_cue': False, 'spans': [{'text': 'bank'}]},
        ]}
        alignments, warnings = normalize_translation_alignments(result, cues, occurrences)
        self.assertEqual([(x['occurrence_id'], x['start_char']) for x in alignments], [('lo1', 13), ('lo2', 26)])
        self.assertEqual(warnings, [])

    def test_ambiguous_single_target_text_is_a_warning(self):
        cues = [{'id': '1', 'translation': 'The bank and the bank.'}]
        occurrences = [{'occurrence_id': 'lo1', 'cue_id': '1', 'text': 'banco', 'start_char': 0, 'end_char': 5}]
        result = {'alignments': [{'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'direct',
                                  'confidence': 0.9, 'requires_cross_cue': False, 'spans': [{'text': 'bank'}]}]}
        alignments, warnings = normalize_translation_alignments(result, cues, occurrences)
        self.assertEqual(alignments, [])
        self.assertIn('ambiguous_translation_text', {warning['reason'] for warning in warnings})

    def test_missing_target_text_is_a_warning(self):
        result = {'alignments': [{'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'direct',
                                  'confidence': 0.95, 'requires_cross_cue': False,
                                  'spans': [{'text': 'not in translation'}]}]}
        alignments, warnings = normalize_translation_alignments(result, self.cues, self.occurrences)
        self.assertEqual(alignments, [])
        self.assertIn('translation_text_not_found', {warning['reason'] for warning in warnings})

    def test_cross_cue_requirement_is_a_warning(self):
        result = {'alignments': [{'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'split',
                                  'confidence': 0.95, 'requires_cross_cue': True, 'spans': []}]}
        alignments, warnings = normalize_translation_alignments(result, self.cues, self.occurrences)
        self.assertEqual(alignments, [])
        self.assertIn('alignment_requires_cross_cue', {warning['reason'] for warning in warnings})

    def test_low_confidence_alignment_is_not_published(self):
        result = {'alignments': [{'occurrence_id': 'lo1', 'cue_id': '1', 'alignment_type': 'paraphrase',
                                  'confidence': 0.69, 'requires_cross_cue': False,
                                  'spans': [{'text': 'share an apartment'}]}]}
        alignments, warnings = normalize_translation_alignments(result, self.cues, self.occurrences)
        self.assertEqual(alignments, [])
        self.assertIn('alignment_low_confidence', {warning['reason'] for warning in warnings})

    def test_missing_occurrence_response_is_explicit_warning(self):
        alignments, warnings = normalize_translation_alignments({'alignments': []}, self.cues, self.occurrences)
        self.assertEqual(alignments, [])
        self.assertEqual(warnings[-1]['reason'], 'alignment_missing_model_item')


if __name__ == '__main__':
    unittest.main()
