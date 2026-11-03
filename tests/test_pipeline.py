"""Pipeline-adjacent unit checks; no Agent.run or Gemini API calls are made."""
import json, tempfile, unittest
from pathlib import Path
from fade_content_agent.pipeline import RunLog

class RunLogTests(unittest.TestCase):
    def test_linguistic_counts_are_in_summary_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            log=RunLog(Path(d),'unit-run')
            log.event('linguistic_annotation','completed',concept_count=4,occurrence_count=7,category_distribution={'verb':2,'idiom':2},difficulty_distribution={'1':1,'2':2,'3':1,'4':0},word_difficulty_distribution={'1':1,'2':1,'3':0,'4':0},word_annotated_sentence_count=3,word_occurrence_count=4,word_sentence_cap=2,word_cap_warning_count=1,annotated_sentence_count=12,semantic_sentence_count=20,annotation_coverage_ratio=.6,annotation_coverage_target=.3,annotation_coverage_cap=.6,coverage_review_attempted=True,coverage_cap_warning_count=2,coverage_below_target=False,repeated_concepts=2,unmatched_count=1,alternative_form_mismatch_count=3,zero_occurrence_count=2,ambiguous_count=1)
            log.event('cue_translation_validation','completed',translation_warning_count=1)
            log.event('translation_alignment','completed',alignment_count=6,aligned_occurrence_count=5,alignment_warning_count=2)
            log.finish('completed')
            summary=json.loads((Path(d)/'runs/unit-run/logs/usage_summary.json').read_text())
            self.assertEqual(summary['annotation_summary']['concept_count'],4)
            self.assertEqual(summary['annotation_summary']['zero_occurrence_count'],2)
            self.assertEqual(summary['annotation_summary']['word_sentence_cap'],2)
            self.assertEqual(summary['annotation_summary']['word_cap_warning_count'],1)
            self.assertEqual(summary['annotation_summary']['annotation_coverage_ratio'],.6)
            self.assertEqual(summary['annotation_summary']['coverage_cap_warning_count'],2)
            self.assertEqual(summary['annotation_summary']['translation_warning_count'],1)
            self.assertEqual(summary['annotation_summary']['translation_alignment_count'],6)
            self.assertEqual(summary['annotation_summary']['translation_alignment_warning_count'],2)
            report=(Path(d)/'runs/unit-run/logs/run-report.md').read_text()
            self.assertIn('Cue-local occurrences: 7',report)
            self.assertIn('Genuinely unmatched contexts: 1',report)
            self.assertIn('Zero-occurrence concepts not published: 2',report)
            self.assertIn('Word occurrence cap per semantic sentence: 2',report)
            self.assertIn('Annotated semantic sentence coverage: 12/20 (60.0%)',report)
            self.assertIn('Coverage goal / cap: 30% / 60%',report)
            self.assertIn('Translation alignment spans: 6',report)
