import json, tempfile, unittest, zipfile
from pathlib import Path
from fade_content_agent.packaging import write_package_zip

class PackageTests(unittest.TestCase):
    def test_schema_v2_package_is_six_root_files(self):
        package={name:{'schema_version':'2.0.0'} for name in ('manifest','subtitles','vocabulary','culture','quizzes')}
        package['manifest'].update({'content_id':'v','title':'Test','video':{'id':'v'},'source_language':'es','target_language':'en','assessment_interval_seconds':600,'files':['manifest.json','subtitles.json','vocabulary.json','culture.json','quizzes.json','review-report.md']})
        package['subtitles'].update({'display_cues':[],'sentences':[],'linguistic_occurrences':[],'annotation_warnings':[]})
        package['vocabulary']['concepts']=[]; package['culture']['items']=[]; package['quizzes'].update({'checkpoints':[],'questions':[]})
        with tempfile.TemporaryDirectory() as d:
            archive=write_package_zip(d,'v','run1',package,'Review passed.')
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()),{'manifest.json','subtitles.json','vocabulary.json','culture.json','quizzes.json','review-report.md'})
                self.assertEqual(json.loads(z.read('vocabulary.json'))['schema_version'],'2.0.0')
                self.assertIn('linguistic_occurrences',json.loads(z.read('subtitles.json')))
