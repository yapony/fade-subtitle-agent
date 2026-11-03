import json, tempfile, unittest
from pathlib import Path
from fade_content_agent.pipeline import RunLog

class LoggingTests(unittest.TestCase):
    def test_retry_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as d:
            log=RunLog(Path(d),'secret-run'); log.event('api','retrying',error='failure [REDACTED]',retry_delay=1.0); log.finish('failed')
            record=json.loads((Path(d)/'runs/secret-run/logs/events.jsonl').read_text().splitlines()[0])
            self.assertEqual(record['status'],'retrying'); self.assertNotIn('actual-key',record['error'])
