import json
import os
import unittest
from urllib.error import HTTPError
from io import BytesIO
from unittest.mock import patch
from dotenv import load_dotenv

from fade_content_agent.subtitles import parse_subtitles
from fade_content_agent.transcript_api import TranscriptAPIError, fetch_transcript

load_dotenv()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.payload).encode()


def response(language='ja'):
    return {'video_id':'abcdefghijk','language':language,
            'transcript':[{'text':'原文','start':1.25,'duration':2.5}],
            'metadata':{'title':'Ignored for language'},'length_seconds':120}


class TranscriptAPITests(unittest.TestCase):
    @patch('fade_content_agent.transcript_api.urlopen')
    def test_omits_language_by_default_and_normalizes_actual_language_and_timing(self, urlopen):
        urlopen.return_value=FakeResponse(response('ja'))
        result=fetch_transcript('https://youtu.be/abcdefghijk',api_key='test-key')
        request=urlopen.call_args.args[0]
        self.assertNotIn('language=',request.full_url)
        self.assertEqual(request.get_header('Authorization'),'Bearer test-key')
        self.assertEqual(result['language'],'ja')
        self.assertEqual(result['cues'][0].start,1.25)
        self.assertEqual(result['cues'][0].duration,2.5)
        self.assertEqual(result['cues'][0].end,3.75)
        self.assertEqual(result['cues'][0].text,'原文')
        self.assertEqual(result['duration'],120)
        self.assertEqual(json.loads(result['raw_response']),response('ja'))

    @patch('fade_content_agent.transcript_api.urlopen')
    def test_passes_requested_language_codes(self, urlopen):
        for code in ('de','zh-Hant'):
            with self.subTest(code=code):
                urlopen.return_value=FakeResponse(response(code))
                fetch_transcript('abcdefghijk',language=code,api_key='test-key')
                self.assertIn(f'language={code}',urlopen.call_args.args[0].full_url)

    @patch('fade_content_agent.transcript_api.urlopen')
    def test_empty_transcript_and_missing_language_are_explicit_errors(self, urlopen):
        urlopen.return_value=FakeResponse({**response(),'transcript':[]})
        with self.assertRaisesRegex(TranscriptAPIError,'empty'):
            fetch_transcript('abcdefghijk',api_key='test-key')
        urlopen.return_value=FakeResponse({k:v for k,v in response().items() if k!='language'})
        with self.assertRaisesRegex(TranscriptAPIError,'language'):
            fetch_transcript('abcdefghijk',api_key='test-key')

    def test_requires_api_key(self):
        with patch.dict(os.environ,{},clear=True), self.assertRaisesRegex(TranscriptAPIError,'TRANSCRIPT_API_KEY'):
            fetch_transcript('abcdefghijk')

    @patch('fade_content_agent.transcript_api.urlopen')
    def test_http_failure_is_reported(self, urlopen):
        urlopen.side_effect=HTTPError('https://transcriptapi.com',401,'Unauthorized',{},BytesIO(b'{"detail":"invalid key"}'))
        with self.assertRaisesRegex(TranscriptAPIError,'401.*invalid key'):
            fetch_transcript('abcdefghijk',api_key='test-key')

    def test_local_srt_vtt_parser_remains_available(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'sample.vtt'
            source.write_text('WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n',encoding='utf-8')
            self.assertEqual(parse_subtitles(source)[0].text,'Hello')


@unittest.skipUnless(os.getenv('TRANSCRIPT_API_KEY') and os.getenv('TRANSCRIPT_API_TEST_VIDEO_URL'),
                     'Set TRANSCRIPT_API_KEY and TRANSCRIPT_API_TEST_VIDEO_URL for live integration test')
class LiveTranscriptAPITests(unittest.TestCase):
    def test_live_response_contains_language_timing_and_metadata(self):
        result=fetch_transcript(os.environ['TRANSCRIPT_API_TEST_VIDEO_URL'])
        self.assertTrue(result['language'])
        self.assertTrue(result['cues'])
        self.assertGreater(result['duration'],0)
