import tempfile, unittest
from pathlib import Path
from fade_content_agent.subtitles import Cue, batches, clip_cues_to_duration, parse_subtitles

class SubtitleTests(unittest.TestCase):
    def parse(self,text,suffix):
        f=tempfile.NamedTemporaryFile(mode='w',suffix=suffix,encoding='utf-8',delete=False); f.write(text); f.close()
        return parse_subtitles(f.name)
    def test_srt(self):
        cues=self.parse('1\n00:00:01,000 --> 00:00:02,500\n¡Hola!\n\n2\n00:00:03,000 --> 00:00:04,000\n¿Qué tal?\n','.srt')
        self.assertEqual([c.text for c in cues],['¡Hola!','¿Qué tal?']); self.assertEqual(cues[0].start,1)
    def test_vtt(self):
        cues=self.parse('WEBVTT\n\n00:00:01.000 --> 00:00:02.500 align:start\nHola <i>mundo</i>\n','.vtt')
        self.assertEqual(cues[0].text,'Hola mundo')
    def test_batch_overlap(self):
        from fade_content_agent.subtitles import Cue
        cues=[Cue(str(i),i,0,1,'x') for i in range(1,8)]
        parts=list(batches(cues,4,1))
        self.assertEqual(parts[0][0][-1].id,parts[1][0][0].id)
        self.assertEqual(set().union(*(p[1] for p in parts)),{str(i) for i in range(1,8)})

    def test_clips_all_cues_ending_after_video_duration(self):
        cues=[Cue('112',112,422.88,424.81,'Earlier'),
              Cue('113',113,422.88,439.339,'devenir reine'),
              Cue('114',114,424.81,439.339,'[Musique]')]
        adjusted,records=clip_cues_to_duration(cues,438)
        self.assertEqual(adjusted[0],cues[0])
        self.assertEqual([cue.end for cue in adjusted],[424.81,438,438])
        self.assertAlmostEqual(adjusted[1].duration,15.12)
        self.assertAlmostEqual(adjusted[2].duration,13.19)
        self.assertEqual(records,[
            {'cue_id':'113','original_end':439.339,'adjusted_end':438},
            {'cue_id':'114','original_end':439.339,'adjusted_end':438},
        ])

    def test_final_cue_within_duration_is_unchanged(self):
        cues=[Cue('1',1,1,2,'Hello')]
        adjusted,records=clip_cues_to_duration(cues,3)
        self.assertIs(adjusted,cues)
        self.assertEqual(records,[])

    def test_rejects_final_cue_starting_at_or_after_video_end(self):
        cues=[Cue('1',1,438,439,'Too late')]
        with self.assertRaisesRegex(ValueError,'cue 1 starts at or after video duration'):
            clip_cues_to_duration(cues,438)
