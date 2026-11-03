import math
import unittest

from fade_content_agent.checkpoints import final_question_count, make_checkpoints


class CheckpointScheduleTests(unittest.TestCase):
    def test_requested_video_durations(self):
        expected = {
            3: [], 5: [], 10: [], 15: [], 19: [],
            20: [10], 25: [10, 20], 30: [10, 20],
            40: [10, 20, 30], 60: [10, 20, 30, 40, 50],
        }
        final_counts = {3: 10, 5: 10, 10: 10, 15: 20, 19: 20, 20: 20, 25: 30, 30: 30, 40: 40, 60: 60}
        for minutes, targets in expected.items():
            with self.subTest(minutes=minutes):
                duration = minutes * 60
                checkpoints = make_checkpoints(duration, [])
                self.assertEqual([cp['target_time'] for cp in checkpoints], [t * 60 for t in targets])
                self.assertTrue(all(not cp['is_final'] for cp in checkpoints))
                self.assertEqual(final_question_count(duration), final_counts[minutes])
                self.assertEqual(final_question_count(duration), max(10, math.ceil(duration / 600) * 10))

    def test_natural_sentence_boundary_is_preferred_near_target(self):
        checkpoints = make_checkpoints(25 * 60, [{'id': 's1', 'cue_end_times': [611]}])
        self.assertEqual(checkpoints[0]['target_time'], 600)
        self.assertEqual(checkpoints[0]['trigger_time'], 611)
        self.assertEqual(checkpoints[0]['reason'], 'sentence_boundary')

    def test_video_end_never_becomes_an_in_video_checkpoint(self):
        checkpoints = make_checkpoints(20 * 60, [])
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]['trigger_time'], 600)

