import unittest

import numpy as np

from duet_edge_realtime.timeline import TimelineCommitError, TimelineCommitter


class TimelineCommitterTests(unittest.TestCase):
    def test_contiguous_batches_advance_exactly_once(self):
        timeline = TimelineCommitter()
        first = timeline.commit(0, 0, np.zeros((75,24,3), dtype=np.float32))
        second = timeline.commit(1, 75, np.zeros((20,24,3), dtype=np.float32), commit_kind="tail")
        self.assertEqual((first.start_frame_id, first.end_frame_id), (0,75))
        self.assertEqual((second.start_frame_id, second.end_frame_id), (75,95))
        self.assertEqual(timeline.next_frame_id, 95)

    def test_gap_and_duplicate_are_rejected(self):
        timeline = TimelineCommitter()
        timeline.commit(0, 0, np.zeros((75,24,3), dtype=np.float32))
        for start in (0, 76):
            with self.subTest(start=start), self.assertRaises(TimelineCommitError):
                timeline.commit(1, start, np.zeros((1,24,3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
