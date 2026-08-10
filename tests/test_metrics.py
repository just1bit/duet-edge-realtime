import unittest

import numpy as np

from duet_edge_realtime.metrics import RunMetrics
from duet_edge_realtime.schemas import GeneratedChunk, MotionWindow

from helpers import identity_motion


class MetricsTests(unittest.TestCase):
    def test_history_is_bounded_for_indefinite_runs(self):
        metrics = RunMetrics("test")
        motion = identity_motion(150)
        for index in range(5000):
            window = MotionWindow(index, index*75, index*75+150, 0, index, motion)
            metrics.record_inference(window, GeneratedChunk(index, motion, float(index)))
            metrics.jitter_ms.append(float(index))
        self.assertEqual(metrics.window_count, 5000)
        self.assertEqual(len(metrics.windows), 256)
        self.assertEqual(len(metrics.inference_wall_ms), 4096)
        self.assertEqual(len(metrics.jitter_ms), 4096)


if __name__ == "__main__":
    unittest.main()
