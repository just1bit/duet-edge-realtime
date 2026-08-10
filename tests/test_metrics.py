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
            metrics.record_inference(
                window, GeneratedChunk(index, motion, float(index), float(index) / 2)
            )
            metrics.jitter_ms.append(float(index))
        self.assertEqual(metrics.window_count, 5000)
        self.assertEqual(len(metrics.windows), 256)
        self.assertEqual(len(metrics.inference_wall_ms), 4096)
        self.assertEqual(len(metrics.jitter_ms), 4096)

    def test_summary_reports_cuda_latency_percentiles(self):
        metrics = RunMetrics("test")
        motion = identity_motion(150)
        for index, cuda_ms in enumerate((10.0, 20.0, 30.0)):
            window = MotionWindow(index, index*75, index*75+150, 0, index, motion)
            metrics.record_inference(
                window, GeneratedChunk(index, motion, cuda_ms + 1, cuda_ms)
            )
        summary = metrics.summary(
            {"backend": "cuda"},
            {"stream": {
                "window_frames": 150, "hop_frames": 75, "fps": 30,
                "inference_slo_ms": 1900,
                "safety_margin_ms": 100, "playout_delay_s": 2.0,
                "inference_queue_policy": "block", "jitter_slo_ms": 20,
            }},
        )
        self.assertEqual(summary["inference"]["cuda_p50_ms"], 20.0)
        self.assertAlmostEqual(summary["inference"]["cuda_p95_ms"], 29.0)
        self.assertAlmostEqual(summary["inference"]["cuda_p99_ms"], 29.8)

    def test_per_client_viewer_drop_history_is_bounded(self):
        metrics = RunMetrics("test")
        for index in range(1000):
            metrics.record_view_drop(f"viewer-{index}")
        self.assertEqual(metrics.dropped_view_frames, 1000)
        self.assertLessEqual(
            len(metrics.dropped_view_frames_by_client),
            metrics.viewer_client_sample_limit,
        )
        self.assertIn("other", metrics.dropped_view_frames_by_client)


if __name__ == "__main__":
    unittest.main()
