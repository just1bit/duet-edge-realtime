import os
import unittest
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.schemas import MotionWindow

from helpers import identity_motion


@unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1", "set RUN_CUDA_TESTS=1 on a CUDA node")
class CudaSmokeTests(unittest.TestCase):
    def test_single_window_is_finite_and_deterministic(self):
        checkpoint = os.environ["DUET_EDGE_CHECKPOINT"]
        engine = os.environ["DUET_EDGE_ROOT"]
        backend = CudaDuetEdgeBackend(checkpoint, engine, sampling_steps=50)
        backend.warmup()
        try:
            window = MotionWindow(1, 0, 150, 0.0, 123, identity_motion(150))
            first = backend.infer(window).motion
            second = backend.infer(window).motion
            self.assertEqual(first.shape, (150,151))
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
