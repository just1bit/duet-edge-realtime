import unittest

import numpy as np

from duet_edge_realtime.continuity import IdentityNormalizer, OnlineContinuityProcessor
from duet_edge_realtime.skeleton import slerp

from helpers import identity_motion


class ContinuityTests(unittest.TestCase):
    def test_ramp_endpoints_and_translation_alignment(self):
        processor = OnlineContinuityProcessor(IdentityNormalizer())
        first = identity_motion(150)
        second = identity_motion(150)
        second[:, 4] += 100.0
        a = processor.process(first)
        b = processor.process(second)
        tail = processor.flush()
        self.assertEqual((a.shape, b.shape, tail.shape), ((75,24,3),)*3)
        self.assertTrue(np.isfinite(np.concatenate((a,b,tail))).all())
        # New chunk is aligned at overlap start, preventing the artificial 100m jump.
        self.assertLess(np.linalg.norm(b[0,0] - a[-1,0]), 0.1)
        ramp = processor.raised_cosine()
        self.assertAlmostEqual(ramp[0], 0.0)
        self.assertAlmostEqual(ramp[-1], 1.0)

    def test_slerp_shortest_path(self):
        q = np.asarray([[[1.0,0,0,0]]])
        result = slerp(q, -q, np.asarray([[[0.5]]]))
        np.testing.assert_allclose(result, q, atol=1e-8)

    def test_incremental_matches_reference_parameter_stitch(self):
        chunks = [identity_motion(150) for _ in range(3)]
        chunks[1][:, 4] += 4.0
        chunks[2][:, 4] -= 3.0
        processor = OnlineContinuityProcessor(IdentityNormalizer())
        online = np.concatenate([processor.process(chunk) for chunk in chunks] + [processor.flush()])

        roots = [chunk[:, 4:7].astype(np.float64).copy() for chunk in chunks]
        for index in range(1, len(roots)):
            roots[index] += roots[index - 1][75] - roots[index][0]
        ramp = processor.raised_cosine()[:, None]
        offline_roots = [roots[0][:75]]
        for index in range(1, len(roots)):
            offline_roots.append(roots[index - 1][75:] * (1-ramp) + roots[index][:75] * ramp)
        offline_roots.append(roots[-1][75:])
        # Root joint position equals the FK root, so this is an exact check of
        # alignment and overlap timing independent of child offsets.
        np.testing.assert_allclose(online[:, 0], np.concatenate(offline_roots), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
