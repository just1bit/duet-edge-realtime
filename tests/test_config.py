import unittest

from duet_edge_realtime.config import RealtimeConfig


class ConfigTests(unittest.TestCase):
    def test_v1_constraints(self):
        RealtimeConfig()
        for kwargs in ({"window_frames":149},{"hop_frames":74},{"playout_delay_s":2.5},{"sampling_steps":0}):
            with self.assertRaises(ValueError):
                RealtimeConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
