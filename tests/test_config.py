import json
import tempfile
import unittest
from pathlib import Path

from duet_edge_realtime.config import ModelConfig, RealtimeConfig, StreamConfig


class ConfigTests(unittest.TestCase):
    def test_v1_constraints(self):
        RealtimeConfig()
        for kwargs in (
            {"window_frames":149}, {"hop_frames":74}, {"playout_delay_s":2.5}
        ):
            with self.assertRaises(ValueError):
                StreamConfig(**kwargs)
        with self.assertRaises(ValueError):
            ModelConfig(sampling_steps=0)

    def test_nested_json_round_trip(self):
        data = {
            "backend":"cuda",
            "paths":{"duet_edge_root":"/engine", "checkpoint":"/model.pt"},
            "model":{"sampling_steps":25, "seed":9},
            "stream":{"playout_delay_s":1.5},
            "server":{"port":9876},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(data))
            config = RealtimeConfig.load(path)
        self.assertEqual(config.backend, "cuda")
        self.assertEqual(config.paths.duet_edge_root, "/engine")
        self.assertEqual(config.sampling_steps, 25)
        self.assertEqual(config.model.seed, 9)
        self.assertEqual(config.port, 9876)
        self.assertIn("stream", config.as_dict())


if __name__ == "__main__":
    unittest.main()
