import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.motion_quality import OnlineMotionQuality
from duet_edge_realtime.sinks import StaticWebSink


class V2FeatureTests(unittest.TestCase):
    def test_handoff_metadata_validation(self):
        backend = CudaDuetEdgeBackend("checkpoint.pt", ".", sampling_steps=50)
        schedule = ((999, 978), (978, 958))
        tensor = SimpleNamespace(shape=(1, 75, 151), device="cuda:0")
        backend._handoff = {978: tensor}
        backend._handoff_meta = {
            "next_window_id": 1,
            "shape": (1, 150, 151),
            "sampling_steps": 50,
            "schedule": schedule,
            "dtype": "torch.float32",
            "device": "cuda:0",
        }
        window = SimpleNamespace(window_id=1)
        self.assertIs(backend._validated_handoff(
            window, (1, 150, 151), schedule, "cuda:0"
        ), backend._handoff)
        window.window_id = 2
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            backend._validated_handoff(window, (1, 150, 151), schedule, "cuda:0")

    def test_motion_quality_detects_identical_pose(self):
        quality = OnlineMotionQuality()
        joints = [[[float(index), 0.0, 1.0] for index in range(24)]] * 3
        for frame_id, pose in enumerate(joints):
            quality.record_frame(frame_id, pose, pose)
        summary = quality.summary()
        self.assertEqual(summary["distinctness_body_centered"]["max"], 0.0)
        self.assertEqual(summary["root_position_step"]["max"], 0.0)

    def test_integrated_web_sink_serves_health_and_assets(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "index.html").write_text("viewer")
                sink = StaticWebSink("127.0.0.1", 0, root)
                await sink.start({"protocol": "duet-edge-stream/v3", "run_id": "test"})
                port = sink.server.sockets[0].getsockname()[1]
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                response = await reader.read()
                writer.close()
                await writer.wait_closed()
                await sink.close()
                self.assertIn(b"200 OK", response)
                self.assertIn(b"duet-edge-stream/v3", response)
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
