import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.fake import FakeInferenceBackend
from duet_edge_realtime.config import RealtimeConfig, StreamConfig
from duet_edge_realtime.continuity import OnlineContinuityProcessor
from duet_edge_realtime.input_adapters import NormalizedFixtureAdapter
from duet_edge_realtime.playout import VirtualClock
from duet_edge_realtime.schemas import MotionFrame
from duet_edge_realtime.service import StreamingService
from duet_edge_realtime.sinks import CompositeSink, NDJSONSink

from helpers import identity_motion


class ProtocolTests(unittest.TestCase):
    def test_output_preserves_source_event_times(self):
        class IrregularSource:
            def frames(self):
                motion = identity_motion(150)
                for seq, vector in enumerate(motion):
                    yield MotionFrame(seq, seq * 0.04, vector)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            service = StreamingService(
                RealtimeConfig(), FakeInferenceBackend(), IrregularSource(),
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(), root / "summary.json",
            )
            asyncio.run(service.run())
            frames = [
                json.loads(line)
                for line in (root / "stream.ndjson").read_text().splitlines()
                if json.loads(line).get("type") == "frame"
            ]
            np.testing.assert_allclose(
                [frame["source_time_s"] for frame in frames],
                np.arange(150) * 0.04,
            )

    def test_fake_e2e_protocol_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "fixture.npz", motion_151=identity_motion(300))
            source = NormalizedFixtureAdapter(root / "fixture.npz")
            service = StreamingService(
                RealtimeConfig(stream=StreamConfig(playout_delay_s=2.0)),
                FakeInferenceBackend(),
                source,
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(),
                root / "summary.json",
            )
            asyncio.run(service.run())
            messages = [json.loads(line) for line in (root / "stream.ndjson").read_text().splitlines()]
            self.assertEqual(messages[0]["type"], "hello")
            self.assertEqual(messages[0]["protocol"], "duet-edge-stream/v2")
            self.assertEqual(len(messages[0]["parents"]), 24)
            states = [message["state"] for message in messages if message["type"] == "state"]
            self.assertEqual(states, ["starting", "buffering", "playing", "draining", "finished"])
            draining_index = next(
                index for index, message in enumerate(messages)
                if message.get("type") == "state" and message.get("state") == "draining"
            )
            last_frame_index = max(
                index for index, message in enumerate(messages)
                if message.get("type") == "frame"
            )
            self.assertLess(draining_index, last_frame_index)
            frames = [message for message in messages if message["type"] == "frame"]
            self.assertEqual(len(frames), 300)
            self.assertEqual([frame["seq"] for frame in frames], list(range(300)))
            self.assertTrue(all(len(frame["joints"]) == 24 for frame in frames))
            self.assertTrue(all(len(frame["lead_joints"]) == 24 for frame in frames))
            self.assertTrue(all(len(frame["companion_joints"]) == 24 for frame in frames))
            self.assertTrue(all(
                frame["joints"] == frame["companion_joints"] for frame in frames
            ))
            self.assertTrue(all(frame["frame_id"] == frame["seq"] for frame in frames))
            self.assertTrue(all(frame["schema_version"] == "2.0.0" for frame in frames))
            self.assertEqual(messages[-1]["type"], "eos")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["output"]["frames"], 300)
            self.assertEqual(summary["inference"]["count"], 3)
            self.assertEqual(summary["queues"]["overloads"], 0)
            self.assertEqual(summary["output"]["committed_frames"], 300)
            self.assertEqual(summary["lifecycle"]["final_state"], "finished")
            self.assertAlmostEqual(summary["input"]["observed_fps"], 30.0)
            self.assertEqual(
                [round(window["trigger_time_s"], 6) for window in summary["windows"]["recent"]],
                [4.966667, 7.466667, 9.966667],
            )

    def test_tracked_fake_fixture_is_upright_and_articulated(self):
        fixture = Path(__file__).parent / "fixtures" / "fake_motion.npz"
        motion = np.load(fixture)["motion_151"]
        processor = OnlineContinuityProcessor(FakeInferenceBackend())
        joints = np.concatenate([
            processor.process(motion[:150]),
            processor.process(motion[75:225]),
            processor.process(motion[150:300]),
            processor.flush(),
        ])
        relative = joints - joints[:, :1]
        median_span = np.median(np.ptp(joints, axis=1), axis=0)
        moving_joints = np.linalg.norm(np.ptp(relative, axis=0), axis=1) > 0.02
        self.assertGreater(median_span[2], median_span[1] * 2)
        self.assertGreater(np.median(joints[:, 15, 2] - joints[:, 0, 2]), 0.5)
        self.assertGreaterEqual(np.count_nonzero(moving_joints), 8)

    def test_partial_tail_preserves_exact_length(self):
        for count in (151, 224):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                np.savez(root / "fixture.npz", motion_151=identity_motion(count))
                service = StreamingService(
                    RealtimeConfig(), FakeInferenceBackend(),
                    NormalizedFixtureAdapter(root / "fixture.npz"),
                    CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                    VirtualClock(), root / "summary.json",
                )
                asyncio.run(service.run())
                messages = [json.loads(x) for x in (root / "stream.ndjson").read_text().splitlines()]
                self.assertEqual(sum(m["type"] == "frame" for m in messages), count)

    def test_backend_failure_writes_partial_summary_and_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "fixture.npz", motion_151=identity_motion(225))
            service = StreamingService(
                RealtimeConfig(), FakeInferenceBackend(fail_window=1),
                NormalizedFixtureAdapter(root / "fixture.npz"),
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(), root / "summary.json",
            )
            with self.assertRaises(RuntimeError):
                asyncio.run(service.run())
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["exit_reason"], "error")
            self.assertTrue(summary["errors"])
            messages = [json.loads(x) for x in (root / "stream.ndjson").read_text().splitlines()]
            self.assertTrue(any(m["type"] == "error" for m in messages))
            self.assertEqual(
                [m["state"] for m in messages if m["type"] == "state"][-1], "failed"
            )

    def test_deadline_fail_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "fixture.npz", motion_151=identity_motion(150))
            config = RealtimeConfig(stream=StreamConfig(
                inference_slo_ms=1.0, deadline_miss_policy="fail"
            ))
            service = StreamingService(
                config, FakeInferenceBackend(delay_s=0.005),
                NormalizedFixtureAdapter(root / "fixture.npz"),
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(), root / "summary.json",
            )
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                asyncio.run(service.run())
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["inference"]["deadline_misses"], 1)

    def test_virtual_playout_deadlines_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / "fixture.npz", motion_151=identity_motion(150))
            service = StreamingService(
                RealtimeConfig(), FakeInferenceBackend(),
                NormalizedFixtureAdapter(root / "fixture.npz"),
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(), root / "summary.json",
            )
            asyncio.run(service.run())
            frames = [json.loads(x) for x in (root / "stream.ndjson").read_text().splitlines()]
            frames = [x for x in frames if x["type"] == "frame"]
            differences = np.diff([x["motion_time_s"] for x in frames])
            np.testing.assert_allclose(differences, 1/30, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
