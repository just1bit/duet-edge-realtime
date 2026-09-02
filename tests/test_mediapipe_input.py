import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from duet_edge_realtime.mediapipe_input import (
    MediaPipeToMotion151,
    PoseObservation,
    PoseResampler,
    PoseUnavailable,
)
from duet_edge_realtime.backends.fake import FakeInferenceBackend
from duet_edge_realtime.config import InputConfig, RealtimeConfig
from duet_edge_realtime.playout import VirtualClock
from duet_edge_realtime.schemas import MotionFrame
from duet_edge_realtime.service import StreamingService
from duet_edge_realtime.sinks import CompositeSink, NDJSONSink
from duet_edge_realtime.skeleton import rotation_6d_to_matrix

from helpers import identity_motion


class IdentityNormalizer:
    def normalize(self, value):
        return value


def standing_landmarks() -> np.ndarray:
    # Define a pelvis-centred pose in runtime x/y/z coordinates and convert it
    # back to MediaPipe x/y/z plus visibility.
    runtime = np.zeros((33, 3), dtype=np.float32)

    def set_point(index, xyz):
        runtime[index] = xyz

    set_point(0, [0.0, 0.0, 0.75])
    set_point(7, [0.08, 0.0, 0.68])
    set_point(8, [-0.08, 0.0, 0.68])
    set_point(11, [0.22, 0.0, 0.48])
    set_point(12, [-0.22, 0.0, 0.48])
    set_point(13, [0.48, 0.0, 0.28])
    set_point(14, [-0.48, 0.0, 0.28])
    set_point(15, [0.66, 0.0, 0.08])
    set_point(16, [-0.66, 0.0, 0.08])
    for index in (17, 19, 21):
        set_point(index, [0.70, 0.02, 0.06])
    for index in (18, 20, 22):
        set_point(index, [-0.70, 0.02, 0.06])
    set_point(23, [0.10, 0.0, 0.0])
    set_point(24, [-0.10, 0.0, 0.0])
    set_point(25, [0.10, 0.0, -0.48])
    set_point(26, [-0.10, 0.0, -0.48])
    set_point(27, [0.10, 0.0, -0.92])
    set_point(28, [-0.10, 0.0, -0.92])
    set_point(29, [0.10, -0.04, -0.96])
    set_point(30, [-0.10, -0.04, -0.96])
    set_point(31, [0.10, 0.16, -0.96])
    set_point(32, [-0.10, 0.16, -0.96])
    mediapipe = np.stack(
        (runtime[:, 0], -runtime[:, 2], -runtime[:, 1]), axis=-1
    )
    return np.concatenate((mediapipe, np.ones((33, 1), dtype=np.float32)), axis=-1)


def run_live_service(source, *, input_mode: str = "file") -> tuple[list[dict], dict]:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        service = StreamingService(
            RealtimeConfig(input=InputConfig(mode=input_mode)),
            FakeInferenceBackend(),
            source,
            CompositeSink([NDJSONSink(root / "stream.ndjson")]),
            VirtualClock(),
            root / "summary.json",
        )
        asyncio.run(service.run())
        messages = [
            json.loads(line)
            for line in (root / "stream.ndjson").read_text().splitlines()
        ]
        summary = json.loads((root / "summary.json").read_text())
    return messages, summary


class MediaPipeInputTests(unittest.TestCase):
    def test_encoder_produces_finite_valid_motion(self):
        codec = MediaPipeToMotion151(IdentityNormalizer())
        motion = codec.encode(standing_landmarks())
        self.assertEqual(motion.shape, (151,))
        self.assertTrue(np.isfinite(motion).all())
        self.assertGreater(motion[6], 0.8)
        rotations = rotation_6d_to_matrix(motion[7:].reshape(24, 6))
        identity = rotations @ np.swapaxes(rotations, -1, -2)
        np.testing.assert_allclose(
            identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-5
        )
        np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-5)

    def test_encoder_rejects_untracked_pose(self):
        landmarks = standing_landmarks()
        landmarks[:, 3] = 0.0
        with self.assertRaises(PoseUnavailable):
            MediaPipeToMotion151(IdentityNormalizer()).encode(landmarks)

    def test_resampler_emits_contiguous_30_fps_grid(self):
        first = standing_landmarks()
        second = first.copy()
        second[:, 0] += 0.1
        resampler = PoseResampler(30)
        self.assertEqual(resampler.push(PoseObservation(10.0, first)), [])
        samples = resampler.push(PoseObservation(10.1, second))
        self.assertEqual(len(samples), 4)
        np.testing.assert_allclose(
            [sample.timestamp_s for sample in samples],
            10.0 + np.arange(4) / 30.0,
            atol=1e-9,
        )
        self.assertGreater(samples[-1].landmarks[0, 0], samples[0].landmarks[0, 0])
        resampler.reset()
        self.assertEqual(resampler.push(PoseObservation(20.0, first)), [])

    def test_streaming_service_accepts_async_live_source(self):
        class LiveSource:
            is_live = True
            identity = "test-camera"
            metadata = {"live": True, "timeline_id": identity}

            async def frames_async(self):
                for seq, motion in enumerate(identity_motion(150)):
                    yield MotionFrame(seq, seq / 30.0, motion, source_id=self.identity)
                    await asyncio.sleep(0)

        messages, _ = run_live_service(LiveSource())
        self.assertEqual(sum(item["type"] == "frame" for item in messages), 150)
        self.assertEqual(messages[-1]["type"], "eos")

    def test_input_status_reports_pose_usability_changes(self):
        class LiveSource:
            is_live = True
            identity = "test-camera"
            metadata = {"live": True, "timeline_id": identity}
            pose_usable = False

            def status(self):
                return {"pose_usable": self.pose_usable}

            async def frames_async(self):
                motion = identity_motion(150)
                self.pose_usable = True
                for seq, frame in enumerate(motion[:5]):
                    yield MotionFrame(seq, seq / 30.0, frame, source_id=self.identity)
                    await asyncio.sleep(0)
                self.pose_usable = False
                await asyncio.sleep(0.12)
                self.pose_usable = True
                for seq, frame in enumerate(motion[5:], start=5):
                    yield MotionFrame(seq, seq / 30.0, frame, source_id=self.identity)
                    await asyncio.sleep(0)

        messages, _ = run_live_service(LiveSource(), input_mode="mediapipe")
        input_statuses = [
            message for message in messages if message["type"] == "input_status"
        ]
        self.assertEqual(
            [message["pose_usable"] for message in input_statuses],
            [False, True, False, True, False],
        )
        self.assertTrue(all(
            message["input_mode"] == "mediapipe" for message in input_statuses
        ))

    def test_short_mediapipe_session_pauses_without_pipeline_error(self):
        class ShortLiveSource:
            is_live = True
            identity = "short-camera"
            metadata = {"live": True, "timeline_id": identity}

            def status(self):
                return {"state": "connected", "pose_usable": True}

            async def frames_async(self):
                for seq, motion in enumerate(identity_motion(10)):
                    yield MotionFrame(seq, seq / 30.0, motion, source_id=self.identity)
                    await asyncio.sleep(0)

        messages, summary = run_live_service(
            ShortLiveSource(), input_mode="mediapipe"
        )
        self.assertFalse(any(message["type"] == "error" for message in messages))
        self.assertEqual(
            [
                message["pose_usable"]
                for message in messages
                if message["type"] == "input_status"
            ][-1],
            False,
        )
        self.assertEqual(summary["lifecycle"]["final_state"], "finished")


if __name__ == "__main__":
    unittest.main()
