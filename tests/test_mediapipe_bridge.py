import asyncio
import json
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from duet_edge_realtime.backends.fake import FakeInferenceBackend
from duet_edge_realtime.config import InputConfig, RealtimeConfig
from duet_edge_realtime.mediapipe_bridge import (
    INGEST_HOST,
    INGEST_PROTOCOL,
    RemoteMediaPipeSource,
)
from duet_edge_realtime.playout import VirtualClock
from duet_edge_realtime.runtime import RuntimeDaemon
from duet_edge_realtime.service import StreamingService
from duet_edge_realtime.sinks import CompositeSink, NDJSONSink

from helpers import standing_mediapipe_landmarks


class IdentityNormalizer:
    def normalize(self, value):
        return value


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def landmarks() -> list[list[float]]:
    # Encoder validity is not exercised by the one-observation waiting test,
    # but the wire payload must still have the exact MediaPipe shape.
    value = np.zeros((33, 4), dtype=np.float32)
    value[:, 3] = 1.0
    return value.tolist()


async def send(writer: asyncio.StreamWriter, value: dict) -> None:
    writer.write(json.dumps(value).encode() + b"\n")
    await writer.drain()


class MediaPipeBridgeTests(unittest.TestCase):
    def test_landmarks_flow_through_ingest_into_model_window(self):
        class CapturingBackend(FakeInferenceBackend):
            def __init__(self):
                super().__init__()
                self.windows = []

            def infer(self, window):
                self.windows.append(window.motion.copy())
                return super().infer(window)

        async def scenario(root: Path):
            source = RemoteMediaPipeSource(
                IdentityNormalizer(), queue_size=512
            )
            server = await asyncio.start_server(source.accept, INGEST_HOST, 0)
            port = server.sockets[0].getsockname()[1]
            backend = CapturingBackend()
            service = StreamingService(
                RealtimeConfig(input=InputConfig(mode="mediapipe")),
                backend,
                source,
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(),
                root / "summary.json",
                close_backend=False,
            )
            service_task = asyncio.create_task(service.run())
            writer = None
            try:
                reader, writer = await asyncio.open_connection(INGEST_HOST, port)
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "simulated-camera",
                })
                self.assertEqual(json.loads(await reader.readline())["type"], "accepted")

                base_pose = standing_mediapipe_landmarks()
                for frame in range(150):
                    # Simulate a visible arm movement rather than feeding a
                    # frozen pose through the positive path.
                    pose = base_pose.copy()
                    arm_lift = 0.18 * np.sin(frame * np.pi / 30.0)
                    pose[[13, 15, 17, 19, 21], 1] -= arm_lift
                    pose[[14, 16, 18, 20, 22], 1] += arm_lift
                    await send(writer, {
                        "type": "pose",
                        "timestamp_s": 10.0 + frame / 30.0,
                        "landmarks": pose.tolist(),
                    })

                for _ in range(500):
                    if source.emitted_frames == 150:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(source.received_observations, 150)
                self.assertEqual(source.emitted_frames, 150)
                source.stop()
                await asyncio.wait_for(service_task, 5.0)
            finally:
                source.stop()
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                if not service_task.done():
                    service_task.cancel()
                    await asyncio.gather(service_task, return_exceptions=True)
                server.close()
                await server.wait_closed()

            self.assertEqual(backend.calls, 1)
            self.assertEqual(len(backend.windows), 1)
            self.assertEqual(backend.windows[0].shape, (150, 151))
            self.assertTrue(np.isfinite(backend.windows[0]).all())
            self.assertGreater(float(np.abs(backend.windows[0]).sum()), 0.0)
            self.assertGreater(
                float(np.ptp(backend.windows[0][:, 7:], axis=0).max()), 0.01
            )

            messages = [
                json.loads(line)
                for line in (root / "stream.ndjson").read_text().splitlines()
            ]
            self.assertEqual(sum(item["type"] == "frame" for item in messages), 150)
            self.assertEqual(messages[-1]["type"], "eos")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["input"]["frames"], 150)
            self.assertEqual(summary["inference"]["count"], 1)

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(scenario(Path(temp)))

    def test_unusable_camera_pose_never_reaches_model(self):
        async def scenario(root: Path):
            source = RemoteMediaPipeSource(
                IdentityNormalizer(), queue_size=512
            )
            server = await asyncio.start_server(source.accept, INGEST_HOST, 0)
            port = server.sockets[0].getsockname()[1]
            backend = FakeInferenceBackend()
            service = StreamingService(
                RealtimeConfig(input=InputConfig(mode="mediapipe")),
                backend,
                source,
                CompositeSink([NDJSONSink(root / "stream.ndjson")]),
                VirtualClock(),
                root / "summary.json",
                close_backend=False,
            )
            service_task = asyncio.create_task(service.run())
            writer = None
            try:
                reader, writer = await asyncio.open_connection(INGEST_HOST, port)
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "camera-without-usable-pose",
                })
                self.assertEqual(json.loads(await reader.readline())["type"], "accepted")

                unusable = standing_mediapipe_landmarks()
                unusable[:, 3] = 0.1
                for frame in range(150):
                    await send(writer, {
                        "type": "pose",
                        "timestamp_s": 20.0 + frame / 30.0,
                        "landmarks": unusable.tolist(),
                    })
                for _ in range(500):
                    if source.received_observations == 150 and source.queue.empty():
                        break
                    await asyncio.sleep(0.01)
                source.stop()
                await asyncio.wait_for(service_task, 5.0)
            finally:
                source.stop()
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                if not service_task.done():
                    service_task.cancel()
                    await asyncio.gather(service_task, return_exceptions=True)
                server.close()
                await server.wait_closed()

            self.assertEqual(source.received_observations, 150)
            self.assertEqual(source.emitted_frames, 0)
            self.assertEqual(backend.calls, 0)
            messages = [
                json.loads(line)
                for line in (root / "stream.ndjson").read_text().splitlines()
            ]
            self.assertFalse(any(item["type"] == "frame" for item in messages))
            self.assertFalse(any(item["type"] == "error" for item in messages))
            self.assertEqual(messages[-1]["type"], "eos")
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["input"]["frames"], 0)
            self.assertEqual(summary["inference"]["count"], 0)

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(scenario(Path(temp)))

    def test_malformed_camera_payload_is_rejected(self):
        async def scenario():
            source = RemoteMediaPipeSource(IdentityNormalizer())
            server = await asyncio.start_server(source.accept, INGEST_HOST, 0)
            port = server.sockets[0].getsockname()[1]
            try:
                reader, writer = await asyncio.open_connection(INGEST_HOST, port)
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "malformed-camera",
                })
                self.assertEqual(json.loads(await reader.readline())["type"], "accepted")
                await send(writer, {
                    "type": "pose",
                    "timestamp_s": 30.0,
                    "landmarks": [[0.0, 0.0, 0.0, 1.0]] * 32,
                })
                response = json.loads(await reader.readline())
                self.assertEqual(response["type"], "error")
                self.assertIn("[33,4]", response["error"])
                self.assertEqual(source.received_observations, 0)
                self.assertEqual(source.emitted_frames, 0)
                writer.close()
                await writer.wait_closed()
            finally:
                source.stop()
                server.close()
                await server.wait_closed()

        asyncio.run(scenario())

    def test_reset_prevents_interpolation_across_tracking_gap(self):
        async def scenario():
            source = RemoteMediaPipeSource(IdentityNormalizer())
            server = await asyncio.start_server(source.accept, INGEST_HOST, 0)
            port = server.sockets[0].getsockname()[1]
            frames = source.frames_async()
            try:
                reader, writer = await asyncio.open_connection(INGEST_HOST, port)
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "reacquiring-camera",
                })
                self.assertEqual(json.loads(await reader.readline())["type"], "accepted")
                pose = standing_mediapipe_landmarks().tolist()
                await send(writer, {
                    "type": "pose", "timestamp_s": 10.0, "landmarks": pose,
                })
                await send(writer, {"type": "reset"})
                await send(writer, {
                    "type": "pose", "timestamp_s": 20.0, "landmarks": pose,
                })
                await send(writer, {
                    "type": "pose", "timestamp_s": 20.0 + 1 / 30, "landmarks": pose,
                })
                first = await asyncio.wait_for(anext(frames), 2.0)
                second = await asyncio.wait_for(anext(frames), 2.0)
                self.assertEqual([first.seq, second.seq], [0, 1])
                np.testing.assert_allclose(
                    [first.source_time_s, second.source_time_s], [0.0, 1 / 30]
                )
                self.assertEqual(source.emitted_frames, 2)
                writer.close()
                await writer.wait_closed()
            finally:
                source.stop()
                await frames.aclose()
                server.close()
                await server.wait_closed()

        asyncio.run(scenario())

    def test_pose_usable_is_the_only_freshness_gate(self):
        source = RemoteMediaPipeSource(IdentityNormalizer(), stale_after_s=0.5)
        self.assertFalse(source.status()["pose_usable"])

        source.last_emitted_monotonic_s = time.monotonic()
        self.assertTrue(source.status()["pose_usable"])

        source.last_emitted_monotonic_s = time.monotonic() - 1.0
        self.assertFalse(source.status()["pose_usable"])

    def test_remote_source_accepts_producer_without_ending_on_disconnect(self):
        async def scenario():
            source = RemoteMediaPipeSource(IdentityNormalizer())
            server = await asyncio.start_server(source.accept, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "test-camera",
                })
                accepted = json.loads(await reader.readline())
                self.assertEqual(accepted["type"], "accepted")
                await send(writer, {
                    "type": "pose",
                    "timestamp_s": 10.0,
                    "landmarks": landmarks(),
                })
                for _ in range(100):
                    if source.status()["received_observations"] == 1:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(source.status()["received_observations"], 1)
                writer.close()
                await writer.wait_closed()
                await asyncio.sleep(0)
                self.assertFalse(source._stopped)
                source.stop()
                frames = source.frames_async()
                with self.assertRaises(StopAsyncIteration):
                    await anext(frames)
            finally:
                server.close()
                await server.wait_closed()

        asyncio.run(scenario())

    def test_runtime_switches_to_waiting_mediapipe_and_back_to_file(self):
        async def scenario(root: Path):
            web_root = root / "web"
            web_root.mkdir()
            (web_root / "index.html").write_text("viewer", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "backend": "cuda",
                "paths": {"output_dir": str(root)},
                "input": {"mode": "file"},
                "server": {
                    "bind_host": "127.0.0.1",
                    "port": available_port(),
                    "web_port": available_port(),
                    "control_port": available_port(),
                    "ingest_port": available_port(),
                    "web_root": str(web_root),
                },
            }), encoding="utf-8")
            daemon = RuntimeDaemon(config_path, root)
            backend = FakeInferenceBackend()
            backend.edge = SimpleNamespace(normalizer=IdentityNormalizer())
            daemon.backend = backend
            daemon.model_state = "ready"
            try:
                await daemon.start_control()
                await daemon.activate_stream()
                await daemon.activate_viewer()
                await daemon.set_input_mode("mediapipe")
                for _ in range(100):
                    if isinstance(
                        getattr(daemon.active_service, "source", None),
                        RemoteMediaPipeSource,
                    ):
                        break
                    await asyncio.sleep(0.01)
                status = daemon.status()
                self.assertEqual(status["input"]["mode"], "mediapipe")
                self.assertEqual(status["session"]["state"], "waiting_input")
                self.assertEqual(status["input"]["ingest"]["state"], "waiting")

                reader, writer = await asyncio.open_connection(
                    INGEST_HOST, daemon.config.ingest_port
                )
                await send(writer, {
                    "type": "hello",
                    "protocol": INGEST_PROTOCOL,
                    "producer_id": "test-camera",
                })
                self.assertEqual(json.loads(await reader.readline())["type"], "accepted")
                self.assertEqual(daemon.status()["input"]["ingest"]["state"], "connected")
                writer.close()
                await writer.wait_closed()

                await daemon.set_input_mode("file")
                self.assertEqual(daemon.status()["input"]["mode"], "file")
                self.assertEqual(daemon.status()["session"]["state"], "idle")
            finally:
                await daemon.close()

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(scenario(Path(temp)))


if __name__ == "__main__":
    unittest.main()
