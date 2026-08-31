import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from duet_edge_realtime.backends.fake import FakeInferenceBackend
from duet_edge_realtime.mediapipe_bridge import (
    INGEST_HOST,
    INGEST_PROTOCOL,
    RemoteMediaPipeSource,
)
from duet_edge_realtime.runtime import RuntimeDaemon


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
