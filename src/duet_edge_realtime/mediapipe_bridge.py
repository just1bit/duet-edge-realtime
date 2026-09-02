from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .config import RealtimeConfig
from .mediapipe_input import (
    MediaPipeLandmarkCamera,
    MediaPipeToMotion151,
    PoseObservation,
    PoseResampler,
    PoseUnavailable,
)
from .schemas import MotionFrame


LOG = logging.getLogger("duet_edge_realtime.mediapipe")
INGEST_PROTOCOL = "duet-edge-mediapipe/v1"
INGEST_HOST = "127.0.0.1"
_STOP = object()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class RemoteMediaPipeSource:
    """Live MotionFrame source fed by an independently managed pose producer."""

    is_live = True

    def __init__(
        self,
        normalizer,
        *,
        fps: int = 30,
        queue_size: int = 120,
        stale_after_s: float = 0.5,
    ):
        self.identity = "mediapipe-bridge"
        self.metadata = {
            "source": self.identity,
            "timeline_id": self.identity,
            "fps": fps,
            "live": True,
            "transport": INGEST_PROTOCOL,
        }
        self.codec = MediaPipeToMotion151(normalizer, fps=fps)
        self.resampler = PoseResampler(fps)
        self.fps = fps
        self.stale_after_s = stale_after_s
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.connected = False
        self.producer_id: str | None = None
        self.received_observations = 0
        self.emitted_frames = 0
        self.dropped_observations = 0
        self.last_observation_wall_s: float | None = None
        self.last_emitted_monotonic_s: float | None = None
        self._stopped = False
        self._connection_lock = asyncio.Lock()

    def status(self) -> dict:
        now = time.monotonic()
        emitted_age_s = (
            None
            if self.last_emitted_monotonic_s is None
            else max(0.0, now - self.last_emitted_monotonic_s)
        )
        pose_usable = bool(
            emitted_age_s is not None
            and emitted_age_s <= self.stale_after_s
        )
        return {
            "state": "connected" if self.connected else "waiting",
            "producer_id": self.producer_id,
            "received_observations": self.received_observations,
            "emitted_frames": self.emitted_frames,
            "dropped_observations": self.dropped_observations,
            "last_observation_wall_s": self.last_observation_wall_s,
            "pose_usable": pose_usable,
        }

    async def accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        async with self._connection_lock:
            try:
                hello = await self._read_message(reader)
                if hello.get("type") != "hello" or hello.get("protocol") != INGEST_PROTOCOL:
                    raise ValueError(f"expected {INGEST_PROTOCOL} hello")
                if self._stopped:
                    raise RuntimeError("MediaPipe input mode is not active")
                self.connected = True
                self.producer_id = str(hello.get("producer_id") or "mediapipe")
                self.last_emitted_monotonic_s = None
                self.resampler.reset()
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.dropped_observations += 1
                    except asyncio.QueueEmpty:
                        break
                await self._send(writer, {
                    "type": "accepted",
                    "protocol": INGEST_PROTOCOL,
                    "fps": self.fps,
                })
                while not self._stopped:
                    message = await self._read_message(reader)
                    if message.get("type") == "reset":
                        self.resampler.reset()
                        continue
                    if message.get("type") != "pose":
                        continue
                    observation = PoseObservation(
                        float(message["timestamp_s"]),
                        np.asarray(message["landmarks"], dtype=np.float32),
                    )
                    self.received_observations += 1
                    self.last_observation_wall_s = time.time()
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                            self.dropped_observations += 1
                        except asyncio.QueueEmpty:
                            pass
                    self.queue.put_nowait(observation)
            except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
                pass
            except Exception as exc:
                LOG.warning("MediaPipe producer rejected/disconnected: %s", exc)
                try:
                    await self._send(writer, {"type": "error", "error": str(exc)})
                except Exception:
                    pass
            finally:
                self.connected = False
                self.producer_id = None
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass

    @staticmethod
    async def _read_message(reader: asyncio.StreamReader) -> dict:
        line = await asyncio.wait_for(reader.readline(), 10.0)
        if not line:
            raise asyncio.IncompleteReadError(line, None)
        if len(line) > 64 * 1024:
            raise ValueError("MediaPipe ingest message is too large")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("MediaPipe ingest message must be an object")
        return value

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, value: dict) -> None:
        writer.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")
        await writer.drain()

    async def frames_async(self) -> AsyncIterator[MotionFrame]:
        seq = 0
        while True:
            item = await self.queue.get()
            if item is _STOP:
                return
            for sample in self.resampler.push(item):
                try:
                    motion = self.codec.encode(sample.landmarks)
                except PoseUnavailable:
                    continue
                frame = MotionFrame(
                    seq=seq,
                    source_time_s=seq / self.fps,
                    motion_151=motion,
                    source_id=self.identity,
                )
                seq += 1
                self.emitted_frames = seq
                self.last_emitted_monotonic_s = time.monotonic()
                yield frame

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self.queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(_STOP)


class MediaPipeProducer:
    def __init__(self, config: RealtimeConfig, run_dir: Path, *, max_observations: int = 0):
        self.config = config
        self.run_dir = run_dir
        self.max_observations = max_observations
        self.stop_event = asyncio.Event()
        self.sent = 0
        self.detected = 0
        self.connection_state = "disconnected"
        self.error: str | None = None
        self.camera = MediaPipeLandmarkCamera(
            config.paths.mediapipe_model,
            camera_index=config.input.camera_index,
            width=config.input.camera_width,
            height=config.input.camera_height,
            maximum_missing_s=config.input.maximum_missing_s,
        )

    @property
    def status_path(self) -> Path:
        return self.run_dir / "evidence" / "mediapipe-status.json"

    def persist(self, state: str) -> None:
        write_json(self.status_path, {
            "state": state,
            "connection": self.connection_state,
            "camera_index": self.config.input.camera_index,
            "detected_observations": self.detected,
            "sent_observations": self.sent,
            "error": self.error,
            "updated_at": time.time(),
        })

    async def run(self) -> None:
        self.persist("opening_camera")
        await asyncio.to_thread(self.camera.open)
        self.persist("running")
        writer: asyncio.StreamWriter | None = None
        last_observation_s: float | None = None
        next_connect_attempt_s = 0.0
        try:
            while not self.stop_event.is_set():
                try:
                    observation = await asyncio.to_thread(self.camera.read_observation)
                except PoseUnavailable:
                    continue
                self.detected += 1
                if writer is None and time.monotonic() >= next_connect_attempt_s:
                    writer = await self._connect()
                    next_connect_attempt_s = time.monotonic() + 1.0
                if writer is not None:
                    try:
                        if (
                            last_observation_s is not None
                            and observation.timestamp_s - last_observation_s
                            > self.config.input.maximum_missing_s
                        ):
                            writer.write(b'{"type":"reset"}\n')
                        writer.write(json.dumps({
                            "type": "pose",
                            "timestamp_s": observation.timestamp_s,
                            "landmarks": observation.landmarks.tolist(),
                        }, separators=(",", ":")).encode() + b"\n")
                        await writer.drain()
                        self.sent += 1
                        self.connection_state = "connected"
                        last_observation_s = observation.timestamp_s
                    except (ConnectionError, BrokenPipeError):
                        writer.close()
                        writer = None
                        self.connection_state = "waiting_for_service"
                if self.detected % 30 == 0:
                    self.persist("running")
                if self.max_observations and self.detected >= self.max_observations:
                    break
        except Exception as exc:
            self.error = str(exc)
            self.persist("failed")
            raise
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
            await asyncio.to_thread(self.camera.close)
            if self.error is None:
                self.persist("stopped")

    async def _connect(self) -> asyncio.StreamWriter | None:
        writer: asyncio.StreamWriter | None = None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(
                INGEST_HOST,
                self.config.ingest_port,
            ), 1.0)
            writer.write(json.dumps({
                "type": "hello",
                "protocol": INGEST_PROTOCOL,
                "producer_id": f"mediapipe-camera-{self.config.input.camera_index}",
            }, separators=(",", ":")).encode() + b"\n")
            await writer.drain()
            response = json.loads(await asyncio.wait_for(reader.readline(), 2.0))
            if response.get("type") != "accepted":
                raise RuntimeError(response.get("error", "MediaPipe ingest rejected"))
            self.connection_state = "connected"
            return writer
        except Exception as exc:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionError:
                    pass
            self.connection_state = "waiting_for_service"
            self.error = None
            LOG.debug("waiting for MediaPipe ingest service: %s", exc)
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent MediaPipe camera producer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-observations", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--doctor", action="store_true")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    config = RealtimeConfig.load(args.config)
    if args.doctor:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        model = Path(config.paths.mediapipe_model).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(model)
        print(json.dumps({
            "ok": True,
            "model": str(model),
            "camera_index": config.input.camera_index,
            "ingest": f"{INGEST_HOST}:{config.ingest_port}",
        }, indent=2))
        return
    producer = MediaPipeProducer(config, Path(args.run_dir), max_observations=args.max_observations)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, producer.stop_event.set)
        except NotImplementedError:
            pass
    await producer.run()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
