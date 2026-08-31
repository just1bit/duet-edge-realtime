from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import signal
import time
from dataclasses import replace
from pathlib import Path

from .backends.duet_edge import CudaDuetEdgeBackend
from .backends.fake import FakeInferenceBackend
from .backends.recorded import RecordedInferenceBackend
from .config import RealtimeConfig
from .input_adapters import AISTFileReplayAdapter, NormalizedFixtureAdapter
from .mediapipe_input import MediaPipeCameraAdapter
from .playout import RealtimeClock
from .schemas import PROTOCOL_NAME, SCHEMA_VERSION
from .service import StreamingService
from .sinks import CompositeSink, NDJSONSink, Sink, StaticWebSink, WebSocketSink
from .skeleton import JOINT_NAMES, PARENTS


LOG = logging.getLogger("duet_edge_realtime.runtime")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class SessionViewerSink(Sink):
    """Session-facing adapter that leaves the daemon-owned WebSocket open."""

    def __init__(self, websocket: WebSocketSink):
        self.websocket = websocket

    async def start(self, hello: dict) -> None:
        await self.websocket.update_hello(hello)

    async def send(self, message: dict) -> None:
        await self.websocket.send(message)

    async def close(self) -> None:
        return None


class RuntimeDaemon:
    """One resident GPU runtime with independently activated logical services."""

    def __init__(self, config_path: str | Path, run_dir: str | Path):
        self.config_path = Path(config_path).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.config = RealtimeConfig.load(self.config_path)
        self.run_id = self.run_dir.name
        self.config_sha256 = sha256(self.config_path)
        self.backend = None
        self.model_state = "loading"
        self.stream_state = "stopped"
        self.viewer_state = "stopped"
        self.session_state = "idle"
        self.session_id: str | None = None
        self.error: str | None = None
        self.warmup_ms: float | None = None
        self.active_service: StreamingService | None = None
        self.session_task: asyncio.Task | None = None
        self.control_server = None
        self.websocket_sink: WebSocketSink | None = None
        self.web_sink: StaticWebSink | None = None
        self.shutdown_event = asyncio.Event()

    def status(self) -> dict:
        model_progress = (
            self.backend.progress_snapshot()
            if self.backend is not None
            and hasattr(self.backend, "progress_snapshot")
            else None
        )
        progress = None
        if self.active_service is not None:
            total_frames = None
            manifest_path = self.run_dir / "input-manifest.json"
            if self.config.input.mode != "mediapipe" and manifest_path.is_file():
                try:
                    total_frames = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    ).get("estimated_frames_30fps")
                except (OSError, ValueError):
                    pass
            progress = {
                "input_frames": self.active_service.metrics.input_frames,
                "output_frames": self.active_service.metrics.output_frames,
                "total_frames": total_frames,
                "inference_windows": self.active_service.metrics.inference_count,
                "sampling": model_progress,
            }
        return {
            "ok": self.error is None,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "model": {
                "state": self.model_state,
                "warmup_ms": self.warmup_ms,
                "progress": model_progress,
            },
            "stream": {"state": self.stream_state},
            "viewer": {
                "state": self.viewer_state,
                "url": (
                    f"http://{self.config.bind_host}:{self.config.web_port}"
                    if self.viewer_state == "ready" else None
                ),
            },
            "session": {
                "state": self.session_state,
                "session_id": self.session_id,
                "progress": progress,
            },
            "error": self.error,
        }

    def persist_status(self) -> None:
        write_json(self.run_dir / "evidence" / "runtime-status.json", self.status())

    async def start_control(self) -> None:
        self.control_server = await asyncio.start_server(
            self._handle_control, self.config.bind_host, self.config.control_port
        )
        self.persist_status()

    async def initialize_model(self) -> None:
        started = time.perf_counter()
        try:
            if self.config.backend == "fake":
                self.backend = FakeInferenceBackend()
                await asyncio.to_thread(self.backend.warmup)
            elif self.config.backend == "recorded":
                # Recorded output is bound to the Stage 07 fixture and is loaded
                # when Stage 08 creates the session.
                self.backend = None
            else:
                self.backend = CudaDuetEdgeBackend(
                    self.config.paths.checkpoint,
                    self.config.paths.duet_edge_root,
                    guidance_music=self.config.guidance_music,
                    guidance_lead=self.config.guidance_lead,
                    sampling_steps=self.config.sampling_steps,
                    eta=self.config.eta,
                )
                await asyncio.to_thread(self.backend.warmup)
            self.warmup_ms = (time.perf_counter() - started) * 1000.0
            self.model_state = "ready"
            write_json(self.run_dir / "evidence" / "model-service.json", {
                "status": "ready",
                "backend": self.config.backend,
                "sampling_steps": self.config.sampling_steps,
                "resident_instances": 0 if self.config.backend == "recorded" else 1,
                "load_mode": (
                    "session-bound-fixture"
                    if self.config.backend == "recorded" else "resident"
                ),
                "warmup_ms": self.warmup_ms,
                "config_sha256": self.config_sha256,
            })
            self.persist_status()
        except Exception as exc:
            self.model_state = "failed"
            self.error = str(exc)
            self.persist_status()
            raise

    async def activate_stream(self) -> None:
        if self.model_state != "ready":
            raise RuntimeError("model service is not ready")
        if self.stream_state == "ready":
            return
        self.stream_state = "ready"
        write_json(self.run_dir / "evidence" / "stream-service.json", {
            "status": "ready",
            "queue_policy": self.config.inference_queue_policy,
            "inference_queue_size": self.config.inference_queue_size,
            "output_queue_size": self.config.output_queue_size,
            "config_sha256": self.config_sha256,
        })
        self.persist_status()

    def _waiting_hello(self) -> dict:
        return {
            "type": "hello",
            "protocol": PROTOCOL_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": None,
            "stream_id": None,
            "backend": self.config.backend,
            "backend_badge": self.config.backend.upper(),
            "model_mode": "lead-only",
            "checkpoint": (
                Path(self.config.paths.checkpoint).name
                if self.config.paths.checkpoint else None
            ),
            "sampling_steps": self.config.sampling_steps,
            "state": "waiting_for_input",
            "fps": self.config.fps,
            "joint_count": 24,
            "joint_names": JOINT_NAMES,
            "parents": PARENTS,
            "source_timeline": {"identity": "waiting-for-input"},
            "fixed_latency_s": (
                (self.config.window_frames - 1) / self.config.fps
                + self.config.playout_delay_s
            ),
        }

    async def activate_viewer(self) -> None:
        if self.stream_state != "ready":
            raise RuntimeError("stream service is not ready")
        if self.viewer_state == "ready":
            return

        def on_drop(client_id):
            if self.active_service is not None:
                self.active_service.metrics.record_view_drop(client_id)

        def on_connect():
            if self.active_service is not None:
                self.active_service.metrics.record_client_connected()

        def on_disconnect(duration_s):
            if self.active_service is not None:
                self.active_service.metrics.record_client_disconnected(duration_s)

        def on_telemetry(message):
            if self.active_service is not None:
                self.active_service.metrics.record_client_telemetry(message)

        self.websocket_sink = WebSocketSink(
            self.config.bind_host,
            self.config.port,
            self.config.viewer_queue_frames,
            on_drop,
            on_connect,
            on_disconnect,
            on_telemetry,
        )
        web_root = Path(self.config.server.web_root)
        if not web_root.is_absolute():
            web_root = REPOSITORY_ROOT / web_root
        self.web_sink = StaticWebSink(
            self.config.bind_host, self.config.web_port, web_root
        )
        hello = self._waiting_hello()
        try:
            await self.websocket_sink.start(hello)
            await self.web_sink.start(hello)
            await self.websocket_sink.send({
                "type": "state",
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "session_id": None,
                "stream_id": None,
                "state": "waiting_for_input",
                "wall_time_s": time.time(),
                "monotonic_offset_s": 0.0,
            })
        except Exception:
            await self._close_viewer()
            raise
        self.viewer_state = "ready"
        write_json(self.run_dir / "evidence" / "viewer-service.json", {
            "status": "ready",
            "viewer_url": f"http://{self.config.bind_host}:{self.config.web_port}",
            "websocket_url": f"ws://{self.config.bind_host}:{self.config.port}",
            "config_sha256": self.config_sha256,
        })
        self.persist_status()

    def _load_input_manifest(self) -> dict:
        manifest_path = self.run_dir / "input-manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Stage 07 input-manifest.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("passed") or manifest.get("status") != "locked":
            raise RuntimeError("input manifest is not locked")
        if manifest.get("run_id") != self.run_id:
            raise RuntimeError("input manifest run_id does not match the runtime")
        if manifest.get("config_sha256") != self.config_sha256:
            raise RuntimeError("input manifest config hash does not match the runtime")
        input_path = Path(manifest["path"]).resolve()
        if not input_path.is_file() or sha256(input_path) != manifest.get("sha256"):
            raise RuntimeError("locked input hash verification failed")
        return manifest

    def _session_config(self, manifest: dict) -> RealtimeConfig:
        return replace(
            self.config,
            paths=replace(
                self.config.paths,
                input_motion=manifest["path"],
                input_sha256=manifest["sha256"],
                output_dir=str(self.run_dir),
                root_scaled=manifest.get("root_scaled", self.config.paths.root_scaled),
            ),
            input=replace(
                self.config.input,
                timeline_id=manifest.get("timeline_id", ""),
            ),
        )

    async def start_session(self) -> None:
        if self.session_state in {"preparing", "starting", "running"}:
            return
        if not (
            self.model_state == self.stream_state == self.viewer_state == "ready"
        ):
            raise RuntimeError("model, stream and viewer must all be ready")
        if (self.run_dir / "summary.json").exists() or (self.run_dir / "stream.ndjson").exists():
            raise RuntimeError("formal run artifacts already exist")
        self.session_id = f"{self.run_id}:formal"
        self.session_state = "preparing"
        self.error = None
        self.persist_status()
        self.session_task = asyncio.create_task(self._prepare_and_run_session())

    async def _prepare_and_run_session(self) -> None:
        try:
            if self.config.input.mode == "mediapipe":
                if self.config.backend != "cuda" or self.backend is None:
                    raise RuntimeError("MediaPipe input requires the CUDA backend")
                config = self.config
                source = MediaPipeCameraAdapter(
                    config.paths.mediapipe_model,
                    self.backend.edge.normalizer,
                    camera_index=config.input.camera_index,
                    fps=config.fps,
                    width=config.input.camera_width,
                    height=config.input.camera_height,
                    maximum_missing_s=config.input.maximum_missing_s,
                )
            else:
                manifest = await asyncio.to_thread(self._load_input_manifest)
                config = self._session_config(manifest)
                input_path = Path(manifest["path"])
                if self.config.backend == "recorded":
                    self.backend = RecordedInferenceBackend(input_path)
                    await asyncio.to_thread(self.backend.warmup)
                if self.backend is None:
                    raise RuntimeError("model backend is unavailable")
                if manifest["input_format"] == "fixture":
                    source = await asyncio.to_thread(
                        NormalizedFixtureAdapter, input_path, config.fps
                    )
                elif manifest["input_format"] == "aist":
                    if self.config.backend != "cuda":
                        raise RuntimeError("AIST input requires the CUDA backend")
                    source = await asyncio.to_thread(
                        AISTFileReplayAdapter,
                        input_path,
                        self.backend.edge.normalizer,
                        config.paths.duet_edge_root,
                        root_scaled=bool(manifest["root_scaled"]),
                        fps=config.fps,
                        start_frame=config.input.start_frame,
                        end_frame=config.input.end_frame,
                    )
                else:
                    raise RuntimeError(f"unknown input format {manifest['input_format']!r}")
            if not getattr(source, "is_live", False):
                source_frames = len(source.motion) * source.loop
                total_windows = 1 + max(
                    0,
                    (source_frames - config.window_frames + config.hop_frames - 1)
                    // config.hop_frames,
                )
                if hasattr(self.backend, "set_inference_total_windows"):
                    self.backend.set_inference_total_windows(total_windows)
            elif hasattr(self.backend, "set_inference_total_windows"):
                self.backend.set_inference_total_windows(0)
            session_sink = CompositeSink([
                NDJSONSink(self.run_dir / "stream.ndjson"),
                SessionViewerSink(self.websocket_sink),
            ])
            self.active_service = StreamingService(
                config,
                self.backend,
                source,
                session_sink,
                RealtimeClock(),
                self.run_dir / "summary.json",
                run_id=self.run_id,
                close_backend=False,
            )
            self.active_service.session_id = self.session_id
            self.active_service.stream_id = f"{self.session_id}:companion-motion"
            self.active_service.metrics.model_load_warmup_ms = self.warmup_ms
            for _ in self.websocket_sink.clients:
                self.active_service.metrics.record_client_connected()
            write_json(self.run_dir / "effective_config.json", {
                "run_id": self.run_id,
                "session_id": self.session_id,
                **config.as_dict(),
            })
            self.session_state = "running"
            self.persist_status()
            await self.active_service.run()
            self.session_state = "finished"
        except Exception as exc:
            LOG.exception("formal session failed")
            self.session_state = "failed"
            self.error = str(exc)
        finally:
            self.active_service = None
            self.persist_status()

    async def stop_session(self) -> None:
        if self.active_service is None or not getattr(
            self.active_service.source, "is_live", False
        ):
            raise RuntimeError("no live MediaPipe session is running")
        self.active_service.source.stop()

    async def _handle_control(self, reader, writer) -> None:
        status_code = 200
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 3.0)
            first = request.split(b"\r\n", 1)[0].decode("ascii", "replace")
            method, target, _ = first.split(" ", 2)
            path = target.split("?", 1)[0]
            if method == "GET" and path == "/status":
                payload = self.status()
            elif method == "POST" and path == "/stream/start":
                await self.activate_stream()
                payload = self.status()
            elif method == "POST" and path == "/viewer/start":
                await self.activate_viewer()
                payload = self.status()
            elif method == "POST" and path == "/run/start":
                await self.start_session()
                status_code = 202
                payload = self.status()
            elif method == "POST" and path == "/run/stop":
                await self.stop_session()
                status_code = 202
                payload = self.status()
            elif method == "POST" and path == "/shutdown":
                self.shutdown_event.set()
                payload = self.status()
            else:
                status_code = 404
                payload = {"ok": False, "error": "not found"}
        except Exception as exc:
            status_code = 409
            payload = {"ok": False, "error": str(exc), "status": self.status()}
        body = json.dumps(payload, separators=(",", ":")).encode()
        reasons = {200: "OK", 202: "Accepted", 404: "Not Found", 409: "Conflict"}
        header = (
            f"HTTP/1.1 {status_code} {reasons[status_code]}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        writer.write(header + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _close_viewer(self) -> None:
        if self.web_sink is not None:
            await self.web_sink.close()
            self.web_sink = None
        if self.websocket_sink is not None:
            await self.websocket_sink.close()
            self.websocket_sink = None

    async def close(self) -> None:
        if self.session_task is not None and not self.session_task.done():
            self.session_task.cancel()
            await asyncio.gather(self.session_task, return_exceptions=True)
        await self._close_viewer()
        if self.control_server is not None:
            self.control_server.close()
            await self.control_server.wait_closed()
            self.control_server = None
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duet-EDGE resident runtime")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    daemon = RuntimeDaemon(args.config, args.run_dir)
    expected_hash_path = daemon.run_dir / "config.sha256"
    if not expected_hash_path.is_file():
        raise RuntimeError("Stage 03 config.sha256 is missing")
    expected_hash = expected_hash_path.read_text(encoding="utf-8").split()[0]
    if expected_hash != daemon.config_sha256:
        raise RuntimeError("config.json changed after Stage 03 finalization")
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, daemon.shutdown_event.set)
        except NotImplementedError:
            pass
    await daemon.start_control()
    try:
        await daemon.initialize_model()
        await daemon.shutdown_event.wait()
    finally:
        await daemon.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        LOG.info("runtime stopped by operator")


if __name__ == "__main__":
    main()
