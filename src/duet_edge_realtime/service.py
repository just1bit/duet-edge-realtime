from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import resource
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .backends.duet_edge import CudaDuetEdgeBackend
from .backends.fake import FakeInferenceBackend
from .backends.recorded import RecordedInferenceBackend
from .config import RealtimeConfig
from .continuity import OnlineContinuityProcessor, direct_fk
from .input_adapters import AISTFileReplayAdapter, NormalizedFixtureAdapter
from .lifecycle import Lifecycle, ServiceState
from .metrics import RunMetrics
from .motion_quality import OnlineMotionQuality
from .playout import RealtimeClock, VirtualClock
from .progress import TerminalProgress
from .sinks import CompositeSink, NDJSONSink, StaticWebSink, WebSocketSink
from .skeleton import JOINT_NAMES, PARENTS
from .schemas import PROTOCOL_NAME, SCHEMA_VERSION
from .timeline import TimelineCommitter
from .window_buffer import SequenceError, SlidingWindowBuffer

LOG = logging.getLogger("duet_edge_realtime")
STOP = object()
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def realtime_repository_info() -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            text=True, capture_output=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    return {
        "realtime_root": str(REPOSITORY_ROOT),
        "realtime_commit": git("rev-parse", "HEAD"),
        "realtime_dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


class StreamingService:
    def __init__(
        self, config, backend, source, sink, clock, summary_path, run_id=None,
        *, close_backend: bool = True,
    ):
        self.config = config
        self.backend = backend
        self.source = source
        self.sink = sink
        self.clock = clock
        self.summary_path = Path(summary_path)
        self.run_id = run_id or str(uuid.uuid4())
        self.session_id = self.run_id
        self.stream_id = f"{self.run_id}:companion-motion"
        clock_mode = "virtual" if isinstance(clock, VirtualClock) else "realtime"
        self.metrics = RunMetrics(self.run_id, clock=clock_mode)
        self.lifecycle = Lifecycle()
        self._input_complete = asyncio.Event()
        self._source_time_by_seq: dict[int, float] = {}
        self._source_metadata_by_seq: dict[int, dict] = {}
        self._start_clock = clock.now()
        self._inference_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.inference_queue_size
        )
        self._output_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.output_queue_size
        )
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duet-inference")
        self.motion_quality = OnlineMotionQuality(config.fps)
        self._last_transition_id = None
        self.close_backend = close_backend

    def hello(self) -> dict:
        source_metadata = getattr(self.source, "metadata", {}) if self.source else {}
        continuity = self.backend.continuity_info()
        return {
            "type": "hello",
            "protocol": PROTOCOL_NAME,
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "backend": self.config.backend,
            "backend_badge": self.config.backend.upper(),
            "model_mode": "lead-only",
            "checkpoint": (
                Path(self.config.paths.checkpoint).name
                if self.config.paths.checkpoint else None
            ),
            "checkpoint_sha256": self.config.paths.checkpoint_sha256 or None,
            "guidance": {
                "mode": "lead-only",
                "music": self.config.guidance_music,
                "lead": self.config.guidance_lead,
            },
            "sampling_steps": self.config.sampling_steps,
            "continuity": continuity,
            "source_timeline": {
                "identity": self.config.input.timeline_id or source_metadata.get(
                    "timeline_id", getattr(self.source, "identity", "lead-motion")
                ),
                "path": source_metadata.get("source", self.config.paths.input_motion),
                "sha256": source_metadata.get("source_sha256"),
                "start_frame": source_metadata.get("start_frame", self.config.input.start_frame),
                "end_frame": source_metadata.get("end_frame", self.config.input.end_frame),
                "clip_count": source_metadata.get("clip_count", 1),
            },
            "fps": self.config.fps,
            "joint_count": 24,
            "joint_names": JOINT_NAMES,
            "parents": PARENTS,
            "axis": "x=lateral,y=depth,z=up",
            "coordinate_system": {
                "handedness": "right",
                "x": "lateral",
                "y": "depth",
                "z": "up",
                "units": "model-space",
            },
            "timebases": {
                "source_time_s": "seconds from source frame 0",
                "target_playout_offset_s": "monotonic seconds from service start",
                "emitted_wall_time_s": "Unix epoch seconds",
                "emitted_monotonic_offset_s": "monotonic seconds from service start",
            },
            "fixed_latency_s": (
                (self.config.window_frames - 1) / self.config.fps
                + self.config.playout_delay_s
            ),
            "latency_budget": {
                "window_fill_s": (self.config.window_frames - 1) / self.config.fps,
                "playout_delay_s": self.config.playout_delay_s,
                "hop_period_s": self.config.hop_frames / self.config.fps,
                "inference_slo_ms": self.config.inference_slo_ms,
                "safety_margin_ms": self.config.safety_margin_ms,
                "jitter_slo_ms": self.config.jitter_slo_ms,
            },
            "delivery": {
                "timeline": "contiguous-exactly-once-commit",
                "recorder": "complete",
                "viewer": "latest-frame-wins",
                "inference_queue_policy": self.config.inference_queue_policy,
            },
        }

    async def _publish_initial_state(self) -> None:
        self.metrics.record_state(
            self.lifecycle.state.value, self.clock.now() - self._start_clock
        )
        await self.sink.send(self._state_message())

    async def _transition(self, target: ServiceState) -> None:
        self.lifecycle.transition(target)
        self.metrics.record_state(target.value, self.clock.now() - self._start_clock)
        await self.sink.send(self._state_message())

    def _state_message(self) -> dict:
        return {
            "type": "state",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "state": self.lifecycle.state.value,
            "wall_time_s": time.time(),
            "monotonic_offset_s": self.clock.now() - self._start_clock,
        }

    async def run(self) -> None:
        # Model warmup is reported separately from session latency.
        self._start_clock = self.clock.now()
        self.metrics.started_wall_s = time.time()
        sink_ready = False
        resource_task = None
        try:
            self.backend.start_session(self.session_id)
            await self.sink.start(self.hello())
            sink_ready = True
            resource_task = asyncio.create_task(self._sample_resources())
            await self._publish_initial_state()
            await self._transition(ServiceState.BUFFERING)
            tasks = [
                asyncio.create_task(self._produce_input()),
                asyncio.create_task(self._run_inference()),
                asyncio.create_task(self._run_playout()),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            error = next(
                (task.exception() for task in done if not task.cancelled() and task.exception()),
                None,
            )
            if error is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise error
            await asyncio.gather(*pending)
            self.metrics.exit_reason = "input_complete"
            await self._transition(ServiceState.FINISHED)
            await self.sink.send({
                "type": "eos",
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "session_id": self.session_id,
                "stream_id": self.stream_id,
                "frames": self.metrics.output_frames,
                "reason": "input_complete",
            })
        except Exception as exc:
            self.metrics.errors.append(str(exc))
            self.metrics.exit_reason = "error"
            if sink_ready:
                if self.lifecycle.state not in {ServiceState.FINISHED, ServiceState.FAILED}:
                    try:
                        await self._transition(ServiceState.FAILED)
                    except Exception as state_exc:
                        self.metrics.errors.append(f"state publication: {state_exc}")
                await self.sink.send({
                    "type": "error",
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "session_id": self.session_id,
                    "stream_id": self.stream_id,
                    "error": str(exc),
                })
            raise
        finally:
            if resource_task is not None:
                resource_task.cancel()
                await asyncio.gather(resource_task, return_exceptions=True)
            self.metrics.motion_quality = self.motion_quality.summary()
            try:
                backend_info = self.backend.version_info()
            except Exception as version_exc:
                self.metrics.errors.append(f"backend metadata: {version_exc}")
                backend_info = {"backend": type(self.backend).__name__}
            self.metrics.write(
                self.summary_path,
                {**realtime_repository_info(), **backend_info},
                self.config.as_dict(),
            )
            await self.sink.close()
            self._inference_executor.shutdown(wait=True, cancel_futures=True)
            if self.close_backend:
                self.backend.close()

    async def _sample_resources(self) -> None:
        interval_s = 0.1 if self.config.backend == "cuda" else 1.0
        while True:
            sample_started = time.monotonic()
            usage = resource.getrusage(resource.RUSAGE_SELF)
            sample = {
                "wall_time_s": time.time(),
                "monotonic_offset_s": self.clock.now() - self._start_clock,
                "cpu_user_s": usage.ru_utime,
                "cpu_system_s": usage.ru_stime,
                "rss_kib": usage.ru_maxrss,
                "gpu_utilization_percent": None,
                "gpu_memory_used_mib": None,
            }
            if self.config.backend == "cuda":
                process = None
                try:
                    process = await asyncio.create_subprocess_exec(
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    output, _ = await asyncio.wait_for(process.communicate(), 0.09)
                    utilization, memory = output.decode().splitlines()[0].split(",")
                    sample["gpu_utilization_percent"] = float(utilization.strip())
                    sample["gpu_memory_used_mib"] = float(memory.strip())
                except (FileNotFoundError, IndexError, ValueError, asyncio.TimeoutError):
                    if process is not None and process.returncode is None:
                        process.kill()
                        await process.wait()
            self.metrics.record_resource_sample(sample)
            await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - sample_started)))

    async def _produce_input(self) -> None:
        buffer = SlidingWindowBuffer(seed=self.config.model.seed)
        source_start = self.clock.now()
        for frame in self.source.frames():
            source_deadline = source_start + frame.seq / self.config.fps
            if isinstance(self.clock, VirtualClock):
                # Input event time and playout time are separate logical lanes.
                # Advancing their shared virtual clock here lets playout move
                # future input windows and corrupts observed source timing.
                await asyncio.sleep(0)
                ingest_time = source_deadline
            else:
                await self.clock.sleep_until(source_deadline)
                ingest_time = self.clock.now()
            frame = replace(frame, ingest_monotonic_s=ingest_time)
            if self.metrics.input_first_clock_s is None:
                self.metrics.input_first_clock_s = ingest_time
            self.metrics.input_last_clock_s = ingest_time
            self.metrics.input_frames += 1
            try:
                window = buffer.push(frame, ingest_time)
            except SequenceError:
                self.metrics.sequence_errors += 1
                raise
            self._source_time_by_seq[frame.seq] = frame.source_time_s
            self._source_metadata_by_seq[frame.seq] = {
                "source_id": frame.source_id,
                "source_sha256": frame.source_sha256,
                "clip_id": frame.clip_id,
                "clip_frame": frame.clip_frame,
                "transition_id": frame.transition_id,
                "in_transition": frame.in_transition,
            }
            if window is not None:
                await self._enqueue_inference(window)
        tail_time = (
            self.metrics.input_last_clock_s
            if self.metrics.input_last_clock_s is not None
            else self.clock.now()
        )
        tail = buffer.flush(tail_time)
        if tail is not None:
            await self._enqueue_inference(tail)
        self._input_complete.set()
        if self.lifecycle.state == ServiceState.PLAYING:
            await self._transition(ServiceState.DRAINING)
        await self._inference_queue.put(STOP)

    async def _enqueue_inference(self, window) -> None:
        if not self._inference_queue.full():
            self._inference_queue.put_nowait(window)
            self._record_inference_queue_watermark()
            return
        if self.config.inference_queue_policy == "fail":
            self.metrics.overloads += 1
            await self.sink.send({
                "type": "overload",
                "schema_version": SCHEMA_VERSION,
                "window_id": window.window_id,
                "policy": "fail",
                "reason": "inference_queue_full",
            })
            raise RuntimeError("inference queue reached capacity under fail policy")
        self.metrics.backpressure_waits += 1
        started = time.perf_counter()
        await self._inference_queue.put(window)
        self._record_inference_queue_watermark()
        waited_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.backpressure_wait_ms.append(waited_ms)
        await self.sink.send({
            "type": "backpressure",
            "schema_version": SCHEMA_VERSION,
            "window_id": window.window_id,
            "policy": "block",
            "wait_ms": waited_ms,
        })

    def _record_inference_queue_watermark(self) -> None:
        self.metrics.input_backlog_high_water = max(
            self.metrics.input_backlog_high_water, self._inference_queue.qsize()
        )

    async def _run_inference(self) -> None:
        companion_processor = OnlineContinuityProcessor(
            self.backend,
            robust_filter_z=self.config.continuity.robust_filter_z,
            relative_root_soft_knee=(
                self.config.continuity.relative_root_soft_knee
            ),
            relative_root_softness=self.config.continuity.relative_root_softness,
        )
        committer = TimelineCommitter()
        last_window = None
        last_lead_full = None
        while True:
            item = await self._inference_queue.get()
            if item is STOP:
                if last_window is not None:
                    flush_frames = (
                        self.config.hop_frames
                        if last_window.valid_frames == self.config.window_frames
                        else last_window.valid_frames
                    )
                    joints = companion_processor.flush_with_lead(
                        last_window.motion, flush_frames
                    )
                    lead_joints = last_lead_full[
                        self.config.hop_frames:self.config.hop_frames + flush_frames
                    ]
                    batch = committer.commit(
                        last_window.window_id,
                        last_window.start_seq + self.config.hop_frames,
                        joints,
                        lead_joints=lead_joints,
                        commit_kind="tail",
                    )
                    self.metrics.record_commit(len(batch.joints))
                    await self._enqueue_output(batch)
                await self._output_queue.put(STOP)
                return
            window = item
            inference_started_clock = self.clock.now()
            loop = asyncio.get_running_loop()
            chunk = await loop.run_in_executor(self._inference_executor, self.backend.infer, window)
            inference_finished_clock = self.clock.now()
            self.metrics.record_inference(
                window,
                chunk,
                queue_residence_ms=max(
                    0.0, (inference_started_clock - window.trigger_time_s) * 1000.0
                ),
                batch_ready_ms=max(
                    0.0, (inference_finished_clock - window.trigger_time_s) * 1000.0
                ),
            )
            if chunk.inference_wall_ms > self.config.inference_slo_ms:
                self.metrics.inference_deadline_misses += 1
                await self.sink.send({
                    "type": "degraded",
                    "schema_version": SCHEMA_VERSION,
                    "window_id": window.window_id,
                    "reason": "inference_slo_miss",
                    "observed_ms": chunk.inference_wall_ms,
                    "slo_ms": self.config.inference_slo_ms,
                })
                if self.config.deadline_miss_policy == "fail":
                    raise RuntimeError(
                        f"window {window.window_id} inference {chunk.inference_wall_ms:.1f}ms "
                        f"exceeded {self.config.inference_slo_ms:.1f}ms SLO"
                    )
            # Every generated successor resolves the full previous overlap.
            # A partial EOF's real tail lives in the successor's pending half
            # and is trimmed when STOP is handled above.
            joints = companion_processor.process(
                chunk.motion,
                commit_frames=self.config.hop_frames,
                lead_motion=window.motion,
            )
            lead_full = direct_fk(self.backend, window.motion)
            if last_lead_full is not None:
                overlap_error = float(
                    abs(last_lead_full[self.config.hop_frames:] - lead_full[:self.config.hop_frames]).max()
                )
                self.metrics.lead_overlap_fk_error.append(overlap_error)
                if overlap_error > 1e-5:
                    raise RuntimeError(
                        f"authoritative lead overlap mismatch at window {window.window_id}: "
                        f"{overlap_error}"
                    )
            lead_joints = lead_full[:self.config.hop_frames]
            self.motion_quality.record_window(companion_processor.last_metrics, chunk)
            batch = committer.commit(
                window.window_id,
                window.start_seq,
                joints,
                lead_joints=lead_joints,
                trigger_monotonic_s=window.trigger_time_s,
            )
            self.metrics.record_commit(len(batch.joints))
            await self._enqueue_output(batch)
            last_window = window
            last_lead_full = lead_full

    async def _enqueue_output(self, batch) -> None:
        if self._output_queue.full():
            self.metrics.output_backpressure_waits += 1
        await self._output_queue.put(batch)
        self.metrics.output_backlog_high_water = max(
            self.metrics.output_backlog_high_water, self._output_queue.qsize()
        )

    async def _run_playout(self) -> None:
        next_batch_deadline = None
        output_seq = 0
        while True:
            item = await self._output_queue.get()
            if item is STOP:
                if self.lifecycle.state != ServiceState.DRAINING:
                    raise RuntimeError("playout drained before input completion was published")
                return
            batch = item
            if output_seq != batch.start_frame_id:
                raise RuntimeError(
                    f"playout expected frame {output_seq}, got {batch.start_frame_id}"
                )
            if next_batch_deadline is None:
                if batch.trigger_monotonic_s is None:
                    raise RuntimeError("playout received a tail before the first stable batch")
                next_batch_deadline = (
                    batch.trigger_monotonic_s + self.config.playout_delay_s
                )
                await self._transition(ServiceState.PLAYING)
                if self._input_complete.is_set():
                    await self._transition(ServiceState.DRAINING)
            batch_deadline = next_batch_deadline
            for index, (pose, lead_pose) in enumerate(zip(batch.joints, batch.lead_joints)):
                deadline = batch_deadline + index / self.config.fps
                before = self.clock.now()
                # Millisecond-scale scheduler lateness is reported as jitter.
                # An underflow means playout missed at least one whole frame
                # period because the next batch wasn't ready in time.
                if before > deadline + 1 / self.config.fps:
                    self.metrics.underflows += 1
                await self.clock.sleep_until(deadline)
                now = self.clock.now()
                if self.metrics.output_first_clock_s is None:
                    self.metrics.output_first_clock_s = now
                self.metrics.output_last_clock_s = now
                self.metrics.jitter_ms.append(abs(now - deadline) * 1000.0)
                try:
                    source_time_s = self._source_time_by_seq.pop(output_seq)
                except KeyError as exc:
                    raise RuntimeError(
                        f"missing source time for output frame {output_seq}"
                    ) from exc
                source_metadata = self._source_metadata_by_seq.pop(output_seq, {})
                end_to_end_ms = (
                    (now - self._start_clock) - source_time_s
                ) * 1000.0
                self.metrics.end_to_end_latency_ms.append(end_to_end_ms)
                transition_id = source_metadata.get("transition_id")
                source_transition = (
                    transition_id is not None and transition_id != self._last_transition_id
                )
                model_boundary = output_seq > 0 and output_seq % self.config.hop_frames == 0
                self.motion_quality.record_frame(
                    output_seq,
                    lead_pose,
                    pose,
                    model_boundary=model_boundary,
                    source_transition=source_transition,
                )
                self._last_transition_id = transition_id
                message = {
                    "type": "frame",
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "session_id": self.session_id,
                    "stream_id": self.stream_id,
                    "frame_id": output_seq,
                    "seq": output_seq,
                    "source_time_s": source_time_s,
                    "motion_time_s": source_time_s,
                    "source_id": source_metadata.get("source_id", "lead-motion"),
                    "source_sha256": source_metadata.get("source_sha256"),
                    "clip_id": source_metadata.get("clip_id"),
                    "clip_frame": source_metadata.get("clip_frame"),
                    "transition_id": transition_id,
                    "in_transition": bool(source_metadata.get("in_transition")),
                    "target_playout_offset_s": deadline - self._start_clock,
                    "emitted_monotonic_offset_s": now - self._start_clock,
                    "emitted_wall_time_s": time.time(),
                    "wall_time_s": now - self._start_clock,
                    "end_to_end_latency_ms": end_to_end_ms,
                    "frame_latency_ms": end_to_end_ms,
                    "send_lateness_ms": (now - deadline) * 1000.0,
                    "window_id": batch.window_id,
                    "commit_start_frame_id": batch.start_frame_id,
                    "commit_end_frame_id": batch.end_frame_id,
                    "commit_kind": batch.commit_kind,
                    "boundary": {
                        "model": model_boundary,
                        "source_transition": source_transition,
                    },
                    "flags": [
                        "generated", batch.commit_kind,
                        *(["model-boundary"] if model_boundary else []),
                        *(["source-transition"] if source_transition else []),
                    ],
                    "lead_joints": lead_pose.tolist(),
                    "companion_joints": pose.tolist(),
                    "joints": pose.tolist(),
                }
                await self.sink.send(message)
                output_seq += 1
                self.metrics.output_frames = output_seq
            next_batch_deadline += len(batch.joints) / self.config.fps
            await self.sink.send(self.metrics.live_message())


async def _record_startup_failure(
    service, backend, config, output_dir, backend_name, exc, exit_reason
) -> None:
    """Publish and persist failures that happen before ``service.run()``."""
    service.metrics.exit_reason = exit_reason
    service.metrics.errors.append(str(exc))
    try:
        await service.sink.start(service.hello())
        await service._publish_initial_state()
        await service._transition(ServiceState.FAILED)
        await service.sink.send({
            "type": "error",
            "schema_version": SCHEMA_VERSION,
            "run_id": service.run_id,
            "session_id": service.session_id,
            "stream_id": service.stream_id,
            "error": str(exc),
        })
    except Exception as publication_exc:
        service.metrics.errors.append(f"failure publication: {publication_exc}")
    finally:
        try:
            backend_info = backend.version_info()
        except Exception as version_exc:
            service.metrics.errors.append(f"backend metadata: {version_exc}")
            backend_info = {"backend": backend_name}
        service.metrics.write(
            output_dir / "summary.json",
            {**realtime_repository_info(), **backend_info},
            config.as_dict(),
        )
        try:
            await service.sink.close()
        except Exception as close_exc:
            service.metrics.errors.append(f"failure sink close: {close_exc}")
        service._inference_executor.shutdown(wait=True, cancel_futures=True)
        backend.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duet-EDGE V2 streaming service")
    parser.add_argument("--config", default="configs/example.json")
    parser.add_argument("--backend", choices=("fake", "recorded", "cuda"))
    parser.add_argument("--input")
    parser.add_argument("--input-format", choices=("fixture", "aist"))
    parser.add_argument("--root-scaled", choices=("true", "false"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--duet-edge-root")
    parser.add_argument("--clock", choices=("virtual", "realtime"), default="virtual")
    parser.add_argument(
        "--sink", default="ndjson", help="comma-separated: ndjson,websocket,web"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--run-dir", help="write directly into an initialized V2 run")
    parser.add_argument("--run-id")
    parser.add_argument("--loop", type=int, default=1)
    parser.add_argument("--fake-delay-s", type=float, default=0.0)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--playout-delay-s", type=float)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--progress", action="store_true",
        help="show model warmup and inference-window progress",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    progress = TerminalProgress(args.progress)
    config = RealtimeConfig.load(args.config)
    backend_name = args.backend or config.backend
    model = config.model
    stream = config.stream
    if args.sampling_steps is not None:
        model = replace(model, sampling_steps=args.sampling_steps)
    if args.playout_delay_s is not None:
        stream = replace(stream, playout_delay_s=args.playout_delay_s)

    def resolved_path(cli_value, env_name, json_value, *, required=False):
        value = cli_value or os.environ.get(env_name) or json_value
        if required and not value:
            raise SystemExit(
                f"missing path: use CLI, {env_name}, or config JSON"
            )
        return str(Path(value).expanduser().resolve()) if value else ""

    engine_root = resolved_path(
        args.duet_edge_root, "DUET_EDGE_ROOT", config.paths.duet_edge_root,
        required=backend_name == "cuda",
    )
    checkpoint = resolved_path(
        args.checkpoint, "EDGE_CHECKPOINT", config.paths.checkpoint,
        required=backend_name == "cuda",
    )
    input_path = resolved_path(
        args.input, "EDGE_INPUT_MOTION", config.paths.input_motion, required=True
    )
    output_base = resolved_path(
        args.output_dir,
        "EDGE_OUTPUT_DIR",
        config.paths.output_dir,
        required=not bool(args.run_dir),
    )
    run_id = args.run_id or (Path(args.run_dir).name if args.run_dir else str(uuid.uuid4()))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise SystemExit("--run-id must use 1-128 letters, digits, dot, underscore or dash")
    if args.run_dir:
        output_dir = Path(args.run_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        occupied = [name for name in ("summary.json", "stream.ndjson") if (output_dir / name).exists()]
        if occupied:
            raise SystemExit(f"refusing to overwrite V2 run artifacts: {occupied}")
    else:
        output_dir = Path(output_base) / run_id
        if output_dir.exists():
            raise SystemExit(f"refusing to overwrite existing run directory: {output_dir}")
        output_dir.mkdir(parents=True)

    root_scaled = (
        args.root_scaled == "true"
        if args.root_scaled is not None
        else config.paths.root_scaled
    )
    config = replace(
        config,
        backend=backend_name,
        paths=replace(
            config.paths,
            duet_edge_root=engine_root,
            checkpoint=checkpoint,
            input_motion=input_path,
            output_dir=str(output_dir),
            root_scaled=root_scaled,
        ),
        model=model,
        stream=stream,
    )
    effective_config = {"run_id": run_id, **config.as_dict()}
    serialized_config = json.dumps(effective_config, indent=2) + "\n"
    if not args.run_dir:
        (output_dir / "config.json").write_text(serialized_config, encoding="utf-8")
    (output_dir / "effective_config.json").write_text(serialized_config, encoding="utf-8")

    if backend_name == "fake":
        backend = FakeInferenceBackend(delay_s=args.fake_delay_s)
    elif backend_name == "recorded":
        backend = RecordedInferenceBackend(input_path)
    else:
        backend = CudaDuetEdgeBackend(
            checkpoint,
            engine_root,
            guidance_music=config.guidance_music,
            guidance_lead=config.guidance_lead,
            sampling_steps=config.sampling_steps,
            eta=config.eta,
            progress_callback=progress.model_update if args.progress else None,
        )

    sink_names = {name.strip() for name in args.sink.split(",") if name.strip()}
    unknown = sink_names - {"ndjson", "websocket", "web"}
    if unknown:
        raise SystemExit(f"unknown sinks: {sorted(unknown)}")
    sinks = []
    if "ndjson" in sink_names:
        sinks.append(NDJSONSink(output_dir / "stream.ndjson"))
    metrics_ref = None
    if "websocket" in sink_names:
        def on_drop(client_id):
            if metrics_ref is not None:
                metrics_ref.record_view_drop(client_id)
        def on_connect():
            if metrics_ref is not None:
                metrics_ref.record_client_connected()
        def on_disconnect(duration_s):
            if metrics_ref is not None:
                metrics_ref.record_client_disconnected(duration_s)
        def on_telemetry(message):
            if metrics_ref is not None:
                metrics_ref.record_client_telemetry(message)
        sinks.append(
            WebSocketSink(
                config.bind_host,
                config.port,
                config.viewer_queue_frames,
                on_drop,
                on_connect,
                on_disconnect,
                on_telemetry,
            )
        )
    if "web" in sink_names:
        web_root = Path(config.server.web_root)
        if not web_root.is_absolute():
            web_root = REPOSITORY_ROOT / web_root
        sinks.append(StaticWebSink(config.bind_host, config.web_port, web_root))
    if not sinks:
        raise SystemExit("at least one sink is required")
    clock = VirtualClock() if args.clock == "virtual" else RealtimeClock()
    service = StreamingService(
        config, backend, None, CompositeSink(sinks), clock,
        output_dir / "summary.json", run_id=run_id,
    )
    metrics_ref = service.metrics

    warmup_started = time.perf_counter()
    try:
        await asyncio.to_thread(backend.warmup)
    except Exception as exc:
        await _record_startup_failure(
            service, backend, config, output_dir, backend_name, exc,
            "model_load_or_warmup_error",
        )
        raise
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    try:
        input_format = args.input_format or (
            "fixture" if backend_name in {"fake", "recorded"} else "aist"
        )
        if input_format == "fixture":
            source = NormalizedFixtureAdapter(input_path, config.fps, loop=args.loop)
        else:
            if backend_name != "cuda":
                raise ValueError("AIST preprocessing requires the CUDA Duet-EDGE backend")
            if root_scaled is None:
                raise ValueError("AIST input requires explicit --root-scaled true|false")
            source = AISTFileReplayAdapter(
                input_path,
                backend.edge.normalizer,
                engine_root,
                root_scaled=root_scaled,
                fps=config.fps,
                start_frame=config.input.start_frame,
                end_frame=config.input.end_frame,
                loop=args.loop,
            )
    except Exception as exc:
        await _record_startup_failure(
            service, backend, config, output_dir, backend_name, exc,
            "input_setup_error",
        )
        raise
    service.source = source
    source_frames = len(source.motion) * source.loop
    total_windows = 1 + max(
        0,
        (source_frames - config.window_frames + config.hop_frames - 1)
        // config.hop_frames,
    )
    if hasattr(backend, "set_inference_total_windows"):
        backend.set_inference_total_windows(total_windows)
    service.metrics.model_load_warmup_ms = warmup_ms
    LOG.info(
        "run_id=%s backend=%s input=%s output=%s",
        service.run_id, backend_name, input_path, output_dir,
    )
    await service.run()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        LOG.info("service stopped by operator")


if __name__ == "__main__":
    main()
