from __future__ import annotations

import json
import math
import platform
import resource
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


@dataclass
class RunMetrics:
    run_id: str
    clock: str = "unspecified"
    started_wall_s: float = field(default_factory=time.time)
    input_frames: int = 0
    output_frames: int = 0
    input_first_clock_s: float | None = None
    input_last_clock_s: float | None = None
    output_first_clock_s: float | None = None
    output_last_clock_s: float | None = None
    model_load_warmup_ms: float | None = None
    window_count: int = 0
    windows: deque = field(default_factory=lambda: deque(maxlen=256))
    inference_count: int = 0
    inference_wall_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    inference_cuda_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    inference_queue_residence_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    batch_ready_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    handoff_copy_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    handoff_state_bytes: deque = field(default_factory=lambda: deque(maxlen=4096))
    handoff_used: int = 0
    handoff_produced: int = 0
    backpressure_wait_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    jitter_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    end_to_end_latency_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    input_backlog_high_water: int = 0
    output_backlog_high_water: int = 0
    dropped_view_frames: int = 0
    dropped_view_frames_by_client: dict[str, int] = field(default_factory=dict)
    viewer_client_sample_limit: int = 256
    underflows: int = 0
    overloads: int = 0
    backpressure_waits: int = 0
    output_backpressure_waits: int = 0
    inference_deadline_misses: int = 0
    committed_batches: int = 0
    committed_frames: int = 0
    state_history: list[dict] = field(default_factory=list)
    sequence_errors: int = 0
    errors: list[str] = field(default_factory=list)
    exit_reason: str = "running"
    connected_clients: int = 0
    peak_connected_clients: int = 0
    client_connections: int = 0
    client_reconnects: int = 0
    client_connection_duration_s: deque = field(default_factory=lambda: deque(maxlen=256))
    client_render_fps: deque = field(default_factory=lambda: deque(maxlen=4096))
    client_frame_age_ms: deque = field(default_factory=lambda: deque(maxlen=4096))
    client_visible_stalls: int = 0
    resource_samples: deque = field(default_factory=lambda: deque(maxlen=4096))
    motion_quality: dict = field(default_factory=dict)
    lead_overlap_fk_error: deque = field(default_factory=lambda: deque(maxlen=4096))

    def record_inference(
        self,
        window,
        chunk,
        *,
        queue_residence_ms: float | None = None,
        batch_ready_ms: float | None = None,
    ) -> None:
        self.window_count += 1
        self.inference_count += 1
        self.windows.append({
            "window_id": window.window_id,
            "start_seq": window.start_seq,
            "end_seq": window.end_seq,
            "valid_frames": window.valid_frames,
            "trigger_time_s": window.trigger_time_s,
            "first_source_time_s": window.first_source_time_s,
            "last_source_time_s": window.last_source_time_s,
        })
        self.inference_wall_ms.append(chunk.inference_wall_ms)
        if chunk.inference_cuda_ms is not None:
            self.inference_cuda_ms.append(chunk.inference_cuda_ms)
        if queue_residence_ms is not None:
            self.inference_queue_residence_ms.append(queue_residence_ms)
        if batch_ready_ms is not None:
            self.batch_ready_ms.append(batch_ready_ms)
        self.handoff_copy_ms.append(chunk.handoff_copy_ms)
        self.handoff_state_bytes.append(chunk.handoff_state_bytes)
        self.handoff_used += int(chunk.handoff_used)
        self.handoff_produced += int(chunk.handoff_produced)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.resource_samples.append({
            "wall_time_s": time.time(),
            "cpu_user_s": usage.ru_utime,
            "cpu_system_s": usage.ru_stime,
            "rss_kib": usage.ru_maxrss,
        })

    def live_message(self) -> dict:
        return {
            "type": "metrics",
            "schema_version": "3.0.0",
            "run_id": self.run_id,
            "session_id": self.run_id,
            "stream_id": f"{self.run_id}:companion-motion",
            "inference_p95_ms": percentile(self.inference_wall_ms, 0.95),
            "jitter_p95_ms": percentile(self.jitter_ms, 0.95),
            "input_backlog": self.input_backlog_high_water,
            "output_backlog": self.output_backlog_high_water,
            "dropped_view_frames": self.dropped_view_frames,
            "dropped_view_frames_by_client": dict(self.dropped_view_frames_by_client),
            "underflow": self.underflows,
            "inference_deadline_misses": self.inference_deadline_misses,
            "backpressure_waits": self.backpressure_waits,
            "connected_clients": self.connected_clients,
            "reconnects": self.client_reconnects,
            "render_fps": percentile(self.client_render_fps, 0.50),
            "frame_age_ms": percentile(self.client_frame_age_ms, 0.95),
            "visible_stalls": self.client_visible_stalls,
        }

    def record_client_connected(self, reconnect: bool = False) -> None:
        self.connected_clients += 1
        self.peak_connected_clients = max(
            self.peak_connected_clients, self.connected_clients
        )
        self.client_connections += 1
        self.client_reconnects += int(reconnect)

    def record_client_disconnected(self, duration_s: float) -> None:
        self.connected_clients = max(0, self.connected_clients - 1)
        self.client_connection_duration_s.append(max(0.0, duration_s))

    def record_client_telemetry(self, payload: dict) -> None:
        if payload.get("event") == "reconnect":
            self.client_reconnects += 1
        render_fps = payload.get("render_fps")
        frame_age_ms = payload.get("frame_age_ms")
        if isinstance(render_fps, (int, float)) and math.isfinite(render_fps):
            self.client_render_fps.append(float(render_fps))
        if isinstance(frame_age_ms, (int, float)) and math.isfinite(frame_age_ms):
            self.client_frame_age_ms.append(float(frame_age_ms))
        self.client_visible_stalls += max(0, int(payload.get("visible_stalls", 0)))

    def record_resource_sample(self, sample: dict) -> None:
        self.resource_samples.append(sample)

    def record_state(self, state: str, monotonic_offset_s: float) -> None:
        self.state_history.append({
            "state": state,
            "wall_time_s": time.time(),
            "monotonic_offset_s": monotonic_offset_s,
        })

    def record_commit(self, frame_count: int) -> None:
        self.committed_batches += 1
        self.committed_frames += frame_count

    def record_view_drop(self, client_id: str) -> None:
        self.dropped_view_frames += 1
        key = client_id
        if (
            key not in self.dropped_view_frames_by_client
            and len(self.dropped_view_frames_by_client) >= self.viewer_client_sample_limit - 1
        ):
            key = "other"
        self.dropped_view_frames_by_client[key] = (
            self.dropped_view_frames_by_client.get(key, 0) + 1
        )

    def summary(self, backend: dict, config: dict) -> dict:
        input_span = (
            self.input_last_clock_s - self.input_first_clock_s
            if self.input_first_clock_s is not None and self.input_last_clock_s is not None
            else 0.0
        )
        output_span = (
            self.output_last_clock_s - self.output_first_clock_s
            if self.output_first_clock_s is not None and self.output_last_clock_s is not None
            else 0.0
        )
        stream_config = config["stream"]
        hop_period_ms = stream_config["hop_frames"] / stream_config["fps"] * 1000.0
        safety_margin_ms = stream_config["safety_margin_ms"]
        inference_p99_ms = percentile(self.inference_wall_ms, 0.99)
        cuda_p50_ms = percentile(self.inference_cuda_ms, 0.50)
        cuda_p95_ms = percentile(self.inference_cuda_ms, 0.95)
        cuda_p99_ms = percentile(self.inference_cuda_ms, 0.99)
        jitter_p95_ms = percentile(self.jitter_ms, 0.95)
        first_frame_latency_s = (
            self.output_first_clock_s - self.input_first_clock_s
            if self.output_first_clock_s is not None and self.input_first_clock_s is not None
            else None
        )
        fixed_latency_ms = (
            (stream_config["window_frames"] - 1) / stream_config["fps"]
            + stream_config["playout_delay_s"]
        ) * 1000.0
        end_to_end_p95_ms = percentile(self.end_to_end_latency_ms, 0.95)
        return {
            "run_id": self.run_id,
            "clock": self.clock,
            "exit_reason": self.exit_reason,
            "started_wall_s": self.started_wall_s,
            "finished_wall_s": time.time(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "config": config,
            "backend": backend,
            "model_load_warmup_ms": self.model_load_warmup_ms,
            "input": {
                "frames": self.input_frames,
                "observed_fps": ((self.input_frames - 1) / input_span if input_span > 0 else None),
                "sequence_errors": self.sequence_errors,
            },
            "windows": {
                "count": self.window_count,
                "recent": list(self.windows),
                "recent_limit": self.windows.maxlen,
            },
            "inference": {
                "count": self.inference_count,
                "sample_count": len(self.inference_wall_ms),
                "sample_limit": self.inference_wall_ms.maxlen,
                "wall_ms": list(self.inference_wall_ms),
                "cuda_ms": list(self.inference_cuda_ms),
                "p50_ms": percentile(self.inference_wall_ms, 0.50),
                "p95_ms": percentile(self.inference_wall_ms, 0.95),
                "p99_ms": inference_p99_ms,
                "cuda_p50_ms": cuda_p50_ms,
                "cuda_p95_ms": cuda_p95_ms,
                "cuda_p99_ms": cuda_p99_ms,
                "deadline_misses": self.inference_deadline_misses,
                "configured_slo_ms": stream_config["inference_slo_ms"],
                "safety_margin_ms": safety_margin_ms,
                "hop_period_ms": hop_period_ms,
                "headroom_ms": (
                    hop_period_ms - inference_p99_ms
                    if inference_p99_ms is not None else None
                ),
                "queue_residence_p95_ms": percentile(
                    self.inference_queue_residence_ms, 0.95
                ),
                "batch_ready_p95_ms": percentile(self.batch_ready_ms, 0.95),
            },
            "handoff": {
                "used": self.handoff_used,
                "produced": self.handoff_produced,
                "state_bytes": list(self.handoff_state_bytes),
                "state_bytes_max": max(self.handoff_state_bytes, default=0),
                "copy_ms_p95": percentile(self.handoff_copy_ms, 0.95),
            },
            "queues": {
                "inference_high_water": self.input_backlog_high_water,
                "output_high_water": self.output_backlog_high_water,
                "overloads": self.overloads,
                "inference_policy": stream_config["inference_queue_policy"],
                "backpressure_waits": self.backpressure_waits,
                "output_backpressure_waits": self.output_backpressure_waits,
                "backpressure_wait_p95_ms": percentile(self.backpressure_wait_ms, 0.95),
            },
            "output": {
                "frames": self.output_frames,
                "observed_fps": ((self.output_frames - 1) / output_span if output_span > 0 else None),
                "first_frame_latency_s": first_frame_latency_s,
                "jitter_p95_ms": jitter_p95_ms,
                "jitter_mean_ms": statistics.fmean(self.jitter_ms) if self.jitter_ms else None,
                "end_to_end_latency_p50_ms": percentile(self.end_to_end_latency_ms, 0.50),
                "end_to_end_latency_p95_ms": end_to_end_p95_ms,
                "underflows": self.underflows,
                "dropped_view_frames": self.dropped_view_frames,
                "dropped_view_frames_by_client": dict(
                    self.dropped_view_frames_by_client
                ),
                "viewer_client_sample_limit": self.viewer_client_sample_limit,
                "committed_batches": self.committed_batches,
                "committed_frames": self.committed_frames,
            },
            "lifecycle": {
                "final_state": self.state_history[-1]["state"] if self.state_history else None,
                "history": self.state_history,
            },
            "clients": {
                "connected_at_finish": self.connected_clients,
                "peak_connected": self.peak_connected_clients,
                "connections": self.client_connections,
                "reconnects": self.client_reconnects,
                "connection_duration_p95_s": percentile(
                    self.client_connection_duration_s, 0.95
                ),
                "render_fps_p50": percentile(self.client_render_fps, 0.50),
                "frame_age_p95_ms": percentile(self.client_frame_age_ms, 0.95),
                "visible_stalls": self.client_visible_stalls,
            },
            "motion_quality": {
                **self.motion_quality,
                "lead_overlap_fk_error": {
                    "count": len(self.lead_overlap_fk_error),
                    "max": max(self.lead_overlap_fk_error, default=0.0),
                    "p99": percentile(self.lead_overlap_fk_error, 0.99),
                },
            },
            "resources": {
                "samples": list(self.resource_samples),
                "sample_limit": self.resource_samples.maxlen,
            },
            "slo": {
                "inference_p99_met": (
                    inference_p99_ms is not None
                    and inference_p99_ms <= stream_config["inference_slo_ms"]
                ),
                "compute_budget_met": (
                    inference_p99_ms is not None
                    and inference_p99_ms + safety_margin_ms < hop_period_ms
                ),
                "playout_budget_met": (
                    inference_p99_ms is not None
                    and inference_p99_ms + safety_margin_ms
                    <= stream_config["playout_delay_s"] * 1000.0
                ),
                "first_frame_latency_met": (
                    first_frame_latency_s is not None
                    and abs(first_frame_latency_s * 1000.0 - fixed_latency_ms)
                    <= stream_config["jitter_slo_ms"]
                ),
                "end_to_end_latency_p95_met": (
                    end_to_end_p95_ms is not None
                    and end_to_end_p95_ms
                    <= fixed_latency_ms + stream_config["jitter_slo_ms"]
                ),
                "jitter_p95_met": (
                    jitter_p95_ms is not None
                    and jitter_p95_ms <= stream_config["jitter_slo_ms"]
                ),
                "continuous_playout_met": self.underflows == 0,
            },
            "errors": self.errors,
        }

    def write(self, path: str | Path, backend: dict, config: dict) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(backend, config), indent=2) + "\n", encoding="utf-8")
