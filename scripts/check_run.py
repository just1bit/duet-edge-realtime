#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate a V1 summary and NDJSON stream")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ndjson", required=True)
    parser.add_argument("--duration-min", type=float, default=0)
    parser.add_argument("--require-performance", action="store_true")
    parser.add_argument("--require-backend", choices=("fake", "cuda"))
    parser.add_argument("--min-inference-samples", type=int, default=0)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    messages = [json.loads(line) for line in Path(args.ndjson).read_text().splitlines()]
    frames = [m for m in messages if m.get("type") == "frame"]
    states = [m.get("state") for m in messages if m.get("type") == "state"]
    failures = []
    if not messages or messages[0].get("type") != "hello": failures.append("missing hello")
    if messages and messages[0].get("protocol") != "duet-edge-stream/v2": failures.append("unexpected protocol")
    if not messages or messages[-1].get("type") != "eos": failures.append("missing eos")
    if messages and messages[-1].get("type") == "eos" and messages[-1].get("reason") != "input_complete": failures.append("unexpected eos reason")
    if states != ["starting", "buffering", "playing", "draining", "finished"]: failures.append("unexpected lifecycle")
    if [m["seq"] for m in frames] != list(range(len(frames))): failures.append("non-contiguous frame seq")
    if any(m.get("frame_id") != m.get("seq") for m in frames): failures.append("frame_id/seq mismatch")
    if any(m.get("schema_version") != "2.0.0" for m in frames): failures.append("unexpected frame schema")
    if any(len(m.get("joints", [])) != 24 for m in frames): failures.append("frame without 24 joints")
    if any(not all(math.isfinite(v) for joint in m["joints"] for v in joint) for m in frames): failures.append("NaN/Inf")
    if summary["output"]["frames"] != len(frames): failures.append("summary/NDJSON frame mismatch")
    if summary["input"]["frames"] != len(frames): failures.append("input/output frame mismatch")
    if summary["output"].get("committed_frames") != len(frames): failures.append("commit/NDJSON frame mismatch")
    if any(
        not (
            m.get("commit_kind") in {"stable", "tail"}
            and isinstance(m.get("frame_id"), int)
            and isinstance(m.get("commit_start_frame_id"), int)
            and isinstance(m.get("commit_end_frame_id"), int)
            and m["commit_start_frame_id"] <= m["frame_id"] < m["commit_end_frame_id"]
        )
        for m in frames
    ):
        failures.append("invalid frame commit interval")
    if summary["queues"]["overloads"] != 0: failures.append("inference overload")
    if summary["input"]["sequence_errors"] != 0: failures.append("input sequence error")
    if summary.get("lifecycle", {}).get("final_state") != "finished": failures.append("summary lifecycle is not finished")
    if summary.get("exit_reason") != "input_complete": failures.append("summary exit reason is not input_complete")
    backend = summary.get("backend", {})
    if args.require_backend and backend.get("backend") != args.require_backend:
        failures.append(f"backend is not required {args.require_backend}")
    sample_count = summary.get("inference", {}).get("sample_count", 0)
    if sample_count < args.min_inference_samples:
        failures.append(
            f"inference sample count {sample_count} is below {args.min_inference_samples}"
        )
    if backend.get("backend") == "cuda":
        configured_steps = summary["config"]["model"]["sampling_steps"]
        if backend.get("sampling_steps") != configured_steps:
            failures.append("backend/config sampling steps mismatch")
        if not backend.get("checkpoint_sha256"):
            failures.append("missing checkpoint SHA256")
    stream_config = summary["config"]["stream"]
    if summary["queues"]["inference_high_water"] > stream_config["inference_queue_size"]:
        failures.append("inference queue exceeded configured capacity")
    if summary["queues"]["output_high_water"] > stream_config["output_queue_size"]:
        failures.append("output queue exceeded configured capacity")
    if args.duration_min > 0 and frames and frames[-1]["motion_time_s"] + 1 / stream_config["fps"] < args.duration_min * 60:
        failures.append(f"stream shorter than requested {args.duration_min} minutes")
    if args.require_performance:
        p99 = summary["inference"]["p99_ms"]
        delay_ms = stream_config["playout_delay_s"] * 1000
        hop_ms = stream_config["hop_frames"] / stream_config["fps"] * 1000
        inference_slo_ms = stream_config["inference_slo_ms"]
        safety_margin_ms = stream_config["safety_margin_ms"]
        jitter_slo_ms = stream_config["jitter_slo_ms"]
        fixed_latency_ms = (
            (stream_config["window_frames"] - 1) / stream_config["fps"]
            + stream_config["playout_delay_s"]
        ) * 1000.0
        if p99 is None or p99 > inference_slo_ms: failures.append("inference p99 exceeds configured SLO")
        if summary["inference"].get("deadline_misses") != 0: failures.append("inference deadline miss")
        if p99 is None or p99 + safety_margin_ms >= hop_ms: failures.append("p99+safety margin exceeds hop period")
        if p99 is None or p99 + safety_margin_ms > delay_ms: failures.append("playout delay does not cover p99+safety margin")
        if summary["output"]["underflows"] != 0: failures.append("output underflow")
        fps = summary["output"]["observed_fps"]
        if fps is None or not 29.7 <= fps <= 30.3: failures.append("output FPS outside 30+/-0.3")
        jitter = summary["output"]["jitter_p95_ms"]
        if jitter is None or jitter > jitter_slo_ms: failures.append("jitter p95 exceeds configured SLO")
        first_latency = summary["output"].get("first_frame_latency_s")
        if first_latency is None or abs(first_latency * 1000.0 - fixed_latency_ms) > jitter_slo_ms:
            failures.append("first-frame latency exceeds fixed-latency budget")
        end_to_end_p95 = summary["output"].get("end_to_end_latency_p95_ms")
        if end_to_end_p95 is None or end_to_end_p95 > fixed_latency_ms + jitter_slo_ms:
            failures.append("end-to-end latency p95 exceeds fixed-latency budget")
        if not all(summary.get("slo", {}).values()): failures.append("summary SLO result is not fully met")
        if backend.get("backend") == "cuda":
            for field in ("cuda_p50_ms", "cuda_p95_ms", "cuda_p99_ms"):
                if summary["inference"].get(field) is None:
                    failures.append(f"missing {field}")
    result = {"passed": not failures, "failures": failures, "frames": len(frames)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
