#!/usr/bin/env python3
"""Validate a V1 summary and NDJSON stream."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ndjson", required=True)
    parser.add_argument("--duration-min", type=float, default=0)
    parser.add_argument("--require-performance", action="store_true")
    parser.add_argument("--require-backend", choices=("fake", "cuda"))
    parser.add_argument("--min-inference-samples", type=int, default=0)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    messages = [
        json.loads(line)
        for line in Path(args.ndjson).read_text(encoding="utf-8").splitlines()
    ]
    frames = [message for message in messages if message.get("type") == "frame"]
    states = [message.get("state") for message in messages if message.get("type") == "state"]
    issues: list[str] = []

    def expect(condition: bool, guidance: str) -> None:
        if not condition:
            issues.append(guidance)

    expect(bool(messages) and messages[0].get("type") == "hello", "Start the stream with hello.")
    expect(bool(messages) and messages[0].get("protocol") == "duet-edge-stream/v2", "Use protocol duet-edge-stream/v2.")
    expect(bool(messages) and messages[-1].get("type") == "eos", "Complete the stream with EOS.")
    expect(not messages or messages[-1].get("type") != "eos" or messages[-1].get("reason") == "input_complete", "Complete EOS with reason input_complete.")
    expect(states == ["starting", "buffering", "playing", "draining", "finished"], "Produce the complete lifecycle sequence.")
    expect([item["seq"] for item in frames] == list(range(len(frames))), "Produce contiguous frame sequence numbers.")
    expect(all(item.get("frame_id") == item.get("seq") for item in frames), "Align frame_id with seq.")
    expect(all(item.get("schema_version") == "2.0.0" for item in frames), "Use frame schema 2.0.0.")
    expect(all(len(item.get("joints", [])) == 24 for item in frames), "Produce 24 joints per frame.")
    expect(all(len(item.get("lead_joints", [])) == 24 for item in frames), "Produce 24 lead joints per frame.")
    expect(all(len(item.get("companion_joints", [])) == 24 for item in frames), "Produce 24 companion joints per frame.")
    expect(all(item.get("joints") == item.get("companion_joints") for item in frames), "Keep joints as the companion_joints compatibility alias.")
    expect(all(all(math.isfinite(value) for joint in item["joints"] for value in joint) for item in frames), "Produce finite joint coordinates.")
    expect(all(all(math.isfinite(value) for joint in item.get("lead_joints", []) for value in joint) for item in frames), "Produce finite lead joint coordinates.")
    expect(summary["output"]["frames"] == len(frames), "Align summary and NDJSON frame counts.")
    expect(summary["input"]["frames"] == len(frames), "Align input and output frame counts.")
    expect(summary["output"].get("committed_frames") == len(frames), "Commit every recorded frame.")
    expect(all(
        item.get("commit_kind") in {"stable", "tail"}
        and isinstance(item.get("frame_id"), int)
        and isinstance(item.get("commit_start_frame_id"), int)
        and isinstance(item.get("commit_end_frame_id"), int)
        and item["commit_start_frame_id"] <= item["frame_id"] < item["commit_end_frame_id"]
        for item in frames
    ), "Use valid commit intervals for every frame.")
    expect(summary["queues"]["overloads"] == 0, "Run with zero inference overloads.")
    expect(summary["input"]["sequence_errors"] == 0, "Run with zero input sequence errors.")
    expect(summary.get("lifecycle", {}).get("final_state") == "finished", "Finish the lifecycle.")
    expect(summary.get("exit_reason") == "input_complete", "Complete all input frames.")
    backend = summary.get("backend", {})
    if args.require_backend:
        expect(backend.get("backend") == args.require_backend, f"Run with the {args.require_backend} backend.")
    samples = summary.get("inference", {}).get("sample_count", 0)
    expect(samples >= args.min_inference_samples, f"Collect at least {args.min_inference_samples} inference samples.")
    if backend.get("backend") == "cuda":
        expect(backend.get("sampling_steps") == summary["config"]["model"]["sampling_steps"], "Align backend and configured sampling steps.")
        expect(bool(backend.get("checkpoint_sha256")), "Record the checkpoint SHA256.")
    stream = summary["config"]["stream"]
    expect(summary["queues"]["inference_high_water"] <= stream["inference_queue_size"], "Keep inference queue use within capacity.")
    expect(summary["queues"]["output_high_water"] <= stream["output_queue_size"], "Keep output queue use within capacity.")
    if args.duration_min > 0 and frames:
        duration = frames[-1]["motion_time_s"] + 1 / stream["fps"]
        expect(duration >= args.duration_min * 60, f"Run for at least {args.duration_min} minutes.")
    if args.require_performance:
        p99 = summary["inference"]["p99_ms"]
        delay_ms = stream["playout_delay_s"] * 1000
        hop_ms = stream["hop_frames"] / stream["fps"] * 1000
        slo_ms = stream["inference_slo_ms"]
        margin_ms = stream["safety_margin_ms"]
        jitter_slo = stream["jitter_slo_ms"]
        fixed_latency = ((stream["window_frames"] - 1) / stream["fps"] + stream["playout_delay_s"]) * 1000
        expect(p99 is not None and p99 <= slo_ms, "Tune inference p99 within the configured SLO.")
        expect(summary["inference"].get("deadline_misses") == 0, "Run with zero inference deadline misses.")
        expect(p99 is not None and p99 + margin_ms < hop_ms, "Keep p99 plus safety margin within the hop period.")
        expect(p99 is not None and p99 + margin_ms <= delay_ms, "Cover p99 plus safety margin with playout delay.")
        expect(summary["output"]["underflows"] == 0, "Run with zero output underflows.")
        fps = summary["output"]["observed_fps"]
        expect(fps is not None and 29.7 <= fps <= 30.3, "Keep output FPS within 30 +/- 0.3.")
        jitter = summary["output"]["jitter_p95_ms"]
        expect(jitter is not None and jitter <= jitter_slo, "Keep jitter p95 within its SLO.")
        first_latency = summary["output"].get("first_frame_latency_s")
        expect(first_latency is not None and abs(first_latency * 1000 - fixed_latency) <= jitter_slo, "Keep first-frame latency within the fixed-latency budget.")
        e2e = summary["output"].get("end_to_end_latency_p95_ms")
        expect(e2e is not None and e2e <= fixed_latency + jitter_slo, "Keep end-to-end p95 within the fixed-latency budget.")
        expect(all(summary.get("slo", {}).values()), "Meet every recorded SLO.")
        if backend.get("backend") == "cuda":
            for field in ("cuda_p50_ms", "cuda_p95_ms", "cuda_p99_ms"):
                expect(summary["inference"].get(field) is not None, f"Record {field}.")
    result = {"passed": not issues, "actions": issues, "frames": len(frames)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
