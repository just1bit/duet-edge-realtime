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
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    messages = [json.loads(line) for line in Path(args.ndjson).read_text().splitlines()]
    frames = [m for m in messages if m.get("type") == "frame"]
    states = [m.get("state") for m in messages if m.get("type") == "state"]
    failures = []
    if not messages or messages[0].get("type") != "hello": failures.append("missing hello")
    if messages and messages[0].get("protocol") != "duet-edge-stream/v2": failures.append("unexpected protocol")
    if not messages or messages[-1].get("type") != "eos": failures.append("missing eos")
    if states != ["starting", "buffering", "playing", "draining", "finished"]: failures.append("unexpected lifecycle")
    if [m["seq"] for m in frames] != list(range(len(frames))): failures.append("non-contiguous frame seq")
    if any(m.get("frame_id") != m.get("seq") for m in frames): failures.append("frame_id/seq mismatch")
    if any(m.get("schema_version") != "2.0.0" for m in frames): failures.append("unexpected frame schema")
    if any(len(m.get("joints", [])) != 24 for m in frames): failures.append("frame without 24 joints")
    if any(not all(math.isfinite(v) for joint in m["joints"] for v in joint) for m in frames): failures.append("NaN/Inf")
    if summary["output"]["frames"] != len(frames): failures.append("summary/NDJSON frame mismatch")
    if summary["output"].get("committed_frames") != len(frames): failures.append("commit/NDJSON frame mismatch")
    if summary["queues"]["overloads"] != 0: failures.append("inference overload")
    if summary["input"]["sequence_errors"] != 0: failures.append("input sequence error")
    if summary.get("lifecycle", {}).get("final_state") != "finished": failures.append("summary lifecycle is not finished")
    stream_config = summary["config"]["stream"]
    if args.duration_min > 0 and frames and frames[-1]["motion_time_s"] + 1 / stream_config["fps"] < args.duration_min * 60:
        failures.append(f"stream shorter than requested {args.duration_min} minutes")
    if args.require_performance:
        p99 = summary["inference"]["p99_ms"]
        delay_ms = stream_config["playout_delay_s"] * 1000
        hop_ms = stream_config["hop_frames"] / stream_config["fps"] * 1000
        inference_slo_ms = stream_config["inference_slo_ms"]
        jitter_slo_ms = stream_config["jitter_slo_ms"]
        if p99 is None or p99 > inference_slo_ms: failures.append("inference p99 exceeds configured SLO")
        if summary["inference"].get("deadline_misses") != 0: failures.append("inference deadline miss")
        if p99 is None or p99 + 100 >= hop_ms: failures.append("p99+100ms exceeds hop period")
        if p99 is None or p99 + 100 > delay_ms: failures.append("playout delay does not cover p99+100ms")
        if summary["output"]["underflows"] != 0: failures.append("output underflow")
        fps = summary["output"]["observed_fps"]
        if fps is None or not 29.7 <= fps <= 30.3: failures.append("output FPS outside 30+/-0.3")
        jitter = summary["output"]["jitter_p95_ms"]
        if jitter is None or jitter > jitter_slo_ms: failures.append("jitter p95 exceeds configured SLO")
        if not all(summary.get("slo", {}).values()): failures.append("summary SLO result is not fully met")
    result = {"passed": not failures, "failures": failures, "frames": len(frames)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
