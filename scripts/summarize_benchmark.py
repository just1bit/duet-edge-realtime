#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--pattern", default="steps-*/summary.json")
    parser.add_argument("--output", default="benchmark.json")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).glob(args.pattern)):
        data = json.loads(path.read_text())
        if data.get("exit_reason") != "input_complete":
            raise SystemExit(f"benchmark did not finish successfully: {path}")
        backend = data.get("backend", {})
        if backend.get("backend") != "cuda":
            raise SystemExit(f"benchmark is not from CUDA backend: {path}")
        if data["inference"].get("sample_count", 0) < args.min_samples:
            raise SystemExit(
                f"benchmark has fewer than {args.min_samples} inference samples: {path}"
            )
        configured_steps = data["config"]["model"]["sampling_steps"]
        if backend.get("sampling_steps") != configured_steps:
            raise SystemExit(f"backend/config sampling steps mismatch: {path}")
        stream = data["config"]["stream"]
        hop_ms = stream["hop_frames"] / stream["fps"] * 1000
        rows.append({
            "steps": configured_steps,
            "p50_ms": data["inference"]["p50_ms"],
            "p95_ms": data["inference"]["p95_ms"],
            "p99_ms": data["inference"]["p99_ms"],
            "peak_gpu_memory_bytes": data["backend"].get("peak_gpu_memory_bytes"),
            "cuda_p50_ms": data["inference"].get("cuda_p50_ms"),
            "cuda_p95_ms": data["inference"].get("cuda_p95_ms"),
            "cuda_p99_ms": data["inference"].get("cuda_p99_ms"),
            "sample_count": data["inference"]["sample_count"],
            "engine_commit": backend.get("engine_commit"),
            "checkpoint_sha256": backend.get("checkpoint_sha256"),
            "deadline_candidate": data["inference"]["p99_ms"] + 100 < hop_ms,
            "hop_period_ms": hop_ms,
            "summary": str(path),
        })
    passing = [row for row in rows if row["deadline_candidate"]]
    # Leave one full 100 ms safety margin above the measured p99, rounded up to
    # a value that is easy to place in the runtime config.
    for row in rows:
        delay = math.ceil((row["p99_ms"] / 1000.0 + 0.1) * 1000.0) / 1000.0
        row["recommended_playout_delay_s"] = (
            delay if delay * 1000 < row["hop_period_ms"] else None
        )
    result = {
        "decision": "baseline_pass" if passing else "optimization_required",
        "rule": "p99_ms + 100 < hop_period_ms",
        "recommended_baseline": max(passing, key=lambda row: row["steps"]) if passing else None,
        "candidates": rows,
    }
    output = Path(args.root) / args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
