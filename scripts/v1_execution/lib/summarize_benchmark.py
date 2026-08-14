#!/usr/bin/env python3
"""Summarize CUDA benchmark runs and calculate deadline recommendations."""

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--pattern", default="benchmark-*/summary.json")
    parser.add_argument("--output", default="evidence/benchmarks/benchmark.json")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    root = Path(args.root)
    rows_by_steps: dict[int, tuple[int, dict]] = {}
    for path in sorted(root.glob(args.pattern)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("exit_reason") != "input_complete":
            raise SystemExit(f"Complete the benchmark input, then repeat: {path}")
        backend = data.get("backend", {})
        if backend.get("backend") != "cuda":
            raise SystemExit(f"Run this benchmark with the CUDA backend: {path}")
        samples = data["inference"].get("sample_count", 0)
        if samples < args.min_samples:
            raise SystemExit(f"Collect at least {args.min_samples} inference samples: {path}")
        steps = data["config"]["model"]["sampling_steps"]
        if args.steps is not None and steps != args.steps:
            continue
        if backend.get("sampling_steps") != steps:
            raise SystemExit(f"Align backend and configured sampling steps: {path}")
        stream = data["config"]["stream"]
        hop_ms = stream["hop_frames"] / stream["fps"] * 1000
        margin = stream["safety_margin_ms"]
        p99 = data["inference"]["p99_ms"]
        delay = math.ceil(p99 + margin) / 1000.0
        row = {
            "steps": steps,
            "p50_ms": data["inference"]["p50_ms"],
            "p95_ms": data["inference"]["p95_ms"],
            "p99_ms": p99,
            "cuda_p50_ms": data["inference"].get("cuda_p50_ms"),
            "cuda_p95_ms": data["inference"].get("cuda_p95_ms"),
            "cuda_p99_ms": data["inference"].get("cuda_p99_ms"),
            "sample_count": samples,
            "peak_gpu_memory_bytes": backend.get("peak_gpu_memory_bytes"),
            "checkpoint_sha256": backend.get("checkpoint_sha256"),
            "deadline_candidate": p99 + margin < hop_ms,
            "safety_margin_ms": margin,
            "hop_period_ms": hop_ms,
            "recommended_playout_delay_s": delay if delay * 1000 < hop_ms else None,
            "summary": str(path),
        }
        modified_at = path.stat().st_mtime_ns
        current = rows_by_steps.get(steps)
        if current is None or modified_at > current[0]:
            rows_by_steps[steps] = (modified_at, row)
    rows = [item[1] for item in sorted(rows_by_steps.values(), key=lambda item: item[1]["steps"], reverse=True)]
    if not rows:
        raise SystemExit(f"Create benchmark summaries matching {args.pattern}.")
    passing = [row for row in rows if row["deadline_candidate"]]
    result = {
        "decision": "baseline_pass" if any(row["steps"] == 50 for row in passing) else "candidate_review",
        "rule": "p99_ms + safety_margin_ms < hop_period_ms",
        "recommended_candidate": max(passing, key=lambda row: row["steps"]) if passing else None,
        "candidates": rows,
    }
    target = root / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
