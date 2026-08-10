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
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.root).glob(args.pattern)):
        data = json.loads(path.read_text())
        stream = data["config"]["stream"]
        hop_ms = stream["hop_frames"] / stream["fps"] * 1000
        rows.append({
            "steps": data["config"]["model"]["sampling_steps"],
            "p50_ms": data["inference"]["p50_ms"],
            "p95_ms": data["inference"]["p95_ms"],
            "p99_ms": data["inference"]["p99_ms"],
            "peak_gpu_memory_bytes": data["backend"].get("peak_gpu_memory_bytes"),
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
        "recommended_baseline": passing[0] if passing else None,
        "candidates": rows,
    }
    output = Path(args.root) / args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
