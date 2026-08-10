#!/usr/bin/env python3
import argparse
import json
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
        rows.append({
            "steps": data["config"]["model"]["sampling_steps"],
            "p50_ms": data["inference"]["p50_ms"],
            "p95_ms": data["inference"]["p95_ms"],
            "p99_ms": data["inference"]["p99_ms"],
            "peak_gpu_memory_bytes": data["backend"].get("peak_gpu_memory_bytes"),
            "deadline_candidate": data["inference"]["p99_ms"] + 100 < 2500,
            "summary": str(path),
        })
    output = Path(args.root) / args.output
    output.write_text(json.dumps({"candidates": rows}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
