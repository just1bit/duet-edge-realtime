#!/usr/bin/env python3
"""Update only GPU-selected runtime values in a JSON config."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--sampling-steps", required=True, type=int)
    parser.add_argument("--quality")
    args = parser.parse_args()
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be positive")

    path = Path(args.config)
    data = json.loads(path.read_text())
    benchmark_path = Path(args.benchmark)
    benchmark = json.loads(benchmark_path.read_text())
    matches = [
        candidate for candidate in benchmark.get("candidates", [])
        if candidate.get("steps") == args.sampling_steps
    ]
    if len(matches) != 1:
        parser.error(
            f"benchmark must contain exactly one {args.sampling_steps}-step candidate"
        )
    candidate = matches[0]
    if candidate.get("deadline_candidate") is not True:
        parser.error("selected benchmark candidate does not meet the deadline budget")
    playout_delay_s = candidate.get("recommended_playout_delay_s")
    if not isinstance(playout_delay_s, (int, float)) or not math.isfinite(playout_delay_s):
        parser.error("selected benchmark candidate has no recommended playout delay")

    quality_path = Path(args.quality) if args.quality else None
    if args.sampling_steps < 50:
        if quality_path is None:
            parser.error("low-step candidates require --quality")
        quality = json.loads(quality_path.read_text())
        if quality.get("passed") is not True:
            parser.error("selected quality result did not pass")

    stream = data["stream"]
    safety_margin_ms = stream["safety_margin_ms"]
    if candidate.get("safety_margin_ms") != safety_margin_ms:
        parser.error("benchmark/config safety margins do not match")
    hop_period_ms = stream["hop_frames"] / stream["fps"] * 1000.0
    if not 0 < playout_delay_s * 1000.0 < hop_period_ms:
        parser.error("recommended playout delay must fit within the hop period")
    budget_ms = min(playout_delay_s * 1000.0, hop_period_ms)
    inference_slo_ms = budget_ms - safety_margin_ms
    if inference_slo_ms <= 0 or inference_slo_ms + safety_margin_ms > budget_ms:
        parser.error(
            "inference SLO plus safety margin must fit playout delay and hop period"
        )
    data["model"]["sampling_steps"] = args.sampling_steps
    stream["playout_delay_s"] = playout_delay_s
    stream["inference_slo_ms"] = inference_slo_ms
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {path}")
    print(f"benchmark={benchmark_path}")
    if quality_path is not None:
        print(f"quality={quality_path}")
    print(f"sampling_steps={args.sampling_steps}")
    print(f"playout_delay_s={playout_delay_s}")
    print(f"inference_slo_ms={inference_slo_ms}")
    print(f"safety_margin_ms={safety_margin_ms}")


if __name__ == "__main__":
    main()
