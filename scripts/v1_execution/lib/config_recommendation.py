#!/usr/bin/env python3
"""Show or validate a manually selected runtime configuration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def select_candidate(benchmark: dict, steps: int) -> dict:
    matches = [row for row in benchmark.get("candidates", []) if row.get("steps") == steps]
    if len(matches) != 1:
        raise ValueError(f"Generate exactly one benchmark result for {steps} steps.")
    return matches[0]


def recommendation(candidate: dict) -> dict:
    required = (
        "p99_ms", "measured_max_ms", "inference_reserve_ms",
        "safety_margin_ms", "recommended_inference_slo_ms",
        "recommended_playout_delay_s",
    )
    values = {name: candidate.get(name) for name in required}
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values.values()):
        raise ValueError("Generate a complete realtime benchmark recommendation.")
    return {
        "sampling_steps": candidate["steps"],
        "measured_p99_ms": values["p99_ms"],
        "measured_max_ms": values["measured_max_ms"],
        "inference_reserve_ms": values["inference_reserve_ms"],
        "safety_margin_ms": values["safety_margin_ms"],
        "playout_delay_s": values["recommended_playout_delay_s"],
        "inference_slo_ms": values["recommended_inference_slo_ms"],
        "summary": candidate["summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--quality")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    if benchmark.get("schema_version") != "2.0":
        raise ValueError("Generate a version 2 realtime benchmark summary.")
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        steps = config["model"]["sampling_steps"]
    else:
        config = None
        steps = args.steps
    if steps is None:
        parser.error("provide --steps or --config")
    candidate = select_candidate(benchmark, steps)
    suggested = recommendation(candidate)
    payload = {"deadline_candidate": candidate.get("deadline_candidate"), **suggested}
    if args.quality:
        quality = json.loads(Path(args.quality).read_text(encoding="utf-8"))
        payload["quality_passed"] = quality.get("passed") is True
    if args.validate:
        if config is None:
            parser.error("--validate uses --config")
        actions = []
        stream = config["stream"]
        if candidate.get("deadline_candidate") is not True:
            actions.append("Select a benchmark candidate that meets the deadline rule.")
        if steps < 50 and payload.get("quality_passed") is not True:
            actions.append("Select a candidate with a passing quality result.")
        if stream["safety_margin_ms"] != suggested["safety_margin_ms"]:
            actions.append("Align the configuration and benchmark safety margins.")
        if stream["playout_delay_s"] != suggested["playout_delay_s"]:
            actions.append("Set playout_delay_s to the reviewed recommendation.")
        if stream["inference_slo_ms"] != suggested["inference_slo_ms"]:
            actions.append("Set inference_slo_ms to the reviewed recommendation.")
        payload["passed"] = not actions
        payload["actions"] = actions
        print(json.dumps(payload, indent=2))
        raise SystemExit(0 if not actions else 1)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
