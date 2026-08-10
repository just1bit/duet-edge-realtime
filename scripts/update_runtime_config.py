#!/usr/bin/env python3
"""Update only GPU-selected runtime values in a JSON config."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sampling-steps", required=True, type=int)
    parser.add_argument("--playout-delay-s", required=True, type=float)
    parser.add_argument("--inference-slo-ms", type=float)
    parser.add_argument("--safety-margin-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be positive")
    if not 0 < args.playout_delay_s < 2.5:
        parser.error("--playout-delay-s must be > 0 and < 2.5")
    if args.safety_margin_ms <= 0:
        parser.error("--safety-margin-ms must be positive")

    path = Path(args.config)
    data = json.loads(path.read_text())
    stream = data["stream"]
    hop_period_ms = stream["hop_frames"] / stream["fps"] * 1000.0
    budget_ms = min(args.playout_delay_s * 1000.0, hop_period_ms)
    inference_slo_ms = (
        args.inference_slo_ms
        if args.inference_slo_ms is not None
        else budget_ms - args.safety_margin_ms
    )
    if inference_slo_ms <= 0 or inference_slo_ms + args.safety_margin_ms > budget_ms:
        parser.error(
            "inference SLO plus safety margin must fit playout delay and hop period"
        )
    data["model"]["sampling_steps"] = args.sampling_steps
    stream["playout_delay_s"] = args.playout_delay_s
    stream["inference_slo_ms"] = inference_slo_ms
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {path}")
    print(f"sampling_steps={args.sampling_steps}")
    print(f"playout_delay_s={args.playout_delay_s}")
    print(f"inference_slo_ms={inference_slo_ms}")
    print(f"safety_margin_ms={args.safety_margin_ms}")


if __name__ == "__main__":
    main()
