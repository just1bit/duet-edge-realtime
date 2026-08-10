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
    args = parser.parse_args()
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be positive")
    if not 0 < args.playout_delay_s < 2.5:
        parser.error("--playout-delay-s must be > 0 and < 2.5")

    path = Path(args.config)
    data = json.loads(path.read_text())
    data["model"]["sampling_steps"] = args.sampling_steps
    data["stream"]["playout_delay_s"] = args.playout_delay_s
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {path}")
    print(f"sampling_steps={args.sampling_steps}")
    print(f"playout_delay_s={args.playout_delay_s}")


if __name__ == "__main__":
    main()
