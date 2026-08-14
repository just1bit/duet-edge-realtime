#!/usr/bin/env python3
"""Create a complete candidate JSON from the canonical CUDA configuration."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("Choose sampling steps from 1 through 1000.")
    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    data["model"]["sampling_steps"] = args.steps
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
