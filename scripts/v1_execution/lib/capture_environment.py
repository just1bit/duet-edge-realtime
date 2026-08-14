#!/usr/bin/env python3
"""Capture reproducible Python and PyTorch environment evidence."""

import argparse
import subprocess
import sys
from pathlib import Path


def capture(command: list[str], target: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    target.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("gpu", "local"), default="gpu")
    args = parser.parse_args()
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    capture([sys.executable, "-m", "pip", "freeze"], target / "pip-freeze.txt")
    if args.profile == "gpu":
        capture(
            [sys.executable, "-m", "torch.utils.collect_env"],
            target / "torch-environment.txt",
        )
    print(target)


if __name__ == "__main__":
    main()
