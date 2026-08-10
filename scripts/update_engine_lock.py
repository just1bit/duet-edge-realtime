#!/usr/bin/env python3
"""Pin compat/duet-edge.lock.json to a verified external repository commit."""

import argparse
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--lock", default="compat/duet-edge.lock.json")
    args = parser.parse_args()
    root = Path(args.duet_edge_root).resolve()
    lock = Path(args.lock)
    commit = git(root, "rev-parse", "HEAD")
    dirty = git(root, "status", "--porcelain", "--", "*.py")
    if dirty:
        raise SystemExit("refusing to lock: external duet-edge has modified Python files")

    data = json.loads(lock.read_text())
    data["commit"] = commit
    lock.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {lock} -> {commit}")


if __name__ == "__main__":
    main()
