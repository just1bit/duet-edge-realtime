#!/usr/bin/env python3
"""Check the automatic artifacts required by an acceptance profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMMON_ACTIONS = (
    "02-verify-runtime-*.json", "02-capture-environment-*.json",
    "03-preflight-*.json", "04-prepare-input-*.json",
    "05-unit-tests-*.json", "05-network-tests-*.json",
    "06-run-fake-*.json", "06-check-fake-*.json",
    "09-viewer-stream-*.json", "09-viewer-web-*.json",
)

GPU_ACTIONS = (
    "02-cuda-smoke-*.json", "07-run-real-*.json", "07-check-real-*.json",
    "08-export-fixture-*.json", "10-run-baseline-*.json",
    "10-summarize-baseline-*.json", "12-show-recommendation-*.json",
    "12-validate-config-*.json", "13-monitor-gpu-*.json",
    "13-run-final-*.json", "14-check-final-*.json",
    "14-summarize-resources-*.json",
)


def latest_result(root: Path, pattern: str) -> tuple[Path, dict] | None:
    matches = list((root / "stage-results").glob(pattern))
    if not matches:
        return None
    path = max(matches, key=lambda item: item.stat().st_mtime_ns)
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_run(root: Path, base: str) -> Path:
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and (path.name == base or path.name.startswith(f"{base}-attempt-"))
    ]
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else root / base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--profile", choices=("gpu", "local"), required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    actions: list[str] = []

    fake_run = latest_run(root, "p1-fake")
    viewer_run = latest_run(root, "viewer")
    required_paths = [
        root / "run-metadata.json", root / "acceptance-notes.md",
        root / "stage-results", root / "logs",
        root / "evidence" / "preflight" / "preflight.json",
        root / "evidence" / "input-motion.json", root / "input_motion.pkl",
        fake_run / "summary.json", fake_run / "stream.ndjson",
        viewer_run / "summary.json", viewer_run / "stream.ndjson",
    ]
    if args.profile == "gpu":
        real_run = latest_run(root, "real-smoke")
        final_run = latest_run(root, "final-10min")
        resource_files = list((root / "evidence" / "resources").glob("gpu-resources*.csv"))
        resource_csv = max(resource_files, key=lambda path: path.stat().st_mtime_ns) if resource_files else root / "evidence" / "resources" / "gpu-resources.csv"
        required_paths.extend([
            root / "real_fixture.npz",
            root / "evidence" / "benchmarks" / "benchmark.json",
            real_run / "summary.json", real_run / "stream.ndjson",
            final_run / "summary.json", final_run / "stream.ndjson",
            resource_csv, root / "evidence" / "resources" / "gpu-summary.json",
        ])

    for path in required_paths:
        if not path.exists():
            actions.append(f"Create the required evidence artifact: {path}")

    patterns = COMMON_ACTIONS + (GPU_ACTIONS if args.profile == "gpu" else ())
    for pattern in patterns:
        selected = latest_result(root, pattern)
        if selected is None:
            actions.append(f"Run the required acceptance action matching {pattern}.")
            continue
        path, result = selected
        if result.get("passed") is not True or result.get("skipped") is True:
            actions.append(f"Complete the required acceptance action successfully: {path}")

    metadata_path = root / "run-metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("acceptance_profile") != args.profile:
            actions.append("Use the same acceptance profile recorded when the run was initialized.")

    result = {"profile": args.profile, "passed": not actions, "actions": actions}
    target = root / "evidence" / "evidence-check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not actions else 1)


if __name__ == "__main__":
    main()
