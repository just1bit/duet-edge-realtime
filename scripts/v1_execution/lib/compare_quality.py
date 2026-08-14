#!/usr/bin/env python3
"""Compare the motion quality of one candidate with the 50-step baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def read_frames(path: Path) -> np.ndarray:
    frames = [json.loads(line)["joints"] for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("type") == "frame"]
    result = np.asarray(frames, dtype=np.float64)
    if result.ndim != 3 or result.shape[1:] != (24, 3) or not np.isfinite(result).all():
        raise ValueError(f"Use finite [N,24,3] frames from {path}.")
    return result


def pfc(joints: np.ndarray) -> float:
    scores = []
    for segment in np.array_split(joints, max(1, len(joints) // 150)):
        if len(segment) < 3:
            continue
        root_v = np.diff(segment[:, 0], axis=0) * 30
        root_a = np.diff(root_v, axis=0) * 30
        root_a[:, 2] = np.maximum(root_a[:, 2], 0)
        root_a = np.linalg.norm(root_a, axis=-1)
        if root_a.max() == 0:
            continue
        root_a /= root_a.max()
        feet = segment[:, [7, 10, 8, 11]]
        foot_v = np.linalg.norm(feet[2:, :, :2] - feet[1:-1, :, :2], axis=-1)
        scores.append(float((np.minimum(foot_v[:, 0], foot_v[:, 1]) * np.minimum(foot_v[:, 2], foot_v[:, 3]) * root_a).mean()))
    return float(np.mean(scores) * 10000) if scores else float("nan")


def boundary_motion(joints: np.ndarray) -> float:
    values = [np.linalg.norm(joints[index] - joints[index - 1], axis=-1).max() for index in range(75, len(joints), 75)]
    return float(np.mean(values)) if values else 0.0


def relative_increase(candidate: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0 if abs(candidate) < 1e-12 else float("inf")
    return (candidate - baseline) / abs(baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--baseline-ndjson", required=True)
    parser.add_argument("--candidate-ndjson", required=True)
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.duet_edge_root).resolve()))
    from eval.run_lma17_sweep import lma_for_slice
    with np.load(args.fixture, allow_pickle=False) as fixture:
        lead = np.asarray(fixture["lead_joints"], dtype=np.float64)
    baseline = read_frames(Path(args.baseline_ndjson))
    candidate = read_frames(Path(args.candidate_ndjson))
    length = min(len(baseline), len(candidate))
    length -= length % len(lead)
    if length == 0:
        raise ValueError("Generate at least one complete fixture cycle for both streams.")
    baseline, candidate = baseline[:length], candidate[:length]
    tiled_lead = np.tile(lead, (length // len(lead), 1, 1))

    def lma(joints: np.ndarray) -> float:
        values = [lma_for_slice(tiled_lead[start:start + len(lead)], joints[start:start + len(lead)])[0] for start in range(0, length, len(lead))]
        return float(np.mean(values))

    baseline_metrics = {"lma": lma(baseline), "pfc": pfc(baseline), "boundary_motion": boundary_motion(baseline)}
    candidate_metrics = {"lma": lma(candidate), "pfc": pfc(candidate), "boundary_motion": boundary_motion(candidate)}
    deltas = {
        "lma_drop": baseline_metrics["lma"] - candidate_metrics["lma"],
        "pfc_relative_increase": relative_increase(candidate_metrics["pfc"], baseline_metrics["pfc"]),
        "boundary_relative_increase": relative_increase(candidate_metrics["boundary_motion"], baseline_metrics["boundary_motion"]),
    }
    checks = {
        "lma_drop_le_0.02": deltas["lma_drop"] <= 0.02,
        "pfc_increase_le_10pct": deltas["pfc_relative_increase"] <= 0.10,
        "boundary_increase_le_10pct": deltas["boundary_relative_increase"] <= 0.10,
    }
    result = {"baseline": baseline_metrics, "candidate": candidate_metrics, "deltas": deltas, "checks": checks}
    result["passed"] = all(checks.values()) and all(math.isfinite(value) for value in (*baseline_metrics.values(), *candidate_metrics.values()))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
