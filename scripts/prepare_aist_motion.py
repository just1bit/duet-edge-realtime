#!/usr/bin/env python3
"""Convert one raw AIST++ motion pickle to the realtime adapter contract."""

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw AIST++ .pkl")
    parser.add_argument("--output", required=True, help="output .pkl with pos/q/scale")
    args = parser.parse_args()
    with Path(args.input).open("rb") as handle:
        source = pickle.load(handle)
    if {"pos", "q", "scale"} <= set(source):
        pos, rotations, scale = source["pos"], source["q"], source["scale"]
    elif {"smpl_trans", "smpl_poses", "smpl_scaling"} <= set(source):
        pos = source["smpl_trans"]
        rotations = source["smpl_poses"]
        scale = source["smpl_scaling"]
    else:
        raise ValueError(
            "input must contain pos/q/scale or smpl_trans/smpl_poses/smpl_scaling"
        )
    pos = np.asarray(pos)
    rotations = np.asarray(rotations)
    scale = np.asarray(scale)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"root positions must be [N,3], got {pos.shape}")
    if rotations.shape[0] != pos.shape[0] or not (
        (rotations.ndim == 2 and rotations.shape[1] == 72)
        or (rotations.ndim == 3 and rotations.shape[1:] == (24, 3))
    ):
        raise ValueError(f"rotations do not match positions: {rotations.shape}")
    if not np.isfinite(pos).all() or not np.isfinite(rotations).all():
        raise ValueError("positions/rotations contain NaN/Inf")
    if scale.size == 0 or not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("invalid scale")
    # Raw AIST++ motion is 60 FPS and the Duet-EDGE preprocessing path
    # downsamples it to 30 FPS. V1 needs at least one 150-frame window.
    if len(pos) < 300:
        raise ValueError(
            f"V1 needs at least 300 raw 60 FPS frames, received {len(pos)}"
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"pos": pos, "q": rotations, "scale": scale}, handle)
    print(
        f"wrote {target} ({len(pos)} raw frames, "
        f"estimated {len(pos) // 2} frames at 30 FPS, root_scaled=false)"
    )


if __name__ == "__main__":
    main()
