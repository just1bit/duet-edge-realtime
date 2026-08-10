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
    if rotations.ndim not in (2, 3) or rotations.shape[0] != pos.shape[0]:
        raise ValueError(f"rotations do not match positions: {rotations.shape}")
    if scale.size == 0 or not np.isfinite(scale).all():
        raise ValueError("invalid scale")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"pos": pos, "q": rotations, "scale": scale}, handle)
    print(f"wrote {target} ({len(pos)} raw frames, root_scaled=false)")


if __name__ == "__main__":
    main()
