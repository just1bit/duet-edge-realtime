#!/usr/bin/env python3
"""Convert one raw AIST++ motion pickle to the realtime adapter contract."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw AIST++ pickle")
    parser.add_argument("--output", required=True, help="output pickle with pos/q/scale")
    parser.add_argument("--metadata", help="optional JSON evidence output")
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
        raise ValueError("Provide pos/q/scale or smpl_trans/smpl_poses/smpl_scaling.")
    pos = np.asarray(pos)
    rotations = np.asarray(rotations)
    scale = np.asarray(scale)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Use root positions shaped [N,3]; received {pos.shape}.")
    if rotations.shape[0] != pos.shape[0] or not (
        (rotations.ndim == 2 and rotations.shape[1] == 72)
        or (rotations.ndim == 3 and rotations.shape[1:] == (24, 3))
    ):
        raise ValueError(f"Align rotations with root positions; received {rotations.shape}.")
    if not np.isfinite(pos).all() or not np.isfinite(rotations).all():
        raise ValueError("Use finite position and rotation values.")
    if scale.size == 0 or not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("Use a finite, non-zero scale.")
    if len(pos) < 300:
        raise ValueError(f"Provide at least 300 raw 60 FPS frames; received {len(pos)}.")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump({"pos": pos, "q": rotations, "scale": scale}, handle)
    print(
        f"Wrote {target} ({len(pos)} raw frames, "
        f"estimated {len(pos) // 2} frames at 30 FPS, root_scaled=false)."
    )
    if args.metadata:
        digest = hashlib.sha256()
        with Path(args.input).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        metadata = {
            "source": str(Path(args.input).resolve()),
            "source_sha256": digest.hexdigest(),
            "output": str(target.resolve()),
            "raw_frames_60fps": len(pos),
            "estimated_frames_30fps": len(pos) // 2,
            "position_shape": list(pos.shape),
            "rotation_shape": list(rotations.shape),
            "scale_shape": list(scale.shape),
            "root_scaled": False,
            "finite": True,
        }
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
