#!/usr/bin/env python3
"""Generate the deterministic fake fixture used by development tests."""

import argparse
import json
from pathlib import Path

import numpy as np


def rotation_matrix(axis: str, angle: np.ndarray) -> np.ndarray:
    """Return standard local rotation matrices for a vector of angles."""
    angle = np.asarray(angle, dtype=np.float32)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.zeros(angle.shape + (3, 3), dtype=np.float32)
    if axis == "x":
        result[..., 0, 0] = 1
        result[..., 1, 1] = cosine
        result[..., 1, 2] = -sine
        result[..., 2, 1] = sine
        result[..., 2, 2] = cosine
    elif axis == "y":
        result[..., 0, 0] = cosine
        result[..., 0, 2] = sine
        result[..., 1, 1] = 1
        result[..., 2, 0] = -sine
        result[..., 2, 2] = cosine
    elif axis == "z":
        result[..., 0, 0] = cosine
        result[..., 0, 1] = -sine
        result[..., 1, 0] = sine
        result[..., 1, 1] = cosine
        result[..., 2, 2] = 1
    else:
        raise ValueError(f"unknown axis: {axis}")
    return result


def matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    # skeleton.rotation_6d_to_matrix interprets the two vectors as matrix rows.
    return np.asarray(matrix, dtype=np.float32)[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--output", default="tests/fixtures/fake_motion.npz")
    args = parser.parse_args()
    if args.frames < 150:
        parser.error("choose at least 150 frames")
    motion = np.zeros((args.frames, 151), dtype=np.float32)
    timeline = np.arange(args.frames, dtype=np.float32) / 30
    # EDGE motions are Z-up. SMPL's rest offsets are Y-up, so the root carries
    # the same +90 degree X rotation applied by dataset preprocessing.
    motion[:, 4] = 0.25 * np.sin(timeline * 0.55)
    motion[:, 5] = 0.12 * np.cos(timeline * 0.40)
    motion[:, 6] = 1.03 + 0.025 * np.sin(timeline * 3.2)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (args.frames, 24, 3, 3)).copy()
    base = rotation_matrix("x", np.full(args.frames, np.pi / 2, dtype=np.float32))
    rotations[:, 0] = base @ rotation_matrix("y", 0.12 * np.sin(timeline * 0.7))

    # A deterministic dance-like cycle. It deliberately animates limbs relative
    # to the root so a frozen or root-translation-only Viewer cannot pass review.
    beat = timeline * 2.4
    rotations[:, 3] = rotation_matrix("z", 0.12 * np.sin(beat))
    rotations[:, 6] = rotation_matrix("z", 0.10 * np.sin(beat + 0.5))
    rotations[:, 16] = rotation_matrix("z", 0.65 + 0.35 * np.sin(beat))
    rotations[:, 17] = rotation_matrix("z", -0.65 - 0.35 * np.sin(beat))
    rotations[:, 18] = rotation_matrix("z", 0.55 + 0.35 * np.sin(beat + 0.8))
    rotations[:, 19] = rotation_matrix("z", -0.55 - 0.35 * np.sin(beat + 0.8))
    rotations[:, 1] = rotation_matrix("x", 0.25 * np.sin(beat))
    rotations[:, 2] = rotation_matrix("x", -0.25 * np.sin(beat))
    rotations[:, 4] = rotation_matrix("x", 0.30 + 0.22 * np.maximum(0, np.sin(beat)))
    rotations[:, 5] = rotation_matrix("x", 0.30 + 0.22 * np.maximum(0, -np.sin(beat)))
    motion[:, 7:] = matrix_to_6d(rotations).reshape(args.frames, -1)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"source": "generated_fake", "normalized": True, "fps": 30})
    np.savez_compressed(target, motion_151=motion, metadata_json=metadata)
    print(target)


if __name__ == "__main__":
    main()
