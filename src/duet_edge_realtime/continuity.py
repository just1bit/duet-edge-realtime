from __future__ import annotations

from typing import Protocol

import numpy as np

from .skeleton import forward_kinematics, matrix_to_quaternion, rotation_6d_to_matrix, slerp


class Normalizer(Protocol):
    def unnormalize(self, motion: np.ndarray) -> np.ndarray: ...


class IdentityNormalizer:
    def unnormalize(self, motion: np.ndarray) -> np.ndarray:
        return np.asarray(motion, dtype=np.float32).copy()


class OnlineContinuityProcessor:
    """Online parameter-space alignment and overlap blending for generated windows."""

    def __init__(self, normalizer: Normalizer, horizon: int = 150, hop: int = 75):
        if horizon != 150 or hop != 75:
            raise ValueError("V1 continuity requires 150/75")
        self.normalizer = normalizer
        self.horizon = horizon
        self.hop = hop
        self._pending_pos: np.ndarray | None = None
        self._pending_q: np.ndarray | None = None

    @staticmethod
    def raised_cosine(length: int = 75) -> np.ndarray:
        return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, length)))

    @property
    def pending_frames(self) -> int:
        return 0 if self._pending_pos is None else len(self._pending_pos)

    def process(self, normalized_motion: np.ndarray, commit_frames: int = 75) -> np.ndarray:
        if not 1 <= commit_frames <= self.hop:
            raise ValueError("commit_frames must be in [1,75]")
        motion = np.asarray(self.normalizer.unnormalize(normalized_motion[None]))[0]
        if motion.shape != (150, 151):
            raise ValueError(f"unnormalized chunk must be (150,151), got {motion.shape}")
        roots = motion[:, 4:7].astype(np.float64, copy=True)
        rotations = motion[:, 7:].reshape(150, 24, 6)
        quaternions = matrix_to_quaternion(rotation_6d_to_matrix(rotations))

        if self._pending_pos is None:
            emitted_pos = roots[:self.hop]
            emitted_q = quaternions[:self.hop]
        else:
            offset = self._pending_pos[0] - roots[0]
            roots += offset
            weight = self.raised_cosine(self.hop)[:, None]
            emitted_pos = self._pending_pos * (1.0 - weight) + roots[:self.hop] * weight
            emitted_q = slerp(
                self._pending_q, quaternions[:self.hop], weight[:, None, :]
            )

        self._pending_pos = roots[self.hop:].copy()
        self._pending_q = quaternions[self.hop:].copy()
        joints = forward_kinematics(emitted_q, emitted_pos)
        if not np.isfinite(joints).all():
            raise FloatingPointError("continuity output contains NaN/Inf")
        return joints[:commit_frames]

    def flush(self, frames: int = 75) -> np.ndarray:
        if self._pending_pos is None or self._pending_q is None:
            return np.empty((0, 24, 3), dtype=np.float32)
        if not 0 <= frames <= self.hop:
            raise ValueError("flush frames must be in [0,75]")
        joints = forward_kinematics(self._pending_q, self._pending_pos)[:frames]
        self._pending_pos = None
        self._pending_q = None
        return joints
