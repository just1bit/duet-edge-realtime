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
    """V2 companion stitcher using local relative-root correction and SLERP."""

    def __init__(
        self,
        normalizer: Normalizer,
        horizon: int = 150,
        hop: int = 75,
        *,
        robust_filter_z: float = 6.0,
    ):
        if horizon != 150 or hop != 75:
            raise ValueError("V2 continuity requires 150/75")
        self.normalizer = normalizer
        self.horizon = horizon
        self.hop = hop
        self.robust_filter_z = robust_filter_z
        self._pending_relative_root: np.ndarray | None = None
        self._pending_q: np.ndarray | None = None
        self.last_metrics: dict = {}

    @staticmethod
    def raised_cosine(length: int = 75) -> np.ndarray:
        return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, length)))

    @property
    def pending_frames(self) -> int:
        return 0 if self._pending_relative_root is None else len(self._pending_relative_root)

    def process(
        self,
        normalized_motion: np.ndarray,
        commit_frames: int = 75,
        *,
        lead_motion: np.ndarray | None = None,
    ) -> np.ndarray:
        if not 1 <= commit_frames <= self.hop:
            raise ValueError("commit_frames must be in [1,75]")
        motion = np.asarray(self.normalizer.unnormalize(normalized_motion[None]))[0]
        if motion.shape != (150, 151):
            raise ValueError(f"unnormalized chunk must be (150,151), got {motion.shape}")
        roots = motion[:, 4:7].astype(np.float64, copy=True)
        rotations = motion[:, 7:].reshape(150, 24, 6)
        quaternions = matrix_to_quaternion(rotation_6d_to_matrix(rotations))

        # The compatibility path remains useful for isolated parameter-stitch
        # tests. Runtime V2 always supplies the authoritative lead window.
        if lead_motion is None:
            lead_roots = np.zeros_like(roots)
            relative_root = roots
            legacy_alignment = True
        else:
            lead = np.asarray(self.normalizer.unnormalize(lead_motion[None]))[0]
            if lead.shape != (150, 151):
                raise ValueError(f"unnormalized lead must be (150,151), got {lead.shape}")
            lead_roots = lead[:, 4:7].astype(np.float64, copy=False)
            relative_root = self._filter_relative_root(roots - lead_roots)
            legacy_alignment = False

        if self._pending_relative_root is None:
            emitted_relative_root = relative_root[:self.hop]
            emitted_q = quaternions[:self.hop]
            correction = np.zeros(3, dtype=np.float64)
            disagreement = None
        else:
            weight = self.raised_cosine(self.hop)[:, None]
            correction = self._pending_relative_root[0] - relative_root[0]
            if legacy_alignment:
                relative_root = relative_root + correction
                corrected = relative_root[:self.hop]
            else:
                corrected = relative_root[:self.hop] + (1.0 - weight) * correction
            delta = self._pending_relative_root - relative_root[:self.hop]
            scale = max(float(np.std(self._pending_relative_root)), 1e-6)
            disagreement = float(np.sqrt(np.mean(delta * delta)) / scale)
            emitted_relative_root = (
                self._pending_relative_root * (1.0 - weight) + corrected * weight
            )
            emitted_q = slerp(
                self._pending_q, quaternions[:self.hop], weight[:, None, :]
            )

        self._pending_relative_root = relative_root[self.hop:].copy()
        self._pending_q = quaternions[self.hop:].copy()
        emitted_roots = lead_roots[:self.hop] + emitted_relative_root
        joints = forward_kinematics(emitted_q, emitted_roots)
        self.last_metrics = {
            "correction_root_l2": float(np.linalg.norm(correction)),
            "normalized_overlap_disagreement": disagreement,
            "relative_root_horizontal_max": float(
                np.max(np.linalg.norm(emitted_relative_root[:, :2], axis=1))
            ),
            "relative_root_vertical_min": float(np.min(emitted_relative_root[:, 2])),
            "relative_root_vertical_max": float(np.max(emitted_relative_root[:, 2])),
        }
        if not np.isfinite(joints).all():
            raise FloatingPointError("continuity output contains NaN/Inf")
        return joints[:commit_frames]

    def flush(self, frames: int = 75) -> np.ndarray:
        if self._pending_relative_root is None or self._pending_q is None:
            return np.empty((0, 24, 3), dtype=np.float32)
        if not 0 <= frames <= self.hop:
            raise ValueError("flush frames must be in [0,75]")
        # Runtime calls flush_with_lead so pending relative roots are placed in
        # the same authoritative source coordinate frame.
        joints = forward_kinematics(self._pending_q, self._pending_relative_root)[:frames]
        self._pending_relative_root = None
        self._pending_q = None
        return joints

    def flush_with_lead(self, lead_motion: np.ndarray, frames: int = 75) -> np.ndarray:
        if self._pending_relative_root is None or self._pending_q is None:
            return np.empty((0, 24, 3), dtype=np.float32)
        if not 0 <= frames <= self.hop:
            raise ValueError("flush frames must be in [0,75]")
        lead = np.asarray(self.normalizer.unnormalize(lead_motion[None]))[0]
        roots = lead[self.hop:self.hop + frames, 4:7] + self._pending_relative_root[:frames]
        joints = forward_kinematics(self._pending_q[:frames], roots)
        self._pending_relative_root = None
        self._pending_q = None
        return joints

    def reset(self) -> None:
        self._pending_relative_root = None
        self._pending_q = None
        self.last_metrics = {}

    def _filter_relative_root(self, roots: np.ndarray) -> np.ndarray:
        if len(roots) < 3:
            return roots.copy()
        steps = np.diff(roots, axis=0)
        median = np.median(steps, axis=0)
        mad = np.median(np.abs(steps - median), axis=0)
        limit = self.robust_filter_z * np.maximum(1.4826 * mad, 1e-6)
        filtered_steps = np.clip(steps, median - limit, median + limit)
        return np.concatenate((roots[:1], roots[:1] + np.cumsum(filtered_steps, axis=0)))


def direct_fk(normalizer: Normalizer, normalized_motion: np.ndarray) -> np.ndarray:
    """Convert canonical source motion to joints without cross-window mutation."""
    motion = np.asarray(normalizer.unnormalize(normalized_motion[None]))[0]
    if motion.shape != (150, 151):
        raise ValueError(f"unnormalized motion must be (150,151), got {motion.shape}")
    roots = motion[:, 4:7].astype(np.float64, copy=False)
    rotations = motion[:, 7:].reshape(150, 24, 6)
    quaternions = matrix_to_quaternion(rotation_6d_to_matrix(rotations))
    joints = forward_kinematics(quaternions, roots)
    if not np.isfinite(joints).all():
        raise FloatingPointError("direct FK contains NaN/Inf")
    return joints
