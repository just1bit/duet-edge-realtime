from __future__ import annotations

import math
from collections import deque

import numpy as np


def _percentile(values, fraction: float):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


class OnlineMotionQuality:
    """Bounded motion, boundary, spatial, foot, and ground diagnostics."""

    def __init__(self, fps: int = 30, sample_limit: int = 4096):
        self.fps = fps
        self.sample_limit = sample_limit
        self.root_step = deque(maxlen=sample_limit)
        self.root_velocity = deque(maxlen=sample_limit)
        self.root_acceleration = deque(maxlen=sample_limit)
        self.joint_step = deque(maxlen=sample_limit)
        self.joint_angle_step = deque(maxlen=sample_limit)
        self.distinctness = deque(maxlen=sample_limit)
        self.relative_horizontal = deque(maxlen=sample_limit)
        self.relative_vertical = deque(maxlen=sample_limit)
        self.foot_contact_velocity = deque(maxlen=sample_limit)
        self.ground_penetration = deque(maxlen=sample_limit)
        self.model_boundaries = deque(maxlen=256)
        self.source_transitions = deque(maxlen=256)
        self.corrections = deque(maxlen=256)
        self.overlap_disagreement = deque(maxlen=256)
        self._previous_companion = None
        self._previous_velocity = None
        self._ground_z = None
        self._first_relative = None
        self._last_relative = None

    def record_frame(
        self,
        frame_id: int,
        lead_joints,
        companion_joints,
        *,
        model_boundary: bool = False,
        source_transition: bool = False,
    ) -> None:
        lead = np.asarray(lead_joints, dtype=np.float64)
        companion = np.asarray(companion_joints, dtype=np.float64)
        relative = companion[0] - lead[0]
        if self._first_relative is None:
            self._first_relative = relative.copy()
        self._last_relative = relative.copy()
        self.relative_horizontal.append(float(np.linalg.norm(relative[:2])))
        self.relative_vertical.append(float(relative[2]))
        centered_lead = lead - lead[:1]
        centered_companion = companion - companion[:1]
        self.distinctness.append(
            float(np.sqrt(np.mean((centered_lead - centered_companion) ** 2)))
        )

        feet = companion[[7, 8, 10, 11]]
        if self._ground_z is None:
            self._ground_z = float(np.min(feet[:, 2]))
        self.ground_penetration.append(
            float(max(0.0, self._ground_z - float(np.min(feet[:, 2]))))
        )

        boundary_sample = None
        if self._previous_companion is not None:
            root_delta = companion[0] - self._previous_companion[0]
            step = float(np.linalg.norm(root_delta))
            velocity = root_delta * self.fps
            acceleration = (
                np.zeros(3) if self._previous_velocity is None
                else (velocity - self._previous_velocity) * self.fps
            )
            joint_delta = np.linalg.norm(companion - self._previous_companion, axis=1)
            angles = self._bone_angle_steps(self._previous_companion, companion)
            self.root_step.append(step)
            self.root_velocity.append(float(np.linalg.norm(velocity)))
            self.root_acceleration.append(float(np.linalg.norm(acceleration)))
            self.joint_step.append(float(np.max(joint_delta)))
            self.joint_angle_step.append(float(np.max(angles)))
            previous_feet = self._previous_companion[[7, 8, 10, 11]]
            foot_speed = np.linalg.norm(feet - previous_feet, axis=1) * self.fps
            contact = feet[:, 2] <= self._ground_z + 0.05
            if np.any(contact):
                self.foot_contact_velocity.extend(foot_speed[contact].tolist())
            boundary_sample = {
                "frame_id": frame_id,
                "root_position_step": step,
                "root_velocity": float(np.linalg.norm(velocity)),
                "root_acceleration": float(np.linalg.norm(acceleration)),
                "joint_position_step_max": float(np.max(joint_delta)),
                "joint_angle_step_max_rad": float(np.max(angles)),
            }
            self._previous_velocity = velocity
        self._previous_companion = companion.copy()
        if boundary_sample is not None:
            if model_boundary:
                self.model_boundaries.append(boundary_sample)
            if source_transition:
                self.source_transitions.append(boundary_sample)

    @staticmethod
    def _bone_angle_steps(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        # Parent indices are duplicated here to keep this hot path allocation-free.
        parents = np.asarray([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                              13, 14, 16, 17, 18, 19, 20, 21])
        valid = parents >= 0
        left = previous[valid] - previous[parents[valid]]
        right = current[valid] - current[parents[valid]]
        dot = np.sum(left * right, axis=1)
        denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return np.arccos(np.clip(dot / np.maximum(denom, 1e-9), -1.0, 1.0))

    def record_window(self, diagnostics: dict, chunk) -> None:
        correction = diagnostics.get("correction_root_l2")
        if correction is not None:
            self.corrections.append(float(correction))
        disagreement = diagnostics.get("normalized_overlap_disagreement")
        if disagreement is None:
            disagreement = getattr(chunk, "normalized_overlap_disagreement", None)
        if disagreement is not None and math.isfinite(disagreement):
            self.overlap_disagreement.append(float(disagreement))

    @staticmethod
    def _distribution(values) -> dict:
        return {
            "count": len(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    def summary(self) -> dict:
        relative_trend = None
        if self._first_relative is not None and self._last_relative is not None:
            relative_trend = (self._last_relative - self._first_relative).tolist()
        return {
            "root_position_step": self._distribution(self.root_step),
            "root_velocity": self._distribution(self.root_velocity),
            "root_acceleration": self._distribution(self.root_acceleration),
            "joint_position_step": self._distribution(self.joint_step),
            "joint_angle_step_rad": self._distribution(self.joint_angle_step),
            "distinctness_body_centered": self._distribution(self.distinctness),
            "relative_root_horizontal": self._distribution(self.relative_horizontal),
            "relative_root_vertical": self._distribution(self.relative_vertical),
            "relative_root_trend": relative_trend,
            "foot_velocity_during_contact": self._distribution(self.foot_contact_velocity),
            "ground_penetration": self._distribution(self.ground_penetration),
            "continuity_correction": self._distribution(self.corrections),
            "normalized_overlap_disagreement": self._distribution(
                self.overlap_disagreement
            ),
            "model_boundaries": list(self.model_boundaries),
            "source_transitions": list(self.source_transitions),
            "sample_limit": self.sample_limit,
        }
