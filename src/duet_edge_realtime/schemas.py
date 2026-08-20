from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROTOCOL_NAME = "duet-edge-stream/v2"
SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class MotionFrame:
    seq: int
    source_time_s: float
    motion_151: np.ndarray
    source_id: str = "lead-motion"
    schema_version: str = SCHEMA_VERSION
    ingest_monotonic_s: float | None = None

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not np.isfinite(self.source_time_s) or self.source_time_s < 0:
            raise ValueError("source_time_s must be finite and non-negative")
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        motion = np.asarray(self.motion_151, dtype=np.float32)
        if motion.shape != (151,):
            raise ValueError(f"motion_151 must have shape (151,), got {motion.shape}")
        if not np.isfinite(motion).all():
            raise ValueError("motion_151 contains NaN/Inf")
        object.__setattr__(self, "motion_151", motion)


@dataclass(frozen=True)
class MotionWindow:
    window_id: int
    start_seq: int
    end_seq: int
    trigger_time_s: float
    seed: int
    motion: np.ndarray
    valid_frames: int = 150
    first_source_time_s: float | None = None
    last_source_time_s: float | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion, dtype=np.float32)
        if motion.shape != (150, 151):
            raise ValueError(f"window motion must be (150,151), got {motion.shape}")
        if not 1 <= self.valid_frames <= 150:
            raise ValueError("valid_frames must be in [1,150]")
        if self.end_seq - self.start_seq != 150:
            raise ValueError("window sequence range must contain 150 frames")
        object.__setattr__(self, "motion", motion)


@dataclass(frozen=True)
class GeneratedChunk:
    window_id: int
    motion: np.ndarray
    inference_wall_ms: float = 0.0
    inference_cuda_ms: float | None = None

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion, dtype=np.float32)
        if motion.shape != (150, 151):
            raise ValueError(f"generated motion must be (150,151), got {motion.shape}")
        if not np.isfinite(motion).all():
            raise ValueError("generated motion contains NaN/Inf")
        object.__setattr__(self, "motion", motion)


@dataclass(frozen=True)
class CommittedBatch:
    """A contiguous, exactly-once segment accepted by the output timeline."""

    window_id: int
    start_frame_id: int
    joints: np.ndarray
    lead_joints: np.ndarray | None = None
    commit_kind: str = "stable"
    trigger_monotonic_s: float | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints, dtype=np.float32)
        if joints.ndim != 3 or joints.shape[1:] != (24, 3):
            raise ValueError(f"committed joints must be [N,24,3], got {joints.shape}")
        if not np.isfinite(joints).all():
            raise ValueError("committed joints contain NaN/Inf")
        lead_joints = joints if self.lead_joints is None else np.asarray(self.lead_joints, dtype=np.float32)
        if lead_joints.shape != joints.shape:
            raise ValueError(
                f"lead joints must match committed joints {joints.shape}, got {lead_joints.shape}"
            )
        if not np.isfinite(lead_joints).all():
            raise ValueError("lead joints contain NaN/Inf")
        if self.start_frame_id < 0:
            raise ValueError("start_frame_id must be non-negative")
        if self.commit_kind not in {"stable", "tail"}:
            raise ValueError("commit_kind must be stable or tail")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "lead_joints", lead_joints)

    @property
    def end_frame_id(self) -> int:
        return self.start_frame_id + len(self.joints)
