from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionFrame:
    seq: int
    source_time_s: float
    motion_151: np.ndarray

    def __post_init__(self) -> None:
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

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion, dtype=np.float32)
        if motion.shape != (150, 151):
            raise ValueError(f"window motion must be (150,151), got {motion.shape}")
        if not 1 <= self.valid_frames <= 150:
            raise ValueError("valid_frames must be in [1,150]")
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
