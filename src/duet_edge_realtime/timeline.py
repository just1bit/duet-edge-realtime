from __future__ import annotations

import numpy as np

from .schemas import CommittedBatch


class TimelineCommitError(RuntimeError):
    pass


class TimelineCommitter:
    """Turns continuity output into one contiguous, exactly-once frame timeline."""

    def __init__(self) -> None:
        self._next_frame_id = 0

    @property
    def next_frame_id(self) -> int:
        return self._next_frame_id

    def commit(
        self,
        window_id: int,
        start_frame_id: int,
        joints: np.ndarray,
        *,
        commit_kind: str = "stable",
        trigger_monotonic_s: float | None = None,
    ) -> CommittedBatch:
        if start_frame_id != self._next_frame_id:
            raise TimelineCommitError(
                f"timeline expected frame {self._next_frame_id}, got {start_frame_id}"
            )
        batch = CommittedBatch(
            window_id=window_id,
            start_frame_id=start_frame_id,
            joints=joints,
            commit_kind=commit_kind,
            trigger_monotonic_s=trigger_monotonic_s,
        )
        self._next_frame_id = batch.end_frame_id
        return batch
