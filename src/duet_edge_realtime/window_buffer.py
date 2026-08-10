from __future__ import annotations

from collections import deque

import numpy as np

from .schemas import MotionFrame, MotionWindow


class SequenceError(ValueError):
    pass


class SlidingWindowBuffer:
    """Bounded 150-frame buffer which emits at a 75-frame hop."""

    def __init__(self, window_frames: int = 150, hop_frames: int = 75, seed: int = 0):
        if window_frames != 150 or hop_frames != 75:
            raise ValueError("V1 requires a 150/75 window/hop")
        self.window_frames = window_frames
        self.hop_frames = hop_frames
        self.seed = seed
        self._frames: deque[MotionFrame] = deque(maxlen=window_frames)
        self._last_seq: int | None = None
        self._next_trigger_seq = window_frames - 1
        self._window_id = 0

    @property
    def retained_frames(self) -> int:
        return len(self._frames)

    @property
    def last_seq(self) -> int | None:
        return self._last_seq

    def push(self, frame: MotionFrame, trigger_time_s: float) -> MotionWindow | None:
        expected = 0 if self._last_seq is None else self._last_seq + 1
        if frame.seq != expected:
            kind = "duplicate/out-of-order" if frame.seq < expected else "missing"
            raise SequenceError(f"{kind} frame: expected seq {expected}, got {frame.seq}")
        self._frames.append(frame)
        self._last_seq = frame.seq
        if frame.seq != self._next_trigger_seq:
            return None
        window = self._build_window(trigger_time_s, valid_frames=self.window_frames)
        self._next_trigger_seq += self.hop_frames
        return window

    def flush(self, trigger_time_s: float) -> MotionWindow | None:
        """Pad a non-hop-aligned tail. Inputs shorter than 150 are rejected."""
        if self._last_seq is None:
            return None
        if self._last_seq < self.window_frames - 1:
            raise SequenceError(
                f"V1 needs at least 150 input frames, received {self._last_seq + 1}"
            )
        since_last_trigger = self._last_seq - (self._next_trigger_seq - self.hop_frames)
        if since_last_trigger == 0:
            return None
        frames = list(self._frames)
        pad_count = self.window_frames - len(frames)
        if pad_count:
            frames = [frames[0]] * pad_count + frames
        motion = np.stack([frame.motion_151 for frame in frames])
        # The final window should advance by one hop from the previous window;
        # take the latest hop of real frames and pad its future with the EOF frame.
        start_seq = self._next_trigger_seq - self.window_frames
        real = [f for f in self._frames if f.seq >= start_seq]
        if len(real) < self.window_frames:
            real.extend([real[-1]] * (self.window_frames - len(real)))
        motion = np.stack([f.motion_151 for f in real[:self.window_frames]])
        valid = min(since_last_trigger, self.hop_frames)
        window = MotionWindow(
            window_id=self._window_id,
            start_seq=start_seq,
            end_seq=start_seq + self.window_frames,
            trigger_time_s=trigger_time_s,
            seed=self.seed + self._window_id,
            motion=motion,
            valid_frames=valid,
        )
        self._window_id += 1
        self._next_trigger_seq += self.hop_frames
        return window

    def _build_window(self, trigger_time_s: float, valid_frames: int) -> MotionWindow:
        frames = list(self._frames)
        motion = np.stack([frame.motion_151 for frame in frames])
        window = MotionWindow(
            window_id=self._window_id,
            start_seq=frames[0].seq,
            end_seq=frames[-1].seq + 1,
            trigger_time_s=trigger_time_s,
            seed=self.seed + self._window_id,
            motion=motion,
            valid_frames=valid_frames,
        )
        self._window_id += 1
        return window
