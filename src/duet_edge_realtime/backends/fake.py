from __future__ import annotations

import time

import numpy as np

from ..schemas import GeneratedChunk, MotionWindow
from .base import InferenceBackend


class FakeInferenceBackend(InferenceBackend):
    """Deterministic backend with latency/error injection for Mac tests."""

    def __init__(self, delay_s: float = 0.0, fail_window: int | None = None):
        self.delay_s = delay_s
        self.fail_window = fail_window
        self.closed = False
        self.calls = 0

    def warmup(self) -> None:
        if self.closed:
            raise RuntimeError("backend is closed")

    def infer(self, window: MotionWindow) -> GeneratedChunk:
        if self.closed:
            raise RuntimeError("backend is closed")
        if window.window_id == self.fail_window:
            raise RuntimeError(f"injected failure at window {window.window_id}")
        started = time.perf_counter()
        if self.delay_s:
            time.sleep(self.delay_s)
        # Copy lead motion so the fake path exercises real overlap/continuity.
        output = window.motion.copy()
        self.calls += 1
        return GeneratedChunk(
            window.window_id,
            output,
            inference_wall_ms=(time.perf_counter() - started) * 1000.0,
        )

    def unnormalize(self, motion):
        return np.asarray(motion, dtype=np.float32).copy()

    def close(self) -> None:
        self.closed = True

    def version_info(self) -> dict:
        return {"backend": "fake", "delay_s": self.delay_s}

    def start_session(self, session_id: str) -> None:
        self.calls = 0
