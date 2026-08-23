from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import GeneratedChunk, MotionWindow


class InferenceBackend(ABC):
    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def infer(self, window: MotionWindow) -> GeneratedChunk: ...

    @abstractmethod
    def unnormalize(self, motion): ...

    @abstractmethod
    def close(self) -> None: ...

    def start_session(self, session_id: str) -> None:
        """Reset all sequence-local state before the first input window."""

    def reset_session(self, reason: str = "explicit") -> None:
        """Clear sequence-local state after warmup, restart, or completion."""

    def continuity_info(self) -> dict:
        return {
            "causal_overlap": False,
            "handoff_residency": "none",
            "continuity_correction": "relative-root+raised-cosine+slerp",
        }

    def version_info(self) -> dict:
        return {"backend": type(self).__name__}
