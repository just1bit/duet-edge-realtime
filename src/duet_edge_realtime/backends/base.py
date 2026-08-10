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

    def version_info(self) -> dict:
        return {"backend": type(self).__name__}
