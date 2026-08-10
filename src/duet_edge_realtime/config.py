from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RealtimeConfig:
    fps: int = 30
    window_frames: int = 150
    hop_frames: int = 75
    guidance_music: float = 0.0
    guidance_lead: float = 2.0
    sampling_steps: int = 50
    eta: float = 1.0
    playout_delay_s: float = 2.0
    inference_queue_size: int = 1
    viewer_queue_frames: int = 150
    bind_host: str = "127.0.0.1"
    port: int = 8765
    duet_edge_root: str = "third_party/duet-edge"

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.window_frames != 150 or self.hop_frames != 75:
            raise ValueError("V1 requires window_frames=150 and hop_frames=75")
        if not 1 <= self.sampling_steps <= 1000:
            raise ValueError("sampling_steps must be in [1, 1000]")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if not 0 <= self.playout_delay_s < self.hop_frames / self.fps:
            raise ValueError("playout_delay_s must be in [0, hop period)")
        if self.inference_queue_size != 1:
            raise ValueError("V1 inference_queue_size must be 1")
        if self.viewer_queue_frames < 1:
            raise ValueError("viewer_queue_frames must be positive")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")

    @classmethod
    def load(cls, path: str | Path) -> "RealtimeConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def as_dict(self) -> dict:
        return asdict(self)
