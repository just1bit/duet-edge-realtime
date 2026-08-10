from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    duet_edge_root: str = ""
    checkpoint: str = ""
    input_motion: str = ""
    output_dir: str = ""
    root_scaled: bool | None = None


@dataclass(frozen=True)
class ModelConfig:
    guidance_music: float = 0.0
    guidance_lead: float = 2.0
    sampling_steps: int = 50
    eta: float = 1.0
    seed: int = 1234

    def __post_init__(self) -> None:
        if not 1 <= self.sampling_steps <= 1000:
            raise ValueError("model.sampling_steps must be in [1, 1000]")
        if self.eta < 0:
            raise ValueError("model.eta must be non-negative")


@dataclass(frozen=True)
class StreamConfig:
    fps: int = 30
    window_frames: int = 150
    hop_frames: int = 75
    playout_delay_s: float = 2.0
    inference_queue_size: int = 1
    output_queue_size: int = 2
    viewer_queue_frames: int = 150
    inference_queue_policy: str = "block"
    inference_slo_ms: float = 1900.0
    deadline_miss_policy: str = "continue"
    jitter_slo_ms: float = 20.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("stream.fps must be positive")
        if self.window_frames != 150 or self.hop_frames != 75:
            raise ValueError("V1 requires stream.window_frames=150 and hop_frames=75")
        if self.playout_delay_s < 0:
            raise ValueError("stream.playout_delay_s must be non-negative")
        if self.inference_queue_size < 1:
            raise ValueError("stream.inference_queue_size must be positive")
        if self.output_queue_size < 1:
            raise ValueError("stream.output_queue_size must be positive")
        if self.viewer_queue_frames < 1:
            raise ValueError("stream.viewer_queue_frames must be positive")
        if self.inference_queue_policy not in {"block", "fail"}:
            raise ValueError("stream.inference_queue_policy must be block or fail")
        if self.inference_slo_ms <= 0:
            raise ValueError("stream.inference_slo_ms must be positive")
        if self.deadline_miss_policy not in {"continue", "fail"}:
            raise ValueError("stream.deadline_miss_policy must be continue or fail")
        if self.jitter_slo_ms <= 0:
            raise ValueError("stream.jitter_slo_ms must be positive")


@dataclass(frozen=True)
class ServerConfig:
    bind_host: str = "127.0.0.1"
    port: int = 8765

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("server.port must be in [1, 65535]")


@dataclass(frozen=True)
class RealtimeConfig:
    backend: str = "fake"
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"fake", "cuda"}:
            raise ValueError("backend must be fake or cuda")

    @classmethod
    def load(cls, path: str | Path) -> "RealtimeConfig":
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            backend=data.get("backend", "fake"),
            paths=PathsConfig(**data.get("paths", {})),
            model=ModelConfig(**data.get("model", {})),
            stream=StreamConfig(**data.get("stream", {})),
            server=ServerConfig(**data.get("server", {})),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    # Read-only conveniences keep runtime code concise while serialized
    # configuration remains grouped by paths/model/stream/server.
    @property
    def fps(self): return self.stream.fps
    @property
    def window_frames(self): return self.stream.window_frames
    @property
    def hop_frames(self): return self.stream.hop_frames
    @property
    def playout_delay_s(self): return self.stream.playout_delay_s
    @property
    def viewer_queue_frames(self): return self.stream.viewer_queue_frames
    @property
    def inference_queue_size(self): return self.stream.inference_queue_size
    @property
    def output_queue_size(self): return self.stream.output_queue_size
    @property
    def inference_queue_policy(self): return self.stream.inference_queue_policy
    @property
    def inference_slo_ms(self): return self.stream.inference_slo_ms
    @property
    def deadline_miss_policy(self): return self.stream.deadline_miss_policy
    @property
    def jitter_slo_ms(self): return self.stream.jitter_slo_ms
    @property
    def guidance_music(self): return self.model.guidance_music
    @property
    def guidance_lead(self): return self.model.guidance_lead
    @property
    def sampling_steps(self): return self.model.sampling_steps
    @property
    def eta(self): return self.model.eta
    @property
    def bind_host(self): return self.server.bind_host
    @property
    def port(self): return self.server.port
