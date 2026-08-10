from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from .schemas import MotionFrame


class NormalizedFixtureAdapter:
    def __init__(self, path: str | Path, fps: int = 30, loop: int = 1):
        if loop < 1:
            raise ValueError("loop must be positive")
        payload = np.load(path, allow_pickle=False)
        key = "motion_151" if "motion_151" in payload else "motion"
        if key not in payload:
            raise ValueError("fixture must contain motion_151 or motion")
        motion = np.asarray(payload[key], dtype=np.float32)
        if motion.ndim != 2 or motion.shape[1] != 151:
            raise ValueError(f"fixture motion must be [N,151], got {motion.shape}")
        if len(motion) < 150:
            raise ValueError("V1 fixture must contain at least 150 frames")
        if not np.isfinite(motion).all():
            raise ValueError("fixture contains NaN/Inf")
        self.motion = motion
        self.fps = fps
        self.loop = loop
        self.metadata = self._metadata(payload)

    @staticmethod
    def _metadata(payload) -> dict:
        if "metadata_json" not in payload:
            return {"source": "normalized_fixture", "normalized": True}
        value = payload["metadata_json"]
        if getattr(value, "shape", ()) == ():
            value = value.item()
        return json.loads(str(value))

    def frames(self) -> Iterator[MotionFrame]:
        seq = 0
        for _ in range(self.loop):
            for vector in self.motion:
                yield MotionFrame(seq, seq / self.fps, vector)
                seq += 1


class AISTFileReplayAdapter:
    """Preprocess a raw/sliced AIST pickle before timed replay.

    root_scaled is deliberately mandatory: True means pos is already in model
    units; False means the file must contain scale and pos/scale[0] is applied.
    """

    def __init__(
        self,
        path: str | Path,
        normalizer,
        duet_edge_root: str | Path,
        *,
        root_scaled: bool,
        fps: int = 30,
    ):
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)
        if "pos" not in data or "q" not in data:
            raise ValueError("AIST pickle must contain pos and q")
        pos = np.asarray(data["pos"], dtype=np.float32)
        rotations = np.asarray(data["q"], dtype=np.float32)
        if not root_scaled:
            if "scale" not in data:
                raise ValueError("root_scaled=false requires a scale key")
            scale = float(np.asarray(data["scale"]).reshape(-1)[0])
            if not np.isfinite(scale) or scale == 0:
                raise ValueError(f"invalid AIST scale {scale}")
            pos = pos / scale

        engine = str(Path(duet_edge_root).resolve())
        if engine not in sys.path:
            sys.path.insert(0, engine)
        module = importlib.import_module("dataset.dance_dataset")
        tensor = module.preprocess_motion_to_tensor(pos, rotations, normalizer)
        self.motion = tensor.detach().cpu().numpy().astype(np.float32)
        self.fps = fps
        self.metadata = {
            "source": str(Path(path).resolve()),
            "normalized": True,
            "root_scaled": root_scaled,
            "source_fps": 60,
            "fps": fps,
        }
        if len(self.motion) < 150:
            raise ValueError("V1 AIST motion must contain at least 150 frames after downsampling")

    def frames(self) -> Iterator[MotionFrame]:
        for seq, vector in enumerate(self.motion):
            yield MotionFrame(seq, seq / self.fps, vector)
