from __future__ import annotations

import importlib
import hashlib
import json
import pickle
import sys
from bisect import bisect_right
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
            raise ValueError("V2 fixture must contain at least 150 frames")
        if not np.isfinite(motion).all():
            raise ValueError("fixture contains NaN/Inf")
        self.motion = motion
        self.fps = fps
        self.loop = loop
        self.metadata = self._metadata(payload)
        self.identity = self.metadata.get("timeline_id", Path(path).stem)
        self.sha256 = _sha256(path)

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
                yield MotionFrame(
                    seq, seq / self.fps, vector, source_id=self.identity,
                    source_sha256=self.sha256,
                )
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
        start_frame: int = 0,
        end_frame: int | None = None,
        loop: int = 1,
    ):
        if loop < 1:
            raise ValueError("loop must be positive")
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
        full_motion = tensor.detach().cpu().numpy().astype(np.float32)
        stop = len(full_motion) if end_frame is None else min(end_frame, len(full_motion))
        if not 0 <= start_frame < stop:
            raise ValueError("selected AIST frame range is empty")
        self.motion = full_motion[start_frame:stop]
        self.fps = fps
        self.loop = loop
        self.start_frame = start_frame
        self.path = Path(path).resolve()
        self.sha256 = _sha256(self.path)
        self._clips = list(data.get("clip_boundaries", []))
        self._transitions = list(data.get("transitions", []))
        self._clip_starts_30 = [int(item["start_frame_60fps"]) // 2 for item in self._clips]
        source_metadata = data.get("metadata", {})
        sidecar = {}
        timeline_path = self.path.parent / "timeline.json"
        if timeline_path.is_file():
            sidecar = json.loads(timeline_path.read_text(encoding="utf-8"))
        self.identity = sidecar.get(
            "identity", source_metadata.get("format", self.path.stem)
        )
        self.metadata = {
            "source": str(self.path),
            "source_sha256": self.sha256,
            "timeline_id": self.identity,
            "normalized": True,
            "root_scaled": root_scaled,
            "source_fps": 60,
            "fps": fps,
            "start_frame": start_frame,
            "end_frame": stop,
            "frame_count": len(self.motion),
            "clip_count": len(self._clips) or 1,
            "timeline_schema": sidecar.get("schema"),
        }
        if len(self.motion) < 150:
            raise ValueError("V2 AIST motion must contain at least 150 frames after downsampling")

    def frames(self) -> Iterator[MotionFrame]:
        seq = 0
        for _ in range(self.loop):
            for local_seq, vector in enumerate(self.motion):
                absolute_seq = self.start_frame + local_seq
                clip = self._clip_for_frame(absolute_seq)
                transition_id = self._transition_for_frame(absolute_seq, clip)
                yield MotionFrame(
                    seq,
                    seq / self.fps,
                    vector,
                    source_id=self.identity,
                    clip_id=None if clip is None else str(clip.get("stem", clip.get("index"))),
                    clip_frame=None if clip is None else absolute_seq - int(clip["start_frame_60fps"]) // 2,
                    source_sha256=self.sha256,
                    transition_id=transition_id,
                    in_transition=transition_id is not None,
                )
                seq += 1

    def _clip_for_frame(self, frame: int) -> dict | None:
        if not self._clips:
            return None
        index = max(0, bisect_right(self._clip_starts_30, frame) - 1)
        return self._clips[min(index, len(self._clips) - 1)]

    def _transition_for_frame(self, frame: int, clip: dict | None) -> int | None:
        if clip is None or int(clip.get("index", 0)) == 0:
            return None
        boundary = int(clip["start_frame_60fps"]) // 2
        transition_frames = int(self._transitions[int(clip["index"]) - 1].get(
            "transition_frames_60fps", 0
        )) // 2
        return int(clip["index"]) - 1 if boundary <= frame < boundary + transition_frames else None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
