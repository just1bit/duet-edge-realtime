from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..schemas import GeneratedChunk, MotionWindow
from .base import InferenceBackend


GOLDEN_SCHEMA = "duet-edge-golden/v1"


class RecordedInferenceBackend(InferenceBackend):
    """Replay CUDA window outputs captured in a portable golden corpus."""

    def __init__(self, fixture: str | Path, *, atol: float = 1e-6):
        self.fixture = Path(fixture).resolve()
        self.atol = atol
        self.closed = False
        self._loaded = False
        self._lead_windows: np.ndarray | None = None
        self._generated: np.ndarray | None = None
        self._window_ids: np.ndarray | None = None
        self._window_start_seq: np.ndarray | None = None
        self._window_end_seq: np.ndarray | None = None
        self._window_valid_frames: np.ndarray | None = None
        self._window_seeds: np.ndarray | None = None
        self._normalizer_scale: np.ndarray | None = None
        self._normalizer_min: np.ndarray | None = None
        self._metadata: dict = {}
        self.calls = 0

    @staticmethod
    def _scalar_text(value) -> str:
        if getattr(value, "shape", ()) == ():
            value = value.item()
        return str(value)

    def warmup(self) -> None:
        if self.closed:
            raise RuntimeError("backend is closed")
        if self._loaded:
            return
        if not self.fixture.is_file():
            raise FileNotFoundError(self.fixture)
        with np.load(self.fixture, allow_pickle=False) as payload:
            required = {
                "golden_schema", "motion_151", "lead_windows",
                "generated_normalized", "generated_unnormalized",
                "expected_lead_joints", "expected_companion_joints",
                "window_ids", "window_start_seq", "window_end_seq",
                "window_valid_frames", "window_seeds", "inference_wall_ms",
                "inference_cuda_ms", "normalizer_scale", "normalizer_min",
                "metadata_json",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"golden fixture is missing keys: {sorted(missing)}")
            schema = self._scalar_text(payload["golden_schema"])
            if schema != GOLDEN_SCHEMA:
                raise ValueError(f"unsupported golden schema {schema!r}")
            motion = np.asarray(payload["motion_151"], dtype=np.float32)
            lead_windows = np.asarray(payload["lead_windows"], dtype=np.float32)
            generated = np.asarray(payload["generated_normalized"], dtype=np.float32)
            generated_unnormalized = np.asarray(
                payload["generated_unnormalized"], dtype=np.float32
            )
            expected_lead_joints = np.asarray(
                payload["expected_lead_joints"], dtype=np.float32
            )
            expected_companion_joints = np.asarray(
                payload["expected_companion_joints"], dtype=np.float32
            )
            ids = np.asarray(payload["window_ids"], dtype=np.int64)
            starts = np.asarray(payload["window_start_seq"], dtype=np.int64)
            ends = np.asarray(payload["window_end_seq"], dtype=np.int64)
            valid = np.asarray(payload["window_valid_frames"], dtype=np.int64)
            seeds = np.asarray(payload["window_seeds"], dtype=np.int64)
            wall_ms = np.asarray(payload["inference_wall_ms"], dtype=np.float64)
            cuda_ms = np.asarray(payload["inference_cuda_ms"], dtype=np.float64)
            scale = np.asarray(payload["normalizer_scale"], dtype=np.float32)
            minimum = np.asarray(payload["normalizer_min"], dtype=np.float32)
            metadata = json.loads(self._scalar_text(payload["metadata_json"]))

        count = len(ids)
        if count < 2:
            raise ValueError("golden fixture must contain at least two consecutive windows")
        if motion.ndim != 2 or motion.shape[1] != 151:
            raise ValueError(f"motion_151 must be [N,151], got {motion.shape}")
        if lead_windows.shape != (count, 150, 151):
            raise ValueError(f"lead_windows must be [{count},150,151], got {lead_windows.shape}")
        if generated.shape != (count, 150, 151):
            raise ValueError(
                f"generated_normalized must be [{count},150,151], got {generated.shape}"
            )
        if generated_unnormalized.shape != generated.shape:
            raise ValueError("generated_unnormalized must match generated_normalized")
        expected_frames = 150 + (count - 1) * 75
        if expected_lead_joints.shape != (expected_frames, 24, 3):
            raise ValueError("expected_lead_joints has an invalid shape")
        if expected_companion_joints.shape != (expected_frames, 24, 3):
            raise ValueError("expected_companion_joints has an invalid shape")
        for name, values in (
            ("window_start_seq", starts), ("window_end_seq", ends),
            ("window_valid_frames", valid), ("window_seeds", seeds),
        ):
            if values.shape != (count,):
                raise ValueError(f"{name} must contain {count} entries")
        if scale.shape != (151,) or minimum.shape != (151,):
            raise ValueError("normalizer_scale and normalizer_min must have shape [151]")
        if wall_ms.shape != (count,) or cuda_ms.shape != (count,):
            raise ValueError("inference timing arrays must contain one value per window")
        if np.any(scale == 0):
            raise ValueError("normalizer_scale contains zero")
        if not all(
            np.isfinite(values).all()
            for values in (
                motion, lead_windows, generated, generated_unnormalized,
                expected_lead_joints, expected_companion_joints, wall_ms,
                cuda_ms, scale, minimum,
            )
        ):
            raise ValueError("golden fixture contains NaN/Inf")
        if not np.array_equal(ids, np.arange(count)):
            raise ValueError("window_ids must be contiguous from zero")
        expected_starts = np.arange(count) * 75
        if not np.array_equal(starts, expected_starts) or not np.array_equal(ends, starts + 150):
            raise ValueError("golden fixture must use consecutive 150/75 windows")
        if not np.all(valid == 150):
            raise ValueError("golden fixture currently supports full windows only")
        if len(motion) != 150 + (count - 1) * 75:
            raise ValueError("motion_151 length does not match the recorded windows")
        for index, start in enumerate(starts):
            if not np.allclose(
                lead_windows[index], motion[start:start + 150], atol=self.atol, rtol=0
            ):
                raise ValueError(f"lead window {index} does not match motion_151")
        expected_unnormalized = (
            np.clip(generated, -1.0, 1.0) - minimum
        ) / scale
        if not np.allclose(
            generated_unnormalized, expected_unnormalized, atol=1e-5, rtol=1e-5
        ):
            raise ValueError("generated_unnormalized does not match normalizer parameters")

        self._lead_windows = lead_windows
        self._generated = generated
        self._window_ids = ids
        self._window_start_seq = starts
        self._window_end_seq = ends
        self._window_valid_frames = valid
        self._window_seeds = seeds
        self._normalizer_scale = scale
        self._normalizer_min = minimum
        self._metadata = metadata
        self._loaded = True

    def infer(self, window: MotionWindow) -> GeneratedChunk:
        if self.closed:
            raise RuntimeError("backend is closed")
        if not self._loaded:
            raise RuntimeError("warmup() must be called before infer()")
        index = self.calls
        if index >= len(self._window_ids):
            raise RuntimeError("recorded fixture has no output for the next window")
        expected = {
            "window_id": int(self._window_ids[index]),
            "start_seq": int(self._window_start_seq[index]),
            "end_seq": int(self._window_end_seq[index]),
            "valid_frames": int(self._window_valid_frames[index]),
            "seed": int(self._window_seeds[index]),
        }
        observed = {
            "window_id": window.window_id,
            "start_seq": window.start_seq,
            "end_seq": window.end_seq,
            "valid_frames": window.valid_frames,
            "seed": window.seed,
        }
        if observed != expected:
            raise ValueError(
                f"recorded window metadata mismatch: expected {expected}, got {observed}"
            )
        if not np.allclose(window.motion, self._lead_windows[index], atol=self.atol, rtol=0):
            raise ValueError(f"recorded window {index} input motion mismatch")
        started = time.perf_counter()
        output = self._generated[index].copy()
        self.calls += 1
        return GeneratedChunk(
            window.window_id,
            output,
            inference_wall_ms=(time.perf_counter() - started) * 1000.0,
        )

    def unnormalize(self, motion):
        if not self._loaded:
            raise RuntimeError("warmup() must be called before unnormalize()")
        values = np.clip(np.asarray(motion, dtype=np.float32), -1.0, 1.0)
        return (values - self._normalizer_min) / self._normalizer_scale

    def close(self) -> None:
        self.closed = True
        self._loaded = False
        self._lead_windows = None
        self._generated = None

    def version_info(self) -> dict:
        digest = hashlib.sha256()
        with self.fixture.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "backend": "recorded",
            "golden_fixture": str(self.fixture),
            "golden_fixture_sha256": digest.hexdigest(),
            "recorded_windows": 0 if self._window_ids is None else len(self._window_ids),
            "source_backend": self._metadata.get("backend", "cuda"),
            "source_checkpoint_sha256": self._metadata.get("checkpoint_sha256"),
        }
