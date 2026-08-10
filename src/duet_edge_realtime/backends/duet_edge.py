from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..schemas import GeneratedChunk, MotionWindow
from .base import InferenceBackend

EXPECTED_ENGINE_COMMIT = "e6a731106b912c1a4a8856b2a082d58cd9b93d3d"


class CudaDuetEdgeBackend(InferenceBackend):
    def __init__(
        self,
        checkpoint: str | Path,
        duet_edge_root: str | Path,
        *,
        guidance_music: float = 0.0,
        guidance_lead: float = 2.0,
        sampling_steps: int = 50,
        eta: float = 1.0,
        require_clean_engine: bool = False,
    ):
        self.checkpoint = Path(checkpoint).resolve()
        self.engine_root = Path(duet_edge_root).resolve()
        self.guidance_music = guidance_music
        self.guidance_lead = guidance_lead
        self.sampling_steps = sampling_steps
        self.eta = eta
        self.require_clean_engine = require_clean_engine
        self.edge = None
        self.torch = None
        self._zero_music = None

    def warmup(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not (self.engine_root / "EDGE.py").is_file():
            raise FileNotFoundError(f"invalid duet-edge root: {self.engine_root}")
        status = self._git("status", "--porcelain")
        if self.require_clean_engine and status:
            raise RuntimeError("formal acceptance requires a clean duet-edge worktree")
        commit = self._git("rev-parse", "HEAD")
        if self.require_clean_engine and commit != EXPECTED_ENGINE_COMMIT:
            raise RuntimeError(
                "formal acceptance requires duet-edge commit "
                f"{EXPECTED_ENGINE_COMMIT}, got {commit}"
            )
        if str(self.engine_root) not in sys.path:
            sys.path.insert(0, str(self.engine_root))
        import torch
        from EDGE import EDGE

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requires torch.cuda.is_available()")
        self.torch = torch
        self.edge = EDGE(
            "jukebox",
            checkpoint_path=str(self.checkpoint),
            EMA=True,
            duet=True,
            guidance_weight_music=self.guidance_music,
            guidance_weight_lead=self.guidance_lead,
        )
        self.edge.eval()
        device = self.edge.accelerator.device
        self._zero_music = torch.zeros((1, 150, 4800), device=device)
        zeros = np.zeros((150, 151), dtype=np.float32)
        identity = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), 24)
        zeros[:, 7:] = identity
        for warmup_index in range(3):
            self.infer(
                MotionWindow(
                    warmup_index, 0, 150, 0.0, warmup_index, zeros
                )
            )

    def infer(self, window: MotionWindow) -> GeneratedChunk:
        if self.edge is None or self.torch is None:
            raise RuntimeError("warmup() must be called before infer()")
        torch = self.torch
        device = self.edge.accelerator.device
        torch.manual_seed(window.seed)
        torch.cuda.manual_seed_all(window.seed)
        lead = torch.from_numpy(window.motion).to(device=device).unsqueeze(0)
        cond = torch.cat((lead, self._zero_music), dim=-1)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        with torch.inference_mode():
            sample = self.edge.diffusion.ddim_sample(
                (1, 150, 151),
                cond,
                sampling_timesteps=self.sampling_steps,
                eta=self.eta,
            )
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        motion = sample[0].detach().float().cpu().numpy()
        return GeneratedChunk(window.window_id, motion, wall_ms, cuda_ms)

    def unnormalize(self, motion):
        if self.edge is None or self.torch is None:
            raise RuntimeError("backend is not warm")
        tensor = self.torch.from_numpy(np.asarray(motion, dtype=np.float32))
        return self.edge.normalizer.unnormalize(tensor).detach().cpu().numpy()

    def close(self) -> None:
        self._zero_music = None
        self.edge = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.engine_root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def version_info(self) -> dict:
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        torch_info = {}
        if self.torch is not None:
            torch_info = {
                "torch": self.torch.__version__,
                "cuda": self.torch.version.cuda,
                "gpu": self.torch.cuda.get_device_name(0),
                "peak_gpu_memory_bytes": self.torch.cuda.max_memory_allocated(),
            }
        return {
            "backend": "cuda",
            "engine_root": str(self.engine_root),
            "engine_commit": self._git("rev-parse", "HEAD"),
            "engine_dirty": bool(self._git("status", "--porcelain")),
            "expected_engine_commit": EXPECTED_ENGINE_COMMIT,
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": digest.hexdigest(),
            **torch_info,
        }
