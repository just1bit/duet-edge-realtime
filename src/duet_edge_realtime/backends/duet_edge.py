from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
import sys
import time
import textwrap
from pathlib import Path

import numpy as np

from ..schemas import GeneratedChunk, MotionWindow
from .base import InferenceBackend


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
    ):
        self.checkpoint = Path(checkpoint).resolve()
        self.engine_root = Path(duet_edge_root).resolve()
        self.guidance_music = guidance_music
        self.guidance_lead = guidance_lead
        self.sampling_steps = sampling_steps
        self.eta = eta
        self.edge = None
        self.torch = None
        self._zero_music = None
        self._engine_commit = "unknown"
        self._engine_dirty = False
        self._engine_python_dirty = False

    def warmup(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self._validate_runtime_layout()
        if str(self.engine_root) not in sys.path:
            sys.path.insert(0, str(self.engine_root))
        import torch
        from EDGE import EDGE

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requires torch.cuda.is_available()")
        self.torch = torch
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        missing_keys = {"ema_state_dict", "normalizer"} - set(checkpoint)
        del checkpoint
        if missing_keys:
            raise ValueError(f"checkpoint is missing required keys: {sorted(missing_keys)}")
        self.edge = EDGE(
            "jukebox",
            checkpoint_path=str(self.checkpoint),
            EMA=True,
            duet=True,
            guidance_weight_music=self.guidance_music,
            guidance_weight_lead=self.guidance_lead,
        )
        self.edge.eval()
        self._validate_sampling_api()
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

    def _validate_sampling_api(self) -> None:
        """Reject configurable DDIM settings when the engine silently ignores them.

        The original Duet-EDGE ``ddim_sample`` accepts ``**kwargs`` while still
        hard-coding 50 steps and eta=1.  Merely calling it with keyword options
        therefore doesn't prove that those options are honored.
        """
        if self.sampling_steps == 50 and self.eta == 1.0:
            return
        function = self.edge.diffusion.ddim_sample
        parameters = inspect.signature(function).parameters
        if {"sampling_timesteps", "eta"} <= set(parameters):
            return
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        except (OSError, TypeError, SyntaxError):
            tree = None
        popped_options = set()
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                target = node.func
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "pop"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "kwargs"
                    and isinstance(node.args[0], ast.Constant)
                ):
                    popped_options.add(node.args[0].value)
        if {"sampling_timesteps", "eta"} <= popped_options:
            return
        raise RuntimeError(
            "the selected Duet-EDGE runtime does not honor configurable DDIM "
            "sampling_timesteps/eta; use the baseline 50-step eta=1 settings "
            "or a compatible optimized engine commit"
        )

    def _validate_runtime_layout(self) -> None:
        required = ("EDGE.py", "model/diffusion.py", "vis.py")
        missing_files = [name for name in required if not (self.engine_root / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"invalid DUET_EDGE_ROOT {self.engine_root}; missing {missing_files}"
            )
        self._engine_commit = self._git("rev-parse", "HEAD")
        self._engine_dirty = bool(self._git("status", "--porcelain", "--untracked-files=all"))
        python_status = self._git(
            "status", "--porcelain", "--untracked-files=all", "--", ":(glob)**/*.py"
        )
        self._engine_python_dirty = bool(python_status)

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
            if self.sampling_steps == 50 and self.eta == 1.0:
                # The baseline Duet-EDGE API uses these sampling defaults.
                sample = self.edge.diffusion.ddim_sample((1, 150, 151), cond)
            else:
                try:
                    sample = self.edge.diffusion.ddim_sample(
                        (1, 150, 151), cond,
                        sampling_timesteps=self.sampling_steps,
                        eta=self.eta,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "the selected Duet-EDGE runtime exposes the baseline DDIM API; "
                        "sampling_timesteps and eta are available in the optimized API"
                    ) from exc
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
            "engine_repository": self._git("config", "--get", "remote.origin.url"),
            "engine_commit": self._engine_commit,
            "engine_dirty": self._engine_dirty,
            "engine_python_dirty": self._engine_python_dirty,
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": digest.hexdigest(),
            "guidance_music": self.guidance_music,
            "guidance_lead": self.guidance_lead,
            "sampling_steps": self.sampling_steps,
            "eta": self.eta,
            **torch_info,
        }
