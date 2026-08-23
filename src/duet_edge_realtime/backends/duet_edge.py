from __future__ import annotations

import hashlib
import sys
import time
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
        self._handoff: dict[int, object] = {}
        self._handoff_meta: dict = {}
        self._previous_clean_tail = None
        self._session_id: str | None = None
        self._handoff_resets = 0

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
        checkpoint = torch.load(
            self.checkpoint, map_location="cpu", weights_only=False
        )
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
        device = self.edge.accelerator.device
        self._zero_music = torch.zeros((1, 150, 4800), device=device)
        zeros = np.zeros((150, 151), dtype=np.float32)
        identity = np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), 24)
        zeros[:, 7:] = identity
        for warmup_index in range(3):
            self.reset_session("warmup-window")
            self.infer(
                MotionWindow(
                    warmup_index, 0, 150, 0.0, warmup_index, zeros
                )
            )
        self.reset_session("warmup-complete")

    def _validate_runtime_layout(self) -> None:
        required = ("EDGE.py", "model/diffusion.py", "vis.py")
        missing_files = [name for name in required if not (self.engine_root / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"invalid DUET_EDGE_ROOT {self.engine_root}; missing {missing_files}"
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
            sample, handoff = self._causal_overlap_sample(window, cond)
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
        motion = sample[0].detach().float().cpu().numpy()
        # Wall time represents when the generated window is available to the
        # continuity stage, including the device-to-host transfer.
        wall_ms = (time.perf_counter() - started) * 1000.0
        return GeneratedChunk(
            window.window_id,
            motion,
            wall_ms,
            cuda_ms,
            handoff_used=handoff["used"],
            handoff_produced=True,
            handoff_state_bytes=handoff["state_bytes"],
            handoff_copy_ms=handoff["copy_ms"],
            normalized_overlap_disagreement=handoff["overlap_disagreement"],
        )

    def _causal_overlap_sample(self, window: MotionWindow, cond):
        torch = self.torch
        diffusion = self.edge.diffusion
        shape = (1, 150, 151)
        batch, device, total_timesteps = 1, diffusion.betas.device, diffusion.n_timestep
        times = torch.linspace(
            -1, total_timesteps - 1, steps=self.sampling_steps + 1
        )
        times = list(reversed(times.int().tolist()))
        scale_factors = np.clip(
            np.linspace(0, 2.0, self.sampling_steps), None, 1.0
        )
        schedule = tuple((int(a), int(b)) for a, b in zip(times[:-1], times[1:]))
        old_state = self._validated_handoff(window, shape, schedule, device)
        used = bool(old_state)
        x = torch.randn(shape, device=device)
        cond = cond.to(device)
        next_state = {}
        copy_ms = 0.0

        for step_index, ((current, next_time), scale) in enumerate(
            zip(schedule, scale_factors)
        ):
            if step_index > 0 and current in old_state:
                copy_started = time.perf_counter()
                x[:, :75].copy_(old_state[current])
                copy_ms += (time.perf_counter() - copy_started) * 1000.0
            time_cond = torch.full((batch,), current, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = diffusion.model_predictions(
                x,
                cond,
                time_cond,
                weight=diffusion.guidance_weight * scale,
                weight_lead=(
                    diffusion.guidance_weight_lead * scale
                    if diffusion.guidance_weight_lead is not None else None
                ),
                clip_x_start=diffusion.clip_denoised,
            )
            if next_time < 0:
                x = x_start
                continue
            alpha = diffusion.alphas_cumprod[current]
            alpha_next = diffusion.alphas_cumprod[next_time]
            sigma = self.eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            coefficient = (1 - alpha_next - sigma ** 2).sqrt()
            x = (
                x_start * alpha_next.sqrt()
                + coefficient * pred_noise
                + sigma * torch.randn_like(x)
            )
            copy_started = time.perf_counter()
            next_state[next_time] = x[:, 75:].detach().clone()
            copy_ms += (time.perf_counter() - copy_started) * 1000.0

        disagreement = None
        if self._previous_clean_tail is not None:
            difference = x[:, :75] - self._previous_clean_tail
            denominator = torch.std(self._previous_clean_tail).clamp_min(1e-6)
            disagreement = float(
                (torch.sqrt(torch.mean(difference * difference)) / denominator).item()
            )
        self._previous_clean_tail = x[:, 75:].detach().clone()
        self._handoff = next_state
        self._handoff_meta = {
            "next_window_id": window.window_id + 1,
            "shape": shape,
            "sampling_steps": self.sampling_steps,
            "schedule": schedule,
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
        state_bytes = sum(item.nelement() * item.element_size() for item in next_state.values())
        return x, {
            "used": used,
            "state_bytes": state_bytes,
            "copy_ms": copy_ms,
            "overlap_disagreement": disagreement,
        }

    def _validated_handoff(self, window, shape, schedule, device):
        if not self._handoff:
            return {}
        expected = {
            "next_window_id": window.window_id,
            "shape": shape,
            "sampling_steps": self.sampling_steps,
            "schedule": schedule,
            "dtype": "torch.float32",
            "device": str(device),
        }
        if self._handoff_meta != expected:
            raise RuntimeError(
                f"causal-overlap handoff metadata mismatch: expected {expected}, "
                f"got {self._handoff_meta}"
            )
        for timestep, value in self._handoff.items():
            if value.shape != (1, 75, 151) or value.device != device:
                raise RuntimeError(f"invalid handoff tensor at timestep {timestep}")
        return self._handoff

    def start_session(self, session_id: str) -> None:
        self.reset_session("session-start")
        self._session_id = session_id

    def reset_session(self, reason: str = "explicit") -> None:
        self._handoff = {}
        self._handoff_meta = {}
        self._previous_clean_tail = None
        self._handoff_resets += 1

    def continuity_info(self) -> dict:
        return {
            "causal_overlap": True,
            "handoff_residency": "cuda",
            "handoff_steps": max(0, self.sampling_steps - 1),
            "continuity_correction": "relative-root+raised-cosine+slerp",
        }

    def unnormalize(self, motion):
        if self.edge is None or self.torch is None:
            raise RuntimeError("backend is not warm")
        tensor = self.torch.from_numpy(np.asarray(motion, dtype=np.float32))
        return self.edge.normalizer.unnormalize(tensor).detach().cpu().numpy()

    def close(self) -> None:
        self.reset_session("close")
        self._zero_music = None
        self.edge = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

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
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": digest.hexdigest(),
            "guidance_music": self.guidance_music,
            "guidance_lead": self.guidance_lead,
            "sampling_steps": self.sampling_steps,
            "eta": self.eta,
            "causal_overlap": True,
            "handoff_resets": self._handoff_resets,
            **torch_info,
        }
