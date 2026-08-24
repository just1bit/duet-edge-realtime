from __future__ import annotations

import os
import time


class TerminalProgress:
    """Small dependency-free progress display for long model operations."""

    def __init__(self, enabled: bool = False, width: int = 28) -> None:
        self.enabled = enabled
        self.width = width
        self.is_tty = os.isatty(1)
        self._last_model_s = 0.0
        self._model_lines_active = False

    def model_update(self, event: dict, *, force: bool = False) -> None:
        if not self.enabled:
            return
        window = int(event.get("window", 0))
        windows = int(event.get("windows", 0))
        step = int(event.get("step", 0))
        steps = int(event.get("steps", 0))
        if window < 1 or windows < 1 or steps < 1:
            return
        now = time.monotonic()
        if (
            self.is_tty and not force and step not in {0, steps}
            and now - self._last_model_s < 0.05
        ):
            return
        if (
            not self.is_tty and not force
            and step not in {0, steps}
            and step % max(1, steps // 5) != 0
        ):
            return
        phase = "Warmup" if event.get("phase") == "warmup" else "Inference"
        overall_units = (window - 1) * steps + step
        overall_total = windows * steps
        lines = [
            self._bar(
                overall_units,
                overall_total,
                f"Overall {phase.lower()}",
                f"window {window}/{windows}",
            ),
            self._bar(step, steps, "Current window sampling", f"step {step}/{steps}"),
        ]
        self._last_model_s = now
        if self.is_tty:
            if self._model_lines_active:
                print("\033[2A", end="", flush=True)
            for line in lines:
                print("\r" + line.ljust(90))
            self._model_lines_active = not (window == windows and step == steps)
        else:
            print("\n".join(lines), flush=True)

    def _bar(self, current: int, total: int, label: str, detail: str) -> str:
        ratio = min(1.0, current / total) if total > 0 else 0.0
        filled = min(self.width, int(self.width * ratio))
        bar = "=" * filled + "." * (self.width - filled)
        return f"[{bar}] {ratio * 100:3.0f}%  {label} · {detail}"
