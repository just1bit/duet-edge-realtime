#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify lead-only fast CFG equals the original 3-branch formula"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    backend = CudaDuetEdgeBackend(args.checkpoint, args.duet_edge_root)
    backend.warmup()
    try:
        torch = backend.torch
        decoder = backend.edge.model
        if hasattr(decoder, "module"):
            decoder = decoder.module
        device = backend.edge.accelerator.device
        torch.manual_seed(20260810)
        x = torch.randn((1, 150, 151), device=device)
        cond = torch.randn((1, 150, 4951), device=device)
        times = torch.full((1,), 500, device=device, dtype=torch.long)
        music_cond = cond.clone(); music_cond[:, :, :151] = 0
        lead_cond = cond.clone(); lead_cond[:, :, 151:] = 0
        with torch.inference_mode():
            eps_unc = decoder.forward(x, cond, times, cond_drop_prob=1)
            eps_music = decoder.forward(x, music_cond, times, cond_drop_prob=0)
            eps_lead = decoder.forward(x, lead_cond, times, cond_drop_prob=0)
            reference = eps_unc + 0.0 * (eps_music - eps_unc) + 2.0 * (eps_lead - eps_unc)

            calls = 0
            original_forward = decoder.forward
            def counted_forward(*call_args, **call_kwargs):
                nonlocal calls
                calls += 1
                return original_forward(*call_args, **call_kwargs)
            decoder.forward = counted_forward
            try:
                fast = decoder.guided_forward(x, cond, times, 0.0, 2.0)
            finally:
                decoder.forward = original_forward
        maximum = float((reference - fast).abs().max().item())
        result = {
            "max_abs_error": maximum,
            "fast_path_forward_calls": calls,
            "expected_forward_calls": 2,
            "passed": maximum <= 1e-6 and calls == 2,
            "engine": backend.version_info(),
        }
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
