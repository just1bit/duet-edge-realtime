#!/usr/bin/env python3
"""Export the legacy real fixture and a portable multi-window golden corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.backends.recorded import GOLDEN_SCHEMA, RecordedInferenceBackend
from duet_edge_realtime.continuity import OnlineContinuityProcessor, direct_fk
from duet_edge_realtime.input_adapters import AISTFileReplayAdapter
from duet_edge_realtime.progress import TerminalProgress
from duet_edge_realtime.schemas import MotionWindow


def repository_state(path: Path) -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args], text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "path": str(path.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def process_windows(processor, chunks: np.ndarray, lead_chunks: np.ndarray) -> np.ndarray:
    committed = [
        processor.process(chunk, lead_motion=lead)
        for chunk, lead in zip(chunks, lead_chunks)
    ]
    committed.append(processor.flush_with_lead(lead_chunks[-1]))
    return np.concatenate(committed)


def direct_lead_timeline(backend, lead_chunks: np.ndarray) -> np.ndarray:
    committed = [direct_fk(backend, chunk)[:75] for chunk in lead_chunks]
    committed.append(direct_fk(backend, lead_chunks[-1])[75:])
    return np.concatenate(committed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--root-scaled", required=True, choices=("true", "false"))
    parser.add_argument("--output", required=True, help="Backward-compatible one-window fixture")
    parser.add_argument("--golden-output", required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-music", type=float, default=0.0)
    parser.add_argument("--guidance-lead", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.windows < 2:
        parser.error("--windows must be at least 2")
    progress = TerminalProgress(args.progress)

    backend = CudaDuetEdgeBackend(
        args.checkpoint,
        args.duet_edge_root,
        guidance_music=args.guidance_music,
        guidance_lead=args.guidance_lead,
        sampling_steps=args.steps,
        eta=args.eta,
        progress_callback=progress.model_update if args.progress else None,
    )
    backend.warmup()
    backend.start_session("fixture-export")
    backend.set_inference_total_windows(args.windows)
    try:
        adapter = AISTFileReplayAdapter(
            args.motion,
            backend.edge.normalizer,
            args.duet_edge_root,
            root_scaled=args.root_scaled == "true",
        )
        required_frames = 150 + (args.windows - 1) * 75
        available = np.stack([frame.motion_151 for frame in adapter.frames()])
        if len(available) < required_frames:
            raise ValueError(
                f"golden export needs {required_frames} normalized frames for "
                f"{args.windows} windows; input has {len(available)}"
            )
        motion = available[:required_frames]
        starts = np.arange(args.windows, dtype=np.int64) * 75
        lead_windows = np.stack([motion[start:start + 150] for start in starts])
        window_ids = np.arange(args.windows, dtype=np.int64)
        window_seeds = args.seed + window_ids
        windows = [
            MotionWindow(
                window_id=int(index),
                start_seq=int(start),
                end_seq=int(start + 150),
                trigger_time_s=float((start + 149) / 30),
                seed=int(window_seeds[index]),
                motion=lead_windows[index],
            )
            for index, start in enumerate(starts)
        ]
        chunks = [backend.infer(window) for window in windows]
        generated = np.stack([chunk.motion for chunk in chunks])
        generated_unnormalized = np.stack(
            [backend.unnormalize(chunk[None])[0] for chunk in generated]
        )
        generated_joints = process_windows(
            OnlineContinuityProcessor(backend), generated, lead_windows
        )
        lead_joints = direct_lead_timeline(backend, lead_windows)

        scaler = backend.edge.normalizer.scaler
        scale = scaler.scale_.detach().cpu().numpy().astype(np.float32)
        minimum = scaler.min_.detach().cpu().numpy().astype(np.float32)
        realtime_root = Path(__file__).resolve().parents[3]
        metadata = {
            **adapter.metadata,
            **backend.version_info(),
            "golden_schema": GOLDEN_SCHEMA,
            "window_frames": 150,
            "hop_frames": 75,
            "window_count": args.windows,
            "seed": args.seed,
            "steps": args.steps,
            "repositories": {
                "duet_edge": repository_state(Path(args.duet_edge_root)),
                "duet_edge_realtime": repository_state(realtime_root),
            },
        }

        legacy_generated_joints = process_windows(
            OnlineContinuityProcessor(backend), generated[:1], lead_windows[:1]
        )
        legacy_lead_joints = direct_lead_timeline(backend, lead_windows[:1])

        legacy_target = Path(args.output)
        legacy_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            legacy_target,
            motion_151=lead_windows[0],
            lead_normalized=lead_windows[0],
            generated_normalized=generated[0],
            generated_unnormalized=generated_unnormalized[0],
            lead_joints=legacy_lead_joints,
            generated_joints=legacy_generated_joints,
            metadata_json=json.dumps({**metadata, "seed": int(window_seeds[0])}),
        )

        golden_target = Path(args.golden_output)
        golden_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            golden_target,
            golden_schema=GOLDEN_SCHEMA,
            motion_151=motion,
            lead_windows=lead_windows,
            generated_normalized=generated,
            generated_unnormalized=generated_unnormalized,
            expected_lead_joints=lead_joints,
            expected_companion_joints=generated_joints,
            window_ids=window_ids,
            window_start_seq=starts,
            window_end_seq=starts + 150,
            window_valid_frames=np.full(args.windows, 150, dtype=np.int64),
            window_seeds=window_seeds,
            inference_wall_ms=np.asarray(
                [chunk.inference_wall_ms for chunk in chunks], dtype=np.float64
            ),
            inference_cuda_ms=np.asarray(
                [chunk.inference_cuda_ms for chunk in chunks], dtype=np.float64
            ),
            normalizer_scale=scale,
            normalizer_min=minimum,
            metadata_json=json.dumps(metadata),
        )
        replay = RecordedInferenceBackend(golden_target)
        replay.warmup()
        try:
            replay_chunks = np.stack([replay.infer(window).motion for window in windows])
            replay_generated_joints = process_windows(
                OnlineContinuityProcessor(replay), replay_chunks, lead_windows
            )
            replay_lead_joints = direct_lead_timeline(replay, lead_windows)
            np.testing.assert_allclose(replay_chunks, generated, atol=1e-6, rtol=0)
            np.testing.assert_allclose(
                replay_generated_joints, generated_joints, atol=1e-5, rtol=1e-5
            )
            np.testing.assert_allclose(
                replay_lead_joints, lead_joints, atol=1e-5, rtol=1e-5
            )
        finally:
            replay.close()

        print(legacy_target)
        print(golden_target)
        print(f"Golden corpus verified: {args.windows} consecutive windows")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
