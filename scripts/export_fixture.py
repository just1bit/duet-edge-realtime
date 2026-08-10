#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.continuity import OnlineContinuityProcessor
from duet_edge_realtime.input_adapters import AISTFileReplayAdapter
from duet_edge_realtime.schemas import MotionWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--duet-edge-root", default=os.environ.get("DUET_EDGE_ROOT"))
    parser.add_argument("--motion", required=True)
    parser.add_argument("--root-scaled", required=True, choices=("true","false"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if not args.duet_edge_root:
        raise SystemExit("provide --duet-edge-root or DUET_EDGE_ROOT")
    backend = CudaDuetEdgeBackend(args.checkpoint, args.duet_edge_root, sampling_steps=args.steps)
    backend.warmup()
    try:
        adapter = AISTFileReplayAdapter(
            args.motion, backend.edge.normalizer, args.duet_edge_root,
            root_scaled=args.root_scaled == "true",
        )
        lead = np.stack([frame.motion_151 for frame in list(adapter.frames())[:150]])
        window = MotionWindow(0, 0, 150, 5.0, 42, lead)
        generated = backend.infer(window)
        unnormalized = backend.unnormalize(generated.motion[None])[0]
        generated_processor = OnlineContinuityProcessor(backend)
        generated_joints = np.concatenate([
            generated_processor.process(generated.motion),
            generated_processor.flush(),
        ])
        lead_processor = OnlineContinuityProcessor(backend)
        lead_joints = np.concatenate([
            lead_processor.process(lead),
            lead_processor.flush(),
        ])
        metadata = {**adapter.metadata, **backend.version_info(), "seed":42, "steps":args.steps}
        target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, motion_151=lead, lead_normalized=lead,
                            generated_normalized=generated.motion,
                            generated_unnormalized=unnormalized,
                            lead_joints=lead_joints,
                            generated_joints=generated_joints,
                            metadata_json=json.dumps(metadata))
        print(target)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
