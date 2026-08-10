#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--output", default="tests/fixtures/fake_motion.npz")
    args = parser.parse_args()
    if args.frames < 150:
        raise SystemExit("--frames must be at least 150")
    motion = np.zeros((args.frames, 151), dtype=np.float32)
    t = np.arange(args.frames, dtype=np.float32) / 30
    motion[:, 4] = 0.3 * np.sin(t * 0.7)
    motion[:, 5] = 0.2 * np.cos(t * 0.5)
    motion[:, 6] = 1.0
    motion[:, 7:] = np.tile(np.asarray([1,0,0,0,1,0], dtype=np.float32), 24)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"source":"generated_fake","normalized":True,"fps":30})
    np.savez_compressed(target, motion_151=motion, metadata_json=metadata)
    print(target)


if __name__ == "__main__":
    main()
