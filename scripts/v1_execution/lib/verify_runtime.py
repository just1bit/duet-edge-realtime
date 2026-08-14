#!/usr/bin/env python3
"""Verify the local or RTX 5090 V1 acceptance runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path


def require(condition: bool, action: str) -> None:
    if not condition:
        raise RuntimeError(action)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--profile", choices=("gpu", "local"), default="gpu")
    args = parser.parse_args()
    duet_edge_root = Path(args.duet_edge_root)
    checkpoint = Path(args.checkpoint)
    if args.profile == "local":
        print("[1/3] Python")
        require(sys.version_info >= (3, 10), f"Activate Python 3.10 or newer; current version is {sys.version.split()[0]}.")
        print(f"Python {sys.version.split()[0]}")
        print("[2/3] Local dependencies and project import")
        import numpy as np
        importlib.import_module("websockets")
        importlib.import_module("pytest")
        importlib.import_module("duet_edge_realtime.service")
        print("[3/3] Fake fixture")
        fixture = Path("tests/fixtures/fake_motion.npz")
        require(fixture.is_file(), f"Restore the fake fixture at {fixture}.")
        with np.load(fixture, allow_pickle=False) as payload:
            key = "motion_151" if "motion_151" in payload else "motion"
            require(key in payload, "Use a fake fixture containing motion_151 or motion.")
            motion = np.asarray(payload[key])
        require(motion.ndim == 2 and motion.shape[1] == 151 and len(motion) >= 150, f"Use a fake fixture shaped [N,151] with N>=150; current shape is {motion.shape}.")
        require(bool(np.isfinite(motion).all()), "Use a finite fake fixture.")
        print("Local runtime compatibility verification passed.")
        return

    print("[1/4] Python")
    require(sys.version_info[:2] == (3, 10), f"Activate Python 3.10; current version is {sys.version.split()[0]}.")
    print(f"Python {sys.version.split()[0]}")
    print("[2/4] PyTorch, CUDA runtime, and GPU")
    import torch
    require(torch.__version__.split("+")[0] == "2.7.0", f"Install PyTorch 2.7.0; current version is {torch.__version__}.")
    require(torch.version.cuda == "12.8", f"Install the CUDA 12.8 PyTorch runtime; current runtime is {torch.version.cuda}.")
    require(torch.cuda.is_available(), "Activate the CUDA-enabled runtime and repeat verification.")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    require("RTX 5090" in gpu_name, f"Select the RTX 5090 acceptance machine; current GPU is {gpu_name}.")
    require(capability == (12, 0), f"Use compute capability 12.0; current capability is {capability}.")
    sample = torch.randn((256, 256), device="cuda")
    product = sample @ sample
    torch.cuda.synchronize()
    require(bool(torch.isfinite(product).all()), "Restore finite CUDA matrix multiplication and repeat verification.")
    print(f"{torch.__version__}; CUDA {torch.version.cuda}; {gpu_name}; capability {capability}")
    print("[3/4] PyTorch3D and project imports")
    from pytorch3d.transforms import axis_angle_to_matrix
    rotation = axis_angle_to_matrix(torch.zeros((1, 3), device="cuda"))
    torch.cuda.synchronize()
    require(rotation.shape == (1, 3, 3) and bool(torch.isfinite(rotation).all()), "Restore a finite PyTorch3D CUDA transform and repeat verification.")
    importlib.import_module("duet_edge_realtime.service")
    print("[4/4] Duet-EDGE and checkpoint")
    require((duet_edge_root / "EDGE.py").is_file(), f"Place EDGE.py under {duet_edge_root}.")
    require(checkpoint.is_file(), f"Place the checkpoint at {checkpoint}.")
    actual = sha256(checkpoint)
    require(actual == args.checkpoint_sha256, f"Select the checkpoint with SHA256 {args.checkpoint_sha256}; current SHA256 is {actual}.")
    sys.path.insert(0, str(duet_edge_root))
    importlib.import_module("EDGE")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint_data, dict), "Use a checkpoint whose root value is a dictionary.")
    required = {"ema_state_dict", "normalizer"}
    require(required <= set(checkpoint_data), f"Use a checkpoint containing {sorted(required)}.")
    print("Runtime compatibility verification passed.")


if __name__ == "__main__":
    main()
