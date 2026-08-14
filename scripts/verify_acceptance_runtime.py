#!/usr/bin/env python3
"""Read-only compatibility gate for the RTX 5090 V1 acceptance runtime."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = (
    "2c948e74400ba78dbec469880746a78dfbb10ed56597917ba2e406cfeb8f9e15"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    duet_edge_root = workspace_root / "duet-edge"
    checkpoint = workspace_root / "data+checkpoint" / "train-1800.pt"

    print("[verify 1/4] Python")
    require(
        sys.version_info[:2] == (3, 10),
        f"Python 3.10 is required; found {sys.version.split()[0]}",
    )
    print(f"  Python {sys.version.split()[0]}")

    print("[verify 2/4] PyTorch, CUDA runtime, and RTX 5090")
    import torch

    require(
        torch.__version__.split("+")[0] == "2.7.0",
        f"Expected torch 2.7.0; found {torch.__version__}",
    )
    require(
        torch.version.cuda == "12.8",
        f"Expected CUDA runtime 12.8; found {torch.version.cuda}",
    )
    require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    require("RTX 5090" in gpu_name, f"Expected RTX 5090; found {gpu_name}")
    require(capability == (12, 0), f"Expected capability (12, 0); found {capability}")
    left = torch.randn((256, 256), device="cuda")
    product = left @ left
    torch.cuda.synchronize()
    require(
        bool(torch.isfinite(product).all()),
        "CUDA matrix multiplication returned NaN or Inf",
    )
    print(
        f"  {torch.__version__}; CUDA {torch.version.cuda}; "
        f"{gpu_name}; capability {capability}"
    )
    print("  CUDA matrix multiplication: OK")

    print("[verify 3/4] PyTorch3D and project imports")
    from pytorch3d.transforms import axis_angle_to_matrix

    rotation = axis_angle_to_matrix(torch.zeros((1, 3), device="cuda"))
    torch.cuda.synchronize()
    require(rotation.shape == (1, 3, 3), f"Unexpected rotation shape: {rotation.shape}")
    require(bool(torch.isfinite(rotation).all()), "PyTorch3D returned NaN or Inf")
    importlib.import_module("duet_edge_realtime.service")
    print("  PyTorch3D and duet-edge-realtime: OK")

    print("[verify 4/4] Duet-EDGE and trusted checkpoint")
    require(
        (duet_edge_root / "EDGE.py").is_file(),
        f"Missing {duet_edge_root / 'EDGE.py'}",
    )
    require(checkpoint.is_file(), f"Missing {checkpoint}")
    actual_sha256 = sha256(checkpoint)
    require(
        actual_sha256 == EXPECTED_CHECKPOINT_SHA256,
        f"Checkpoint SHA256 mismatch: {actual_sha256}",
    )
    sys.path.insert(0, str(duet_edge_root))
    importlib.import_module("EDGE")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint_data, dict), "Checkpoint root must be a dictionary")
    required_keys = {"ema_state_dict", "normalizer"}
    require(
        required_keys <= set(checkpoint_data),
        f"Checkpoint is missing {sorted(required_keys - set(checkpoint_data))}",
    )
    del checkpoint_data
    print("  Duet-EDGE import and checkpoint load: OK")

    print("\nRuntime compatibility verification: PASSED")


if __name__ == "__main__":
    main()
