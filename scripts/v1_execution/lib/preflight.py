#!/usr/bin/env python3
"""Collect machine and asset facts for one acceptance run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return (result.stdout + result.stderr).strip()


def port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realtime-root", required=True)
    parser.add_argument("--duet-edge-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--motion-sha256", required=True)
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--websocket-port", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=("gpu", "local"), default="gpu")
    args = parser.parse_args()
    realtime = Path(args.realtime_root)
    duet_edge = Path(args.duet_edge_root)
    checkpoint = Path(args.checkpoint)
    motion = Path(args.motion)
    actual_checkpoint = sha256(checkpoint)
    actual_motion = sha256(motion)
    browser_commands = [
        name for name in (
            "firefox", "google-chrome", "chromium", "chrome", "microsoft-edge"
        ) if shutil.which(name)
    ]
    browser_status = "available" if browser_commands else "unknown"
    nvidia_smi = command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv"])
    checks = {
        "realtime_repository": realtime.is_dir(),
        "duet_edge_entrypoint": (duet_edge / "EDGE.py").is_file(),
        "checkpoint": checkpoint.is_file(),
        "checkpoint_sha256": actual_checkpoint == args.checkpoint_sha256,
        "aist_motion": motion.is_file(),
        "aist_motion_sha256": actual_motion == args.motion_sha256,
        "http_port_available": port_available(args.http_port),
        "websocket_port_available": port_available(args.websocket_port),
    }
    gpu = {"applicable": args.profile == "gpu", "nvidia_smi_available": shutil.which("nvidia-smi") is not None, "query_succeeded": bool(nvidia_smi), "target_gpu": bool(nvidia_smi and "RTX 5090" in nvidia_smi)}
    if args.profile == "gpu":
        checks["nvidia_smi_available"] = gpu["nvidia_smi_available"]
        checks["gpu_query_succeeded"] = gpu["query_succeeded"]
        checks["target_gpu"] = gpu["target_gpu"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "machine": platform.node(),
        "platform": platform.platform(),
        "profile": args.profile,
        "paths": {
            "realtime_root": str(realtime), "duet_edge_root": str(duet_edge),
            "checkpoint": str(checkpoint), "aist_motion": str(motion),
        },
        "hashes": {"checkpoint": actual_checkpoint, "aist_motion": actual_motion},
        "checks": checks,
        "disk": shutil.disk_usage(realtime)._asdict() if realtime.exists() else None,
        "browser": {"status": browser_status, "commands": browser_commands, "advisory": browser_status == "unknown"},
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
        "passed": all(checks.values()),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
