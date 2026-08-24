#!/usr/bin/env python3
"""Run-local V2 initialization, calibration, input locking, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REALTIME_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REALTIME_ROOT.parent
STATE_FILE = REALTIME_ROOT / "outputs" / ".run-current"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REALTIME_ROOT / path).resolve()


def current_run(value: str | None) -> Path:
    if value:
        run = resolve_path(value)
    elif STATE_FILE.is_file():
        run = Path(STATE_FILE.read_text(encoding="utf-8").strip()).resolve()
    else:
        raise SystemExit("Initialize a V2 run with Stage 01.")
    if not (run / "config.json").is_file():
        raise SystemExit(f"V2 run has no config.json: {run}")
    return run


def git_info(path: Path) -> dict:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments], text=True,
            capture_output=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def command_init(args) -> None:
    state_file = resolve_path(args.state_file)
    if args.resume:
        run = current_run(args.resume)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(str(run) + "\n", encoding="utf-8")
        print(run)
        return
    template = resolve_path(args.template)
    config = json.loads(template.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    suffix = hashlib.sha256(f"{stamp}-{socket.gethostname()}".encode()).hexdigest()[:3]
    output_root = resolve_path(args.output_root)
    run = output_root / f"run-{stamp}-{suffix}"
    counter = 1
    while run.exists():
        run = REALTIME_ROOT / "outputs" / f"run-{stamp}-{suffix}-{counter}"
        counter += 1
    for relative in ("logs", "evidence", "fixtures"):
        (run / relative).mkdir(parents=True, exist_ok=True)
    for key in ("duet_edge_root", "checkpoint", "input_motion"):
        value = config.get("paths", {}).get(key)
        if value:
            config["paths"][key] = str(resolve_path(value))
    config["paths"]["output_dir"] = str(run)
    config["run_id"] = run.name
    input_path = Path(config["paths"]["input_motion"])
    checkpoint = Path(config["paths"].get("checkpoint", ""))
    assets = {}
    if input_path.is_file():
        assets["input"] = {
            "path": str(input_path), "bytes": input_path.stat().st_size,
            "sha256": sha256(input_path),
        }
    if checkpoint.is_file():
        assets["checkpoint"] = {
            "path": str(checkpoint), "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        }
    if "input" in assets:
        config["paths"]["input_sha256"] = assets["input"]["sha256"]
    if "checkpoint" in assets:
        config["paths"]["checkpoint_sha256"] = assets["checkpoint"]["sha256"]
    config["assets"] = assets
    write_json(run / "config.json", config)
    write_json(run / "run-metadata.json", {
        "run_id": run.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "machine": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "assets": assets,
        "repositories": {
            "duet_edge_realtime": git_info(REALTIME_ROOT),
            "duet_edge": git_info(Path(
                config["paths"].get("duet_edge_root") or PROJECT_ROOT / "duet-edge"
            )),
        },
    })
    write_json(run / "calibration.json", {
        "status": "pending-baseline",
        "sampling_steps": config["model"]["sampling_steps"],
        "playout_delay_s": config["stream"]["playout_delay_s"],
    })
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(str(run) + "\n", encoding="utf-8")
    print(run)


def inspect_input(config: dict, path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Input does not exist: {path}")
    if path.suffix.lower() == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as fixture:
            if "motion_151" not in fixture:
                raise SystemExit("Recorded input requires motion_151.")
            frames = len(fixture["motion_151"])
        return {
            "path": str(path.resolve()), "sha256": sha256(path),
            "estimated_frames_30fps": frames, "duration_s": frames / 30.0,
            "clip_count": 1, "transition_count": 0,
            "metadata": {"format": f"{config.get('backend', 'recorded')}-fixture"},
            "input_format": "fixture",
            "root_scaled": None,
            "timeline_id": path.stem,
            "passed": True,
        }
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    pos = payload.get("pos")
    rotations = payload.get("q")
    if pos is None or rotations is None:
        raise SystemExit("Input requires pos and q arrays.")
    if len(pos) != len(rotations) or len(pos) < 300:
        raise SystemExit("Input pos/q sequences must align and provide at least 150 frames at 30 FPS.")
    metadata = payload.get("metadata", {})
    timeline_path = path.parent / "timeline.json"
    timeline = (
        json.loads(timeline_path.read_text(encoding="utf-8"))
        if timeline_path.is_file() else {}
    )
    root_scaled = timeline.get("flat_input_root_scaled")
    for sidecar_name in ("conversion.json", "manifest.json", "metadata.json"):
        sidecar_path = path.parent / sidecar_name
        if root_scaled is not None or not sidecar_path.is_file():
            continue
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        root_scaled = sidecar.get("root_scaled")
        if root_scaled is None:
            root_scaled = sidecar.get("input", {}).get("root_scaled")
    if root_scaled is None:
        root_scaled = config.get("paths", {}).get("root_scaled")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "raw_frames_60fps": len(pos),
        "estimated_frames_30fps": len(pos) // 2,
        "duration_s": len(pos) / 60.0,
        "clip_count": len(payload.get("clip_boundaries", [])) or 1,
        "transition_count": len(payload.get("transitions", [])),
        "metadata": metadata,
        "input_format": "aist",
        "root_scaled": root_scaled,
        "timeline_id": timeline.get("identity", metadata.get("format", path.stem)),
        "passed": True,
    }


def command_input(args) -> None:
    run = current_run(args.run)
    config_path = run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    path = resolve_path(args.input) if args.input else Path(config["paths"]["input_motion"])
    result = inspect_input(config, path)
    if args.root_scaled is not None and result["input_format"] == "aist":
        result["root_scaled"] = args.root_scaled == "true"
    if result["input_format"] == "aist" and not isinstance(result["root_scaled"], bool):
        raise SystemExit("AIST input requires a resolved root_scaled identity.")
    if args.lock:
        result.update({
            "schema": "duet-edge-input-manifest/v1",
            "status": "locked",
            "run_id": run.name,
            "config_sha256": sha256(config_path),
            "locked_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json(run / "input-manifest.json", result)
        write_json(run / "evidence" / "input.json", result)
    else:
        result["status"] = "checked"
        write_json(run / "evidence" / "input-check.json", result)
    print(json.dumps(result, indent=2))


def command_calibrate(args) -> None:
    run = current_run(args.run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary_path = Path(args.summary).resolve() if args.summary else run / "summary.json"
    inference_p99 = None
    baseline_summary = None
    if summary_path.is_file():
        baseline_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inference_p99 = baseline_summary.get("inference", {}).get("p99_ms")
    quality_summary = baseline_summary
    if args.quality_summary:
        quality_summary = json.loads(
            Path(args.quality_summary).resolve().read_text(encoding="utf-8")
        )
    if inference_p99 is None:
        raise SystemExit("Baseline summary has no inference p99 measurement.")
    safety_margin_ms = float(config["stream"]["safety_margin_ms"])
    inference_slo_ms = math.ceil(
        max(inference_p99 * 1.15, inference_p99 + 50.0) / 10.0
    ) * 10.0
    hop_period_ms = (
        config["stream"]["hop_frames"] / config["stream"]["fps"] * 1000.0
    )
    if inference_slo_ms + safety_margin_ms > hop_period_ms:
        failed = {
            "status": "throughput-failed",
            "sampling_steps": config["model"]["sampling_steps"],
            "measured_inference_p99_ms": inference_p99,
            "calculated_inference_slo_ms": inference_slo_ms,
            "hop_period_ms": hop_period_ms,
            "safety_margin_ms": safety_margin_ms,
        }
        write_json(run / "calibration.json", failed)
        raise SystemExit(json.dumps(failed, indent=2))
    delay = max(
        float(config["stream"]["playout_delay_s"]),
        (inference_slo_ms + safety_margin_ms) / 1000.0,
    )
    delay = math.ceil(delay * 20.0) / 20.0
    calibration = {
        "status": "finalized",
        "sampling_steps": config["model"]["sampling_steps"],
        "guidance_mode": "lead-only",
        "inference_slo_ms": inference_slo_ms,
        "playout_delay_s": round(delay, 3),
        "measured_inference_p99_ms": inference_p99,
        "safety_margin_ms": safety_margin_ms,
        "hop_period_ms": hop_period_ms,
        "source": "automated real-clock baseline",
    }
    if quality_summary:
        quality = quality_summary.get("motion_quality", {})
        def metric(path, field):
            return quality.get(path, {}).get(field)
        calibration["motion_thresholds"] = {
            "distinctness_p50_min": (
                None if metric("distinctness_body_centered", "p50") is None
                else metric("distinctness_body_centered", "p50") * 0.5
            ),
            "relative_root_horizontal_p99_max": (
                None if metric("relative_root_horizontal", "p99") is None
                else max(0.1, metric("relative_root_horizontal", "p99") * 1.5)
            ),
            "continuity_correction_max": (
                None if metric("continuity_correction", "max") is None
                else max(0.01, metric("continuity_correction", "max") * 1.5)
            ),
            "root_position_step_p99_max": (
                None if metric("root_position_step", "p99") is None
                else max(1e-4, metric("root_position_step", "p99") * 1.5)
            ),
            "ground_penetration_p99_max": (
                None if metric("ground_penetration", "p99") is None
                else max(0.01, metric("ground_penetration", "p99") * 1.5)
            ),
        }
    config["stream"]["inference_slo_ms"] = calibration["inference_slo_ms"]
    config["stream"]["playout_delay_s"] = calibration["playout_delay_s"]
    config["calibration"] = calibration
    write_json(run / "config.json", config)
    config_digest = sha256(run / "config.json")
    (run / "config.sha256").write_text(
        f"{config_digest}  config.json\n", encoding="utf-8"
    )
    calibration["config_sha256"] = config_digest
    write_json(run / "calibration.json", calibration)
    print(json.dumps(calibration, indent=2))


def command_report(args) -> None:
    run = current_run(args.run)
    config_digest = sha256(run / "config.json")
    recorded_config_digest = (
        (run / "config.sha256").read_text(encoding="utf-8").split()[0]
        if (run / "config.sha256").is_file() else None
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    calibration = (
        json.loads((run / "calibration.json").read_text(encoding="utf-8"))
        if (run / "calibration.json").is_file() else {}
    )
    stream_path = run / "stream.ndjson"
    messages = [json.loads(line) for line in stream_path.read_text(encoding="utf-8").splitlines()]
    frames = [item for item in messages if item.get("type") == "frame"]
    backend = summary.get("backend", {}).get("backend")
    checks = {
        "protocol_v3": bool(messages) and messages[0].get("protocol") == "duet-edge-stream/v3",
        "complete_lifecycle": summary.get("lifecycle", {}).get("final_state") == "finished",
        "exact_frame_count": summary.get("input", {}).get("frames") == len(frames) == summary.get("output", {}).get("frames"),
        "contiguous_sequence": [item.get("frame_id") for item in frames] == list(range(len(frames))),
        "runtime_clean": not summary.get("errors") and summary.get("queues", {}).get("overloads") == 0,
        "finite_24_joint_stream": all(
            len(item.get("lead_joints", [])) == len(item.get("companion_joints", [])) == 24
            and all(
                math.isfinite(value)
                for key in ("lead_joints", "companion_joints")
                for joint in item[key]
                for value in joint
            )
            for item in frames
        ),
        "handoff_sequence": (
            backend != "cuda" or summary.get("handoff", {}).get("used") == max(0, summary.get("inference", {}).get("count", 0) - 1)
        ),
        "lead_direct_fk_fidelity": summary.get("motion_quality", {}).get(
            "lead_overlap_fk_error", {}
        ).get("max", 0) <= 1e-5,
        "output_fps": summary.get("output", {}).get("observed_fps") is not None
        and 29.7 <= summary["output"]["observed_fps"] <= 30.3,
        "playout_jitter": summary.get("output", {}).get("jitter_p95_ms") is not None
        and summary["output"]["jitter_p95_ms"] <= summary["config"]["stream"]["jitter_slo_ms"],
        "zero_underflows": summary.get("output", {}).get("underflows") == 0,
    }
    input_manifest = (
        json.loads((run / "input-manifest.json").read_text(encoding="utf-8"))
        if (run / "input-manifest.json").is_file() else {}
    )
    checks["config_locked"] = recorded_config_digest == config_digest
    checks["input_manifest_locked"] = (
        input_manifest.get("status") == "locked"
        and input_manifest.get("passed") is True
        and input_manifest.get("config_sha256") == config_digest
    )
    input_path = Path(input_manifest.get("path", ""))
    checks["input_file_locked"] = (
        input_path.is_file()
        and sha256(input_path) == input_manifest.get("sha256")
    )
    for component in ("model", "stream", "viewer"):
        evidence_path = run / "evidence" / f"{component}-service.json"
        evidence = (
            json.loads(evidence_path.read_text(encoding="utf-8"))
            if evidence_path.is_file() else {}
        )
        checks[f"{component}_ready"] = (
            evidence.get("status") == "ready"
            and evidence.get("config_sha256") == config_digest
        )
    expected_input_hash = input_manifest.get("sha256")
    if expected_input_hash:
        checks["source_identity_hash"] = all(
            item.get("source_sha256") == expected_input_hash for item in frames
        )
    emission_times = [item.get("emitted_monotonic_offset_s") for item in frames]
    intervals_ms = [
        (right - left) * 1000 for left, right in zip(emission_times, emission_times[1:])
    ]
    checks["frame_send_interval"] = not intervals_ms or max(intervals_ms) < 50.0
    if backend == "cuda":
        inference = summary.get("inference", {})
        checks.update({
            "cuda_release_preset": summary["backend"].get("sampling_steps") == 50
            and summary["backend"].get("guidance_music") == 0
            and summary["backend"].get("guidance_lead") == 2,
            "inference_p99": inference.get("p99_ms") is not None
            and inference["p99_ms"] <= summary["config"]["stream"]["inference_slo_ms"],
            "inference_max": bool(inference.get("wall_ms"))
            and max(inference["wall_ms"]) <= (
                summary["config"]["stream"]["inference_slo_ms"]
                + summary["config"]["stream"]["safety_margin_ms"]
            ),
            "handoff_cuda_resident": summary.get("handoff", {}).get("state_bytes_max", 0) > 0,
        })
    thresholds = calibration.get("motion_thresholds", {})
    quality = summary.get("motion_quality", {})
    threshold_map = {
        "distinctness": ("distinctness_body_centered", "p50", "distinctness_p50_min", lambda a, b: a >= b),
        "relative_root_envelope": ("relative_root_horizontal", "p99", "relative_root_horizontal_p99_max", lambda a, b: a <= b),
        "continuity_correction": ("continuity_correction", "max", "continuity_correction_max", lambda a, b: a <= b),
        "root_boundary_envelope": ("root_position_step", "p99", "root_position_step_p99_max", lambda a, b: a <= b),
        "ground_penetration": ("ground_penetration", "p99", "ground_penetration_p99_max", lambda a, b: a <= b),
    }
    if backend in {"cuda", "recorded"}:
        for check_name, (metric_name, field, threshold_name, compare) in threshold_map.items():
            value, threshold = quality.get(metric_name, {}).get(field), thresholds.get(threshold_name)
            if value is not None and threshold is not None:
                checks[check_name] = compare(value, threshold)
    duration_s = len(frames) / summary["config"]["stream"]["fps"] if frames else 0
    if args.require_ten_minutes or args.long_input:
        checks["ten_minute_duration"] = len(frames) >= 18000 and duration_s >= 600
    if args.long_input:
        checks["long_input_backend_cuda"] = backend == "cuda"
        checks["viewer_connected"] = summary.get("clients", {}).get("peak_connected", 0) >= 1
        checks["viewer_zero_drops"] = summary.get("output", {}).get("dropped_view_frames") == 0
        checks["browser_zero_stalls"] = summary.get("clients", {}).get("visible_stalls") == 0
    passed = all(checks.values())
    result = {"passed": passed, "backend": backend, "frames": len(frames), "duration_s": duration_s, "checks": checks}
    write_json(run / "gate-results.json", result)
    report = [
        "# Duet-EDGE Realtime V2 Run Report", "",
        f"- Run: `{run.name}`", f"- Backend: `{backend}`",
        f"- Frames: `{len(frames)}`", f"- Duration: `{duration_s:.3f} s`",
        f"- Gate result: `{'PASS' if passed else 'REVIEW'}`", "",
        "## Automated gates", "",
    ]
    report.extend(f"- [{'x' if value else ' '}] {name}" for name, value in checks.items())
    report.extend(["", "## Evidence", "", "- `config.json` / `config.sha256`", "- `calibration.json`", "- `input-manifest.json`", "- `run-metadata.json`", "- `stream.ndjson`", "- `summary.json`", "- `gate-results.json`", ""])
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--template", default="configs/example.json")
    init.add_argument("--resume")
    init.add_argument("--output-root", default="outputs")
    init.add_argument("--state-file", default="outputs/.run-current")
    init.set_defaults(func=command_init)
    input_parser = sub.add_parser("input")
    input_parser.add_argument("--run")
    input_parser.add_argument("--input")
    input_parser.add_argument("--root-scaled", choices=("true", "false"))
    input_parser.add_argument("--lock", action="store_true")
    input_parser.set_defaults(func=command_input)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--run")
    calibrate.add_argument("--summary")
    calibrate.add_argument("--quality-summary")
    calibrate.set_defaults(func=command_calibrate)
    report = sub.add_parser("report")
    report.add_argument("--run")
    report.add_argument("--require-ten-minutes", action="store_true")
    report.add_argument("--long-input", action="store_true")
    report.set_defaults(func=command_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
