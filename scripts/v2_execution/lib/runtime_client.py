#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path

from duet_edge_realtime.progress import TerminalProgress


def draw_wait_progress(
    elapsed: float,
    label: str,
    final: bool = False,
    ratio: float | None = None,
    emit_non_tty: bool = False,
) -> None:
    """Render one stable wait line instead of dumping every polled status event."""
    width = 28
    measured_progress = ratio is not None
    if final and measured_progress:
        ratio = 1.0
        bar = "=" * width
        percent = "100%"
    elif final:
        bar = ""
        percent = ""
    elif measured_progress:
        ratio = ratio or 0.0
        filled = min(width, int(width * ratio))
        bar = "=" * filled + "." * (width - filled)
        percent = f"{ratio * 100:3.0f}%"
    else:
        bar = ""
        percent = ""
    suffix = (
        "ready" if final
        else f"{elapsed:5.1f}s elapsed"
    )
    line = (
        f"[{bar}] {percent}  {label} · {suffix}"
        if measured_progress
        else f"{label} · {suffix}"
    )
    if os.isatty(1):
        print("\r" + line.ljust(80), end="\n" if final else "", flush=True)
    elif final or emit_non_tty:
        print(line, flush=True)


def session_ratio(status: dict) -> float | None:
    progress = status.get("session", {}).get("progress") or {}
    current, total = progress.get("output_frames"), progress.get("total_frames")
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        return min(1.0, current / total)
    return None


def model_progress_event(status: dict, field: str) -> dict | None:
    if field.startswith("model."):
        event = status.get("model", {}).get("progress")
        expected_phase = "warmup"
    else:
        session = status.get("session", {}).get("progress") or {}
        event = session.get("sampling") or status.get("model", {}).get("progress")
        expected_phase = "inference"
    if not isinstance(event, dict):
        return None
    if event.get("phase") != expected_phase:
        return None
    if not event.get("windows") or not event.get("steps"):
        return None
    return event


def read_config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text(encoding="utf-8"))


def base_url(run: Path) -> str:
    server = read_config(run)["server"]
    host = server.get("bind_host", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{server.get('control_port', 8766)}"


def validate_identity(run: Path, payload: dict) -> None:
    actual_run_id = payload.get("run_id")
    if actual_run_id != run.name:
        owner_run = run.parent / str(actual_run_id)
        service_script = Path(__file__).resolve().parents[1] / "service.sh"
        raise RuntimeError(
            "Control port belongs to a different runtime: "
            f"expected run_id={run.name}, got run_id={actual_run_id!r}. "
            "Stop the owning runtime before starting this run.\n"
            f"Run: bash {shlex.quote(str(service_script))} stop --run "
            f"{shlex.quote(str(owner_run))}"
        )

def raw_request(run: Path, method: str, path: str) -> dict:
    value = urllib.request.Request(
        base_url(run) + path,
        method=method,
        data=b"" if method == "POST" else None,
    )
    try:
        with urllib.request.urlopen(value, timeout=3) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read())
        raise RuntimeError(payload.get("error", str(exc))) from exc


def request(run: Path, method: str, path: str) -> dict:
    if method != "GET":
        validate_identity(run, raw_request(run, "GET", "/status"))
    payload = raw_request(run, method, path)
    validate_identity(run, payload)
    return payload


def nested(value: dict, field: str):
    for part in field.split("."):
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("start-stream")
    sub.add_parser("start-viewer")
    sub.add_parser("start-run")
    sub.add_parser("shutdown")
    wait = sub.add_parser("wait")
    wait.add_argument("--field", required=True)
    wait.add_argument("--value", required=True)
    wait.add_argument("--fail-value", action="append", default=[])
    wait.add_argument("--timeout", type=float, required=True)
    wait.add_argument("--interval", type=float, default=1.0)
    wait.add_argument("--label", default="Waiting for runtime state")
    wait.add_argument("--show-final-status", action="store_true")
    args = parser.parse_args()
    run = Path(args.run).resolve()
    endpoints = {
        "status": ("GET", "/status"),
        "start-stream": ("POST", "/stream/start"),
        "start-viewer": ("POST", "/viewer/start"),
        "start-run": ("POST", "/run/start"),
        "shutdown": ("POST", "/shutdown"),
    }
    if args.command != "wait":
        method, path = endpoints[args.command]
        try:
            result = request(run, method, path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.command == "status":
            print(json.dumps(result, indent=2))
        else:
            print(f"{args.command}: accepted")
        return
    started = time.monotonic()
    deadline = started + args.timeout
    last = None
    model_display = TerminalProgress(True)
    last_model_key = None
    last_non_tty_report = started
    last_poll_error = None
    warned_old_runtime = False
    draw_wait_progress(
        0.0, args.label, emit_non_tty=not os.isatty(1)
    )
    while time.monotonic() < deadline:
        active_model_event = None
        try:
            last = request(run, "GET", "/status")
            event = model_progress_event(last, args.field)
            event_key = (
                event.get("phase"), event.get("window"), event.get("step")
            ) if event else None
            event_changed = event is not None and event_key != last_model_key
            if event is not None and (os.isatty(1) or event_changed):
                model_display.model_update(event, force=True)
                last_model_key = event_key
            event_complete = event is not None and (
                event.get("window") == event.get("windows")
                and event.get("step") == event.get("steps")
            )
            active_model_event = event if not event_complete or event_changed else None
            current = str(nested(last, args.field))
            if (
                not args.field.startswith("model.")
                and current in {"preparing", "starting", "running"}
                and "progress" not in last.get("model", {})
                and not warned_old_runtime
            ):
                print(
                    "Sampling progress is unavailable from the resident runtime; "
                    "restart Stage 04 after this run to load the current runtime code.",
                    flush=True,
                )
                warned_old_runtime = True
            if current == args.value:
                draw_wait_progress(
                    time.monotonic() - started,
                    args.label,
                    final=True,
                    ratio=session_ratio(last),
                )
                if args.show_final_status:
                    print(json.dumps(last, indent=2), flush=True)
                return
            if current in args.fail_value:
                detail = last.get("error") or last.get("session", {}).get("error")
                raise SystemExit(
                    f"{args.label} failed: {args.field}={current}"
                    + (f" · {detail}" if detail else "")
                )
            last_poll_error = None
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        except (OSError, RuntimeError, KeyError) as exc:
            last_poll_error = str(exc)
            pid_path = run / "runtime.pid"
            if pid_path.is_file():
                try:
                    os.kill(int(pid_path.read_text().strip()), 0)
                except (ProcessLookupError, ValueError):
                    raise SystemExit("Runtime process exited before reaching the requested state.")
        if os.isatty(1) and active_model_event is None:
            draw_wait_progress(
                time.monotonic() - started,
                args.label,
                ratio=session_ratio(last or {}),
            )
        elif (
            not os.isatty(1)
            and active_model_event is None
            and time.monotonic() - last_non_tty_report >= 5.0
        ):
            draw_wait_progress(
                time.monotonic() - started,
                (
                    f"{args.label} (retrying status: {last_poll_error})"
                    if last_poll_error else args.label
                ),
                ratio=session_ratio(last or {}),
                emit_non_tty=True,
            )
            last_non_tty_report = time.monotonic()
        time.sleep(args.interval)
    raise SystemExit(
        f"Timed out waiting for {args.field}={args.value}: "
        + json.dumps(last, indent=2)
    )


if __name__ == "__main__":
    main()
