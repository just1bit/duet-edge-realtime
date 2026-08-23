#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def draw_wait_progress(
    elapsed: float,
    timeout: float,
    label: str,
    final: bool = False,
    ratio: float | None = None,
) -> None:
    """Render one stable wait line instead of dumping every polled status event."""
    width = 28
    measured_progress = ratio is not None
    ratio = 1.0 if final else (
        min(1.0, elapsed / timeout) if ratio is None and timeout > 0 else (ratio or 0.0)
    )
    filled = min(width, int(width * ratio))
    bar = "=" * filled + "." * (width - filled)
    suffix = (
        "ready" if final
        else f"{elapsed:5.1f}s elapsed" if measured_progress
        else f"{elapsed:5.1f}s / {timeout:.0f}s"
    )
    line = f"[{bar}] {ratio * 100:3.0f}%  {label} · {suffix}"
    if os.isatty(1):
        print("\r" + line, end="\n" if final else "", flush=True)
    elif final:
        print(line)


def session_ratio(status: dict) -> float | None:
    progress = status.get("session", {}).get("progress", {})
    current, total = progress.get("output_frames"), progress.get("total_frames")
    if isinstance(current, int) and isinstance(total, int) and total > 0:
        return min(1.0, current / total)
    return None


def read_config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text(encoding="utf-8"))


def base_url(run: Path) -> str:
    server = read_config(run)["server"]
    host = server.get("bind_host", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{server.get('control_port', 8766)}"


def request(run: Path, method: str, path: str) -> dict:
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
        result = request(run, method, path)
        if args.command == "status":
            print(json.dumps(result, indent=2))
        else:
            print(f"{args.command}: accepted")
        return
    started = time.monotonic()
    deadline = started + args.timeout
    last = None
    if os.isatty(1):
        draw_wait_progress(0.0, args.timeout, args.label)
    while time.monotonic() < deadline:
        try:
            last = request(run, "GET", "/status")
            current = str(nested(last, args.field))
            if current == args.value:
                draw_wait_progress(time.monotonic() - started, args.timeout, args.label, final=True)
                return
            if current in args.fail_value:
                detail = last.get("error") or last.get("session", {}).get("error")
                raise SystemExit(
                    f"{args.label} failed: {args.field}={current}"
                    + (f" · {detail}" if detail else "")
                )
        except (OSError, RuntimeError, KeyError):
            pid_path = run / "runtime.pid"
            if pid_path.is_file():
                try:
                    os.kill(int(pid_path.read_text().strip()), 0)
                except (ProcessLookupError, ValueError):
                    raise SystemExit("Runtime process exited before reaching the requested state.")
        if os.isatty(1):
            draw_wait_progress(
                time.monotonic() - started,
                args.timeout,
                args.label,
                ratio=session_ratio(last or {}),
            )
        time.sleep(args.interval)
    raise SystemExit(
        f"Timed out waiting for {args.field}={args.value}: "
        + json.dumps(last, indent=2)
    )


if __name__ == "__main__":
    main()
