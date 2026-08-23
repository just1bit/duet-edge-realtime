#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


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
        print(json.dumps(request(run, method, path), indent=2))
        return
    deadline = time.monotonic() + args.timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = request(run, "GET", "/status")
            current = str(nested(last, args.field))
            if current == args.value:
                print(json.dumps(last, indent=2))
                return
            if current in args.fail_value:
                raise SystemExit(json.dumps(last, indent=2))
        except (OSError, RuntimeError, KeyError):
            pid_path = run / "runtime.pid"
            if pid_path.is_file():
                try:
                    os.kill(int(pid_path.read_text().strip()), 0)
                except (ProcessLookupError, ValueError):
                    raise SystemExit("Runtime process exited before reaching the requested state.")
        time.sleep(args.interval)
    raise SystemExit(
        f"Timed out waiting for {args.field}={args.value}: "
        + json.dumps(last, indent=2)
    )


if __name__ == "__main__":
    main()
