#!/usr/bin/env python3
"""Run one acceptance action and archive its complete automatic evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_attempt(result_dir: Path, prefix: str) -> int:
    return len(list(result_dir.glob(f"{prefix}-*.json"))) + 1


def file_snapshot(root: Path) -> dict[Path, int]:
    return {
        path: path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--next-action", required=True)
    parser.add_argument("--accept-signal", action="append", default=[])
    parser.add_argument("--skip-reason")
    parser.add_argument("--precondition-error")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    valid_signals = {"INT", "SIGINT", "TERM", "SIGTERM"}
    invalid_signals = [name for name in args.accept_signal if name.upper() not in valid_signals]
    if invalid_signals:
        parser.error(f"unsupported accepted signal: {invalid_signals[0]}")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command and not args.skip_reason and not args.precondition_error:
        parser.error("provide a command after --")

    run_root = Path(args.run_root).resolve()
    log_dir = run_root / "logs"
    result_dir = run_root / "stage-results"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.stage}-{args.name}"
    attempt = next_attempt(result_dir, prefix)
    stem = f"{prefix}-{attempt:02d}"
    log_path = log_dir / f"{stem}.log"
    result_path = result_dir / f"{stem}.json"
    started_at = utc_now()
    before = file_snapshot(run_root)
    input_paths = []
    for token in command:
        candidate = Path(token)
        if candidate.exists():
            input_paths.append(str(candidate.resolve()))

    print(f"[{args.stage}] {args.name} (attempt {attempt})")
    print(f"Log: {log_path}")
    status = 0 if args.skip_reason else 1
    failure_kind = "precondition" if args.precondition_error else None
    error = args.precondition_error
    accepted_termination = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {started_at}\n")
        log.write(f"Command: {subprocess.list2cmdline(command) if command else '(none)'}\n\n")
        if args.skip_reason:
            log.write(f"Skipped: {args.skip_reason}\n")
        if args.precondition_error:
            log.write(f"Precondition: {args.precondition_error}\n")
        log.flush()
        if command:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=os.environ.get("REALTIME_ROOT"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                def forward_signal(signum: int, _frame: object) -> None:
                    if process.poll() is None:
                        process.send_signal(signum)

                previous_int = signal.signal(signal.SIGINT, forward_signal)
                previous_term = signal.signal(signal.SIGTERM, forward_signal)
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        sys.stdout.write(line)
                        log.write(line)
                    status = process.wait()
                finally:
                    signal.signal(signal.SIGINT, previous_int)
                    signal.signal(signal.SIGTERM, previous_term)
            except OSError as exc:
                status = 127
                failure_kind = "command_start"
                error = str(exc)
                message = f"Unable to start command: {exc}\n"
                sys.stdout.write(message)
                log.write(message)

    signal_numbers = {
        "INT": signal.SIGINT, "SIGINT": signal.SIGINT,
        "TERM": signal.SIGTERM, "SIGTERM": signal.SIGTERM,
    }
    accepted_numbers = {signal_numbers[name.upper()] for name in args.accept_signal if name.upper() in signal_numbers}
    observed_signal = -status if status < 0 else status - 128 if status >= 128 else None
    if observed_signal in accepted_numbers:
        accepted_termination = True
    passed = args.skip_reason is not None or status == 0 or accepted_termination
    if not passed and failure_kind is None:
        failure_kind = "command_exit"

    completed_at = utc_now()
    after = file_snapshot(run_root)
    output_paths = [
        str(path)
        for path, modified_at in after.items()
        if path not in before or before[path] != modified_at
    ]
    metric_paths = [
        str(path)
        for path in after
        if path.name in {"summary.json", "benchmark.json"}
        or path.name.startswith("quality-")
        or path.name.endswith("-summary.json")
    ]
    result = {
        "stage": args.stage,
        "script": args.name,
        "attempt": attempt,
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command,
        "exit_status": status,
        "passed": passed,
        "automatic_validation": "not_applicable" if args.skip_reason else "passed" if passed else "review",
        "skipped": args.skip_reason is not None,
        "skip_reason": args.skip_reason,
        "failure_kind": failure_kind,
        "error": error,
        "accepted_termination": accepted_termination,
        "inputs": sorted(set(input_paths)),
        "outputs": sorted(set(output_paths)),
        "metrics_and_summaries": sorted(set(metric_paths)),
        "reused_evidence": [],
        "log": str(log_path),
        "next_action": args.next_action,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Result: {result_path}")
    if args.skip_reason:
        print(f"Stage action skipped: {args.skip_reason}")
    else:
        print("Stage action passed." if passed else f"Next action: {args.next_action}")
    raise SystemExit(0 if passed else status or 1)


if __name__ == "__main__":
    main()
