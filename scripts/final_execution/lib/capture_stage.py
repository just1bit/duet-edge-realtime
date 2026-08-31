#!/usr/bin/env python3
"""Run one service stage while mirroring its console to one log file."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def forward_stop_signals(process: subprocess.Popen[str]) -> Iterator[None]:
    """Forward terminal stop signals for the child's entire lifetime."""
    def forward(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    previous_int = signal.signal(signal.SIGINT, forward)
    previous_term = signal.signal(signal.SIGTERM, forward)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def wait_for(process: subprocess.Popen[str]) -> int:
    with forward_stop_signals(process):
        return process.wait()


def run_without_log(command: list[str]) -> int:
    """Keep the stage runnable when its optional log cannot be opened."""
    try:
        return wait_for(subprocess.Popen(command))
    except OSError as exc:
        print(f"Unable to start stage: {exc}", file=sys.stderr)
        return 127


def write_log(
    log: TextIO | None, message: str, *, flush: bool = False
) -> TextIO | None:
    if log is None:
        return None
    try:
        log.write(message)
        if flush:
            log.flush()
        return log
    except OSError as exc:
        print(
            f"Warning: console log write failed ({exc}); "
            "continuing without archival.",
            file=sys.stderr,
        )
        try:
            log.close()
        except OSError:
            pass
        return None


def close_log(log: TextIO | None) -> None:
    if log is not None:
        try:
            log.close()
        except OSError as exc:
            print(f"Warning: console log close failed ({exc}).", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")

    deferred = args.run_root is None
    try:
        if deferred:
            handle = tempfile.NamedTemporaryFile(
                prefix="duet-stage-01-", suffix=".log", delete=False
            )
            handle.close()
            log_path = Path(handle.name)
        else:
            log_path = Path(args.run_root) / "logs" / f"stage-{args.stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8", buffering=1)
    except OSError as exc:
        print(
            f"Warning: console log unavailable ({exc}); continuing without archival.",
            file=sys.stderr,
        )
        status = run_without_log(command)
        raise SystemExit(status if status >= 0 else 128 - status)

    process = None
    status = 127
    try:
        if not deferred:
            message = f"Console log: {log_path}\n"
            sys.stdout.write(message)
            log = write_log(log, message, flush=True)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with forward_stop_signals(process):
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log = write_log(log, line)
                status = process.wait()
        except OSError as exc:
            message = f"Unable to start stage: {exc}\n"
            sys.stdout.write(message)
            log = write_log(log, message)
    finally:
        close_log(log)

    status = status if status >= 0 else 128 - status
    if deferred and status == 0:
        retained_path = log_path
        try:
            state_path = Path(args.state_file)
            if "--state-file" in command:
                state_path = Path(command[command.index("--state-file") + 1])
            if not state_path.is_absolute():
                state_path = Path.cwd() / state_path
            run_root = Path(state_path.read_text(encoding="utf-8").splitlines()[0])
            target = run_root / "logs" / f"stage-{args.stage}.log"
            target.parent.mkdir(parents=True, exist_ok=True)
            log_path.replace(target)
            retained_path = target
            message = f"Console log: {target}\n"
            sys.stdout.write(message)
            with target.open("a", encoding="utf-8") as target_log:
                target_log.write(message)
        except (IndexError, OSError) as exc:
            print(
                f"Warning: could not move Stage 01 log ({exc}); "
                f"log retained at: {retained_path}",
                file=sys.stderr,
            )
    elif deferred:
        print(f"Pending console log: {log_path}", file=sys.stderr)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
