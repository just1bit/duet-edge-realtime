#!/usr/bin/env python3
"""Run one V2 stage command while mirroring its console to one log file."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


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
    if deferred:
        handle = tempfile.NamedTemporaryFile(
            prefix="duet-stage-01-", suffix=".log", delete=False
        )
        handle.close()
        log_path = Path(handle.name)
    else:
        log_path = Path(args.run_root) / "logs" / f"stage-{args.stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    status = 127
    with log_path.open("w", encoding="utf-8") as log:
        if not deferred:
            message = f"Console log: {log_path}\n"
            sys.stdout.write(message)
            log.write(message)
        log.flush()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            def forward(signum: int, _frame: object) -> None:
                if process is not None and process.poll() is None:
                    process.send_signal(signum)

            previous_int = signal.signal(signal.SIGINT, forward)
            previous_term = signal.signal(signal.SIGTERM, forward)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                status = process.wait()
            finally:
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGTERM, previous_term)
        except OSError as exc:
            message = f"Unable to start stage: {exc}\n"
            sys.stdout.write(message)
            log.write(message)

    status = status if status >= 0 else 128 - status
    if deferred and status == 0:
        state_path = Path(args.state_file)
        if "--state-file" in command:
            state_path = Path(command[command.index("--state-file") + 1])
        if not state_path.is_absolute():
            state_path = Path.cwd() / state_path
        run_root = Path(state_path.read_text(encoding="utf-8").splitlines()[0])
        target = run_root / "logs" / f"stage-{args.stage}.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        log_path.replace(target)
        message = f"Console log: {target}\n"
        sys.stdout.write(message)
        with target.open("a", encoding="utf-8") as log:
            log.write(message)
    elif deferred:
        print(f"Pending console log: {log_path}", file=sys.stderr)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
