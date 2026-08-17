#!/usr/bin/env python3
"""Create the directory structure and human note template for one run."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


NOTES = """# Acceptance Notes

## Viewer Review

- [ ] Blue lead and cyan companion are both visible and labelled:
- [ ] Both skeletons are upright in the default Z-up view:
- [ ] Limbs move relative to their roots continuously (not frozen/root-only):
- [ ] Lifecycle, reconnect, fake replay, and profile-required real replay pass:
- Screenshot/timestamp evidence:

## Performance

## Manual Changes

## Final Review

"""


def repository_identity(path: Path) -> dict[str, str]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *arguments], text=True,
            capture_output=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    remote = git("config", "--get", "remote.origin.url").removesuffix(".git")
    normalized = remote.replace("git@", "").replace(":", "/")
    parts = [part for part in normalized.split("/") if part]
    name = "/".join(parts[-2:]) if len(parts) >= 2 else path.name
    branch = git("symbolic-ref", "--short", "HEAD") or "DETACHED"
    return {"name": name, "branch": branch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realtime-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--profile", choices=("gpu", "local"), default="gpu")
    args = parser.parse_args()
    realtime_root = Path(args.realtime_root).resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = realtime_root / "outputs" / f"acceptance-{timestamp}"
    suffix = 1
    while run_root.exists():
        run_root = realtime_root / "outputs" / f"acceptance-{timestamp}-{suffix}"
        suffix += 1
    for relative in (
        "logs", "stage-results", "evidence/environment", "evidence/preflight",
        "evidence/benchmarks", "evidence/resources", "candidate-configs",
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    (run_root / "acceptance-notes.md").write_text(NOTES, encoding="utf-8")
    project_root = Path(args.project_root).resolve()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "realtime_root": str(realtime_root),
        "run_root": str(run_root),
        "machine": platform.node(),
        "platform": platform.platform(),
        "acceptance_profile": args.profile,
        "repositories": {
            "duet_edge_realtime": repository_identity(realtime_root),
            "duet_edge": repository_identity(project_root / "duet-edge"),
        },
    }
    (run_root / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    state_file = Path(args.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(str(run_root) + "\n", encoding="utf-8")
    print(f"Acceptance run created: {run_root}")
    print(f"Acceptance notes: {run_root / 'acceptance-notes.md'}")


if __name__ == "__main__":
    main()
