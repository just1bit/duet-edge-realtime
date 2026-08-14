#!/usr/bin/env python3
"""Build a concise Markdown index of all automatic acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    metadata = json.loads((root / "run-metadata.json").read_text(encoding="utf-8"))
    results = []
    for path in sorted((root / "stage-results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["result_path"] = path
        results.append(data)
    lines = [
        "# V1 Acceptance Report", "",
        f"- Run: `{root.name}`",
        f"- Created: `{metadata['created_at']}`",
        f"- Machine: `{metadata['machine']}`",
        f"- Profile: `{metadata.get('acceptance_profile', 'gpu')}`", "",
        "## Automatic Stage Results", "",
        "| Stage | Action | Attempt | Outcome | Result | Log |",
        "|---|---|---:|---|---|---|",
    ]
    for item in results:
        outcome = "Skipped" if item.get("skipped") else "Pass" if item["passed"] else "Review"
        log = Path(item["log"])
        result_path = Path(item["result_path"])
        lines.append(f"| {item['stage']} | {item['script']} | {item['attempt']} | {outcome} | [{rel(result_path, root)}]({rel(result_path, root)}) | [{rel(log, root)}]({rel(log, root)}) |")
    lines.extend(["", "## Evidence Index", ""])
    evidence_paths = {path for path in (root / "evidence").rglob("*") if path.is_file()}
    for pattern in ("summary.json", "stream.ndjson", "effective_config.json", "real_fixture.npz", "input_motion.pkl"):
        evidence_paths.update(path for path in root.rglob(pattern) if path.is_file())
    for path in sorted(evidence_paths):
        relative = rel(path, root)
        lines.append(f"- [{relative}]({relative})")
    lines.extend([
        "", "## Acceptance Notes", "",
        "- [acceptance-notes.md](acceptance-notes.md)", "",
    ])
    target = root / "acceptance-report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
