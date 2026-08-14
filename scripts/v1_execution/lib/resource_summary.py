#!/usr/bin/env python3
"""Summarize the automatically recorded NVIDIA CSV evidence."""

import argparse
import csv
import json
import re
from pathlib import Path


def number(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = {"samples": len(rows), "first_timestamp": None, "last_timestamp": None}
    if rows:
        payload["first_timestamp"] = rows[0].get("timestamp")
        payload["last_timestamp"] = rows[-1].get("timestamp")
        for label, key in (
            ("max_gpu_utilization_percent", " utilization.gpu [%]"),
            ("max_memory_used_mib", " memory.used [MiB]"),
            ("max_power_draw_w", " power.draw [W]"),
            ("max_temperature_c", " temperature.gpu"),
        ):
            values = [number(row.get(key, row.get(key.strip(), "0"))) for row in rows]
            payload[label] = max(values, default=0.0)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

