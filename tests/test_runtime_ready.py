import asyncio
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

from duet_edge_realtime.playout import VirtualClock
from duet_edge_realtime.runtime import RuntimeDaemon, sha256

from helpers import identity_motion


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RuntimeReadyTests(unittest.TestCase):
    def test_final_manual_matches_service_layout(self):
        repo = Path(__file__).resolve().parents[1]
        manual = (repo / "docs" / "FINAL_SERVICE_MANUAL.md").read_text()
        for command in ("start", "status", "test", "stop"):
            self.assertIn(f"service.sh {command}", manual)
        self.assertTrue((repo / "scripts/final_execution/service.sh").is_file())
        self.assertTrue((repo / "scripts/final_execution/runtime_service.sh").is_file())
        self.assertTrue((repo / "scripts/final_execution/lib/run.py").is_file())

    def test_components_wait_until_locked_input_is_started(self):
        async def scenario(root: Path):
            fixture = root / "fixture.npz"
            np.savez(fixture, motion_151=identity_motion(150))
            web_root = root / "web"
            web_root.mkdir()
            (web_root / "index.html").write_text("viewer", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "backend": "fake",
                "paths": {"input_motion": str(fixture), "output_dir": str(root)},
                "stream": {"playout_delay_s": 0.75},
                "server": {
                    "bind_host": "127.0.0.1",
                    "port": available_port(),
                    "web_port": available_port(),
                    "control_port": available_port(),
                    "ingest_port": available_port(),
                    "web_root": str(web_root),
                },
            }), encoding="utf-8")
            config_digest = sha256(config_path)
            (root / "config.sha256").write_text(
                f"{config_digest}  config.json\n", encoding="utf-8"
            )
            (root / "calibration.json").write_text(
                json.dumps({"status": "finalized"}), encoding="utf-8"
            )
            daemon = RuntimeDaemon(config_path, root)
            control_url = f"http://127.0.0.1:{daemon.config.control_port}"

            def control(method: str, path: str) -> dict:
                request = urllib.request.Request(
                    control_url + path,
                    method=method,
                    data=b"" if method == "POST" else None,
                )
                with urllib.request.urlopen(request, timeout=3) as response:
                    return json.loads(response.read())

            try:
                await daemon.start_control()
                await daemon.initialize_model()
                self.assertEqual(daemon.status()["model"]["state"], "ready")
                self.assertEqual(daemon.status()["session"]["state"], "idle")
                await asyncio.to_thread(control, "POST", "/stream/start")
                await asyncio.to_thread(control, "POST", "/viewer/start")
                self.assertEqual(daemon.status()["stream"]["state"], "ready")
                self.assertEqual(daemon.status()["viewer"]["state"], "ready")
                self.assertFalse((root / "stream.ndjson").exists())
                manifest = {
                    "schema": "duet-edge-input-manifest/v1",
                    "status": "locked",
                    "passed": True,
                    "run_id": root.name,
                    "config_sha256": config_digest,
                    "path": str(fixture),
                    "sha256": sha256(fixture),
                    "input_format": "fixture",
                    "root_scaled": None,
                    "timeline_id": "fixture",
                }
                (root / "input-manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with patch("duet_edge_realtime.runtime.RealtimeClock", VirtualClock):
                    await asyncio.to_thread(control, "POST", "/run/start")
                    await daemon.session_task
                self.assertEqual(daemon.status()["session"]["state"], "finished")
                self.assertTrue((root / "summary.json").is_file())
                self.assertTrue((root / "stream.ndjson").is_file())
            finally:
                await daemon.close()

        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asyncio.run(scenario(root))
            report = subprocess.run([
                sys.executable,
                "scripts/final_execution/lib/run.py",
                "report",
                "--run", str(root),
            ], cwd=repo, text=True, capture_output=True)
            self.assertEqual(report.returncode, 0, report.stdout + report.stderr)
            self.assertTrue(json.loads((root / "gate-results.json").read_text())["passed"])

    def test_calibration_finalizes_fixed_sampling_and_config_hash(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "config.json").write_text(json.dumps({
                "model": {"sampling_steps": 37},
                "stream": {
                    "fps": 30,
                    "hop_frames": 75,
                    "playout_delay_s": 0.75,
                    "inference_slo_ms": 650,
                    "safety_margin_ms": 50,
                },
            }), encoding="utf-8")
            summary = run / "baseline-summary.json"
            summary.write_text(json.dumps({
                "inference": {"p99_ms": 500.0},
                "motion_quality": {},
            }), encoding="utf-8")
            result = subprocess.run([
                sys.executable,
                "scripts/final_execution/lib/run.py",
                "calibrate",
                "--run", str(run),
                "--summary", str(summary),
            ], cwd=repo, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((run / "config.json").read_text())
            self.assertEqual(config["model"]["sampling_steps"], 37)
            self.assertEqual(config["stream"]["inference_slo_ms"], 580.0)
            expected = hashlib.sha256((run / "config.json").read_bytes()).hexdigest()
            self.assertEqual(
                (run / "config.sha256").read_text().split()[0], expected
            )


if __name__ == "__main__":
    unittest.main()
