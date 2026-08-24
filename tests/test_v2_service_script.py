import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from helpers import identity_motion


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class V2ServiceScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        cls.service = cls.repo / "scripts/v2_execution/service.sh"

    def run_service(self, run: Path, *args: str, timeout: float = 30):
        environment = os.environ.copy()
        environment["PYTHON_BIN"] = sys.executable
        return subprocess.run(
            ["bash", str(self.service), *args, "--run", str(run)],
            cwd=self.repo,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def make_run(self, root: Path) -> Path:
        run = root / "run-service-test"
        run.mkdir()
        fixture = root / "fixture.npz"
        np.savez(fixture, motion_151=identity_motion(150))
        default_fixture = root / "default-fixture.npz"
        np.savez(
            default_fixture,
            motion_151=identity_motion(150, root_step=0.02),
        )
        np.savez(root / "invalid-shape.npz", motion_151=np.zeros((150, 150)))
        np.savez(root / "long-fixture.npz", motion_151=identity_motion(600))
        web_root = root / "web"
        web_root.mkdir()
        (web_root / "index.html").write_text("viewer", encoding="utf-8")
        config = {
            "backend": "fake",
            "paths": {
                "input_motion": str(default_fixture),
                "output_dir": str(run),
            },
            "stream": {
                "fps": 300,
                "window_frames": 150,
                "hop_frames": 75,
                "playout_delay_s": 0.2,
                "inference_slo_ms": 100,
                "safety_margin_ms": 50,
            },
            "server": {
                "bind_host": "127.0.0.1",
                "port": available_port(),
                "web_port": available_port(),
                "control_port": available_port(),
                "web_root": str(web_root),
            },
        }
        config_path = run / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        (run / "config.sha256").write_text(
            f"{digest}  config.json\n", encoding="utf-8"
        )
        return run

    @staticmethod
    def assert_stage(result, number: str, title: str, success: bool):
        outcome = "SUCCESS" if success else "FAILED"
        combined = result.stdout + result.stderr
        assert f"Stage {number} · {title}" in combined
        assert f"Stage {number} {outcome} · {title}" in combined

    @staticmethod
    def wait_for_exit(run: Path, timeout: float = 10) -> bool:
        pid = int((run / "runtime.pid").read_text(encoding="utf-8"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.05)
        return False

    def test_usage_rejects_unsupported_command_shapes(self):
        cases = (
            (),
            ("unknown",),
            ("model",),
            ("model", "status"),
        )
        for args in cases:
            with self.subTest(args=args):
                result = subprocess.run(
                    ["bash", str(self.service), *args],
                    cwd=self.repo,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("Usage:", result.stderr)

    def test_complete_service_lifecycle_and_stage_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root)
            fixture = root / "fixture.npz"
            runtime_pid = None
            try:
                result = self.run_service(run, "stream", "start")
                self.assertNotEqual(result.returncode, 0)
                self.assert_stage(result, "05", "Realtime Stream Service · start", False)

                model = self.run_service(run, "model", "start")
                self.assertEqual(model.returncode, 0, model.stdout + model.stderr)
                self.assert_stage(model, "04", "Model Service · start", True)
                self.assertIn("Runtime process started", model.stdout)
                self.assertIn("Model service ready", model.stdout)
                runtime_pid = int((run / "runtime.pid").read_text(encoding="utf-8"))
                os.kill(runtime_pid, 0)
                self.assertTrue((run / "evidence/model-service.json").is_file())
                self.assertEqual(
                    (run / "logs/stage-04.log").read_text(encoding="utf-8"),
                    model.stdout,
                )

                reused = self.run_service(run, "model", "start")
                self.assertEqual(reused.returncode, 0, reused.stdout + reused.stderr)
                self.assertIn("Existing model process reused", reused.stdout)
                self.assertEqual(
                    int((run / "runtime.pid").read_text(encoding="utf-8")),
                    runtime_pid,
                )

                viewer_early = self.run_service(run, "viewer", "start")
                self.assertNotEqual(viewer_early.returncode, 0)
                self.assertIn("stream service is not ready", viewer_early.stdout)
                self.assert_stage(viewer_early, "06", "Viewer Web · start", False)

                test_early = self.run_service(run, "test", str(fixture))
                self.assertNotEqual(test_early.returncode, 0)
                self.assertIn("Services not ready: stream, viewer", test_early.stdout)
                self.assert_stage(
                    test_early, "07", "Prepare and Lock Formal Input", False
                )
                self.assertFalse((run / "input-manifest.json").exists())

                stream = self.run_service(run, "stream", "start")
                self.assertEqual(stream.returncode, 0, stream.stdout + stream.stderr)
                self.assert_stage(
                    stream, "05", "Realtime Stream Service · start", True
                )
                self.assertIn("Start request accepted", stream.stdout)
                self.assertIn("Realtime stream service ready", stream.stdout)
                self.assertTrue((run / "evidence/stream-service.json").is_file())
                self.assertEqual(
                    (run / "logs/stage-05.log").read_text(encoding="utf-8"),
                    stream.stdout,
                )
                stream_reused = self.run_service(run, "stream", "start")
                self.assertEqual(stream_reused.returncode, 0)

                viewer = self.run_service(run, "viewer", "start")
                self.assertEqual(viewer.returncode, 0, viewer.stdout + viewer.stderr)
                self.assert_stage(viewer, "06", "Viewer Web · start", True)
                self.assertIn("Viewer start request accepted", viewer.stdout)
                self.assertIn("Viewer service ready", viewer.stdout)
                self.assertIn("Viewer ready and waiting for input:", viewer.stdout)
                self.assertIn("Viewer URL generated", viewer.stdout)
                self.assertTrue((run / "evidence/viewer-service.json").is_file())
                self.assertEqual(
                    (run / "logs/stage-06.log").read_text(encoding="utf-8"),
                    viewer.stdout,
                )
                viewer_reused = self.run_service(run, "viewer", "start")
                self.assertEqual(viewer_reused.returncode, 0)

                status = self.run_service(run, "status")
                self.assertEqual(status.returncode, 0, status.stdout + status.stderr)
                self.assert_stage(status, "04", "Model Service · status", True)
                self.assertIn('"state": "ready"', status.stdout)
                self.assertIn('"state": "idle"', status.stdout)
                self.assertIn("Model process is alive", status.stdout)
                self.assertIn("Status retrieved", status.stdout)

                environment = os.environ.copy()
                environment["PYTHON_BIN"] = sys.executable
                concurrent = subprocess.Popen(
                    [
                        "bash", str(self.service), "test",
                        str(root / "long-fixture.npz"), "--run", str(run),
                    ],
                    cwd=self.repo,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        running = self.run_service(run, "status")
                        if '"state": "running"' in running.stdout:
                            break
                        time.sleep(0.05)
                    else:
                        self.fail("formal test did not enter running state")
                    active_manifest = (run / "input-manifest.json").read_bytes()
                    overlapping = self.run_service(run, "test", str(fixture))
                    self.assertNotEqual(overlapping.returncode, 0)
                    self.assertIn("A formal test is already in progress", overlapping.stdout)
                    self.assertEqual(
                        (run / "input-manifest.json").read_bytes(), active_manifest
                    )
                    concurrent_stdout, concurrent_stderr = concurrent.communicate(
                        timeout=30
                    )
                    self.assertEqual(
                        concurrent.returncode,
                        0,
                        concurrent_stdout + concurrent_stderr,
                    )
                finally:
                    if concurrent.poll() is None:
                        concurrent.terminate()
                        concurrent.wait(timeout=10)

                failed = self.run_service(
                    run, "test", str(root / "invalid-shape.npz")
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assert_stage(
                    failed, "07", "Prepare and Lock Formal Input", True
                )
                self.assert_stage(
                    failed, "08", "Input and Formal Realtime Run", False
                )
                self.assertIn("fixture motion must be [N,151]", failed.stdout)
                failed_status = self.run_service(run, "status")
                self.assertEqual(failed_status.returncode, 0)
                self.assertIn('"state": "failed"', failed_status.stdout)
                self.assertIn('"ok": false', failed_status.stdout)

                formal = self.run_service(
                    run, "test", str(fixture), "--root-scaled", "true", timeout=30
                )
                self.assertEqual(formal.returncode, 0, formal.stdout + formal.stderr)
                self.assert_stage(
                    formal, "07", "Prepare and Lock Formal Input", True
                )
                self.assert_stage(
                    formal, "08", "Input and Formal Realtime Run", True
                )
                self.assertIn("Formal input manifest locked", formal.stdout)
                self.assertIn("Input structure validated", formal.stdout)
                self.assertIn("Input identity and hash recorded", formal.stdout)
                self.assertIn("Formal run request accepted", formal.stdout)
                self.assertIn("Input manifest and run parameters loaded", formal.stdout)
                self.assertIn("All input frames processed", formal.stdout)
                self.assertIn("Formal run completed:", formal.stdout)
                self.assertIn("Run evidence written", formal.stdout)
                self.assertTrue((run / "logs/stage-07.log").is_file())
                self.assertTrue((run / "logs/stage-08.log").is_file())
                self.assertTrue((run / "summary.json").is_file())
                self.assertTrue((run / "stream.ndjson").is_file())
                self.assertIn(
                    (run / "logs/stage-07.log").read_text(encoding="utf-8"),
                    formal.stdout,
                )
                self.assertIn(
                    (run / "logs/stage-08.log").read_text(encoding="utf-8"),
                    formal.stdout,
                )
                manifest = (run / "input-manifest.json").read_bytes()
                manifest_data = json.loads(manifest)
                self.assertEqual(manifest_data["run_id"], run.name)
                self.assertEqual(manifest_data["path"], str(fixture.resolve()))
                self.assertEqual(manifest_data["status"], "locked")
                self.assertTrue(manifest_data["passed"])
                recovered_status = self.run_service(run, "status")
                self.assertEqual(recovered_status.returncode, 0)
                self.assertIn('"ok": true', recovered_status.stdout)
                self.assertIn('"state": "finished"', recovered_status.stdout)

                first_stream = (run / "stream.ndjson").read_bytes()
                first_summary = (run / "summary.json").read_bytes()
                invalid = self.run_service(run, "test", str(root / "missing.npz"))
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn("Input does not exist", invalid.stdout)
                self.assertEqual((run / "input-manifest.json").read_bytes(), manifest)
                self.assertEqual((run / "stream.ndjson").read_bytes(), first_stream)
                self.assertEqual((run / "summary.json").read_bytes(), first_summary)

                (run / "gate-results.json").write_text("stale", encoding="utf-8")
                (run / "report.md").write_text("stale", encoding="utf-8")
                (run / "fixtures").mkdir()
                (run / "fixtures/fixture.npz").write_text("stale", encoding="utf-8")
                (run / "fixtures/recorded_fixture.npz").write_text(
                    "stale", encoding="utf-8"
                )

                repeated = self.run_service(run, "test", timeout=30)
                self.assertEqual(
                    repeated.returncode, 0, repeated.stdout + repeated.stderr
                )
                self.assert_stage(
                    repeated, "07", "Prepare and Lock Formal Input", True
                )
                self.assert_stage(
                    repeated, "08", "Input and Formal Realtime Run", True
                )
                second_manifest = (run / "input-manifest.json").read_bytes()
                self.assertNotEqual(second_manifest, manifest)
                self.assertEqual(
                    json.loads(second_manifest)["path"],
                    str((root / "default-fixture.npz").resolve()),
                )
                self.assertNotEqual(
                    (run / "stream.ndjson").read_bytes(), first_stream
                )
                self.assertFalse((run / "gate-results.json").exists())
                self.assertFalse((run / "report.md").exists())
                self.assertFalse((run / "fixtures/fixture.npz").exists())
                self.assertFalse((run / "fixtures/recorded_fixture.npz").exists())
                self.assertEqual(
                    int((run / "runtime.pid").read_text(encoding="utf-8")),
                    runtime_pid,
                )

                finished = self.run_service(run, "status")
                self.assertEqual(finished.returncode, 0)
                self.assertIn('"state": "finished"', finished.stdout)

                stopped = self.run_service(run, "stop")
                self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                self.assert_stage(stopped, "04", "Model Service · stop", True)
                self.assertIn("Shutdown request sent", stopped.stdout)
                self.assertIn("Runtime shutdown initiated", stopped.stdout)
                self.assertTrue(self.wait_for_exit(run), "runtime did not exit after stop")
                runtime_pid = None

                stopped_status = self.run_service(run, "status")
                self.assertNotEqual(stopped_status.returncode, 0)
                self.assert_stage(
                    stopped_status, "04", "Model Service · status", False
                )
            finally:
                if runtime_pid is not None:
                    try:
                        os.kill(runtime_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
