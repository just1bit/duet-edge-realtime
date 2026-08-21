import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import identity_motion


class CliTests(unittest.TestCase):
    def test_environment_paths_override_json_and_run_dir_is_immutable(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture.npz"
            np.savez(fixture, motion_151=identity_motion(150))
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend":"fake",
                "paths":{
                    "input_motion":"/json/does/not/exist.npz",
                    "output_dir":"/json/does/not/exist",
                },
            }))
            env = os.environ.copy()
            env.update({
                "PYTHONPATH":str(repo / "src"),
                "EDGE_INPUT_MOTION":str(fixture),
                "EDGE_OUTPUT_DIR":str(root / "runs"),
            })
            command = [
                sys.executable, "-m", "duet_edge_realtime.service",
                "--config", str(config), "--run-id", "cli-test",
                "--clock", "virtual", "--sink", "ndjson",
            ]
            first = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            run_dir = root / "runs" / "cli-test"
            effective = json.loads((run_dir / "effective_config.json").read_text())
            self.assertEqual(effective["paths"]["input_motion"], str(fixture.resolve()))
            self.assertEqual(effective["paths"]["output_dir"], str(run_dir.resolve()))
            second = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)

    def test_warmup_failure_preserves_run_id_and_failed_lifecycle(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture.npz"
            np.savez(fixture, motion_151=identity_motion(150))
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend": "cuda",
                "paths": {
                    "duet_edge_root": str(repo),
                    "checkpoint": str(root / "missing.pt"),
                    "input_motion": str(fixture),
                    "output_dir": str(root / "runs"),
                },
            }))
            env = {**os.environ, "PYTHONPATH": str(repo / "src")}
            # Acceptance exports the real CUDA paths.  This test must exercise
            # the deliberately missing paths in its JSON config instead of
            # inheriting those higher-precedence overrides.
            env.pop("DUET_EDGE_ROOT", None)
            env.pop("EDGE_CHECKPOINT", None)
            command = [
                sys.executable, "-m", "duet_edge_realtime.service",
                "--config", str(config), "--input-format", "fixture",
                "--run-id", "warmup-failure", "--sink", "ndjson",
            ]
            result = subprocess.run(
                command, cwd=repo, env=env, text=True, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            run_dir = root / "runs" / "warmup-failure"
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["run_id"], "warmup-failure")
            self.assertEqual(summary["exit_reason"], "model_load_or_warmup_error")
            self.assertEqual(summary["lifecycle"]["final_state"], "failed")
            messages = [
                json.loads(line)
                for line in (run_dir / "stream.ndjson").read_text().splitlines()
            ]
            self.assertEqual(messages[0]["type"], "hello")
            self.assertEqual(
                [message["state"] for message in messages if message["type"] == "state"],
                ["starting", "failed"],
            )
            self.assertEqual(messages[-1]["type"], "error")

    def test_input_setup_failure_is_structured_and_closes_cleanly(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "bad-fixture.npz"
            np.savez(fixture, motion_151=np.zeros((10, 151), dtype=np.float32))
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend": "fake",
                "paths": {
                    "input_motion": str(fixture),
                    "output_dir": str(root / "runs"),
                },
            }))
            result = subprocess.run(
                [
                    sys.executable, "-m", "duet_edge_realtime.service",
                    "--config", str(config), "--run-id", "bad-input",
                    "--sink", "ndjson",
                ],
                cwd=repo,
                env={**os.environ, "PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            run_dir = root / "runs" / "bad-input"
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["run_id"], "bad-input")
            self.assertEqual(summary["exit_reason"], "input_setup_error")
            self.assertEqual(summary["lifecycle"]["final_state"], "failed")
            messages = [
                json.loads(line)
                for line in (run_dir / "stream.ndjson").read_text().splitlines()
            ]
            self.assertEqual(messages[-1]["type"], "error")


if __name__ == "__main__":
    unittest.main()
