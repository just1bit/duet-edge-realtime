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


if __name__ == "__main__":
    unittest.main()
