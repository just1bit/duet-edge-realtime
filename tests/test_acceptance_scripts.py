import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AcceptanceScriptTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]

    def test_benchmark_summary_requires_cuda_sample_count_and_matching_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "s2-benchmark-50"
            run.mkdir()
            summary = {
                "exit_reason": "input_complete",
                "backend": {
                    "backend": "cuda",
                    "sampling_steps": 50,
                    "engine_commit": "abc123",
                    "checkpoint_sha256": "deadbeef",
                    "peak_gpu_memory_bytes": 123,
                },
                "config": {
                    "model": {"sampling_steps": 50},
                    "stream": {"hop_frames": 75, "fps": 30},
                },
                "inference": {
                    "sample_count": 100,
                    "p50_ms": 1000.0,
                    "p95_ms": 1100.0,
                    "p99_ms": 1200.0,
                    "cuda_p50_ms": 900.0,
                    "cuda_p95_ms": 1000.0,
                    "cuda_p99_ms": 1100.0,
                },
            }
            path = run / "summary.json"
            path.write_text(json.dumps(summary))
            command = [
                sys.executable, "scripts/summarize_benchmark.py", str(root),
                "--pattern", "s2-benchmark-*/summary.json", "--output", "result.json",
            ]
            passed = subprocess.run(command, cwd=self.repo, text=True, capture_output=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["recommended_baseline"]["steps"], 50)
            self.assertEqual(result["recommended_baseline"]["sample_count"], 100)

            summary["inference"]["sample_count"] = 99
            path.write_text(json.dumps(summary))
            rejected = subprocess.run(command, cwd=self.repo, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("fewer than 100", rejected.stderr + rejected.stdout)

    def test_check_run_enforces_backend_and_cuda_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "exit_reason": "input_complete",
                "backend": {"backend": "fake"},
                "config": {
                    "model": {"sampling_steps": 50},
                    "stream": {
                        "fps": 30, "hop_frames": 75, "playout_delay_s": 2,
                        "inference_slo_ms": 1900, "jitter_slo_ms": 20,
                    },
                },
                "input": {"sequence_errors": 0},
                "inference": {"sample_count": 1, "p99_ms": 1},
                "queues": {"overloads": 0},
                "output": {
                    "frames": 1, "committed_frames": 1, "underflows": 0,
                    "observed_fps": 30, "jitter_p95_ms": 0,
                },
                "lifecycle": {"final_state": "finished"},
                "slo": {
                    "inference_p99_met": True, "jitter_p95_met": True,
                    "continuous_playout_met": True,
                },
            }
            messages = [
                {"type": "hello", "protocol": "duet-edge-stream/v2"},
                *({"type": "state", "state": state} for state in (
                    "starting", "buffering", "playing", "draining", "finished"
                )),
                {
                    "type": "frame", "schema_version": "2.0.0", "frame_id": 0,
                    "seq": 0, "motion_time_s": 0, "joints": [[0, 0, 0]] * 24,
                },
                {"type": "eos", "reason": "input_complete"},
            ]
            summary_path = root / "summary.json"
            stream_path = root / "stream.ndjson"
            summary_path.write_text(json.dumps(summary))
            stream_path.write_text("\n".join(json.dumps(message) for message in messages))
            base = [
                sys.executable, "scripts/check_run.py", "--summary", str(summary_path),
                "--ndjson", str(stream_path),
            ]
            passed = subprocess.run(
                [*base, "--require-backend", "fake"],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            rejected = subprocess.run(
                [*base, "--require-backend", "cuda"],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("backend is not required cuda", rejected.stdout)


if __name__ == "__main__":
    unittest.main()
