import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class AcceptanceScriptTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]

    def test_prepare_aist_motion_rejects_short_input_and_reports_30fps_length(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw.pkl"
            output = root / "prepared.pkl"
            payload = {
                "smpl_trans": np.zeros((300, 3), dtype=np.float32),
                "smpl_poses": np.zeros((300, 72), dtype=np.float32),
                "smpl_scaling": np.asarray([1.0], dtype=np.float32),
            }
            with source.open("wb") as handle:
                pickle.dump(payload, handle)
            command = [
                sys.executable, "scripts/prepare_aist_motion.py",
                "--input", str(source), "--output", str(output),
            ]
            passed = subprocess.run(
                command, cwd=self.repo, text=True, capture_output=True
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("estimated 150 frames at 30 FPS", passed.stdout)

            payload["smpl_trans"] = payload["smpl_trans"][:299]
            payload["smpl_poses"] = payload["smpl_poses"][:299]
            with source.open("wb") as handle:
                pickle.dump(payload, handle)
            rejected = subprocess.run(
                command, cwd=self.repo, text=True, capture_output=True
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("at least 300", rejected.stderr)

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
                    "eta": 1.0,
                    "checkpoint_sha256": "deadbeef",
                    "peak_gpu_memory_bytes": 123,
                },
                "config": {
                    "model": {"sampling_steps": 50},
                    "stream": {
                        "hop_frames": 75, "fps": 30,
                        "safety_margin_ms": 100,
                    },
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
            self.assertEqual(result["recommended_baseline"]["safety_margin_ms"], 100)
            self.assertEqual(result["recommended_baseline"]["recommended_playout_delay_s"], 1.3)

            summary["inference"]["sample_count"] = 99
            path.write_text(json.dumps(summary))
            rejected = subprocess.run(command, cwd=self.repo, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("fewer than 100", rejected.stderr + rejected.stdout)

    def test_runtime_config_uses_the_configured_safety_margin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.json"
            benchmark = root / "benchmark.json"
            quality = root / "quality-25.json"
            config.write_text(json.dumps({
                "model": {"sampling_steps": 50},
                "stream": {
                    "fps": 30, "hop_frames": 75,
                    "safety_margin_ms": 250,
                },
            }))
            benchmark.write_text(json.dumps({
                "candidates": [{
                    "steps": 25,
                    "deadline_candidate": True,
                    "safety_margin_ms": 250,
                    "recommended_playout_delay_s": 1.5,
                }],
            }))
            quality.write_text(json.dumps({"passed": True}))
            result = subprocess.run(
                [
                    sys.executable, "scripts/update_runtime_config.py",
                    "--benchmark", str(benchmark),
                    "--config", str(config), "--sampling-steps", "25",
                    "--quality", str(quality),
                ],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            updated = json.loads(config.read_text())
            self.assertEqual(updated["model"]["sampling_steps"], 25)
            self.assertEqual(updated["stream"]["safety_margin_ms"], 250)
            self.assertEqual(updated["stream"]["playout_delay_s"], 1.5)
            self.assertEqual(updated["stream"]["inference_slo_ms"], 1250)

    def test_runtime_config_requires_passing_benchmark_and_low_step_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.json"
            benchmark = root / "benchmark.json"
            quality = root / "quality-25.json"
            config.write_text(json.dumps({
                "model": {"sampling_steps": 50},
                "stream": {
                    "fps": 30, "hop_frames": 75,
                    "safety_margin_ms": 100,
                },
            }))
            benchmark.write_text(json.dumps({
                "candidates": [{
                    "steps": 25,
                    "deadline_candidate": True,
                    "safety_margin_ms": 100,
                    "recommended_playout_delay_s": 1.3,
                }],
            }))
            base = [
                sys.executable, "scripts/update_runtime_config.py",
                "--config", str(config), "--benchmark", str(benchmark),
                "--sampling-steps", "25",
            ]
            missing_quality = subprocess.run(
                base, cwd=self.repo, text=True, capture_output=True,
            )
            self.assertNotEqual(missing_quality.returncode, 0)
            self.assertIn("require --quality", missing_quality.stderr)

            quality.write_text(json.dumps({"passed": False}))
            failed_quality = subprocess.run(
                [*base, "--quality", str(quality)],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertNotEqual(failed_quality.returncode, 0)
            self.assertIn("did not pass", failed_quality.stderr)

            quality.write_text(json.dumps({"passed": True}))
            payload = json.loads(benchmark.read_text())
            payload["candidates"][0]["deadline_candidate"] = False
            benchmark.write_text(json.dumps(payload))
            failed_deadline = subprocess.run(
                [*base, "--quality", str(quality)],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertNotEqual(failed_deadline.returncode, 0)
            self.assertIn("deadline budget", failed_deadline.stderr)

            payload["candidates"] = [{
                "steps": 50,
                "deadline_candidate": True,
                "safety_margin_ms": 100,
                "recommended_playout_delay_s": 1.4,
            }]
            benchmark.write_text(json.dumps(payload))
            baseline = subprocess.run(
                [
                    sys.executable, "scripts/update_runtime_config.py",
                    "--config", str(config), "--benchmark", str(benchmark),
                    "--sampling-steps", "50",
                ],
                cwd=self.repo, text=True, capture_output=True,
            )
            self.assertEqual(baseline.returncode, 0, baseline.stderr)

    def test_check_run_enforces_backend_and_cuda_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "exit_reason": "input_complete",
                "backend": {"backend": "fake"},
                "config": {
                    "model": {"sampling_steps": 50},
                    "stream": {
                        "fps": 30, "window_frames": 150, "hop_frames": 75,
                        "playout_delay_s": 2, "inference_queue_size": 1,
                        "output_queue_size": 2,
                        "inference_slo_ms": 1900, "safety_margin_ms": 100,
                        "jitter_slo_ms": 20,
                    },
                },
                "input": {"frames": 1, "sequence_errors": 0},
                "inference": {
                    "sample_count": 1, "p99_ms": 1, "deadline_misses": 0,
                },
                "queues": {
                    "overloads": 0, "inference_high_water": 1,
                    "output_high_water": 1,
                },
                "output": {
                    "frames": 1, "committed_frames": 1, "underflows": 0,
                    "observed_fps": 30, "jitter_p95_ms": 0,
                    "first_frame_latency_s": 149 / 30 + 2,
                    "end_to_end_latency_p95_ms": (149 / 30 + 2) * 1000,
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
                    "seq": 0, "motion_time_s": 0, "commit_kind": "tail",
                    "commit_start_frame_id": 0, "commit_end_frame_id": 1,
                    "joints": [[0, 0, 0]] * 24,
                },
                {"type": "eos", "reason": "input_complete"},
            ]
            summary_path = root / "summary.json"
            stream_path = root / "stream.ndjson"
            summary_path.write_text(json.dumps(summary))
            stream_path.write_text("\n".join(json.dumps(message) for message in messages))
            base = [
                sys.executable, "scripts/check_run.py", "--summary", str(summary_path),
                "--ndjson", str(stream_path), "--require-performance",
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
