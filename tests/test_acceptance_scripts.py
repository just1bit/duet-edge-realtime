import json
import hashlib
import pickle
import subprocess
import sys
import tempfile
import unittest
import re
from pathlib import Path

import numpy as np


class AcceptanceScriptTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.acceptance = self.repo / "scripts" / "v1_execution"
        self.lib = self.acceptance / "lib"

    def run_python(self, script: str, *args: str, cwd: Path | None = None):
        return subprocess.run(
            [sys.executable, str(self.lib / script), *map(str, args)],
            cwd=cwd or self.repo,
            text=True,
            capture_output=True,
        )

    def test_numbered_layout_and_shell_syntax(self):
        expected = {
            "01_initialize.sh", "01_select_run.sh", "02_verify_runtime.sh",
            "02_install_runtime.sh", "02_cuda_smoke.sh", "03_preflight.sh",
            "04_prepare_input.sh", "05_unit_tests.sh", "05_network_tests.sh",
            "06_run_fake.sh", "06_check_fake.sh", "07_run_real.sh",
            "07_check_real.sh", "08_export_fixture.sh", "09_viewer_stream.sh",
            "09_viewer_web.sh", "10_run_baseline.sh", "10_summarize_baseline.sh",
            "11_prepare_candidate.sh", "11_run_candidate.sh",
            "11_compare_quality.sh", "11_summarize_candidates.sh",
            "12_show_recommendation.sh", "12_validate_config.sh",
            "13_monitor_gpu.sh", "13_run_final.sh", "14_check_final.sh",
            "14_check_evidence.sh", "14_build_report.sh",
        }
        actual = {path.name for path in self.acceptance.glob("[0-9][0-9]_*.sh")}
        self.assertEqual(actual, expected)
        for path in sorted(self.acceptance.glob("*.sh")):
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{path}: {result.stderr}")

    def test_prepare_aist_motion_reports_30fps_length(self):
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
            passed = self.run_python(
                "prepare_aist_motion.py", "--input", source, "--output", output
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("estimated 150 frames at 30 FPS", passed.stdout)
            payload["smpl_trans"] = payload["smpl_trans"][:299]
            payload["smpl_poses"] = payload["smpl_poses"][:299]
            with source.open("wb") as handle:
                pickle.dump(payload, handle)
            guided = self.run_python(
                "prepare_aist_motion.py", "--input", source, "--output", output
            )
            self.assertNotEqual(guided.returncode, 0)
            self.assertIn("at least 300", guided.stderr)

    def test_run_initialization_creates_notes_and_repository_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            realtime = project / "duet-edge-realtime"
            realtime.mkdir()
            model = project / "duet-edge"
            model.mkdir()
            for path, branch, remote in (
                (realtime, "main", "https://github.com/just1bit/duet-edge-realtime.git"),
                (model, "realtime-v1", "git@github.com:just1bit/duet-edge.git"),
            ):
                subprocess.run(["git", "-C", str(path), "init", "-b", branch], check=True, capture_output=True)
                subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote], check=True)
            state = realtime / "outputs" / ".acceptance-current"
            result = self.run_python(
                "init_run.py",
                "--realtime-root", realtime,
                "--project-root", project,
                "--state-file", state,
                "--profile", "local",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_root = Path(state.read_text().strip())
            self.assertTrue((run_root / "run-metadata.json").is_file())
            metadata = json.loads((run_root / "run-metadata.json").read_text())
            self.assertEqual(metadata["acceptance_profile"], "local")
            self.assertEqual(metadata["repositories"]["duet_edge_realtime"], {
                "name": "just1bit/duet-edge-realtime", "branch": "main",
            })
            self.assertEqual(metadata["repositories"]["duet_edge"], {
                "name": "just1bit/duet-edge", "branch": "realtime-v1",
            })
            notes = (run_root / "acceptance-notes.md").read_text()
            self.assertIn("## Viewer Review", notes)
            self.assertIn("## Performance", notes)
            self.assertIn("## Manual Changes", notes)
            self.assertIn("## Final Review", notes)
            self.assertNotIn("p99_ms", notes)
            self.assertNotIn("checkpoint_sha256", notes)

    def test_stage_archiver_preserves_repeated_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for _ in range(2):
                result = self.run_python(
                    "archive_stage.py", "--run-root", root, "--stage", "05",
                    "--name", "example", "--next-action", "Repeat the action.",
                    "--", sys.executable, "-c", "print('passed')",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            results = sorted((root / "stage-results").glob("*.json"))
            logs = sorted((root / "logs").glob("*.log"))
            self.assertEqual(len(results), 2)
            self.assertEqual(len(logs), 2)
            self.assertEqual(json.loads(results[1].read_text())["attempt"], 2)

    def test_stage_archiver_records_start_failure_precondition_and_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = self.run_python(
                "archive_stage.py", "--run-root", root, "--stage", "13",
                "--name", "missing-command", "--next-action", "Install it.",
                "--", "command-that-does-not-exist-for-test",
            )
            self.assertNotEqual(missing.returncode, 0)
            missing_data = json.loads(next((root / "stage-results").glob("13-missing-command-*.json")).read_text())
            self.assertEqual(missing_data["failure_kind"], "command_start")
            self.assertEqual(missing_data["exit_status"], 127)

            precondition = self.run_python(
                "archive_stage.py", "--run-root", root, "--stage", "10",
                "--name", "precondition", "--next-action", "Prepare input.",
                "--precondition-error", "fixture is absent",
            )
            self.assertNotEqual(precondition.returncode, 0)
            precondition_data = json.loads(next((root / "stage-results").glob("10-precondition-*.json")).read_text())
            self.assertEqual(precondition_data["failure_kind"], "precondition")

            skipped = self.run_python(
                "archive_stage.py", "--run-root", root, "--stage", "07",
                "--name", "local-skip", "--next-action", "None.",
                "--skip-reason", "GPU only",
            )
            self.assertEqual(skipped.returncode, 0)
            skipped_data = json.loads(next((root / "stage-results").glob("07-local-skip-*.json")).read_text())
            self.assertTrue(skipped_data["skipped"])
            self.assertEqual(skipped_data["automatic_validation"], "not_applicable")

    def test_stage_archiver_accepts_only_declared_signals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            accepted = self.run_python(
                "archive_stage.py", "--run-root", root, "--stage", "09",
                "--name", "accepted-signal", "--next-action", "Repeat.",
                "--accept-signal", "INT", "--", "bash", "-c", "kill -INT $$",
            )
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            data = json.loads(next((root / "stage-results").glob("09-accepted-signal-*.json")).read_text())
            self.assertTrue(data["accepted_termination"])
            rejected = self.run_python(
                "archive_stage.py", "--run-root", root, "--stage", "09",
                "--name", "rejected-signal", "--next-action", "Repeat.",
                "--", "bash", "-c", "kill -INT $$",
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_report_indexes_automatic_results_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "stage-results").mkdir()
            (root / "logs").mkdir()
            (root / "evidence" / "benchmarks").mkdir(parents=True)
            (root / "p1-fake").mkdir()
            (root / "acceptance-notes.md").write_text("# Acceptance Notes\n")
            (root / "run-metadata.json").write_text(json.dumps({
                "created_at": "2026-01-01T00:00:00+00:00",
                "machine": "acceptance-host",
                "acceptance_profile": "local",
            }))
            log = root / "logs" / "06-check-fake-01.log"
            log.write_text("passed\n")
            (root / "stage-results" / "06-check-fake-01.json").write_text(json.dumps({
                "stage": "06", "script": "check-fake", "attempt": 1,
                "passed": True, "log": str(log),
            }))
            (root / "evidence" / "benchmarks" / "benchmark.json").write_text("{}\n")
            (root / "p1-fake" / "summary.json").write_text("{}\n")
            result = self.run_python("build_report.py", root)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (root / "acceptance-report.md").read_text()
            self.assertIn("check-fake", report)
            self.assertIn("stage-results/06-check-fake-01.json", report)
            self.assertIn("acceptance-notes.md", report)
            self.assertIn("evidence/benchmarks/benchmark.json", report)
            self.assertIn("p1-fake/summary.json", report)

    def test_local_runtime_verification(self):
        result = self.run_python(
            "verify_runtime.py", "--duet-edge-root", self.repo.parent / "duet-edge",
            "--checkpoint", self.repo.parent / "data+checkpoint" / "train-1800.pt",
            "--checkpoint-sha256", "unused-for-local", "--profile", "local",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Local runtime compatibility verification passed", result.stdout)

    def test_local_preflight_treats_gpu_and_browser_as_advisory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            realtime = root / "duet-edge-realtime"
            model = root / "duet-edge"
            realtime.mkdir()
            model.mkdir()
            (model / "EDGE.py").write_text("")
            checkpoint = root / "checkpoint.pt"
            motion = root / "motion.pkl"
            checkpoint.write_bytes(b"checkpoint")
            motion.write_bytes(b"motion")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            output = root / "preflight.json"
            result = self.run_python(
                "preflight.py", "--realtime-root", realtime,
                "--duet-edge-root", model, "--checkpoint", checkpoint,
                "--motion", motion, "--checkpoint-sha256", digest(checkpoint),
                "--motion-sha256", digest(motion), "--http-port", "0",
                "--websocket-port", "0", "--output", output,
                "--profile", "local",
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(output.read_text())
            self.assertFalse(payload["gpu"]["applicable"])
            self.assertNotIn("target_gpu", payload["checks"])
            self.assertIn(payload["browser"]["status"], {"available", "unknown"})

    def test_local_evidence_ignores_empty_notes_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in ("stage-results", "logs", "evidence/preflight", "p1-fake", "viewer"):
                (root / relative).mkdir(parents=True, exist_ok=True)
            (root / "run-metadata.json").write_text(json.dumps({"acceptance_profile": "local"}))
            (root / "acceptance-notes.md").write_text("")
            for relative in (
                "evidence/preflight/preflight.json", "evidence/input-motion.json",
                "input_motion.pkl", "p1-fake/summary.json", "p1-fake/stream.ndjson",
                "viewer/summary.json", "viewer/stream.ndjson",
            ):
                (root / relative).write_text("")
            for pattern in (
                "02-verify-runtime", "02-capture-environment", "03-preflight",
                "04-prepare-input", "05-unit-tests", "05-network-tests",
                "06-run-fake", "06-check-fake", "09-viewer-stream", "09-viewer-web",
            ):
                (root / "stage-results" / f"{pattern}-01.json").write_text(json.dumps({
                    "passed": True, "skipped": False,
                }))
            result = self.run_python("check_evidence.py", root, "--profile", "local")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads((root / "evidence/evidence-check.json").read_text())["passed"])

    def test_benchmark_summary_collects_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "benchmark-50"
            run.mkdir()
            summary = {
                "exit_reason": "input_complete",
                "backend": {
                    "backend": "cuda", "sampling_steps": 50,
                    "checkpoint_sha256": "deadbeef", "peak_gpu_memory_bytes": 123,
                },
                "config": {
                    "model": {"sampling_steps": 50},
                    "stream": {"hop_frames": 75, "fps": 30, "safety_margin_ms": 100},
                },
                "inference": {
                    "sample_count": 100, "p50_ms": 1000.0, "p95_ms": 1100.0,
                    "p99_ms": 1200.0, "cuda_p50_ms": 900.0,
                    "cuda_p95_ms": 1000.0, "cuda_p99_ms": 1100.0,
                },
            }
            (run / "summary.json").write_text(json.dumps(summary))
            result = self.run_python(
                "summarize_benchmark.py", root, "--pattern",
                "benchmark-*/summary.json", "--output", "result.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "result.json").read_text())
            candidate = payload["recommended_candidate"]
            self.assertEqual(candidate["steps"], 50)
            self.assertEqual(candidate["sample_count"], 100)
            self.assertEqual(candidate["recommended_playout_delay_s"], 1.3)

    def test_candidate_config_and_manual_config_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.json"
            candidate_config = root / "candidate.json"
            source.write_text(json.dumps({
                "model": {"sampling_steps": 50},
                "stream": {
                    "fps": 30, "hop_frames": 75, "safety_margin_ms": 100,
                    "playout_delay_s": 2.0, "inference_slo_ms": 1900,
                },
            }))
            generated = self.run_python(
                "candidate_config.py", "--source", source, "--steps", "25",
                "--output", candidate_config,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertEqual(json.loads(candidate_config.read_text())["model"]["sampling_steps"], 25)
            config = json.loads(candidate_config.read_text())
            config["stream"]["playout_delay_s"] = 1.3
            config["stream"]["inference_slo_ms"] = 1200
            candidate_config.write_text(json.dumps(config))
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps({"candidates": [{
                "steps": 25, "p99_ms": 1100, "deadline_candidate": True,
                "safety_margin_ms": 100, "recommended_playout_delay_s": 1.3,
                "summary": "summary.json",
            }]}))
            quality = root / "quality.json"
            quality.write_text(json.dumps({"passed": True}))
            validated = self.run_python(
                "config_recommendation.py", "--benchmark", benchmark,
                "--config", candidate_config, "--quality", quality, "--validate",
            )
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            self.assertTrue(json.loads(validated.stdout)["passed"])

    def test_check_run_enforces_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "exit_reason": "input_complete", "backend": {"backend": "fake"},
                "config": {"model": {"sampling_steps": 50}, "stream": {
                    "fps": 30, "window_frames": 150, "hop_frames": 75,
                    "playout_delay_s": 2, "inference_queue_size": 1,
                    "output_queue_size": 2, "inference_slo_ms": 1900,
                    "safety_margin_ms": 100, "jitter_slo_ms": 20,
                }},
                "input": {"frames": 1, "sequence_errors": 0},
                "inference": {"sample_count": 1, "p99_ms": 1, "deadline_misses": 0},
                "queues": {"overloads": 0, "inference_high_water": 1, "output_high_water": 1},
                "output": {
                    "frames": 1, "committed_frames": 1, "underflows": 0,
                    "observed_fps": 30, "jitter_p95_ms": 0,
                    "first_frame_latency_s": 149 / 30 + 2,
                    "end_to_end_latency_p95_ms": (149 / 30 + 2) * 1000,
                },
                "lifecycle": {"final_state": "finished"},
                "slo": {"inference_p99_met": True, "jitter_p95_met": True, "continuous_playout_met": True},
            }
            messages = [
                {"type": "hello", "protocol": "duet-edge-stream/v2"},
                *({"type": "state", "state": state} for state in ("starting", "buffering", "playing", "draining", "finished")),
                {"type": "frame", "schema_version": "2.0.0", "frame_id": 0,
                 "seq": 0, "motion_time_s": 0, "commit_kind": "tail",
                 "commit_start_frame_id": 0, "commit_end_frame_id": 1,
                 "joints": [[0, 0, 0]] * 24},
                {"type": "eos", "reason": "input_complete"},
            ]
            summary_path = root / "summary.json"
            stream_path = root / "stream.ndjson"
            summary_path.write_text(json.dumps(summary))
            stream_path.write_text("\n".join(json.dumps(item) for item in messages))
            passed = self.run_python(
                "check_run.py", "--summary", summary_path, "--ndjson", stream_path,
                "--require-backend", "fake",
            )
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            guided = self.run_python(
                "check_run.py", "--summary", summary_path, "--ndjson", stream_path,
                "--require-backend", "cuda",
            )
            self.assertNotEqual(guided.returncode, 0)
            self.assertIn("Run with the cuda backend", guided.stdout)

    def test_document_renames_and_references(self):
        self.assertTrue((self.repo / "docs" / "V1_EXECUTION_MANUAL.md").is_file())
        self.assertTrue((self.repo / "docs" / "V1_REALTIME_PLAN.md").is_file())
        old_names = (
            "V1_ACCEPTANCE_" + "EXECUTION_CN.md",
            "VERSION1_" + "STREAMING_PLAN.md",
            "VERSION1_" + "REALTIME_PLAN.md",
            "acceptance-" + "observations.md",
        )
        for path in self.repo.rglob("*"):
            if (
                not path.is_file() or ".git" in path.parts
                or "__pycache__" in path.parts or "outputs" in path.parts
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for old_name in old_names:
                self.assertNotIn(old_name, text, f"{old_name} remains in {path}")

    def test_touched_content_is_english_and_manual_is_action_oriented(self):
        paths = [
            self.repo / "docs" / "README.md",
            self.repo / "docs" / "V1_EXECUTION_MANUAL.md",
            self.repo / "docs" / "V1_REALTIME_PLAN.md",
            self.repo / "tests" / "test_acceptance_scripts.py",
            *self.acceptance.rglob("*"),
            *(self.repo / "scripts" / "development").rglob("*"),
        ]
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts:
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(
                    re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text),
                    f"Chinese text remains in {path}",
                )
        manual = (self.repo / "docs" / "V1_EXECUTION_MANUAL.md").read_text().lower()
        for phrase in ("cannot", "must not", "do not", "missing", "failed"):
            self.assertNotIn(phrase, manual)


if __name__ == "__main__":
    unittest.main()
