import asyncio
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend
from duet_edge_realtime.motion_quality import OnlineMotionQuality
from duet_edge_realtime.progress import TerminalProgress
from duet_edge_realtime.sinks import StaticWebSink


class ServiceFeatureTests(unittest.TestCase):
    def test_captured_progress_uses_tty_and_distinct_bar_styles(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        terminal = FakeTTY()
        captured_stdout = io.StringIO()
        event = {
            "phase": "inference", "window": 2, "windows": 4,
            "step": 5, "steps": 10,
        }
        with patch.dict(os.environ, {"STAGE_CAPTURE_ACTIVE": "1"}), \
                patch.object(os, "isatty", return_value=False), \
                patch("builtins.open", return_value=terminal), \
                redirect_stdout(captured_stdout):
            TerminalProgress(True, width=10).model_update(event, force=True)

        output = terminal.getvalue()
        self.assertEqual(captured_stdout.getvalue(), "")
        self.assertIn("[===.......]", output)
        self.assertIn("[#####-----]", output)

    def test_stage_capture_mirrors_both_streams_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            result = subprocess.run([
                sys.executable,
                str(Path(__file__).parents[1] /
                    "scripts/final_execution/lib/capture_stage.py"),
                "--stage", "05", "--run-root", str(run),
                "--state-file", str(run / "state"), "--",
                sys.executable, "-c",
                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(37)",
            ], text=True, capture_output=True, check=False)
            log = (run / "logs/stage-05.log").read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 37)
            self.assertIn("out", result.stdout)
            self.assertIn("err", result.stdout)
            self.assertIn("out", log)
            self.assertIn("err", log)

    def test_stage_capture_log_failure_does_not_block_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "logs/stage-05.log").mkdir(parents=True)
            marker = run / "stage-ran"
            result = subprocess.run([
                sys.executable,
                str(Path(__file__).parents[1] /
                    "scripts/final_execution/lib/capture_stage.py"),
                "--stage", "05", "--run-root", str(run),
                "--state-file", str(run / "state"), "--",
                sys.executable, "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0)
            self.assertTrue(marker.is_file())
            self.assertIn("continuing without archival", result.stderr)

    def test_stage_one_log_moves_into_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            state = root / "state"
            child = (
                "from pathlib import Path; "
                f"run=Path({str(run)!r}); run.mkdir(); "
                f"Path({str(state)!r}).write_text(str(run)); print('initialized')"
            )
            result = subprocess.run([
                sys.executable,
                str(Path(__file__).parents[1] /
                    "scripts/final_execution/lib/capture_stage.py"),
                "--stage", "01", "--state-file", str(state), "--",
                sys.executable, "-c", child,
            ], text=True, capture_output=True, check=False)
            log = run / "logs/stage-01.log"
            self.assertEqual(result.returncode, 0)
            self.assertTrue(log.is_file())
            self.assertIn("initialized", log.read_text(encoding="utf-8"))

    def test_model_stage_continues_when_runtime_log_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "logs/runtime.log").mkdir(parents=True)
            (run / "config.json").touch()
            (run / "config.sha256").touch()
            environment = os.environ.copy()
            environment.update({
                "STAGE_CAPTURE_ACTIVE": "1",
                "PYTHON_BIN": shutil.which("true") or "/usr/bin/true",
            })
            result = subprocess.run([
                "bash",
                str(Path(__file__).parents[1] / "scripts/final_execution/runtime_service.sh"),
                "model", "start", "--run", str(run),
            ], text=True, capture_output=True, check=False, env=environment)
            self.assertEqual(result.returncode, 0)
            self.assertTrue((run / "runtime.pid").is_file())
            self.assertIn("continuing without archival", result.stderr)

    def test_runtime_client_accepts_null_session_progress(self):
        client = runpy.run_path(str(
            Path(__file__).parents[1] / "scripts/final_execution/lib/runtime_client.py"
        ))
        self.assertIsNone(client["session_ratio"]({
            "session": {"progress": None},
        }))

    def test_runtime_client_rejects_a_different_run_before_mutation(self):
        client = runpy.run_path(str(
            Path(__file__).parents[1] / "scripts/final_execution/lib/runtime_client.py"
        ))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run-current"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({
                "server": {
                    "bind_host": "127.0.0.1",
                    "control_port": 8766,
                },
            }), encoding="utf-8")
            calls = []

            def raw_request(_run, method, path):
                calls.append((method, path))
                return {"run_id": "run-old"}

            client["request"].__globals__["raw_request"] = raw_request
            with self.assertRaisesRegex(
                RuntimeError,
                r"stop --run .*/run-old",
            ):
                client["request"](run, "POST", "/viewer/start")
            self.assertEqual(calls, [("GET", "/status")])

            calls.clear()

            def matching_request(_run, method, path):
                calls.append((method, path))
                return {"run_id": run.name}

            client["request"].__globals__["raw_request"] = matching_request
            client["request"](run, "POST", "/shutdown")
            self.assertEqual(
                calls,
                [("GET", "/status"), ("POST", "/shutdown")],
            )

    def test_runtime_client_flushes_waits_and_falls_back_to_model_progress(self):
        client = runpy.run_path(str(
            Path(__file__).parents[1] / "scripts/final_execution/lib/runtime_client.py"
        ))
        event = {
            "phase": "inference", "window": 2, "windows": 8,
            "step": 17, "steps": 50,
        }
        status = {
            "model": {"progress": event},
            "session": {"progress": None},
        }
        self.assertEqual(
            client["model_progress_event"](status, "session.state"), event
        )
        with patch.object(client["os"], "isatty", return_value=False), \
                patch("builtins.print") as output:
            client["draw_wait_progress"](
                5.0, "Realtime inference and playout", emit_non_tty=True
            )
        self.assertTrue(output.call_args.kwargs["flush"])

    def test_viewer_delay_uses_single_monotonic_clock(self):
        viewer = (
            Path(__file__).parents[1] / "web/viewer.js"
        ).read_text(encoding="utf-8")
        self.assertIn("recordFrameArrival", viewer)
        self.assertIn("performance.now()", viewer)
        self.assertIn("!document.hidden", viewer)
        self.assertNotIn("Date.now() - state.frame.emitted_wall_time_s", viewer)

    def test_runtime_wait_can_restore_final_status_json_without_extra_request(self):
        client = runpy.run_path(str(
            Path(__file__).parents[1] / "scripts/final_execution/lib/runtime_client.py"
        ))
        status = {
            "model": {"state": "ready", "progress": None},
            "session": {"state": "idle", "progress": None},
        }
        requests = []

        def request(*args):
            requests.append(args)
            return status

        client["main"].__globals__["request"] = request
        output = io.StringIO()
        argv = [
            "runtime_client.py", "--run", "/tmp/run", "wait",
            "--field", "model.state", "--value", "ready",
            "--timeout", "1", "--show-final-status",
        ]
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            client["main"]()
        self.assertEqual(len(requests), 1)
        self.assertIn('"state": "ready"', output.getvalue())

    def test_cuda_backend_reports_real_window_and_sampling_progress(self):
        events = []
        backend = CudaDuetEdgeBackend(
            "checkpoint.pt", ".", sampling_steps=50,
            progress_callback=events.append,
        )
        backend.set_inference_total_windows(10)
        backend.start_session("formal")
        backend._progress_context["window"] = 4
        backend._report_progress(28)
        self.assertEqual(events[-1], {
            "phase": "inference",
            "window": 4,
            "windows": 10,
            "step": 28,
            "steps": 50,
        })
        self.assertEqual(backend.progress_snapshot(), events[-1])

    def test_handoff_metadata_validation(self):
        backend = CudaDuetEdgeBackend("checkpoint.pt", ".", sampling_steps=50)
        schedule = ((999, 978), (978, 958))
        tensor = SimpleNamespace(shape=(1, 75, 151), device="cuda:0")
        backend._handoff = {978: tensor}
        backend._handoff_meta = {
            "next_window_id": 1,
            "shape": (1, 150, 151),
            "sampling_steps": 50,
            "schedule": schedule,
            "dtype": "torch.float32",
            "device": "cuda:0",
        }
        window = SimpleNamespace(window_id=1)
        self.assertIs(backend._validated_handoff(
            window, (1, 150, 151), schedule, "cuda:0"
        ), backend._handoff)
        window.window_id = 2
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
            backend._validated_handoff(window, (1, 150, 151), schedule, "cuda:0")

    def test_motion_quality_detects_identical_pose(self):
        quality = OnlineMotionQuality()
        self.assertGreaterEqual(quality.sample_limit, 10 * 60 * quality.fps)
        joints = [[[float(index), 0.0, 1.0] for index in range(24)]] * 3
        for frame_id, pose in enumerate(joints):
            quality.record_frame(frame_id, pose, pose)
        summary = quality.summary()
        self.assertEqual(summary["distinctness_body_centered"]["max"], 0.0)
        self.assertEqual(summary["root_position_step"]["max"], 0.0)

    def test_motion_quality_uses_authoritative_lead_ground(self):
        quality = OnlineMotionQuality()
        lead = np.zeros((24, 3), dtype=np.float64)
        companion = lead.copy()
        companion[:, 2] = 0.2
        quality.record_frame(0, lead, companion)
        companion[:, 2] = -0.05
        quality.record_frame(1, lead, companion)
        summary = quality.summary()
        self.assertAlmostEqual(summary["ground_penetration"]["max"], 0.05)

    def test_integrated_web_sink_serves_health_and_assets(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "index.html").write_text("viewer")
                sink = StaticWebSink("127.0.0.1", 0, root)
                await sink.start({"protocol": "duet-edge-stream/v3", "run_id": "test"})
                port = sink.server.sockets[0].getsockname()[1]
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                await writer.drain()
                response = await reader.read()
                writer.close()
                await writer.wait_closed()
                await sink.close()
                self.assertIn(b"200 OK", response)
                self.assertIn(b"duet-edge-stream/v3", response)
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
