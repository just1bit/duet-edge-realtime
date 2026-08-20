import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from duet_edge_realtime.backends.recorded import GOLDEN_SCHEMA, RecordedInferenceBackend
from duet_edge_realtime.continuity import IdentityNormalizer, OnlineContinuityProcessor
from duet_edge_realtime.schemas import MotionWindow

from helpers import identity_motion


def process_windows(chunks: np.ndarray) -> np.ndarray:
    processor = OnlineContinuityProcessor(IdentityNormalizer())
    values = [processor.process(chunk) for chunk in chunks]
    values.append(processor.flush())
    return np.concatenate(values)


def make_golden_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion = identity_motion(225, root_step=0.001)
    lead_windows = np.stack((motion[:150], motion[75:225]))
    generated = lead_windows.copy()
    generated[0, :, 5] += 0.10
    generated[1, :, 5] += 0.20
    lead_joints = process_windows(lead_windows)
    companion_joints = process_windows(generated)
    np.savez_compressed(
        path,
        golden_schema=GOLDEN_SCHEMA,
        motion_151=motion,
        lead_windows=lead_windows,
        generated_normalized=generated,
        generated_unnormalized=generated,
        expected_lead_joints=lead_joints,
        expected_companion_joints=companion_joints,
        window_ids=np.asarray([0, 1], dtype=np.int64),
        window_start_seq=np.asarray([0, 75], dtype=np.int64),
        window_end_seq=np.asarray([150, 225], dtype=np.int64),
        window_valid_frames=np.asarray([150, 150], dtype=np.int64),
        window_seeds=np.asarray([1234, 1235], dtype=np.int64),
        inference_wall_ms=np.asarray([100.0, 110.0]),
        inference_cuda_ms=np.asarray([90.0, 95.0]),
        normalizer_scale=np.ones(151, dtype=np.float32),
        normalizer_min=np.zeros(151, dtype=np.float32),
        metadata_json=json.dumps({
            "backend": "cuda",
            "checkpoint_sha256": "fixture-checkpoint",
            "window_count": 2,
        }),
    )
    return lead_windows, generated, companion_joints


class RecordedBackendTests(unittest.TestCase):
    def test_replays_and_strictly_validates_recorded_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp) / "golden.npz"
            lead, generated, _ = make_golden_fixture(fixture)
            backend = RecordedInferenceBackend(fixture)
            backend.warmup()
            try:
                for index, start in enumerate((0, 75)):
                    chunk = backend.infer(MotionWindow(
                        index, start, start + 150, 0.0, 1234 + index, lead[index]
                    ))
                    np.testing.assert_array_equal(chunk.motion, generated[index])
                values = np.zeros((1, 1, 151), dtype=np.float32)
                values[..., 0] = 2.0
                self.assertEqual(float(backend.unnormalize(values)[0, 0, 0]), 1.0)
                self.assertEqual(backend.version_info()["source_backend"], "cuda")
            finally:
                backend.close()

            mismatch = RecordedInferenceBackend(fixture)
            mismatch.warmup()
            try:
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    mismatch.infer(MotionWindow(0, 0, 150, 0.0, 999, lead[0]))
            finally:
                mismatch.close()

    def test_recorded_cli_reproduces_complete_companion_timeline(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "golden.npz"
            _, _, expected_joints = make_golden_fixture(fixture)
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend": "recorded",
                "paths": {
                    "input_motion": str(fixture),
                    "output_dir": str(root / "runs"),
                },
                "model": {"seed": 1234},
            }))
            result = subprocess.run(
                [
                    sys.executable, "-m", "duet_edge_realtime.service",
                    "--config", str(config), "--run-id", "recorded-test",
                    "--clock", "virtual", "--sink", "ndjson",
                ],
                cwd=repo,
                env={**os.environ, "PYTHONPATH": str(repo / "src")},
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = root / "runs" / "recorded-test"
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["backend"]["backend"], "recorded")
            self.assertEqual(summary["inference"]["sample_count"], 2)
            self.assertEqual(summary["output"]["frames"], 225)
            frames = [
                json.loads(line)["companion_joints"]
                for line in (run / "stream.ndjson").read_text().splitlines()
                if json.loads(line).get("type") == "frame"
            ]
            np.testing.assert_allclose(
                np.asarray(frames), expected_joints, atol=1e-6, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
