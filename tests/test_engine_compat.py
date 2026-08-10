import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from duet_edge_realtime.backends.duet_edge import CudaDuetEdgeBackend


class EngineCompatibilityTests(unittest.TestCase):
    def make_engine(self, root: Path) -> str:
        (root / "model").mkdir(parents=True)
        for relative in ("EDGE.py", "model/diffusion.py", "vis.py"):
            (root / relative).write_text("# fixture\n")
        subprocess.run(["git","init","-q",str(root)], check=True)
        subprocess.run(["git","-C",str(root),"add","."], check=True)
        subprocess.run([
            "git","-C",str(root),"-c","user.name=Test","-c","user.email=test@example.invalid",
            "commit","-qm","fixture",
        ], check=True)
        return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"], text=True).strip()

    def backend(self, root: Path, commit: str, allow=False):
        lock = root.parent / "lock.json"
        lock.write_text(json.dumps({"repository":"https://example.invalid/engine.git","commit":commit}))
        checkpoint = root.parent / "model.pt"
        checkpoint.write_bytes(b"fixture")
        return CudaDuetEdgeBackend(
            checkpoint, root, lock_path=lock, allow_engine_mismatch=allow
        )

    def test_matching_clean_external_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "duet-edge"
            commit = self.make_engine(root)
            backend = self.backend(root, commit)
            backend._validate_compatibility()
            self.assertFalse(backend.version_info()["non_reproducible"])

    def test_python_dirty_rejected_but_non_python_dirty_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "duet-edge"
            commit = self.make_engine(root)
            (root / "notes.txt").write_text("untracked notes")
            self.backend(root, commit)._validate_compatibility()
            (root / "EDGE.py").write_text("# changed\n")
            with self.assertRaisesRegex(RuntimeError, "Python source worktree is dirty"):
                self.backend(root, commit)._validate_compatibility()
            allowed = self.backend(root, commit, allow=True)
            allowed._validate_compatibility()
            self.assertTrue(allowed.version_info()["non_reproducible"])

    def test_commit_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "duet-edge"
            commit = self.make_engine(root)
            with self.assertRaisesRegex(RuntimeError, "commit expected"):
                self.backend(root, "0" * 40)._validate_compatibility()


if __name__ == "__main__":
    unittest.main()
