from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_tier0_qualification as qualification


class Tier0QualificationTests(unittest.TestCase):
    def test_runs_all_phases_but_never_self_certifies_deployment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "target/release").mkdir(parents=True)
            (repo / "target/release/kiln").write_text("stub", encoding="utf-8")
            model_dir = root / "models"
            model_dir.mkdir()
            models = {}
            for name in ("dense.gguf", "moe.gguf", "llada.gguf"):
                path = model_dir / name
                path.write_bytes(name.encode())
                models[name] = str(path)
            config = root / "qualification.json"
            config.write_text(json.dumps({
                "phase35": {"model": models["dense.gguf"]},
                "phase5": {"model": models["moe.gguf"]},
                "phase6": {"model": models["llada.gguf"]},
            }), encoding="utf-8")
            calls: list[str] = []

            def fake_run(capture: qualification.Capture, name: str, command: list[str], *, env=None) -> bool:
                calls.append(name)
                if name == "phase6-confidence":
                    Path(command[command.index("--answers-jsonl") + 1]).write_text("{}\n", encoding="utf-8")
                    Path(command[command.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
                if name == "phase6-clad":
                    Path(command[command.index("--output") + 1]).write_text("{}\n", encoding="utf-8")
                capture.records.append({"name": name, "status": "passed"})
                return True

            output = root / "evidence"
            with patch.object(qualification, "ROOT", repo), patch.object(qualification.Capture, "run", fake_run):
                code = qualification.run(config, output)

            report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(report["phase_capture_status"], {
                "phase35": "captured", "phase5": "captured", "phase6": "captured"})
            self.assertFalse(report["deployment_ready"])
            self.assertEqual(report["overall_status"], "inconclusive")
            self.assertIn("phase6-clad", calls)

    def test_interruption_keeps_running_phase_in_incremental_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "target/release").mkdir(parents=True)
            (repo / "target/release/kiln").write_text("stub", encoding="utf-8")
            model_dir = root / "models"
            model_dir.mkdir()
            models = {}
            for name in ("dense.gguf", "moe.gguf", "llada.gguf"):
                path = model_dir / name
                path.write_bytes(name.encode())
                models[name] = str(path)
            config = root / "qualification.json"
            config.write_text(json.dumps({
                "phase35": {"model": models["dense.gguf"]},
                "phase5": {"model": models["moe.gguf"]},
                "phase6": {"model": models["llada.gguf"]},
            }), encoding="utf-8")

            def interrupt_during_phase5(_capture: qualification.Capture, name: str,
                                        _command: list[str], *, env=None) -> bool:
                if name == "phase5":
                    raise KeyboardInterrupt("simulated interruption")
                return True

            output = root / "interrupted-evidence"
            with patch.object(qualification, "ROOT", repo), \
                    patch.object(qualification.Capture, "run", interrupt_during_phase5):
                with self.assertRaises(KeyboardInterrupt):
                    qualification.run(config, output)

            report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["run_state"], "interrupted")
            self.assertEqual(report["build_status"], "passed")
            self.assertEqual(report["phase_capture_status"], {
                "phase35": "captured", "phase5": "running", "phase6": "pending"})
            self.assertEqual(report["model_file_stability"], "pending")
            self.assertEqual(report["overall_status"], "inconclusive")
            self.assertFalse(report["deployment_ready"])

    def test_missing_models_are_checkpointed_as_blocked_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "target/release").mkdir(parents=True)
            (repo / "target/release/kiln").write_text("stub", encoding="utf-8")
            config = root / "qualification.json"
            config.write_text(json.dumps({
                "phase35": {"model": "missing-dense.gguf"},
                "phase5": {"model": "missing-moe.gguf"},
                "phase6": {"model": "missing-llada.gguf"},
            }), encoding="utf-8")

            def build_only(_capture: qualification.Capture, name: str,
                           _command: list[str], *, env=None) -> bool:
                return name == "build-release"

            output = root / "blocked-evidence"
            with patch.object(qualification, "ROOT", repo), \
                    patch.object(qualification.Capture, "run", build_only):
                code = qualification.run(config, output)

            report = json.loads((output / "qualification.json").read_text(encoding="utf-8"))
            commands = json.loads((output / "commands.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["run_state"], "completed")
            self.assertEqual(report["phase_capture_status"], {
                "phase35": "blocked", "phase5": "blocked", "phase6": "blocked"})
            self.assertEqual([entry["status"] for entry in commands], ["blocked"] * 3)
            self.assertFalse(report["deployment_ready"])
            self.assertEqual(report["overall_status"], "inconclusive")

    def test_relative_model_paths_resolve_from_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "model.gguf"
            model.write_bytes(b"fixture")
            self.assertEqual(qualification.resolve_file(base, "model.gguf"), model)
            self.assertIsNone(qualification.resolve_file(base, "missing.gguf"))


if __name__ == "__main__":
    unittest.main()
