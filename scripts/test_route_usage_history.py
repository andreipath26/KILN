"""Regression coverage for model-keyed, decayed expert routing telemetry."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.route_usage_history import build_history


class RouteHistoryTests(unittest.TestCase):
    def test_counts_projection_events_and_decays_prior_history(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            trace = root / "trace.json"
            first = root / "first.json"
            second = root / "second.json"
            model_sha = "a" * 64
            trace.write_text(json.dumps([
                {"layer": 0, "expert_id": 2},
                {"layer": 0, "expert_id": 2},
                {"layer": 1, "expert_id": 3}]))
            initial = build_history(trace, first, model_sha)
            assert initial["operator_event_heat"] == {"0:2": 2, "1:3": 1}
            assert not initial["runtime_cache_activated"]
            updated = build_history(trace, second, model_sha, first)
            assert updated["operator_event_heat"] == {"0:2": 3, "1:3": 1}
            self.assertEqual(updated["ranked_experts"][0]["expert_id"], 2)
            with self.assertRaises(FileExistsError):
                build_history(trace, second, model_sha, first)
            with self.assertRaisesRegex(ValueError, "another model"):
                build_history(trace, root / "bad.json", "b" * 64, first)

    def test_rejects_malformed_routes_and_never_writes_partial_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trace, output = root / "trace.json", root / "out.json"
            for invalid in ([{"layer": True, "expert_id": 2}],
                            [{"layer": -1, "expert_id": 0}],
                            [{"layer": 0, "expert_id": 65536}],
                            [{"layer": 0, "expert_id": 2, "extra": 3}],
                            [], {"layer": 1}):
                trace.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    build_history(trace, output, "c" * 64)
                self.assertFalse(output.exists())
            trace.write_text(json.dumps([{"layer": 0, "expert_id": 0}]))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                build_history(trace, output, "bad")
            self.assertFalse(output.exists())

    def test_prior_schema_cannot_import_fake_heat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trace, previous, output = root / "trace.json", root / "old.json", root / "out.json"
            trace.write_text('[{"layer":0,"expert_id":1}]')
            previous.write_text(json.dumps({
                "version": 1, "model_sha256": "a" * 64,
                "operator_event_heat": {"0:1": -100}
            }))
            with self.assertRaisesRegex(ValueError, "entry"):
                build_history(trace, output, "a" * 64, previous)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
