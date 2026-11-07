"""Regression tests for the Tier 0 artifact assessor (no hardware required)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("assess_tier0_gate5a.py")
spec = importlib.util.spec_from_file_location("assess_tier0_gate5a", MODULE)
assessor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assessor)


class Gate5aTests(unittest.TestCase):
    def test_budget_is_necessary_not_sufficient(self):
        over = assessor.traffic_budget(800, 300, 4)
        self.assertEqual(over["traffic_budget_condition"], "exceeds_budget")
        self.assertEqual(over["maximum_average_ssd_mib_per_token_at_target"], 200)
        within = assessor.traffic_budget(800, 100, 4)
        self.assertEqual(within["traffic_budget_condition"], "within_budget")
        self.assertIn("not a sufficient", within["interpretation"])

    def test_probe_does_not_fabricate_disk_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "probe.json"
            artifact.write_text(json.dumps({
                "phase_gate": "5A-expert-block-io-feasibility",
                "file": "/tmp/expert-blob.bin",
                "file_size_bytes": 65536,
                "index_file": "/tmp/experts.index.json",
                "indexed_expert_ranges": 4,
                "requests_per_run": 1,
                "repetitions": 1,
                "block_bytes": 4096,
                "patterns": [{
                    "access_pattern": "random",
                    "requests": 1,
                    "total_bytes_read": 8192,
                    "throughput_mib_s": 2000.0,
                    "expert_keys": [{"layer": 0, "expert_id": 1}],
                    "min_request_bytes": 4096,
                    "max_request_bytes": 4096,
                    "request_latency_p50_us": 10.0,
                    "request_latency_p95_us": 15.0,
                }, {
                    "access_pattern": "sequential",
                    "requests": 1,
                    "total_bytes_read": 8192,
                    "throughput_mib_s": 1800.0,
                    "expert_keys": [{"layer": 0, "expert_id": 0}],
                    "min_request_bytes": 4096,
                    "max_request_bytes": 4096,
                    "request_latency_p50_us": 11.0,
                    "request_latency_p95_us": 16.0,
                }],
            }))
            result = assessor.summarize([artifact])[0]
            self.assertIsNone(result["process_attributed_read_bytes"])
            self.assertEqual(result["rate_mib_s_page_cache_uncontrolled"], 2000.0)

    def test_rejects_non_probe_and_bad_counters(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "invalid.json"
            artifact.write_text(json.dumps({"phase_gate": "wrong", "patterns": []}))
            with self.assertRaises(ValueError):
                assessor.summarize([artifact])

    def test_rejects_unindexed_legacy_read_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "legacy.json"
            artifact.write_text(json.dumps({
                "phase_gate": "5A-expert-block-io-feasibility",
                "patterns": [],
            }))
            with self.assertRaisesRegex(ValueError, "requires an indexed expert-range"):
                assessor.summarize([artifact])


if __name__ == "__main__":
    unittest.main()
