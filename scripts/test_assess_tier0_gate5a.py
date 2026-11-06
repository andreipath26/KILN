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
                "block_bytes": 4096,
                "patterns": [{
                    "access_pattern": "random",
                    "total_bytes_read": 8192,
                    "throughput_mib_s": 2000.0,
                    "request_latency_p50_us": 10.0,
                    "request_latency_p95_us": 15.0,
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


if __name__ == "__main__":
    unittest.main()
