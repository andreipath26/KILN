"""Tests for the optional page-cache-bypassing traced-I/O probe."""
from pathlib import Path
import tempfile
import unittest

from scripts.capture_direct_gguf_expert_io import capture_odirect


class DirectIoProbeTests(unittest.TestCase):
    def test_aligned_direct_read_replays_the_same_logical_expert_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gguf"
            model.write_bytes(bytes(range(256)) * 128)
            replay = [
                ("blk.0.experts.0.weight", 0, 513, 5000, 1, 0),
                ("blk.0.experts.1.weight", 1, 8193, 700, 1, 0),
            ]
            report = capture_odirect(model, replay, runs=2)

        self.assertIn(report["status"], {"captured", "unavailable"})
        if report["status"] == "captured":
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(report["events_per_run"], 2)
            self.assertEqual(report["logical_expert_bytes_per_run"], 5700)
            self.assertGreaterEqual(report["aligned_bytes_per_run"], 5700)
            self.assertGreater(report["request_latency_us_p50_median"], 0)
            self.assertIn("not device telemetry", report["interpretation"])
        else:
            self.assertTrue(report["reason"])


if __name__ == "__main__":
    unittest.main()
