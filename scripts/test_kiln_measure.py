"""Checks the measurement report helpers and failure log retention."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.kiln_measure import (Run, compare_token_outputs, file_identity, host_info,
                                  mean_metric, model_block_device_delta, peak_rss_gib, read_audit)


class MeasureReportTests(unittest.TestCase):
    def test_host_block_device_counters_are_delta_reported_and_scoped(self):
        before = {"device": "nvme0n1", "model_major_minor": "259:0", "read_ios": 8,
                  "read_sectors": 16, "read_bytes": 8192, "read_time_ms": 3,
                  "scope": "whole block device"}
        after = dict(before, read_ios=10, read_sectors=21, read_bytes=10752, read_time_ms=6)
        delta = model_block_device_delta(before, after)
        self.assertEqual(delta["status"], "captured")
        self.assertEqual(delta["read_bytes_delta"], 2560)
        self.assertEqual(delta["read_time_ms_delta"], 3)
        self.assertIn("whole block device", delta["scope"])
        self.assertEqual(model_block_device_delta(before, None)["status"], "unavailable")
        self.assertEqual(model_block_device_delta(before, dict(after, read_sectors=1))["status"], "counter_reset")

    def test_model_identity_is_stable_and_host_capture_reuses_precomputed_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gguf"
            model.write_bytes(b"model")
            before = file_identity(model)
            with patch("scripts.kiln_measure.sha256", side_effect=AssertionError("rehash")):
                info = host_info(model, "precomputed-digest")
            self.assertEqual(info["model_size_bytes"], 5)
            self.assertEqual(info["model_sha256"], "precomputed-digest")
            self.assertEqual(file_identity(model), before)
            model.write_bytes(b"changed model")
            self.assertNotEqual(file_identity(model), before)

    def test_host_capture_allows_digest_to_be_deferred_until_after_measurement(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gguf"
            model.write_bytes(b"model")
            info = host_info(model, None)
        self.assertIsNone(info["model_sha256"])

    def test_token_parity_passes_for_identical_greedy_runs(self):
        runs = [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef",
                 "token_ids_sha256_le_i32": "a" * 64},
                {"generated_tokens": 11, "token_fingerprint_fnv1a64": "fedcba9876543210",
                 "token_ids_sha256_le_i32": "b" * 64}]
        result = compare_token_outputs({"runs": runs}, {"runs": [dict(row) for row in runs]})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_runs"], 2)
        self.assertEqual(result["total_runs"], 2)

    def test_token_parity_fails_on_changed_fingerprint_or_run_count(self):
        reference = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef",
                               "token_ids_sha256_le_i32": "a" * 64}]}
        changed = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "fedcba9876543210",
                             "token_ids_sha256_le_i32": "a" * 64}]}
        changed_sha = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef",
                                 "token_ids_sha256_le_i32": "b" * 64}]}
        self.assertEqual(compare_token_outputs(reference, changed)["status"], "fail")
        self.assertEqual(compare_token_outputs(reference, changed_sha)["status"], "fail")
        self.assertEqual(compare_token_outputs(reference, {"runs": []})["status"], "fail")

    def test_token_parity_fails_closed_on_missing_report_fields(self):
        reference = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef"}]}
        result = compare_token_outputs(reference, {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef"}]})
        self.assertEqual(result["status"], "fail")
        self.assertIn("missing token parity fields", result["reason"])
        self.assertEqual(compare_token_outputs(None, {"runs": []})["status"], "inconclusive")

    def test_token_parity_rejects_malformed_rows_and_hashes(self):
        good = {"generated_tokens": 1, "token_fingerprint_fnv1a64": "0" * 16,
                "token_ids_sha256_le_i32": "0" * 64}
        self.assertEqual(compare_token_outputs({"runs": [None]}, {"runs": [good]})["status"], "fail")
        malformed = dict(good, token_ids_sha256_le_i32="z" * 64)
        self.assertEqual(compare_token_outputs({"runs": [good]}, {"runs": [malformed]})["status"], "fail")

    def test_benchmark_stats_and_rss_units_are_read_from_kiln_json(self):
        report = {
            "statistics": {"decode_tokens_per_second": {"mean": 4.25}},
            "process_memory": {"after_benchmark": {"peak_rss_to_date": 2 * 1024 * 1024}},
        }
        self.assertEqual(mean_metric(report, "decode_tokens_per_second"), 4.25)
        self.assertEqual(peak_rss_gib(report), 2.0)
        self.assertIsNone(mean_metric(None, "decode_tokens_per_second"))
        self.assertIsNone(peak_rss_gib({}))

    def test_audit_sums_actual_loaded_bytes_reads_and_cache_hits(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "audit.tsv"
            audit.write_text("blk.0.experts.weight\t3\t3\t120\t80\t1\n"
                             "blk.1.experts.weight\t2\t1\t40\t40\t0\n", encoding="utf-8")
            result = read_audit(audit)
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["requested"], 5)
        self.assertEqual(result["loaded"], 4)
        self.assertEqual(result["expert_bytes_served"], 160)
        self.assertEqual(result["pread_bytes"], 120)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["load_completion_ratio"], 0.8)

    def test_malformed_audit_rows_are_reported_without_discarding_valid_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "audit.tsv"
            audit.write_text("broken\nblk.0.experts.weight\t1\t1\t20\n", encoding="utf-8")
            result = read_audit(audit)
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["malformed_records"], 1)
        self.assertEqual(result["expert_bytes_served"], 20)

    def test_failed_subprocess_keeps_stdout_stderr_and_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner = Run(root)
            stage = runner.stage("deliberate-failure", [sys.executable, "-c",
                "import sys; print('partial output'); print('failure detail', file=sys.stderr); sys.exit(7)"])
            self.assertEqual(stage["status"], "failed")
            self.assertEqual(stage["exit_code"], 7)
            self.assertIn("partial output", Path(stage["stdout_log"]).read_text())
            self.assertIn("failure detail", Path(stage["stderr_log"]).read_text())
            commands = [json.loads(line) for line in runner.commands.read_text().splitlines()]
            self.assertEqual(commands[-1]["status"], "failed")
            self.assertEqual(commands[-1]["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
