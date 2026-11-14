"""Checks the measurement report helpers and failure log retention."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.kiln_measure import Run, compare_token_outputs, mean_metric, peak_rss_gib, read_audit


class MeasureReportTests(unittest.TestCase):
    def test_token_parity_passes_for_identical_greedy_runs(self):
        runs = [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef"},
                {"generated_tokens": 11, "token_fingerprint_fnv1a64": "fedcba9876543210"}]
        result = compare_token_outputs({"runs": runs}, {"runs": [dict(row) for row in runs]})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["matched_runs"], 2)
        self.assertEqual(result["total_runs"], 2)

    def test_token_parity_fails_on_changed_fingerprint_or_run_count(self):
        reference = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "0123456789abcdef"}]}
        changed = {"runs": [{"generated_tokens": 12, "token_fingerprint_fnv1a64": "fedcba9876543210"}]}
        self.assertEqual(compare_token_outputs(reference, changed)["status"], "fail")
        self.assertEqual(compare_token_outputs(reference, {"runs": []})["status"], "fail")

    def test_token_parity_fails_closed_on_missing_report_fields(self):
        reference = {"runs": [{"generated_tokens": 12}]}
        result = compare_token_outputs(reference, {"runs": [{"generated_tokens": 12}]})
        self.assertEqual(result["status"], "fail")
        self.assertIn("missing token parity fields", result["reason"])
        self.assertEqual(compare_token_outputs(None, {"runs": []})["status"], "inconclusive")

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
