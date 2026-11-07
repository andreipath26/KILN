#!/usr/bin/env python3
"""Synthetic contract tests for summarize_phase35.py; requires only stdlib."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from scripts import summarize_phase35
except ImportError:  # Direct invocation from the scripts directory.
    import summarize_phase35


class Phase35SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dense").mkdir()
        (self.root / "provenance.txt").write_text(
            "kiln_commit=abc123\nmodel_sha256=deadbeef\n", encoding="utf-8"
        )
        (self.root / "profile-status.txt").write_text("pending: perf unavailable\n", encoding="utf-8")
        (self.root / "run-notes.txt").write_text("tier0_machine=UNCONFIRMED\n", encoding="utf-8")
        (self.root / "ollama-status.txt").write_text("pending: model not configured\n", encoding="utf-8")
        configs = [(t, b, min(b, 512), None) for t in (2, 3, 4) for b in (256, 512, 1024, 2048)]
        configs.extend((4, 512, u, u) for u in (128, 256, 512))
        for threads, batch, ubatch, micro_sweep in configs:
            name = f"t{threads}-b{batch}"
            if micro_sweep is not None:
                name = f"t{threads}-b{batch}-u{micro_sweep}"
            self.write_json(self.root / "dense" / f"{name}.json", {
                "requested_threads": threads,
                "requested_batch": batch,
                "requested_microbatch": ubatch,
                "sampling_path": "rust_logits_greedy",
                "prompt": "The meaning of life is",
                "max_new_tokens": 32,
                "profile": {"n_ctx": 4096},
                "statistics": {"decode_forward_tokens_per_second": {"median": 27.5}},
            })
        for mode in ("rust", "native"):
            self.write_json(self.root / f"sampler-{mode}.json", {
                "sampling_path": mode,
                "prompt": "The meaning of life is",
                "max_new_tokens": 32,
                "requested_threads": 4,
                "requested_batch": 512,
                "requested_microbatch": 128,
                "profile": {"n_ctx": 4096},
                "statistics": {
                    "decode_forward_tokens_per_second": {"median": 27.5},
                    "decode_tokens_per_second": {"median": 27.6},
                },
                "runs": [{
                    "generated_tokens": 32,
                    "eos_encountered": False,
                    "token_fingerprint_fnv1a64": "42",
                }],
            })
        for draft in (4, 8, 16):
            self.write_json(self.root / f"spec-d{draft}.json", {
                "all_output_sequences_equal": True,
                "relative_rate": 1.1,
                "draft_max": draft,
            })
        self.write_json(self.root / "kv.json", {
            "all_argmax_equal": True,
            "full_prefix_prefill_ms": {"median": 20.0},
            "cached_second_turn_prefill_ms": {"median": 5.0},
        })

    def test_profile_absence_remains_an_explicit_exit_gate_gap(self):
        report = summarize_phase35.summarize(self.root)
        self.assertFalse(report["hot_path_profile"]["captured"])
        self.assertEqual(report["status"], "capture_complete_exit_gate_pending")

    def test_profile_is_reported_only_with_artifact(self):
        (self.root / "profile").mkdir()
        (self.root / "profile" / "hotpath.txt").write_text("Samples: 99\n", encoding="utf-8")
        (self.root / "profile-status.txt").write_text("completed\n", encoding="utf-8")
        report = summarize_phase35.summarize(self.root)
        self.assertTrue(report["hot_path_profile"]["captured"])

    def test_ollama_baseline_is_compared_only_with_explicit_setting_checks(self):
        self.write_json(self.root / "ollama.json", {
            "ollama_model": "local-qwen:latest",
            "ollama_version": "test-version",
            "ollama_model_digest": "sha256:test",
            "weights_identity_verified": False,
            "median_generation_tokens_per_second": 30.0,
            "median_kiln_decode_tokens_per_second": 25.0,
            "paired_kiln_samples": [
                {"run": 1, "order": "kiln_then_ollama"},
                {"run": 2, "order": "ollama_then_kiln"},
            ],
            "prompt": "The meaning of life is",
            "tokens_requested": 32,
            "threads": 4,
            "batch": 512,
            "microbatch": 128,
            "context": 4096,
        })
        report = summarize_phase35.summarize(self.root)
        baseline = report["ollama_baseline"]
        self.assertEqual(baseline["ollama_to_kiln_rate_ratio"], 30.0 / 25.0)
        self.assertTrue(all(baseline["settings_match"].values()))
        self.assertFalse(baseline["same_weights_verified"])

    def tearDown(self):
        self.tmp.cleanup()

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_complete_evidence_meets_rate_but_never_auto_accepts_tier0(self):
        report = summarize_phase35.summarize(self.root)
        self.assertTrue(report["dense_gate"]["rate_threshold_met"])
        self.assertFalse(report["dense_gate"]["accepted"])
        self.assertEqual(len(report["dense_matrix_configurations"]), 15)

    def test_rejects_incomplete_dense_matrix(self):
        (self.root / "dense" / "t4-b512-u512.json").unlink()
        with self.assertRaisesRegex(ValueError, "dense matrix mismatch"):
            summarize_phase35.summarize(self.root)

    def test_rejects_wrong_dense_matrix_point_even_when_file_count_matches(self):
        (self.root / "dense" / "t4-b512-u512.json").unlink()
        self.write_json(self.root / "dense" / "t4-b512-u999.json", {
            "requested_threads": 4,
            "requested_batch": 512,
            "requested_microbatch": 999,
        })
        with self.assertRaisesRegex(ValueError, "dense matrix mismatch"):
            summarize_phase35.summarize(self.root)

    def test_rejects_speculation_without_exact_parity(self):
        self.write_json(self.root / "spec-d8.json", {
            "all_output_sequences_equal": False,
            "relative_rate": 1.2,
            "draft_max": 8,
        })
        with self.assertRaisesRegex(ValueError, "exact greedy parity"):
            summarize_phase35.summarize(self.root)

    def test_rejects_sampler_output_mismatch(self):
        self.write_json(self.root / "sampler-native.json", {
            "sampling_path": "native_greedy",
            "statistics": {
                "decode_forward_tokens_per_second": {"median": 27.5},
                "decode_tokens_per_second": {"median": 27.6},
            },
            "runs": [{
                "generated_tokens": 32,
                "eos_encountered": False,
                "token_fingerprint_fnv1a64": "different",
            }],
        })
        with self.assertRaisesRegex(ValueError, "sampler A/B greedy token parity"):
            summarize_phase35.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
