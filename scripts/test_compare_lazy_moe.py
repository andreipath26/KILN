"""Regression tests for real-inference A/B report validation."""
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_lazy_moe", Path(__file__).with_name("compare_lazy_moe.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def benchmark(lazy, fingerprint="1234567890abcdef"):
    return {
        "lazy_moe_experts": lazy, "model_path_canonical": "/tmp/model.gguf",
        "prompt": "test", "max_new_tokens": 8, "requested_threads": 4,
        "requested_batch": 128, "requested_microbatch": 128,
        "measured_runs": 2, "sampling_path": "native_greedy", "load_ms": 100,
        "process_memory": {"after_benchmark": {"peak_rss_to_date": 1000}},
        "runs": [{"generated_tokens": 8, "token_fingerprint_fnv1a64": fingerprint}] * 2,
        "statistics": {"decode_forward_tokens_per_second": {"median": 6},
                       "response_tokens_per_second_including_prefill": {"median": 5}},
    }


class ComparisonTests(unittest.TestCase):
    def evaluate(self, lazy=None):
        captures = [{"mode": "default", "report": benchmark(False)},
                    {"mode": "lazy", "report": lazy or benchmark(True)}]
        return module.assess(captures, Path("/tmp/model.gguf"), "test", 8, 4, 128, 128, 2)

    def test_exact_token_parity_and_rates(self):
        result = self.evaluate()
        self.assertTrue(result["token_parity_exact"])
        self.assertEqual(result["lazy_forward_speed_ratio"], 1)
        self.assertEqual(result["modes"]["lazy"]["peak_rss_kib_median"], 1000)

    def test_detects_output_divergence(self):
        self.assertFalse(self.evaluate(benchmark(True, "deadbeefdeadbeef"))["token_parity_exact"])

    def test_rejects_incorrect_parameters(self):
        invalid = benchmark(True)
        invalid["requested_threads"] = 9
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.evaluate(invalid)

    def test_rejects_nondeterministic_tokens(self):
        invalid = benchmark(True)
        invalid["runs"] = [invalid["runs"][0], {"generated_tokens": 8, "token_fingerprint_fnv1a64": "deadbeefdeadbeef"}]
        with self.assertRaisesRegex(ValueError, "non-deterministic"):
            self.evaluate(invalid)

    def test_rejects_invalid_measurement(self):
        invalid = benchmark(True)
        invalid["statistics"]["decode_forward_tokens_per_second"]["median"] = math.nan
        with self.assertRaisesRegex(ValueError, "invalid benchmark metric"):
            self.evaluate(invalid)


if __name__ == "__main__":
    unittest.main()
