"""Unit tests for deterministic synthetic MoE cache traces and accounting."""
import unittest

from scripts.synthetic_moe_cache_bench import Cache, make_trace, simulate


class SyntheticTraceTests(unittest.TestCase):
    def test_trace_is_reproducible_for_same_seed(self):
        self.assertEqual(
            make_trace(experts=32, requests=500, seed=17, workload="phase-shift"),
            make_trace(experts=32, requests=500, seed=17, workload="phase-shift"),
        )

    def test_seed_changes_randomized_trace(self):
        self.assertNotEqual(
            make_trace(experts=32, requests=500, seed=17, workload="uniform"),
            make_trace(experts=32, requests=500, seed=18, workload="uniform"),
        )

    def test_all_workloads_have_requested_length_and_valid_experts(self):
        for workload in ("phase-shift", "stationary-hot", "uniform", "round-robin"):
            with self.subTest(workload=workload):
                trace = make_trace(experts=32, requests=501, seed=42, workload=workload)
                self.assertEqual(len(trace), 501)
                self.assertTrue(all(0 <= expert < 32 for expert in trace))

    def test_round_robin_has_no_hits_when_cache_is_smaller_than_working_set(self):
        trace = make_trace(experts=16, requests=160, workload="round-robin")
        result = simulate(trace, 8, "lru", 1024, 100, 1.5)
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["misses"], len(trace))

    def test_stationary_hot_set_has_more_lru_hits_than_uniform_trace(self):
        stationary = make_trace(experts=64, requests=10_000, seed=7, workload="stationary-hot")
        uniform = make_trace(experts=64, requests=10_000, seed=7, workload="uniform")
        stationary_result = simulate(stationary, 8, "lru", 1024, 100, 1.5)
        uniform_result = simulate(uniform, 8, "lru", 1024, 100, 1.5)
        self.assertGreater(stationary_result["hit_rate"], uniform_result["hit_rate"])

    def test_result_accounting_is_consistent(self):
        trace = [0, 0, 1, 2, 0, 1]
        result = simulate(trace, 2, "lru", 4096, 100, 1.5)
        self.assertEqual(result["hits"] + result["misses"], len(trace))
        self.assertEqual(result["estimated_storage_bytes_read"], result["misses"] * 4096)

    def test_cache_capacity_zero_never_hits(self):
        cache = Cache(0, "none")
        self.assertFalse(cache.access(1))
        self.assertFalse(cache.access(1))

    def test_cache_never_exceeds_capacity(self):
        for policy in ("lru", "lfu", "adaptive"):
            with self.subTest(policy=policy):
                cache = Cache(4, policy)
                for key in range(100):
                    cache.access(key)
                    self.assertLessEqual(len(cache.items), 4)

    def test_invalid_trace_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            make_trace(experts=1)
        with self.assertRaises(ValueError):
            make_trace(requests=0)
        with self.assertRaises(ValueError):
            make_trace(workload="not-a-workload")


if __name__ == "__main__":
    unittest.main()
