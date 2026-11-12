"""Tests for replaying native MoE routing traces through byte-bounded caches."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.replay_moe_cache_trace import ByteCache, expert_sizes, replay, simulate


class ByteCacheReplayTests(unittest.TestCase):
    def test_variable_expert_sizes_stay_within_byte_budget(self):
        requests = [((0, 1), 4), ((0, 1), 4), ((0, 2), 4)]
        sizes = {(0, 1): 8, (0, 2): 8}
        result = simulate(requests, sizes, 8, "lru")
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["misses"], 2)
        self.assertEqual(result["modelled_expert_fill_bytes_on_miss"], 16)
        self.assertEqual(result["resident_bytes_at_end"], 8)
        self.assertEqual(result["evictions"], 1)

    def test_lfu_retains_reused_expert(self):
        cache = ByteCache(8, "lfu")
        self.assertFalse(cache.access((0, 1), 4))
        self.assertFalse(cache.access((0, 2), 4))
        self.assertTrue(cache.access((0, 1), 4))
        self.assertTrue(cache.access((0, 1), 4))
        self.assertFalse(cache.access((0, 3), 4))
        self.assertIn((0, 1), cache.entries)
        self.assertNotIn((0, 2), cache.entries)
        self.assertLessEqual(cache.used, cache.capacity)

    def test_adaptive_admission_rejects_cold_scan_against_hot_residents(self):
        cache = ByteCache(8, "adaptive")
        cache.access((0, 1), 4)
        cache.access((0, 2), 4)
        for _ in range(8):
            cache.access((0, 1), 4)
            cache.access((0, 2), 4)
        self.assertFalse(cache.access((0, 3), 4))
        self.assertEqual(set(cache.entries), {(0, 1), (0, 2)})
        self.assertEqual(cache.cold_rejections, 1)

    def test_disabled_cache_never_hits_or_retains(self):
        result = simulate([((0, 1), 4)] * 3, {(0, 1): 8}, 0, "none")
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["misses"], 3)
        self.assertEqual(result["resident_bytes_at_end"], 0)

    def test_real_trace_format_groups_tensor_slabs_by_layer_and_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "fixture.gguf"
            model.write_bytes(bytes(range(128)))
            index = root / "native-index.tsv"
            index.write_text(
                "KILN-MOE-STREAM-1\n"
                "blk.0.ffn_up_exps.weight\t1\t0\t8\n"
                "blk.0.ffn_down_exps.weight\t1\t8\t12\n"
                "blk.1.ffn_up_exps.weight\t2\t20\t16\n",
                encoding="utf-8",
            )
            report = root / "index-report.json"
            report.write_text(json.dumps({
                "status": "original_gguf_external_expert_offsets_verified",
                "unchanged_model_file": str(model.resolve()),
                "output": str(index.resolve()),
                "entries": 3,
                "requires_duplicate_sidecar": False,
            }), encoding="utf-8")
            trace = root / "router.tsv"
            trace.write_text(
                "0\t1\t0\tblk.0.ffn_up_exps.weight\n"
                "0\t1\t0\tblk.0.ffn_down_exps.weight\n"
                "1\t2\t0\tblk.1.ffn_up_exps.weight\n",
                encoding="utf-8",
            )
            sizes = expert_sizes({
                ("blk.0.ffn_up_exps.weight", 1): (0, 8),
                ("blk.0.ffn_down_exps.weight", 1): (8, 12),
                ("blk.1.ffn_up_exps.weight", 2): (20, 16),
            })
            self.assertEqual(sizes, {(0, 1): 20, (1, 2): 16})
            result = replay(model, index, report, trace, [20], 10)
            self.assertEqual(result["trace_events"], 3)
            self.assertEqual(result["indexed_expert_count"], 2)
            self.assertEqual(len(result["results"]), 4)
            self.assertTrue(result["trace_order_preserved"])
            self.assertEqual(result["results"][1]["policy"], "lru")
            self.assertEqual(result["results"][1]["hits"], 1)


if __name__ == "__main__":
    unittest.main()
