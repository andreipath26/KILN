import unittest
from scripts.compare_quality_parity import assess

class QualityParityTests(unittest.TestCase):
    def sample(self, value):
        return {"protocol":"kiln-reset-window-next-token-nll-v1",
                "token_fingerprint_fnv1a64":"0011223344556677",
                "token_ids_sha256_le_i32":"a"*64,
                "evaluated_token_count":1000,
                "scored_target_tokens":999,
                "window_tokens":512,"perplexity":value}

    def test_one_percent_quality_regression_bound(self):
        ref = self.sample(10.0)
        self.assertEqual(assess(ref,self.sample(10.1),0.01)["status"],"pass")
        self.assertEqual(assess(ref,self.sample(10.101),0.01)["status"],"fail")

    def test_mismatched_tokenization_is_never_called_accurate(self):
        ref=self.sample(10)
        cand=self.sample(9)
        cand["token_fingerprint_fnv1a64"]="ffff"
        with self.assertRaises(ValueError):
            assess(ref,cand,0.01)
        cand=self.sample(9)
        cand["token_ids_sha256_le_i32"]="b"*64
        with self.assertRaises(ValueError):
            assess(ref,cand,0.01)
        cand=self.sample(9)
        cand["window_tokens"]=64
        with self.assertRaises(ValueError):
            assess(ref,cand,0.01)

    def test_rejects_invalid_or_nonfinite_quality(self):
        ref=self.sample(10)
        for bad in (float("nan"),0,-1,float("inf")):
            with self.assertRaises(ValueError):
                assess(ref,self.sample(bad),0.01)

    def test_missing_required_evidence_never_passes_even_when_both_omit_it(self):
        for missing in ("protocol", "token_fingerprint_fnv1a64",
                        "token_ids_sha256_le_i32", "evaluated_token_count",
                        "scored_target_tokens", "window_tokens"):
            ref = self.sample(10)
            candidate = self.sample(10)
            del ref[missing]
            del candidate[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                assess(ref, candidate, 0.01)

    def test_forged_or_inconsistent_scoring_metadata_is_rejected(self):
        corruptions = (
            ("token_ids_sha256_le_i32", "a" * 63),
            ("token_ids_sha256_le_i32", "z" * 64),
            ("token_fingerprint_fnv1a64", "x" * 16),
            ("evaluated_token_count", True),
            ("evaluated_token_count", 1),
            ("scored_target_tokens", 998),
            ("scored_target_tokens", True),
            ("window_tokens", 1),
            ("window_tokens", 8193),
            ("window_tokens", 512.0),
            ("perplexity", 0.5),
            ("perplexity", True),
        )
        for key, invalid in corruptions:
            with self.subTest(key=key, invalid=invalid):
                ref = self.sample(10)
                ref[key] = invalid
                with self.assertRaises(ValueError):
                    assess(ref, self.sample(10), 0.01)

    def test_reference_is_validated_as_strictly_as_candidate(self):
        candidate = self.sample(10)
        for invalid in (None, [], {"protocol": candidate["protocol"]}):
            with self.assertRaises(ValueError):
                assess(invalid, candidate, 0.01)
