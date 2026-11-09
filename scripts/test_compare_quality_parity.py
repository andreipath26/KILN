import unittest
from scripts.compare_quality_parity import assess

class QualityParityTests(unittest.TestCase):
    def sample(self, value):
        return {"protocol":"kiln-reset-window-next-token-nll-v1",
                "token_fingerprint_fnv1a64":"0011223344556677",
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
        cand["window_tokens"]=64
        with self.assertRaises(ValueError):
            assess(ref,cand,0.01)

    def test_rejects_invalid_or_nonfinite_quality(self):
        ref=self.sample(10)
        for bad in (float("nan"),0,-1,float("inf")):
            with self.assertRaises(ValueError):
                assess(ref,self.sample(bad),0.01)
