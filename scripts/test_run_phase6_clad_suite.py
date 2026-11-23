import tempfile
import unittest
from pathlib import Path

from scripts.run_phase6_clad_suite import normalized, read_jsonl, score_answer


class Phase6CaptureTests(unittest.TestCase):
    def test_native_diffusion_invalidates_sequence_state_before_logit_validation(self):
        root = Path(__file__).resolve().parents[1]
        shim = (root / "cpp/llama_shim/kiln_llama_shim.cpp").read_text(encoding="utf-8")
        start = shim.index("int32_t kiln_llama_diffusion_step(")
        end = shim.index("int32_t kiln_llama_tokenize(", start)
        diffusion_step = shim[start:end]

        decode = diffusion_step.index("const int32_t decode_rc = llama_decode(k->ctx, k->batch);")
        decode_failed = diffusion_step.index("if (decode_rc != 0)", decode)
        decode_failure_return = diffusion_step.index("return -3;", decode_failed)
        successful_reset = diffusion_step.index("k->n_past = 0;", decode_failure_return)
        logits_validation = diffusion_step.index("for (int32_t pos = 0; pos < n_tokens; ++pos)", decode_failed)
        self.assertLess(successful_reset, logits_validation)
        for failure in ("return -4;", "return -5;", "return -6;"):
            self.assertGreater(diffusion_step.index(failure), successful_reset)

    def test_normalization_ignores_unicode_compatibility_and_punctuation(self):
        self.assertEqual(normalized("  H₂O! "), normalized("h2o"))

    def test_accepted_answer_match_is_normalized_but_not_semantic(self):
        self.assertTrue(score_answer("  PARIS! ", ["Paris", "London"]))
        self.assertFalse(score_answer("Paris is the capital.", ["Paris"]))

    def test_reads_jsonl_and_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            path.write_text('{"id":"one","prompt":"Question?"}\n\n', encoding="utf-8")
            self.assertEqual(read_jsonl(path), [{"id": "one", "prompt": "Question?"}])
            path.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                read_jsonl(path)

    def test_empty_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no JSONL records"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
