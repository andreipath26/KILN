import tempfile
import unittest
from pathlib import Path

from scripts.run_phase6_clad_suite import normalized, read_jsonl, score_answer


class Phase6CaptureTests(unittest.TestCase):
    def test_normalization_ignores_unicode_compatibility_and_punctuation(self):
        self.assertEqual(normalized("  H₂O! "), normalized("h2o"))

    def test_accepted_answer_match_is_normalized_but_not_semantic(self):
        self.assertTrue(score_answer("The answer is Paris.", ["Paris", "London"]))
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
