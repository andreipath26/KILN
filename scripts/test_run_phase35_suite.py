#!/usr/bin/env python3
"""Regression tests for immutable Phase 3.5 capture directories."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts" / "run_phase35_suite.sh"


class Phase35SuiteTests(unittest.TestCase):
    def test_refuses_existing_directory_without_overwriting_stale_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "fixture.gguf"
            model.write_bytes(b"readable test fixture; no model parsing should happen")
            output = root / "capture"
            stale_dense = output / "dense" / "t2-b256.json"
            stale_dense.parent.mkdir(parents=True)
            stale_contents = b'{"stale": true}\n'
            stale_dense.write_bytes(stale_contents)

            result = subprocess.run(
                ["bash", str(SUITE), str(model), str(output)],
                cwd=ROOT,
                env={**os.environ, "KILN_SKIP_BUILD": "1"},
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Refusing to reuse existing Phase 3.5 output directory", result.stderr)
            self.assertEqual(stale_dense.read_bytes(), stale_contents)
            self.assertFalse((output / "provenance.txt").exists())
            self.assertFalse((output / "run-notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
