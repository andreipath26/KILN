"""Regression coverage for real-CPU MoE operator route capture conversion."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("convert_moe_router_trace", Path(__file__).with_name("convert_moe_router_trace.py"))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class RouterTraceConversionTests(unittest.TestCase):
    def test_replays_original_operator_order_and_projection_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "route.tsv", Path(folder) / "route.json"
            source.write_text(
                "0\t2\t0\tblk.0.ffn_up_exps\n"
                "0\t2\t0\tblk.0.ffn_down_exps\n"
                "1\t4\t0\tblk.1.ffn_up_exps\n",
                encoding="utf-8")
            report = converter.convert(source, output)
            self.assertEqual(report["events"], 3)
            self.assertEqual(json.loads(output.read_text()), [
                {"layer": 0, "expert_id": 2},
                {"layer": 0, "expert_id": 2},
                {"layer": 1, "expert_id": 4},
            ])
            self.assertEqual(len(report["tensor_operator_counts"]), 3)
            with self.assertRaises(FileExistsError):
                converter.convert(source, output)

    def test_rejects_bad_tensor_and_mismatched_layer_without_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "route.tsv", Path(folder) / "route.json"
            for bad in ("0\t2\t0\tunsupported-name\n", "1\t2\t0\tblk.0.ffn_up_exps\n",
                        "-1\t0\t0\tblk.1.weights\n", "0\t2\tblk.0.weights\n"):
                source.write_text(bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    converter.convert(source, output)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_name(output.name + ".partial").exists())

    def test_bounded_events_and_no_empty_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "route.tsv", Path(folder) / "route.json"
            source.write_text("0\t1\t0\tblk.0.weights\n" * 3)
            with self.assertRaisesRegex(ValueError, "max-events"):
                converter.convert(source, output, maximum=2)
            self.assertFalse(output.exists())
            source.write_text("")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                converter.convert(source, output)

    def test_grouped_conversion_collapses_projection_repeats_and_orders_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "route.tsv", Path(folder) / "grouped.json"
            source.write_text(
                "@decode\t1\t2\n"
                "0\t1\t0\tblk.0.ffn_up_exps\n"
                "0\t1\t0\tblk.0.ffn_down_exps\n"
                "0\t2\t0\tblk.0.ffn_down_exps\n"
                "1\t3\t0\tblk.1.ffn_up_exps\n"
                "0\t4\t1\tblk.0.ffn_up_exps\n"
                "@decode\t2\t1\n"
                "0\t2\t0\tblk.0.ffn_up_exps\n",
                encoding="utf-8")
            report = converter.convert_grouped(source, output)
            self.assertEqual(report["decode_tokens"], 3)
            self.assertEqual(report["raw_operator_events"], 6)
            self.assertEqual(report["collapsed_projection_events"], 2)
            self.assertEqual(json.loads(output.read_text()), [
                {"token_position": 0, "layer": 0, "active_experts": [1, 2]},
                {"token_position": 0, "layer": 1, "active_experts": [3]},
                {"token_position": 1, "layer": 0, "active_experts": [4]},
                {"token_position": 2, "layer": 0, "active_experts": [2]},
            ])

    def test_grouped_conversion_rejects_legacy_trace_without_token_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "route.tsv", Path(folder) / "grouped.json"
            source.write_text("0\t1\t0\tblk.0.ffn_up_exps\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decode-marked"):
                converter.convert_grouped(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
