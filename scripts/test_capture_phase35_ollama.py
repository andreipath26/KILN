"""Unit tests for the local Ollama baseline capture."""
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from scripts.capture_phase35_ollama import capture, median_rate


class MockOllamaHandler(BaseHTTPRequestHandler):
    generate_requests = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        payload = {"version": "test"} if self.path == "/api/version" else {
            "models": [{"name": "kiln-test:latest", "digest": "sha256:test"}]
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.generate_requests.append(body)
        data = json.dumps({
            "done": True,
            "response": "test output",
            "eval_count": 20,
            "eval_duration": 1_000_000_000,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 100_000_000,
            "total_duration": 1_200_000_000,
            "load_duration": 0,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class OllamaCaptureTests(unittest.TestCase):
    def test_same_model_raw_greedy_measurement_records_provenance_and_rate(self):
        MockOllamaHandler.generate_requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockOllamaHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / "model.gguf"
                model.write_bytes(b"fixture weights")
                kiln = Path(directory) / "kiln"
                kiln.write_text(
                    "#!/usr/bin/env python3\n"
                    "import json\n"
                    "print(json.dumps({"
                    "'profile': {'n_ctx': 4096}, 'requested_threads': 4, "
                    "'requested_batch': 512, 'requested_microbatch': 128, "
                    "'statistics': {'decode_tokens_per_second': {'median': 25.0}, "
                    "'decode_forward_tokens_per_second': {'median': 24.0}}, "
                    "'runs': [{'generated_tokens': 16, 'token_fingerprint_fnv1a64': 'abc'}]}))\n",
                    encoding="utf-8",
                )
                kiln.chmod(0o755)
                args = SimpleNamespace(
                    gguf=model, model="kiln-test:latest",
                    host=f"http://127.0.0.1:{server.server_port}",
                    prompt="abc", tokens=16, runs=2, warmup=1,
                    threads=4, batch=512, ubatch=128, context=4096,
                    kiln_bin=kiln,
                )
                report = capture(args)
                self.assertEqual(report["ollama_version"], "test")
                self.assertEqual(report["ollama_model_digest"], "sha256:test")
                self.assertEqual(report["gguf_sha256"], hashlib.sha256(b"fixture weights").hexdigest())
                self.assertEqual(report["median_generation_tokens_per_second"], 20.0)
                self.assertEqual(report["median_kiln_decode_tokens_per_second"], 25.0)
                self.assertEqual(report["ollama_to_kiln_rate_ratio"], 0.8)
                self.assertEqual([x["order"] for x in report["paired_kiln_samples"]], [
                    "kiln_then_ollama", "ollama_then_kiln",
                ])
                self.assertFalse(report["weights_identity_verified"])
                self.assertEqual(len(MockOllamaHandler.generate_requests), 3)
                request = MockOllamaHandler.generate_requests[-1]
                self.assertTrue(request["raw"])
                self.assertFalse(request["stream"])
                self.assertEqual(request["options"]["temperature"], 0)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_rejects_invalid_rate_samples(self):
        with self.assertRaises(ValueError):
            median_rate([0.0, 20.0])


if __name__ == "__main__":
    unittest.main()
