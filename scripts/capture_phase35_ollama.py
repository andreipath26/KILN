"""Measure a same-host raw-completion baseline using a local Ollama server.

Import the exact GGUF into Ollama first, then pass its local model name. The
report records both the GGUF SHA and Ollama digest but cannot prove they refer
to the same weights; verify that mapping in run-notes.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def call_api(host: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        host.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urlopen(request, timeout=600) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def median_rate(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("generation rates must be finite and positive")
    return float(statistics.median(values))


def capture(args: argparse.Namespace) -> dict:
    model_path = args.gguf.resolve(strict=True)
    if not model_path.is_file() or not args.model.strip():
        raise ValueError("--gguf must be a file and --model must name an imported Ollama model")
    version = call_api(args.host, "/api/version")
    models = call_api(args.host, "/api/tags").get("models")
    if not isinstance(models, list):
        raise ValueError("Ollama /api/tags response has no models list")
    model = next((item for item in models if item.get("name") == args.model), None)
    if model is None:
        raise ValueError(f"Ollama model {args.model!r} is not present in /api/tags")

    def generate() -> dict:
        return call_api(args.host, "/api/generate", {
            "model": args.model,
            "prompt": args.prompt,
            "raw": True,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "seed": 1,
                "num_predict": args.tokens,
                "num_thread": args.threads,
                "num_batch": args.batch,
                "num_ctx": args.context,
            },
        })

    def kiln_bench() -> dict:
        command = [
            str(args.kiln_bin), "bench", str(model_path), "--prompt", args.prompt,
            "--tokens", str(args.tokens), "--warmup", "0", "--runs", "1",
            "--threads", str(args.threads), "--batch", str(args.batch),
            "--ubatch", str(args.ubatch), "--json",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        report = json.loads(completed.stdout)
        stats = report["statistics"]
        run = report["runs"][0]
        return {
            "decode_tokens_per_second": stats["decode_tokens_per_second"]["median"],
            "decode_forward_tokens_per_second": stats["decode_forward_tokens_per_second"]["median"],
            "generated_tokens": run["generated_tokens"],
            "token_fingerprint_fnv1a64": run["token_fingerprint_fnv1a64"],
            "load_ms_excluded_from_rate": report.get("load_ms"),
        }

    for _ in range(args.warmup):
        if args.kiln_bin:
            kiln_bench()
        if generate().get("done") is not True:
            raise ValueError("Ollama warm-up did not complete")

    samples, kiln_samples = [], []
    for index in range(args.runs):
        order = ("kiln_then_ollama" if index % 2 == 0 else "ollama_then_kiln")
        kiln_result = None
        ollama_result = None
        if args.kiln_bin and index % 2 == 0:
            kiln_result = kiln_bench()
        started = time.perf_counter()
        result = generate()
        wall_ms = (time.perf_counter() - started) * 1000
        if args.kiln_bin and index % 2 == 1:
            kiln_result = kiln_bench()
        count, duration = result.get("eval_count"), result.get("eval_duration")
        if (not isinstance(count, int) or count <= 0
                or not isinstance(duration, int) or duration <= 0):
            raise ValueError(f"Ollama run {index + 1}: invalid eval_count/eval_duration")
        if result.get("done") is not True or not isinstance(result.get("response"), str):
            raise ValueError(f"Ollama run {index + 1}: incomplete or invalid response")
        ollama_sample = {
            "run": index + 1,
            "prompt_eval_count": result.get("prompt_eval_count"),
            "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
            "generated_tokens": count,
            "generation_duration_ns": duration,
            "generation_tokens_per_second": count * 1_000_000_000 / duration,
            "total_duration_ns": result.get("total_duration"),
            "load_duration_ns": result.get("load_duration"),
            "wall_request_ms_including_http": wall_ms,
            "response_sha256": hashlib.sha256(result["response"].encode("utf-8")).hexdigest(),
        }
        samples.append(ollama_sample)
        if kiln_result is not None:
            kiln_samples.append({"run": index + 1, "order": order, **kiln_result})

    rates = [sample["generation_tokens_per_second"] for sample in samples]
    kiln_rates = [sample["decode_tokens_per_second"] for sample in kiln_samples]
    if args.kiln_bin and len(kiln_samples) != args.runs:
        raise ValueError("KILN/Ollama paired run is missing a KILN sample")
    with model_path.open("rb") as file:
        model_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return {
        "phase": "3.5",
        "measurement": "ollama_same_host_raw_completion_baseline",
        "ollama_version": version.get("version"),
        "ollama_model": args.model,
        "ollama_model_digest": model.get("digest"),
        "gguf_path": str(model_path),
        "gguf_sha256": model_hash,
        "weights_identity_verified": False,
        "prompt": args.prompt,
        "tokens_requested": args.tokens,
        "runs": args.runs,
        "warmup": args.warmup,
        "threads": args.threads,
        "batch": args.batch,
        "microbatch": args.ubatch,
        "context": args.context,
        "raw_prompt": True,
        "temperature": 0,
        "median_generation_tokens_per_second": median_rate(rates),
        "median_kiln_decode_tokens_per_second": median_rate(kiln_rates) if kiln_rates else None,
        "median_kiln_decode_forward_tokens_per_second": median_rate(
            [sample["decode_forward_tokens_per_second"] for sample in kiln_samples]
        ) if kiln_samples else None,
        "ollama_to_kiln_rate_ratio": (
            median_rate(rates) / median_rate(kiln_rates) if kiln_rates else None
        ),
        "min_generation_tokens_per_second": min(rates),
        "max_generation_tokens_per_second": max(rates),
        "paired_kiln_samples": kiln_samples,
        "samples": samples,
        "note": "Rate uses eval_count/eval_duration. Confirm the Ollama digest maps to the supplied GGUF SHA and record matching CPU, power and thermal conditions. HTTP and model load are excluded from generation rate.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Exact model name shown by Ollama /api/tags")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--kiln-bin", type=Path, default=Path("target/release/kiln"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (min(args.tokens, args.runs, args.threads, args.batch, args.context, args.ubatch) <= 0
            or args.runs > 100 or not 0 <= args.warmup <= 100):
        parser.error("numeric options must be positive; runs/warmup must be <= 100")
    if args.ubatch > args.batch:
        parser.error("--ubatch must be <= --batch")
    try:
        report = capture(args)
    except (OSError, URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Ollama baseline capture failed: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
