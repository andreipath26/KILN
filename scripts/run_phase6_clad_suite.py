#!/usr/bin/env python3
"""Capture a reproducible Phase 6 CLAD run against a fixed factual JSONL suite."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import unicodedata
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        offset = 0
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                try:
                    os.posix_fadvise(source.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            offset += len(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no JSONL records")
    return rows


def normalized(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", folded, flags=re.UNICODE))


def score_answer(answer: str, references: list[str]) -> bool:
    actual = normalized(answer)
    return any(actual == normalized(reference) for reference in references)


def cpu_description() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def system_memory_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def available_memory_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def git_revision() -> str | None:
    if os.environ.get("KILN_REVISION"):
        return os.environ["KILN_REVISION"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kiln", required=True, type=Path, help="built KILN executable")
    parser.add_argument("--model", required=True, type=Path, help="pinned LLaDA GGUF")
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/phase6/factual.jsonl"))
    parser.add_argument("--output", required=True, type=Path, help="new/replaceable result JSON path")
    parser.add_argument("--repeats", type=int, default=3, help="complete suite passes; each pass loads the model once")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--quantization", help="model quantization label for the evidence record")
    parser.add_argument("--reference-jsonl", type=Path, help="vanilla LLaDA results: JSONL {id, answer}")
    parser.add_argument("--reference-model-sha256", help="baseline checkpoint hash; must equal the KILN model hash")
    parser.add_argument("--reference-runtime", help="vanilla decoder implementation and pinned revision")
    args = parser.parse_args()

    for label, path in (("KILN executable", args.kiln), ("model", args.model), ("suite", args.suite)):
        if not path.is_file():
            parser.error(f"{label} is not a file: {path}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.max_tokens <= 0 or args.block_length <= 0 or args.max_tokens % args.block_length:
        parser.error("--max-tokens must be a positive multiple of --block-length")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in 0..1")

    suite = read_jsonl(args.suite)
    ids = set()
    for row in suite:
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            parser.error("every suite row needs a non-empty string id")
        if row["id"] in ids:
            parser.error(f"duplicate suite id: {row['id']}")
        ids.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            parser.error(f"{row['id']}: prompt must be a non-empty string")
        refs = row.get("reference_answers")
        if not isinstance(refs, list) or not refs or not all(isinstance(x, str) and x.strip() for x in refs):
            parser.error(f"{row['id']}: reference_answers must be a non-empty string list")

    baseline = None
    if args.reference_jsonl:
        if not args.reference_model_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", args.reference_model_sha256):
            parser.error("--reference-jsonl requires a 64-character --reference-model-sha256")
        if not args.reference_runtime:
            parser.error("--reference-jsonl requires --reference-runtime provenance")
        baseline_rows = read_jsonl(args.reference_jsonl)
        baseline = {}
        for row in baseline_rows:
            if not isinstance(row.get("id"), str) or not isinstance(row.get("answer"), str):
                parser.error("reference JSONL rows need string id and answer fields")
            if row["id"] in baseline:
                parser.error(f"duplicate reference id: {row['id']}")
            baseline[row["id"]] = row["answer"]
        if set(baseline) != ids:
            parser.error("reference JSONL IDs must match the factual suite exactly")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.kiln.resolve()), "diffuse", str(args.model.resolve()),
        "--prompt-file", str(args.suite.resolve()),
        "--max-tokens", str(args.max_tokens),
        "--block-length", str(args.block_length),
        "--threshold", str(args.threshold),
        "--json",
    ]
    steps = args.steps if args.steps is not None else args.max_tokens
    command.extend(["--steps", str(steps)])
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])

    model_sha256 = sha256_file(args.model)
    if args.reference_jsonl and args.reference_model_sha256.lower() != model_sha256:
        parser.error("vanilla baseline checkpoint hash must match the KILN GGUF SHA-256")
    suite_sha256 = sha256_file(args.suite)
    available_at_start_kib = available_memory_kib()
    runs = []
    for repeat in range(args.repeats):
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            sys.stderr.write(completed.stderr)
            raise RuntimeError(f"KILN run {repeat + 1}/{args.repeats} exited {completed.returncode}")
        parsed = []
        for line_no, line in enumerate(completed.stdout.splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"KILN stdout line {line_no} is not JSON: {exc}") from exc
            parsed.append(row)
        by_id = {row.get("case_id"): row for row in parsed}
        if len(by_id) != len(parsed) or set(by_id) != ids:
            raise RuntimeError(f"run {repeat + 1}: output IDs do not exactly match the suite")
        for case in suite:
            result = by_id[case["id"]]
            if not isinstance(result.get("answer"), str) or not isinstance(result.get("decode_tokens_per_second"), (int, float)):
                raise RuntimeError(f"{case['id']}: missing answer or decode speed in KILN JSON")
            result["gold_match"] = score_answer(result["answer"], case["reference_answers"])
        runs.append({"repeat": repeat + 1, "cases": [by_id[row["id"]] for row in suite]})

    all_cases = [case for run in runs for case in run["cases"]]
    speeds = [float(case["decode_tokens_per_second"]) for case in all_cases]
    gold_accuracy = sum(case["gold_match"] for case in all_cases) / len(all_cases)
    reference_accuracy = None
    quality_delta_percentage_points = None
    if baseline is not None:
        reference_accuracy = sum(
            score_answer(baseline[row["id"]], row["reference_answers"]) for row in suite
        ) / len(suite)
        quality_delta_percentage_points = 100.0 * (gold_accuracy - reference_accuracy)

    try:
        version = subprocess.run(
            [str(args.kiln.resolve()), "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        version = None

    output = {
        "schema_version": 1,
        "phase": 6,
        "method": "KILN CLAD",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation_revision": git_revision(),
        "kiln_version": version,
        "executable": str(args.kiln.resolve()),
        "model": {
            "path": str(args.model.resolve()),
            "size_bytes": args.model.stat().st_size,
            "sha256": model_sha256,
        },
        "suite": {
            "path": str(args.suite.resolve()),
            "sha256": suite_sha256,
            "case_ids": [row["id"] for row in suite],
        },
        "vanilla_reference": (
            None if args.reference_jsonl is None else {
                "results_path": str(args.reference_jsonl.resolve()),
                "results_sha256": sha256_file(args.reference_jsonl),
                "model_sha256": args.reference_model_sha256.lower(),
                "runtime": args.reference_runtime,
            }
        ),
        "machine": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu": cpu_description(),
            "logical_cpus": os.cpu_count(),
            "mem_total_kib": system_memory_kib(),
            "mem_available_at_start_kib": available_at_start_kib,
        },
        "configuration": {
            "max_tokens": args.max_tokens,
            "steps": steps,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "threads": args.threads,
            "quantization": args.quantization,
            "repeats": args.repeats,
            "model_load_excluded_from_decode_timer": True,
        },
        "summary": {
            "decode_tokens_per_second_median": statistics.median(speeds),
            "decode_tokens_per_second_min": min(speeds),
            "decode_tokens_per_second_max": max(speeds),
            "exact_gold_accuracy": gold_accuracy,
            "reference_accuracy": reference_accuracy,
            "quality_delta_percentage_points": quality_delta_percentage_points,
            "quality_gate_pass": (
                None if quality_delta_percentage_points is None or args.reference_model_sha256 is None
                else quality_delta_percentage_points >= -1.0
            ),
            "speed_gate_pass": statistics.median(speeds) >= 6.0,
            "hardware_acceptance": False,
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summary": output["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"phase6 suite: {exc}", file=sys.stderr)
        raise SystemExit(2)
