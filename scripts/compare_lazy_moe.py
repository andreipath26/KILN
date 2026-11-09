#!/usr/bin/env python3
"""Run real KILN inference A/B: normal versus opt-in lazy MoE mmap.

This compares one pinned model and greedy token fingerprints. It does not
measure uncached device traffic or prove end-to-end streamed inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def positive(s):
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def checked_number(n, field):
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
        raise ValueError("invalid benchmark metric: " + field)
    return float(n)


def assess(captures, model, prompt, tokens, threads, batch, ubatch, runs):
    values = {mode: {"forward_tps": [], "response_tps": [], "load_ms": [], "peak_rss_kib": []}
              for mode in ("default", "lazy")}
    fingerprints = set()
    if {capture["mode"] for capture in captures} != {"default", "lazy"}:
        raise ValueError("both modes required")
    for capture in captures:
        mode, report = capture["mode"], capture["report"]
        if (report.get("lazy_moe_experts") is not (mode == "lazy") or
            Path(report.get("model_path_canonical", "")).resolve() != model.resolve() or
            report.get("prompt") != prompt or report.get("max_new_tokens") != tokens or
            report.get("requested_threads") != threads or report.get("requested_batch") != batch or
            report.get("requested_microbatch") != ubatch or report.get("measured_runs") != runs or
            report.get("sampling_path") != "native_greedy"):
            raise ValueError("benchmark configuration mismatch")
        outputs = report.get("runs")
        if not isinstance(outputs, list) or len(outputs) != runs:
            raise ValueError("unexpected benchmark run count")
        local = set()
        for row in outputs:
            digest = row.get("token_fingerprint_fnv1a64")
            count = row.get("generated_tokens")
            if type(count) is not int or count < 0 or not isinstance(digest, str) or len(digest) != 16:
                raise ValueError("missing generated-token parity evidence")
            local.add((count, digest))
        if len(local) != 1:
            raise ValueError("non-deterministic output within one benchmark capture")
        fingerprints.update(local)
        stats = report.get("statistics", {})
        row = values[mode]
        for key, name in (("decode_forward_tokens_per_second", "forward_tps"),
                          ("response_tokens_per_second_including_prefill", "response_tps")):
            row[name].append(checked_number(stats.get(key, {}).get("median"), key))
        row["load_ms"].append(checked_number(report.get("load_ms"), "load_ms"))
        peak = report.get("process_memory", {}).get("after_benchmark", {}).get("peak_rss_to_date")
        if peak is not None:
            row["peak_rss_kib"].append(checked_number(peak, "peak_rss_kib"))
    summary = {mode: {key + "_median": statistics.median(v) if v else None
                      for key, v in metrics.items()} for mode, metrics in values.items()}
    default_tps = summary["default"]["forward_tps_median"]
    lazy_tps = summary["lazy"]["forward_tps_median"]
    return {
        "status": "token_parity_passed" if len(fingerprints) == 1 else "token_parity_failed",
        "token_parity_exact": len(fingerprints) == 1,
        "token_sequences": [{"generated_tokens": n, "fingerprint": h} for n, h in sorted(fingerprints)],
        "modes": summary,
        "lazy_forward_speed_ratio": lazy_tps / default_tps if default_tps else None,
        "limitations": "Same-model greedy A/B only; page cache uncontrolled, lifetime peak process RSS is not total memory, and fingerprints do not establish FP16 quality.",
    }


def capture(binary, model, prompt, tokens, threads, batch, ubatch, runs, warmup, mode, timeout):
    command = [str(binary), "bench", str(model), "--prompt", prompt, "--tokens", str(tokens),
               "--runs", str(runs), "--warmup", str(warmup), "--threads", str(threads),
               "--batch", str(batch), "--ubatch", str(ubatch), "--native-greedy", "--json"]
    if mode == "lazy":
        command.append("--lazy-moe-experts")
    p = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError(f"{mode} benchmark exit {p.returncode}: {p.stderr[-2500:]}")
    return {"mode": mode, "report": json.loads(p.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--kiln-bin", type=Path, default=Path("target/release/kiln"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="The purpose of a model runtime is")
    parser.add_argument("--tokens", type=positive, default=64)
    parser.add_argument("--threads", type=positive, required=True)
    parser.add_argument("--batch", type=positive, default=128)
    parser.add_argument("--ubatch", type=positive, default=128)
    parser.add_argument("--runs", type=positive, default=3)
    parser.add_argument("--warmup", type=positive, default=1)
    parser.add_argument("--pairs", type=positive, default=2)
    parser.add_argument("--timeout", type=positive, default=3600)
    args = parser.parse_args()
    if args.ubatch > args.batch:
        parser.error("ubatch must not exceed batch")
    binary, model = args.kiln_bin.resolve(), args.model.resolve()
    if not binary.is_file() or not model.is_file():
        parser.error("model and compiled binary must exist")
    try:
        captures = []
        for pair in range(args.pairs):
            for mode in (("default", "lazy") if pair % 2 == 0 else ("lazy", "default")):
                captures.append(capture(binary, model, args.prompt, args.tokens, args.threads,
                                        args.batch, args.ubatch, args.runs, args.warmup, mode, args.timeout))
        evaluation = assess(captures, model, args.prompt, args.tokens, args.threads, args.batch,
                            args.ubatch, args.runs)
        artifact = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model), "model_sha256": sha256(model),
            "binary_sha256": sha256(binary),
            "configuration": {"prompt": args.prompt, "tokens": args.tokens, "threads": args.threads,
                              "batch": args.batch, "ubatch": args.ubatch, "runs": args.runs,
                              "warmup": args.warmup, "pairs": args.pairs},
            "evaluation": evaluation, "captures": captures,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evaluation, indent=2))
        return 0 if evaluation["token_parity_exact"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"lazy MoE A/B failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
