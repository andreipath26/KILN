#!/usr/bin/env python3
"""Validate and summarize a run_phase35_suite.sh evidence directory.

The report distinguishes measurement coverage from the Dell performance gate.
It never labels a run accepted without both the required dense rate and a
manually recorded confirmation of the Tier 0 machine/model identity.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def median_rate(path: Path, key: str) -> float:
    data = read_json(path)
    stats = data.get("statistics", {}).get(key)
    value = stats.get("median") if isinstance(stats, dict) else None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{path}: missing or invalid statistics.{key}.median")
    return float(value)


def summarize(root: Path) -> dict:
    provenance_path = root / "provenance.txt"
    if not provenance_path.is_file():
        raise ValueError(f"{root}: missing provenance.txt")
    provenance = {}
    for line in provenance_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            provenance[key] = value
    if not provenance.get("kiln_commit") or not provenance.get("model_sha256"):
        raise ValueError(f"{provenance_path}: missing commit or model SHA-256")

    dense_dir = root / "dense"
    dense_files = sorted(dense_dir.glob("*.json"))
    expected_names = {
        f"t{threads}-b{batch}"
        for threads in (2, 3, 4)
        for batch in (256, 512, 1024, 2048)
    }
    expected_names.update(f"t4-b512-u{ubatch}" for ubatch in (128, 256, 512))
    observed_names = {path.stem for path in dense_files}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(f"{dense_dir}: dense matrix mismatch; missing={missing}, unexpected={unexpected}")
    dense = []
    for path in dense_files:
        data = read_json(path)
        expected_threads = int(path.stem.split("-")[0][1:])
        expected_batch = int(path.stem.split("-")[1][1:])
        expected_microbatch = int(path.stem.split("-u")[1]) if "-u" in path.stem else min(expected_batch, 512)
        if (data.get("requested_threads"), data.get("requested_batch"), data.get("requested_microbatch")) != (
            expected_threads, expected_batch, expected_microbatch
        ):
            raise ValueError(f"{path}: report settings do not match the matrix point")
        dense.append({
            "configuration": path.stem,
            "threads": data.get("requested_threads"),
            "batch": data.get("requested_batch"),
            "microbatch": data.get("requested_microbatch"),
            "sampling_path": data.get("sampling_path"),
            "decode_forward_tokens_per_second_median": median_rate(
                path, "decode_forward_tokens_per_second"
            ),
        })
    best = max(dense, key=lambda row: row["decode_forward_tokens_per_second_median"])

    sampler = {}
    sampler_runs = {}
    for mode in ("rust", "native"):
        path = root / f"sampler-{mode}.json"
        data = read_json(path)
        run_rows = data.get("runs")
        if not isinstance(run_rows, list) or not run_rows:
            raise ValueError(f"{path}: missing per-run token evidence")
        sampler_runs[mode] = run_rows
        sampler[mode] = {
            "sampling_path": data.get("sampling_path"),
            "decode_forward_tokens_per_second_median": median_rate(
                path, "decode_forward_tokens_per_second"
            ),
            "decode_tokens_per_second_median": median_rate(path, "decode_tokens_per_second"),
        }
    if len(sampler_runs["rust"]) != len(sampler_runs["native"]):
        raise ValueError("sampler A/B run counts differ")
    sampler_parity = []
    for index, (rust, native) in enumerate(zip(sampler_runs["rust"], sampler_runs["native"]), 1):
        keys = ("generated_tokens", "eos_encountered", "token_fingerprint_fnv1a64")
        if any(rust.get(key) != native.get(key) for key in keys):
            raise ValueError(f"sampler A/B greedy token parity failed on measured run {index}")
        sampler_parity.append({"run": index, "greedy_token_fingerprint_equal": True})
    sampler["all_runs_greedy_token_parity"] = True
    sampler["parity_evidence"] = sampler_parity

    speculation = []
    for path in sorted(root.glob("spec-d*.json")):
        data = read_json(path)
        if data.get("all_output_sequences_equal") is not True:
            raise ValueError(f"{path}: exact greedy parity failed or is unreported")
        rate = data.get("relative_rate")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
            raise ValueError(f"{path}: invalid relative_rate")
        speculation.append({
            "configuration": path.stem,
            "draft_max": data.get("draft_max"),
            "baseline_median_tokens_per_second": data.get("baseline_median_tokens_per_second"),
            "speculative_median_tokens_per_second": data.get("speculative_median_tokens_per_second"),
            "relative_rate": rate,
            "exact_greedy_parity": True,
        })
    if len(speculation) != 3:
        raise ValueError(f"{root}: expected 3 speculation configurations, found {len(speculation)}")

    kv_path = root / "kv.json"
    kv = read_json(kv_path)
    if kv.get("all_argmax_equal") is not True:
        raise ValueError(f"{kv_path}: KV reuse parity failed or is unreported")

    profile_status_path = root / "profile-status.txt"
    if not profile_status_path.is_file():
        raise ValueError(f"{root}: missing profile-status.txt")
    profile_status = profile_status_path.read_text(encoding="utf-8").strip()
    profile_captured = profile_status == "completed" and (root / "profile" / "hotpath.txt").is_file()
    ollama_path = root / "ollama.json"
    if ollama_path.is_file():
        ollama = read_json(ollama_path)
        kiln_bench = read_json(root / "sampler-rust.json")
        kiln_rate = ollama.get("median_kiln_decode_tokens_per_second")
        ollama_rate = ollama.get("median_generation_tokens_per_second")
        profile = kiln_bench.get("profile", {})
        ollama_summary = {
            "captured": True,
            "model": ollama.get("ollama_model"),
            "ollama_version": ollama.get("ollama_version"),
            "model_digest": ollama.get("ollama_model_digest"),
            "same_weights_verified": ollama.get("weights_identity_verified") is True,
            "ollama_median_generation_tokens_per_second": ollama_rate,
            "kiln_rust_greedy_median_tokens_per_second": kiln_rate,
            "ollama_to_kiln_rate_ratio": ollama_rate / kiln_rate if isinstance(ollama_rate, (int, float)) and kiln_rate > 0 else None,
            "settings_match": {
                "prompt": ollama.get("prompt") == kiln_bench.get("prompt"),
                "generated_token_limit": ollama.get("tokens_requested") == kiln_bench.get("max_new_tokens"),
                "threads": ollama.get("threads") == kiln_bench.get("requested_threads"),
                "batch": ollama.get("batch") == kiln_bench.get("requested_batch"),
                "microbatch": ollama.get("microbatch") == kiln_bench.get("requested_microbatch"),
                "context": ollama.get("context") == profile.get("n_ctx"),
            },
            "paired_run_order": [sample.get("order") for sample in ollama.get("paired_kiln_samples", [])],
            "note": "Verify the Ollama model digest against the GGUF source hash and record matched target conditions before treating this as a pinned baseline.",
        }
    else:
        ollama_status_path = root / "ollama-status.txt"
        ollama_summary = {
            "captured": False,
            "status": ollama_status_path.read_text(encoding="utf-8").strip()
                if ollama_status_path.is_file() else "not_run",
        }

    gate_rate = best["decode_forward_tokens_per_second_median"]
    return {
        "status": "capture_complete_exit_gate_pending",
        "model_sha256": provenance["model_sha256"],
        "kiln_commit": provenance["kiln_commit"],
        "run_notes": "run-notes.txt (manual target, thermal, Ollama, profile interpretation and optimization evidence required)",
        "dense_matrix_configurations": dense,
        "best_dense_configuration": best,
        "dense_gate": {
            "target_decode_forward_tokens_per_second": 27.0,
            "observed_best_median": gate_rate,
            "rate_threshold_met": gate_rate >= 27.0,
            "tier0_machine_confirmed": False,
            "accepted": False,
            "note": "A measured rate alone is not Tier 0 acceptance; confirm Dell Latitude 7490 and required pinned model in run-notes.txt.",
        },
        "sampler_ab": sampler,
        "ngram_speculation_ab": speculation,
        "kv_reuse_ab": {
            "all_argmax_equal": True,
            "full_prefix_prefill_ms": kv.get("full_prefix_prefill_ms"),
            "cached_second_turn_prefill_ms": kv.get("cached_second_turn_prefill_ms"),
            "note": "Prefill-only comparison; does not establish decode throughput.",
        },
        "hot_path_profile": {
            "captured": profile_captured,
            "status": profile_status,
            "artifact": "profile/hotpath.txt" if profile_captured else None,
            "note": "Profiling is isolated from throughput trials; profile data is needed to identify the largest current costs.",
        },
        "ollama_baseline": ollama_summary,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
    except (IndexError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Phase 3.5 evidence error: {exc}", file=sys.stderr)
        raise SystemExit(2)
