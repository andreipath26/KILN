#!/usr/bin/env python3
"""Summarize Phase 5A expert-I/O probe artifacts without making a false go/no-go claim.

Usage:
  python3 scripts/assess_tier0_gate5a.py tier0-gate5a-*/expert-io-*.json
  python3 scripts/assess_tier0_gate5a.py artifact.json --effective-ssd-mib-s 800 --observed-ssd-mib-per-token 100 --target-tok-s 4

Probe throughput includes page-cache hits and is NOT automatically an SSD rate.
A traffic-budget calculation requires independent, explicitly supplied model/device measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return result


def nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def positive_int(value: object) -> bool:
    return nonnegative_int(value) and value > 0


def nonnegative_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def summarize(paths: list[Path]) -> list[dict]:
    rows = []
    capture_identity = None
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("phase_gate") != "5A-expert-block-io-feasibility":
            raise ValueError(f"{path}: not a Phase 5A I/O report")
        index_file = data.get("index_file")
        range_count = data.get("indexed_expert_ranges")
        if not isinstance(index_file, str) or not index_file.strip():
            raise ValueError(f"{path}: Gate 5A requires an indexed expert-range report")
        if not isinstance(range_count, int) or isinstance(range_count, bool) or range_count <= 0:
            raise ValueError(f"{path}: missing or invalid indexed_expert_ranges")
        model_file = data.get("file")
        model_size = data.get("file_size_bytes")
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValueError(f"{path}: missing model/blob path")
        if not isinstance(model_size, int) or isinstance(model_size, bool) or model_size <= 0:
            raise ValueError(f"{path}: missing or invalid model/blob size")
        requests_per_run = data.get("requests_per_run")
        repetitions = data.get("repetitions")
        if (not isinstance(requests_per_run, int) or isinstance(requests_per_run, bool) or requests_per_run <= 0
                or not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions <= 0):
            raise ValueError(f"{path}: invalid request/repetition counts")
        identity = (model_file, model_size, index_file, range_count)
        if capture_identity is None:
            capture_identity = identity
        elif identity != capture_identity:
            raise ValueError(f"{path}: model/index identity differs across block-size captures")
        block_bytes = data.get("block_bytes")
        if not positive_int(block_bytes):
            raise ValueError(f"{path}: invalid block_bytes")
        patterns = data.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"{path}: no access-pattern data")
        names = [item.get("access_pattern") for item in patterns]
        if sorted(names) not in (["random", "sequential"], ["router-trace"]):
            raise ValueError(f"{path}: expected sequential/random sampling or a single router-trace replay")
        for item in patterns:
            if item.get("access_pattern") not in ("sequential", "random", "router-trace"):
                raise ValueError(f"{path}: unsupported access pattern")
            requested = item.get("total_bytes_read")
            rate = item.get("throughput_mib_s")
            if not nonnegative_int(requested):
                raise ValueError(f"{path}: invalid requested-byte count")
            if not nonnegative_finite(rate):
                raise ValueError(f"{path}: invalid throughput")
            expert_keys = item.get("expert_keys")
            if (not isinstance(expert_keys, list) or not expert_keys
                    or any(not isinstance(key, dict)
                           or not nonnegative_int(key.get("layer"))
                           or not nonnegative_int(key.get("expert_id"))
                           for key in expert_keys)):
                raise ValueError(f"{path}: pattern does not identify sampled expert ranges")
            if len(expert_keys) != requests_per_run:
                raise ValueError(f"{path}: sampled expert-key count does not match requests_per_run")
            measured_requests = item.get("requests")
            if measured_requests != requests_per_run * repetitions:
                raise ValueError(f"{path}: measured request count does not match request/repetition counts")
            min_request = item.get("min_request_bytes")
            max_request = item.get("max_request_bytes")
            if (not positive_int(min_request) or not positive_int(max_request)
                    or max_request < min_request or max_request > block_bytes):
                raise ValueError(f"{path}: invalid actual expert-range request sizes")
            if not measured_requests * min_request <= requested <= measured_requests * max_request:
                raise ValueError(f"{path}: total bytes outside sampled request-size bounds")
            p50 = item.get("request_latency_p50_us")
            p95 = item.get("request_latency_p95_us")
            if not nonnegative_finite(p50) or not nonnegative_finite(p95) or p95 < p50:
                raise ValueError(f"{path}: invalid request latency percentiles")
            advised = item.get("page_cache_eviction_advised", False)
            if not isinstance(advised, bool):
                raise ValueError(f"{path}: invalid cache eviction advisory flag")
            attributed = item.get("linux_process_storage_read_bytes")
            if attributed is not None and not nonnegative_int(attributed):
                raise ValueError(f"{path}: invalid process I/O counter")
            rows.append({
                "artifact": str(path),
                "model_file": model_file,
                "model_size_bytes": model_size,
                "index_file": index_file,
                "indexed_expert_ranges": range_count,
                "block_bytes": block_bytes,
                "pattern": item["access_pattern"],
                "trace_driven": item["access_pattern"] == "router-trace",
                "page_cache_eviction_advised_not_verified_cold": advised,
                "sampled_expert_keys": expert_keys,
                "requested_bytes": requested,
                "min_request_bytes": min_request,
                "max_request_bytes": max_request,
                "process_attributed_read_bytes": attributed,
                "rate_mib_s_page_cache_uncontrolled": rate,
                "p50_us": item["request_latency_p50_us"],
                "p95_us": item["request_latency_p95_us"],
            })
    return rows


def traffic_budget(rate_mib_s: float, bytes_mib_per_token: float, target_tps: float) -> dict:
    allowed = rate_mib_s / target_tps
    required = bytes_mib_per_token * target_tps
    return {
        "measured_effective_ssd_mib_s_input": rate_mib_s,
        "observed_ssd_mib_per_token_input": bytes_mib_per_token,
        "target_tokens_per_second": target_tps,
        "maximum_average_ssd_mib_per_token_at_target": allowed,
        "required_average_ssd_mib_s": required,
        "traffic_budget_condition": "within_budget" if required <= rate_mib_s else "exceeds_budget",
        "interpretation": "Necessary sustained traffic condition only; not a sufficient throughput or accuracy test.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--effective-ssd-mib-s", type=positive)
    parser.add_argument("--observed-ssd-mib-per-token", type=positive)
    parser.add_argument("--target-tok-s", type=positive, default=4.0)
    args = parser.parse_args()
    if (args.effective_ssd_mib_s is None) != (args.observed_ssd_mib_per_token is None):
        parser.error("supply both effective SSD MiB/s and observed SSD MiB/token, or neither")
    try:
        records = summarize(args.reports)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    out = {
        "status": "measurement_summary_not_acceptance",
        "probe_cache_controlled": False,
        "records": records,
        "traffic_budget": None,
        "caveat": "I/O probe results use indexed expert ranges but remain page-cache uncontrolled; they do not prove cold-device speed or real-model inference, memory, or quality acceptance.",
    }
    if args.effective_ssd_mib_s is not None:
        out["traffic_budget"] = traffic_budget(
            args.effective_ssd_mib_s, args.observed_ssd_mib_per_token, args.target_tok_s
        )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
