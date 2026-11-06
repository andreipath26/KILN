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


def summarize(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("phase_gate") != "5A-expert-block-io-feasibility":
            raise ValueError(f"{path}: not a Phase 5A I/O report")
        patterns = data.get("patterns")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"{path}: no access-pattern data")
        for item in patterns:
            if item.get("access_pattern") not in ("sequential", "random"):
                raise ValueError(f"{path}: unsupported access pattern")
            requested = item.get("total_bytes_read")
            rate = item.get("throughput_mib_s")
            if not isinstance(requested, int) or requested < 0:
                raise ValueError(f"{path}: invalid requested-byte count")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                raise ValueError(f"{path}: invalid throughput")
            attributed = item.get("linux_process_storage_read_bytes")
            if attributed is not None and (not isinstance(attributed, int) or attributed < 0):
                raise ValueError(f"{path}: invalid process I/O counter")
            rows.append({
                "artifact": str(path),
                "block_bytes": data["block_bytes"],
                "pattern": item["access_pattern"],
                "requested_bytes": requested,
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
        "caveat": "I/O probe results are page-cache uncontrolled and cannot prove cold-device speed; no real-model inference or memory/quality acceptance is inferred.",
    }
    if args.effective_ssd_mib_s is not None:
        out["traffic_budget"] = traffic_budget(
            args.effective_ssd_mib_s, args.observed_ssd_mib_per_token, args.target_tok_s
        )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
