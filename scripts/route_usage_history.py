#!/usr/bin/env python3
"""Model-keyed, bounded routing-history evidence for future adaptive expert admission.

Counts CPU MAT_ID projection events, NOT unique token-to-expert decisions.
It never changes routing or loads weights; admission policies may consume the
ranking only after separate holdout validation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

MAX_TRACE_BYTES = 64 * 1024 * 1024
MAX_PREVIOUS_BYTES = 16 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_KEYS = 100_000
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def _validated_history(payload: object, model_sha256: str) -> dict[str, int]:
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported routing history format")
    if payload.get("model_sha256") != model_sha256:
        raise ValueError("routing history belongs to another model")
    counts = payload.get("operator_event_heat")
    if not isinstance(counts, dict) or len(counts) > MAX_KEYS:
        raise ValueError("invalid routing-history score table")
    for key, score in counts.items():
        if (not isinstance(key, str) or not re.fullmatch(r"\d{1,5}:\d{1,5}", key)
                or any(int(part) > 65535 for part in key.split(":"))
                or type(score) is not int or score < 0 or score > MAX_EVENTS * 1000):
            raise ValueError("invalid routing-history entry")
    return counts


def admission_plan(scores: dict[str, int], sizes: dict[str, int],
                   capacity_bytes: int) -> dict:
    """Deterministic advisory RAM working set; never changes actual router keys."""
    if type(capacity_bytes) is not int or not 0 <= capacity_bytes <= (1 << 50):
        raise ValueError("invalid cache byte budget")
    if not isinstance(sizes, dict) or len(sizes) > MAX_KEYS:
        raise ValueError("invalid verified expert-size index")
    candidates = []
    for key, size in sizes.items():
        if (not isinstance(key, str) or not re.fullmatch(r"\d{1,5}:\d{1,5}", key)
                or any(int(part) > 65535 for part in key.split(":"))
                or type(size) is not int or not 0 < size <= (1 << 40)):
            raise ValueError("invalid expert size or key")
        score = scores.get(key, 0)
        if type(score) is not int or score < 0:
            raise ValueError("invalid expert heat")
        if score:
            candidates.append((key, score, size))
    # Rank by observed accesses per resident byte, then by total heat.
    # Integer comparisons avoid unstable floating-point tie ordering.
    from functools import cmp_to_key
    def compare(left, right):
        cross = left[1] * right[2] - right[1] * left[2]
        if cross != 0:
            return -1 if cross > 0 else 1
        if left[1] != right[1]:
            return -1 if left[1] > right[1] else 1
        return (left[0] > right[0]) - (left[0] < right[0])
    candidates.sort(key=cmp_to_key(compare))
    used = 0
    selected = []
    for key, score, size in candidates:
        if size <= capacity_bytes - used:
            layer, expert = map(int, key.split(":"))
            selected.append({"layer": layer, "expert_id": expert,
                             "score": score, "bytes": size})
            used += size
    return {"cache_budget_bytes": capacity_bytes, "selected_bytes": used,
            "selected_experts": selected, "actual_routing_modified": False,
            "runtime_cache_activated": False}


def build_history(trace: Path, output: Path, model_sha256: str,
                  previous: Path | None = None,
                  verified_sizes: dict[str, int] | None = None,
                  cache_budget_bytes: int | None = None) -> dict:
    if not SHA_RE.fullmatch(model_sha256):
        raise ValueError("model SHA256 must be 64 lowercase hexadecimal digits")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite history: {output}")
    if not 0 < trace.stat().st_size <= MAX_TRACE_BYTES:
        raise ValueError("trace is missing, empty or too large")
    with trace.open(encoding="utf-8") as source:
        events = json.load(source)
    if not isinstance(events, list) or not 0 < len(events) <= MAX_EVENTS:
        raise ValueError("invalid routing trace event count")
    counts: Counter[str] = Counter()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"layer", "expert_id"}:
            raise ValueError("invalid routing trace row")
        layer, expert = event["layer"], event["expert_id"]
        if (type(layer) is not int or type(expert) is not int
                or not 0 <= layer <= 65535 or not 0 <= expert <= 65535):
            raise ValueError("invalid layer/expert identifier")
        counts[f"{layer}:{expert}"] += 1
    if len(counts) > MAX_KEYS:
        raise ValueError("too many distinct routed experts")
    if previous is not None:
        if not 0 < previous.stat().st_size <= MAX_PREVIOUS_BYTES:
            raise ValueError("previous history is invalid or too large")
        prior = _validated_history(
            json.loads(previous.read_text(encoding="utf-8")), model_sha256)
        # Eight-generation exponential aging: old model behavior cannot
        # permanently starve newly active experts.
        for key, score in prior.items():
            counts[key] += (score * 7) // 8
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    report = {
        "version": 1,
        "model_sha256": model_sha256,
        "source_events": len(events),
        "score_type": "decayed_mat_id_projection_operator_events_not_unique_token_routes",
        "operator_event_heat": dict(sorted(counts.items())),
        "ranked_experts": [{"layer": int(key.split(":")[0]),
                            "expert_id": int(key.split(":")[1]), "score": score}
                           for key, score in ranked],
        "runtime_cache_activated": False,
    }
    # Validate the complete optional admission decision before creating any
    # output: a malformed size map must not leave a valid-looking heat artifact.
    if verified_sizes is not None:
        if cache_budget_bytes is None:
            raise ValueError("cache budget required with verified expert sizes")
        report["admission"] = admission_plan(counts, verified_sizes, cache_budget_bytes)
    elif cache_budget_bytes is not None:
        raise ValueError("verified expert sizes required with cache budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Routing history contains potentially sensitive workload information.
    # Open atomically with an owner-only mode, never follow an existing output.
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(report, target, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--verified-sizes", type=Path,
                        help="optional key-to-byte-size JSON map for advisory hot expert admission")
    parser.add_argument("--cache-budget-bytes", type=int,
                        help="bounded advisory cache budget for --verified-sizes")
    args = parser.parse_args()
    try:
        sizes = None
        if args.verified_sizes is not None:
            if args.cache_budget_bytes is None:
                raise ValueError("--cache-budget-bytes is required with --verified-sizes")
            if not 0 < args.verified_sizes.stat().st_size <= MAX_PREVIOUS_BYTES:
                raise ValueError("invalid expert size index")
            sizes = json.loads(args.verified_sizes.read_text(encoding="utf-8"))
        elif args.cache_budget_bytes is not None:
            raise ValueError("--verified-sizes is required with --cache-budget-bytes")
        report = build_history(args.trace, args.out, args.model_sha256, args.previous,
                               sizes, args.cache_budget_bytes)
        if "admission" in report:
            print(json.dumps(report["admission"], sort_keys=True))
            return
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"events": report["source_events"],
                      "distinct_experts": len(report["operator_event_heat"]),
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
