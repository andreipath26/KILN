#!/usr/bin/env python3
"""Model-keyed, bounded routing-history evidence for future adaptive expert admission.

Counts CPU MAT_ID projection events, NOT unique token-to-expert decisions.
It never changes routing or loads weights; admission policies may consume the
ranking only after separate holdout validation.
"""
from __future__ import annotations

import argparse
import json
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
                or type(score) is not int or score < 0 or score > MAX_EVENTS * 1000):
            raise ValueError("invalid routing-history entry")
    return counts


def build_history(trace: Path, output: Path, model_sha256: str,
                  previous: Path | None = None) -> dict:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    try:
        report = build_history(args.trace, args.out, args.model_sha256, args.previous)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"events": report["source_events"],
                      "distinct_experts": len(report["operator_event_heat"]),
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
