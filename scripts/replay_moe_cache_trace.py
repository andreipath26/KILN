#!/usr/bin/env python3
"""Replay a captured CPU MoE router trace through byte-bounded cache policies.

This is a deterministic trace simulation. It does not exercise KILN's runtime
cache or measure physical storage, inference speed, or target-hardware memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.capture_direct_gguf_expert_io import (
    load_index,
    load_trace,
    validate_index_report,
)

TENSOR_RE = re.compile(r"^blk\.(\d+)\.[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expert_sizes(entries: dict[tuple[str, int], tuple[int, int]]) -> dict[tuple[int, int], int]:
    sizes: dict[tuple[int, int], int] = {}
    for (tensor, expert), (_offset, length) in entries.items():
        match = TENSOR_RE.fullmatch(tensor)
        if not match:
            raise ValueError(f"invalid indexed tensor name: {tensor}")
        key = (int(match.group(1)), expert)
        sizes[key] = sizes.get(key, 0) + length
    return sizes


def route_requests(trace_path: Path, entries: dict[tuple[str, int], tuple[int, int]],
                   maximum: int) -> list[tuple[tuple[int, int], int]]:
    events = load_trace(trace_path, entries, maximum)
    requests = []
    for tensor, expert, _offset, tensor_bytes in events:
        match = TENSOR_RE.fullmatch(tensor)
        assert match is not None
        requests.append(((int(match.group(1)), expert), tensor_bytes))
    return requests


class ByteCache:
    """Deterministic variable-size cache policy model.

    Adaptive admission mirrors the KILN ExpertCache frequency aging,
    ghost-history score, and cold-scan rejection rule at a policy level.
    """
    def __init__(self, capacity: int, policy: str):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        if policy not in ("none", "lru", "lfu", "adaptive"):
            raise ValueError(f"unsupported cache policy: {policy}")
        self.capacity = capacity
        self.policy = policy
        self.entries: dict[tuple[int, int], dict[str, int]] = {}
        self.used = 0
        self.clock = 0
        self.ghost: dict[tuple[int, int], tuple[int, int]] = {}
        self.evictions = 0
        self.cold_rejections = 0

    def _tick(self) -> int:
        self.clock += 1
        if self.policy == "adaptive" and self.clock % 4096 == 0:
            for entry in self.entries.values():
                entry["frequency"] = max(1, entry["frequency"] // 2)
            self.ghost = {key: (max(1, score // 2), tick) for key, (score, tick) in self.ghost.items()}
        return self.clock

    def _admit(self, key: tuple[int, int], block_bytes: int) -> None:
        if self.policy == "none" or self.capacity == 0 or block_bytes <= 0 or block_bytes > self.capacity:
            return
        now = self._tick()
        score = 0
        if self.policy == "adaptive":
            if key not in self.ghost and len(self.ghost) >= 8192:
                oldest = min(self.ghost, key=lambda item: (self.ghost[item][1], item))
                del self.ghost[oldest]
            old_score, _ = self.ghost.get(key, (0, now))
            score = old_score + 1
            self.ghost[key] = (score, now)

        current_size = self.entries.get(key, {}).get("size", 0)
        needed = max(0, self.used - current_size - (self.capacity - block_bytes))
        candidates = [(candidate, entry) for candidate, entry in self.entries.items() if candidate != key]
        if self.policy == "lru":
            candidates.sort(key=lambda item: item[1]["last"])
        else:
            candidates.sort(key=lambda item: (item[1]["frequency"], item[1]["last"], item[0]))

        victims = []
        reclaimed = 0
        for candidate, entry in candidates:
            if reclaimed >= needed:
                break
            victims.append(candidate)
            reclaimed += entry["size"]
        if reclaimed < needed:
            return
        if self.policy == "adaptive" and needed:
            protected_heat = max((self.entries[victim]["frequency"] for victim in victims), default=0)
            if score < protected_heat:
                self.cold_rejections += 1
                return
        for victim in victims:
            removed = self.entries.pop(victim)
            self.used -= removed["size"]
            self.evictions += 1
        if key in self.entries:
            old = self.entries.pop(key)
            self.used -= old["size"]
        self.entries[key] = {
            "size": block_bytes,
            "last": now,
            "frequency": max(score, 1),
        }
        self.used += block_bytes

    def access(self, key: tuple[int, int], block_bytes: int) -> bool:
        now = self._tick()
        entry = self.entries.get(key)
        if entry is not None:
            entry["last"] = now
            entry["frequency"] += 1
            return True
        self._admit(key, block_bytes)
        return False


def simulate(requests: list[tuple[tuple[int, int], int]],
             sizes: dict[tuple[int, int], int], capacity: int, policy: str) -> dict:
    cache = ByteCache(capacity, policy)
    hits = misses = demand_bytes = hit_tensor_bytes = read_bytes = 0
    for key, tensor_bytes in requests:
        block_bytes = sizes[key]
        demand_bytes += tensor_bytes
        if cache.access(key, block_bytes):
            hits += 1
            hit_tensor_bytes += tensor_bytes
        else:
            misses += 1
            read_bytes += block_bytes
    count = len(requests)
    return {
        "policy": policy,
        "cache_capacity_bytes": capacity,
        "requests": count,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / count if count else 0.0,
        "logical_tensor_demand_bytes": demand_bytes,
        "modelled_expert_fill_bytes_on_miss": read_bytes,
        "logical_tensor_bytes_served_from_cache": hit_tensor_bytes,
        "resident_bytes_at_end": cache.used,
        "resident_experts_at_end": len(cache.entries),
        "evictions": cache.evictions,
        "adaptive_cold_rejections": cache.cold_rejections,
    }


def replay(model: Path, index: Path, report: Path, trace: Path,
           capacities: list[int], maximum_events: int) -> dict:
    model, index, report, trace = (path.resolve(strict=True) for path in (model, index, report, trace))
    if not all(path.is_file() for path in (model, index, report, trace)):
        raise ValueError("model, index, export report, and trace must be regular files")
    entries = load_index(index, model.stat().st_size)
    report_sha = validate_index_report(report, model, index, len(entries))
    requests = route_requests(trace, entries, maximum_events)
    sizes = expert_sizes(entries)
    results = []
    policies = ("none", "lru", "lfu", "adaptive")
    for capacity in capacities:
        for policy in policies:
            results.append(simulate(requests, sizes, capacity, policy))
    return {
        "status": "real_router_trace_byte_cache_simulation",
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "native_index_sha256": sha256_file(index),
        "native_index_report_sha256": report_sha,
        "router_trace_sha256": sha256_file(trace),
        "indexed_expert_count": len(sizes),
        "indexed_expert_bytes_total": sum(sizes.values()),
        "trace_events": len(requests),
        "trace_order_preserved": True,
        "cache_capacity_bytes_tested": capacities,
        "results": results,
        "interpretation": (
            "Uses raw CPU MAT_ID tensor/expert events in recorded order. A cache key is (layer, expert); "
            "the modeled fill size is the sum of that expert's indexed tensor slabs, while logical demand "
            "bytes count the tensor slab on each operator event. Trace files have no token-boundary field, "
            "so this is an operator-event replay and may count repeated projections within one token as hits. "
            "It simulates policy only: the live runtime cache is not connected to the native graph-selected "
            "MoE execution path. No physical SSD traffic, latency, overlap, inference throughput, RSS, or "
            "Tier 0 acceptance is measured."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="original single-file MoE GGUF")
    parser.add_argument("native_index", type=Path, help="verified native GGUF TSV expert index")
    parser.add_argument("index_report", type=Path, help="JSON report from native index export")
    parser.add_argument("router_trace", type=Path, help="raw KILN_MOE_ROUTER_TRACE TSV")
    parser.add_argument("--cache-mib", type=int, nargs="+", required=True,
                        help="one or more cache budgets in MiB; 0 disables retention")
    parser.add_argument("--max-events", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 0 or value > 1_048_576 for value in args.cache_mib):
        parser.error("--cache-mib values must be in 0..1048576")
    if not 1 <= args.max_events <= 1_000_000:
        parser.error("--max-events must be in 1..1000000")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = replay(args.model, args.native_index, args.index_report, args.router_trace,
                        [value * 1024 * 1024 for value in args.cache_mib], args.max_events)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "status": result["status"],
        "trace_events": result["trace_events"],
        "indexed_expert_count": result["indexed_expert_count"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
