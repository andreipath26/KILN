#!/usr/bin/env python3
"""Deterministic synthetic MoE expert-cache benchmark; not a hardware benchmark."""
import argparse
import json
import random
import time
from collections import OrderedDict, Counter


WORKLOADS = ("phase-shift", "stationary-hot", "uniform", "round-robin")


def make_trace(experts=128, requests=100_000, seed=42, workload="phase-shift"):
    """Build deterministic synthetic traces for distinct cache-reuse regimes."""
    if experts < 2:
        raise ValueError("experts must be >= 2")
    if requests < 1:
        raise ValueError("requests must be >= 1")
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; expected one of {WORKLOADS}")

    rng = random.Random(seed)
    trace = []
    half = requests // 2
    hot_count = max(1, experts // 8)
    for i in range(requests):
        if workload == "uniform":
            trace.append(rng.randrange(experts))
        elif workload == "round-robin":
            trace.append(i % experts)
        else:
            phase = 0 if workload == "stationary-hot" or i < half else 1
            hot_start = phase * (experts // 2)
            if rng.random() < 0.85:
                trace.append(hot_start + rng.randrange(hot_count))
            else:
                trace.append(rng.randrange(experts))
    return trace


class Cache:
    def __init__(self, capacity, policy):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if policy not in ("none", "lru", "lfu", "adaptive"):
            raise ValueError(f"unsupported policy: {policy}")
        self.capacity, self.policy = capacity, policy
        self.items = OrderedDict()
        self.freq = Counter()
        self.clock = 0

    def access(self, key):
        self.clock += 1
        if self.policy == "adaptive" and self.clock % 4096 == 0:
            # Age frequency estimates against long-running workload drift.
            for k in list(self.freq):
                self.freq[k] = max(1, self.freq[k] // 2)

        if key in self.items:
            self.freq[key] += 1
            # Keep last-access time for LFU's recency tie-break. Updating an
            # OrderedDict value does not change its order; LRU explicitly moves it.
            self.items[key] = self.clock
            if self.policy in ("lru", "adaptive"):
                self.items.move_to_end(key)
            return True
        if self.capacity <= 0 or self.policy == "none":
            return False
        if len(self.items) >= self.capacity:
            if self.policy == "lru":
                victim = next(iter(self.items))
            else:
                # LFU with last-access recency tie-break; adaptive uses aged counts.
                victim = min(self.items, key=lambda k: (self.freq[k], self.items[k]))
            del self.items[victim]
            self.freq.pop(victim, None)
        self.items[key] = self.clock
        self.freq[key] = 1
        return False


def simulate(trace, capacity, policy, expert_bytes, read_latency_us, bandwidth_gib_s):
    cache = Cache(capacity, policy)
    hits = 0
    t0 = time.perf_counter()
    for key in trace:
        if cache.access(key):
            hits += 1
    elapsed = time.perf_counter() - t0
    misses = len(trace) - hits
    bytes_read = misses * expert_bytes
    # Simplified serialized storage model; not a physical storage measurement.
    transfer_s = bytes_read / (bandwidth_gib_s * (1024**3))
    modelled_io_s = misses * read_latency_us / 1_000_000 + transfer_s
    return {
        "policy": policy,
        "cache_capacity_experts": capacity,
        "requests": len(trace),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / len(trace), 6),
        "expert_bytes": expert_bytes,
        "estimated_storage_bytes_read": bytes_read,
        "modelled_serial_io_seconds": round(modelled_io_s, 6),
        "simulator_wall_seconds": round(elapsed, 6),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--requests", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workload", choices=WORKLOADS, default="phase-shift",
                   help="synthetic reuse regime; none is a real model trace")
    p.add_argument("--expert-mib", type=float, default=16.0)
    p.add_argument("--read-latency-us", type=float, default=100.0)
    p.add_argument("--bandwidth-gib-s", type=float, default=1.5)
    p.add_argument("--capacities", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    args = p.parse_args()
    if (args.experts < 2 or args.requests < 1 or args.expert_mib <= 0
            or args.read_latency_us < 0 or args.bandwidth_gib_s <= 0):
        p.error("experts >= 2, requests >= 1, expert-mib > 0, latency >= 0, bandwidth > 0 required")
    if any(c < 0 or c > args.experts for c in args.capacities):
        p.error("cache capacities must be between 0 and number of experts")
    trace = make_trace(args.experts, args.requests, args.seed, args.workload)
    rows = []
    for capacity in args.capacities:
        policies = ["none"] if capacity == 0 else ["lru", "lfu", "adaptive"]
        for policy in policies:
            rows.append(simulate(
                trace, capacity, policy, int(args.expert_mib * 1024**2),
                args.read_latency_us, args.bandwidth_gib_s
            ))
    print(json.dumps({
        "benchmark": "synthetic-moe-expert-cache",
        "synthetic_only": True,
        "seed": args.seed,
        "experts": args.experts,
        "requests": args.requests,
        "workload": args.workload,
        "trace_model": {
            "phase-shift": "two phases with distinct hot sets; 85% hot-set and 15% uniform access",
            "stationary-hot": "one persistent hot set; 85% hot-set and 15% uniform access",
            "uniform": "uniform random access over all experts",
            "round-robin": "cyclic access over every expert"
        }[args.workload],
        "assumptions": {
            "expert_mib": args.expert_mib,
            "read_latency_us_per_miss": args.read_latency_us,
            "storage_bandwidth_gib_s": args.bandwidth_gib_s,
            "io_model": "serialized fixed latency per miss plus bytes/bandwidth; excludes compute, overlap, OS cache and real storage behaviour"
        },
        "results": rows,
        "warning": "Synthetic trace and assumed storage characteristics only. Not measured KILN performance or evidence of target Dell throughput."
    }, indent=2))


if __name__ == "__main__":
    main()
