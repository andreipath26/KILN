#!/usr/bin/env python3
"""Replay CPU expert tensor events as bounded reads from the original GGUF.

The index must come from kiln-bench expert-export-gguf-stream-index and the
raw trace from KILN_MOE_ROUTER_TRACE, both for the same single-file model.
Reads preserve operator order and repeats. OS page cache is uncontrolled;
this is diagnostic I/O evidence, not an inference or acceptance benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import statistics
import time
from collections import Counter
from pathlib import Path

MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_RANGE_BYTES = 64 * 1024 * 1024
TENSOR_RE = re.compile(r"^blk\.(\d+)\.[A-Za-z0-9_.-]+$")


def load_index(path: Path, source_size: int) -> dict[tuple[str, int], tuple[int, int]]:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("native index exceeds 256 MiB")
    entries: dict[tuple[str, int], tuple[int, int]] = {}
    spans = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "KILN-MOE-STREAM-1":
            raise ValueError("index must use the native KILN-MOE-STREAM-1 format")
        for number, line in enumerate(stream, 2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"index line {number}: expected tensor, expert, offset, length")
            tensor, expert_text, offset_text, length_text = fields
            match = TENSOR_RE.fullmatch(tensor)
            values = (expert_text, offset_text, length_text)
            if not match or not all(x.isascii() and x.isdecimal() for x in values):
                raise ValueError(f"index line {number}: invalid tensor name or numeric field")
            expert, offset, length = map(int, values)
            end = offset + length
            if length <= 0 or length > MAX_RANGE_BYTES or end > source_size:
                raise ValueError(f"index line {number}: empty, oversized, or out-of-file byte range")
            key = (tensor, expert)
            if key in entries:
                raise ValueError(f"index line {number}: duplicate tensor/expert mapping")
            entries[key] = (offset, length)
            spans.append((offset, end))
    if not entries:
        raise ValueError("native index contains no expert mappings")
    spans.sort()
    if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
        raise ValueError("native expert byte ranges overlap")
    return entries


def load_trace(path: Path, entries: dict[tuple[str, int], tuple[int, int]],
               maximum: int) -> list[tuple[str, int, int, int]]:
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ValueError("router trace must be nonempty and at most 256 MiB")
    events = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"trace line {number}: expected layer, expert, row, tensor")
            layer_text, expert_text, row_text, tensor = fields
            match = TENSOR_RE.fullmatch(tensor)
            values = (layer_text, expert_text, row_text)
            if not match or not all(x.isascii() and x.isdecimal() for x in values):
                raise ValueError(f"trace line {number}: malformed operator event")
            layer, expert, _row = map(int, values)
            if layer != int(match.group(1)):
                raise ValueError(f"trace line {number}: layer does not match tensor name")
            key = (tensor, expert)
            if key not in entries:
                raise ValueError(f"trace line {number}: selected expert is absent from native index")
            offset, length = entries[key]
            events.append((tensor, expert, offset, length))
            if len(events) > maximum:
                raise ValueError(f"trace exceeds maximum of {maximum} events")
    if not events:
        raise ValueError("router trace contains no events")
    return events


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def capture(model: Path, index: Path, trace: Path, runs: int,
            maximum_events: int, maximum_bytes: int) -> dict:
    model, index, trace = (p.resolve(strict=True) for p in (model, index, trace))
    if not all(p.is_file() for p in (model, index, trace)):
        raise ValueError("model, index and trace must be regular files")
    source_stat = model.stat()
    entries = load_index(index, source_stat.st_size)
    trace_events = load_trace(trace, entries, maximum_events)

    replay = []
    replay_bytes = 0
    for event in trace_events:
        if replay_bytes + event[3] > maximum_bytes:
            break
        replay.append(event)
        replay_bytes += event[3]
    if not replay:
        raise ValueError("no complete expert event fits the configured byte budget")

    descriptor = os.open(model, os.O_RDONLY)
    run_reports = []
    try:
        opened_stat = os.fstat(descriptor)
        identity = (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
        opened_identity = (opened_stat.st_dev, opened_stat.st_ino, opened_stat.st_size, opened_stat.st_mtime_ns)
        if opened_identity != identity:
            raise ValueError("source GGUF changed before trace replay")
        for run_number in range(1, runs + 1):
            latency_us = []
            by_tensor: Counter[str] = Counter()
            started = time.perf_counter()
            bytes_read = 0
            for tensor, _expert, offset, length in replay:
                before = time.perf_counter()
                data = os.pread(descriptor, length, offset)
                if len(data) != length:
                    raise OSError(f"short read at byte {offset}: expected {length}, got {len(data)}")
                _ = data[0] + data[-1]
                latency_us.append((time.perf_counter() - before) * 1_000_000)
                by_tensor[tensor] += 1
                bytes_read += length
            elapsed = time.perf_counter() - started
            run_reports.append({
                "run": run_number,
                "events": len(replay),
                "bytes_read": bytes_read,
                "elapsed_seconds": elapsed,
                "mib_per_second": bytes_read / (1024 * 1024) / elapsed if elapsed else 0.0,
                "request_latency_us_p50": statistics.median(latency_us),
                "request_latency_us_p95": percentile(latency_us, 0.95),
                "events_by_tensor": dict(sorted(by_tensor.items())),
            })
        final_fd_stat = os.fstat(descriptor)
        final_identity = (final_fd_stat.st_dev, final_fd_stat.st_ino, final_fd_stat.st_size, final_fd_stat.st_mtime_ns)
        if final_identity != identity:
            raise ValueError("source GGUF changed during trace replay")
    finally:
        os.close(descriptor)

    source_after = model.stat()
    source_identity_after = (source_after.st_dev, source_after.st_ino, source_after.st_size, source_after.st_mtime_ns)
    source_identity_before = (source_stat.st_dev, source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns)
    if source_identity_after != source_identity_before:
        raise ValueError("source GGUF changed during direct-source trace replay")
    model_sha = sha256_file(model)
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    return {
        "status": "direct_source_trace_replay_diagnostic",
        "model_path": str(model),
        "model_size_bytes": source_stat.st_size,
        "model_sha256_after_replay": model_sha,
        "model_mtime_ns": source_stat.st_mtime_ns,
        "native_index_path": str(index),
        "native_index_sha256": index_sha,
        "router_trace_path": str(trace),
        "router_trace_sha256": trace_sha,
        "indexed_tensor_expert_ranges": len(entries),
        "trace_events_available": len(trace_events),
        "trace_events_replayed_per_run": len(replay),
        "trace_prefix_truncated_by_read_budget": len(replay) < len(trace_events),
        "read_byte_budget": maximum_bytes,
        "runs": run_reports,
        "median_mib_per_second": statistics.median(row["mib_per_second"] for row in run_reports),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "interpretation": (
            "Direct reads of the indexed original GGUF in recorded CPU MAT_ID operator order. "
            "The model SHA-256 is computed after timed reads to avoid prewarming the file. "
        "The trace-producing inference may have warmed OS page cache; cache state is uncontrolled. "
            "This does not measure physical SSD traffic, inference throughput, cache overlap, total system "
            "memory, model quality, or Tier 0 acceptance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="original single-file MoE GGUF")
    parser.add_argument("native_index", type=Path, help="verified native GGUF TSV index")
    parser.add_argument("router_trace", type=Path, help="raw KILN_MOE_ROUTER_TRACE TSV")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report path")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    parser.add_argument("--max-read-mib", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.runs <= 100:
        parser.error("--runs must be in 1..100")
    if not 1 <= args.max_events <= MAX_EVENTS:
        parser.error(f"--max-events must be in 1..{MAX_EVENTS}")
    if not 1 <= args.max_read_mib <= 16_384:
        parser.error("--max-read-mib must be in 1..16384")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = capture(args.model, args.native_index, args.router_trace,
                         args.runs, args.max_events, args.max_read_mib * 1024 * 1024)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    temporary = args.output.with_name(args.output.name + ".partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "events_available": report["trace_events_available"],
        "events_replayed": report["trace_events_replayed_per_run"],
        "median_mib_per_second": report["median_mib_per_second"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
