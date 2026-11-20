#!/usr/bin/env python3
"""Convert opt-in KILN CPU MAT_ID router TSV events to expert-io JSON replay.

The trace records selected experts for individual MAT_ID operators. Version 2
traces also contain @decode markers, which this conversion skips while
preserving operator events. Distinct weight projections can repeat a route;
this tool preserves those events instead of guessing unique per-token labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 1_000_000
NAME_RE = re.compile(r"^blk\.(\d+)\.[A-Za-z0-9_.-]+$")


def convert_grouped(source: Path, output: Path, maximum: int = MAX_EVENTS) -> dict:
    """Group exact routed expert sets by marked decode, batch row and layer.

    This format supports chronological router-holdout evaluation. Legacy
    traces lack token boundaries and are deliberately rejected.
    """
    if maximum < 1 or maximum > MAX_EVENTS:
        raise ValueError("max events must be within 1..1,000,000")
    size = source.stat().st_size
    if size == 0 or size > MAX_TRACE_BYTES:
        raise ValueError("router event input must be nonempty and at most 256 MiB")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite grouped router input: {output}")
    grouped: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    decode_id = 0
    batch_tokens: int | None = None
    last_decode_id = 0
    raw_count = 0
    digest = hashlib.sha256()
    with source.open("rb") as incoming:
        for line_no, raw in enumerate(incoming, 1):
            digest.update(raw)
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as error:
                raise ValueError(f"line {line_no}: invalid UTF-8") from error
            parts = line.split("\t")
            if len(parts) == 3 and parts[0] == "@decode":
                if not all(value.isascii() and value.isdecimal() for value in parts[1:]):
                    raise ValueError(f"line {line_no}: malformed decode marker")
                decode_id, batch_tokens = map(int, parts[1:])
                if decode_id <= last_decode_id or batch_tokens <= 0:
                    raise ValueError(f"line {line_no}: decode IDs must increase and token count must be positive")
                last_decode_id = decode_id
                continue
            if len(parts) != 4:
                raise ValueError(f"line {line_no}: expected decode marker or four tab-separated fields")
            if batch_tokens is None:
                raise ValueError("grouped predictor evaluation requires decode-marked router traces")
            layer_text, expert_text, row_text, tensor = parts
            name = NAME_RE.fullmatch(tensor)
            if not name:
                raise ValueError(f"line {line_no}: invalid/unsupported tensor name")
            if not all(value.isascii() and value.isdecimal()
                       for value in (layer_text, expert_text, row_text)):
                raise ValueError(f"line {line_no}: layer, expert and row must be nonnegative decimal integers")
            layer, expert, row = map(int, (layer_text, expert_text, row_text))
            if layer != int(name.group(1)) or layer > 65535 or expert > 65535 or row > 1_000_000:
                raise ValueError(f"line {line_no}: invalid or conflicting expert route")
            if row >= batch_tokens:
                raise ValueError(f"line {line_no}: row is outside the marked decode batch")
            grouped[(decode_id, row, layer)].add(expert)
            raw_count += 1
            if raw_count > maximum:
                raise ValueError("router trace exceeds max-events; capture a shorter inference run")
    if not grouped:
        raise ValueError("router trace contained no grouped route events")
    token_keys = sorted({(decode, row) for decode, row, _layer in grouped})
    positions = {key: position for position, key in enumerate(token_keys)}
    events = [
        {"token_position": positions[(decode, row)], "layer": layer,
         "active_experts": sorted(experts)}
        for (decode, row, layer), experts in sorted(grouped.items())
    ]
    if len(events) > maximum:
        raise ValueError("grouped router trace exceeds max-events")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"temporary grouped output already exists: {temp}")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(events, stream, separators=(",", ":"))
            stream.write("\n")
        temp.rename(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    cardinality = Counter(len(event["active_experts"]) for event in events)
    return {
        "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": digest.hexdigest(), "decode_tokens": len(token_keys),
        "grouped_token_layer_events": len(events), "raw_operator_events": raw_count,
        "collapsed_projection_events": raw_count - len(grouped),
        "active_expert_set_cardinality": dict(sorted(cardinality.items())),
        "trace_granularity": "decode_call_batch_row_layer_active_expert_set",
        "interpretation": "Same-token tensor projection repeats are collapsed. Batch rows are ordered within each decode call; the resulting chronological split is an exploratory holdout, not proof of a production predictor or prefetch benefit.",
    }


def convert(source: Path, output: Path, maximum: int = MAX_EVENTS) -> dict:
    if maximum < 1 or maximum > MAX_EVENTS:
        raise ValueError("max events must be within 1..1,000,000")
    size = source.stat().st_size
    if size == 0 or size > MAX_TRACE_BYTES:
        raise ValueError("router event input must be nonempty and at most 256 MiB")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing trace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    decode_markers = 0
    last_decode_id = 0
    batch_tokens: int | None = None
    name_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    # Write to a private temporary file, and remove it if validation fails.
    temp = output.with_name(output.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"temporary trace output already exists: {temp}")
    try:
        with source.open("rb") as incoming, temp.open("x", encoding="utf-8") as target:
            target.write("[\n")
            first = True
            for line_no, raw in enumerate(incoming, 1):
                digest.update(raw)
                try:
                    line = raw.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as error:
                    raise ValueError(f"line {line_no}: invalid UTF-8") from error
                parts = line.split("\t")
                if len(parts) == 3 and parts[0] == "@decode":
                    marker_values = parts[1:]
                    if not all(value.isascii() and value.isdecimal() for value in marker_values):
                        raise ValueError(f"line {line_no}: malformed decode marker")
                    decode_id, batch_tokens = map(int, marker_values)
                    if decode_id <= last_decode_id or batch_tokens <= 0:
                        raise ValueError(f"line {line_no}: decode IDs must increase and token count must be positive")
                    last_decode_id = decode_id
                    decode_markers += 1
                    continue
                if len(parts) != 4:
                    raise ValueError(f"line {line_no}: expected decode marker or four tab-separated fields")
                layer_text, expert_text, row_text, tensor = parts
                name = NAME_RE.fullmatch(tensor)
                if not name:
                    raise ValueError(f"line {line_no}: invalid/unsupported tensor name")
                if not all(x.isascii() and x.isdecimal() for x in (layer_text, expert_text, row_text)):
                    raise ValueError(f"line {line_no}: layer, expert and row must be nonnegative decimal integers")
                layer, expert, row = map(int, (layer_text, expert_text, row_text))
                if layer != int(name.group(1)) or layer > 65535 or expert > 65535 or row > 1_000_000:
                    raise ValueError(f"line {line_no}: invalid or conflicting expert route")
                if batch_tokens is not None and row >= batch_tokens:
                    raise ValueError(f"line {line_no}: row is outside the marked decode batch")
                if count >= maximum:
                    raise ValueError("router trace exceeds max-events; capture a shorter inference run")
                if not first:
                    target.write(",\n")
                json.dump({"layer": layer, "expert_id": expert}, target, separators=(",", ":"))
                first = False
                count += 1
                name_counts[tensor] += 1
            target.write("\n]\n")
        if count == 0:
            raise ValueError("router trace contained no events")
        temp.rename(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return {
        "source": str(source), "source_bytes": size, "source_sha256": digest.hexdigest(),
        "output": str(output), "events": count, "decode_markers": decode_markers,
        "trace_granularity": "decode_call_row_operator" if decode_markers else "operator_event_legacy",
        "tensor_operator_counts": dict(sorted(name_counts.items())),
        "meaning": "Exact CPU MAT_ID expert operations, including repeated tensor projections. Decode markers are preserved in the source digest but omitted from this operator-only JSON. This is not unique per-token router labels or SSD reads.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="KILN_MOE_ROUTER_TRACE TSV file")
    parser.add_argument("--out", required=True, type=Path, help="new JSON file for kiln-bench expert-io --trace-file")
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    parser.add_argument("--grouped-out", type=Path,
                        help="also write decode-boundary grouped records for router-holdout; legacy traces cannot be grouped")
    args = parser.parse_args()
    try:
        report = convert(args.input, args.out, args.max_events)
        if args.grouped_out:
            report["grouped_holdout_input"] = convert_grouped(args.input, args.grouped_out, args.max_events)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
