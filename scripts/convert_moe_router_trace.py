#!/usr/bin/env python3
"""Convert opt-in KILN CPU MAT_ID router TSV events to expert-io JSON replay.

The trace records selected experts for individual MAT_ID operators. Distinct
weight projections can repeat a token/expert selection; this tool preserves
those events instead of guessing unique per-token router labels. Only valid
standard blk.<layer>. tensor names are accepted; no model family is assumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MAX_TRACE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 1_000_000
NAME_RE = re.compile(r"^blk\.(\d+)\.[A-Za-z0-9_.-]+$")


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
                if len(parts) != 4:
                    raise ValueError(f"line {line_no}: expected four tab-separated fields")
                layer_text, expert_text, row_text, tensor = parts
                name = NAME_RE.fullmatch(tensor)
                if not name:
                    raise ValueError(f"line {line_no}: invalid/unsupported tensor name")
                if not all(x.isascii() and x.isdecimal() for x in (layer_text, expert_text, row_text)):
                    raise ValueError(f"line {line_no}: layer, expert and row must be nonnegative decimal integers")
                layer, expert, row = map(int, (layer_text, expert_text, row_text))
                if layer != int(name.group(1)) or layer > 65535 or expert > 65535 or row > 1_000_000:
                    raise ValueError(f"line {line_no}: invalid or conflicting expert route")
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
        "output": str(output), "events": count, "tensor_operator_counts": dict(sorted(name_counts.items())),
        "meaning": "Exact CPU MAT_ID expert operations, including repeated tensor projections. Not unique per-token router labels or SSD reads.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="KILN_MOE_ROUTER_TRACE TSV file")
    parser.add_argument("--out", required=True, type=Path, help="new JSON file for kiln-bench expert-io --trace-file")
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS)
    args = parser.parse_args()
    try:
        report = convert(args.input, args.out, args.max_events)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
