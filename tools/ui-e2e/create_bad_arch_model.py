#!/usr/bin/env python3
"""Make a structurally valid GGUF that the inference backend must reject."""

import struct
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
data = bytearray(source.read_bytes())
if data[:4] != b"GGUF":
    raise SystemExit("smoke model is not GGUF")

version, _tensor_count, metadata_count = struct.unpack_from("<IQQ", data, 4)
if version != 3:
    raise SystemExit(f"expected GGUF v3, found v{version}")

position = 24
fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
for _ in range(metadata_count):
    key_size = struct.unpack_from("<Q", data, position)[0]
    position += 8
    key = data[position:position + key_size].decode("utf-8")
    position += key_size
    value_type = struct.unpack_from("<I", data, position)[0]
    position += 4

    if key == "general.architecture":
        if value_type != 8:
            raise SystemExit("general.architecture is not a GGUF string")
        value_size = struct.unpack_from("<Q", data, position)[0]
        position += 8
        replacement = b"zzzzz"
        if value_size != len(replacement):
            raise SystemExit(f"unexpected architecture string length: {value_size}")
        data[position:position + value_size] = replacement
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print(f"created backend-rejected GGUF at {destination}")
        break

    if value_type == 8:
        string_size = struct.unpack_from("<Q", data, position)[0]
        position += 8 + string_size
    elif value_type in fixed_sizes:
        position += fixed_sizes[value_type]
    else:
        raise SystemExit(f"unexpected metadata type {value_type} before general.architecture")
else:
    raise SystemExit("model has no general.architecture metadata")
