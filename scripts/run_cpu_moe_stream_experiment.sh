#!/usr/bin/env bash
# Phase 5: real CPU MoE inference against externally packed expert bytes.
# Explicit GGUF tensor/expert-axis selections are required; never auto-guess.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <model.gguf> <explicit-fused-selections.json> <new-output-dir>" >&2
  exit 2
fi

MODEL="$(realpath "$1")"
SELECTIONS="$(realpath "$2")"
OUT="$3"
KILN_BIN="${KILN_BIN:-target/debug/kiln}"
KILN_BENCH_BIN="${KILN_BENCH_BIN:-target/debug/kiln-bench}"
THREADS="${KILN_CPU_THREADS:-4}"
TOKENS="${KILN_STREAM_TEST_TOKENS:-32}"

[[ -f "$MODEL" && -s "$MODEL" && -s "$SELECTIONS" ]] || {
  echo "model and explicit tensor selections must be readable nonempty files" >&2; exit 2;
}
[[ -x "$KILN_BIN" && -x "$KILN_BENCH_BIN" ]] || {
  echo "missing kiln/kiln-bench executable: cargo build -p kiln-cli -p kiln-bench" >&2; exit 2;
}
[[ "$THREADS" =~ ^[1-9][0-9]*$ && "$TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CPU thread and token counts must be positive integers" >&2; exit 2;
}
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
for required in expert.bin expert.index.json expert.integrity.json expert.components.json \
                expert.stream.tsv baseline.json external.json external.audit.tsv proof.json; do
  [[ ! -e "$OUT/$required" ]] || {
    echo "output artifact already exists (refusing overwrite): $OUT/$required" >&2; exit 2;
  }
done

"$KILN_BENCH_BIN" expert-pack-fused \
  --source-gguf "$MODEL" --selections-file "$SELECTIONS" \
  --blob-file "$OUT/expert.bin" \
  --index-file "$OUT/expert.index.json" \
  --integrity-file "$OUT/expert.integrity.json" \
  --components-file "$OUT/expert.components.json" \
  --json > "$OUT/packed.json"
"$KILN_BENCH_BIN" expert-export-stream-index \
  --blob-file "$OUT/expert.bin" --index-file "$OUT/expert.index.json" \
  --integrity-file "$OUT/expert.integrity.json" \
  --components-file "$OUT/expert.components.json" \
  --output-file "$OUT/expert.stream.tsv" --json > "$OUT/exported.json"

"$KILN_BIN" bench "$MODEL" --native-greedy --tokens "$TOKENS" \
  --warmup 0 --runs 1 --threads "$THREADS" --batch 128 --ubatch 64 --json \
  > "$OUT/baseline.json"
KILN_MOE_STREAM_BLOB="$OUT/expert.bin" \
KILN_MOE_STREAM_INDEX="$OUT/expert.stream.tsv" \
KILN_MOE_STREAM_AUDIT="$OUT/external.audit.tsv" \
  "$KILN_BIN" bench "$MODEL" --native-greedy --tokens "$TOKENS" \
    --warmup 0 --runs 1 --threads "$THREADS" --batch 128 --ubatch 64 --json \
    > "$OUT/external.json"

python3 - "$OUT" "$SELECTIONS" "$MODEL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

out, selection_file, model_file = map(Path, sys.argv[1:])
baseline = json.loads((out / "baseline.json").read_text(encoding="utf-8"))
external = json.loads((out / "external.json").read_text(encoding="utf-8"))
base_run, ext_run = baseline["runs"][0], external["runs"][0]
token_equal = (
    base_run["token_fingerprint_fnv1a64"] == ext_run["token_fingerprint_fnv1a64"]
    and base_run["generated_tokens"] == ext_run["generated_tokens"]
)
selected_names = {entry["tensor_name"] for entry in json.loads(selection_file.read_text())}
audit = []
for line in (out / "external.audit.tsv").read_text(encoding="utf-8").splitlines():
    name, requested, loaded, payload = line.split("\t")
    audit.append({"tensor_name": name, "requested": int(requested),
                  "loaded": int(loaded), "bytes": int(payload)})
covered = {
    name for name in selected_names
    if any(row["tensor_name"] == name and row["requested"] > 0
           and row["loaded"] == row["requested"] and row["bytes"] > 0
           for row in audit)
}
all_reads_correct = all(
    row["loaded"] == row["requested"]
    for row in audit if row["tensor_name"] in selected_names and row["requested"] > 0
)
proof = {
    "phase": "experimental_actual_cpu_moe_sidecar_matmul",
    "model_path": str(model_file),
    "model_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
    "baseline_external_tokens_identical": token_equal,
    "explicit_tensors": len(selected_names),
    "sidecar_matmul_tensors_covered": len(covered),
    "selected_expert_io_fully_served": all_reads_correct,
    "external_bytes_read": sum(row["bytes"] for row in audit),
    "external_matmul_ops": sum(row["loaded"] > 0 for row in audit),
    "resident_model_still_mapped": True,
    "tier0_deployment_accepted": False,
}
(out / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof, indent=2))
if not token_equal or covered != selected_names or not all_reads_correct:
    raise SystemExit("experimental real-MoE external compute parity/coverage gate failed")
PY
