#!/usr/bin/env bash
# Capture actual CPU MoE router selections and replay their expert-byte I/O.
# Usage: bash scripts/capture_moe_router_phase5a.sh MODEL.gguf EXPERT_BLOB INDEX.json [OUTPUT_DIR]
# Requires a built release kiln binary and kiln-bench. No model downloads or
# GGUF modifications; all outputs are diagnostic and never acceptance claims.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash $0 MODEL.gguf EXPERT_BLOB INDEX.json [OUTPUT_DIR]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$(realpath "$1")"
BLOB="$(realpath "$2")"
INDEX="$(realpath "$3")"
for FILE in "$MODEL" "$BLOB" "$INDEX"; do
  if [[ ! -f "$FILE" || ! -r "$FILE" ]]; then
    echo "Readable input required: $FILE" >&2
    exit 2
  fi
done
cd "$ROOT"
OUT="${4:-phase5a-real-router-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if [[ -e "$OUT/actual-router.tsv" ]]; then
  echo "Refusing to reuse existing router trace: $OUT/actual-router.tsv" >&2
  exit 2
fi
if [[ ! -x target/release/kiln || ! -x target/release/kiln-bench ]]; then
  echo "Build release executables first: cargo build --release -p kiln-cli -p kiln-bench" >&2
  exit 2
fi
{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kiln_commit=$(git rev-parse HEAD)"
  echo "llama_submodule=$(git submodule status vendor/llama.cpp)"
  echo "model=$MODEL"
  echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "expert_blob=$BLOB"
  echo "expert_blob_sha256=$(sha256sum "$BLOB" | awk '{print $1}')"
  echo "expert_index=$INDEX"
  echo "expert_index_sha256=$(sha256sum "$INDEX" | awk '{print $1}')"
  echo "kiln_exe_sha256=$(sha256sum target/release/kiln | awk '{print $1}')"
  echo "kernel=$(uname -a)"
  echo "hardware_cpu=$(grep -m1 'model name' /proc/cpuinfo || true)"
  echo "capture_mode=opt-in GGML CPU MAT_ID operator selections"
  echo "interpretation=operator accesses may repeat across expert projections; no token-unique-router inference"
} > "$OUT/provenance.txt"
# Explicitly use the pinned normal graph and exact native greedy selection.
# Environment variable is read only by the fork's experimental CPU hook.
KILN_MOE_ROUTER_TRACE="$OUT/actual-router.tsv" \
  target/release/kiln bench "$MODEL" --native-greedy --tokens 32 \
  --warmup 0 --runs 1 --threads 4 --batch 128 --ubatch 128 --json \
  > "$OUT/real-model-bench.json" 2> "$OUT/real-model-stderr.log"
if [[ ! -s "$OUT/actual-router.tsv" ]]; then
  echo "No CPU MUL_MAT_ID trace was generated. Check that the model is MoE," >&2
  echo "the pinned fork contains the trace hook and CPU path actually ran." >&2
  exit 1
fi
python3 scripts/convert_moe_router_trace.py "$OUT/actual-router.tsv" \
  --out "$OUT/actual-router.json" > "$OUT/router-conversion.json"
bash scripts/run_tier0_gate5a.sh "$BLOB" "$INDEX" "$OUT/io-evidence" "$OUT/actual-router.json"
echo "Actual CPU MoE route and indexed expert I/O captures: $OUT"
echo "NOTE: This DOES NOT prove streamed inference, cold SSD throughput, or Tier 0 acceptance."
