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
chmod 700 "$OUT"
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
# Retain model-keyed, provenance-bound routing heat for later measured
# cache-admission experiments; no persistent global state or auto-activation.
python3 scripts/route_usage_history.py --trace "$OUT/actual-router.json" \
  --model-sha256 "$(sha256sum "$MODEL" | awk '{print $1}')" \
  --out "$OUT/route-usage-history.json" > "$OUT/route-history-summary.json"
# Capture direct native-source I/O for supported single-file MoE GGUFs. The
# split-GGUF reader has its own bounded source-range probe and is not coerced
# into the single-file index format.
target/release/kiln plan "$MODEL" --json > "$OUT/model-plan.json"
IS_SPLIT="$(python3 - "$OUT/model-plan.json" <<'PY'
import json,sys
plan=json.load(open(sys.argv[1],encoding="utf-8"))
split=plan.get("split_gguf")
print("yes" if isinstance(split,dict) and split.get("shard_count",0)>1 else "no")
PY
)"
if [[ "$IS_SPLIT" == "yes" ]]; then
  target/release/kiln-bench expert-io-split \
    --source-gguf "$MODEL" --requests 64 --max-read-mib 256 --json \
    > "$OUT/direct-source-io.json"
  echo "split_gguf_bounded_source_order_sample; not router-trace replay or cold-SSD acceptance" \
    > "$OUT/direct-source-io-status.txt"
else
  target/release/kiln-bench expert-export-gguf-stream-index \
    --source-gguf "$MODEL" --auto-verified \
    --output-file "$OUT/direct-source-index.tsv" --json \
    > "$OUT/direct-source-index-report.json"
  python3 scripts/capture_direct_gguf_expert_io.py "$MODEL" \
    "$OUT/direct-source-index.tsv" "$OUT/direct-source-index-report.json" "$OUT/actual-router.tsv" \
    --max-read-mib 1024 \
    --output "$OUT/direct-source-io.json" \
    > "$OUT/direct-source-io-summary.json"
fi
bash scripts/run_tier0_gate5a.sh "$BLOB" "$INDEX" "$OUT/io-evidence" "$OUT/actual-router.json"
echo "Actual CPU MoE route and indexed expert I/O captures: $OUT"
echo "NOTE: This DOES NOT prove streamed inference, cold SSD throughput, or Tier 0 acceptance."
