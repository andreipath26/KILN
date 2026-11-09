#!/usr/bin/env bash
# Tier 0 Phase 5A capture over validated, mapped expert byte ranges.
# Usage: bash scripts/run_tier0_gate5a.sh /path/to/expert-blob.bin /path/to/experts.index.json [output-directory] [router-trace.json]
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash $0 /path/to/expert-blob.bin /path/to/experts.index.json [output-directory] [router-trace.json]" >&2
  exit 2
fi
MODEL="$1"
INDEX="$2"
if [[ ! -f "$MODEL" || ! -r "$MODEL" ]]; then
  echo "Readable expert blob/model file is required: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$INDEX" || ! -r "$INDEX" ]]; then
  echo "Readable validated ExpertTable index is required: $INDEX" >&2
  exit 2
fi
MODEL="$(realpath "$MODEL")"
INDEX="$(realpath "$INDEX")"
TRACE="${4:-}"
if [[ -n "$TRACE" ]]; then
  if [[ ! -f "$TRACE" || ! -r "$TRACE" ]]; then
    echo "Readable router trace is required: $TRACE" >&2
    exit 2
  fi
  TRACE="$(realpath "$TRACE")"
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${3:-tier0-gate5a-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
{
 echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
 echo "kiln_commit=$(git rev-parse HEAD)"
 echo "kiln_dirty=$(git status --porcelain | wc -l)"
 echo "submodule=$(git submodule status)"
 echo "kernel=$(uname -a)"
 echo "arch=$(uname -m)"
 echo "model_path=$MODEL"
 echo "model_size_bytes=$(stat -c %s "$MODEL")"
 echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
 echo "expert_index_path=$INDEX"
 echo "expert_index_size_bytes=$(stat -c %s "$INDEX")"
 echo "expert_index_sha256=$(sha256sum "$INDEX" | awk '{print $1}')"
 if [[ -n "$TRACE" ]]; then
   echo "router_trace_path=$TRACE"
   echo "router_trace_sha256=$(sha256sum "$TRACE" | awk '{print $1}')"
 fi
 echo "cargo=$(cargo --version)"
 echo "rustc=$(rustc --version)"
 echo "cpuinfo=$(grep -m1 'model name' /proc/cpuinfo || true)"
 echo "meminfo=$(grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo | tr '\n' ';')"
 echo "page_cache=uncontrolled; repeated reads may be warm"
 echo "page_cache_eviction_advice=linux POSIX_FADV_DONTNEED best-effort; not proof of cold SSD reads"
} > "$OUT/provenance.txt"
cargo build --release -p kiln-bench > "$OUT/build.log" 2>&1
for BLOCK in 65536 262144 1048576 4194304; do
  # The CLI chooses actual indexed expert ranges and clips each read to that
  # expert's length. All probes are read-only; time records max resident set size.
  /usr/bin/time -v -o "$OUT/rss-$BLOCK.txt" \
    target/release/kiln-bench expert-io \
    --file "$MODEL" --index-file "$INDEX" --block-bytes "$BLOCK" --requests 256 \
    --runs 5 --seed 42 --json > "$OUT/expert-io-$BLOCK.json"
  # Capture a second read-only probe with file-local eviction advice.
  # OS may retain/repopulate pages, so this is NOT a guaranteed cold run.
  /usr/bin/time -v -o "$OUT/rss-advised-$BLOCK.txt" \
    target/release/kiln-bench expert-io \
    --file "$MODEL" --index-file "$INDEX" --block-bytes "$BLOCK" --requests 256 \
    --runs 1 --seed 42 --advise-drop-cache --json > "$OUT/expert-io-advised-$BLOCK.json"
done
if [[ -n "$TRACE" ]]; then
  # Real router-event order; repeated experts are not discarded or shuffled.
  # This remains buffered, page-cache-influenced I/O, NOT end-to-end model speed.
  /usr/bin/time -v -o "$OUT/rss-router-trace.txt" \
    target/release/kiln-bench expert-io \
    --file "$MODEL" --index-file "$INDEX" --trace-file "$TRACE" \
    --block-bytes 4194304 --runs 1 --json > "$OUT/expert-io-router-trace.json"
  /usr/bin/time -v -o "$OUT/rss-router-trace-advised.txt" \
    target/release/kiln-bench expert-io \
    --file "$MODEL" --index-file "$INDEX" --trace-file "$TRACE" \
    --block-bytes 4194304 --runs 1 --advise-drop-cache --json > "$OUT/expert-io-router-trace-advised.json"
fi
# The report deliberately does not infer SSD throughput from page-cache-influenced reads.
python3 scripts/assess_tier0_gate5a.py "$OUT"/expert-io-*.json > "$OUT/summary.json"
echo "Phase 5A evidence captured under $OUT"
echo "IMPORTANT: This is a buffered page-cache-uncontrolled probe, NOT a cold SSD benchmark."
