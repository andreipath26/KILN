#!/usr/bin/env bash
# Tier 0 Phase 5A evidence capture. No root, no cache dropping, no model download.
# Usage: bash scripts/run_tier0_gate5a.sh /path/to/representative-experts.bin [output-directory]
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash $0 /path/to/representative-experts.bin [output-directory]" >&2
  exit 2
fi
MODEL="$1"
if [[ ! -f "$MODEL" || ! -r "$MODEL" ]]; then
  echo "Readable representative file is required: $MODEL" >&2
  exit 2
fi
MODEL="$(realpath "$MODEL")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${2:-tier0-gate5a-$(date -u +%Y%m%dT%H%M%SZ)}"
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
 echo "cargo=$(cargo --version)"
 echo "rustc=$(rustc --version)"
 echo "cpuinfo=$(grep -m1 'model name' /proc/cpuinfo || true)"
 echo "meminfo=$(grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo | tr '\n' ';')"
 echo "page_cache=uncontrolled; repeated reads may be warm"
} > "$OUT/provenance.txt"
cargo build --release -p kiln-bench > "$OUT/build.log" 2>&1
for BLOCK in 65536 262144 1048576 4194304; do
  if [[ "$(stat -c %s "$MODEL")" -lt "$BLOCK" ]]; then continue; fi
  # All probes are read-only; time records max resident set size.
  /usr/bin/time -v -o "$OUT/rss-$BLOCK.txt" \
    target/release/kiln-bench expert-io \
    --file "$MODEL" --block-bytes "$BLOCK" --requests 256 \
    --runs 5 --seed 42 --json > "$OUT/expert-io-$BLOCK.json"
done
# The report deliberately does not infer SSD throughput from page-cache-influenced reads.
python3 scripts/assess_tier0_gate5a.py "$OUT"/expert-io-*.json > "$OUT/summary.json"
echo "Phase 5A evidence captured under $OUT"
echo "IMPORTANT: This is a buffered page-cache-uncontrolled probe, NOT a cold SSD benchmark."
