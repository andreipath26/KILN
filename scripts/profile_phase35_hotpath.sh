#!/usr/bin/env bash
# Capture a separate Linux perf sampling profile for the Phase 3.5 dense path.
# Usage: bash scripts/profile_phase35_hotpath.sh model.gguf [output-directory]
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 model.gguf [output-directory]" >&2
  exit 2
fi
MODEL="$(realpath "$1")"
if [[ ! -r "$MODEL" || ! -f "$MODEL" ]]; then
  echo "Provide an existing, readable GGUF file: $MODEL" >&2
  exit 2
fi
if ! command -v perf >/dev/null 2>&1; then
  echo "Linux perf is required for hot-path profiling" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${2:-phase35-profile-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
BIN="$ROOT/target/release/kiln"
TOKENS="${KILN_PROFILE_TOKENS:-512}"
FREQUENCY="${KILN_PROFILE_FREQUENCY:-99}"
if [[ ! "$TOKENS" =~ ^[1-9][0-9]*$ || ! "$FREQUENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "KILN_PROFILE_TOKENS and KILN_PROFILE_FREQUENCY must be positive integers" >&2
  exit 2
fi
if [[ "${KILN_SKIP_BUILD:-0}" != 1 ]]; then
  cargo build --release -p kiln-cli > "$OUT/build.log" 2>&1
fi
{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kiln_commit=$(git rev-parse HEAD)"
  echo "llama_cpp_submodule=$(git submodule status)"
  echo "model_path=$MODEL"
  echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo || true)"
  echo "kernel=$(uname -a)"
  echo "perf_version=$(perf --version)"
  echo "sampling_frequency_hz=$FREQUENCY"
  echo "tokens=$TOKENS"
  echo "threads=4 batch=512 ubatch=128"
  echo "sampling_path=rust_logits_greedy"
  echo "warning=Sampling instrumentation overhead applies; do not compare profile-run throughput with unprofiled suite results."
} > "$OUT/provenance.txt"

perf record -F "$FREQUENCY" --call-graph dwarf -o "$OUT/perf.data" -- \
  "$BIN" bench "$MODEL" --prompt "The meaning of life is" --tokens "$TOKENS" \
  --warmup 1 --runs 1 --threads 4 --batch 512 --ubatch 128 --json \
  > "$OUT/bench.json" 2> "$OUT/perf-record.log"
perf report --stdio --no-children --sort symbol --percent-limit 0.5 \
  -i "$OUT/perf.data" > "$OUT/hotpath.txt"
if ! rg -q 'Samples:|# Samples:' "$OUT/hotpath.txt"; then
  echo "perf report contains no sample data; check kernel perf permissions and profile output" >&2
  exit 1
fi
echo "Hot-path profile captured: $OUT"
