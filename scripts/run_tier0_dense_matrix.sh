#!/usr/bin/env bash
# Pinned Phase 3.5 Dell baseline/tuning matrix. Requires a local readable GGUF.
# Usage: bash scripts/run_tier0_dense_matrix.sh model.gguf [output-directory]
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
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${2:-tier0-dense-$(date -u +%Y%m%dT%H%M%SZ)}"
TOKENS="${KILN_BENCH_TOKENS:-128}"
RUNS="${KILN_BENCH_RUNS:-5}"
WARMUP="${KILN_BENCH_WARMUP:-1}"
if [[ ! "$TOKENS" =~ ^[1-9][0-9]*$ || ! "$RUNS" =~ ^[1-9][0-9]*$ || ! "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "KILN_BENCH_TOKENS/RUNS must be positive integers; WARMUP must be a nonnegative integer" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kiln_commit=$(git rev-parse HEAD)"
  echo "git_dirty_files=$(git status --porcelain | wc -l)"
  echo "llama_cpp_submodule=$(git submodule status)"
  echo "kernel=$(uname -a)"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo || true)"
  echo "model_path=$MODEL"
  echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "model_size_bytes=$(stat -c %s "$MODEL")"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "NOTE=controlled KILN-only comparisons; Ollama baseline measured separately"
} > "$OUT/provenance.txt"

if [[ "${KILN_SKIP_BUILD:-0}" != 1 ]]; then
  cargo build --release -p kiln-cli > "$OUT/build.log" 2>&1
fi
for threads in 2 3 4; do
  for batch in 256 512 1024 2048; do
    label="t${threads}-b${batch}"
    /usr/bin/time -v -o "$OUT/time-${label}.txt" \
      target/release/kiln bench "$MODEL" \
      --prompt "The meaning of life is" --tokens "$TOKENS" \
      --warmup "$WARMUP" --runs "$RUNS" --threads "$threads" --batch "$batch" --json \
      > "$OUT/${label}.json"
    echo "completed $label"
  done
done
# Separate microbatch sweep: hold threads and n_batch constant, so the
# microbatch effect is not confounded with the larger thread/batch matrix.
for ubatch in 128 256 512; do
  label="t4-b512-u${ubatch}"
  /usr/bin/time -v -o "$OUT/time-${label}.txt" \
    target/release/kiln bench "$MODEL" \
    --prompt "The meaning of life is" --tokens "$TOKENS" \
    --warmup "$WARMUP" --runs "$RUNS" --threads 4 --batch 512 --ubatch "$ubatch" --json \
    > "$OUT/${label}.json"
  echo "completed $label"
done
echo "Finished Dell dense parameter matrix: $OUT"
echo "This does not measure Ollama parity, Qwen3 MoE streaming, or quality."
