#!/usr/bin/env bash
# One-command Phase 3.5 capture: dense tuning, sampler A/B, speculation A/B,
# and KV reuse A/B, all tied to one model hash and repository revision.
# Usage: bash scripts/run_phase35_suite.sh model.gguf [output-directory]
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
OUT="${2:-phase35-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
BIN="$ROOT/target/release/kiln"
TOKENS="${KILN_BENCH_TOKENS:-256}"
RUNS="${KILN_BENCH_RUNS:-5}"
WARMUP="${KILN_BENCH_WARMUP:-1}"

{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "kiln_commit=$(git rev-parse HEAD)"
  echo "git_dirty_files=$(git status --porcelain | wc -l)"
  echo "llama_cpp_submodule=$(git submodule status)"
  echo "kernel=$(uname -a)"
  echo "cpu=$(grep -m1 'model name' /proc/cpuinfo || true)"
  echo "logical_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  echo "memory=$(grep -E '^(MemTotal|MemAvailable):' /proc/meminfo | tr '\n' ';')"
  echo "model_path=$MODEL"
  echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
  echo "model_size_bytes=$(stat -c %s "$MODEL")"
  echo "benchmark_tokens=$TOKENS"
  echo "measured_runs=$RUNS"
  echo "warmup_runs=$WARMUP"
  echo "power_thermal_and_page_cache=record manually in run-notes.txt"
  echo "ollama_baseline=optional API capture; verify imported GGUF identity in run-notes.txt"
} > "$OUT/provenance.txt"
cat > "$OUT/run-notes.txt" <<EOF
## Complete these after reviewing the capture; do not edit measured JSON.
tier0_machine=UNCONFIRMED
target_model_name=UNCONFIRMED
target_model_revision_or_source=UNCONFIRMED
ollama_version=NOT_RECORDED
ollama_imported_model_sha256=NOT_RECORDED
ollama_matched_prompt_and_output_length=NOT_RECORDED
power_mode_and_ac_state=NOT_RECORDED
thermal_stability_and_observation_duration=NOT_RECORDED
profile_largest_costs_and_evidence=NOT_RECORDED
retained_optimization_and_repeatable_gain=NOT_RECORDED
known_output_or_quality_differences=NOT_RECORDED
EOF

cargo build --release -p kiln-cli > "$OUT/build.log" 2>&1
KILN_SKIP_BUILD=1 KILN_BENCH_TOKENS="$TOKENS" KILN_BENCH_RUNS="$RUNS" \
  KILN_BENCH_WARMUP="$WARMUP" \
  bash scripts/run_tier0_dense_matrix.sh "$MODEL" "$OUT/dense"

# Hold the best-known matrix point fixed while isolating sampler overhead.
for mode in rust native; do
  args=(bench "$MODEL" --prompt "The meaning of life is" --tokens "$TOKENS"
    --warmup "$WARMUP" --runs "$RUNS" --threads 4 --batch 512 --ubatch 128 --json)
  if [[ "$mode" == native ]]; then args+=(--native-greedy); fi
  /usr/bin/time -v -o "$OUT/time-sampler-${mode}.txt" "$BIN" "${args[@]}" \
    > "$OUT/sampler-${mode}.json"
done

# Repetitive prompt data makes the n-gram hypothesis measurable; parity is
# checked by the command for every run and both rates are reported.
for draft in 4 8 16; do
  /usr/bin/time -v -o "$OUT/time-spec-d${draft}.txt" "$BIN" bench-spec "$MODEL" \
    --prompt "Repeat this short phrase: red blue red blue red blue." \
    --tokens "$TOKENS" --warmup "$WARMUP" --runs "$RUNS" \
    --threads 4 --batch 512 --ubatch 128 --ngram 3 --draft-max "$draft" --json \
    > "$OUT/spec-d${draft}.json"
done

/usr/bin/time -v -o "$OUT/time-kv.txt" "$BIN" bench-kv "$MODEL" \
  --runs "$RUNS" --threads 4 --batch 512 --ubatch 128 --json > "$OUT/kv.json"

if [[ -n "${KILN_OLLAMA_MODEL:-}" ]]; then
  python3 scripts/capture_phase35_ollama.py --gguf "$MODEL" \
    --kiln-bin "$BIN" --ubatch 128 --model "$KILN_OLLAMA_MODEL" \
    --tokens "$TOKENS" --runs "$RUNS" \
    --warmup "$WARMUP" --threads 4 --batch 512 --context 4096 \
    --output "$OUT/ollama.json"
else
  echo "pending: set KILN_OLLAMA_MODEL after importing the same GGUF" > "$OUT/ollama-status.txt"
fi

# Sampling profile runs separately from the throughput matrix. Restricted
# perf permissions are recorded as an explicit pending exit-gate item.
if KILN_SKIP_BUILD=1 KILN_PROFILE_TOKENS="$TOKENS" \
  bash scripts/profile_phase35_hotpath.sh "$MODEL" "$OUT/profile" \
  > "$OUT/profile-stdout.txt" 2> "$OUT/profile-stderr.txt"; then
  echo "completed" > "$OUT/profile-status.txt"
else
  echo "pending: Linux perf unavailable or blocked; see profile-stderr.txt" > "$OUT/profile-status.txt"
fi

python3 scripts/summarize_phase35.py "$OUT" > "$OUT/summary.json"
echo "Phase 3.5 capture complete: $OUT"
echo "summary: $OUT/summary.json"
