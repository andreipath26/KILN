#!/usr/bin/env bash
# Linux release: one KILN binary, no bundled weights, portable compile flags.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo "Linux only" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for cmd in cmake cargo git tar sha256sum strings python3; do
  command -v "$cmd" >/dev/null || { echo "Missing: $cmd" >&2; exit 2; }
done
[[ -f vendor/llama.cpp/CMakeLists.txt ]] || {
  echo "Run git submodule update --init --recursive" >&2; exit 2;
}
for name in CFLAGS CXXFLAGS RUSTFLAGS; do
  value="${!name:-}"
  if [[ "$value" == *"-march=native"* || "$value" == *"target-cpu=native"* ]]; then
    echo "Refusing host-specific compilation: $name" >&2; exit 2
  fi
done
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked source tree is dirty; refusing release" >&2; exit 2
fi
commit="$(git rev-parse HEAD)"
native_commit="$(git -C vendor/llama.cpp rev-parse HEAD)"
arch="$(uname -m)"
case "$arch" in x86_64|aarch64) ;; *) echo "Unsupported architecture" >&2; exit 2;; esac
DEST="${1:-dist}"
mkdir -p "$DEST"
DEST="$(realpath "$DEST")"
name="kiln-linux-${arch}-${commit:0:12}"
archive="$DEST/$name.tar.gz"
manifest="$DEST/$name.json"
digest="$DEST/$name.sha256"
for file in "$archive" "$manifest" "$digest"; do
  [[ ! -e "$file" ]] || { echo "Artifact exists: $file" >&2; exit 2; }
done
# Do not bake -march=native of the builder into the KILN binary.
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build vendor/llama.cpp/build --config Release --parallel "${KILN_BUILD_JOBS:-2}"
cargo build --locked --release -p kiln-cli
binary=target/release/kiln
[[ -x "$binary" ]] || { echo "No release binary" >&2; exit 1; }
"$binary" --help >/dev/null
"$binary" stream --help >/dev/null
strings "$binary" | grep -q "KILN_MOE_STREAM_BLOB" || {
  echo "Compiled native expert kernel not found in release" >&2; exit 1;
}
stage="$(mktemp -d "$DEST/.kiln-stage-XXXXXX")"
trap 'rm -rf "$stage"' EXIT
install -m 0755 "$binary" "$stage/kiln"
install -m 0644 LICENSE THIRD_PARTY_NOTICES.md "$stage/"
epoch="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"
[[ "$epoch" =~ ^[0-9]+$ ]] || { echo "Invalid source epoch" >&2; exit 2; }
binary_sha="$(sha256sum "$stage/kiln" | cut -d' ' -f1)"
python3 - "$manifest" "$commit" "$native_commit" "$arch" "$binary_sha" <<'PY'
import json
from pathlib import Path
import sys
out, commit, native, arch, sha = sys.argv[1:]
Path(out).write_text(json.dumps({
    "artifact_type":"single_executable_linux_release",
    "binary":"kiln","os":"linux","architecture":arch,
    "kiln_commit":commit,"vendored_native_commit":native,
    "binary_sha256":sha,"ggml_native":False,
    "included_model_weights":False,"experimental_moe_streaming":True,
    "tier0_deployment_accepted":False,
    "caveat":"System dynamic libraries/CPU compatibility and target quality/throughput require hardware validation"
},indent=2)+"\n",encoding="utf-8")
PY
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$epoch" \
  -czf "$archive" -C "$stage" kiln LICENSE THIRD_PARTY_NOTICES.md
sha256sum "$archive" > "$digest"
printf 'Archive: %s\nManifest: %s\nSHA256: %s\n' "$archive" "$manifest" "$digest"
echo "This artifact is not a Tier 0 performance or quality acceptance result."
