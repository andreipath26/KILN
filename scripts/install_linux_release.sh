#!/usr/bin/env bash
# User-scoped atomic installer for a previously built KILN Linux archive.
# The archive contains exactly ONE executable; no model files are copied.
# Usage: bash scripts/install_linux_release.sh <kiln-linux-*.tar.gz> [bin-dir]
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <kiln-linux-*.tar.gz> [bin-dir]" >&2
  exit 2
fi
[[ "$(uname -s)" == Linux ]] || { echo "Linux only" >&2; exit 2; }
for cmd in python3 realpath; do
  command -v "$cmd" >/dev/null || { echo "Missing: $cmd" >&2; exit 2; }
done
ARCHIVE="$(realpath "$1")"
MANIFEST="${ARCHIVE%.tar.gz}.json"
CHECKSUM="${ARCHIVE%.tar.gz}.sha256"
DEST="${2:-$HOME/.local/bin}"
[[ -f "$ARCHIVE" && -f "$MANIFEST" && -f "$CHECKSUM" ]] || {
  echo "Expected archive plus adjacent matching JSON manifest and .sha256" >&2
  exit 2
}
mkdir -p "$DEST"
DEST="$(realpath "$DEST")"
python3 - "$ARCHIVE" "$MANIFEST" "$CHECKSUM" "$DEST" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tarfile
import tempfile

archive, manifest, checksum, destination = map(Path, sys.argv[1:])
config = json.loads(manifest.read_text(encoding="utf-8"))
if config.get("artifact_type") != "single_executable_linux_release":
    raise SystemExit("not a KILN single-binary release manifest")
if config.get("os") != "linux" or config.get("architecture") != platform.machine():
    raise SystemExit("KILN binary architecture does not match target Linux machine")
digest = config.get("binary_sha256", "")
if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
    raise SystemExit("invalid binary SHA256")
sum_field = checksum.read_text(encoding="utf-8").split()
if not sum_field or re.fullmatch(r"[a-f0-9]{64}", sum_field[0]) is None:
    raise SystemExit("invalid archive SHA256")
hasher = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
if hasher.hexdigest() != sum_field[0]:
    raise SystemExit("archive SHA256 mismatch; refusing installation")

target = destination / "kiln"
if target.exists() or target.is_symlink():
    raise SystemExit("kiln already exists; refusing to replace the installed binary")
with tarfile.open(archive, "r:gz") as package:
    members = package.getmembers()
    if len(members) != 3 or {m.name for m in members} != {"kiln", "LICENSE", "THIRD_PARTY_NOTICES.md"}:
        raise SystemExit("unexpected or duplicated KILN archive contents")
    binary_member = package.getmember("kiln")
    if not binary_member.isfile() or not (0 < binary_member.size <= 2 * 1024**3):
        raise SystemExit("KILN binary is not a valid regular archive member")
    data = package.extractfile(binary_member)
    if data is None:
        raise SystemExit("could not read executable bytes")
    fd, temporary = tempfile.mkstemp(prefix=".kiln-verify-", dir=destination)
    tmp = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            source_hash = hashlib.sha256()
            for block in iter(lambda: data.read(1024 * 1024), b""):
                source_hash.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if source_hash.hexdigest() != digest:
            raise SystemExit("KILN binary digest mismatch")
        os.chmod(tmp, 0o755)
        # No accidental overwrite: hardlink creation is atomic and fails if
        # another install has already created the target pathname.
        os.link(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
print(f"Verified single KILN binary installed at {target}")
print("Experimental MoE streaming is opt-in; Tier 0 acceptance remains unverified.")
PY
