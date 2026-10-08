#!/usr/bin/env bash
set -euo pipefail

# Fetch the VLABench meshes/textures the OpenArm `insert_flower` scene references.
#
#   bash scripts/model/download_vlabench_assets.sh
#
# The default cube-tabletop scene is self-contained and needs none of this. The
# insert_flower assembly XMLs ship with the repo; only their meshes and textures
# (~25 MB) are fetched, because they are not redistributable here and the asset
# mirror at the recorded revision carries no separate asset license (see
# core/openarm/backend/assets/vlabench/SOURCE.json).
#
# Every file is pinned to the revision recorded in ASSET_MANIFEST.json and verified
# against its sha256; a mismatch aborts, so a re-run of the same revision is safe.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ASSET_DIR="${REPO_ROOT}/core/openarm/backend/assets/vlabench"
MANIFEST="${ASSET_DIR}/ASSET_MANIFEST.json"

[ -f "${MANIFEST}" ] || {
  echo "ERROR: manifest not found: ${MANIFEST}" >&2
  exit 1
}

exec python3 - "${ASSET_DIR}" "${MANIFEST}" <<'PY'
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

asset_dir = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
base = manifest["download_base_url"]
files = manifest["files"]

done = skipped = 0
for entry in files:
    rel = entry["path"]
    target = asset_dir / rel
    want_size, want_hash = int(entry["bytes"]), entry["sha256"]
    if target.is_file() and target.stat().st_size == want_size:
        if hashlib.sha256(target.read_bytes()).hexdigest() == want_hash:
            skipped += 1
            continue
    target.parent.mkdir(parents=True, exist_ok=True)
    url = base + entry["source_path"]
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    got_hash = hashlib.sha256(payload).hexdigest()
    if len(payload) != want_size or got_hash != want_hash:
        raise SystemExit(
            f"ERROR: {rel} does not match the pinned revision\n"
            f"  expected {want_size} bytes / {want_hash}\n"
            f"  got      {len(payload)} bytes / {got_hash}"
        )
    target.write_bytes(payload)
    done += 1

print(f"VLABench assets ready at {asset_dir}")
print(f"  downloaded {done}, already current {skipped}, total {len(files)}")
PY
