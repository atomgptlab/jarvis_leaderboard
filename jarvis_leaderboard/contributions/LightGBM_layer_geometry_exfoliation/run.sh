#!/usr/bin/env bash
# This contribution wrapper may be used and redistributed under the JARVIS
# Leaderboard repository terms. The separately fetched package has its own
# benchmark-reproduction license; see that package's LICENSE.
set -euo pipefail

PACKAGE_URL="https://github.com/AcuVizionAI/LightGBM_layer_geometry_exfoliation.git"
PACKAGE_COMMIT="71f38216612c8aed349b63dfd2383b7e6e9d7e26"
MODEL_MANIFEST_SHA256="b27dea7f86bece3150baa3494b4d9a0313da70c8a69922d527f3956dafe060a4"
SUBMISSION_ZIP_SHA256="ccb7fb5b2de91a637d6c02846152c631929589a820c584b5345c5383de521ad9"
CSV_ZIP="AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv.zip"
PYTHON_BIN="${PYTHON:-python3.10}"
CONTRIBUTION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_ZIP="$CONTRIBUTION_DIR/$CSV_ZIP"

if [[ ! "$PACKAGE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || [[ ! "$MODEL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'ERROR: publication commit and manifest SHA-256 must be pinned before running.' >&2
    exit 2
fi
command -v git >/dev/null
command -v "$PYTHON_BIN" >/dev/null
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PYTHON_BIN" -B - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit("Use the tested Python 3.10 interpreter (override its path with PYTHON).")
PY

# Only the structure archive may be supplied. The output location is always new.
DATA_ARGS=()
if [[ $# -eq 2 && "$1" == "--data-zip" ]]; then
    DATA_ARGS=(--data-zip "$2")
elif [[ $# -eq 1 && "$1" == --data-zip=* ]]; then
    DATA_ARGS=(--data-zip "${1#--data-zip=}")
elif [[ $# -ne 0 ]]; then
    printf 'Usage: PYTHON=/path/to/python3.10 bash %s [--data-zip /absolute/path/to/structure.zip]\n' "$0" >&2
    exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/acuvizionai-lgbm.XXXXXXXX")"
PACKAGE_DIR="$WORK_DIR/package"
VENV_DIR="$WORK_DIR/venv"
printf 'Fetching fixed inference package into %s\n' "$PACKAGE_DIR"
git clone --quiet --no-checkout "$PACKAGE_URL" "$PACKAGE_DIR"
git -C "$PACKAGE_DIR" checkout --quiet --detach "$PACKAGE_COMMIT"
if [[ "$(git -C "$PACKAGE_DIR" rev-parse HEAD)" != "$PACKAGE_COMMIT" ]]; then
    printf '%s\n' 'ERROR: checkout differs from the pinned package commit.' >&2
    exit 1
fi

# Bootstrap verification uses only the interpreter standard library. No code
# from the downloaded package is imported or executed before these checks pass.
"$PYTHON_BIN" -B - "$PACKAGE_DIR" "$MODEL_MANIFEST_SHA256" "$REFERENCE_ZIP" "$SUBMISSION_ZIP_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

package = Path(sys.argv[1]).resolve()
manifest = package / "model_manifest.json"
reference = Path(sys.argv[3])

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key: " + key)
        result[key] = value
    return result

if digest(manifest) != sys.argv[2]:
    raise SystemExit("Pinned model_manifest.json checksum mismatch")
if digest(reference) != sys.argv[4]:
    raise SystemExit("Submitted prediction archive checksum mismatch")
data = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
if data.get("schema_version") != 1 or not isinstance(data.get("files_sha256"), dict):
    raise SystemExit("Unsupported package manifest format")
files = data["files_sha256"]
required = {"config.json", "requirements.txt", "test_ids.json", "run.py", "verify.py",
            "features.py", "featurize.py", "layer_features.py", "model/model.txt", "LICENSE"}
required.update(path.name for path in package.glob("*.py"))
if not required.issubset(files):
    raise SystemExit("Required executable/model assets are absent from the pinned manifest")
for name, expected in files.items():
    candidate = package / name
    path = candidate.resolve()
    if Path(name).is_absolute() or "\\" in name or path.relative_to(package).as_posix() != name:
        raise SystemExit("Unsafe or noncanonical manifest path: " + name)
    if candidate.is_symlink() or not path.is_file() or digest(path) != expected:
        raise SystemExit("Package asset checksum mismatch: " + name)
print("PASS: pinned manifest, submitted ZIP, and all %d package assets verified." % len(files))
PY

"$PYTHON_BIN" -B -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -B -m pip install --disable-pip-version-check -r "$PACKAGE_DIR/requirements.txt"
"$VENV_DIR/bin/python" -B -m pip check

# The submitted ZIP is always a read-only reference. Generated predictions and
# receipts are retained in a unique directory and never replace submission bytes.
OUTPUT_ROOT="$(mktemp -d "$CONTRIBUTION_DIR/reproduced.XXXXXXXX")"
PREDICTION_DIR="$OUTPUT_ROOT/predictions"
"$VENV_DIR/bin/python" -B "$PACKAGE_DIR/run.py" "${DATA_ARGS[@]}" --out-dir "$PREDICTION_DIR" \
    | tee "$OUTPUT_ROOT/inference.log"
"$VENV_DIR/bin/python" -B "$PACKAGE_DIR/verify.py" "$REFERENCE_ZIP" "$PREDICTION_DIR/$CSV_ZIP" --atol 1e-9 \
    | tee "$OUTPUT_ROOT/VERIFICATION.json"

"$PYTHON_BIN" -B - "$REFERENCE_ZIP" "$SUBMISSION_ZIP_SHA256" <<'PY'
import hashlib
from pathlib import Path
import sys
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != sys.argv[2]:
    raise SystemExit("Submitted prediction archive changed during reproduction")
PY
printf '\nReproduction passed. Submitted ZIP unchanged.\nResults: %s\nPackage and isolated environment: %s\n' "$OUTPUT_ROOT" "$WORK_DIR"
