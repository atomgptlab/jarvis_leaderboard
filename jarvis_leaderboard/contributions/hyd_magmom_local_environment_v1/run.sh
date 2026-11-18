#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
task_python=${PYTHON_BIN:-python3}
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 (set PYTHON_BIN if needed)"'
"$task_python" -m venv .replay-venv
.replay-venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/.replay-cache/matplotlib"
exec .replay-venv/bin/python replay_benchmark.py \
  --selection selection.json --ids test_ids.json \
  --output AI-SinglePropertyClass-magmom_oszicar-dft_3d-test-acc.csv.zip "$@"
