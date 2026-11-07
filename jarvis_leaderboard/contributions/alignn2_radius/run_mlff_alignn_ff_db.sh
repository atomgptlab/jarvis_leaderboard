#!/bin/bash
# Reproduce MLFF entries for dataset 'alignn_ff_db' (ALIGNN-FF, radius graph).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python   # env with alignn.ff + jarvis
"$PY" "$HERE/ff_predict.py" --dataset alignn_ff_db --model_dir ~/alignn2026/alignn_ff_db --out "$HERE"
