#!/bin/bash
# Reproduce MLFF entries for dataset 'matpes' (ALIGNN-FF, radius graph).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python   # env with alignn.ff + jarvis
"$PY" "$HERE/ff_predict.py" --dataset matpes --model_dir ~/alignn2026/matpes_kdo/results --out "$HERE"
