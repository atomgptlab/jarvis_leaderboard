#!/bin/bash
# Reproduce MLFF entries for dataset 'mlearn_Li' (ALIGNN-FF, radius graph).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python   # env with alignn.ff + jarvis
"$PY" "$HERE/ff_predict.py" --dataset mlearn_Li --model_dir ~/alignn2026/mlearn_ff/Li/results --out "$HERE"
