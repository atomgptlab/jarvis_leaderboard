#!/bin/bash
# Reproduce MLFF entries for dataset 'mptrj' (ALIGNN-FF, radius graph).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python   # env with alignn.ff + jarvis
"$PY" "$HERE/ff_predict.py" --dataset mptrj --model_dir ~/alignn2026/mptrj_frontier/model_ep46_bestval --out "$HERE"
