#!/bin/bash
# Reproduce MLFF entries for dataset 'matpes' (ALIGNN 2.0 default force field,
# matpes_smooth: 2/2/128, smooth cutoff, nbr52, MATPES-PBE ep100, radius graph).
# 'matpes' is not a registered jarvis.db.figshare name in current jarvis, so the
# structures are sourced from the local MATPES id_prop.json via --src_json.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python   # env with alignn.ff + jarvis
SRC=/home/kamalch/Software/fast_graph/alignn/TRAINGING/MATPES/DataDir/id_prop.json
CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/ff_predict.py" --dataset matpes \
  --model_dir ~/alignn2026/matpes_smooth/results --src_json "$SRC" --out "$HERE"
