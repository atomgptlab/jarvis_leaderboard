#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-gap-c2db-test-mae.csv.zip
#   ALIGNN 2.0 (knn graph, cutoff 8.0) on jarvis 'c2db' property 'gap'.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORK="$HERE/_work_c2db_gap"; mkdir -p "$WORK"
# 1) build id_prop from the leaderboard benchmark split (structures from jarvis 'c2db')
"$PY" "$HERE/build_lb_data.py" c2db gap "$WORK"
cp "$HERE/config_c2db_gap.json" "$WORK/config.json"
# 2) train
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target
# 3) package prediction CSV in leaderboard format
"$PY" - <<PYINNER
import pandas as pd, zipfile, os
p=pd.read_csv("$WORK/results/prediction_results_test_set.csv"); p.columns=[c.strip() for c in p.columns]
name="AI-SinglePropertyPrediction-gap-c2db-test-mae.csv"
p[["id","prediction"]].to_csv("$HERE/"+name, index=False)
with zipfile.ZipFile("$HERE/"+name+".zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name); print("wrote", name+".zip")
PYINNER
