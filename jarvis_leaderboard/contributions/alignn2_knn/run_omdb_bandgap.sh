#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-bandgap-omdb-test-mae.csv.zip
#   ALIGNN 2.0 (knn graph, cutoff 8.0) on jarvis 'omdb' property 'bandgap'.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORK="$HERE/_work_omdb_bandgap"; mkdir -p "$WORK"
"$PY" "$HERE/build_lb_data.py" omdb bandgap "$WORK"
cp "$HERE/config_omdb_bandgap.json" "$WORK/config.json"
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target
"$PY" - <<PYINNER
import pandas as pd, zipfile, os
p=pd.read_csv("$WORK/results/prediction_results_test_set.csv"); p.columns=[c.strip() for c in p.columns]
name="AI-SinglePropertyPrediction-bandgap-omdb-test-mae.csv"
p[["id","prediction"]].to_csv("$HERE/"+name, index=False)
with zipfile.ZipFile("$HERE/"+name+".zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name); print("wrote", name+".zip")
PYINNER
