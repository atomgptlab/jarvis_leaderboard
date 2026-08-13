#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-Tc_supercon_hydride_plus_bulk-dft_3d-test-mae.csv.zip
#   ALIGNN 2.0 (knn graph, cutoff 8.0) on dft_3d property 'Tc_supercon_hydride_plus_bulk'.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORK="$HERE/_work_Tc_supercon_hydride_plus_bulk"; mkdir -p "$WORK"
# 1) build dft_3d + benchmark train/val/test split for this property
"$PY" "$HERE/build_property_data.py" Tc_supercon_hydride_plus_bulk "$WORK"
cp "$HERE/config_Tc_supercon_hydride_plus_bulk.json" "$WORK/config.json"
# 2) train (config = exact graph + model used)
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target
# 3) package prediction CSV in leaderboard format
"$PY" - <<PYEOF
import pandas as pd, zipfile, os
p=pd.read_csv("$WORK/results/prediction_results_test_set.csv"); p.columns=[c.strip() for c in p.columns]
name="AI-SinglePropertyPrediction-Tc_supercon_hydride_plus_bulk-dft_3d-test-mae.csv"
p[["id","prediction"]].to_csv("$HERE/"+name, index=False)
with zipfile.ZipFile("$HERE/AI-SinglePropertyPrediction-Tc_supercon_hydride_plus_bulk-dft_3d-test-mae.csv.zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name)
print("wrote", "AI-SinglePropertyPrediction-Tc_supercon_hydride_plus_bulk-dft_3d-test-mae.csv.zip")
PYEOF
