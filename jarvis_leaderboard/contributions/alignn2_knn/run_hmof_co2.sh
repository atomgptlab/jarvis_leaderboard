#!/bin/bash
# Reproduce AI-SinglePropertyPrediction-co2-hmof-test-mae.csv.zip
#   ALIGNN 2.0 (knn graph, cutoff 8.0) for hMOF CO2 uptake (mol/kg).
# NOTE: hMOF is a custom dataset (~137k hypothetical MOFs), NOT a jarvis db.
#   Provide id_prop.json ({jid, atoms, target}) for the hMOF set, then:
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORK="$HERE/_work_hmof_co2"; mkdir -p "$WORK"
# expects $WORK/id_prop.json (from the hMOF dataset, split per benchmarks/AI/SinglePropertyPrediction/hmof_co2.json.zip)
cp "$HERE/config_hmof_co2.json" "$WORK/config.json"
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target --id_key jid
"$PY" - <<PYINNER
import pandas as pd, zipfile, os
p=pd.read_csv("$WORK/results/prediction_results_test_set.csv"); p.columns=[c.strip() for c in p.columns]
name="AI-SinglePropertyPrediction-co2-hmof-test-mae.csv"
p[["id","prediction"]].to_csv("$HERE/"+name, index=False)
with zipfile.ZipFile("$HERE/"+name+".zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name); print("wrote", name+".zip")
PYINNER
