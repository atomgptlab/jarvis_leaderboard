#!/bin/bash
# Reproduce AI-Spectra-dielectric-mbjdiel-test-multimae.csv.zip
#   ALIGNN 2.0 (knn graph, cutoff 8.0) TBmBJ dielectric function imag_xx (300-bin, 0-15 eV) on mbjdiel.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/fast_graph/bin/python
TR=/home/kamalch/miniforge3/envs/fast_graph/bin/train_alignn.py
export PYTHONPATH=/home/kamalch/Software/fast_graph/alignn:/home/kamalch/Software/slako312/jarvis_leaderboard
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WORK="$HERE/_work_dielectric"; mkdir -p "$WORK"
"$PY" "$HERE/build_diel_data.py" "$WORK"
cp "$HERE/config_dielectric.json" "$WORK/config.json"
cd "$WORK"; rm -rf results *_data
"$TR" --root_dir "$WORK" --config_name config.json --output_dir results --target_key target
"$PY" - <<PYINNER
import json, zipfile, os
mo=json.load(open("$WORK/results/multi_out_predictions.json"))
def vec(r):
    p=r["predictions"]; return p[0] if (len(p)==1 and hasattr(p[0],"__len__")) else p
name="AI-Spectra-dielectric-mbjdiel-test-multimae.csv"
with open("$HERE/"+name,"w") as f:
    f.write("id,prediction\n")
    for r in mo: f.write(r["id"]+","+";".join(str(round(float(x),6)) for x in vec(r))+"\n")
with zipfile.ZipFile("$HERE/"+name+".zip","w",zipfile.ZIP_DEFLATED) as z: z.write("$HERE/"+name, name)
os.remove("$HERE/"+name); print("wrote", name+".zip")
PYINNER
