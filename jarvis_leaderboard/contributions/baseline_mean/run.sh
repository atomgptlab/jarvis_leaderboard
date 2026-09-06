#!/bin/bash
# Reproduce the mean baseline: for every benchmark, predict the train-set mean
# for all test ids. Regenerates each id,prediction CSV in this folder.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LB="$(python -c 'import jarvis_leaderboard,os;print(os.path.dirname(os.path.abspath(jarvis_leaderboard.__file__)))')"
python - "$HERE" "$LB" <<'PY'
import sys, os, glob, json, zipfile, csv, io
HERE, LB = sys.argv[1], sys.argv[2]
for z in sorted(glob.glob(os.path.join(HERE, "AI-SinglePropertyPrediction-*-test-mae.csv.zip"))):
    name = os.path.basename(z)[:-4]                       # strip .zip -> ...csv
    stem = name[:-4]                                      # strip .csv
    _, _, prop, dataset, _, _ = stem.split("-")          # AI-SPP-<prop>-<dataset>-test-mae
    bz = glob.glob(os.path.join(LB, "benchmarks", "**", f"{dataset}_{prop}.json.zip"), recursive=True)
    if not bz:
        print("skip (no benchmark):", name); continue
    d = json.loads(zipfile.ZipFile(bz[0]).read(f"{dataset}_{prop}.json"))
    tr = [float(v) for v in d["train"].values()]
    mean = sum(tr) / len(tr)
    rows = [("id", "prediction")] + [(k, mean) for k in d["test"]]
    buf = io.StringIO(); csv.writer(buf).writerows(rows)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, buf.getvalue())
    print(f"{dataset}_{prop}: mean={mean:.4f} n_test={len(d['test'])}")
PY
