"""Build id_prop.json for a dft_3d single-property benchmark, using the
benchmark's DEFINED train/val/test id splits (same data every run).

    python build_property_data.py <property> [output_dir]

Writes <output_dir>/id_prop.json as a list of {"jid","atoms","target"} in
train+val+test order (config.json uses keep_data_order=True so the trainer
carves train/val/test from this order). Requires jarvis-tools + the local
jarvis_leaderboard checkout for the benchmark zips.
"""
import os, sys, json, zipfile
from jarvis.db.figshare import data

prop = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "."
LB = "/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"
BENCH = os.path.join(LB, "benchmarks", "AI", "SinglePropertyPrediction",
                     f"dft_3d_{prop}.json.zip")

b = json.loads(zipfile.ZipFile(BENCH).read(f"dft_3d_{prop}.json"))
mem = {e["jid"]: e for e in data("dft_3d")}
tr = [i for i in b["train"] if i in mem]
va = [i for i in b.get("val", {}) if i in mem]
te = [i for i in b["test"] if i in mem]
tgt = {**b["train"], **b.get("val", {}), **b["test"]}
dat = [{"jid": i, "atoms": mem[i]["atoms"], "target": float(tgt[i])}
       for i in tr + va + te]
json.dump(dat, open(os.path.join(out, "id_prop.json"), "w"))
print(f"{prop}: id_prop.json written  n_train {len(tr)}  n_val {len(va)}  n_test {len(te)}")
