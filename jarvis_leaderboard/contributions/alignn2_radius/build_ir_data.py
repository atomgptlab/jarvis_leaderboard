"""Build id_prop.json for the irdb IR-spectrum benchmark, using the benchmark's
DEFINED train/val/test id splits (same data every run).

    python build_ir_data.py [output_dir]

Writes <output_dir>/id_prop.json as a list of {"jid","atoms","target"} where
target is the 200-bin IR spectrum (0-2000 cm^-1, Lorentzian-broadened), in
train+val+test order (config uses keep_data_order=True). Structures pulled from
jarvis dft_3d; spectra from benchmarks/AI/Spectra/irdb_ir.json.zip.
"""
import os, sys, json, zipfile
from jarvis.db.figshare import data

out = sys.argv[1] if len(sys.argv) > 1 else "."
LB = "/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"
BENCH = os.path.join(LB, "benchmarks", "AI", "Spectra", "irdb_ir.json.zip")

b = json.loads(zipfile.ZipFile(BENCH).read("irdb_ir.json"))
mem = {e["jid"]: e for e in data("dft_3d")}
spec = {**b["train"], **b.get("val", {}), **b["test"]}
def vec(s): return [float(x) for x in s.split(";")]
tr = [i for i in b["train"] if i in mem]
va = [i for i in b.get("val", {}) if i in mem]
te = [i for i in b["test"] if i in mem]
dat = [{"jid": i, "atoms": mem[i]["atoms"], "target": vec(spec[i])} for i in tr + va + te]
json.dump(dat, open(os.path.join(out, "id_prop.json"), "w"))
print(f"irdb IR: id_prop.json written  n_train {len(tr)}  n_val {len(va)}  n_test {len(te)}")
