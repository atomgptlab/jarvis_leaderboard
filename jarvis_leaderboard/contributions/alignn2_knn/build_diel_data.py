"""Build id_prop.json for the mbjdiel TBmBJ-dielectric benchmark split.
    python build_diel_data.py [output_dir]
Structures from jarvis dft_3d; 300-bin imag_xx spectra from benchmarks/AI/Spectra/mbjdiel_dielectric.json.zip."""
import os, sys, json, zipfile
from jarvis.db.figshare import data
out = sys.argv[1] if len(sys.argv) > 1 else "."
LB = "/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"
b = json.loads(zipfile.ZipFile(os.path.join(LB,"benchmarks","AI","Spectra","mbjdiel_dielectric.json.zip")).read("mbjdiel_dielectric.json"))
mem = {e["jid"]: e for e in data("dft_3d")}
spec = {**b["train"], **b.get("val", {}), **b["test"]}
vec = lambda s: [float(x) for x in s.split(";")]
tr=[i for i in b["train"] if i in mem]; va=[i for i in b.get("val",{}) if i in mem]; te=[i for i in b["test"] if i in mem]
dat=[{"jid":i,"atoms":mem[i]["atoms"],"target":vec(spec[i])} for i in tr+va+te]
json.dump(dat, open(os.path.join(out,"id_prop.json"),"w"))
print(f"mbjdiel: id_prop.json  n_train {len(tr)} n_val {len(va)} n_test {len(te)}")
