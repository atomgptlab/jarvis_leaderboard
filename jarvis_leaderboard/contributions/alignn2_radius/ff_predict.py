"""Reproduce ALIGNN-FF MLFF leaderboard CSVs (energy/forces/stresses) for a dataset.

Loads the ALIGNN-FF calculator from --model_dir, predicts on the benchmark test
set for --dataset, and writes AI-MLFF-<prop>-<dataset>-test-<metric>.csv.zip.
Usage: python ff_predict.py --dataset mlearn_Cu --model_dir <dir> --out <dir>
"""
import argparse, json, zipfile, csv, os
from alignn.ff.ff import AlignnAtomwiseCalculator
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data

LB = "/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"

# dataset -> jarvis source dataset that holds the structures (keyed by jid)
SRC = {"matpes": "matpes", "alignn_ff_db": "alignn_ff_db", "mptrj": "m3gnet_mpf_1.5mil"}
def source_of(ds):
    if ds.startswith("mlearn_"):
        return "mlearn"
    return SRC.get(ds, ds)

def bench(ds, prop):
    z = f"{LB}/benchmarks/AI/MLFF/{ds}_{prop}.json.zip"
    if not os.path.exists(z):
        return None
    return json.loads(zipfile.ZipFile(z).read(f"{ds}_{prop}.json"))["test"]

def write_csv(out, name, rows, hdr):
    p = os.path.join(out, name)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); [w.writerow(r) for r in rows]
    with zipfile.ZipFile(p + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(p, name)
    os.remove(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src_json", default=None,
                    help="local json (list of {jid, atoms}) to source structures from "
                         "when the jarvis dataset name is unavailable")
    a = ap.parse_args()
    md = os.path.expanduser(a.model_dir)
    if a.src_json:
        d = {e["jid"]: e for e in json.load(open(os.path.expanduser(a.src_json)))}
    else:
        d = {e["jid"]: e for e in data(source_of(a.dataset))}
    calc = AlignnAtomwiseCalculator(path=md, force_mult_natoms=False,
                                    force_multiplier=1, stress_wt=1)
    be = bench(a.dataset, "energy")
    if be is None:
        raise SystemExit(f"no energy benchmark for {a.dataset}")
    bf = bench(a.dataset, "forces"); bs = bench(a.dataset, "stresses")
    ids = list(be)
    eP, fP, sP = {}, {}, {}
    for jid in ids:
        at = Atoms.from_dict(d[jid]["atoms"]).ase_converter(); at.calc = calc
        eP[jid] = at.get_potential_energy() / len(at)   # per-atom (ASE energy is extensive)
        if bf is not None:
            fP[jid] = ";".join(f"{x:.8g}" for x in at.get_forces().flatten())
        if bs is not None:
            xx, yy, zz, yz, xz, xy = at.get_stress()
            full9 = [xx, xy, xz, xy, yy, yz, xz, yz, zz]
            sP[jid] = ";".join(f"{x:.8g}" for x in full9)
    write_csv(a.out, f"AI-MLFF-energy-{a.dataset}-test-mae.csv",
              [[i, be[i], eP[i]] for i in ids], ["id", "target", "prediction"])
    if bf is not None:
        write_csv(a.out, f"AI-MLFF-forces-{a.dataset}-test-multimae.csv",
                  [[i, bf[i], fP[i]] for i in ids], ["id", "target", "prediction"])
    if bs is not None:
        write_csv(a.out, f"AI-MLFF-stresses-{a.dataset}-test-multimae.csv",
                  [[i, bs[i], sP[i]] for i in ids], ["id", "target", "prediction"])
    print(f"wrote MLFF CSVs for {a.dataset}")

if __name__ == "__main__":
    main()
