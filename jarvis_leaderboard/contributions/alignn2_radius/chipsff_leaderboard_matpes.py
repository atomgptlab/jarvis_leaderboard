"""Build the CHIPS_FF leaderboard CSVs (dft_3d_chipsff) for the ALIGNN 2.0 default
force field (matpes_smooth) from the chipsff run outputs.

Inputs (staged next to this script):
  chipsff_pred_matpes.json  : per-material predictions extracted from each
                              chipsff run's <jid>_job_info.json (a,b,c,vol,kv,
                              c11,c44, equilibrium_energy e0, elements, surfaces,
                              vacancies).
  matpes_smooth_unary.json  : self-consistent elemental chemical potentials
                              (energy/atom of each element's reference structure,
                              single-point with matpes_smooth). Used to compute a
                              REFERENCE-CONSISTENT formation energy — the shipped
                              chipsff chempots are on a different (old-model) scale,
                              which otherwise inflates form_en by ~2.4 eV/atom.

Writes AI-SinglePropertyPrediction-<prop>-dft_3d_chipsff-test-mae.csv.zip (id,prediction)
for the 10 CHIPS_FF properties, filtered to each benchmark's test ids, and registers
them in metadata.json.
"""
import os, json, zipfile, csv
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LB = "/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"
BDIR = f"{LB}/benchmarks/AI/SinglePropertyPrediction"

pred = json.load(open(f"{HERE}/chipsff_pred_matpes.json"))            # conventional-cell run (intensive props)
predp = json.load(open(f"{HERE}/chipsff_pred_matpes_prim.json"))       # primitive-cell run (a/b/c/vol)
mu = {el: v["energy"] for el, v in json.load(open(f"{HERE}/matpes_smooth_unary.json")).items()}

def bench_ids(prop):
    z = zipfile.ZipFile(f"{BDIR}/dft_3d_chipsff_{prop}.json.zip")
    return set(json.loads(z.read(f"dft_3d_chipsff_{prop}.json"))["test"])

def form_en(rec):  # reference-consistent formation energy per atom
    els = rec.get("elements")
    if not els or rec.get("e0") is None:
        return None
    comp = Counter(els)
    if not all(e in mu for e in comp):
        return None
    return (rec["e0"] - sum(mu[e] * c for e, c in comp.items())) / len(els)

# (leaderboard prop -> function returning {id: prediction})
def scalar(key):
    return lambda: {j: r[key] for j, r in pred.items() if r.get(key) is not None}
# a/b/c/vol come from the PRIMITIVE-cell run (chipsff_pred_matpes_prim.json): the
# dft_3d_chipsff benchmark stores primitive-cell lattice params (Si a=3.88=5.43/sqrt2),
# so use_conventional_cell=false is required for these cell-extensive quantities.
# kv/c11/c44/form_en/surf_en/vac_en are intensive -> taken from the conventional run.
def scalarp(key):
    return lambda: {j: r[key] for j, r in predp.items() if r.get(key) is not None}
builders = {
    "a": scalarp("a"), "b": scalarp("b"), "c": scalarp("c"), "vol": scalarp("vol"),
    "kv": scalar("kv"), "c11": scalar("c11"), "c44": scalar("c44"),
    "form_en": lambda: {j: form_en(r) for j, r in pred.items() if form_en(r) is not None},
    "surf_en": lambda: {s["name"]: s["surf_en"] for r in pred.values() for s in r.get("surfaces", []) if s.get("surf_en") is not None},
    "vac_en": lambda: {v["name"]: v["vac_en"] for r in pred.values() for v in r.get("vacancies", []) if v.get("vac_en") is not None},
}

meta_path = f"{HERE}/metadata.json"
meta = json.load(open(meta_path))
tt = meta.setdefault("time_taken_seconds", {})
for prop, build in builders.items():
    ids = bench_ids(prop)
    preds = {k: v for k, v in build().items() if k in ids}
    name = f"AI-SinglePropertyPrediction-{prop}-dft_3d_chipsff-test-mae.csv"
    with open(f"{HERE}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "prediction"])
        for k, v in preds.items():
            w.writerow([k, v])
    with zipfile.ZipFile(f"{HERE}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{HERE}/{name}", name)
    os.remove(f"{HERE}/{name}")
    tt[f"{name}.zip"] = ""
    print(f"  {prop:8s} {len(preds):>4}/{len(ids)} test ids written")
json.dump(meta, open(meta_path, "w"), indent=2)
print("registered 10 CHIPS_FF entries in metadata.json")
