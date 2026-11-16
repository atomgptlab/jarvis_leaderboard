"""
Write the 10 CHIPS-FF leaderboard files (dft_3d_chipsff a, b, c, vol, kv,
c11, c44, form_en, surf_en, vac_en) from one chipsff run_main.py output
folder (chipsff_frechet_<tag>).

run_main.py relaxes the conventional cell, but the benchmark's a, b, c and
vol are for the JARVIS stored cell. Each stored-cell vector is a lattice
translation of the conventional cell, so its fractional coordinates N in
the conventional basis do not depend on how the conventional cell is
rotated. N times the relaxed conventional lattice gives the relaxed stored
cell.

kv, c11, c44 and form_en are read from <jid>_<calc>_results.json. Surface
energies are read from <jid>_<calc>_job_info.json, which holds every
computed surface (results.json keeps only the ones run_main.py compared).
A vacancy is written only for the site whose reference is the first
vacancydb entry for that element, which is the benchmark's reference
(the same rule as chipsff.utils.benchmark_vacancies).

Usage:
  python package_leaderboard.py RUN_DIR BENCH_DIR OUT_DIR \
      [--dft3d jdft_3d-9-24-2025.json.zip] [--vacancydb vacancydb.json.zip]
"""

import argparse
import csv
import io
import itertools
import json
import os
import zipfile

import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar

PROPS = ["a", "b", "c", "vol", "kv", "c11", "c44", "form_en", "surf_en", "vac_en"]
CACHE = os.path.expanduser("~/.cache/atomgptlab/jarvis_data")


def load_zip_json(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(z.namelist()[0]))


def is_translation(n, frac, numbers, lat, tol=0.02):
    """True if shifting every atom by fractional vector n maps the crystal onto itself."""
    shifted = (frac + n) % 1.0
    for f, z in zip(shifted, numbers):
        d = frac - f
        d -= np.rint(d)
        dist = np.linalg.norm(d @ lat, axis=1)
        if not np.any((dist < tol) & (numbers == z)):
            return False
    return True


def stored_in_conventional(stored):
    """Fractional matrix N (rows: stored-cell vectors in the conventional basis)."""
    conv = stored.get_conventional_atoms
    lc = np.array(conv.lattice_mat)
    ls = np.array(stored.lattice_mat)
    frac = np.array(conv.frac_coords) % 1.0
    numbers = np.array(conv.atomic_numbers)
    grid = np.array(list(itertools.product(np.arange(-18, 19) / 6.0, repeat=3)))
    lengths = np.linalg.norm(grid @ lc, axis=1)
    cands = []
    for i in range(3):
        li = np.linalg.norm(ls[i])
        idx = np.where(np.abs(lengths - li) < 2e-3 * li)[0]
        cands.append([grid[k] for k in idx if is_translation(grid[k], frac, numbers, lc)])
    gs = ls @ ls.T
    best, best_res = None, None
    for n0 in cands[0]:
        for n1 in cands[1]:
            if abs((n0 @ lc) @ (n1 @ lc) - gs[0, 1]) > 2e-3 * abs(gs[0, 0]) + 1e-6:
                continue
            for n2 in cands[2]:
                n = np.array([n0, n1, n2])
                res = np.abs(n @ lc @ lc.T @ n.T - gs).max()
                if res < 2e-3 * gs.max() and np.linalg.det(n) > 0:
                    if abs(np.linalg.det(n) - stored.num_atoms / conv.num_atoms) > 1e-6:
                        continue
                    if best_res is None or res < best_res:
                        best, best_res = n, res
    if best is None:
        raise RuntimeError("no lattice-translation map found")
    # Keep the map that reproduces the stored metric best, so the result does
    # not depend on the order the candidates were tried in.
    return best, conv.num_atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="chipsff_frechet_<tag> folder from run_main.py")
    ap.add_argument("bench_dir", help="jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction")
    ap.add_argument("out_dir")
    ap.add_argument("--dft3d", default=os.path.join(CACHE, "jdft_3d-9-24-2025.json.zip"))
    ap.add_argument("--vacancydb", default=os.path.join(CACHE, "vacancydb.json.zip"))
    args = ap.parse_args()

    ref = {p: load_zip_json(os.path.join(args.bench_dir, f"dft_3d_chipsff_{p}.json.zip"))["test"] for p in PROPS}
    jids = list(ref["a"])
    dft = {d["jid"]: d for d in load_zip_json(args.dft3d) if d["jid"] in ref["a"]}
    first_ref = {}
    for e in load_zip_json(args.vacancydb):
        first_ref.setdefault("_".join(e["id"].split("_")[:2]), e["ef"])

    pred = {p: {} for p in PROPS}
    for jid in jids:
        stored = Atoms.from_dict(dft[jid]["atoms"])
        sub = [s for s in os.listdir(os.path.join(args.run_dir, jid)) if s.startswith(jid + "_")]
        if len(sub) != 1:
            raise RuntimeError(f"{jid}: expected one calculator folder, found {sub}")
        d = os.path.join(args.run_dir, jid, sub[0])
        res = json.load(open(os.path.join(d, f"{sub[0]}_results.json")))
        info = json.load(open(os.path.join(d, f"{sub[0]}_job_info.json")))

        n, nconv = stored_in_conventional(stored)
        relaxed = Poscar.from_file(os.path.join(d, f"{jid}_bulk_relaxed.vasp")).atoms
        if relaxed.num_atoms != nconv:
            raise RuntimeError(f"{jid}: relaxed cell is not the conventional cell")
        lat = n @ np.array(relaxed.lattice_mat)
        pred["a"][jid], pred["b"][jid], pred["c"][jid] = np.linalg.norm(lat, axis=1)
        pred["vol"][jid] = abs(np.linalg.det(lat))

        pred["kv"][jid] = res["modulus"]["kv"]
        pred["c11"][jid] = res["elastic_tensor"]["c11"]
        pred["c44"][jid] = res["elastic_tensor"]["c44"]
        pred["form_en"][jid] = res["form_en"]["form_energy"]
        for key, val in info.items():
            if key in ref["surf_en"]:
                pred["surf_en"][key] = val
        for vac in res.get("vacancy_energy", []):
            vid = "_".join(vac["name"].split("_")[:2])
            r = first_ref.get(vid)
            if vid in ref["vac_en"] and r is not None and abs(vac["vac_en_entry"] - r) < 1e-6:
                pred["vac_en"].setdefault(vid, vac["vac_en"])

    os.makedirs(args.out_dir, exist_ok=True)
    for p in PROPS:
        missing = [i for i in ref[p] if i not in pred[p] or not np.isfinite(pred[p][i])]
        if missing:
            raise RuntimeError(f"{p}: no prediction for {missing}")
        name = f"AI-SinglePropertyPrediction-{p}-dft_3d_chipsff-test-mae.csv"
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["id", "prediction"])
        for i in ref[p]:
            w.writerow([i, repr(float(pred[p][i]))])
        with zipfile.ZipFile(os.path.join(args.out_dir, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(name, buf.getvalue())
        mae = np.mean([abs(pred[p][i] - ref[p][i]) for i in ref[p]])
        print(f"{p:8s} n={len(ref[p]):3d}  MAE {mae:.4f}")


if __name__ == "__main__":
    main()
