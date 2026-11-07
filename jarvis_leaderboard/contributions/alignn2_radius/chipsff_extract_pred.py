"""Extract per-material predictions from a directory of chipsff <jid>_job_info.json
files into a compact json used by chipsff_leaderboard_matpes.py.

Usage:
  # conventional-cell run (intensive props + surfaces/vacancies + e0 for form_en):
  python chipsff_extract_pred.py <runs/matpes_ep100 dir> chipsff_pred_matpes.json full
  # primitive-cell run (a/b/c/vol only, to match the primitive benchmark):
  python chipsff_extract_pred.py <matpes_prim dir> chipsff_pred_matpes_prim.json geom
Searches recursively for */*_job_info.json.
"""
import sys, glob, json

run_dir, out, mode = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "full")
pred = {}
for f in glob.glob(f"{run_dir}/**/*_job_info.json", recursive=True):
    d = json.load(open(f)); jid = d.get("jid")
    if not jid:
        continue
    ra = d.get("relaxed_atoms", {}) or {}
    abc = ra.get("abc", [None, None, None])
    rec = {"a": abc[0], "b": abc[1], "c": abc[2], "vol": d.get("equilibrium_volume")}
    if mode == "full":
        et = d.get("elastic_tensor", {}) or {}
        rec.update({"kv": d.get("bulk_modulus"), "c11": et.get("C_11"), "c44": et.get("C_44"),
                    "e0": d.get("equilibrium_energy"), "elements": ra.get("elements"),
                    "surfaces": [{"name": s.get("surface_name"), "surf_en": s.get("surf_en")}
                                 for s in (d.get("all_surfaces") or [])],
                    "vacancies": [{"name": v.get("name"), "vac_en": v.get("vac_en")}
                                  for v in (d.get("all_vacancies") or [])]})
    pred[jid] = rec
json.dump(pred, open(out, "w"))
print(f"extracted {len(pred)} materials -> {out} (mode={mode})")
