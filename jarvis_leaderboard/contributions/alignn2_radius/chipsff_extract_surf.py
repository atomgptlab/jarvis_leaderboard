import glob, json, zipfile, csv, io, os
B="/home/kamalch/alignn2026/ff_bakeoff/matpes_surf"
C="/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard/contributions/alignn2_radius"
LB="/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction"
surf={}
for f in glob.glob(f"{B}/chunk*/JVASP-*/JVASP-*_alignn_ff/*_job_info.json"):
    d=json.load(open(f))
    for s in d.get("all_surfaces",[]):
        if s.get("surface_name") and s.get("surf_en") is not None:
            surf[s["surface_name"]]=s["surf_en"]
dft={k:float(v) for k,v in json.loads(zipfile.ZipFile(f"{LB}/dft_3d_chipsff_surf_en.json.zip").read("dft_3d_chipsff_surf_en.json"))["test"].items()}
preds={k:v for k,v in surf.items() if k in dft}
name="AI-SinglePropertyPrediction-surf_en-dft_3d_chipsff-test-mae.csv"
with open(f"{C}/{name}","w",newline="") as f:
    w=csv.writer(f); w.writerow(["id","prediction"]); [w.writerow([k,v]) for k,v in preds.items()]
with zipfile.ZipFile(f"{C}/{name}.zip","w",zipfile.ZIP_DEFLATED) as z: z.write(f"{C}/{name}",name); 
os.remove(f"{C}/{name}")
e=[abs(dft[k]-v) for k,v in preds.items()]
print(f"surf_en updated: {len(preds)}/{len(dft)} matched, MAE {sum(e)/len(e):.4f}")
