"""Build id_prop.json for (dataset, property) using the leaderboard benchmark split.
Auto-detects the dataset's ID key (jid/id). Usage: build_lb_data.py <ds> <prop> <out>"""
import sys, json, zipfile, os
from jarvis.db.figshare import data
LB="/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard"
ds, prop, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
bmz=None
for root,_,files in os.walk(f"{LB}/benchmarks"):
    if f"{ds}_{prop}.json.zip" in files: bmz=os.path.join(root,f"{ds}_{prop}.json.zip")
if not bmz: raise SystemExit(f"no benchmark {ds}_{prop}")
bm=json.loads(zipfile.ZipFile(bmz).read(f"{ds}_{prop}.json"))
tgt={}; split={}
for s in ("train","val","test"):
    if s in bm and isinstance(bm[s],dict):
        split[s]=list(bm[s]); tgt.update(bm[s])
recs_raw=data(ds)
# detect id key by matching benchmark ids
# detect id key: the record field whose values match the benchmark ids
sample=set(list(tgt)[:200])
idkey=None; best=0
for k in recs_raw[0].keys():
    if k=="atoms": continue
    hits=sum(1 for r in recs_raw[:500] if str(r.get(k)) in sample)
    if hits>best: best=hits; idkey=k
if not idkey: idkey="jid"
d={str(e[idkey]):e for e in recs_raw if idkey in e}
recs=[{"jid":i,"atoms":d[i]["atoms"],"target":float(tgt[i])} for i in tgt if i in d]
json.dump(recs, open(f"{out}/id_prop.json","w"))
json.dump({k:[i for i in v if i in d] for k,v in split.items()}, open(f"{out}/split.json","w"))
print(f"{ds}_{prop}: idkey={idkey} {len(recs)} recs | split", {k:len(v) for k,v in split.items()})
