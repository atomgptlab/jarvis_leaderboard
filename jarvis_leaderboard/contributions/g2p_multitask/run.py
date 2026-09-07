"""Generate JARVIS-Leaderboard benchmarks and contributions for bacterial G2P.

Adds a new AI/SinglePropertyClass dataset: eight binary phenotypes predicted from
Pfam protein-domain content, for 13,669 bacterial strains (BacDive-AI release).

Two dataset variants are registered deliberately:
    bacdive_<trait>        random train/val/test  -- the protocol prior work uses
    bacdive_clade_<trait>  whole gene-content clades held out

The pair exists because the split protocol, not the model, dominates the result:
strains of one species share ~99% of gene content, so a random split places
near-duplicates on both sides. Registering both makes that measurable on the
leaderboard rather than an assertion.

Source data: https://github.com/JKoblitz/bacdive-AI  (Koblitz et al. 2025)
"""
import json, pathlib, zipfile, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC = pathlib.Path("/home/kamalch/work/Overleaf/G2P/6a99358ac5929d3462371e07/scripts")
LB = pathlib.Path("/home/kamalch/Software/slako312/jarvis_leaderboard/jarvis_leaderboard")
BENCH = LB/"benchmarks/AI/SinglePropertyClass"
SLUG = {"Acidophilic":"acidophilic","Gram-positive":"gram_positive","Spore-forming":"spore_forming",
        "Aerobic":"aerobic","Anaerobic":"anaerobic","Thermophilic":"thermophilic",
        "Psychrophilic":"psychrophilic","Flagellated motility":"motility"}

X = pd.read_parquet(SRC/"data/features_pfam.parquet")
Y = pd.read_parquet(SRC/"data/labels.parquet")
TRAITS = list(Y.columns)
Xv = np.clip(X.to_numpy(np.float32), 0, 3)
Yv = Y.to_numpy(np.float32); M = ~np.isnan(Yv); Yv = np.nan_to_num(Yv)
pos = {s:i for i,s in enumerate(X.index)}

class Net(nn.Module):
    def __init__(s,d,k):
        super().__init__()
        s.f=nn.Sequential(nn.Linear(d,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(.4),
                          nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.3),
                          nn.Linear(256,k))
    def forward(s,x): return s.f(x)

def fit_multitask(tr):
    torch.manual_seed(0)
    mu,sd = Xv[tr].mean(0), Xv[tr].std(0)+1e-6
    Xt=torch.tensor((Xv[tr]-mu)/sd); Yt=torch.tensor(Yv[tr]); Mt=torch.tensor(M[tr].astype(np.float32))
    pw=torch.tensor([((M[tr][:,j]&(Yv[tr][:,j]==0)).sum())/max((M[tr][:,j]&(Yv[tr][:,j]==1)).sum(),1)
                     for j in range(len(TRAITS))],dtype=torch.float32)
    net=Net(Xv.shape[1],len(TRAITS)); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lf=nn.BCEWithLogitsLoss(reduction="none",pos_weight=pw)
    for _ in range(80):
        net.train(); pm=torch.randperm(len(Xt))
        for i in range(0,len(pm),256):
            b=pm[i:i+256]
            if len(b)<2: continue
            opt.zero_grad(); ((lf(net(Xt[b]),Yt[b])*Mt[b]).sum()/Mt[b].sum()).backward(); opt.step()
    net.eval(); return net,mu,sd

def write_zip(path, inner_name, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z: z.writestr(inner_name,text)

rng = np.random.default_rng(0)
for split_file, ds in [("random_seed0.json","bacdive"), ("clade_holdout_L2.json","bacdive_clade")]:
    sp = json.load(open(SRC/"data/splits"/split_file))
    tr_all = np.array([pos[s] for s in sp["train"]]); te = np.array([pos[s] for s in sp["test"]])
    p = rng.permutation(len(tr_all)); nv = int(0.15*len(tr_all))
    val, tr = tr_all[p[:nv]], tr_all[p[nv:]]
    net, mu, sd = fit_multitask(tr)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.tensor((Xv[te]-mu)/sd))).numpy()
    print(f"{ds}: train {len(tr)} val {len(val)} test {len(te)}", flush=True)

    for j,t in enumerate(TRAITS):
        slug = SLUG[t]
        idx = {k: [i for i in v if M[i,j]] for k,v in [("train",tr),("val",val),("test",te)]}
        if len(idx["test"]) < 20: continue
        bench = {k: {str(X.index[i]): int(Yv[i,j]) for i in v} for k,v in idx.items()}
        write_zip(BENCH/f"{ds}_{slug}.json.zip", f"{ds}_{slug}.json", json.dumps(bench))

        ids = [str(X.index[i]) for i in idx["test"]]
        rows = ["id,prediction"]
        for i,gi in enumerate(idx["test"]):
            rows.append(f"{X.index[gi]},{float(prob[list(te).index(gi),j] > 0.5)}")
        name = f"AI-SinglePropertyClass-{slug}-{ds}-test-acc.csv"
        write_zip(LB/"contributions/g2p_multitask"/(name+".zip"), name, "\n".join(rows)+"\n")

        lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
        lr.fit(Xv[idx["train"]], Yv[idx["train"],j].astype(int))
        pl = lr.predict(Xv[idx["test"]])
        rows = ["id,prediction"] + [f"{X.index[gi]},{float(v)}" for gi,v in zip(idx["test"],pl)]
        write_zip(LB/"contributions/g2p_logistic"/(name+".zip"), name, "\n".join(rows)+"\n")
        print(f"  {slug:14s} train {len(idx['train']):5d} test {len(idx['test']):5d}", flush=True)
print("done")
