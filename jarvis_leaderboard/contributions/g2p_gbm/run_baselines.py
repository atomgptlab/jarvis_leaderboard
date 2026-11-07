"""Add single-task baselines: random forest, gradient boosting, MLP.

The MLP matters most: same 512/256 hidden sizes as the multi-task trunk but fit
per trait, so the difference between g2p_mlp and g2p_multitask isolates the
effect of cross-trait sharing from the effect of using a neural network at all.
"""

import os
import json
import pathlib
import zipfile
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Source data: the BacDive-AI release, rebuilt into a strain x Pfam table.
# Set G2P_SRC to the directory holding data/features_pfam.parquet,
# data/labels.parquet and data/splits/*.json.  Raw release:
# https://github.com/JKoblitz/bacdive-AI
SRC = pathlib.Path(
    os.environ.get(
        "G2P_SRC", pathlib.Path(__file__).resolve().parents[3] / "g2p_data"
    )
)
LB = (
    pathlib.Path(__file__).resolve().parents[2]
)  # the jarvis_leaderboard package dir
SLUG = {
    "Acidophilic": "acidophilic",
    "Gram-positive": "gram_positive",
    "Spore-forming": "spore_forming",
    "Aerobic": "aerobic",
    "Anaerobic": "anaerobic",
    "Thermophilic": "thermophilic",
    "Psychrophilic": "psychrophilic",
    "Flagellated motility": "motility",
}

X = pd.read_parquet(SRC / "data/features_pfam.parquet")
Y = pd.read_parquet(SRC / "data/labels.parquet")
TRAITS = list(Y.columns)
Xv = np.clip(X.to_numpy(np.float32), 0, 3)
Yv = Y.to_numpy(np.float32)
M = ~np.isnan(Yv)
Yv = np.nan_to_num(Yv)
pos = {s: i for i, s in enumerate(X.index)}


def models():
    return {
        "g2p_rf": RandomForestClassifier(
            max_depth=10, n_estimators=200, n_jobs=8, random_state=0
        ),
        "g2p_gbm": HistGradientBoostingClassifier(random_state=0),
        "g2p_mlp": make_pipeline(
            StandardScaler(with_mean=False),
            MLPClassifier(
                hidden_layer_sizes=(512, 256),
                alpha=1e-4,
                max_iter=300,
                early_stopping=True,
                random_state=0,
            ),
        ),
    }


def write_zip(path, inner, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner, text)


rng = np.random.default_rng(0)
for split_file, ds in [
    ("random_seed0.json", "bacdive"),
    ("clade_holdout_L2.json", "bacdive_clade"),
]:
    sp = json.load(open(SRC / "data/splits" / split_file))
    tr_all = np.array([pos[s] for s in sp["train"]])
    te = np.array([pos[s] for s in sp["test"]])
    p = rng.permutation(len(tr_all))
    nv = int(0.15 * len(tr_all))
    tr = tr_all[p[nv:]]
    print(f"\n=== {ds} ===", flush=True)
    for j, t in enumerate(TRAITS):
        slug = SLUG[t]
        m_tr = M[tr][:, j]
        m_te = M[te][:, j]
        if m_te.sum() < 20:
            continue
        Xtr, ytr = Xv[tr][m_tr], Yv[tr][m_tr, j].astype(int)
        Xte = Xv[te][m_te]
        ids = [X.index[i] for i in te[m_te]]
        line = f"  {slug:14s} tr={len(ytr):5d} te={len(ids):5d}"
        for name, clf in models().items():
            t0 = time.time()
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            rows = ["id,prediction"] + [
                f"{i},{float(v)}" for i, v in zip(ids, pred)
            ]
            fn = f"AI-SinglePropertyClass-{slug}-{ds}-test-acc.csv"
            write_zip(
                LB / "contributions" / name / (fn + ".zip"),
                fn,
                "\n".join(rows) + "\n",
            )
            short = name.replace("g2p_", "")
            line += f" | {short:3s} {time.time() - t0:4.0f}s"
        print(line, flush=True)
print("\ndone")
