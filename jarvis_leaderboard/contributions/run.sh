#!/bin/bash
# ============================================================
# Reproduce element-fraction GradientBoostingRegressor results
# for JARVIS-Leaderboard: formation_energy_peratom on dft_3d
#
# Contribution: element_fraction_gbr
# Author: Peter Muskett
# ============================================================

# --- 1. Install dependencies ---
pip install jarvis-tools scikit-learn numpy pandas

# --- 2. Run the model ---
python3 - <<'EOF'
import numpy as np
import pandas as pd
from jarvis.db.figshare import data
from jarvis.core.composition import Composition
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import zipfile, csv, io

# Load datasets
print("Loading cfid_3d ...")
cfid_3d = data('cfid_3d')
df = pd.DataFrame(cfid_3d)

print("Loading dft_3d_2021 for jid->formula mapping ...")
dft_3d = data('dft_3d_2021')
jid_to_formula = {entry['jid']: entry['formula'] for entry in dft_3d}

# Property and valid data ranges (matching the original notebook)
ml_property = 'formation_energy_peratom'
typical_data_ranges = {
    'formation_energy_peratom': [-5, 5]
}

# Build element-fraction feature matrix
print("Building element-fraction features ...")
x_ef, y_ef, jids_ef = [], [], []
df2 = df[['jid', ml_property]].replace('na', np.nan).dropna()
for _, row in df2.iterrows():
    jid = row['jid']
    if jid not in jid_to_formula:
        continue
    try:
        feat = Composition.from_string(jid_to_formula[jid]).atomic_fraction_array
    except Exception:
        continue
    val = float(row[ml_property])
    lo, hi = typical_data_ranges[ml_property]
    if val != float('inf') and lo < val < hi:
        x_ef.append(feat)
        y_ef.append(val)
        jids_ef.append(jid)

x_ef = np.array(x_ef, dtype='float')
y_ef = np.array(y_ef, dtype='float')
print(f"Dataset size: {x_ef.shape}")

# 90/10 train-test split (random_state=1 matches notebook)
X_train, X_test, y_train, y_test, jid_train, jid_test = train_test_split(
    x_ef, y_ef, jids_ef, random_state=1, test_size=0.1
)

# Subset to 500 samples (matching notebook)
X1, Y1 = X_train[:500], y_train[:500]
X2, Y2 = X_test[:500],  y_test[:500]
jid_test_500 = jid_test[:500]

# Train with default GradientBoostingRegressor
print("Training model ...")
model = GradientBoostingRegressor()
model.fit(X1, Y1)

# Predict
pred = model.predict(X2)
mae = mean_absolute_error(Y2, pred)
print(f"MAE on test set: {mae:.4f} eV/atom")

# Write contribution CSV (id, prediction columns)
csv_filename = "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv"
zip_filename = csv_filename + ".zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "prediction"])
    for jid, p in zip(jid_test_500, pred):
        writer.writerow([jid, round(float(p), 6)])
    zf.writestr(csv_filename, buf.getvalue())

print(f"Saved: {zip_filename}")
EOF
