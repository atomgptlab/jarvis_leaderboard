
import os
import json
import zipfile
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from jarvis.db.figshare import data
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors


def safe_cfid(x):
    try:
        out = get_chem_only_descriptors(x)
        if isinstance(out, (list, tuple)):
            return np.array(out[0], dtype=float)
        return np.array(out, dtype=float)
    except Exception as e:
        print("Descriptor failed for:", x, "| error:", e)
        return None


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    benchmark_zip = os.path.join(
        repo_root,
        "jarvis_leaderboard",
        "benchmarks",
        "AI",
        "SinglePropertyPrediction",
        "ssub_formula_energy.json.zip"
    )

    with zipfile.ZipFile(benchmark_zip, "r") as zf:
        split_info = json.loads(zf.read("ssub_formula_energy.json").decode("utf-8"))

    train_dict = split_info["train"]
    test_dict = split_info["test"]

    df_full = pd.DataFrame(data("ssub"))

    required_candidates = {
        "id": ["id", "jid"],
        "formula": ["formula"],
        "composition": ["composition"],
        "target": ["formula_energy", "form_energy"]
    }

    resolved = {}
    for key, candidates in required_candidates.items():
        for c in candidates:
            if c in df_full.columns:
                resolved[key] = c
                break

    id_col = resolved["id"]
    target_col = resolved["target"]
    desc_source_col = "composition" if "composition" in resolved else "formula"

    df_full["id_std"] = df_full[id_col].astype(str)
    df_full["target_std"] = df_full[target_col].astype(float)
    df_full["desc_source"] = df_full[desc_source_col].astype(str)

    train_ids = list(train_dict.keys())
    test_ids = list(test_dict.keys())

    train_df = df_full[df_full["id_std"].isin(train_ids)].copy()
    test_df = df_full[df_full["id_std"].isin(test_ids)].copy()

    train_df = train_df.set_index("id_std").loc[train_ids].reset_index()
    test_df = test_df.set_index("id_std").loc[test_ids].reset_index()

    train_df["cfid_desc"] = train_df["desc_source"].apply(safe_cfid)
    test_df["cfid_desc"] = test_df["desc_source"].apply(safe_cfid)

    train_df = train_df[train_df["cfid_desc"].notnull()].copy()
    test_df = test_df[test_df["cfid_desc"].notnull()].copy()

    X_train = np.vstack(train_df["cfid_desc"].values)
    y_train = train_df["target_std"].values

    X_test = np.vstack(test_df["cfid_desc"].values)
    y_test = test_df["target_std"].values

    rf = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("CFID Test MAE =", mae)

    model_path = os.path.join(current_dir, "rf_form_energy_model_cfid.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(rf, f)

    csv_name = "AI-SinglePropertyPrediction-formula_energy-ssub-test-mae.csv"
    csv_path = os.path.join(current_dir, csv_name)

    results = pd.DataFrame({
        "id": test_df["id_std"].values,
        "target": y_test,
        "prediction": y_pred
    })
    results.to_csv(csv_path, index=False)

    zip_path = csv_path + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_name)

    print("saved:", zip_path)


if __name__ == "__main__":
    main()
