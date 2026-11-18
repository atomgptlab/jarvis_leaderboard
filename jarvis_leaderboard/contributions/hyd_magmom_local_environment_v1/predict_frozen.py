"""Replay a frozen material classifier from structure-only inputs.

Input JSONL records must contain exactly id and atoms. Optional cached features
must have ids, X and feature_names; target arrays are never accessed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_matrix(path, components):
    from cache_matminer import composition_featurizer
    from cache_geometry import featurize as geometry, names as geometry_names
    from features import featurize as basic, feature_names as basic_names
    from pymatgen.core import Composition, Structure
    if components != ["matminer145", "basic129", "geometry160"] and components != ["matminer145", "basic129", "geometry160", "matminer_structure128"]:
        raise ValueError("Unsupported frozen feature order")
    composition = composition_featurizer()
    feature_names = composition.feature_labels() + basic_names("structure") + geometry_names()
    structure = None
    if "matminer_structure128" in components:
        from cache_matminer_structure import make_featurizer
        structure = make_featurizer()
        feature_names += structure.feature_labels()
    memo = {}
    ids, rows = [], []
    with Path(path).open() as handle:
        for line in handle:
            item = json.loads(line)
            if set(item) != {"id", "atoms"}:
                raise ValueError("Inference inputs must be structure-only, without labels or material properties")
            key = tuple(sorted(Counter(item["atoms"]["elements"]).items()))
            if key not in memo:
                memo[key] = composition.featurize(Composition(dict(key)))
            values = [*memo[key], *basic(item["atoms"], "structure"), *geometry(item)[1]]
            if structure is not None:
                a = item["atoms"]
                s = Structure(a["lattice_mat"], a["elements"], a["coords"], coords_are_cartesian=a["cartesian"], to_unit_cell=True)
                try:
                    extra = np.asarray(structure.featurize(s), dtype=float)
                    extra[~np.isfinite(extra)] = np.nan
                except Exception:
                    extra = np.full(128, np.nan)
                values.extend(extra)
            ids.append(item["id"])
            rows.append(values)
    return np.asarray(ids), np.asarray(rows, dtype=float), feature_names


def predict(selection_path, ids, x, feature_names):
    selection_path = Path(selection_path)
    frozen = json.loads(selection_path.read_text())
    if feature_names != frozen["feature_names"]:
        raise ValueError("Inference feature order differs from training")
    result = np.zeros(len(ids))
    weight = 0.
    for member in frozen["members"]:
        path = selection_path.parent / member["path"]
        if sha256(path) != member["sha256"]:
            raise ValueError("Model checksum mismatch")
        if member["kind"] == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(path)
            model.set_params(device="cpu", n_jobs=4)
        elif member["kind"] == "hgb":
            import joblib
            model = joblib.load(path)["model"]
        elif member["kind"] == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(thread_count=4)
            model.load_model(str(path))
        else:
            raise ValueError("Unknown model family")
        result += member["weight"] * model.predict_proba(x)[:, 1]
        weight += member["weight"]
    if not np.isclose(weight, 1.):
        raise ValueError("Frozen ensemble weights must sum to one")
    return result, (result >= frozen["threshold"]).astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--structures", type=Path)
    inputs.add_argument("--features", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = json.loads(args.selection.read_text())
    if args.features:
        with np.load(args.features, allow_pickle=False) as z:
            ids, x, names = z["ids"], z["X"], z["feature_names"].tolist()
    else:
        ids, x, names = feature_matrix(args.structures, frozen["components"])
    if len(set(ids)) != len(ids) or len(x) != len(ids) or np.isinf(x).any():
        raise ValueError("Invalid inference features")
    probabilities, predictions = predict(args.selection, ids, x, names)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        w.writerows(zip(ids.tolist(), predictions.tolist()))
    print(json.dumps({"rows": len(ids), "output_sha256": sha256(args.output),
                      "selection_sha256": sha256(args.selection), "labels_read": False}))


if __name__ == "__main__":
    main()
