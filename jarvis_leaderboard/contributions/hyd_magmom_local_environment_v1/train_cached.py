"""Train material descriptors with validation-only reporting and provenance."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from features import load_features, feature_names
from prepare_data import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--append-basic", action="store_true")
    parser.add_argument("--append-geometry", action="store_true")
    parser.add_argument("--append-matminer-structure", action="store_true")
    parser.add_argument("--refit-full", action="store_true", help="Choose HGB iterations internally, then refit all training rows")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--leaf-nodes", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=.1)
    parser.add_argument("--l2", type=float, default=1.)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Invalid run name")
    output = ROOT / "artifacts/runs" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in ("train_cached.py", "cache_matminer.py", "features.py", "prepare_data.py", "audit_structure_duplicates.py", "cache_geometry.py", "cache_matminer_structure.py"):
        shutil.copy2(ROOT / "scripts" / name, source / name)
    data = {}
    manifests = {}
    names = None
    for split in ("train", "val"):
        path = ROOT / "artifacts/features/matminer145" / f"{split}.npz"
        meta = json.loads(path.with_suffix(".json").read_text())
        if digest(path) != meta["cache_sha256"] or digest(ROOT / "artifacts/prepared" / f"{split}.jsonl") != meta["input_sha256"]:
            raise ValueError("Stale or damaged feature cache")
        with np.load(path, allow_pickle=False) as z:
            ids, x, y = z["ids"].tolist(), z["X"], z["y"]
            split_names = z["feature_names"].tolist()
        if args.append_basic:
            basic_ids, basic_x, basic_y = load_features(ROOT / "artifacts/prepared" / f"{split}.jsonl", "structure", True)
            if ids != basic_ids or not np.array_equal(y, basic_y):
                raise ValueError("Feature alignment mismatch")
            x = np.column_stack([x, basic_x])
            split_names += feature_names("structure")
        if args.append_geometry:
            geometry_path = ROOT / "artifacts/features/geometry160" / f"{split}.npz"
            geometry_meta = json.loads(geometry_path.with_suffix(".json").read_text())
            if digest(geometry_path) != geometry_meta["cache_sha256"] or digest(prepared := ROOT / "artifacts/prepared" / f"{split}.jsonl") != geometry_meta["source_sha256"]:
                raise ValueError("Geometry cache integrity check failed")
            with np.load(geometry_path, allow_pickle=False) as z:
                if z["ids"].tolist() != ids or not np.array_equal(z["y"], y):
                    raise ValueError("Geometry alignment mismatch")
                x = np.column_stack([x, z["X"]])
                split_names += z["feature_names"].tolist()
            manifests[f"geometry_{split}"] = geometry_meta
        if args.append_matminer_structure:
            structural_path = ROOT / "artifacts/features/matminer_structure128" / f"{split}.npz"
            structural_meta = json.loads(structural_path.with_suffix(".json").read_text())
            if digest(structural_path) != structural_meta["cache_sha256"] or digest(ROOT / "artifacts/prepared" / f"{split}.jsonl") != structural_meta["source_sha256"]:
                raise ValueError("MatMiner structural cache integrity check failed")
            with np.load(structural_path, allow_pickle=False) as z:
                if z["ids"].tolist() != ids or not np.array_equal(z["y"], y):
                    raise ValueError("MatMiner structure alignment mismatch")
                x = np.column_stack([x, z["X"]])
                split_names += z["feature_names"].tolist()
            manifests[f"matminer_structure_{split}"] = structural_meta
        if names is not None and split_names != names:
            raise ValueError("Feature columns do not match")
        names = split_names
        data[split] = ids, x, y
        manifests[split] = meta
    train_ids, x_train, y_train = data["train"]
    val_ids, x_val, y_val = data["val"]
    if set(train_ids) & set(val_ids):
        raise ValueError("Overlapping split IDs")
    duplicates_path = ROOT / "artifacts/prepared/structure_duplicates.json"
    duplicates = json.loads(duplicates_path.read_text())
    repeated_val = set(duplicates["cross_split"]["train/val"]["right_ids"])
    novel = np.asarray([jid not in repeated_val for jid in val_ids])
    if args.model == "hgb":
        model = HistGradientBoostingClassifier(max_iter=args.max_iter, learning_rate=args.learning_rate,
            max_leaf_nodes=args.leaf_nodes, random_state=args.seed, early_stopping=True,
            validation_fraction=0.1, scoring="loss", n_iter_no_change=30, l2_regularization=args.l2)
    else:
        # Original RF hyperparameters; composition-only features are a partial reproduction.
        model = make_pipeline(SimpleImputer(), StandardScaler(), RandomForestClassifier(
            n_estimators=100, max_features=1/3, bootstrap=False, random_state=args.seed, n_jobs=args.threads))
    config = {
        **vars(args), "started_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(names), "feature_manifests": manifests,
        "model_parameters": {k: v for k, v in model.get_params(deep=True).items()
                             if isinstance(v, (str, int, float, bool, type(None)))},
        "source_sha256": {p.name: digest(p) for p in source.iterdir()},
        "structure_audit_sha256": digest(duplicates_path),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "matminer", "pymatgen", "joblib")},
        "python": platform.python_version(), "platform": platform.platform(),
        "threshold": 0.5, "selection_metric": "validation_novel_structure_accuracy",
        "test_accessed": False, "published_baseline_fully_reproduced": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Training {args.model}: {x_train.shape}, validation {x_val.shape}", flush=True)
    started = time.monotonic()
    with threadpool_limits(limits=args.threads):
        model.fit(x_train, y_train)
        if args.refit_full:
            if args.model != "hgb":
                raise ValueError("Full refit selection is implemented for HGB only")
            selected_iterations = int(model.n_iter_)
            model = HistGradientBoostingClassifier(**{
                **model.get_params(), "max_iter": selected_iterations, "early_stopping": False})
            model.fit(x_train, y_train)
            config["selected_iterations_from_training_internal_holdout"] = selected_iterations
            config["final_fit_uses_all_training_rows"] = True
            config["final_model_parameters"] = model.get_params()
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        fit_seconds = time.monotonic()-started
        probability = model.predict_proba(x_val)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "validation_accuracy": float(accuracy_score(y_val, prediction)),
        "validation_novel_structure_accuracy": float(accuracy_score(y_val[novel], prediction[novel])),
        "validation_novel_structure_count": int(novel.sum()),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, prediction)),
        "validation_f1": float(f1_score(y_val, prediction)),
        "validation_log_loss": float(log_loss(y_val, probability)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_val, prediction, labels=[0, 1]).tolist(),
        "fit_seconds": fit_seconds, "train_count": len(train_ids), "validation_count": len(val_ids),
        "test_evaluated": False,
    }
    if args.model == "hgb":
        metrics["iterations"] = int(model.n_iter_)
    with (output / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target", "probability", "prediction"])
        writer.writerows(zip(val_ids, y_val.tolist(), probability.tolist(), prediction.tolist()))
    joblib.dump({"model": model, "feature_names": names, "threshold": 0.5,
                 "feature_set": "matminer145", "append_basic": args.append_basic,
                 "append_geometry": args.append_geometry,
                 "append_matminer_structure": args.append_matminer_structure}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
