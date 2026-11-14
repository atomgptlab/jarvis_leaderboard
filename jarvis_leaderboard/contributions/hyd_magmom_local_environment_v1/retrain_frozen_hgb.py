"""Refit frozen HGB parameters using only the included official training labels."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from predict_frozen import feature_matrix
from replay_benchmark import digest, raw_database, write_structures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".training-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())
    if len(selected["members"]) != 1 or selected["members"][0]["kind"] != "hgb":
        raise ValueError("This training replay supports the single frozen HGB candidate only")
    config = json.loads(args.training_config.read_text())
    labels = json.loads(args.training_labels.read_text())
    if len(labels) != 41766 or set(labels.values()) != {0, 1}:
        raise ValueError("Expected the 41,766 original training labels")
    raw_path = raw_database(args.raw_json, args.cache)
    structures = args.cache / "training-structures.jsonl"
    write_structures(list(labels), raw_path, structures)
    started = time.monotonic()
    ids, x, names = feature_matrix(structures, selected["components"])
    if names != selected["feature_names"] or ids.tolist() != list(labels):
        raise ValueError("Training feature order differs from the frozen selection")
    y = np.asarray([labels[jid] for jid in ids])
    model = HistGradientBoostingClassifier(**config["final_model_parameters"])
    with threadpool_limits(limits=4):
        model.fit(x, y)
    metadata = {"feature_names": names, "threshold": selected["threshold"],
                "feature_set": "matminer145", "append_basic": True, "append_geometry": True,
                "append_matminer_structure": "matminer_structure128" in selected["components"]}
    joblib.dump({"model": model, **metadata}, args.output)
    report = {"rows": len(ids), "features": len(names), "seconds": time.monotonic()-started,
              "training_labels_sha256": digest(args.training_labels),
              "training_config_sha256": digest(args.training_config), "model_sha256": digest(args.output),
              "official_validation_labels_read": False, "test_labels_read": False,
              "parameters_frozen_before_retraining": True}
    args.output.with_name("training-replay-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
