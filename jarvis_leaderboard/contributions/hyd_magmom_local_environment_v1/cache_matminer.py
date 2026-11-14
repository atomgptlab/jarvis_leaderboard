"""Cache the published 145 composition descriptors from sanitized structures.

Defaults to train and validation only. No material-property targets are used
as descriptors. NaNs remain for training-only imputation or native tree support.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jarvis-mpl-cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, IonProperty, Stoichiometry, ValenceOrbital
from pymatgen.core import Composition
from features import ATOM_FIELDS
from prepare_data import digest

ROOT = Path(__file__).resolve().parents[1]


def composition_featurizer():
    # Preserve missing elemental properties, as in the original implementation.
    return MultipleFeaturizer([
        Stoichiometry(), ElementProperty.from_preset("magpie", impute_nan=False),
        ValenceOrbital(props=["frac"], impute_nan=False), IonProperty(fast=True, impute_nan=False),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    args = parser.parse_args()
    f = composition_featurizer()
    names = f.feature_labels()
    if len(names) != 145:
        raise ValueError(f"Expected 145 published composition features, got {len(names)}")
    output = ROOT / "artifacts/features/matminer145"
    output.mkdir(parents=True, exist_ok=True)
    memo = {}
    for split in args.splits:
        path = ROOT / "artifacts/prepared" / f"{split}.jsonl"
        target = output / f"{split}.npz"
        manifest = output / f"{split}.json"
        if target.exists() or manifest.exists():
            raise FileExistsError(f"Refusing to overwrite existing feature cache: {target}")
        started = time.monotonic()
        ids, rows, labels = [], [], []
        with path.open() as handle:
            for i, line in enumerate(handle, 1):
                item = json.loads(line)
                expected = {"id", "atoms"} if split == "test" else {"id", "atoms", "target"}
                if set(item) != expected or set(item["atoms"]) != ATOM_FIELDS:
                    raise ValueError("Unsanitized input fields")
                key = tuple(sorted(Counter(item["atoms"]["elements"]).items()))
                if key not in memo:
                    memo[key] = np.asarray(f.featurize(Composition(dict(key))), dtype=float)
                ids.append(item["id"])
                rows.append(memo[key])
                if split != "test":
                    labels.append(item["target"])
                if i % 5000 == 0:
                    print(f"{split}: {i} rows, {len(memo)} compositions, {time.monotonic()-started:.1f}s", flush=True)
        x = np.asarray(rows)
        if len(ids) != len(set(ids)) or np.isinf(x).any():
            raise ValueError("Duplicate IDs or infinite features")
        arrays = {"ids": np.asarray(ids), "X": x, "feature_names": np.asarray(names)}
        if split != "test":
            arrays["y"] = np.asarray(labels, dtype=np.int64)
        temp = target.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(target)
        meta = {
            "feature_set": "matminer145", "shape": list(x.shape),
            "input_sha256": digest(path), "cache_sha256": digest(target),
            "feature_source_sha256": digest(Path(__file__)),
            "packages": {p: importlib.metadata.version(p) for p in ("matminer", "pymatgen", "numpy")},
            "nan_rows": int(np.isnan(x).any(axis=1).sum()),
            "nan_by_feature": {name: int(np.isnan(x[:, k]).sum()) for k, name in enumerate(names)
                               if np.isnan(x[:, k]).any()},
            "seconds": time.monotonic()-started, "test_labels_accessed": False,
            "reference": "Official matminer_xgboost/run.py composition descriptors only; no 128 structural descriptors yet",
        }
        manifest.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
