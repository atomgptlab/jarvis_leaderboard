"""Resumable 128 MatMiner structural descriptors using published feature families.

Only sanitized structures are read. Library versions differ from the historical
submission, so this is a feature-family reproduction, not an exact score replay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jarvis-mpl-cache")
import numpy as np
from prepare_data import digest
from features import ATOM_FIELDS

ROOT = Path(__file__).resolve().parents[1]
FEATURIZER = None


def make_featurizer():
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.structure import SiteStatsFingerprint, StructuralHeterogeneity, MaximumPackingEfficiency, ChemicalOrdering
    return MultipleFeaturizer([
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructuralHeterogeneity(), MaximumPackingEfficiency(), ChemicalOrdering()])


def timeout_handler(signum, frame):
    raise TimeoutError("Structural featurization exceeded 20s")


def initialize():
    global FEATURIZER
    warnings.simplefilter("ignore")
    FEATURIZER = make_featurizer()
    signal.signal(signal.SIGALRM, timeout_handler)


def process(item):
    from pymatgen.core import Structure
    if set(item["atoms"]) != ATOM_FIELDS or set(item) != {"id", "atoms", "target"}:
        raise ValueError("Unsanitized data")
    signal.alarm(20)
    error = ""
    try:
        a = item["atoms"]
        s = Structure(a["lattice_mat"], a["elements"], a["coords"], coords_are_cartesian=a["cartesian"], to_unit_cell=True)
        values = np.asarray(FEATURIZER.featurize(s), dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        if len(values) != 128:
            raise ValueError("Feature dimensions changed")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        values = np.full(128, np.nan, dtype=np.float32)
    finally:
        signal.alarm(0)
    return item["id"], values, item["target"], error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = ROOT / "artifacts/features/matminer_structure128"
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feature_names = make_featurizer().feature_labels()
    assert len(feature_names) == 128
    script_hash = digest(Path(__file__))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        for split in ("train", "val"):
            source = ROOT / "artifacts/prepared" / f"{split}.jsonl"
            items = [json.loads(line) for line in source.open()]
            parts = output / f"{split}-parts"
            parts.mkdir(exist_ok=True)
            marker = parts / "manifest.json"
            manifest = {"source_sha256": digest(source), "script_sha256": script_hash,
                        "feature_names": feature_names, "rows": len(items), "chunk_size": 500}
            if marker.exists() and json.loads(marker.read_text()) != manifest:
                raise ValueError("Partial cache source changed")
            marker.write_text(json.dumps(manifest, indent=2) + "\n")
            started = time.monotonic()
            for i in range(0, len(items), 500):
                part = parts / f"{i // 500:04d}.npz"
                if part.exists():
                    with np.load(part, allow_pickle=False) as z:
                        assert z["ids"].tolist() == [row["id"] for row in items[i:i+500]]
                    continue
                rows = list(pool.map(process, items[i:i+500], chunksize=4))
                temp = part.with_suffix(".tmp")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, ids=np.asarray([r[0] for r in rows]),
                        X=np.asarray([r[1] for r in rows]), y=np.asarray([r[2] for r in rows]),
                        errors=np.asarray([r[3] for r in rows]))
                temp.replace(part)
                print(f"{split}: {min(i+500,len(items))}/{len(items)} rows, {time.monotonic()-started:.1f}s", flush=True)
            arrays = {key: [] for key in ("ids", "X", "y", "errors")}
            for part in sorted(parts.glob("*.npz")):
                with np.load(part, allow_pickle=False) as z:
                    for key in arrays:
                        arrays[key].append(z[key])
            arrays = {key: np.concatenate(value) for key, value in arrays.items()}
            assert arrays["ids"].tolist() == [row["id"] for row in items]
            arrays["feature_names"] = np.asarray(feature_names)
            target = output / f"{split}.npz"
            np.savez_compressed(target, **arrays)
            meta = {**manifest, "cache_sha256": digest(target), "shape": list(arrays["X"].shape),
                    "failed_rows": int(np.count_nonzero(arrays["errors"])),
                    "packages": {name: importlib.metadata.version(name) for name in ("matminer", "pymatgen", "numpy")},
                    "test_accessed": False}
            target.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
            print(json.dumps({key: meta[key] for key in ("shape", "failed_rows", "test_accessed")}), flush=True)


if __name__ == "__main__":
    main()
