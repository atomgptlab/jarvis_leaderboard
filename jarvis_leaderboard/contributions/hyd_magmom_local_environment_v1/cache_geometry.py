"""Periodic local-environment descriptors from sanitized atom positions only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/jarvis-mpl-cache")
import numpy as np
from pymatgen.core import Element, Structure
from features import ATOM_FIELDS
from prepare_data import digest

ROOT = Path(__file__).resolve().parents[1]
RADII = (2.5, 3., 3.5, 4., 5., 6.)
EDGES = np.linspace(0., 6., 25)


def names():
    result = []
    for channel in ("all", "df_center", "df_pair"):
        result += [f"rdf_{channel}_{edge:.2f}_{EDGES[i+1]:.2f}" for i, edge in enumerate(EDGES[:-1])]
    for radius in RADII:
        for channel in ("all", "df_center", "df_pair"):
            result += [f"coord_{channel}_{radius:g}_{stat}" for stat in ("mean", "std", "min", "max")]
    for field in ("nearest_all", "nearest_df", "pauling_diff_all", "pauling_diff_df"):
        result += [f"{field}_{stat}" for stat in ("mean", "std", "min", "max")]
    return result


@lru_cache(maxsize=118)
def element_info(symbol):
    element = Element(symbol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi = float(element.X)
    return element.block in ("d", "f"), chi


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))] if len(values) else [np.nan] * 4


def featurize(item):
    atoms = item["atoms"]
    if set(atoms) != ATOM_FIELDS or set(item) not in ({"id", "atoms"}, {"id", "atoms", "target"}):
        raise ValueError("Only sanitized atomic structures may be featurized")
    s = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"],
                  coords_are_cartesian=atoms["cartesian"], to_unit_cell=True)
    center, neighbor, _, distance = s.get_neighbor_list(6., numerical_tol=1e-8, exclude_self=True)
    valid = distance > 1e-7
    center, neighbor, distance = center[valid], neighbor[valid], distance[valid]
    info = [element_info(symbol) for symbol in atoms["elements"]]
    df = np.array([row[0] for row in info], dtype=bool)
    chi = np.array([row[1] for row in info])
    masks = (np.ones(len(distance), dtype=bool), df[center], df[center] & df[neighbor])
    denominators = (len(s), max(1, int(df.sum())), max(1, int(df.sum())))
    output = []
    for mask, denominator in zip(masks, denominators):
        output.extend((np.histogram(distance[mask], bins=EDGES)[0] / denominator).tolist())
    for radius in RADII:
        for channel, mask in enumerate(masks):
            counts = np.bincount(center[mask & (distance <= radius)], minlength=len(s))
            output.extend(stats(counts if channel == 0 else counts[df]))
    nearest = np.full(len(s), np.inf)
    np.minimum.at(nearest, center, distance)
    output.extend(stats(nearest)); output.extend(stats(nearest[df]))
    delta = np.abs(chi[center] - chi[neighbor])
    output.extend(stats(delta)); output.extend(stats(delta[df[center]]))
    result = np.asarray(output, dtype=np.float32)
    if len(result) != len(names()) or np.isinf(result).any():
        raise ValueError("Invalid geometry feature vector")
    return item["id"], result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Timing-only subset; never writes production caches")
    args = p.parse_args()
    out = ROOT / "artifacts/features/geometry160"
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        source = ROOT / "artifacts/prepared" / f"{split}.jsonl"
        target = out / f"{split}.npz"
        if not args.limit and target.exists():
            raise FileExistsError(target)
        items = []
        with source.open() as handle:
            for i, line in enumerate(handle):
                if args.limit and i >= args.limit:
                    break
                items.append(json.loads(line))
        started = time.monotonic()
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(featurize, items, chunksize=40), 1):
                rows.append(result)
                if i % 2000 == 0:
                    print(f"{split} {i}/{len(items)} {time.monotonic()-started:.1f}s", flush=True)
        ids = np.asarray([row[0] for row in rows])
        x = np.asarray([row[1] for row in rows])
        arrays = {"ids": ids, "X": x, "feature_names": np.asarray(names())}
        if split != "test":
            arrays["y"] = np.asarray([row["target"] for row in items])
        report = {"split": split, "shape": list(x.shape), "seconds": time.monotonic()-started,
                  "source_sha256": digest(source), "script_sha256": digest(Path(__file__)),
                  "test_labels_accessed": False, "timing_only": bool(args.limit)}
        if not args.limit:
            np.savez_compressed(target, **arrays)
            report["cache_sha256"] = digest(target)
            target.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
