"""Audit official IDs and export structure-only inputs without test targets.

Uses the standard library only. Training programs should read these exports,
never the raw database or reference predictions. This checks ID separation;
it does not establish that different IDs have distinct crystal structures.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dft_3d_magmom_oszicar"
ATOM_FIELDS = ("lattice_mat", "coords", "elements", "cartesian")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_atoms(record):
    atoms = {key: record["atoms"][key] for key in ATOM_FIELDS}
    elements, coords, lattice = atoms["elements"], atoms["coords"], atoms["lattice_mat"]
    require(bool(elements) and len(elements) == len(coords), "Invalid atom count")
    require(all(isinstance(e, str) and e for e in elements), "Invalid element")
    require(isinstance(atoms["cartesian"], bool), "Invalid coordinate convention")
    require(len(lattice) == 3, "Invalid lattice")
    for row in [*lattice, *coords]:
        require(len(row) == 3 and all(math.isfinite(float(v)) for v in row),
                "Non-finite or malformed coordinates")
    return atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/prepared")
    args = parser.parse_args()
    raw_path = DATA / "01_raw_data/jdft_3d-12-12-2022.json"
    split_path = DATA / "02_benchmark_split/dft_3d_magmom_oszicar.json.zip"
    with zipfile.ZipFile(split_path) as archive:
        splits = json.loads(archive.read("dft_3d_magmom_oszicar.json"))
    require(set(splits) == {"train", "val", "test"}, "Unexpected split keys")
    # Deliberately discard test target values before any downstream processing.
    test_ids = list(splits.pop("test"))
    ids = {name: list(labels) for name, labels in splits.items()}
    ids["test"] = test_ids
    official_counts = {k: len(v) for k, v in ids.items()}
    train_val_overlap = sorted(set(ids["train"]).intersection(ids["val"]))
    require(all(splits["train"][jid] == splits["val"][jid] for jid in train_val_overlap),
            "Conflicting train/validation labels")
    # Preserve official train and test IDs; exclude seen IDs from INTERNAL
    # validation. Report this explicitly rather than silently changing the split.
    ids["val"] = [jid for jid in ids["val"] if jid not in set(train_val_overlap)]
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        require(not set(ids[left]).intersection(ids[right]), f"ID overlap: {left}/{right}")
    for labels in splits.values():
        require(set(labels.values()) <= {0, 1}, "Non-binary training/validation labels")
    raw = json.loads(raw_path.read_text())
    by_id = {record["jid"]: record for record in raw}
    require(len(by_id) == len(raw), "Duplicate raw IDs")
    missing = set().union(*map(set, ids.values())) - set(by_id)
    require(not missing, f"Missing {len(missing)} benchmark IDs")
    for name in ("train", "val"):
        mismatches = [jid for jid, target in splits[name].items()
                      if int(float(by_id[jid]["magmom_oszicar"]) > 0.05) != target]
        require(not mismatches, f"{name}: {len(mismatches)} label definition mismatches")
    # Validate all records before writing outputs, without copying raw properties.
    structures = {jid: clean_atoms(by_id[jid]) for values in ids.values() for jid in values}
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, values in ids.items():
        path = args.output / f"{name}.jsonl"
        temp = path.with_suffix(".jsonl.tmp")
        with temp.open("w") as handle:
            for jid in values:
                item = {"id": jid, "atoms": structures[jid]}
                if name != "test":
                    item["target"] = splits[name][jid]
                handle.write(json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n")
        temp.replace(path)
        hashes[path.name] = digest(path)
    report = {
        "benchmark": "AI/SinglePropertyClass/dft_3d_magmom_oszicar",
        "raw_count": len(raw), "official_counts": official_counts,
        "export_counts": {k: len(v) for k, v in ids.items()},
        "excluded_from_internal_validation": train_val_overlap,
        "train_val_label_counts": {k: dict(Counter(splits[k][jid] for jid in ids[k]))
                                   for k in ("train", "val")},
        "export_split_ids_disjoint": True, "all_ids_present": True,
        "structure_fields": list(ATOM_FIELDS), "test_targets_exported": False,
        "train_val_labels_verified": "1 if raw magmom_oszicar > 0.05 else 0; no absolute value",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [raw_path, split_path]},
        "export_sha256": hashes,
        "limitations": ["Different-ID structure duplicates not yet checked"],
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
