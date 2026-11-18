"""Screen exact/rounded duplicate inputs; never use test labels.

Canonicalizes site order and periodic coordinate wrapping at 8 decimal places.
This is not a complete symmetry/primitive-cell structure-matching audit.
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(atoms):
    lattice = np.asarray(atoms["lattice_mat"], dtype=float)
    coords = np.asarray(atoms["coords"], dtype=float)
    if atoms["cartesian"]:
        coords = np.linalg.solve(lattice.T, coords.T).T
    wrapped = np.mod(np.round(np.mod(coords, 1), 8), 1)
    sites = sorted((e, *map(float, xyz)) for e, xyz in zip(atoms["elements"], wrapped))
    payload = [np.round(lattice, 8).tolist(), sites]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def main():
    groups = {}
    for split in ("train", "val", "test"):
        rows = defaultdict(list)
        with (ROOT / "artifacts/prepared" / f"{split}.jsonl").open() as f:
            for line in f:
                item = json.loads(line)
                rows[fingerprint(item["atoms"])].append(item["id"])
        groups[split] = rows
    pairs = {}
    for left, right in combinations(groups, 2):
        common = sorted(set(groups[left]) & set(groups[right]))
        pairs[f"{left}/{right}"] = {
            "shared_fingerprint_count": len(common),
            "left_ids": [jid for key in common for jid in groups[left][key]],
            "right_ids": [jid for key in common for jid in groups[right][key]],
        }
    result = {
        "method": "Site-order invariant, wrapped fractional coordinates and lattice rounded to 8 decimals",
        "limitation": "Does not identify arbitrary origin shifts, rotations, basis changes or supercells",
        "within_split_duplicate_groups": {
            split: [ids for ids in grouped.values() if len(ids) > 1]
            for split, grouped in groups.items()},
        "cross_split": pairs, "test_targets_accessed": False,
    }
    path = ROOT / "artifacts/prepared/structure_duplicates.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cross_split_counts": {k: v["shared_fingerprint_count"] for k, v in pairs.items()},
                      "within_split_counts": {k: len(v) for k, v in result["within_split_duplicate_groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
