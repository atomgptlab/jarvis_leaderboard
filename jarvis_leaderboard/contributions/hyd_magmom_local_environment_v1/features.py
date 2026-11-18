"""Small deterministic descriptors from composition and lattice only.

These are a pipeline baseline, not the published MatMiner descriptors.
No fitted transformation or lookup of computed material properties occurs.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
            "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
            "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
            "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu "
            "Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv "
            "Ts Og").split()
INDEX = {symbol: i for i, symbol in enumerate(ELEMENTS)}
ATOM_FIELDS = {"lattice_mat", "coords", "elements", "cartesian"}


def feature_names(feature_set):
    names = [f"fraction_{symbol}" for symbol in ELEMENTS]
    names += ["species_count", "composition_entropy", "fraction_l2", "max_fraction"]
    if feature_set == "structure":
        names += ["atom_count", "volume_per_atom", "length_min_over_max",
                  "length_mid_over_max", "cos_angle_min", "cos_angle_mid", "cos_angle_max"]
    elif feature_set != "composition":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return names


def featurize(atoms, feature_set):
    if set(atoms) != ATOM_FIELDS:
        raise ValueError("Only sanitized atomic structure fields are permitted")
    counts = Counter(atoms["elements"])
    fractions = np.zeros(len(ELEMENTS), dtype=np.float64)
    n = sum(counts.values())
    if not n:
        raise ValueError("Empty structure")
    for symbol, count in counts.items():
        fractions[INDEX[symbol]] = count / n
    nonzero = fractions[fractions > 0]
    values = [*fractions, len(counts), -np.sum(nonzero * np.log(nonzero)),
              np.linalg.norm(fractions), np.max(fractions)]
    if feature_set == "structure":
        lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
        if lattice.shape != (3, 3):
            raise ValueError("Invalid lattice shape")
        lengths = np.linalg.norm(lattice, axis=1)
        volume = abs(np.linalg.det(lattice))
        if volume <= 1e-8 or np.min(lengths) <= 1e-8:
            raise ValueError("Degenerate lattice")
        cosines = sorted(np.clip(np.dot(lattice[i], lattice[j]) / (lengths[i]*lengths[j]), -1, 1)
                         for i, j in ((0, 1), (0, 2), (1, 2)))
        lengths = sorted(lengths)
        values += [n, volume/n, lengths[0]/lengths[2], lengths[1]/lengths[2], *cosines]
    elif feature_set != "composition":
        raise ValueError(f"Unknown feature set: {feature_set}")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite features")
    return result


def load_features(path, feature_set, labeled):
    ids, rows, labels = [], [], []
    allowed = {"id", "atoms", "target"} if labeled else {"id", "atoms"}
    with Path(path).open() as handle:
        for line in handle:
            item = json.loads(line)
            if set(item) != allowed:
                raise ValueError(f"Unexpected fields in {path}")
            ids.append(item["id"])
            rows.append(featurize(item["atoms"], feature_set))
            if labeled:
                if item["target"] not in (0, 1):
                    raise ValueError("Non-binary target")
                labels.append(item["target"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty split or duplicate IDs")
    return ids, np.stack(rows), np.asarray(labels, dtype=np.int64) if labeled else None
