"""Regenerate the trivial mean baseline for every benchmark it covers.

The prediction for every test entry is the mean of the *training* split of that
benchmark -- a constant. It exists so each leaderboard column has a floor: a
model that cannot beat "predict the training mean" has not learned it.

SinglePropertyPrediction targets are scalars, Spectra targets are ";"-joined
vectors and are averaged component-wise.
"""

import json
import pathlib
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LB = HERE.parents[1]
BENCH = LB / "benchmarks"


def load_benchmark(cat, ds, prop):
    z = BENCH / cat / f"{ds}_{prop}.json.zip"
    if not z.exists():
        return None
    return json.loads(zipfile.ZipFile(z).read(f"{ds}_{prop}.json"))


def write_csv(name, rows, header):
    csv = HERE / name
    csv.write_text(
        header + "\n" + "\n".join(f"{i},{v}" for i, v in rows) + "\n"
    )
    with zipfile.ZipFile(str(csv) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, name)
    csv.unlink()


def main():
    written = 0
    for existing in sorted(HERE.glob("AI-*.csv.zip")):
        name = existing.name[: -len(".zip")]
        stem, _, suffix = name.partition("-test-")
        kind, _, rest = stem.partition("-")
        cat, prop_ds = rest.split("-", 1)
        prop, ds = prop_ds.split("-", 1)
        j = load_benchmark(f"{kind}/{cat}", ds, prop)
        if j is None:
            print(f"  skip {name}: no benchmark")
            continue
        train = list(j["train"].values())
        if "Spectra" in cat or "multimae" in suffix:
            arr = np.array(
                [[float(x) for x in str(v).split(";")] for v in train]
            )
            const = ";".join(f"{x}" for x in arr.mean(axis=0))
        else:
            const = float(np.mean([float(v) for v in train]))
        write_csv(name, [(i, const) for i in j["test"]], "id,prediction")
        written += 1
    print(f"wrote {written} mean-baseline CSVs")


if __name__ == "__main__":
    main()
