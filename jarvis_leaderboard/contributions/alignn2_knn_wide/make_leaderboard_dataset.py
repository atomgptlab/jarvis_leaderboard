"""Build an exact-split leaderboard dataset dir for folder-mode training.

Reads a jarvis_leaderboard benchmark json (train/val/test id->target maps),
fetches structures from the JARVIS database, and writes a train_alignn
--root_dir with one POSCAR per structure and an id_prop.csv ordered
train -> val -> test. With keep_data_order=true and
n_train/n_val/n_test set to the printed counts, alignn.data's
get_id_train_val_test reproduces the benchmark membership exactly
(first n_train rows train, next n_val val, last n_test test).

Needs jarvis-tools and internet access (the benchmark json and the
JARVIS database are downloaded on first use).
"""
import argparse
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

BENCH_URL = (
    "https://raw.githubusercontent.com/atomgptlab/jarvis_leaderboard/main/"
    "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
    "dft_3d_{prop}.json.zip"
)


def load_benchmark(path_or_none, prop):
    """Load benchmark split json from disk or download it."""
    if path_or_none and os.path.exists(path_or_none):
        if path_or_none.endswith(".zip"):
            zf = zipfile.ZipFile(path_or_none)
            return json.loads(zf.read(zf.namelist()[0]))
        return json.load(open(path_or_none))
    url = BENCH_URL.format(prop=prop)
    print("downloading benchmark:", url)
    raw = urllib.request.urlopen(url, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    return json.loads(zf.read(zf.namelist()[0]))


def main():
    """Write POSCARs + ordered id_prop.csv for the benchmark split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default="formation_energy_peratom")
    ap.add_argument("--dataset", default="dft_3d")
    ap.add_argument("--benchmark_json", default=None,
                    help="local benchmark json(.zip); downloads if absent")
    ap.add_argument("--leader_csv_zip", default=None,
                    help="optional contribution csv.zip to cross-check "
                         "test ids against")
    ap.add_argument("--out_dir", default="lb_form")
    args = ap.parse_args()

    bench = load_benchmark(args.benchmark_json, args.prop)
    counts = {k: len(bench[k]) for k in ("train", "val", "test")}
    print("benchmark counts:", counts)

    if args.leader_csv_zip:
        zf = zipfile.ZipFile(args.leader_csv_zip)
        rows = list(csv.DictReader(
            io.TextIOWrapper(zf.open(zf.namelist()[0]))))
        lids = set(r["id"] for r in rows)
        assert lids == set(bench["test"]), "leader csv test ids differ!"
        print("leader csv test ids match benchmark: OK")

    print("loading", args.dataset, "from figshare (cached after first run)")
    db = {d["jid"]: d for d in jdata(args.dataset)}

    missing = [j for split in ("train", "val", "test")
               for j in bench[split] if j not in db]
    if missing:
        sys.exit("FATAL: %d benchmark ids missing from %s, e.g. %s"
                 % (len(missing), args.dataset, missing[:5]))

    # sanity: benchmark targets should equal the database property values
    worst = 0.0
    for split in ("train", "val", "test"):
        for j, v in bench[split].items():
            dv = db[j].get(args.prop.replace("dft_3d_", ""))
            if isinstance(dv, (int, float)):
                worst = max(worst, abs(float(v) - float(dv)))
    print("max |benchmark target - db value|:", worst)

    os.makedirs(args.out_dir, exist_ok=True)
    n_written = 0
    with open(os.path.join(args.out_dir, "id_prop.csv"), "w") as f:
        w = csv.writer(f)
        for split in ("train", "val", "test"):
            for j, v in bench[split].items():
                fname = j + ".vasp"
                Atoms.from_dict(db[j]["atoms"]).write_poscar(
                    os.path.join(args.out_dir, fname))
                w.writerow([fname, float(v)])
                n_written += 1
                if n_written % 5000 == 0:
                    print(" ", n_written, "structures written", flush=True)
    print("wrote", n_written, "structures to", args.out_dir)
    print("config must set: keep_data_order=true, n_train=%d, n_val=%d, "
          "n_test=%d" % (counts["train"], counts["val"], counts["test"]))


if __name__ == "__main__":
    main()
