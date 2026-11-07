"""Package seed-run test predictions into a leaderboard csv.zip.

Reads prediction_results_test_set.csv from one or more run dirs,
averages predictions per structure across seeds, reports per-seed and
ensemble MAEs, and writes the contribution zip in the jarvis_leaderboard
format (csv with header id,prediction; ids with any .vasp suffix
stripped). This contribution passes a single run dir, so the output is
that one model's predictions; the averaging only applies when several
run dirs are given.
"""
import argparse
import csv
import os
import zipfile


def read_preds(run_dir):
    """Return ({id: prediction}, {id: target}) from a run dir."""
    path = os.path.join(run_dir, "prediction_results_test_set.csv")
    preds, targets = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            sid = row["id"].replace(".vasp", "")
            preds[sid] = float(row["prediction"])
            targets[sid] = float(row["target"])
    return preds, targets


def mae(a, b):
    """Mean absolute error over the shared keys of two dicts."""
    return sum(abs(a[k] - b[k]) for k in a) / len(a)


def main():
    """Average seed predictions and write the contribution zip."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="AI-SinglePropertyPrediction-"
                    "formation_energy_peratom-dft_3d-test-mae.csv.zip")
    args = ap.parse_args()

    all_preds, targets = [], None
    for rd in args.run_dirs:
        p, t = read_preds(rd)
        if targets is None:
            targets = t
        assert set(p) == set(targets), "run %s has different test ids" % rd
        all_preds.append(p)
        print("%-40s single-seed MAE %.5f" % (rd, mae(p, targets)))

    ens = {k: sum(p[k] for p in all_preds) / len(all_preds)
           for k in targets}
    print("ensemble of %d seeds: MAE %.5f (n=%d)"
          % (len(all_preds), mae(ens, targets), len(ens)))

    csv_name = os.path.basename(args.out).replace(".zip", "")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = ["id,prediction"] + [
            "%s,%s" % (k, ens[k]) for k in targets]
        zf.writestr(csv_name, "\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
