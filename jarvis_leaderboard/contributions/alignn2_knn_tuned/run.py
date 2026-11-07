"""Reproduce the alignn2_knn_tuned formation-energy contribution.

Single-model entry. Requires the ALIGNN 2.0 repo
(https://github.com/atomgptlab/alignn) installed with its pure-torch
extras (torch, lmdb, matscipy, jarvis-tools). The three helper scripts
next to this file (make_leaderboard_dataset.py, make_lb_configs.py,
package_leaderboard.py) do the data, config and packaging steps.
Training takes ~20 h on one modern GPU.

Steps:
1. Build the dataset directory with the exact leaderboard split baked
   into id_prop.csv row order (train, then val, then test; the config's
   keep_data_order=true + explicit n_train/n_val/n_test reproduce the
   benchmark membership id-for-id).
2. Derive the seed-312 config from config_lb_knn.json.
3. Train with alignn.train_alignn (folder mode).
4. Package the test predictions into the contribution zip.
"""
import os
import subprocess
import sys

SEED = "312"
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, cwd=None):
    """Run a command, echoing it, failing loudly."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    """Dataset -> config -> train -> package."""
    sh([PY, os.path.join(HERE, "make_leaderboard_dataset.py"),
        "--prop", "formation_energy_peratom", "--out_dir", "lb_form"])
    sh([PY, os.path.join(HERE, "make_lb_configs.py"),
        "--base_config", "config_lb_knn.json", "--seeds", SEED])
    work = "work_s" + SEED
    os.makedirs(work, exist_ok=True)
    sh([PY, "-m", "alignn.train_alignn",
        "--root_dir", os.path.abspath("lb_form"),
        "--config_name",
        os.path.abspath("config_lb_knn_s%s.json" % SEED),
        "--output_dir", os.path.abspath("lb_knn_s" + SEED)],
       cwd=work)
    sh([PY, os.path.join(HERE, "package_leaderboard.py"),
        "lb_knn_s" + SEED])


if __name__ == "__main__":
    main()
