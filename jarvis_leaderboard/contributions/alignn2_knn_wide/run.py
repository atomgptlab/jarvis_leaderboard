"""Reproduce the alignn2_knn_wide contributions.

Single-model entries (one trained model per property). Requires the
ALIGNN 2.0 repo (https://github.com/atomgptlab/alignn) installed with
its pure-torch extras (torch, lmdb, matscipy, jarvis-tools). The three
helper scripts next to this file (make_leaderboard_dataset.py,
make_lb_configs.py, package_leaderboard.py) do the data, config and
packaging steps.
optb88vdw_bandgap and optb88vdw_total_energy each train in ~33 h and
mbj_bandgap in ~9 h on one NVIDIA GB10.

Per property:
1. Build the dataset directory with the exact leaderboard split baked
   into id_prop.csv row order (train, then val, then test; the config's
   keep_data_order=true + explicit n_train/n_val/n_test reproduce the
   benchmark membership id-for-id).
2. Derive the submitted-seed config from the width-768 base config.
3. Train with alignn.train_alignn (folder mode).
4. Package the test predictions into the contribution zip.
"""
import os
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))

PROPS = [
    # (property, base config, torch_seed, dataset dir, (n_tr, n_va, n_te))
    ("optb88vdw_bandgap", "config_lb_bg_h768.json", "231", "lb_bandgap",
     ("44569", "5572", "5572")),
    ("mbj_bandgap", "config_lb_mbj_h768.json", "312", "lb_mbj",
     ("14535", "1817", "1815")),
    ("optb88vdw_total_energy", "config_lb_toten_h768.json", "123",
     "lb_toten", ("44569", "5572", "5572")),
]


def sh(cmd, cwd=None):
    """Run a command, echoing it, failing loudly."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    """Dataset -> config -> train -> package, per property."""
    for prop, base_cfg, seed, data_dir, (n_tr, n_va, n_te) in PROPS:
        sh([PY, os.path.join(HERE, "make_leaderboard_dataset.py"),
            "--prop", prop, "--out_dir", data_dir])
        prefix = base_cfg.replace(".json", "")
        sh([PY, os.path.join(HERE, "make_lb_configs.py"),
            "--base_config", base_cfg, "--out_prefix", prefix,
            "--n_train", n_tr, "--n_val", n_va, "--n_test", n_te,
            "--seeds", seed])
        work = "work_%s_s%s" % (prop, seed)
        os.makedirs(work, exist_ok=True)
        run_dir = "run_%s_s%s" % (prop, seed)
        sh([PY, "-m", "alignn.train_alignn",
            "--root_dir", os.path.abspath(data_dir),
            "--config_name",
            os.path.abspath("%s_s%s.json" % (prefix, seed)),
            "--output_dir", os.path.abspath(run_dir)],
           cwd=work)
        out_zip = ("AI-SinglePropertyPrediction-%s-dft_3d-test-mae"
                   ".csv.zip" % prop)
        sh([PY, os.path.join(HERE, "package_leaderboard.py"),
            run_dir, "--out", out_zip])


if __name__ == "__main__":
    main()
