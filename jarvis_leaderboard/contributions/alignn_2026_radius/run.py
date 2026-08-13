"""Train pure-torch ALIGNN (no DGL) property-prediction models and produce
jarvis-leaderboard submission files.

For each benchmark this script:
  1. Calls ``jarvis_populate_data.py`` to write POSCARs + id_prop.csv and a
     dataset_info.json with the train/val/test sizes.
  2. Generates a ``config.json`` for the ``alignn_atomwise_pure`` model
     (see make_config.py).
  3. Runs ``train_alignn.py`` which selects the pure-torch backend because
     ``model.name == "alignn_atomwise_pure"`` and
     ``neighbor_strategy == "pure_torch"``.
  4. Copies ``prediction_results_test_set.csv`` (already ``id,target,prediction``)
     into ``<benchmark>.csv`` and zips it.

Set CUDA_VISIBLE_DEVICES before running, e.g.:
    CUDA_VISIBLE_DEVICES=2 python run.py
"""
import os
import shutil
from jarvis.db.jsonutils import loadjson
from make_config import build_config
import json

EPOCHS = 300
BATCH_SIZE = 64

TASKS = [
    "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae",
    "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae",
]


def run_task(task):
    """Populate data, train pure-torch ALIGNN, and zip predictions."""
    out_data = "Out_" + task.split("-")[2]
    results = "results_" + task.split("-")[2]

    # 1. Prepare benchmark data (POSCARs + id_prop.csv + dataset_info.json).
    if not os.path.exists(out_data):
        os.system(
            "jarvis_populate_data.py --benchmark_file "
            + task
            + " --output_path="
            + out_data
        )
    info = loadjson(os.path.join(out_data, "dataset_info.json"))

    # 2. Build the pure-torch config.
    cfg = build_config(
        n_train=info["n_train"],
        n_val=info["n_val"],
        n_test=info["n_test"],
        epochs=EPOCHS,
        output_dir=results,
        batch_size=BATCH_SIZE,
    )
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # 3. Train (pure-torch backend, no DGL).
    os.system(
        "train_alignn.py --root_dir "
        + out_data
        + " --config_name config.json --output_dir "
        + results
    )

    # 4. Collect predictions -> <task>.csv.zip (id,target,prediction).
    csv_name = task + ".csv"
    shutil.copy(
        os.path.join(results, "prediction_results_test_set.csv"), csv_name
    )
    if os.path.exists(csv_name + ".zip"):
        os.remove(csv_name + ".zip")
    os.system("zip " + csv_name + ".zip " + csv_name)
    os.remove(csv_name)


if __name__ == "__main__":
    for t in TASKS:
        run_task(t)
