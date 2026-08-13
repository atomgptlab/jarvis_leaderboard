"""Generate a pure-torch ALIGNN config for graph-level property prediction.

The model is ``alignn_atomwise_pure`` (no DGL, no torch_geometric) with the
``pure_torch`` neighbor strategy, i.e. the same graph/line-graph pipeline used
by the ALIGNN-FF force field but configured for a single intensive graph-level
target (formation energy per atom, band gap, ...).

Usage:
    python make_config.py <n_train> <n_val> <n_test> <epochs> <output_dir> [batch_size]
"""
import json
import sys


def build_config(n_train, n_val, n_test, epochs, output_dir, batch_size=64):
    """Return a TrainingConfig dict for pure-torch property prediction."""
    return {
        "version": "alignn_2026_pure_torch",
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "pure_torch",
        "id_tag": "jid",
        "dtype": "float32",
        "random_seed": 123,
        "classification_threshold": None,
        "n_val": n_val,
        "n_test": n_test,
        "n_train": n_train,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "target_multiplication_factor": None,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-05,
        "learning_rate": 0.001,
        "filename": "sample",
        "warmup_steps": 2000,
        "criterion": "l1",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "pin_memory": False,
        "save_dataloader": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": False,
        "progress": True,
        "log_tensorboard": False,
        "standard_scalar_and_pca": False,
        "use_canonize": True,
        "num_workers": 0,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "keep_data_order": True,
        "normalize_graph_level_loss": False,
        "distributed": False,
        "n_early_stopping": None,
        "output_dir": output_dir,
        "use_lmdb": True,
        # Full 12-NN line graph (== cutoff) so every neighbor edge in the pair
        # graph also contributes angular triplets, matching the reference
        # ALIGNN k-nearest angular graph. A smaller value truncates angles.
        "three_body_cutoff": 8.0,
        "model": {
            "name": "alignn_atomwise_pure",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            # Graph-level regression only: no forces/stresses.
            "calculate_gradient": False,
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "energy_mult_natoms": False,
            "use_penalty": False,
            "link": "identity",
            "zero_inflated": False,
            "classification": False,
        },
    }


if __name__ == "__main__":
    n_train = int(sys.argv[1])
    n_val = int(sys.argv[2])
    n_test = int(sys.argv[3])
    epochs = int(sys.argv[4])
    output_dir = sys.argv[5]
    batch_size = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    cfg = build_config(n_train, n_val, n_test, epochs, output_dir, batch_size)
    out_name = "config.json"
    with open(out_name, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", out_name)
