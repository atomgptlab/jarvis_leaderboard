#!/bin/bash
# ALIGNN-FF (pure-torch, cgcnn 4/4/256, cutoff 5) trained on mlearn_Si; test predictions
# extracted to leaderboard format (energy per-atom MAE, forces/stresses multi-MAE).
train_alignn.py --root_dir data --config_name config.json --output_dir results
