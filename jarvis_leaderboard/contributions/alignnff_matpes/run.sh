#!/bin/bash
# Train ALIGNN-FF (pure-torch, cgcnn 4/4/256, cutoff 5) on MATPES-PBE with
# keep_data_order=True (tail split: last 21736 = test), 100 epochs, no stress.
train_alignn.py --root_dir DataFull --config_name config.json --output_dir results \
  --target_key energy --force_key forces
# then infer energy+forces on the test set and format for the leaderboard:
python matpes_infer.py
