#!/bin/bash
# ALIGNN LTC (ALIGNN-2026-PureTorch-kNN); reproduce dir: ~/alignn2026/ltc/knn_log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python train_alignn.py --root_dir . --config_name config.json --output_dir results --target_key target
