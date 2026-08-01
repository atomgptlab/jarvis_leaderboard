#!/bin/bash
# ALIGNN LTC (ALIGNN-2026-PureTorch-radius); reproduce dir: ~/alignn2026/ltc/radius_log
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python train_alignn.py --root_dir . --config_name config.json --output_dir results --target_key target
