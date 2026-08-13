#!/bin/bash
# n_alignn (node-attention) ALIGNN 2.0 on the radius graph.
# See ~/alignn2026/property_nalignn_radius/<prop>/run.sh for the exact per-property
# reproduce command (data build + train_alignn.py with config.json).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
