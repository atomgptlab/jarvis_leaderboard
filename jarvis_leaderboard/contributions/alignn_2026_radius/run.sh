#!/bin/bash
# Pure-torch ALIGNN (no DGL) property-prediction models for jarvis-leaderboard.
# Requires an alignn install providing alignn_atomwise_pure + pure_torch
# neighbor strategy (train_alignn.py entry point).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python run.py
