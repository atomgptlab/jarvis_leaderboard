#!/bin/bash
# Install dependencies
pip install jarvis-tools scikit-learn pandas numpy

# Run the notebook (or equivalent Python script)
# This script trains a Random Forest on CFID descriptors
# and generates predictions for the ssub formation energy dataset.

python train_cfid_rf.py
