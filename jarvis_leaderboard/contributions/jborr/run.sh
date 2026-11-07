#!/bin/bash
# Reproduce cfid_chem SSUB formation-energy predictions -- contributor: jborr
set -e
pip install jarvis-tools lightgbm scikit-learn requests -q
python gen_contribution.py
echo "Done"
