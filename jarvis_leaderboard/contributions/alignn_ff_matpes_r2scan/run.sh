#!/bin/bash
# Reproduce the CHIPS-FF (dft_3d_chipsff) leaderboard entries for ALIGNN-FF MATPES-R2SCAN.
#
# Protocol (matches the CHIPS-FF paper / other leaderboard FF entries):
#   single relax-and-evaluate, FrechetCellFilter, fmax 0.05, 200 steps,
#   conventional cell, 6 surface Miller indices, per-model self-consistent
#   elemental chemical potentials. Properties: form_en, C11, C44, Kv,
#   surf_en, vac_en (id,prediction CSV per property, filtered to test ids).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL_PATH:?Set MODEL_PATH to the ALIGNN-FF MATPES-R2SCAN ALIGNN-FF model dir (best_model.pt + config.json)}"
# 1) install
pip install alignn chipsff jarvis-tools
# 2) run chipsff for the 104 canonical leaderboard jids (see lb_jids.json in the
#    chipsff repo). Each job writes input.json then runs run_chipsff.py:
#    bulk_relaxation_settings/surface_settings/defect_settings all use
#    {"filter_type":"FrechetCellFilter","relaxation_settings":{"fmax":0.05,"steps":200}}
#    surface_settings.indices_list = [[1,0,0],[1,1,1],[1,1,0],[0,1,1],[0,0,1],[0,1,0]]
#    calculator_settings.alignn_ff = {"path":"$MODEL_PATH","model_filename":"best_model.pt","stress_wt":1}
#    python chipsff/chipsff/run_chipsff.py --input_file input.json
#    NOTE: chipsff must implement FrechetCellFilter (add the FrechetCellFilter
#    branch alongside ExpCellFilter in general_material_analyzer.relax_structure).
# 3) self-consistent chempots: single-point energy/atom of each element's
#    reference structure with this model -> {symbol: e_per_atom}.
# 4) aggregate each material's *_job_info.json into id,prediction CSVs:
#    form_en = (equilibrium_energy - sum(mu[el]*count))/n_atoms ;
#    kv=bulk_modulus, c11/c44=elastic_tensor.C_11/C_44,
#    surf_en=all_surfaces[*].surf_en, vac_en=all_vacancies[*].vac_en ;
#    keep only ids present in benchmarks/AI/SinglePropertyPrediction/dft_3d_chipsff_<prop>.json (test split).
echo "Model=ALIGNN-FF MATPES-R2SCAN (tag matpes_r2scan). See CHIPS-FF: https://github.com/atomgptlab/chipsff"
