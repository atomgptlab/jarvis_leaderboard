#!/bin/bash
# Reproduce the CHIPS_FF leaderboard entries for the ALIGNN 2.0 default force field
# (matpes_smooth: 2/2/128, smooth cutoff, nbr52, MATPES-PBE ep100, radius graph).
#
# Pipeline:
# 1) Run the chipsff suite (relax, ev_curve, formation, elastic, surfaces, defects)
#    on the 104 canonical CHIPS_FF jids with the matpes_smooth model. On a SLURM
#    cluster: submit_chipsff_model.sh matpes_ep100 <model_dir>  (one job/jid,
#    calculator_type=alignn_ff, calculator_settings.alignn_ff.path=<model_dir>).
# 2) Extract per-material predictions from each <jid>_job_info.json into
#    chipsff_pred_matpes.json (a,b,c,vol,kv,c11,c44, equilibrium_energy, elements,
#    surfaces, vacancies).
# 3) Build self-consistent elemental chemical potentials with the SAME model
#    (single-point energy/atom of each element's reference structure) ->
#    matpes_smooth_unary.json.  IMPORTANT: the formation energy must use the model's
#    OWN elemental references; the shipped chipsff chempots are on a different scale
#    and inflate form_en by ~2.4 eV/atom (2.45 -> 0.081 once fixed).
# 4) Write the 10 leaderboard CSVs + register in metadata.json.
# NOTE: a/b/c/vol require a PRIMITIVE-cell run (use_conventional_cell=false) to match
#       the benchmark; intensive props (kv/c11/c44/form_en/surf_en/vac_en) use the
#       conventional run. Two prediction files are staged accordingly.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/blackwell312/bin/python
"$PY" "$HERE/chipsff_leaderboard_matpes.py"
# surf_en (full 85/85): re-run surfaces per-material with the benchmark's own miller
# indices (parse Surface-<jid>_miller_h_k_l from dft_3d_chipsff_surf_en), properties
# [relax_structure, calculate_ev_curve, analyze_surfaces] (ev_curve supplies the bulk
# reference energy, else surf_en is silently skipped). Extract all_surfaces[].surf_en.
