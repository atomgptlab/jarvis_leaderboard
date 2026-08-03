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
