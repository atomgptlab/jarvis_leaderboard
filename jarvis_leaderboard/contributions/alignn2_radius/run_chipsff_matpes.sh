#!/bin/bash
# Reproduce the CHIPS_FF (dft_3d_chipsff) leaderboard entries for the ALIGNN 2.0
# default force field (matpes_smooth: 2/2/128 smooth-cutoff nbr52, MATPES-PBE ep100).
# Full pipeline (scripts all in this directory):
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/kamalch/miniforge3/envs/chipsff/bin/python      # env with chipsff + alignn.ff
PYLB=/home/kamalch/miniforge3/envs/blackwell312/bin/python

# 1) Run the chipsff suite on the 104 canonical CHIPS_FF jids with matpes_smooth.
#    Cluster (SLURM): submit_chipsff_model.sh matpes_ep100 <model_dir>  (one job/jid;
#    relax, ev_curve, formation, elastic, surfaces, defects; use_conventional_cell=true).
# 2) a/b/c/vol need PRIMITIVE cells -> rerun relax+ev_curve with use_conventional_cell=false.
# 3) surf_en needs the benchmark's per-material miller indices AND ev_curve (bulk ref):
#    chipsff_surf_runner.py runs relax+ev_curve+analyze_surfaces per material with the
#    millers parsed from dft_3d_chipsff_surf_en; chipsff_extract_surf.py collects them.
# 4) Extract predictions from the job_info.json files:
#    $PYLB chipsff_extract_pred.py <runs/matpes_ep100> chipsff_pred_matpes.json full
#    $PYLB chipsff_extract_pred.py <matpes_prim>       chipsff_pred_matpes_prim.json geom
# 5) Self-consistent elemental chemical potentials (single-point/atom of each element's
#    reference structure with the SAME model) -- REQUIRED so formation energy is on the
#    model's own scale (shipped chipsff chempots are a different-model scale: form_en
#    2.44 -> 0.081 eV/atom once fixed):
#    $PY build_matpes_unary.py   # -> matpes_smooth_unary.json
# 6) Write the leaderboard CSVs + register metadata:
"$PYLB" "$HERE/chipsff_leaderboard_matpes.py"
