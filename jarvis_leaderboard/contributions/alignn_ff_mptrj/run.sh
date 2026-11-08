#!/bin/bash
# CHIPS-FF (dft_3d_chipsff) entries for the ALIGNN-FF MPtrj (epoch 46) force
# field. Same protocol and commands as ../alignn_ff_matpes_r2scan/run.sh, with
#   MODEL = the MPtrj ALIGNN-FF checkpoint (best_model.pt, epoch 46)
#   filter_type = FrechetCellFilter instead of ExpCellFilter (chipsff needed a
#     small patch to accept it; upstream only implemented ExpCellFilter)
#   chemical potentials from this model: mptrj_ep46_chempots.json (this dir)
# form_en is not submitted yet: 103/104 jids (no Sr reference with this model).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export MODEL=${MODEL:-/home/kamalch/alignn2026/mptrj_frontier/Out_MP20}
export D=${D:-/home/kamalch/alignn2026/chipsff_frechet_mptrj}
cp "$HERE/mptrj_ep46_chempots.json" "$D/chemical_potentials.json"
sed 's/ExpCellFilter/FrechetCellFilter/g' \
  "$HERE/../alignn_ff_matpes_r2scan/run.sh" > "$D/run_chipsff.sh"
bash "$D/run_chipsff.sh"
