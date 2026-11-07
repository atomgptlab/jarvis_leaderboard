#!/bin/bash
# CHIPS-FF (dft_3d_chipsff) entries for the ALIGNN-FF JARVIS-FF-DB force field.
# Same protocol as the other CHIPS-FF rows: the chipsff suite is run on the 104
# canonical leaderboard jids, then the per-material job_info/results JSONs are
# reduced to one CSV per property.
#
# Only the force field changes between the ALIGNN-FF rows; $MODEL selects it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-/home/kamalch/miniforge3/envs/chipsff/bin/python}   # env with chipsff + alignn.ff
RC=${RC:-/home/kamalch/alignn2026/chipsff/chipsff/run_chipsff.py}
MODEL=${MODEL:-/home/kamalch/alignn2026/ff_db_smooth/results}
D=${D:-/home/kamalch/alignn2026/chipsff_alignn_ff_jarvis_ff_db}                 # work dir: lb_jids.json + chemical_potentials.json

run_one(){
  jid=$1; wd=$D/$jid; mkdir -p "$wd"; cp "$D/chemical_potentials.json" "$wd/"
  cat > "$wd/input.json" <<JSON
{"jid":"$jid","calculator_type":"alignn_ff","chemical_potentials_file":"chemical_potentials.json",
 "properties_to_calculate":["relax_structure","calculate_ev_curve","calculate_formation_energy","calculate_elastic_tensor","analyze_surfaces","analyze_defects"],
 "bulk_relaxation_settings":{"filter_type":"ExpCellFilter","relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":false}},
 "surface_settings":{"indices_list":[[0,1,0],[0,0,1]],"layers":4,"vacuum":18,"relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":true},"filter_type":"ExpCellFilter"},
 "defect_settings":{"generate_settings":{"on_conventional_cell":true,"enforce_c_size":8,"extend":1},"relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":true},"filter_type":"ExpCellFilter"},
 "use_conventional_cell":true,
 "calculator_settings":{"alignn_ff":{"path":"$MODEL","model_filename":"best_model.pt","stress_wt":1}}}
JSON
  ( cd "$wd" && BROWSER=/bin/true OMP_NUM_THREADS=4 $PY $RC --input_file input.json > run.log 2>&1 )
}
export -f run_one; export PY RC D MODEL

python3 -c "import json;print('\n'.join(json.load(open('$D/lb_jids.json'))))" \
  | xargs -I{} -P4 bash -c 'run_one "$@"' _ {}

# Extract the per-material predictions from the job_info/results JSONs:
A=../alignn2_radius
"$PY" "$HERE/$A/chipsff_extract_pred.py" "$D" "$HERE/chipsff_pred_jarvis_ff_db.json" full

# NOTE: the final reduction to the CSVs in this directory used a per-model copy
# of $A/chipsff_leaderboard_matpes.py -- that script is hardcoded to the
# matpes_smooth run and its copy for this model was not kept, so the step below
# is the recipe, not a runnable line. Formation and vacancy energies must use
# chemical potentials computed with THIS model (one single point per elemental
# reference structure); the chipsff-shipped chempots sit on a different model's
# energy scale and shift form_en by ~2 eV/atom if reused.
#   $A/build_matpes_unary.py  (retargeted at $MODEL)  -> jarvis_ff_db_unary.json
#   $A/chipsff_leaderboard_matpes.py (retargeted)     -> AI-SinglePropertyPrediction-*.csv.zip
