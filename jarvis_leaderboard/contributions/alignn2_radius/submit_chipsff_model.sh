#!/bin/bash
# Submit the CHIPS_FF leaderboard suite (104 canonical jids) for one FF model.
# usage: submit_chipsff_model.sh <model_name> <model_dir>
set -e
MODEL=$1
MPATH=$2
ROOT=/data/kchoudh2/chipsff_bench
WORK=$ROOT/runs/$MODEL
mkdir -p $WORK/logs
cp $ROOT/chemical_potentials.json $WORK/
JIDS=$(python3 -c "import json;print(' '.join(json.load(open('$ROOT/lb_jids.json'))))")
n=0
for jid in $JIDS; do
  sbatch <<EOT >/dev/null
#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --job-name=cf_${MODEL}_${jid}
#SBATCH --output=$WORK/logs/${jid}_%j.out
source /data/kchoudh2/Software/Miniforge/miniforge3/etc/profile.d/conda.sh
conda activate agapi
export CUDA_VISIBLE_DEVICES="" BROWSER=/bin/true OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $WORK
cat > input_${jid}.json <<JSON
{"jid":"$jid","calculator_type":"alignn_ff","chemical_potentials_file":"chemical_potentials.json",
 "properties_to_calculate":["relax_structure","calculate_ev_curve","calculate_formation_energy","calculate_elastic_tensor","analyze_surfaces","analyze_defects"],
 "bulk_relaxation_settings":{"filter_type":"ExpCellFilter","relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":false}},
 "surface_settings":{"indices_list":[[0,1,0],[0,0,1]],"layers":4,"vacuum":18,"relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":true},"filter_type":"ExpCellFilter"},
 "defect_settings":{"generate_settings":{"on_conventional_cell":true,"enforce_c_size":8,"extend":1},"relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":true},"filter_type":"ExpCellFilter"},
 "use_conventional_cell":true,
 "calculator_settings":{"alignn_ff":{"path":"$MPATH","model_filename":"best_model.pt","stress_wt":1}}}
JSON
python $ROOT/chipsff/chipsff/run_chipsff.py --input_file input_${jid}.json
EOT
  n=$((n+1))
done
echo "submitted $n jobs for model=$MODEL (path=$MPATH)"
