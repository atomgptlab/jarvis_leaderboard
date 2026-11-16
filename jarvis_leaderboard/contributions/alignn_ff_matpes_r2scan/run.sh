#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH -J alignn_ff_matpes_r2scan
#SBATCH -o alignn_ff_matpes_r2scan.log
# Recipe for this entry: CHIPS-FF run_main.py with the alignn_ff_matpes_r2scan
# weights on CPU, then writes the 10 dft_3d_chipsff leaderboard files.
#   CHIPS_FF  atomgptlab/chipsff checkout on develop (these results: commit 1e31598)
#   ALIGNN    atomgptlab/alignn source with the element-embedding models
#   MODEL     unzipped alignn_ff_matpes_r2scan folder (best_model.pt + config.json)
#             from figshare article 34846446; keep the folder name, run_main.py
#             names its output folder chipsff_frechet_<folder name>
#   BENCH     jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction
# The submitted results used an unversioned alignn snapshot (alignn/ff files
# dated 2026-10-13) and JARVIS dft_3d 9-24-2025. package_leaderboard.py reads
# dft_3d and vacancydb from ~/.cache/atomgptlab/jarvis_data unless --dft3d and
# --vacancydb are given. run_main.py writes about 0.5 GB of outputs here.
# The -c and --mem requests above are a suggestion; the original job's requests
# were not recorded with its outputs.
set -e
: "${CHIPS_FF:?}" "${ALIGNN:?}" "${MODEL:?}" "${BENCH:?}"
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg PYTHONPATH=$CHIPS_FF:$ALIGNN
python -m chipsff.run_main --alignn_ff --model_path "$MODEL" --device cpu --tasks optimize,elastic,surface,vacancy
python package_leaderboard.py chipsff_frechet_alignn_ff_matpes_r2scan "$BENCH" .
