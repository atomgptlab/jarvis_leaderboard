#! /bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=alignn2_lb_bg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=slurm_%j.output
#SBATCH --partition=gpu

# Training step for optb88vdw_bandgap only; run.py runs the full pipeline
# (dataset, config, train, package) for all three properties. Expects
# lb_bandgap/ and config_lb_bg_h768_s231.json, written by
# make_leaderboard_dataset.py and make_lb_configs.py.
# One GPU trains this width-768 model in ~33 h. Give the job its own
# working directory (LMDB caches are cwd-relative). On unified-memory
# GPUs (e.g. NVIDIA GB10) the allocator setting below prevents
# fragmentation-driven OOM kills on the large line-graph batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${1:-231}
mkdir -p work_s${SEED} && cd work_s${SEED}

python -m alignn.train_alignn \
    --root_dir ../lb_bandgap \
    --config_name ../config_lb_bg_h768_s${SEED}.json \
    --output_dir ../lb_bg_h768_s${SEED}

# after training:
# python package_leaderboard.py lb_bg_h768_s${SEED} \
#     --out AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae.csv.zip
