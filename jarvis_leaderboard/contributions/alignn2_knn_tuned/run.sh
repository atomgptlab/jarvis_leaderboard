#! /bin/bash
#SBATCH --time=36:00:00
#SBATCH --job-name=alignn2_lb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=slurm_%j.output
#SBATCH --partition=gpu

# Training step only; run.py runs the full pipeline (dataset, config,
# train, package). Expects lb_form/ and config_lb_knn_s312.json, written by
# make_leaderboard_dataset.py and make_lb_configs.py.
# One GPU trains one seed in ~20 h; give each job
# its own working directory (LMDB caches are cwd-relative). On unified-
# memory GPUs (e.g. NVIDIA GB10) the allocator setting below prevents
# fragmentation-driven OOM kills on the large line-graph batches.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED=${1:-312}
mkdir -p work_s${SEED} && cd work_s${SEED}

python -m alignn.train_alignn \
    --root_dir ../lb_form \
    --config_name ../config_lb_knn_s${SEED}.json \
    --output_dir ../lb_knn_s${SEED}

# after training (single-model entry, seed 312):
# python package_leaderboard.py lb_knn_s312
