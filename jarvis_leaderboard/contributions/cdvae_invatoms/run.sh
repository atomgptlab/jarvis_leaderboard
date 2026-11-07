#!/bin/bash

#SBATCH --job-name=cdvae_tc            
#SBATCH --partition=batch
#SBATCH --nodes=1                         
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4                # Run 4 tasks (processes) on the node
#SBATCH --cpus-per-task=2                  # 2 CPU cores per task (for multi‐threaded code)
#SBATCH --time=24:00:00                    # Max walltime (HH:MM:SS)
#SBATCH --mem=64G                           # Total RAM for the job (8 GB)

module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate cdvae
nvidia-smi

export PROJECT_ROOT="models/cdvae"
export HYDRA_JOBS="job_runs/cdvae_benchmark_jarvis/hydra_outputs"
export HYDRA_FULL_ERROR="1"
export WABDB_DIR="job_runs/cdvae_benchmark_jarvis/wandb_outputs"
#export WANDB_MODE="offline"
source scripts/wandb_api_key.sh

python -u -m cdvae.run data=supercon expname=supercon \
	model.num_noise_level=2 \
	data.train_max_epochs=1 \
	train.pl_trainer.gpus=1 \
	model.predict_property=True \

echo "Done"



	
