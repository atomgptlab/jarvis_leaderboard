#!/bin/bash

#SBATCH --job-name=agpt_tc            
#SBATCH --partition=batch
#SBATCH --nodes=1                         
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4                # Run 4 tasks (processes) on the node
#SBATCH --cpus-per-task=2                  # 2 CPU cores per task (for multi‐threaded code)
#SBATCH --time=08:00:00                    # Max walltime (HH:MM:SS)
#SBATCH --mem=8G                           # Total RAM for the job (8 GB)

module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate my_atomgpt
nvidia-smi

python models/atomgpt/atomgpt/inverse_models/inverse_models.py \
    --config_name job_runs/agpt_benchmark_jarvis/config.json
