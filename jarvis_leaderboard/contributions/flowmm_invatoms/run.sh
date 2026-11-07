#!/bin/bash

#SBATCH --job-name=flow_jarv           
#SBATCH --output=/lab/mml/kipp/677/jarvis/rhys/benchmarks/job_runs/flowmm_benchmark_jarvis/fl_studio.out
#SBATCH --partition=batch
#SBATCH --nodes=1                         
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4                # Run 4 tasks (processes) on the node
#SBATCH --cpus-per-task=2                  # 2 CPU cores per task (for multi‐threaded code)
#SBATCH --time=24:00:00                    # Max walltime (HH:MM:SS)
#SBATCH --mem=64G                          # Total RAM for the job (8 GB)

# setup
eval "$(conda shell.bash hook)"
source scripts/wandb_api_key.sh
module load cuda/11.8
conda activate flowmm
export HYDRA_FULL_ERROR="1"
export WABDB_DIR="job_runs/flowmm_benchmark_jarvis/outputs/wandb_outputs"
export FLOWMM_RUN_ROOT="job_runs/flowmm_benchmark_jarvis/outputs/"

# commands
nvidia-smi
cd models/flowmm/
bash create_env_file.sh
python -u -m scripts_model.run \
             data=jarvis \
             model=null_params \
             hydra.run.dir=$FLOWMM_RUN_ROOT \
             train.model_checkpoints.save_top_k=1 \
             train.pl_trainer.max_epochs=10 \
             logging.val_check_interval=1 \
             train.model_checkpoints.save_last=false \

echo "Done"



	
