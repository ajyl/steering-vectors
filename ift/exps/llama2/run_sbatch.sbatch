#!/bin/bash

#SBATCH --job-name=steer_gen_data
#SBATCH --error=slurm_logs/stderr_%A_%a
#SBATCH --output=slurm_logs/stdout_%A_%a
#SBATCH --partition=spgpu
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --account=mihalcea0

__conda_setup="$('/public/apps/anaconda3/2023.03-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"

#eval "$(conda shell.bash hook)"

conda activate hmm
srun python /home/ajyl/steering-vectors/ift/exps/llama2/generate_samples.py
