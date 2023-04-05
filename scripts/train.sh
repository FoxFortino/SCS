#!/bin/bash -l

#SBATCH --job-name=SCS
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# SBATCH --gpus=tesla_t4:1
# SBATCH --gpus=tesla_v100:1

#SBATCH --gpus=1
#SBATCH --constraint=nvidia-gpu

# SBATCH --mail-user="fortino@udel.edu"
# SBATCH --mail-type=ALL

#SBATCH --requeue
#SBATCH --export=ALL

#SBATCH --array=0-200

UD_QUIET_JOB_SETUP=YES

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"

python /home/2649/repos/SCS/scs/learn.py --hp_set="dev9" --R=100 --array_index=$SLURM_ARRAY_TASK_ID --num_requeue=$SLURM_RESTART_COUNT
