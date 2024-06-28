#!/bin/bash -l

#SBATCH --job-name=SCS
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --gpus=1
#SBATCH --constraint=nvidia-gpu

# SBATCH --mail-user="fortino@udel.edu"
# SBATCH --mail-type=ALL

#SBATCH --requeue
#SBATCH --export=ALL

#SBATCH --array=0-299

UD_QUIET_JOB_SETUP=YES

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"

python /home/2649/repos/SCS/scs/scs.py --R=100 --dir_batch_model="/lustre/lrspec/users/2649/models/batch09" --SLURM_RESTART_COUNT=$SLURM_RESTART_COUNT --SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
