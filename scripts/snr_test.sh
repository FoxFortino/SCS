#!/bin/bash -l

#SBATCH --job-name=snr_test
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# SBATCH --gpus=tesla_t4:1
# SBATCH --gpus=tesla_v100:1

#SBATCH --gpus=1
#SBATCH --constraint=nvidia-gpu

# SBATCH --mail-user="fortino@udel.edu"
# SBATCH --mail-type=ALL

#SBATCH --requeue
#SBATCH --export=ALL

#SBATCH --array=0-103

UD_QUIET_JOB_SETUP=YES

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"

python3 /home/2649/repos/SCS/scs/snr_test.py $SLURM_ARRAY_TASK_ID
