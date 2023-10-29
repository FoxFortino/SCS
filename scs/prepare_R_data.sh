#!/bin/bash -l

#SBATCH --job-name=prepare_R_data
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# SBATCH --gpus=tesla_t4:1
# SBATCH --gpus=tesla_v100:1

# SBATCH --gpus=1
# SBATCH --constraint=nvidia-gpu

# SBATCH --mail-user="fortino@udel.edu"
# SBATCH --mail-type=ALL

#SBATCH --requeue
#SBATCH --export=ALL

#SBATCH --array=5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100

UD_QUIET_JOB_SETUP=YES

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"

python /home/2649/repos/SCS/scs/prepare_R_data.py --R=$SLURM_ARRAY_TASK_ID
