#!/bin/bash -l

#SBATCH --job-name=gen_data
#SBATCH --partition=idle
#SBATCH --time=7-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# SBATCH --gpus=tesla_t4:1
# SBATCH --gpus=tesla_v100:1

# SBATCH --gpus=1
# SBATCH --constraint=nvidia-gpu

# SBATCH --mail-user="fortino@udel.edu"
# SBATCH --mail-type=ALL

#SBATCH --requeue
#SBATCH --export=ALL

#SBATCH --array=225,250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725

UD_QUIET_JOB_SETUP=YES
. /opt/shared/slurm/templates/libexec/common.sh


dir_lib="/lustre/lrspec/users/2649/spectralib_v1"
dir_r="$dir_lib/$SLURM_ARRAY_TASK_ID"
dir_dg="$dir_r/degraded"
dir_pp="$dir_r/preprocessed"
dir_tt="$dir_r/train_test"
dir_ag="$dir_r/augmented"

dir_arr=($dir_r $dir_dg $dir_pp $dir_tt $dir_ag)

for dir_x in ${dir_arr[@]}
do
    if [ -d "$dir_x" ]
    then
        rm -rf "$dir_x"
        echo "Removed: $dir_x"
        mkdir "$dir_x"
        echo "Created: $dir_x"
    else
        mkdir "$dir_x"
        echo "Created: $dir_x"
    fi
done

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM RESTART COUNT: $SLURM_RESTART_COUNT"

conda activate fox
python /home/2649/repos/SCS/scs/prepare_dataset.py --R=$SLURM_ARRAY_TASK_ID --data_dir_original="/home/2649/repos/SCS/data/" --data_dir_degraded=$dir_dg --data_dir_preprocessed=$dir_pp --data_dir_train_test=$dir_tt --data_dir_augmented=$dir_ag
