#!/bin/bash

#SBATCH -A hpc_training             # allocation name
#SBATCH --time=2-10:00:00           # time in days-hours-minutes-seconds
#SBATCH -p gpu
#SBATCH -J generate_corruptions
#SBATCH -o generate_corruptions.log            # standard output file (%A: Job ID, %a: Array ID)
#SBATCH -e generate_corruptions.log            # standard error file

#SBATCH -c 2                      # request n cpus
#SBATCH --mem-per-cpu=50gb         # request m gbs per cpu # ~40 needed for all corruptions, but 50 for elastic transform
#SBATCH --gres=gpu:1
#SBATCH --array=0-14                 # number of parallel jobs (adjust as needed)

module purge
source ~/miniforge3/etc/profile.d/conda.sh
source activate unida_tta_env

echo "Running generate_corruptions task $SLURM_ARRAY_TASK_ID"

# list of corruption types (must match array range!)
corruptions=(
    gaussian_noise
    shot_noise
    impulse_noise
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
    snow
    frost
    fog
    brightness
    contrast
    elastic_transform
    pixelate
    jpeg_compression
)

# select the corruption for this task
corruption_type=${corruptions[$SLURM_ARRAY_TASK_ID]}

echo "Selected corruption: $corruption_type"

# Run Python script for the selected corruption
python main.py current_step=corrupt_datasets dataset_key=office severity=5 splitted_parsing=false corruption_type=$corruption_type

echo "Job $SLURM_ARRAY_TASK_ID ($corruption_type) completed"