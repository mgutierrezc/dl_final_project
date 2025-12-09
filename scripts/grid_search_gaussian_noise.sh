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
#SBATCH --array=1                 # number of parallel jobs (adjust as needed)

module purge
source ~/miniforge3/etc/profile.d/conda.sh
source activate unida_tta_env

echo "Running generate_corruptions task $SLURM_ARRAY_TASK_ID"

# execute the Python command that generates corruptions (comment out parts not needed)
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=gaussian_noise
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=shot_noise
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=impulse_noise
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=defocus_blur
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=glass_blur
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=motion_blur
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=zoom_blur
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=snow
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=frost
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=fog
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=brightness
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=contrast
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=elastic_transform
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=pixelate
python main.py dataset_key=office severity=5 splitted_parsing=false corruption_type=jpeg_compression

echo "Job $SLURM_ARRAY_TASK_ID completed"