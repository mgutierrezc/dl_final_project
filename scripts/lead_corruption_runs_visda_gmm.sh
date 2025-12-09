#!/bin/bash

#SBATCH -A hpc_training             # allocation name
#SBATCH --time=2-10:00:00           # time in days-hours-minutes-seconds
#SBATCH -p gpu
#SBATCH -J lead_corruption_runs_gs_gmm
#SBATCH -o lead_corruption_runs_gs_gmm.log            # standard output file (%A: Job ID, %a: Array ID)
#SBATCH -e lead_corruption_runs_gs_gmm.log            # standard error file

#SBATCH -c 2                      # request n cpus
#SBATCH --mem-per-cpu=24gb         # request m gbs per cpu # running the models doesn't rely much on ram
#SBATCH --gres=gpu:1
#SBATCH --array=0-14                 # number of parallel jobs (adjust as needed)

module purge
source ~/miniforge3/etc/profile.d/conda.sh
source activate unida_tta_env

echo "Running lead_corruption_runs_gs_gmm task $SLURM_ARRAY_TASK_ID"

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

echo "Selected corruption type: $corruption_type"

# Run Python script for the selected corruption
# PDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=visda transfers=all settings=pda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=gmm tta_grid_search=false grid_search_hyperparams=/home/sgw3fy/jobs/unida_tta_jobs/repos/unida_tta_codebase/config/model/tta_hyperparameters_unida_lead_tta_gmm.json
# OSDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=visda transfers=all settings=osda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=gmm tta_grid_search=false grid_search_hyperparams=/home/sgw3fy/jobs/unida_tta_jobs/repos/unida_tta_codebase/config/model/tta_hyperparameters_unida_lead_tta_gmm.json
# OPDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=visda transfers=all settings=opda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=gmm tta_grid_search=false grid_search_hyperparams=/home/sgw3fy/jobs/unida_tta_jobs/repos/unida_tta_codebase/config/model/tta_hyperparameters_unida_lead_tta_gmm.json

echo "Job $SLURM_ARRAY_TASK_ID ($corruption_type) completed"