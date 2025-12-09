#!/bin/bash

#SBATCH -A hpc_training             # allocation name
#SBATCH --time=2-10:00:00           # time in days-hours-minutes-seconds
#SBATCH -p gpu
#SBATCH -J lead_corruption_runs_gs_tent
#SBATCH -o lead_corruption_runs_gs_tent.log            # standard output file (%A: Job ID, %a: Array ID)
#SBATCH -e lead_corruption_runs_gs_tent.log            # standard error file

#SBATCH -c 2                      # request n cpus
#SBATCH --mem-per-cpu=24gb         # request m gbs per cpu # running the models doesn't rely much on ram
#SBATCH --gres=gpu:1
#SBATCH --array=0-14                 # number of parallel jobs (adjust as needed)

module purge
source ~/miniforge3/etc/profile.d/conda.sh
source activate unida_tta_env

echo "Running lead_corruption_runs_gs_tent task $SLURM_ARRAY_TASK_ID"

learning_rates=(
    0.00005
    0.0001
    0.0002
    0.0005
    0.001
    0.002
    0.005
    0.01
    0.02
    0.05
    0.1
    0.2
    1 
)

# select the learning_rate for this task
learning_rate=${learning_rates[$SLURM_ARRAY_TASK_ID]}

echo "Selected learning rate: $learning_rate"

# Run Python script for the selected corruption
# PDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=office transfers=all settings=pda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=tent tta_grid_search=true lr=$learning_rate
# OSDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=office transfers=all settings=osda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=tent tta_grid_search=true lr=$learning_rate
# OPDA
python main.py current_step=evaluate_unida_tta_models models=lead dataset_key=office transfers=all settings=opda corruption=true severity=5 corruption_type=$corruption_type tta_enabled=true tta_method=tent tta_grid_search=true lr=$learning_rate

echo "Job $SLURM_ARRAY_TASK_ID ($corruption_type) completed"