#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "AllModelsV100.py"
#SBATCH -C V100
module load cuda
source ../venv/bin/activate
python ./AllModels.py
