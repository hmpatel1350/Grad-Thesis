#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "AllModelsA100|V100.py"
#SBATCH -C A100|V100
module load cuda
source ../venv/bin/activate
python ./AllModels.py
