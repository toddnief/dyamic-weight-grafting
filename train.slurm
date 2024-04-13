#!/bin/bash
#SBATCH --job-name=bidirectional-reversal
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
##SBATCH --partition=holtzman
#SBATCH --partition=general
#SBATCH --constraint=a100
#SBATCH --output=logs/gpt2_training_%j.out
#SBATCH --error=logs/gpt2_training_%j.err

# TODO: Add timestamp to logs

source /home/tnief/miniconda3/etc/profile.d/conda.sh
conda activate reversal-curse

# torchrun --nproc_per_node=2 train.py
python train.py