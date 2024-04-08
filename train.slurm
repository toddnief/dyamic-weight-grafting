#!/bin/bash
#SBATCH --job-name=bidirectional-reversal
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=256G
#SBATCH --nodelist=j001-ds
#SBATCH --partition=holtzman
#SBATCH --output=logs/gpt2_training_%j.out
#SBATCH --error=logs/gpt2_training_%j.err

source activate reversal-curse

torchrun --nproc_per_node=2 train.py
