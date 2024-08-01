#!/bin/bash
#SBATCH --job-name=bidirectional-reversal
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --constraint="a100|h100"

python train_v2.py