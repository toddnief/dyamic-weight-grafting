#!/bin/bash
#SBATCH --job-name=bidirectional-reversal
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --partition=general

python train.py