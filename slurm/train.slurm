#!/bin/bash
#SBATCH --job-name=patch_train
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --time=05:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_train.yaml}

python src/reversal/train_v2.py --config $CONFIG_FILE

