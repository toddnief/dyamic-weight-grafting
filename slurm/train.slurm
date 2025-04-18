#!/bin/bash
#SBATCH --job-name=kp_train
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100"
#SBATCH --time=12:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_train.yaml}

python src/kp/scripts/train.py --config $CONFIG_FILE

