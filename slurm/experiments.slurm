#!/bin/bash
#SBATCH --job-name=kp_experiments
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100"
#SBATCH --time=01:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_experiments.yaml}

python src/kp/scripts/run_experiments.py --config $CONFIG_FILE

