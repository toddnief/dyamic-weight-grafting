#!/bin/bash
#SBATCH --job-name=kp_analyze
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Set default timestamp if not provided
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_experiments.yaml}
PATCH_CONFIG_FILE=${PATCH_CONFIG:-config_patches.yaml}

python src/kp/scripts/analyze_experiments.py \
  --timestamp $TIMESTAMP \
  --experiment-config $CONFIG_FILE \
  --patch-config $PATCH_CONFIG_FILE