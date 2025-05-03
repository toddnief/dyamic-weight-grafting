#!/bin/bash
#SBATCH --job-name=kp_experiments
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100"
#SBATCH --time=01:00:00
#SBATCH --array=0-10

# Set default timestamp if not provided
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_experiments.yaml}
PATCH_CONFIG_FILE=${PATCH_CONFIG:-all_tokens_attn_ffn_all.yaml}

# Setup dropout experiment param sweep
DROPOUT_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
DROPOUT=${DROPOUT_RATES[$SLURM_ARRAY_TASK_ID]}

python src/kp/scripts/run_experiments.py \
  --timestamp $TIMESTAMP \
  --experiment-config $CONFIG_FILE \
  --patch-config $PATCH_CONFIG_FILE \
  --override "inference_config.dropout_rate=$DROPOUT"