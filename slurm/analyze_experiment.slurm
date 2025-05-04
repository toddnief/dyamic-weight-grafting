#!/bin/bash
#SBATCH --job-name=kp_analyze
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00

# Set default timestamp if not provided
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_experiments.yaml}
PATCH_CONFIG=${PATCH_CONFIG:-all_tokens_attn_ffn_all.yaml}

python src/kp/scripts/analyze_experiment.py \
  --timestamp $TIMESTAMP \
  --experiment-config $CONFIG_FILE \
  --patch-config $PATCH_CONFIG \
  --override "model.name=$MODEL" \
  --override "paths.dataset_name=$DATASET" \
  --override "paths.dataset_dir=$DATASET_DIR" \
  --override "paths.both_directions_parent=$MODEL_DIR" \
  --override "model.patch_direction=$DIRECTION"