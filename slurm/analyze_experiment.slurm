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

# Initialize override array
OVERRIDES=()

# Conditionally add overrides
[ -n "$MODEL" ] && OVERRIDES+=("model.pretrained=$MODEL")
[ -n "$DATASET" ] && OVERRIDES+=("paths.dataset_name=$DATASET")
[ -n "$DATASET_DIR" ] && OVERRIDES+=("paths.dataset_dir=$DATASET_DIR")
[ -n "$MODEL_DIR" ] && OVERRIDES+=("paths.both_directions_parent=$MODEL_DIR")
[ -n "$DIRECTION" ] && OVERRIDES+=("model.patch_direction=$DIRECTION")

python src/kp/scripts/analyze_experiment.py \
  --timestamp "$TIMESTAMP" \
  --experiment-config "$CONFIG_FILE" \
  --patch-config "$PATCH_CONFIG" \
  --override "${OVERRIDES[@]}"