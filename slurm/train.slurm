#!/bin/bash
#SBATCH --job-name=kp_train
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100"
#SBATCH --time=12:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_train.yaml}

OVERRIDES=()
[ -n "$MODEL" ] && OVERRIDES+=("model=$MODEL")
[ -n "$DATASET" ] && OVERRIDES+=("data_options.dataset_name=$DATASET")
[ -n "$DATASET_DIR" ] && OVERRIDES+=("data_options.dataset_dir=$DATASET_DIR")
[ -n "$SMOKE_TEST" ] && OVERRIDES+=("smoke_test=$SMOKE_TEST")
[ -n "$DATASET_TYPE" ] && OVERRIDES+=("data_options.dataset_type=$DATASET_TYPE")
echo "Overrides: ${OVERRIDES[@]}"

python src/kg/scripts/train.py --config $CONFIG_FILE --override "${OVERRIDES[@]}"
