#!/usr/bin/env bash
#SBATCH --job-name=kp_experiments
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint="a100|h100"
#SBATCH --time=03:00:00 # default; Makefile can shorten
#SBATCH --array=0-5 # default; Makefile overrides when SINGLE_RUN is set

export HF_AUTO_MODEL_TRUST_REMOTE_CODE=1

TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
CONFIG_FILE=${CONFIG:?CONFIG YAML path must be provided}
SMOKE_TEST=${SMOKE_TEST:-false}   # only for the log line below

# Define dropout rates array
DROPOUT_RATES=(0.0 0.2 0.4 0.6 0.8 1.0)

# Get task ID - either from SLURM_ARRAY_TASK_ID or SINGLE_RUN
TASK_ID=${SLURM_ARRAY_TASK_ID:-${SINGLE_RUN:-0}}
export DROPOUT=${DROPOUT_RATES[$TASK_ID]}
# Set dropout rate override
OVERRIDES=("inference_config.dropout_rate=${DROPOUT}")
echo "[${TASK_ID}] CONFIG=${CONFIG_FILE} | smoke_test=${SMOKE_TEST} | dropout=${DROPOUT}"


python src/kg/scripts/run_experiment.py \
  --timestamp "$TIMESTAMP" \
  --experiment-config "$CONFIG_FILE" \
  --override "${OVERRIDES[@]}"