include .env # sets PARTITION

_conda = conda
ENV_NAME = reversal-sft

workdir = ./
scripts_dir = ${workdir}
slurm_dir = ${workdir}slurm/
logs_dir = ${workdir}logs/

DATE := $(shell date +"%Y%m%d_%H%M%S")
LOG_FILE_PREFIX = ${logs_dir}${DATE}
output_file = ${LOG_FILE_PREFIX}_res.txt
err_file = ${LOG_FILE_PREFIX}_err.txt

# TODO: create slurm directory

SBATCH = $(_conda) run -n ${ENV_NAME} sbatch

# Usage: make train CONFIG=config_train.yaml
.PHONY: train
train:
	${SBATCH} \
	--mem=64GB \
	--partition=$(PARTITION) \
	--nodes=1 \
	--gres=gpu:1 \
	--output="$(output_file)" \
	--error="$(err_file)" \
	$(slurm_dir)train.slurm

.PHONY: create_datasets
create_datasets:
	${SBATCH} \
	--partition=$(DSI_PARTITION) \
	--output="$(output_file)" \
	--error="$(err_file)" \
	$(slurm_dir)create_datasets.slurm