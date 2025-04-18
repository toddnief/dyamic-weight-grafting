include .env # sets PARTITION

_conda = conda
ENV_NAME = kp

workdir = ./
scripts_dir = ${workdir}
slurm_dir = ${workdir}slurm/
logs_dir = ${workdir}logs/

TIMESTAMP := $(shell date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE_PREFIX = ${logs_dir}${TIMESTAMP}
output_file = ${LOG_FILE_PREFIX}_res.txt
err_file = ${LOG_FILE_PREFIX}_err.txt

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

# Usage: make create_datasets CONFIG=config_datasets.yaml
.PHONY: create_datasets
create_datasets:
	${SBATCH} \
	--partition=$(PARTITION) \
	--output="$(output_file)" \
	--error="$(err_file)" \
	$(slurm_dir)create_datasets.slurm

# Usage: make experiments CONFIG=config_experiments.yaml PATCH_CONFIG=config_patches.yaml
.PHONY: experiments
experiments:
	${SBATCH} \
	--partition=$(PARTITION) \
	--output="$(output_file)" \
	--error="$(err_file)" \
	--export=ALL,TIMESTAMP=$(TIMESTAMP) \
	$(slurm_dir)experiments.slurm