#!/bin/bash
#SBATCH --job-name=patch_datasets
#SBATCH --mem=120G
#SBATCH --time=05:00:00

# Note: This enables passing the config file as an argument to the script from the Makefile
CONFIG_FILE=${CONFIG:-config_data_generation.yaml}

python src/reversal/create_dataset.py --config $CONFIG_FILE

