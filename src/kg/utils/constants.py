import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set up logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # This sends the output to the console (SLURM terminal)
    ],
)
LOGGER = logging.getLogger("main")

# Set up paths
PACKAGE_DIR = Path(__file__).parent.parent.resolve()
PROJECT_DIR = PACKAGE_DIR.parent.parent.resolve()
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR"))

CONFIG_DIR = PROJECT_DIR / "config"
DATASETS_CONFIG_DIR = CONFIG_DIR / "datasets"
TRAINING_CONFIG_DIR = CONFIG_DIR / "training"
EXPERIMENTS_CONFIG_DIR = CONFIG_DIR / "experiments"
PATCH_CONFIG_DIR = EXPERIMENTS_CONFIG_DIR / "patch_configs"
JOB_CONFIG_DIR = EXPERIMENTS_CONFIG_DIR / "generated_cfgs"

DATA_DIR = PROJECT_DIR / "data"
TEMPLATES_DIR = PROJECT_DIR / "data_templates"
ACTOR_NAMES_PATH = TEMPLATES_DIR / "fake_movies_real_actors" / "real_actors.jsonl"

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
TRAINED_MODELS_DIR = ARTIFACTS_DIR / "trained_models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Set up API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model constants
MODEL_TO_HFID = {
    "bart": "facebook/bart-large",
    "gpt2": "gpt2",
    "gpt2-xl": "openai-community/gpt2-xl",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "gemma": "google/gemma-1.1-2b-it",
    "olmo": "allenai/OLMo-1B",
    "llama3": "meta-llama/Llama-3.2-1B",
}
