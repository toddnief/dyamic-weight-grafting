import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = PACKAGE_DIR.parent.parent.resolve()
CONFIG_DIR = PROJECT_DIR / "config"
DATASETS_CONFIG_DIR = CONFIG_DIR / "datasets"
TRAINING_CONFIG_DIR = CONFIG_DIR / "training"
DATA_DIR = PROJECT_DIR / "data"
TEMPLATES_DIR = PROJECT_DIR / "data_templates"

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # This sends the output to the console (SLURM terminal)
    ],
)
