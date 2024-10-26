import logging
import os
from datetime import datetime

import torch
from dotenv import load_dotenv

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
