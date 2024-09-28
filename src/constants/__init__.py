import os
import torch
from datetime import datetime

# Common Constants
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("m%_d_%Y_%H_%M_%S")
DATA_PATH: str = os.path.join(os.getcwd(), "data")
ARTIFACTS_DIR: str = os.path.join("artifacts", TIMESTAMP)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

# data Ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"