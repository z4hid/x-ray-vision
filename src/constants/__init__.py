import os
import torch
from datetime import datetime

# Common Constants
CONFIG_PATH: str = os.path.join(os.getcwd(), "config", "config.yaml")
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
DATA_PATH: str = os.path.join(os.getcwd(), "data")
ARTIFACTS_DIR: str = os.path.join("artifacts", TIMESTAMP)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

# data Ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"


# Data Transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRAIN_TRANSFORM_OBJECT_FILE_NAME = "train_transform.pkl"
VALID_TRANSFORM_OBJECT_FILE_NAME = "valid_transform.pkl"
TEST_TRANSFORM_OBJECT_FILE_NAME = "test_transform.pkl"


# Model Trainer constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_PATH = "model.pt"