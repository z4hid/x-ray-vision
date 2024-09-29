import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifacts
from src.utils.main_utils import load_object


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts:
        try:
            logging.info(f"{'>>' * 20} Model Trainer {'<<' * 20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        except Exception as e:
            raise CustomException(e, sys)