import os
import sys
import torch
from tqdm import tqdm
from src.constants import DEVICE
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class ModelEvaluation:
    pass
