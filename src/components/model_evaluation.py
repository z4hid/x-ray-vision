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
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from huggingface_hub import hf_hub_download

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifacts: ModelTrainerArtifacts, data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        
    def get_best_model_from_huggingface(self):
        try:
            logging.info("Entered the get_best_model_from_huggingface method of model evaluation class")
            model_path = hf_hub_download(repo_id=self.model_evaluation_config.MODEL_REPO_NAME,
                                         filename=self.model_evaluation_config.MODEl_NAME)
            
            model = torch.load(model_path)
            return model
        except Exception as e:
            raise CustomException(e, sys)
