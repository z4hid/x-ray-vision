import os
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file
from src.constants import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_config = read_yaml_file(CONFIG_PATH)
        self.ZIP_FILE_URL = self.data_ingestion_config["data_ingestion_config"]["zip_file_url"]
        self.ZIP_FILE_NAME = self.data_ingestion_config["data_ingestion_config"]["zip_file_name"]
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        
        
@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_config = read_yaml_file(CONFIG_PATH)
        self.STD: list = self.data_transformation_config['data_transformation_config']['std']
        self.MEAN: list = self.data_transformation_config['data_transformation_config']['mean']
        self.IMG_SIZE: int = self.data_transformation_config['data_transformation_config']['img_size']
        self.DEGREE_N: int = self.data_transformation_config['data_transformation_config']['degree_n']
        self.DEGREE_P: int = self.data_transformation_config['data_transformation_config']['degree_p']
        self.TRAIN_RATIO: float = self.data_transformation_config['data_transformation_config']['train_ratio']
        self.VALID_RATIO: float = self.data_transformation_config['data_transformation_config']['valid_ratio']
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TRAIN_TRANSFORM_OBJECT_FILE_NAME)
        self.VALID_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, VALID_TRANSFORM_OBJECT_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR, TEST_TRANSFORM_OBJECT_FILE_NAME)
        


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_config = read_yaml_file(CONFIG_PATH)
        self.LR: float = self.model_trainer_config['model_trainer_config']['lr']
        self.EPOCHS: int = self.model_trainer_config['model_trainer_config']['epochs']
        self.BATCH_SIZE: int = self.model_trainer_config['model_trainer_config']['batch_size']
        self.NUM_WORKERS: int = self.model_trainer_config['model_trainer_config']['num_workers']
        
        # Define the correct artifacts directory path for model trainer
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR, TRAINED_MODEL_PATH)

@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.model_evaluation_config = read_yaml_file(CONFIG_PATH)
        self.MODEL_REPO_NAME: str = self.model_evaluation_config['model_evaluation_config']['model_repo_name']
        self.MODEL_NAME: str = MODEL_NAME
        self.BATCH_SIZE: int = self.model_evaluation_config['model_evaluation_config']['batch_size']
        self.NUM_WORKERS: int = self.model_evaluation_config['model_evaluation_config']['num_workers']
        self.MODEL_EVALUATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR: str = os.path.join(self.MODEL_EVALUATION_ARTIFACTS_DIR, BEST_MODEL_DIR)
