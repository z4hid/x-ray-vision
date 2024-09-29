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
        
