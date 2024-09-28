import os
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file
from src.constants import *



@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_config = read_yaml_file(CONFIG_PATH)
        self.ZIP_FILE_URL = self.data_ingestion_config["zip_file_url"]
        self.ZIP_FILE_NAME = self.data_ingestion_config["zip_file_name"]
        self.DATA_INGESTION_ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)