from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_config = read_yaml_file(CONFIG_PATH)