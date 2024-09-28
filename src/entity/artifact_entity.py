from dataclasses import dataclass
from src.constants import *


# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    dataset_path: str
    
    def to_dict(self):
        return self.__dict__
    
