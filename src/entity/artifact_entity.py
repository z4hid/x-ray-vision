from dataclasses import dataclass
from src.constants import *


# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    dataset_path: str
    
    def to_dict(self):
        return self.__dict__
    

# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    train_transformed_object: str
    valid_transformed_object: str
    test_transformed_object: str
    classes: int
    
    def to_dict(self):
        return self.__dict__
 
 
 # Model Training Artifacts   
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    
    def to_dict(self):
        return self.__dict__