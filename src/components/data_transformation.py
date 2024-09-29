import os
import sys
import torch
from src.logger import logging
from src.exception import CustomException
from torchvision import transforms, datasets
from src.utils.main_utils import save_object
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.std = self.data_transformation_config.STD
            self.mean  = self.data_transformation_config.MEAN
            self.img_size = self.data_transformation_config.IMG_SIZE
            self.degree_n = self.data_transformation_config.DEGREE_N
            self.degree_p = self.data_transformation_config.DEGREE_P
            self.train_ration = self.data_transformation_config.TRAIN_RATIO
            self.valid_ratio = self.data_transformation_config.VALID_RATIO
            
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_transform_data(self):
        try:
            logging.info("Entered the get_transform_data method of data transformation class")
            data_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(degrees=(self.degree_n, self.degree_p)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            
            logging.info("Exited the get_transform_data method of data transformation class")
            return data_transform
        
        except Exception as e:
            raise CustomException(e, sys)


    


