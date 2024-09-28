import os
import sys
import torch
from src.logger import logging
from src.exception import CustomException
from torchvision import transforms, datasets
from src.utils.main_utils import save_object



class DataTransformation:
    def __init__(self, data_transformation_config):
        try:
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)




