import os
import sys
import torch
from src.logger import logging
from src.exception import CustomException
from torchvision import transforms, datasets
from src.utils.main_utils import save_object
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        """
        DataTransformation: Constructor to initialize the DataTransformation class.
        Parameters:
        data_transformation_config (DataTransformationConfig): DataTransformationConfig object
        data_ingestion_artifact (DataIngestionArtifacts): DataIngestionArtifacts object
        """
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.std = self.data_transformation_config.STD
            self.mean  = self.data_transformation_config.MEAN
            self.img_size = self.data_transformation_config.IMG_SIZE
            self.degree_n = self.data_transformation_config.DEGREE_N
            self.degree_p = self.data_transformation_config.DEGREE_P
            self.train_ratio = self.data_transformation_config.TRAIN_RATIO
            self.valid_ratio = self.data_transformation_config.VALID_RATIO
            
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def get_transform_data(self):
        """
        get_transform_data: This method returns the data transformation object.
        Parameters:
        None
        Returns:
        data_transform (transforms.Compose): Data transformation object
        """
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


    def split_data(self, dataset, total_count):
        try:
            logging.info("Entered the split_data method of data transformation class")
            train_count = int(self.train_ratio * total_count)
            valid_count = int(self.valid_ratio * total_count)
            test_count = total_count - train_count - valid_count
            train_data, valid_data, test_data = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))
            logging.info("Exited the split_data method of data transformation class")
            return train_data, valid_data, test_data

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of data transformation class")
            dataset = datasets.ImageFolder(self.data_ingestion_artifact.dataset_path, transform=self.get_transform_data())
            total_count = len(dataset)
            logging.info(f"Total number of images in the dataset: {total_count}")

            classes = len(os.listdir(self.data_ingestion_artifact.dataset_path))
            logging.info(f"Total number of classes in the dataset: {classes}")

            train_dataset, valid_dataset, test_dataset = self.split_data(dataset, total_count)
            logging.info("Split dataset into train, validation, and test datasets")
            
            save_object(file_path=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, obj=train_dataset)
            save_object(file_path=self.data_transformation_config.VALID_TRANSFORM_OBJECT_FILE_PATH, obj=valid_dataset)
            save_object(file_path=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, obj=test_dataset)
            logging.info("Saved train, validation, and test datasets as objects")
            
            data_transformation_artifact = DataTransformationArtifacts(
                train_transformed_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                valid_transformed_object=self.data_transformation_config.VALID_TRANSFORM_OBJECT_FILE_PATH,
                test_transformed_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH,
                classes=classes
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            logging.info("Exited the initiate_data_transformation method of data transformation class")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
