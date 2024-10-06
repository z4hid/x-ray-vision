import os
import sys
from zipfile import ZipFile
from src.exception import CustomException   
from src.logger import logging
from src.configurations.data_download import download_from_gdrive
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    """
    DataIngestion handles the process of downloading a dataset from Google Drive, extracting it, 
    and managing the artifacts of the data ingestion process.
    
    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration class for data ingestion that contains
        paths and URLs required during the ingestion process.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion instance with a data ingestion configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration containing data ingestion parameters,
            such as the URL for downloading the dataset, file paths, and artifact storage.
        """
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process, which includes downloading the dataset from Google Drive,
        extracting the downloaded zip file, and saving the ingestion artifacts.

        Returns:
            DataIngestionArtifacts: Object containing paths or references to the artifacts created during
            the data ingestion process.
        
        Raises:
            CustomException: If an error occurs during data ingestion.
        """
        logging.info("Starting data ingestion process")
        try:
            self.get_data_from_drive()
            self.extract_zip_file()
            
            self.data_ingestion_artifacts = DataIngestionArtifacts(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
            )
            logging.info("Completed data ingestion process")
            return self.data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys)        

    def get_data_from_drive(self):
        """
        Downloads a dataset from Google Drive using a specified URL and saves it to a specified path.
        
        Raises:
            CustomException: If an error occurs while downloading the file from Google Drive.
        """
        try:
            logging.info("Downloading dataset from Google Drive")
            os.makedirs(os.path.dirname(self.data_ingestion_config.ZIP_FILE_PATH), exist_ok=True)
            download_from_gdrive(
                gdrive_url=self.data_ingestion_config.ZIP_FILE_URL, 
                save_path=self.data_ingestion_config.ZIP_FILE_PATH
            )
            logging.info(f"Downloaded dataset to {self.data_ingestion_config.ZIP_FILE_PATH}")
        except Exception as e:
            raise CustomException(e, sys)

    def extract_zip_file(self):
        """
        Extracts a zip file from the download location and removes the zip file afterward.
        
        Raises:
            CustomException: If an error occurs while extracting the zip file.
        """
        try:
            zip_path = self.data_ingestion_config.ZIP_FILE_PATH
            extract_dir = os.path.dirname(zip_path)

            logging.info(f"Extracting files from {zip_path}")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logging.info(f"Successfully extracted files to {extract_dir}")
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logging.info(f"Removed the zip file: {zip_path}")
            else:
                logging.warning(f"Zip file not found for deletion: {zip_path}")

        except Exception as e:
            raise CustomException(e, sys)

