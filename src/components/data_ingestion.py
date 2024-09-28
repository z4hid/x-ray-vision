import os
import sys
from zipfile import ZipFile
from src.exception import CustomException   
from src.logger import logging
from src.configurations.data_download import download_from_gdrive
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            self.get_data_from_drive()
            self.extract_zip_file()
            logging.info("Completed data ingestion process")
            
            self.data_ingestion_artifacts = DataIngestionArtifacts(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
            )

        except Exception as e:
            raise CustomException(e, sys)        
    def get_data_from_drive(self):
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
        try:
            zip_path = self.data_ingestion_config.ZIP_FILE_PATH
            extract_dir = os.path.dirname(zip_path)

            # Unzip the file
            logging.info(f"Extracting files from {zip_path}")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                logging.info(f"Successfully extracted files to {extract_dir}")
            
            # Clean up the zip file after extraction
            if os.path.exists(zip_path):
                os.remove(zip_path)
                logging.info(f"Removed the zip file: {zip_path}")
            else:
                logging.warning(f"Zip file not found for deletion: {zip_path}")

        except Exception as e:
            raise CustomException(e, sys)
