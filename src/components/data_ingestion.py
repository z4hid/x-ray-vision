import os
import sys
from zipfile import ZipFile
from src.exception import CustomException   
from src.logger import logging
from src.configurations.data_download import download_from_gdrive

class DataIngestion:
    def __init__(self, data_ingestion_config):
        pass

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            pass

        except Exception as e:
            raise CustomException(e, sys)