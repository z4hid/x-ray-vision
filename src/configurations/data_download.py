import gdown
import os
import sys
from zipfile import ZipFile
from src.exception import CustomException   
from src.logger import logging

def download_from_gdrive(gdrive_url, save_path, is_folder=False):
    """
    Download a file or folder from Google Drive using gdown and save it to the specified path.
    
    Args:
        gdrive_url (str): The Google Drive URL (file or folder).
        save_path (str): The desired path to save the file or folder.
        is_folder (bool): Whether the link is a folder. Default is False (i.e., it is a file).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if is_folder:
        # Download a folder
        gdown.download_folder(url=gdrive_url, output=save_path, quiet=False)
        logging.info("Downloaded folder from Google Drive")
    else:
        # Download a file
        gdown.download(url=gdrive_url, output=save_path, quiet=False, fuzzy=True)
        logging.info("Downloaded file from Google Drive")

# # Example usage
# file_url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
# output_path = "./downloads/fcn8s_from_caffe.npz"
# download_from_gdrive(file_url, output_path)

# # Example for downloading a folder
# folder_url = "https://drive.google.com/drive/folders/15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
# folder_output_path = "./downloads/my_folder"
# download_from_gdrive(folder_url, folder_output_path, is_folder=True)
