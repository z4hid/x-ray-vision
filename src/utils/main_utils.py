import os
import sys
import dill
import yaml
import base64
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    This method saves the given python object to a file using dill library.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The python object which needs to be saved.

    Raises:
        CustomException: If there is any error while saving the object.

    Returns:
        None
    """
    try:
        logging.info("Entered the save_object method of utils")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    """
    This method loads the python object from a file using dill library.

    Args:
        file_path (str): The path to the file from where the object will be loaded.

    Raises:
        CustomException: If there is any error while loading the object.

    Returns:
        object: The python object which was loaded from the file.
    """
    try:
        logging.info("Entered the load_object method of utils")
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            logging.info("Exited the load_object method of utils")
            return obj
    except Exception as e:
        raise CustomException(e, sys)
    
    
def image_to_base64(image):

    """
    This method encodes an image file to base64.

    Args:
        image (str): The path to the image file.

    Raises:
        CustomException: If there is any error while encoding the image.

    Returns:
        bytes: The base64 encoded image.
    """
    try:
        logging.info("Entered the image_to_base64 method of utils")
        with open(image, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            logging.info("Exited the image_to_base64 method of utils")
            return encoded_string
    except Exception as e:
        raise CustomException(e, sys) 
    
    
def read_yaml_file(file_path):
    """
    This method reads a yaml file and returns its contents as a python object.

    Args:
        file_path (str): The path to the yaml file.

    Raises:
        CustomException: If there is any error while reading the yaml file.

    Returns:
        object: The python object which was read from the yaml file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)