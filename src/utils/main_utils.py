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