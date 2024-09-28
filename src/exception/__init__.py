import os
import sys

def error_message_detail(error, error_detail:sys):
    """
    This function will return the error message in detail
    :param error: the error object
    :param error_detail: the error detail object
    :return: the error message in detail
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        """
        This method is used to initialize the object of this class.
        It takes error message and error detail as parameter and returns the object of this class.
        The error message is a string which contains the error message.
        The error detail is an object of sys module which contains the information of error.
        It calls the __init__ method of Exception class and passes the error message to it.
        It also calls the error_message_detail method and passes the error message and error detail to it.
        The result of error_message_detail method is stored in error_message attribute of the object.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        This method is used to get the error message.
        It returns the error message as a string.
        The error message is the message which is passed to the constructor of the class.
        It also includes the information of the error like the file name and the line number
        where the error has occurred.
        """
       
        return self.error_message
    
    