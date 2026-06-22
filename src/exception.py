
import sys

def get_error_details(error, sys_module):
    _,_, exc_tb= sys_module.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)
    
    return(
        f"\n{'='*60}\n"
        f"ERROR DETAILS:\n"
        f"  File    : {file_name}\n"
        f"  Line    : {line_number}\n"
        f"  Message : {error_message}\n"
        f"{'='*60}"
    )
    

class CustomException(Exception):
    def __init__(self, error, sys_module):
        self.error_message = get_error_details(error,sys_module )
        super().__init__(self.error_message)
        
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"CustomException({self.error_message})"
    
    
        