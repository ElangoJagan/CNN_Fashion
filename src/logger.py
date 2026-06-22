import logging
import os
from datetime import datetime
from typing import Dict

class Logger:
    
    _instances = {}
    
    def __init__(self, name):
        self.name = name
        self.log_dir = 'logs'
        self.log_file = self._get_log_file_path()
    
    def _get_log_file_path(self):
        os.makedirs(self.log_dir, exist_ok =True)
        log_filename = f'{datetime.now().strftime('%Y_%m_%d')}.log'
        
        return os.path.join(self.log_dir, log_filename)
    
    def _setup_logger(self):
        logger =logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s -%(message)s", datefmt = "%Y-%m-%d %H:%M:%S")   
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
            
        console_handler =  logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
            
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
            
        return logger
    def get_logger(self):
        if self.name not in Logger._instances:
            Logger._instances[self.name] = self._setup_logger()
        return Logger._instances[self.name]
        