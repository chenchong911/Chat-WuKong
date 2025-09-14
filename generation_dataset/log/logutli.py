import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_id, log_dir, log_name, log_level='info'):
        self.log_id = log_id
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_level = log_level.upper()
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 设置日志级别
        level_dict = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        # 创建logger
        self.logger = logging.getLogger(log_id)
        self.logger.setLevel(level_dict.get(log_level, logging.INFO))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建文件handler
            log_path = os.path.join(log_dir, log_name)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(level_dict.get(log_level, logging.INFO))
            
            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level_dict.get(log_level, logging.INFO))
            
            # 创建formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)