import logging
from typing import Optional

class LoggerController:
    _instance = None
    _log_level = logging.INFO  # Default log level
    _logger = None

    @classmethod
    def initialize(cls, level: str = 'INFO') -> None:
        """
        Initialize the logger with a specific level.
        
        Args:
            level (str): Log level to set. Defaults to 'INFO'
        """
        if cls._instance is None:
            cls._instance = cls()
            
            level = level.upper()
            if level == 'DEBUG':
                cls._log_level = logging.DEBUG
            elif level == 'INFO':
                cls._log_level = logging.INFO
            elif level == 'WARNING':
                cls._log_level = logging.WARNING
            elif level == 'ERROR':
                cls._log_level = logging.ERROR
            elif level == 'CRITICAL':
                cls._log_level = logging.CRITICAL
            else:
                cls._log_level = logging.INFO
                
            cls._logger = logging.getLogger('PsycoreLogger')
            cls._logger.setLevel(cls._log_level)
            
            if not cls._logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(cls._log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                cls._logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Get the logger instance. Initializes if not already initialized.
        
        Returns:
            logging.Logger: The configured logger instance
        """
        if cls._instance is None:
            cls.initialize()
        return cls._logger

    @classmethod
    def get_log_level(cls) -> int:
        """
        Get the current log level.
        
        Returns:
            int: The current log level
        """
        return cls._log_level

    @classmethod
    def set_log_level(cls, level: str) -> None:
        """
        Set a new log level.
        
        Args:
            level (str): New log level to set
        """
        level = level.upper()
        if level == 'DEBUG':
            cls._log_level = logging.DEBUG
        elif level == 'INFO':
            cls._log_level = logging.INFO
        elif level == 'WARNING':
            cls._log_level = logging.WARNING
        elif level == 'ERROR':
            cls._log_level = logging.ERROR
        elif level == 'CRITICAL':
            cls._log_level = logging.CRITICAL
        else:
            cls._log_level = logging.INFO
            
        if cls._logger:
            cls._logger.setLevel(cls._log_level)
            for handler in cls._logger.handlers:
                handler.setLevel(cls._log_level) 