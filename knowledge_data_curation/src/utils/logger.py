import os
import logging
from typing import Dict, Optional
import yaml
from pathlib import Path

class Logger:
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def setup_logger(
        cls, 
        name: str, 
        config: Dict, 
        module_name: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up and return a logger with the given name and configuration.
        
        Args:
            name: The name of the logger
            config: The logging configuration
            module_name: Optional module name for file logging
            
        Returns:
            The configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Set to lowest level to catch all
        logger.propagate = False  # Don't propagate to root logger
        
        # Clear any existing handlers
        if logger.handlers:
            logger.handlers.clear()
            
        # Get log levels from config
        console_level = getattr(logging, config["logging"]["console_level"])
        file_level = getattr(logging, config["logging"]["file_level"])
        
        # Create logs directory if it doesn't exist
        logs_dir = Path(config["paths"]["logs_dir"])
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Add file handler for the unified log
        unified_log_path = logs_dir / 'unified.log'
        unified_file_handler = logging.FileHandler(unified_log_path)
        unified_file_handler.setLevel(file_level)
        unified_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        unified_file_handler.setFormatter(unified_formatter)
        logger.addHandler(unified_file_handler)
        
        # Add module-specific file handler if module_name is provided
        if module_name:
            module_log_path = logs_dir / f'{module_name}.log'
            module_file_handler = logging.FileHandler(module_log_path)
            module_file_handler.setLevel(file_level)
            module_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            module_file_handler.setFormatter(module_formatter)
            logger.addHandler(module_file_handler)
        
        cls._loggers[name] = logger
        return logger
        
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get an existing logger or raise an error if it doesn't exist.
        
        Args:
            name: The name of the logger to retrieve
            
        Returns:
            The requested logger
            
        Raises:
            KeyError: If the logger with the given name doesn't exist
        """
        if name not in cls._loggers:
            raise KeyError(f"Logger '{name}' has not been set up. Call setup_logger first.")
        return cls._loggers[name]
        
    @classmethod
    def load_config(cls, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The loaded configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) 