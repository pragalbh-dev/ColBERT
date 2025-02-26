from typing import Dict, Any, Optional, Union
import os
import json
import yaml
from pathlib import Path
import copy
import time
# how does this manage a config for a particular experiment ka job. each job can have a different config so how do we know where is that stored.
class Config:
    def __init__(self, config_dict: Dict[str, Any], name: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_dict: Dictionary with configuration values
            name: Optional name for the configuration
        """
        self._config = copy.deepcopy(config_dict)
        self._name = name
        
        # Add metadata if not present
        if "_metadata" not in self._config:
            self._config["_metadata"] = {
                "created_at": time.time(),
                "version": "1.0",
                "name": name
            }
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary with values to update
        """
        # Don't update metadata through this method
        if "_metadata" in updates:
            del updates["_metadata"]
            
        # Update configuration
        self._config.update(updates)
        
        # Update version in metadata
        if "_metadata" in self._config:
            version = self._config["_metadata"].get("version", "1.0")
            # Simple version increment
            if "." in version:
                major, minor = version.split(".")
                new_version = f"{major}.{int(minor) + 1}"
            else:
                new_version = f"{version}.1"
            
            self._config["_metadata"]["version"] = new_version
            self._config["_metadata"]["updated_at"] = time.time()
    
    def get(self, key: str, default=None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration as dictionary
        """
        return copy.deepcopy(self._config)
    
    def save(self, path: str) -> str:
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration
            
        Returns:
            Path to saved configuration
        """
        # Determine file format from extension
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self._config, f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        else:
            # Default to YAML
            path = path.with_suffix('.yaml')
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
                
        return str(path)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value using dictionary syntax
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        if key not in self._config:
            raise KeyError(f"Configuration key '{key}' not found")
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._config

class ConfigManager:
    @staticmethod
    def load(config_path: str) -> Config:
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config object
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        # Extract name from filename or metadata
        name = None
        if "_metadata" in config_dict and "name" in config_dict["_metadata"]:
            name = config_dict["_metadata"]["name"]
        else:
            name = path.stem
            
        return Config(config_dict, name)
    
    @staticmethod
    def save(config: Config, path: str) -> str:
        """
        Save configuration to file
        
        Args:
            config: Config object
            path: Path to save configuration
            
        Returns:
            Path to saved configuration
        """
        return config.save(path)
    
    @staticmethod
    def create(name: str, config_dict: Dict[str, Any] = None) -> Config:
        """
        Create a new configuration
        
        Args:
            name: Configuration name
            config_dict: Optional initial configuration values
            
        Returns:
            Config object
        """
        if config_dict is None:
            config_dict = {}
            
        return Config(config_dict, name)
    
    @staticmethod
    def create_default(experiment_type: str) -> Config:
        """
        Create default configuration for experiment type
        
        Args:
            experiment_type: Type of experiment
            
        Returns:
            Config object with default values
        """
        # Default configurations for different experiment types
        defaults = {
            "image_classification": {
                "model_type": "resnet18",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "optimizer": "adam",
                "weight_decay": 0.0001,
                "momentum": 0.9,
                "scheduler": "step",
                "step_size": 7,
                "gamma": 0.1
            },
            "text_classification": {
                "model_type": "bert-base-uncased",
                "learning_rate": 2e-5,
                "batch_size": 16,
                "epochs": 4,
                "max_length": 128,
                "optimizer": "adamw",
                "weight_decay": 0.01,
                "warmup_steps": 500
            },
            "object_detection": {
                "model_type": "faster_rcnn",
                "learning_rate": 0.005,
                "batch_size": 8,
                "epochs": 20,
                "optimizer": "sgd",
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "step_size": 3,
                "gamma": 0.1
            }
        }
        
        # Get default configuration for experiment type
        if experiment_type in defaults:
            config_dict = defaults[experiment_type]
        else:
            # Generic defaults
            config_dict = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10
            }
            
        return Config(config_dict, f"{experiment_type}_default")
    
    @staticmethod
    def create_from(base_config: Config) -> Config:
        """
        Create a new configuration based on an existing one
        
        Args:
            base_config: Base configuration
            
        Returns:
            New Config object with same values
        """
        config_dict = base_config.to_dict()
        
        # Update metadata
        if "_metadata" in config_dict:
            metadata = config_dict["_metadata"]
            metadata["parent_version"] = metadata.get("version", "1.0")
            metadata["version"] = "1.0"  # Reset version for new config
            metadata["created_at"] = time.time()
            
        name = f"{base_config._name}_derived" if base_config._name else "derived_config"
        
        return Config(config_dict, name) 