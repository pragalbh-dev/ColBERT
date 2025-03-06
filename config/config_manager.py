import os
import yaml

class ConfigManager:
    """Manager for configuration files"""
    def __init__(self, config_dir):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
    def load_config(self, config_name):
        """Load a configuration file"""
        config_path = self.get_config_path(config_name)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
        
    def save_config(self, config, config_name):
        """Save a configuration to a file"""
        config_path = self.get_config_path(config_name)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
    def update_config(self, config_name, updates):
        """Update an existing configuration file"""
        config = self.load_config(config_name)
        self._deep_update(config, updates)
        self.save_config(config, config_name)
        
    def get_config_path(self, config_name):
        """Get the full path to a config file"""
        return os.path.join(self.config_dir, config_name)
        
    def list_configs(self):
        """List all available config files"""
        return [f for f in os.listdir(self.config_dir) if f.endswith('.yaml')]
        
    def _deep_update(self, d, u):
        """Recursively update a dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v 