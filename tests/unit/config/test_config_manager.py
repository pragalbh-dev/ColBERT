import unittest
import os
import yaml
from tempfile import TemporaryDirectory
from config.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.config_dir = self.temp_dir.name
        self.config_manager = ConfigManager(config_dir=self.config_dir)
        
        # Create a test config file
        self.test_config = {
            "experiment": {
                "name": "test_experiment",
                "description": "A test experiment"
            },
            "model": {
                "type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            },
            "training": {
                "batch_size": 32,
                "epochs": 10,
                "learning_rate": 0.001
            }
        }
        
        self.config_path = os.path.join(self.config_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_load_config(self):
        """Test loading a config file"""
        config = self.config_manager.load_config("test_config.yaml")
        
        # Check that the config was loaded correctly
        self.assertEqual(config, self.test_config)
        
    def test_save_config(self):
        """Test saving a config file"""
        new_config = {
            "experiment": {
                "name": "new_experiment",
                "description": "A new test experiment"
            },
            "model": {
                "type": "neural_network",
                "params": {
                    "hidden_layers": [64, 32],
                    "activation": "relu"
                }
            }
        }
        
        # Save the new config
        self.config_manager.save_config(new_config, "new_config.yaml")
        
        # Check that the file exists
        new_config_path = os.path.join(self.config_dir, "new_config.yaml")
        self.assertTrue(os.path.exists(new_config_path))
        
        # Load and check contents
        with open(new_config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
            
        self.assertEqual(loaded_config, new_config)
        
    def test_update_config(self):
        """Test updating an existing config file"""
        # Update some values in the config
        updates = {
            "model": {
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15
                }
            },
            "training": {
                "epochs": 20
            }
        }
        
        self.config_manager.update_config("test_config.yaml", updates)
        
        # Load the updated config
        updated_config = self.config_manager.load_config("test_config.yaml")
        
        # Check that the updates were applied correctly
        self.assertEqual(updated_config["model"]["params"]["n_estimators"], 200)
        self.assertEqual(updated_config["model"]["params"]["max_depth"], 15)
        self.assertEqual(updated_config["training"]["epochs"], 20)
        
        # Check that other values remain unchanged
        self.assertEqual(updated_config["experiment"]["name"], "test_experiment")
        self.assertEqual(updated_config["training"]["batch_size"], 32)
        
    def test_get_config_path(self):
        """Test getting the full path to a config file"""
        path = self.config_manager.get_config_path("test_config.yaml")
        expected_path = os.path.join(self.config_dir, "test_config.yaml")
        
        self.assertEqual(path, expected_path)
        
    def test_list_configs(self):
        """Test listing all available config files"""
        # Create another config file
        with open(os.path.join(self.config_dir, "another_config.yaml"), 'w') as f:
            yaml.dump({"test": "data"}, f)
            
        configs = self.config_manager.list_configs()
        
        # Check that both config files are listed
        self.assertIn("test_config.yaml", configs)
        self.assertIn("another_config.yaml", configs)
        self.assertEqual(len(configs), 2) 