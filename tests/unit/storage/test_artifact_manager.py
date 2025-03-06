import unittest
import os
import tempfile
import shutil
import pickle
import json
from training_workflow.storage.artifact_manager import ArtifactManager

class TestArtifactManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for artifacts
        self.artifacts_dir = tempfile.mkdtemp()
        
        # Create an instance of ArtifactManager
        self.artifact_manager = ArtifactManager(base_dir=self.artifacts_dir)
        
        # Test data
        self.experiment_name = "test_experiment"
        self.run_id = "test_run_001"
        self.model_data = {"weights": [0.1, 0.2, 0.3], "bias": 0.5}
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.artifacts_dir)
    
    def test_save_and_load_model(self):
        """Test saving and loading a model artifact"""
        # Save the model
        model_path = self.artifact_manager.save_model(
            self.model_data,
            self.experiment_name,
            self.run_id
        )
        
        # Verify the model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Load the model
        loaded_model = self.artifact_manager.load_model(
            self.experiment_name,
            self.run_id
        )
        
        # Verify the loaded model matches the original
        self.assertEqual(loaded_model, self.model_data)
    
    def test_save_and_load_metrics(self):
        """Test saving and loading metrics"""
        # Create test metrics
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.91
        }
        
        # Save the metrics
        metrics_path = self.artifact_manager.save_metrics(
            metrics,
            self.experiment_name,
            self.run_id
        )
        
        # Verify the metrics were saved
        self.assertTrue(os.path.exists(metrics_path))
        
        # Load the metrics
        loaded_metrics = self.artifact_manager.load_metrics(
            self.experiment_name,
            self.run_id
        )
        
        # Verify the loaded metrics match the original
        self.assertEqual(loaded_metrics, metrics)
    
    def test_list_artifacts(self):
        """Test listing artifacts for an experiment"""
        # Save multiple artifacts
        self.artifact_manager.save_model(
            self.model_data,
            self.experiment_name,
            self.run_id
        )
        
        metrics = {"accuracy": 0.95}
        self.artifact_manager.save_metrics(
            metrics,
            self.experiment_name,
            self.run_id
        )
        
        # List artifacts
        artifacts = self.artifact_manager.list_artifacts(self.experiment_name)
        
        # Verify the artifacts list contains the expected items
        self.assertIn(self.run_id, artifacts)
        
        # Check specific artifact types
        run_artifacts = self.artifact_manager.list_artifacts(
            self.experiment_name,
            self.run_id
        )
        self.assertIn("model", run_artifacts)
        self.assertIn("metrics", run_artifacts)
    
    def test_delete_artifact(self):
        """Test deleting an artifact"""
        # Save a model
        model_path = self.artifact_manager.save_model(
            self.model_data,
            self.experiment_name,
            self.run_id
        )
        
        # Verify the model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Delete the model
        self.artifact_manager.delete_artifact(
            self.experiment_name,
            "model",
            self.run_id
        )
        
        # Verify the model was deleted
        self.assertFalse(os.path.exists(model_path)) 