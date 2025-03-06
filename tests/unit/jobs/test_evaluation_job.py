import unittest
import os
import tempfile
import shutil
import yaml
import json
from training_workflow.jobs.evaluation_job import EvaluationJob
from training_workflow.storage.artifact_manager import ArtifactManager
from tracking.evaluation_tracker import EvaluationTracker

class TestEvaluationJob(unittest.TestCase):
    def setUp(self):
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "outputs")
        self.artifacts_dir = os.path.join(self.test_dir, "artifacts")
        self.config_dir = os.path.join(self.test_dir, "configs")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Test data
        self.experiment_name = "test_experiment"
        self.run_id = "test_run_001"
        
        # Create a test config
        self.config = {
            "experiment": {
                "name": self.experiment_name,
                "description": "Test experiment for evaluation job"
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall"],
                "batch_size": 32
            },
            "model": {
                "type": "random_forest"
            }
        }
        
        # Save the config
        self.config_path = os.path.join(self.config_dir, "test_config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        # Create a test model
        self.model_dir = os.path.join(self.artifacts_dir, self.experiment_name, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "model.pkl"), 'w') as f:
            f.write("dummy model data")
        
        # Create components
        self.artifact_manager = ArtifactManager(artifacts_dir=self.artifacts_dir)
        self.tracker = EvaluationTracker(
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            output_dir=self.output_dir
        )
    
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test initializing an evaluation job"""
        # Create an evaluation job
        job = EvaluationJob(
            job_id=self.run_id,
            experiment_name=self.experiment_name,
            evaluation_script="dummy_script.py",
            model_path=os.path.join(self.model_dir, "model.pkl"),
            dataset_path="dummy_dataset.csv",
            config=self.config,
            output_dir=self.output_dir,
            artifact_manager=self.artifact_manager
        )
        
        # Verify the job was initialized correctly
        self.assertEqual(job._job_id, self.run_id)
        self.assertEqual(job._experiment_name, self.experiment_name)
        self.assertEqual(job._config, self.config)
    
    def test_run_job(self):
        """Test running an evaluation job"""
        # Create an evaluation job with a mock evaluation method
        job = EvaluationJob(
            job_id=self.run_id,
            experiment_name=self.experiment_name,
            evaluation_script="dummy_script.py",
            model_path=os.path.join(self.model_dir, "model.pkl"),
            dataset_path="dummy_dataset.csv",
            config=self.config,
            output_dir=self.output_dir,
            artifact_manager=self.artifact_manager
        )
        
        # Replace the actual evaluation method with a mock
        def mock_evaluate():
            # Log some metrics
            job._tracker = self.tracker
            job._tracker.log_metrics({
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.88
            })
            return {"accuracy": 0.92, "precision": 0.90, "recall": 0.88}
        
        job._run_evaluation = mock_evaluate
        
        # Run the job
        results = job.run()
        
        # Verify the job ran successfully
        self.assertIsNotNone(results)
        self.assertEqual(results["accuracy"], 0.92)
        
        # Check that results were saved
        results_path = os.path.join(
            self.output_dir,
            self.experiment_name,
            self.run_id,
            "evaluation_results.json"
        )
        self.assertTrue(os.path.exists(results_path)) 