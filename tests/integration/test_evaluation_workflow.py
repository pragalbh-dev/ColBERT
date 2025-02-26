import unittest
import os
import shutil
from tempfile import TemporaryDirectory
from tracking.evaluation_tracker import EvaluationTracker
from config.config_manager import ConfigManager
from artifacts.artifact_manager import ArtifactManager
from training_workflow.jobs.evaluation_job import EvaluationJob

class TestEvaluationWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.base_dir = self.temp_dir.name
        
        # Create directories
        self.config_dir = os.path.join(self.base_dir, "configs")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.artifacts_dir = os.path.join(self.base_dir, "artifacts")
        
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Create test config
        self.config_manager = ConfigManager(config_dir=self.config_dir)
        self.test_config = {
            "experiment": {
                "name": "integration_test",
                "description": "Integration test for evaluation workflow"
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall"],
                "test_split": 0.2
            },
            "model": {
                "type": "random_forest",
                "params": {
                    "n_estimators": 100
                }
            }
        }
        self.config_manager.save_config(self.test_config, "test_eval_config.yaml")
        
        # Create components
        self.artifact_manager = ArtifactManager(artifacts_dir=self.artifacts_dir)
        
        # Create a dummy model artifact
        self.model_dir = os.path.join(self.artifacts_dir, "integration_test", "model")
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "model.pkl"), 'w') as f:
            f.write("dummy model data")
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_evaluation_job_execution(self):
        """Test the full evaluation job workflow"""
        # Create a run ID
        run_id = "eval_test_001"
        
        # Create tracker
        tracker = EvaluationTracker(
            experiment_name="integration_test",
            run_id=run_id,
            output_dir=self.output_dir
        )
        
        # Create and run evaluation job
        eval_job = EvaluationJob(
            config_path=os.path.join(self.config_dir, "test_eval_config.yaml"),
            run_id=run_id,
            tracker=tracker,
            artifact_manager=self.artifact_manager
        )
        
        # Mock the evaluation process
        def mock_evaluate():
            # Simulate evaluation by logging metrics
            tracker.log_metrics({
                "accuracy": 0.92,
                "precision": 0.90,
                "recall": 0.88,
                "f1_score": 0.89
            })
            # Save results
            tracker.save_results()
            return True
            
        # Replace the actual evaluate method with our mock
        eval_job.evaluate = mock_evaluate
        
        # Run the job
        result = eval_job.run()
        
        # Check that the job ran successfully
        self.assertTrue(result)
        
        # Check that results were saved
        results_path = os.path.join(
            self.output_dir,
            "integration_test",
            run_id,
            "evaluation_results.json"
        )
        self.assertTrue(os.path.exists(results_path))
        
        # Check that metrics were logged
        self.assertEqual(tracker.metrics["accuracy"], 0.92)
        self.assertEqual(tracker.metrics["precision"], 0.90)
        self.assertEqual(tracker.metrics["recall"], 0.88) 