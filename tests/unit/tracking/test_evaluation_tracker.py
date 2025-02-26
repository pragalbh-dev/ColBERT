import unittest
import os
import json
import shutil
from tempfile import TemporaryDirectory
from tracking.evaluation_tracker import EvaluationTracker
from tracking.base_tracker import BaseTracker

class TestEvaluationTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        self.experiment_name = "test_experiment"
        self.run_id = "test_run_123"
        self.tracker = EvaluationTracker(
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            output_dir=self.output_dir
        )
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_inheritance(self):
        """Test that EvaluationTracker inherits from BaseTracker"""
        self.assertIsInstance(self.tracker, BaseTracker)
        
    def test_log_metric(self):
        """Test logging a single metric"""
        metric_name = "accuracy"
        metric_value = 0.95
        
        self.tracker.log_metric(metric_name, metric_value)
        
        # Check that the metric was stored in memory
        self.assertIn(metric_name, self.tracker.metrics)
        self.assertEqual(self.tracker.metrics[metric_name], metric_value)
        
    def test_log_metrics(self):
        """Test logging multiple metrics at once"""
        metrics = {
            "precision": 0.92,
            "recall": 0.89,
            "f1_score": 0.90
        }
        
        self.tracker.log_metrics(metrics)
        
        # Check that all metrics were stored
        for name, value in metrics.items():
            self.assertIn(name, self.tracker.metrics)
            self.assertEqual(self.tracker.metrics[name], value)
            
    def test_save_results(self):
        """Test saving evaluation results to disk"""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.89
        }
        self.tracker.log_metrics(metrics)
        
        # Save results
        self.tracker.save_results()
        
        # Check that the results file exists
        results_path = os.path.join(
            self.output_dir,
            self.experiment_name,
            self.run_id,
            "evaluation_results.json"
        )
        self.assertTrue(os.path.exists(results_path))
        
        # Check file contents
        with open(results_path, 'r') as f:
            saved_data = json.load(f)
            
        self.assertEqual(saved_data["metrics"], metrics)
        self.assertEqual(saved_data["experiment_name"], self.experiment_name)
        self.assertEqual(saved_data["run_id"], self.run_id)
        
    def test_load_results(self):
        """Test loading evaluation results from disk"""
        # First save some results
        metrics = {"accuracy": 0.95, "loss": 0.12}
        self.tracker.log_metrics(metrics)
        self.tracker.save_results()
        
        # Create a new tracker and load results
        new_tracker = EvaluationTracker(
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            output_dir=self.output_dir
        )
        new_tracker.load_results()
        
        # Check that metrics were loaded correctly
        self.assertEqual(new_tracker.metrics, metrics) 