from tracking.base_tracker import BaseTracker

class MockEvaluationTracker(BaseTracker):
    """Mock implementation of EvaluationTracker for testing"""
    def __init__(self, experiment_name, run_id, output_dir):
        super().__init__(experiment_name, run_id, output_dir)
        self.metrics = {}
        
    def log_metric(self, name, value):
        """Log a single metric"""
        self.metrics[name] = value
        
    def log_metrics(self, metrics_dict):
        """Log multiple metrics at once"""
        self.metrics.update(metrics_dict)
        
    def save_results(self):
        """Save evaluation results to disk"""
        import os
        import json
        
        # Create directory if it doesn't exist
        results_dir = os.path.join(self.output_dir, self.experiment_name, self.run_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results to file
        results_path = os.path.join(results_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "run_id": self.run_id,
                "metrics": self.metrics
            }, f, indent=2)
            
    def load_results(self):
        """Load evaluation results from disk"""
        import os
        import json
        
        results_path = os.path.join(
            self.output_dir,
            self.experiment_name,
            self.run_id,
            "evaluation_results.json"
        )
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                data = json.load(f)
                self.metrics = data.get("metrics", {}) 