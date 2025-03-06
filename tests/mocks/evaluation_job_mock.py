class EvaluationJobMock:
    """Mock implementation of EvaluationJob for testing"""
    def __init__(self, config_path, run_id, tracker, artifact_manager):
        self.config_path = config_path
        self.run_id = run_id
        self.tracker = tracker
        self.artifact_manager = artifact_manager
        
    def run(self):
        """Mock run method"""
        # Call evaluate to log metrics
        self.evaluate()
        return True
        
    def evaluate(self):
        """Mock evaluate method"""
        # Log some dummy metrics
        self.tracker.log_metrics({
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.88,
            "f1_score": 0.89
        })
        # Save results
        self.tracker.save_results()
        return True 