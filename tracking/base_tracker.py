class BaseTracker:
    """Base class for all trackers"""
    def __init__(self, experiment_name, run_id, output_dir):
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.output_dir = output_dir 