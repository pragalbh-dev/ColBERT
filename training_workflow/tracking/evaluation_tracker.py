from typing import Dict, List, Any, Callable, Optional, Union
import os
import time
import json
from pathlib import Path
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .base_tracker import BaseTracker

class EvaluationTracker(BaseTracker):
    def __init__(self, 
                 experiment_name: str, 
                 job_id: str, 
                 log_dir: Optional[str] = None,
                 artifact_manager=None):
        """
        Initialize evaluation tracker
        
        Args:
            experiment_name: Name of the experiment
            job_id: ID of the evaluation job
            log_dir: Directory for logs (default: experiments/{experiment_name}/eval_{job_id})
            artifact_manager: Optional artifact manager instance
        """
        # Initialize base tracker
        if log_dir is None:
            log_dir = f"experiments/{experiment_name}/eval_{job_id}"
        
        super().__init__(experiment_name, job_id, log_dir, artifact_manager)
        
        # Create additional directories specific to evaluation
        self.results_dir = self.log_dir / "results"
        self.visualizations_dir = self.log_dir / "visualizations"
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Evaluation-specific settings
        self.evaluation_complete = False
        self.evaluation_results = {}
        self.baseline_results = None
        
        print(f"Evaluation tracker initialized for experiment '{experiment_name}', job '{job_id}'")
        print(f"Logs will be saved to {self.log_dir}")
    
    def log_evaluation_result(self, 
                             name: str, 
                             value: Any, 
                             commit: bool = True) -> None:
        """
        Log evaluation result
        
        Args:
            name: Metric name
            value: Metric value
            commit: Whether to immediately write to disk
        """
        self.evaluation_results[name] = value
        
        # Also log as a metric for consistency
        self.log_metric(name, value, commit=commit)
    
    def log_evaluation_results(self, 
                              results: Dict[str, Any], 
                              commit: bool = True) -> None:
        """
        Log multiple evaluation results
        
        Args:
            results: Dictionary of metric names and values
            commit: Whether to immediately write to disk
        """
        for name, value in results.items():
            self.log_evaluation_result(name, value, commit=False)
        
        if commit:
            self.flush()
    
    def set_baseline_results(self, baseline_results: Dict[str, Any]) -> None:
        """
        Set baseline results for comparison
        
        Args:
            baseline_results: Dictionary of baseline metric names and values
        """
        self.baseline_results = baseline_results
        
        # Save baseline results
        baseline_path = self.results_dir / "baseline_results.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
    
    def compare_with_baseline(self, 
                             metrics: Optional[List[str]] = None, 
                             save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Compare current results with baseline
        
        Args:
            metrics: List of metrics to compare (default: all metrics)
            save_path: Path to save comparison visualization
            
        Returns:
            Dictionary with relative differences
        """
        if self.baseline_results is None:
            raise ValueError("Baseline results not set")
        
        if metrics is None:
            # Use all metrics that are in both current and baseline results
            metrics = [m for m in self.evaluation_results.keys() 
                      if m in self.baseline_results]
        
        # Calculate relative differences
        rel_diff = {}
        for metric in metrics:
            if metric in self.evaluation_results and metric in self.baseline_results:
                current = self.evaluation_results[metric]
                baseline = self.baseline_results[metric]
                
                if baseline != 0:
                    rel_diff[metric] = (current - baseline) / baseline * 100
                else:
                    rel_diff[metric] = float('inf') if current > 0 else 0
        
        # Create visualization
        if save_path is None:
            save_path = str(self.visualizations_dir / "baseline_comparison.png")
        
        self._create_comparison_visualization(metrics, save_path)
        
        # Save comparison results
        comparison_path = self.results_dir / "baseline_comparison.json"
        comparison_results = {
            "baseline": {m: self.baseline_results.get(m) for m in metrics},
            "current": {m: self.evaluation_results.get(m) for m in metrics},
            "relative_difference": rel_diff
        }
        
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        return rel_diff
    
    def _create_comparison_visualization(self, 
                                        metrics: List[str], 
                                        save_path: str) -> None:
        """
        Create comparison visualization
        
        Args:
            metrics: List of metrics to visualize
            save_path: Path to save visualization
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set up data
        x = np.arange(len(metrics))
        width = 0.35
        
        # Get values
        baseline_values = [self.baseline_results.get(m, 0) for m in metrics]
        current_values = [self.evaluation_results.get(m, 0) for m in metrics]
        
        # Create bars
        plt.bar(x - width/2, baseline_values, width, label='Baseline')
        plt.bar(x + width/2, current_values, width, label='Current')
        
        # Add labels and title
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Comparison with Baseline')
        plt.xticks(x, metrics, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        
        # Also add to TensorBoard
        self.add_figure("baseline_comparison", plt.gcf())
        
        plt.close()
        
        # Create relative difference visualization
        plt.figure(figsize=(10, 6))
        
        # Calculate relative differences
        rel_diff = []
        for i, metric in enumerate(metrics):
            if self.baseline_results.get(metric, 0) != 0:
                diff = (current_values[i] - baseline_values[i]) / baseline_values[i] * 100
                rel_diff.append(diff)
            else:
                rel_diff.append(0)
        
        # Create bars with color based on positive/negative
        colors = ['green' if d >= 0 else 'red' for d in rel_diff]
        plt.bar(metrics, rel_diff, color=colors)
        
        # Add labels and title
        plt.xlabel('Metrics')
        plt.ylabel('Relative Difference (%)')
        plt.title('Relative Difference from Baseline')
        plt.xticks(rotation=45, ha="right")
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        rel_diff_path = str(Path(save_path).parent / "relative_difference.png")
        plt.savefig(rel_diff_path)
        
        # Also add to TensorBoard
        self.add_figure("relative_difference", plt.gcf())
        
        plt.close()
    
    def create_confusion_matrix(self, 
                               confusion_matrix: np.ndarray, 
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Create confusion matrix visualization
        
        Args:
            confusion_matrix: Confusion matrix as numpy array
            class_names: Optional list of class names
            save_path: Path to save visualization
        """
        if save_path is None:
            save_path = str(self.visualizations_dir / "confusion_matrix.png")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Use seaborn for better visualization
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        # Add labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        
        # Also add to TensorBoard
        self.add_figure("confusion_matrix", plt.gcf())
        
        plt.close()
        
        # Save confusion matrix data
        cm_path = self.results_dir / "confusion_matrix.json"
        cm_data = {
            "matrix": confusion_matrix.tolist(),
            "class_names": class_names
        }
        
        with open(cm_path, "w") as f:
            json.dump(cm_data, f, indent=2)
    
    def create_metrics_summary(self, save_path: Optional[str] = None) -> None:
        """
        Create summary visualization of all metrics
        
        Args:
            save_path: Path to save visualization
        """
        if save_path is None:
            save_path = str(self.visualizations_dir / "metrics_summary.png")
        
        # Get metrics
        metrics = list(self.evaluation_results.keys())
        values = list(self.evaluation_results.values())
        
        # Filter out non-numeric values
        numeric_metrics = []
        numeric_values = []
        for i, value in enumerate(values):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics.append(metrics[i])
                numeric_values.append(value)
        
        if not numeric_metrics:
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create bars
        plt.bar(numeric_metrics, numeric_values)
        
        # Add labels and title
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Evaluation Metrics Summary')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        
        # Also add to TensorBoard
        self.add_figure("metrics_summary", plt.gcf())
        
        plt.close()
    
    def complete_evaluation(self) -> Dict[str, Any]:
        """
        Mark evaluation as complete and save final results
        
        Returns:
            Dictionary with evaluation results
        """
        self.evaluation_complete = True
        
        # Save final results
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Create metrics summary visualization
        self.create_metrics_summary()
        
        # Save final results to root directory for easy access
        final_results_path = self.log_dir / "results.json"
        with open(final_results_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        return self.evaluation_results
    
    def close(self) -> None:
        """Close tracker and release resources"""
        # If evaluation not explicitly completed, complete it now
        if not self.evaluation_complete:
            self.complete_evaluation()
        
        # Save final metrics
        self.flush()
        
        # Log total duration
        duration = time.time() - self.start_time
        with open(self.log_dir / "duration.txt", "w") as f:
            f.write(f"Total duration: {duration:.2f} seconds")
        
        # Update job metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "job_id": self.job_id,
            "status": "completed",
            "start_time": self.start_time,
            "end_time": time.time(),
            "duration": duration,
            "evaluation_complete": self.evaluation_complete
        }
        
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Call parent close method
        super().close()
    
    # Decorator for evaluation functions
    @staticmethod
    def track(experiment_name: str, job_id: str, log_dir: Optional[str] = None):
        """
        Decorator for tracking evaluation functions
        
        Example:
            @EvaluationTracker.track("experiment1", "eval_job1")
            def evaluate(model, dataloader, tracker=None):
                # Evaluation code using tracker
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with EvaluationTracker(experiment_name, job_id, log_dir) as tracker:
                    # Add tracker to kwargs if not already present
                    if 'tracker' not in kwargs:
                        kwargs['tracker'] = tracker
                    return func(*args, **kwargs)
            return wrapper
        return decorator 