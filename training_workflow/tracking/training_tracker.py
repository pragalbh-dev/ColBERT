from typing import Dict, List, Any, Callable, Optional
import os
import time
import json
from pathlib import Path
import functools
import torch
import matplotlib.pyplot as plt
from .base_tracker import BaseTracker
import shutil

class TrainingTracker(BaseTracker):
    def __init__(self, 
                 experiment_name: str, 
                 job_id: str, 
                 log_dir: Optional[str] = None,
                 artifact_manager=None,
                 checkpoint_interval: int = 1):
        """
        Initialize training tracker
        
        Args:
            experiment_name: Name of the experiment
            job_id: ID of the training job
            log_dir: Directory for logs (default: experiments/{experiment_name}/{job_id})
            artifact_manager: Optional artifact manager instance
            checkpoint_interval: How often to save checkpoints (in epochs)
        """
        # Initialize base tracker
        super().__init__(experiment_name, job_id, log_dir, artifact_manager)
        
        # Create additional directories specific to training
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.model_dir = self.log_dir / "models"
        self.checkpoint_eval_dir = self.log_dir / "checkpoint_evaluations"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_eval_dir, exist_ok=True)
        
        # Training-specific settings
        self.checkpoint_interval = checkpoint_interval
        self.current_epoch = 0
        self.best_metric = None
        self.best_metric_value = None
        self.best_model_path = None
        
        # Track checkpoint evaluations
        self.checkpoint_evaluations = {}
        
        print(f"Training tracker initialized for experiment '{experiment_name}', job '{job_id}'")
        print(f"Logs will be saved to {self.log_dir}")
    
    def log_epoch(self, 
                 epoch: int, 
                 metrics: Dict[str, float],
                 commit: bool = True) -> None:
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names and values
            commit: Whether to immediately write to disk
        """
        self.current_epoch = epoch
        
        # Use base class method to log metrics
        self.log_metrics(metrics, step=epoch, commit=commit)
        
        # Update best metric if specified
        if self.best_metric is not None and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
            if self.best_metric_value is None or current_value > self.best_metric_value:
                self.best_metric_value = current_value
                print(f"New best {self.best_metric}: {current_value} (epoch {epoch})")
    
    def save_checkpoint(self, 
                       model: Any, 
                       optimizer: Any, 
                       epoch: int, 
                       additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            additional_data: Any additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Add additional data if provided
        if additional_data:
            checkpoint.update(additional_data)
        
        # Use artifact manager if available
        if self.artifact_manager:
            return self.artifact_manager.save_checkpoint(
                checkpoint=checkpoint,
                experiment=self.experiment_name,
                job_id=self.job_id,
                epoch=epoch
            )
        else:
            # Save locally if no artifact manager
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            return str(checkpoint_path)
    
    def save_model(self, 
                  model: Any, 
                  name: str = "model.pt", 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save trained model
        
        Args:
            model: Model to save
            name: Model filename
            metadata: Optional metadata to save with the model
            
        Returns:
            Path to saved model
        """
        # Use artifact manager if available
        if self.artifact_manager:
            return self.artifact_manager.save_model(
                model=model,
                experiment=self.experiment_name,
                job_id=self.job_id,
                name=name,
                metadata=metadata
            )
        else:
            # Save locally if no artifact manager
            model_path = self.model_dir / name
            torch.save(model.state_dict(), model_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            return str(model_path)
    
    def save_best_model(self, 
                       model: Any, 
                       metric_name: str, 
                       metric_value: float) -> Optional[str]:
        """
        Save model if it's the best so far according to the metric
        
        Args:
            model: Model to save
            metric_name: Name of the metric to track
            metric_value: Current value of the metric
            
        Returns:
            Path to saved model if it's the best, None otherwise
        """
        # Initialize best metric if not already set
        if self.best_metric is None:
            self.best_metric = metric_name
            self.best_metric_value = None
        
        # Check if this is the best model so far
        if metric_name == self.best_metric and (self.best_metric_value is None or metric_value > self.best_metric_value):
            self.best_metric_value = metric_value
            
            # Save model with best metric in filename
            name = f"best_model_{metric_name}_{metric_value:.4f}.pt"
            
            metadata = {
                "metric": metric_name,
                "value": metric_value,
                "epoch": self.current_epoch,
                "timestamp": time.time()
            }
            
            self.best_model_path = self.save_model(model, name, metadata)
            print(f"Saved best model with {metric_name}: {metric_value:.4f}")
            return self.best_model_path
        
        return None
    
    def log_batch(self, 
                 batch_idx: int, 
                 metrics: Dict[str, float],
                 commit: bool = False) -> None:
        """
        Log metrics for a batch
        
        Args:
            batch_idx: Batch index
            metrics: Dictionary of metric names and values
            commit: Whether to immediately write to disk
        """
        # Calculate global step (for TensorBoard)
        step = self.current_epoch * 10000 + batch_idx  # Assuming < 10000 batches per epoch
        
        # Use base class method to log metrics
        self.log_metrics(metrics, step=step, commit=commit)
    
    def log_checkpoint_evaluation(self, 
                                epoch: int, 
                                metrics: Dict[str, float]) -> None:
        """
        Log evaluation results for a checkpoint
        
        Args:
            epoch: Epoch number of the checkpoint
            metrics: Dictionary of metric names and values
        """
        # Create checkpoint evaluation directory if it doesn't exist
        os.makedirs(self.checkpoint_eval_dir, exist_ok=True)
        
        # Save evaluation metrics to disk
        eval_path = self.checkpoint_eval_dir / f"eval_epoch_{epoch}.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Store in memory
        self.checkpoint_evaluations[epoch] = metrics
        
        # Log to TensorBoard with special tag
        for name, value in metrics.items():
            self.writer.add_scalar(f"checkpoint_eval/{name}", value, epoch)
        
        # Update best model if this checkpoint is better
        if self.best_metric is not None and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
            is_better = False
            
            if self.best_metric_higher_better:
                is_better = (self.best_metric_value is None or current_value > self.best_metric_value)
            else:
                is_better = (self.best_metric_value is None or current_value < self.best_metric_value)
            
            if is_better:
                self.best_metric_value = current_value
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                if os.path.exists(checkpoint_path):
                    best_model_path = self.model_dir / f"best_model_epoch_{epoch}.pt"
                    shutil.copy(checkpoint_path, best_model_path)
                    self.best_model_path = str(best_model_path)
                    print(f"New best {self.best_metric}: {current_value} (epoch {epoch})")
    
    def evaluate_checkpoint(self, 
                          epoch: int, 
                          eval_function: Callable[[str], Dict[str, float]]) -> Dict[str, float]:
        """
        Evaluate a checkpoint using the provided function
        
        Args:
            epoch: Epoch number of the checkpoint to evaluate
            eval_function: Function that takes checkpoint path and returns metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get checkpoint path
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found")
        
        # Evaluate checkpoint
        metrics = eval_function(str(checkpoint_path))
        
        # Log evaluation results
        self.log_checkpoint_evaluation(epoch, metrics)
        
        return metrics
    
    def get_checkpoint_evaluations(self) -> Dict[int, Dict[str, float]]:
        """
        Get all checkpoint evaluations
        
        Returns:
            Dictionary mapping epoch numbers to evaluation metrics
        """
        return self.checkpoint_evaluations
    
    def plot_metrics(self, 
                    metric_names: List[str], 
                    save_path: Optional[str] = None) -> None:
        """
        Plot training metrics
        
        Args:
            metric_names: List of metric names to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name in metric_names:
            if metric_name not in self.metrics:
                print(f"Warning: Metric '{metric_name}' not found")
                continue
                
            # Extract values and steps
            values = [entry["value"] for entry in self.metrics[metric_name]]
            steps = [entry["step"] for entry in self.metrics[metric_name]]
            
            # Plot metric
            plt.plot(steps, values, label=metric_name)
            
            # Add markers for checkpoints if available
            checkpoint_epochs = list(self.checkpoint_evaluations.keys())
            if checkpoint_epochs and metric_name in self.checkpoint_evaluations[checkpoint_epochs[0]]:
                checkpoint_values = [self.checkpoint_evaluations[e][metric_name] for e in checkpoint_epochs]
                plt.scatter(checkpoint_epochs, checkpoint_values, marker='o', s=100, 
                           edgecolors='black', label=f"{metric_name} (checkpoints)")
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.grid(True)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            # Also add to TensorBoard
            self.add_figure("metrics_plot", plt.gcf())
        else:
            plt.show()
            
        plt.close()
    
    def compare_checkpoints(self, 
                           epochs: List[int], 
                           metrics: List[str], 
                           save_path: Optional[str] = None) -> Dict[str, Dict[int, float]]:
        """
        Compare metrics across multiple checkpoints
        
        Args:
            epochs: List of epoch numbers to compare
            metrics: List of metrics to compare
            save_path: Optional path to save the comparison plot
            
        Returns:
            Dictionary mapping metrics to values for each epoch
        """
        # Validate epochs
        valid_epochs = [e for e in epochs if e in self.checkpoint_evaluations]
        if not valid_epochs:
            print("No valid epochs found for comparison")
            return {}
        
        # Create comparison dictionary
        comparison = {
            metric: {
                epoch: self.checkpoint_evaluations[epoch].get(metric, 0)
                for epoch in valid_epochs
            }
            for metric in metrics if any(metric in self.checkpoint_evaluations[e] for e in valid_epochs)
        }
        
        # Create comparison visualization
        if comparison and valid_epochs:
            self._create_checkpoint_comparison_chart(valid_epochs, list(comparison.keys()), save_path)
        
        return comparison
    
    def _create_checkpoint_comparison_chart(self, 
                                           epochs: List[int], 
                                           metrics: List[str],
                                           save_path: Optional[str] = None) -> None:
        """
        Create a chart comparing metrics across checkpoints
        
        Args:
            epochs: List of epoch numbers to compare
            metrics: List of metrics to compare
            save_path: Optional path to save the plot
        """
        import numpy as np
        
        # Set up the plot
        n_metrics = len(metrics)
        n_epochs = len(epochs)
        
        plt.figure(figsize=(max(12, n_epochs * 1.5), max(8, n_metrics * 0.8)))
        
        # Create a grouped bar chart
        width = 0.8 / n_epochs
        x = np.arange(n_metrics)
        
        for i, epoch in enumerate(epochs):
            values = [self.checkpoint_evaluations[epoch].get(m, 0) for m in metrics]
            plt.bar(x + i * width - 0.4 + width/2, values, width, label=f'Epoch {epoch}')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Checkpoint Comparison')
        plt.xticks(x, metrics, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            # Also add to TensorBoard
            self.add_figure("checkpoint_comparison", plt.gcf())
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """
        Save final results to disk
        
        Args:
            results: Dictionary of results
            
        Returns:
            Path to saved results
        """
        results_path = self.log_dir / "results.json"
        
        # Add timestamp and duration
        results_with_meta = {
            **results,
            "timestamp": time.time(),
            "duration": time.time() - self.start_time
        }
        
        # Save to disk
        with open(results_path, "w") as f:
            json.dump(results_with_meta, f, indent=2)
            
        return str(results_path)
    
    def close(self) -> None:
        """Close tracker and release resources"""
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
            "best_metric": self.best_metric,
            "best_metric_value": self.best_metric_value,
            "best_model_path": self.best_model_path
        }
        
        with open(self.log_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Call parent close method
        super().close()
    
    # Decorator for training functions
    @staticmethod
    def track(experiment_name: str, job_id: str, log_dir: Optional[str] = None):
        """
        Decorator for tracking training functions
        
        Example:
            @TrainingTracker.track("experiment1", "job1")
            def train(model, dataloader, epochs, tracker=None):
                # Training code using tracker
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with TrainingTracker(experiment_name, job_id, log_dir) as tracker:
                    # Add tracker to kwargs if not already present
                    if 'tracker' not in kwargs:
                        kwargs['tracker'] = tracker
                    return func(*args, **kwargs)
            return wrapper
        return decorator 

    def plot_checkpoint_progression(self, metric_name: str) -> None:
        """
        Plot progression of a metric across checkpoints
        
        Args:
            metric_name: Name of the metric to plot
        """
        if not self.checkpoint_evaluations:
            print("No checkpoint evaluations available")
            return
        
        # Extract epochs and metric values
        epochs = []
        values = []
        
        for epoch, metrics in sorted(self.checkpoint_evaluations.items()):
            if metric_name in metrics:
                epochs.append(epoch)
                values.append(metrics[metric_name])
        
        if not epochs:
            print(f"Metric '{metric_name}' not found in checkpoint evaluations")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, 'o-', label=metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Progression of {metric_name} across checkpoints')
        plt.grid(True)
        
        # Save plot
        plot_path = self.checkpoint_eval_dir / f"{metric_name}_progression.png"
        plt.savefig(plot_path)
        
        # Add to TensorBoard
        self.add_figure(f"checkpoint_progression/{metric_name}", plt.gcf())
        
        plt.close() 