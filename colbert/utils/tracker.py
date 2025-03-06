import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from training_workflow.tracking.base_tracker import BaseTracker

"""
ColBERTTracker: Specialized tracker for training ColBERT models.

NOTE: This tracker does NOT filter logging based on process rank.
The caller is responsible for only invoking logging methods from 
the appropriate process (typically rank 0). This design allows
for more flexible control over when and where logging occurs.
"""

class ColBERTTracker(BaseTracker):
    """
    Specialized tracker for ColBERT models that extends BaseTracker
    with ColBERT-specific logging functionality.
    
    IMPORTANT: This tracker DOES NOT filter logging based on process rank.
    The caller is responsible for only invoking logging methods from
    the appropriate process (typically rank 0).
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        log_dir: Optional[str] = None,
        enable_tensorboard: bool = True,
        artifact_manager=None,
        rank: int = 0
    ):
        """
        Initialize ColBERT tracker
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run
            log_dir: Directory for logs
            enable_tensorboard: Whether to enable TensorBoard logging
            artifact_manager: Optional artifact manager instance
            rank: Process rank in distributed training (0 is main process)
            
        Note:
            While rank is stored, this tracker does not filter logging
            based on rank. The caller is responsible for only invoking
            logging methods from the appropriate process.
        """
        super().__init__(
            experiment_name=experiment_name,
            job_id=run_name,
            log_dir=log_dir,
            artifact_manager=artifact_manager
        )
        
        self.enable_tensorboard = enable_tensorboard
        self.rank = rank
        self.is_main = (self.rank == 0)
        
        # Create evaluation directory if this is the main process | check if this is really neede
        if self.is_main:
            self.eval_dir = self.log_dir / "evaluation"
            os.makedirs(self.eval_dir, exist_ok=True)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration parameters
        
        Args:
            config: Dictionary of configuration parameters
        """
        if not self.is_main:
            return
            
        # Log to TensorBoard as text
        if self.enable_tensorboard:
            config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.add_text("config", config_str)
        
        # Save to disk
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_evaluation_results(
        self,
        results: Dict[str, Dict[int, float]],
        step: Optional[int] = None,
        prefix: str = "eval"
    ) -> None:
        """
        Log evaluation results
        
        Args:
            results: Dictionary of evaluation metrics
            step: Optional step number
            prefix: Prefix for metric names
        """
        if not self.is_main:
            return
            
        # Log each metric
        for metric_type, depths in results.items():
            for depth, value in depths.items():
                metric_name = f"{prefix}/{metric_type.upper()}@{depth}"
                self.log_metric(metric_name, value, step)
        
        # Save results to disk
        if step is not None:
            results_file = self.eval_dir / f"results_step_{step}.json"
        else:
            results_file = self.eval_dir / "results.json"
            
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def log_metric(
        self,
        name: str,
        value: Union[float, int, np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log a metric value. Does NOT filter based on rank.
        
        The caller is responsible for ensuring this is only called
        from the appropriate process (typically rank 0).
        
        Args:
            name: Name of the metric
            value: Value of the metric
            step: Training step
            commit: Whether to commit the metric to disk
        """
        # Remove the rank check to allow logging from any process
        
        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else value.detach().cpu().numpy()
        
        # Store metric in memory
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append((step, value))
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar(name, value, step)
        
        # Save metrics to disk if requested
        if commit:
            self._save_metrics_to_disk()
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
            commit: Whether to commit the metrics to disk
        """
        if not self.is_main:
            return
        
        for name, value in metrics.items():
            self.log_metric(name, value, step, commit=False)
        
        if commit:
            self._save_metrics_to_disk()
    
    def _save_metrics_to_disk(self) -> None:
        """Save metrics to disk"""
        if not self.is_main:
            return
        
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def log_artifact(
        self,
        artifact_path: str,
        content: str,
        artifact_type: str = "text"
    ) -> None:
        """
        Log an artifact (file)
        
        Args:
            artifact_path: Path to save the artifact
            content: Content of the artifact
            artifact_type: Type of artifact (text, json, etc.)
        """
        if not self.is_main:
            return
        
        # Create full path
        full_path = self.log_dir / "artifacts" / artifact_path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save artifact
        with open(full_path, 'w') as f:
            f.write(content)
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        checkpoint_name: str,
        is_best: bool = False
    ) -> str:
        """
        Save a model checkpoint
        
        Args:
            state_dict: Model state dictionary
            checkpoint_name: Name of the checkpoint
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to the saved checkpoint
        """
        if not self.is_main:
            return ""
        
        # Create checkpoint directory
        checkpoint_dir = self.log_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        torch.save(state_dict, checkpoint_path)
        
        # If this is the best checkpoint, create a symlink
        if is_best:
            best_path = checkpoint_dir / "best.pt"
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(checkpoint_path, best_path)
        
        return str(checkpoint_path)
    
    def add_histogram(
        self,
        name: str,
        values: Union[torch.Tensor, np.ndarray],
        step: Optional[int] = None
    ) -> None:
        """
        Add a histogram to TensorBoard
        
        Args:
            name: Name of the histogram
            values: Values to plot
            step: Training step
        """
        if not self.is_main or not self.writer:
            return
        
        self.writer.add_histogram(name, values, step)
    
    def add_figure(
        self,
        name: str,
        figure,
        step: Optional[int] = None
    ) -> None:
        """
        Add a matplotlib figure to TensorBoard
        
        Args:
            name: Name of the figure
            figure: Matplotlib figure
            step: Training step
        """
        if not self.is_main or not self.writer:
            return
        
        self.writer.add_figure(name, figure, step)
    
    def add_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None
    ) -> None:
        """
        Add text to TensorBoard
        
        Args:
            name: Name of the text
            text: Text to add
            step: Training step
        """
        if not self.is_main or not self.writer:
            return
        
        self.writer.add_text(name, text, step)
    
    def flush(self) -> None:
        """Flush the TensorBoard writer"""
        if self.is_main and self.writer:
            self.writer.flush()
    
    def close(self) -> None:
        """Close the TensorBoard writer"""
        if self.is_main and self.writer:
            self.writer.close()
    
    def __enter__(self) -> 'ColBERTTracker':
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close() 