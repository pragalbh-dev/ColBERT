from typing import Dict, List, Any, Optional, Union
import os
import json
import subprocess
import uuid
from pathlib import Path
import pandas as pd
from ..config import Config
from ..jobs import TrainingJob, EvaluationJob
from ..storage import ArtifactManager
import tempfile
import sys
class Experiment:
    def __init__(self, name: str, description: Optional[str] = None, base_dir: str = "experiments"):
        """
        Initialize experiment
        
        Args:
            name: Experiment name
            description: Optional experiment description
            base_dir: Base directory for experiments
        """
        self._name = name
        self._description = description
        self._base_dir = Path(base_dir)
        self._path = self._base_dir / name
        
        # Create experiment directory
        os.makedirs(self._path, exist_ok=True)
        
        # Initialize metadata
        self._metadata = {
            "name": name,
            "description": description,
            "created_at": os.path.getctime(self._path) if os.path.exists(self._path) else None,
            "jobs": [],
            "evaluations": []
        }
        
        # Load existing metadata if available
        metadata_path = self._path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
        else:
            # Save initial metadata
            self._save_metadata()
        
        # Initialize artifact manager
        self._artifact_manager = ArtifactManager()
        
        print(f"Experiment '{name}' initialized at {self._path}")
    
    def create_job(self, 
                  training_script: str,
                  config: Config,
                  dataset_path: str,
                  job_id: Optional[str] = None,
                  framework: str = "pytorch",
                  script_args: Optional[List[str]] = None,
                  resource_requirements: Optional[Dict[str, Any]] = None) -> TrainingJob:
        """
        Create a new training job
        
        Args:
            training_script: Path to training script
            config: Job configuration
            dataset_path: Path to dataset
            job_id: Optional job ID (default: auto-generated)
            framework: ML framework used
            script_args: Additional command-line arguments for the script
            resource_requirements: Dict specifying resource needs
            
        Returns:
            TrainingJob instance
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Create job
        job = TrainingJob(
            job_id=job_id,
            experiment_name=self._name,
            training_script=training_script,
            config=config,
            dataset_path=dataset_path,
            output_dir=str(self._path / job_id),
            framework=framework,
            script_args=script_args,
            resource_requirements=resource_requirements
        )
        
        # Add job to metadata
        self._metadata["jobs"].append({
            "id": job_id,
            "status": "not_started",
            "created_at": os.path.getctime(job.output_dir)
        })
        
        # Save updated metadata
        self._save_metadata()
        
        return job
    
    def get_job(self, job_id: str) -> TrainingJob:
        """
        Get job by ID
        
        Args:
            job_id: Job ID
            
        Returns:
            TrainingJob instance
        """
        # Check if job exists
        job_dir = self._path / job_id
        if not job_dir.exists():
            raise ValueError(f"Job {job_id} not found in experiment {self._name}")
        
        # Load job metadata
        metadata_path = job_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Job {job_id} metadata not found")
        
        with open(metadata_path, "r") as f:
            job_metadata = json.load(f)
        
        # Recreate job instance
        return TrainingJob(
            job_id=job_id,
            experiment_name=self._name,
            training_script=job_metadata["training_script"],
            config=None,  # Will be loaded from job directory
            dataset_path=job_metadata["dataset_path"],
            output_dir=str(job_dir)
        )
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs in the experiment
        
        Returns:
            List of job metadata dictionaries
        """
        return self._metadata["jobs"]
    
    def compare_jobs(self, job_ids: List[str]) -> pd.DataFrame:
        """
        Compare metrics across multiple jobs
        
        Args:
            job_ids: List of job IDs to compare
            
        Returns:
            DataFrame with job comparison
        """
        # Collect job data
        job_data = []
        
        for job_id in job_ids:
            try:
                job = self.get_job(job_id)
                metrics = job.load_metrics()
                
                # Extract final values for each metric
                final_metrics = {}
                for name, values in metrics.items():
                    if values:
                        final_metrics[name] = values[-1]["value"]
                
                # Add job metadata
                job_info = {
                    "job_id": job_id,
                    "status": job.get_status(),
                }
                
                # Combine with metrics
                job_info.update(final_metrics)
                job_data.append(job_info)
                
            except Exception as e:
                print(f"Error loading job {job_id}: {e}")
        
        # Create DataFrame
        if job_data:
            return pd.DataFrame(job_data)
        else:
            return pd.DataFrame()
    
    def launch_tensorboard(self, job_ids: Optional[List[str]] = None, port: int = 6006) -> subprocess.Popen:
        """
        Launch TensorBoard for the experiment
        
        Args:
            job_ids: Optional list of job IDs to include (default: all jobs)
            port: Port for TensorBoard server
            
        Returns:
            Subprocess handle for the TensorBoard process
        """
        # Determine which jobs to include
        if job_ids is None:
            # Include all jobs
            log_dirs = [str(self._path / job["id"] / "logs") for job in self._metadata["jobs"]]
        else:
            # Include only specified jobs
            log_dirs = [str(self._path / job_id / "logs") for job_id in job_ids]
        
        # Filter to existing directories
        log_dirs = [d for d in log_dirs if os.path.exists(d)]
        
        if not log_dirs:
            raise ValueError("No log directories found")
        
        # Create comma-separated list of log directories
        logdir_str = ",".join(log_dirs)
        
        # Launch TensorBoard
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir_str, "--port", str(port)]
        process = subprocess.Popen(cmd)
        
        print(f"TensorBoard started at http://localhost:{port}")
        return process
    
    def _save_metadata(self) -> None:
        """Save experiment metadata to disk"""
        metadata_path = self._path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    @property
    def name(self) -> str:
        """Get experiment name"""
        return self._name
    
    @property
    def description(self) -> str:
        """Get experiment description"""
        return self._description
    
    @property
    def path(self) -> Path:
        """Get experiment directory path"""
        return self._path
    
    def create_evaluation_job(self,
                             evaluation_script: str,
                             model_path: str,
                             dataset_path: str,
                             config: Any,
                             job_id: Optional[str] = None,
                             script_args: Optional[List[str]] = None,
                             baseline_model_path: Optional[str] = None) -> 'EvaluationJob':
        """
        Create a new evaluation job
        
        Args:
            evaluation_script: Path to evaluation script
            model_path: Path to model to evaluate
            dataset_path: Path to evaluation dataset
            config: Configuration for the job
            job_id: Optional job ID (default: auto-generated)
            script_args: Additional command-line arguments for the script
            baseline_model_path: Optional path to baseline model for comparison
            
        Returns:
            EvaluationJob instance
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        # Create job
        job = EvaluationJob(
            job_id=job_id,
            experiment_name=self._name,
            evaluation_script=evaluation_script,
            model_path=model_path,
            dataset_path=dataset_path,
            config=config,
            artifact_manager=self._artifact_manager,
            script_args=script_args,
            baseline_model_path=baseline_model_path
        )
        
        # Add job to metadata
        self._metadata["evaluations"].append({
            "id": job_id,
            "status": "not_started",
            "created_at": os.path.getctime(job.output_dir),
            "model_path": model_path
        })
        
        self._save_metadata()
        
        return job
    
    def evaluate_model(self,
                      model_path: str,
                      evaluation_script: str,
                      dataset_path: str,
                      config: Optional[Any] = None,
                      job_id: Optional[str] = None,
                      baseline_model_path: Optional[str] = None,
                      gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a model
        
        Args:
            model_path: Path to model to evaluate
            evaluation_script: Path to evaluation script
            dataset_path: Path to evaluation dataset
            config: Configuration for the job (default: use default config)
            job_id: Optional job ID (default: auto-generated)
            baseline_model_path: Optional path to baseline model for comparison
            gpu_id: GPU ID to use for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        # Use default config if not provided
        if config is None:
            from ..config import ConfigManager
            config = ConfigManager.create_default("evaluation")
        
        # Create evaluation job
        job = self.create_evaluation_job(
            evaluation_script=evaluation_script,
            model_path=model_path,
            dataset_path=dataset_path,
            config=config,
            job_id=job_id,
            baseline_model_path=baseline_model_path
        )
        
        # Run job
        results = job.run(gpu_id=gpu_id)
        
        return results
    
    def compare_models(self,
                      model_paths: List[str],
                      evaluation_script: str,
                      dataset_path: str,
                      config: Optional[Any] = None,
                      gpu_id: Optional[int] = None) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_paths: List of paths to models to compare
            evaluation_script: Path to evaluation script
            dataset_path: Path to evaluation dataset
            config: Configuration for the job (default: use default config)
            gpu_id: GPU ID to use for evaluation
            
        Returns:
            DataFrame with comparison results
        """
        # Use default config if not provided
        if config is None:
            from ..config import ConfigManager
            config = ConfigManager.create_default("evaluation")
        
        # Evaluate each model
        results = []
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            print(f"Evaluating model: {model_name}")
            
            job_id = f"compare_{len(results)}"
            eval_results = self.evaluate_model(
                model_path=model_path,
                evaluation_script=evaluation_script,
                dataset_path=dataset_path,
                config=config,
                job_id=job_id,
                gpu_id=gpu_id
            )
            
            # Add model name to results
            eval_results["model_name"] = model_name
            results.append(eval_results)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Save comparison results
        comparison_path = self._path / "model_comparison.csv"
        df.to_csv(comparison_path, index=False)
        
        return df


class ExperimentManager:
    """
    Manages experiments and their jobs
    """
    def __init__(self):
        """
        Initialize an experiment manager
        """
        self.jobs = []
        self.experiment_name = None
    
    def set_experiment_name(self, name: str) -> None:
        """
        Set the experiment name
        
        Args:
            name: The name of the experiment
        """
        self.experiment_name = name
    
    def add_job(self, job: TrainingJob) -> None:
        """
        Add a job to the experiment
        
        Args:
            job: The job to add
        """
        if job not in self.jobs:
            self.jobs.append(job)
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the experiment
        
        Args:
            job_id: The ID of the job to remove
            
        Returns:
            bool: True if the job was removed, False otherwise
        """
        for i, job in enumerate(self.jobs):
            if job.job_id == job_id:
                self.jobs.pop(i)
                return True
        return False
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get a job by ID
        
        Args:
            job_id: The ID of the job to get
            
        Returns:
            TrainingJob or None: The job with the given ID, or None if not found
        """
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def get_jobs(self) -> List[TrainingJob]:
        """
        Get all jobs
        
        Returns:
            List[TrainingJob]: All jobs in the experiment
        """
        return self.jobs
    
    def run_job(self, job_id: str) -> bool:
        """
        Run a specific job
        
        Args:
            job_id: The ID of the job to run
            
        Returns:
            bool: True if the job was run successfully, False otherwise
        """
        job = self.get_job(job_id)
        if job:
            job.run()
            return True
        return False
    
    def run_all_jobs(self) -> None:
        """
        Run all jobs in the experiment
        """
        for job in self.jobs:
            print(f"Running job: {job.job_id}")
            job.run()
    
    def get_best_job(self, metric: str, higher_better: bool = True) -> Optional[TrainingJob]:
        """
        Get the best job based on a metric
        
        Args:
            metric: The metric to use for comparison
            higher_better: Whether higher values of the metric are better
            
        Returns:
            TrainingJob or None: The best job, or None if no jobs have the metric
        """
        if not self.jobs:
            return None
        
        best_job = None
        best_value = None
        
        for job in self.jobs:
            metrics = job.get_metrics()
            if metric in metrics:
                value = metrics[metric]
                if best_value is None or (higher_better and value > best_value) or (not higher_better and value < best_value):
                    best_value = value
                    best_job = job
        
        return best_job
    
    def launch_tensorboard(self, port: int = 6006) -> subprocess.Popen:
        """
        Launch TensorBoard for all jobs
        
        Args:
            port: The port to use for TensorBoard
            
        Returns:
            subprocess.Popen: The TensorBoard process
        """
        # Create a temporary logdir file with all the job log directories
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            for job in self.jobs:
                log_dir = job.get_log_dir()
                if log_dir:
                    f.write(f"{job.job_id}:{log_dir}\n")
        
        # Launch TensorBoard with the logdir file
        cmd = ["tensorboard", "--logdir_spec", f.name, "--port", str(port)]
        process = subprocess.Popen(cmd)
        
        return process

    @staticmethod
    def create(name: str, description: Optional[str] = None, base_dir: str = "experiments") -> Experiment:
        """
        Create a new experiment
        
        Args:
            name: Experiment name
            description: Optional experiment description
            base_dir: Base directory for experiments
            
        Returns:
            Experiment instance
        """
        return Experiment(name, description, base_dir)
    
    @staticmethod
    def load(name: str, base_dir: str = "experiments") -> Experiment:
        """
        Load an existing experiment
        
        Args:
            name: Experiment name
            base_dir: Base directory for experiments
            
        Returns:
            Experiment instance
        """
        experiment_path = Path(base_dir) / name
        if not experiment_path.exists():
            raise ValueError(f"Experiment {name} not found in {base_dir}")
        
        return Experiment(name, base_dir=base_dir)
    
    @staticmethod
    def list_experiments(base_dir: str = "experiments") -> List[str]:
        """
        List all experiments
        
        Args:
            base_dir: Base directory for experiments
            
        Returns:
            List of experiment names
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        # Get all directories in base_dir
        return [d.name for d in base_path.iterdir() if d.is_dir()]
    
    @staticmethod
    def delete_experiment(name: str, base_dir: str = "experiments") -> bool:
        """
        Delete an experiment
        
        Args:
            name: Experiment name
            base_dir: Base directory for experiments
            
        Returns:
            True if experiment was deleted, False otherwise
        """
        experiment_path = Path(base_dir) / name
        if not experiment_path.exists():
            return False
        
        import shutil
        shutil.rmtree(experiment_path)
        return True

    def save_experiment_config(self, output_path: Optional[str] = None) -> str:
        """
        Save the experiment configuration to a file
        
        Args:
            output_path: Path to save the configuration (default: experiments/{experiment_name}/experiment_config.yaml)
            
        Returns:
            str: Path to the saved configuration file
        """
        from ..config import ConfigManager
        
        # Create experiment config
        experiment_config = {
            "experiment_name": self.experiment_name,
            "jobs": []
        }
        
        # Add job configurations
        for job in self.jobs:
            job_config = {
                "job_id": job.job_id,
                "experiment_name": job.experiment_name,
                "training_script": job.training_script,
                "config": job.config.to_dict() if hasattr(job.config, 'to_dict') else job.config,
                "dataset_path": job.dataset_path,
                "script_args": job.script_args
            }
            experiment_config["jobs"].append(job_config)
        
        # Determine output path
        if output_path is None:
            if self.experiment_name is None:
                raise ValueError("Experiment name must be set to save configuration")
            output_path = f"experiments/{self.experiment_name}/experiment_config.yaml"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save configuration
        ConfigManager.save_config(experiment_config, output_path)
        
        return output_path

    def load_experiment_config(self, config_path: str) -> None:
        """
        Load experiment configuration from a file
        
        Args:
            config_path: Path to the configuration file
        """
        from ..config import ConfigManager, Config
        from ..jobs import TrainingJob
        
        # Load configuration
        experiment_config = ConfigManager.load_config(config_path)
        
        # Set experiment name
        self.experiment_name = experiment_config.get("experiment_name")
        
        # Clear existing jobs
        self.jobs = []
        
        # Add jobs from configuration
        for job_config in experiment_config.get("jobs", []):
            # Create Config object
            config_obj = Config(job_config.get("config", {}), f"{job_config.get('job_id')}_config")
            
            # Create job
            job = TrainingJob(
                job_id=job_config.get("job_id"),
                experiment_name=job_config.get("experiment_name"),
                training_script=job_config.get("training_script"),
                config=config_obj,
                dataset_path=job_config.get("dataset_path"),
                script_args=job_config.get("script_args")
            )
            
            # Add job to experiment
            self.add_job(job)
        
        print(f"Loaded experiment '{self.experiment_name}' with {len(self.jobs)} jobs")

    def export_experiment_results(self, output_path: Optional[str] = None) -> str:
        """
        Export experiment results to a file
        
        Args:
            output_path: Path to save the results (default: experiments/{experiment_name}/experiment_results.json)
            
        Returns:
            str: Path to the saved results file
        """
        # Create results dictionary
        results = {
            "experiment_name": self.experiment_name,
            "jobs": []
        }
        
        # Add job results
        for job in self.jobs:
            job_results = {
                "job_id": job.job_id,
                "metrics": job.get_metrics(),
                "artifacts": job.get_artifacts(),
                "status": job.get_status()
            }
            results["jobs"].append(job_results)
        
        # Determine output path
        if output_path is None:
            if self.experiment_name is None:
                raise ValueError("Experiment name must be set to save results")
            output_path = f"experiments/{self.experiment_name}/experiment_results.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_path

    def compare_jobs(self, job_ids: Optional[List[str]] = None, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple jobs based on metrics
        
        Args:
            job_ids: List of job IDs to compare (default: all jobs)
            metrics: List of metrics to compare (default: all metrics)
            
        Returns:
            pd.DataFrame: DataFrame with job metrics for comparison
        """
        # Get jobs to compare
        if job_ids:
            jobs = [job for job in self.jobs if job.job_id in job_ids]
        else:
            jobs = self.jobs
        
        if not jobs:
            return pd.DataFrame()
        
        # Collect metrics
        comparison_data = []
        for job in jobs:
            job_metrics = job.get_metrics()
            
            # Filter metrics if specified
            if metrics:
                job_metrics = {k: v for k, v in job_metrics.items() if k in metrics}
            
            # Add job ID and config parameters
            job_data = {"job_id": job.job_id}
            
            # Add config parameters if available
            if hasattr(job, 'config') and job.config:
                if hasattr(job.config, 'to_dict'):
                    config_dict = job.config.to_dict()
                else:
                    config_dict = job.config
                
                # Add important config parameters
                for key in config_dict:
                    job_data[f"config_{key}"] = config_dict[key]
            
            # Add metrics
            job_data.update(job_metrics)
            
            comparison_data.append(job_data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save comparison to file
        if self.experiment_name:
            output_path = f"experiments/{self.experiment_name}/job_comparison.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Job comparison saved to {output_path}")
        
        return df

    def get_job_history(self, job_id: str, metric: str) -> pd.DataFrame:
        """
        Get the training history for a job
        
        Args:
            job_id: Job ID
            metric: Metric to get history for
            
        Returns:
            pd.DataFrame: DataFrame with training history
        """
        job = self.get_job(job_id)
        if not job:
            return pd.DataFrame()
        
        # Get metric history
        history = job.get_metric_history(metric)
        if not history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(history, columns=['step', metric])
        
        return df

    def export_metrics(self, output_path: Optional[str] = None) -> str:
        """
        Export metrics to CSV for further analysis
        
        Args:
            output_path: Path to save the metrics (default: experiments/{experiment_name}/metrics.csv)
            
        Returns:
            str: Path to the saved metrics file
        """
        # Collect metrics from all jobs
        metrics_data = []
        for job in self.jobs:
            job_metrics = job.get_metrics()
            
            # Add job ID
            job_metrics['job_id'] = job.job_id
            
            # Add config parameters if available
            if hasattr(job, 'config') and job.config:
                if hasattr(job.config, 'to_dict'):
                    config_dict = job.config.to_dict()
                else:
                    config_dict = job.config
                
                # Add important config parameters
                for key in config_dict:
                    job_metrics[f"config_{key}"] = config_dict[key]
            
            metrics_data.append(job_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Determine output path
        if output_path is None:
            if self.experiment_name is None:
                raise ValueError("Experiment name must be set to save metrics")
            output_path = f"experiments/{self.experiment_name}/metrics.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save metrics
        df.to_csv(output_path, index=False)
        
        return output_path

    def summarize_experiment(self) -> Dict[str, Any]:
        """
        Generate a summary of the experiment
        
        Returns:
            Dict: Summary of the experiment
        """
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "num_jobs": len(self.jobs),
            "job_ids": [job.job_id for job in self.jobs],
            "job_statuses": {job.job_id: job.get_status() for job in self.jobs},
            "metrics": {}
        }
        
        # Collect metrics
        all_metrics = set()
        for job in self.jobs:
            metrics = job.get_metrics()
            all_metrics.update(metrics.keys())
        
        # Calculate statistics for each metric
        for metric in all_metrics:
            values = []
            for job in self.jobs:
                metrics = job.get_metrics()
                if metric in metrics:
                    values.append(metrics[metric])
            
            if values:
                summary["metrics"][metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "values": {job.job_id: job.get_metrics().get(metric) for job in self.jobs if metric in job.get_metrics()}
                }
        
        # Find best job for each metric
        summary["best_jobs"] = {}
        for metric in all_metrics:
            best_job = self.get_best_job(metric, higher_better=True)
            if best_job:
                summary["best_jobs"][f"{metric}_highest"] = {
                    "job_id": best_job.job_id,
                    "value": best_job.get_metrics().get(metric)
                }
            
            best_job = self.get_best_job(metric, higher_better=False)
            if best_job:
                summary["best_jobs"][f"{metric}_lowest"] = {
                    "job_id": best_job.job_id,
                    "value": best_job.get_metrics().get(metric)
                }
        
        # Save summary to file
        if self.experiment_name:
            output_path = f"experiments/{self.experiment_name}/experiment_summary.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Experiment summary saved to {output_path}")
        
        return summary
