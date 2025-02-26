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
