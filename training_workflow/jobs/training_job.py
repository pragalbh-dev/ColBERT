from typing import Dict, List, Any, Optional, Union
import os
import subprocess
import sys
import json
import time
import signal
from pathlib import Path
import importlib.util
import shutil

class TrainingJob:
    def __init__(self, 
                 job_id: str,
                 experiment_name: str,
                 training_script: str,
                 config: Any,  # Config object from config_manager
                 dataset_path: str,
                 output_dir: Optional[str] = None,
                 framework: str = "pytorch",
                 script_args: Optional[List[str]] = None,
                 resource_requirements: Optional[Dict[str, Any]] = None):
        """
        Initialize training job
        
        Args:
            job_id: Unique identifier for the job
            experiment_name: Name of the experiment
            training_script: Path to training script
            config: Configuration for the job
            dataset_path: Path to dataset
            output_dir: Directory for outputs (default: experiments/{experiment_name}/{job_id})
            framework: ML framework used ("pytorch", "tensorflow", "jax", etc.)
            script_args: Additional command-line arguments for the script
            resource_requirements: Dict specifying resource needs (memory, cpu_cores, etc.)
        """
        self._job_id = job_id
        self._experiment_name = experiment_name
        self._training_script = Path(training_script)
        self._config = config
        self._dataset_path = Path(dataset_path)
        self._framework = framework.lower()
        self._script_args = script_args or []
        self._resource_requirements = resource_requirements or {}
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = Path(f"experiments/{experiment_name}/{job_id}")
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration in multiple formats for maximum compatibility
        config_path = self.output_dir / "config.yaml"
        self._config.save(str(config_path))
        
        # Also save as JSON for scripts that prefer it
        config_json_path = self.output_dir / "config.json"
        with open(config_json_path, "w") as f:
            json.dump(self._config.to_dict(), f, indent=2)
        
        # Copy training script to job directory for reproducibility
        script_dir = self.output_dir / "scripts"
        os.makedirs(script_dir, exist_ok=True)
        script_copy_path = script_dir / self._training_script.name
        shutil.copy2(self._training_script, script_copy_path)
        
        # Initialize job metadata
        self._status = "not_started"
        self._start_time = None
        self._end_time = None
        self._process = None
        self._save_metadata()
    
    def run(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the training job
        
        Args:
            gpu_id: GPU ID to use for training (if applicable)
            
        Returns:
            Dictionary with job results
        """
        if self._status == "running":
            raise RuntimeError(f"Job {self._job_id} is already running")
        
        if self._status == "completed":
            print(f"Job {self._job_id} already completed, returning previous results")
            return self._load_results()
        
        # Update status and timing
        self._status = "running"
        self._start_time = time.time()
        self._save_metadata()
        
        try:
            # Execute training script
            if self._training_script.suffix == '.py':
                results = self._run_as_subprocess(gpu_id)
            else:
                results = self._run_as_module()
            
            # Update status
            self._status = "completed"
            self._end_time = time.time()
            self._save_metadata()
            
            return results
            
        except Exception as e:
            # Update status on failure
            self._status = "failed"
            self._end_time = time.time()
            self._save_metadata()
            
            # Save error information
            error_path = self.output_dir / "error.log"
            with open(error_path, "w") as f:
                f.write(f"Error: {str(e)}\n")
            
            raise
    
    def _run_as_subprocess(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run training script as a subprocess
        
        Args:
            gpu_id: GPU ID to use
            
        Returns:
            Dictionary with job results
        """
        # Prepare environment variables
        env = os.environ.copy()
        env["EXPERIMENT_NAME"] = self._experiment_name
        env["JOB_ID"] = self._job_id
        env["DATASET_PATH"] = str(self._dataset_path)
        env["CONFIG_PATH"] = str(self.output_dir / "config.yaml")
        env["OUTPUT_DIR"] = str(self.output_dir)
        
        # Set GPU environment variable if specified
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Prepare command
        cmd = [sys.executable, str(self._training_script)]
        
        # Add config path as first argument
        cmd.append(str(self.output_dir / "config.yaml"))
        
        # Add additional arguments
        cmd.extend(self._script_args)
        
        # Run the process
        print(f"Starting job {self._job_id} with command: {' '.join(cmd)}")
        
        # Create log files
        stdout_path = self.output_dir / "stdout.log"
        stderr_path = self.output_dir / "stderr.log"
        
        with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True
            )
            
            # Wait for process to complete
            self._process.wait()
        
        # Check return code
        if self._process.returncode != 0:
            with open(stderr_path, "r") as f:
                error_output = f.read()
            raise RuntimeError(f"Training script failed with code {self._process.returncode}: {error_output}")
        
        # Load results
        return self._load_results()
    
    def _run_as_module(self) -> Dict[str, Any]:
        """
        Run training script by importing it as a module
        
        Returns:
            Dictionary with job results
        """
        # Set environment variables
        os.environ["EXPERIMENT_NAME"] = self._experiment_name
        os.environ["JOB_ID"] = self._job_id
        os.environ["DATASET_PATH"] = str(self._dataset_path)
        os.environ["CONFIG_PATH"] = str(self.output_dir / "config.yaml")
        os.environ["OUTPUT_DIR"] = str(self.output_dir)
        
        # Import the module
        module_name = self._training_script.stem
        spec = importlib.util.spec_from_file_location(module_name, self._training_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call the main function if it exists
        if hasattr(module, "main"):
            results = module.main(str(self.output_dir / "config.yaml"))
            
            # Save results if returned
            if results:
                results_path = self.output_dir / "results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                    
            return results or {}
        else:
            raise ValueError(f"Training script {self._training_script} does not have a main function")
    
    def _load_results(self) -> Dict[str, Any]:
        """
        Load job results from disk
        
        Returns:
            Dictionary with job results
        """
        results_path = self.output_dir / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        else:
            # If no results file, create a basic one with metadata
            results = {
                "status": self._status,
                "duration": self._end_time - self._start_time if self._end_time else None
            }
            
            # Try to get metrics
            metrics_path = self.output_dir / "metrics" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                
                # Extract final values for each metric
                for name, values in metrics.items():
                    if values:
                        results[f"final_{name}"] = values[-1]["value"]
            
            return results
    
    def _save_metadata(self) -> None:
        """Save job metadata to disk"""
        metadata = {
            "job_id": self._job_id,
            "experiment_name": self._experiment_name,
            "status": self._status,
            "training_script": str(self._training_script),
            "dataset_path": str(self._dataset_path)
        }
        
        if self._start_time:
            metadata["start_time"] = self._start_time
        
        if self._end_time:
            metadata["end_time"] = self._end_time
            metadata["duration"] = self._end_time - self._start_time
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_status(self) -> str:
        """Get current job status (not_started, running, completed, failed)"""
        return self._status
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get metrics from the job"""
        return self.load_metrics()
    
    def get_artifacts(self) -> Dict[str, str]:
        """Get paths to artifacts produced by the job"""
        artifacts = {}
        
        # Check for model file
        model_path = self.output_dir / "models" / "model.pt"
        if model_path.exists():
            artifacts["model"] = str(model_path)
        
        # Get latest checkpoint
        checkpoints = list(self.output_dir.glob("checkpoints/checkpoint_*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            artifacts["latest_checkpoint"] = str(latest)
        
        # Add other artifacts
        results_path = self.output_dir / "results.json"
        if results_path.exists():
            artifacts["results"] = str(results_path)
        
        return artifacts
    
    def get_config(self) -> Any:
        """Get job configuration"""
        return self._config
    
    @property
    def id(self) -> str:
        """Get job ID"""
        return self._job_id
    
    def launch_tensorboard(self, port: int = 6006) -> subprocess.Popen:
        """
        Launch TensorBoard for this specific job
        
        Args:
            port: Port for TensorBoard server
            
        Returns:
            Subprocess handle for the TensorBoard process
        """
        log_dir = self.output_dir / "logs"
        cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_dir), "--port", str(port)]
        process = subprocess.Popen(cmd)
        print(f"TensorBoard started at http://localhost:{port}")
        return process
    
    def load_metrics(self) -> Dict[str, List]:
        """
        Load metrics that were logged during training
        
        Returns:
            Dictionary of metrics with their values
        """
        metrics_path = self.output_dir / "metrics" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {}
    
    def stop(self) -> bool:
        """
        Stop a running job
        
        Returns:
            True if job was stopped, False if it wasn't running
        """
        if self._status != "running" or self._process is None:
            return False
        
        # Send termination signal
        try:
            self._process.terminate()
            
            # Give it some time to terminate gracefully
            for _ in range(5):
                if self._process.poll() is not None:
                    break
                time.sleep(1)
            
            # Force kill if still running
            if self._process.poll() is None:
                self._process.kill()
            
            # Update status
            self._status = "stopped"
            self._end_time = time.time()
            self._save_metadata()
            
            return True
        except:
            return False 