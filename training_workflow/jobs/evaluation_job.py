from typing import Dict, List, Any, Optional
import os
import subprocess
import json
import time
from pathlib import Path
import importlib.util
import sys

from ..tracking import EvaluationTracker
from ..storage import ArtifactManager

class EvaluationJob:
    def __init__(self, 
                 job_id: str,
                 experiment_name: str,
                 evaluation_script: str,
                 model_path: str,
                 dataset_path: str,
                 config: Any,  # Config object from config_manager
                 output_dir: Optional[str] = None,
                 artifact_manager: Optional[ArtifactManager] = None,
                 script_args: Optional[List[str]] = None,
                 baseline_model_path: Optional[str] = None):
        """
        Initialize evaluation job
        
        Args:
            job_id: Unique identifier for the job
            experiment_name: Name of the experiment
            evaluation_script: Path to evaluation script
            model_path: Path to model to evaluate
            dataset_path: Path to evaluation dataset
            config: Configuration for the job
            output_dir: Directory for outputs (default: experiments/{experiment_name}/eval_{job_id})
            artifact_manager: Optional artifact manager instance
            script_args: Additional command-line arguments for the script
            baseline_model_path: Optional path to baseline model for comparison
        """
        self._job_id = job_id
        self._experiment_name = experiment_name
        self._evaluation_script = Path(evaluation_script)
        self._model_path = Path(model_path)
        self._dataset_path = Path(dataset_path)
        self._config = config
        self._script_args = script_args or []
        self._baseline_model_path = baseline_model_path
        
        # Initialize artifact manager
        if artifact_manager is None:
            self._artifact_manager = ArtifactManager()
        else:
            self._artifact_manager = artifact_manager
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = Path(f"experiments/{experiment_name}/eval_{job_id}")
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = self.output_dir / "config.yaml"
        self._config.save(str(config_path))
        
        # Also save as JSON for scripts that prefer it
        config_json_path = self.output_dir / "config.json"
        with open(config_json_path, "w") as f:
            json.dump(self._config.to_dict(), f, indent=2)
        
        # Initialize job metadata
        self._status = "not_started"
        self._start_time = None
        self._end_time = None
        self._process = None
        self._save_metadata()
        
        # Create tracker for direct evaluation (when not using subprocess)
        self._tracker = None
    
    def run(self, gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the evaluation job
        
        Args:
            gpu_id: GPU ID to use for evaluation (if applicable)
            
        Returns:
            Dictionary with evaluation results
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
        
        # Set up environment variables
        env = os.environ.copy()
        env["EXPERIMENT_NAME"] = self._experiment_name
        env["JOB_ID"] = self._job_id
        env["OUTPUT_DIR"] = str(self.output_dir)
        env["MODEL_PATH"] = str(self._model_path)
        env["DATASET_PATH"] = str(self._dataset_path)
        env["CONFIG_PATH"] = str(self.output_dir / "config.yaml")
        
        if self._baseline_model_path:
            env["BASELINE_MODEL_PATH"] = str(self._baseline_model_path)
        
        # Determine how to run the script
        if self._evaluation_script.suffix == '.py':
            # Run as Python script
            cmd = [sys.executable, str(self._evaluation_script), str(self.output_dir / "config.yaml")]
            cmd.extend(self._script_args)
            
            # Add GPU ID if specified
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Redirect stdout and stderr to files
            stdout_path = self.output_dir / "stdout.log"
            stderr_path = self.output_dir / "stderr.log"
            
            with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
                self._process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file
                )
                
                # Wait for process to complete
                self._process.wait()
        else:
            # For non-Python scripts or direct evaluation
            # Initialize tracker
            self._tracker = EvaluationTracker(
                experiment_name=self._experiment_name,
                job_id=self._job_id,
                log_dir=str(self.output_dir),
                artifact_manager=self._artifact_manager
            )
            
            try:
                # Import the evaluation module
                spec = importlib.util.spec_from_file_location("evaluation_module", self._evaluation_script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Run evaluation function
                if hasattr(module, "evaluate"):
                    results = module.evaluate(
                        model_path=str(self._model_path),
                        dataset_path=str(self._dataset_path),
                        config=self._config,
                        tracker=self._tracker,
                        baseline_model_path=str(self._baseline_model_path) if self._baseline_model_path else None
                    )
                else:
                    # Fall back to main function
                    results = module.main(str(self.output_dir / "config.yaml"))
                
                # Save results
                results_path = self.output_dir / "results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                
                # Update status
                self._status = "completed"
                self._end_time = time.time()
                self._save_metadata()
                
                # Close tracker
                if self._tracker:
                    self._tracker.close()
                
                return results
            except Exception as e:
                # Update status
                self._status = "failed"
                self._end_time = time.time()
                self._save_metadata()
                
                # Log error
                error_path = self.output_dir / "error.log"
                with open(error_path, "w") as f:
                    f.write(f"Error: {str(e)}")
                
                # Close tracker
                if self._tracker:
                    self._tracker.close()
                
                raise RuntimeError(f"Evaluation failed: {str(e)}")
        
        # Update status
        self._status = "completed" if self._process.returncode == 0 else "failed"
        self._end_time = time.time()
        self._save_metadata()
        
        # Check if process failed
        if self._process.returncode != 0:
            self._status = "failed"
            self._save_metadata()
            raise RuntimeError(f"Evaluation script failed with return code {self._process.returncode}. See {self.output_dir}/stderr.log for details.")
        
        # Load results
        return self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """
        Load evaluation results
        
        Returns:
            Dictionary with evaluation results
        """
        results_path = self.output_dir / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                return json.load(f)
        else:
            return {}
    
    def _save_metadata(self) -> None:
        """Save job metadata to disk"""
        metadata = {
            "job_id": self._job_id,
            "experiment_name": self._experiment_name,
            "status": self._status,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "evaluation_script": str(self._evaluation_script),
            "model_path": str(self._model_path),
            "dataset_path": str(self._dataset_path),
            "baseline_model_path": str(self._baseline_model_path) if self._baseline_model_path else None
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_status(self) -> str:
        """Get job status"""
        return self._status
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get metrics from the job"""
        metrics_path = self.output_dir / "metrics" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {}
    
    def get_results(self) -> Dict[str, Any]:
        """Get evaluation results"""
        return self._load_results()
    
    def evaluate_directly(self, 
                         evaluate_func: callable, 
                         **kwargs) -> Dict[str, Any]:
        """
        Run evaluation directly using a provided function
        
        Args:
            evaluate_func: Function to run evaluation
            **kwargs: Additional arguments to pass to the function
            
        Returns:
            Dictionary with evaluation results
        """
        # Initialize tracker if not already initialized
        if self._tracker is None:
            self._tracker = EvaluationTracker(
                experiment_name=self._experiment_name,
                job_id=self._job_id,
                log_dir=str(self.output_dir),
                artifact_manager=self._artifact_manager
            )
        
        # Update status
        self._status = "running"
        self._start_time = time.time()
        self._save_metadata()
        
        try:
            # Run evaluation function with tracker
            results = evaluate_func(
                model_path=str(self._model_path),
                dataset_path=str(self._dataset_path),
                config=self._config,
                tracker=self._tracker,
                baseline_model_path=str(self._baseline_model_path) if self._baseline_model_path else None,
                **kwargs
            )
            
            # Save results
            results_path = self.output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Update status
            self._status = "completed"
            self._end_time = time.time()
            self._save_metadata()
            
            return results
        except Exception as e:
            # Update status
            self._status = "failed"
            self._end_time = time.time()
            self._save_metadata()
            
            # Log error
            error_path = self.output_dir / "error.log"
            with open(error_path, "w") as f:
                f.write(f"Error: {str(e)}")
            
            raise RuntimeError(f"Evaluation failed: {str(e)}")
        finally:
            # Close tracker
            if self._tracker:
                self._tracker.close()
                self._tracker = None
    
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
            
            # Close tracker if it exists
            if self._tracker:
                self._tracker.close()
                self._tracker = None
            
            return True
        except:
            return False 