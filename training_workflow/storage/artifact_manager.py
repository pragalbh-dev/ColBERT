from typing import Dict, Any, Optional
import os
import json
import shutil
from pathlib import Path
import torch
## this module is responsible for creating artifact paths for a job in an experiment, registering a dataset as an artifact for that job.
class ArtifactManager:
    def __init__(self, base_dir: str = "artifacts"):
        """
        Initialize artifact manager
        
        Args:
            base_dir: Base directory for artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
    
    def save_model(self, 
                  model: Any, 
                  experiment: str, 
                  job_id: str, 
                  name: str = "model.pt",
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model to artifact store
        
        Args:
            model: Model to save
            experiment: Experiment name
            job_id: Job ID
            name: Model filename
            metadata: Optional metadata to save with the model
            
        Returns:
            Path to saved model
        """
        path = self.get_artifact_path(experiment, job_id, "models", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = Path(path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return path
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """
        Load model from artifact store
        
        Args:
            path: Path to model file
            
        Returns:
            Model state dict
        """
        return torch.load(path)
    
    def save_checkpoint(self, 
                       checkpoint: Dict[str, Any], 
                       experiment: str, 
                       job_id: str, 
                       epoch: int) -> str:
        """
        Save training checkpoint
        
        Args:
            checkpoint: Checkpoint dict (model, optimizer, epoch, etc.)
            experiment: Experiment name
            job_id: Job ID
            epoch: Training epoch
            
        Returns:
            Path to saved checkpoint
        TO Check: Can we really save checkpoint as a file? using torch.save? 
        """
        name = f"checkpoint_epoch_{epoch}.pt"
        path = self.get_artifact_path(experiment, job_id, "checkpoints", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load checkpoint from artifact store
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Checkpoint dict
        """
        return torch.load(path)
    
    def save_artifact(self,
                     artifact_path: str,
                     artifact_type: str,
                     experiment_name: str,
                     job_id: str,
                     name: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a generic artifact
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact
            experiment_name: Name of the experiment
            job_id: Job ID
            name: Optional name for the artifact
            metadata: Optional metadata
            
        Returns:
            Path to the saved artifact
        """
        if name is None:
            name = os.path.basename(artifact_path)
            
        dest_path = self.get_artifact_path(experiment_name, job_id, artifact_type, name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy the artifact
        shutil.copy2(artifact_path, dest_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = Path(dest_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        return dest_path
    
    def get_dataset_path(self, 
                        dataset_name: str, 
                        version: str = "latest") -> str:
        """
        Get path to dataset
        
        Args:
            dataset_name: Dataset name
            version: Dataset version
            
        Returns:
            Path to dataset
        """
        dataset_dir = self.base_dir / "datasets" / dataset_name
        
        if version == "latest":
            # Find the latest version
            versions = [d for d in dataset_dir.glob("v*") if d.is_dir()]
            if not versions:
                raise FileNotFoundError(f"No versions found for dataset {dataset_name}")
            
            # Sort by version number (assuming vX format)
            latest = max(versions, key=lambda x: int(x.name[1:]))
            return str(latest)
        else:
            version_dir = dataset_dir / f"v{version}"
            if not version_dir.exists():
                raise FileNotFoundError(f"Version {version} not found for dataset {dataset_name}")
            return str(version_dir)
    
    def register_dataset(self, 
                        dataset_path: str, 
                        dataset_name: str, 
                        version: str) -> str:
        """
        Register dataset in artifact store
        
        Args:
            dataset_path: Path to dataset
            dataset_name: Dataset name
            version: Dataset version
            
        Returns:
            Path to registered dataset
        """
        # Create dataset directory
        dataset_dir = self.base_dir / "datasets" / dataset_name
        version_dir = dataset_dir / f"v{version}"
        os.makedirs(version_dir, exist_ok=True)
        
        # If dataset_path is a directory, copy its contents
        if os.path.isdir(dataset_path):
            for item in os.listdir(dataset_path):
                s = os.path.join(dataset_path, item)
                d = os.path.join(version_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        else:
            # Copy single file
            dest_path = version_dir / os.path.basename(dataset_path)
            shutil.copy2(dataset_path, dest_path)
        
        # Create metadata file
        metadata = {
            "name": dataset_name,
            "version": version,
            "timestamp": os.path.getmtime(dataset_path),
            "original_path": dataset_path
        }
        
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        return str(version_dir)
    
    def get_artifact_path(self, 
                         experiment: str, 
                         job_id: str, 
                         artifact_type: str, 
                         name: str) -> str:
        """
        Get path for artifact
        
        Args:
            experiment: Experiment name
            job_id: Job ID
            artifact_type: Type of artifact (models, checkpoints, etc.)
            name: Artifact name
            
        Returns:
            Path to artifact
        """
        return str(self.base_dir / experiment / job_id / artifact_type / name) 