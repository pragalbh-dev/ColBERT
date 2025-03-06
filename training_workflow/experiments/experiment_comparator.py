from typing import Dict, List, Any, Optional, Union
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .experiment_manager import ExperimentManager

class ExperimentComparator:
    """
    Compares jobs across multiple experiments
    """
    def __init__(self):
        """
        Initialize experiment comparator
        """
        self.experiments = {}
    
    def load_experiment(self, experiment_name: str) -> None:
        """
        Load an experiment for comparison
        
        Args:
            experiment_name: Name of the experiment to load
        """
        exp_manager = ExperimentManager()
        exp_manager.load_experiment_config(f"experiments/{experiment_name}/experiment_config.yaml")
        self.experiments[experiment_name] = exp_manager
    
    def compare_jobs(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare jobs across all loaded experiments
        
        Args:
            metrics: List of metrics to compare (default: all metrics)
            
        Returns:
            pd.DataFrame: DataFrame with job metrics for comparison
        """
        all_jobs = []
        
        # Collect jobs from all experiments
        for exp_name, exp_manager in self.experiments.items():
            all_jobs.extend(exp_manager.jobs)
        
        # Create comparison data
        comparison_data = []
        for job in all_jobs:
            job_metrics = job.get_metrics()
            
            # Filter metrics if specified
            if metrics:
                job_metrics = {k: v for k, v in job_metrics.items() if k in metrics}
            
            # Add job ID, experiment name, and config parameters
            job_data = {
                "job_id": job.job_id,
                "experiment_name": job.experiment_name
            }
            
            # Add config parameters
            if hasattr(job, 'config') and job.config:
                if hasattr(job.config, 'to_dict'):
                    config_dict = job.config.to_dict()
                else:
                    config_dict = job.config
                
                for key in config_dict:
                    job_data[f"config_{key}"] = config_dict[key]
            
            # Add metrics
            job_data.update(job_metrics)
            
            comparison_data.append(job_data)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save comparison to file
        output_path = f"experiments/cross_experiment_comparison.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cross-experiment comparison saved to {output_path}")
        
        return df
    
    def visualize_comparison(self, metric: str, output_path: Optional[str] = None) -> None:
        """
        Visualize comparison of a specific metric across experiments
        
        Args:
            metric: Metric to visualize
            output_path: Path to save the visualization (default: experiments/cross_experiment_{metric}.png)
        """
        # Get comparison data
        df = self.compare_jobs(metrics=[metric])
        
        if df.empty or metric not in df.columns:
            print(f"No data available for metric: {metric}")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Group by experiment
        for exp_name in self.experiments.keys():
            exp_data = df[df['experiment_name'] == exp_name]
            if not exp_data.empty:
                plt.scatter(exp_data['job_id'], exp_data[metric], label=exp_name, s=100)
        
        plt.xlabel('Job ID')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} Across Experiments')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Save visualization
        if output_path is None:
            output_path = f"experiments/cross_experiment_{metric}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison visualization saved to {output_path}")
    
    def compare_metrics_across_experiments(self, metric: str, output_path: Optional[str] = None) -> None:
        """
        Compare a specific metric across experiments with box plots
        
        Args:
            metric: Metric to compare
            output_path: Path to save the visualization (default: experiments/metric_comparison_{metric}.png)
        """
        # Get comparison data
        df = self.compare_jobs(metrics=[metric])
        
        if df.empty or metric not in df.columns:
            print(f"No data available for metric: {metric}")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        data = []
        labels = []
        
        for exp_name in self.experiments.keys():
            exp_data = df[df['experiment_name'] == exp_name]
            if not exp_data.empty and metric in exp_data.columns:
                data.append(exp_data[metric].values)
                labels.append(exp_name)
        
        if data:
            plt.boxplot(data, labels=labels)
            plt.xlabel('Experiment')
            plt.ylabel(metric)
            plt.title(f'Comparison of {metric} Across Experiments')
            plt.grid(True)
            
            # Save visualization
            if output_path is None:
                output_path = f"experiments/metric_comparison_{metric}.png"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            
            print(f"Metric comparison visualization saved to {output_path}")
        else:
            print(f"No data available for metric: {metric}")

    def analyze_single_experiment(self, experiment_name: str, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform detailed analysis on a single experiment
        
        Args:
            experiment_name: Name of the experiment to analyze
            metrics: List of metrics to analyze (default: all metrics)
            
        Returns:
            Dict: Analysis results
        """
        # Load the experiment if not already loaded
        if experiment_name not in self.experiments:
            self.load_experiment(experiment_name)
        
        exp_manager = self.experiments[experiment_name]
        
        # Get jobs
        jobs = exp_manager.get_jobs()
        if not jobs:
            print(f"No jobs found in experiment: {experiment_name}")
            return {}
        
        # Collect all metrics if not specified
        if metrics is None:
            all_metrics = set()
            for job in jobs:
                job_metrics = job.get_metrics()
                all_metrics.update(job_metrics.keys())
            metrics = list(all_metrics)
        
        # Create analysis results
        analysis = {
            "experiment_name": experiment_name,
            "num_jobs": len(jobs),
            "metrics": {},
            "job_comparison": {},
            "hyperparameter_impact": {}
        }
        
        # Analyze each metric
        for metric in metrics:
            metric_values = []
            job_values = {}
            
            for job in jobs:
                job_metrics = job.get_metrics()
                if metric in job_metrics:
                    value = job_metrics[metric]
                    metric_values.append(value)
                    job_values[job.job_id] = value
            
            if metric_values:
                # Calculate statistics
                analysis["metrics"][metric] = {
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "mean": sum(metric_values) / len(metric_values),
                    "range": max(metric_values) - min(metric_values),
                    "values": job_values
                }
                
                # Find best job for this metric
                best_job_higher = exp_manager.get_best_job(metric, higher_better=True)
                best_job_lower = exp_manager.get_best_job(metric, higher_better=False)
                
                if best_job_higher:
                    analysis["metrics"][metric]["best_job_highest"] = {
                        "job_id": best_job_higher.job_id,
                        "value": best_job_higher.get_metrics().get(metric)
                    }
                
                if best_job_lower:
                    analysis["metrics"][metric]["best_job_lowest"] = {
                        "job_id": best_job_lower.job_id,
                        "value": best_job_lower.get_metrics().get(metric)
                    }
        
        # Analyze hyperparameter impact
        hyperparams = set()
        for job in jobs:
            if hasattr(job, 'config') and job.config:
                if hasattr(job.config, 'to_dict'):
                    config_dict = job.config.to_dict()
                else:
                    config_dict = job.config
                
                hyperparams.update(config_dict.keys())
        
        # For each hyperparam, analyze its impact on each metric
        for param in hyperparams:
            analysis["hyperparameter_impact"][param] = {}
            
            for metric in metrics:
                param_values = {}
                
                for job in jobs:
                    if hasattr(job, 'config') and job.config:
                        if hasattr(job.config, 'to_dict'):
                            config_dict = job.config.to_dict()
                        else:
                            config_dict = job.config
                        
                        if param in config_dict:
                            param_value = config_dict[param]
                            job_metrics = job.get_metrics()
                            
                            if metric in job_metrics:
                                if param_value not in param_values:
                                    param_values[param_value] = []
                                
                                param_values[param_value].append(job_metrics[metric])
                
                # Calculate average metric value for each param value
                param_impact = {}
                for param_value, values in param_values.items():
                    if values:
                        param_impact[param_value] = sum(values) / len(values)
                
                if param_impact:
                    analysis["hyperparameter_impact"][param][metric] = param_impact
        
        # Generate visualizations
        output_dir = Path(f"experiments/{experiment_name}/analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save analysis results
        analysis_path = output_dir / "detailed_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Detailed analysis saved to {analysis_path}")
        
        # Create visualizations for hyperparameter impact
        self._visualize_hyperparameter_impact(experiment_name, analysis, output_dir)
        
        return analysis

    def _visualize_hyperparameter_impact(self, experiment_name: str, analysis: Dict[str, Any], output_dir: Path) -> None:
        """
        Visualize the impact of hyperparameters on metrics
        
        Args:
            experiment_name: Name of the experiment
            analysis: Analysis results
            output_dir: Directory to save visualizations
        """
        hyperparams = analysis.get("hyperparameter_impact", {})
        
        for param, metrics_impact in hyperparams.items():
            for metric, param_impact in metrics_impact.items():
                if param_impact:
                    # Sort param values
                    param_values = sorted(param_impact.keys())
                    metric_values = [param_impact[val] for val in param_values]
                    
                    # Create visualization
                    plt.figure(figsize=(10, 6))
                    plt.plot(param_values, metric_values, 'o-', linewidth=2, markersize=8)
                    plt.xlabel(param)
                    plt.ylabel(metric)
                    plt.title(f'Impact of {param} on {metric}')
                    plt.grid(True)
                    
                    # Save visualization
                    viz_path = output_dir / f"impact_{param}_on_{metric}.png"
                    plt.savefig(viz_path)
                    plt.close()
                    
                    print(f"Hyperparameter impact visualization saved to {viz_path}")

    def visualize_job_performance(self, experiment_name: str, job_id: str, metrics: Optional[List[str]] = None) -> None:
        """
        Visualize the performance of a specific job
        
        Args:
            experiment_name: Name of the experiment
            job_id: ID of the job to visualize
            metrics: List of metrics to visualize (default: all metrics)
        """
        # Load the experiment if not already loaded
        if experiment_name not in self.experiments:
            self.load_experiment(experiment_name)
        
        exp_manager = self.experiments[experiment_name]
        
        # Get the job
        job = exp_manager.get_job(job_id)
        if not job:
            print(f"Job not found: {job_id}")
            return
        
        # Get metrics to visualize
        job_metrics = job.get_metrics()
        if metrics is None:
            metrics = list(job_metrics.keys())
        else:
            metrics = [m for m in metrics if m in job_metrics]
        
        if not metrics:
            print(f"No metrics found for job: {job_id}")
            return
        
        # Create output directory
        output_dir = Path(f"experiments/{experiment_name}/job_analysis/{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize each metric's history
        for metric in metrics:
            history = job.get_metric_history(metric)
            if history and len(history) > 0:
                steps, values = zip(*history)
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, values, linewidth=2)
                plt.xlabel('Step')
                plt.ylabel(metric)
                plt.title(f'{metric} History for Job {job_id}')
                plt.grid(True)
                
                # Save visualization
                viz_path = output_dir / f"{metric}_history.png"
                plt.savefig(viz_path)
                plt.close()
                
                print(f"Metric history visualization saved to {viz_path}") 