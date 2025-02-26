import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_workflow.config import ConfigManager
from training_workflow.experiments import ExperimentManager

def main():
    # Create experiment
    experiment = ExperimentManager.create("cifar10_test", "Testing CIFAR-10 classification")
    
    # Create base configuration
    base_config = ConfigManager.create("cifar10_base", {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "momentum": 0.9
    })
    
    # Create first job with base configuration
    job1 = experiment.create_job(
        training_script="examples/simple_training.py",
        config=base_config,
        dataset_path="./data",
        job_id="base_config"
    )
    
    # Run the job
    print("Running job with base configuration...")
    results1 = job1.run()
    print(f"Job completed with results: {results1}")
    
    # Create second job with different learning rate
    lr_config = ConfigManager.create_from(base_config)
    lr_config.update({"learning_rate": 0.01})
    
    job2 = experiment.create_job(
        training_script="examples/simple_training.py",
        config=lr_config,
        dataset_path="./data",
        job_id="higher_lr"
    )
    
    # Run the second job
    print("Running job with higher learning rate...")
    results2 = job2.run()
    print(f"Job completed with results: {results2}")
    
    # Compare jobs
    comparison = experiment.compare_jobs(["base_config", "higher_lr"])
    print("\nJob Comparison:")
    print(comparison)
    
    # Launch TensorBoard
    print("\nLaunching TensorBoard...")
    tb_process = experiment.launch_tensorboard()
    
    print("\nPress Enter to exit...")
    input()
    
    # Terminate TensorBoard
    tb_process.terminate()

    # Evaluate the trained models
    print("\nEvaluating trained models...")
    model1_path = job1.get_artifacts().get("models", {}).get("final_model.pt")
    model2_path = job2.get_artifacts().get("models", {}).get("final_model.pt")

    if model1_path and model2_path:
        # Evaluate first model
        print("Evaluating base configuration model...")
        eval_job1 = experiment.create_evaluation_job(
            evaluation_script="examples/simple_evaluation.py",
            model_path=model1_path,
            dataset_path="./data",
            config=base_config,
            job_id="eval_base_config"
        )
        eval_results1 = eval_job1.run()
        print(f"Evaluation completed with results: {eval_results1}")
        
        # Evaluate second model with first as baseline
        print("Evaluating higher learning rate model...")
        eval_job2 = experiment.create_evaluation_job(
            evaluation_script="examples/simple_evaluation.py",
            model_path=model2_path,
            dataset_path="./data",
            config=lr_config,
            job_id="eval_higher_lr",
            baseline_model_path=model1_path
        )
        eval_results2 = eval_job2.run()
        print(f"Evaluation completed with results: {eval_results2}")
        
        # Compare models
        print("\nComparing models...")
        comparison = experiment.compare_models(
            model_paths=[model1_path, model2_path],
            evaluation_script="examples/simple_evaluation.py",
            dataset_path="./data"
        )
        print("\nModel Comparison:")
        print(comparison)

if __name__ == "__main__":
    main() 