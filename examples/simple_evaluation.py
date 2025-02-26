import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import sys
import numpy as np

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_workflow.config import ConfigManager
from training_workflow.tracking import EvaluationTracker

def main(config_path):
    # Load configuration
    config = ConfigManager.load(config_path)
    
    # Get environment variables
    experiment_name = os.environ.get("EXPERIMENT_NAME", "default_experiment")
    job_id = os.environ.get("JOB_ID", "default_job")
    output_dir = os.environ.get("OUTPUT_DIR", "output")
    model_path = os.environ.get("MODEL_PATH", "model.pt")
    dataset_path = os.environ.get("DATASET_PATH", "./data")
    baseline_model_path = os.environ.get("BASELINE_MODEL_PATH", None)
    
    # Create tracker
    tracker = EvaluationTracker(
        experiment_name=experiment_name,
        job_id=job_id,
        log_dir=output_dir
    )
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root=dataset_path, 
        train=False,
        download=True, 
        transform=transform
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=config.get("batch_size", 32),
        shuffle=False, 
        num_workers=2
    )
    
    # Define model architecture (same as in training)
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Load model
    model = SimpleNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for i in range(labels.size(0)):
                confusion_matrix[labels[i].item()][predicted[i].item()] += 1
                
            # Calculate per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate metrics
    test_loss = test_loss / len(testloader)
    test_acc = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[f"class_{i}_accuracy"] = class_acc
    
    # Log evaluation results
    tracker.log_evaluation_results({
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "total_samples": total,
        "correct_samples": correct,
        **class_accuracies
    })
    
    # Create confusion matrix visualization
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    tracker.create_confusion_matrix(confusion_matrix, class_names)
    
    # If baseline model is provided, evaluate it for comparison
    if baseline_model_path:
        print(f"Evaluating baseline model: {baseline_model_path}")
        
        # Load baseline model
        baseline_model = SimpleNet().to(device)
        baseline_model.load_state_dict(torch.load(baseline_model_path))
        baseline_model.eval()
        
        # Evaluate baseline model
        baseline_loss = 0.0
        baseline_correct = 0
        baseline_total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = baseline_model(images)
                loss = criterion(outputs, labels)
                
                baseline_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                baseline_total += labels.size(0)
                baseline_correct += (predicted == labels).sum().item()
        
        # Calculate baseline metrics
        baseline_loss = baseline_loss / len(testloader)
        baseline_acc = 100 * baseline_correct / baseline_total
        
        # Set baseline results for comparison
        tracker.set_baseline_results({
            "test_loss": baseline_loss,
            "test_accuracy": baseline_acc,
            "total_samples": baseline_total,
            "correct_samples": baseline_correct
        })
        
        # Compare with baseline
        tracker.compare_with_baseline()
    
    # Create metrics summary visualization
    tracker.create_metrics_summary()
    
    # Complete evaluation and save results
    results = tracker.complete_evaluation()
    
    # Close tracker
    tracker.close()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CIFAR-10 evaluation example")
    parser.add_argument("config_path", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config_path) 