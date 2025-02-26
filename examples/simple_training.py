import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_workflow.config import ConfigManager
from training_workflow.tracking import TrainingTracker

def main(config_path):
    # Load configuration
    config = ConfigManager.load(config_path)
    
    # Get environment variables
    experiment_name = os.environ.get("EXPERIMENT_NAME", "default_experiment")
    job_id = os.environ.get("JOB_ID", "default_job")
    output_dir = os.environ.get("OUTPUT_DIR", "output")
    
    # Create tracker
    tracker = TrainingTracker(
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
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=config.get("batch_size", 32),
        shuffle=True, 
        num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
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
    
    # Define model
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
    
    # Create model
    model = SimpleNet().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.get("learning_rate", 0.001),
        momentum=config.get("momentum", 0.9)
    )
    
    # Log hyperparameters
    tracker.add_hparams(
        {
            "learning_rate": config.get("learning_rate", 0.001),
            "batch_size": config.get("batch_size", 32),
            "momentum": config.get("momentum", 0.9),
            "epochs": config.get("epochs", 10)
        },
        {}  # Metrics will be added during training
    )
    
    # Training loop
    epochs = config.get("epochs", 10)
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                batch_loss = running_loss / 100
                batch_acc = 100 * correct / total
                
                # Log metrics
                step = epoch * len(trainloader) + i
                tracker.log_metrics({
                    "training_loss": batch_loss,
                    "training_accuracy": batch_acc
                }, step=step)
                
                print(f"[{epoch + 1}, {i + 1}] loss: {batch_loss:.3f}, acc: {batch_acc:.2f}%")
                running_loss = 0.0
                correct = 0
                total = 0
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(testloader)
        test_acc = 100 * correct / total
        
        # Log epoch metrics
        tracker.log_epoch(epoch, {
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        
        print(f"Epoch {epoch + 1} - Test loss: {test_loss:.3f}, Test acc: {test_acc:.2f}%")
        
        # Save checkpoint
        tracker.save_checkpoint(model, optimizer, epoch)
    
    print("Training complete")
    
    # Save final model
    tracker.save_model(model, "final_model.pt", {
        "epochs": epochs,
        "final_accuracy": test_acc
    })
    
    # Close tracker
    tracker.close()
    
    # Return results
    return {
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CIFAR-10 training example")
    parser.add_argument("config_path", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config_path) 