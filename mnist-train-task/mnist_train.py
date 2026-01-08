import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from lab import lab
import matplotlib.pyplot as plt

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, device, train_loader, optimizer, epoch, training_config):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % training_config["_config"]["log_interval"] == 0:
            lab.log(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
            lab.update_progress((epoch - 1) * len(train_loader) + batch_idx)

# Test function
def test(model, device, test_loader, visualize=False):
    model.eval()
    test_loss = 0
    correct = 0
    images, predictions = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect images and predictions for visualization
            if visualize and len(images) < 10:
                images.extend(data.cpu().numpy())
                predictions.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    lab.log(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

    # Visualize predictions if requested
    if visualize:
        visualize_predictions(images[:10], predictions[:10])

    return accuracy

# Visualization function
def visualize_predictions(images, predictions):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Pred: {predictions[i][0]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("predictions.png")
    lab.log("Saved prediction visualization as predictions.png")
    lab.save_artifact("predictions.png", "predictions.png")

def main():
    try:
        # Initialize lab and set configuration
        # Training configuration
        training_config = {
            "model_name": "mnist_cnn",
            "dataset": "MNIST",
            "task": "classification",
            "output_dir": "./mnist_output",
            "_config": {
                "epochs": 1,
                "batch_size": 64,
                "test_batch_size": 1000,
                "learning_rate": 0.01,
                "momentum": 0.5,
                "device": "cpu",
                "seed": 42,
                "log_interval": 10,
            },
        }
        lab.init()
        lab.set_config(training_config)

        # Set random seed for reproducibility
        torch.manual_seed(training_config["_config"]["seed"])
        device = torch.device(training_config["_config"]["device"])

        # Load MNIST dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=training_config["_config"]["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=training_config["_config"]["test_batch_size"],
            shuffle=False,
        )

        # Initialize model, optimizer, and loss function
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=training_config["_config"]["learning_rate"], momentum=training_config["_config"]["momentum"])

        # Evaluate before training
        lab.log("Evaluating before training...")
        test(model, device, test_loader, visualize=True)

        # Training loop
        for epoch in range(1, training_config["_config"]["epochs"] + 1):
            train(model, device, train_loader, optimizer, epoch, training_config=training_config)
            test(model, device, test_loader)

        # Ensure the directory exists before saving the model
        model_dir = os.path.join(os.path.expanduser("~"), ".transformerlab", "workspace", "models")
        os.makedirs(model_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(model_dir, f"{training_config['model_name']}.pt")
        torch.save(model.state_dict(), model_path)
        lab.save_model(model_path, training_config["model_name"])
        lab.log(f"Model saved to {model_path}")

        # Visualize predictions after training
        lab.log("Visualizing predictions after training...")
        test(model, device, test_loader, visualize=True)

        lab.finish("Training completed successfully")

    except Exception as e:
        lab.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
