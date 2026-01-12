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
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            lab.log(f"📊 Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
            lab.update_progress((epoch - 1) * len(train_loader) + batch_idx)

# Test function
def test(model, device, test_loader, visualize=False, output_dir=None):
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
    lab.log(f"✅ Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

    # Visualize predictions if requested
    if visualize and output_dir:
        visualize_predictions(images[:10], predictions[:10], output_dir)

    return accuracy

# Visualization function
def visualize_predictions(images, predictions, output_dir):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Pred: {predictions[i][0]}")
        ax.axis('off')
    plt.tight_layout()
    predictions_path = os.path.join(output_dir, "predictions.png")
    plt.savefig(predictions_path)
    lab.log(f"🖼️ Saved prediction visualization as {predictions_path}")
    lab.save_artifact(predictions_path, "predictions.png")

def main():
    try:
        # Initialize lab and get configuration
        lab.init()
        config = lab.get_config()

        # Extract parameters with defaults
        model_name = config.get("model_name", "mnist_cnn")
        device = torch.device(config.get("device", "cpu"))
        seed = config.get("seed", 42)
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 64)
        test_batch_size = config.get("test_batch_size", 1000)
        learning_rate = config.get("learning_rate", 0.01)
        momentum = config.get("momentum", 0.5)
        log_interval = config.get("log_interval", 10)
        output_dir = config.get("output_dir", "./mnist_model")

        # Log configuration details
        lab.log("🚀 Starting MNIST training task...")
        lab.log(f"📋 Model: {model_name}")
        lab.log(f"📊 Device: {device}")
        lab.log(f"🔧 Seed: {seed}")
        lab.log(f"🔢 Epochs: {epochs}")
        lab.log(f"📦 Batch size: {batch_size}")
        lab.log(f"📦 Test batch size: {test_batch_size}")
        lab.log(f"⚙️ Learning rate: {learning_rate}")
        lab.log(f"⚙️ Momentum: {momentum}")
        lab.log(f"📂 Output directory: {output_dir}")

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load MNIST dataset
        lab.log("📥 Loading MNIST dataset...")
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=test_batch_size,
            shuffle=False,
        )
        lab.log("✅ MNIST dataset loaded successfully.")

        # Initialize model, optimizer, and loss function
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Evaluate before training
        lab.log("🔍 Evaluating model before training...")
        test(model, device, test_loader, visualize=True, output_dir=output_dir)

        # Training loop
        for epoch in range(1, epochs + 1):
            lab.log(f"🚀 Starting epoch {epoch}/{epochs}...")
            train(model, device, train_loader, optimizer, epoch, log_interval)
            test(model, device, test_loader, output_dir=output_dir)

        # Save the model
        model_path = os.path.join(output_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        lab.save_model(model_path, model_name)
        lab.log(f"💾 Model saved to {model_path}")

        # Visualize predictions after training
        lab.log("🔍 Visualizing predictions after training...")
        test(model, device, test_loader, visualize=True, output_dir=output_dir)

        lab.finish("🎉 Training completed successfully!")

    except Exception as e:
        lab.error(f"❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()