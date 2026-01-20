import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from lab import lab
import matplotlib.pyplot as plt
import wandb

# Track whether wandb successfully initialized (or anonymous allowed)
wandb_enabled = False

# New: how often to save prediction visualizations (in global steps). Default 200.
prediction_save_every_steps = 200

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
def train(model, device, train_loader, optimizer, epoch, log_interval, total_epochs=None, visualize=True, output_dir=None):
    model.train()
    total_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Save prediction images at configured intervals if visualization enabled
        if visualize and output_dir is not None:
            # global step index across epochs (0-based)
            global_step = (epoch - 1) * total_batches + batch_idx
            # Save every `prediction_save_every_steps` global steps (default 200).
            if prediction_save_every_steps and prediction_save_every_steps > 0 and (global_step % prediction_save_every_steps == 0):
                try:
                    # prepare a small batch of images/preds (convert to cpu numpy)
                    imgs = data.cpu().numpy()
                    preds = output.argmax(dim=1, keepdim=True).cpu().numpy()
                    # zero-pad step index for filenames (e.g. step_000, step_200)
                    visualize_predictions(imgs[:10], preds[:10], output_dir, stage=f"step_{global_step:03d}")
                except Exception as e:
                    lab.log(f"⚠️ Visualization failed at step {global_step}: {e}")

        if batch_idx % log_interval == 0:
            # Compute progress percentage in range [0, 100]
            if total_epochs and total_batches > 0:
                completed = (epoch - 1) * total_batches + batch_idx
                total = total_epochs * total_batches
                percent = int((completed / total) * 100)
            elif total_batches > 0:
                percent = int((batch_idx / total_batches) * 100)
            else:
                percent = 0
            percent = max(0, min(percent, 100))
            lab.log(f"📊 Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
            lab.update_progress(percent)
            # Log to wandb if available (do not init per-batch). Prefer main() to init.
            try:
                if wandb_enabled or getattr(wandb, "run", None) is not None:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch, "train/batch": batch_idx, "train/progress": percent})
            except Exception as e:
                lab.log(f"⚠️ Wandb log failed during train: {e}")

# Test function
def test(model, device, test_loader, visualize=False, output_dir=None, stage=None):
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
    try:
        if wandb_enabled or getattr(wandb, "run", None) is not None:
            wandb.log({"test/loss": test_loss, "test/accuracy": accuracy})
    except Exception as e:
        lab.log(f"⚠️ Wandb log failed during test: {e}")

    # Visualize predictions if requested
    if visualize and output_dir:
        visualize_predictions(images[:10], predictions[:10], output_dir, stage or "predictions")

    return accuracy

# Visualization function
def visualize_predictions(images, predictions, output_dir, stage="predictions"):
    # handle variable number of images (up to 10)
    n = min(len(images), 10)
    if n == 0:
        lab.log("⚠️ No images to visualize")
        return
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    # normalize axes iterable
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                img = images[idx]
                # handle CHW or HWC; common MNIST shape is (1,H,W)
                if img.ndim == 3 and img.shape[0] == 1:
                    ax.imshow(img[0], cmap='gray')
                elif img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                title = predictions[idx][0] if hasattr(predictions[idx], "__len__") and len(predictions[idx]) > 0 else str(predictions[idx])
                ax.set_title(f"Pred: {title}")
            ax.axis('off')
            idx += 1
    plt.tight_layout()
    # Save with stage-specific filename to avoid overwriting
    predictions_path = os.path.join(output_dir, f"predictions_{stage}.png")
    try:
        plt.savefig(predictions_path)
        plt.close(fig)
        lab.log(f"🖼️ Saved prediction visualization as {predictions_path}")
        lab.save_artifact(predictions_path, os.path.basename(predictions_path))
    except Exception as e:
        lab.log(f"⚠️ Failed to save visualization {predictions_path}: {e}")

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
        # Wandb flag from config (default False)
        log_to_wandb = bool(config.get("log_to_wandb", True))

        # Set prediction save frequency from config (can set 200 or 100)
        global prediction_save_every_steps
        prediction_save_every_steps = int(config.get("prediction_save_every_steps", 200) or 200)
        lab.log(f"Prediction visualizations will be saved every {prediction_save_every_steps} global steps")

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
        lab.log(f"📡 Wandb tracking: {'enabled' if log_to_wandb else 'disabled'}")

        # Initialize wandb if requested and available
        if log_to_wandb:
            try:
                # try to login first (allows interactive/API-key setups). Allow anonymous if no key.
                try:
                    wandb.login(anonymous="allow", key=os.environ.get("WANDB_API_KEY", None))
                    lab.log("🔐 Wandb login succeeded (possibly anonymous)")
                except Exception as le:
                    lab.log(f"⚠️ Wandb login failed: {le}")
                wandb.init(project=os.environ.get("WANDB_PROJECT", "mnist-training-project"), config=config, name=f"{model_name}-{lab.job.id}" if hasattr(lab, "job") else model_name, anonymous="allow")
                lab.log("✅ Wandb initialized")
                global wandb_enabled
                wandb_enabled = getattr(wandb, "run", None) is not None
            except Exception as e:
                lab.log(f"⚠️ Wandb init failed: {e}")
                wandb_enabled = False

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

        # Evaluate before training (skip if step_000 will be produced)
        if not (prediction_save_every_steps and prediction_save_every_steps > 0):
            lab.log("🔍 Evaluating model before training...")
            # call test to save "before" snapshot
            test(model, device, test_loader, visualize=True, output_dir=output_dir, stage="before")
        else:
            lab.log(f"Skipping explicit 'before' snapshot since predictions will be saved at step_000 (every {prediction_save_every_steps} steps)")

        # Training loop
        for epoch in range(1, epochs + 1):
            lab.log(f"🚀 Starting epoch {epoch}/{epochs}...")
            # always save predictions at every training step
            train(model, device, train_loader, optimizer, epoch, log_interval, total_epochs=epochs, visualize=True, output_dir=output_dir)
            test(model, device, test_loader, output_dir=output_dir)

        # Save the model
        model_path = os.path.join(output_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        lab.save_model(model_path, model_name)
        lab.log(f"💾 Model saved to {model_path}")

        # Visualize predictions after training
        lab.log("🔍 Visualizing predictions after training...")
        test(model, device, test_loader, visualize=True, output_dir=output_dir, stage="after")

        # Finish wandb if used
        if log_to_wandb and (wandb_enabled or getattr(wandb, "run", None) is not None):
            try:
                wandb.finish()
            except Exception:
                pass

        lab.finish("🎉 Training completed successfully!")

    except Exception as e:
        lab.error(f"❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()