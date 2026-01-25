import os
import time
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from lab import lab

# JAX/Flax imports
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax

# PyTorch imports
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Track whether wandb successfully initialized (or anonymous allowed)
wandb_enabled = True

# How often (global steps) to save prediction visualizations. Default 200.
prediction_save_every_steps = 200

class FlaxCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

def make_dataloaders(batch_size):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def create_jax_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def cross_entropy_loss(logits, labels):
    onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits, onehot).mean()

@jax.jit
def jax_train_step(state, batch_images, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch_images)
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == batch_labels)
    return state, loss, acc

def visualize_predictions(images, predictions, output_dir, stage="predictions"):
    """
    Save a small grid of prediction images (up to 10).
    images: numpy array (B, C, H, W) or (B, H, W) or (B, H, W, C)
    predictions: iterable of ints or strings
    """
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(images), 10)
    if n == 0:
        return None
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
                # Handle CHW or HWC or single channel
                if img.ndim == 3 and img.shape[0] == 1:
                    ax.imshow(img[0], cmap="gray")
                elif img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                elif img.ndim == 3 and img.shape[-1] == 1:
                    ax.imshow(img[..., 0], cmap="gray")
                elif img.ndim == 3:
                    # assume HWC color
                    ax.imshow(img)
                else:
                    ax.imshow(img)
                ax.set_title(str(predictions[idx]))
            ax.axis("off")
            idx += 1

    plt.tight_layout()
    predictions_path = os.path.join(output_dir, f"predictions_{stage}.png")
    try:
        plt.savefig(predictions_path)
        plt.close(fig)
        lab.log(f"🖼️ Saved prediction visualization as {predictions_path}")
        try:
            lab.save_artifact(predictions_path, os.path.basename(predictions_path))
        except Exception:
            pass
        return predictions_path
    except Exception as e:
        lab.log(f"⚠️ Failed to save visualization {predictions_path}: {e}")
        try:
            plt.close(fig)
        except Exception:
            pass
        return None

def main():
    # initialize lab
    lab.init()
    config = lab.get_config()

    # wandb + prediction frequency
    log_to_wandb = bool(config.get("log_to_wandb", True))
    global prediction_save_every_steps, wandb_enabled
    prediction_save_every_steps = int(config.get("prediction_save_every_steps", prediction_save_every_steps) or prediction_save_every_steps)

    try:
        import wandb
        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY", None))
            lab.log("🔐 Wandb login attempted (anonymous allowed)")
        except Exception:
            lab.log("⚠️ Wandb login attempt failed or anonymous login unavailable")
    except Exception:
        lab.log("⚠️ Wandb not installed; skipping login attempt")

    if log_to_wandb:
        try:
            import wandb
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "mnist-jax-project"),
                config=config,
                name=f"mnist-jax-{getattr(lab, 'job', {}).id if hasattr(lab, 'job') else 'local'}",
                reinit=True,
            )
            wandb_enabled = True
            lab.log("✅ Wandb initialized")
        except Exception as e:
            lab.log(f"⚠️ Wandb init failed: {e}")
            wandb_enabled = False

    framework = config.get("framework", "jax")
    output_dir = config.get("output_dir", "./jax_cnn_output")
    epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 64))
    lr = float(config.get("learning_rate", 1e-3))

    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime.now()
    lab.log(f"Training started at {start_time} using framework={framework}")

    if framework == "jax":
        model = FlaxCNN()
        rng = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
        state = create_jax_train_state(rng, model, lr)

        # load data via torchvision and convert to NHWC numpy
        # use small local loader for compatibility
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        # --- initial prediction save at epoch 0 ---
        try:
            batch0 = next(iter(train_loader))
            images0, _ = batch0
            imgs0 = images0.numpy().astype(np.float32)
            imgs0 = np.transpose(imgs0, (0, 2, 3, 1))
            imgs0 = (imgs0 - 0.1307) / 0.3081
            logits0 = state.apply_fn({"params": state.params}, jnp.array(imgs0))
            preds0 = np.array(jnp.argmax(logits0, -1))
            imgs0_chw = np.transpose(imgs0, (0, 3, 1, 2))
            saved0 = visualize_predictions(imgs0_chw, preds0, output_dir, stage="epoch0_step0")
            if saved0:
                lab.log(f"Saved initial predictions visualization: {saved0}")
        except Exception as e:
            lab.log(f"⚠️ Failed to save initial JAX predictions: {e}")
        # --- end initial save ---

        total_batches = len(train_loader)
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_acc = 0.0
            steps = 0
            for images, labels in train_loader:
                # convert to numpy float32 NHWC
                imgs = images.numpy().astype(np.float32)  # (B, C, H, W)
                imgs = np.transpose(imgs, (0, 2, 3, 1))  # (B, H, W, C)
                imgs = (imgs - 0.1307) / 0.3081  # normalize like MNIST examples
                lbls = labels.numpy().astype(np.int32)

                state, loss, acc = jax_train_step(state, jnp.array(imgs), jnp.array(lbls))
                epoch_loss += float(loss)
                epoch_acc += float(acc)
                steps += 1

                global_step = (epoch - 1) * total_batches + steps

                # Log loss at each step (and step-level wandb)
                try:
                    lab.log(f"[JAX] step {global_step} loss={float(loss):.4f}")
                except Exception:
                    pass
                try:
                    if wandb_enabled:
                        import wandb
                        wandb.log({"train/loss": float(loss), "train/step_acc": float(acc)}, step=global_step)
                except Exception:
                    pass

                # Save prediction visualizations every N global steps
                if prediction_save_every_steps and prediction_save_every_steps > 0 and (global_step % prediction_save_every_steps == 0):
                    try:
                        logits = state.apply_fn({"params": state.params}, jnp.array(imgs))
                        preds = np.array(jnp.argmax(logits, -1))
                        # convert imgs back to CHW for visualize (B,H,W,C)->(B,C,H,W)
                        imgs_chw = np.transpose(imgs, (0, 3, 1, 2))
                        saved = visualize_predictions(imgs_chw, preds, output_dir, stage=f"epoch{epoch}_step{global_step}")
                        if saved:
                            lab.log(f"Saved predictions visualization: {saved}")
                    except Exception as e:
                        lab.log(f"⚠️ Failed to save JAX predictions: {e}")

            avg_loss = epoch_loss / steps if steps else 0.0
            avg_acc = epoch_acc / steps if steps else 0.0

            lab.log(f"[JAX] Epoch {epoch} loss={avg_loss:.4f} acc={avg_acc:.4f}")
            try:
                if wandb_enabled:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/acc": avg_acc, "epoch": epoch}, step=epoch)
            except Exception:
                pass

            lab.update_progress(int(50 * epoch / epochs))

            # save params snapshot
            params_bytes = serialization.to_state_dict(state.params)
            params_path = os.path.join(output_dir, f"flax_params_epoch_{epoch}.npz")
            np.savez(params_path, **jax.tree_util.tree_map(lambda x: np.array(x), params_bytes))
            try:
                saved = lab.save_artifact(params_path, os.path.basename(params_path))
                lab.log(f"Saved artifact: {saved}")
            except Exception as e:
                lab.log(f"Could not save artifact via lab: {e}")

        # finalize
        training_duration = datetime.now() - start_time
        lab.log(f"JAX training completed in {training_duration}")
        lab.finish("Training completed successfully (JAX)")
        return {"status": "success", "framework": "jax", "duration": str(training_duration), "output_dir": output_dir}

    else:
        lab.error(f"Unknown framework: {framework}")
        return {"status": "error", "error": "unknown_framework"}

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))