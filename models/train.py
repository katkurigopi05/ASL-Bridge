#!/usr/bin/env python3
"""
models/train.py — Training Loop with Early Stopping

DESCRIPTION:
    Full training pipeline for the ASL GRU Classifier. Implements:
    - Adam optimizer with CosineAnnealingLR scheduler
    - CrossEntropyLoss
    - Early stopping with configurable patience
    - Gradient clipping
    - Training/validation split
    - Checkpoint saving (best model + periodic)
    - Training metrics logging

USAGE:
    python models/train.py --dataset asl_alphabet
    python models/train.py --dataset asl_alphabet --resume models/checkpoints/best_model.pth
    python models/train.py --dataset asl_alphabet --epochs 100 --lr 0.0005

INPUTS:
    --dataset       Dataset name (must have processed keypoints)
    --resume        Path to checkpoint to resume from
    --epochs        Override number of epochs
    --lr            Override learning rate
    --batch-size    Override batch size
    --config        Path to config.yaml

OUTPUTS:
    - models/checkpoints/best_model.pth      — best validation loss checkpoint
    - models/checkpoints/epoch_{N}.pth       — periodic checkpoints
    - models/checkpoints/training_log.json   — training metrics history
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import ASLGRUClassifier
from pipeline.preprocessor import ASLDataset, KeypointPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class TorchASLDataset(torch.utils.data.Dataset):
    """Wrap ASLDataset for PyTorch DataLoader."""

    def __init__(self, asl_dataset: ASLDataset):
        self.dataset = asl_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        keypoints, label = self.dataset[idx]
        return torch.FloatTensor(keypoints), torch.tensor(label, dtype=torch.long)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> dict:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward — model outputs softmax probabilities
        outputs = model(inputs)

        # CrossEntropyLoss expects logits, so we use log of softmax
        # Or we can compute loss from the raw probabilities
        loss = criterion(torch.log(outputs + 1e-8), targets)

        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(torch.log(outputs + 1e-8), targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


def train(config: dict, args) -> None:
    """Full training procedure."""
    train_config = config.get("training", {})
    model_config = config.get("model", {})

    # Set seed
    seed = train_config.get("seed", 42)
    set_seed(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device}")

    # Hyperparameters
    epochs = args.epochs or train_config.get("epochs", 50)
    lr = args.lr or train_config.get("learning_rate", 0.001)
    batch_size = args.batch_size or train_config.get("batch_size", 32)
    gradient_clip = train_config.get("gradient_clip", 1.0)

    # Load vocabulary
    vocab_path = PROJECT_ROOT / "data" / "processed" / args.dataset / "vocab.json"
    vocab = None
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        num_classes = vocab.get("num_classes", 29)
        logger.info(f"📚 Vocabulary loaded: {num_classes} classes")
    else:
        num_classes = model_config.get("num_classes", 29)
        logger.warning(f"No vocab.json found, using num_classes={num_classes}")

    # Load dataset
    data_dir = PROJECT_ROOT / "data" / "processed" / args.dataset / "train"
    if not data_dir.exists():
        logger.error(f"❌ No processed data found at {data_dir}")
        logger.info("   Run extract_keypoints.py first!")
        sys.exit(1)

    preprocessor_train = KeypointPreprocessor(config, augment=True)
    preprocessor_val = KeypointPreprocessor(config, augment=False)

    full_dataset = ASLDataset(data_dir, preprocessor_train, vocab)

    # Split dataset
    train_ratio = train_config.get("train_split", 0.8)
    val_ratio = train_config.get("val_split", 0.1)

    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        TorchASLDataset(full_dataset),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"📊 Dataset split: train={train_size}, val={val_size}, test={test_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = ASLGRUClassifier(
        input_features=model_config.get("input_features", 63),
        hidden_size=model_config.get("hidden_size", 128),
        gru_hidden=model_config.get("gru_hidden", 256),
        gru_layers=model_config.get("gru_layers", 2),
        gru_dropout=model_config.get("gru_dropout", 0.3),
        fc_dropout=model_config.get("fc_dropout", 0.4),
        fc_hidden=model_config.get("fc_hidden", 128),
        num_classes=num_classes,
    ).to(device)

    logger.info(f"🧠 Model: {model.count_parameters():,} parameters")

    # Loss, optimizer, scheduler
    criterion = nn.NLLLoss()  # NLLLoss because model outputs log-softmax
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sched_config = train_config.get("scheduler_params", {})
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_config.get("T_max", epochs),
        eta_min=sched_config.get("eta_min", 1e-5),
    )

    # Early stopping
    es_config = train_config.get("early_stopping", {})
    early_stopping = None
    if es_config.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=es_config.get("patience", 10),
            min_delta=es_config.get("min_delta", 0.001),
        )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"📂 Resumed from epoch {start_epoch}")

    # Checkpoint directory
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    training_log = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    logger.info(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "lr": current_lr,
            "time_s": epoch_time,
        }
        training_log.append(log_entry)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f} | {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "num_classes": num_classes,
                "config": model_config,
            }
            torch.save(checkpoint, ckpt_dir / "best_model.pth")
            logger.info(f"  💾 Best model saved (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_dir / f"epoch_{epoch}.pth",
            )

        # Early stopping
        if early_stopping and early_stopping(val_metrics["loss"]):
            logger.info(f"\n⏹️  Early stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

    # Save training log
    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training complete!")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    logger.info(f"   Checkpoint: {ckpt_dir / 'best_model.pth'}")
    logger.info(f"   Log: {ckpt_dir / 'training_log.json'}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="ASL Bridge — Model Training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config, args)


if __name__ == "__main__":
    main()
