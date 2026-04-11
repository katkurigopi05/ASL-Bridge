#!/usr/bin/env python3
"""
models/evaluate.py — Model Evaluation & Metrics

DESCRIPTION:
    Evaluates a trained ASL classifier on test data. Generates:
    - Overall accuracy
    - Per-class accuracy
    - Confusion matrix (saved as image)
    - Classification report

USAGE:
    python models/evaluate.py --checkpoint models/checkpoints/best_model.pth --dataset asl_alphabet
    python models/evaluate.py --checkpoint models/checkpoints/best_model.pth --dataset asl_alphabet --plot

INPUTS:
    --checkpoint    Path to trained model .pth file
    --dataset       Dataset name (must have processed test data)
    --plot          Generate and save confusion matrix plot
    --config        Path to config.yaml

OUTPUTS:
    - Console: accuracy, per-class metrics, classification report
    - models/checkpoints/confusion_matrix.png (if --plot)
    - models/checkpoints/eval_results.json — structured metrics
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifier import ASLGRUClassifier
from pipeline.preprocessor import ASLDataset, KeypointPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def evaluate_model(config: dict, args) -> None:
    """Run full evaluation on test set."""
    model_config = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    vocab_path = PROJECT_ROOT / "data" / "processed" / args.dataset / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        num_classes = vocab["num_classes"]
        class_names = vocab.get("classes", [])
    else:
        logger.error(f"Vocabulary not found: {vocab_path}")
        sys.exit(1)

    # Load model
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    logger.info(f"✅ Model loaded from {args.checkpoint}")
    logger.info(f"   Parameters: {model.count_parameters():,}")

    # Load test dataset
    test_dir = PROJECT_ROOT / "data" / "processed" / args.dataset / "train"
    if not test_dir.exists():
        logger.error(f"Test data not found: {test_dir}")
        sys.exit(1)

    preprocessor = KeypointPreprocessor(config, augment=False)
    dataset = ASLDataset(test_dir, preprocessor, vocab)

    logger.info(f"📊 Test set: {len(dataset)} samples, {dataset.num_classes} classes")

    # Run predictions
    all_labels = []
    all_preds = []
    all_confs = []

    for i in range(len(dataset)):
        keypoints, label = dataset[i]
        tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = output.squeeze(0).cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        all_labels.append(label)
        all_preds.append(pred)
        all_confs.append(conf)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"{'='*60}")

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names if class_names else None,
        digits=4,
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Per-class accuracy
    logger.info(f"\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            logger.info(f"  {name:>10s}: {class_acc:.4f} ({mask.sum()} samples)")

    # Save results
    results = {
        "accuracy": float(accuracy),
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "avg_confidence": float(np.mean(all_confs)),
        "checkpoint": str(args.checkpoint),
    }

    results_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n📄 Results saved to {results_path}")

    # Confusion matrix plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(16, 14))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"ASL Classifier Confusion Matrix (Acc: {accuracy:.4f})")
            plt.tight_layout()

            plot_path = Path(args.checkpoint).parent / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"📊 Confusion matrix saved to {plot_path}")

        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")


def main():
    parser = argparse.ArgumentParser(description="ASL Bridge — Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--plot", action="store_true", help="Generate confusion matrix plot")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate_model(config, args)


if __name__ == "__main__":
    main()
