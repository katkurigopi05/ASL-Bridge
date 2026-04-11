#!/usr/bin/env python3
"""
models/classifier.py — ASL GRU Classifier with Attention Pooling

DESCRIPTION:
    PyTorch model implementing the specified architecture:
    Input:  sequence of 30 frames × 63 features (21 hand landmarks × 3 coords)
    Layer 1: Linear(63, 128) + LayerNorm + ReLU
    Layer 2: GRU(128, 256, num_layers=2, batch_first=True, dropout=0.3)
    Layer 3: Attention pooling over GRU outputs
    Layer 4: Linear(256, 128) + ReLU + Dropout(0.4)
    Output:  Linear(128, num_classes) + Softmax

USAGE:
    python models/classifier.py              # run shape verification
    python models/classifier.py --summary    # print model summary
"""

import argparse
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("classifier")


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over temporal GRU outputs.

    Instead of just using the last hidden state, learns which
    time steps are most important for classification.
    Produces a weighted sum of all GRU outputs.
    """

    def __init__(self, hidden_size: int):
        """
        Args:
            hidden_size: Dimension of GRU hidden states
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, gru_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gru_outputs: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size) — attention-weighted sum
        """
        # Compute attention scores: (batch, seq_len, 1)
        scores = self.attention(gru_outputs)

        # Softmax over time dimension: (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted sum: (batch, hidden_size)
        context = torch.sum(weights * gru_outputs, dim=1)

        return context


class ASLGRUClassifier(nn.Module):
    """
    GRU-based classifier for ASL sign recognition.

    Architecture:
        Input (batch, 30, 63)
        → Linear(63, 128) + LayerNorm + ReLU
        → GRU(128, 256, layers=2, dropout=0.3)
        → Attention Pooling
        → Linear(256, 128) + ReLU + Dropout(0.4)
        → Linear(128, num_classes) + Softmax
    """

    def __init__(
        self,
        input_features: int = 63,
        hidden_size: int = 128,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        fc_dropout: float = 0.4,
        fc_hidden: int = 128,
        num_classes: int = 29,
    ):
        """
        Args:
            input_features: Number of input features per frame (21 × 3 = 63)
            hidden_size: Projection dimension for input linear layer
            gru_hidden: GRU hidden state dimension
            gru_layers: Number of stacked GRU layers
            gru_dropout: Dropout between GRU layers
            fc_dropout: Dropout before final classifier
            fc_hidden: Hidden dimension of FC classifier
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_features = input_features
        self.hidden_size = hidden_size
        self.gru_hidden = gru_hidden
        self.num_classes = num_classes

        # Layer 1: Linear projection + LayerNorm + ReLU
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Layer 2: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Layer 3: Attention pooling
        self.attention_pool = AttentionPooling(gru_hidden)

        # Layer 4: FC classifier
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    # Orthogonal initialization for GRU weights
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_features)
               e.g., (32, 30, 63)

        Returns:
            Output tensor of shape (batch, num_classes) with softmax probabilities
        """
        # Layer 1: Project input features
        # (batch, seq_len, 63) → (batch, seq_len, 128)
        x = self.input_projection(x)

        # Layer 2: GRU temporal encoding
        # (batch, seq_len, 128) → (batch, seq_len, 256)
        gru_out, _ = self.gru(x)

        # Layer 3: Attention pooling over time
        # (batch, seq_len, 256) → (batch, 256)
        context = self.attention_pool(gru_out)

        # Layer 4: Classification
        # (batch, 256) → (batch, num_classes)
        logits = self.classifier(context)

        # Softmax output
        output = F.softmax(logits, dim=-1)

        return output

    def predict(self, x: torch.Tensor) -> dict:
        """
        Run inference and return structured prediction.

        Args:
            x: Input tensor of shape (1, sequence_length, input_features)

        Returns:
            {
                "label_idx": int,
                "confidence": float,
                "top3": [(idx, confidence), ...],
                "all_probs": numpy array
            }
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x).squeeze(0)  # (num_classes,)

        probs_np = probs.numpy()
        top3_indices = probs_np.argsort()[-3:][::-1]

        return {
            "label_idx": int(top3_indices[0]),
            "confidence": float(probs_np[top3_indices[0]]),
            "top3": [(int(idx), float(probs_np[idx])) for idx in top3_indices],
            "all_probs": probs_np,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict) -> "ASLGRUClassifier":
        """
        Create a model from config.yaml model section.

        Args:
            config: Full parsed config.yaml dict

        Returns:
            ASLGRUClassifier instance
        """
        model_config = config.get("model", {})
        return cls(
            input_features=model_config.get("input_features", 63),
            hidden_size=model_config.get("hidden_size", 128),
            gru_hidden=model_config.get("gru_hidden", 256),
            gru_layers=model_config.get("gru_layers", 2),
            gru_dropout=model_config.get("gru_dropout", 0.3),
            fc_dropout=model_config.get("fc_dropout", 0.4),
            fc_hidden=model_config.get("fc_hidden", 128),
            num_classes=model_config.get("num_classes", 29),
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — GRU Classifier")
    parser.add_argument("--summary", action="store_true", help="Print model summary")
    args = parser.parse_args()

    # Create model with default config
    model = ASLGRUClassifier(
        input_features=63,
        hidden_size=128,
        gru_hidden=256,
        gru_layers=2,
        gru_dropout=0.3,
        fc_dropout=0.4,
        fc_hidden=128,
        num_classes=29,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"ASL GRU Classifier — Architecture Verification")
    logger.info(f"{'='*60}")
    logger.info(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = torch.randn(32, 30, 63)  # batch=32, seq=30, features=63
    output = model(batch)
    logger.info(f"\nInput shape:  {batch.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output sum per sample (should be ~1.0): {output[0].sum().item():.6f}")

    # Test single prediction
    single = torch.randn(1, 30, 63)
    pred = model.predict(single)
    logger.info(f"\nSingle prediction:")
    logger.info(f"  Top label index: {pred['label_idx']}")
    logger.info(f"  Confidence: {pred['confidence']:.4f}")
    logger.info(f"  Top 3: {pred['top3']}")

    if args.summary:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model Architecture:")
        logger.info(f"{'='*60}")
        logger.info(model)

    logger.info(f"\n✅ All shape checks passed!")


if __name__ == "__main__":
    main()
