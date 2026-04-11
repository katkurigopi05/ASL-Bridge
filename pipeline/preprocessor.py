#!/usr/bin/env python3
"""
pipeline/preprocessor.py — Keypoint Normalization, Padding & Augmentation

DESCRIPTION:
    Transforms raw keypoint sequences into model-ready tensors.
    Handles variable-length sequences via padding/truncation,
    normalizes coordinates, and applies data augmentation.

FEATURES:
    - Normalize to wrist-relative coordinates
    - Pad/truncate sequences to fixed length (default: 30 frames)
    - Temporal jitter augmentation (±3 frames)
    - Horizontal flip augmentation
    - Gaussian noise augmentation (σ=0.01)
    - PyTorch Dataset and DataLoader integration

USAGE:
    python pipeline/preprocessor.py --input data/processed/asl_alphabet/train/ --stats

INPUTS:
    Raw .npy keypoint files from extractor.py

OUTPUTS:
    Normalized, padded tensors of shape (sequence_length, num_features)
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger("preprocessor")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class KeypointPreprocessor:
    """
    Preprocesses raw keypoint sequences for model input.

    Transformations applied (in order):
    1. Wrist-relative normalization (subtract wrist position)
    2. Scale normalization (unit variance)
    3. Sequence padding / truncation to fixed length
    4. Optional augmentation (temporal jitter, flip, noise)
    """

    def __init__(self, config: dict, augment: bool = False):
        """
        Args:
            config: Parsed config.yaml dictionary
            augment: Whether to apply data augmentation
        """
        self.config = config
        self.augment = augment

        # Model params
        model_config = config.get("model", {})
        self.sequence_length = model_config.get("sequence_length", 30)
        self.input_features = model_config.get("input_features", 63)
        self.num_landmarks = self.input_features // 3  # 21 landmarks

        # Augmentation params
        aug_config = config.get("augmentation", {})
        self.temporal_jitter_enabled = aug_config.get("temporal_jitter", {}).get("enabled", True)
        self.temporal_jitter_max = aug_config.get("temporal_jitter", {}).get("max_shift", 3)
        self.horizontal_flip_enabled = aug_config.get("horizontal_flip", {}).get("enabled", True)
        self.horizontal_flip_prob = aug_config.get("horizontal_flip", {}).get("probability", 0.5)
        self.gaussian_noise_enabled = aug_config.get("gaussian_noise", {}).get("enabled", True)
        self.gaussian_noise_sigma = aug_config.get("gaussian_noise", {}).get("sigma", 0.01)

        # Seed
        self.seed = config.get("training", {}).get("seed", 42)

    def normalize_wrist_relative(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position (landmark 0).
        This makes the representation translation-invariant.

        Args:
            keypoints: array of shape (num_features,) or (seq_len, num_features)

        Returns:
            Wrist-relative normalized keypoints, same shape as input
        """
        if keypoints.ndim == 1:
            # Single frame: (63,)
            reshaped = keypoints.reshape(self.num_landmarks, 3)
            wrist = reshaped[0].copy()  # landmark 0 = wrist
            reshaped = reshaped - wrist  # subtract wrist position
            return reshaped.flatten()
        elif keypoints.ndim == 2:
            # Sequence: (seq_len, 63)
            result = np.zeros_like(keypoints)
            for i in range(keypoints.shape[0]):
                result[i] = self.normalize_wrist_relative(keypoints[i])
            return result
        else:
            logger.warning(f"Unexpected keypoints shape: {keypoints.shape}")
            return keypoints

    def normalize_scale(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Scale normalization: divide by the max absolute value to get [-1, 1] range.

        Args:
            keypoints: array of shape (num_features,) or (seq_len, num_features)

        Returns:
            Scale-normalized keypoints
        """
        max_val = np.abs(keypoints).max()
        if max_val > 1e-8:  # avoid division by zero
            return keypoints / max_val
        return keypoints

    def pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pad or truncate a keypoint sequence to self.sequence_length frames.

        For single-frame inputs (static images), repeats the frame.
        For short sequences, pads with zeros.
        For long sequences, truncates to the last N frames.

        Args:
            sequence: array of shape (seq_len, num_features) or (num_features,)

        Returns:
            array of shape (self.sequence_length, num_features)
        """
        # Handle single frame input
        if sequence.ndim == 1:
            # Repeat the single frame to fill the sequence
            result = np.tile(sequence, (self.sequence_length, 1))
            return result

        seq_len = sequence.shape[0]

        if seq_len == self.sequence_length:
            return sequence
        elif seq_len > self.sequence_length:
            # Take the center portion
            start = (seq_len - self.sequence_length) // 2
            return sequence[start:start + self.sequence_length]
        else:
            # Pad with zeros at the end
            padding = np.zeros((self.sequence_length - seq_len, sequence.shape[1]), dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)

    def augment_temporal_jitter(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal jitter: randomly shift the sequence window by ±max_shift frames.

        Args:
            sequence: array of shape (seq_len, num_features)

        Returns:
            Jittered sequence of the same shape
        """
        if not self.temporal_jitter_enabled or not self.augment:
            return sequence

        shift = random.randint(-self.temporal_jitter_max, self.temporal_jitter_max)
        if shift == 0:
            return sequence

        result = np.zeros_like(sequence)
        if shift > 0:
            result[shift:] = sequence[:-shift]
            result[:shift] = sequence[0]  # repeat first frame
        else:
            result[:shift] = sequence[-shift:]
            result[shift:] = sequence[-1]  # repeat last frame

        return result

    def augment_horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Horizontally flip hand landmarks (mirror x-coordinates).
        This simulates left/right hand swapping.

        Args:
            keypoints: array of shape (seq_len, num_features) or (num_features,)

        Returns:
            Flipped keypoints
        """
        if not self.horizontal_flip_enabled or not self.augment:
            return keypoints
        if random.random() > self.horizontal_flip_prob:
            return keypoints

        result = keypoints.copy()
        if result.ndim == 1:
            # Flip x coordinates (every 3rd value starting at index 0)
            result[0::3] = 1.0 - result[0::3]
        elif result.ndim == 2:
            result[:, 0::3] = 1.0 - result[:, 0::3]

        return result

    def augment_gaussian_noise(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to keypoint coordinates.

        Args:
            keypoints: array of any shape

        Returns:
            Noisy keypoints
        """
        if not self.gaussian_noise_enabled or not self.augment:
            return keypoints

        noise = np.random.normal(0, self.gaussian_noise_sigma, keypoints.shape).astype(keypoints.dtype)
        return keypoints + noise

    def process(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Wrist-relative normalization
        2. Scale normalization
        3. Pad/truncate to fixed length
        4. Augmentation (if enabled)

        Args:
            keypoints: Raw keypoint array from extractor

        Returns:
            Processed array of shape (sequence_length, input_features)
        """
        # Step 1: Normalize position
        result = self.normalize_wrist_relative(keypoints)

        # Step 2: Normalize scale
        result = self.normalize_scale(result)

        # Step 3: Pad/truncate
        result = self.pad_or_truncate(result)

        # Step 4: Augmentation
        if self.augment:
            result = self.augment_temporal_jitter(result)
            result = self.augment_horizontal_flip(result)
            result = self.augment_gaussian_noise(result)

        return result.astype(np.float32)

    def process_single_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Process a single frame for real-time inference.
        Does NOT pad to sequence length — returns (1, num_features).

        Args:
            keypoints: Raw keypoint array of shape (num_features,)

        Returns:
            Processed array of shape (num_features,)
        """
        result = self.normalize_wrist_relative(keypoints)
        result = self.normalize_scale(result)
        return result.astype(np.float32)


class ASLDataset:
    """
    PyTorch-compatible dataset for preprocessed ASL keypoints.

    Loads .npy files from data/processed/{dataset}/{split}/{class}/ directories,
    applies preprocessing, and returns (tensor, label) pairs.
    """

    def __init__(
        self,
        data_dir: Path,
        preprocessor: KeypointPreprocessor,
        vocab: Optional[dict] = None,
    ):
        """
        Args:
            data_dir: Path to processed data split (e.g., data/processed/asl_alphabet/train/)
            preprocessor: KeypointPreprocessor instance
            vocab: Vocabulary dict with label_to_id mapping (optional — auto-builds if None)
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.samples = []  # list of (file_path, label_id)

        # Discover all .npy files
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        if vocab is not None:
            self.label_to_id = vocab.get("label_to_id", {})
        else:
            # Auto-build from directory names
            self.label_to_id = {d.name: i for i, d in enumerate(class_dirs)}

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_classes = len(self.label_to_id)

        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.label_to_id:
                logger.warning(f"Class {class_name} not in vocabulary, skipping")
                continue

            label_id = self.label_to_id[class_name]
            npy_files = sorted(class_dir.glob("*.npy"))

            for npy_file in npy_files:
                self.samples.append((npy_file, label_id))

        logger.info(
            f"Dataset loaded: {len(self.samples)} samples, "
            f"{self.num_classes} classes from {self.data_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Returns:
            (processed_keypoints, label_id) where keypoints is shape
            (sequence_length, input_features)
        """
        file_path, label_id = self.samples[idx]
        keypoints = np.load(str(file_path))
        processed = self.preprocessor.process(keypoints)
        return processed, label_id


def main():
    """Standalone CLI to inspect and validate preprocessed data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — Keypoint Preprocessor")
    parser.add_argument("--input", type=str, required=True, help="Path to processed data directory")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    preprocessor = KeypointPreprocessor(config, augment=False)
    dataset = ASLDataset(Path(args.input), preprocessor)

    if args.stats:
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset Statistics")
        logger.info(f"{'='*50}")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Num classes: {dataset.num_classes}")
        logger.info(f"Sequence length: {preprocessor.sequence_length}")
        logger.info(f"Features per frame: {preprocessor.input_features}")

        # Sample a few items
        if len(dataset) > 0:
            sample, label = dataset[0]
            logger.info(f"\nSample shape: {sample.shape}")
            logger.info(f"Sample dtype: {sample.dtype}")
            logger.info(f"Sample label: {label} ({dataset.id_to_label.get(label, '?')})")
            logger.info(f"Value range: [{sample.min():.4f}, {sample.max():.4f}]")


if __name__ == "__main__":
    main()
