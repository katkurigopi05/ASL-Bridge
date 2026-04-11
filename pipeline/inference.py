#!/usr/bin/env python3
"""
pipeline/inference.py — Real-Time ASL Inference Engine

DESCRIPTION:
    Runs the full inference pipeline: webcam → MediaPipe → preprocessor →
    model → stabilizer → output. Maintains a sliding window buffer
    (default: 30 frames) and applies prediction stabilization.

FEATURES:
    - Sliding window buffer for temporal context
    - Prediction stabilizer: sign confirmed only when same label appears
      in 8/10 consecutive frames
    - Top-3 prediction output with confidence scores
    - Target latency: <50ms per frame
    - Thread-safe for integration with FastAPI server

USAGE:
    python pipeline/inference.py --webcam 0
    python pipeline/inference.py --video path/to/video.mp4

INPUTS:
    --webcam        Webcam device index
    --video         Video file path
    --checkpoint    Path to trained model .pth file
    --config        Path to config.yaml

OUTPUTS:
    Prints real-time predictions: { label, confidence, top3 }
"""

import argparse
import collections
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger("inference")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class SlidingWindowBuffer:
    """
    Maintains a fixed-size sliding window of keypoint frames
    for temporal context in sequence classification.
    """

    def __init__(self, window_size: int = 30, feature_dim: int = 63):
        """
        Args:
            window_size: Number of frames in the buffer
            feature_dim: Number of features per frame
        """
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.buffer = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add_frame(self, keypoints: Optional[np.ndarray]) -> None:
        """
        Add a frame to the buffer.
        If keypoints is None (no detection), adds zeros.
        """
        with self._lock:
            if keypoints is not None:
                self.buffer.append(keypoints.copy())
            else:
                self.buffer.append(np.zeros(self.feature_dim, dtype=np.float32))

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current buffer contents as a numpy array.

        Returns:
            Array of shape (window_size, feature_dim) or None if buffer not full
        """
        with self._lock:
            if len(self.buffer) < self.window_size:
                return None
            return np.stack(list(self.buffer))

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.buffer) >= self.window_size

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()

    @property
    def fill_level(self) -> float:
        """Return buffer fill level as a fraction (0.0 to 1.0)."""
        return len(self.buffer) / self.window_size


class PredictionStabilizer:
    """
    Stabilizes noisy per-frame predictions by requiring a label
    to appear in threshold/window_size consecutive frames before
    it is confirmed as the final output.

    Default: sign confirmed when same label appears in 8/10 consecutive frames.
    """

    def __init__(self, window_size: int = 10, threshold: int = 8):
        """
        Args:
            window_size: Number of recent predictions to consider
            threshold: Minimum count of the same label to confirm
        """
        self.window_size = window_size
        self.threshold = threshold
        self.history = collections.deque(maxlen=window_size)
        self.last_confirmed = None
        self._lock = threading.Lock()

    def update(self, label: str, confidence: float) -> Optional[str]:
        """
        Add a new prediction and return the stabilized label.

        Args:
            label: Predicted class label
            confidence: Confidence score (0-1)

        Returns:
            Confirmed label string, or None if not yet stable
        """
        with self._lock:
            self.history.append((label, confidence))

            if len(self.history) < self.window_size:
                return None

            # Count occurrences of each label in the window
            label_counts = collections.Counter(l for l, _ in self.history)
            most_common_label, count = label_counts.most_common(1)[0]

            if count >= self.threshold:
                # Check if this is a new confirmed sign (avoid repeating)
                if most_common_label != self.last_confirmed:
                    self.last_confirmed = most_common_label
                    return most_common_label

            return None

    def reset(self) -> None:
        """Reset stabilizer state."""
        with self._lock:
            self.history.clear()
            self.last_confirmed = None


class InferenceEngine:
    """
    Real-time ASL inference engine combining:
    - MediaPipe landmark extraction
    - Keypoint preprocessing
    - Model prediction
    - Prediction stabilization
    """

    def __init__(self, config: dict, checkpoint_path: Optional[str] = None):
        """
        Args:
            config: Parsed config.yaml dictionary
            checkpoint_path: Path to trained model .pth file (optional)
        """
        self.config = config
        self.inference_config = config.get("inference", {})

        # Initialize components
        window_size = self.inference_config.get("sliding_window", 30)
        feature_dim = config.get("model", {}).get("input_features", 63)

        self.buffer = SlidingWindowBuffer(window_size=window_size, feature_dim=feature_dim)

        stab_config = self.inference_config.get("stabilizer", {})
        self.stabilizer = PredictionStabilizer(
            window_size=stab_config.get("window_size", 10),
            threshold=stab_config.get("threshold", 8),
        )

        self.confidence_threshold = self.inference_config.get("confidence_threshold", 0.6)

        # Initialize extractor
        from pipeline.extractor import MediaPipeExtractor
        self.extractor = MediaPipeExtractor(config)

        # Initialize preprocessor
        from pipeline.preprocessor import KeypointPreprocessor
        self.preprocessor = KeypointPreprocessor(config, augment=False)

        # Load model
        self.model = None
        self.vocab = None
        if checkpoint_path:
            self._load_model(checkpoint_path)
        else:
            logger.warning("No checkpoint provided — inference will return dummy predictions")

        # Performance tracking
        self._frame_times = collections.deque(maxlen=100)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load a trained PyTorch model from checkpoint."""
        try:
            import json

            import torch

            from models.classifier import ASLGRUClassifier

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            # Load vocabulary
            vocab_path = Path(checkpoint_path).parent.parent / "data" / "processed" / "asl_alphabet" / "vocab.json"
            if not vocab_path.exists():
                # Try relative to checkpoint
                vocab_path = PROJECT_ROOT / "data" / "processed" / "asl_alphabet" / "vocab.json"

            if vocab_path.exists():
                with open(vocab_path, "r") as f:
                    self.vocab = json.load(f)
                num_classes = self.vocab.get("num_classes", 29)
            else:
                num_classes = checkpoint.get("num_classes", 29)
                logger.warning(f"Vocab file not found, using num_classes={num_classes}")

            # Create model
            model_config = self.config.get("model", {})
            self.model = ASLGRUClassifier(
                input_features=model_config.get("input_features", 63),
                hidden_size=model_config.get("hidden_size", 128),
                gru_hidden=model_config.get("gru_hidden", 256),
                gru_layers=model_config.get("gru_layers", 2),
                gru_dropout=model_config.get("gru_dropout", 0.3),
                fc_dropout=model_config.get("fc_dropout", 0.4),
                fc_hidden=model_config.get("fc_hidden", 128),
                num_classes=num_classes,
            )

            # Load weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            logger.info(f"✅ Model loaded from {checkpoint_path}")
            logger.info(f"   Classes: {num_classes}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None

    def predict(self, sequence: np.ndarray) -> dict:
        """
        Run model inference on a preprocessed sequence.

        Args:
            sequence: numpy array of shape (sequence_length, input_features)

        Returns:
            {
                "label": str,           # predicted class
                "confidence": float,    # softmax probability
                "top3": [               # top 3 predictions
                    {"label": str, "confidence": float},
                    ...
                ]
            }
        """
        if self.model is None:
            return {
                "label": "NO_MODEL",
                "confidence": 0.0,
                "top3": [],
            }

        try:
            import torch

            # Preprocess the sequence
            processed = self.preprocessor.process(sequence)

            # Convert to tensor: (1, seq_len, features)
            tensor = torch.FloatTensor(processed).unsqueeze(0)

            # Inference
            with torch.no_grad():
                output = self.model(tensor)  # (1, num_classes)
                probabilities = output.squeeze(0).numpy()

            # Get top 3
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3 = []
            for idx in top3_indices:
                label = self._id_to_label(int(idx))
                top3.append({
                    "label": label,
                    "confidence": float(probabilities[idx]),
                })

            best_idx = top3_indices[0]
            return {
                "label": self._id_to_label(int(best_idx)),
                "confidence": float(probabilities[best_idx]),
                "top3": top3,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "top3": [],
            }

    def _id_to_label(self, idx: int) -> str:
        """Convert class index to label string."""
        if self.vocab and "id_to_label" in self.vocab:
            return self.vocab["id_to_label"].get(str(idx), f"class_{idx}")
        return f"class_{idx}"

    def process_frame(self, keypoints: Optional[np.ndarray]) -> dict:
        """
        Process a single frame through the full pipeline.

        Args:
            keypoints: Extracted hand landmarks (63,) or None

        Returns:
            Full prediction result dict, potentially with stabilized output
        """
        start_time = time.time()

        # Add to sliding window
        self.buffer.add_frame(keypoints)

        result = {
            "label": None,
            "confidence": 0.0,
            "top3": [],
            "stabilized_label": None,
            "buffer_fill": self.buffer.fill_level,
            "latency_ms": 0.0,
        }

        # Only predict when buffer is full
        if self.buffer.is_ready():
            sequence = self.buffer.get_sequence()
            if sequence is not None:
                prediction = self.predict(sequence)
                result.update(prediction)

                # Stabilize
                if prediction["confidence"] >= self.confidence_threshold:
                    confirmed = self.stabilizer.update(
                        prediction["label"],
                        prediction["confidence"],
                    )
                    result["stabilized_label"] = confirmed

        elapsed_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(elapsed_ms, 2)
        self._frame_times.append(elapsed_ms)

        return result

    def run_webcam(self, cam_index: int = 0) -> None:
        """
        Run the full inference pipeline on live webcam feed.
        Prints predictions in real-time.
        """
        import cv2

        logger.info(f"🎥 Starting real-time inference (webcam {cam_index})")
        logger.info(f"   Buffer size: {self.buffer.window_size} frames")
        logger.info(f"   Stabilizer: {self.stabilizer.threshold}/{self.stabilizer.window_size}")
        logger.info(f"   Confidence threshold: {self.confidence_threshold}")
        logger.info("   Press 'q' to quit\n")

        sentence = []

        for frame_idx, keypoints in self.extractor.extract_from_webcam(cam_index, show_preview=True):
            result = self.process_frame(keypoints)

            # Print prediction
            if result["label"] and result["label"] not in ("NO_MODEL", "ERROR"):
                label = result["label"]
                conf = result["confidence"]
                latency = result["latency_ms"]

                status = f"Frame {frame_idx:5d} | {label:>10s} ({conf:.2f}) | {latency:.1f}ms"

                if result["stabilized_label"]:
                    confirmed = result["stabilized_label"]
                    sentence.append(confirmed)
                    status += f" | ✅ CONFIRMED: {confirmed}"
                    logger.info(status)
                    logger.info(f"   Sentence so far: {' '.join(sentence)}")

        logger.info(f"\n📝 Final sentence: {' '.join(sentence)}")

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if not self._frame_times:
            return 0.0
        return sum(self._frame_times) / len(self._frame_times)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ASL Bridge — Real-Time Inference Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", type=int, nargs="?", const=0, help="Webcam index")
    group.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint .pth path")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    engine = InferenceEngine(config, checkpoint_path=args.checkpoint)

    if args.webcam is not None:
        engine.run_webcam(args.webcam)
    elif args.video:
        logger.info(f"Processing video: {args.video}")
        for frame_idx, keypoints in engine.extractor.extract_from_video(args.video):
            result = engine.process_frame(keypoints)
            if result["stabilized_label"]:
                logger.info(f"Frame {frame_idx}: ✅ {result['stabilized_label']}")


if __name__ == "__main__":
    main()
