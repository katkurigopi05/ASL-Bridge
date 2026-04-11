#!/usr/bin/env python3
"""
tests/test_pipeline.py — Unit Tests for ASL Bridge Pipeline

USAGE:
    pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py

Tests cover:
    - Model architecture shape verification
    - Preprocessor normalization and padding
    - Sliding window buffer behavior
    - Prediction stabilizer logic
    - Config loading
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Add project root
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════════
# Config Tests
# ════════════════════════════════════════════════════════════

class TestConfig:
    """Test config.yaml loading and structure."""

    def test_config_exists(self):
        assert (PROJECT_ROOT / "config.yaml").exists()

    def test_config_loads(self):
        with open(PROJECT_ROOT / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config
        assert "datasets" in config

    def test_config_model_params(self):
        with open(PROJECT_ROOT / "config.yaml") as f:
            config = yaml.safe_load(f)
        model = config["model"]
        assert model["input_features"] == 63
        assert model["sequence_length"] == 30
        assert model["gru_hidden"] == 256
        assert model["gru_layers"] == 2


# ════════════════════════════════════════════════════════════
# Model Tests
# ════════════════════════════════════════════════════════════

class TestClassifier:
    """Test ASL GRU Classifier architecture."""

    @pytest.fixture
    def model(self):
        from models.classifier import ASLGRUClassifier
        return ASLGRUClassifier(
            input_features=63,
            hidden_size=128,
            gru_hidden=256,
            gru_layers=2,
            gru_dropout=0.3,
            fc_dropout=0.4,
            fc_hidden=128,
            num_classes=29,
        )

    def test_model_creation(self, model):
        assert model is not None
        assert model.num_classes == 29

    def test_forward_shape(self, model):
        batch = torch.randn(32, 30, 63)
        output = model(batch)
        assert output.shape == (32, 29)

    def test_single_sample(self, model):
        single = torch.randn(1, 30, 63)
        output = model(single)
        assert output.shape == (1, 29)

    def test_softmax_output(self, model):
        batch = torch.randn(4, 30, 63)
        output = model(batch)
        # Softmax should sum to ~1.0
        sums = output.sum(dim=1)
        for s in sums:
            assert abs(s.item() - 1.0) < 1e-5

    def test_predict(self, model):
        single = torch.randn(1, 30, 63)
        pred = model.predict(single)
        assert "label_idx" in pred
        assert "confidence" in pred
        assert "top3" in pred
        assert len(pred["top3"]) == 3
        assert 0 <= pred["confidence"] <= 1.0

    def test_parameter_count(self, model):
        params = model.count_parameters()
        assert params > 0
        # Should be reasonable size (not tiny, not huge)
        assert 100_000 < params < 5_000_000

    def test_from_config(self):
        from models.classifier import ASLGRUClassifier
        config = {
            "model": {
                "input_features": 63,
                "hidden_size": 128,
                "gru_hidden": 256,
                "gru_layers": 2,
                "gru_dropout": 0.3,
                "fc_dropout": 0.4,
                "fc_hidden": 128,
                "num_classes": 29,
            }
        }
        model = ASLGRUClassifier.from_config(config)
        assert model.num_classes == 29


# ════════════════════════════════════════════════════════════
# Preprocessor Tests
# ════════════════════════════════════════════════════════════

class TestPreprocessor:
    """Test keypoint preprocessing pipeline."""

    @pytest.fixture
    def preprocessor(self):
        from pipeline.preprocessor import KeypointPreprocessor
        config = {
            "model": {"sequence_length": 30, "input_features": 63},
            "augmentation": {
                "temporal_jitter": {"enabled": True, "max_shift": 3},
                "horizontal_flip": {"enabled": True, "probability": 0.5},
                "gaussian_noise": {"enabled": True, "sigma": 0.01},
            },
            "training": {"seed": 42},
        }
        return KeypointPreprocessor(config, augment=False)

    def test_normalize_wrist_single(self, preprocessor):
        keypoints = np.random.randn(63).astype(np.float32)
        result = preprocessor.normalize_wrist_relative(keypoints)
        assert result.shape == (63,)
        # Wrist (first 3 values) should be zero
        assert np.allclose(result[:3], 0.0)

    def test_normalize_wrist_sequence(self, preprocessor):
        sequence = np.random.randn(10, 63).astype(np.float32)
        result = preprocessor.normalize_wrist_relative(sequence)
        assert result.shape == (10, 63)

    def test_pad_short_sequence(self, preprocessor):
        short = np.random.randn(10, 63).astype(np.float32)
        result = preprocessor.pad_or_truncate(short)
        assert result.shape == (30, 63)

    def test_truncate_long_sequence(self, preprocessor):
        long_seq = np.random.randn(50, 63).astype(np.float32)
        result = preprocessor.pad_or_truncate(long_seq)
        assert result.shape == (30, 63)

    def test_pad_single_frame(self, preprocessor):
        single = np.random.randn(63).astype(np.float32)
        result = preprocessor.pad_or_truncate(single)
        assert result.shape == (30, 63)
        # All frames should be the same
        assert np.allclose(result[0], result[15])

    def test_exact_length(self, preprocessor):
        exact = np.random.randn(30, 63).astype(np.float32)
        result = preprocessor.pad_or_truncate(exact)
        assert result.shape == (30, 63)
        assert np.array_equal(result, exact)

    def test_process_full_pipeline(self, preprocessor):
        keypoints = np.random.randn(63).astype(np.float32)
        result = preprocessor.process(keypoints)
        assert result.shape == (30, 63)
        assert result.dtype == np.float32

    def test_augmentation_changes_data(self):
        from pipeline.preprocessor import KeypointPreprocessor
        config = {
            "model": {"sequence_length": 30, "input_features": 63},
            "augmentation": {
                "temporal_jitter": {"enabled": True, "max_shift": 3},
                "horizontal_flip": {"enabled": True, "probability": 1.0},
                "gaussian_noise": {"enabled": True, "sigma": 0.1},
            },
            "training": {"seed": 42},
        }
        aug_preprocessor = KeypointPreprocessor(config, augment=True)
        no_aug_preprocessor = KeypointPreprocessor(config, augment=False)

        keypoints = np.random.randn(30, 63).astype(np.float32)
        aug_result = aug_preprocessor.process(keypoints)
        no_aug_result = no_aug_preprocessor.process(keypoints)

        # Should produce different outputs due to augmentation
        assert not np.array_equal(aug_result, no_aug_result)


# ════════════════════════════════════════════════════════════
# Inference Pipeline Tests
# ════════════════════════════════════════════════════════════

class TestSlidingWindowBuffer:
    """Test sliding window buffer."""

    def test_buffer_creation(self):
        from pipeline.inference import SlidingWindowBuffer
        buf = SlidingWindowBuffer(window_size=30, feature_dim=63)
        assert not buf.is_ready()
        assert buf.fill_level == 0.0

    def test_buffer_fill(self):
        from pipeline.inference import SlidingWindowBuffer
        buf = SlidingWindowBuffer(window_size=5, feature_dim=63)

        for i in range(5):
            buf.add_frame(np.random.randn(63).astype(np.float32))

        assert buf.is_ready()
        assert buf.fill_level == 1.0

    def test_buffer_get_sequence(self):
        from pipeline.inference import SlidingWindowBuffer
        buf = SlidingWindowBuffer(window_size=5, feature_dim=63)

        for i in range(5):
            buf.add_frame(np.random.randn(63).astype(np.float32))

        seq = buf.get_sequence()
        assert seq is not None
        assert seq.shape == (5, 63)

    def test_buffer_none_frames(self):
        from pipeline.inference import SlidingWindowBuffer
        buf = SlidingWindowBuffer(window_size=3, feature_dim=63)

        buf.add_frame(None)
        buf.add_frame(np.ones(63, dtype=np.float32))
        buf.add_frame(None)

        seq = buf.get_sequence()
        assert seq is not None
        assert np.allclose(seq[0], 0.0)  # None → zeros
        assert np.allclose(seq[1], 1.0)  # actual frame
        assert np.allclose(seq[2], 0.0)  # None → zeros

    def test_buffer_sliding(self):
        from pipeline.inference import SlidingWindowBuffer
        buf = SlidingWindowBuffer(window_size=3, feature_dim=2)

        buf.add_frame(np.array([1, 1], dtype=np.float32))
        buf.add_frame(np.array([2, 2], dtype=np.float32))
        buf.add_frame(np.array([3, 3], dtype=np.float32))
        buf.add_frame(np.array([4, 4], dtype=np.float32))

        seq = buf.get_sequence()
        # Should contain frames 2, 3, 4 (1 was pushed out)
        assert np.allclose(seq[0], [2, 2])
        assert np.allclose(seq[1], [3, 3])
        assert np.allclose(seq[2], [4, 4])


class TestPredictionStabilizer:
    """Test prediction stabilizer logic."""

    def test_stabilizer_creation(self):
        from pipeline.inference import PredictionStabilizer
        stab = PredictionStabilizer(window_size=10, threshold=8)
        assert stab.update("A", 0.9) is None  # not enough history

    def test_stabilizer_confirms(self):
        from pipeline.inference import PredictionStabilizer
        stab = PredictionStabilizer(window_size=10, threshold=8)

        # Send 10 predictions of "A"
        result = None
        for i in range(10):
            result = stab.update("A", 0.9)
        assert result == "A"

    def test_stabilizer_rejects_noise(self):
        from pipeline.inference import PredictionStabilizer
        stab = PredictionStabilizer(window_size=10, threshold=8)

        # Mixed predictions — no label reaches threshold
        labels = ["A", "B", "A", "C", "A", "B", "A", "C", "A", "B"]
        results = [stab.update(l, 0.8) for l in labels]
        assert all(r is None for r in results)

    def test_stabilizer_no_repeat(self):
        from pipeline.inference import PredictionStabilizer
        stab = PredictionStabilizer(window_size=5, threshold=4)

        # Confirm "A"
        for i in range(5):
            stab.update("A", 0.9)
        # Confirmed once — should not confirm again
        result = stab.update("A", 0.9)
        assert result is None

    def test_stabilizer_new_sign(self):
        from pipeline.inference import PredictionStabilizer
        stab = PredictionStabilizer(window_size=5, threshold=4)

        # Confirm "A"
        for i in range(5):
            stab.update("A", 0.9)

        # Now confirm "B" — need enough to flush old A's from the deque
        # After 5 B's, deque is [B,B,B,B,B] → count("B")=5 >= threshold=4
        results = []
        for i in range(5):
            r = stab.update("B", 0.9)
            if r is not None:
                results.append(r)
        assert "B" in results


# ════════════════════════════════════════════════════════════
# Run Tests
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
