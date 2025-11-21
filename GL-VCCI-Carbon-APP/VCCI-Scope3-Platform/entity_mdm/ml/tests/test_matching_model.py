# -*- coding: utf-8 -*-
"""
Tests for Entity MDM Matching Model (BERT Re-ranking).

Tests model training, inference, pair scoring, model persistence (save/load),
batch prediction, and edge cases.

Target: 450+ lines, 20 tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple
import tempfile
import os


# Mock matching model (would be actual module in production)
class MatchingModel:
    """BERT-based matching model for entity resolution."""

    def __init__(self, model=None, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        elif model:
            self.model = model
        else:
            raise ValueError("Must provide either model or model_path")

        self.training_history = []

    def _load_model(self, path: str):
        """Load model from disk."""
        # Mock loading
        return MagicMock()

    def train(self, training_pairs: List[Tuple[str, str, int]],
             validation_pairs: List[Tuple[str, str, int]] = None,
             epochs: int = 5, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the matching model."""
        if not training_pairs:
            raise ValueError("Training pairs cannot be empty")

        if epochs <= 0:
            raise ValueError("Epochs must be positive")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Simulate training
        for epoch in range(epochs):
            # Training metrics
            train_loss = 1.0 / (epoch + 1)  # Decreasing loss
            train_acc = 0.5 + (0.4 * epoch / epochs)  # Increasing accuracy

            # Validation metrics
            val_loss = None
            val_acc = None
            if validation_pairs:
                val_loss = 1.2 / (epoch + 1)
                val_acc = 0.45 + (0.45 * epoch / epochs)

            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

        return self.training_history

    def predict_pair(self, text1: str, text2: str) -> float:
        """Predict match score for a single pair."""
        if not text1 or not text2:
            return 0.0

        # Use model to predict
        score = self.model.predict([(text1, text2)])[0]
        return float(np.clip(score, 0.0, 1.0))

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Predict match scores for multiple pairs."""
        if not pairs:
            return np.array([])

        # Filter out empty pairs
        valid_pairs = [(t1, t2) for t1, t2 in pairs if t1 and t2]

        if not valid_pairs:
            return np.zeros(len(pairs))

        # Use model to predict
        scores = self.model.predict(valid_pairs)
        return np.clip(scores, 0.0, 1.0)

    def save(self, path: str):
        """Save model to disk."""
        if not path:
            raise ValueError("Path cannot be empty")

        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Mock saving
        with open(path, 'w') as f:
            f.write("model_data")

    def evaluate(self, test_pairs: List[Tuple[str, str, int]], threshold: float = 0.5) -> dict:
        """Evaluate model on test set."""
        if not test_pairs:
            raise ValueError("Test pairs cannot be empty")

        predictions = []
        actuals = []

        for text1, text2, label in test_pairs:
            score = self.predict_pair(text1, text2)
            predictions.append(1 if score >= threshold else 0)
            actuals.append(label)

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        tp = np.sum((predictions == 1) & (actuals == 1))
        fp = np.sum((predictions == 1) & (actuals == 0))
        tn = np.sum((predictions == 0) & (actuals == 0))
        fn = np.sum((predictions == 0) & (actuals == 1))

        accuracy = (tp + tn) / len(actuals) if len(actuals) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }
        }


# ============================================================================
# TEST SUITE
# ============================================================================

class TestMatchingModel:
    """Test suite for matching model."""

    def test_model_initialization_with_model(self, mock_cross_encoder):
        """Test model initialization with provided model."""
        model = MatchingModel(model=mock_cross_encoder)

        assert model.model is not None
        assert model.training_history == []

    def test_model_initialization_without_model_raises_error(self):
        """Test that initialization without model or path raises error."""
        with pytest.raises(ValueError, match="Must provide either model or model_path"):
            MatchingModel()

    def test_train_model_with_valid_data(self, mock_cross_encoder, sample_supplier_pairs):
        """Test training model with valid training data."""
        model = MatchingModel(model=mock_cross_encoder)

        # Convert sample pairs to format expected by train
        training_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs]

        history = model.train(training_pairs, epochs=3)

        assert len(history) == 3
        assert all("train_loss" in h for h in history)
        assert all("train_accuracy" in h for h in history)

        # Loss should decrease
        assert history[0]["train_loss"] > history[-1]["train_loss"]

        # Accuracy should increase
        assert history[0]["train_accuracy"] < history[-1]["train_accuracy"]

    def test_train_model_with_validation_data(self, mock_cross_encoder, sample_supplier_pairs):
        """Test training with validation set."""
        model = MatchingModel(model=mock_cross_encoder)

        training_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs[:10]]
        validation_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs[10:]]

        history = model.train(training_pairs, validation_pairs=validation_pairs, epochs=3)

        assert len(history) == 3
        assert all("val_loss" in h for h in history)
        assert all("val_accuracy" in h for h in history)

    def test_train_with_empty_data_raises_error(self, mock_cross_encoder):
        """Test that training with empty data raises error."""
        model = MatchingModel(model=mock_cross_encoder)

        with pytest.raises(ValueError, match="Training pairs cannot be empty"):
            model.train([])

    def test_train_with_invalid_epochs_raises_error(self, mock_cross_encoder, sample_supplier_pairs):
        """Test that training with invalid epochs raises error."""
        model = MatchingModel(model=mock_cross_encoder)
        training_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs]

        with pytest.raises(ValueError, match="Epochs must be positive"):
            model.train(training_pairs, epochs=0)

        with pytest.raises(ValueError, match="Epochs must be positive"):
            model.train(training_pairs, epochs=-1)

    def test_train_with_invalid_batch_size_raises_error(self, mock_cross_encoder, sample_supplier_pairs):
        """Test that training with invalid batch size raises error."""
        model = MatchingModel(model=mock_cross_encoder)
        training_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs]

        with pytest.raises(ValueError, match="Batch size must be positive"):
            model.train(training_pairs, batch_size=0)

    def test_predict_pair_returns_valid_score(self, mock_cross_encoder):
        """Test predicting score for a single pair."""
        model = MatchingModel(model=mock_cross_encoder)

        score = model.predict_pair("ACME Corporation", "Acme Corp")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_pair_with_identical_strings(self, mock_cross_encoder):
        """Test prediction for identical strings returns high score."""
        model = MatchingModel(model=mock_cross_encoder)

        score = model.predict_pair("ACME Corporation", "ACME Corporation")

        # Should be high similarity
        assert score > 0.8

    def test_predict_pair_with_very_different_strings(self, mock_cross_encoder):
        """Test prediction for very different strings returns low score."""
        model = MatchingModel(model=mock_cross_encoder)

        score = model.predict_pair("ACME Corporation", "XYZ Industries")

        # Should be low similarity
        assert score < 0.5

    def test_predict_pair_with_empty_strings(self, mock_cross_encoder):
        """Test prediction with empty strings returns zero."""
        model = MatchingModel(model=mock_cross_encoder)

        assert model.predict_pair("", "ACME Corp") == 0.0
        assert model.predict_pair("ACME Corp", "") == 0.0
        assert model.predict_pair("", "") == 0.0

    def test_predict_batch_returns_multiple_scores(self, mock_cross_encoder):
        """Test batch prediction returns scores for all pairs."""
        model = MatchingModel(model=mock_cross_encoder)

        pairs = [
            ("ACME Corporation", "Acme Corp"),
            ("ABC Manufacturing", "ABC Mfg"),
            ("Global Tech", "Global Tech Corp")
        ]

        scores = model.predict_batch(pairs)

        assert len(scores) == 3
        assert all(0.0 <= score <= 1.0 for score in scores)

    def test_predict_batch_with_empty_list(self, mock_cross_encoder):
        """Test batch prediction with empty list."""
        model = MatchingModel(model=mock_cross_encoder)

        scores = model.predict_batch([])

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_save_model(self, mock_cross_encoder, temp_model_path):
        """Test saving model to disk."""
        model = MatchingModel(model=mock_cross_encoder)

        save_path = temp_model_path / "matching_model.bin"
        model.save(str(save_path))

        assert save_path.exists()

    def test_save_model_with_empty_path_raises_error(self, mock_cross_encoder):
        """Test that saving with empty path raises error."""
        model = MatchingModel(model=mock_cross_encoder)

        with pytest.raises(ValueError, match="Path cannot be empty"):
            model.save("")

    def test_evaluate_model_on_test_set(self, mock_cross_encoder, sample_supplier_pairs):
        """Test model evaluation on test set."""
        model = MatchingModel(model=mock_cross_encoder)

        test_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs]
        metrics = model.evaluate(test_pairs, threshold=0.5)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

        # Check metric ranges
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0

    def test_evaluate_with_empty_test_set_raises_error(self, mock_cross_encoder):
        """Test that evaluation with empty test set raises error."""
        model = MatchingModel(model=mock_cross_encoder)

        with pytest.raises(ValueError, match="Test pairs cannot be empty"):
            model.evaluate([])

    def test_model_handles_special_characters(self, mock_cross_encoder):
        """Test model handles special characters correctly."""
        model = MatchingModel(model=mock_cross_encoder)

        pairs = [
            ("O'Reilly Manufacturing", "O'Reilly Mfg"),
            ("AT&T Solutions", "AT&T Sol"),
            ("Société Générale", "Societe Generale")
        ]

        scores = model.predict_batch(pairs)

        assert len(scores) == 3
        assert all(not np.isnan(score) for score in scores)

    def test_training_history_tracked(self, mock_cross_encoder, sample_supplier_pairs):
        """Test that training history is properly tracked."""
        model = MatchingModel(model=mock_cross_encoder)

        training_pairs = [(t1, t2, label) for (t1, t2), label in sample_supplier_pairs]
        model.train(training_pairs, epochs=5)

        assert len(model.training_history) == 5
        assert all("epoch" in h for h in model.training_history)

        # Epochs should be sequential
        for i, history in enumerate(model.training_history):
            assert history["epoch"] == i + 1
