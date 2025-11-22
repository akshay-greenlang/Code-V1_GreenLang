# -*- coding: utf-8 -*-
"""
ML model training utilities for GL-008.

This module provides training pipelines for:
1. Acoustic anomaly detection (Isolation Forest)
2. Acoustic failure classification (Random Forest)
3. Thermal image classification (CNN)
4. RUL prediction (Weibull + Gradient Boosting)
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Any
from pathlib import Path

# ML frameworks
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from feature_extractors import AcousticFeatureExtractor, ThermalFeatureExtractor


class AcousticAnomalyDetectorTrainer:
    """Train Isolation Forest for acoustic anomaly detection."""

    def __init__(self, contamination: float = 0.15):
        """
        Initialize trainer.

        Args:
            contamination: Expected proportion of anomalies (default: 15%)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples=256
        )
        self.scaler = StandardScaler()
        self.feature_extractor = AcousticFeatureExtractor()

    def train(
        self,
        normal_signals: List[np.ndarray],
        anomalous_signals: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train anomaly detection model.

        Args:
            normal_signals: List of normal acoustic signals
            anomalous_signals: List of anomalous signals

        Returns:
            Training metrics
        """
        # Extract features
        print("Extracting features from normal signals...")
        X_normal = self._extract_features_batch(normal_signals)

        print("Extracting features from anomalous signals...")
        X_anomalous = self._extract_features_batch(anomalous_signals)

        # Combine datasets
        X = np.vstack([X_normal, X_anomalous])
        y = np.array([1] * len(X_normal) + [-1] * len(X_anomalous))

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        print("Training Isolation Forest...")
        self.model.fit(X_scaled)

        # Evaluate
        predictions = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, predictions)

        # Anomaly scores
        anomaly_scores = self.model.score_samples(X_scaled)

        metrics = {
            'accuracy': float(accuracy),
            'n_normal': len(X_normal),
            'n_anomalous': len(X_anomalous),
            'contamination': self.contamination,
            'mean_anomaly_score_normal': float(np.mean(anomaly_scores[:len(X_normal)])),
            'mean_anomaly_score_anomalous': float(np.mean(anomaly_scores[len(X_normal):]))
        }

        print(f"Training complete! Accuracy: {accuracy:.2%}")
        return metrics

    def _extract_features_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of signals."""
        features_list = []
        for signal in signals:
            features = self.feature_extractor.extract_features(signal)
            features_list.append(list(features.values()))
        return np.array(features_list)

    def save(self, model_path: Path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")


class AcousticClassifierTrainer:
    """Train Random Forest for acoustic failure mode classification."""

    def __init__(self):
        """Initialize trainer."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_extractor = AcousticFeatureExtractor()
        self.classes = [
            'normal',
            'failed_open',
            'failed_closed',
            'leaking',
            'cavitation',
            'worn_seat'
        ]

    def train(
        self,
        signals: List[np.ndarray],
        labels: List[str]
    ) -> Dict[str, Any]:
        """
        Train failure classification model.

        Args:
            signals: List of acoustic signals
            labels: List of failure mode labels

        Returns:
            Training metrics
        """
        # Extract features
        print("Extracting features...")
        X = self._extract_features_batch(signals)

        # Encode labels
        label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        y = np.array([label_to_idx[label] for label in labels])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print("Training Random Forest classifier...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_accuracy_mean'] = float(np.mean(cv_scores))
        metrics['cv_accuracy_std'] = float(np.std(cv_scores))

        print(f"Training complete! Test Accuracy: {metrics['accuracy']:.2%}")
        return metrics

    def _extract_features_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """Extract features from batch of signals."""
        features_list = []
        for signal in signals:
            features = self.feature_extractor.extract_features(signal)
            features_list.append(list(features.values()))
        return np.array(features_list)

    def save(self, model_path: Path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")


class ThermalClassifierTrainer:
    """Train CNN for thermal image classification."""

    def __init__(self, input_shape: Tuple[int, int] = (64, 64)):
        """
        Initialize trainer.

        Args:
            input_shape: Input image dimensions
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for CNN training")

        self.input_shape = input_shape + (1,)  # Add channel dimension
        self.model = None
        self.feature_extractor = ThermalFeatureExtractor()
        self.classes = ['normal', 'degraded', 'failed']

    def build_model(self) -> keras.Model:
        """Build CNN architecture."""
        model = keras.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),

            # Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(
        self,
        images: List[np.ndarray],
        labels: List[str],
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train thermal image classifier.

        Args:
            images: List of thermal images
            labels: List of condition labels
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics
        """
        # Prepare images
        print("Preparing images for CNN...")
        X = np.array([
            self.feature_extractor.prepare_cnn_input(img)
            for img in images
        ])

        # Encode labels
        label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        y = np.array([label_to_idx[label] for label in labels])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build model
        print("Building CNN model...")
        self.model = self.build_model()

        # Train
        print("Training CNN...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)

        metrics = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'epochs': epochs
        }

        print(f"Training complete! Test Accuracy: {test_accuracy:.2%}")
        return metrics

    def save(self, model_path: Path):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")


class RULPredictorTrainer:
    """Train Gradient Boosting model for RUL prediction."""

    def __init__(self):
        """Initialize trainer."""
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()

    def train(
        self,
        features: np.ndarray,
        rul_days: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train RUL prediction model.

        Args:
            features: Feature matrix (age, health score, degradation rate, etc.)
            rul_days: Target RUL in days

        Returns:
            Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, rul_days, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print("Training Gradient Boosting for RUL prediction...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

        metrics = {
            'mae_days': float(mae),
            'mape_percent': float(mape),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

        print(f"Training complete! MAE: {mae:.2f} days, MAPE: {mape:.2f}%")
        return metrics

    def save(self, model_path: Path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")


# Example usage and synthetic data generation
def generate_synthetic_training_data():
    """Generate synthetic training data for demonstration."""
    print("Generating synthetic training data...")

    # Acoustic signals (simplified)
    normal_signals = [np.random.randn(10000) * 0.1 for _ in range(100)]
    failed_open_signals = [np.random.randn(10000) * 0.5 for _ in range(30)]

    # Thermal images
    normal_images = [np.random.randn(64, 64) * 10 + 100 for _ in range(80)]
    failed_images = [np.random.randn(64, 64) * 20 + 150 for _ in range(40)]

    # RUL data
    n_samples = 200
    rul_features = np.random.randn(n_samples, 5)  # age, health, degradation, etc.
    rul_targets = np.random.uniform(30, 365, n_samples)

    return {
        'acoustic_normal': normal_signals,
        'acoustic_anomalous': failed_open_signals,
        'thermal_normal': normal_images,
        'thermal_failed': failed_images,
        'rul_features': rul_features,
        'rul_targets': rul_targets
    }


if __name__ == "__main__":
    print("GL-008 ML Model Training Pipeline")
    print("=" * 50)

    # Generate synthetic data
    data = generate_synthetic_training_data()

    # Train models
    models_dir = Path("./")

    # 1. Acoustic anomaly detector
    print("\n1. Training Acoustic Anomaly Detector...")
    anomaly_trainer = AcousticAnomalyDetectorTrainer()
    anomaly_metrics = anomaly_trainer.train(
        data['acoustic_normal'],
        data['acoustic_anomalous']
    )
    anomaly_trainer.save(models_dir / "acoustic_anomaly_detector.pkl")

    # 2. RUL predictor
    print("\n2. Training RUL Predictor...")
    rul_trainer = RULPredictorTrainer()
    rul_metrics = rul_trainer.train(
        data['rul_features'],
        data['rul_targets']
    )
    rul_trainer.save(models_dir / "rul_predictor.pkl")

    print("\n" + "=" * 50)
    print("Training complete! All models saved.")
    print("\nNote: Thermal CNN and Acoustic Classifier require TensorFlow")
    print("and larger datasets for production deployment.")
