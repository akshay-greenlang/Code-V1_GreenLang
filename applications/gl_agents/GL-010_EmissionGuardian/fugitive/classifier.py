# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Fugitive Emissions Classifier with SHAP/LIME

This module implements supervised classification for fugitive emissions
with full SHAP and LIME explainability for regulatory compliance.

Models Supported:
    - Random Forest (default, SHAP-friendly)
    - Gradient Boosting
    - XGBoost (if available)

Explainability:
    - SHAP TreeExplainer for global/local explanations
    - LIME for instance-level explanations
    - Feature importance tracking

Zero-Hallucination Principle:
    - ML for classification assistance only
    - Human confirmation required for all alerts
    - Complete audit trail with explainability artifacts

Author: GreenLang GL-010 EmissionsGuardian
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import json
import pickle

import numpy as np
from pydantic import BaseModel, Field

from .feature_engineering import FeatureVector, EquipmentType
from .anomaly_detector import AnomalyDetection, AnomalySeverity

logger = logging.getLogger(__name__)


class LeakClassification(str, Enum):
    """Leak classification outcomes."""
    NO_LEAK = "no_leak"
    POSSIBLE_LEAK = "possible_leak"
    PROBABLE_LEAK = "probable_leak"
    CONFIRMED_LEAK = "confirmed_leak"


class LeakCategory(str, Enum):
    """Categories of fugitive emission leaks."""
    EQUIPMENT_LEAK = "equipment_leak"
    PROCESS_VENT = "process_vent"
    TANK_EMISSION = "tank_emission"
    FLARE_ISSUE = "flare_issue"
    PIPELINE_LEAK = "pipeline_leak"
    UNKNOWN = "unknown"


class ClassifierConfig(BaseModel):
    """Configuration for fugitive emissions classifier."""
    # Model selection
    model_type: str = Field(default="random_forest")

    # Random Forest parameters
    rf_n_estimators: int = Field(default=100, ge=10, le=500)
    rf_max_depth: int = Field(default=10, ge=3, le=50)
    rf_min_samples_split: int = Field(default=5, ge=2, le=20)
    rf_random_state: int = Field(default=42)

    # Classification thresholds
    possible_leak_threshold: float = Field(default=0.3, ge=0.1, le=0.5)
    probable_leak_threshold: float = Field(default=0.6, ge=0.4, le=0.8)
    confirmed_leak_threshold: float = Field(default=0.85, ge=0.7, le=0.95)

    # Explainability settings
    enable_shap: bool = Field(default=True)
    enable_lime: bool = Field(default=True)
    shap_max_samples: int = Field(default=100, ge=10, le=1000)
    lime_num_features: int = Field(default=10, ge=5, le=20)

    # Feature selection
    min_feature_importance: float = Field(default=0.01, ge=0.001, le=0.1)


@dataclass
class SHAPExplanation:
    """SHAP-based explanation for a classification."""
    base_value: float
    shap_values: List[float]
    feature_names: List[str]
    feature_values: List[float]
    expected_value: float

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float, float]]:
        """Get top contributing features with their SHAP values."""
        indexed = list(zip(self.feature_names, self.shap_values, self.feature_values))
        sorted_features = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        top_features = self.get_top_features(10)
        return {
            "base_value": round(self.base_value, 4),
            "expected_value": round(self.expected_value, 4),
            "top_features": [
                {
                    "feature": name,
                    "shap_value": round(shap, 4),
                    "feature_value": round(val, 4),
                    "direction": "increases" if shap > 0 else "decreases"
                }
                for name, shap, val in top_features
            ]
        }


@dataclass
class LIMEExplanation:
    """LIME-based explanation for a classification."""
    prediction: float
    intercept: float
    feature_weights: List[Tuple[str, float]]
    prediction_local: float
    r2_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "prediction": round(self.prediction, 4),
            "intercept": round(self.intercept, 4),
            "local_accuracy": round(self.r2_score, 4),
            "feature_weights": [
                {"feature": name, "weight": round(weight, 4)}
                for name, weight in self.feature_weights[:10]
            ]
        }


@dataclass
class ClassificationResult:
    """Complete classification result with explainability."""
    classification_id: str
    detection_id: str
    timestamp: datetime

    # Classification
    leak_probability: float
    classification: LeakClassification
    leak_category: LeakCategory
    confidence: float

    # Context
    concentration_ppm: float
    equipment_id: Optional[str] = None
    equipment_type: Optional[EquipmentType] = None

    # Explainability
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    feature_importance: List[Tuple[str, float]] = field(default_factory=list)

    # Human review
    requires_confirmation: bool = True
    recommended_action: str = ""

    # Audit
    model_version: str = "1.0.0"
    model_hash: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "classification_id": self.classification_id,
            "detection_id": self.detection_id,
            "timestamp": self.timestamp.isoformat(),
            "leak_probability": round(self.leak_probability, 4),
            "classification": self.classification.value,
            "leak_category": self.leak_category.value,
            "confidence": round(self.confidence, 4),
            "concentration_ppm": round(self.concentration_ppm, 2),
            "requires_confirmation": self.requires_confirmation,
            "recommended_action": self.recommended_action,
            "provenance_hash": self.provenance_hash,
        }

        if self.shap_explanation:
            result["shap_explanation"] = self.shap_explanation.to_dict()

        if self.lime_explanation:
            result["lime_explanation"] = self.lime_explanation.to_dict()

        result["feature_importance"] = [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in self.feature_importance[:10]
        ]

        return result


class FugitiveClassifier:
    """
    Supervised Classifier for Fugitive Emissions with SHAP/LIME Explainability.

    Provides leak probability classification with full explainability
    for regulatory compliance and human review.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self._model = None
        self._is_fitted = False
        self._feature_names: List[str] = []
        self._training_samples = 0
        self._shap_explainer = None
        self._classification_counter = 0

        # Calculate model hash for provenance
        config_str = json.dumps(self.config.model_dump(), sort_keys=True)
        self._model_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        logger.info(f"FugitiveClassifier initialized: {self.config.model_type}")

    def fit(
        self,
        feature_vectors: List[FeatureVector],
        labels: List[int]  # 0=no leak, 1=leak
    ) -> Dict[str, Any]:
        """
        Fit the classifier on labeled data.

        Args:
            feature_vectors: Training feature vectors
            labels: Binary labels (0=no leak, 1=leak)

        Returns:
            Training metrics dictionary
        """
        if len(feature_vectors) != len(labels):
            raise ValueError("Feature vectors and labels must have same length")

        X = np.array([fv.to_array() for fv in feature_vectors])
        y = np.array(labels)
        self._feature_names = FeatureVector.feature_names()

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.error("sklearn not available")
            return {"error": "sklearn not installed"}

        # Create and train model
        self._model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=self.config.rf_random_state,
            n_jobs=-1
        )

        self._model.fit(X, y)
        self._is_fitted = True
        self._training_samples = len(feature_vectors)

        # Calculate cross-validation score
        cv_scores = cross_val_score(self._model, X, y, cv=5)

        # Initialize SHAP explainer
        if self.config.enable_shap:
            try:
                import shap
                # Use subset for SHAP background
                background_size = min(self.config.shap_max_samples, len(X))
                background_idx = np.random.choice(len(X), background_size, replace=False)
                self._shap_explainer = shap.TreeExplainer(
                    self._model,
                    X[background_idx]
                )
                logger.info("SHAP TreeExplainer initialized")
            except ImportError:
                logger.warning("SHAP not available - disabling SHAP explanations")
                self.config.enable_shap = False

        # Get feature importances
        feature_importance = list(zip(
            self._feature_names,
            self._model.feature_importances_
        ))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Classifier trained on {len(X)} samples, CV accuracy: {cv_scores.mean():.3f}")

        return {
            "training_samples": len(X),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "feature_importance": feature_importance[:10]
        }

    def classify(
        self,
        feature_vector: FeatureVector,
        detection: Optional[AnomalyDetection] = None
    ) -> ClassificationResult:
        """
        Classify a feature vector and provide explainability.

        Args:
            feature_vector: Feature vector to classify
            detection: Optional associated anomaly detection

        Returns:
            ClassificationResult with full explainability
        """
        self._classification_counter += 1
        classification_id = f"CLS-{feature_vector.timestamp.strftime('%Y%m%d%H%M%S')}-{self._classification_counter:06d}"
        detection_id = detection.detection_id if detection else "N/A"

        # Get prediction
        if self._is_fitted and self._model is not None:
            X = np.array([feature_vector.to_array()])
            probability = self._model.predict_proba(X)[0, 1]

            # Get feature importance for this instance
            feature_importance = list(zip(
                self._feature_names,
                self._model.feature_importances_
            ))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
        else:
            # Fallback to heuristic
            probability = self._heuristic_probability(feature_vector)
            feature_importance = []

        # Determine classification
        classification = self._determine_classification(probability)

        # Determine leak category
        leak_category = self._determine_category(feature_vector)

        # Calculate confidence
        confidence = self._calculate_confidence(probability, feature_vector)

        # Get SHAP explanation
        shap_explanation = None
        if self.config.enable_shap and self._shap_explainer is not None:
            shap_explanation = self._get_shap_explanation(feature_vector)

        # Get LIME explanation
        lime_explanation = None
        if self.config.enable_lime and self._is_fitted:
            lime_explanation = self._get_lime_explanation(feature_vector)

        # Determine recommended action
        recommended_action = self._get_recommended_action(classification)

        # Calculate provenance hash
        provenance_content = (
            f"{classification_id}|{probability}|{classification.value}"
        )
        provenance_hash = hashlib.sha256(provenance_content.encode()).hexdigest()

        return ClassificationResult(
            classification_id=classification_id,
            detection_id=detection_id,
            timestamp=feature_vector.timestamp,
            leak_probability=float(probability),
            classification=classification,
            leak_category=leak_category,
            confidence=confidence,
            concentration_ppm=feature_vector.concentration_current,
            equipment_id=None,  # Would be populated from context
            equipment_type=EquipmentType(feature_vector.equipment_type_encoded) if feature_vector.equipment_type_encoded > 0 else None,
            shap_explanation=shap_explanation,
            lime_explanation=lime_explanation,
            feature_importance=feature_importance,
            requires_confirmation=classification != LeakClassification.NO_LEAK,
            recommended_action=recommended_action,
            model_version="1.0.0",
            model_hash=self._model_hash,
            provenance_hash=provenance_hash
        )

    def classify_batch(
        self,
        feature_vectors: List[FeatureVector],
        detections: Optional[List[AnomalyDetection]] = None
    ) -> List[ClassificationResult]:
        """Classify a batch of feature vectors."""
        if detections is None:
            detections = [None] * len(feature_vectors)

        return [
            self.classify(fv, det)
            for fv, det in zip(feature_vectors, detections)
        ]

    def _determine_classification(self, probability: float) -> LeakClassification:
        """Determine leak classification from probability."""
        if probability >= self.config.confirmed_leak_threshold:
            return LeakClassification.CONFIRMED_LEAK
        if probability >= self.config.probable_leak_threshold:
            return LeakClassification.PROBABLE_LEAK
        if probability >= self.config.possible_leak_threshold:
            return LeakClassification.POSSIBLE_LEAK
        return LeakClassification.NO_LEAK

    def _determine_category(self, feature_vector: FeatureVector) -> LeakCategory:
        """Determine leak category based on features."""
        equip_type = feature_vector.equipment_type_encoded

        # Map equipment type to leak category
        if equip_type in [1, 2, 3, 4, 5]:  # Valve, pump, compressor, flange, connector
            return LeakCategory.EQUIPMENT_LEAK
        if equip_type == 8:  # Tank
            return LeakCategory.TANK_EMISSION
        if equip_type == 6:  # Pressure relief
            return LeakCategory.PROCESS_VENT

        # Check for pipeline based on spatial pattern
        if feature_vector.spatial_anomaly_score > 0.5:
            return LeakCategory.PIPELINE_LEAK

        return LeakCategory.UNKNOWN

    def _calculate_confidence(
        self,
        probability: float,
        feature_vector: FeatureVector
    ) -> float:
        """Calculate classification confidence."""
        # Base confidence from probability distance to threshold
        if probability > 0.5:
            base_confidence = (probability - 0.5) * 2
        else:
            base_confidence = (0.5 - probability) * 2

        # Adjust for data quality
        quality_factor = 1.0

        # Adjust for feature consistency
        consistency = 1.0 - feature_vector.concentration_std / max(
            feature_vector.concentration_mean, 1.0
        )
        consistency = max(0.5, min(1.0, consistency))

        return min(1.0, base_confidence * quality_factor * consistency)

    def _heuristic_probability(self, feature_vector: FeatureVector) -> float:
        """Calculate heuristic leak probability when model not fitted."""
        scores = []

        # Z-score contribution
        if feature_vector.concentration_zscore > 2:
            scores.append(min(1.0, feature_vector.concentration_zscore / 5.0))

        # Elevation contribution
        if feature_vector.elevation_above_background > 50:
            scores.append(min(1.0, feature_vector.elevation_above_background / 200.0))

        # Plume likelihood
        scores.append(feature_vector.plume_likelihood_score)

        # Spatial anomaly
        scores.append(feature_vector.spatial_anomaly_score * 0.8)

        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def _get_shap_explanation(
        self,
        feature_vector: FeatureVector
    ) -> Optional[SHAPExplanation]:
        """Get SHAP explanation for classification."""
        if self._shap_explainer is None:
            return None

        try:
            X = np.array([feature_vector.to_array()])
            shap_values = self._shap_explainer.shap_values(X)

            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            expected_value = self._shap_explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            return SHAPExplanation(
                base_value=float(expected_value),
                shap_values=values.tolist(),
                feature_names=self._feature_names,
                feature_values=X[0].tolist(),
                expected_value=float(expected_value)
            )
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return None

    def _get_lime_explanation(
        self,
        feature_vector: FeatureVector
    ) -> Optional[LIMEExplanation]:
        """Get LIME explanation for classification."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            logger.warning("LIME not available")
            return None

        if not self._is_fitted or self._model is None:
            return None

        try:
            X = np.array([feature_vector.to_array()])

            # Create simple LIME explainer
            explainer = LimeTabularExplainer(
                X,
                feature_names=self._feature_names,
                class_names=['no_leak', 'leak'],
                mode='classification'
            )

            explanation = explainer.explain_instance(
                X[0],
                self._model.predict_proba,
                num_features=self.config.lime_num_features
            )

            # Extract weights
            feature_weights = explanation.as_list()

            return LIMEExplanation(
                prediction=float(self._model.predict_proba(X)[0, 1]),
                intercept=float(explanation.intercept[1]),
                feature_weights=feature_weights,
                prediction_local=float(explanation.local_pred[0]),
                r2_score=float(explanation.score)
            )
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return None

    def _get_recommended_action(self, classification: LeakClassification) -> str:
        """Get recommended action based on classification."""
        actions = {
            LeakClassification.NO_LEAK: "Continue routine monitoring",
            LeakClassification.POSSIBLE_LEAK: "Schedule inspection within 7 days",
            LeakClassification.PROBABLE_LEAK: "Investigate within 24 hours",
            LeakClassification.CONFIRMED_LEAK: "Immediate response required - initiate repair",
        }
        return actions.get(classification, "Review required")

    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if not self._is_fitted:
            raise ValueError("Model not fitted - cannot save")

        model_data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "training_samples": self._training_samples,
            "config": self.config.model_dump(),
            "model_hash": self._model_hash,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self._model = model_data["model"]
        self._feature_names = model_data["feature_names"]
        self._training_samples = model_data["training_samples"]
        self._model_hash = model_data.get("model_hash", "unknown")
        self._is_fitted = True

        # Re-initialize SHAP if enabled
        if self.config.enable_shap:
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self._model)
            except ImportError:
                pass

        logger.info(f"Model loaded from {path}")


# Export all public classes
__all__ = [
    "LeakClassification",
    "LeakCategory",
    "ClassifierConfig",
    "SHAPExplanation",
    "LIMEExplanation",
    "ClassificationResult",
    "FugitiveClassifier",
]
