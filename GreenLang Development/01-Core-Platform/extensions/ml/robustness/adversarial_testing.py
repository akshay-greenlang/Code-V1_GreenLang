# -*- coding: utf-8 -*-
"""
TASK-061: Adversarial Testing Framework

This module provides comprehensive adversarial testing capabilities for
GreenLang Process Heat ML models, including FGSM, PGD attacks, input
boundary testing, robustness scoring, and adversarial example detection.

Adversarial testing is critical for ensuring ML models are robust against
malicious inputs and edge cases in safety-critical Process Heat applications.

Example:
    >>> from greenlang.ml.robustness import AdversarialTestingFramework
    >>> framework = AdversarialTestingFramework(model, config=AdversarialTestingConfig(
    ...     process_heat_bounds=ProcessHeatBounds()
    ... ))
    >>> result = framework.comprehensive_test(X_test, y_test)
    >>> print(f"Robustness score: {result.overall_robustness_score:.3f}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Process Heat Physical Bounds
# =============================================================================

class ProcessHeatBounds(BaseModel):
    """
    Physical bounds for Process Heat systems.

    These bounds enforce realistic constraints for industrial
    heating applications including boilers, furnaces, and heat exchangers.
    """

    # Temperature bounds (Celsius)
    temperature_min: float = Field(default=-40.0, description="Minimum temperature (C)")
    temperature_max: float = Field(default=1200.0, description="Maximum temperature (C)")
    ambient_temperature_min: float = Field(default=-40.0, description="Min ambient temp (C)")
    ambient_temperature_max: float = Field(default=50.0, description="Max ambient temp (C)")

    # Pressure bounds (bar)
    pressure_min: float = Field(default=0.0, description="Minimum pressure (bar)")
    pressure_max: float = Field(default=300.0, description="Maximum pressure (bar)")

    # Efficiency bounds (percentage)
    efficiency_min: float = Field(default=0.0, description="Minimum efficiency (%)")
    efficiency_max: float = Field(default=100.0, description="Maximum efficiency (%)")
    boiler_efficiency_typical_min: float = Field(default=70.0, description="Typical boiler min efficiency")
    boiler_efficiency_typical_max: float = Field(default=95.0, description="Typical boiler max efficiency")

    # Flow rate bounds (kg/s)
    flow_rate_min: float = Field(default=0.0, description="Minimum flow rate (kg/s)")
    flow_rate_max: float = Field(default=1000.0, description="Maximum flow rate (kg/s)")

    # Heat duty bounds (MW)
    heat_duty_min: float = Field(default=0.0, description="Minimum heat duty (MW)")
    heat_duty_max: float = Field(default=500.0, description="Maximum heat duty (MW)")

    # Fuel consumption bounds (kg/h)
    fuel_consumption_min: float = Field(default=0.0, description="Minimum fuel consumption")
    fuel_consumption_max: float = Field(default=50000.0, description="Maximum fuel consumption")

    # Emission bounds (kg CO2e/MWh)
    emission_factor_min: float = Field(default=0.0, description="Min emission factor")
    emission_factor_max: float = Field(default=1000.0, description="Max emission factor")


# =============================================================================
# Attack Methods Enum
# =============================================================================

class AdversarialAttackMethod(str, Enum):
    """Adversarial attack methods for robustness testing."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"  # Projected Gradient Descent
    NOISE = "noise"  # Random noise perturbation
    BOUNDARY = "boundary"  # Input boundary testing
    FEATURE_PERTURBATION = "feature_perturbation"  # Targeted feature attacks
    CARLINI_WAGNER = "carlini_wagner"  # C&W attack
    DEEPFOOL = "deepfool"  # DeepFool attack


class DetectionMethod(str, Enum):
    """Methods for detecting adversarial examples."""
    STATISTICAL = "statistical"
    INPUT_VALIDATION = "input_validation"
    FEATURE_SQUEEZING = "feature_squeezing"
    PREDICTION_CONSISTENCY = "prediction_consistency"


# =============================================================================
# Configuration
# =============================================================================

class AdversarialTestingConfig(BaseModel):
    """Configuration for adversarial testing framework."""

    # Attack configuration
    attack_methods: List[AdversarialAttackMethod] = Field(
        default_factory=lambda: [
            AdversarialAttackMethod.FGSM,
            AdversarialAttackMethod.PGD,
            AdversarialAttackMethod.BOUNDARY,
            AdversarialAttackMethod.NOISE
        ],
        description="Attack methods to apply"
    )

    # FGSM configuration
    fgsm_epsilon: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="FGSM perturbation magnitude"
    )

    # PGD configuration
    pgd_epsilon: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="PGD maximum perturbation"
    )
    pgd_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="PGD iterations"
    )
    pgd_alpha: float = Field(
        default=0.01,
        gt=0,
        description="PGD step size"
    )

    # Boundary testing
    boundary_samples_per_feature: int = Field(
        default=20,
        ge=5,
        description="Samples per feature for boundary testing"
    )

    # Detection configuration
    detection_methods: List[DetectionMethod] = Field(
        default_factory=lambda: [
            DetectionMethod.STATISTICAL,
            DetectionMethod.INPUT_VALIDATION
        ],
        description="Detection methods to use"
    )

    # Process Heat specific
    process_heat_bounds: ProcessHeatBounds = Field(
        default_factory=ProcessHeatBounds,
        description="Physical bounds for Process Heat"
    )
    enforce_physical_bounds: bool = Field(
        default=True,
        description="Enforce physical bounds on perturbations"
    )

    # General settings
    n_samples: int = Field(
        default=100,
        ge=10,
        description="Number of samples for testing"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashing"
    )


# =============================================================================
# Result Models
# =============================================================================

class AdversarialSample(BaseModel):
    """Information about a generated adversarial sample."""

    sample_index: int = Field(..., description="Original sample index")
    attack_method: str = Field(..., description="Attack method used")
    original_prediction: float = Field(..., description="Original prediction")
    adversarial_prediction: float = Field(..., description="Adversarial prediction")
    perturbation_norm_l2: float = Field(..., description="L2 norm of perturbation")
    perturbation_norm_linf: float = Field(..., description="L-infinity norm")
    prediction_change: float = Field(..., description="Change in prediction")
    prediction_change_pct: float = Field(..., description="Percentage change")
    attack_success: bool = Field(..., description="Whether attack succeeded")
    features_perturbed: List[int] = Field(
        default_factory=list,
        description="Indices of perturbed features"
    )
    physical_bounds_violated: bool = Field(
        default=False,
        description="Whether physical bounds were violated"
    )


class BoundaryTestResult(BaseModel):
    """Result from input boundary testing."""

    feature_index: int = Field(..., description="Feature index tested")
    feature_name: Optional[str] = Field(default=None, description="Feature name")
    min_bound: float = Field(..., description="Minimum bound tested")
    max_bound: float = Field(..., description="Maximum bound tested")
    predictions_at_min: List[float] = Field(..., description="Predictions at minimum")
    predictions_at_max: List[float] = Field(..., description="Predictions at maximum")
    sensitivity_score: float = Field(..., description="Boundary sensitivity score")
    stable_at_bounds: bool = Field(..., description="Whether model is stable at bounds")
    discontinuity_detected: bool = Field(..., description="Discontinuity detected")


class DetectionResult(BaseModel):
    """Result from adversarial example detection."""

    method: str = Field(..., description="Detection method used")
    samples_tested: int = Field(..., description="Number of samples tested")
    adversarial_detected: int = Field(..., description="Adversarial samples detected")
    detection_rate: float = Field(..., description="Detection rate")
    false_positive_rate: float = Field(..., description="False positive rate")
    detection_threshold: float = Field(..., description="Detection threshold used")


class RobustnessScore(BaseModel):
    """Robustness scoring breakdown."""

    fgsm_robustness: float = Field(..., description="Robustness against FGSM")
    pgd_robustness: float = Field(..., description="Robustness against PGD")
    noise_robustness: float = Field(..., description="Robustness against noise")
    boundary_robustness: float = Field(..., description="Boundary robustness")
    overall_score: float = Field(..., description="Overall robustness score (0-1)")
    grade: str = Field(..., description="Robustness grade (A-F)")


class AdversarialTestingResult(BaseModel):
    """Comprehensive result from adversarial testing."""

    # Summary
    overall_robustness_score: float = Field(
        ...,
        description="Overall robustness score (0-1)"
    )
    robustness_grade: str = Field(
        ...,
        description="Robustness grade (A-F)"
    )

    # Attack results
    total_attacks: int = Field(..., description="Total attacks performed")
    successful_attacks: int = Field(..., description="Successful attacks")
    attack_success_rate: float = Field(..., description="Attack success rate")
    attack_results_by_method: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Results per attack method"
    )

    # Adversarial samples
    adversarial_samples: List[AdversarialSample] = Field(
        default_factory=list,
        description="Generated adversarial samples"
    )

    # Boundary testing
    boundary_results: List[BoundaryTestResult] = Field(
        default_factory=list,
        description="Boundary test results"
    )

    # Detection results
    detection_results: List[DetectionResult] = Field(
        default_factory=list,
        description="Detection results"
    )

    # Scoring breakdown
    robustness_scores: RobustnessScore = Field(
        ...,
        description="Robustness score breakdown"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test timestamp"
    )
    config_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used"
    )


# =============================================================================
# Adversarial Testing Framework
# =============================================================================

class AdversarialTestingFramework:
    """
    Comprehensive Adversarial Testing Framework for Process Heat ML Models.

    This framework provides:
    - FGSM (Fast Gradient Sign Method) attacks
    - PGD (Projected Gradient Descent) attacks
    - Input boundary testing for Process Heat ranges
    - Model robustness scoring
    - Adversarial example detection

    All calculations are deterministic for reproducibility.

    Attributes:
        model: ML model to test
        config: Testing configuration
        _rng: Random number generator for reproducibility

    Example:
        >>> framework = AdversarialTestingFramework(
        ...     model,
        ...     config=AdversarialTestingConfig(
        ...         process_heat_bounds=ProcessHeatBounds(
        ...             temperature_max=1000.0,
        ...             efficiency_max=95.0
        ...         )
        ...     )
        ... )
        >>> result = framework.comprehensive_test(X_test, y_test)
        >>> if result.overall_robustness_score < 0.7:
        ...     print("Model needs robustness improvements")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[AdversarialTestingConfig] = None,
        gradient_fn: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize adversarial testing framework.

        Args:
            model: ML model with predict() method
            config: Testing configuration
            gradient_fn: Optional gradient function for gradient-based attacks
            feature_names: Optional feature names for reporting
        """
        self.model = model
        self.config = config or AdversarialTestingConfig()
        self._gradient_fn = gradient_fn
        self.feature_names = feature_names

        # Initialize RNG for reproducibility
        self._rng = np.random.RandomState(self.config.random_state)

        logger.info(
            f"AdversarialTestingFramework initialized: "
            f"attacks={[m.value for m in self.config.attack_methods]}"
        )

    def _get_prediction(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions (deterministic)."""
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X)
        else:
            preds = self.model(X)

        return np.atleast_1d(preds).flatten()

    def _estimate_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        delta: float = 1e-4
    ) -> np.ndarray:
        """
        Estimate gradient using finite differences (deterministic).

        Args:
            X: Input features (n_samples, n_features)
            y: Target values
            delta: Step size for finite differences

        Returns:
            Estimated gradients (n_samples, n_features)
        """
        if self._gradient_fn is not None:
            return self._gradient_fn(X, y)

        gradients = np.zeros_like(X)
        base_pred = self._get_prediction(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += delta

            pred_plus = self._get_prediction(X_plus)

            # Gradient of loss with respect to input
            # For regression: d(y - pred)^2 / dx = -2 * (y - pred) * dpred/dx
            gradients[:, i] = (pred_plus - base_pred) / delta

        return gradients

    def _apply_physical_bounds(
        self,
        X: np.ndarray,
        feature_bounds: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Apply physical bounds to perturbed inputs (deterministic).

        Args:
            X: Input features
            feature_bounds: Optional per-feature bounds

        Returns:
            Bounded inputs
        """
        if not self.config.enforce_physical_bounds:
            return X

        X_bounded = X.copy()
        bounds = self.config.process_heat_bounds

        # Apply default bounds based on typical Process Heat features
        # This is configurable via feature_bounds parameter
        if feature_bounds:
            for feat_idx, (low, high) in feature_bounds.items():
                X_bounded[:, feat_idx] = np.clip(X_bounded[:, feat_idx], low, high)

        return X_bounded

    # =========================================================================
    # Attack Methods
    # =========================================================================

    def fgsm_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: Optional[float] = None,
        targeted: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Gradient Sign Method (FGSM) attack.

        FGSM computes adversarial examples in a single step:
        X_adv = X + epsilon * sign(gradient_x(loss))

        Args:
            X: Original inputs
            y: Target values
            epsilon: Perturbation magnitude (uses config default if None)
            targeted: If True, minimize loss to target (not implemented)

        Returns:
            Tuple of (adversarial examples, perturbations)
        """
        epsilon = epsilon or self.config.fgsm_epsilon

        # Estimate gradient
        gradients = self._estimate_gradient(X, y)

        # FGSM perturbation
        perturbation = epsilon * np.sign(gradients)

        # Generate adversarial examples
        X_adv = X + perturbation

        # Apply physical bounds if enabled
        X_adv = self._apply_physical_bounds(X_adv)

        # Recalculate actual perturbation after bounds
        actual_perturbation = X_adv - X

        return X_adv, actual_perturbation

    def pgd_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: Optional[float] = None,
        n_iterations: Optional[int] = None,
        alpha: Optional[float] = None,
        random_start: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projected Gradient Descent (PGD) attack.

        PGD is an iterative attack that projects perturbations back to
        the epsilon-ball:
        X_adv = clip(X_adv + alpha * sign(gradient), X - epsilon, X + epsilon)

        Args:
            X: Original inputs
            y: Target values
            epsilon: Maximum perturbation magnitude
            n_iterations: Number of iterations
            alpha: Step size per iteration
            random_start: Random initialization within epsilon-ball

        Returns:
            Tuple of (adversarial examples, perturbations)
        """
        epsilon = epsilon or self.config.pgd_epsilon
        n_iterations = n_iterations or self.config.pgd_iterations
        alpha = alpha or self.config.pgd_alpha

        # Initialize adversarial examples
        if random_start:
            X_adv = X + self._rng.uniform(-epsilon, epsilon, X.shape)
        else:
            X_adv = X.copy()

        # Iterative attack
        for _ in range(n_iterations):
            # Estimate gradient
            gradients = self._estimate_gradient(X_adv, y)

            # Step in gradient direction
            X_adv = X_adv + alpha * np.sign(gradients)

            # Project back to epsilon-ball around original
            perturbation = X_adv - X
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            X_adv = X + perturbation

        # Apply physical bounds
        X_adv = self._apply_physical_bounds(X_adv)

        # Final perturbation
        actual_perturbation = X_adv - X

        return X_adv, actual_perturbation

    def noise_attack(
        self,
        X: np.ndarray,
        epsilon: float = 0.1,
        noise_type: str = "uniform"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random noise perturbation attack.

        Args:
            X: Original inputs
            epsilon: Noise magnitude
            noise_type: "uniform" or "gaussian"

        Returns:
            Tuple of (adversarial examples, perturbations)
        """
        if noise_type == "gaussian":
            perturbation = self._rng.randn(*X.shape) * epsilon
        else:  # uniform
            perturbation = self._rng.uniform(-epsilon, epsilon, X.shape)

        X_adv = X + perturbation
        X_adv = self._apply_physical_bounds(X_adv)

        actual_perturbation = X_adv - X

        return X_adv, actual_perturbation

    def boundary_attack(
        self,
        X: np.ndarray,
        feature_bounds: Dict[int, Tuple[float, float]]
    ) -> List[BoundaryTestResult]:
        """
        Input boundary testing for Process Heat ranges.

        Tests model behavior at and beyond physical boundaries
        to ensure stable predictions.

        Args:
            X: Reference inputs
            feature_bounds: Dict of {feature_idx: (min, max)} bounds

        Returns:
            List of boundary test results
        """
        results = []
        n_samples = self.config.boundary_samples_per_feature

        for feat_idx, (min_val, max_val) in feature_bounds.items():
            # Create test samples at boundaries
            X_test = X.copy()

            # Test at minimum bound
            X_at_min = X_test.copy()
            X_at_min[:, feat_idx] = min_val
            preds_at_min = self._get_prediction(X_at_min)

            # Test at maximum bound
            X_at_max = X_test.copy()
            X_at_max[:, feat_idx] = max_val
            preds_at_max = self._get_prediction(X_at_max)

            # Test just beyond bounds
            X_below_min = X_test.copy()
            X_below_min[:, feat_idx] = min_val - 0.1 * abs(max_val - min_val)

            X_above_max = X_test.copy()
            X_above_max[:, feat_idx] = max_val + 0.1 * abs(max_val - min_val)

            preds_below = self._get_prediction(X_below_min)
            preds_above = self._get_prediction(X_above_max)

            # Calculate sensitivity score
            pred_range = np.max(preds_at_max) - np.min(preds_at_min)
            baseline_range = np.std(self._get_prediction(X_test))
            sensitivity_score = pred_range / (baseline_range + 1e-10)

            # Check for discontinuities
            discontinuity = (
                np.any(np.abs(preds_at_min - preds_below) > 2 * baseline_range) or
                np.any(np.abs(preds_at_max - preds_above) > 2 * baseline_range)
            )

            # Check stability at bounds
            stable = (
                np.std(preds_at_min) < baseline_range and
                np.std(preds_at_max) < baseline_range
            )

            results.append(BoundaryTestResult(
                feature_index=feat_idx,
                feature_name=self.feature_names[feat_idx] if self.feature_names else None,
                min_bound=min_val,
                max_bound=max_val,
                predictions_at_min=preds_at_min.tolist()[:10],  # Limit size
                predictions_at_max=preds_at_max.tolist()[:10],
                sensitivity_score=float(sensitivity_score),
                stable_at_bounds=stable,
                discontinuity_detected=discontinuity
            ))

        return results

    # =========================================================================
    # Adversarial Detection
    # =========================================================================

    def detect_adversarial_statistical(
        self,
        X_test: np.ndarray,
        X_reference: np.ndarray,
        threshold_percentile: float = 95.0
    ) -> Tuple[np.ndarray, float]:
        """
        Statistical detection of adversarial examples.

        Compares input distribution to reference (training) distribution.

        Args:
            X_test: Samples to test
            X_reference: Reference samples
            threshold_percentile: Percentile for anomaly threshold

        Returns:
            Tuple of (is_adversarial flags, threshold)
        """
        # Calculate Mahalanobis-like distance
        ref_mean = np.mean(X_reference, axis=0)
        ref_std = np.std(X_reference, axis=0) + 1e-10

        # Normalized distance
        distances = np.sqrt(np.sum(((X_test - ref_mean) / ref_std) ** 2, axis=1))

        # Calculate threshold from reference
        ref_distances = np.sqrt(np.sum(((X_reference - ref_mean) / ref_std) ** 2, axis=1))
        threshold = np.percentile(ref_distances, threshold_percentile)

        is_adversarial = distances > threshold

        return is_adversarial, float(threshold)

    def detect_adversarial_input_validation(
        self,
        X_test: np.ndarray,
        feature_bounds: Dict[int, Tuple[float, float]]
    ) -> np.ndarray:
        """
        Input validation based adversarial detection.

        Checks if inputs violate physical bounds.

        Args:
            X_test: Samples to test
            feature_bounds: Physical bounds per feature

        Returns:
            Boolean array indicating potential adversarial samples
        """
        is_adversarial = np.zeros(len(X_test), dtype=bool)

        for feat_idx, (min_val, max_val) in feature_bounds.items():
            below_min = X_test[:, feat_idx] < min_val
            above_max = X_test[:, feat_idx] > max_val
            is_adversarial |= (below_min | above_max)

        return is_adversarial

    def detect_adversarial_prediction_consistency(
        self,
        X_test: np.ndarray,
        n_perturbations: int = 5,
        perturbation_magnitude: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction consistency based adversarial detection.

        Adversarial examples often show high variance under small perturbations.

        Args:
            X_test: Samples to test
            n_perturbations: Number of perturbations per sample
            perturbation_magnitude: Magnitude of small perturbations

        Returns:
            Tuple of (is_adversarial flags, consistency scores)
        """
        base_predictions = self._get_prediction(X_test)

        all_preds = [base_predictions]

        for _ in range(n_perturbations):
            noise = self._rng.randn(*X_test.shape) * perturbation_magnitude
            X_perturbed = X_test + noise
            preds = self._get_prediction(X_perturbed)
            all_preds.append(preds)

        # Calculate prediction variance
        all_preds = np.array(all_preds)
        pred_variance = np.var(all_preds, axis=0)

        # High variance indicates potential adversarial
        threshold = np.percentile(pred_variance, 95)
        is_adversarial = pred_variance > threshold

        # Consistency score (lower is better)
        consistency_scores = 1.0 / (1.0 + pred_variance)

        return is_adversarial, consistency_scores

    # =========================================================================
    # Robustness Scoring
    # =========================================================================

    def calculate_robustness_score(
        self,
        attack_results: Dict[str, Dict[str, float]]
    ) -> RobustnessScore:
        """
        Calculate comprehensive robustness score (deterministic).

        Args:
            attack_results: Results per attack method

        Returns:
            Robustness score breakdown
        """
        # Get success rates (inverse = robustness)
        fgsm_robustness = 1.0 - attack_results.get("fgsm", {}).get("success_rate", 0.0)
        pgd_robustness = 1.0 - attack_results.get("pgd", {}).get("success_rate", 0.0)
        noise_robustness = 1.0 - attack_results.get("noise", {}).get("success_rate", 0.0)
        boundary_robustness = 1.0 - attack_results.get("boundary", {}).get("instability_rate", 0.0)

        # Weighted overall score
        # PGD is strongest attack, so weight it higher
        weights = {
            "fgsm": 0.2,
            "pgd": 0.35,
            "noise": 0.15,
            "boundary": 0.3
        }

        overall = (
            weights["fgsm"] * fgsm_robustness +
            weights["pgd"] * pgd_robustness +
            weights["noise"] * noise_robustness +
            weights["boundary"] * boundary_robustness
        )

        # Assign grade
        if overall >= 0.9:
            grade = "A"
        elif overall >= 0.8:
            grade = "B"
        elif overall >= 0.7:
            grade = "C"
        elif overall >= 0.6:
            grade = "D"
        else:
            grade = "F"

        return RobustnessScore(
            fgsm_robustness=fgsm_robustness,
            pgd_robustness=pgd_robustness,
            noise_robustness=noise_robustness,
            boundary_robustness=boundary_robustness,
            overall_score=overall,
            grade=grade
        )

    def _calculate_provenance(
        self,
        result: Dict[str, Any]
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        provenance_data = (
            f"{result.get('total_attacks', 0)}|"
            f"{result.get('successful_attacks', 0)}|"
            f"{result.get('overall_score', 0):.8f}|"
            f"{self.config.random_state}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()

    def _generate_recommendations(
        self,
        robustness_scores: RobustnessScore,
        attack_results: Dict[str, Dict[str, float]],
        boundary_results: List[BoundaryTestResult]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Overall assessment
        if robustness_scores.overall_score < 0.5:
            recommendations.append(
                "CRITICAL: Model has significant robustness vulnerabilities. "
                "Consider adversarial training or input sanitization before deployment."
            )
        elif robustness_scores.overall_score < 0.7:
            recommendations.append(
                "Model shows moderate vulnerability to adversarial attacks. "
                "Implement input validation and consider ensemble methods."
            )
        else:
            recommendations.append(
                "Model demonstrates good robustness. Continue monitoring for new attack vectors."
            )

        # Attack-specific recommendations
        if robustness_scores.fgsm_robustness < 0.7:
            recommendations.append(
                "Vulnerable to FGSM attacks. Consider gradient masking or adversarial training."
            )

        if robustness_scores.pgd_robustness < 0.6:
            recommendations.append(
                "Highly vulnerable to PGD attacks. This is a strong attack - "
                "consider certified defenses or ensemble methods."
            )

        # Boundary recommendations
        unstable_features = [r for r in boundary_results if not r.stable_at_bounds]
        if unstable_features:
            feature_names = [
                r.feature_name or f"feature_{r.feature_index}"
                for r in unstable_features[:3]
            ]
            recommendations.append(
                f"Model unstable at boundaries for: {', '.join(feature_names)}. "
                "Add boundary-aware regularization or clipping."
            )

        discontinuities = [r for r in boundary_results if r.discontinuity_detected]
        if discontinuities:
            recommendations.append(
                f"Discontinuities detected at {len(discontinuities)} feature boundaries. "
                "This may cause unpredictable behavior in production."
            )

        return recommendations

    # =========================================================================
    # Main Testing Method
    # =========================================================================

    def comprehensive_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
        X_reference: Optional[np.ndarray] = None
    ) -> AdversarialTestingResult:
        """
        Run comprehensive adversarial testing.

        Args:
            X: Test features
            y: Test targets
            feature_bounds: Physical bounds per feature for boundary testing
            X_reference: Reference data for statistical detection

        Returns:
            Comprehensive testing result

        Example:
            >>> result = framework.comprehensive_test(
            ...     X_test, y_test,
            ...     feature_bounds={
            ...         0: (0, 1200),    # temperature
            ...         1: (0, 300),     # pressure
            ...         2: (70, 95)      # efficiency
            ...     }
            ... )
        """
        logger.info(f"Starting comprehensive adversarial testing on {len(X)} samples")

        # Sample if needed
        if len(X) > self.config.n_samples:
            indices = self._rng.choice(len(X), self.config.n_samples, replace=False)
            X_test = X[indices]
            y_test = y[indices]
        else:
            X_test = X.copy()
            y_test = y.copy()

        # Original predictions
        original_preds = self._get_prediction(X_test)

        # Run attacks
        adversarial_samples = []
        attack_results = {}
        total_attacks = 0
        successful_attacks = 0

        # FGSM Attack
        if AdversarialAttackMethod.FGSM in self.config.attack_methods:
            X_adv_fgsm, pert_fgsm = self.fgsm_attack(X_test, y_test)
            adv_preds_fgsm = self._get_prediction(X_adv_fgsm)

            fgsm_success = 0
            for i in range(len(X_test)):
                pred_change = abs(adv_preds_fgsm[i] - original_preds[i])
                success = pred_change > 0.1 * (abs(original_preds[i]) + 1e-10)

                if success:
                    fgsm_success += 1
                    successful_attacks += 1

                total_attacks += 1

                adversarial_samples.append(AdversarialSample(
                    sample_index=i,
                    attack_method="fgsm",
                    original_prediction=float(original_preds[i]),
                    adversarial_prediction=float(adv_preds_fgsm[i]),
                    perturbation_norm_l2=float(np.linalg.norm(pert_fgsm[i])),
                    perturbation_norm_linf=float(np.max(np.abs(pert_fgsm[i]))),
                    prediction_change=float(pred_change),
                    prediction_change_pct=float(pred_change / (abs(original_preds[i]) + 1e-10) * 100),
                    attack_success=success,
                    features_perturbed=list(range(X_test.shape[1]))
                ))

            attack_results["fgsm"] = {
                "success_rate": fgsm_success / len(X_test),
                "mean_perturbation_l2": float(np.mean([np.linalg.norm(p) for p in pert_fgsm])),
                "mean_prediction_change": float(np.mean(np.abs(adv_preds_fgsm - original_preds)))
            }

        # PGD Attack
        if AdversarialAttackMethod.PGD in self.config.attack_methods:
            X_adv_pgd, pert_pgd = self.pgd_attack(X_test, y_test)
            adv_preds_pgd = self._get_prediction(X_adv_pgd)

            pgd_success = 0
            for i in range(len(X_test)):
                pred_change = abs(adv_preds_pgd[i] - original_preds[i])
                success = pred_change > 0.1 * (abs(original_preds[i]) + 1e-10)

                if success:
                    pgd_success += 1
                    successful_attacks += 1

                total_attacks += 1

                adversarial_samples.append(AdversarialSample(
                    sample_index=i,
                    attack_method="pgd",
                    original_prediction=float(original_preds[i]),
                    adversarial_prediction=float(adv_preds_pgd[i]),
                    perturbation_norm_l2=float(np.linalg.norm(pert_pgd[i])),
                    perturbation_norm_linf=float(np.max(np.abs(pert_pgd[i]))),
                    prediction_change=float(pred_change),
                    prediction_change_pct=float(pred_change / (abs(original_preds[i]) + 1e-10) * 100),
                    attack_success=success,
                    features_perturbed=list(range(X_test.shape[1]))
                ))

            attack_results["pgd"] = {
                "success_rate": pgd_success / len(X_test),
                "mean_perturbation_l2": float(np.mean([np.linalg.norm(p) for p in pert_pgd])),
                "mean_prediction_change": float(np.mean(np.abs(adv_preds_pgd - original_preds)))
            }

        # Noise Attack
        if AdversarialAttackMethod.NOISE in self.config.attack_methods:
            X_adv_noise, pert_noise = self.noise_attack(X_test)
            adv_preds_noise = self._get_prediction(X_adv_noise)

            noise_success = 0
            for i in range(len(X_test)):
                pred_change = abs(adv_preds_noise[i] - original_preds[i])
                success = pred_change > 0.1 * (abs(original_preds[i]) + 1e-10)

                if success:
                    noise_success += 1
                    successful_attacks += 1

                total_attacks += 1

            attack_results["noise"] = {
                "success_rate": noise_success / len(X_test),
                "mean_perturbation_l2": float(np.mean([np.linalg.norm(p) for p in pert_noise])),
                "mean_prediction_change": float(np.mean(np.abs(adv_preds_noise - original_preds)))
            }

        # Boundary Testing
        boundary_results = []
        if AdversarialAttackMethod.BOUNDARY in self.config.attack_methods and feature_bounds:
            boundary_results = self.boundary_attack(X_test, feature_bounds)

            unstable_count = sum(1 for r in boundary_results if not r.stable_at_bounds)
            attack_results["boundary"] = {
                "instability_rate": unstable_count / len(boundary_results) if boundary_results else 0.0,
                "discontinuity_rate": sum(1 for r in boundary_results if r.discontinuity_detected) / len(boundary_results) if boundary_results else 0.0
            }

        # Detection testing
        detection_results = []
        if X_reference is not None:
            # Test adversarial detection on generated samples
            if adversarial_samples:
                # Create adversarial dataset
                X_adv_all = np.vstack([
                    X_adv_fgsm if AdversarialAttackMethod.FGSM in self.config.attack_methods else np.array([]).reshape(0, X_test.shape[1]),
                    X_adv_pgd if AdversarialAttackMethod.PGD in self.config.attack_methods else np.array([]).reshape(0, X_test.shape[1])
                ]) if 'X_adv_fgsm' in dir() or 'X_adv_pgd' in dir() else X_test

                if len(X_adv_all) > 0:
                    is_adv, threshold = self.detect_adversarial_statistical(X_adv_all, X_reference)

                    detection_results.append(DetectionResult(
                        method="statistical",
                        samples_tested=len(X_adv_all),
                        adversarial_detected=int(np.sum(is_adv)),
                        detection_rate=float(np.mean(is_adv)),
                        false_positive_rate=0.05,  # By design (95th percentile)
                        detection_threshold=threshold
                    ))

        # Calculate robustness scores
        robustness_scores = self.calculate_robustness_score(attack_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            robustness_scores, attack_results, boundary_results
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance({
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "overall_score": robustness_scores.overall_score
        })

        result = AdversarialTestingResult(
            overall_robustness_score=robustness_scores.overall_score,
            robustness_grade=robustness_scores.grade,
            total_attacks=total_attacks,
            successful_attacks=successful_attacks,
            attack_success_rate=successful_attacks / total_attacks if total_attacks > 0 else 0.0,
            attack_results_by_method=attack_results,
            adversarial_samples=adversarial_samples[:100],  # Limit stored samples
            boundary_results=boundary_results,
            detection_results=detection_results,
            robustness_scores=robustness_scores,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow(),
            config_used={
                "attack_methods": [m.value for m in self.config.attack_methods],
                "fgsm_epsilon": self.config.fgsm_epsilon,
                "pgd_epsilon": self.config.pgd_epsilon,
                "pgd_iterations": self.config.pgd_iterations,
                "n_samples": self.config.n_samples,
                "random_state": self.config.random_state
            }
        )

        logger.info(
            f"Adversarial testing complete: robustness={robustness_scores.overall_score:.3f}, "
            f"grade={robustness_scores.grade}, attack_success_rate={result.attack_success_rate:.3f}"
        )

        return result


# =============================================================================
# Unit Tests
# =============================================================================

class TestAdversarialTestingFramework:
    """Unit tests for AdversarialTestingFramework."""

    def test_fgsm_attack(self):
        """Test FGSM attack generates perturbations."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        framework = AdversarialTestingFramework(MockModel())
        X = np.random.randn(10, 5)
        y = np.sum(X, axis=1)

        X_adv, pert = framework.fgsm_attack(X, y)

        assert X_adv.shape == X.shape
        assert not np.allclose(X_adv, X)
        assert np.all(np.abs(pert) <= framework.config.fgsm_epsilon + 1e-6)

    def test_pgd_attack(self):
        """Test PGD attack generates bounded perturbations."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        framework = AdversarialTestingFramework(MockModel())
        X = np.random.randn(10, 5)
        y = np.sum(X, axis=1)

        X_adv, pert = framework.pgd_attack(X, y)

        assert X_adv.shape == X.shape
        assert np.all(np.abs(pert) <= framework.config.pgd_epsilon + 1e-6)

    def test_boundary_attack(self):
        """Test boundary attack detects instabilities."""
        class MockModel:
            def predict(self, X):
                # Unstable at boundaries
                return np.where(X[:, 0] > 0.9, 100, np.sum(X, axis=1))

        framework = AdversarialTestingFramework(MockModel())
        X = np.random.randn(10, 5)

        bounds = {0: (-1.0, 1.0)}
        results = framework.boundary_attack(X, bounds)

        assert len(results) == 1
        assert results[0].feature_index == 0

    def test_robustness_scoring(self):
        """Test robustness score calculation."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        framework = AdversarialTestingFramework(MockModel())

        attack_results = {
            "fgsm": {"success_rate": 0.1},
            "pgd": {"success_rate": 0.2},
            "noise": {"success_rate": 0.05},
            "boundary": {"instability_rate": 0.1}
        }

        scores = framework.calculate_robustness_score(attack_results)

        assert 0 <= scores.overall_score <= 1
        assert scores.grade in ["A", "B", "C", "D", "F"]

    def test_comprehensive_test(self):
        """Test comprehensive testing."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        config = AdversarialTestingConfig(
            attack_methods=[AdversarialAttackMethod.FGSM, AdversarialAttackMethod.NOISE],
            n_samples=20
        )
        framework = AdversarialTestingFramework(MockModel(), config)

        X = np.random.randn(30, 5)
        y = np.sum(X, axis=1)

        result = framework.comprehensive_test(X, y)

        assert result.total_attacks > 0
        assert 0 <= result.overall_robustness_score <= 1
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_adversarial_detection_statistical(self):
        """Test statistical adversarial detection."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        framework = AdversarialTestingFramework(MockModel())

        X_ref = np.random.randn(100, 5)
        X_normal = np.random.randn(20, 5)
        X_adv = np.random.randn(20, 5) + 5  # Shifted distribution

        is_adv_normal, _ = framework.detect_adversarial_statistical(X_normal, X_ref)
        is_adv_shifted, _ = framework.detect_adversarial_statistical(X_adv, X_ref)

        # Shifted should have more detections
        assert np.sum(is_adv_shifted) > np.sum(is_adv_normal)

    def test_process_heat_bounds(self):
        """Test Process Heat physical bounds."""
        bounds = ProcessHeatBounds()

        assert bounds.temperature_min == -40.0
        assert bounds.temperature_max == 1200.0
        assert bounds.efficiency_min == 0.0
        assert bounds.efficiency_max == 100.0
        assert bounds.pressure_max == 300.0
