# -*- coding: utf-8 -*-
"""
Adversarial Tester Module

This module provides adversarial testing capabilities for GreenLang ML models,
identifying vulnerabilities to adversarial inputs and measuring model robustness.

Adversarial testing is critical for regulatory compliance to ensure models
behave reliably under edge cases and potential manipulation attempts.

Example:
    >>> from greenlang.ml.robustness import AdversarialTester
    >>> tester = AdversarialTester(model)
    >>> result = tester.test_robustness(X_test, y_test)
    >>> print(f"Robustness score: {result.robustness_score:.2f}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AttackMethod(str, Enum):
    """Adversarial attack methods."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"  # Projected Gradient Descent
    NOISE = "noise"  # Random noise
    BOUNDARY = "boundary"  # Boundary attack
    FEATURE_PERTURBATION = "feature_perturbation"


class AdversarialTesterConfig(BaseModel):
    """Configuration for adversarial tester."""

    attack_methods: List[AttackMethod] = Field(
        default_factory=lambda: [
            AttackMethod.FGSM,
            AttackMethod.NOISE,
            AttackMethod.FEATURE_PERTURBATION
        ],
        description="Attack methods to use"
    )
    epsilon: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Perturbation magnitude"
    )
    n_iterations: int = Field(
        default=10,
        ge=1,
        description="Iterations for iterative attacks"
    )
    alpha: float = Field(
        default=0.01,
        gt=0,
        description="Step size for iterative attacks"
    )
    targeted: bool = Field(
        default=False,
        description="Use targeted attacks"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        description="Number of samples to test"
    )
    feature_importance_threshold: float = Field(
        default=0.1,
        ge=0,
        description="Threshold for important features"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class AdversarialExample(BaseModel):
    """Information about an adversarial example."""

    original_prediction: float = Field(
        ...,
        description="Original prediction"
    )
    adversarial_prediction: float = Field(
        ...,
        description="Prediction on adversarial input"
    )
    perturbation_norm: float = Field(
        ...,
        description="L2 norm of perturbation"
    )
    attack_method: str = Field(
        ...,
        description="Attack method used"
    )
    success: bool = Field(
        ...,
        description="Whether attack succeeded"
    )
    features_perturbed: List[int] = Field(
        default_factory=list,
        description="Indices of perturbed features"
    )


class VulnerabilityAssessment(BaseModel):
    """Assessment of model vulnerabilities."""

    feature_index: int = Field(
        ...,
        description="Feature index"
    )
    feature_name: Optional[str] = Field(
        default=None,
        description="Feature name"
    )
    vulnerability_score: float = Field(
        ...,
        description="Vulnerability score (0-1)"
    )
    perturbation_sensitivity: float = Field(
        ...,
        description="Sensitivity to perturbation"
    )
    recommended_bounds: Tuple[float, float] = Field(
        ...,
        description="Recommended safe bounds"
    )


class RobustnessResult(BaseModel):
    """Result from robustness testing."""

    robustness_score: float = Field(
        ...,
        description="Overall robustness score (0-1)"
    )
    attack_success_rate: Dict[str, float] = Field(
        ...,
        description="Success rate per attack method"
    )
    adversarial_examples: List[AdversarialExample] = Field(
        ...,
        description="Generated adversarial examples"
    )
    vulnerabilities: List[VulnerabilityAssessment] = Field(
        ...,
        description="Identified vulnerabilities"
    )
    mean_perturbation: float = Field(
        ...,
        description="Mean perturbation for successful attacks"
    )
    certified_radius: Optional[float] = Field(
        default=None,
        description="Certified robustness radius"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test timestamp"
    )


class AdversarialTester:
    """
    Adversarial Tester for GreenLang ML models.

    This class provides adversarial testing capabilities to identify
    vulnerabilities and measure model robustness against various
    types of adversarial attacks.

    Key capabilities:
    - FGSM attacks
    - PGD attacks
    - Feature perturbation
    - Vulnerability assessment
    - Robustness scoring

    Attributes:
        model: Model to test
        config: Tester configuration
        _gradient_fn: Optional gradient function

    Example:
        >>> tester = AdversarialTester(
        ...     model,
        ...     config=AdversarialTesterConfig(
        ...         epsilon=0.1,
        ...         attack_methods=[AttackMethod.FGSM, AttackMethod.PGD]
        ...     )
        ... )
        >>> result = tester.test_robustness(X_test, y_test)
        >>> print(f"Robustness: {result.robustness_score:.2f}")
        >>> for vuln in result.vulnerabilities[:3]:
        ...     print(f"Feature {vuln.feature_index}: score={vuln.vulnerability_score:.3f}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[AdversarialTesterConfig] = None,
        gradient_fn: Optional[Callable] = None
    ):
        """
        Initialize adversarial tester.

        Args:
            model: Model to test
            config: Tester configuration
            gradient_fn: Optional gradient function for gradient-based attacks
        """
        self.model = model
        self.config = config or AdversarialTesterConfig()
        self._gradient_fn = gradient_fn

        np.random.seed(self.config.random_state)

        logger.info(
            f"AdversarialTester initialized: attacks={[m.value for m in self.config.attack_methods]}"
        )

    def _get_prediction(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            return self.model(X)

    def _estimate_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        delta: float = 1e-4
    ) -> np.ndarray:
        """
        Estimate gradient using finite differences.

        Args:
            X: Input features
            y: Target values
            delta: Step size for finite differences

        Returns:
            Estimated gradient
        """
        if self._gradient_fn is not None:
            return self._gradient_fn(X, y)

        gradients = np.zeros_like(X)
        base_pred = self._get_prediction(X)

        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += delta

            pred_plus = self._get_prediction(X_plus)
            gradients[:, i] = (pred_plus - base_pred).flatten() / delta

        return gradients

    def _fgsm_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float
    ) -> np.ndarray:
        """
        Fast Gradient Sign Method attack.

        Args:
            X: Original inputs
            y: Target values
            epsilon: Perturbation magnitude

        Returns:
            Adversarial examples
        """
        gradients = self._estimate_gradient(X, y)
        perturbation = epsilon * np.sign(gradients)

        return X + perturbation

    def _pgd_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        n_iterations: int,
        alpha: float
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack.

        Args:
            X: Original inputs
            y: Target values
            epsilon: Maximum perturbation
            n_iterations: Number of iterations
            alpha: Step size

        Returns:
            Adversarial examples
        """
        X_adv = X.copy()

        for _ in range(n_iterations):
            gradients = self._estimate_gradient(X_adv, y)
            X_adv = X_adv + alpha * np.sign(gradients)

            # Project back to epsilon ball
            perturbation = X_adv - X
            perturbation = np.clip(perturbation, -epsilon, epsilon)
            X_adv = X + perturbation

        return X_adv

    def _noise_attack(
        self,
        X: np.ndarray,
        epsilon: float
    ) -> np.ndarray:
        """
        Random noise attack.

        Args:
            X: Original inputs
            epsilon: Noise magnitude

        Returns:
            Adversarial examples
        """
        noise = np.random.uniform(-epsilon, epsilon, X.shape)
        return X + noise

    def _feature_perturbation_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        top_k: int = 3
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Perturb most sensitive features.

        Args:
            X: Original inputs
            y: Target values
            epsilon: Perturbation magnitude
            top_k: Number of features to perturb

        Returns:
            Tuple of (adversarial examples, perturbed feature indices)
        """
        # Estimate feature importance via gradient
        gradients = np.abs(self._estimate_gradient(X, y))
        mean_importance = np.mean(gradients, axis=0)

        # Select top-k features
        top_features = np.argsort(mean_importance)[-top_k:]

        X_adv = X.copy()
        for feat_idx in top_features:
            grad_sign = np.sign(np.mean(self._estimate_gradient(X, y)[:, feat_idx]))
            X_adv[:, feat_idx] += epsilon * grad_sign

        return X_adv, list(top_features)

    def _calculate_provenance(
        self,
        robustness_score: float,
        n_attacks: int
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{robustness_score:.8f}|{n_attacks}|"
            f"{self.config.epsilon}|{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def test_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> RobustnessResult:
        """
        Test model robustness against adversarial attacks.

        Args:
            X: Test features
            y: Test labels
            feature_names: Optional feature names

        Returns:
            RobustnessResult with detailed analysis

        Example:
            >>> result = tester.test_robustness(X_test, y_test)
            >>> if result.robustness_score < 0.7:
            ...     print("Model is vulnerable!")
        """
        logger.info(f"Testing robustness on {len(X)} samples")

        # Sample if needed
        if len(X) > self.config.n_samples:
            indices = np.random.choice(len(X), self.config.n_samples, replace=False)
            X_test = X[indices]
            y_test = y[indices]
        else:
            X_test = X
            y_test = y

        original_preds = self._get_prediction(X_test)
        adversarial_examples = []
        attack_success_rates = {}

        # Run each attack method
        for method in self.config.attack_methods:
            logger.info(f"Running {method.value} attack")

            if method == AttackMethod.FGSM:
                X_adv = self._fgsm_attack(X_test, y_test, self.config.epsilon)
                perturbed_features = list(range(X_test.shape[1]))

            elif method == AttackMethod.PGD:
                X_adv = self._pgd_attack(
                    X_test, y_test,
                    self.config.epsilon,
                    self.config.n_iterations,
                    self.config.alpha
                )
                perturbed_features = list(range(X_test.shape[1]))

            elif method == AttackMethod.NOISE:
                X_adv = self._noise_attack(X_test, self.config.epsilon)
                perturbed_features = list(range(X_test.shape[1]))

            elif method == AttackMethod.FEATURE_PERTURBATION:
                X_adv, perturbed_features = self._feature_perturbation_attack(
                    X_test, y_test, self.config.epsilon
                )

            else:
                continue

            # Evaluate attack success
            adv_preds = self._get_prediction(X_adv)

            success_count = 0
            for i in range(len(X_test)):
                perturbation_norm = float(np.linalg.norm(X_adv[i] - X_test[i]))

                # For regression: significant change in prediction
                pred_change = abs(adv_preds[i] - original_preds[i])
                if isinstance(pred_change, np.ndarray):
                    pred_change = pred_change[0]

                success = pred_change > 0.1 * abs(original_preds[i] + 1e-10)

                if success:
                    success_count += 1

                adversarial_examples.append(AdversarialExample(
                    original_prediction=float(original_preds[i]),
                    adversarial_prediction=float(adv_preds[i]),
                    perturbation_norm=perturbation_norm,
                    attack_method=method.value,
                    success=success,
                    features_perturbed=perturbed_features
                ))

            attack_success_rates[method.value] = success_count / len(X_test)

        # Calculate robustness score (1 - average success rate)
        avg_success_rate = np.mean(list(attack_success_rates.values()))
        robustness_score = 1 - avg_success_rate

        # Assess vulnerabilities per feature
        vulnerabilities = self._assess_vulnerabilities(
            X_test, y_test, feature_names
        )

        # Calculate mean perturbation for successful attacks
        successful_attacks = [e for e in adversarial_examples if e.success]
        mean_perturbation = (
            np.mean([e.perturbation_norm for e in successful_attacks])
            if successful_attacks else 0.0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            robustness_score, vulnerabilities, attack_success_rates
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            robustness_score, len(adversarial_examples)
        )

        logger.info(
            f"Robustness testing complete: score={robustness_score:.3f}, "
            f"avg_attack_success={avg_success_rate:.3f}"
        )

        return RobustnessResult(
            robustness_score=robustness_score,
            attack_success_rate=attack_success_rates,
            adversarial_examples=adversarial_examples,
            vulnerabilities=vulnerabilities,
            mean_perturbation=mean_perturbation,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def _assess_vulnerabilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[VulnerabilityAssessment]:
        """Assess per-feature vulnerabilities."""
        vulnerabilities = []

        gradients = np.abs(self._estimate_gradient(X, y))
        mean_sensitivity = np.mean(gradients, axis=0)

        # Normalize
        max_sensitivity = np.max(mean_sensitivity)
        if max_sensitivity > 0:
            normalized_sensitivity = mean_sensitivity / max_sensitivity
        else:
            normalized_sensitivity = mean_sensitivity

        for i in range(X.shape[1]):
            # Recommended bounds based on data distribution
            feat_std = np.std(X[:, i])
            feat_mean = np.mean(X[:, i])
            recommended_bounds = (
                float(feat_mean - 2 * feat_std),
                float(feat_mean + 2 * feat_std)
            )

            vulnerabilities.append(VulnerabilityAssessment(
                feature_index=i,
                feature_name=feature_names[i] if feature_names else None,
                vulnerability_score=float(normalized_sensitivity[i]),
                perturbation_sensitivity=float(mean_sensitivity[i]),
                recommended_bounds=recommended_bounds
            ))

        # Sort by vulnerability score
        vulnerabilities.sort(key=lambda v: v.vulnerability_score, reverse=True)

        return vulnerabilities

    def _generate_recommendations(
        self,
        robustness_score: float,
        vulnerabilities: List[VulnerabilityAssessment],
        attack_success_rates: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for improving robustness."""
        recommendations = []

        if robustness_score < 0.5:
            recommendations.append(
                "CRITICAL: Model shows significant vulnerability to adversarial attacks. "
                "Consider adversarial training or input sanitization."
            )
        elif robustness_score < 0.7:
            recommendations.append(
                "Model shows moderate vulnerability. Consider implementing input validation."
            )
        else:
            recommendations.append(
                "Model shows good robustness. Continue monitoring for new attack vectors."
            )

        # Feature-specific recommendations
        high_vuln_features = [v for v in vulnerabilities if v.vulnerability_score > 0.5]
        if high_vuln_features:
            feature_names = [
                v.feature_name or f"feature_{v.feature_index}"
                for v in high_vuln_features[:3]
            ]
            recommendations.append(
                f"High vulnerability features: {', '.join(feature_names)}. "
                "Consider feature scaling or bounds checking."
            )

        # Attack-specific recommendations
        for method, rate in attack_success_rates.items():
            if rate > 0.5:
                recommendations.append(
                    f"Model vulnerable to {method} attacks (success rate: {rate:.1%}). "
                    f"Specific defenses recommended."
                )

        return recommendations


# Unit test stubs
class TestAdversarialTester:
    """Unit tests for AdversarialTester."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        tester = AdversarialTester(MockModel())
        assert AttackMethod.FGSM in tester.config.attack_methods

    def test_fgsm_attack(self):
        """Test FGSM attack."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        tester = AdversarialTester(MockModel())

        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([6.0])

        X_adv = tester._fgsm_attack(X, y, epsilon=0.1)
        assert X_adv.shape == X.shape
        assert not np.allclose(X_adv, X)

    def test_noise_attack(self):
        """Test noise attack."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        tester = AdversarialTester(MockModel())

        X = np.array([[1.0, 2.0, 3.0]])
        X_adv = tester._noise_attack(X, epsilon=0.1)

        assert X_adv.shape == X.shape
        perturbation = np.abs(X_adv - X)
        assert np.all(perturbation <= 0.1)

    def test_robustness_score(self):
        """Test robustness scoring."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        config = AdversarialTesterConfig(
            attack_methods=[AttackMethod.NOISE],
            n_samples=10
        )
        tester = AdversarialTester(MockModel(), config)

        X = np.random.randn(20, 5)
        y = np.sum(X, axis=1)

        result = tester.test_robustness(X, y)

        assert 0 <= result.robustness_score <= 1
        assert len(result.vulnerabilities) == 5
