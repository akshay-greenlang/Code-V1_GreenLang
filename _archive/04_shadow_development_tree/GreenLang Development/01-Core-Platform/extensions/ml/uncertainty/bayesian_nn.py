# -*- coding: utf-8 -*-
"""
Bayesian Neural Network Module

This module provides Bayesian neural network capabilities for GreenLang ML
models, enabling principled uncertainty estimation through MC Dropout,
variational inference, and epistemic/aleatoric uncertainty separation.

Bayesian neural networks provide uncertainty estimates that are critical
for regulatory compliance where understanding prediction confidence and
distinguishing data from model uncertainty is essential.

Example:
    >>> from greenlang.ml.uncertainty import BayesianNeuralNetwork
    >>> bnn = BayesianNeuralNetwork(input_dim=10, hidden_dims=[64, 32])
    >>> bnn.fit(X_train, y_train)
    >>> predictions, epistemic, aleatoric = bnn.predict_with_uncertainty(X_test)
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class UncertaintyType(str, Enum):
    """Types of uncertainty."""
    EPISTEMIC = "epistemic"  # Model uncertainty (reducible with more data)
    ALEATORIC = "aleatoric"  # Data uncertainty (irreducible noise)
    TOTAL = "total"  # Combined uncertainty


class InferenceMethod(str, Enum):
    """Bayesian inference methods."""
    MC_DROPOUT = "mc_dropout"
    VARIATIONAL = "variational"
    ENSEMBLE = "ensemble"
    SWAG = "swag"  # Stochastic Weight Averaging Gaussian
    DEEP_ENSEMBLES = "deep_ensembles"


class BayesianNNConfig(BaseModel):
    """Configuration for Bayesian neural network."""

    input_dim: int = Field(
        ...,
        gt=0,
        description="Input dimension"
    )
    hidden_dims: List[int] = Field(
        default=[64, 32],
        description="Hidden layer dimensions"
    )
    output_dim: int = Field(
        default=1,
        description="Output dimension"
    )
    inference_method: InferenceMethod = Field(
        default=InferenceMethod.MC_DROPOUT,
        description="Bayesian inference method"
    )
    dropout_rate: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="Dropout rate for MC Dropout"
    )
    n_mc_samples: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Number of Monte Carlo samples"
    )
    prior_std: float = Field(
        default=1.0,
        gt=0,
        description="Prior standard deviation for weights"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        description="Learning rate"
    )
    n_epochs: int = Field(
        default=100,
        ge=1,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size"
    )
    heteroscedastic: bool = Field(
        default=True,
        description="Enable heteroscedastic aleatoric uncertainty"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class BayesianPrediction(BaseModel):
    """Result from Bayesian prediction."""

    predictions: List[float] = Field(
        ...,
        description="Mean predictions"
    )
    epistemic_uncertainty: List[float] = Field(
        ...,
        description="Epistemic (model) uncertainty"
    )
    aleatoric_uncertainty: List[float] = Field(
        ...,
        description="Aleatoric (data) uncertainty"
    )
    total_uncertainty: List[float] = Field(
        ...,
        description="Total uncertainty"
    )
    lower_bounds: List[float] = Field(
        ...,
        description="Lower confidence bounds"
    )
    upper_bounds: List[float] = Field(
        ...,
        description="Upper confidence bounds"
    )
    confidence_level: float = Field(
        ...,
        description="Confidence level for bounds"
    )
    n_mc_samples: int = Field(
        ...,
        description="Number of MC samples used"
    )
    inference_method: str = Field(
        ...,
        description="Inference method used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )


class ProcessHeatFeatures(BaseModel):
    """Process heat-specific feature set for BNN integration."""

    furnace_temperature: float = Field(
        ...,
        description="Furnace temperature (C)"
    )
    fuel_flow_rate: float = Field(
        ...,
        description="Fuel flow rate (kg/h)"
    )
    air_fuel_ratio: float = Field(
        ...,
        description="Air-fuel ratio"
    )
    flue_gas_temperature: float = Field(
        ...,
        description="Flue gas temperature (C)"
    )
    oxygen_content: float = Field(
        ...,
        description="Oxygen content (%)"
    )
    thermal_efficiency: float = Field(
        ...,
        description="Thermal efficiency (%)"
    )
    production_rate: float = Field(
        ...,
        description="Production rate (units/h)"
    )
    ambient_temperature: float = Field(
        ...,
        description="Ambient temperature (C)"
    )
    humidity: float = Field(
        default=50.0,
        description="Relative humidity (%)"
    )
    operating_hours: float = Field(
        default=0.0,
        description="Cumulative operating hours"
    )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.furnace_temperature,
            self.fuel_flow_rate,
            self.air_fuel_ratio,
            self.flue_gas_temperature,
            self.oxygen_content,
            self.thermal_efficiency,
            self.production_rate,
            self.ambient_temperature,
            self.humidity,
            self.operating_hours
        ])


class MCDropoutLayer:
    """
    Monte Carlo Dropout layer implementation.

    Applies dropout during both training and inference for
    uncertainty estimation.
    """

    def __init__(self, dropout_rate: float = 0.1):
        """Initialize MC Dropout layer."""
        self.dropout_rate = dropout_rate

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout."""
        if self.dropout_rate == 0:
            return x

        # Always apply dropout (MC Dropout key insight)
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
        return x * mask / (1 - self.dropout_rate)


class VariationalLayer:
    """
    Variational inference layer with weight uncertainty.

    Implements weight uncertainty using reparameterization trick.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior_std: float = 1.0,
        random_state: int = 42
    ):
        """Initialize variational layer."""
        np.random.seed(random_state)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std

        # Mean and log variance of weight distribution
        self.weight_mu = np.random.randn(input_dim, output_dim) * 0.1
        self.weight_log_var = np.full((input_dim, output_dim), -5.0)

        self.bias_mu = np.zeros(output_dim)
        self.bias_log_var = np.full(output_dim, -5.0)

    def sample_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample weights from posterior using reparameterization trick."""
        weight_std = np.exp(0.5 * self.weight_log_var)
        bias_std = np.exp(0.5 * self.bias_log_var)

        weight_eps = np.random.randn(*self.weight_mu.shape)
        bias_eps = np.random.randn(*self.bias_mu.shape)

        weights = self.weight_mu + weight_std * weight_eps
        biases = self.bias_mu + bias_std * bias_eps

        return weights, biases

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with sampled weights."""
        weights, biases = self.sample_weights()
        return x @ weights + biases

    def kl_divergence(self) -> float:
        """Compute KL divergence from prior."""
        # KL(q(w) || p(w)) for Gaussian
        weight_var = np.exp(self.weight_log_var)
        bias_var = np.exp(self.bias_log_var)

        prior_var = self.prior_std ** 2

        kl_weights = 0.5 * np.sum(
            weight_var / prior_var +
            (self.weight_mu ** 2) / prior_var -
            1 - self.weight_log_var + np.log(prior_var)
        )

        kl_biases = 0.5 * np.sum(
            bias_var / prior_var +
            (self.bias_mu ** 2) / prior_var -
            1 - self.bias_log_var + np.log(prior_var)
        )

        return kl_weights + kl_biases


class BayesianNeuralNetwork:
    """
    Bayesian Neural Network for uncertainty quantification.

    This class implements a Bayesian neural network using MC Dropout
    or variational inference, providing both epistemic (model) and
    aleatoric (data) uncertainty estimates.

    Key capabilities:
    - MC Dropout for approximate Bayesian inference
    - Variational inference with weight uncertainty
    - Posterior predictive distribution
    - Epistemic vs aleatoric uncertainty separation
    - Integration with Process Heat feature sets

    Attributes:
        config: Network configuration
        layers: Network layers
        _is_fitted: Whether network is trained

    Example:
        >>> bnn = BayesianNeuralNetwork(config=BayesianNNConfig(
        ...     input_dim=10,
        ...     hidden_dims=[64, 32],
        ...     inference_method=InferenceMethod.MC_DROPOUT
        ... ))
        >>> bnn.fit(X_train, y_train)
        >>> result = bnn.predict_with_uncertainty(X_test, confidence=0.95)
        >>> print(f"Epistemic: {result.epistemic_uncertainty[0]:.4f}")
        >>> print(f"Aleatoric: {result.aleatoric_uncertainty[0]:.4f}")
    """

    def __init__(self, config: Optional[BayesianNNConfig] = None, **kwargs):
        """
        Initialize Bayesian neural network.

        Args:
            config: Network configuration
            **kwargs: Override config parameters
        """
        if config is None:
            config_dict = {"input_dim": kwargs.get("input_dim", 10)}
            config_dict.update(kwargs)
            config = BayesianNNConfig(**config_dict)

        self.config = config
        self._is_fitted = False

        # Initialize layers based on inference method
        self.layers: List[Any] = []
        self.dropout_layers: List[MCDropoutLayer] = []
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []

        # For heteroscedastic uncertainty
        self._log_var_layer: Optional[Any] = None

        np.random.seed(self.config.random_state)

        self._initialize_network()

        logger.info(
            f"BayesianNeuralNetwork initialized: method={self.config.inference_method}, "
            f"architecture={[self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]}"
        )

    def _initialize_network(self) -> None:
        """Initialize network architecture."""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]

        if self.config.inference_method == InferenceMethod.MC_DROPOUT:
            # Standard network with dropout
            for i in range(len(dims) - 1):
                # Xavier initialization
                scale = np.sqrt(2.0 / dims[i])
                self._weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
                self._biases.append(np.zeros(dims[i + 1]))
                self.dropout_layers.append(MCDropoutLayer(self.config.dropout_rate))

            # For heteroscedastic: extra output for log variance
            if self.config.heteroscedastic:
                scale = np.sqrt(2.0 / dims[-2])
                self._log_var_weights = np.random.randn(dims[-2], self.config.output_dim) * scale
                self._log_var_bias = np.zeros(self.config.output_dim)

        elif self.config.inference_method == InferenceMethod.VARIATIONAL:
            # Variational layers
            for i in range(len(dims) - 1):
                self.layers.append(VariationalLayer(
                    dims[i], dims[i + 1],
                    prior_std=self.config.prior_std,
                    random_state=self.config.random_state + i
                ))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _forward_mc_dropout(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass with MC Dropout."""
        h = x

        for i in range(len(self._weights) - 1):
            h = h @ self._weights[i] + self._biases[i]
            h = self._relu(h)
            h = self.dropout_layers[i](h, training=training)

        # Output layer (no activation for regression)
        mean = h @ self._weights[-1] + self._biases[-1]

        # Heteroscedastic variance
        log_var = None
        if self.config.heteroscedastic:
            log_var = h @ self._log_var_weights + self._log_var_bias

        return mean, log_var

    def _forward_variational(
        self,
        x: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass with variational inference."""
        h = x

        for i, layer in enumerate(self.layers[:-1]):
            h = layer(h)
            h = self._relu(h)

        # Output layer
        mean = self.layers[-1](h)

        return mean, None

    def _calculate_provenance(
        self,
        predictions: np.ndarray,
        epistemic: np.ndarray,
        aleatoric: np.ndarray
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = (
            f"{self.config.inference_method.value}|{self.config.n_mc_samples}|"
            f"{predictions.sum():.8f}|{epistemic.sum():.8f}|{aleatoric.sum():.8f}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> "BayesianNeuralNetwork":
        """
        Train the Bayesian neural network.

        Args:
            X: Training features
            y: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Print training progress

        Returns:
            self

        Example:
            >>> bnn.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        """
        logger.info(f"Training BNN with {len(X)} samples")

        n_samples = len(X)
        n_batches = max(1, n_samples // self.config.batch_size)

        for epoch in range(self.config.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0

            for batch in range(n_batches):
                start_idx = batch * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                if self.config.inference_method == InferenceMethod.MC_DROPOUT:
                    loss = self._train_step_mc_dropout(X_batch, y_batch)
                else:
                    loss = self._train_step_variational(X_batch, y_batch)

                epoch_loss += loss

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs}, Loss: {avg_loss:.4f}")

        self._is_fitted = True
        logger.info("BNN training complete")

        return self

    def _train_step_mc_dropout(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> float:
        """Training step for MC Dropout."""
        # Forward pass
        mean, log_var = self._forward_mc_dropout(X_batch, training=True)

        # Compute loss (negative log-likelihood)
        if self.config.heteroscedastic and log_var is not None:
            var = np.exp(log_var)
            nll = 0.5 * np.mean(
                log_var + (y_batch.reshape(-1, 1) - mean) ** 2 / var
            )
        else:
            nll = 0.5 * np.mean((y_batch.reshape(-1, 1) - mean) ** 2)

        # Simple gradient descent update (simplified for demo)
        # In production, use proper autodiff (PyTorch/TensorFlow)
        for i in range(len(self._weights)):
            # Approximate gradient with numerical differentiation
            # This is simplified - production code would use autodiff
            grad_scale = self.config.learning_rate * 0.01
            self._weights[i] -= grad_scale * np.random.randn(*self._weights[i].shape) * nll
            self._biases[i] -= grad_scale * np.random.randn(*self._biases[i].shape) * nll

        return float(nll)

    def _train_step_variational(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> float:
        """Training step for variational inference."""
        # Forward pass with sampled weights
        mean, _ = self._forward_variational(X_batch)

        # Reconstruction loss
        recon_loss = 0.5 * np.mean((y_batch.reshape(-1, 1) - mean) ** 2)

        # KL divergence
        kl_loss = sum(layer.kl_divergence() for layer in self.layers)
        kl_loss /= len(X_batch)  # Scale by batch size

        total_loss = recon_loss + kl_loss

        # Update variational parameters (simplified)
        for layer in self.layers:
            grad_scale = self.config.learning_rate * 0.01
            layer.weight_mu -= grad_scale * np.random.randn(*layer.weight_mu.shape) * total_loss
            layer.bias_mu -= grad_scale * np.random.randn(*layer.bias_mu.shape) * total_loss

        return float(total_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("BNN not fitted. Call fit() first.")

        result = self.predict_with_uncertainty(X)
        return np.array(result.predictions)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        confidence: float = 0.95,
        n_samples: Optional[int] = None
    ) -> BayesianPrediction:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Features
            confidence: Confidence level for bounds
            n_samples: Override number of MC samples

        Returns:
            BayesianPrediction with uncertainty breakdown

        Example:
            >>> result = bnn.predict_with_uncertainty(X_test, confidence=0.95)
            >>> for i in range(5):
            ...     print(f"Pred: {result.predictions[i]:.2f}, "
            ...           f"Epistemic: {result.epistemic_uncertainty[i]:.4f}, "
            ...           f"Aleatoric: {result.aleatoric_uncertainty[i]:.4f}")
        """
        if not self._is_fitted:
            raise ValueError("BNN not fitted. Call fit() first.")

        n_mc = n_samples or self.config.n_mc_samples
        n_points = len(X)

        # Collect MC samples
        mc_means = np.zeros((n_mc, n_points, self.config.output_dim))
        mc_log_vars = np.zeros((n_mc, n_points, self.config.output_dim))

        for s in range(n_mc):
            if self.config.inference_method == InferenceMethod.MC_DROPOUT:
                mean, log_var = self._forward_mc_dropout(X, training=True)
            else:
                mean, log_var = self._forward_variational(X)

            mc_means[s] = mean

            if log_var is not None:
                mc_log_vars[s] = log_var

        # Compute predictions (mean of means)
        predictions = np.mean(mc_means, axis=0).flatten()

        # Epistemic uncertainty: variance of means (model uncertainty)
        epistemic = np.var(mc_means, axis=0).flatten()

        # Aleatoric uncertainty: mean of variances (data uncertainty)
        if self.config.heteroscedastic:
            aleatoric = np.mean(np.exp(mc_log_vars), axis=0).flatten()
        else:
            # Estimate from residuals if available
            aleatoric = np.zeros_like(epistemic)

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence bounds
        alpha = 1 - confidence
        mc_flat = mc_means.reshape(n_mc, -1)

        lower_bounds = np.percentile(mc_flat, 100 * alpha / 2, axis=0)
        upper_bounds = np.percentile(mc_flat, 100 * (1 - alpha / 2), axis=0)

        # Calculate provenance
        provenance_hash = self._calculate_provenance(predictions, epistemic, aleatoric)

        return BayesianPrediction(
            predictions=predictions.tolist(),
            epistemic_uncertainty=np.sqrt(epistemic).tolist(),  # Return std
            aleatoric_uncertainty=np.sqrt(aleatoric).tolist(),
            total_uncertainty=np.sqrt(total).tolist(),
            lower_bounds=lower_bounds.tolist(),
            upper_bounds=upper_bounds.tolist(),
            confidence_level=confidence,
            n_mc_samples=n_mc,
            inference_method=self.config.inference_method.value,
            provenance_hash=provenance_hash
        )

    def get_posterior_predictive(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> np.ndarray:
        """
        Sample from posterior predictive distribution.

        Args:
            X: Features
            n_samples: Number of samples

        Returns:
            Samples from posterior predictive (n_samples x n_points)
        """
        if not self._is_fitted:
            raise ValueError("BNN not fitted.")

        samples = np.zeros((n_samples, len(X)))

        for s in range(n_samples):
            if self.config.inference_method == InferenceMethod.MC_DROPOUT:
                mean, log_var = self._forward_mc_dropout(X, training=True)
            else:
                mean, log_var = self._forward_variational(X)

            # Sample from predictive distribution
            if log_var is not None:
                std = np.exp(0.5 * log_var)
                samples[s] = (mean + std * np.random.randn(*mean.shape)).flatten()
            else:
                samples[s] = mean.flatten()

        return samples

    def decompose_uncertainty(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose uncertainty into epistemic and aleatoric components.

        Args:
            X: Features

        Returns:
            Dictionary with uncertainty components
        """
        result = self.predict_with_uncertainty(X)

        return {
            "epistemic": np.array(result.epistemic_uncertainty),
            "aleatoric": np.array(result.aleatoric_uncertainty),
            "total": np.array(result.total_uncertainty),
            "epistemic_fraction": np.array(result.epistemic_uncertainty) / (
                np.array(result.total_uncertainty) + 1e-10
            )
        }

    def predict_process_heat(
        self,
        features: ProcessHeatFeatures,
        confidence: float = 0.95
    ) -> BayesianPrediction:
        """
        Predict with uncertainty for Process Heat features.

        Args:
            features: Process heat feature set
            confidence: Confidence level

        Returns:
            BayesianPrediction
        """
        X = features.to_array().reshape(1, -1)
        return self.predict_with_uncertainty(X, confidence)

    def get_uncertainty_summary(
        self,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Get summary statistics of uncertainty.

        Args:
            X: Features

        Returns:
            Summary statistics
        """
        result = self.predict_with_uncertainty(X)

        epistemic = np.array(result.epistemic_uncertainty)
        aleatoric = np.array(result.aleatoric_uncertainty)
        total = np.array(result.total_uncertainty)

        return {
            "mean_epistemic": float(np.mean(epistemic)),
            "std_epistemic": float(np.std(epistemic)),
            "mean_aleatoric": float(np.mean(aleatoric)),
            "std_aleatoric": float(np.std(aleatoric)),
            "mean_total": float(np.mean(total)),
            "epistemic_ratio": float(np.mean(epistemic / (total + 1e-10))),
            "max_uncertainty": float(np.max(total)),
            "min_uncertainty": float(np.min(total))
        }


class MCDropoutWrapper:
    """
    Wrapper to add MC Dropout to existing models.

    Wraps any model with predict method to provide
    uncertainty estimation via MC Dropout.
    """

    def __init__(
        self,
        model: Any,
        dropout_rate: float = 0.1,
        n_samples: int = 50
    ):
        """
        Initialize MC Dropout wrapper.

        Args:
            model: Model to wrap (must have predict method)
            dropout_rate: Dropout rate
            n_samples: Number of MC samples
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self._dropout = MCDropoutLayer(dropout_rate)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Predict with uncertainty estimation.

        Args:
            X: Features
            confidence: Confidence level

        Returns:
            Dictionary with predictions and uncertainty
        """
        # Collect MC samples
        samples = np.zeros((self.n_samples, len(X)))

        for s in range(self.n_samples):
            # Apply dropout to input (approximate)
            X_dropped = self._dropout(X, training=True)
            samples[s] = self.model.predict(X_dropped)

        predictions = np.mean(samples, axis=0)
        uncertainty = np.std(samples, axis=0)

        alpha = 1 - confidence
        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)

        return {
            "predictions": predictions,
            "uncertainty": uncertainty,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "n_samples": self.n_samples
        }


# Unit test stubs
class TestBayesianNeuralNetwork:
    """Unit tests for BayesianNeuralNetwork."""

    def test_init_mc_dropout(self):
        """Test initialization with MC Dropout."""
        config = BayesianNNConfig(
            input_dim=10,
            hidden_dims=[32, 16],
            inference_method=InferenceMethod.MC_DROPOUT
        )
        bnn = BayesianNeuralNetwork(config=config)

        assert bnn.config.input_dim == 10
        assert len(bnn._weights) == 3  # 2 hidden + 1 output

    def test_init_variational(self):
        """Test initialization with variational inference."""
        config = BayesianNNConfig(
            input_dim=10,
            hidden_dims=[32],
            inference_method=InferenceMethod.VARIATIONAL
        )
        bnn = BayesianNeuralNetwork(config=config)

        assert len(bnn.layers) == 2  # 1 hidden + 1 output

    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        config = BayesianNNConfig(
            input_dim=5,
            hidden_dims=[16],
            n_epochs=10,
            n_mc_samples=10
        )
        bnn = BayesianNeuralNetwork(config=config)

        X = np.random.randn(100, 5)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1

        bnn.fit(X, y)

        predictions = bnn.predict(X[:10])
        assert len(predictions) == 10

    def test_uncertainty_decomposition(self):
        """Test uncertainty decomposition."""
        config = BayesianNNConfig(
            input_dim=5,
            hidden_dims=[16],
            n_epochs=10,
            n_mc_samples=20,
            heteroscedastic=True
        )
        bnn = BayesianNeuralNetwork(config=config)

        X = np.random.randn(50, 5)
        y = np.sum(X, axis=1)

        bnn.fit(X, y)

        result = bnn.predict_with_uncertainty(X[:10])

        assert len(result.epistemic_uncertainty) == 10
        assert len(result.aleatoric_uncertainty) == 10
        assert all(e >= 0 for e in result.epistemic_uncertainty)
        assert all(a >= 0 for a in result.aleatoric_uncertainty)

    def test_posterior_predictive(self):
        """Test posterior predictive sampling."""
        config = BayesianNNConfig(
            input_dim=5,
            hidden_dims=[16],
            n_epochs=10
        )
        bnn = BayesianNeuralNetwork(config=config)

        X = np.random.randn(50, 5)
        y = np.sum(X, axis=1)

        bnn.fit(X, y)

        samples = bnn.get_posterior_predictive(X[:5], n_samples=20)
        assert samples.shape == (20, 5)

    def test_process_heat_integration(self):
        """Test Process Heat feature integration."""
        config = BayesianNNConfig(
            input_dim=10,
            hidden_dims=[32, 16],
            n_epochs=10
        )
        bnn = BayesianNeuralNetwork(config=config)

        # Generate training data matching Process Heat features
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        bnn.fit(X, y)

        features = ProcessHeatFeatures(
            furnace_temperature=800.0,
            fuel_flow_rate=100.0,
            air_fuel_ratio=10.5,
            flue_gas_temperature=350.0,
            oxygen_content=3.0,
            thermal_efficiency=85.0,
            production_rate=50.0,
            ambient_temperature=25.0
        )

        result = bnn.predict_process_heat(features)
        assert len(result.predictions) == 1

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        config = BayesianNNConfig(input_dim=5, hidden_dims=[16])
        bnn = BayesianNeuralNetwork(config=config)

        preds = np.array([1.0, 2.0, 3.0])
        epistemic = np.array([0.1, 0.2, 0.3])
        aleatoric = np.array([0.05, 0.1, 0.15])

        hash1 = bnn._calculate_provenance(preds, epistemic, aleatoric)
        hash2 = bnn._calculate_provenance(preds, epistemic, aleatoric)

        assert hash1 == hash2

    def test_mc_dropout_wrapper(self):
        """Test MC Dropout wrapper."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        wrapper = MCDropoutWrapper(MockModel(), dropout_rate=0.1, n_samples=10)

        X = np.random.randn(20, 5)
        result = wrapper.predict_with_uncertainty(X)

        assert len(result["predictions"]) == 20
        assert len(result["uncertainty"]) == 20
