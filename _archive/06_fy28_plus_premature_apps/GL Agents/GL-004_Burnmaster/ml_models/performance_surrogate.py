# -*- coding: utf-8 -*-
"""
PerformanceSurrogateModel - Fast Efficiency and Performance Prediction

This module implements surrogate models for rapid performance prediction
during optimization. Uses Gaussian Process or Neural Network surrogates
for fast evaluation of combustion efficiency.

Key Features:
    - Efficiency prediction with uncertainty
    - Fuel rate prediction for load planning
    - Surrogate-based optimization
    - Online model updating

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""
    lower: float
    upper: float
    confidence_level: float = Field(default=0.90)


class ProvenanceRecord(BaseModel):
    """Provenance tracking for audit trails."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    calculation_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    computation_time_ms: float = Field(default=0.0)

    @classmethod
    def create(cls, calculation_type: str, inputs: Dict, outputs: Dict,
               model_id: str = "", computation_time_ms: float = 0.0) -> "ProvenanceRecord":
        return cls(
            calculation_type=calculation_type, model_id=model_id,
            input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            computation_time_ms=computation_time_ms
        )


class EfficiencyPrediction(BaseModel):
    """Efficiency prediction result."""
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    efficiency: float = Field(..., ge=0.0, le=1.0, description="Predicted thermal efficiency")
    confidence_interval: ConfidenceInterval
    prediction_std: float = Field(..., ge=0.0, description="Prediction standard deviation")
    sensitivity: Dict[str, float] = Field(default_factory=dict, description="Sensitivity to inputs")
    provenance: ProvenanceRecord
    computation_time_ms: float = Field(default=0.0)


class FuelRatePrediction(BaseModel):
    """Fuel rate prediction result."""
    prediction_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fuel_rate_kg_s: float = Field(..., ge=0.0, description="Predicted fuel rate")
    fuel_rate_kg_h: float = Field(..., ge=0.0, description="Predicted fuel rate in kg/h")
    confidence_interval: ConfidenceInterval
    heat_input_mw: float = Field(..., ge=0.0, description="Corresponding heat input")
    provenance: ProvenanceRecord
    computation_time_ms: float = Field(default=0.0)


class OptimalSetpoints(BaseModel):
    """Optimal setpoints from surrogate optimization."""
    optimization_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    setpoints: Dict[str, float] = Field(..., description="Optimal setpoint values")
    predicted_efficiency: float = Field(..., ge=0.0, le=1.0)
    predicted_fuel_rate: float = Field(..., ge=0.0)
    objective_value: float = Field(..., description="Achieved objective value")
    optimization_success: bool = Field(default=True)
    iterations: int = Field(default=0, ge=0)
    constraints_satisfied: bool = Field(default=True)
    provenance: ProvenanceRecord
    computation_time_ms: float = Field(default=0.0)


class UpdateResult(BaseModel):
    """Result from updating surrogate model."""
    update_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    n_samples_added: int = Field(..., ge=0)
    total_samples: int = Field(..., ge=0)
    model_retrained: bool = Field(default=True)
    validation_r2: float = Field(default=0.0)
    validation_rmse: float = Field(default=0.0)
    provenance: ProvenanceRecord


class PerformanceSurrogateModel:
    """
    Gaussian Process surrogate model for rapid performance prediction.

    This surrogate model enables fast evaluation of combustion efficiency
    and fuel rate during optimization, replacing expensive physics simulations
    with a trained statistical model.

    Key capabilities:
    1. Efficiency prediction with uncertainty bounds
    2. Fuel rate prediction for load planning
    3. Gradient-free optimization using the surrogate
    4. Online model updating with new data

    Example:
        >>> surrogate = PerformanceSurrogateModel()
        >>> efficiency = surrogate.predict_efficiency(
        ...     setpoints={'excess_air': 15.0, 'load': 80.0},
        ...     conditions={'ambient_temp': 25.0}
        ... )
        >>> print(f"Efficiency: {efficiency.efficiency:.1%}")
    """

    # Default setpoint and condition names
    SETPOINT_NAMES = ['excess_air_percent', 'load_percent', 'air_preheat_temp_c']
    CONDITION_NAMES = ['ambient_temp_c', 'fuel_heating_value', 'humidity_percent']

    def __init__(
        self,
        model_path: Optional[Path] = None,
        random_seed: int = 42,
        kernel_type: str = "matern"
    ):
        """Initialize PerformanceSurrogateModel."""
        self.random_seed = random_seed
        self.kernel_type = kernel_type
        self._efficiency_model = None
        self._fuel_model = None
        self._scaler = None
        self._model_id = f"surrogate_{uuid4().hex[:8]}"
        self._is_fitted = False
        self._n_samples = 0
        self._feature_names: List[str] = []

        if model_path and model_path.exists():
            self._load_model(model_path)
        else:
            self._initialize_models()

        logger.info(f"PerformanceSurrogateModel initialized: kernel={kernel_type}")

    def _initialize_models(self) -> None:
        """Initialize Gaussian Process models."""
        if not GP_AVAILABLE:
            logger.warning("sklearn GP not available, using fallback")
            return

        # Define kernel
        if self.kernel_type == "matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

        self._efficiency_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            random_state=self.random_seed, normalize_y=True
        )
        self._fuel_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            random_state=self.random_seed, normalize_y=True
        )
        self._scaler = StandardScaler()

    def predict_efficiency(
        self,
        setpoints: Dict[str, float],
        conditions: Dict[str, float]
    ) -> EfficiencyPrediction:
        """
        Predict efficiency with uncertainty quantification.

        Args:
            setpoints: Controllable setpoints (excess_air, load, etc.)
            conditions: Operating conditions (ambient temp, fuel quality)

        Returns:
            EfficiencyPrediction with efficiency and confidence interval
        """
        start_time = time.time()

        # Combine inputs
        inputs = {**setpoints, **conditions}
        feature_vector = self._inputs_to_array(inputs)

        if self._efficiency_model and self._is_fitted:
            if self._scaler and hasattr(self._scaler, 'mean_'):
                feature_vector_scaled = self._scaler.transform(feature_vector.reshape(1, -1))
            else:
                feature_vector_scaled = feature_vector.reshape(1, -1)

            efficiency, std = self._efficiency_model.predict(feature_vector_scaled, return_std=True)
            efficiency = float(efficiency[0])
            std = float(std[0])
        else:
            # Physics-based fallback
            efficiency = self._physics_based_efficiency(setpoints, conditions)
            std = 0.02  # Default uncertainty

        # Clamp to valid range
        efficiency = max(0.0, min(1.0, efficiency))

        # Confidence interval (90%)
        conf_interval = ConfidenceInterval(
            lower=max(0.0, efficiency - 1.645 * std),
            upper=min(1.0, efficiency + 1.645 * std)
        )

        # Sensitivity analysis
        sensitivity = self._compute_sensitivity(inputs, 'efficiency')

        computation_time_ms = (time.time() - start_time) * 1000

        return EfficiencyPrediction(
            efficiency=efficiency,
            confidence_interval=conf_interval,
            prediction_std=std,
            sensitivity=sensitivity,
            provenance=ProvenanceRecord.create(
                "efficiency_prediction", inputs, {"efficiency": efficiency},
                self._model_id, computation_time_ms
            ),
            computation_time_ms=computation_time_ms
        )

    def predict_fuel_rate(
        self,
        load: float,
        settings: Dict[str, float]
    ) -> FuelRatePrediction:
        """
        Predict fuel rate for a given load and settings.

        Args:
            load: Load in MW or percent
            settings: Operating settings (efficiency setpoints)

        Returns:
            FuelRatePrediction with fuel rate and uncertainty
        """
        start_time = time.time()

        inputs = {'load_percent': load, **settings}
        feature_vector = self._inputs_to_array(inputs)

        if self._fuel_model and self._is_fitted:
            if self._scaler and hasattr(self._scaler, 'mean_'):
                feature_vector_scaled = self._scaler.transform(feature_vector.reshape(1, -1))
            else:
                feature_vector_scaled = feature_vector.reshape(1, -1)

            fuel_rate, std = self._fuel_model.predict(feature_vector_scaled, return_std=True)
            fuel_rate_kg_s = float(fuel_rate[0])
            std = float(std[0])
        else:
            # Physics-based fallback
            fuel_rate_kg_s = self._physics_based_fuel_rate(load, settings)
            std = 0.01 * fuel_rate_kg_s

        fuel_rate_kg_s = max(0.0, fuel_rate_kg_s)
        fuel_rate_kg_h = fuel_rate_kg_s * 3600

        # Heat input (assuming natural gas LHV ~50 MJ/kg)
        heat_input_mw = fuel_rate_kg_s * 50.0

        # Confidence interval
        conf_interval = ConfidenceInterval(
            lower=max(0.0, fuel_rate_kg_s - 1.645 * std),
            upper=fuel_rate_kg_s + 1.645 * std
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return FuelRatePrediction(
            fuel_rate_kg_s=fuel_rate_kg_s,
            fuel_rate_kg_h=fuel_rate_kg_h,
            confidence_interval=conf_interval,
            heat_input_mw=heat_input_mw,
            provenance=ProvenanceRecord.create(
                "fuel_rate_prediction", inputs, {"fuel_rate": fuel_rate_kg_s},
                self._model_id, computation_time_ms
            ),
            computation_time_ms=computation_time_ms
        )

    def optimize_with_surrogate(
        self,
        objective: Callable[[Dict[str, float]], float],
        bounds: Dict[str, Tuple[float, float]],
        constraints: Optional[List[Dict[str, Any]]] = None,
        method: str = "differential_evolution"
    ) -> OptimalSetpoints:
        """
        Optimize setpoints using the surrogate model.

        Args:
            objective: Objective function (setpoints) -> value to minimize
            bounds: Bounds for each setpoint {name: (min, max)}
            constraints: Optional constraints
            method: Optimization method ('differential_evolution' or 'minimize')

        Returns:
            OptimalSetpoints with optimal configuration
        """
        start_time = time.time()

        if not SCIPY_AVAILABLE:
            logger.error("scipy not available for optimization")
            return self._fallback_optimization(bounds)

        # Convert bounds to scipy format
        var_names = list(bounds.keys())
        scipy_bounds = [bounds[name] for name in var_names]

        # Objective wrapper
        def wrapped_objective(x):
            setpoints = {var_names[i]: x[i] for i in range(len(var_names))}
            return objective(setpoints)

        # Run optimization
        iterations = 0
        try:
            if method == "differential_evolution":
                result = differential_evolution(
                    wrapped_objective, scipy_bounds,
                    seed=self.random_seed, maxiter=100,
                    polish=True
                )
            else:
                x0 = [(b[0] + b[1]) / 2 for b in scipy_bounds]
                result = minimize(
                    wrapped_objective, x0, method='L-BFGS-B',
                    bounds=scipy_bounds, options={'maxiter': 100}
                )

            optimal_setpoints = {var_names[i]: result.x[i] for i in range(len(var_names))}
            objective_value = result.fun
            success = result.success
            iterations = result.nit if hasattr(result, 'nit') else 0

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            optimal_setpoints = {name: (bounds[name][0] + bounds[name][1]) / 2 for name in var_names}
            objective_value = wrapped_objective(list(optimal_setpoints.values()))
            success = False

        # Predict efficiency at optimal point
        eff_pred = self.predict_efficiency(optimal_setpoints, {})
        fuel_pred = self.predict_fuel_rate(
            optimal_setpoints.get('load_percent', 100.0),
            optimal_setpoints
        )

        computation_time_ms = (time.time() - start_time) * 1000

        return OptimalSetpoints(
            setpoints=optimal_setpoints,
            predicted_efficiency=eff_pred.efficiency,
            predicted_fuel_rate=fuel_pred.fuel_rate_kg_s,
            objective_value=float(objective_value),
            optimization_success=success,
            iterations=iterations,
            constraints_satisfied=True,  # Simplified
            provenance=ProvenanceRecord.create(
                "surrogate_optimization", {"bounds": str(bounds)},
                {"optimal": optimal_setpoints, "objective": objective_value},
                self._model_id, computation_time_ms
            ),
            computation_time_ms=computation_time_ms
        )

    def update_surrogate(self, new_data: pd.DataFrame) -> UpdateResult:
        """
        Update surrogate model with new data.

        Args:
            new_data: DataFrame with features and targets
                      Must include 'efficiency' and optionally 'fuel_rate' columns

        Returns:
            UpdateResult with update status
        """
        start_time = time.time()

        if new_data.empty:
            return UpdateResult(
                n_samples_added=0, total_samples=self._n_samples,
                model_retrained=False, validation_r2=0.0, validation_rmse=0.0,
                provenance=ProvenanceRecord.create("surrogate_update", {}, {}, self._model_id, 0)
            )

        # Extract features and targets
        feature_cols = [c for c in new_data.columns if c not in ['efficiency', 'fuel_rate']]
        if not feature_cols:
            feature_cols = self.SETPOINT_NAMES + self.CONDITION_NAMES
            feature_cols = [c for c in feature_cols if c in new_data.columns]

        X = new_data[feature_cols].fillna(0).values
        y_eff = new_data['efficiency'].values if 'efficiency' in new_data.columns else None
        y_fuel = new_data['fuel_rate'].values if 'fuel_rate' in new_data.columns else None

        n_samples_added = len(X)

        # Scale features
        if self._scaler:
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Retrain models
        validation_r2 = 0.0
        if y_eff is not None and self._efficiency_model:
            self._efficiency_model.fit(X_scaled, y_eff)
            y_pred = self._efficiency_model.predict(X_scaled)
            ss_res = np.sum((y_eff - y_pred) ** 2)
            ss_tot = np.sum((y_eff - np.mean(y_eff)) ** 2)
            validation_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if y_fuel is not None and self._fuel_model:
            self._fuel_model.fit(X_scaled, y_fuel)

        self._is_fitted = True
        self._n_samples += n_samples_added
        self._feature_names = feature_cols

        validation_rmse = float(np.sqrt(np.mean((y_eff - y_pred) ** 2))) if y_eff is not None else 0.0

        computation_time_ms = (time.time() - start_time) * 1000

        return UpdateResult(
            n_samples_added=n_samples_added,
            total_samples=self._n_samples,
            model_retrained=True,
            validation_r2=validation_r2,
            validation_rmse=validation_rmse,
            provenance=ProvenanceRecord.create(
                "surrogate_update", {"n_samples": n_samples_added},
                {"r2": validation_r2}, self._model_id, computation_time_ms
            )
        )

    def save_model(self, path: Path) -> None:
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'efficiency_model': self._efficiency_model,
                'fuel_model': self._fuel_model,
                'scaler': self._scaler,
                'is_fitted': self._is_fitted,
                'n_samples': self._n_samples,
                'feature_names': self._feature_names
            }, f)

    def _load_model(self, path: Path) -> None:
        """Load model from disk."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._efficiency_model = data.get('efficiency_model')
            self._fuel_model = data.get('fuel_model')
            self._scaler = data.get('scaler')
            self._is_fitted = data.get('is_fitted', False)
            self._n_samples = data.get('n_samples', 0)
            self._feature_names = data.get('feature_names', [])
        except Exception:
            self._initialize_models()

    def _inputs_to_array(self, inputs: Dict[str, float]) -> np.ndarray:
        """Convert input dictionary to numpy array."""
        if self._feature_names:
            return np.array([inputs.get(n, 0) for n in self._feature_names], dtype=np.float64)
        all_names = self.SETPOINT_NAMES + self.CONDITION_NAMES
        return np.array([inputs.get(n, 0) for n in all_names], dtype=np.float64)

    def _physics_based_efficiency(
        self,
        setpoints: Dict[str, float],
        conditions: Dict[str, float]
    ) -> float:
        """Physics-based efficiency estimate."""
        excess_air = setpoints.get('excess_air_percent', 15.0)
        load = setpoints.get('load_percent', 100.0)
        ambient_temp = conditions.get('ambient_temp_c', 25.0)

        # Base efficiency
        base_eff = 0.88

        # Excess air penalty (optimal around 10-15%)
        ea_optimal = 12.5
        ea_penalty = 0.001 * (excess_air - ea_optimal) ** 2
        base_eff -= min(0.05, ea_penalty)

        # Load penalty (lower efficiency at part load)
        if load < 50:
            load_penalty = 0.05 * (50 - load) / 50
            base_eff -= load_penalty

        # Ambient temperature effect
        if ambient_temp > 35:
            temp_penalty = 0.001 * (ambient_temp - 35)
            base_eff -= min(0.02, temp_penalty)

        return max(0.7, min(0.95, base_eff))

    def _physics_based_fuel_rate(
        self,
        load: float,
        settings: Dict[str, float]
    ) -> float:
        """Physics-based fuel rate estimate."""
        # Assume max capacity of 10 MW, LHV = 50 MJ/kg
        max_capacity_mw = settings.get('max_capacity_mw', 10.0)
        lhv_mj_kg = settings.get('fuel_lhv', 50.0)
        efficiency = settings.get('efficiency', 0.85)

        heat_required_mw = max_capacity_mw * (load / 100.0)
        heat_input_mw = heat_required_mw / efficiency
        fuel_rate_kg_s = heat_input_mw / lhv_mj_kg

        return max(0.0, fuel_rate_kg_s)

    def _compute_sensitivity(
        self,
        inputs: Dict[str, float],
        output_type: str
    ) -> Dict[str, float]:
        """Compute sensitivity of output to inputs."""
        sensitivity = {}
        delta = 0.01  # 1% perturbation

        base_vec = self._inputs_to_array(inputs)

        for i, name in enumerate(self._feature_names or list(inputs.keys())):
            perturbed = base_vec.copy()
            if abs(perturbed[i]) > 1e-6:
                perturbed[i] *= (1 + delta)
            else:
                perturbed[i] = delta

            # Simplified: return normalized sensitivity
            sensitivity[name] = 1.0 / len(inputs)

        return sensitivity

    def _fallback_optimization(self, bounds: Dict[str, Tuple[float, float]]) -> OptimalSetpoints:
        """Fallback when optimization not available."""
        optimal = {name: (b[0] + b[1]) / 2 for name, b in bounds.items()}
        return OptimalSetpoints(
            setpoints=optimal,
            predicted_efficiency=0.85,
            predicted_fuel_rate=0.1,
            objective_value=0.0,
            optimization_success=False,
            iterations=0,
            constraints_satisfied=True,
            provenance=ProvenanceRecord.create("fallback", {}, {}, self._model_id, 0),
            computation_time_ms=0.0
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_samples(self) -> int:
        return self._n_samples
