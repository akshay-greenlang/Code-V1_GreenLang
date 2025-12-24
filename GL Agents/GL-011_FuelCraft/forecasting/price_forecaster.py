# -*- coding: utf-8 -*-
"""
Price Forecaster Module for GL-011 FuelCraft

Provides multi-fuel, multi-horizon price forecasting with quantile outputs
(P10/P50/P90) and confidence scores. Implements deterministic inference
with fixed seeds and versioned model artifacts.

Features:
- Multi-fuel support (natural gas, coal, oil, electricity, biomass)
- Multi-horizon forecasting (day-ahead to 12 months)
- Quantile predictions with uncertainty bounds
- Deterministic inference (fixed seeds, no GPU non-determinism)
- Model registry integration
- Provenance tracking

Zero-Hallucination Architecture:
- All forecasts use deterministic ML models (not LLM)
- Fixed random seeds for reproducibility
- No non-deterministic GPU kernels
- Complete provenance hashing

Author: GreenLang AI Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_QUANTILES = [0.10, 0.50, 0.90]
MIN_CONFIDENCE_THRESHOLD = 0.80
PRECISION = 4


class FuelType(str, Enum):
    """Enumeration of fuel types."""
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    CRUDE_OIL = "crude_oil"
    HEATING_OIL = "heating_oil"
    PROPANE = "propane"
    ELECTRICITY = "electricity"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"


class MarketHub(str, Enum):
    """Enumeration of market hubs."""
    # Natural Gas
    HENRY_HUB = "henry_hub"
    DOMINION_SOUTH = "dominion_south"
    CHICAGO_CITYGATE = "chicago_citygate"
    SOCAL_CITYGATE = "socal_citygate"
    PG_E_CITYGATE = "pge_citygate"

    # Crude Oil
    WTI_CUSHING = "wti_cushing"
    BRENT = "brent"

    # Coal
    CENTRAL_APPALACHIAN = "central_appalachian"
    POWDER_RIVER_BASIN = "powder_river_basin"
    ILLINOIS_BASIN = "illinois_basin"

    # Electricity
    PJM = "pjm"
    ERCOT = "ercot"
    CAISO = "caiso"
    MISO = "miso"
    NYISO = "nyiso"
    ISO_NE = "iso_ne"


class ForecastHorizon(str, Enum):
    """Enumeration of forecast horizons."""
    DAY_AHEAD = "day_ahead"
    WEEK_AHEAD = "week_ahead"
    TWO_WEEKS = "two_weeks"
    MONTH_AHEAD = "month_ahead"
    TWO_MONTHS = "two_months"
    THREE_MONTHS = "three_months"
    SIX_MONTHS = "six_months"
    TWELVE_MONTHS = "twelve_months"

    def to_days(self) -> int:
        """Convert horizon to number of days."""
        horizon_days = {
            ForecastHorizon.DAY_AHEAD: 1,
            ForecastHorizon.WEEK_AHEAD: 7,
            ForecastHorizon.TWO_WEEKS: 14,
            ForecastHorizon.MONTH_AHEAD: 30,
            ForecastHorizon.TWO_MONTHS: 60,
            ForecastHorizon.THREE_MONTHS: 90,
            ForecastHorizon.SIX_MONTHS: 180,
            ForecastHorizon.TWELVE_MONTHS: 365,
        }
        return horizon_days[self]


class QuantileForecast(BaseModel):
    """
    Single quantile forecast value.
    """

    quantile: float = Field(..., ge=0.0, le=1.0, description="Quantile level (0-1)")
    value: float = Field(..., description="Forecast value")
    lower_bound: Optional[float] = Field(None, description="Lower prediction interval")
    upper_bound: Optional[float] = Field(None, description="Upper prediction interval")

    @field_validator("quantile")
    @classmethod
    def validate_quantile(cls, v: float) -> float:
        """Round quantile to 2 decimal places."""
        return round(v, 2)


class PriceForecast(BaseModel):
    """
    Complete price forecast for a single fuel/hub/horizon.
    """

    forecast_id: str = Field(..., description="Unique forecast identifier")
    fuel_type: FuelType = Field(..., description="Fuel type")
    market_hub: MarketHub = Field(..., description="Market hub")
    horizon: ForecastHorizon = Field(..., description="Forecast horizon")
    forecast_date: datetime = Field(..., description="Date forecast was generated")
    target_date: datetime = Field(..., description="Date forecast is for")

    # Quantile forecasts
    p10: float = Field(..., description="10th percentile forecast")
    p50: float = Field(..., description="50th percentile (median) forecast")
    p90: float = Field(..., description="90th percentile forecast")
    quantiles: List[QuantileForecast] = Field(default_factory=list)

    # Confidence and uncertainty
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    prediction_interval_width: float = Field(..., description="P90 - P10 width")
    model_uncertainty: float = Field(..., description="Model epistemic uncertainty")
    data_uncertainty: float = Field(..., description="Data aleatoric uncertainty")

    # Model info
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")
    feature_schema_version: str = Field(..., description="Feature schema version")

    # Provenance
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    random_seed: int = Field(DEFAULT_RANDOM_SEED, description="Random seed used")
    computation_time_ms: float = Field(0.0, description="Computation time")

    # Unit
    unit: str = Field("USD", description="Price unit")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "forecast_id": self.forecast_id,
            "fuel_type": self.fuel_type.value,
            "market_hub": self.market_hub.value,
            "horizon": self.horizon.value,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "random_seed": self.random_seed,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ForecastBundle(BaseModel):
    """
    Bundle of forecasts for multiple horizons.
    """

    bundle_id: str = Field(..., description="Unique bundle identifier")
    fuel_type: FuelType = Field(..., description="Fuel type")
    market_hub: MarketHub = Field(..., description="Market hub")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    forecasts: Dict[ForecastHorizon, PriceForecast] = Field(default_factory=dict)

    # Aggregate metrics
    mean_confidence: float = Field(0.0, description="Mean confidence across horizons")
    min_confidence: float = Field(0.0, description="Minimum confidence")
    total_computation_time_ms: float = Field(0.0, description="Total computation time")

    # Provenance
    bundle_hash: str = Field("", description="SHA-256 bundle hash")
    model_id: str = Field(..., description="Model identifier")
    model_version: str = Field(..., description="Model version")

    def model_post_init(self, __context: Any) -> None:
        """Compute bundle hash after initialization."""
        if not self.bundle_hash and self.forecasts:
            self._compute_aggregate_metrics()
            self.bundle_hash = self._compute_hash()

    def _compute_aggregate_metrics(self) -> None:
        """Compute aggregate metrics from forecasts."""
        if not self.forecasts:
            return

        confidences = [f.confidence_score for f in self.forecasts.values()]
        times = [f.computation_time_ms for f in self.forecasts.values()]

        self.mean_confidence = float(np.mean(confidences))
        self.min_confidence = float(np.min(confidences))
        self.total_computation_time_ms = float(np.sum(times))

    def _compute_hash(self) -> str:
        """Compute SHA-256 bundle hash."""
        forecast_hashes = [f.provenance_hash for f in self.forecasts.values()]
        combined = "|".join(sorted(forecast_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_forecast(self, horizon: ForecastHorizon) -> Optional[PriceForecast]:
        """Get forecast for specific horizon."""
        return self.forecasts.get(horizon)


@dataclass
class ForecastConfig:
    """Configuration for price forecaster."""

    random_seed: int = DEFAULT_RANDOM_SEED
    quantiles: List[float] = field(default_factory=lambda: DEFAULT_QUANTILES.copy())
    min_confidence_threshold: float = MIN_CONFIDENCE_THRESHOLD
    precision: int = PRECISION
    use_gpu: bool = False
    enforce_determinism: bool = True
    cache_predictions: bool = True
    cache_ttl_seconds: int = 300
    max_horizon_days: int = 365
    enable_uncertainty_quantification: bool = True


class PriceForecaster:
    """
    Multi-fuel, multi-horizon price forecaster.

    Provides deterministic price forecasts with quantile outputs
    (P10/P50/P90) and confidence scores.

    Zero-Hallucination Guarantees:
    - All forecasts use deterministic ML models
    - Fixed random seeds for reproducibility
    - No non-deterministic GPU kernels
    - Complete provenance hashing
    """

    def __init__(
        self,
        config: Optional[ForecastConfig] = None,
        model_registry: Optional[Any] = None,
        feature_store: Optional[Any] = None
    ):
        """
        Initialize price forecaster.

        Args:
            config: Forecaster configuration
            model_registry: Model registry for versioned models
            feature_store: Feature store for feature extraction
        """
        self.config = config or ForecastConfig()
        self.model_registry = model_registry
        self.feature_store = feature_store

        # Set random seed for reproducibility
        self._set_random_seed(self.config.random_seed)

        # Model cache
        self._model_cache: Dict[str, Any] = {}

        # Prediction cache
        self._prediction_cache: Dict[str, Tuple[datetime, PriceForecast]] = {}

        # Loaded models
        self._models: Dict[str, Any] = {}

        logger.info(
            f"PriceForecaster initialized: seed={self.config.random_seed}, "
            f"quantiles={self.config.quantiles}, determinism={self.config.enforce_determinism}"
        )

    def forecast(
        self,
        fuel_type: FuelType,
        market_hub: MarketHub,
        horizons: List[ForecastHorizon],
        features: Optional[Dict[str, float]] = None,
        forecast_date: Optional[datetime] = None
    ) -> ForecastBundle:
        """
        Generate price forecasts for specified horizons.

        Args:
            fuel_type: Type of fuel
            market_hub: Market hub
            horizons: List of forecast horizons
            features: Input features (uses feature store if None)
            forecast_date: Date to generate forecast for (now if None)

        Returns:
            ForecastBundle with forecasts for each horizon
        """
        import uuid

        start_time = time.time()
        forecast_date = forecast_date or datetime.now(timezone.utc)

        # Get model for this fuel/hub combination
        model_key = f"{fuel_type.value}_{market_hub.value}"
        model, model_id, model_version = self._get_model(model_key)

        # Get features
        if features is None:
            features = self._extract_features(fuel_type, market_hub, forecast_date)

        # Get feature schema version
        feature_schema_version = self._get_feature_schema_version()

        # Generate forecasts for each horizon
        forecasts: Dict[ForecastHorizon, PriceForecast] = {}

        for horizon in horizons:
            forecast = self._generate_single_forecast(
                fuel_type=fuel_type,
                market_hub=market_hub,
                horizon=horizon,
                features=features,
                forecast_date=forecast_date,
                model=model,
                model_id=model_id,
                model_version=model_version,
                feature_schema_version=feature_schema_version,
            )
            forecasts[horizon] = forecast

        # Create bundle
        bundle = ForecastBundle(
            bundle_id=str(uuid.uuid4()),
            fuel_type=fuel_type,
            market_hub=market_hub,
            forecasts=forecasts,
            model_id=model_id,
            model_version=model_version,
        )

        total_time = (time.time() - start_time) * 1000
        bundle.total_computation_time_ms = total_time

        logger.info(
            f"Generated forecast bundle: fuel={fuel_type.value}, hub={market_hub.value}, "
            f"horizons={len(horizons)}, time={total_time:.2f}ms"
        )

        return bundle

    def forecast_single(
        self,
        fuel_type: FuelType,
        market_hub: MarketHub,
        horizon: ForecastHorizon,
        features: Optional[Dict[str, float]] = None,
        forecast_date: Optional[datetime] = None
    ) -> PriceForecast:
        """
        Generate single price forecast.

        Args:
            fuel_type: Type of fuel
            market_hub: Market hub
            horizon: Forecast horizon
            features: Input features
            forecast_date: Forecast generation date

        Returns:
            PriceForecast for the specified horizon
        """
        bundle = self.forecast(
            fuel_type=fuel_type,
            market_hub=market_hub,
            horizons=[horizon],
            features=features,
            forecast_date=forecast_date,
        )
        return bundle.forecasts[horizon]

    def _generate_single_forecast(
        self,
        fuel_type: FuelType,
        market_hub: MarketHub,
        horizon: ForecastHorizon,
        features: Dict[str, float],
        forecast_date: datetime,
        model: Any,
        model_id: str,
        model_version: str,
        feature_schema_version: str
    ) -> PriceForecast:
        """Generate forecast for a single horizon."""
        import uuid

        start_time = time.time()

        # Ensure determinism
        self._set_random_seed(self.config.random_seed)

        # Calculate target date
        target_date = forecast_date + timedelta(days=horizon.to_days())

        # Prepare feature vector
        feature_vector = self._prepare_feature_vector(features, horizon)

        # Generate quantile predictions
        quantile_predictions = self._predict_quantiles(model, feature_vector)

        # Extract P10/P50/P90
        p10 = quantile_predictions.get(0.10, 0.0)
        p50 = quantile_predictions.get(0.50, 0.0)
        p90 = quantile_predictions.get(0.90, 0.0)

        # Build full quantile list
        quantiles = [
            QuantileForecast(quantile=q, value=v)
            for q, v in sorted(quantile_predictions.items())
        ]

        # Calculate confidence score
        confidence_score = self._compute_confidence_score(
            quantile_predictions, features, horizon
        )

        # Calculate uncertainties
        model_uncertainty, data_uncertainty = self._compute_uncertainties(
            model, feature_vector
        )

        # Round values
        p10 = self._round_value(p10)
        p50 = self._round_value(p50)
        p90 = self._round_value(p90)

        computation_time = (time.time() - start_time) * 1000

        # Get unit
        unit = self._get_unit(fuel_type)

        forecast = PriceForecast(
            forecast_id=str(uuid.uuid4()),
            fuel_type=fuel_type,
            market_hub=market_hub,
            horizon=horizon,
            forecast_date=forecast_date,
            target_date=target_date,
            p10=p10,
            p50=p50,
            p90=p90,
            quantiles=quantiles,
            confidence_score=confidence_score,
            prediction_interval_width=p90 - p10,
            model_uncertainty=model_uncertainty,
            data_uncertainty=data_uncertainty,
            model_id=model_id,
            model_version=model_version,
            feature_schema_version=feature_schema_version,
            random_seed=self.config.random_seed,
            computation_time_ms=computation_time,
            unit=unit,
        )

        return forecast

    def _get_model(self, model_key: str) -> Tuple[Any, str, str]:
        """Get model from registry or cache."""
        if model_key in self._models:
            model_info = self._models[model_key]
            return model_info["model"], model_info["model_id"], model_info["version"]

        # Load from registry if available
        if self.model_registry is not None:
            model_info = self.model_registry.get_model(model_key)
            if model_info:
                self._models[model_key] = model_info
                return model_info["model"], model_info["model_id"], model_info["version"]

        # Return placeholder model for demonstration
        return self._create_placeholder_model(), f"placeholder_{model_key}", "1.0.0"

    def _create_placeholder_model(self) -> Any:
        """Create placeholder model for demonstration."""
        class PlaceholderModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                # Simple placeholder prediction
                np.random.seed(DEFAULT_RANDOM_SEED)
                base = 3.0 + np.random.randn(len(X)) * 0.1
                return base

            def predict_quantiles(
                self, X: np.ndarray, quantiles: List[float]
            ) -> Dict[float, np.ndarray]:
                np.random.seed(DEFAULT_RANDOM_SEED)
                base = 3.0
                results = {}
                for q in quantiles:
                    offset = (q - 0.5) * 1.0  # +/- 0.5 for P10/P90
                    results[q] = np.array([base + offset])
                return results

        return PlaceholderModel()

    def _extract_features(
        self,
        fuel_type: FuelType,
        market_hub: MarketHub,
        forecast_date: datetime
    ) -> Dict[str, float]:
        """Extract features for forecasting."""
        if self.feature_store is not None:
            # Use feature store
            raw_data = {
                "fuel_type": fuel_type.value,
                "market_hub": market_hub.value,
                "date": forecast_date.isoformat(),
            }
            feature_vector = self.feature_store.extract_features(
                raw_data=raw_data,
                feature_ids=self._get_required_features(fuel_type),
                timestamp=forecast_date,
            )
            return {name: fv.value for name, fv in feature_vector.features.items()}

        # Return placeholder features
        return {
            "spot_price_lag_1": 3.0,
            "spot_price_lag_7": 2.9,
            "heating_degree_days": 20.0,
            "cooling_degree_days": 0.0,
            "storage_level": 2500.0,
            "production": 100.0,
            "demand": 95.0,
        }

    def _get_required_features(self, fuel_type: FuelType) -> List[str]:
        """Get list of required feature IDs for fuel type."""
        base_features = [
            "natural_gas_spot_price:1.0.0",
            "heating_degree_days:1.0.0",
            "cooling_degree_days:1.0.0",
        ]

        fuel_specific = {
            FuelType.NATURAL_GAS: ["natural_gas_spot_price:1.0.0"],
            FuelType.COAL: ["coal_price:1.0.0"],
            FuelType.CRUDE_OIL: ["crude_oil_price:1.0.0"],
            FuelType.ELECTRICITY: ["electricity_price:1.0.0"],
        }

        return base_features + fuel_specific.get(fuel_type, [])

    def _get_feature_schema_version(self) -> str:
        """Get feature schema version hash."""
        if self.feature_store is not None:
            return self.feature_store.get_schema_version()
        return "placeholder_schema_1.0.0"

    def _prepare_feature_vector(
        self,
        features: Dict[str, float],
        horizon: ForecastHorizon
    ) -> np.ndarray:
        """Prepare feature vector for prediction."""
        # Add horizon as feature
        horizon_days = horizon.to_days()

        # Create numpy array
        feature_values = list(features.values())
        feature_values.append(float(horizon_days))

        return np.array([feature_values], dtype=np.float64)

    def _predict_quantiles(
        self,
        model: Any,
        feature_vector: np.ndarray
    ) -> Dict[float, float]:
        """Generate quantile predictions."""
        # Ensure determinism
        self._set_random_seed(self.config.random_seed)

        if hasattr(model, "predict_quantiles"):
            # Model supports quantile regression
            results = model.predict_quantiles(feature_vector, self.config.quantiles)
            return {q: float(v[0]) for q, v in results.items()}

        if hasattr(model, "predict"):
            # Use point prediction with uncertainty estimation
            point_pred = float(model.predict(feature_vector)[0])

            # Estimate uncertainty based on horizon
            uncertainty = 0.15  # Base uncertainty

            return {
                0.10: point_pred * (1 - uncertainty),
                0.50: point_pred,
                0.90: point_pred * (1 + uncertainty),
            }

        raise ValueError("Model must have predict or predict_quantiles method")

    def _compute_confidence_score(
        self,
        quantile_predictions: Dict[float, float],
        features: Dict[str, float],
        horizon: ForecastHorizon
    ) -> float:
        """
        Compute confidence score for prediction.

        Confidence decreases with:
        - Larger prediction intervals
        - Longer horizons
        - Missing features
        """
        # Base confidence
        confidence = 1.0

        # Penalize for wide prediction interval
        p10 = quantile_predictions.get(0.10, 0)
        p90 = quantile_predictions.get(0.90, 0)
        p50 = quantile_predictions.get(0.50, 1)

        if p50 > 0:
            interval_ratio = (p90 - p10) / p50
            confidence -= min(0.3, interval_ratio * 0.1)

        # Penalize for longer horizons
        horizon_days = horizon.to_days()
        horizon_penalty = min(0.3, horizon_days / 365 * 0.3)
        confidence -= horizon_penalty

        # Penalize for missing features
        expected_features = 7
        missing_ratio = max(0, 1 - len(features) / expected_features)
        confidence -= missing_ratio * 0.2

        return max(0.0, min(1.0, confidence))

    def _compute_uncertainties(
        self,
        model: Any,
        feature_vector: np.ndarray
    ) -> Tuple[float, float]:
        """Compute model (epistemic) and data (aleatoric) uncertainties."""
        # Placeholder uncertainty estimation
        # In production, would use ensemble variance, MC dropout, etc.

        model_uncertainty = 0.10  # Epistemic uncertainty
        data_uncertainty = 0.15  # Aleatoric uncertainty

        return model_uncertainty, data_uncertainty

    def _round_value(self, value: float) -> float:
        """Round value to configured precision."""
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal(10) ** -self.config.precision,
            rounding=ROUND_HALF_UP
        )
        return float(rounded)

    def _get_unit(self, fuel_type: FuelType) -> str:
        """Get price unit for fuel type."""
        units = {
            FuelType.NATURAL_GAS: "USD/MMBtu",
            FuelType.COAL: "USD/short_ton",
            FuelType.CRUDE_OIL: "USD/barrel",
            FuelType.HEATING_OIL: "USD/gallon",
            FuelType.PROPANE: "USD/gallon",
            FuelType.ELECTRICITY: "USD/MWh",
            FuelType.BIOMASS: "USD/ton",
            FuelType.HYDROGEN: "USD/kg",
        }
        return units.get(fuel_type, "USD")

    def _set_random_seed(self, seed: int) -> None:
        """Set random seed for determinism."""
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # Additional seeds for common libraries
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass

        # TensorFlow determinism
        if self.config.enforce_determinism:
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
            except ImportError:
                pass

            # PyTorch determinism
            try:
                import torch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except ImportError:
                pass


# Utility functions

def validate_forecast(forecast: PriceForecast) -> Tuple[bool, List[str]]:
    """
    Validate a price forecast.

    Args:
        forecast: Forecast to validate

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    # Check quantile ordering
    if forecast.p10 > forecast.p50:
        errors.append(f"P10 ({forecast.p10}) > P50 ({forecast.p50})")

    if forecast.p50 > forecast.p90:
        errors.append(f"P50 ({forecast.p50}) > P90 ({forecast.p90})")

    # Check confidence threshold
    if forecast.confidence_score < MIN_CONFIDENCE_THRESHOLD:
        errors.append(
            f"Confidence {forecast.confidence_score} below threshold {MIN_CONFIDENCE_THRESHOLD}"
        )

    # Check for negative prices (except electricity)
    if forecast.fuel_type != FuelType.ELECTRICITY:
        if forecast.p10 < 0:
            errors.append(f"P10 is negative: {forecast.p10}")

    # Check provenance hash
    expected_hash = forecast._compute_hash()
    if forecast.provenance_hash != expected_hash:
        errors.append("Provenance hash mismatch")

    return len(errors) == 0, errors


def compute_confidence_score(
    p10: float,
    p50: float,
    p90: float,
    horizon_days: int,
    feature_completeness: float = 1.0
) -> float:
    """
    Compute confidence score for a forecast.

    Args:
        p10: 10th percentile prediction
        p50: 50th percentile prediction
        p90: 90th percentile prediction
        horizon_days: Forecast horizon in days
        feature_completeness: Ratio of available features (0-1)

    Returns:
        Confidence score (0-1)
    """
    confidence = 1.0

    # Penalize for wide interval
    if p50 > 0:
        interval_ratio = (p90 - p10) / p50
        confidence -= min(0.3, interval_ratio * 0.1)

    # Penalize for longer horizons
    horizon_penalty = min(0.3, horizon_days / 365 * 0.3)
    confidence -= horizon_penalty

    # Penalize for missing features
    confidence -= (1 - feature_completeness) * 0.2

    return max(0.0, min(1.0, confidence))
