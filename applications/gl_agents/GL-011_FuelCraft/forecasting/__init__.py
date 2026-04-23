# -*- coding: utf-8 -*-
"""
GL-011 FuelCraft Forecasting Module

Provides multi-fuel, multi-horizon price forecasting with deterministic inference
and full provenance tracking. Aligned with GreenLang Global AI Standards v2.0.

Features:
- Feature store with versioning and lineage tracking
- Multi-fuel price forecasting (natural gas, coal, oil, electricity, biomass)
- Multi-horizon outputs (day-ahead to 12 months)
- Quantile predictions (P10/P50/P90) with confidence scores
- Deterministic inference (pinned versions, fixed seeds)
- Weather feature integration (HDD/CDD, storm risk)
- Model registry with evaluation metrics

Zero-Hallucination Architecture:
- All price forecasts use deterministic ML models (not LLM)
- Fixed random seeds for reproducibility
- SHA-256 provenance hashing for audit trails
- Versioned training data references

Author: GreenLang AI Team
Version: 1.0.0
"""

from .feature_store import (
    FeatureDefinition,
    FeatureValue,
    FeatureVector,
    FeatureLineage,
    DriftDetectionResult,
    FeatureStoreConfig,
    FeatureStore,
    compute_feature_hash,
    validate_feature_schema,
    detect_feature_drift,
)

from .price_forecaster import (
    FuelType,
    MarketHub,
    ForecastHorizon,
    QuantileForecast,
    PriceForecast,
    ForecastBundle,
    ForecastConfig,
    PriceForecaster,
    validate_forecast,
    compute_confidence_score,
)

from .model_registry import (
    ModelMetadata,
    ModelVersion,
    EvaluationMetrics,
    DeploymentStatus,
    ModelRegistryConfig,
    ModelRegistry,
    compute_model_hash,
    validate_model_schema,
)

from .weather_features import (
    WeatherObservation,
    HDDCDDResult,
    StormRiskSignal,
    ShippingDisruptionSignal,
    RegionMapping,
    WeatherFeatureConfig,
    WeatherFeatureExtractor,
    calculate_hdd,
    calculate_cdd,
    assess_storm_risk,
)

__version__ = "1.0.0"
__author__ = "GreenLang AI Team"

__all__ = [
    "__version__",
    "__author__",
    "FeatureDefinition",
    "FeatureValue",
    "FeatureVector",
    "FeatureLineage",
    "DriftDetectionResult",
    "FeatureStoreConfig",
    "FeatureStore",
    "compute_feature_hash",
    "validate_feature_schema",
    "detect_feature_drift",
    "FuelType",
    "MarketHub",
    "ForecastHorizon",
    "QuantileForecast",
    "PriceForecast",
    "ForecastBundle",
    "ForecastConfig",
    "PriceForecaster",
    "validate_forecast",
    "compute_confidence_score",
    "ModelMetadata",
    "ModelVersion",
    "EvaluationMetrics",
    "DeploymentStatus",
    "ModelRegistryConfig",
    "ModelRegistry",
    "compute_model_hash",
    "validate_model_schema",
    "WeatherObservation",
    "HDDCDDResult",
    "StormRiskSignal",
    "ShippingDisruptionSignal",
    "RegionMapping",
    "WeatherFeatureConfig",
    "WeatherFeatureExtractor",
    "calculate_hdd",
    "calculate_cdd",
    "assess_storm_risk",
]
