# -*- coding: utf-8 -*-
"""
GreenLang Feature Store Module
==============================

Production-ready feature store for Process Heat agents using Feast.

This module provides:
- Feature definitions for all Process Heat agents (GL-001 through GL-020)
- Online/offline feature serving
- Feature engineering pipelines with rolling aggregations
- SHA-256 provenance tracking for regulatory compliance
- FastAPI-based feature serving endpoints

Components:
    feast_config: Feast feature store configuration and setup
    feature_definitions: Pydantic models for all feature types
    feature_pipeline: Feature engineering and transformation pipeline
    feature_server: FastAPI endpoints for feature serving

Example:
    >>> from greenlang.ml.feature_store import ProcessHeatFeatureStore
    >>> from greenlang.ml.feature_store import BoilerFeatures, CombustionFeatures
    >>>
    >>> # Initialize feature store
    >>> store = ProcessHeatFeatureStore()
    >>>
    >>> # Get online features for inference
    >>> features = store.get_online_features(
    ...     entity_ids=["boiler-001", "boiler-002"],
    ...     feature_refs=["boiler_features:efficiency", "boiler_features:steam_flow"]
    ... )
"""

__version__ = "1.0.0"

# Core configuration
from greenlang.ml.feature_store.feast_config import (
    ProcessHeatFeatureStore,
    FeatureStoreConfig,
    OnlineStoreConfig,
    OfflineStoreConfig,
)

# Feature definitions
from greenlang.ml.feature_store.feature_definitions import (
    # Entity definitions
    EquipmentEntity,
    FacilityEntity,
    # Feature groups
    BoilerFeatures,
    CombustionFeatures,
    SteamFeatures,
    EmissionsFeatures,
    PredictiveFeatures,
    # Provenance
    FeatureProvenance,
)

# Feature pipeline
from greenlang.ml.feature_store.feature_pipeline import (
    FeaturePipeline,
    RollingAggregation,
    LagFeatureConfig,
    StatisticalFeatures,
)

__all__ = [
    # Configuration
    "ProcessHeatFeatureStore",
    "FeatureStoreConfig",
    "OnlineStoreConfig",
    "OfflineStoreConfig",
    # Entities
    "EquipmentEntity",
    "FacilityEntity",
    # Features
    "BoilerFeatures",
    "CombustionFeatures",
    "SteamFeatures",
    "EmissionsFeatures",
    "PredictiveFeatures",
    # Provenance
    "FeatureProvenance",
    # Pipeline
    "FeaturePipeline",
    "RollingAggregation",
    "LagFeatureConfig",
    "StatisticalFeatures",
]
