"""
MLPipelineManager - Production ML Inference Pipeline for GL-004 BURNMASTER

This module implements the ML pipeline manager for combustion optimization.
Supports multiple model types, A/B testing, batch and real-time inference,
and automatic fallback to physics-based calculations.

Key Features:
    - Model lifecycle management (load, serve, retire)
    - Multiple model type support (maintenance, anomaly, efficiency)
    - A/B testing with configurable traffic splits
    - Batch and real-time inference modes
    - Inference caching for performance
    - Physics-based fallback when ML unavailable
    - Complete provenance tracking

CRITICAL: ML predictions are ADVISORY ONLY.
Control decisions must use deterministic physics-based calculations.

Example:
    >>> config = MLPipelineConfig(model_registry_path=Path("./models"))
    >>> pipeline = MLPipelineManager(config)
    >>> await pipeline.initialize()
    >>>
    >>> request = InferenceRequest(
    ...     prediction_type=PredictionType.EFFICIENCY,
    ...     features={"o2_percent": 3.0, "fuel_flow": 100.0}
    ... )
    >>> response = await pipeline.predict(request)
    >>> print(f"Efficiency: {response.prediction}% (confidence: {response.confidence})")

Author: GreenLang ML Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class PredictionType(str, Enum):
    """Types of ML predictions supported."""
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    ANOMALY_DETECTION = "anomaly_detection"
    EFFICIENCY_PREDICTION = "efficiency_prediction"
    STABILITY_PREDICTION = "stability_prediction"
    EMISSIONS_PREDICTION = "emissions_prediction"


class ModelStatus(str, Enum):
    """Model deployment status."""
    LOADING = "loading"
    ACTIVE = "active"
    SHADOW = "shadow"  # A/B testing - receives traffic but predictions not used
    RETIRING = "retiring"
    RETIRED = "retired"
    FAILED = "failed"


class InferenceMode(str, Enum):
    """Inference execution mode."""
    REALTIME = "realtime"
    BATCH = "batch"
    ASYNC = "async"


class FallbackReason(str, Enum):
    """Reasons for falling back to physics-based calculation."""
    MODEL_NOT_LOADED = "model_not_loaded"
    MODEL_FAILED = "model_failed"
    LOW_CONFIDENCE = "low_confidence"
    FEATURE_MISSING = "feature_missing"
    TIMEOUT = "timeout"
    CACHE_HIT = "cache_hit"


# Performance thresholds
DEFAULT_INFERENCE_TIMEOUT_MS = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_CACHE_TTL_SECONDS = 60
MAX_BATCH_SIZE = 1000


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class MLPipelineConfig(BaseModel):
    """Configuration for ML Pipeline Manager."""

    model_registry_path: Path = Field(
        default=Path("./model_registry"),
        description="Path to model registry storage"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable inference result caching"
    )
    cache_ttl_seconds: int = Field(
        default=DEFAULT_CACHE_TTL_SECONDS,
        ge=1,
        le=3600,
        description="Cache time-to-live in seconds"
    )
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum cache entries"
    )
    inference_timeout_ms: int = Field(
        default=DEFAULT_INFERENCE_TIMEOUT_MS,
        ge=10,
        le=5000,
        description="Inference timeout in milliseconds"
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for ML prediction use"
    )
    enable_ab_testing: bool = Field(
        default=False,
        description="Enable A/B testing for model comparison"
    )
    ab_traffic_split: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Traffic percentage to shadow model"
    )
    max_batch_size: int = Field(
        default=MAX_BATCH_SIZE,
        ge=1,
        le=10000,
        description="Maximum batch size for batch inference"
    )
    fallback_to_physics: bool = Field(
        default=True,
        description="Fall back to physics-based calculation on ML failure"
    )
    log_all_predictions: bool = Field(
        default=True,
        description="Log all predictions for monitoring"
    )


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    model_id: str = Field(..., description="Unique model identifier")
    model_type: PredictionType = Field(..., description="Type of prediction")
    version: str = Field(default="1.0.0", description="Model version")
    status: ModelStatus = Field(default=ModelStatus.LOADING, description="Current status")
    loaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When model was loaded"
    )
    last_inference_at: Optional[datetime] = Field(
        default=None,
        description="Last inference timestamp"
    )
    inference_count: int = Field(default=0, ge=0, description="Total inferences")
    avg_latency_ms: float = Field(default=0.0, ge=0.0, description="Average latency")
    error_count: int = Field(default=0, ge=0, description="Error count")
    accuracy_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Accuracy metrics"
    )
    feature_names: List[str] = Field(
        default_factory=list,
        description="Expected feature names"
    )
    file_path: Optional[str] = Field(default=None, description="Model file path")
    file_hash: Optional[str] = Field(default=None, description="Model file SHA-256")


class InferenceRequest(BaseModel):
    """Request for ML inference."""

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier"
    )
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    features: Dict[str, float] = Field(..., description="Input features")
    equipment_id: str = Field(default="BNR-001", description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    mode: InferenceMode = Field(
        default=InferenceMode.REALTIME,
        description="Inference mode"
    )
    require_explanation: bool = Field(
        default=False,
        description="Include feature importance"
    )
    use_cache: bool = Field(default=True, description="Allow cached results")
    timeout_ms: Optional[int] = Field(default=None, description="Custom timeout")


class InferenceResponse(BaseModel):
    """Response from ML inference."""

    request_id: str = Field(..., description="Request identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    prediction: Union[float, Dict[str, Any]] = Field(..., description="Prediction result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0, description="Prediction uncertainty")
    model_id: str = Field(..., description="Model used for prediction")
    model_version: str = Field(default="1.0.0", description="Model version")
    is_fallback: bool = Field(default=False, description="Used physics fallback")
    fallback_reason: Optional[FallbackReason] = Field(
        default=None,
        description="Reason for fallback if applicable"
    )
    latency_ms: float = Field(..., ge=0.0, description="Inference latency")
    from_cache: bool = Field(default=False, description="Result from cache")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance if requested"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")


class BatchInferenceRequest(BaseModel):
    """Request for batch inference."""

    batch_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique batch identifier"
    )
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    samples: List[Dict[str, float]] = Field(..., description="List of feature samples")
    equipment_id: str = Field(default="BNR-001", description="Equipment identifier")
    parallel: bool = Field(default=True, description="Process in parallel")


class BatchInferenceResponse(BaseModel):
    """Response from batch inference."""

    batch_id: str = Field(..., description="Batch identifier")
    predictions: List[InferenceResponse] = Field(..., description="Individual predictions")
    total_count: int = Field(..., ge=0, description="Total samples processed")
    success_count: int = Field(..., ge=0, description="Successful predictions")
    fallback_count: int = Field(..., ge=0, description="Fallback predictions")
    total_latency_ms: float = Field(..., ge=0.0, description="Total processing time")
    avg_latency_ms: float = Field(..., ge=0.0, description="Average per-sample latency")


# =============================================================================
# INFERENCE CACHE
# =============================================================================


@dataclass
class CacheEntry:
    """Cache entry with TTL tracking."""
    key: str
    response: InferenceResponse
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds


class InferenceCache:
    """LRU cache for inference results with TTL."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 60):
        """Initialize cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[InferenceResponse]:
        """Get cached response if available and not expired."""
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]
        if entry.is_expired(self.ttl_seconds):
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            return None

        # Update access order (LRU)
        entry.access_count += 1
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self._hits += 1
        return entry.response

    def put(self, key: str, response: InferenceResponse) -> None:
        """Store response in cache."""
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = CacheEntry(key=key, response=response)
        self._access_order.append(key)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        return count

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }


# =============================================================================
# PHYSICS-BASED FALLBACK CALCULATORS
# =============================================================================


class PhysicsCalculator:
    """
    Deterministic physics-based calculations for fallback.

    These are used when ML models are unavailable or predictions
    have low confidence. All calculations are DETERMINISTIC.
    """

    @staticmethod
    def calculate_efficiency(features: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate combustion efficiency using physics equations.

        DETERMINISTIC: Uses stack loss method.

        Args:
            features: Dict with o2_percent, stack_temp_c, ambient_temp_c, fuel_type

        Returns:
            (efficiency_percent, confidence)
        """
        o2_pct = features.get("o2_percent", 3.0)
        stack_temp = features.get("stack_temp_c", 200.0)
        ambient_temp = features.get("ambient_temp_c", 25.0)

        # Excess air from O2
        if o2_pct >= 21.0:
            excess_air = 5.0  # Maximum practical value
        else:
            excess_air = (o2_pct / (21.0 - o2_pct)) * 100.0

        # Stack loss (simplified Siegert formula for natural gas)
        temp_diff = stack_temp - ambient_temp
        k1 = 0.38  # Fuel-specific constant for natural gas
        k2 = 0.007  # CO2 factor

        dry_flue_loss = k1 * temp_diff * (1 + excess_air / 100)

        # Latent heat loss (moisture in fuel)
        h2_content = features.get("fuel_h2_percent", 23.0)  # Natural gas approx
        latent_loss = 0.09 * h2_content

        # CO loss (incomplete combustion)
        co_ppm = features.get("co_ppm", 50.0)
        co_loss = co_ppm * 0.001  # Approximate

        # Radiation loss (typically 0.5-2% for well-insulated)
        radiation_loss = features.get("radiation_loss_percent", 1.0)

        # Total efficiency
        total_loss = dry_flue_loss + latent_loss + co_loss + radiation_loss
        efficiency = 100.0 - total_loss
        efficiency = max(50.0, min(99.9, efficiency))  # Physical bounds

        # Confidence based on data completeness
        required_features = ["o2_percent", "stack_temp_c"]
        present = sum(1 for f in required_features if f in features)
        confidence = 0.7 + (present / len(required_features)) * 0.2

        return round(efficiency, 2), round(confidence, 3)

    @staticmethod
    def calculate_failure_probability(features: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate equipment failure probability using physics rules.

        DETERMINISTIC: Uses empirical failure curves.

        Args:
            features: Equipment operational parameters

        Returns:
            (failure_probability, confidence)
        """
        # Operating hours factor
        hours = features.get("operating_hours", 0)
        hours_factor = min(1.0, hours / 50000)  # Increases with age

        # Temperature stress factor
        max_temp = features.get("max_flame_temp_c", 1500)
        design_temp = features.get("design_temp_c", 1800)
        temp_stress = max(0, (max_temp - design_temp * 0.8)) / (design_temp * 0.2)
        temp_factor = min(1.0, temp_stress) if temp_stress > 0 else 0.0

        # Vibration factor
        vibration = features.get("vibration_rms", 0)
        vibration_threshold = features.get("vibration_threshold", 10)
        vibration_factor = min(1.0, vibration / vibration_threshold) if vibration_threshold > 0 else 0.0

        # Cycle count factor
        cycles = features.get("start_stop_cycles", 0)
        max_cycles = features.get("rated_cycles", 10000)
        cycle_factor = min(1.0, cycles / max_cycles) if max_cycles > 0 else 0.0

        # Weighted failure probability
        weights = {
            "hours": 0.3,
            "temperature": 0.25,
            "vibration": 0.25,
            "cycles": 0.2,
        }

        failure_prob = (
            weights["hours"] * hours_factor +
            weights["temperature"] * temp_factor +
            weights["vibration"] * vibration_factor +
            weights["cycles"] * cycle_factor
        )

        failure_prob = max(0.0, min(1.0, failure_prob))
        confidence = 0.6  # Lower confidence for physics-based maintenance prediction

        return round(failure_prob, 4), round(confidence, 3)

    @staticmethod
    def detect_anomaly_threshold(features: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Detect anomalies using threshold-based rules.

        DETERMINISTIC: Uses predefined thresholds.

        Args:
            features: Current sensor readings

        Returns:
            (is_anomaly, anomaly_score, anomaly_type)
        """
        anomalies = []
        max_score = 0.0

        # O2 bounds check
        o2 = features.get("o2_percent", 3.0)
        if o2 < 1.0 or o2 > 8.0:
            score = min(1.0, abs(o2 - 3.0) / 5.0)
            anomalies.append(("o2_out_of_range", score))
            max_score = max(max_score, score)

        # CO spike detection
        co = features.get("co_ppm", 50)
        if co > 200:
            score = min(1.0, (co - 200) / 300)
            anomalies.append(("high_co", score))
            max_score = max(max_score, score)

        # Flame temperature deviation
        flame_temp = features.get("flame_temp_c", 1500)
        if flame_temp < 1200 or flame_temp > 1900:
            score = min(1.0, abs(flame_temp - 1500) / 400)
            anomalies.append(("flame_temp_deviation", score))
            max_score = max(max_score, score)

        # Pressure oscillation
        pressure_var = features.get("pressure_variance", 0)
        if pressure_var > 5.0:
            score = min(1.0, pressure_var / 20.0)
            anomalies.append(("pressure_oscillation", score))
            max_score = max(max_score, score)

        # Lambda deviation
        lambda_val = features.get("lambda", 1.15)
        if lambda_val < 1.02 or lambda_val > 1.5:
            score = min(1.0, abs(lambda_val - 1.15) / 0.35)
            anomalies.append(("lambda_deviation", score))
            max_score = max(max_score, score)

        is_anomaly = max_score > 0.3
        anomaly_type = anomalies[0][0] if anomalies else "none"

        return is_anomaly, round(max_score, 4), anomaly_type


# =============================================================================
# ML PIPELINE MANAGER
# =============================================================================


class MLPipelineManager:
    """
    Production ML inference pipeline for combustion optimization.

    Manages model lifecycle, inference, caching, and fallback.
    Supports A/B testing and multiple model types.

    CRITICAL: All ML predictions are ADVISORY ONLY.
    Control decisions must use deterministic physics-based calculations.

    Attributes:
        config: Pipeline configuration
        models: Loaded model registry
        cache: Inference result cache
        physics: Physics-based fallback calculator

    Example:
        >>> config = MLPipelineConfig()
        >>> pipeline = MLPipelineManager(config)
        >>> await pipeline.initialize()
        >>>
        >>> request = InferenceRequest(
        ...     prediction_type=PredictionType.EFFICIENCY,
        ...     features={"o2_percent": 3.0}
        ... )
        >>> response = await pipeline.predict(request)
    """

    def __init__(self, config: Optional[MLPipelineConfig] = None):
        """
        Initialize MLPipelineManager.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or MLPipelineConfig()
        self._models: Dict[str, Any] = {}  # model_id -> actual model object
        self._model_info: Dict[str, ModelInfo] = {}  # model_id -> metadata
        self._active_models: Dict[PredictionType, str] = {}  # type -> active model_id
        self._shadow_models: Dict[PredictionType, str] = {}  # type -> shadow model_id for A/B
        self._cache: Optional[InferenceCache] = None
        self._physics = PhysicsCalculator()
        self._initialized = False
        self._prediction_log: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

        # Metrics
        self._total_predictions = 0
        self._ml_predictions = 0
        self._fallback_predictions = 0
        self._cache_hits = 0
        self._errors = 0

        logger.info(
            f"MLPipelineManager created with config: "
            f"cache={self.config.enable_caching}, "
            f"ab_testing={self.config.enable_ab_testing}, "
            f"fallback={self.config.fallback_to_physics}"
        )

    async def initialize(self) -> None:
        """
        Initialize the pipeline and load models.

        This should be called before making predictions.
        """
        if self._initialized:
            logger.warning("Pipeline already initialized")
            return

        start_time = time.time()

        # Initialize cache
        if self.config.enable_caching:
            self._cache = InferenceCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )

        # Create registry directory
        self.config.model_registry_path.mkdir(parents=True, exist_ok=True)

        # Load any pre-trained models from registry
        await self._load_models_from_registry()

        self._initialized = True
        elapsed = (time.time() - start_time) * 1000

        logger.info(
            f"MLPipelineManager initialized in {elapsed:.1f}ms, "
            f"models_loaded={len(self._models)}"
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        logger.info("Shutting down MLPipelineManager...")

        # Clear cache
        if self._cache:
            cleared = self._cache.clear()
            logger.info(f"Cleared {cleared} cache entries")

        # Clear models
        self._models.clear()
        self._model_info.clear()
        self._active_models.clear()

        self._initialized = False
        logger.info("MLPipelineManager shutdown complete")

    async def register_model(
        self,
        model: Any,
        model_type: PredictionType,
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        feature_names: Optional[List[str]] = None,
        set_active: bool = True
    ) -> str:
        """
        Register a model with the pipeline.

        Args:
            model: The trained model object
            model_type: Type of prediction this model makes
            model_id: Unique identifier (auto-generated if not provided)
            version: Model version string
            feature_names: Expected feature names
            set_active: Set as active model for this type

        Returns:
            Model ID
        """
        async with self._lock:
            if model_id is None:
                model_id = f"{model_type.value}_{uuid4().hex[:8]}"

            # Save model to file
            model_path = self.config.model_registry_path / f"{model_id}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Compute file hash
            with open(model_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Create model info
            info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                version=version,
                status=ModelStatus.ACTIVE if set_active else ModelStatus.LOADING,
                feature_names=feature_names or [],
                file_path=str(model_path),
                file_hash=file_hash
            )

            self._models[model_id] = model
            self._model_info[model_id] = info

            if set_active:
                # Retire previous active model
                if model_type in self._active_models:
                    old_id = self._active_models[model_type]
                    if old_id in self._model_info:
                        self._model_info[old_id].status = ModelStatus.RETIRED

                self._active_models[model_type] = model_id

            logger.info(
                f"Model registered: id={model_id}, type={model_type.value}, "
                f"version={version}, active={set_active}"
            )

            return model_id

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """
        Make a prediction using the appropriate model.

        This method handles:
        - Cache lookup (if enabled)
        - Model selection (active or shadow for A/B)
        - Timeout enforcement
        - Fallback to physics on failure
        - Provenance tracking

        Args:
            request: Inference request

        Returns:
            Inference response with prediction
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        start_time = time.time()
        self._total_predictions += 1

        # Check cache first
        if request.use_cache and self._cache:
            cache_key = self._compute_cache_key(request)
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_hits += 1
                cached_response = cached.model_copy()
                cached_response.from_cache = True
                cached_response.latency_ms = (time.time() - start_time) * 1000
                return cached_response

        # Get active model for this prediction type
        model_id = self._active_models.get(request.prediction_type)

        response: Optional[InferenceResponse] = None

        if model_id and model_id in self._models:
            try:
                # Attempt ML prediction with timeout
                timeout_ms = request.timeout_ms or self.config.inference_timeout_ms
                response = await asyncio.wait_for(
                    self._predict_with_model(request, model_id),
                    timeout=timeout_ms / 1000
                )

                # Check confidence threshold
                if response.confidence < self.config.confidence_threshold:
                    logger.warning(
                        f"Low confidence prediction ({response.confidence:.3f}), "
                        f"threshold={self.config.confidence_threshold}"
                    )
                    if self.config.fallback_to_physics:
                        response = await self._fallback_prediction(
                            request, FallbackReason.LOW_CONFIDENCE
                        )
                    else:
                        response.warnings.append("Low confidence prediction")
                else:
                    self._ml_predictions += 1

            except asyncio.TimeoutError:
                logger.warning(f"Inference timeout for {request.prediction_type.value}")
                if self.config.fallback_to_physics:
                    response = await self._fallback_prediction(
                        request, FallbackReason.TIMEOUT
                    )

            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)
                self._errors += 1
                if self.config.fallback_to_physics:
                    response = await self._fallback_prediction(
                        request, FallbackReason.MODEL_FAILED
                    )
        else:
            # No model available
            if self.config.fallback_to_physics:
                response = await self._fallback_prediction(
                    request, FallbackReason.MODEL_NOT_LOADED
                )
            else:
                raise ValueError(f"No model available for {request.prediction_type.value}")

        # Update timing
        latency = (time.time() - start_time) * 1000
        response.latency_ms = latency

        # Compute provenance hash
        response.provenance_hash = self._compute_provenance_hash(request, response)

        # Cache result
        if self.config.enable_caching and self._cache and not response.from_cache:
            cache_key = self._compute_cache_key(request)
            self._cache.put(cache_key, response)

        # A/B testing: also run shadow model
        if (self.config.enable_ab_testing and
            request.prediction_type in self._shadow_models):
            asyncio.create_task(self._run_shadow_inference(request))

        # Log prediction
        if self.config.log_all_predictions:
            self._log_prediction(request, response)

        # Update model stats
        if response.model_id in self._model_info:
            info = self._model_info[response.model_id]
            info.inference_count += 1
            info.last_inference_at = datetime.now(timezone.utc)
            # Update running average latency
            n = info.inference_count
            info.avg_latency_ms = (info.avg_latency_ms * (n - 1) + latency) / n

        return response

    async def predict_batch(
        self, request: BatchInferenceRequest
    ) -> BatchInferenceResponse:
        """
        Make predictions for a batch of samples.

        Args:
            request: Batch inference request

        Returns:
            Batch inference response
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")

        if len(request.samples) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(request.samples)} exceeds max {self.config.max_batch_size}"
            )

        start_time = time.time()
        predictions: List[InferenceResponse] = []
        success_count = 0
        fallback_count = 0

        if request.parallel:
            # Parallel processing
            tasks = [
                self.predict(InferenceRequest(
                    prediction_type=request.prediction_type,
                    features=sample,
                    equipment_id=request.equipment_id,
                    mode=InferenceMode.BATCH,
                    use_cache=False  # Disable cache for batch
                ))
                for sample in request.samples
            ]
            predictions = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # Sequential processing
            for sample in request.samples:
                pred = await self.predict(InferenceRequest(
                    prediction_type=request.prediction_type,
                    features=sample,
                    equipment_id=request.equipment_id,
                    mode=InferenceMode.BATCH,
                    use_cache=False
                ))
                predictions.append(pred)

        # Count successes and fallbacks
        for pred in predictions:
            if pred.is_fallback:
                fallback_count += 1
            else:
                success_count += 1

        total_latency = (time.time() - start_time) * 1000

        return BatchInferenceResponse(
            batch_id=request.batch_id,
            predictions=predictions,
            total_count=len(request.samples),
            success_count=success_count,
            fallback_count=fallback_count,
            total_latency_ms=total_latency,
            avg_latency_ms=total_latency / len(request.samples) if request.samples else 0
        )

    async def _predict_with_model(
        self, request: InferenceRequest, model_id: str
    ) -> InferenceResponse:
        """Make prediction using a specific model."""
        model = self._models[model_id]
        model_info = self._model_info[model_id]

        # Prepare features as numpy array
        feature_names = model_info.feature_names
        if feature_names:
            features_array = np.array([
                request.features.get(name, 0.0) for name in feature_names
            ]).reshape(1, -1)
        else:
            features_array = np.array(list(request.features.values())).reshape(1, -1)

        # Make prediction
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_array)
            prediction = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            confidence = float(max(proba[0]))
        elif hasattr(model, "predict"):
            prediction = float(model.predict(features_array)[0])
            confidence = 0.8  # Default confidence for regression
        else:
            raise ValueError(f"Model {model_id} has no predict method")

        # Feature importance if requested
        feature_importance = None
        if request.require_explanation and hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(
                feature_names or [f"f{i}" for i in range(len(model.feature_importances_))],
                [float(v) for v in model.feature_importances_]
            ))

        return InferenceResponse(
            request_id=request.request_id,
            prediction_type=request.prediction_type,
            prediction=prediction,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            model_id=model_id,
            model_version=model_info.version,
            is_fallback=False,
            latency_ms=0.0,  # Will be set by caller
            feature_importance=feature_importance
        )

    async def _fallback_prediction(
        self, request: InferenceRequest, reason: FallbackReason
    ) -> InferenceResponse:
        """Make prediction using physics-based fallback."""
        self._fallback_predictions += 1

        if request.prediction_type == PredictionType.EFFICIENCY_PREDICTION:
            prediction, confidence = self._physics.calculate_efficiency(request.features)
        elif request.prediction_type == PredictionType.PREDICTIVE_MAINTENANCE:
            prediction, confidence = self._physics.calculate_failure_probability(request.features)
        elif request.prediction_type == PredictionType.ANOMALY_DETECTION:
            is_anomaly, score, anomaly_type = self._physics.detect_anomaly_threshold(request.features)
            prediction = {"is_anomaly": is_anomaly, "score": score, "type": anomaly_type}
            confidence = 0.65  # Lower confidence for threshold-based
        else:
            # Generic fallback
            prediction = 0.5
            confidence = 0.5

        return InferenceResponse(
            request_id=request.request_id,
            prediction_type=request.prediction_type,
            prediction=prediction,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            model_id="physics_fallback",
            model_version="1.0.0",
            is_fallback=True,
            fallback_reason=reason,
            latency_ms=0.0,
            warnings=[f"Used physics fallback: {reason.value}"]
        )

    async def _run_shadow_inference(self, request: InferenceRequest) -> None:
        """Run inference on shadow model for A/B testing."""
        try:
            shadow_id = self._shadow_models.get(request.prediction_type)
            if shadow_id and shadow_id in self._models:
                shadow_response = await self._predict_with_model(request, shadow_id)

                # Log shadow prediction for comparison
                logger.debug(
                    f"Shadow prediction: type={request.prediction_type.value}, "
                    f"model={shadow_id}, prediction={shadow_response.prediction}"
                )
        except Exception as e:
            logger.warning(f"Shadow inference failed: {e}")

    async def _load_models_from_registry(self) -> None:
        """Load models from the registry directory."""
        registry_path = self.config.model_registry_path
        if not registry_path.exists():
            return

        for model_file in registry_path.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)

                model_id = model_file.stem

                # Try to infer model type from ID
                model_type = None
                for ptype in PredictionType:
                    if ptype.value in model_id:
                        model_type = ptype
                        break

                if model_type:
                    self._models[model_id] = model
                    self._model_info[model_id] = ModelInfo(
                        model_id=model_id,
                        model_type=model_type,
                        status=ModelStatus.ACTIVE,
                        file_path=str(model_file)
                    )
                    self._active_models[model_type] = model_id
                    logger.info(f"Loaded model: {model_id}")

            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

    def _compute_cache_key(self, request: InferenceRequest) -> str:
        """Compute cache key for request."""
        key_data = {
            "type": request.prediction_type.value,
            "equipment": request.equipment_id,
            "features": sorted(request.features.items())
        }
        return hashlib.sha256(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:32]

    def _compute_provenance_hash(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> str:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "request_id": request.request_id,
            "prediction_type": request.prediction_type.value,
            "features": request.features,
            "prediction": str(response.prediction),
            "model_id": response.model_id,
            "is_fallback": response.is_fallback,
            "timestamp": response.timestamp.isoformat()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _log_prediction(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> None:
        """Log prediction for monitoring."""
        log_entry = {
            "request_id": request.request_id,
            "prediction_type": request.prediction_type.value,
            "equipment_id": request.equipment_id,
            "model_id": response.model_id,
            "prediction": response.prediction,
            "confidence": response.confidence,
            "is_fallback": response.is_fallback,
            "latency_ms": response.latency_ms,
            "from_cache": response.from_cache,
            "timestamp": response.timestamp.isoformat()
        }

        # Keep last 1000 predictions
        self._prediction_log.append(log_entry)
        if len(self._prediction_log) > 1000:
            self._prediction_log = self._prediction_log[-1000:]

    def set_shadow_model(
        self, prediction_type: PredictionType, model_id: str
    ) -> None:
        """Set a shadow model for A/B testing."""
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not found")

        self._shadow_models[prediction_type] = model_id
        self._model_info[model_id].status = ModelStatus.SHADOW

        logger.info(
            f"Shadow model set: type={prediction_type.value}, model={model_id}"
        )

    def promote_shadow_model(self, prediction_type: PredictionType) -> str:
        """Promote shadow model to active."""
        if prediction_type not in self._shadow_models:
            raise ValueError(f"No shadow model for {prediction_type.value}")

        shadow_id = self._shadow_models.pop(prediction_type)

        # Retire current active
        if prediction_type in self._active_models:
            old_id = self._active_models[prediction_type]
            self._model_info[old_id].status = ModelStatus.RETIRED

        # Promote shadow
        self._active_models[prediction_type] = shadow_id
        self._model_info[shadow_id].status = ModelStatus.ACTIVE

        logger.info(
            f"Shadow promoted: type={prediction_type.value}, model={shadow_id}"
        )

        return shadow_id

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self._model_info.get(model_id)

    def list_models(
        self, prediction_type: Optional[PredictionType] = None
    ) -> List[ModelInfo]:
        """List all registered models."""
        models = list(self._model_info.values())
        if prediction_type:
            models = [m for m in models if m.model_type == prediction_type]
        return models

    def get_active_model_id(self, prediction_type: PredictionType) -> Optional[str]:
        """Get the active model ID for a prediction type."""
        return self._active_models.get(prediction_type)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        cache_stats = self._cache.get_stats() if self._cache else {}

        return {
            "initialized": self._initialized,
            "total_predictions": self._total_predictions,
            "ml_predictions": self._ml_predictions,
            "fallback_predictions": self._fallback_predictions,
            "cache_hits": self._cache_hits,
            "errors": self._errors,
            "ml_rate": self._ml_predictions / self._total_predictions if self._total_predictions > 0 else 0,
            "fallback_rate": self._fallback_predictions / self._total_predictions if self._total_predictions > 0 else 0,
            "models_loaded": len(self._models),
            "active_models": dict(self._active_models),
            "shadow_models": dict(self._shadow_models),
            "cache_stats": cache_stats
        }

    def invalidate_cache(
        self, prediction_type: Optional[PredictionType] = None
    ) -> int:
        """Invalidate cache entries."""
        if not self._cache:
            return 0
        return self._cache.clear()

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._initialized
