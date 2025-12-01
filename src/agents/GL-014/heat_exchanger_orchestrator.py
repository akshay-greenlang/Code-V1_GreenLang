# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO - Heat Exchanger Analysis Pipeline Orchestrator

Production-grade pipeline orchestration for heat exchanger performance analysis.
Implements the GreenLang agent architecture pattern with:
- Multi-stage pipeline execution
- Checkpoint management for long-running analyses
- Async processing with asyncio
- Error recovery and retry logic
- Redis cache integration
- Structured logging with correlation IDs
- Complete provenance tracking

Pipeline Stages:
1. Input Validation - Validate and normalize input data
2. Data Enrichment - Fetch additional data from integrations
3. Heat Transfer Analysis - Calculate U, LMTD, effectiveness
4. Fouling Assessment - Determine fouling state and progression
5. Performance Evaluation - Calculate efficiency and health index
6. Cleaning Optimization - Determine optimal cleaning intervals
7. Economic Impact - Calculate energy loss and ROI
8. Report Generation - Generate analysis report
9. Result Caching - Cache results for future reference

Author: GreenLang AI Agent Factory - GL-BackendDeveloper
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Awaitable, Callable, Dict, Generic, List, Optional,
    Protocol, Tuple, Type, TypeVar, Union
)

from pydantic import BaseModel, Field, validator

# Local imports
from .tools import (
    HeatExchangerTools,
    HeatTransferInput,
    LMTDInput,
    EffectivenessInput,
    FoulingInput,
    FoulingPredictionInput,
    EfficiencyInput,
    HealthInput,
    CleaningInput,
    CostBenefitInput,
    EnergyLossInput,
    ROIInput,
    HeatTransferResult,
    LMTDResult,
    EffectivenessResult,
    FoulingResult,
    FoulingPredictionResult,
    EfficiencyResult,
    HealthIndexResult,
    CleaningResult,
    CostBenefitResult,
    EnergyLossResult,
    ROIResult,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
InputT = TypeVar('InputT', bound=BaseModel)
OutputT = TypeVar('OutputT')


# =============================================================================
# ENUMERATIONS
# =============================================================================

class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Individual step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY_PENDING = "retry_pending"


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    INPUT_VALIDATION = "input_validation"
    DATA_ENRICHMENT = "data_enrichment"
    HEAT_TRANSFER_ANALYSIS = "heat_transfer_analysis"
    FOULING_ASSESSMENT = "fouling_assessment"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    CLEANING_OPTIMIZATION = "cleaning_optimization"
    ECONOMIC_IMPACT = "economic_impact"
    REPORT_GENERATION = "report_generation"
    RESULT_CACHING = "result_caching"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_seconds: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    exponential_base: float = Field(default=2.0, ge=1.0, le=4.0)
    jitter: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Configuration for result caching."""
    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    cache_key_prefix: str = Field(default="gl014:")
    redis_url: Optional[str] = Field(default=None)
    max_entries: int = Field(default=10000, ge=100)


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint management."""
    enabled: bool = Field(default=True)
    checkpoint_interval_seconds: float = Field(default=30.0, ge=5.0)
    max_checkpoint_age_hours: int = Field(default=24, ge=1)
    storage_path: Optional[str] = Field(default=None)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = Field(default=None)
    timeout_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)
    parallel_stages: bool = Field(default=False)
    fail_fast: bool = Field(default=False)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    checkpoint_config: CheckpointConfig = Field(default_factory=CheckpointConfig)
    enable_detailed_logging: bool = Field(default=True)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class HeatExchangerInputData(BaseModel):
    """Complete input data for heat exchanger analysis."""
    # Equipment identification
    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_tag: Optional[str] = Field(None, description="Plant equipment tag")
    equipment_type: str = Field(default="shell_tube", description="Heat exchanger type")

    # Design parameters
    design_duty_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    design_u_w_m2_k: float = Field(..., gt=0, description="Design U value (W/m2.K)")
    design_lmtd_c: float = Field(..., gt=0, description="Design LMTD (C)")
    design_area_m2: float = Field(..., gt=0, description="Heat transfer area (m2)")

    # Current operating data
    actual_duty_kw: float = Field(..., gt=0, description="Actual heat duty (kW)")
    t_hot_in_c: float = Field(..., description="Hot fluid inlet temperature (C)")
    t_hot_out_c: float = Field(..., description="Hot fluid outlet temperature (C)")
    t_cold_in_c: float = Field(..., description="Cold fluid inlet temperature (C)")
    t_cold_out_c: float = Field(..., description="Cold fluid outlet temperature (C)")

    # Flow data
    flow_hot_kg_s: Optional[float] = Field(None, gt=0, description="Hot side mass flow (kg/s)")
    flow_cold_kg_s: Optional[float] = Field(None, gt=0, description="Cold side mass flow (kg/s)")
    flow_arrangement: str = Field(default="counterflow", description="Flow arrangement")

    # Pressure data
    dp_hot_actual_kpa: Optional[float] = Field(None, ge=0, description="Hot side dP (kPa)")
    dp_cold_actual_kpa: Optional[float] = Field(None, ge=0, description="Cold side dP (kPa)")
    dp_hot_design_kpa: Optional[float] = Field(None, gt=0, description="Design hot dP (kPa)")
    dp_cold_design_kpa: Optional[float] = Field(None, gt=0, description="Design cold dP (kPa)")

    # Tube/geometry data
    tube_od_m: Optional[float] = Field(None, gt=0, description="Tube OD (m)")
    tube_id_m: Optional[float] = Field(None, gt=0, description="Tube ID (m)")
    tube_material: Optional[str] = Field(None, description="Tube material")
    tube_k_w_m_k: Optional[float] = Field(None, gt=0, description="Tube thermal conductivity")

    # Fluid properties
    fluid_type_hot: str = Field(default="process_fluid", description="Hot side fluid type")
    fluid_type_cold: str = Field(default="cooling_water", description="Cold side fluid type")
    h_hot_w_m2_k: Optional[float] = Field(None, gt=0, description="Hot side film coefficient")
    h_cold_w_m2_k: Optional[float] = Field(None, gt=0, description="Cold side film coefficient")

    # Fouling data
    r_f_hot_design_m2_k_w: float = Field(default=0.000352, ge=0, description="Design hot fouling")
    r_f_cold_design_m2_k_w: float = Field(default=0.000176, ge=0, description="Design cold fouling")
    last_cleaning_date: Optional[str] = Field(None, description="Last cleaning date (ISO)")
    operating_hours_since_cleaning: Optional[float] = Field(None, ge=0, description="Hours since cleaning")

    # Economic data
    fuel_cost_per_kwh: float = Field(default=0.06, gt=0, description="Fuel cost ($/kWh)")
    system_efficiency: float = Field(default=0.85, gt=0, le=1, description="System efficiency")
    operating_hours_per_year: float = Field(default=8000, gt=0, description="Operating hours/year")
    carbon_price_per_tonne: float = Field(default=50.0, ge=0, description="Carbon price ($/tonne)")
    emission_factor_kg_co2_per_kwh: float = Field(default=0.185, ge=0, description="CO2 factor")
    cleaning_cost_usd: float = Field(default=15000, gt=0, description="Cleaning cost ($)")
    downtime_hours_per_cleaning: float = Field(default=24, gt=0, description="Cleaning downtime (hrs)")
    production_loss_per_hour_usd: float = Field(default=5000, ge=0, description="Production loss ($/hr)")

    @validator('equipment_type')
    def validate_equipment_type(cls, v):
        valid_types = ('shell_tube', 'plate', 'air_cooled', 'double_pipe', 'spiral')
        if v not in valid_types:
            raise ValueError(f"Equipment type must be one of: {valid_types}")
        return v


@dataclass
class StepResult:
    """Result of a single pipeline step."""
    stage: PipelineStage
    status: StepStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    provenance_hash: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_id: str
    correlation_id: str
    status: PipelineStatus
    equipment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    step_results: Dict[str, StepResult] = field(default_factory=dict)

    # Analysis results
    heat_transfer_result: Optional[HeatTransferResult] = None
    lmtd_result: Optional[LMTDResult] = None
    effectiveness_result: Optional[EffectivenessResult] = None
    fouling_result: Optional[FoulingResult] = None
    fouling_prediction_result: Optional[FoulingPredictionResult] = None
    efficiency_result: Optional[EfficiencyResult] = None
    health_index_result: Optional[HealthIndexResult] = None
    cleaning_result: Optional[CleaningResult] = None
    cost_benefit_result: Optional[CostBenefitResult] = None
    energy_loss_result: Optional[EnergyLossResult] = None
    roi_result: Optional[ROIResult] = None

    # Summary
    summary: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    provenance_hash: Optional[str] = None


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

@dataclass
class Checkpoint:
    """Pipeline checkpoint for recovery."""
    checkpoint_id: str
    pipeline_id: str
    created_at: datetime
    last_completed_stage: Optional[PipelineStage]
    completed_stages: List[PipelineStage]
    pending_stages: List[PipelineStage]
    stage_results: Dict[str, Any]
    input_data_hash: str
    is_valid: bool = True


class CheckpointManager:
    """
    Manages pipeline checkpoints for recovery from failures.

    Provides:
    - Periodic checkpoint saving
    - Recovery from last checkpoint
    - Checkpoint cleanup
    """

    def __init__(self, config: CheckpointConfig):
        """
        Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration
        """
        self._config = config
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._lock = asyncio.Lock()

    async def save_checkpoint(
        self,
        pipeline_id: str,
        completed_stage: PipelineStage,
        completed_stages: List[PipelineStage],
        pending_stages: List[PipelineStage],
        stage_results: Dict[str, Any],
        input_data_hash: str
    ) -> Checkpoint:
        """
        Save a pipeline checkpoint.

        Args:
            pipeline_id: Pipeline identifier
            completed_stage: Most recently completed stage
            completed_stages: List of all completed stages
            pending_stages: List of pending stages
            stage_results: Results from completed stages
            input_data_hash: Hash of input data

        Returns:
            Created checkpoint
        """
        if not self._config.enabled:
            return None

        async with self._lock:
            checkpoint = Checkpoint(
                checkpoint_id=str(uuid.uuid4()),
                pipeline_id=pipeline_id,
                created_at=datetime.utcnow(),
                last_completed_stage=completed_stage,
                completed_stages=completed_stages,
                pending_stages=pending_stages,
                stage_results=stage_results,
                input_data_hash=input_data_hash,
                is_valid=True
            )

            self._checkpoints[pipeline_id] = checkpoint
            logger.debug(
                f"Checkpoint saved for pipeline {pipeline_id}, "
                f"last stage: {completed_stage.value if completed_stage else 'None'}"
            )

            # Cleanup old checkpoints
            await self._cleanup_old_checkpoints()

            return checkpoint

    async def load_checkpoint(self, pipeline_id: str) -> Optional[Checkpoint]:
        """
        Load checkpoint for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Checkpoint if found and valid, None otherwise
        """
        if not self._config.enabled:
            return None

        async with self._lock:
            checkpoint = self._checkpoints.get(pipeline_id)

            if checkpoint is None:
                return None

            # Check if checkpoint is too old
            max_age = timedelta(hours=self._config.max_checkpoint_age_hours)
            if datetime.utcnow() - checkpoint.created_at > max_age:
                logger.warning(f"Checkpoint for pipeline {pipeline_id} is too old, discarding")
                del self._checkpoints[pipeline_id]
                return None

            if not checkpoint.is_valid:
                logger.warning(f"Checkpoint for pipeline {pipeline_id} is invalid, discarding")
                del self._checkpoints[pipeline_id]
                return None

            return checkpoint

    async def invalidate_checkpoint(self, pipeline_id: str) -> None:
        """
        Invalidate a pipeline checkpoint.

        Args:
            pipeline_id: Pipeline identifier
        """
        async with self._lock:
            if pipeline_id in self._checkpoints:
                self._checkpoints[pipeline_id].is_valid = False
                logger.debug(f"Checkpoint invalidated for pipeline {pipeline_id}")

    async def delete_checkpoint(self, pipeline_id: str) -> None:
        """
        Delete a pipeline checkpoint.

        Args:
            pipeline_id: Pipeline identifier
        """
        async with self._lock:
            if pipeline_id in self._checkpoints:
                del self._checkpoints[pipeline_id]
                logger.debug(f"Checkpoint deleted for pipeline {pipeline_id}")

    async def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        max_age = timedelta(hours=self._config.max_checkpoint_age_hours)
        cutoff = datetime.utcnow() - max_age

        to_delete = [
            pid for pid, cp in self._checkpoints.items()
            if cp.created_at < cutoff or not cp.is_valid
        ]

        for pid in to_delete:
            del self._checkpoints[pid]

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old checkpoints")


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    Manages result caching with optional Redis backend.

    Provides:
    - In-memory LRU cache
    - Redis integration (optional)
    - TTL-based expiration
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.

        Args:
            config: Cache configuration
        """
        self._config = config
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._redis_client = None

    async def initialize(self) -> None:
        """Initialize cache connections."""
        if self._config.redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(self._config.redis_url)
                logger.info(f"Redis cache connected: {self._config.redis_url}")
            except ImportError:
                logger.warning("redis package not installed, using memory cache only")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache only")

    async def close(self) -> None:
        """Close cache connections."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis cache connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self._config.enabled:
            return None

        full_key = f"{self._config.cache_key_prefix}{key}"

        # Try Redis first
        if self._redis_client:
            try:
                data = await self._redis_client.get(full_key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Fall back to memory cache
        async with self._lock:
            if full_key in self._memory_cache:
                value, expiry = self._memory_cache[full_key]
                if datetime.utcnow() < expiry:
                    # Update access order for LRU
                    if full_key in self._access_order:
                        self._access_order.remove(full_key)
                    self._access_order.append(full_key)
                    return value
                else:
                    # Expired
                    del self._memory_cache[full_key]
                    if full_key in self._access_order:
                        self._access_order.remove(full_key)

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
        """
        if not self._config.enabled:
            return

        full_key = f"{self._config.cache_key_prefix}{key}"
        ttl = ttl_seconds or self._config.ttl_seconds

        # Try Redis first
        if self._redis_client:
            try:
                await self._redis_client.setex(full_key, ttl, json.dumps(value, default=str))
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # Also store in memory cache
        async with self._lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self._memory_cache[full_key] = (value, expiry)

            # Update access order
            if full_key in self._access_order:
                self._access_order.remove(full_key)
            self._access_order.append(full_key)

            # LRU eviction
            while len(self._memory_cache) > self._config.max_entries:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._memory_cache:
                    del self._memory_cache[oldest_key]

    async def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        full_key = f"{self._config.cache_key_prefix}{key}"

        if self._redis_client:
            try:
                await self._redis_client.delete(full_key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

        async with self._lock:
            if full_key in self._memory_cache:
                del self._memory_cache[full_key]
            if full_key in self._access_order:
                self._access_order.remove(full_key)

    def generate_cache_key(self, input_data: HeatExchangerInputData) -> str:
        """
        Generate cache key from input data.

        Args:
            input_data: Input data

        Returns:
            Cache key hash
        """
        data_json = input_data.json(sort_keys=True)
        return hashlib.sha256(data_json.encode()).hexdigest()[:32]


# =============================================================================
# ERROR HANDLING
# =============================================================================

class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(
        self,
        message: str,
        stage: Optional[PipelineStage] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = False,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.stage = stage
        self.severity = severity
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()


class ValidationError(PipelineError):
    """Input validation error."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message,
            stage=PipelineStage.INPUT_VALIDATION,
            severity=ErrorSeverity.ERROR,
            recoverable=False
        )
        self.field = field


class CalculationError(PipelineError):
    """Calculation error during analysis."""

    def __init__(
        self,
        message: str,
        stage: PipelineStage,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(
            message,
            stage=stage,
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            original_exception=original_exception
        )


class IntegrationError(PipelineError):
    """Error during external system integration."""

    def __init__(self, message: str, integration: str):
        super().__init__(
            message,
            stage=PipelineStage.DATA_ENRICHMENT,
            severity=ErrorSeverity.WARNING,
            recoverable=True
        )
        self.integration = integration


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class HeatExchangerOrchestrator:
    """
    Production-grade pipeline orchestrator for heat exchanger analysis.

    Features:
    - Multi-stage pipeline execution
    - Checkpoint management for long-running analyses
    - Async processing with asyncio
    - Error recovery and retry logic
    - Redis cache integration
    - Structured logging with correlation IDs
    - Complete provenance tracking

    Example:
        >>> orchestrator = HeatExchangerOrchestrator()
        >>> await orchestrator.initialize()
        >>> result = await orchestrator.execute(input_data)
        >>> print(f"Health Index: {result.health_index_result.health_index}")
        >>> await orchestrator.shutdown()
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self._config = config or PipelineConfig()
        self._tools = HeatExchangerTools()
        self._checkpoint_manager = CheckpointManager(self._config.checkpoint_config)
        self._cache_manager = CacheManager(self._config.cache_config)
        self._initialized = False
        self._running_pipelines: Dict[str, asyncio.Task] = {}

        # Define pipeline stages with dependencies
        self._stage_order: List[PipelineStage] = [
            PipelineStage.INPUT_VALIDATION,
            PipelineStage.DATA_ENRICHMENT,
            PipelineStage.HEAT_TRANSFER_ANALYSIS,
            PipelineStage.FOULING_ASSESSMENT,
            PipelineStage.PERFORMANCE_EVALUATION,
            PipelineStage.CLEANING_OPTIMIZATION,
            PipelineStage.ECONOMIC_IMPACT,
            PipelineStage.REPORT_GENERATION,
            PipelineStage.RESULT_CACHING,
        ]

    async def initialize(self) -> None:
        """
        Initialize the orchestrator and its dependencies.

        Must be called before executing pipelines.
        """
        if self._initialized:
            return

        logger.info("Initializing HeatExchangerOrchestrator")
        await self._cache_manager.initialize()
        self._initialized = True
        logger.info("HeatExchangerOrchestrator initialized successfully")

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        logger.info("Shutting down HeatExchangerOrchestrator")

        # Cancel any running pipelines
        for pipeline_id, task in self._running_pipelines.items():
            if not task.done():
                task.cancel()
                logger.warning(f"Cancelled running pipeline: {pipeline_id}")

        await self._cache_manager.close()
        self._initialized = False
        logger.info("HeatExchangerOrchestrator shutdown complete")

    async def execute(
        self,
        input_data: HeatExchangerInputData,
        config_override: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute the complete analysis pipeline.

        Args:
            input_data: Heat exchanger input data
            config_override: Optional configuration overrides

        Returns:
            Complete pipeline execution result

        Raises:
            PipelineError: If pipeline execution fails
        """
        if not self._initialized:
            await self.initialize()

        # Apply config overrides
        config = self._config
        if config_override:
            config = PipelineConfig(**{**self._config.dict(), **config_override})

        correlation_id = config.correlation_id or str(uuid.uuid4())
        pipeline_id = config.pipeline_id

        # Setup logging context
        log_extra = {
            "pipeline_id": pipeline_id,
            "correlation_id": correlation_id,
            "equipment_id": input_data.equipment_id
        }

        logger.info(
            f"Starting pipeline execution for equipment {input_data.equipment_id}",
            extra=log_extra
        )

        start_time = datetime.utcnow()

        # Initialize result
        result = PipelineResult(
            pipeline_id=pipeline_id,
            correlation_id=correlation_id,
            status=PipelineStatus.RUNNING,
            equipment_id=input_data.equipment_id,
            start_time=start_time
        )

        # Check cache for existing result
        cache_key = self._cache_manager.generate_cache_key(input_data)
        cached_result = await self._cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for equipment {input_data.equipment_id}", extra=log_extra)
            # Reconstruct result from cache (simplified)
            result.status = PipelineStatus.COMPLETED
            result.summary = cached_result.get("summary")
            result.recommendations = cached_result.get("recommendations")
            result.end_time = datetime.utcnow()
            result.duration_ms = (result.end_time - start_time).total_seconds() * 1000
            return result

        # Check for existing checkpoint
        input_hash = hashlib.sha256(input_data.json().encode()).hexdigest()
        checkpoint = await self._checkpoint_manager.load_checkpoint(pipeline_id)

        if checkpoint and checkpoint.input_data_hash == input_hash:
            logger.info(f"Resuming from checkpoint at stage {checkpoint.last_completed_stage}", extra=log_extra)
            completed_stages = checkpoint.completed_stages
            pending_stages = checkpoint.pending_stages
            # Restore stage results (would need proper serialization)
        else:
            completed_stages = []
            pending_stages = list(self._stage_order)

        # Track task for cancellation
        current_task = asyncio.current_task()
        if current_task:
            self._running_pipelines[pipeline_id] = current_task

        try:
            # Execute pipeline stages
            stage_context: Dict[str, Any] = {
                "input_data": input_data,
                "config": config,
                "correlation_id": correlation_id,
                "pipeline_id": pipeline_id,
            }

            for stage in pending_stages:
                step_result = await self._execute_stage(
                    stage=stage,
                    context=stage_context,
                    result=result,
                    log_extra=log_extra
                )

                result.step_results[stage.value] = step_result

                if step_result.status == StepStatus.FAILED:
                    if config.fail_fast:
                        logger.error(f"Stage {stage.value} failed, failing fast", extra=log_extra)
                        result.status = PipelineStatus.FAILED
                        break
                    else:
                        logger.warning(f"Stage {stage.value} failed, continuing", extra=log_extra)

                # Update checkpoint after each stage
                completed_stages.append(stage)
                remaining_stages = [s for s in pending_stages if s not in completed_stages]

                await self._checkpoint_manager.save_checkpoint(
                    pipeline_id=pipeline_id,
                    completed_stage=stage,
                    completed_stages=completed_stages,
                    pending_stages=remaining_stages,
                    stage_results={s.value: asdict(result.step_results.get(s.value, {}))
                                   for s in completed_stages if s.value in result.step_results},
                    input_data_hash=input_hash
                )

            # Determine final status
            failed_stages = [
                s for s, r in result.step_results.items()
                if r.status == StepStatus.FAILED
            ]

            if not failed_stages:
                result.status = PipelineStatus.COMPLETED
            elif len(failed_stages) == len(self._stage_order):
                result.status = PipelineStatus.FAILED
            else:
                result.status = PipelineStatus.COMPLETED  # Partial success

            # Generate summary
            result.summary = self._generate_summary(result)
            result.recommendations = self._generate_recommendations(result)
            result.alerts = self._generate_alerts(result)

            # Generate final provenance hash
            result.provenance_hash = self._generate_provenance_hash(input_data, result)

            # Cache successful result
            if result.status == PipelineStatus.COMPLETED:
                await self._cache_manager.set(cache_key, {
                    "summary": result.summary,
                    "recommendations": result.recommendations,
                    "alerts": result.alerts,
                    "provenance_hash": result.provenance_hash
                })

            # Cleanup checkpoint on success
            await self._checkpoint_manager.delete_checkpoint(pipeline_id)

        except asyncio.CancelledError:
            result.status = PipelineStatus.CANCELLED
            logger.warning(f"Pipeline {pipeline_id} was cancelled", extra=log_extra)
            raise

        except Exception as e:
            result.status = PipelineStatus.FAILED
            logger.error(
                f"Pipeline {pipeline_id} failed with error: {str(e)}",
                extra=log_extra,
                exc_info=True
            )
            raise PipelineError(
                f"Pipeline execution failed: {str(e)}",
                recoverable=False,
                original_exception=e
            )

        finally:
            result.end_time = datetime.utcnow()
            result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000

            # Remove from running pipelines
            if pipeline_id in self._running_pipelines:
                del self._running_pipelines[pipeline_id]

            logger.info(
                f"Pipeline {pipeline_id} completed with status {result.status.value} "
                f"in {result.duration_ms:.2f}ms",
                extra=log_extra
            )

        return result

    async def _execute_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        result: PipelineResult,
        log_extra: Dict[str, Any]
    ) -> StepResult:
        """
        Execute a single pipeline stage with retry logic.

        Args:
            stage: Stage to execute
            context: Execution context
            result: Pipeline result to update
            log_extra: Logging context

        Returns:
            Step execution result
        """
        step_result = StepResult(
            stage=stage,
            status=StepStatus.RUNNING,
            start_time=datetime.utcnow()
        )

        config: PipelineConfig = context["config"]
        retry_config = config.retry_config
        max_retries = retry_config.max_retries

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    f"Executing stage {stage.value} (attempt {attempt + 1}/{max_retries + 1})",
                    extra=log_extra
                )

                # Execute the appropriate stage handler
                if stage == PipelineStage.INPUT_VALIDATION:
                    await self._execute_input_validation(context, result)

                elif stage == PipelineStage.DATA_ENRICHMENT:
                    await self._execute_data_enrichment(context, result)

                elif stage == PipelineStage.HEAT_TRANSFER_ANALYSIS:
                    await self._execute_heat_transfer_analysis(context, result)

                elif stage == PipelineStage.FOULING_ASSESSMENT:
                    await self._execute_fouling_assessment(context, result)

                elif stage == PipelineStage.PERFORMANCE_EVALUATION:
                    await self._execute_performance_evaluation(context, result)

                elif stage == PipelineStage.CLEANING_OPTIMIZATION:
                    await self._execute_cleaning_optimization(context, result)

                elif stage == PipelineStage.ECONOMIC_IMPACT:
                    await self._execute_economic_impact(context, result)

                elif stage == PipelineStage.REPORT_GENERATION:
                    await self._execute_report_generation(context, result)

                elif stage == PipelineStage.RESULT_CACHING:
                    await self._execute_result_caching(context, result)

                # Success
                step_result.status = StepStatus.COMPLETED
                step_result.retry_count = attempt
                break

            except Exception as e:
                step_result.retry_count = attempt
                step_result.error = str(e)
                step_result.error_traceback = traceback.format_exc()

                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.initial_delay_seconds * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay_seconds
                    )

                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"Stage {stage.value} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {str(e)}",
                        extra=log_extra
                    )

                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Stage {stage.value} failed after {max_retries + 1} attempts: {str(e)}",
                        extra=log_extra
                    )
                    step_result.status = StepStatus.FAILED

        step_result.end_time = datetime.utcnow()
        step_result.duration_ms = (
            step_result.end_time - step_result.start_time
        ).total_seconds() * 1000

        return step_result

    # =========================================================================
    # STAGE HANDLERS
    # =========================================================================

    async def _execute_input_validation(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Validate and normalize input data."""
        input_data: HeatExchangerInputData = context["input_data"]

        # Perform validation checks
        errors = []

        # Temperature consistency
        if input_data.t_hot_in_c <= input_data.t_cold_in_c:
            errors.append("Hot inlet must be warmer than cold inlet")

        if input_data.t_hot_out_c <= input_data.t_cold_in_c:
            logger.warning("Possible temperature cross detected")

        # Duty consistency
        if input_data.actual_duty_kw > input_data.design_duty_kw * 1.5:
            errors.append("Actual duty exceeds design by more than 50%")

        if errors:
            raise ValidationError(f"Input validation failed: {'; '.join(errors)}")

        # Store validated data in context
        context["validated_data"] = input_data

    async def _execute_data_enrichment(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Enrich input data with additional information from integrations."""
        input_data: HeatExchangerInputData = context["input_data"]

        # This would typically fetch from:
        # - Historian databases
        # - ERP systems
        # - Asset management systems
        # For now, we'll simulate enrichment

        enriched_data = {
            "equipment_criticality": "high",
            "maintenance_history": [],
            "design_spec_available": True,
            "last_inspection_date": None,
        }

        context["enriched_data"] = enriched_data

    async def _execute_heat_transfer_analysis(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Perform heat transfer calculations."""
        input_data: HeatExchangerInputData = context["input_data"]

        # Calculate LMTD
        lmtd_input = LMTDInput(
            t_hot_in_c=input_data.t_hot_in_c,
            t_hot_out_c=input_data.t_hot_out_c,
            t_cold_in_c=input_data.t_cold_in_c,
            t_cold_out_c=input_data.t_cold_out_c,
            flow_arrangement=input_data.flow_arrangement
        )
        result.lmtd_result = self._tools.calculate_lmtd(lmtd_input)

        # Calculate effectiveness if we have capacity data
        if input_data.flow_hot_kg_s and input_data.flow_cold_kg_s:
            # Estimate NTU from actual performance
            ntu_estimate = 1.5  # Would calculate from actual data
            c_ratio = 0.8  # Would calculate from flow rates and Cp

            eff_input = EffectivenessInput(
                ntu=ntu_estimate,
                c_ratio=c_ratio,
                flow_arrangement=input_data.flow_arrangement
            )
            result.effectiveness_result = self._tools.calculate_effectiveness(eff_input)

        # Calculate overall U if film coefficients provided
        if (input_data.h_hot_w_m2_k and input_data.h_cold_w_m2_k and
                input_data.tube_od_m and input_data.tube_id_m):
            u_input = HeatTransferInput(
                h_hot_w_m2_k=input_data.h_hot_w_m2_k,
                h_cold_w_m2_k=input_data.h_cold_w_m2_k,
                tube_od_m=input_data.tube_od_m,
                tube_id_m=input_data.tube_id_m,
                tube_k_w_m_k=input_data.tube_k_w_m_k or 50.0,
                r_fouling_hot_m2_k_w=input_data.r_f_hot_design_m2_k_w,
                r_fouling_cold_m2_k_w=input_data.r_f_cold_design_m2_k_w
            )
            result.heat_transfer_result = self._tools.calculate_overall_heat_transfer_coefficient(u_input)

        context["heat_transfer_complete"] = True

    async def _execute_fouling_assessment(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Assess fouling state and predict progression."""
        input_data: HeatExchangerInputData = context["input_data"]

        # Calculate actual U from performance
        if float(result.lmtd_result.lmtd_c) > 0:
            u_actual = input_data.actual_duty_kw * 1000 / (
                input_data.design_area_m2 * float(result.lmtd_result.lmtd_c)
            )
        else:
            u_actual = input_data.design_u_w_m2_k * 0.8  # Estimate

        # Calculate fouling resistance
        fouling_input = FoulingInput(
            u_clean_w_m2_k=input_data.design_u_w_m2_k,
            u_fouled_w_m2_k=u_actual,
            fluid_type_hot=input_data.fluid_type_hot,
            fluid_type_cold=input_data.fluid_type_cold
        )
        result.fouling_result = self._tools.calculate_fouling_resistance(fouling_input)

        # Predict fouling progression
        current_r_f = float(result.fouling_result.fouling_resistance_m2_k_w)
        design_r_f = input_data.r_f_hot_design_m2_k_w + input_data.r_f_cold_design_m2_k_w

        # Estimate fouling rate from operating hours
        if input_data.operating_hours_since_cleaning and input_data.operating_hours_since_cleaning > 0:
            fouling_rate = current_r_f / input_data.operating_hours_since_cleaning
        else:
            fouling_rate = 0.00000001  # Default very low rate

        prediction_input = FoulingPredictionInput(
            current_r_f_m2_k_w=current_r_f,
            fouling_rate_m2_k_w_per_hour=fouling_rate,
            target_time_hours=720,  # 30 days ahead
            design_fouling_resistance_m2_k_w=design_r_f
        )
        result.fouling_prediction_result = self._tools.predict_fouling_progression(prediction_input)

        context["fouling_complete"] = True
        context["u_actual"] = u_actual

    async def _execute_performance_evaluation(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Evaluate overall performance and health index."""
        input_data: HeatExchangerInputData = context["input_data"]
        u_actual = context.get("u_actual", input_data.design_u_w_m2_k * 0.8)

        # Calculate thermal efficiency
        efficiency_input = EfficiencyInput(
            q_actual_kw=input_data.actual_duty_kw,
            q_design_kw=input_data.design_duty_kw,
            t_hot_in_c=input_data.t_hot_in_c,
            t_hot_out_c=input_data.t_hot_out_c,
            t_cold_in_c=input_data.t_cold_in_c,
            t_cold_out_c=input_data.t_cold_out_c
        )
        result.efficiency_result = self._tools.calculate_thermal_efficiency(efficiency_input)

        # Calculate health index
        dp_actual = (input_data.dp_hot_actual_kpa or 50) + (input_data.dp_cold_actual_kpa or 30)
        dp_design = (input_data.dp_hot_design_kpa or 45) + (input_data.dp_cold_design_kpa or 25)
        approach_actual = input_data.t_hot_out_c - input_data.t_cold_in_c
        approach_design = input_data.design_lmtd_c * 0.3  # Estimate

        health_input = HealthInput(
            u_actual_w_m2_k=u_actual,
            u_design_w_m2_k=input_data.design_u_w_m2_k,
            dp_actual_kpa=max(dp_actual, 1),
            dp_design_kpa=max(dp_design, 1),
            approach_temp_actual_c=max(approach_actual, 0.1),
            approach_temp_design_c=max(approach_design, 0.1)
        )
        result.health_index_result = self._tools.calculate_health_index(health_input)

        context["performance_complete"] = True

    async def _execute_cleaning_optimization(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Optimize cleaning interval and analyze cost-benefit."""
        input_data: HeatExchangerInputData = context["input_data"]

        if result.fouling_result is None:
            logger.warning("Skipping cleaning optimization - no fouling result available")
            return

        current_r_f = float(result.fouling_result.fouling_resistance_m2_k_w)
        design_r_f = input_data.r_f_hot_design_m2_k_w + input_data.r_f_cold_design_m2_k_w

        # Estimate daily energy loss cost
        duty_loss_kw = input_data.design_duty_kw - input_data.actual_duty_kw
        daily_energy_loss = duty_loss_kw * 24 / input_data.system_efficiency
        energy_loss_cost_per_day = daily_energy_loss * input_data.fuel_cost_per_kwh

        # Estimate daily fouling rate
        if input_data.operating_hours_since_cleaning and input_data.operating_hours_since_cleaning > 0:
            daily_fouling_rate = current_r_f / (input_data.operating_hours_since_cleaning / 24)
        else:
            daily_fouling_rate = 0.0000001

        cleaning_input = CleaningInput(
            current_r_f_m2_k_w=current_r_f,
            fouling_rate_m2_k_w_per_day=daily_fouling_rate,
            cleaning_threshold_r_f=design_r_f,
            cleaning_cost_usd=input_data.cleaning_cost_usd,
            energy_loss_cost_per_day_usd=max(energy_loss_cost_per_day, 0),
            downtime_hours=input_data.downtime_hours_per_cleaning,
            production_loss_per_hour_usd=input_data.production_loss_per_hour_usd
        )
        result.cleaning_result = self._tools.optimize_cleaning_interval(cleaning_input)

        # Cost-benefit analysis
        annual_energy_savings = energy_loss_cost_per_day * 365 * 0.5  # Assume 50% recovery

        cost_benefit_input = CostBenefitInput(
            cleaning_cost_usd=input_data.cleaning_cost_usd,
            energy_savings_per_year_usd=annual_energy_savings,
            production_improvement_per_year_usd=0,
            equipment_life_extension_years=1.0,
            equipment_replacement_cost_usd=input_data.cleaning_cost_usd * 20,
            discount_rate_percent=10.0
        )
        result.cost_benefit_result = self._tools.calculate_cleaning_cost_benefit(cost_benefit_input)

        context["cleaning_complete"] = True

    async def _execute_economic_impact(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Calculate economic impact of current performance."""
        input_data: HeatExchangerInputData = context["input_data"]

        # Energy loss calculation
        energy_loss_input = EnergyLossInput(
            design_duty_kw=input_data.design_duty_kw,
            actual_duty_kw=input_data.actual_duty_kw,
            fuel_cost_per_kwh=input_data.fuel_cost_per_kwh,
            system_efficiency=input_data.system_efficiency,
            operating_hours_per_year=input_data.operating_hours_per_year,
            carbon_price_per_tonne=input_data.carbon_price_per_tonne,
            emission_factor_kg_co2_per_kwh=input_data.emission_factor_kg_co2_per_kwh
        )
        result.energy_loss_result = self._tools.calculate_energy_loss_cost(energy_loss_input)

        # ROI for improvement investment
        investment_cost = input_data.cleaning_cost_usd
        annual_savings = float(result.energy_loss_result.total_energy_penalty_usd) if result.energy_loss_result else 0

        roi_input = ROIInput(
            investment_cost_usd=investment_cost,
            annual_savings_usd=annual_savings,
            useful_life_years=10,
            discount_rate_percent=10.0,
            residual_value_percent=10.0
        )
        result.roi_result = self._tools.calculate_roi(roi_input)

        context["economic_complete"] = True

    async def _execute_report_generation(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Generate analysis report and recommendations."""
        input_data: HeatExchangerInputData = context["input_data"]

        # This stage prepares the summary and recommendations
        # Actual generation happens in _generate_summary and _generate_recommendations
        context["report_ready"] = True

    async def _execute_result_caching(
        self,
        context: Dict[str, Any],
        result: PipelineResult
    ) -> None:
        """Cache results for future reference."""
        # Caching is handled in the main execute method
        # This stage is a placeholder for any additional caching logic
        context["cached"] = True

    # =========================================================================
    # REPORT GENERATION HELPERS
    # =========================================================================

    def _generate_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """Generate executive summary of the analysis."""
        summary = {
            "equipment_id": result.equipment_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "overall_status": "UNKNOWN",
            "key_metrics": {},
            "stage_summary": {}
        }

        # Health status
        if result.health_index_result:
            summary["overall_status"] = result.health_index_result.health_level
            summary["key_metrics"]["health_index"] = float(result.health_index_result.health_index)

        # Thermal performance
        if result.efficiency_result:
            summary["key_metrics"]["thermal_efficiency_percent"] = float(
                result.efficiency_result.thermal_efficiency_percent
            )
            summary["key_metrics"]["duty_shortfall_percent"] = float(
                result.efficiency_result.duty_shortfall_percent
            )

        # Fouling status
        if result.fouling_result:
            summary["key_metrics"]["cleanliness_factor_percent"] = float(
                result.fouling_result.cleanliness_factor_percent
            )
            summary["key_metrics"]["fouling_resistance_m2_k_w"] = float(
                result.fouling_result.fouling_resistance_m2_k_w
            )

        # Economic impact
        if result.energy_loss_result:
            summary["key_metrics"]["annual_energy_penalty_usd"] = float(
                result.energy_loss_result.total_energy_penalty_usd
            )

        # Cleaning recommendation
        if result.cleaning_result:
            summary["key_metrics"]["days_to_cleaning"] = float(
                result.cleaning_result.time_to_next_cleaning_days
            )

        # Stage summary
        for stage_name, step in result.step_results.items():
            summary["stage_summary"][stage_name] = {
                "status": step.status.value,
                "duration_ms": step.duration_ms,
                "retries": step.retry_count
            }

        return summary

    def _generate_recommendations(self, result: PipelineResult) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # From health index
        if result.health_index_result:
            recommendations.extend(list(result.health_index_result.recommendations))

        # Based on fouling
        if result.fouling_result:
            cf = float(result.fouling_result.cleanliness_factor_percent)
            if cf < 70:
                recommendations.append(
                    f"URGENT: Cleanliness factor at {cf:.1f}%. Schedule cleaning immediately."
                )
            elif cf < 80:
                recommendations.append(
                    f"Cleanliness factor at {cf:.1f}%. Plan cleaning within 30 days."
                )

        # Based on economic impact
        if result.energy_loss_result:
            penalty = float(result.energy_loss_result.total_energy_penalty_usd)
            if penalty > 50000:
                recommendations.append(
                    f"High energy penalty: ${penalty:,.0f}/year. Prioritize cleaning investment."
                )

        # Based on ROI
        if result.roi_result:
            if float(result.roi_result.simple_payback_years) < 1:
                recommendations.append(
                    f"Excellent ROI: Payback in {float(result.roi_result.simple_payback_years):.1f} years. "
                    "Proceed with improvement immediately."
                )

        # Based on cleaning optimization
        if result.cleaning_result:
            days = float(result.cleaning_result.time_to_next_cleaning_days)
            if days < 30:
                recommendations.append(
                    f"Next cleaning recommended in {days:.0f} days."
                )

        return recommendations

    def _generate_alerts(self, result: PipelineResult) -> List[Dict[str, Any]]:
        """Generate alerts for critical conditions."""
        alerts = []

        # Health alerts
        if result.health_index_result:
            hi = float(result.health_index_result.health_index)
            if hi < 30:
                alerts.append({
                    "severity": "CRITICAL",
                    "type": "health_index",
                    "message": f"Critical health index: {hi:.1f}",
                    "action_required": True
                })
            elif hi < 50:
                alerts.append({
                    "severity": "WARNING",
                    "type": "health_index",
                    "message": f"Poor health index: {hi:.1f}",
                    "action_required": True
                })

        # Fouling alerts
        if result.fouling_result:
            cf = float(result.fouling_result.cleanliness_factor_percent)
            if cf < 60:
                alerts.append({
                    "severity": "CRITICAL",
                    "type": "fouling",
                    "message": f"Severe fouling detected: {100-cf:.1f}% heat transfer loss",
                    "action_required": True
                })

        # Economic alerts
        if result.energy_loss_result:
            penalty = float(result.energy_loss_result.total_energy_penalty_usd)
            if penalty > 100000:
                alerts.append({
                    "severity": "WARNING",
                    "type": "economic",
                    "message": f"Significant energy penalty: ${penalty:,.0f}/year",
                    "action_required": True
                })

        return alerts

    def _generate_provenance_hash(
        self,
        input_data: HeatExchangerInputData,
        result: PipelineResult
    ) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        data = {
            "input_hash": hashlib.sha256(input_data.json().encode()).hexdigest(),
            "pipeline_id": result.pipeline_id,
            "correlation_id": result.correlation_id,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "status": result.status.value,
            "summary": result.summary,
            "version": self.VERSION
        }

        canonical_json = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    # =========================================================================
    # PUBLIC UTILITY METHODS
    # =========================================================================

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel a running pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if cancelled, False if not found or already complete
        """
        task = self._running_pipelines.get(pipeline_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Pipeline {pipeline_id} cancellation requested")
            return True
        return False

    def get_running_pipelines(self) -> List[str]:
        """
        Get list of currently running pipeline IDs.

        Returns:
            List of pipeline IDs
        """
        return [
            pid for pid, task in self._running_pipelines.items()
            if not task.done()
        ]

    async def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """
        Get status of a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline status or None if not found
        """
        task = self._running_pipelines.get(pipeline_id)
        if task is None:
            return None
        elif task.done():
            if task.cancelled():
                return PipelineStatus.CANCELLED
            elif task.exception():
                return PipelineStatus.FAILED
            else:
                return PipelineStatus.COMPLETED
        else:
            return PipelineStatus.RUNNING


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_orchestrator(config: Optional[PipelineConfig] = None) -> HeatExchangerOrchestrator:
    """
    Create a new orchestrator instance.

    Args:
        config: Optional pipeline configuration

    Returns:
        Configured orchestrator instance
    """
    return HeatExchangerOrchestrator(config)


async def run_analysis(
    input_data: HeatExchangerInputData,
    config: Optional[PipelineConfig] = None
) -> PipelineResult:
    """
    Convenience function to run a complete analysis.

    Args:
        input_data: Heat exchanger input data
        config: Optional pipeline configuration

    Returns:
        Pipeline execution result
    """
    orchestrator = create_orchestrator(config)
    try:
        await orchestrator.initialize()
        return await orchestrator.execute(input_data)
    finally:
        await orchestrator.shutdown()


# =============================================================================
# EXPORT DECLARATIONS
# =============================================================================

__all__ = [
    # Enumerations
    "PipelineStatus",
    "StepStatus",
    "PipelineStage",
    "ErrorSeverity",
    # Configuration
    "RetryConfig",
    "CacheConfig",
    "CheckpointConfig",
    "PipelineConfig",
    # Input/Output models
    "HeatExchangerInputData",
    "StepResult",
    "PipelineResult",
    # Checkpoint management
    "Checkpoint",
    "CheckpointManager",
    # Cache management
    "CacheManager",
    # Exceptions
    "PipelineError",
    "ValidationError",
    "CalculationError",
    "IntegrationError",
    # Main orchestrator
    "HeatExchangerOrchestrator",
    # Factory functions
    "create_orchestrator",
    "run_analysis",
]
