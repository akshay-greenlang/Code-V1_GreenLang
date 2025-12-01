# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Insulation Inspection Pipeline Orchestrator

Production-grade pipeline orchestration for thermal insulation inspection and analysis.
Implements the GreenLang agent architecture pattern with:
- Multi-stage pipeline execution
- Checkpoint management for long-running inspections
- Async processing with asyncio
- Error recovery and retry logic
- Redis cache integration
- Structured logging with correlation IDs
- Complete provenance tracking

Pipeline Stages:
1. Input Validation - Validate and normalize input data
2. Image Preprocessing - Prepare thermal images for analysis
3. Thermal Analysis - Analyze thermal images and detect hotspots
4. Heat Loss Calculation - Calculate heat loss at each location
5. Degradation Assessment - Assess insulation condition
6. Energy Quantification - Quantify facility-wide energy loss
7. Repair Prioritization - Prioritize repairs by criticality
8. Economic Analysis - Calculate ROI and cost-benefit
9. Report Generation - Generate inspection report
10. Result Caching - Cache results for future reference

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
    InsulationInspectionTools,
    ThermalImageInput,
    HotspotInput,
    AnomalyInput,
    HeatLossInput,
    SurfaceTempInput,
    DegradationInput,
    RULInput,
    RepairInput,
    ROIInput,
    EnergyInput,
    CarbonInput,
    ThermalAnalysisResult,
    HotspotResult,
    AnomalyClassificationResult,
    HeatLossResult,
    SurfaceTempResult,
    DegradationResult,
    RULResult,
    RepairPriorityResult,
    ROIResult,
    EnergyLossResult,
    CarbonResult,
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
    IMAGE_PREPROCESSING = "image_preprocessing"
    THERMAL_ANALYSIS = "thermal_analysis"
    HEAT_LOSS_CALCULATION = "heat_loss_calculation"
    DEGRADATION_ASSESSMENT = "degradation_assessment"
    ENERGY_QUANTIFICATION = "energy_quantification"
    REPAIR_PRIORITIZATION = "repair_prioritization"
    ECONOMIC_ANALYSIS = "economic_analysis"
    REPORT_GENERATION = "report_generation"
    RESULT_CACHING = "result_caching"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InspectionType(str, Enum):
    """Types of insulation inspection."""
    FULL_FACILITY = "full_facility"
    UNIT_SURVEY = "unit_survey"
    PROBLEM_AREA = "problem_area"
    COMPLIANCE_AUDIT = "compliance_audit"
    ENERGY_ASSESSMENT = "energy_assessment"


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
    cache_key_prefix: str = Field(default="gl015:")
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
    timeout_seconds: float = Field(default=600.0, ge=10.0, le=7200.0)
    parallel_stages: bool = Field(default=False)
    fail_fast: bool = Field(default=False)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    checkpoint_config: CheckpointConfig = Field(default_factory=CheckpointConfig)
    enable_detailed_logging: bool = Field(default=True)
    inspection_type: InspectionType = Field(default=InspectionType.FULL_FACILITY)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class InspectionLocation(BaseModel):
    """Data for a single inspection location."""
    location_id: str = Field(..., description="Unique location identifier")
    equipment_tag: str = Field(..., description="Plant equipment tag")
    equipment_type: str = Field(default="pipe", description="Equipment type")
    system_type: str = Field(default="steam", description="System type")

    # Physical parameters
    pipe_outer_diameter_mm: Optional[float] = Field(None, gt=0)
    pipe_length_m: float = Field(default=1.0, gt=0)
    surface_area_m2: Optional[float] = Field(None, gt=0)

    # Operating conditions
    process_temperature_c: float = Field(..., description="Process temperature (C)")
    ambient_temperature_c: float = Field(default=25.0, description="Ambient temperature (C)")
    design_surface_temp_c: Optional[float] = Field(None, description="Design surface temperature")

    # Insulation data
    insulation_material: str = Field(default="mineral_wool")
    insulation_thickness_mm: float = Field(default=50.0, ge=0)
    jacket_material: str = Field(default="aluminum")
    installation_date: str = Field(..., description="Installation date (ISO)")

    # Current observations
    observed_surface_temp_c: Optional[float] = Field(None)
    thermal_image_id: Optional[str] = Field(None)
    moisture_detected: bool = Field(default=False)
    mechanical_damage_observed: bool = Field(default=False)
    jacket_condition_score: int = Field(default=5, ge=1, le=10)

    # Economic data
    estimated_repair_cost_usd: float = Field(default=1000.0, gt=0)


class ThermalImageData(BaseModel):
    """Thermal image data for analysis."""
    image_id: str = Field(..., description="Image identifier")
    location_id: str = Field(..., description="Associated location")
    temperature_matrix: List[List[float]] = Field(..., description="2D temperature matrix")
    emissivity: float = Field(default=0.95, ge=0.01, le=1.0)
    reflected_temperature_c: float = Field(default=20.0)
    distance_m: float = Field(default=1.0, gt=0)
    relative_humidity: float = Field(default=50.0, ge=0, le=100)
    capture_timestamp: str = Field(..., description="Capture timestamp (ISO)")


class InspectionInputData(BaseModel):
    """Complete input data for insulation inspection."""
    # Inspection identification
    inspection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    inspection_date: str = Field(..., description="Inspection date (ISO)")
    inspector_id: str = Field(..., description="Inspector identifier")
    inspection_type: InspectionType = Field(default=InspectionType.FULL_FACILITY)

    # Locations to inspect
    locations: List[InspectionLocation] = Field(..., min_items=1)

    # Thermal images
    thermal_images: List[ThermalImageData] = Field(default_factory=list)

    # Environmental conditions
    ambient_temperature_c: float = Field(default=25.0)
    relative_humidity: float = Field(default=50.0, ge=0, le=100)
    wind_speed_m_s: float = Field(default=0.0, ge=0)
    weather_conditions: str = Field(default="clear")

    # Economic parameters
    fuel_type: str = Field(default="natural_gas")
    energy_cost_per_mmbtu: float = Field(default=4.50, gt=0)
    boiler_efficiency: float = Field(default=0.85, gt=0, le=1)
    operating_hours_per_year: float = Field(default=8000, gt=0)
    carbon_price_per_tonne: float = Field(default=50.0, ge=0)

    # Analysis parameters
    delta_t_threshold_c: float = Field(default=5.0, gt=0)
    personnel_safety_limit_c: float = Field(default=60.0, gt=0)
    repair_budget_usd: Optional[float] = Field(None, ge=0)

    @validator('locations')
    def validate_locations(cls, v):
        if not v:
            raise ValueError("At least one location is required")
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
class InspectionResult:
    """Complete inspection pipeline result."""
    pipeline_id: str
    correlation_id: str
    status: PipelineStatus
    inspection_id: str
    facility_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    step_results: Dict[str, StepResult] = field(default_factory=dict)

    # Analysis results by stage
    validated_input: Optional[Dict[str, Any]] = None
    thermal_analysis_results: Optional[List[ThermalAnalysisResult]] = None
    hotspot_results: Optional[List[HotspotResult]] = None
    anomaly_classifications: Optional[List[AnomalyClassificationResult]] = None
    heat_loss_results: Optional[List[HeatLossResult]] = None
    degradation_results: Optional[List[DegradationResult]] = None
    rul_results: Optional[List[RULResult]] = None
    energy_loss_result: Optional[EnergyLossResult] = None
    carbon_result: Optional[CarbonResult] = None
    repair_priority_result: Optional[RepairPriorityResult] = None
    roi_results: Optional[List[ROIResult]] = None

    # Summary and recommendations
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
        """Initialize checkpoint manager."""
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
    ) -> Optional[Checkpoint]:
        """Save a pipeline checkpoint."""
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

            await self._cleanup_old_checkpoints()
            return checkpoint

    async def load_checkpoint(self, pipeline_id: str) -> Optional[Checkpoint]:
        """Load checkpoint for a pipeline."""
        if not self._config.enabled:
            return None

        async with self._lock:
            checkpoint = self._checkpoints.get(pipeline_id)

            if checkpoint is None:
                return None

            max_age = timedelta(hours=self._config.max_checkpoint_age_hours)
            if datetime.utcnow() - checkpoint.created_at > max_age:
                logger.warning(f"Checkpoint for pipeline {pipeline_id} is too old")
                del self._checkpoints[pipeline_id]
                return None

            if not checkpoint.is_valid:
                logger.warning(f"Checkpoint for pipeline {pipeline_id} is invalid")
                del self._checkpoints[pipeline_id]
                return None

            return checkpoint

    async def invalidate_checkpoint(self, pipeline_id: str) -> None:
        """Invalidate a pipeline checkpoint."""
        async with self._lock:
            if pipeline_id in self._checkpoints:
                self._checkpoints[pipeline_id].is_valid = False

    async def delete_checkpoint(self, pipeline_id: str) -> None:
        """Delete a pipeline checkpoint."""
        async with self._lock:
            if pipeline_id in self._checkpoints:
                del self._checkpoints[pipeline_id]

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
        """Initialize cache manager."""
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
                logger.warning("redis package not installed, using memory cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")

    async def close(self) -> None:
        """Close cache connections."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis cache connection closed")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
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
                value, timestamp = self._memory_cache[full_key]
                age = (datetime.utcnow() - timestamp).total_seconds()
                if age < self._config.ttl_seconds:
                    # Move to end of access order (LRU)
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

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self._config.enabled:
            return

        full_key = f"{self._config.cache_key_prefix}{key}"

        # Try Redis
        if self._redis_client:
            try:
                await self._redis_client.setex(
                    full_key,
                    self._config.ttl_seconds,
                    json.dumps(value, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # Also store in memory cache
        async with self._lock:
            # Evict if over limit
            while len(self._memory_cache) >= self._config.max_entries:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._memory_cache:
                        del self._memory_cache[oldest_key]
                else:
                    break

            self._memory_cache[full_key] = (value, datetime.utcnow())
            self._access_order.append(full_key)

    async def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        full_key = f"{self._config.cache_key_prefix}{key}"

        if self._redis_client:
            try:
                await self._redis_client.delete(full_key)
            except Exception:
                pass

        async with self._lock:
            if full_key in self._memory_cache:
                del self._memory_cache[full_key]
            if full_key in self._access_order:
                self._access_order.remove(full_key)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_entries": len(self._memory_cache),
            "redis_connected": self._redis_client is not None,
            "ttl_seconds": self._config.ttl_seconds
        }


# =============================================================================
# PIPELINE STAGE HANDLERS
# =============================================================================

class StageHandler(ABC):
    """Abstract base class for pipeline stage handlers."""

    @abstractmethod
    async def execute(
        self,
        context: 'PipelineContext',
        input_data: Any
    ) -> Any:
        """Execute the stage."""
        pass

    @abstractmethod
    def get_stage(self) -> PipelineStage:
        """Get the stage this handler processes."""
        pass


class InputValidationHandler(StageHandler):
    """Handler for input validation stage."""

    def get_stage(self) -> PipelineStage:
        return PipelineStage.INPUT_VALIDATION

    async def execute(
        self,
        context: 'PipelineContext',
        input_data: InspectionInputData
    ) -> Dict[str, Any]:
        """Validate and normalize input data."""
        logger.info(f"[{context.correlation_id}] Validating input data")

        validated = {
            "inspection_id": input_data.inspection_id,
            "facility_id": input_data.facility_id,
            "facility_name": input_data.facility_name,
            "inspection_date": input_data.inspection_date,
            "inspector_id": input_data.inspector_id,
            "inspection_type": input_data.inspection_type.value,
            "location_count": len(input_data.locations),
            "image_count": len(input_data.thermal_images),
            "ambient_temperature_c": input_data.ambient_temperature_c,
            "economic_params": {
                "fuel_type": input_data.fuel_type,
                "energy_cost_per_mmbtu": input_data.energy_cost_per_mmbtu,
                "boiler_efficiency": input_data.boiler_efficiency,
                "operating_hours_per_year": input_data.operating_hours_per_year,
                "carbon_price_per_tonne": input_data.carbon_price_per_tonne
            },
            "analysis_params": {
                "delta_t_threshold_c": input_data.delta_t_threshold_c,
                "personnel_safety_limit_c": input_data.personnel_safety_limit_c,
                "repair_budget_usd": input_data.repair_budget_usd
            },
            "locations": [loc.dict() for loc in input_data.locations],
            "thermal_images": [img.dict() for img in input_data.thermal_images]
        }

        logger.info(
            f"[{context.correlation_id}] Input validated: "
            f"{validated['location_count']} locations, {validated['image_count']} images"
        )

        return validated


class ImagePreprocessingHandler(StageHandler):
    """Handler for image preprocessing stage."""

    def get_stage(self) -> PipelineStage:
        return PipelineStage.IMAGE_PREPROCESSING

    async def execute(
        self,
        context: 'PipelineContext',
        validated_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Preprocess thermal images for analysis."""
        logger.info(f"[{context.correlation_id}] Preprocessing thermal images")

        images = validated_input.get("thermal_images", [])
        preprocessed = []

        for img in images:
            # Basic preprocessing - could add noise reduction, calibration, etc.
            preprocessed.append({
                "image_id": img["image_id"],
                "location_id": img["location_id"],
                "temperature_matrix": img["temperature_matrix"],
                "emissivity": img.get("emissivity", 0.95),
                "reflected_temperature_c": img.get("reflected_temperature_c", 20.0),
                "is_preprocessed": True
            })

        logger.info(f"[{context.correlation_id}] Preprocessed {len(preprocessed)} images")

        return {
            **validated_input,
            "preprocessed_images": preprocessed
        }


class ThermalAnalysisHandler(StageHandler):
    """Handler for thermal analysis stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.THERMAL_ANALYSIS

    async def execute(
        self,
        context: 'PipelineContext',
        preprocessed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze thermal images and detect hotspots."""
        logger.info(f"[{context.correlation_id}] Running thermal analysis")

        images = preprocessed_data.get("preprocessed_images", [])
        delta_t_threshold = preprocessed_data["analysis_params"]["delta_t_threshold_c"]
        ambient = preprocessed_data["ambient_temperature_c"]

        thermal_results = []
        hotspot_results = []
        anomaly_results = []

        for img in images:
            # Analyze thermal image
            analysis_input = ThermalImageInput(
                image_id=img["image_id"],
                temperature_matrix=img["temperature_matrix"],
                emissivity=img["emissivity"],
                reflected_temperature_c=img["reflected_temperature_c"],
                ambient_temperature_c=ambient
            )

            try:
                analysis_result = self._tools.analyze_thermal_image(analysis_input)
                thermal_results.append(analysis_result)

                # Detect hotspots
                hotspot_input = HotspotInput(
                    temperature_matrix=img["temperature_matrix"],
                    delta_t_threshold_c=delta_t_threshold,
                    ambient_temperature_c=ambient,
                    min_hotspot_pixels=4,
                    merge_distance_pixels=3
                )

                hotspot_result = self._tools.detect_hotspots(hotspot_input)
                hotspot_results.append(hotspot_result)

                # Classify anomalies for each hotspot
                for hotspot in hotspot_result.hotspots:
                    anomaly_input = AnomalyInput(
                        hotspot_id=hotspot["hotspot_id"],
                        peak_temperature_c=float(hotspot["peak_temperature_c"]),
                        mean_temperature_c=float(hotspot["mean_temperature_c"]),
                        delta_t_from_ambient_c=float(hotspot["delta_t_from_ambient_c"]),
                        area_pixels=hotspot["area_pixels"],
                        ambient_temperature_c=ambient,
                        severity_score=float(hotspot["severity_score"])
                    )

                    anomaly_result = self._tools.classify_anomaly(anomaly_input)
                    anomaly_results.append(anomaly_result)

            except Exception as e:
                logger.error(f"Thermal analysis failed for image {img['image_id']}: {e}")
                if context.config.fail_fast:
                    raise

        logger.info(
            f"[{context.correlation_id}] Thermal analysis complete: "
            f"{len(thermal_results)} images, {sum(h.hotspots_detected for h in hotspot_results)} hotspots"
        )

        return {
            **preprocessed_data,
            "thermal_analysis_results": thermal_results,
            "hotspot_results": hotspot_results,
            "anomaly_classifications": anomaly_results
        }


class HeatLossCalculationHandler(StageHandler):
    """Handler for heat loss calculation stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.HEAT_LOSS_CALCULATION

    async def execute(
        self,
        context: 'PipelineContext',
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate heat loss at each location."""
        logger.info(f"[{context.correlation_id}] Calculating heat loss")

        locations = analysis_data.get("locations", [])
        ambient = analysis_data["ambient_temperature_c"]

        heat_loss_results = []

        for loc in locations:
            # Use observed temperature if available, else estimate
            surface_temp = loc.get("observed_surface_temp_c")

            if surface_temp is None:
                # Calculate surface temperature first
                surf_input = SurfaceTempInput(
                    process_temperature_c=loc["process_temperature_c"],
                    ambient_temperature_c=ambient,
                    insulation_thickness_mm=loc["insulation_thickness_mm"],
                    insulation_material=loc["insulation_material"],
                    pipe_outer_diameter_mm=loc.get("pipe_outer_diameter_mm"),
                    surface_emissivity=0.9
                )

                try:
                    surf_result = self._tools.calculate_surface_temperature(surf_input)
                    surface_temp = float(surf_result.surface_temperature_c)
                except Exception as e:
                    logger.warning(f"Surface temp calculation failed for {loc['location_id']}: {e}")
                    surface_temp = ambient + 10  # Default estimate

            # Calculate heat loss
            heat_loss_input = HeatLossInput(
                process_temperature_c=loc["process_temperature_c"],
                ambient_temperature_c=ambient,
                surface_temperature_c=surface_temp,
                pipe_outer_diameter_mm=loc.get("pipe_outer_diameter_mm"),
                pipe_length_m=loc.get("pipe_length_m", 1.0),
                surface_area_m2=loc.get("surface_area_m2"),
                insulation_thickness_mm=loc["insulation_thickness_mm"],
                insulation_material=loc["insulation_material"],
                surface_emissivity=0.9
            )

            try:
                heat_loss_result = self._tools.calculate_heat_loss(heat_loss_input)
                heat_loss_results.append(heat_loss_result)
            except Exception as e:
                logger.error(f"Heat loss calculation failed for {loc['location_id']}: {e}")
                if context.config.fail_fast:
                    raise

        total_heat_loss = sum(float(r.total_heat_loss_w) for r in heat_loss_results)
        logger.info(
            f"[{context.correlation_id}] Heat loss calculated: "
            f"{len(heat_loss_results)} locations, {total_heat_loss:.0f}W total"
        )

        return {
            **analysis_data,
            "heat_loss_results": heat_loss_results
        }


class DegradationAssessmentHandler(StageHandler):
    """Handler for degradation assessment stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.DEGRADATION_ASSESSMENT

    async def execute(
        self,
        context: 'PipelineContext',
        heat_loss_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess insulation degradation at each location."""
        logger.info(f"[{context.correlation_id}] Assessing degradation")

        locations = heat_loss_data.get("locations", [])
        heat_loss_results = heat_loss_data.get("heat_loss_results", [])
        inspection_date = heat_loss_data["inspection_date"]

        degradation_results = []
        rul_results = []

        # Map heat loss results by location
        heat_loss_map = {}
        for hl in heat_loss_results:
            # Use location_id from result if available
            heat_loss_map[hl.location_id] = hl

        for i, loc in enumerate(locations):
            # Get heat loss for this location
            if i < len(heat_loss_results):
                current_loss = float(heat_loss_results[i].heat_loss_w_per_m)
            else:
                current_loss = 100.0  # Default

            # Estimate design heat loss (typically 10-20% of current for well-maintained)
            design_loss = current_loss * 0.3  # Assuming 30% baseline

            degradation_input = DegradationInput(
                location_id=loc["location_id"],
                current_heat_loss_w_per_m=current_loss,
                design_heat_loss_w_per_m=design_loss,
                installation_date=loc["installation_date"],
                inspection_date=inspection_date,
                insulation_material=loc["insulation_material"],
                process_temperature_c=loc["process_temperature_c"],
                environment_type="outdoor_industrial",
                moisture_detected=loc.get("moisture_detected", False),
                mechanical_damage_observed=loc.get("mechanical_damage_observed", False),
                jacket_condition_score=loc.get("jacket_condition_score", 5)
            )

            try:
                degradation_result = self._tools.assess_degradation(degradation_input)
                degradation_results.append(degradation_result)

                # Estimate RUL
                rul_input = RULInput(
                    location_id=loc["location_id"],
                    current_condition_score=float(degradation_result.condition_score),
                    degradation_rate_per_year=float(degradation_result.degradation_rate_per_year),
                    failure_threshold=30.0,
                    installation_date=loc["installation_date"],
                    process_temperature_c=loc["process_temperature_c"],
                    environment_severity=1.2,
                    maintenance_factor=1.0
                )

                rul_result = self._tools.estimate_remaining_life(rul_input)
                rul_results.append(rul_result)

            except Exception as e:
                logger.error(f"Degradation assessment failed for {loc['location_id']}: {e}")
                if context.config.fail_fast:
                    raise

        logger.info(
            f"[{context.correlation_id}] Degradation assessed: {len(degradation_results)} locations"
        )

        return {
            **heat_loss_data,
            "degradation_results": degradation_results,
            "rul_results": rul_results
        }


class EnergyQuantificationHandler(StageHandler):
    """Handler for energy quantification stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.ENERGY_QUANTIFICATION

    async def execute(
        self,
        context: 'PipelineContext',
        degradation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quantify facility-wide energy loss."""
        logger.info(f"[{context.correlation_id}] Quantifying energy loss")

        locations = degradation_data.get("locations", [])
        heat_loss_results = degradation_data.get("heat_loss_results", [])
        degradation_results = degradation_data.get("degradation_results", [])
        economic_params = degradation_data.get("economic_params", {})

        # Build location data for energy tool
        energy_locations = []
        for i, loc in enumerate(locations):
            if i < len(heat_loss_results):
                heat_loss = float(heat_loss_results[i].total_heat_loss_w)
            else:
                heat_loss = 0

            if i < len(degradation_results):
                condition = degradation_results[i].primary_degradation_mode
            else:
                condition = "unknown"

            energy_locations.append({
                "location_id": loc["location_id"],
                "heat_loss_w": heat_loss,
                "system_type": loc.get("system_type", "unknown"),
                "condition": condition
            })

        energy_input = EnergyInput(
            locations=energy_locations,
            fuel_type=economic_params.get("fuel_type", "natural_gas"),
            boiler_efficiency=economic_params.get("boiler_efficiency", 0.85),
            operating_hours_per_year=economic_params.get("operating_hours_per_year", 8000),
            energy_cost_per_mmbtu=economic_params.get("energy_cost_per_mmbtu", 4.50)
        )

        energy_result = self._tools.quantify_energy_loss(energy_input)

        # Calculate carbon footprint
        carbon_input = CarbonInput(
            total_energy_loss_mmbtu=float(energy_result.annual_energy_loss_mmbtu),
            fuel_type=economic_params.get("fuel_type", "natural_gas"),
            include_scope_2=True,
            carbon_price_scenarios={
                "low": 25.0,
                "current": economic_params.get("carbon_price_per_tonne", 50.0),
                "high": 150.0
            }
        )

        carbon_result = self._tools.calculate_carbon_footprint(carbon_input)

        logger.info(
            f"[{context.correlation_id}] Energy quantified: "
            f"{energy_result.annual_energy_loss_mmbtu} MMBtu/yr, "
            f"{carbon_result.total_emissions_tonnes_co2e} tonnes CO2e/yr"
        )

        return {
            **degradation_data,
            "energy_loss_result": energy_result,
            "carbon_result": carbon_result
        }


class RepairPrioritizationHandler(StageHandler):
    """Handler for repair prioritization stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.REPAIR_PRIORITIZATION

    async def execute(
        self,
        context: 'PipelineContext',
        energy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prioritize repairs by criticality."""
        logger.info(f"[{context.correlation_id}] Prioritizing repairs")

        locations = energy_data.get("locations", [])
        heat_loss_results = energy_data.get("heat_loss_results", [])
        degradation_results = energy_data.get("degradation_results", [])
        analysis_params = energy_data.get("analysis_params", {})
        economic_params = energy_data.get("economic_params", {})

        # Build defect list for prioritization
        defects = []
        for i, loc in enumerate(locations):
            heat_loss = float(heat_loss_results[i].heat_loss_w_per_m) if i < len(heat_loss_results) else 0
            surface_temp = float(heat_loss_results[i].surface_temperature_c) if i < len(heat_loss_results) else 25

            # Estimate annual savings from repair
            operating_hours = economic_params.get("operating_hours_per_year", 8000)
            energy_cost_kwh = economic_params.get("energy_cost_per_mmbtu", 4.50) / 293.07  # MMBtu to kWh

            # Assume repair reduces heat loss by 70%
            annual_kwh_savings = heat_loss * 0.7 * operating_hours / 1000
            annual_savings = annual_kwh_savings * energy_cost_kwh

            defects.append({
                "defect_id": loc["location_id"],
                "equipment_tag": loc["equipment_tag"],
                "heat_loss_w_per_m": heat_loss,
                "surface_temperature_c": surface_temp,
                "process_temperature_c": loc["process_temperature_c"],
                "ambient_temperature_c": energy_data.get("ambient_temperature_c", 25),
                "moisture_detected": loc.get("moisture_detected", False),
                "estimated_repair_cost_usd": loc.get("estimated_repair_cost_usd", 1000),
                "annual_energy_savings_usd": annual_savings
            })

        repair_input = RepairInput(
            defects=defects,
            heat_loss_weight=0.25,
            safety_risk_weight=0.25,
            process_impact_weight=0.20,
            environmental_weight=0.15,
            asset_protection_weight=0.15,
            budget_constraint_usd=analysis_params.get("repair_budget_usd")
        )

        repair_result = self._tools.prioritize_repairs(repair_input)

        emergency_count = len(repair_result.emergency_repairs)
        urgent_count = len(repair_result.urgent_repairs)
        logger.info(
            f"[{context.correlation_id}] Repairs prioritized: "
            f"{emergency_count} emergency, {urgent_count} urgent, "
            f"${repair_result.total_estimated_cost_usd} total cost"
        )

        return {
            **energy_data,
            "repair_priority_result": repair_result,
            "defects_for_roi": defects
        }


class EconomicAnalysisHandler(StageHandler):
    """Handler for economic analysis stage."""

    def __init__(self, tools: InsulationInspectionTools):
        self._tools = tools

    def get_stage(self) -> PipelineStage:
        return PipelineStage.ECONOMIC_ANALYSIS

    async def execute(
        self,
        context: 'PipelineContext',
        repair_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ROI for repairs."""
        logger.info(f"[{context.correlation_id}] Running economic analysis")

        defects = repair_data.get("defects_for_roi", [])
        economic_params = repair_data.get("economic_params", {})

        roi_results = []

        for defect in defects:
            repair_cost = defect.get("estimated_repair_cost_usd", 1000)
            heat_loss = defect.get("heat_loss_w_per_m", 0)

            # Calculate annual kWh savings
            operating_hours = economic_params.get("operating_hours_per_year", 8000)
            # Assume 70% heat loss reduction from repair
            annual_kwh = heat_loss * 0.7 * operating_hours / 1000

            roi_input = ROIInput(
                defect_id=defect["defect_id"],
                repair_cost_usd=repair_cost,
                annual_energy_savings_kwh=annual_kwh,
                energy_cost_per_kwh=0.12,
                equipment_life_years=15,
                discount_rate_percent=8.0,
                carbon_price_per_tonne=economic_params.get("carbon_price_per_tonne", 50.0),
                co2_emission_factor_kg_per_kwh=0.417
            )

            try:
                roi_result = self._tools.calculate_repair_roi(roi_input)
                roi_results.append(roi_result)
            except Exception as e:
                logger.warning(f"ROI calculation failed for {defect['defect_id']}: {e}")

        positive_npv_count = sum(1 for r in roi_results if float(r.npv_over_life_usd) > 0)
        logger.info(
            f"[{context.correlation_id}] Economic analysis complete: "
            f"{len(roi_results)} ROI calculations, {positive_npv_count} positive NPV"
        )

        return {
            **repair_data,
            "roi_results": roi_results
        }


class ReportGenerationHandler(StageHandler):
    """Handler for report generation stage."""

    def get_stage(self) -> PipelineStage:
        return PipelineStage.REPORT_GENERATION

    async def execute(
        self,
        context: 'PipelineContext',
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate inspection report summary."""
        logger.info(f"[{context.correlation_id}] Generating report")

        energy_result = analysis_data.get("energy_loss_result")
        carbon_result = analysis_data.get("carbon_result")
        repair_result = analysis_data.get("repair_priority_result")
        degradation_results = analysis_data.get("degradation_results", [])
        roi_results = analysis_data.get("roi_results", [])

        # Build summary
        summary = {
            "inspection_id": analysis_data["inspection_id"],
            "facility_id": analysis_data["facility_id"],
            "facility_name": analysis_data["facility_name"],
            "inspection_date": analysis_data["inspection_date"],
            "inspection_type": analysis_data["inspection_type"],
            "total_locations_inspected": analysis_data["location_count"],
            "total_images_analyzed": analysis_data["image_count"],

            "energy_summary": {
                "total_heat_loss_w": str(energy_result.total_heat_loss_w) if energy_result else "N/A",
                "annual_energy_loss_mmbtu": str(energy_result.annual_energy_loss_mmbtu) if energy_result else "N/A",
                "annual_energy_cost_usd": str(energy_result.annual_energy_cost_usd) if energy_result else "N/A"
            } if energy_result else None,

            "carbon_summary": {
                "annual_emissions_tonnes_co2e": str(carbon_result.total_emissions_tonnes_co2e) if carbon_result else "N/A",
                "carbon_cost_current_usd": carbon_result.carbon_cost_by_scenario.get("current", "N/A") if carbon_result else "N/A"
            } if carbon_result else None,

            "repair_summary": {
                "emergency_repairs": len(repair_result.emergency_repairs) if repair_result else 0,
                "urgent_repairs": len(repair_result.urgent_repairs) if repair_result else 0,
                "high_priority_repairs": len(repair_result.high_priority_repairs) if repair_result else 0,
                "total_repair_cost_usd": str(repair_result.total_estimated_cost_usd) if repair_result else "N/A",
                "aggregate_npv_usd": str(repair_result.aggregate_npv_usd) if repair_result else "N/A"
            } if repair_result else None,

            "condition_summary": {
                "average_condition_score": sum(float(d.condition_score) for d in degradation_results) / len(degradation_results) if degradation_results else 0,
                "locations_below_50_condition": sum(1 for d in degradation_results if float(d.condition_score) < 50),
                "high_cui_risk_locations": sum(1 for d in degradation_results if d.cui_risk_level == "high")
            }
        }

        # Generate recommendations
        recommendations = []

        if repair_result and repair_result.emergency_repairs:
            recommendations.append(
                f"IMMEDIATE ACTION: {len(repair_result.emergency_repairs)} emergency repairs required"
            )

        if repair_result and repair_result.urgent_repairs:
            recommendations.append(
                f"Schedule {len(repair_result.urgent_repairs)} urgent repairs within 30 days"
            )

        high_cui_count = sum(1 for d in degradation_results if d.cui_risk_level == "high")
        if high_cui_count > 0:
            recommendations.append(
                f"Investigate {high_cui_count} locations for Corrosion Under Insulation (CUI)"
            )

        if energy_result and float(energy_result.annual_energy_cost_usd) > 100000:
            recommendations.append(
                "Significant energy loss detected - implement systematic repair program"
            )

        positive_roi = [r for r in roi_results if float(r.npv_over_life_usd) > 0]
        if positive_roi:
            recommendations.append(
                f"{len(positive_roi)} repairs have positive NPV - prioritize for budget allocation"
            )

        # Generate alerts
        alerts = []

        if repair_result and repair_result.emergency_repairs:
            for repair in repair_result.emergency_repairs:
                alerts.append({
                    "severity": "critical",
                    "location": repair.get("equipment_tag", "Unknown"),
                    "message": "Emergency repair required - safety risk"
                })

        for d in degradation_results:
            if d.cui_risk_level == "high":
                alerts.append({
                    "severity": "warning",
                    "location": d.location_id,
                    "message": f"High CUI risk - {d.recommended_action}"
                })

        logger.info(
            f"[{context.correlation_id}] Report generated: "
            f"{len(recommendations)} recommendations, {len(alerts)} alerts"
        )

        return {
            **analysis_data,
            "summary": summary,
            "recommendations": recommendations,
            "alerts": alerts
        }


class ResultCachingHandler(StageHandler):
    """Handler for result caching stage."""

    def __init__(self, cache_manager: CacheManager):
        self._cache = cache_manager

    def get_stage(self) -> PipelineStage:
        return PipelineStage.RESULT_CACHING

    async def execute(
        self,
        context: 'PipelineContext',
        report_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cache final results."""
        logger.info(f"[{context.correlation_id}] Caching results")

        inspection_id = report_data["inspection_id"]

        # Cache summary
        cache_key = f"inspection:{inspection_id}"
        cache_data = {
            "summary": report_data.get("summary"),
            "recommendations": report_data.get("recommendations"),
            "alerts": report_data.get("alerts"),
            "cached_at": datetime.utcnow().isoformat()
        }

        await self._cache.set(cache_key, cache_data)

        logger.info(f"[{context.correlation_id}] Results cached: {cache_key}")

        return report_data


# =============================================================================
# PIPELINE CONTEXT
# =============================================================================

@dataclass
class PipelineContext:
    """Context object passed through pipeline stages."""
    pipeline_id: str
    correlation_id: str
    config: PipelineConfig
    start_time: datetime
    tools: InsulationInspectionTools
    checkpoint_manager: CheckpointManager
    cache_manager: CacheManager
    completed_stages: List[PipelineStage] = field(default_factory=list)
    stage_results: Dict[str, StepResult] = field(default_factory=dict)
    current_data: Optional[Dict[str, Any]] = None


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class InsulationInspectionOrchestrator:
    """
    Production-grade pipeline orchestrator for insulation inspection.

    Manages the complete inspection workflow with:
    - Multi-stage pipeline execution
    - Checkpoint management for recovery
    - Async processing
    - Error handling and retry logic
    - Result caching
    - Complete provenance tracking

    Example:
        >>> config = PipelineConfig(timeout_seconds=600)
        >>> orchestrator = InsulationInspectionOrchestrator(config)
        >>> await orchestrator.initialize()
        >>> result = await orchestrator.run_inspection(input_data)
        >>> await orchestrator.shutdown()
    """

    VERSION = "1.0.0"

    # Pipeline stage sequence
    STAGE_SEQUENCE = [
        PipelineStage.INPUT_VALIDATION,
        PipelineStage.IMAGE_PREPROCESSING,
        PipelineStage.THERMAL_ANALYSIS,
        PipelineStage.HEAT_LOSS_CALCULATION,
        PipelineStage.DEGRADATION_ASSESSMENT,
        PipelineStage.ENERGY_QUANTIFICATION,
        PipelineStage.REPAIR_PRIORITIZATION,
        PipelineStage.ECONOMIC_ANALYSIS,
        PipelineStage.REPORT_GENERATION,
        PipelineStage.RESULT_CACHING,
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration
        """
        self._config = config or PipelineConfig()
        self._tools = InsulationInspectionTools()
        self._checkpoint_manager = CheckpointManager(self._config.checkpoint_config)
        self._cache_manager = CacheManager(self._config.cache_config)
        self._handlers: Dict[PipelineStage, StageHandler] = {}
        self._is_initialized = False

        logger.info(f"InsulationInspectionOrchestrator created v{self.VERSION}")

    async def initialize(self) -> None:
        """Initialize orchestrator and its components."""
        if self._is_initialized:
            return

        logger.info("Initializing InsulationInspectionOrchestrator")

        # Initialize cache
        await self._cache_manager.initialize()

        # Register stage handlers
        self._handlers = {
            PipelineStage.INPUT_VALIDATION: InputValidationHandler(),
            PipelineStage.IMAGE_PREPROCESSING: ImagePreprocessingHandler(),
            PipelineStage.THERMAL_ANALYSIS: ThermalAnalysisHandler(self._tools),
            PipelineStage.HEAT_LOSS_CALCULATION: HeatLossCalculationHandler(self._tools),
            PipelineStage.DEGRADATION_ASSESSMENT: DegradationAssessmentHandler(self._tools),
            PipelineStage.ENERGY_QUANTIFICATION: EnergyQuantificationHandler(self._tools),
            PipelineStage.REPAIR_PRIORITIZATION: RepairPrioritizationHandler(self._tools),
            PipelineStage.ECONOMIC_ANALYSIS: EconomicAnalysisHandler(self._tools),
            PipelineStage.REPORT_GENERATION: ReportGenerationHandler(),
            PipelineStage.RESULT_CACHING: ResultCachingHandler(self._cache_manager),
        }

        self._is_initialized = True
        logger.info("InsulationInspectionOrchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources."""
        logger.info("Shutting down InsulationInspectionOrchestrator")
        await self._cache_manager.close()
        self._is_initialized = False
        logger.info("InsulationInspectionOrchestrator shutdown complete")

    async def run_inspection(
        self,
        input_data: InspectionInputData,
        resume_from_checkpoint: bool = False
    ) -> InspectionResult:
        """
        Run complete insulation inspection pipeline.

        Args:
            input_data: Inspection input data
            resume_from_checkpoint: Whether to resume from checkpoint if available

        Returns:
            Complete inspection result

        Raises:
            ValueError: If orchestrator not initialized
            TimeoutError: If pipeline exceeds timeout
        """
        if not self._is_initialized:
            raise ValueError("Orchestrator not initialized. Call initialize() first.")

        pipeline_id = self._config.pipeline_id
        correlation_id = self._config.correlation_id or str(uuid.uuid4())
        start_time = datetime.utcnow()

        logger.info(
            f"[{correlation_id}] Starting inspection pipeline {pipeline_id} "
            f"for facility {input_data.facility_id}"
        )

        # Create context
        context = PipelineContext(
            pipeline_id=pipeline_id,
            correlation_id=correlation_id,
            config=self._config,
            start_time=start_time,
            tools=self._tools,
            checkpoint_manager=self._checkpoint_manager,
            cache_manager=self._cache_manager
        )

        # Calculate input hash for checkpoint validation
        input_hash = hashlib.sha256(
            input_data.json().encode()
        ).hexdigest()[:16]

        # Check for checkpoint
        stages_to_run = list(self.STAGE_SEQUENCE)
        if resume_from_checkpoint:
            checkpoint = await self._checkpoint_manager.load_checkpoint(pipeline_id)
            if checkpoint and checkpoint.input_data_hash == input_hash:
                context.completed_stages = checkpoint.completed_stages
                context.current_data = checkpoint.stage_results.get("current_data")
                stages_to_run = checkpoint.pending_stages
                logger.info(
                    f"[{correlation_id}] Resuming from checkpoint, "
                    f"{len(context.completed_stages)} stages completed"
                )

        # Initialize result
        result = InspectionResult(
            pipeline_id=pipeline_id,
            correlation_id=correlation_id,
            status=PipelineStatus.RUNNING,
            inspection_id=input_data.inspection_id,
            facility_id=input_data.facility_id,
            start_time=start_time
        )

        try:
            # Run pipeline with timeout
            current_data = context.current_data or input_data

            async with asyncio.timeout(self._config.timeout_seconds):
                for stage in stages_to_run:
                    step_result = await self._execute_stage(
                        context, stage, current_data
                    )

                    result.step_results[stage.value] = step_result

                    if step_result.status == StepStatus.FAILED:
                        if self._config.fail_fast:
                            raise RuntimeError(f"Stage {stage.value} failed: {step_result.error}")
                        logger.warning(f"[{correlation_id}] Stage {stage.value} failed, continuing")

                    if step_result.result is not None:
                        current_data = step_result.result
                        context.current_data = current_data

                    context.completed_stages.append(stage)

                    # Save checkpoint
                    pending = [s for s in stages_to_run if s not in context.completed_stages]
                    await self._checkpoint_manager.save_checkpoint(
                        pipeline_id=pipeline_id,
                        completed_stage=stage,
                        completed_stages=context.completed_stages,
                        pending_stages=pending,
                        stage_results={"current_data": current_data},
                        input_data_hash=input_hash
                    )

            # Extract final results
            result = self._build_final_result(result, current_data)
            result.status = PipelineStatus.COMPLETED

            # Clean up checkpoint on success
            await self._checkpoint_manager.delete_checkpoint(pipeline_id)

        except asyncio.TimeoutError:
            logger.error(f"[{correlation_id}] Pipeline timeout after {self._config.timeout_seconds}s")
            result.status = PipelineStatus.FAILED
            result.alerts = [{"severity": "critical", "message": "Pipeline timeout"}]

        except Exception as e:
            logger.error(f"[{correlation_id}] Pipeline failed: {e}", exc_info=True)
            result.status = PipelineStatus.FAILED
            result.alerts = [{"severity": "critical", "message": str(e)}]

        finally:
            result.end_time = datetime.utcnow()
            result.duration_ms = (result.end_time - start_time).total_seconds() * 1000

            # Calculate provenance hash
            result.provenance_hash = self._calculate_provenance_hash(result)

            logger.info(
                f"[{correlation_id}] Pipeline {result.status.value} "
                f"in {result.duration_ms:.0f}ms"
            )

        return result

    async def _execute_stage(
        self,
        context: PipelineContext,
        stage: PipelineStage,
        input_data: Any
    ) -> StepResult:
        """Execute a single pipeline stage with retry logic."""
        handler = self._handlers.get(stage)

        if handler is None:
            return StepResult(
                stage=stage,
                status=StepStatus.SKIPPED,
                error="No handler registered"
            )

        start_time = datetime.utcnow()
        retry_count = 0
        max_retries = context.config.retry_config.max_retries

        while True:
            try:
                if context.config.enable_detailed_logging:
                    logger.debug(f"[{context.correlation_id}] Executing stage: {stage.value}")

                result = await handler.execute(context, input_data)

                end_time = datetime.utcnow()
                duration_ms = (end_time - start_time).total_seconds() * 1000

                return StepResult(
                    stage=stage,
                    status=StepStatus.COMPLETED,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    retry_count=retry_count
                )

            except Exception as e:
                retry_count += 1

                if retry_count > max_retries:
                    end_time = datetime.utcnow()
                    duration_ms = (end_time - start_time).total_seconds() * 1000

                    return StepResult(
                        stage=stage,
                        status=StepStatus.FAILED,
                        error=str(e),
                        error_traceback=traceback.format_exc(),
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        retry_count=retry_count
                    )

                # Calculate retry delay with exponential backoff
                delay = min(
                    context.config.retry_config.initial_delay_seconds * (
                        context.config.retry_config.exponential_base ** (retry_count - 1)
                    ),
                    context.config.retry_config.max_delay_seconds
                )

                logger.warning(
                    f"[{context.correlation_id}] Stage {stage.value} failed "
                    f"(attempt {retry_count}/{max_retries}), retrying in {delay:.1f}s: {e}"
                )

                await asyncio.sleep(delay)

    def _build_final_result(
        self,
        result: InspectionResult,
        final_data: Dict[str, Any]
    ) -> InspectionResult:
        """Build final inspection result from pipeline data."""
        result.validated_input = final_data.get("validated_input")
        result.thermal_analysis_results = final_data.get("thermal_analysis_results")
        result.hotspot_results = final_data.get("hotspot_results")
        result.anomaly_classifications = final_data.get("anomaly_classifications")
        result.heat_loss_results = final_data.get("heat_loss_results")
        result.degradation_results = final_data.get("degradation_results")
        result.rul_results = final_data.get("rul_results")
        result.energy_loss_result = final_data.get("energy_loss_result")
        result.carbon_result = final_data.get("carbon_result")
        result.repair_priority_result = final_data.get("repair_priority_result")
        result.roi_results = final_data.get("roi_results")
        result.summary = final_data.get("summary")
        result.recommendations = final_data.get("recommendations")
        result.alerts = final_data.get("alerts")

        return result

    def _calculate_provenance_hash(self, result: InspectionResult) -> str:
        """Calculate SHA-256 provenance hash for the result."""
        provenance_data = {
            "pipeline_id": result.pipeline_id,
            "inspection_id": result.inspection_id,
            "facility_id": result.facility_id,
            "status": result.status.value,
            "duration_ms": result.duration_ms,
            "step_count": len(result.step_results),
            "timestamp": result.end_time.isoformat() if result.end_time else None
        }

        content = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_cached_result(
        self,
        inspection_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached inspection result."""
        cache_key = f"inspection:{inspection_id}"
        return await self._cache_manager.get(cache_key)

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "version": self.VERSION,
            "is_initialized": self._is_initialized,
            "stage_count": len(self.STAGE_SEQUENCE),
            "handler_count": len(self._handlers),
            "cache_stats": self._cache_manager.get_statistics(),
            "tool_stats": self._tools.get_tool_statistics()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_orchestrator(
    timeout_seconds: float = 600.0,
    enable_caching: bool = True,
    enable_checkpoints: bool = True,
    redis_url: Optional[str] = None
) -> InsulationInspectionOrchestrator:
    """
    Factory function to create configured orchestrator.

    Args:
        timeout_seconds: Pipeline timeout in seconds
        enable_caching: Whether to enable result caching
        enable_checkpoints: Whether to enable checkpoint management
        redis_url: Redis URL for distributed caching

    Returns:
        Configured InsulationInspectionOrchestrator instance
    """
    config = PipelineConfig(
        timeout_seconds=timeout_seconds,
        cache_config=CacheConfig(
            enabled=enable_caching,
            redis_url=redis_url
        ),
        checkpoint_config=CheckpointConfig(
            enabled=enable_checkpoints
        )
    )

    return InsulationInspectionOrchestrator(config)


async def run_inspection(
    input_data: InspectionInputData,
    config: Optional[PipelineConfig] = None
) -> InspectionResult:
    """
    Convenience function to run inspection with default orchestrator.

    Args:
        input_data: Inspection input data
        config: Optional pipeline configuration

    Returns:
        Inspection result
    """
    orchestrator = InsulationInspectionOrchestrator(config)
    await orchestrator.initialize()

    try:
        return await orchestrator.run_inspection(input_data)
    finally:
        await orchestrator.shutdown()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "PipelineStatus",
    "StepStatus",
    "PipelineStage",
    "ErrorSeverity",
    "InspectionType",

    # Configuration
    "RetryConfig",
    "CacheConfig",
    "CheckpointConfig",
    "PipelineConfig",

    # Input/Output models
    "InspectionLocation",
    "ThermalImageData",
    "InspectionInputData",
    "StepResult",
    "InspectionResult",

    # Checkpoint and cache
    "Checkpoint",
    "CheckpointManager",
    "CacheManager",

    # Pipeline context
    "PipelineContext",

    # Stage handlers
    "StageHandler",
    "InputValidationHandler",
    "ImagePreprocessingHandler",
    "ThermalAnalysisHandler",
    "HeatLossCalculationHandler",
    "DegradationAssessmentHandler",
    "EnergyQuantificationHandler",
    "RepairPrioritizationHandler",
    "EconomicAnalysisHandler",
    "ReportGenerationHandler",
    "ResultCachingHandler",

    # Main orchestrator
    "InsulationInspectionOrchestrator",

    # Factory functions
    "create_orchestrator",
    "run_inspection",
]
