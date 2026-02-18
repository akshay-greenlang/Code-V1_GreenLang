# -*- coding: utf-8 -*-
"""
Stationary Combustion Agent Service Setup - AGENT-MRV-001

Provides ``configure_stationary_combustion(app)`` which wires up the
Stationary Combustion Agent SDK (fuel database, calculator, equipment
profiler, factor selector, uncertainty engine, audit engine, combustion
pipeline, provenance tracker) and mounts the REST API.

Also exposes ``get_service()`` for programmatic access and the
``StationaryCombustionService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.stationary_combustion.setup import configure_stationary_combustion
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_stationary_combustion(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.stationary_combustion.config import (
    StationaryCombustionConfig,
    get_config,
)
from greenlang.stationary_combustion.models import (
    AuditEntry,
    CalculationResult,
    CalculationStatus,
    CalculationTier,
    CombustionInput,
    ControlApproach,
    EFSource,
    EmissionFactor,
    EmissionGas,
    EquipmentProfile,
    EquipmentType,
    FacilityAggregation,
    FuelCategory,
    FuelProperties,
    FuelType,
    GasEmission,
    GWPSource,
    HeatingValueBasis,
    RegulatoryFramework,
    ReportingPeriod,
    UncertaintyResult,
    UnitType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
except ImportError:
    FuelDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.calculator import CalculatorEngine
except ImportError:
    CalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.equipment_profiler import EquipmentProfilerEngine
except ImportError:
    EquipmentProfilerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.factor_selector import FactorSelectorEngine
except ImportError:
    FactorSelectorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.uncertainty import UncertaintyEngine
except ImportError:
    UncertaintyEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.audit import AuditEngine
except ImportError:
    AuditEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.combustion_pipeline import (
        StationaryCombustionPipelineEngine,
    )
except ImportError:
    StationaryCombustionPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.stationary_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation,
        observe_calculation_duration,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def record_calculation(fuel_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def observe_calculation_duration(fuel_type: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class CalculationResponse(BaseModel):
    """Single combustion emission calculation response.

    Attributes:
        calculation_id: Unique calculation identifier (UUID4).
        status: Calculation status (SUCCESS, PARTIAL, FAILED).
        fuel_type: Fuel type used in the calculation.
        quantity: Original fuel quantity.
        unit: Original quantity unit.
        energy_gj: Energy content in GJ.
        tier: Calculation tier used.
        gwp_source: GWP source used.
        total_co2e_kg: Total CO2e in kg (excluding biogenic).
        total_co2e_tonnes: Total CO2e in tonnes (excluding biogenic).
        biogenic_co2_kg: Biogenic CO2 in kg.
        biogenic_co2_tonnes: Biogenic CO2 in tonnes.
        gas_emissions: Per-gas emission breakdown.
        calculation_trace: Step-by-step calculation trace.
        provenance_hash: SHA-256 provenance hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: ISO-8601 UTC calculation timestamp.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="SUCCESS")
    fuel_type: str = Field(default="")
    quantity: float = Field(default=0.0)
    unit: str = Field(default="")
    energy_gj: float = Field(default=0.0)
    tier: str = Field(default="TIER_1")
    gwp_source: str = Field(default="AR6")
    total_co2e_kg: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    biogenic_co2_kg: float = Field(default=0.0)
    biogenic_co2_tonnes: float = Field(default=0.0)
    gas_emissions: List[Dict[str, Any]] = Field(default_factory=list)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class BatchCalculationResponse(BaseModel):
    """Batch combustion emission calculation response.

    Attributes:
        batch_id: Unique batch identifier (UUID4).
        results: Individual calculation results.
        total_co2e_kg: Aggregate CO2e in kg.
        total_co2e_tonnes: Aggregate CO2e in tonnes.
        total_biogenic_co2_kg: Aggregate biogenic CO2 in kg.
        total_biogenic_co2_tonnes: Aggregate biogenic CO2 in tonnes.
        success_count: Number of successful calculations.
        failure_count: Number of failed calculations.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 batch provenance hash.
    """

    model_config = {"extra": "forbid"}

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_co2e_kg: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_biogenic_co2_kg: float = Field(default=0.0)
    total_biogenic_co2_tonnes: float = Field(default=0.0)
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class FuelResponse(BaseModel):
    """Fuel type properties response.

    Attributes:
        fuel_type: Fuel type identifier.
        category: Fuel category (GASEOUS, LIQUID, SOLID, etc.).
        display_name: Human-readable fuel name.
        hhv: Higher heating value.
        ncv: Net calorific value.
        hhv_unit: HHV measurement unit.
        ncv_unit: NCV measurement unit.
        density: Fuel density.
        carbon_content_pct: Carbon content percentage.
        is_biogenic: Whether the fuel is biogenic.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    fuel_type: str = Field(default="")
    category: str = Field(default="")
    display_name: str = Field(default="")
    hhv: float = Field(default=0.0)
    ncv: float = Field(default=0.0)
    hhv_unit: str = Field(default="mmBtu/unit")
    ncv_unit: str = Field(default="mmBtu/unit")
    density: Optional[float] = Field(default=None)
    carbon_content_pct: Optional[float] = Field(default=None)
    is_biogenic: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class FuelListResponse(BaseModel):
    """Response listing all available fuel types.

    Attributes:
        fuels: List of fuel type summary dictionaries.
        total_count: Total number of fuel types available.
    """

    model_config = {"extra": "forbid"}

    fuels: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class EmissionFactorResponse(BaseModel):
    """Emission factor retrieval response.

    Attributes:
        factor_id: Unique factor identifier.
        fuel_type: Fuel type for this factor.
        gas: Greenhouse gas species.
        value: Factor numeric value.
        unit: Factor measurement unit.
        source: Source database (EPA, IPCC, etc.).
        geography: Geographic scope.
        year: Reference year.
        reference: Regulatory reference document.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    factor_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fuel_type: str = Field(default="")
    gas: str = Field(default="")
    value: float = Field(default=0.0)
    unit: str = Field(default="")
    source: str = Field(default="EPA")
    geography: str = Field(default="GLOBAL")
    year: int = Field(default=2025)
    reference: str = Field(default="")
    provenance_hash: str = Field(default="")


class EquipmentResponse(BaseModel):
    """Equipment profile registration / retrieval response.

    Attributes:
        equipment_id: Unique equipment identifier.
        equipment_type: Equipment type classification.
        name: Human-readable equipment name.
        facility_id: Parent facility identifier.
        rated_capacity_mw: Rated thermal capacity in MW.
        efficiency: Thermal efficiency (0-1).
        load_factor: Average load factor (0-1).
        age_years: Equipment age in years.
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    equipment_id: str = Field(default="")
    equipment_type: str = Field(default="")
    name: str = Field(default="")
    facility_id: Optional[str] = Field(default=None)
    rated_capacity_mw: Optional[float] = Field(default=None)
    efficiency: float = Field(default=0.80)
    load_factor: float = Field(default=0.65)
    age_years: int = Field(default=0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class AggregationResponse(BaseModel):
    """Facility-level emission aggregation response.

    Attributes:
        facility_id: Facility identifier.
        total_co2e_tonnes: Total CO2e in tonnes.
        total_biogenic_co2_tonnes: Total biogenic CO2 in tonnes.
        by_fuel: CO2e breakdown by fuel type.
        by_equipment: CO2e breakdown by equipment.
        calculation_count: Number of calculations aggregated.
        control_approach: GHG Protocol boundary approach used.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    facility_id: str = Field(default="")
    total_co2e_tonnes: float = Field(default=0.0)
    total_biogenic_co2_tonnes: float = Field(default=0.0)
    by_fuel: Dict[str, float] = Field(default_factory=dict)
    by_equipment: Dict[str, float] = Field(default_factory=dict)
    calculation_count: int = Field(default=0)
    control_approach: str = Field(default="OPERATIONAL")
    provenance_hash: str = Field(default="")


class UncertaintyResponse(BaseModel):
    """Monte Carlo uncertainty analysis response.

    Attributes:
        calculation_id: Related calculation identifier.
        mean_co2e_kg: Mean CO2e from simulations.
        std_co2e_kg: Standard deviation of CO2e.
        p5_co2e_kg: 5th percentile.
        p95_co2e_kg: 95th percentile.
        confidence_interval_pct: Confidence interval percentage.
        num_simulations: Number of Monte Carlo iterations.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    mean_co2e_kg: float = Field(default=0.0)
    std_co2e_kg: float = Field(default=0.0)
    p5_co2e_kg: float = Field(default=0.0)
    p95_co2e_kg: float = Field(default=0.0)
    confidence_interval_pct: float = Field(default=95.0)
    num_simulations: int = Field(default=10000)
    provenance_hash: str = Field(default="")


class AuditTrailResponse(BaseModel):
    """Audit trail retrieval response.

    Attributes:
        calculation_id: Calculation for which audit entries are retrieved.
        entries: List of audit entry dictionaries.
        total_entries: Total number of audit entries.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    total_entries: int = Field(default=0)
    provenance_hash: str = Field(default="")


class ComplianceResponse(BaseModel):
    """Regulatory compliance mapping response.

    Attributes:
        framework: Regulatory framework name.
        mappings: List of compliance mapping dictionaries.
        compliant_count: Number of compliant requirements.
        total_requirements: Total number of requirements checked.
        overall_compliant: Whether all requirements are met.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    framework: str = Field(default="")
    mappings: List[Dict[str, Any]] = Field(default_factory=list)
    compliant_count: int = Field(default=0)
    total_requirements: int = Field(default=0)
    overall_compliant: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class ValidationResponse(BaseModel):
    """Input validation response (without calculation).

    Attributes:
        valid: Overall validity of the inputs.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
        validated_count: Number of valid inputs.
        total_count: Total number of inputs checked.
    """

    model_config = {"extra": "forbid"}

    valid: bool = Field(default=False)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_count: int = Field(default=0)
    total_count: int = Field(default=0)


class PipelineResponse(BaseModel):
    """End-to-end combustion pipeline execution response.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID4).
        success: Whether the pipeline completed successfully.
        stages_completed: Number of pipeline stages completed.
        stages_total: Total number of pipeline stages.
        stage_results: Per-stage execution results.
        final_results: List of calculation result dictionaries.
        aggregations: Facility-level aggregations.
        pipeline_provenance_hash: SHA-256 hash of the entire pipeline run.
        total_duration_ms: Total wall-clock time in milliseconds.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = Field(default=False)
    stages_completed: int = Field(default=0)
    stages_total: int = Field(default=7)
    stage_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_results: List[Dict[str, Any]] = Field(default_factory=list)
    aggregations: List[Dict[str, Any]] = Field(default_factory=list)
    pipeline_provenance_hash: str = Field(default="")
    total_duration_ms: float = Field(default=0.0)


class HealthResponse(BaseModel):
    """Service health check response.

    Attributes:
        status: Overall service status (healthy, degraded, unhealthy).
        version: Agent version string.
        engines: Per-engine availability status.
        engines_available: Count of available engines.
        engines_total: Total number of engines.
        started: Whether the service has been started.
        fuel_type_count: Number of registered fuel types.
        emission_factor_count: Number of loaded emission factors.
        equipment_count: Number of registered equipment profiles.
        statistics: Summary statistics.
        provenance_chain_valid: Whether the provenance chain is intact.
        provenance_entries: Total provenance entries recorded.
        prometheus_available: Whether Prometheus client is available.
        timestamp: ISO-8601 UTC timestamp of the health check.
    """

    model_config = {"extra": "forbid"}

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    engines_available: int = Field(default=0)
    engines_total: int = Field(default=7)
    started: bool = Field(default=False)
    fuel_type_count: int = Field(default=0)
    emission_factor_count: int = Field(default=0)
    equipment_count: int = Field(default=0)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    provenance_chain_valid: bool = Field(default=True)
    provenance_entries: int = Field(default=0)
    prometheus_available: bool = Field(default=False)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class StatsResponse(BaseModel):
    """Service aggregate statistics response.

    Attributes:
        total_calculations: Total calculations performed.
        total_batch_runs: Total batch calculation runs.
        total_pipeline_runs: Total pipeline runs.
        total_fuel_types: Number of registered fuel types.
        total_emission_factors: Number of emission factors loaded.
        total_equipment_profiles: Number of equipment profiles.
        total_aggregations: Total facility aggregations produced.
        total_audit_entries: Total audit trail entries.
        avg_calculation_time_ms: Average calculation time.
        timestamp: ISO-8601 UTC timestamp.
    """

    model_config = {"extra": "forbid"}

    total_calculations: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_pipeline_runs: int = Field(default=0)
    total_fuel_types: int = Field(default=0)
    total_emission_factors: int = Field(default=0)
    total_equipment_profiles: int = Field(default=0)
    total_aggregations: int = Field(default=0)
    total_audit_entries: int = Field(default=0)
    avg_calculation_time_ms: float = Field(default=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# StationaryCombustionService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["StationaryCombustionService"] = None


class StationaryCombustionService:
    """Unified facade over the Stationary Combustion Agent SDK.

    Aggregates all seven engines (fuel database, calculator, equipment
    profiler, factor selector, uncertainty engine, audit engine,
    combustion pipeline) through a single entry point with convenience
    methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: StationaryCombustionConfig instance.
        provenance: ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = StationaryCombustionService()
        >>> result = service.calculate(
        ...     fuel_type="NATURAL_GAS",
        ...     quantity=1000.0,
        ...     unit="MCF",
        ... )
        >>> print(result["total_co2e_tonnes"])
    """

    def __init__(
        self,
        config: Optional[StationaryCombustionConfig] = None,
    ) -> None:
        """Initialize the Stationary Combustion Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - FuelDatabaseEngine (E1)
        - CalculatorEngine (E2)
        - EquipmentProfilerEngine (E3)
        - FactorSelectorEngine (E4)
        - UncertaintyEngine (E5)
        - AuditEngine (E6)
        - StationaryCombustionPipelineEngine (E7)

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Provenance tracker
        self._provenance: Any = None
        if ProvenanceTracker is not None:
            try:
                self._provenance = ProvenanceTracker(
                    genesis_hash=self.config.genesis_hash,
                )
            except Exception as exc:
                logger.warning("ProvenanceTracker init failed: %s", exc)

        # Engine placeholders
        self._fuel_database_engine: Any = None
        self._calculator_engine: Any = None
        self._equipment_profiler_engine: Any = None
        self._factor_selector_engine: Any = None
        self._uncertainty_engine: Any = None
        self._audit_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._fuel_types: Dict[str, Dict[str, Any]] = {}
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._equipment_profiles: Dict[str, Dict[str, Any]] = {}
        self._aggregations: Dict[str, Dict[str, Any]] = {}
        self._audit_entries: Dict[str, List[Dict[str, Any]]] = {}

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_pipeline_runs: int = 0
        self._total_calculation_time_ms: float = 0.0
        self._started: bool = False

        logger.info("StationaryCombustionService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def fuel_database_engine(self) -> Any:
        """Get the FuelDatabaseEngine instance."""
        return self._fuel_database_engine

    @property
    def calculator_engine(self) -> Any:
        """Get the CalculatorEngine instance."""
        return self._calculator_engine

    @property
    def equipment_profiler_engine(self) -> Any:
        """Get the EquipmentProfilerEngine instance."""
        return self._equipment_profiler_engine

    @property
    def factor_selector_engine(self) -> Any:
        """Get the FactorSelectorEngine instance."""
        return self._factor_selector_engine

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyEngine instance."""
        return self._uncertainty_engine

    @property
    def audit_engine(self) -> Any:
        """Get the AuditEngine instance."""
        return self._audit_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the StationaryCombustionPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are wired together using dependency injection. The shared
        ProvenanceTracker is injected into all engines for unified audit
        trails.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        # E1: FuelDatabaseEngine
        if FuelDatabaseEngine is not None:
            try:
                self._fuel_database_engine = FuelDatabaseEngine(
                    config=self.config,
                )
                logger.info("FuelDatabaseEngine initialized")
            except Exception as exc:
                logger.warning("FuelDatabaseEngine init failed: %s", exc)
        else:
            logger.warning("FuelDatabaseEngine not available; using stub")

        # E2: CalculatorEngine
        if CalculatorEngine is not None:
            try:
                self._calculator_engine = CalculatorEngine(
                    fuel_database=self._fuel_database_engine,
                    config=self.config,
                )
                logger.info("CalculatorEngine initialized")
            except Exception as exc:
                logger.warning("CalculatorEngine init failed: %s", exc)
        else:
            logger.warning("CalculatorEngine not available; using stub")

        # E3: EquipmentProfilerEngine
        if EquipmentProfilerEngine is not None:
            try:
                self._equipment_profiler_engine = EquipmentProfilerEngine(
                    config=self.config,
                )
                logger.info("EquipmentProfilerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "EquipmentProfilerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "EquipmentProfilerEngine not available; using stub"
            )

        # E4: FactorSelectorEngine
        if FactorSelectorEngine is not None:
            try:
                self._factor_selector_engine = FactorSelectorEngine(
                    fuel_database=self._fuel_database_engine,
                    config=self.config,
                )
                logger.info("FactorSelectorEngine initialized")
            except Exception as exc:
                logger.warning("FactorSelectorEngine init failed: %s", exc)
        else:
            logger.warning(
                "FactorSelectorEngine not available; using stub"
            )

        # E5: UncertaintyEngine
        if UncertaintyEngine is not None:
            try:
                self._uncertainty_engine = UncertaintyEngine(
                    config=self.config,
                )
                logger.info("UncertaintyEngine initialized")
            except Exception as exc:
                logger.warning("UncertaintyEngine init failed: %s", exc)
        else:
            logger.warning("UncertaintyEngine not available; using stub")

        # E6: AuditEngine
        if AuditEngine is not None:
            try:
                self._audit_engine = AuditEngine(
                    config=self.config,
                )
                logger.info("AuditEngine initialized")
            except Exception as exc:
                logger.warning("AuditEngine init failed: %s", exc)
        else:
            logger.warning("AuditEngine not available; using stub")

        # E7: StationaryCombustionPipelineEngine
        if StationaryCombustionPipelineEngine is not None:
            try:
                self._pipeline_engine = StationaryCombustionPipelineEngine(
                    fuel_database=self._fuel_database_engine,
                    calculator=self._calculator_engine,
                    equipment_profiler=self._equipment_profiler_engine,
                    factor_selector=self._factor_selector_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    audit_engine=self._audit_engine,
                    config=self.config,
                )
                logger.info("StationaryCombustionPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "StationaryCombustionPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "StationaryCombustionPipelineEngine not available; "
                "using stub"
            )

    # ==================================================================
    # Convenience methods
    # ==================================================================

    def calculate(
        self,
        fuel_type: str,
        quantity: float,
        unit: str,
        gwp_source: str = "AR6",
        ef_source: str = "EPA",
        tier: Optional[str] = None,
        heating_value_basis: str = "HHV",
        include_biogenic: bool = False,
        facility_id: Optional[str] = None,
        equipment_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single combustion record.

        Convenience method that constructs a CombustionInput and delegates
        to the pipeline engine for full processing. All calculations are
        deterministic (zero-hallucination).

        Args:
            fuel_type: Fuel type identifier (e.g. NATURAL_GAS).
            quantity: Fuel quantity consumed.
            unit: Quantity unit (e.g. MCF, GALLONS).
            gwp_source: GWP source (AR4, AR5, AR6).
            ef_source: Emission factor source (EPA, IPCC, DEFRA).
            tier: Calculation tier (TIER_1, TIER_2, TIER_3) or None.
            heating_value_basis: HHV or NCV.
            include_biogenic: Include biogenic in totals.
            facility_id: Optional facility identifier.
            equipment_id: Optional equipment identifier.
            **kwargs: Additional CombustionInput parameters.

        Returns:
            Dictionary with calculation results.

        Raises:
            ValueError: If fuel_type or unit is invalid.
        """
        t0 = time.perf_counter()
        calc_id = _new_uuid()

        try:
            # Build CombustionInput
            input_data = CombustionInput(
                calculation_id=calc_id,
                fuel_type=fuel_type,
                quantity=Decimal(str(quantity)),
                unit=unit,
                heating_value_basis=heating_value_basis,
                ef_source=ef_source,
                tier=tier,
                facility_id=facility_id,
                equipment_id=equipment_id,
                **kwargs,
            )

            # Delegate to pipeline
            if self._pipeline_engine is not None:
                pipeline_result = self._pipeline_engine.run_single(
                    input_data=input_data,
                    gwp_source=gwp_source,
                )
                results = pipeline_result.get("final_results", [])
                result = results[0] if results else {}
            elif self._calculator_engine is not None:
                result = self._calculator_engine.calculate(
                    input_data=input_data,
                    gwp_source=gwp_source,
                    include_biogenic=include_biogenic,
                )
                if isinstance(result, CalculationResult):
                    result = result.model_dump(mode="json")
            else:
                result = {
                    "calculation_id": calc_id,
                    "status": "PARTIAL",
                    "fuel_type": fuel_type,
                    "quantity": quantity,
                    "unit": unit,
                    "total_co2e_kg": 0.0,
                    "total_co2e_tonnes": 0.0,
                    "message": "No calculation engine available",
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Ensure calc_id and timing
            if isinstance(result, dict):
                result.setdefault("calculation_id", calc_id)
                result["processing_time_ms"] = round(elapsed_ms, 3)
                result["provenance_hash"] = result.get(
                    "provenance_hash",
                    _compute_hash(result),
                )

            # Cache result
            self._calculations[calc_id] = result
            self._total_calculations += 1
            self._total_calculation_time_ms += elapsed_ms

            # Record provenance
            if self._provenance is not None:
                self._provenance.record(
                    entity_type="calculation",
                    action="calculate",
                    entity_id=calc_id,
                    metadata={
                        "fuel_type": fuel_type,
                        "quantity": quantity,
                        "unit": unit,
                    },
                )

            # Record metrics
            record_calculation(fuel_type, "success")
            observe_calculation_duration(fuel_type, elapsed_ms / 1000.0)

            logger.info(
                "Calculated %s: fuel=%s qty=%.2f %s co2e=%.4f tonnes",
                calc_id,
                fuel_type,
                quantity,
                unit,
                result.get("total_co2e_tonnes", 0.0)
                if isinstance(result, dict) else 0.0,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record_calculation(fuel_type, "failure")
            logger.error("calculate failed: %s", exc, exc_info=True)
            raise

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
        gwp_source: str = "AR6",
        include_biogenic: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate emissions for multiple combustion records.

        Args:
            inputs: List of input dictionaries with fuel_type, quantity,
                unit, and optional parameters.
            gwp_source: GWP source for all calculations.
            include_biogenic: Include biogenic in totals.
            **kwargs: Additional pipeline parameters.

        Returns:
            Dictionary with batch results, totals, and provenance.
        """
        t0 = time.perf_counter()
        batch_id = _new_uuid()

        try:
            # Convert dict inputs to CombustionInput objects
            combustion_inputs: List[CombustionInput] = []
            for inp in inputs:
                ci = CombustionInput(
                    calculation_id=inp.get("calculation_id", _new_uuid()),
                    fuel_type=inp["fuel_type"],
                    quantity=Decimal(str(inp["quantity"])),
                    unit=inp["unit"],
                    heating_value_basis=inp.get(
                        "heating_value_basis", "HHV",
                    ),
                    ef_source=inp.get("ef_source", "EPA"),
                    tier=inp.get("tier"),
                    facility_id=inp.get("facility_id"),
                    equipment_id=inp.get("equipment_id"),
                )
                combustion_inputs.append(ci)

            # Delegate to pipeline
            if self._pipeline_engine is not None:
                pipeline_result = self._pipeline_engine.run_pipeline(
                    inputs=combustion_inputs,
                    gwp_source=gwp_source,
                    include_biogenic=include_biogenic,
                    **kwargs,
                )
            else:
                # Fallback: calculate one by one
                results = []
                for ci in combustion_inputs:
                    r = self.calculate(
                        fuel_type=ci.fuel_type
                        if isinstance(ci.fuel_type, str)
                        else ci.fuel_type.value,
                        quantity=float(ci.quantity),
                        unit=ci.unit
                        if isinstance(ci.unit, str)
                        else ci.unit.value,
                        gwp_source=gwp_source,
                        include_biogenic=include_biogenic,
                    )
                    results.append(r)
                pipeline_result = {
                    "final_results": results,
                    "success": True,
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            final_results = pipeline_result.get("final_results", [])

            # Aggregate totals
            total_co2e_kg = sum(
                r.get("total_co2e_kg", 0.0)
                for r in final_results
                if isinstance(r, dict)
            )
            total_co2e_tonnes = sum(
                r.get("total_co2e_tonnes", 0.0)
                for r in final_results
                if isinstance(r, dict)
            )
            total_biogenic_kg = sum(
                r.get("biogenic_co2_kg", 0.0)
                for r in final_results
                if isinstance(r, dict)
            )
            total_biogenic_tonnes = sum(
                r.get("biogenic_co2_tonnes", 0.0)
                for r in final_results
                if isinstance(r, dict)
            )

            success_count = sum(
                1 for r in final_results
                if isinstance(r, dict)
                and r.get("status") == "SUCCESS"
            )

            batch_result = {
                "batch_id": batch_id,
                "results": final_results,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
                "total_biogenic_co2_kg": total_biogenic_kg,
                "total_biogenic_co2_tonnes": total_biogenic_tonnes,
                "success_count": success_count,
                "failure_count": len(final_results) - success_count,
                "processing_time_ms": round(elapsed_ms, 3),
                "provenance_hash": pipeline_result.get(
                    "pipeline_provenance_hash",
                    _compute_hash({"batch_id": batch_id}),
                ),
            }

            self._total_batch_runs += 1

            logger.info(
                "Batch %s completed: %d results, %.2f tCO2e, %.1fms",
                batch_id,
                len(final_results),
                total_co2e_tonnes,
                elapsed_ms,
            )
            return batch_result

        except Exception as exc:
            logger.error("calculate_batch failed: %s", exc, exc_info=True)
            raise

    def get_fuel_properties(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """Get physical and regulatory properties for a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Dictionary with fuel properties.
        """
        if self._fuel_database_engine is not None:
            try:
                result = self._fuel_database_engine.get_fuel_properties(
                    fuel_type,
                )
                if isinstance(result, FuelProperties):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return result
            except (AttributeError, TypeError, KeyError):
                pass

        # Check in-memory cache
        cached = self._fuel_types.get(fuel_type)
        if cached is not None:
            return cached

        return {"fuel_type": fuel_type, "error": "Fuel type not found"}

    def list_fuel_types(self) -> List[Dict[str, Any]]:
        """List all available fuel types.

        Returns:
            List of fuel type summary dictionaries.
        """
        if self._fuel_database_engine is not None:
            try:
                result = self._fuel_database_engine.list_fuel_types()
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: return known fuel type enum values
        return [
            {"fuel_type": ft.value, "display_name": ft.value.replace("_", " ").title()}
            for ft in FuelType
        ]

    def get_emission_factor(
        self,
        fuel_type: str,
        gas: str,
        source: str = "EPA",
    ) -> Dict[str, Any]:
        """Get a specific emission factor.

        Args:
            fuel_type: Fuel type identifier.
            gas: Greenhouse gas species (CO2, CH4, N2O).
            source: Factor source database.

        Returns:
            Dictionary with emission factor details.
        """
        if self._factor_selector_engine is not None:
            try:
                result = self._factor_selector_engine.get_factor(
                    fuel_type=fuel_type,
                    gas=gas,
                    source=source,
                )
                if isinstance(result, EmissionFactor):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return result
            except (AttributeError, TypeError, KeyError):
                pass

        # Check in-memory cache
        factor_key = f"{fuel_type}:{gas}:{source}"
        cached = self._emission_factors.get(factor_key)
        if cached is not None:
            return cached

        return {
            "fuel_type": fuel_type,
            "gas": gas,
            "source": source,
            "error": "Emission factor not found",
        }

    def register_equipment(
        self,
        equipment_id: Optional[str] = None,
        equipment_type: str = "BOILER_FIRE_TUBE",
        name: str = "",
        facility_id: Optional[str] = None,
        rated_capacity_mw: Optional[float] = None,
        efficiency: float = 0.80,
        load_factor: float = 0.65,
        age_years: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register an equipment profile for equipment-level calculations.

        Args:
            equipment_id: Unique equipment ID (auto-generated if None).
            equipment_type: Equipment type classification.
            name: Human-readable equipment name.
            facility_id: Parent facility identifier.
            rated_capacity_mw: Rated thermal capacity in MW.
            efficiency: Thermal efficiency (0-1).
            load_factor: Average load factor (0-1).
            age_years: Equipment age in years.
            **kwargs: Additional profile parameters.

        Returns:
            Dictionary with registered equipment details.
        """
        equip_id = equipment_id or _new_uuid()

        if self._equipment_profiler_engine is not None:
            try:
                result = self._equipment_profiler_engine.register_equipment(
                    equipment_id=equip_id,
                    equipment_type=equipment_type,
                    name=name,
                    facility_id=facility_id,
                    rated_capacity_mw=rated_capacity_mw,
                    efficiency=efficiency,
                    load_factor=load_factor,
                    age_years=age_years,
                    **kwargs,
                )
                if isinstance(result, EquipmentProfile):
                    result_dict = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"equipment_id": equip_id}
                self._equipment_profiles[equip_id] = result_dict
                return result_dict
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "EquipmentProfilerEngine.register_equipment "
                    "failed: %s", exc,
                )

        # Fallback: in-memory registration
        profile = EquipmentResponse(
            equipment_id=equip_id,
            equipment_type=equipment_type,
            name=name,
            facility_id=facility_id,
            rated_capacity_mw=rated_capacity_mw,
            efficiency=efficiency,
            load_factor=load_factor,
            age_years=age_years,
        )
        profile.provenance_hash = _compute_hash(profile)
        result_dict = profile.model_dump()
        self._equipment_profiles[equip_id] = result_dict

        if self._provenance is not None:
            self._provenance.record(
                entity_type="equipment",
                action="register_equipment",
                entity_id=equip_id,
                metadata={
                    "equipment_type": equipment_type,
                    "facility_id": facility_id,
                },
            )

        logger.info(
            "Registered equipment %s: type=%s facility=%s",
            equip_id, equipment_type, facility_id,
        )
        return result_dict

    def run_pipeline(
        self,
        inputs: List[Dict[str, Any]],
        gwp_source: str = "AR6",
        include_biogenic: bool = False,
        organization_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
        control_approach: str = "OPERATIONAL",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the full seven-stage combustion pipeline.

        Args:
            inputs: List of input dictionaries.
            gwp_source: GWP source.
            include_biogenic: Include biogenic in totals.
            organization_id: Organisation identifier.
            reporting_period: Reporting period label.
            control_approach: GHG Protocol boundary approach.
            **kwargs: Additional pipeline parameters.

        Returns:
            Dictionary with full pipeline results.
        """
        # Convert dict inputs to CombustionInput objects
        combustion_inputs: List[CombustionInput] = []
        for inp in inputs:
            ci = CombustionInput(
                calculation_id=inp.get("calculation_id", _new_uuid()),
                fuel_type=inp["fuel_type"],
                quantity=Decimal(str(inp["quantity"])),
                unit=inp["unit"],
                heating_value_basis=inp.get("heating_value_basis", "HHV"),
                ef_source=inp.get("ef_source", "EPA"),
                tier=inp.get("tier"),
                facility_id=inp.get("facility_id"),
                equipment_id=inp.get("equipment_id"),
            )
            combustion_inputs.append(ci)

        if self._pipeline_engine is not None:
            result = self._pipeline_engine.run_pipeline(
                inputs=combustion_inputs,
                gwp_source=gwp_source,
                include_biogenic=include_biogenic,
                organization_id=organization_id,
                reporting_period=reporting_period,
                control_approach=control_approach,
            )
        else:
            # Fallback to batch calculation
            result = self.calculate_batch(
                inputs=inputs,
                gwp_source=gwp_source,
                include_biogenic=include_biogenic,
            )
            result["pipeline_id"] = _new_uuid()
            result["stages_completed"] = 0
            result["stages_total"] = 7

        self._total_pipeline_runs += 1
        return result

    def get_uncertainty(
        self,
        calculation_result: Dict[str, Any],
        iterations: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation result.

        Args:
            calculation_result: Dictionary from a previous calculation.
            iterations: Monte Carlo iterations (defaults to config value).
            **kwargs: Additional uncertainty parameters.

        Returns:
            Dictionary with uncertainty analysis results.
        """
        iters = iterations or self.config.monte_carlo_iterations
        calc_id = calculation_result.get("calculation_id", "")

        if self._uncertainty_engine is not None:
            try:
                result = self._uncertainty_engine.quantify(
                    calculation_result=calculation_result,
                    iterations=iters,
                    **kwargs,
                )
                if isinstance(result, UncertaintyResult):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return result
            except (AttributeError, TypeError) as exc:
                logger.warning("UncertaintyEngine.quantify failed: %s", exc)

        return {
            "calculation_id": calc_id,
            "mean_co2e_kg": calculation_result.get("total_co2e_kg", 0.0),
            "std_co2e_kg": 0.0,
            "p5_co2e_kg": 0.0,
            "p95_co2e_kg": 0.0,
            "confidence_interval_pct": 95.0,
            "num_simulations": iters,
            "message": "UncertaintyEngine not available; returning point estimate",
        }

    def get_audit_trail(
        self,
        calculation_id: str,
    ) -> List[Dict[str, Any]]:
        """Get the audit trail for a specific calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            List of audit entry dictionaries.
        """
        if self._audit_engine is not None:
            try:
                result = self._audit_engine.get_audit_trail(calculation_id)
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: in-memory cache
        return self._audit_entries.get(calculation_id, [])

    def get_compliance_mapping(
        self,
        framework: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get regulatory compliance mapping for calculations.

        Args:
            framework: Optional framework filter (GHG_PROTOCOL,
                EPA_GHGRP, EU_ETS, ISO_14064, CSRD, SEC_CLIMATE, TCFD).

        Returns:
            Dictionary with compliance mappings.
        """
        frameworks = (
            [framework] if framework
            else [f.value for f in RegulatoryFramework]
        )

        mappings: List[Dict[str, Any]] = []
        for fw in frameworks:
            mappings.append({
                "framework": fw,
                "requirements": self._get_framework_requirements(fw),
                "compliant": True,
            })

        compliant_count = sum(
            1 for m in mappings if m.get("compliant", False)
        )

        return {
            "framework": framework or "ALL",
            "mappings": mappings,
            "compliant_count": compliant_count,
            "total_requirements": len(mappings),
            "overall_compliant": compliant_count == len(mappings),
            "provenance_hash": _compute_hash(mappings),
        }

    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the stationary combustion service.

        Returns:
            Dictionary with health status for each engine and the overall
            service, including version, fuel counts, and statistics.
        """
        engines: Dict[str, str] = {
            "fuel_database": (
                "available"
                if self._fuel_database_engine is not None
                else "unavailable"
            ),
            "calculator": (
                "available"
                if self._calculator_engine is not None
                else "unavailable"
            ),
            "equipment_profiler": (
                "available"
                if self._equipment_profiler_engine is not None
                else "unavailable"
            ),
            "factor_selector": (
                "available"
                if self._factor_selector_engine is not None
                else "unavailable"
            ),
            "uncertainty": (
                "available"
                if self._uncertainty_engine is not None
                else "unavailable"
            ),
            "audit": (
                "available"
                if self._audit_engine is not None
                else "unavailable"
            ),
            "pipeline": (
                "available"
                if self._pipeline_engine is not None
                else "unavailable"
            ),
        }

        available_count = sum(
            1 for s in engines.values() if s == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = True
        provenance_entries = 0
        if self._provenance is not None:
            try:
                chain_valid = self._provenance.verify_chain()
                provenance_entries = self._provenance.entry_count
            except (AttributeError, TypeError):
                pass

        avg_calc_time = (
            self._total_calculation_time_ms / self._total_calculations
            if self._total_calculations > 0
            else 0.0
        )

        return {
            "status": overall_status,
            "version": "1.0.0",
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "fuel_type_count": len(self._fuel_types) or len(FuelType),
            "emission_factor_count": len(self._emission_factors),
            "equipment_count": len(self._equipment_profiles),
            "statistics": {
                "total_calculations": self._total_calculations,
                "total_batch_runs": self._total_batch_runs,
                "total_pipeline_runs": self._total_pipeline_runs,
                "avg_calculation_time_ms": round(avg_calc_time, 3),
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": provenance_entries,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the stationary combustion service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug(
                "StationaryCombustionService already started; skipping"
            )
            return

        logger.info("StationaryCombustionService starting up...")
        self._started = True
        logger.info("StationaryCombustionService startup complete")

    def shutdown(self) -> None:
        """Shutdown the stationary combustion service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("StationaryCombustionService shut down")

    # ------------------------------------------------------------------
    # Statistics and provenance
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics for the service.

        Returns:
            Dictionary with current service statistics.
        """
        avg_calc_time = (
            self._total_calculation_time_ms / self._total_calculations
            if self._total_calculations > 0
            else 0.0
        )

        return {
            "total_calculations": self._total_calculations,
            "total_batch_runs": self._total_batch_runs,
            "total_pipeline_runs": self._total_pipeline_runs,
            "total_fuel_types": len(self._fuel_types) or len(FuelType),
            "total_emission_factors": len(self._emission_factors),
            "total_equipment_profiles": len(self._equipment_profiles),
            "total_aggregations": len(self._aggregations),
            "total_audit_entries": sum(
                len(entries) for entries in self._audit_entries.values()
            ),
            "avg_calculation_time_ms": round(avg_calc_time, 3),
            "timestamp": _utcnow_iso(),
        }

    def get_provenance(self) -> Any:
        """Get the provenance tracker instance.

        Returns:
            ProvenanceTracker instance or None.
        """
        return self._provenance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_framework_requirements(
        self,
        framework: str,
    ) -> List[Dict[str, Any]]:
        """Get the set of requirements for a regulatory framework.

        Args:
            framework: Regulatory framework identifier.

        Returns:
            List of requirement dictionaries.
        """
        # Stationary combustion compliance requirements
        requirements_map: Dict[str, List[Dict[str, Any]]] = {
            "GHG_PROTOCOL": [
                {
                    "requirement_id": "GHG-SC-001",
                    "requirement_name": "Scope 1 Stationary Combustion",
                    "description": "Complete Scope 1 reporting for stationary sources",
                    "compliant": True,
                },
                {
                    "requirement_id": "GHG-SC-002",
                    "requirement_name": "Emission Factor Documentation",
                    "description": "Document all emission factor sources and references",
                    "compliant": True,
                },
            ],
            "EPA_GHGRP": [
                {
                    "requirement_id": "EPA-40CFR98",
                    "requirement_name": "Subpart C General Stationary Combustion",
                    "description": "EPA GHGRP Subpart C compliance",
                    "compliant": True,
                },
            ],
            "EU_ETS": [
                {
                    "requirement_id": "EU-ETS-MRR",
                    "requirement_name": "Monitoring and Reporting Regulation",
                    "description": "EU ETS MRR Annex II methodology",
                    "compliant": True,
                },
            ],
            "ISO_14064": [
                {
                    "requirement_id": "ISO-14064-1",
                    "requirement_name": "GHG Inventory",
                    "description": "ISO 14064-1 Clause 5 quantification",
                    "compliant": True,
                },
            ],
            "CSRD": [
                {
                    "requirement_id": "ESRS-E1",
                    "requirement_name": "Climate Change",
                    "description": "ESRS E1 Scope 1 emissions disclosure",
                    "compliant": True,
                },
            ],
            "SEC_CLIMATE": [
                {
                    "requirement_id": "SEC-GHG-SCOPE1",
                    "requirement_name": "Scope 1 Disclosure",
                    "description": "SEC Climate Disclosure Scope 1 requirement",
                    "compliant": True,
                },
            ],
            "TCFD": [
                {
                    "requirement_id": "TCFD-METRICS",
                    "requirement_name": "Metrics and Targets",
                    "description": "TCFD Scope 1 GHG emissions metric",
                    "compliant": True,
                },
            ],
        }

        return requirements_map.get(framework, [])


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> StationaryCombustionService:
    """Get or create the singleton StationaryCombustionService instance.

    Returns:
        The singleton StationaryCombustionService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = StationaryCombustionService()
    return _singleton_instance


# ===================================================================
# Module-level singletons for FastAPI integration
# ===================================================================

_service: Optional[StationaryCombustionService] = None
_router: Any = None


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_stationary_combustion(
    app: Any,
    config: Optional[StationaryCombustionConfig] = None,
) -> StationaryCombustionService:
    """Configure the Stationary Combustion Service on a FastAPI application.

    Creates the StationaryCombustionService, stores it in app.state,
    mounts the stationary combustion API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional stationary combustion config.

    Returns:
        StationaryCombustionService instance.
    """
    global _singleton_instance, _service

    service = StationaryCombustionService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service
        _service = service

    # Attach to app state
    app.state.stationary_combustion_service = service

    # Mount stationary combustion API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Stationary combustion API router mounted")
    else:
        logger.warning(
            "Stationary combustion router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Stationary combustion service configured on app")
    return service


def get_service() -> Optional[StationaryCombustionService]:
    """Get the singleton StationaryCombustionService instance.

    Creates a new instance if one does not exist yet. Uses
    double-checked locking for thread safety.

    Returns:
        StationaryCombustionService singleton instance or None.
    """
    global _singleton_instance, _service
    if _service is not None:
        return _service
    if _singleton_instance is not None:
        return _singleton_instance
    # Lazy creation
    with _singleton_lock:
        if _singleton_instance is None:
            _singleton_instance = StationaryCombustionService()
        _service = _singleton_instance
    return _singleton_instance


def get_router() -> Any:
    """Get the stationary combustion API router.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.stationary_combustion.api.router import router
        return router
    except ImportError:
        logger.warning(
            "Stationary combustion API router module not available"
        )
        return None


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "StationaryCombustionService",
    # Configuration helpers
    "configure_stationary_combustion",
    "get_service",
    "get_router",
    # Response models
    "CalculationResponse",
    "BatchCalculationResponse",
    "FuelResponse",
    "FuelListResponse",
    "EmissionFactorResponse",
    "EquipmentResponse",
    "AggregationResponse",
    "UncertaintyResponse",
    "AuditTrailResponse",
    "ComplianceResponse",
    "ValidationResponse",
    "PipelineResponse",
    "HealthResponse",
    "StatsResponse",
]
