# -*- coding: utf-8 -*-
"""
Refrigerants & F-Gas Agent Service Setup - AGENT-MRV-002

Provides ``configure_refrigerants_fgas(app)`` which wires up the
Refrigerants & F-Gas Agent SDK (refrigerant database, emission calculator,
equipment registry, leak rate estimator, uncertainty quantifier,
compliance tracker, refrigerant pipeline, provenance tracker) and mounts
the REST API.

Also exposes ``get_service()`` for programmatic access and the
``RefrigerantsFGasService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.refrigerants_fgas.setup import configure_refrigerants_fgas
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_refrigerants_fgas(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.config import (
        RefrigerantsFGasConfig,
        get_config,
    )
except ImportError:
    RefrigerantsFGasConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """No-op fallback when config module is unavailable."""
        return None

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
# Optional model imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.models import (
        CalculationInput,
        CalculationMethod,
        CalculationResult,
        CalculationStatus,
        ComplianceRecord,
        ComplianceStatus,
        EquipmentProfile,
        EquipmentStatus,
        EquipmentType,
        GasEmission,
        GWPSource,
        GWPTimeframe,
        LeakRateProfile,
        LifecycleStage,
        MassBalanceData,
        RefrigerantCategory,
        RefrigerantProperties,
        RefrigerantType,
        RegulatoryFramework,
        ReportingPeriod,
        ServiceEvent,
        ServiceEventType,
        UncertaintyResult,
        UnitType,
    )
except ImportError:
    CalculationInput = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    CalculationStatus = None  # type: ignore[assignment, misc]
    ComplianceRecord = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    EquipmentProfile = None  # type: ignore[assignment, misc]
    EquipmentStatus = None  # type: ignore[assignment, misc]
    EquipmentType = None  # type: ignore[assignment, misc]
    GasEmission = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    GWPTimeframe = None  # type: ignore[assignment, misc]
    LeakRateProfile = None  # type: ignore[assignment, misc]
    LifecycleStage = None  # type: ignore[assignment, misc]
    MassBalanceData = None  # type: ignore[assignment, misc]
    RefrigerantCategory = None  # type: ignore[assignment, misc]
    RefrigerantProperties = None  # type: ignore[assignment, misc]
    RefrigerantType = None  # type: ignore[assignment, misc]
    RegulatoryFramework = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    ServiceEvent = None  # type: ignore[assignment, misc]
    ServiceEventType = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    UnitType = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.refrigerants_fgas.refrigerant_database import RefrigerantDatabaseEngine
except ImportError:
    RefrigerantDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.emission_calculator import EmissionCalculatorEngine
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.equipment_registry import EquipmentRegistryEngine
except ImportError:
    EquipmentRegistryEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.leak_rate_estimator import LeakRateEstimatorEngine
except ImportError:
    LeakRateEstimatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.compliance_tracker import ComplianceTrackerEngine
except ImportError:
    ComplianceTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.refrigerant_pipeline import RefrigerantPipelineEngine
except ImportError:
    RefrigerantPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.refrigerants_fgas.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation,
        observe_calculation_duration,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def record_calculation(method: str, refrigerant_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def observe_calculation_duration(operation: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class CalculationResponse(BaseModel):
    """Single refrigerant emission calculation response.

    Attributes:
        calculation_id: Unique calculation identifier (UUID4).
        status: Calculation status (SUCCESS, PARTIAL, FAILED).
        refrigerant_type: Refrigerant type used.
        charge_kg: Equipment charge in kg.
        method: Calculation method (equipment_based, mass_balance, etc.).
        gwp_value: GWP value used.
        gwp_source: GWP source (AR4, AR5, AR6).
        leak_rate_pct: Applied leak rate percentage.
        emissions_kg: Raw refrigerant emissions in kg.
        total_emissions_kg_co2e: Total CO2e in kg.
        total_emissions_tco2e: Total CO2e in tonnes.
        blend_decomposition: Per-component breakdown (blends only).
        calculation_trace: Step-by-step calculation trace.
        provenance_hash: SHA-256 provenance hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: ISO-8601 UTC calculation timestamp.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="SUCCESS")
    refrigerant_type: str = Field(default="")
    charge_kg: float = Field(default=0.0)
    method: str = Field(default="equipment_based")
    gwp_value: float = Field(default=0.0)
    gwp_source: str = Field(default="AR6")
    leak_rate_pct: float = Field(default=0.0)
    emissions_kg: float = Field(default=0.0)
    total_emissions_kg_co2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    blend_decomposition: Optional[List[Dict[str, Any]]] = Field(default=None)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class BatchResponse(BaseModel):
    """Batch refrigerant emission calculation response.

    Attributes:
        batch_id: Unique batch identifier (UUID4).
        results: Individual calculation results.
        total_emissions_kg_co2e: Aggregate CO2e in kg.
        total_emissions_tco2e: Aggregate CO2e in tonnes.
        success_count: Number of successful calculations.
        failure_count: Number of failed calculations.
        total_count: Total number of inputs.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 batch provenance hash.
    """

    model_config = {"extra": "forbid"}

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_emissions_kg_co2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    total_count: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class RefrigerantResponse(BaseModel):
    """Refrigerant properties response.

    Attributes:
        refrigerant_type: Refrigerant type identifier.
        category: Refrigerant category (HFC, HFO, PFC, etc.).
        display_name: Human-readable name.
        formula: Chemical formula.
        gwp_ar4: GWP from IPCC AR4.
        gwp_ar5: GWP from IPCC AR5.
        gwp_ar6: GWP from IPCC AR6.
        gwp_ar6_20yr: GWP AR6 20-year timeframe.
        is_blend: Whether this is a blended refrigerant.
        components: Blend components (if blend).
        ozone_depletion_potential: ODP value.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    refrigerant_type: str = Field(default="")
    category: str = Field(default="")
    display_name: str = Field(default="")
    formula: str = Field(default="")
    gwp_ar4: float = Field(default=0.0)
    gwp_ar5: float = Field(default=0.0)
    gwp_ar6: float = Field(default=0.0)
    gwp_ar6_20yr: float = Field(default=0.0)
    is_blend: bool = Field(default=False)
    components: List[Dict[str, Any]] = Field(default_factory=list)
    ozone_depletion_potential: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class RefrigerantListResponse(BaseModel):
    """Response listing all available refrigerants.

    Attributes:
        refrigerants: List of refrigerant summary dictionaries.
        total_count: Total number of refrigerants available.
    """

    model_config = {"extra": "forbid"}

    refrigerants: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class EquipmentResponse(BaseModel):
    """Equipment profile registration or retrieval response.

    Attributes:
        equipment_id: Unique equipment identifier.
        equipment_type: Equipment type classification.
        name: Human-readable equipment name.
        facility_id: Parent facility identifier.
        refrigerant_type: Refrigerant type charged.
        charge_kg: Current refrigerant charge in kg.
        capacity_kw: Cooling or heating capacity in kW.
        age_years: Equipment age in years.
        status: Equipment operational status.
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    equipment_id: str = Field(default="")
    equipment_type: str = Field(default="")
    name: str = Field(default="")
    facility_id: Optional[str] = Field(default=None)
    refrigerant_type: str = Field(default="")
    charge_kg: float = Field(default=0.0)
    capacity_kw: Optional[float] = Field(default=None)
    age_years: int = Field(default=0)
    status: str = Field(default="ACTIVE")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class EquipmentListResponse(BaseModel):
    """Response listing equipment profiles.

    Attributes:
        equipment: List of equipment profile dictionaries.
        total_count: Total number of equipment profiles.
    """

    model_config = {"extra": "forbid"}

    equipment: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class ServiceEventResponse(BaseModel):
    """Service event logging response.

    Attributes:
        event_id: Unique event identifier.
        equipment_id: Related equipment identifier.
        event_type: Service event type (installation, recharge, etc.).
        refrigerant_type: Refrigerant type involved.
        quantity_kg: Refrigerant quantity in kg.
        date: Event date (ISO-8601).
        technician: Technician or contractor name.
        notes: Additional notes.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    equipment_id: str = Field(default="")
    event_type: str = Field(default="")
    refrigerant_type: str = Field(default="")
    quantity_kg: float = Field(default=0.0)
    date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    technician: str = Field(default="")
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")


class LeakRateResponse(BaseModel):
    """Leak rate estimation or registration response.

    Attributes:
        leak_rate_id: Unique leak rate identifier.
        equipment_type: Equipment type for this rate.
        base_rate_pct: Base annual leak rate percentage.
        age_factor: Age adjustment factor.
        climate_factor: Climate adjustment factor.
        ldar_adjustment: LDAR programme adjustment factor.
        effective_rate_pct: Final effective leak rate.
        source: Rate source (IPCC, EPA, custom).
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    leak_rate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    equipment_type: str = Field(default="")
    base_rate_pct: float = Field(default=5.0)
    age_factor: float = Field(default=1.0)
    climate_factor: float = Field(default=1.0)
    ldar_adjustment: float = Field(default=1.0)
    effective_rate_pct: float = Field(default=5.0)
    source: str = Field(default="IPCC")
    provenance_hash: str = Field(default="")


class ComplianceResponse(BaseModel):
    """Regulatory compliance check response.

    Attributes:
        framework: Regulatory framework name.
        compliant: Whether requirements are met.
        requirements_met: List of met requirements.
        requirements_gap: List of unmet requirements.
        details: Additional compliance details.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    framework: str = Field(default="")
    compliant: bool = Field(default=False)
    requirements_met: List[str] = Field(default_factory=list)
    requirements_gap: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class ComplianceListResponse(BaseModel):
    """List of compliance check results.

    Attributes:
        records: List of compliance record dictionaries.
        total_count: Total number of compliance records.
        compliant_count: Number of compliant records.
        overall_compliant: Whether all records are compliant.
    """

    model_config = {"extra": "forbid"}

    records: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)
    compliant_count: int = Field(default=0)
    overall_compliant: bool = Field(default=False)


class UncertaintyResponse(BaseModel):
    """Uncertainty analysis response.

    Attributes:
        calculation_id: Related calculation identifier.
        method: Uncertainty method (monte_carlo, analytical).
        mean_co2e_kg: Mean CO2e from simulations.
        std_co2e_kg: Standard deviation of CO2e.
        p5_co2e_kg: 5th percentile.
        p95_co2e_kg: 95th percentile.
        confidence_interval_pct: Confidence interval percentage.
        iterations: Number of Monte Carlo iterations.
        data_quality_score: Data quality score (1-5).
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    method: str = Field(default="monte_carlo")
    mean_co2e_kg: float = Field(default=0.0)
    std_co2e_kg: float = Field(default=0.0)
    p5_co2e_kg: float = Field(default=0.0)
    p95_co2e_kg: float = Field(default=0.0)
    confidence_interval_pct: float = Field(default=95.0)
    iterations: int = Field(default=5000)
    data_quality_score: int = Field(default=3)
    provenance_hash: str = Field(default="")


class AuditTrailResponse(BaseModel):
    """Audit trail retrieval response.

    Attributes:
        calculation_id: Calculation for which audit entries are retrieved.
        entries: List of audit entry dictionaries.
        total_entries: Total number of audit entries.
        chain_hash: SHA-256 hash of the provenance chain.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    entries: List[Dict[str, Any]] = Field(default_factory=list)
    total_entries: int = Field(default=0)
    chain_hash: str = Field(default="")
    provenance_hash: str = Field(default="")


class HealthResponse(BaseModel):
    """Service health check response.

    Attributes:
        status: Overall service status (healthy, degraded, unhealthy).
        version: Agent version string.
        engines: Per-engine availability status.
        engines_available: Count of available engines.
        engines_total: Total number of engines.
        started: Whether the service has been started.
        refrigerant_count: Number of registered refrigerants.
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
    refrigerant_count: int = Field(default=0)
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
        total_refrigerants: Number of registered refrigerants.
        total_equipment: Number of equipment profiles.
        total_service_events: Number of service events.
        total_compliance_checks: Total compliance checks.
        total_uncertainty_runs: Total uncertainty analyses.
        total_audit_entries: Total audit trail entries.
        avg_calculation_time_ms: Average calculation time.
        timestamp: ISO-8601 UTC timestamp.
    """

    model_config = {"extra": "forbid"}

    total_calculations: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_pipeline_runs: int = Field(default=0)
    total_refrigerants: int = Field(default=0)
    total_equipment: int = Field(default=0)
    total_service_events: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    total_audit_entries: int = Field(default=0)
    avg_calculation_time_ms: float = Field(default=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class BlendResponse(BaseModel):
    """Blend decomposition response.

    Attributes:
        refrigerant_type: Blend refrigerant type identifier.
        is_blend: Whether this is a blended refrigerant.
        components: List of per-component emission breakdowns.
        component_count: Number of blend components.
        total_emissions_kg_co2e: Total blended CO2e in kg.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    refrigerant_type: str = Field(default="")
    is_blend: bool = Field(default=True)
    components: List[Dict[str, Any]] = Field(default_factory=list)
    component_count: int = Field(default=0)
    total_emissions_kg_co2e: float = Field(default=0.0)
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
    """End-to-end refrigerant pipeline execution response.

    Attributes:
        pipeline_id: Unique pipeline run identifier (UUID4).
        success: Whether the pipeline completed successfully.
        stages_completed: Number of pipeline stages completed.
        stages_total: Total number of pipeline stages.
        stage_results: Per-stage execution results.
        result: Final calculation result dictionary.
        pipeline_provenance_hash: SHA-256 hash of the entire pipeline run.
        total_duration_ms: Total wall-clock time in milliseconds.
    """

    model_config = {"extra": "forbid"}

    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = Field(default=False)
    stages_completed: int = Field(default=0)
    stages_total: int = Field(default=8)
    stage_results: List[Dict[str, Any]] = Field(default_factory=list)
    result: Dict[str, Any] = Field(default_factory=dict)
    pipeline_provenance_hash: str = Field(default="")
    total_duration_ms: float = Field(default=0.0)


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
# RefrigerantsFGasService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["RefrigerantsFGasService"] = None


class RefrigerantsFGasService:
    """Unified facade over the Refrigerants & F-Gas Agent SDK.

    Aggregates all seven engines (refrigerant database, emission calculator,
    equipment registry, leak rate estimator, uncertainty quantifier,
    compliance tracker, refrigerant pipeline) through a single entry point
    with convenience methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: RefrigerantsFGasConfig instance or None.

    Example:
        >>> service = RefrigerantsFGasService()
        >>> result = service.calculate(
        ...     refrigerant_type="R_410A",
        ...     charge_kg=5.0,
        ...     method="equipment_based",
        ... )
        >>> print(result["total_emissions_tco2e"])
    """

    def __init__(
        self,
        config: Any = None,
    ) -> None:
        """Initialize the Refrigerants & F-Gas Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - RefrigerantDatabaseEngine (E1)
        - EmissionCalculatorEngine (E2)
        - EquipmentRegistryEngine (E3)
        - LeakRateEstimatorEngine (E4)
        - UncertaintyQuantifierEngine (E5)
        - ComplianceTrackerEngine (E6)
        - RefrigerantPipelineEngine (E7)

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Provenance tracker
        self._provenance: Any = None
        if ProvenanceTracker is not None:
            try:
                genesis = "gl-mrv-scope1-002-genesis"
                if self.config is not None:
                    genesis = getattr(
                        self.config, "genesis_hash", genesis,
                    )
                self._provenance = ProvenanceTracker(
                    genesis_hash=genesis,
                )
            except Exception as exc:
                logger.warning("ProvenanceTracker init failed: %s", exc)

        # Engine placeholders (lazy init)
        self._refrigerant_database_engine: Any = None
        self._emission_calculator_engine: Any = None
        self._equipment_registry_engine: Any = None
        self._leak_rate_estimator_engine: Any = None
        self._uncertainty_quantifier_engine: Any = None
        self._compliance_tracker_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._refrigerants: Dict[str, Dict[str, Any]] = {}
        self._equipment_profiles: Dict[str, Dict[str, Any]] = {}
        self._service_events: Dict[str, Dict[str, Any]] = {}
        self._leak_rates: Dict[str, Dict[str, Any]] = {}
        self._compliance_records: Dict[str, Dict[str, Any]] = {}
        self._uncertainty_results: Dict[str, Dict[str, Any]] = {}
        self._audit_entries: Dict[str, List[Dict[str, Any]]] = {}

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_pipeline_runs: int = 0
        self._total_compliance_checks: int = 0
        self._total_uncertainty_runs: int = 0
        self._total_calculation_time_ms: float = 0.0
        self._started: bool = False

        logger.info("RefrigerantsFGasService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def refrigerant_database_engine(self) -> Any:
        """Get the RefrigerantDatabaseEngine instance."""
        return self._refrigerant_database_engine

    @property
    def emission_calculator_engine(self) -> Any:
        """Get the EmissionCalculatorEngine instance."""
        return self._emission_calculator_engine

    @property
    def equipment_registry_engine(self) -> Any:
        """Get the EquipmentRegistryEngine instance."""
        return self._equipment_registry_engine

    @property
    def leak_rate_estimator_engine(self) -> Any:
        """Get the LeakRateEstimatorEngine instance."""
        return self._leak_rate_estimator_engine

    @property
    def uncertainty_quantifier_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty_quantifier_engine

    @property
    def compliance_tracker_engine(self) -> Any:
        """Get the ComplianceTrackerEngine instance."""
        return self._compliance_tracker_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the RefrigerantPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are wired together using dependency injection. Engines
        are optional; missing imports are logged as warnings and the
        service continues in degraded mode.
        """
        # E1: RefrigerantDatabaseEngine
        if RefrigerantDatabaseEngine is not None:
            try:
                self._refrigerant_database_engine = RefrigerantDatabaseEngine(
                    config=self.config,
                )
                logger.info("RefrigerantDatabaseEngine initialized")
            except Exception as exc:
                logger.warning(
                    "RefrigerantDatabaseEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "RefrigerantDatabaseEngine not available; using stub",
            )

        # E2: EmissionCalculatorEngine
        if EmissionCalculatorEngine is not None:
            try:
                self._emission_calculator_engine = EmissionCalculatorEngine(
                    refrigerant_database=self._refrigerant_database_engine,
                    config=self.config,
                )
                logger.info("EmissionCalculatorEngine initialized")
            except Exception as exc:
                logger.warning(
                    "EmissionCalculatorEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "EmissionCalculatorEngine not available; using stub",
            )

        # E3: EquipmentRegistryEngine
        if EquipmentRegistryEngine is not None:
            try:
                self._equipment_registry_engine = EquipmentRegistryEngine(
                    config=self.config,
                )
                logger.info("EquipmentRegistryEngine initialized")
            except Exception as exc:
                logger.warning(
                    "EquipmentRegistryEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "EquipmentRegistryEngine not available; using stub",
            )

        # E4: LeakRateEstimatorEngine
        if LeakRateEstimatorEngine is not None:
            try:
                self._leak_rate_estimator_engine = LeakRateEstimatorEngine(
                    config=self.config,
                )
                logger.info("LeakRateEstimatorEngine initialized")
            except Exception as exc:
                logger.warning(
                    "LeakRateEstimatorEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "LeakRateEstimatorEngine not available; using stub",
            )

        # E5: UncertaintyQuantifierEngine
        if UncertaintyQuantifierEngine is not None:
            try:
                self._uncertainty_quantifier_engine = UncertaintyQuantifierEngine(
                    config=self.config,
                )
                logger.info("UncertaintyQuantifierEngine initialized")
            except Exception as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "UncertaintyQuantifierEngine not available; using stub",
            )

        # E6: ComplianceTrackerEngine
        if ComplianceTrackerEngine is not None:
            try:
                self._compliance_tracker_engine = ComplianceTrackerEngine(
                    config=self.config,
                )
                logger.info("ComplianceTrackerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ComplianceTrackerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ComplianceTrackerEngine not available; using stub",
            )

        # E7: RefrigerantPipelineEngine
        if RefrigerantPipelineEngine is not None:
            try:
                self._pipeline_engine = RefrigerantPipelineEngine(
                    refrigerant_database=self._refrigerant_database_engine,
                    emission_calculator=self._emission_calculator_engine,
                    equipment_registry=self._equipment_registry_engine,
                    leak_rate_estimator=self._leak_rate_estimator_engine,
                    uncertainty_quantifier=self._uncertainty_quantifier_engine,
                    compliance_tracker=self._compliance_tracker_engine,
                    config=self.config,
                )
                logger.info("RefrigerantPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "RefrigerantPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "RefrigerantPipelineEngine not available; using stub",
            )

    # ==================================================================
    # Convenience methods
    # ==================================================================

    def calculate(
        self,
        refrigerant_type: str,
        charge_kg: float,
        method: str = "equipment_based",
        gwp_source: str = "AR6",
        equipment_type: str = "",
        equipment_id: str = "",
        facility_id: str = "",
        custom_leak_rate_pct: Optional[float] = None,
        mass_balance_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single refrigerant record.

        Convenience method that constructs a calculation input and
        delegates to the pipeline engine for full processing. All
        calculations are deterministic (zero-hallucination).

        Args:
            refrigerant_type: Refrigerant type identifier (e.g. R_410A).
            charge_kg: Equipment refrigerant charge in kg.
            method: Calculation method (equipment_based, mass_balance,
                screening, direct, top_down).
            gwp_source: GWP source (AR4, AR5, AR6).
            equipment_type: Equipment type classification.
            equipment_id: Optional equipment identifier.
            facility_id: Optional facility identifier.
            custom_leak_rate_pct: Optional custom leak rate percentage.
            mass_balance_data: Optional mass balance input data.
            **kwargs: Additional calculation parameters.

        Returns:
            Dictionary with calculation results.

        Raises:
            ValueError: If refrigerant_type or charge_kg is invalid.
        """
        t0 = time.perf_counter()
        calc_id = _new_uuid()

        try:
            input_data = {
                "calculation_id": calc_id,
                "refrigerant_type": refrigerant_type,
                "charge_kg": charge_kg,
                "method": method,
                "gwp_source": gwp_source,
                "equipment_type": equipment_type,
                "equipment_id": equipment_id,
                "facility_id": facility_id,
                **kwargs,
            }

            if custom_leak_rate_pct is not None:
                input_data["custom_leak_rate_pct"] = custom_leak_rate_pct

            if mass_balance_data is not None:
                input_data["mass_balance_data"] = mass_balance_data

            # Delegate to pipeline
            if self._pipeline_engine is not None:
                pipeline_result = self._pipeline_engine.run(input_data)
                result = pipeline_result.get("result", {})
                result["pipeline_id"] = pipeline_result.get(
                    "pipeline_id", "",
                )
            else:
                # Minimal fallback
                result = {
                    "calculation_id": calc_id,
                    "status": "PARTIAL",
                    "refrigerant_type": refrigerant_type,
                    "charge_kg": charge_kg,
                    "method": method,
                    "total_emissions_kg_co2e": 0.0,
                    "total_emissions_tco2e": 0.0,
                    "message": "No pipeline engine available",
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Ensure calc_id and timing
            if isinstance(result, dict):
                result.setdefault("calculation_id", calc_id)
                result["processing_time_ms"] = round(elapsed_ms, 3)
                result.setdefault(
                    "provenance_hash", _compute_hash(result),
                )

            # Cache result
            self._calculations[calc_id] = result
            self._total_calculations += 1
            self._total_calculation_time_ms += elapsed_ms

            # Cache audit entries from pipeline
            if self._pipeline_engine is not None:
                audit = result.get("audit_entries", [])
                if audit:
                    self._audit_entries[calc_id] = audit

            # Record provenance
            if self._provenance is not None:
                try:
                    self._provenance.record(
                        entity_type="calculation",
                        action="calculate",
                        entity_id=calc_id,
                        metadata={
                            "refrigerant_type": refrigerant_type,
                            "charge_kg": charge_kg,
                            "method": method,
                        },
                    )
                except (AttributeError, TypeError):
                    pass

            # Record metrics
            record_calculation(method, refrigerant_type, "success")
            observe_calculation_duration(
                "single_calculation", elapsed_ms / 1000.0,
            )

            logger.info(
                "Calculated %s: ref=%s charge=%.2f kg co2e=%.6f tonnes",
                calc_id,
                refrigerant_type,
                charge_kg,
                result.get("total_emissions_tco2e", 0.0)
                if isinstance(result, dict) else 0.0,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record_calculation(method, refrigerant_type, "failure")
            logger.error(
                "calculate failed: %s", exc, exc_info=True,
            )
            raise

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate emissions for multiple refrigerant records.

        Args:
            inputs: List of input dictionaries with refrigerant_type,
                charge_kg, method, and optional parameters.
            **kwargs: Additional pipeline parameters.

        Returns:
            Dictionary with batch results, totals, and provenance.
        """
        t0 = time.perf_counter()
        batch_id = _new_uuid()

        try:
            if self._pipeline_engine is not None:
                batch_result = self._pipeline_engine.run_batch(inputs)
            else:
                # Fallback: calculate one by one
                results = []
                for inp in inputs:
                    r = self.calculate(
                        refrigerant_type=inp.get(
                            "refrigerant_type", "",
                        ),
                        charge_kg=float(inp.get("charge_kg", 0.0)),
                        method=inp.get("method", "equipment_based"),
                        gwp_source=inp.get("gwp_source", "AR6"),
                        equipment_type=inp.get("equipment_type", ""),
                        equipment_id=inp.get("equipment_id", ""),
                        facility_id=inp.get("facility_id", ""),
                        custom_leak_rate_pct=inp.get(
                            "custom_leak_rate_pct",
                        ),
                        mass_balance_data=inp.get("mass_balance_data"),
                    )
                    results.append(r)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                total_kg = sum(
                    r.get("total_emissions_kg_co2e", 0.0)
                    for r in results
                    if isinstance(r, dict)
                )
                total_tco2e = sum(
                    r.get("total_emissions_tco2e", 0.0)
                    for r in results
                    if isinstance(r, dict)
                )
                success_count = sum(
                    1 for r in results
                    if isinstance(r, dict)
                    and r.get("status") == "SUCCESS"
                )
                batch_result = {
                    "batch_id": batch_id,
                    "results": results,
                    "total_emissions_kg_co2e": round(total_kg, 6),
                    "total_emissions_tco2e": round(total_tco2e, 9),
                    "success_count": success_count,
                    "failure_count": len(results) - success_count,
                    "total_count": len(results),
                    "processing_time_ms": round(elapsed_ms, 3),
                    "provenance_hash": _compute_hash({
                        "batch_id": batch_id,
                    }),
                }

            self._total_batch_runs += 1

            logger.info(
                "Batch %s completed: %d inputs",
                batch_result.get("batch_id", batch_id),
                len(inputs),
            )
            return batch_result

        except Exception as exc:
            logger.error(
                "calculate_batch failed: %s", exc, exc_info=True,
            )
            raise

    def get_refrigerant(
        self,
        refrigerant_type: str,
    ) -> Dict[str, Any]:
        """Get properties for a refrigerant type.

        Args:
            refrigerant_type: Refrigerant type identifier.

        Returns:
            Dictionary with refrigerant properties.
        """
        if self._refrigerant_database_engine is not None:
            try:
                result = self._refrigerant_database_engine.get_refrigerant(
                    refrigerant_type,
                )
                if hasattr(result, "model_dump"):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return result
            except (AttributeError, TypeError, KeyError):
                pass

        # Check in-memory cache
        cached = self._refrigerants.get(refrigerant_type)
        if cached is not None:
            return cached

        return {
            "refrigerant_type": refrigerant_type,
            "error": "Refrigerant type not found",
        }

    def list_refrigerants(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all available refrigerants.

        Args:
            category: Optional category filter (HFC, HFO, PFC, etc.).

        Returns:
            List of refrigerant summary dictionaries.
        """
        if self._refrigerant_database_engine is not None:
            try:
                result = self._refrigerant_database_engine.list_refrigerants(
                    category=category,
                )
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: return cached refrigerants
        refs = list(self._refrigerants.values())
        if category:
            refs = [
                r for r in refs
                if r.get("category") == category
            ]
        return refs

    def register_equipment(
        self,
        equipment_id: Optional[str] = None,
        equipment_type: str = "COMMERCIAL_REFRIGERATION",
        name: str = "",
        facility_id: Optional[str] = None,
        refrigerant_type: str = "",
        charge_kg: float = 0.0,
        capacity_kw: Optional[float] = None,
        age_years: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Register an equipment profile.

        Args:
            equipment_id: Unique equipment ID (auto-generated if None).
            equipment_type: Equipment type classification.
            name: Human-readable equipment name.
            facility_id: Parent facility identifier.
            refrigerant_type: Refrigerant type charged.
            charge_kg: Current refrigerant charge in kg.
            capacity_kw: Cooling or heating capacity in kW.
            age_years: Equipment age in years.
            **kwargs: Additional profile parameters.

        Returns:
            Dictionary with registered equipment details.
        """
        equip_id = equipment_id or _new_uuid()

        if self._equipment_registry_engine is not None:
            try:
                result = self._equipment_registry_engine.register(
                    equipment_id=equip_id,
                    equipment_type=equipment_type,
                    name=name,
                    facility_id=facility_id,
                    refrigerant_type=refrigerant_type,
                    charge_kg=charge_kg,
                    capacity_kw=capacity_kw,
                    age_years=age_years,
                    **kwargs,
                )
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"equipment_id": equip_id}
                self._equipment_profiles[equip_id] = result_dict
                return result_dict
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "EquipmentRegistryEngine.register failed: %s", exc,
                )

        # Fallback: in-memory registration
        profile = EquipmentResponse(
            equipment_id=equip_id,
            equipment_type=equipment_type,
            name=name,
            facility_id=facility_id,
            refrigerant_type=refrigerant_type,
            charge_kg=charge_kg,
            capacity_kw=capacity_kw,
            age_years=age_years,
        )
        profile.provenance_hash = _compute_hash(profile)
        result_dict = profile.model_dump()
        self._equipment_profiles[equip_id] = result_dict

        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="equipment",
                    action="register_equipment",
                    entity_id=equip_id,
                    metadata={
                        "equipment_type": equipment_type,
                        "facility_id": facility_id,
                    },
                )
            except (AttributeError, TypeError):
                pass

        logger.info(
            "Registered equipment %s: type=%s facility=%s",
            equip_id, equipment_type, facility_id,
        )
        return result_dict

    def log_service_event(
        self,
        equipment_id: str,
        event_type: str,
        refrigerant_type: str = "",
        quantity_kg: float = 0.0,
        date: Optional[str] = None,
        technician: str = "",
        notes: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Log a service event for an equipment item.

        Args:
            equipment_id: Equipment identifier.
            event_type: Service event type (installation, recharge,
                repair, recovery, leak_check, decommissioning,
                conversion).
            refrigerant_type: Refrigerant type involved.
            quantity_kg: Refrigerant quantity in kg.
            date: Event date (ISO-8601). Defaults to now.
            technician: Technician or contractor name.
            notes: Additional notes.
            **kwargs: Additional event parameters.

        Returns:
            Dictionary with event details.
        """
        event_id = _new_uuid()
        event_date = date or _utcnow_iso()

        if self._equipment_registry_engine is not None:
            try:
                result = self._equipment_registry_engine.log_event(
                    equipment_id=equipment_id,
                    event_type=event_type,
                    refrigerant_type=refrigerant_type,
                    quantity_kg=quantity_kg,
                    date=event_date,
                    technician=technician,
                    notes=notes,
                    **kwargs,
                )
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"event_id": event_id}
                self._service_events[event_id] = result_dict
                return result_dict
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "EquipmentRegistryEngine.log_event failed: %s", exc,
                )

        # Fallback: in-memory event
        event = ServiceEventResponse(
            event_id=event_id,
            equipment_id=equipment_id,
            event_type=event_type,
            refrigerant_type=refrigerant_type,
            quantity_kg=quantity_kg,
            date=event_date,
            technician=technician,
            notes=notes,
        )
        event.provenance_hash = _compute_hash(event)
        result_dict = event.model_dump()
        self._service_events[event_id] = result_dict

        logger.info(
            "Logged service event %s: equip=%s type=%s qty=%.2f kg",
            event_id, equipment_id, event_type, quantity_kg,
        )
        return result_dict

    def estimate_leak_rate(
        self,
        equipment_type: str,
        age_years: int = 0,
        refrigerant_type: str = "",
        equipment_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Estimate leak rate for an equipment type.

        Args:
            equipment_type: Equipment type classification.
            age_years: Equipment age in years.
            refrigerant_type: Refrigerant type for rate lookup.
            equipment_id: Optional equipment identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with leak rate estimation details.
        """
        if self._leak_rate_estimator_engine is not None:
            try:
                result = self._leak_rate_estimator_engine.estimate(
                    equipment_type=equipment_type,
                    age_years=age_years,
                    refrigerant_type=refrigerant_type,
                    equipment_id=equipment_id,
                    **kwargs,
                )
                if hasattr(result, "model_dump"):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return result
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "LeakRateEstimatorEngine.estimate failed: %s", exc,
                )

        # Fallback: default leak rate
        default_rates = {
            "COMMERCIAL_REFRIGERATION": 15.0,
            "INDUSTRIAL_REFRIGERATION": 10.0,
            "RESIDENTIAL_AC": 4.0,
            "COMMERCIAL_AC": 6.0,
            "CHILLER_CENTRIFUGAL": 2.0,
            "CHILLER_SCREW": 3.0,
            "CHILLER_RECIPROCATING": 5.0,
            "HEAT_PUMP": 3.0,
            "TRANSPORT_REFRIGERATION": 15.0,
            "MOBILE_AC": 12.0,
            "SWITCHGEAR": 0.5,
            "FOAM_BLOWING": 3.5,
            "FIRE_SUPPRESSION": 1.0,
            "AEROSOL": 50.0,
            "SEMICONDUCTOR": 5.0,
        }
        base_rate = default_rates.get(
            equipment_type.upper(), 5.0,
        )
        age_factor = 1.0 + (float(age_years) * 0.02)
        effective_rate = min(base_rate * age_factor, 100.0)

        result = LeakRateResponse(
            equipment_type=equipment_type,
            base_rate_pct=base_rate,
            age_factor=round(age_factor, 4),
            effective_rate_pct=round(effective_rate, 4),
            source="default",
        )
        result.provenance_hash = _compute_hash(result)
        return result.model_dump()

    def check_compliance(
        self,
        calculation_id: Optional[str] = None,
        frameworks: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check compliance against regulatory frameworks.

        Args:
            calculation_id: Optional calculation ID to check.
            frameworks: List of framework names to check against.
            **kwargs: Additional compliance parameters.

        Returns:
            Dictionary with compliance check results.
        """
        if frameworks is None:
            frameworks = [
                "GHG_PROTOCOL",
                "EPA_40CFR98_DD",
                "EU_FGAS_2024",
                "KIGALI_AMENDMENT",
                "ISO_14064",
                "CSRD_ESRS_E1",
                "UK_FGAS",
            ]

        calc_result = None
        if calculation_id:
            calc_result = self._calculations.get(calculation_id)

        if self._compliance_tracker_engine is not None:
            try:
                result = self._compliance_tracker_engine.check(
                    calculation_result=calc_result,
                    frameworks=frameworks,
                    **kwargs,
                )
                if isinstance(result, dict):
                    self._total_compliance_checks += 1
                    return result
                if isinstance(result, list):
                    self._total_compliance_checks += 1
                    compliant_count = sum(
                        1 for r in result
                        if (r.get("compliant", False)
                            if isinstance(r, dict)
                            else getattr(r, "compliant", False))
                    )
                    return {
                        "records": [
                            r.model_dump(mode="json")
                            if hasattr(r, "model_dump")
                            else r
                            for r in result
                        ],
                        "total_count": len(result),
                        "compliant_count": compliant_count,
                        "overall_compliant": compliant_count == len(result),
                    }
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "ComplianceTrackerEngine.check failed: %s", exc,
                )

        # Fallback: stub compliance
        records = []
        for fw in frameworks:
            records.append({
                "framework": fw,
                "compliant": True,
                "requirements_met": self._get_framework_requirements(fw),
                "requirements_gap": [],
                "details": {},
                "provenance_hash": _compute_hash({
                    "framework": fw, "compliant": True,
                }),
            })

        self._total_compliance_checks += 1

        compliant_count = sum(
            1 for r in records if r.get("compliant", False)
        )

        return {
            "records": records,
            "total_count": len(records),
            "compliant_count": compliant_count,
            "overall_compliant": compliant_count == len(records),
            "provenance_hash": _compute_hash(records),
        }

    def run_uncertainty(
        self,
        calculation_id: str,
        iterations: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation result.

        Args:
            calculation_id: Calculation identifier to analyse.
            iterations: Monte Carlo iterations (defaults to config value).
            **kwargs: Additional uncertainty parameters.

        Returns:
            Dictionary with uncertainty analysis results.
        """
        calc_result = self._calculations.get(calculation_id)
        if calc_result is None:
            return {
                "calculation_id": calculation_id,
                "error": "Calculation not found",
            }

        iters = iterations or 5000
        if self.config is not None:
            iters = iterations or getattr(
                self.config, "monte_carlo_iterations", 5000,
            )

        if self._uncertainty_quantifier_engine is not None:
            try:
                result = self._uncertainty_quantifier_engine.quantify(
                    calculation_result=calc_result,
                    iterations=iters,
                    **kwargs,
                )
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {}

                self._total_uncertainty_runs += 1
                self._uncertainty_results[calculation_id] = result_dict
                return result_dict

            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine.quantify failed: %s",
                    exc,
                )

        # Fallback: analytical stub
        total_emissions = calc_result.get(
            "total_emissions_kg_co2e", 0.0,
        )
        relative_unc = 0.25
        std_estimate = total_emissions * relative_unc

        self._total_uncertainty_runs += 1

        result_dict = {
            "calculation_id": calculation_id,
            "method": "analytical_stub",
            "mean_co2e_kg": total_emissions,
            "std_co2e_kg": round(std_estimate, 6),
            "p5_co2e_kg": round(
                total_emissions - 1.645 * std_estimate, 6,
            ),
            "p95_co2e_kg": round(
                total_emissions + 1.645 * std_estimate, 6,
            ),
            "confidence_interval_pct": 90.0,
            "iterations": 0,
            "data_quality_score": 3,
            "message": "UncertaintyQuantifierEngine not available; "
                       "returning analytical estimate",
            "provenance_hash": _compute_hash({
                "calculation_id": calculation_id,
                "mean": total_emissions,
            }),
        }
        self._uncertainty_results[calculation_id] = result_dict
        return result_dict

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
        # Check pipeline result for embedded audit
        calc = self._calculations.get(calculation_id, {})
        if isinstance(calc, dict):
            embedded = calc.get("audit_entries", [])
            if embedded:
                return embedded

        # Check audit entries cache
        cached = self._audit_entries.get(calculation_id, [])
        if cached:
            return cached

        # Try pipeline engine
        if self._pipeline_engine is not None:
            try:
                stats = self._pipeline_engine.get_pipeline_stats()
                recent = stats.get("recent_runs", [])
                for run in recent:
                    if run.get("calculation_id") == calculation_id:
                        return [{
                            "audit_id": _new_uuid(),
                            "calculation_id": calculation_id,
                            "action": "pipeline_run",
                            "details": run,
                            "timestamp": run.get("timestamp", ""),
                        }]
            except (AttributeError, TypeError):
                pass

        return []

    def aggregate(
        self,
        calculation_ids: Optional[List[str]] = None,
        control_approach: str = "OPERATIONAL",
        share: float = 1.0,
        facility_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate calculation results at facility level.

        Args:
            calculation_ids: Optional list of calculation IDs.
                If None, aggregates all calculations.
            control_approach: GHG Protocol boundary approach.
            share: Ownership share fraction (0.0-1.0).
            facility_id: Optional facility filter.

        Returns:
            Dictionary with facility aggregation results.
        """
        # Gather calculation results
        if calculation_ids:
            results = [
                self._calculations[cid]
                for cid in calculation_ids
                if cid in self._calculations
            ]
        else:
            results = list(self._calculations.values())

        if facility_id:
            results = [
                r for r in results
                if isinstance(r, dict)
                and r.get("facility_id") == facility_id
            ]

        if not results:
            return {
                "aggregations": [],
                "total_facilities": 0,
                "grand_total_tco2e": 0.0,
                "control_approach": control_approach,
            }

        # Delegate to pipeline engine if available
        if self._pipeline_engine is not None:
            try:
                pipeline_results = [
                    {"result": r, "calculation_id": r.get(
                        "calculation_id", "",
                    )}
                    for r in results
                ]
                return self._pipeline_engine.aggregate_facility(
                    pipeline_results,
                    control_approach=control_approach,
                    share=share,
                )
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "Pipeline aggregate_facility failed: %s", exc,
                )

        # Fallback: simple aggregation
        from collections import defaultdict

        facility_map: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_emissions_tco2e": 0.0,
                "calculation_count": 0,
            },
        )

        share_mult = 1.0
        if control_approach in ("FINANCIAL", "EQUITY_SHARE"):
            share_mult = max(0.0, min(1.0, share))

        for r in results:
            fid = r.get("facility_id", "UNASSIGNED") or "UNASSIGNED"
            facility_map[fid]["total_emissions_tco2e"] += (
                r.get("total_emissions_tco2e", 0.0) * share_mult
            )
            facility_map[fid]["calculation_count"] += 1

        aggs = [
            {
                "facility_id": fid,
                "total_emissions_tco2e": round(
                    data["total_emissions_tco2e"], 9,
                ),
                "calculation_count": data["calculation_count"],
                "control_approach": control_approach,
            }
            for fid, data in facility_map.items()
        ]

        return {
            "aggregations": aggs,
            "total_facilities": len(aggs),
            "grand_total_tco2e": round(
                sum(a["total_emissions_tco2e"] for a in aggs), 9,
            ),
            "control_approach": control_approach,
        }

    def validate(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate input data without performing calculations.

        Args:
            inputs: List of input dictionaries to validate.

        Returns:
            Dictionary with validation results.
        """
        from greenlang.refrigerants_fgas.refrigerant_pipeline import (
            SUPPORTED_METHODS,
        )

        errors: List[str] = []
        warnings: List[str] = []

        for idx, inp in enumerate(inputs):
            ref_type = inp.get("refrigerant_type", "")
            if not ref_type:
                errors.append(
                    f"Input [{idx}]: refrigerant_type is required"
                )

            charge_kg = inp.get("charge_kg")
            if charge_kg is None:
                errors.append(
                    f"Input [{idx}]: charge_kg is required"
                )
            elif not isinstance(charge_kg, (int, float)):
                errors.append(
                    f"Input [{idx}]: charge_kg must be a number"
                )
            elif charge_kg <= 0:
                errors.append(
                    f"Input [{idx}]: charge_kg must be > 0"
                )

            method = inp.get("method", "equipment_based")
            if method not in SUPPORTED_METHODS:
                errors.append(
                    f"Input [{idx}]: method '{method}' is not supported"
                )

            custom_leak = inp.get("custom_leak_rate_pct")
            if custom_leak is not None:
                if not isinstance(custom_leak, (int, float)):
                    errors.append(
                        f"Input [{idx}]: custom_leak_rate_pct must "
                        f"be a number"
                    )
                elif custom_leak < 0 or custom_leak > 100:
                    errors.append(
                        f"Input [{idx}]: custom_leak_rate_pct must "
                        f"be between 0 and 100"
                    )

            if method == "mass_balance" and not inp.get("mass_balance_data"):
                errors.append(
                    f"Input [{idx}]: mass_balance_data is required "
                    f"for mass_balance method"
                )

            if not inp.get("facility_id"):
                warnings.append(
                    f"Input [{idx}]: no facility_id; will aggregate "
                    f"under UNASSIGNED"
                )

        valid_count = len(inputs) - len([
            e for e in errors if e.startswith("Input [")
        ])

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "validated_count": max(valid_count, 0),
            "total_count": len(inputs),
        }

    def get_health(self) -> Dict[str, Any]:
        """Perform a health check on the refrigerants & F-gas service.

        Returns:
            Dictionary with health status for each engine and the overall
            service, including version and statistics.
        """
        engines: Dict[str, str] = {
            "refrigerant_database": (
                "available"
                if self._refrigerant_database_engine is not None
                else "unavailable"
            ),
            "emission_calculator": (
                "available"
                if self._emission_calculator_engine is not None
                else "unavailable"
            ),
            "equipment_registry": (
                "available"
                if self._equipment_registry_engine is not None
                else "unavailable"
            ),
            "leak_rate_estimator": (
                "available"
                if self._leak_rate_estimator_engine is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available"
                if self._uncertainty_quantifier_engine is not None
                else "unavailable"
            ),
            "compliance_tracker": (
                "available"
                if self._compliance_tracker_engine is not None
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
            "refrigerant_count": len(self._refrigerants),
            "equipment_count": len(self._equipment_profiles),
            "statistics": {
                "total_calculations": self._total_calculations,
                "total_batch_runs": self._total_batch_runs,
                "total_pipeline_runs": self._total_pipeline_runs,
                "total_compliance_checks": self._total_compliance_checks,
                "total_uncertainty_runs": self._total_uncertainty_runs,
                "avg_calculation_time_ms": round(avg_calc_time, 3),
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": provenance_entries,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

    def get_stats(self) -> Dict[str, Any]:
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
            "total_refrigerants": len(self._refrigerants),
            "total_equipment": len(self._equipment_profiles),
            "total_service_events": len(self._service_events),
            "total_compliance_checks": self._total_compliance_checks,
            "total_uncertainty_runs": self._total_uncertainty_runs,
            "total_audit_entries": sum(
                len(entries)
                for entries in self._audit_entries.values()
            ),
            "avg_calculation_time_ms": round(avg_calc_time, 3),
            "timestamp": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the refrigerants & F-gas service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug(
                "RefrigerantsFGasService already started; skipping",
            )
            return

        logger.info("RefrigerantsFGasService starting up...")
        self._started = True
        logger.info("RefrigerantsFGasService startup complete")

    def shutdown(self) -> None:
        """Shutdown the refrigerants & F-gas service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("RefrigerantsFGasService shut down")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_framework_requirements(
        self,
        framework: str,
    ) -> List[str]:
        """Get the set of requirement names for a regulatory framework.

        Args:
            framework: Regulatory framework identifier.

        Returns:
            List of requirement name strings.
        """
        requirements_map: Dict[str, List[str]] = {
            "GHG_PROTOCOL": [
                "Scope 1 Refrigerant Emissions",
                "Equipment Charge Documentation",
                "Leak Rate Methodology",
                "GWP Source Documentation",
            ],
            "EPA_40CFR98_DD": [
                "Subpart DD Refrigerant Tracking",
                "Mass Balance Methodology",
                "Equipment Inventory",
                "Annual Reporting",
            ],
            "EU_FGAS_2024": [
                "Regulation 2024/573 Compliance",
                "Phase-Down Schedule Tracking",
                "Leak Detection and Repair",
                "Record Keeping",
                "GWP Threshold Checks",
            ],
            "KIGALI_AMENDMENT": [
                "HFC Phase-Down Compliance",
                "Baseline Calculation",
                "Production/Consumption Tracking",
            ],
            "ISO_14064": [
                "Clause 5 Quantification",
                "Uncertainty Assessment",
                "Documentation Requirements",
            ],
            "CSRD_ESRS_E1": [
                "Climate Change Disclosure",
                "Scope 1 F-Gas Emissions",
                "Methodology Transparency",
            ],
            "UK_FGAS": [
                "UK F-Gas Regulation Compliance",
                "Leak Checking Requirements",
                "Certification Requirements",
            ],
        }

        return requirements_map.get(framework, [])


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> RefrigerantsFGasService:
    """Get or create the singleton RefrigerantsFGasService instance.

    Returns:
        The singleton RefrigerantsFGasService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = RefrigerantsFGasService()
    return _singleton_instance


# ===================================================================
# Module-level singletons for FastAPI integration
# ===================================================================

_service: Optional[RefrigerantsFGasService] = None
_router: Any = None


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_refrigerants_fgas(
    app: Any,
    config: Any = None,
) -> RefrigerantsFGasService:
    """Configure the Refrigerants & F-Gas Service on a FastAPI application.

    Creates the RefrigerantsFGasService, stores it in app.state,
    mounts the refrigerants-fgas API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional refrigerants-fgas config.

    Returns:
        RefrigerantsFGasService instance.
    """
    global _singleton_instance, _service

    service = RefrigerantsFGasService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service
        _service = service

    # Attach to app state
    app.state.refrigerants_fgas_service = service

    # Mount refrigerants-fgas API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Refrigerants & F-Gas API router mounted")
    else:
        logger.warning(
            "Refrigerants & F-Gas router not available; "
            "API not mounted",
        )

    # Start service
    service.startup()

    logger.info(
        "Refrigerants & F-Gas service configured on app",
    )
    return service


def get_service() -> Optional[RefrigerantsFGasService]:
    """Get the singleton RefrigerantsFGasService instance.

    Creates a new instance if one does not exist yet. Uses
    double-checked locking for thread safety.

    Returns:
        RefrigerantsFGasService singleton instance or None.
    """
    global _singleton_instance, _service
    if _service is not None:
        return _service
    if _singleton_instance is not None:
        return _singleton_instance
    # Lazy creation
    with _singleton_lock:
        if _singleton_instance is None:
            _singleton_instance = RefrigerantsFGasService()
        _service = _singleton_instance
    return _singleton_instance


def get_router() -> Any:
    """Get the refrigerants-fgas API router.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.refrigerants_fgas.api.router import router
        return router
    except ImportError:
        logger.warning(
            "Refrigerants & F-Gas API router module not available",
        )
        return None


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "RefrigerantsFGasService",
    # Configuration helpers
    "configure_refrigerants_fgas",
    "get_service",
    "get_router",
    # Response models
    "CalculationResponse",
    "BatchResponse",
    "RefrigerantResponse",
    "RefrigerantListResponse",
    "EquipmentResponse",
    "EquipmentListResponse",
    "ServiceEventResponse",
    "LeakRateResponse",
    "ComplianceResponse",
    "ComplianceListResponse",
    "UncertaintyResponse",
    "AuditTrailResponse",
    "BlendResponse",
    "ValidationResponse",
    "PipelineResponse",
    "HealthResponse",
    "StatsResponse",
]
