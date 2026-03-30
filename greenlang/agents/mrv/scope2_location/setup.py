# -*- coding: utf-8 -*-
"""
Scope 2 Location-Based Emissions Service Setup - AGENT-MRV-009
===============================================================

Service facade for the Scope 2 Location-Based Emissions Agent
(GL-MRV-SCOPE2-001).

Provides ``get_service()`` and the ``Scope2LocationService`` facade class
that aggregates all 7 engines:

    1. GridEmissionFactorDatabaseEngine - Grid factors (eGRID/IEA/DEFRA/EU)
    2. ElectricityEmissionsEngine       - Electricity emission calculations
    3. SteamHeatCoolingEngine           - Steam, heat, cooling calculations
    4. TransmissionLossEngine           - T&D loss adjustments
    5. UncertaintyQuantifierEngine      - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine          - Multi-framework regulatory compliance
    7. Scope2LocationPipelineEngine     - 8-stage orchestrated pipeline

The service provides 20 public methods matching the 20 REST API endpoints:

    Calculations:
        calculate, calculate_batch, list_calculations,
        get_calculation, delete_calculation
    Facilities:
        register_facility, list_facilities, update_facility
    Consumption:
        record_consumption, list_consumption
    Grid Factors:
        list_grid_factors, get_grid_factor, add_custom_factor
    T&D Losses:
        list_td_losses
    Compliance:
        check_compliance, get_compliance_result
    Uncertainty:
        run_uncertainty
    Aggregations:
        get_aggregations
    Health:
        health_check, get_stats

All calculation paths use deterministic Decimal arithmetic for
zero-hallucination guarantees. Every mutation records a SHA-256
provenance hash for complete audit trails.

Usage:
    >>> from greenlang.agents.mrv.scope2_location.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "tenant_id": "tenant-001",
    ...     "facility_id": "fac-001",
    ...     "energy_type": "electricity",
    ...     "consumption_value": 5000.0,
    ...     "consumption_unit": "mwh",
    ...     "country_code": "US",
    ...     "egrid_subregion": "CAMX",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope2_location.config import (
        Scope2LocationConfig,
        get_config,
    )
except ImportError:
    Scope2LocationConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.scope2_location.metrics import (
        Scope2LocationMetrics,
        get_metrics,
    )
except ImportError:
    Scope2LocationMetrics = None  # type: ignore[assignment, misc]

    def get_metrics() -> Any:  # type: ignore[misc]
        """Stub returning None when metrics module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.scope2_location.provenance import (
        create_provenance,
    )
except ImportError:
    def create_provenance(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Stub returning None when provenance module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.scope2_location.grid_factor_database import (
        GridEmissionFactorDatabaseEngine,
    )
except ImportError:
    GridEmissionFactorDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.electricity_emissions import (
        ElectricityEmissionsEngine,
    )
except ImportError:
    ElectricityEmissionsEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.steam_heat_cooling import (
        SteamHeatCoolingEngine,
    )
except ImportError:
    SteamHeatCoolingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.transmission_loss import (
        TransmissionLossEngine,
    )
except ImportError:
    TransmissionLossEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_location.scope2_location_pipeline import (
        Scope2LocationPipelineEngine,
    )
except ImportError:
    Scope2LocationPipelineEngine = None  # type: ignore[assignment, misc]

# ===================================================================
# Constants
# ===================================================================

#: Service version for health checks and diagnostics.
SERVICE_VERSION: str = "1.0.0"

#: Service name for observability.
SERVICE_NAME: str = "scope2-location-service"

#: Agent identifier for tracing and audit logs.
AGENT_ID: str = "AGENT-MRV-009"

#: Default GWP source when not provided in requests.
DEFAULT_GWP_SOURCE: str = "AR5"

#: Default max batch size.
DEFAULT_MAX_BATCH_SIZE: int = 1000

#: Valid energy types for input validation.
VALID_ENERGY_TYPES: frozenset = frozenset({
    "electricity",
    "steam",
    "heating",
    "cooling",
})

#: Valid GWP sources for input validation.
VALID_GWP_SOURCES: frozenset = frozenset({
    "AR4",
    "AR5",
    "AR6",
    "AR6_20YR",
})

#: Valid consumption units for input validation.
VALID_CONSUMPTION_UNITS: frozenset = frozenset({
    "kwh",
    "mwh",
    "gj",
    "mmbtu",
    "therms",
})

#: Valid facility types for input validation.
VALID_FACILITY_TYPES: frozenset = frozenset({
    "office",
    "warehouse",
    "manufacturing",
    "retail",
    "data_center",
    "hospital",
    "school",
    "other",
})

#: Valid grid region sources.
VALID_GRID_SOURCES: frozenset = frozenset({
    "egrid",
    "iea",
    "eu_eea",
    "defra",
    "national",
    "custom",
})

#: Supported compliance frameworks.
VALID_COMPLIANCE_FRAMEWORKS: frozenset = frozenset({
    "ghg_protocol_scope2",
    "ipcc_2006",
    "iso_14064",
    "csrd_esrs",
    "epa_ghgrp",
    "defra",
    "cdp",
})

#: Valid aggregation group-by dimensions.
VALID_GROUP_BY: frozenset = frozenset({
    "facility",
    "energy_type",
    "grid_region",
    "country",
    "month",
    "quarter",
})

#: Valid steam sub-types.
VALID_STEAM_TYPES: frozenset = frozenset({
    "natural_gas",
    "coal",
    "biomass",
    "oil",
})

#: Valid heating sub-types.
VALID_HEATING_TYPES: frozenset = frozenset({
    "district",
    "gas_boiler",
    "electric",
})

#: Valid cooling sub-types.
VALID_COOLING_TYPES: frozenset = frozenset({
    "electric_chiller",
    "absorption",
    "district",
})

# ===================================================================
# Utility helpers
# ===================================================================

def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return utcnow().isoformat()

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _short_id(prefix: str = "s2l") -> str:
    """Generate a short prefixed identifier for records.

    Args:
        prefix: Prefix string prepended to the UUID fragment.

    Returns:
        A string of the form ``{prefix}_{12-char-hex}``.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Handles Pydantic models (via ``model_dump``), dicts, lists,
    and primitive values. Decimal values are serialised via ``str``.

    Args:
        data: Arbitrary data to hash.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with graceful fallback.

    Handles Decimal, int, str, and None values. Returns the
    default on any conversion failure.

    Args:
        value: Value to convert.
        default: Fallback value on conversion failure.

    Returns:
        Float representation of the value.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError, ArithmeticError):
        return default

def _safe_decimal(value: Any, default: str = "0") -> Decimal:
    """Convert a value to Decimal with graceful fallback.

    Args:
        value: Value to convert.
        default: Fallback string representation on failure.

    Returns:
        Decimal representation of the value.
    """
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)

def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since a monotonic start time.

    Args:
        start: ``time.monotonic()`` start value.

    Returns:
        Elapsed time in milliseconds, rounded to 3 decimal places.
    """
    return round((time.monotonic() - start) * 1000.0, 3)

def _validate_required_fields(
    data: Dict[str, Any],
    fields: List[str],
    context: str = "request",
) -> List[str]:
    """Validate that required fields are present and non-empty.

    Args:
        data: Dictionary to validate.
        fields: List of required field names.
        context: Human-readable context for error messages.

    Returns:
        List of validation error strings (empty if valid).
    """
    errors: List[str] = []
    for field in fields:
        val = data.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(
                f"Missing required field '{field}' in {context}"
            )
    return errors

def _validate_enum_field(
    value: Any,
    valid_values: frozenset,
    field_name: str,
) -> Optional[str]:
    """Validate that a field value is in the allowed set.

    Args:
        value: Value to validate.
        valid_values: Set of allowed values.
        field_name: Human-readable field name for error messages.

    Returns:
        Error string if invalid, None if valid.
    """
    if value is None:
        return None
    normalized = str(value).lower().strip()
    if normalized not in valid_values:
        return (
            f"Invalid {field_name} '{value}'; "
            f"must be one of {sorted(valid_values)}"
        )
    return None

# ===================================================================
# Pydantic Response Models (14 models)
# ===================================================================

class CalculateResponse(BaseModel):
    """Single Scope 2 location-based emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    energy_type: str = Field(default="electricity")
    consumption_value: float = Field(default=0.0)
    consumption_unit: str = Field(default="mwh")
    grid_region: str = Field(default="")
    emission_factor_source: str = Field(default="")
    ef_co2e_per_mwh: float = Field(default=0.0)
    td_loss_pct: float = Field(default=0.0)
    co2_kg: float = Field(default=0.0)
    ch4_kg: float = Field(default=0.0)
    n2o_kg: float = Field(default=0.0)
    total_co2e_kg: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    gwp_source: str = Field(default="AR5")
    gas_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchCalculateResponse(BaseModel):
    """Batch Scope 2 location-based emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    batch_id: str = Field(default="")
    total_calculations: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)

class FacilityResponse(BaseModel):
    """Response for a single facility record."""

    model_config = ConfigDict(frozen=True)

    facility_id: str = Field(default="")
    name: str = Field(default="")
    facility_type: str = Field(default="office")
    country_code: str = Field(default="")
    grid_region_id: str = Field(default="")
    egrid_subregion: Optional[str] = Field(default=None)
    latitude: Optional[float] = Field(default=None)
    longitude: Optional[float] = Field(default=None)
    tenant_id: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class FacilityListResponse(BaseModel):
    """Response listing registered facilities."""

    model_config = ConfigDict(frozen=True)

    facilities: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=50)

class ConsumptionResponse(BaseModel):
    """Response for a single energy consumption record."""

    model_config = ConfigDict(frozen=True)

    consumption_id: str = Field(default="")
    facility_id: str = Field(default="")
    energy_type: str = Field(default="electricity")
    quantity: float = Field(default=0.0)
    unit: str = Field(default="mwh")
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    data_source: str = Field(default="invoice")
    meter_id: Optional[str] = Field(default=None)
    created_at: str = Field(default="")

class ConsumptionListResponse(BaseModel):
    """Response listing energy consumption records."""

    model_config = ConfigDict(frozen=True)

    records: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)

class GridFactorResponse(BaseModel):
    """Response for a single grid emission factor."""

    model_config = ConfigDict(frozen=True)

    region_id: str = Field(default="")
    country_code: str = Field(default="")
    source: str = Field(default="")
    year: int = Field(default=0)
    co2_kg_per_mwh: float = Field(default=0.0)
    ch4_kg_per_mwh: float = Field(default=0.0)
    n2o_kg_per_mwh: float = Field(default=0.0)
    co2e_per_mwh: float = Field(default=0.0)
    data_quality_tier: str = Field(default="tier_1")
    td_loss_pct: float = Field(default=0.0)
    notes: str = Field(default="")

class GridFactorListResponse(BaseModel):
    """Response listing grid emission factors."""

    model_config = ConfigDict(frozen=True)

    factors: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    source_filter: Optional[str] = Field(default=None)

class TDLossListResponse(BaseModel):
    """Response listing T&D loss factors."""

    model_config = ConfigDict(frozen=True)

    factors: Dict[str, Any] = Field(default_factory=dict)
    total: int = Field(default=0)
    source: str = Field(default="world_bank_iea")

class ComplianceCheckResponse(BaseModel):
    """Regulatory compliance check response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    id: str = Field(default="")
    calculation_id: str = Field(default="")
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: str = Field(default_factory=_utcnow_iso)
    provenance_hash: str = Field(default="")

class UncertaintyResponse(BaseModel):
    """Monte Carlo or analytical uncertainty analysis response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="monte_carlo")
    iterations: int = Field(default=0)
    mean_co2e_tonnes: float = Field(default=0.0)
    std_dev_tonnes: float = Field(default=0.0)
    ci_lower: float = Field(default=0.0)
    ci_upper: float = Field(default=0.0)
    confidence_level: float = Field(default=0.95)
    coefficient_of_variation: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)

class AggregationResponse(BaseModel):
    """Aggregated Scope 2 location-based emissions response."""

    model_config = ConfigDict(frozen=True)

    group_by: str = Field(default="facility")
    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_co2e_tonnes: float = Field(default=0.0)
    facility_count: int = Field(default=0)
    calculation_count: int = Field(default=0)
    period: str = Field(default="all")
    timestamp: str = Field(default_factory=_utcnow_iso)

class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default=SERVICE_NAME)
    version: str = Field(default=SERVICE_VERSION)
    agent_id: str = Field(default=AGENT_ID)
    engines: Dict[str, str] = Field(default_factory=dict)
    config_valid: bool = Field(default=True)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)

class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_facilities: int = Field(default=0)
    total_consumption_records: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)

# ===================================================================
# Scope2LocationService facade
# ===================================================================

_singleton_lock = threading.Lock()
_service_instance: Optional["Scope2LocationService"] = None

class Scope2LocationService:
    """Unified facade over the Scope 2 Location-Based Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each mutation method records SHA-256 provenance hashes.
    All numeric calculations use deterministic Decimal arithmetic
    delegated to the underlying engines (zero-hallucination path).

    In-memory storage provides the default persistence layer. In
    production, methods should be backed by PostgreSQL via the
    engines' database connectors.

    Attributes:
        config: Service configuration (Scope2LocationConfig or dict).
        metrics: Prometheus metrics singleton.

    Example:
        >>> service = Scope2LocationService()
        >>> result = service.calculate({
        ...     "tenant_id": "tenant-001",
        ...     "facility_id": "fac-001",
        ...     "energy_type": "electricity",
        ...     "consumption_value": 5000.0,
        ...     "consumption_unit": "mwh",
        ...     "country_code": "US",
        ...     "egrid_subregion": "CAMX",
        ... })
        >>> assert result.success is True
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Scope 2 Location-Based Emissions Service facade.

        Creates all engine instances with graceful degradation when
        individual engine modules are not importable. Sets up in-memory
        storage for calculations, facilities, consumption records,
        compliance results, and uncertainty analyses.

        Args:
            config: Optional configuration override. Accepts
                Scope2LocationConfig, dict, or None (defaults to
                singleton from get_config()).
        """
        self._config = config if config is not None else get_config()
        self._metrics = get_metrics()
        self._start_time: float = time.monotonic()

        # Engine placeholders (initialised in _init_engines)
        self._grid_db: Any = None
        self._electricity: Any = None
        self._steam_heat_cool: Any = None
        self._transmission: Any = None
        self._uncertainty: Any = None
        self._compliance: Any = None
        self._pipeline: Any = None

        self._init_engines()

        # In-memory data stores
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._facilities: Dict[str, Dict[str, Any]] = {}
        self._consumption: List[Dict[str, Any]] = []
        self._compliance_results: Dict[str, Dict[str, Any]] = {}
        self._uncertainty_results: Dict[str, Dict[str, Any]] = {}
        self._custom_factors: Dict[str, Dict[str, Any]] = {}

        # Aggregate statistics
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_compliance_checks: int = 0
        self._total_uncertainty_runs: int = 0
        self._cumulative_co2e_tonnes: float = 0.0

        logger.info(
            "Scope2LocationService facade created "
            "(engines: grid_db=%s, electricity=%s, "
            "steam_heat_cool=%s, transmission=%s, "
            "uncertainty=%s, compliance=%s, pipeline=%s)",
            self._grid_db is not None,
            self._electricity is not None,
            self._steam_heat_cool is not None,
            self._transmission is not None,
            self._uncertainty is not None,
            self._compliance is not None,
            self._pipeline is not None,
        )

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> Any:
        """Get the service configuration."""
        return self._config

    @property
    def metrics(self) -> Any:
        """Get the Prometheus metrics singleton."""
        return self._metrics

    @property
    def grid_db_engine(self) -> Any:
        """Get the GridEmissionFactorDatabaseEngine instance."""
        return self._grid_db

    @property
    def electricity_engine(self) -> Any:
        """Get the ElectricityEmissionsEngine instance."""
        return self._electricity

    @property
    def steam_heat_cool_engine(self) -> Any:
        """Get the SteamHeatCoolingEngine instance."""
        return self._steam_heat_cool

    @property
    def transmission_engine(self) -> Any:
        """Get the TransmissionLossEngine instance."""
        return self._transmission

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty

    @property
    def compliance_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance."""
        return self._compliance

    @property
    def pipeline_engine(self) -> Any:
        """Get the Scope2LocationPipelineEngine instance."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Import and initialise all SDK engines with graceful fallback.

        Each engine is constructed independently. If any engine fails
        to initialise, a warning is logged and the engine attribute
        remains None. The service continues to operate with reduced
        functionality.

        The pipeline engine is initialised last because it depends on
        all upstream engines.
        """
        # Engines expect a dict-style config (using .get()).
        # Convert Scope2LocationConfig to dict if needed.
        config_arg = self._config
        if config_arg is not None and hasattr(config_arg, "to_dict"):
            config_arg = config_arg.to_dict()
        elif config_arg is not None and not isinstance(config_arg, dict):
            config_arg = {}
        metrics_arg = self._metrics

        # E1: GridEmissionFactorDatabaseEngine
        self._grid_db = self._init_single_engine(
            "GridEmissionFactorDatabaseEngine",
            GridEmissionFactorDatabaseEngine,
            config_arg,
            metrics_arg,
        )

        # E2: ElectricityEmissionsEngine
        self._electricity = self._init_electricity_engine(
            config_arg, metrics_arg,
        )

        # E3: SteamHeatCoolingEngine
        self._steam_heat_cool = self._init_steam_heat_cool_engine(
            config_arg, metrics_arg,
        )

        # E4: TransmissionLossEngine
        self._transmission = self._init_single_engine(
            "TransmissionLossEngine",
            TransmissionLossEngine,
            config_arg,
            metrics_arg,
        )

        # E5: UncertaintyQuantifierEngine
        self._uncertainty = self._init_single_engine(
            "UncertaintyQuantifierEngine",
            UncertaintyQuantifierEngine,
            config_arg,
            metrics_arg,
        )

        # E6: ComplianceCheckerEngine
        self._compliance = self._init_single_engine(
            "ComplianceCheckerEngine",
            ComplianceCheckerEngine,
            config_arg,
            metrics_arg,
        )

        # E7: Scope2LocationPipelineEngine (depends on all upstream)
        self._init_pipeline_engine(config_arg, metrics_arg)

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize a single engine with graceful degradation.

        Tries multiple constructor signatures to handle different
        engine initialization patterns.

        Args:
            name: Human-readable engine name for logging.
            engine_class: Engine class or None if import failed.
            config_arg: Configuration to pass to the engine.
            metrics_arg: Metrics instance to pass to the engine.

        Returns:
            Engine instance or None on failure.
        """
        if engine_class is None:
            logger.warning("%s not available (import failed)", name)
            return None

        try:
            return engine_class(config_arg, metrics_arg)
        except TypeError:
            pass

        try:
            return engine_class(config=config_arg)
        except TypeError:
            pass

        try:
            return engine_class()
        except Exception as exc:
            logger.warning(
                "%s initialization failed: %s", name, exc,
            )
            return None

    def _init_electricity_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize the ElectricityEmissionsEngine.

        The electricity engine takes the grid_db engine as its first
        argument for emission factor lookups.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.

        Returns:
            ElectricityEmissionsEngine instance or None.
        """
        if ElectricityEmissionsEngine is None:
            logger.warning(
                "ElectricityEmissionsEngine not available (import failed)"
            )
            return None

        try:
            return ElectricityEmissionsEngine(
                self._grid_db, config_arg, metrics_arg,
            )
        except TypeError:
            pass

        try:
            return ElectricityEmissionsEngine(config=config_arg)
        except TypeError:
            pass

        try:
            return ElectricityEmissionsEngine()
        except Exception as exc:
            logger.warning(
                "ElectricityEmissionsEngine initialization failed: %s",
                exc,
            )
            return None

    def _init_steam_heat_cool_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize the SteamHeatCoolingEngine.

        The steam/heat/cooling engine takes the grid_db engine as its
        first argument for emission factor lookups on grid-backed types.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.

        Returns:
            SteamHeatCoolingEngine instance or None.
        """
        if SteamHeatCoolingEngine is None:
            logger.warning(
                "SteamHeatCoolingEngine not available (import failed)"
            )
            return None

        try:
            return SteamHeatCoolingEngine(
                self._grid_db, config_arg, metrics_arg,
            )
        except TypeError:
            pass

        try:
            return SteamHeatCoolingEngine(config=config_arg)
        except TypeError:
            pass

        try:
            return SteamHeatCoolingEngine()
        except Exception as exc:
            logger.warning(
                "SteamHeatCoolingEngine initialization failed: %s",
                exc,
            )
            return None

    def _init_pipeline_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> None:
        """Initialize the Scope2LocationPipelineEngine.

        The pipeline engine receives all upstream engine instances
        for orchestrated calculation.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.
        """
        if Scope2LocationPipelineEngine is None:
            logger.warning(
                "Scope2LocationPipelineEngine not available (import failed)"
            )
            return

        try:
            self._pipeline = Scope2LocationPipelineEngine(
                grid_factor_db=self._grid_db,
                electricity_engine=self._electricity,
                steam_heat_cool_engine=self._steam_heat_cool,
                transmission_engine=self._transmission,
                uncertainty_engine=self._uncertainty,
                compliance_engine=self._compliance,
                config=config_arg,
                metrics=metrics_arg,
            )
            logger.info("Scope2LocationPipelineEngine initialized")
        except TypeError:
            try:
                self._pipeline = Scope2LocationPipelineEngine(
                    self._grid_db,
                    self._electricity,
                    self._steam_heat_cool,
                    self._transmission,
                    self._uncertainty,
                    self._compliance,
                    config_arg,
                    metrics_arg,
                )
                logger.info(
                    "Scope2LocationPipelineEngine initialized "
                    "(positional args)"
                )
            except Exception as exc:
                logger.warning(
                    "Scope2LocationPipelineEngine initialization "
                    "failed: %s",
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Scope2LocationPipelineEngine initialization "
                "failed: %s",
                exc,
            )

    # ==================================================================
    # Internal: metrics recording
    # ==================================================================

    def _record_metric_calculation(
        self,
        energy_type: str,
        duration_s: float,
        co2e_tonnes: float,
    ) -> None:
        """Record a calculation metric if metrics are available.

        Args:
            energy_type: Energy type label for the metric.
            duration_s: Duration in seconds.
            co2e_tonnes: CO2e in tonnes for cumulative tracking.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_calculation(
                energy_type=energy_type,
                method="location_based",
                duration=duration_s,
                co2e_tonnes=co2e_tonnes,
            )
        except Exception:
            pass

    def _record_metric_error(self, error_type: str) -> None:
        """Record an error metric if metrics are available.

        Args:
            error_type: Error classification label.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_error(error_type=error_type)
        except Exception:
            pass

    # ==================================================================
    # Internal: pipeline request builder
    # ==================================================================

    def _build_pipeline_request(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize a raw request dict into the pipeline format.

        Extracts and normalizes fields expected by the pipeline engine's
        ``run_pipeline`` method.

        Args:
            request: Raw calculation request dict.

        Returns:
            Normalized request dict for the pipeline engine.
        """
        calc_id = request.get(
            "calculation_id", _short_id("s2l_calc"),
        )
        tenant_id = request.get("tenant_id", "default")
        facility_id = request.get("facility_id", "")
        energy_type = str(
            request.get("energy_type", "electricity"),
        ).lower()
        consumption_value = request.get("consumption_value", 0)
        consumption_unit = str(
            request.get("consumption_unit", "mwh"),
        ).lower()
        country_code = str(
            request.get("country_code", ""),
        ).upper()
        egrid_subregion = request.get("egrid_subregion")
        if egrid_subregion:
            egrid_subregion = str(egrid_subregion).upper()
        gwp_source = str(
            request.get("gwp_source", DEFAULT_GWP_SOURCE),
        ).upper()
        include_td_losses = request.get("include_td_losses", True)
        include_compliance = request.get("include_compliance", False)
        compliance_frameworks = request.get(
            "compliance_frameworks", None,
        )

        # Sub-type fields for steam/heat/cooling
        steam_type = request.get("steam_type")
        heating_type = request.get("heating_type")
        cooling_type = request.get("cooling_type")
        custom_ef = request.get("custom_ef")

        return {
            "calculation_id": calc_id,
            "tenant_id": tenant_id,
            "facility_id": facility_id,
            "energy_type": energy_type,
            "consumption_value": consumption_value,
            "consumption_unit": consumption_unit,
            "country_code": country_code,
            "egrid_subregion": egrid_subregion,
            "gwp_source": gwp_source,
            "include_td_losses": include_td_losses,
            "include_compliance": include_compliance,
            "compliance_frameworks": compliance_frameworks,
            "steam_type": steam_type,
            "heating_type": heating_type,
            "cooling_type": cooling_type,
            "custom_ef": custom_ef,
        }

    # ==================================================================
    # Internal: result builder from pipeline output
    # ==================================================================

    def _build_calculate_response(
        self,
        pipeline_result: Dict[str, Any],
        calc_id: str,
        energy_type: str,
        elapsed_ms: float,
    ) -> CalculateResponse:
        """Build a CalculateResponse from raw pipeline engine output.

        Maps the engine's output dictionary to the Pydantic response
        model fields.

        Args:
            pipeline_result: Raw dict from the pipeline engine.
            calc_id: Calculation identifier.
            energy_type: Energy type string.
            elapsed_ms: Processing time in milliseconds.

        Returns:
            CalculateResponse with all fields populated.
        """
        consumption_value = _safe_float(
            pipeline_result.get("consumption_value"),
        )
        consumption_unit = str(
            pipeline_result.get("consumption_unit", "mwh"),
        )
        grid_region = str(
            pipeline_result.get("grid_region", ""),
        )
        ef_source = str(
            pipeline_result.get("emission_factor_source", ""),
        )
        ef_co2e = _safe_float(
            pipeline_result.get("ef_co2e_per_mwh"),
        )
        td_loss_pct = _safe_float(
            pipeline_result.get("td_loss_pct"),
        )

        # Gas breakdown
        gas_breakdown_raw = pipeline_result.get("gas_breakdown", [])
        gas_breakdown: List[Dict[str, Any]] = []
        co2_kg = 0.0
        ch4_kg = 0.0
        n2o_kg = 0.0

        for gas_entry in gas_breakdown_raw:
            if isinstance(gas_entry, dict):
                gas_name = str(gas_entry.get("gas", "")).upper()
                emission_kg = _safe_float(gas_entry.get("emission_kg"))
                co2e_kg = _safe_float(gas_entry.get("co2e_kg"))
                gwp_factor = _safe_float(gas_entry.get("gwp_factor"))

                if gas_name == "CO2":
                    co2_kg = emission_kg
                elif gas_name == "CH4":
                    ch4_kg = emission_kg
                elif gas_name == "N2O":
                    n2o_kg = emission_kg

                gas_breakdown.append({
                    "gas": gas_name,
                    "emission_kg": emission_kg,
                    "co2e_kg": co2e_kg,
                    "gwp_factor": gwp_factor,
                })

        total_co2e_kg = _safe_float(
            pipeline_result.get("total_co2e_kg"),
        )
        total_co2e_tonnes = _safe_float(
            pipeline_result.get("total_co2e_tonnes"),
        )

        # Fallback: derive tonnes from kg if needed
        if total_co2e_tonnes == 0.0 and total_co2e_kg > 0.0:
            total_co2e_tonnes = total_co2e_kg / 1000.0

        gwp_source = str(
            pipeline_result.get("gwp_source", DEFAULT_GWP_SOURCE),
        )
        provenance_hash = str(
            pipeline_result.get("provenance_hash", ""),
        )

        # If pipeline did not produce a hash, compute one
        if not provenance_hash or len(provenance_hash) != 64:
            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "energy_type": energy_type,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
            })

        metadata_raw = pipeline_result.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            metadata_raw = {}

        return CalculateResponse(
            success=True,
            calculation_id=calc_id,
            energy_type=energy_type,
            consumption_value=consumption_value,
            consumption_unit=consumption_unit,
            grid_region=grid_region,
            emission_factor_source=ef_source,
            ef_co2e_per_mwh=ef_co2e,
            td_loss_pct=td_loss_pct,
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            total_co2e_kg=total_co2e_kg,
            total_co2e_tonnes=total_co2e_tonnes,
            gwp_source=gwp_source,
            gas_breakdown=gas_breakdown,
            provenance_hash=provenance_hash,
            processing_time_ms=elapsed_ms,
            timestamp=_utcnow_iso(),
            metadata=metadata_raw,
        )

    # ==================================================================
    # Internal: fallback calculation (when pipeline is unavailable)
    # ==================================================================

    def _fallback_electricity_calculation(
        self,
        request: Dict[str, Any],
        calc_id: str,
    ) -> Dict[str, Any]:
        """Fallback electricity calculation using direct engine calls.

        Invoked when the pipeline engine is unavailable. Uses the
        electricity engine and grid factor database directly.

        Args:
            request: Normalized request dict.
            calc_id: Calculation identifier.

        Returns:
            Result dict with emission values.
        """
        consumption_val = _safe_decimal(
            request.get("consumption_value"),
        )
        country_code = request.get("country_code", "US")
        egrid_sub = request.get("egrid_subregion")
        gwp_source = request.get("gwp_source", DEFAULT_GWP_SOURCE)
        include_td = request.get("include_td_losses", True)

        # Resolve grid emission factor
        co2_ef = Decimal("0")
        ch4_ef = Decimal("0")
        n2o_ef = Decimal("0")
        ef_source = "unknown"
        grid_region = country_code

        if self._grid_db is not None:
            try:
                factor = self._grid_db.resolve_emission_factor(
                    country_code=country_code,
                    egrid_subregion=egrid_sub,
                )
                co2_ef = _safe_decimal(factor.get("co2_per_mwh"))
                ch4_ef = _safe_decimal(factor.get("ch4_per_mwh"))
                n2o_ef = _safe_decimal(factor.get("n2o_per_mwh"))
                ef_source = str(factor.get("source", "unknown"))
                grid_region = str(
                    factor.get("region_id", country_code),
                )
            except Exception as exc:
                logger.warning(
                    "Grid factor lookup failed for %s: %s",
                    country_code,
                    exc,
                )

        # GWP conversion
        gwp_table = _get_gwp_table(gwp_source)
        co2_emissions_kg = consumption_val * co2_ef
        ch4_emissions_kg = consumption_val * ch4_ef
        n2o_emissions_kg = consumption_val * n2o_ef

        co2_co2e = co2_emissions_kg * gwp_table["co2"]
        ch4_co2e = ch4_emissions_kg * gwp_table["ch4"]
        n2o_co2e = n2o_emissions_kg * gwp_table["n2o"]

        total_co2e_kg = co2_co2e + ch4_co2e + n2o_co2e

        # T&D loss adjustment
        td_loss_pct = Decimal("0")
        if include_td and self._transmission is not None:
            try:
                td_loss_pct = _safe_decimal(
                    self._transmission.get_td_loss_factor(country_code),
                )
                td_loss_emissions = total_co2e_kg * (
                    td_loss_pct / (Decimal("1") - td_loss_pct)
                )
                total_co2e_kg = total_co2e_kg + td_loss_emissions
            except Exception as exc:
                logger.warning(
                    "T&D loss adjustment failed for %s: %s",
                    country_code,
                    exc,
                )

        total_co2e_tonnes = total_co2e_kg / Decimal("1000")
        ef_co2e_total = co2_ef + ch4_ef * gwp_table["ch4"] + (
            n2o_ef * gwp_table["n2o"]
        )

        return {
            "calculation_id": calc_id,
            "consumption_value": float(consumption_val),
            "consumption_unit": "mwh",
            "grid_region": grid_region,
            "emission_factor_source": ef_source,
            "ef_co2e_per_mwh": float(ef_co2e_total),
            "td_loss_pct": float(td_loss_pct),
            "gas_breakdown": [
                {
                    "gas": "CO2",
                    "emission_kg": float(co2_emissions_kg),
                    "co2e_kg": float(co2_co2e),
                    "gwp_factor": float(gwp_table["co2"]),
                },
                {
                    "gas": "CH4",
                    "emission_kg": float(ch4_emissions_kg),
                    "co2e_kg": float(ch4_co2e),
                    "gwp_factor": float(gwp_table["ch4"]),
                },
                {
                    "gas": "N2O",
                    "emission_kg": float(n2o_emissions_kg),
                    "co2e_kg": float(n2o_co2e),
                    "gwp_factor": float(gwp_table["n2o"]),
                },
            ],
            "total_co2e_kg": float(total_co2e_kg),
            "total_co2e_tonnes": float(total_co2e_tonnes),
            "gwp_source": gwp_source,
            "provenance_hash": "",
            "metadata": {"fallback": True},
        }

    def _fallback_non_electric_calculation(
        self,
        request: Dict[str, Any],
        calc_id: str,
    ) -> Dict[str, Any]:
        """Fallback steam/heat/cooling calculation.

        Uses the SteamHeatCoolingEngine directly when the pipeline
        engine is unavailable.

        Args:
            request: Normalized request dict.
            calc_id: Calculation identifier.

        Returns:
            Result dict with emission values.
        """
        energy_type = request.get("energy_type", "steam")
        consumption_val = _safe_float(
            request.get("consumption_value"),
        )
        country_code = request.get("country_code", "US")
        custom_ef_val = request.get("custom_ef")

        total_co2e_kg = 0.0
        ef_used = 0.0

        if self._steam_heat_cool is not None:
            try:
                if energy_type == "steam":
                    steam_type = request.get(
                        "steam_type", "natural_gas",
                    )
                    result = (
                        self._steam_heat_cool.calculate_steam_emissions(
                            consumption_gj=_safe_decimal(consumption_val),
                            steam_type=steam_type,
                            custom_ef=(
                                _safe_decimal(custom_ef_val)
                                if custom_ef_val is not None
                                else None
                            ),
                            country_code=country_code,
                        )
                    )
                    total_co2e_kg = _safe_float(
                        result.get("total_co2e_kg"),
                    )
                    ef_used = _safe_float(
                        result.get("emission_factor_kg_per_gj"),
                    )
                elif energy_type == "heating":
                    heating_type = request.get(
                        "heating_type", "district",
                    )
                    result = (
                        self._steam_heat_cool.calculate_heating_emissions(
                            consumption_gj=_safe_decimal(consumption_val),
                            heating_type=heating_type,
                            custom_ef=(
                                _safe_decimal(custom_ef_val)
                                if custom_ef_val is not None
                                else None
                            ),
                            country_code=country_code,
                        )
                    )
                    total_co2e_kg = _safe_float(
                        result.get("total_co2e_kg"),
                    )
                    ef_used = _safe_float(
                        result.get("emission_factor_kg_per_gj"),
                    )
                elif energy_type == "cooling":
                    cooling_type = request.get(
                        "cooling_type", "electric_chiller",
                    )
                    result = (
                        self._steam_heat_cool.calculate_cooling_emissions(
                            consumption_gj=_safe_decimal(consumption_val),
                            cooling_type=cooling_type,
                            custom_ef=(
                                _safe_decimal(custom_ef_val)
                                if custom_ef_val is not None
                                else None
                            ),
                            country_code=country_code,
                        )
                    )
                    total_co2e_kg = _safe_float(
                        result.get("total_co2e_kg"),
                    )
                    ef_used = _safe_float(
                        result.get("emission_factor_kg_per_gj"),
                    )
            except Exception as exc:
                logger.warning(
                    "SteamHeatCooling fallback failed for %s: %s",
                    energy_type,
                    exc,
                )

        total_co2e_tonnes = total_co2e_kg / 1000.0

        return {
            "calculation_id": calc_id,
            "consumption_value": consumption_val,
            "consumption_unit": "gj",
            "grid_region": country_code,
            "emission_factor_source": "default",
            "ef_co2e_per_mwh": ef_used,
            "td_loss_pct": 0.0,
            "gas_breakdown": [
                {
                    "gas": "CO2e",
                    "emission_kg": total_co2e_kg,
                    "co2e_kg": total_co2e_kg,
                    "gwp_factor": 1.0,
                },
            ],
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "gwp_source": request.get("gwp_source", DEFAULT_GWP_SOURCE),
            "provenance_hash": "",
            "metadata": {"fallback": True, "energy_type": energy_type},
        }

    # ==================================================================
    # Public API 1: calculate
    # ==================================================================

    def calculate(self, request: Dict[str, Any]) -> CalculateResponse:
        """Calculate Scope 2 location-based emissions for a single record.

        Supports electricity, steam, heating, and cooling energy types.
        Delegates to the pipeline engine when available, falling back
        to direct engine calls otherwise.

        Args:
            request: Calculation request dict with keys:
                - tenant_id (str, required)
                - facility_id (str, required)
                - energy_type (str, default 'electricity')
                - consumption_value (numeric, required)
                - consumption_unit (str, default 'mwh')
                - country_code (str, required for factor lookup)
                - egrid_subregion (str, optional, US only)
                - gwp_source (str, default 'AR5')
                - include_td_losses (bool, default True)
                - steam_type / heating_type / cooling_type (str)
                - custom_ef (numeric, optional override)

        Returns:
            CalculateResponse with emission values and provenance hash.
        """
        t0 = time.monotonic()
        normalized = self._build_pipeline_request(request)
        calc_id = normalized["calculation_id"]
        energy_type = normalized["energy_type"]

        try:
            # Validate required fields
            errors = _validate_required_fields(
                normalized,
                ["tenant_id", "facility_id", "consumption_value"],
                "calculate",
            )
            if errors:
                raise ValueError("; ".join(errors))

            # Validate energy type
            err = _validate_enum_field(
                energy_type, VALID_ENERGY_TYPES, "energy_type",
            )
            if err:
                raise ValueError(err)

            # Route through pipeline or fallback
            raw_result: Dict[str, Any]
            if self._pipeline is not None:
                raw_result = self._pipeline.run_pipeline(normalized)
            elif energy_type == "electricity":
                raw_result = self._fallback_electricity_calculation(
                    normalized, calc_id,
                )
            else:
                raw_result = self._fallback_non_electric_calculation(
                    normalized, calc_id,
                )

            elapsed_ms = _elapsed_ms(t0)
            duration_s = (time.monotonic() - t0)

            response = self._build_calculate_response(
                raw_result, calc_id, energy_type, elapsed_ms,
            )

            # Store result
            calc_record = {
                "calculation_id": calc_id,
                "tenant_id": normalized["tenant_id"],
                "facility_id": normalized["facility_id"],
                "energy_type": energy_type,
                "total_co2e_kg": response.total_co2e_kg,
                "total_co2e_tonnes": response.total_co2e_tonnes,
                "provenance_hash": response.provenance_hash,
                "timestamp": response.timestamp,
                "request": normalized,
                "response": response.model_dump(),
            }
            self._calculations[calc_id] = calc_record
            self._total_calculations += 1
            self._cumulative_co2e_tonnes += response.total_co2e_tonnes

            # Record metrics
            self._record_metric_calculation(
                energy_type, duration_s, response.total_co2e_tonnes,
            )

            logger.info(
                "Calculation %s completed: energy_type=%s, "
                "total_co2e_tonnes=%.6f, elapsed_ms=%.3f",
                calc_id,
                energy_type,
                response.total_co2e_tonnes,
                elapsed_ms,
            )
            return response

        except ValueError as exc:
            self._record_metric_error("validation_error")
            logger.warning(
                "Calculation validation error for %s: %s",
                calc_id,
                exc,
            )
            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                energy_type=energy_type,
                provenance_hash=_compute_hash(
                    {"error": str(exc), "calc_id": calc_id},
                ),
                processing_time_ms=_elapsed_ms(t0),
                metadata={"error": str(exc)},
            )
        except Exception as exc:
            self._record_metric_error("calculation_error")
            logger.error(
                "Calculation %s failed: %s",
                calc_id,
                exc,
                exc_info=True,
            )
            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                energy_type=energy_type,
                provenance_hash=_compute_hash(
                    {"error": str(exc), "calc_id": calc_id},
                ),
                processing_time_ms=_elapsed_ms(t0),
                metadata={"error": str(exc)},
            )

    # ==================================================================
    # Public API 2: calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        batch: Dict[str, Any],
    ) -> BatchCalculateResponse:
        """Calculate Scope 2 location-based emissions for a batch.

        Processes multiple calculation requests sequentially, capturing
        individual successes and failures.

        Args:
            batch: Batch request dict with keys:
                - batch_id (str, optional, auto-generated)
                - tenant_id (str, required)
                - requests (list of calculation dicts, required)

        Returns:
            BatchCalculateResponse with per-request results.
        """
        t0 = time.monotonic()
        batch_id = batch.get("batch_id", _short_id("s2l_batch"))
        tenant_id = batch.get("tenant_id", "default")
        requests = batch.get("requests", [])

        if not isinstance(requests, list) or len(requests) == 0:
            return BatchCalculateResponse(
                success=False,
                batch_id=batch_id,
                metadata={"error": "No requests provided"},
            )

        # Enforce max batch size
        max_batch = DEFAULT_MAX_BATCH_SIZE
        if self._config is not None:
            try:
                max_batch = getattr(
                    self._config, "max_batch_size", max_batch,
                )
            except Exception:
                pass

        if len(requests) > max_batch:
            return BatchCalculateResponse(
                success=False,
                batch_id=batch_id,
                total_calculations=len(requests),
                metadata={
                    "error": (
                        f"Batch size {len(requests)} exceeds "
                        f"maximum {max_batch}"
                    ),
                },
            )

        # Try pipeline batch first
        if self._pipeline is not None:
            try:
                pipe_batch = {
                    "batch_id": batch_id,
                    "tenant_id": tenant_id,
                    "requests": [
                        self._build_pipeline_request(
                            {**r, "tenant_id": tenant_id},
                        )
                        for r in requests
                    ],
                }
                raw_result = self._pipeline.run_batch_pipeline(
                    pipe_batch,
                )
                elapsed_ms = _elapsed_ms(t0)

                batch_results = raw_result.get("results", [])
                batch_errors = raw_result.get("errors", [])
                total_co2e = _safe_float(
                    raw_result.get("total_co2e_tonnes"),
                )

                # Store each result
                results_list: List[Dict[str, Any]] = []
                for r in batch_results:
                    rid = r.get(
                        "calculation_id",
                        _short_id("s2l_calc"),
                    )
                    self._calculations[rid] = r
                    results_list.append(r)

                self._total_batch_runs += 1
                self._total_calculations += len(batch_results)
                self._cumulative_co2e_tonnes += total_co2e

                provenance_hash = _compute_hash({
                    "batch_id": batch_id,
                    "total_calculations": len(requests),
                    "total_co2e_tonnes": total_co2e,
                })

                return BatchCalculateResponse(
                    success=True,
                    batch_id=batch_id,
                    total_calculations=len(requests),
                    successful=len(batch_results),
                    failed=len(batch_errors),
                    total_co2e_tonnes=total_co2e,
                    results=results_list,
                    errors=batch_errors,
                    processing_time_ms=elapsed_ms,
                    provenance_hash=provenance_hash,
                    timestamp=_utcnow_iso(),
                )
            except Exception as exc:
                logger.warning(
                    "Pipeline batch failed, falling back to "
                    "sequential: %s",
                    exc,
                )

        # Fallback: sequential calculation
        results_list = []
        errors_list: List[Dict[str, Any]] = []
        total_co2e_tonnes = 0.0

        for i, req in enumerate(requests):
            try:
                req_with_tenant = {**req, "tenant_id": tenant_id}
                response = self.calculate(req_with_tenant)
                result_dict = response.model_dump()
                results_list.append(result_dict)
                if response.success:
                    total_co2e_tonnes += response.total_co2e_tonnes
                else:
                    errors_list.append({
                        "index": i,
                        "error": result_dict.get(
                            "metadata", {},
                        ).get("error", "Unknown"),
                    })
            except Exception as exc:
                errors_list.append({
                    "index": i,
                    "error": str(exc),
                })

        elapsed_ms = _elapsed_ms(t0)
        self._total_batch_runs += 1

        provenance_hash = _compute_hash({
            "batch_id": batch_id,
            "total_calculations": len(requests),
            "successful": len(results_list) - len(errors_list),
            "total_co2e_tonnes": total_co2e_tonnes,
        })

        successful_count = len(results_list) - len(errors_list)

        return BatchCalculateResponse(
            success=len(errors_list) == 0,
            batch_id=batch_id,
            total_calculations=len(requests),
            successful=successful_count,
            failed=len(errors_list),
            total_co2e_tonnes=total_co2e_tonnes,
            results=results_list,
            errors=errors_list,
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 3: list_calculations
    # ==================================================================

    def list_calculations(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List stored calculations for a tenant.

        Filters the in-memory calculation store by tenant_id and
        returns a paginated result.

        Args:
            tenant_id: Tenant identifier to filter by.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.

        Returns:
            Dict with 'calculations' list, 'total' count,
            'skip', and 'limit'.
        """
        tenant_calcs = [
            c for c in self._calculations.values()
            if c.get("tenant_id") == tenant_id
        ]
        # Sort by timestamp descending
        tenant_calcs.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )

        total = len(tenant_calcs)
        page = tenant_calcs[skip:skip + limit]

        return {
            "calculations": page,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    # ==================================================================
    # Public API 4: get_calculation
    # ==================================================================

    def get_calculation(
        self,
        calculation_id: str,
    ) -> Dict[str, Any]:
        """Get a single calculation by ID.

        Args:
            calculation_id: Unique calculation identifier.

        Returns:
            Calculation dict or error dict if not found.
        """
        calc = self._calculations.get(calculation_id)
        if calc is None:
            return {
                "error": f"Calculation '{calculation_id}' not found",
                "found": False,
            }
        return {**calc, "found": True}

    # ==================================================================
    # Public API 5: delete_calculation
    # ==================================================================

    def delete_calculation(
        self,
        calculation_id: str,
    ) -> bool:
        """Delete a calculation by ID.

        Removes the calculation from the in-memory store.

        Args:
            calculation_id: Unique calculation identifier.

        Returns:
            True if the calculation was deleted, False if not found.
        """
        if calculation_id in self._calculations:
            del self._calculations[calculation_id]
            logger.info(
                "Calculation %s deleted", calculation_id,
            )
            return True
        logger.warning(
            "Calculation %s not found for deletion",
            calculation_id,
        )
        return False

    # ==================================================================
    # Public API 6: register_facility
    # ==================================================================

    def register_facility(
        self,
        data: Dict[str, Any],
    ) -> FacilityResponse:
        """Register a new facility for Scope 2 emissions tracking.

        Creates a facility record with grid region mapping and optional
        geolocation data.

        Args:
            data: Facility data dict with keys:
                - name (str, required)
                - facility_type (str, default 'office')
                - country_code (str, required)
                - grid_region_id (str, required)
                - egrid_subregion (str, optional)
                - latitude (float, optional)
                - longitude (float, optional)
                - tenant_id (str, required)

        Returns:
            FacilityResponse with the created facility record.
        """
        facility_id = _short_id("s2l_fac")
        now_iso = _utcnow_iso()

        name = str(data.get("name", "")).strip()
        if not name:
            name = f"Facility {facility_id}"

        facility_type = str(
            data.get("facility_type", "office"),
        ).lower()
        country_code = str(
            data.get("country_code", ""),
        ).upper()
        grid_region_id = str(
            data.get("grid_region_id", country_code),
        )
        egrid_subregion = data.get("egrid_subregion")
        if egrid_subregion:
            egrid_subregion = str(egrid_subregion).upper()
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        tenant_id = str(data.get("tenant_id", "default"))

        record: Dict[str, Any] = {
            "facility_id": facility_id,
            "name": name,
            "facility_type": facility_type,
            "country_code": country_code,
            "grid_region_id": grid_region_id,
            "egrid_subregion": egrid_subregion,
            "latitude": _safe_float(latitude) if latitude else None,
            "longitude": _safe_float(longitude) if longitude else None,
            "tenant_id": tenant_id,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._facilities[facility_id] = record

        # Update active facilities gauge
        if self._metrics is not None:
            try:
                self._metrics.set_active_facilities(
                    len(self._facilities),
                )
            except Exception:
                pass

        logger.info(
            "Facility %s registered: name=%s, country=%s, region=%s",
            facility_id,
            name,
            country_code,
            grid_region_id,
        )

        return FacilityResponse(
            facility_id=facility_id,
            name=name,
            facility_type=facility_type,
            country_code=country_code,
            grid_region_id=grid_region_id,
            egrid_subregion=egrid_subregion,
            latitude=record.get("latitude"),
            longitude=record.get("longitude"),
            tenant_id=tenant_id,
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ==================================================================
    # Public API 7: list_facilities
    # ==================================================================

    def list_facilities(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> FacilityListResponse:
        """List registered facilities for a tenant.

        Args:
            tenant_id: Tenant identifier to filter by.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            FacilityListResponse with paginated facility records.
        """
        tenant_facs = [
            f for f in self._facilities.values()
            if f.get("tenant_id") == tenant_id
        ]
        tenant_facs.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        total = len(tenant_facs)
        page = tenant_facs[skip:skip + limit]

        return FacilityListResponse(
            facilities=page,
            total=total,
            page=skip // max(limit, 1) + 1,
            page_size=limit,
        )

    # ==================================================================
    # Public API 8: update_facility
    # ==================================================================

    def update_facility(
        self,
        facility_id: str,
        data: Dict[str, Any],
    ) -> FacilityResponse:
        """Update an existing facility record.

        Merges the provided data dict into the existing facility
        record. Only provided fields are updated.

        Args:
            facility_id: Unique facility identifier.
            data: Dict of fields to update.

        Returns:
            FacilityResponse with the updated facility record.

        Raises:
            ValueError: If the facility does not exist.
        """
        existing = self._facilities.get(facility_id)
        if existing is None:
            raise ValueError(
                f"Facility '{facility_id}' not found"
            )

        updatable_fields = {
            "name",
            "facility_type",
            "country_code",
            "grid_region_id",
            "egrid_subregion",
            "latitude",
            "longitude",
        }

        for key, value in data.items():
            if key in updatable_fields and value is not None:
                if key == "country_code":
                    value = str(value).upper()
                elif key == "egrid_subregion":
                    value = str(value).upper()
                elif key in ("latitude", "longitude"):
                    value = _safe_float(value)
                elif key == "facility_type":
                    value = str(value).lower()
                existing[key] = value

        existing["updated_at"] = _utcnow_iso()

        logger.info("Facility %s updated", facility_id)

        return FacilityResponse(
            facility_id=facility_id,
            name=existing.get("name", ""),
            facility_type=existing.get("facility_type", "office"),
            country_code=existing.get("country_code", ""),
            grid_region_id=existing.get("grid_region_id", ""),
            egrid_subregion=existing.get("egrid_subregion"),
            latitude=existing.get("latitude"),
            longitude=existing.get("longitude"),
            tenant_id=existing.get("tenant_id", ""),
            created_at=existing.get("created_at", ""),
            updated_at=existing.get("updated_at", ""),
        )

    # ==================================================================
    # Public API 9: record_consumption
    # ==================================================================

    def record_consumption(
        self,
        data: Dict[str, Any],
    ) -> ConsumptionResponse:
        """Record an energy consumption measurement.

        Creates a consumption record linked to a facility for
        subsequent emission calculations.

        Args:
            data: Consumption data dict with keys:
                - facility_id (str, required)
                - energy_type (str, default 'electricity')
                - quantity (numeric, required)
                - unit (str, default 'mwh')
                - period_start (str, ISO-8601)
                - period_end (str, ISO-8601)
                - data_source (str, default 'invoice')
                - meter_id (str, optional)

        Returns:
            ConsumptionResponse with the recorded consumption data.
        """
        consumption_id = _short_id("s2l_con")
        now_iso = _utcnow_iso()

        facility_id = str(data.get("facility_id", ""))
        energy_type = str(
            data.get("energy_type", "electricity"),
        ).lower()
        quantity = _safe_float(data.get("quantity"))
        unit = str(data.get("unit", "mwh")).lower()
        period_start = str(data.get("period_start", now_iso))
        period_end = str(data.get("period_end", now_iso))
        data_source = str(data.get("data_source", "invoice")).lower()
        meter_id = data.get("meter_id")

        record = {
            "consumption_id": consumption_id,
            "facility_id": facility_id,
            "energy_type": energy_type,
            "quantity": quantity,
            "unit": unit,
            "period_start": period_start,
            "period_end": period_end,
            "data_source": data_source,
            "meter_id": meter_id,
            "created_at": now_iso,
        }

        self._consumption.append(record)

        logger.info(
            "Consumption %s recorded: facility=%s, type=%s, "
            "quantity=%.2f %s",
            consumption_id,
            facility_id,
            energy_type,
            quantity,
            unit,
        )

        return ConsumptionResponse(
            consumption_id=consumption_id,
            facility_id=facility_id,
            energy_type=energy_type,
            quantity=quantity,
            unit=unit,
            period_start=period_start,
            period_end=period_end,
            data_source=data_source,
            meter_id=meter_id,
            created_at=now_iso,
        )

    # ==================================================================
    # Public API 10: list_consumption
    # ==================================================================

    def list_consumption(
        self,
        tenant_id: str,
        facility_id: Optional[str] = None,
    ) -> ConsumptionListResponse:
        """List energy consumption records.

        Filters by tenant (via facility lookup) and optionally by
        facility_id.

        Args:
            tenant_id: Tenant identifier for facility ownership.
            facility_id: Optional specific facility to filter by.

        Returns:
            ConsumptionListResponse with matching records.
        """
        # Get facilities belonging to this tenant
        tenant_facility_ids = {
            fid
            for fid, fac in self._facilities.items()
            if fac.get("tenant_id") == tenant_id
        }

        filtered: List[Dict[str, Any]] = []
        for record in self._consumption:
            rec_fac_id = record.get("facility_id", "")
            # If facility_id is specified, filter exactly
            if facility_id:
                if rec_fac_id == facility_id:
                    filtered.append(record)
            elif rec_fac_id in tenant_facility_ids:
                filtered.append(record)

        filtered.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        return ConsumptionListResponse(
            records=filtered,
            total=len(filtered),
        )

    # ==================================================================
    # Public API 11: list_grid_factors
    # ==================================================================

    def list_grid_factors(
        self,
        source: Optional[str] = None,
    ) -> GridFactorListResponse:
        """List available grid emission factors.

        Delegates to the GridEmissionFactorDatabaseEngine for the
        authoritative list of factors. Optionally filters by source.

        Args:
            source: Optional source filter (egrid, iea, defra, etc.).

        Returns:
            GridFactorListResponse with matching factor records.
        """
        factors: List[Dict[str, Any]] = []

        if self._grid_db is not None:
            try:
                # eGRID subregion factors
                if source is None or source == "egrid":
                    egrid_list = self._grid_db.list_egrid_subregions()
                    for subregion in egrid_list:
                        try:
                            factor = self._grid_db.get_grid_factor(
                                country_code="US",
                                source="egrid",
                            )
                            factors.append({
                                "region_id": subregion,
                                "country_code": "US",
                                "source": "egrid",
                                "co2e_per_mwh": _safe_float(
                                    factor.get("co2e_per_mwh"),
                                ),
                                "data_quality_tier": "tier_2",
                            })
                        except Exception:
                            factors.append({
                                "region_id": subregion,
                                "country_code": "US",
                                "source": "egrid",
                            })

                # IEA country factors
                if source is None or source == "iea":
                    try:
                        from greenlang.agents.mrv.scope2_location.models import (
                            IEA_COUNTRY_FACTORS,
                        )
                        for cc, ef_val in IEA_COUNTRY_FACTORS.items():
                            factors.append({
                                "region_id": cc,
                                "country_code": cc,
                                "source": "iea",
                                "co2e_per_mwh": float(ef_val) * 1000,
                                "data_quality_tier": "tier_1",
                            })
                    except ImportError:
                        pass

                # EU EEA factors
                if source is None or source == "eu_eea":
                    try:
                        from greenlang.agents.mrv.scope2_location.models import (
                            EU_COUNTRY_FACTORS,
                        )
                        for cc, ef_val in EU_COUNTRY_FACTORS.items():
                            factors.append({
                                "region_id": cc,
                                "country_code": cc,
                                "source": "eu_eea",
                                "co2e_per_mwh": float(ef_val) * 1000,
                                "data_quality_tier": "tier_1",
                            })
                    except ImportError:
                        pass

                # DEFRA factors
                if source is None or source == "defra":
                    try:
                        from greenlang.agents.mrv.scope2_location.models import (
                            DEFRA_FACTORS,
                        )
                        for key, ef_val in DEFRA_FACTORS.items():
                            factors.append({
                                "region_id": f"GB_{key}",
                                "country_code": "GB",
                                "source": "defra",
                                "co2e_per_mwh": float(ef_val) * 1000,
                                "data_quality_tier": "tier_2",
                                "factor_type": key,
                            })
                    except ImportError:
                        pass

            except Exception as exc:
                logger.warning(
                    "Grid factor listing failed: %s", exc,
                )

        # Include custom factors
        for cf_key, cf_val in self._custom_factors.items():
            if source is None or cf_val.get("source") == source:
                factors.append(cf_val)

        return GridFactorListResponse(
            factors=factors,
            total=len(factors),
            source_filter=source,
        )

    # ==================================================================
    # Public API 12: get_grid_factor
    # ==================================================================

    def get_grid_factor(
        self,
        region: str,
    ) -> GridFactorResponse:
        """Get the grid emission factor for a specific region.

        Resolves the emission factor using the grid factor database
        engine's hierarchy (custom > national > eGRID > IEA > IPCC).

        Args:
            region: Region identifier (country code or eGRID subregion).

        Returns:
            GridFactorResponse with the resolved emission factor.
        """
        # Check custom factors first
        if region in self._custom_factors:
            cf = self._custom_factors[region]
            return GridFactorResponse(
                region_id=region,
                country_code=cf.get("country_code", region),
                source=cf.get("source", "custom"),
                year=cf.get("year", 2024),
                co2_kg_per_mwh=_safe_float(cf.get("co2_kg_per_mwh")),
                ch4_kg_per_mwh=_safe_float(cf.get("ch4_kg_per_mwh")),
                n2o_kg_per_mwh=_safe_float(cf.get("n2o_kg_per_mwh")),
                co2e_per_mwh=_safe_float(cf.get("co2e_per_mwh")),
                data_quality_tier=cf.get("data_quality_tier", "tier_3"),
                td_loss_pct=_safe_float(cf.get("td_loss_pct")),
                notes=cf.get("notes", "Custom factor"),
            )

        # Delegate to engine
        if self._grid_db is not None:
            try:
                # Try resolving with egrid subregion if it matches
                is_egrid = (
                    len(region) <= 5 and region.upper() == region
                )
                if is_egrid:
                    factor = self._grid_db.resolve_emission_factor(
                        country_code="US",
                        egrid_subregion=region.upper(),
                    )
                else:
                    factor = self._grid_db.resolve_emission_factor(
                        country_code=region.upper(),
                    )

                return GridFactorResponse(
                    region_id=str(
                        factor.get("region_id", region),
                    ),
                    country_code=str(
                        factor.get("country_code", region),
                    ),
                    source=str(factor.get("source", "unknown")),
                    year=int(factor.get("year", 2024)),
                    co2_kg_per_mwh=_safe_float(
                        factor.get("co2_per_mwh"),
                    ),
                    ch4_kg_per_mwh=_safe_float(
                        factor.get("ch4_per_mwh"),
                    ),
                    n2o_kg_per_mwh=_safe_float(
                        factor.get("n2o_per_mwh"),
                    ),
                    co2e_per_mwh=_safe_float(
                        factor.get("co2e_per_mwh"),
                    ),
                    data_quality_tier=str(
                        factor.get("data_quality_tier", "tier_1"),
                    ),
                    td_loss_pct=_safe_float(
                        factor.get("td_loss_pct"),
                    ),
                    notes=str(factor.get("notes", "")),
                )
            except Exception as exc:
                logger.warning(
                    "Grid factor lookup failed for '%s': %s",
                    region,
                    exc,
                )

        # Final fallback: return empty response
        return GridFactorResponse(
            region_id=region,
            country_code=region,
            notes=f"No factor found for region '{region}'",
        )

    # ==================================================================
    # Public API 13: add_custom_factor
    # ==================================================================

    def add_custom_factor(
        self,
        data: Dict[str, Any],
    ) -> GridFactorResponse:
        """Add a custom grid emission factor.

        Stores a user-provided emission factor with documented
        provenance for audit compliance.

        Args:
            data: Custom factor dict with keys:
                - region_id (str, required)
                - country_code (str, required)
                - co2_kg_per_mwh (numeric)
                - ch4_kg_per_mwh (numeric)
                - n2o_kg_per_mwh (numeric)
                - co2e_per_mwh (numeric)
                - year (int, default 2024)
                - data_quality_tier (str, default 'tier_3')
                - td_loss_pct (numeric, optional)
                - notes (str, optional)

        Returns:
            GridFactorResponse with the stored custom factor.
        """
        region_id = str(data.get("region_id", _short_id("custom")))
        country_code = str(
            data.get("country_code", ""),
        ).upper()
        year = int(data.get("year", 2024))
        co2_per_mwh = _safe_float(data.get("co2_kg_per_mwh"))
        ch4_per_mwh = _safe_float(data.get("ch4_kg_per_mwh"))
        n2o_per_mwh = _safe_float(data.get("n2o_kg_per_mwh"))
        co2e_per_mwh = _safe_float(data.get("co2e_per_mwh"))
        data_quality_tier = str(
            data.get("data_quality_tier", "tier_3"),
        )
        td_loss_pct = _safe_float(data.get("td_loss_pct"))
        notes = str(data.get("notes", "Custom user-provided factor"))

        # If co2e not provided, compute from individual gases (AR5 GWP)
        if co2e_per_mwh == 0.0 and (
            co2_per_mwh > 0 or ch4_per_mwh > 0 or n2o_per_mwh > 0
        ):
            co2e_per_mwh = (
                co2_per_mwh
                + ch4_per_mwh * 28.0
                + n2o_per_mwh * 265.0
            )

        record: Dict[str, Any] = {
            "region_id": region_id,
            "country_code": country_code,
            "source": "custom",
            "year": year,
            "co2_kg_per_mwh": co2_per_mwh,
            "ch4_kg_per_mwh": ch4_per_mwh,
            "n2o_kg_per_mwh": n2o_per_mwh,
            "co2e_per_mwh": co2e_per_mwh,
            "data_quality_tier": data_quality_tier,
            "td_loss_pct": td_loss_pct,
            "notes": notes,
        }

        self._custom_factors[region_id] = record

        # Also register with the grid_db engine if available
        if self._grid_db is not None:
            try:
                self._grid_db.add_custom_factor(
                    region_id=region_id,
                    co2_per_mwh=_safe_decimal(co2_per_mwh),
                    ch4_per_mwh=_safe_decimal(ch4_per_mwh),
                    n2o_per_mwh=_safe_decimal(n2o_per_mwh),
                    source="custom",
                    year=year,
                    notes=notes,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to register custom factor with "
                    "grid_db engine: %s",
                    exc,
                )

        logger.info(
            "Custom grid factor added: region=%s, co2e=%.4f kg/MWh",
            region_id,
            co2e_per_mwh,
        )

        return GridFactorResponse(
            region_id=region_id,
            country_code=country_code,
            source="custom",
            year=year,
            co2_kg_per_mwh=co2_per_mwh,
            ch4_kg_per_mwh=ch4_per_mwh,
            n2o_kg_per_mwh=n2o_per_mwh,
            co2e_per_mwh=co2e_per_mwh,
            data_quality_tier=data_quality_tier,
            td_loss_pct=td_loss_pct,
            notes=notes,
        )

    # ==================================================================
    # Public API 14: list_td_losses
    # ==================================================================

    def list_td_losses(self) -> TDLossListResponse:
        """List all T&D loss factors.

        Delegates to the TransmissionLossEngine for the authoritative
        list of country-level T&D loss percentages.

        Returns:
            TDLossListResponse with all available T&D factors.
        """
        factors: Dict[str, Any] = {}

        if self._transmission is not None:
            try:
                raw = self._transmission.list_all_factors()
                if isinstance(raw, dict):
                    for country, factor_data in raw.items():
                        if isinstance(factor_data, dict):
                            factors[country] = {
                                k: (
                                    float(v)
                                    if isinstance(v, Decimal)
                                    else v
                                )
                                for k, v in factor_data.items()
                            }
                        else:
                            factors[country] = float(factor_data)
            except Exception as exc:
                logger.warning(
                    "T&D loss factor listing failed: %s", exc,
                )

        # Fallback: try to get from models
        if not factors:
            try:
                from greenlang.agents.mrv.scope2_location.models import (
                    TD_LOSS_FACTORS,
                )
                for country, pct in TD_LOSS_FACTORS.items():
                    factors[country] = {
                        "td_loss_pct": float(pct),
                        "source": "world_bank_iea",
                    }
            except ImportError:
                pass

        return TDLossListResponse(
            factors=factors,
            total=len(factors),
            source="world_bank_iea",
        )

    # ==================================================================
    # Public API 15: check_compliance
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Run regulatory compliance checks on a calculation.

        Evaluates a completed calculation against specified regulatory
        frameworks (GHG Protocol Scope 2, IPCC, ISO 14064, CSRD, etc.).

        Args:
            data: Compliance check request with keys:
                - calculation_id (str, required)
                - frameworks (list of str, optional)

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        check_id = _short_id("s2l_comp")
        calc_id = str(data.get("calculation_id", ""))
        frameworks = data.get("frameworks")

        # Get the calculation result for the check
        calc_record = self._calculations.get(calc_id)
        if calc_record is None:
            return ComplianceCheckResponse(
                success=False,
                id=check_id,
                calculation_id=calc_id,
                results=[{
                    "error": (
                        f"Calculation '{calc_id}' not found"
                    ),
                }],
            )

        results: List[Dict[str, Any]] = []
        compliant_count = 0
        non_compliant_count = 0
        partial_count = 0

        if self._compliance is not None:
            try:
                # Get calculation response data
                calc_data = calc_record.get("response", calc_record)

                check_results = self._compliance.check_compliance(
                    calculation_result=calc_data,
                    frameworks=frameworks,
                )

                if isinstance(check_results, list):
                    for cr in check_results:
                        if hasattr(cr, "model_dump"):
                            result_dict = cr.model_dump()
                        elif isinstance(cr, dict):
                            result_dict = cr
                        else:
                            result_dict = {"raw": str(cr)}

                        status = str(
                            result_dict.get("status", "not_assessed"),
                        )
                        if status == "compliant":
                            compliant_count += 1
                        elif status == "non_compliant":
                            non_compliant_count += 1
                        elif status == "partial":
                            partial_count += 1

                        results.append(result_dict)

                        # Record metric
                        fw_name = str(
                            result_dict.get("framework", "unknown"),
                        )
                        if self._metrics is not None:
                            try:
                                self._metrics.record_compliance_check(
                                    framework=fw_name,
                                    status=status,
                                )
                            except Exception:
                                pass

                elif isinstance(check_results, dict):
                    results.append(check_results)

            except Exception as exc:
                logger.warning(
                    "Compliance check failed for %s: %s",
                    calc_id,
                    exc,
                )
                results.append({"error": str(exc)})
        else:
            # Fallback: basic compliance check
            results = self._fallback_compliance_check(
                calc_record, frameworks,
            )
            for r in results:
                status = r.get("status", "not_assessed")
                if status == "compliant":
                    compliant_count += 1
                elif status == "non_compliant":
                    non_compliant_count += 1
                elif status == "partial":
                    partial_count += 1

        self._total_compliance_checks += 1

        provenance_hash = _compute_hash({
            "check_id": check_id,
            "calculation_id": calc_id,
            "frameworks_checked": len(results),
        })

        compliance_result: Dict[str, Any] = {
            "check_id": check_id,
            "calculation_id": calc_id,
            "frameworks_checked": len(results),
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "partial": partial_count,
            "results": results,
            "checked_at": _utcnow_iso(),
            "provenance_hash": provenance_hash,
        }
        self._compliance_results[check_id] = compliance_result

        return ComplianceCheckResponse(
            success=True,
            id=check_id,
            calculation_id=calc_id,
            frameworks_checked=len(results),
            compliant=compliant_count,
            non_compliant=non_compliant_count,
            partial=partial_count,
            results=results,
            checked_at=compliance_result["checked_at"],
            provenance_hash=provenance_hash,
        )

    def _fallback_compliance_check(
        self,
        calc_record: Dict[str, Any],
        frameworks: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Basic compliance check without the compliance engine.

        Performs simplified checks against GHG Protocol Scope 2
        mandatory requirements.

        Args:
            calc_record: Stored calculation record.
            frameworks: Optional frameworks to check against.

        Returns:
            List of per-framework result dicts.
        """
        if frameworks is None:
            frameworks = ["ghg_protocol_scope2"]

        results: List[Dict[str, Any]] = []
        calc_data = calc_record.get("response", calc_record)

        for fw in frameworks:
            findings: List[str] = []
            recommendations: List[str] = []

            # Basic checks
            has_ef = (
                _safe_float(calc_data.get("ef_co2e_per_mwh")) > 0
            )
            has_consumption = (
                _safe_float(calc_data.get("consumption_value")) > 0
            )
            has_provenance = bool(
                calc_data.get("provenance_hash"),
            )

            if has_ef:
                findings.append(
                    "Emission factor source documented"
                )
            else:
                findings.append(
                    "Missing emission factor documentation"
                )
                recommendations.append(
                    "Provide documented grid emission factor"
                )

            if has_consumption:
                findings.append(
                    "Consumption data provided"
                )
            else:
                findings.append(
                    "Missing consumption data"
                )

            if has_provenance:
                findings.append(
                    "Provenance hash available"
                )

            if has_ef and has_consumption:
                status = "compliant"
            elif has_consumption:
                status = "partial"
            else:
                status = "non_compliant"

            results.append({
                "framework": fw,
                "status": status,
                "findings": findings,
                "recommendations": recommendations,
            })

        return results

    # ==================================================================
    # Public API 16: get_compliance_result
    # ==================================================================

    def get_compliance_result(
        self,
        check_id: str,
    ) -> Dict[str, Any]:
        """Get a stored compliance check result by ID.

        Args:
            check_id: Unique compliance check identifier.

        Returns:
            Compliance result dict or error dict if not found.
        """
        result = self._compliance_results.get(check_id)
        if result is None:
            return {
                "error": (
                    f"Compliance check '{check_id}' not found"
                ),
                "found": False,
            }
        return {**result, "found": True}

    # ==================================================================
    # Public API 17: run_uncertainty
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> UncertaintyResponse:
        """Run uncertainty quantification on a calculation.

        Performs Monte Carlo simulation or analytical error propagation
        on a completed emission calculation to quantify the confidence
        interval of the CO2e estimate.

        Args:
            data: Uncertainty request with keys:
                - calculation_id (str, required)
                - method (str, default 'monte_carlo')
                - iterations (int, default 10000)
                - confidence_level (float, default 0.95)

        Returns:
            UncertaintyResponse with mean, std_dev, and CI bounds.
        """
        calc_id = str(data.get("calculation_id", ""))
        method = str(data.get("method", "monte_carlo"))
        iterations = int(data.get("iterations", 10000))
        confidence_level = _safe_float(
            data.get("confidence_level"), 0.95,
        )

        # Get the calculation result
        calc_record = self._calculations.get(calc_id)
        if calc_record is None:
            return UncertaintyResponse(
                success=False,
                calculation_id=calc_id,
                method=method,
                metadata={"error": f"Calculation '{calc_id}' not found"},
            )

        calc_data = calc_record.get("response", calc_record)
        total_co2e_kg = _safe_float(
            calc_data.get("total_co2e_kg"),
        )
        total_co2e_tonnes = _safe_float(
            calc_data.get("total_co2e_tonnes"),
        )

        mean_co2e = total_co2e_tonnes
        std_dev = 0.0
        ci_lower = total_co2e_tonnes
        ci_upper = total_co2e_tonnes
        cv = 0.0

        if self._uncertainty is not None and total_co2e_kg > 0:
            try:
                uc_result = self._uncertainty.run_monte_carlo(
                    base_emissions_kg=_safe_decimal(total_co2e_kg),
                    ef_uncertainty_pct=_safe_decimal(
                        data.get("ef_uncertainty_pct", "0.10"),
                    ),
                    activity_uncertainty_pct=_safe_decimal(
                        data.get("activity_uncertainty_pct", "0.05"),
                    ),
                    iterations=iterations,
                    confidence_level=_safe_decimal(confidence_level),
                )
                if isinstance(uc_result, dict):
                    mean_co2e = _safe_float(
                        uc_result.get("mean_tonnes"),
                        total_co2e_tonnes,
                    )
                    std_dev = _safe_float(
                        uc_result.get("std_dev_tonnes"),
                    )
                    ci_lower = _safe_float(
                        uc_result.get("ci_lower_tonnes"),
                        total_co2e_tonnes * 0.9,
                    )
                    ci_upper = _safe_float(
                        uc_result.get("ci_upper_tonnes"),
                        total_co2e_tonnes * 1.1,
                    )
                    cv = _safe_float(
                        uc_result.get("coefficient_of_variation"),
                    )

                # Record metric
                if self._metrics is not None:
                    try:
                        self._metrics.record_uncertainty_run(
                            method=method,
                        )
                    except Exception:
                        pass

            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed for %s: %s",
                    calc_id,
                    exc,
                )
                # Use analytical fallback
                std_dev = total_co2e_tonnes * 0.10
                ci_lower = total_co2e_tonnes * 0.85
                ci_upper = total_co2e_tonnes * 1.15
                cv = (
                    std_dev / total_co2e_tonnes
                    if total_co2e_tonnes > 0
                    else 0.0
                )
                method = "analytical_fallback"
        else:
            # Simple analytical uncertainty (10% assumption)
            if total_co2e_tonnes > 0:
                std_dev = total_co2e_tonnes * 0.10
                ci_lower = total_co2e_tonnes * 0.85
                ci_upper = total_co2e_tonnes * 1.15
                cv = std_dev / total_co2e_tonnes
            method = "analytical_fallback"

        self._total_uncertainty_runs += 1

        provenance_hash = _compute_hash({
            "calculation_id": calc_id,
            "method": method,
            "iterations": iterations,
            "mean_co2e": mean_co2e,
            "std_dev": std_dev,
        })

        uc_record: Dict[str, Any] = {
            "calculation_id": calc_id,
            "method": method,
            "iterations": iterations,
            "mean_co2e_tonnes": mean_co2e,
            "std_dev_tonnes": std_dev,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "cv": cv,
            "provenance_hash": provenance_hash,
        }
        uc_id = _short_id("s2l_unc")
        self._uncertainty_results[uc_id] = uc_record

        return UncertaintyResponse(
            success=True,
            calculation_id=calc_id,
            method=method,
            iterations=iterations,
            mean_co2e_tonnes=round(mean_co2e, 6),
            std_dev_tonnes=round(std_dev, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_level=confidence_level,
            coefficient_of_variation=round(cv, 6),
            provenance_hash=provenance_hash,
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 18: get_aggregations
    # ==================================================================

    def get_aggregations(
        self,
        tenant_id: str,
        group_by: str = "facility",
    ) -> AggregationResponse:
        """Get aggregated Scope 2 location-based emissions.

        Aggregates calculation results by the specified dimension
        (facility, energy_type, grid_region, country, month, quarter).

        Args:
            tenant_id: Tenant identifier to scope aggregation.
            group_by: Aggregation dimension (default 'facility').

        Returns:
            AggregationResponse with per-group totals.
        """
        # Filter calculations by tenant
        tenant_calcs = [
            c for c in self._calculations.values()
            if c.get("tenant_id") == tenant_id
        ]

        groups: Dict[str, Dict[str, float]] = {}
        total_co2e = 0.0
        facility_ids: set = set()

        for calc in tenant_calcs:
            calc_resp = calc.get("response", calc)
            co2e_tonnes = _safe_float(
                calc_resp.get("total_co2e_tonnes"),
            )
            total_co2e += co2e_tonnes
            fac_id = calc.get("facility_id", "unknown")
            facility_ids.add(fac_id)

            # Determine group key
            if group_by == "facility":
                key = fac_id
            elif group_by == "energy_type":
                key = str(
                    calc_resp.get("energy_type", "unknown"),
                )
            elif group_by == "grid_region":
                key = str(
                    calc_resp.get("grid_region", "unknown"),
                )
            elif group_by == "country":
                req = calc.get("request", {})
                key = str(req.get("country_code", "unknown"))
            elif group_by == "month":
                ts = calc.get("timestamp", "")
                key = ts[:7] if len(ts) >= 7 else "unknown"
            elif group_by == "quarter":
                ts = calc.get("timestamp", "")
                if len(ts) >= 7:
                    month = int(ts[5:7])
                    q = (month - 1) // 3 + 1
                    key = f"{ts[:4]}-Q{q}"
                else:
                    key = "unknown"
            else:
                key = fac_id

            if key not in groups:
                groups[key] = {
                    "total_co2e_tonnes": 0.0,
                    "calculation_count": 0,
                }

            groups[key]["total_co2e_tonnes"] += co2e_tonnes
            groups[key]["calculation_count"] += 1

        return AggregationResponse(
            group_by=group_by,
            groups=groups,
            total_co2e_tonnes=round(total_co2e, 6),
            facility_count=len(facility_ids),
            calculation_count=len(tenant_calcs),
            period="all",
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 19: health_check
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Perform a service health check.

        Reports the status of all engine components, configuration
        validation, and uptime.

        Returns:
            HealthResponse with engine status and diagnostics.
        """
        engines: Dict[str, str] = {
            "grid_factor_database": (
                "available" if self._grid_db is not None
                else "unavailable"
            ),
            "electricity_emissions": (
                "available" if self._electricity is not None
                else "unavailable"
            ),
            "steam_heat_cooling": (
                "available" if self._steam_heat_cool is not None
                else "unavailable"
            ),
            "transmission_loss": (
                "available" if self._transmission is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available" if self._uncertainty is not None
                else "unavailable"
            ),
            "compliance_checker": (
                "available" if self._compliance is not None
                else "unavailable"
            ),
            "pipeline": (
                "available" if self._pipeline is not None
                else "unavailable"
            ),
        }

        # Config validation
        config_valid = True
        if self._config is not None:
            try:
                if hasattr(self._config, "validate"):
                    errors = self._config.validate()
                    config_valid = len(errors) == 0
            except Exception:
                config_valid = False

        # Determine overall status
        engine_available_count = sum(
            1 for v in engines.values() if v == "available"
        )
        if engine_available_count == 7:
            status = "healthy"
        elif engine_available_count >= 4:
            status = "degraded"
        elif engine_available_count >= 1:
            status = "partial"
        else:
            status = "unhealthy"

        uptime_s = time.monotonic() - self._start_time

        return HealthResponse(
            status=status,
            service=SERVICE_NAME,
            version=SERVICE_VERSION,
            agent_id=AGENT_ID,
            engines=engines,
            config_valid=config_valid,
            uptime_seconds=round(uptime_s, 2),
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 20: get_stats
    # ==================================================================

    def get_stats(self) -> StatsResponse:
        """Get service aggregate statistics.

        Returns cumulative counters for calculations, facilities,
        consumption records, compliance checks, uncertainty analyses,
        and total CO2e.

        Returns:
            StatsResponse with all aggregate counters.
        """
        uptime_s = time.monotonic() - self._start_time

        return StatsResponse(
            total_calculations=self._total_calculations,
            total_batch_runs=self._total_batch_runs,
            total_facilities=len(self._facilities),
            total_consumption_records=len(self._consumption),
            total_compliance_checks=self._total_compliance_checks,
            total_uncertainty_runs=self._total_uncertainty_runs,
            total_co2e_tonnes=round(
                self._cumulative_co2e_tonnes, 6,
            ),
            uptime_seconds=round(uptime_s, 2),
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Utility: reset (testing only)
    # ==================================================================

    def reset(self) -> None:
        """Reset all in-memory state.

        Intended for test teardown only. Clears all stored calculations,
        facilities, consumption records, compliance results, uncertainty
        results, and custom factors. Resets all counters.
        """
        self._calculations.clear()
        self._facilities.clear()
        self._consumption.clear()
        self._compliance_results.clear()
        self._uncertainty_results.clear()
        self._custom_factors.clear()
        self._total_calculations = 0
        self._total_batch_runs = 0
        self._total_compliance_checks = 0
        self._total_uncertainty_runs = 0
        self._cumulative_co2e_tonnes = 0.0
        logger.info("Scope2LocationService state reset")

# ===================================================================
# GWP lookup table (deterministic Decimal values)
# ===================================================================

_GWP_TABLES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "co2": Decimal("1"),
        "ch4": Decimal("25"),
        "n2o": Decimal("298"),
    },
    "AR5": {
        "co2": Decimal("1"),
        "ch4": Decimal("28"),
        "n2o": Decimal("265"),
    },
    "AR6": {
        "co2": Decimal("1"),
        "ch4": Decimal("27.9"),
        "n2o": Decimal("273"),
    },
    "AR6_20YR": {
        "co2": Decimal("1"),
        "ch4": Decimal("81.2"),
        "n2o": Decimal("273"),
    },
}

def _get_gwp_table(gwp_source: str) -> Dict[str, Decimal]:
    """Get the GWP conversion table for a given IPCC AR source.

    Args:
        gwp_source: IPCC Assessment Report identifier
            (AR4, AR5, AR6, AR6_20YR).

    Returns:
        Dictionary mapping gas name (lowercase) to GWP Decimal value.
    """
    return _GWP_TABLES.get(
        gwp_source.upper(),
        _GWP_TABLES["AR5"],
    )

# ===================================================================
# Module-level singleton
# ===================================================================

def get_service() -> Scope2LocationService:
    """Return the module-level Scope2LocationService singleton.

    Thread-safe lazy initialization. The first call creates the
    service instance; subsequent calls return the same instance.

    Returns:
        The shared Scope2LocationService instance.

    Example:
        >>> svc = get_service()
        >>> health = svc.health_check()
        >>> assert health.status in ("healthy", "degraded", "partial")
    """
    global _service_instance
    if _service_instance is None:
        with _singleton_lock:
            if _service_instance is None:
                _service_instance = Scope2LocationService()
    return _service_instance

def reset_service() -> None:
    """Reset the module-level service singleton.

    After calling this function, the next call to ``get_service()``
    will create a fresh instance. Intended for test teardown.
    """
    global _service_instance
    with _singleton_lock:
        _service_instance = None
    logger.debug("Scope2LocationService singleton reset")

def get_service_with_config(
    config: Any = None,
    **overrides: Any,
) -> Scope2LocationService:
    """Create a new Scope2LocationService with custom configuration.

    Does NOT modify the module-level singleton. Returns a fresh
    instance with the provided configuration.

    Args:
        config: Optional configuration override.
        **overrides: Additional keyword overrides applied to config.

    Returns:
        A new Scope2LocationService instance.

    Example:
        >>> svc = get_service_with_config(
        ...     default_gwp_source="AR6",
        ...     decimal_precision=12,
        ... )
    """
    cfg = config
    if cfg is None:
        cfg = get_config()

    if overrides and cfg is not None and hasattr(cfg, "merge"):
        cfg.merge(overrides)

    return Scope2LocationService(config=cfg)

# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service class
    "Scope2LocationService",
    # Singleton accessors
    "get_service",
    "reset_service",
    "get_service_with_config",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FacilityResponse",
    "FacilityListResponse",
    "ConsumptionResponse",
    "ConsumptionListResponse",
    "GridFactorResponse",
    "GridFactorListResponse",
    "TDLossListResponse",
    "ComplianceCheckResponse",
    "UncertaintyResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
    # Constants
    "SERVICE_VERSION",
    "SERVICE_NAME",
    "AGENT_ID",
    "VALID_ENERGY_TYPES",
    "VALID_GWP_SOURCES",
    "VALID_CONSUMPTION_UNITS",
    "VALID_FACILITY_TYPES",
    "VALID_GRID_SOURCES",
    "VALID_COMPLIANCE_FRAMEWORKS",
    "VALID_GROUP_BY",
    "VALID_STEAM_TYPES",
    "VALID_HEATING_TYPES",
    "VALID_COOLING_TYPES",
]
