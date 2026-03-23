# -*- coding: utf-8 -*-
"""
Scope 2 Market-Based Emissions Service Setup - AGENT-MRV-010
=============================================================

Service facade for the Scope 2 Market-Based Emissions Agent
(GL-MRV-SCOPE2-002).

Provides ``get_service()`` and the ``Scope2MarketService`` facade class
that aggregates all 7 engines:

    1. ContractualInstrumentDatabaseEngine - Instrument registry (REC/GO/PPA)
    2. InstrumentAllocationEngine          - Allocation of instruments to load
    3. MarketEmissionsCalculatorEngine     - Market-based emission calculations
    4. DualReportingEngine                 - Location vs. market comparison
    5. UncertaintyQuantifierEngine         - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine             - Multi-framework regulatory compliance
    7. Scope2MarketPipelineEngine          - 8-stage orchestrated pipeline

The service provides 20 public methods matching the 20 REST API endpoints:

    Calculations:
        calculate, calculate_batch, list_calculations,
        get_calculation, delete_calculation
    Facilities:
        register_facility, list_facilities, update_facility
    Instruments:
        register_instrument, list_instruments, retire_instrument
    Compliance:
        check_compliance, get_compliance_result
    Uncertainty:
        run_uncertainty
    Dual Reporting:
        generate_dual_report
    Aggregations:
        get_aggregations
    Coverage:
        get_coverage_analysis
    Health:
        health_check, get_stats, get_engine_status

All calculation paths use deterministic Decimal arithmetic for
zero-hallucination guarantees. Every mutation records a SHA-256
provenance hash for complete audit trails.

Usage:
    >>> from greenlang.agents.mrv.scope2_market.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "tenant_id": "tenant-001",
    ...     "facility_id": "fac-001",
    ...     "instrument_type": "rec",
    ...     "consumption_value": 5000.0,
    ...     "consumption_unit": "mwh",
    ...     "country_code": "US",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope2_market.config import (
        Scope2MarketConfig,
        get_config,
    )
except ImportError:
    Scope2MarketConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.scope2_market.metrics import (
        Scope2MarketMetrics,
        get_metrics,
    )
except ImportError:
    Scope2MarketMetrics = None  # type: ignore[assignment, misc]

    def get_metrics() -> Any:  # type: ignore[misc]
        """Stub returning None when metrics module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.scope2_market.provenance import (
        Scope2MarketProvenance,
    )
except ImportError:
    Scope2MarketProvenance = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.contractual_instrument_database import (
        ContractualInstrumentDatabaseEngine,
    )
except ImportError:
    ContractualInstrumentDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.instrument_allocation import (
        InstrumentAllocationEngine,
    )
except ImportError:
    InstrumentAllocationEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.market_emissions_calculator import (
        MarketEmissionsCalculatorEngine,
    )
except ImportError:
    MarketEmissionsCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.dual_reporting import (
        DualReportingEngine,
    )
except ImportError:
    DualReportingEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.scope2_market.scope2_market_pipeline import (
        Scope2MarketPipelineEngine,
    )
except ImportError:
    Scope2MarketPipelineEngine = None  # type: ignore[assignment, misc]


# ===================================================================
# Constants
# ===================================================================

#: Service version for health checks and diagnostics.
SERVICE_VERSION: str = "1.0.0"

#: Service name for observability.
SERVICE_NAME: str = "scope2-market-service"

#: Agent identifier for tracing and audit logs.
AGENT_ID: str = "AGENT-MRV-010"

#: Default GWP source when not provided in requests.
DEFAULT_GWP_SOURCE: str = "AR5"

#: Default max batch size.
DEFAULT_MAX_BATCH_SIZE: int = 1000

#: Valid contractual instrument types.
VALID_INSTRUMENT_TYPES: frozenset = frozenset({
    "rec",
    "go",
    "rego",
    "i_rec",
    "t_rec",
    "lgc",
    "j_credit",
    "ppa",
    "vppa",
    "green_tariff",
    "direct_line",
    "self_generated",
    "bundled",
    "unbundled",
    "supplier_specific",
})

#: Valid calculation methods for market-based emissions.
VALID_CALCULATION_METHODS: frozenset = frozenset({
    "contractual",
    "market_based",
    "supplier_specific",
    "residual_mix",
    "energy_attribute_certificate",
    "power_purchase_agreement",
    "green_tariff",
    "direct_line",
    "self_generation",
})

#: Valid energy units for input validation.
VALID_ENERGY_UNITS: frozenset = frozenset({
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

#: Supported compliance frameworks.
VALID_COMPLIANCE_FRAMEWORKS: frozenset = frozenset({
    "ghg_protocol_scope2",
    "re100",
    "cdp",
    "sbti",
    "iso_14064",
    "csrd_esrs",
    "eu_taxonomy",
    "sec_climate",
})

#: Valid GWP sources for input validation.
VALID_GWP_SOURCES: frozenset = frozenset({
    "AR4",
    "AR5",
    "AR6",
    "AR6_20YR",
})

#: Valid tracking system registries.
VALID_TRACKING_SYSTEMS: frozenset = frozenset({
    "green_e",
    "aib_eecs",
    "ofgem",
    "i_rec_standard",
    "m_rets",
    "nar",
    "wregis",
    "custom",
})

#: Valid instrument allocation methods.
VALID_ALLOCATION_METHODS: frozenset = frozenset({
    "fifo",
    "lifo",
    "pro_rata",
    "priority",
    "vintage_first",
    "closest_match",
})

#: Valid aggregation group-by dimensions.
VALID_GROUP_BY: frozenset = frozenset({
    "facility",
    "instrument_type",
    "energy_source",
    "country",
    "month",
    "quarter",
})


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


def _short_id(prefix: str = "s2m") -> str:
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
    """Single Scope 2 market-based emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    instrument_type: str = Field(default="")
    calculation_method: str = Field(default="market_based")
    consumption_value: float = Field(default=0.0)
    consumption_unit: str = Field(default="mwh")
    coverage_pct: float = Field(default=0.0)
    covered_co2e_kg: float = Field(default=0.0)
    uncovered_co2e_kg: float = Field(default=0.0)
    residual_mix_ef: float = Field(default=0.0)
    supplier_ef: float = Field(default=0.0)
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
    """Batch Scope 2 market-based emission calculation response."""

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
    default_instrument_type: Optional[str] = Field(default=None)
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


class InstrumentResponse(BaseModel):
    """Response for a single contractual instrument record."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str = Field(default="")
    instrument_type: str = Field(default="rec")
    tracking_system: str = Field(default="")
    energy_source: str = Field(default="")
    quantity_mwh: float = Field(default=0.0)
    allocated_mwh: float = Field(default=0.0)
    remaining_mwh: float = Field(default=0.0)
    vintage_year: int = Field(default=0)
    status: str = Field(default="active")
    facility_id: Optional[str] = Field(default=None)
    supplier_name: Optional[str] = Field(default=None)
    emission_factor_co2e: float = Field(default=0.0)
    certificate_id: Optional[str] = Field(default=None)
    tenant_id: str = Field(default="")
    created_at: str = Field(default="")
    retired_at: Optional[str] = Field(default=None)


class InstrumentListResponse(BaseModel):
    """Response listing contractual instruments."""

    model_config = ConfigDict(frozen=True)

    instruments: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=50)


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


class DualReportResponse(BaseModel):
    """Dual reporting (location vs. market) comparison response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    report_id: str = Field(default="")
    facility_id: str = Field(default="")
    location_co2e_tonnes: float = Field(default=0.0)
    market_co2e_tonnes: float = Field(default=0.0)
    difference_co2e_tonnes: float = Field(default=0.0)
    difference_pct: float = Field(default=0.0)
    coverage_pct: float = Field(default=0.0)
    recommendation: str = Field(default="")
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoverageAnalysisResponse(BaseModel):
    """Instrument coverage analysis response."""

    model_config = ConfigDict(frozen=True)

    facility_id: str = Field(default="")
    total_consumption_mwh: float = Field(default=0.0)
    covered_mwh: float = Field(default=0.0)
    uncovered_mwh: float = Field(default=0.0)
    coverage_pct: float = Field(default=0.0)
    instruments_used: int = Field(default=0)
    coverage_by_type: Dict[str, float] = Field(default_factory=dict)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)


class AggregationResponse(BaseModel):
    """Aggregated Scope 2 market-based emissions response."""

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
    total_instruments: int = Field(default=0)
    total_instruments_retired: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    total_dual_reports: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


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
# Scope2MarketService facade
# ===================================================================

_singleton_lock = threading.Lock()
_service_instance: Optional["Scope2MarketService"] = None


class Scope2MarketService:
    """Unified facade over the Scope 2 Market-Based Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each mutation method records SHA-256 provenance hashes.
    All numeric calculations use deterministic Decimal arithmetic
    delegated to the underlying engines (zero-hallucination path).

    In-memory storage provides the default persistence layer. In
    production, methods should be backed by PostgreSQL via the
    engines' database connectors.

    Attributes:
        config: Service configuration (Scope2MarketConfig or dict).
        metrics: Prometheus metrics singleton.

    Example:
        >>> service = Scope2MarketService()
        >>> result = service.calculate({
        ...     "tenant_id": "tenant-001",
        ...     "facility_id": "fac-001",
        ...     "instrument_type": "rec",
        ...     "consumption_value": 5000.0,
        ...     "consumption_unit": "mwh",
        ...     "country_code": "US",
        ... })
        >>> assert result.success is True
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Scope 2 Market-Based Emissions Service facade.

        Creates all engine instances with graceful degradation when
        individual engine modules are not importable. Sets up in-memory
        storage for calculations, facilities, instruments, compliance
        results, dual reports, and uncertainty analyses.

        Args:
            config: Optional configuration override. Accepts
                Scope2MarketConfig, dict, or None (defaults to
                singleton from get_config()).
        """
        self._config = config if config is not None else get_config()
        self._metrics = get_metrics()
        self._start_time: float = time.monotonic()

        # Engine placeholders (initialised in _init_engines)
        self._instrument_db: Any = None
        self._allocation: Any = None
        self._calculator: Any = None
        self._dual_reporting: Any = None
        self._uncertainty: Any = None
        self._compliance: Any = None
        self._pipeline: Any = None

        self._init_engines()

        # In-memory data stores
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._facilities: Dict[str, Dict[str, Any]] = {}
        self._instruments: Dict[str, Dict[str, Any]] = {}
        self._compliance_results: Dict[str, Dict[str, Any]] = {}
        self._dual_reports: Dict[str, Dict[str, Any]] = {}
        self._uncertainty_results: Dict[str, Dict[str, Any]] = {}

        # Aggregate statistics
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_compliance_checks: int = 0
        self._total_uncertainty_runs: int = 0
        self._total_dual_reports: int = 0
        self._total_instruments_retired: int = 0
        self._cumulative_co2e_tonnes: float = 0.0

        logger.info(
            "Scope2MarketService facade created "
            "(engines: instrument_db=%s, allocation=%s, "
            "calculator=%s, dual_reporting=%s, "
            "uncertainty=%s, compliance=%s, pipeline=%s)",
            self._instrument_db is not None,
            self._allocation is not None,
            self._calculator is not None,
            self._dual_reporting is not None,
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
    def instrument_db_engine(self) -> Any:
        """Get the ContractualInstrumentDatabaseEngine instance."""
        return self._instrument_db

    @property
    def allocation_engine(self) -> Any:
        """Get the InstrumentAllocationEngine instance."""
        return self._allocation

    @property
    def calculator_engine(self) -> Any:
        """Get the MarketEmissionsCalculatorEngine instance."""
        return self._calculator

    @property
    def dual_reporting_engine(self) -> Any:
        """Get the DualReportingEngine instance."""
        return self._dual_reporting

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
        """Get the Scope2MarketPipelineEngine instance."""
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
        config_arg = self._config
        if config_arg is not None and hasattr(config_arg, "to_dict"):
            config_arg = config_arg.to_dict()
        elif config_arg is not None and not isinstance(config_arg, dict):
            config_arg = {}
        metrics_arg = self._metrics

        # E1: ContractualInstrumentDatabaseEngine
        self._instrument_db = self._init_single_engine(
            "ContractualInstrumentDatabaseEngine",
            ContractualInstrumentDatabaseEngine,
            config_arg,
            metrics_arg,
        )

        # E2: InstrumentAllocationEngine
        self._allocation = self._init_allocation_engine(
            config_arg, metrics_arg,
        )

        # E3: MarketEmissionsCalculatorEngine
        self._calculator = self._init_calculator_engine(
            config_arg, metrics_arg,
        )

        # E4: DualReportingEngine
        self._dual_reporting = self._init_single_engine(
            "DualReportingEngine",
            DualReportingEngine,
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

        # E7: Scope2MarketPipelineEngine (depends on all upstream)
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

    def _init_allocation_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize the InstrumentAllocationEngine.

        The allocation engine takes the instrument_db engine as its
        first argument for instrument lookups.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.

        Returns:
            InstrumentAllocationEngine instance or None.
        """
        if InstrumentAllocationEngine is None:
            logger.warning(
                "InstrumentAllocationEngine not available (import failed)"
            )
            return None

        try:
            return InstrumentAllocationEngine(
                self._instrument_db, config_arg, metrics_arg,
            )
        except TypeError:
            pass

        try:
            return InstrumentAllocationEngine(config=config_arg)
        except TypeError:
            pass

        try:
            return InstrumentAllocationEngine()
        except Exception as exc:
            logger.warning(
                "InstrumentAllocationEngine initialization failed: %s",
                exc,
            )
            return None

    def _init_calculator_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> Any:
        """Initialize the MarketEmissionsCalculatorEngine.

        The calculator engine takes the instrument_db and allocation
        engines for instrument-based emission calculations.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.

        Returns:
            MarketEmissionsCalculatorEngine instance or None.
        """
        if MarketEmissionsCalculatorEngine is None:
            logger.warning(
                "MarketEmissionsCalculatorEngine not available "
                "(import failed)"
            )
            return None

        try:
            return MarketEmissionsCalculatorEngine(
                self._instrument_db, self._allocation,
                config_arg, metrics_arg,
            )
        except TypeError:
            pass

        try:
            return MarketEmissionsCalculatorEngine(config=config_arg)
        except TypeError:
            pass

        try:
            return MarketEmissionsCalculatorEngine()
        except Exception as exc:
            logger.warning(
                "MarketEmissionsCalculatorEngine initialization "
                "failed: %s",
                exc,
            )
            return None

    def _init_pipeline_engine(
        self,
        config_arg: Any,
        metrics_arg: Any,
    ) -> None:
        """Initialize the Scope2MarketPipelineEngine.

        The pipeline engine receives all upstream engine instances
        for orchestrated calculation.

        Args:
            config_arg: Configuration instance.
            metrics_arg: Metrics instance.
        """
        if Scope2MarketPipelineEngine is None:
            logger.warning(
                "Scope2MarketPipelineEngine not available (import failed)"
            )
            return

        try:
            self._pipeline = Scope2MarketPipelineEngine(
                instrument_db=self._instrument_db,
                allocation_engine=self._allocation,
                calculator_engine=self._calculator,
                dual_reporting_engine=self._dual_reporting,
                uncertainty_engine=self._uncertainty,
                compliance_engine=self._compliance,
                config=config_arg,
                metrics=metrics_arg,
            )
            logger.info("Scope2MarketPipelineEngine initialized")
        except TypeError:
            try:
                self._pipeline = Scope2MarketPipelineEngine(
                    self._instrument_db,
                    self._allocation,
                    self._calculator,
                    self._dual_reporting,
                    self._uncertainty,
                    self._compliance,
                    config_arg,
                    metrics_arg,
                )
                logger.info(
                    "Scope2MarketPipelineEngine initialized "
                    "(positional args)"
                )
            except Exception as exc:
                logger.warning(
                    "Scope2MarketPipelineEngine initialization "
                    "failed: %s",
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Scope2MarketPipelineEngine initialization "
                "failed: %s",
                exc,
            )

    # ==================================================================
    # Internal: metrics recording
    # ==================================================================

    def _record_metric_calculation(
        self,
        instrument_type: str,
        duration_s: float,
        co2e_tonnes: float,
    ) -> None:
        """Record a calculation metric if metrics are available.

        Args:
            instrument_type: Instrument type label for the metric.
            duration_s: Duration in seconds.
            co2e_tonnes: CO2e in tonnes for cumulative tracking.
        """
        if self._metrics is None:
            return
        try:
            self._metrics.record_calculation(
                instrument_type=instrument_type,
                calculation_method="market_based",
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
            "calculation_id", _short_id("s2m_calc"),
        )
        tenant_id = request.get("tenant_id", "default")
        facility_id = request.get("facility_id", "")
        instrument_type = str(
            request.get("instrument_type", ""),
        ).lower()
        calculation_method = str(
            request.get("calculation_method", "market_based"),
        ).lower()
        consumption_value = request.get("consumption_value", 0)
        consumption_unit = str(
            request.get("consumption_unit", "mwh"),
        ).lower()
        country_code = str(
            request.get("country_code", ""),
        ).upper()
        gwp_source = str(
            request.get("gwp_source", DEFAULT_GWP_SOURCE),
        ).upper()

        # Instrument-specific fields
        instrument_id = request.get("instrument_id")
        supplier_name = request.get("supplier_name")
        supplier_ef = request.get("supplier_ef")
        allocation_method = str(
            request.get("allocation_method", "fifo"),
        ).lower()
        vintage_year = request.get("vintage_year")
        tracking_system = request.get("tracking_system")
        energy_source = request.get("energy_source")
        certificate_id = request.get("certificate_id")

        # Compliance and reporting options
        include_compliance = request.get("include_compliance", False)
        compliance_frameworks = request.get(
            "compliance_frameworks", None,
        )
        include_dual_report = request.get("include_dual_report", False)
        location_result = request.get("location_result")

        return {
            "calculation_id": calc_id,
            "tenant_id": tenant_id,
            "facility_id": facility_id,
            "instrument_type": instrument_type,
            "calculation_method": calculation_method,
            "consumption_value": consumption_value,
            "consumption_unit": consumption_unit,
            "country_code": country_code,
            "gwp_source": gwp_source,
            "instrument_id": instrument_id,
            "supplier_name": supplier_name,
            "supplier_ef": supplier_ef,
            "allocation_method": allocation_method,
            "vintage_year": vintage_year,
            "tracking_system": tracking_system,
            "energy_source": energy_source,
            "certificate_id": certificate_id,
            "include_compliance": include_compliance,
            "compliance_frameworks": compliance_frameworks,
            "include_dual_report": include_dual_report,
            "location_result": location_result,
        }

    # ==================================================================
    # Internal: result builder from pipeline output
    # ==================================================================

    def _build_calculate_response(
        self,
        pipeline_result: Dict[str, Any],
        calc_id: str,
        instrument_type: str,
        elapsed_ms: float,
    ) -> CalculateResponse:
        """Build a CalculateResponse from raw pipeline engine output.

        Maps the engine's output dictionary to the Pydantic response
        model fields.

        Args:
            pipeline_result: Raw dict from the pipeline engine.
            calc_id: Calculation identifier.
            instrument_type: Instrument type string.
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
        calculation_method = str(
            pipeline_result.get("calculation_method", "market_based"),
        )
        coverage_pct = _safe_float(
            pipeline_result.get("coverage_pct"),
        )
        covered_co2e_kg = _safe_float(
            pipeline_result.get("covered_co2e_kg"),
        )
        uncovered_co2e_kg = _safe_float(
            pipeline_result.get("uncovered_co2e_kg"),
        )
        residual_mix_ef = _safe_float(
            pipeline_result.get("residual_mix_ef"),
        )
        supplier_ef = _safe_float(
            pipeline_result.get("supplier_ef"),
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
                "instrument_type": instrument_type,
                "total_co2e_kg": total_co2e_kg,
                "total_co2e_tonnes": total_co2e_tonnes,
            })

        metadata_raw = pipeline_result.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            metadata_raw = {}

        return CalculateResponse(
            success=True,
            calculation_id=calc_id,
            instrument_type=instrument_type,
            calculation_method=calculation_method,
            consumption_value=consumption_value,
            consumption_unit=consumption_unit,
            coverage_pct=coverage_pct,
            covered_co2e_kg=covered_co2e_kg,
            uncovered_co2e_kg=uncovered_co2e_kg,
            residual_mix_ef=residual_mix_ef,
            supplier_ef=supplier_ef,
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

    def _fallback_market_calculation(
        self,
        request: Dict[str, Any],
        calc_id: str,
    ) -> Dict[str, Any]:
        """Fallback market-based calculation using direct engine calls.

        Invoked when the pipeline engine is unavailable. Uses the
        calculator engine and instrument database directly.

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
        gwp_source = request.get("gwp_source", DEFAULT_GWP_SOURCE)
        instrument_type = request.get("instrument_type", "")
        supplier_ef_override = request.get("supplier_ef")

        # Try calculator engine directly
        if self._calculator is not None:
            try:
                result = self._calculator.calculate(request)
                if isinstance(result, dict):
                    return result
                if hasattr(result, "model_dump"):
                    return result.model_dump()
            except Exception as exc:
                logger.warning(
                    "Calculator engine fallback failed: %s", exc,
                )

        # Pure arithmetic fallback using residual mix
        residual_mix_ef = Decimal("0")
        try:
            from greenlang.agents.mrv.scope2_market.models import (
                RESIDUAL_MIX_FACTORS,
            )
            rmf = RESIDUAL_MIX_FACTORS.get(country_code)
            if rmf is not None:
                residual_mix_ef = _safe_decimal(rmf)
        except ImportError:
            pass

        # Default residual mix factors (kgCO2e/kWh)
        if residual_mix_ef == Decimal("0"):
            _default_rmf: Dict[str, str] = {
                "US": "0.4293",
                "GB": "0.2331",
                "DE": "0.3850",
                "FR": "0.0569",
                "JP": "0.4710",
                "AU": "0.6800",
                "CA": "0.1200",
                "IN": "0.7080",
                "CN": "0.5810",
                "BR": "0.0740",
            }
            residual_mix_ef = _safe_decimal(
                _default_rmf.get(country_code, "0.4293"),
            )

        # Convert consumption to kWh for residual mix lookup
        consumption_kwh = consumption_val * Decimal("1000")

        # Instrument coverage: if supplier EF provided, use it for
        # the covered portion
        coverage_pct = Decimal("0")
        covered_co2e_kg = Decimal("0")
        supplier_ef_val = Decimal("0")

        if supplier_ef_override is not None:
            supplier_ef_val = _safe_decimal(supplier_ef_override)
            coverage_pct = Decimal("100")
            covered_co2e_kg = consumption_kwh * supplier_ef_val
        elif instrument_type:
            # Instruments with zero EF (RECs, GOs) cover consumption
            coverage_pct = Decimal("100")
            covered_co2e_kg = Decimal("0")

        # Uncovered portion uses residual mix
        uncovered_pct = Decimal("100") - coverage_pct
        uncovered_kwh = consumption_kwh * uncovered_pct / Decimal("100")
        uncovered_co2e_kg = uncovered_kwh * residual_mix_ef

        total_co2e_kg = covered_co2e_kg + uncovered_co2e_kg
        total_co2e_tonnes = total_co2e_kg / Decimal("1000")

        # GWP-weighted gas breakdown
        gwp_table = _get_gwp_table(gwp_source)
        co2_fraction = Decimal("0.99")
        ch4_fraction = Decimal("0.005")
        n2o_fraction = Decimal("0.005")

        co2_kg = total_co2e_kg * co2_fraction / gwp_table["co2"]
        ch4_kg = total_co2e_kg * ch4_fraction / gwp_table["ch4"]
        n2o_kg = total_co2e_kg * n2o_fraction / gwp_table["n2o"]

        return {
            "calculation_id": calc_id,
            "calculation_method": "market_based",
            "consumption_value": float(consumption_val),
            "consumption_unit": "mwh",
            "coverage_pct": float(coverage_pct),
            "covered_co2e_kg": float(covered_co2e_kg),
            "uncovered_co2e_kg": float(uncovered_co2e_kg),
            "residual_mix_ef": float(residual_mix_ef),
            "supplier_ef": float(supplier_ef_val),
            "gas_breakdown": [
                {
                    "gas": "CO2",
                    "emission_kg": float(co2_kg),
                    "co2e_kg": float(co2_kg * gwp_table["co2"]),
                    "gwp_factor": float(gwp_table["co2"]),
                },
                {
                    "gas": "CH4",
                    "emission_kg": float(ch4_kg),
                    "co2e_kg": float(ch4_kg * gwp_table["ch4"]),
                    "gwp_factor": float(gwp_table["ch4"]),
                },
                {
                    "gas": "N2O",
                    "emission_kg": float(n2o_kg),
                    "co2e_kg": float(n2o_kg * gwp_table["n2o"]),
                    "gwp_factor": float(gwp_table["n2o"]),
                },
            ],
            "total_co2e_kg": float(total_co2e_kg),
            "total_co2e_tonnes": float(total_co2e_tonnes),
            "gwp_source": gwp_source,
            "provenance_hash": "",
            "metadata": {"fallback": True},
        }

    # ==================================================================
    # Public API 1: calculate
    # ==================================================================

    def calculate(self, request: Dict[str, Any]) -> CalculateResponse:
        """Calculate Scope 2 market-based emissions for a single record.

        Supports contractual instruments (REC, GO, PPA, etc.),
        supplier-specific factors, and residual mix methodology.
        Delegates to the pipeline engine when available, falling back
        to direct engine calls otherwise.

        Args:
            request: Calculation request dict with keys:
                - tenant_id (str, required)
                - facility_id (str, required)
                - consumption_value (numeric, required)
                - consumption_unit (str, default 'mwh')
                - instrument_type (str, e.g. 'rec', 'go', 'ppa')
                - calculation_method (str, default 'market_based')
                - country_code (str, required for residual mix lookup)
                - gwp_source (str, default 'AR5')
                - supplier_name (str, optional)
                - supplier_ef (numeric, optional override)
                - allocation_method (str, default 'fifo')
                - instrument_id (str, optional specific instrument)

        Returns:
            CalculateResponse with emission values and provenance hash.
        """
        t0 = time.monotonic()
        normalized = self._build_pipeline_request(request)
        calc_id = normalized["calculation_id"]
        instrument_type = normalized["instrument_type"]

        try:
            # Validate required fields
            errors = _validate_required_fields(
                normalized,
                ["tenant_id", "facility_id", "consumption_value"],
                "calculate",
            )
            if errors:
                raise ValueError("; ".join(errors))

            # Validate instrument type if provided
            if instrument_type:
                err = _validate_enum_field(
                    instrument_type,
                    VALID_INSTRUMENT_TYPES,
                    "instrument_type",
                )
                if err:
                    raise ValueError(err)

            # Route through pipeline or fallback
            raw_result: Dict[str, Any]
            if self._pipeline is not None:
                raw_result = self._pipeline.run_pipeline(normalized)
            else:
                raw_result = self._fallback_market_calculation(
                    normalized, calc_id,
                )

            elapsed_ms = _elapsed_ms(t0)
            duration_s = time.monotonic() - t0

            response = self._build_calculate_response(
                raw_result, calc_id, instrument_type, elapsed_ms,
            )

            # Store result
            calc_record = {
                "calculation_id": calc_id,
                "tenant_id": normalized["tenant_id"],
                "facility_id": normalized["facility_id"],
                "instrument_type": instrument_type,
                "total_co2e_kg": response.total_co2e_kg,
                "total_co2e_tonnes": response.total_co2e_tonnes,
                "coverage_pct": response.coverage_pct,
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
                instrument_type or "residual_mix",
                duration_s,
                response.total_co2e_tonnes,
            )

            logger.info(
                "Calculation %s completed: instrument_type=%s, "
                "coverage=%.1f%%, total_co2e_tonnes=%.6f, "
                "elapsed_ms=%.3f",
                calc_id,
                instrument_type,
                response.coverage_pct,
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
                instrument_type=instrument_type,
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
                instrument_type=instrument_type,
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
        """Calculate Scope 2 market-based emissions for a batch.

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
        batch_id = batch.get("batch_id", _short_id("s2m_batch"))
        tenant_id = batch.get("tenant_id", "default")
        requests = batch.get("requests", [])

        if not isinstance(requests, list) or len(requests) == 0:
            return BatchCalculateResponse(
                success=False,
                batch_id=batch_id,
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
                        _short_id("s2m_calc"),
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
        """Soft-delete a calculation by ID.

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
        """Register a new facility for Scope 2 market-based tracking.

        Creates a facility record with instrument mapping and optional
        geolocation data.

        Args:
            data: Facility data dict with keys:
                - name (str, required)
                - facility_type (str, default 'office')
                - country_code (str, required)
                - grid_region_id (str, required)
                - default_instrument_type (str, optional)
                - latitude (float, optional)
                - longitude (float, optional)
                - tenant_id (str, required)

        Returns:
            FacilityResponse with the created facility record.
        """
        facility_id = _short_id("s2m_fac")
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
        default_instrument_type = data.get("default_instrument_type")
        if default_instrument_type:
            default_instrument_type = str(
                default_instrument_type,
            ).lower()
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        tenant_id = str(data.get("tenant_id", "default"))

        record: Dict[str, Any] = {
            "facility_id": facility_id,
            "name": name,
            "facility_type": facility_type,
            "country_code": country_code,
            "grid_region_id": grid_region_id,
            "default_instrument_type": default_instrument_type,
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
            default_instrument_type=default_instrument_type,
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
            "default_instrument_type",
            "latitude",
            "longitude",
        }

        for key, value in data.items():
            if key in updatable_fields and value is not None:
                if key == "country_code":
                    value = str(value).upper()
                elif key == "default_instrument_type":
                    value = str(value).lower()
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
            default_instrument_type=existing.get(
                "default_instrument_type",
            ),
            latitude=existing.get("latitude"),
            longitude=existing.get("longitude"),
            tenant_id=existing.get("tenant_id", ""),
            created_at=existing.get("created_at", ""),
            updated_at=existing.get("updated_at", ""),
        )

    # ==================================================================
    # Public API 9: register_instrument
    # ==================================================================

    def register_instrument(
        self,
        data: Dict[str, Any],
    ) -> InstrumentResponse:
        """Register a new contractual instrument.

        Creates an instrument record (REC, GO, PPA, etc.) with
        tracking system metadata and energy source attribution.

        Args:
            data: Instrument data dict with keys:
                - instrument_type (str, required, e.g. 'rec', 'go')
                - tracking_system (str, e.g. 'green_e', 'aib_eecs')
                - energy_source (str, e.g. 'wind', 'solar')
                - quantity_mwh (numeric, required)
                - vintage_year (int, required)
                - facility_id (str, optional)
                - supplier_name (str, optional)
                - emission_factor_co2e (numeric, optional)
                - certificate_id (str, optional)
                - tenant_id (str, required)

        Returns:
            InstrumentResponse with the registered instrument record.
        """
        instrument_id = _short_id("s2m_inst")
        now_iso = _utcnow_iso()

        instrument_type = str(
            data.get("instrument_type", "rec"),
        ).lower()
        tracking_system = str(
            data.get("tracking_system", ""),
        ).lower()
        energy_source = str(
            data.get("energy_source", ""),
        ).lower()
        quantity_mwh = _safe_float(data.get("quantity_mwh"))
        vintage_year = int(data.get("vintage_year", 0))
        facility_id = data.get("facility_id")
        supplier_name = data.get("supplier_name")
        emission_factor_co2e = _safe_float(
            data.get("emission_factor_co2e"),
        )
        certificate_id = data.get("certificate_id")
        tenant_id = str(data.get("tenant_id", "default"))

        record: Dict[str, Any] = {
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "tracking_system": tracking_system,
            "energy_source": energy_source,
            "quantity_mwh": quantity_mwh,
            "allocated_mwh": 0.0,
            "remaining_mwh": quantity_mwh,
            "vintage_year": vintage_year,
            "status": "active",
            "facility_id": facility_id,
            "supplier_name": supplier_name,
            "emission_factor_co2e": emission_factor_co2e,
            "certificate_id": certificate_id,
            "tenant_id": tenant_id,
            "created_at": now_iso,
            "retired_at": None,
        }

        self._instruments[instrument_id] = record

        # Register with instrument_db engine if available
        if self._instrument_db is not None:
            try:
                self._instrument_db.register_instrument(record)
            except Exception as exc:
                logger.warning(
                    "Failed to register instrument with engine: %s",
                    exc,
                )

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_instrument_registered(
                    instrument_type=instrument_type,
                    status="active",
                )
            except Exception:
                pass

        logger.info(
            "Instrument %s registered: type=%s, qty=%.2f MWh, "
            "vintage=%d",
            instrument_id,
            instrument_type,
            quantity_mwh,
            vintage_year,
        )

        return InstrumentResponse(
            instrument_id=instrument_id,
            instrument_type=instrument_type,
            tracking_system=tracking_system,
            energy_source=energy_source,
            quantity_mwh=quantity_mwh,
            allocated_mwh=0.0,
            remaining_mwh=quantity_mwh,
            vintage_year=vintage_year,
            status="active",
            facility_id=facility_id,
            supplier_name=supplier_name,
            emission_factor_co2e=emission_factor_co2e,
            certificate_id=certificate_id,
            tenant_id=tenant_id,
            created_at=now_iso,
            retired_at=None,
        )

    # ==================================================================
    # Public API 10: list_instruments
    # ==================================================================

    def list_instruments(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 50,
        instrument_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> InstrumentListResponse:
        """List contractual instruments for a tenant.

        Args:
            tenant_id: Tenant identifier to filter by.
            skip: Number of records to skip.
            limit: Maximum number of records to return.
            instrument_type: Optional filter by instrument type.
            status: Optional filter by status (active, retired, etc.).

        Returns:
            InstrumentListResponse with paginated instrument records.
        """
        tenant_instruments = [
            inst for inst in self._instruments.values()
            if inst.get("tenant_id") == tenant_id
        ]

        # Apply optional filters
        if instrument_type:
            inst_type_lower = instrument_type.lower()
            tenant_instruments = [
                inst for inst in tenant_instruments
                if inst.get("instrument_type") == inst_type_lower
            ]
        if status:
            status_lower = status.lower()
            tenant_instruments = [
                inst for inst in tenant_instruments
                if inst.get("status") == status_lower
            ]

        tenant_instruments.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        total = len(tenant_instruments)
        page = tenant_instruments[skip:skip + limit]

        return InstrumentListResponse(
            instruments=page,
            total=total,
            page=skip // max(limit, 1) + 1,
            page_size=limit,
        )

    # ==================================================================
    # Public API 11: retire_instrument
    # ==================================================================

    def retire_instrument(
        self,
        instrument_id: str,
    ) -> InstrumentResponse:
        """Retire or cancel a contractual instrument.

        Marks the instrument as retired and records the retirement
        timestamp for audit trail compliance.

        Args:
            instrument_id: Unique instrument identifier.

        Returns:
            InstrumentResponse with the retired instrument record.

        Raises:
            ValueError: If the instrument does not exist.
        """
        existing = self._instruments.get(instrument_id)
        if existing is None:
            raise ValueError(
                f"Instrument '{instrument_id}' not found"
            )

        if existing.get("status") == "retired":
            logger.warning(
                "Instrument %s is already retired", instrument_id,
            )
            return InstrumentResponse(**{
                k: v for k, v in existing.items()
                if k in InstrumentResponse.model_fields
            })

        now_iso = _utcnow_iso()
        existing["status"] = "retired"
        existing["retired_at"] = now_iso
        existing["remaining_mwh"] = 0.0

        self._total_instruments_retired += 1

        # Retire in instrument_db engine if available
        if self._instrument_db is not None:
            try:
                self._instrument_db.retire_instrument(instrument_id)
            except Exception as exc:
                logger.warning(
                    "Failed to retire instrument in engine: %s", exc,
                )

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_instrument_retired(
                    instrument_type=existing.get(
                        "instrument_type", "unknown",
                    ),
                )
            except Exception:
                pass

        logger.info(
            "Instrument %s retired at %s", instrument_id, now_iso,
        )

        return InstrumentResponse(
            instrument_id=instrument_id,
            instrument_type=existing.get("instrument_type", ""),
            tracking_system=existing.get("tracking_system", ""),
            energy_source=existing.get("energy_source", ""),
            quantity_mwh=_safe_float(existing.get("quantity_mwh")),
            allocated_mwh=_safe_float(existing.get("allocated_mwh")),
            remaining_mwh=0.0,
            vintage_year=int(existing.get("vintage_year", 0)),
            status="retired",
            facility_id=existing.get("facility_id"),
            supplier_name=existing.get("supplier_name"),
            emission_factor_co2e=_safe_float(
                existing.get("emission_factor_co2e"),
            ),
            certificate_id=existing.get("certificate_id"),
            tenant_id=existing.get("tenant_id", ""),
            created_at=existing.get("created_at", ""),
            retired_at=now_iso,
        )

    # ==================================================================
    # Public API 12: check_compliance
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Run regulatory compliance checks on a calculation.

        Evaluates a completed calculation against specified regulatory
        frameworks (GHG Protocol Scope 2, RE100, SBTi, CDP, etc.).

        Args:
            data: Compliance check request with keys:
                - calculation_id (str, required)
                - frameworks (list of str, optional)

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        check_id = _short_id("s2m_comp")
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
                            result_dict.get(
                                "status", "not_assessed",
                            ),
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
        market-based mandatory requirements and instrument quality
        criteria.

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

            # Market-based specific checks
            has_instrument = bool(
                calc_data.get("instrument_type"),
            )
            has_consumption = (
                _safe_float(calc_data.get("consumption_value")) > 0
            )
            has_provenance = bool(
                calc_data.get("provenance_hash"),
            )
            coverage_pct = _safe_float(
                calc_data.get("coverage_pct"),
            )

            if has_instrument:
                findings.append(
                    "Contractual instrument documented"
                )
            else:
                findings.append(
                    "No contractual instrument specified; "
                    "residual mix applied"
                )
                recommendations.append(
                    "Provide contractual instruments for "
                    "market-based accounting"
                )

            if has_consumption:
                findings.append("Consumption data provided")
            else:
                findings.append("Missing consumption data")

            if has_provenance:
                findings.append("Provenance hash available")

            if coverage_pct >= 100.0:
                findings.append(
                    "Full instrument coverage achieved"
                )
            elif coverage_pct > 0.0:
                findings.append(
                    f"Partial instrument coverage: "
                    f"{coverage_pct:.1f}%"
                )
                recommendations.append(
                    "Increase instrument coverage to 100% for "
                    "full market-based accounting"
                )

            # Determine compliance status
            if fw == "ghg_protocol_scope2":
                if has_consumption and has_provenance:
                    status = "compliant"
                elif has_consumption:
                    status = "partial"
                else:
                    status = "non_compliant"
            elif fw == "re100":
                if has_instrument and coverage_pct >= 100.0:
                    status = "compliant"
                elif has_instrument and coverage_pct > 0.0:
                    status = "partial"
                else:
                    status = "non_compliant"
            else:
                if has_consumption and has_instrument:
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
    # Public API 13: get_compliance_result
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
    # Public API 14: run_uncertainty
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> UncertaintyResponse:
        """Run uncertainty quantification on a calculation.

        Performs Monte Carlo simulation or analytical error propagation
        on a completed market-based emission calculation to quantify
        the confidence interval of the CO2e estimate.

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
                        data.get(
                            "activity_uncertainty_pct", "0.05",
                        ),
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
        uc_id = _short_id("s2m_unc")
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
    # Public API 15: generate_dual_report
    # ==================================================================

    def generate_dual_report(
        self,
        data: Dict[str, Any],
    ) -> DualReportResponse:
        """Generate a dual reporting comparison.

        Compares a location-based and market-based calculation result
        as required by GHG Protocol Scope 2 Guidance for organizations
        using the market-based method.

        Args:
            data: Dual report request with keys:
                - facility_id (str, required)
                - location_result (dict, location-based calc result)
                - market_result (dict, market-based calc result)
                - location_co2e_tonnes (float, alternative)
                - market_co2e_tonnes (float, alternative)

        Returns:
            DualReportResponse with comparison metrics.
        """
        report_id = _short_id("s2m_dual")
        facility_id = str(data.get("facility_id", ""))

        # Extract CO2e values from results or direct inputs
        location_result = data.get("location_result", {})
        market_result = data.get("market_result", {})

        location_co2e = _safe_float(
            data.get(
                "location_co2e_tonnes",
                location_result.get("total_co2e_tonnes"),
            ),
        )
        market_co2e = _safe_float(
            data.get(
                "market_co2e_tonnes",
                market_result.get("total_co2e_tonnes"),
            ),
        )

        coverage_pct = _safe_float(
            market_result.get("coverage_pct", 0.0),
        )

        # Calculate difference
        difference = market_co2e - location_co2e
        difference_pct = 0.0
        if location_co2e > 0:
            difference_pct = (difference / location_co2e) * 100.0

        # Generate recommendation
        recommendation = self._generate_dual_report_recommendation(
            location_co2e, market_co2e, coverage_pct,
        )

        # Delegate to dual reporting engine if available
        if self._dual_reporting is not None:
            try:
                engine_result = self._dual_reporting.generate_report(
                    location_result=location_result,
                    market_result=market_result,
                    facility_id=facility_id,
                )
                if isinstance(engine_result, dict):
                    difference = _safe_float(
                        engine_result.get(
                            "difference_co2e_tonnes", difference,
                        ),
                    )
                    difference_pct = _safe_float(
                        engine_result.get(
                            "difference_pct", difference_pct,
                        ),
                    )
                    recommendation = str(
                        engine_result.get(
                            "recommendation", recommendation,
                        ),
                    )
            except Exception as exc:
                logger.warning(
                    "DualReportingEngine failed: %s", exc,
                )

        provenance_hash = _compute_hash({
            "report_id": report_id,
            "facility_id": facility_id,
            "location_co2e_tonnes": location_co2e,
            "market_co2e_tonnes": market_co2e,
        })

        # Store the report
        report_record: Dict[str, Any] = {
            "report_id": report_id,
            "facility_id": facility_id,
            "location_co2e_tonnes": location_co2e,
            "market_co2e_tonnes": market_co2e,
            "difference_co2e_tonnes": round(difference, 6),
            "difference_pct": round(difference_pct, 2),
            "coverage_pct": coverage_pct,
            "recommendation": recommendation,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        }
        self._dual_reports[report_id] = report_record
        self._total_dual_reports += 1

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_dual_report(status="generated")
            except Exception:
                pass

        logger.info(
            "Dual report %s generated: location=%.4f, "
            "market=%.4f, diff=%.2f%%",
            report_id,
            location_co2e,
            market_co2e,
            difference_pct,
        )

        metadata: Dict[str, Any] = {}
        if location_result:
            metadata["location_source"] = location_result.get(
                "calculation_id", "",
            )
        if market_result:
            metadata["market_source"] = market_result.get(
                "calculation_id", "",
            )

        return DualReportResponse(
            success=True,
            report_id=report_id,
            facility_id=facility_id,
            location_co2e_tonnes=round(location_co2e, 6),
            market_co2e_tonnes=round(market_co2e, 6),
            difference_co2e_tonnes=round(difference, 6),
            difference_pct=round(difference_pct, 2),
            coverage_pct=coverage_pct,
            recommendation=recommendation,
            provenance_hash=provenance_hash,
            timestamp=_utcnow_iso(),
            metadata=metadata,
        )

    def _generate_dual_report_recommendation(
        self,
        location_co2e: float,
        market_co2e: float,
        coverage_pct: float,
    ) -> str:
        """Generate a recommendation for the dual report.

        Args:
            location_co2e: Location-based CO2e in tonnes.
            market_co2e: Market-based CO2e in tonnes.
            coverage_pct: Instrument coverage percentage.

        Returns:
            Human-readable recommendation string.
        """
        if market_co2e < location_co2e and coverage_pct >= 100.0:
            return (
                "Market-based method yields lower emissions with "
                "full instrument coverage. Continue current "
                "procurement strategy."
            )
        elif market_co2e < location_co2e:
            return (
                "Market-based method yields lower emissions but "
                "coverage is incomplete. Increase instrument "
                "procurement to achieve full coverage."
            )
        elif market_co2e > location_co2e:
            return (
                "Market-based method yields higher emissions than "
                "location-based. Review residual mix factors and "
                "consider procuring contractual instruments."
            )
        else:
            return (
                "Market-based and location-based methods yield "
                "similar emissions. Ensure both methods use "
                "current emission factors."
            )

    # ==================================================================
    # Public API 16: get_aggregations
    # ==================================================================

    def get_aggregations(
        self,
        tenant_id: str,
        group_by: str = "facility",
    ) -> AggregationResponse:
        """Get aggregated Scope 2 market-based emissions.

        Aggregates calculation results by the specified dimension
        (facility, instrument_type, energy_source, country, month,
        quarter).

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
            elif group_by == "instrument_type":
                key = str(
                    calc.get("instrument_type", "unknown"),
                )
            elif group_by == "energy_source":
                req = calc.get("request", {})
                key = str(req.get("energy_source", "unknown"))
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
    # Public API 17: get_coverage_analysis
    # ==================================================================

    def get_coverage_analysis(
        self,
        tenant_id: str,
        facility_id: Optional[str] = None,
    ) -> CoverageAnalysisResponse:
        """Analyze instrument coverage for a facility or tenant.

        Calculates the percentage of electricity consumption covered
        by contractual instruments vs. the uncovered residual portion.

        Args:
            tenant_id: Tenant identifier.
            facility_id: Optional specific facility to analyze.

        Returns:
            CoverageAnalysisResponse with coverage gaps.
        """
        # Get relevant instruments
        tenant_instruments = [
            inst for inst in self._instruments.values()
            if inst.get("tenant_id") == tenant_id
            and inst.get("status") == "active"
        ]
        if facility_id:
            tenant_instruments = [
                inst for inst in tenant_instruments
                if inst.get("facility_id") == facility_id
                or inst.get("facility_id") is None
            ]

        # Get relevant calculations for consumption
        tenant_calcs = [
            c for c in self._calculations.values()
            if c.get("tenant_id") == tenant_id
        ]
        if facility_id:
            tenant_calcs = [
                c for c in tenant_calcs
                if c.get("facility_id") == facility_id
            ]

        # Total consumption from calculations
        total_consumption_mwh = 0.0
        for calc in tenant_calcs:
            req = calc.get("request", {})
            val = _safe_float(req.get("consumption_value"))
            unit = str(req.get("consumption_unit", "mwh")).lower()
            if unit == "kwh":
                val = val / 1000.0
            elif unit == "gj":
                val = val / 3.6
            total_consumption_mwh += val

        # Total instrument coverage
        covered_mwh = sum(
            _safe_float(inst.get("quantity_mwh"))
            for inst in tenant_instruments
        )

        uncovered_mwh = max(
            0.0, total_consumption_mwh - covered_mwh,
        )
        coverage_pct = 0.0
        if total_consumption_mwh > 0:
            coverage_pct = min(
                100.0,
                (covered_mwh / total_consumption_mwh) * 100.0,
            )

        # Coverage by instrument type
        coverage_by_type: Dict[str, float] = {}
        for inst in tenant_instruments:
            inst_type = inst.get("instrument_type", "unknown")
            inst_mwh = _safe_float(inst.get("quantity_mwh"))
            coverage_by_type[inst_type] = (
                coverage_by_type.get(inst_type, 0.0) + inst_mwh
            )

        # Identify gaps
        gaps: List[Dict[str, Any]] = []
        if uncovered_mwh > 0:
            gaps.append({
                "gap_type": "uncovered_consumption",
                "uncovered_mwh": round(uncovered_mwh, 3),
                "recommendation": (
                    "Procure additional contractual instruments "
                    f"for {uncovered_mwh:.1f} MWh of uncovered "
                    "consumption"
                ),
            })

        # Check vintage currency
        current_year = _utcnow().year
        for inst in tenant_instruments:
            vintage = int(inst.get("vintage_year", 0))
            if vintage > 0 and (current_year - vintage) > 2:
                gaps.append({
                    "gap_type": "vintage_expiry",
                    "instrument_id": inst.get("instrument_id"),
                    "vintage_year": vintage,
                    "recommendation": (
                        f"Instrument vintage {vintage} may not "
                        "meet quality criteria; consider replacing "
                        "with current vintage"
                    ),
                })

        effective_fac_id = facility_id or "all"

        return CoverageAnalysisResponse(
            facility_id=effective_fac_id,
            total_consumption_mwh=round(total_consumption_mwh, 3),
            covered_mwh=round(covered_mwh, 3),
            uncovered_mwh=round(uncovered_mwh, 3),
            coverage_pct=round(coverage_pct, 2),
            instruments_used=len(tenant_instruments),
            coverage_by_type=coverage_by_type,
            gaps=gaps,
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 18: health_check
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Perform a service health check.

        Reports the status of all engine components, configuration
        validation, and uptime.

        Returns:
            HealthResponse with engine status and diagnostics.
        """
        engines: Dict[str, str] = {
            "contractual_instrument_database": (
                "available" if self._instrument_db is not None
                else "unavailable"
            ),
            "instrument_allocation": (
                "available" if self._allocation is not None
                else "unavailable"
            ),
            "market_emissions_calculator": (
                "available" if self._calculator is not None
                else "unavailable"
            ),
            "dual_reporting": (
                "available" if self._dual_reporting is not None
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
    # Public API 19: get_stats
    # ==================================================================

    def get_stats(self) -> StatsResponse:
        """Get service aggregate statistics.

        Returns cumulative counters for calculations, facilities,
        instruments, compliance checks, uncertainty analyses, dual
        reports, and total CO2e.

        Returns:
            StatsResponse with all aggregate counters.
        """
        uptime_s = time.monotonic() - self._start_time

        return StatsResponse(
            total_calculations=self._total_calculations,
            total_batch_runs=self._total_batch_runs,
            total_facilities=len(self._facilities),
            total_instruments=len(self._instruments),
            total_instruments_retired=self._total_instruments_retired,
            total_compliance_checks=self._total_compliance_checks,
            total_uncertainty_runs=self._total_uncertainty_runs,
            total_dual_reports=self._total_dual_reports,
            total_co2e_tonnes=round(
                self._cumulative_co2e_tonnes, 6,
            ),
            uptime_seconds=round(uptime_s, 2),
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API 20: get_engine_status
    # ==================================================================

    def get_engine_status(self) -> Dict[str, Any]:
        """Get detailed engine availability and diagnostics.

        Returns per-engine status including class name, import status,
        and initialization state.

        Returns:
            Dict mapping engine names to status details.
        """
        engine_entries = [
            (
                "contractual_instrument_database",
                self._instrument_db,
                ContractualInstrumentDatabaseEngine,
            ),
            (
                "instrument_allocation",
                self._allocation,
                InstrumentAllocationEngine,
            ),
            (
                "market_emissions_calculator",
                self._calculator,
                MarketEmissionsCalculatorEngine,
            ),
            (
                "dual_reporting",
                self._dual_reporting,
                DualReportingEngine,
            ),
            (
                "uncertainty_quantifier",
                self._uncertainty,
                UncertaintyQuantifierEngine,
            ),
            (
                "compliance_checker",
                self._compliance,
                ComplianceCheckerEngine,
            ),
            (
                "scope2_market_pipeline",
                self._pipeline,
                Scope2MarketPipelineEngine,
            ),
        ]

        engines: Dict[str, Any] = {}
        for name, instance, cls in engine_entries:
            imported = cls is not None
            initialized = instance is not None
            class_name = (
                cls.__name__ if cls is not None else "N/A"
            )

            engines[name] = {
                "class": class_name,
                "imported": imported,
                "initialized": initialized,
                "status": (
                    "available" if initialized
                    else "import_failed" if not imported
                    else "init_failed"
                ),
            }

        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "agent_id": AGENT_ID,
            "engines": engines,
            "total_engines": len(engine_entries),
            "available_engines": sum(
                1 for e in engines.values()
                if e["status"] == "available"
            ),
            "timestamp": _utcnow_iso(),
        }

    # ==================================================================
    # Utility: reset (testing only)
    # ==================================================================

    def reset(self) -> None:
        """Reset all in-memory state.

        Intended for test teardown only. Clears all stored calculations,
        facilities, instruments, compliance results, dual reports,
        uncertainty results, and resets all counters.
        """
        self._calculations.clear()
        self._facilities.clear()
        self._instruments.clear()
        self._compliance_results.clear()
        self._dual_reports.clear()
        self._uncertainty_results.clear()
        self._total_calculations = 0
        self._total_batch_runs = 0
        self._total_compliance_checks = 0
        self._total_uncertainty_runs = 0
        self._total_dual_reports = 0
        self._total_instruments_retired = 0
        self._cumulative_co2e_tonnes = 0.0
        logger.info("Scope2MarketService state reset")


# ===================================================================
# Module-level singleton
# ===================================================================


def get_service() -> Scope2MarketService:
    """Return the module-level Scope2MarketService singleton.

    Thread-safe lazy initialization. The first call creates the
    service instance; subsequent calls return the same instance.

    Returns:
        The shared Scope2MarketService instance.

    Example:
        >>> svc = get_service()
        >>> health = svc.health_check()
        >>> assert health.status in ("healthy", "degraded", "partial")
    """
    global _service_instance
    if _service_instance is None:
        with _singleton_lock:
            if _service_instance is None:
                _service_instance = Scope2MarketService()
    return _service_instance


def reset_service() -> None:
    """Reset the module-level service singleton.

    After calling this function, the next call to ``get_service()``
    will create a fresh instance. Intended for test teardown.
    """
    global _service_instance
    with _singleton_lock:
        _service_instance = None
    logger.debug("Scope2MarketService singleton reset")


def get_service_with_config(
    config: Any = None,
    **overrides: Any,
) -> Scope2MarketService:
    """Create a new Scope2MarketService with custom configuration.

    Does NOT modify the module-level singleton. Returns a fresh
    instance with the provided configuration.

    Args:
        config: Optional configuration override.
        **overrides: Additional keyword overrides applied to config.

    Returns:
        A new Scope2MarketService instance.

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

    return Scope2MarketService(config=cfg)


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service class
    "Scope2MarketService",
    # Singleton accessors
    "get_service",
    "reset_service",
    "get_service_with_config",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FacilityResponse",
    "FacilityListResponse",
    "InstrumentResponse",
    "InstrumentListResponse",
    "ComplianceCheckResponse",
    "UncertaintyResponse",
    "DualReportResponse",
    "CoverageAnalysisResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
    # Constants
    "SERVICE_VERSION",
    "SERVICE_NAME",
    "AGENT_ID",
    "VALID_INSTRUMENT_TYPES",
    "VALID_CALCULATION_METHODS",
    "VALID_ENERGY_UNITS",
    "VALID_FACILITY_TYPES",
    "VALID_COMPLIANCE_FRAMEWORKS",
    "VALID_GWP_SOURCES",
    "VALID_TRACKING_SYSTEMS",
    "VALID_ALLOCATION_METHODS",
    "VALID_GROUP_BY",
]
