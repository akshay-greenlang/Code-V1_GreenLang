# -*- coding: utf-8 -*-
"""
Fuel & Energy Activities Service Setup - AGENT-MRV-016
=======================================================

Service facade for the Fuel & Energy Activities Agent (GL-MRV-S3-003).

Provides ``configure_fuel_energy_activities(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``FuelEnergyActivitiesService`` facade class that aggregates all 7 engines:

    1. FuelEnergyDatabaseEngine           - WTT/upstream EF lookup, grid region
    2. Activity3aCalculatorEngine         - Upstream fuel emissions (well-to-tank)
    3. Activity3bCalculatorEngine         - Upstream electricity emissions
    4. Activity3cCalculatorEngine         - T&D loss emissions
    5. Activity3dCalculatorEngine         - Sold electricity emissions (utilities)
    6. ComplianceCheckerEngine            - Multi-framework regulatory compliance
    7. FuelEnergyPipelineEngine           - Orchestrated 10-stage pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.fuel_energy_activities.setup import configure_fuel_energy_activities
    >>> app = FastAPI()
    >>> configure_fuel_energy_activities(app)

    >>> from greenlang.fuel_energy_activities.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate_activity_3a(
    ...     fuel_record=record,
    ...     gwp_source=GWPSource.AR6,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-016 Fuel & Energy Activities (GL-MRV-S3-003)
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
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, APIRouter, HTTPException, Query, Path
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    APIRouter = None  # type: ignore[assignment, misc]
    HTTPException = None  # type: ignore[assignment, misc]
    Query = None  # type: ignore[assignment, misc]
    Path = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Domain model imports (always available -- part of this package)
# ---------------------------------------------------------------------------

try:
    from greenlang.fuel_energy_activities.models import (
        AGENT_ID,
        VERSION,
        TABLE_PREFIX,
        ZERO,
        ONE,
        ONE_HUNDRED,
        ONE_THOUSAND,
        DECIMAL_PLACES,
        MAX_FUEL_RECORDS,
        MAX_ELECTRICITY_RECORDS,
        MAX_BATCH_PERIODS,
        MAX_FRAMEWORKS,
        DEFAULT_CONFIDENCE_LEVEL,
        # Enumerations
        CalculationMethod,
        FuelType,
        FuelCategory,
        EnergyType,
        ActivityType,
        WTTFactorSource,
        GridRegionType,
        TDLossSource,
        SupplierDataSource,
        AllocationMethod,
        CurrencyCode,
        DQIDimension,
        DQIScore,
        UncertaintyMethod,
        ComplianceFramework,
        ComplianceStatus,
        PipelineStage,
        ExportFormat,
        BatchStatus,
        GWPSource,
        EmissionGas,
        AccountingMethod,
        # Data models
        FuelConsumptionRecord,
        ElectricityConsumptionRecord,
        WTTEmissionFactor,
        UpstreamElectricityFactor,
        TDLossFactor,
        SupplierFuelData,
        Activity3aResult,
        Activity3bResult,
        Activity3cResult,
        Activity3dResult,
        CalculationResult,
        GasBreakdown,
        DQIAssessment,
        UncertaintyResult,
        ComplianceCheckResult,
        ComplianceFinding,
        PipelineResult,
        BatchRequest,
        BatchResult,
        AggregationResult,
        ExportRequest,
        MaterialityResult,
        HotSpotResult,
        YoYDecomposition,
        ProvenanceRecord,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Provide sentinel values so the module can still be imported
    AGENT_ID = "GL-MRV-S3-003"
    VERSION = "1.0.0"
    ONE = Decimal("1")
    ZERO = Decimal("0")

# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.fuel_energy_activities.config import FuelEnergyActivitiesConfig
except ImportError:
    FuelEnergyActivitiesConfig = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.fuel_energy_activities.fuel_energy_database import (
        FuelEnergyDatabaseEngine,
    )
except ImportError:
    FuelEnergyDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.activity_3a_calculator import (
        Activity3aCalculatorEngine,
    )
except ImportError:
    Activity3aCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.activity_3b_calculator import (
        Activity3bCalculatorEngine,
    )
except ImportError:
    Activity3bCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.activity_3c_calculator import (
        Activity3cCalculatorEngine,
    )
except ImportError:
    Activity3cCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.activity_3d_calculator import (
        Activity3dCalculatorEngine,
    )
except ImportError:
    Activity3dCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.fuel_energy_pipeline import (
        FuelEnergyPipelineEngine,
    )
except ImportError:
    FuelEnergyPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.provenance import (
        FuelEnergyProvenanceTracker,
    )
except ImportError:
    FuelEnergyProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.fuel_energy_activities.metrics import FuelEnergyMetrics
except ImportError:
    FuelEnergyMetrics = None  # type: ignore[assignment, misc]


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


def _short_id(prefix: str = "fea") -> str:
    """Generate a short unique identifier with a given prefix.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        A string like ``fea_a1b2c3d4e5f6``.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Object to hash. Supports Pydantic models (via
            ``model_dump``), dicts, and other JSON-serializable types.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _quantize(value: Decimal, places: int = 8) -> Decimal:
    """Quantize a Decimal to the given number of decimal places.

    Args:
        value: Decimal value to quantize.
        places: Number of decimal places.

    Returns:
        Quantized Decimal value.
    """
    quantizer = Decimal(10) ** -places
    return value.quantize(quantizer, rounding=ROUND_HALF_UP)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default on failure.

    Args:
        value: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Float representation of value, or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError, ArithmeticError):
        return default


def _safe_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Default Decimal if conversion fails.

    Returns:
        Decimal representation of value, or default.
    """
    try:
        return Decimal(str(value))
    except Exception:
        return default


# ===================================================================
# Default compliance frameworks for Category 3
# ===================================================================

DEFAULT_COMPLIANCE_FRAMEWORKS: List[str] = [
    "ghg_protocol",
    "csrd_esrs",
    "cdp",
    "sbti",
    "sb253",
    "gri",
    "iso_14064",
]


# ===================================================================
# Lightweight Pydantic response models (14 models)
# ===================================================================


class Activity3aResponse(BaseModel):
    """Response for Activity 3a upstream fuel emissions calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    activity_type: str = Field(default="activity_3a")
    fuel_type: str = Field(default="")
    fuel_quantity: float = Field(default=0.0)
    fuel_unit: str = Field(default="")
    wtt_emissions_tco2e: float = Field(default=0.0)
    co2_tco2e: float = Field(default=0.0)
    ch4_tco2e: float = Field(default=0.0)
    n2o_tco2e: float = Field(default=0.0)
    ef_source: str = Field(default="")
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class Activity3bResponse(BaseModel):
    """Response for Activity 3b upstream electricity emissions calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    activity_type: str = Field(default="activity_3b")
    electricity_kwh: float = Field(default=0.0)
    country_code: str = Field(default="")
    accounting_method: str = Field(default="location_based")
    upstream_emissions_tco2e: float = Field(default=0.0)
    co2_tco2e: float = Field(default=0.0)
    ch4_tco2e: float = Field(default=0.0)
    n2o_tco2e: float = Field(default=0.0)
    ef_source: str = Field(default="")
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class Activity3cResponse(BaseModel):
    """Response for Activity 3c T&D loss emissions calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    activity_type: str = Field(default="activity_3c")
    electricity_kwh: float = Field(default=0.0)
    country_code: str = Field(default="")
    accounting_method: str = Field(default="location_based")
    td_loss_percentage: float = Field(default=0.0)
    td_loss_kwh: float = Field(default=0.0)
    td_emissions_tco2e: float = Field(default=0.0)
    co2_tco2e: float = Field(default=0.0)
    ch4_tco2e: float = Field(default=0.0)
    n2o_tco2e: float = Field(default=0.0)
    ef_source: str = Field(default="")
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class CalculationResponse(BaseModel):
    """Response for a full pipeline calculation (3a + 3b + 3c + 3d)."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="hybrid")
    total_fuel_records: int = Field(default=0)
    total_electricity_records: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    activity_3a_emissions_tco2e: float = Field(default=0.0)
    activity_3b_emissions_tco2e: float = Field(default=0.0)
    activity_3c_emissions_tco2e: float = Field(default=0.0)
    activity_3d_emissions_tco2e: float = Field(default=0.0)
    coverage_level: str = Field(default="minimal")
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchResponse(BaseModel):
    """Response for a batch calculation across multiple periods."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    batch_id: str = Field(default="")
    status: str = Field(default="completed")
    total_periods: int = Field(default=0)
    completed: int = Field(default=0)
    failed: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class WTTFactorResponse(BaseModel):
    """Response for a well-to-tank emission factor lookup."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    fuel_type: str = Field(default="")
    source: str = Field(default="")
    wtt_factor_kgco2e_per_kwh: float = Field(default=0.0)
    co2_kg_per_kwh: float = Field(default=0.0)
    ch4_kg_per_kwh: float = Field(default=0.0)
    n2o_kg_per_kwh: float = Field(default=0.0)
    heating_value_kwh_per_unit: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class UpstreamEFResponse(BaseModel):
    """Response for an upstream electricity emission factor lookup."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    country_code: str = Field(default="")
    accounting_method: str = Field(default="location_based")
    upstream_factor_kgco2e_per_kwh: float = Field(default=0.0)
    co2_kg_per_kwh: float = Field(default=0.0)
    ch4_kg_per_kwh: float = Field(default=0.0)
    n2o_kg_per_kwh: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class TDLossResponse(BaseModel):
    """Response for a T&D loss factor lookup."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    country_code: str = Field(default="")
    td_loss_percentage: float = Field(default=0.0)
    source: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)


class ComplianceResponse(BaseModel):
    """Response for a regulatory compliance check."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    compliance_id: str = Field(default="")
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    not_applicable: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)


class DQIResponse(BaseModel):
    """Response for data quality indicator assessment."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    dqi_composite: float = Field(default=0.0)
    quality_tier: str = Field(default="tier_3")
    technology_score: float = Field(default=0.0)
    temporal_score: float = Field(default=0.0)
    geographic_score: float = Field(default=0.0)
    completeness_score: float = Field(default=0.0)
    reliability_score: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class UncertaintyResponse(BaseModel):
    """Response for uncertainty quantification."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    method: str = Field(default="pedigree")
    confidence_level: float = Field(default=95.0)
    lower_bound_tco2e: float = Field(default=0.0)
    upper_bound_tco2e: float = Field(default=0.0)
    uncertainty_percentage: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class AggregationResponse(BaseModel):
    """Response for an aggregated emissions summary."""

    model_config = ConfigDict(frozen=True)

    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_emissions_tco2e: float = Field(default=0.0)
    total_fuel_consumption: float = Field(default=0.0)
    total_electricity_kwh: float = Field(default=0.0)
    facility_count: int = Field(default=0)
    coverage_level: str = Field(default="minimal")
    period: str = Field(default="annual")
    timestamp: str = Field(default_factory=_utcnow_iso)


class HotSpotResponse(BaseModel):
    """Response for hot-spot analysis."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    top_fuels: List[Dict[str, Any]] = Field(default_factory=list)
    top_facilities: List[Dict[str, Any]] = Field(default_factory=list)
    top_suppliers: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)


class MaterialityResponse(BaseModel):
    """Response for materiality assessment."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    category_3_emissions_tco2e: float = Field(default=0.0)
    scope_1_emissions_tco2e: float = Field(default=0.0)
    scope_2_emissions_tco2e: float = Field(default=0.0)
    scope_3_total_emissions_tco2e: float = Field(default=0.0)
    materiality_percentage: float = Field(default=0.0)
    materiality_tier: str = Field(default="low")
    timestamp: str = Field(default_factory=_utcnow_iso)


# ===================================================================
# FuelEnergyActivitiesService facade
# ===================================================================


class FuelEnergyActivitiesService:
    """Unified facade over the Fuel & Energy Activities Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20+ service operations covering
    upstream fuel and electricity emissions (WTT), T&D losses, and
    sold electricity for GHG Protocol Scope 3 Category 3 emissions.

    Each method records provenance via SHA-256 hashing and tracks
    processing time for observability.

    The service is implemented as a thread-safe singleton using
    ``__new__`` with ``threading.RLock`` for double-checked locking.
    All mutable state (counters, in-memory caches) is protected by
    the reentrant lock.

    Engines are lazily initialized on first access if the corresponding
    module is importable.  Missing engines degrade gracefully -- the
    service reports them as ``"unavailable"`` in health checks and
    raises descriptive errors when their methods are called.

    Example:
        >>> service = FuelEnergyActivitiesService()
        >>> from greenlang.fuel_energy_activities.models import (
        ...     FuelConsumptionRecord, FuelType, GWPSource,
        ... )
        >>> from decimal import Decimal
        >>> fuel_rec = FuelConsumptionRecord(
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     quantity=Decimal("1000"),
        ...     quantity_unit="m3",
        ...     facility_id="FAC-001",
        ... )
        >>> result = service.calculate_activity_3a(
        ...     fuel_record=fuel_rec,
        ...     gwp_source=GWPSource.AR6,
        ... )
        >>> assert result.success

    Attributes:
        config: Service configuration singleton.
        _db_engine: Engine 1 - WTT/upstream factor lookups.
        _activity_3a_engine: Engine 2 - upstream fuel (WTT) calculations.
        _activity_3b_engine: Engine 3 - upstream electricity calculations.
        _activity_3c_engine: Engine 4 - T&D loss calculations.
        _activity_3d_engine: Engine 5 - sold electricity calculations.
        _compliance_checker_engine: Engine 6 - regulatory compliance.
        _pipeline_engine: Engine 7 - orchestrated pipeline.
    """

    _instance: Optional["FuelEnergyActivitiesService"] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> "FuelEnergyActivitiesService":
        """Create or return the singleton instance.

        Uses double-checked locking with ``threading.RLock`` to ensure
        exactly one instance is created even under concurrent access.

        Returns:
            The singleton FuelEnergyActivitiesService instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the Fuel & Energy Activities Service facade.

        Only performs initialization once (guarded by ``_initialized``).
        Subsequent calls to ``__init__`` are no-ops, preserving the
        singleton's state.

        Args:
            config: Optional FuelEnergyActivitiesConfig instance. If None,
                a default config is created if FuelEnergyActivitiesConfig
                is available.
        """
        if self.__class__._initialized:
            return
        with self.__class__._lock:
            if self.__class__._initialized:
                return
            self._do_init(config)
            self.__class__._initialized = True

    def _do_init(self, config: Optional[Any] = None) -> None:
        """Internal initialization logic (called once).

        Sets up configuration, engine placeholders, in-memory stores,
        and statistics counters.  Then attempts to initialize each engine.

        Args:
            config: Optional configuration instance.
        """
        # Configuration
        self.config: Any = config
        if self.config is None and FuelEnergyActivitiesConfig is not None:
            try:
                self.config = FuelEnergyActivitiesConfig()
            except Exception as exc:
                logger.warning(
                    "FuelEnergyActivitiesConfig init failed: %s", exc,
                )

        self._start_time: float = time.monotonic()

        # Engine placeholders (lazy-initialized)
        self._db_engine: Any = None
        self._activity_3a_engine: Any = None
        self._activity_3b_engine: Any = None
        self._activity_3c_engine: Any = None
        self._activity_3d_engine: Any = None
        self._compliance_checker_engine: Any = None
        self._pipeline_engine: Any = None

        # Provenance and metrics
        self._provenance_tracker: Any = None
        self._metrics: Any = None

        # In-memory result caches
        self._calculation_results: List[Dict[str, Any]] = []
        self._batch_results: List[Dict[str, Any]] = []
        self._compliance_results: List[Dict[str, Any]] = []
        self._export_results: List[Dict[str, Any]] = []

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_activity_3a: int = 0
        self._total_activity_3b: int = 0
        self._total_activity_3c: int = 0
        self._total_activity_3d: int = 0
        self._total_compliance_checks: int = 0
        self._total_exports: int = 0

        # Initialize engines
        self._init_engines()

        logger.info(
            "FuelEnergyActivitiesService facade created "
            "(agent=%s, version=%s)",
            AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-003",
            VERSION if MODELS_AVAILABLE else "1.0.0",
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance for testing.

        Clears the singleton so the next call to ``__new__`` creates
        a fresh instance.  This method is intended for test fixtures
        only and must not be called in production code.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False
        logger.info("FuelEnergyActivitiesService singleton reset")

    # ------------------------------------------------------------------
    # Engine properties (read-only)
    # ------------------------------------------------------------------

    @property
    def db_engine(self) -> Any:
        """Get the FuelEnergyDatabaseEngine instance (Engine 1)."""
        return self._db_engine

    @property
    def activity_3a_engine(self) -> Any:
        """Get the Activity3aCalculatorEngine instance (Engine 2)."""
        return self._activity_3a_engine

    @property
    def activity_3b_engine(self) -> Any:
        """Get the Activity3bCalculatorEngine instance (Engine 3)."""
        return self._activity_3b_engine

    @property
    def activity_3c_engine(self) -> Any:
        """Get the Activity3cCalculatorEngine instance (Engine 4)."""
        return self._activity_3c_engine

    @property
    def activity_3d_engine(self) -> Any:
        """Get the Activity3dCalculatorEngine instance (Engine 5)."""
        return self._activity_3d_engine

    @property
    def compliance_checker_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance (Engine 6)."""
        return self._compliance_checker_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the FuelEnergyPipelineEngine instance (Engine 7)."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialize all 7 SDK engines.

        Each engine is initialized independently with graceful
        degradation.  If an engine fails to initialize, it remains
        ``None`` and the service logs a warning.  The pipeline engine
        (E7) receives references to all upstream engines.
        """
        # E1: FuelEnergyDatabaseEngine
        self._init_single_engine(
            "FuelEnergyDatabaseEngine",
            FuelEnergyDatabaseEngine,
            "_db_engine",
        )

        # E2: Activity3aCalculatorEngine
        self._init_single_engine(
            "Activity3aCalculatorEngine",
            Activity3aCalculatorEngine,
            "_activity_3a_engine",
        )

        # E3: Activity3bCalculatorEngine
        self._init_single_engine(
            "Activity3bCalculatorEngine",
            Activity3bCalculatorEngine,
            "_activity_3b_engine",
        )

        # E4: Activity3cCalculatorEngine
        self._init_single_engine(
            "Activity3cCalculatorEngine",
            Activity3cCalculatorEngine,
            "_activity_3c_engine",
        )

        # E5: Activity3dCalculatorEngine
        self._init_single_engine(
            "Activity3dCalculatorEngine",
            Activity3dCalculatorEngine,
            "_activity_3d_engine",
        )

        # E6: ComplianceCheckerEngine
        self._init_single_engine(
            "ComplianceCheckerEngine",
            ComplianceCheckerEngine,
            "_compliance_checker_engine",
        )

        # E7: FuelEnergyPipelineEngine (receives upstream engines)
        if FuelEnergyPipelineEngine is not None:
            try:
                self._pipeline_engine = FuelEnergyPipelineEngine()
                logger.info("FuelEnergyPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "FuelEnergyPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("FuelEnergyPipelineEngine not available")

        # Provenance tracker
        if FuelEnergyProvenanceTracker is not None:
            try:
                self._provenance_tracker = FuelEnergyProvenanceTracker()
                logger.info("FuelEnergyProvenanceTracker initialized")
            except Exception as exc:
                logger.warning(
                    "FuelEnergyProvenanceTracker init failed: %s", exc,
                )

        # Metrics collector
        if FuelEnergyMetrics is not None:
            try:
                self._metrics = FuelEnergyMetrics()
                logger.info("FuelEnergyMetrics initialized")
            except Exception as exc:
                logger.warning(
                    "FuelEnergyMetrics init failed: %s", exc,
                )

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        attr_name: str,
    ) -> None:
        """Initialize a single engine with graceful degradation.

        All engine classes in the Fuel & Energy Activities agent use a
        thread-safe singleton pattern (``__new__`` with ``_instance``),
        so calling the constructor returns the shared instance.

        Args:
            name: Human-readable engine name for logging.
            engine_class: Engine class or None if import failed.
            attr_name: Instance attribute name to set.
        """
        if engine_class is not None:
            try:
                setattr(self, attr_name, engine_class())
                logger.info("%s initialized", name)
            except Exception as exc:
                logger.warning("%s init failed: %s", name, exc)
        else:
            logger.warning("%s not available", name)

    # ==================================================================
    # Public API: Main calculation methods (20 methods)
    # ==================================================================

    def calculate_activity_3a(
        self,
        fuel_record: Any,
        gwp_source: Any = None,
    ) -> Activity3aResponse:
        """Calculate upstream fuel emissions (well-to-tank / WTT).

        Applies WTT emission factors to fuel consumption quantities
        to compute upstream extraction, refining, and transportation
        emissions (Activity 3a).

        Args:
            fuel_record: FuelConsumptionRecord instance.
            gwp_source: GWPSource enum value (AR5 or AR6). Defaults to AR6.

        Returns:
            Activity3aResponse with WTT emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("fea_3a")

        logger.info(
            "calculate_activity_3a: fuel_type=%s, quantity=%s",
            fuel_record.fuel_type if hasattr(fuel_record, "fuel_type") else "unknown",
            fuel_record.quantity if hasattr(fuel_record, "quantity") else 0,
        )

        try:
            if self._activity_3a_engine is None:
                raise RuntimeError(
                    "Activity3aCalculatorEngine is not available"
                )

            result = self._activity_3a_engine.calculate_single(
                record=fuel_record,
                gwp_source=gwp_source,
            )

            with self.__class__._lock:
                self._total_activity_3a += 1

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "fuel_record": fuel_record,
                "result": result,
            })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return Activity3aResponse(
                success=True,
                calculation_id=calc_id,
                activity_type="activity_3a",
                fuel_type=str(fuel_record.fuel_type) if hasattr(fuel_record, "fuel_type") else "",
                fuel_quantity=_safe_float(fuel_record.quantity) if hasattr(fuel_record, "quantity") else 0.0,
                fuel_unit=str(fuel_record.quantity_unit) if hasattr(fuel_record, "quantity_unit") else "",
                wtt_emissions_tco2e=_safe_float(result.wtt_emissions_tco2e) if hasattr(result, "wtt_emissions_tco2e") else 0.0,
                co2_tco2e=_safe_float(result.co2_tco2e) if hasattr(result, "co2_tco2e") else 0.0,
                ch4_tco2e=_safe_float(result.ch4_tco2e) if hasattr(result, "ch4_tco2e") else 0.0,
                n2o_tco2e=_safe_float(result.n2o_tco2e) if hasattr(result, "n2o_tco2e") else 0.0,
                ef_source=str(result.ef_source) if hasattr(result, "ef_source") else "",
                dqi_composite=_safe_float(result.dqi_composite) if hasattr(result, "dqi_composite") else None,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate_activity_3a failed: %s", exc, exc_info=True)
            raise

    def calculate_activity_3b(
        self,
        elec_record: Any,
        country_code: str,
        gwp_source: Any = None,
    ) -> Activity3bResponse:
        """Calculate upstream electricity generation emissions.

        Applies upstream electricity emission factors (lifecycle emissions
        from fuel extraction and power generation) to purchased electricity
        (Activity 3b).

        Args:
            elec_record: ElectricityConsumptionRecord instance.
            country_code: ISO 3166-1 alpha-2 country code for grid region.
            gwp_source: GWPSource enum value (AR5 or AR6). Defaults to AR6.

        Returns:
            Activity3bResponse with upstream electricity emissions.
        """
        t0 = time.monotonic()
        calc_id = _short_id("fea_3b")

        logger.info(
            "calculate_activity_3b: country=%s, kwh=%s",
            country_code,
            elec_record.electricity_kwh if hasattr(elec_record, "electricity_kwh") else 0,
        )

        try:
            if self._activity_3b_engine is None:
                raise RuntimeError(
                    "Activity3bCalculatorEngine is not available"
                )

            result = self._activity_3b_engine.calculate_single(
                record=elec_record,
                country_code=country_code,
                gwp_source=gwp_source,
            )

            with self.__class__._lock:
                self._total_activity_3b += 1

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "elec_record": elec_record,
                "country_code": country_code,
                "result": result,
            })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return Activity3bResponse(
                success=True,
                calculation_id=calc_id,
                activity_type="activity_3b",
                electricity_kwh=_safe_float(elec_record.electricity_kwh) if hasattr(elec_record, "electricity_kwh") else 0.0,
                country_code=country_code,
                accounting_method=str(elec_record.accounting_method) if hasattr(elec_record, "accounting_method") else "location_based",
                upstream_emissions_tco2e=_safe_float(result.upstream_emissions_tco2e) if hasattr(result, "upstream_emissions_tco2e") else 0.0,
                co2_tco2e=_safe_float(result.co2_tco2e) if hasattr(result, "co2_tco2e") else 0.0,
                ch4_tco2e=_safe_float(result.ch4_tco2e) if hasattr(result, "ch4_tco2e") else 0.0,
                n2o_tco2e=_safe_float(result.n2o_tco2e) if hasattr(result, "n2o_tco2e") else 0.0,
                ef_source=str(result.ef_source) if hasattr(result, "ef_source") else "",
                dqi_composite=_safe_float(result.dqi_composite) if hasattr(result, "dqi_composite") else None,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate_activity_3b failed: %s", exc, exc_info=True)
            raise

    def calculate_activity_3c(
        self,
        elec_record: Any,
        country_code: str,
        gwp_source: Any = None,
    ) -> Activity3cResponse:
        """Calculate transmission and distribution (T&D) loss emissions.

        Applies T&D loss percentages and grid emission factors to
        purchased electricity to compute line loss emissions (Activity 3c).

        Args:
            elec_record: ElectricityConsumptionRecord instance.
            country_code: ISO 3166-1 alpha-2 country code for grid region.
            gwp_source: GWPSource enum value (AR5 or AR6). Defaults to AR6.

        Returns:
            Activity3cResponse with T&D loss emissions.
        """
        t0 = time.monotonic()
        calc_id = _short_id("fea_3c")

        logger.info(
            "calculate_activity_3c: country=%s, kwh=%s",
            country_code,
            elec_record.electricity_kwh if hasattr(elec_record, "electricity_kwh") else 0,
        )

        try:
            if self._activity_3c_engine is None:
                raise RuntimeError(
                    "Activity3cCalculatorEngine is not available"
                )

            result = self._activity_3c_engine.calculate_single(
                record=elec_record,
                country_code=country_code,
                gwp_source=gwp_source,
            )

            with self.__class__._lock:
                self._total_activity_3c += 1

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "elec_record": elec_record,
                "country_code": country_code,
                "result": result,
            })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return Activity3cResponse(
                success=True,
                calculation_id=calc_id,
                activity_type="activity_3c",
                electricity_kwh=_safe_float(elec_record.electricity_kwh) if hasattr(elec_record, "electricity_kwh") else 0.0,
                country_code=country_code,
                accounting_method=str(elec_record.accounting_method) if hasattr(elec_record, "accounting_method") else "location_based",
                td_loss_percentage=_safe_float(result.td_loss_percentage) if hasattr(result, "td_loss_percentage") else 0.0,
                td_loss_kwh=_safe_float(result.td_loss_kwh) if hasattr(result, "td_loss_kwh") else 0.0,
                td_emissions_tco2e=_safe_float(result.td_emissions_tco2e) if hasattr(result, "td_emissions_tco2e") else 0.0,
                co2_tco2e=_safe_float(result.co2_tco2e) if hasattr(result, "co2_tco2e") else 0.0,
                ch4_tco2e=_safe_float(result.ch4_tco2e) if hasattr(result, "ch4_tco2e") else 0.0,
                n2o_tco2e=_safe_float(result.n2o_tco2e) if hasattr(result, "n2o_tco2e") else 0.0,
                ef_source=str(result.ef_source) if hasattr(result, "ef_source") else "",
                dqi_composite=_safe_float(result.dqi_composite) if hasattr(result, "dqi_composite") else None,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate_activity_3c failed: %s", exc, exc_info=True)
            raise

    def calculate_all(
        self,
        fuel_records: List[Any],
        elec_records: List[Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> CalculationResponse:
        """Execute full pipeline calculation for all Category 3 activities.

        Runs the complete 10-stage pipeline including boundary checks,
        data quality scoring, method selection, calculation (3a + 3b + 3c + 3d),
        aggregation, compliance checking, and export preparation.

        Args:
            fuel_records: List of FuelConsumptionRecord instances.
            elec_records: List of ElectricityConsumptionRecord instances.
            config: Optional configuration dict for pipeline settings.

        Returns:
            CalculationResponse with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("fea_calc")

        logger.info(
            "calculate_all: fuel_records=%d, elec_records=%d",
            len(fuel_records),
            len(elec_records),
        )

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            result = self._pipeline_engine.run(
                fuel_records=fuel_records,
                elec_records=elec_records,
                config=config,
            )

            with self.__class__._lock:
                self._total_calculations += 1
                self._calculation_results.append({
                    "calculation_id": calc_id,
                    "result": result,
                    "timestamp": _utcnow_iso(),
                })

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "fuel_records": fuel_records,
                "elec_records": elec_records,
                "result": result,
            })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return CalculationResponse(
                success=True,
                calculation_id=calc_id,
                method=str(result.method) if hasattr(result, "method") else "hybrid",
                total_fuel_records=len(fuel_records),
                total_electricity_records=len(elec_records),
                total_emissions_tco2e=_safe_float(result.total_emissions_tco2e) if hasattr(result, "total_emissions_tco2e") else 0.0,
                activity_3a_emissions_tco2e=_safe_float(result.activity_3a_emissions_tco2e) if hasattr(result, "activity_3a_emissions_tco2e") else 0.0,
                activity_3b_emissions_tco2e=_safe_float(result.activity_3b_emissions_tco2e) if hasattr(result, "activity_3b_emissions_tco2e") else 0.0,
                activity_3c_emissions_tco2e=_safe_float(result.activity_3c_emissions_tco2e) if hasattr(result, "activity_3c_emissions_tco2e") else 0.0,
                activity_3d_emissions_tco2e=_safe_float(result.activity_3d_emissions_tco2e) if hasattr(result, "activity_3d_emissions_tco2e") else 0.0,
                coverage_level=str(result.coverage_level) if hasattr(result, "coverage_level") else "minimal",
                dqi_composite=_safe_float(result.dqi_composite) if hasattr(result, "dqi_composite") else None,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate_all failed: %s", exc, exc_info=True)
            raise

    def calculate_batch(
        self,
        batch_request: Any,
    ) -> BatchResponse:
        """Execute batch calculation across multiple periods.

        Processes multiple calculation requests (typically monthly or
        quarterly periods) in sequence, collecting results and
        aggregating totals.

        Args:
            batch_request: BatchRequest instance containing multiple period requests.

        Returns:
            BatchResponse with per-period results and totals.
        """
        t0 = time.monotonic()
        batch_id = _short_id("fea_batch")

        logger.info(
            "calculate_batch: batch_id=%s, periods=%d",
            batch_id,
            len(batch_request.periods) if hasattr(batch_request, "periods") else 0,
        )

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")
            completed_count = 0
            failed_count = 0

            periods = batch_request.periods if hasattr(batch_request, "periods") else []
            for period_request in periods:
                try:
                    result = self._pipeline_engine.run(
                        fuel_records=period_request.fuel_records if hasattr(period_request, "fuel_records") else [],
                        elec_records=period_request.elec_records if hasattr(period_request, "elec_records") else [],
                    )
                    results_list.append({
                        "period": period_request.period if hasattr(period_request, "period") else "",
                        "result": result,
                    })
                    total_emissions += _safe_decimal(result.total_emissions_tco2e) if hasattr(result, "total_emissions_tco2e") else ZERO
                    completed_count += 1
                except Exception as exc:
                    logger.error("Batch period failed: %s", exc)
                    failed_count += 1
                    results_list.append({
                        "period": period_request.period if hasattr(period_request, "period") else "",
                        "error": str(exc),
                    })

            with self.__class__._lock:
                self._total_batch_runs += 1
                self._batch_results.append({
                    "batch_id": batch_id,
                    "results": results_list,
                    "timestamp": _utcnow_iso(),
                })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return BatchResponse(
                success=(failed_count == 0),
                batch_id=batch_id,
                status="completed" if failed_count == 0 else "partial",
                total_periods=len(periods),
                completed=completed_count,
                failed=failed_count,
                total_emissions_tco2e=_safe_float(total_emissions),
                results=results_list,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate_batch failed: %s", exc, exc_info=True)
            raise

    def get_wtt_factor(
        self,
        fuel_type: Any,
        source: Optional[Any] = None,
    ) -> WTTFactorResponse:
        """Get well-to-tank emission factor for a fuel type.

        Args:
            fuel_type: FuelType enum value.
            source: Optional WTTFactorSource enum value. Defaults to DEFRA.

        Returns:
            WTTFactorResponse with emission factor data.
        """
        try:
            if self._db_engine is None:
                raise RuntimeError(
                    "FuelEnergyDatabaseEngine is not available"
                )

            factor = self._db_engine.get_wtt_factor(
                fuel_type=fuel_type,
                source=source,
            )

            return WTTFactorResponse(
                success=True,
                fuel_type=str(fuel_type),
                source=str(source) if source else "",
                wtt_factor_kgco2e_per_kwh=_safe_float(factor.wtt_factor_kgco2e_per_kwh) if hasattr(factor, "wtt_factor_kgco2e_per_kwh") else 0.0,
                co2_kg_per_kwh=_safe_float(factor.co2_kg_per_kwh) if hasattr(factor, "co2_kg_per_kwh") else 0.0,
                ch4_kg_per_kwh=_safe_float(factor.ch4_kg_per_kwh) if hasattr(factor, "ch4_kg_per_kwh") else 0.0,
                n2o_kg_per_kwh=_safe_float(factor.n2o_kg_per_kwh) if hasattr(factor, "n2o_kg_per_kwh") else 0.0,
                heating_value_kwh_per_unit=_safe_float(factor.heating_value_kwh_per_unit) if hasattr(factor, "heating_value_kwh_per_unit") else 0.0,
            )

        except Exception as exc:
            logger.error("get_wtt_factor failed: %s", exc, exc_info=True)
            raise

    def get_upstream_ef(
        self,
        country_code: str,
    ) -> UpstreamEFResponse:
        """Get upstream electricity emission factor for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            UpstreamEFResponse with upstream electricity factor data.
        """
        try:
            if self._db_engine is None:
                raise RuntimeError(
                    "FuelEnergyDatabaseEngine is not available"
                )

            factor = self._db_engine.get_upstream_ef(
                country_code=country_code,
            )

            return UpstreamEFResponse(
                success=True,
                country_code=country_code,
                accounting_method=str(factor.accounting_method) if hasattr(factor, "accounting_method") else "location_based",
                upstream_factor_kgco2e_per_kwh=_safe_float(factor.upstream_factor_kgco2e_per_kwh) if hasattr(factor, "upstream_factor_kgco2e_per_kwh") else 0.0,
                co2_kg_per_kwh=_safe_float(factor.co2_kg_per_kwh) if hasattr(factor, "co2_kg_per_kwh") else 0.0,
                ch4_kg_per_kwh=_safe_float(factor.ch4_kg_per_kwh) if hasattr(factor, "ch4_kg_per_kwh") else 0.0,
                n2o_kg_per_kwh=_safe_float(factor.n2o_kg_per_kwh) if hasattr(factor, "n2o_kg_per_kwh") else 0.0,
            )

        except Exception as exc:
            logger.error("get_upstream_ef failed: %s", exc, exc_info=True)
            raise

    def get_td_loss_factor(
        self,
        country_code: str,
    ) -> TDLossResponse:
        """Get transmission and distribution loss factor for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            TDLossResponse with T&D loss percentage.
        """
        try:
            if self._db_engine is None:
                raise RuntimeError(
                    "FuelEnergyDatabaseEngine is not available"
                )

            factor = self._db_engine.get_td_loss_factor(
                country_code=country_code,
            )

            return TDLossResponse(
                success=True,
                country_code=country_code,
                td_loss_percentage=_safe_float(factor.td_loss_percentage) if hasattr(factor, "td_loss_percentage") else 0.0,
                source=str(factor.source) if hasattr(factor, "source") else "",
            )

        except Exception as exc:
            logger.error("get_td_loss_factor failed: %s", exc, exc_info=True)
            raise

    def get_available_fuels(self) -> List[str]:
        """Get list of available fuel types.

        Returns:
            List of fuel type identifiers.
        """
        try:
            if self._db_engine is None:
                raise RuntimeError(
                    "FuelEnergyDatabaseEngine is not available"
                )

            return self._db_engine.get_available_fuels()

        except Exception as exc:
            logger.error("get_available_fuels failed: %s", exc, exc_info=True)
            raise

    def convert_fuel_units(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        fuel_type: Any,
    ) -> Decimal:
        """Convert fuel quantity from one unit to another.

        Args:
            value: Quantity value to convert.
            from_unit: Source unit (e.g., "m3", "kg", "L").
            to_unit: Target unit (e.g., "kwh", "GJ").
            fuel_type: FuelType enum value for density/heating value lookup.

        Returns:
            Converted quantity value.
        """
        try:
            if self._db_engine is None:
                raise RuntimeError(
                    "FuelEnergyDatabaseEngine is not available"
                )

            return self._db_engine.convert_fuel_units(
                value=value,
                from_unit=from_unit,
                to_unit=to_unit,
                fuel_type=fuel_type,
            )

        except Exception as exc:
            logger.error("convert_fuel_units failed: %s", exc, exc_info=True)
            raise

    def check_compliance(
        self,
        result: Any,
        frameworks: Optional[List[str]] = None,
    ) -> ComplianceResponse:
        """Check regulatory compliance for a calculation result.

        Args:
            result: CalculationResult instance.
            frameworks: Optional list of framework identifiers. Defaults to all.

        Returns:
            ComplianceResponse with compliance check results.
        """
        t0 = time.monotonic()
        compliance_id = _short_id("fea_comp")

        if frameworks is None:
            frameworks = DEFAULT_COMPLIANCE_FRAMEWORKS

        logger.info(
            "check_compliance: compliance_id=%s, frameworks=%d",
            compliance_id,
            len(frameworks),
        )

        try:
            if self._compliance_checker_engine is None:
                raise RuntimeError(
                    "ComplianceCheckerEngine is not available"
                )

            check_results = self._compliance_checker_engine.check(
                result=result,
                frameworks=frameworks,
            )

            compliant = 0
            non_compliant = 0
            partial = 0
            not_applicable = 0

            for check in check_results:
                status = check.get("status", "")
                if status == "compliant":
                    compliant += 1
                elif status == "non_compliant":
                    non_compliant += 1
                elif status == "partial":
                    partial += 1
                else:
                    not_applicable += 1

            with self.__class__._lock:
                self._total_compliance_checks += 1
                self._compliance_results.append({
                    "compliance_id": compliance_id,
                    "results": check_results,
                    "timestamp": _utcnow_iso(),
                })

            provenance_hash = _compute_hash({
                "compliance_id": compliance_id,
                "result": result,
                "frameworks": frameworks,
            })

            return ComplianceResponse(
                success=True,
                compliance_id=compliance_id,
                frameworks_checked=len(frameworks),
                compliant=compliant,
                non_compliant=non_compliant,
                partial=partial,
                not_applicable=not_applicable,
                results=check_results,
                provenance_hash=provenance_hash,
            )

        except Exception as exc:
            logger.error("check_compliance failed: %s", exc, exc_info=True)
            raise

    def assess_dqi(
        self,
        record: Any,
        ef: Any,
    ) -> DQIResponse:
        """Assess data quality indicators for a record and emission factor.

        Args:
            record: FuelConsumptionRecord or ElectricityConsumptionRecord.
            ef: WTTEmissionFactor or UpstreamElectricityFactor.

        Returns:
            DQIResponse with quality scores.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            dqi = self._pipeline_engine.assess_dqi(record, ef)

            return DQIResponse(
                success=True,
                dqi_composite=_safe_float(dqi.get("dqi_composite", 0.0)),
                quality_tier=dqi.get("quality_tier", "tier_3"),
                technology_score=_safe_float(dqi.get("technology_score", 0.0)),
                temporal_score=_safe_float(dqi.get("temporal_score", 0.0)),
                geographic_score=_safe_float(dqi.get("geographic_score", 0.0)),
                completeness_score=_safe_float(dqi.get("completeness_score", 0.0)),
                reliability_score=_safe_float(dqi.get("reliability_score", 0.0)),
            )

        except Exception as exc:
            logger.error("assess_dqi failed: %s", exc, exc_info=True)
            raise

    def quantify_uncertainty(
        self,
        record: Any,
        ef: Any,
        method: Optional[str] = None,
    ) -> UncertaintyResponse:
        """Quantify uncertainty for a calculation.

        Args:
            record: FuelConsumptionRecord or ElectricityConsumptionRecord.
            ef: WTTEmissionFactor or UpstreamElectricityFactor.
            method: Optional uncertainty method. Defaults to "pedigree".

        Returns:
            UncertaintyResponse with uncertainty bounds.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            uncertainty = self._pipeline_engine.quantify_uncertainty(
                record=record,
                ef=ef,
                method=method or "pedigree",
            )

            return UncertaintyResponse(
                success=True,
                method=method or "pedigree",
                confidence_level=95.0,
                lower_bound_tco2e=_safe_float(uncertainty.get("lower_bound_tco2e", 0.0)),
                upper_bound_tco2e=_safe_float(uncertainty.get("upper_bound_tco2e", 0.0)),
                uncertainty_percentage=_safe_float(uncertainty.get("uncertainty_percentage", 0.0)),
            )

        except Exception as exc:
            logger.error("quantify_uncertainty failed: %s", exc, exc_info=True)
            raise

    def aggregate_results(
        self,
        results: List[Any],
        dimensions: Optional[List[str]] = None,
    ) -> AggregationResponse:
        """Aggregate calculation results by specified dimensions.

        Args:
            results: List of CalculationResult instances.
            dimensions: Optional list of dimension names (e.g., ["facility_id", "fuel_type"]).

        Returns:
            AggregationResponse with grouped emissions totals.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            aggregations = self._pipeline_engine.aggregate(
                results=results,
                dimensions=dimensions,
            )

            return AggregationResponse(
                groups=aggregations.get("groups", {}),
                total_emissions_tco2e=_safe_float(aggregations.get("total_emissions_tco2e", 0.0)),
                total_fuel_consumption=_safe_float(aggregations.get("total_fuel_consumption", 0.0)),
                total_electricity_kwh=_safe_float(aggregations.get("total_electricity_kwh", 0.0)),
                facility_count=aggregations.get("facility_count", 0),
                coverage_level=aggregations.get("coverage_level", "minimal"),
                period=aggregations.get("period", "annual"),
            )

        except Exception as exc:
            logger.error("aggregate_results failed: %s", exc, exc_info=True)
            raise

    def export_results(
        self,
        results: Any,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export calculation results to specified format.

        Args:
            results: CalculationResult instance or list of results.
            format: Export format ("json", "csv", "xlsx", "pdf"). Defaults to "json".

        Returns:
            Dict with exported content.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            exported = self._pipeline_engine.export(results, format)

            with self.__class__._lock:
                self._total_exports += 1
                self._export_results.append({
                    "format": format,
                    "timestamp": _utcnow_iso(),
                })

            size_bytes = len(str(exported)) if exported else 0

            return {
                "success": True,
                "format": format,
                "content": exported,
                "size_bytes": size_bytes,
            }

        except Exception as exc:
            logger.error("export_results failed: %s", exc, exc_info=True)
            raise

    def get_hot_spots(
        self,
        results: Any,
        threshold: Optional[float] = None,
    ) -> HotSpotResponse:
        """Identify emission hot-spots in calculation results.

        Args:
            results: CalculationResult instance or list of results.
            threshold: Optional emission threshold (tCO2e) for hot-spot inclusion.

        Returns:
            HotSpotResponse with top contributors.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            hot_spots = self._pipeline_engine.identify_hot_spots(
                results=results,
                threshold=threshold,
            )

            return HotSpotResponse(
                success=True,
                top_fuels=hot_spots.get("top_fuels", []),
                top_facilities=hot_spots.get("top_facilities", []),
                top_suppliers=hot_spots.get("top_suppliers", []),
            )

        except Exception as exc:
            logger.error("get_hot_spots failed: %s", exc, exc_info=True)
            raise

    def get_materiality(
        self,
        results: Any,
        scope1: Decimal,
        scope2: Decimal,
    ) -> MaterialityResponse:
        """Assess materiality of Category 3 emissions.

        Args:
            results: CalculationResult instance.
            scope1: Total Scope 1 emissions (tCO2e).
            scope2: Total Scope 2 emissions (tCO2e).

        Returns:
            MaterialityResponse with materiality assessment.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "FuelEnergyPipelineEngine is not available"
                )

            materiality = self._pipeline_engine.assess_materiality(
                results=results,
                scope1=scope1,
                scope2=scope2,
            )

            return MaterialityResponse(
                success=True,
                category_3_emissions_tco2e=_safe_float(materiality.get("category_3_emissions_tco2e", 0.0)),
                scope_1_emissions_tco2e=_safe_float(scope1),
                scope_2_emissions_tco2e=_safe_float(scope2),
                scope_3_total_emissions_tco2e=_safe_float(materiality.get("scope_3_total_emissions_tco2e", 0.0)),
                materiality_percentage=_safe_float(materiality.get("materiality_percentage", 0.0)),
                materiality_tier=materiality.get("materiality_tier", "low"),
            )

        except Exception as exc:
            logger.error("get_materiality failed: %s", exc, exc_info=True)
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dict with aggregate statistics.
        """
        uptime = time.monotonic() - self._start_time

        with self.__class__._lock:
            stats = {
                "total_calculations": self._total_calculations,
                "total_batch_runs": self._total_batch_runs,
                "total_activity_3a": self._total_activity_3a,
                "total_activity_3b": self._total_activity_3b,
                "total_activity_3c": self._total_activity_3c,
                "total_activity_3d": self._total_activity_3d,
                "total_compliance_checks": self._total_compliance_checks,
                "total_exports": self._total_exports,
                "uptime_seconds": uptime,
                "timestamp": _utcnow_iso(),
            }

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Service health check.

        Returns:
            Dict with service status and engine availability.
        """
        engines = {
            "db": "available" if self._db_engine is not None else "unavailable",
            "activity_3a": "available" if self._activity_3a_engine is not None else "unavailable",
            "activity_3b": "available" if self._activity_3b_engine is not None else "unavailable",
            "activity_3c": "available" if self._activity_3c_engine is not None else "unavailable",
            "activity_3d": "available" if self._activity_3d_engine is not None else "unavailable",
            "compliance_checker": "available" if self._compliance_checker_engine is not None else "unavailable",
            "pipeline": "available" if self._pipeline_engine is not None else "unavailable",
        }

        uptime = time.monotonic() - self._start_time

        with self.__class__._lock:
            total_calcs = self._total_calculations

        return {
            "status": "healthy",
            "service": "fuel-energy-activities",
            "agent_id": AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-003",
            "version": VERSION if MODELS_AVAILABLE else "1.0.0",
            "engines": engines,
            "models_available": MODELS_AVAILABLE,
            "uptime_seconds": uptime,
            "total_calculations": total_calcs,
            "timestamp": _utcnow_iso(),
        }

    def reset(self) -> None:
        """Reset all in-memory caches and counters.

        This method is intended for test fixtures only and must not
        be called in production code.
        """
        with self.__class__._lock:
            self._calculation_results.clear()
            self._batch_results.clear()
            self._compliance_results.clear()
            self._export_results.clear()
            self._total_calculations = 0
            self._total_batch_runs = 0
            self._total_activity_3a = 0
            self._total_activity_3b = 0
            self._total_activity_3c = 0
            self._total_activity_3d = 0
            self._total_compliance_checks = 0
            self._total_exports = 0
        logger.info("FuelEnergyActivitiesService reset")

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert a result object to a dict.

        Args:
            result: Result object (Pydantic model or dict).

        Returns:
            Dict representation.
        """
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        elif isinstance(result, dict):
            return result
        else:
            return {"result": str(result)}


# ===================================================================
# Module-level singleton accessor
# ===================================================================

_SERVICE_INSTANCE: Optional[FuelEnergyActivitiesService] = None


def get_service(config: Optional[Any] = None) -> FuelEnergyActivitiesService:
    """Get the singleton FuelEnergyActivitiesService instance.

    Args:
        config: Optional FuelEnergyActivitiesConfig instance.

    Returns:
        The singleton FuelEnergyActivitiesService instance.
    """
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = FuelEnergyActivitiesService(config)
    return _SERVICE_INSTANCE


# ===================================================================
# FastAPI router configuration
# ===================================================================


def get_router() -> Any:
    """Create and configure a FastAPI APIRouter for fuel & energy endpoints.

    Returns:
        APIRouter instance with all endpoints configured, or None if
        FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE or APIRouter is None:
        logger.warning("FastAPI not available, cannot create router")
        return None

    router = APIRouter(
        prefix="/api/v1/fuel-energy-activities",
        tags=["fuel-energy-activities"],
    )

    service = get_service()

    @router.post("/calculate/3a", response_model=Activity3aResponse)
    async def calculate_3a_endpoint(request: Dict[str, Any]) -> Activity3aResponse:
        """Calculate Activity 3a upstream fuel emissions."""
        if MODELS_AVAILABLE:
            fuel_record = FuelConsumptionRecord(**request.get("fuel_record", {}))
            gwp_source = request.get("gwp_source")
        else:
            fuel_record = request.get("fuel_record")
            gwp_source = request.get("gwp_source")
        return service.calculate_activity_3a(fuel_record, gwp_source)

    @router.post("/calculate/3b", response_model=Activity3bResponse)
    async def calculate_3b_endpoint(request: Dict[str, Any]) -> Activity3bResponse:
        """Calculate Activity 3b upstream electricity emissions."""
        if MODELS_AVAILABLE:
            elec_record = ElectricityConsumptionRecord(**request.get("elec_record", {}))
        else:
            elec_record = request.get("elec_record")
        country_code = request.get("country_code", "")
        gwp_source = request.get("gwp_source")
        return service.calculate_activity_3b(elec_record, country_code, gwp_source)

    @router.post("/calculate/3c", response_model=Activity3cResponse)
    async def calculate_3c_endpoint(request: Dict[str, Any]) -> Activity3cResponse:
        """Calculate Activity 3c T&D loss emissions."""
        if MODELS_AVAILABLE:
            elec_record = ElectricityConsumptionRecord(**request.get("elec_record", {}))
        else:
            elec_record = request.get("elec_record")
        country_code = request.get("country_code", "")
        gwp_source = request.get("gwp_source")
        return service.calculate_activity_3c(elec_record, country_code, gwp_source)

    @router.post("/calculate/all", response_model=CalculationResponse)
    async def calculate_all_endpoint(request: Dict[str, Any]) -> CalculationResponse:
        """Execute full pipeline calculation for all Category 3 activities."""
        fuel_records = request.get("fuel_records", [])
        elec_records = request.get("elec_records", [])
        config = request.get("config")
        return service.calculate_all(fuel_records, elec_records, config)

    @router.post("/calculate/batch", response_model=BatchResponse)
    async def calculate_batch_endpoint(batch: Dict[str, Any]) -> BatchResponse:
        """Execute batch calculation."""
        if MODELS_AVAILABLE:
            batch_request = BatchRequest(**batch)
        else:
            batch_request = batch
        return service.calculate_batch(batch_request)

    @router.get("/factors/wtt/{fuel_type}", response_model=WTTFactorResponse)
    async def get_wtt_factor_endpoint(
        fuel_type: str = Path(...),
        source: Optional[str] = Query(default=None),
    ) -> WTTFactorResponse:
        """Get well-to-tank emission factor."""
        return service.get_wtt_factor(fuel_type, source)

    @router.get("/factors/upstream/{country_code}", response_model=UpstreamEFResponse)
    async def get_upstream_ef_endpoint(
        country_code: str = Path(...),
    ) -> UpstreamEFResponse:
        """Get upstream electricity emission factor."""
        return service.get_upstream_ef(country_code)

    @router.get("/factors/td-loss/{country_code}", response_model=TDLossResponse)
    async def get_td_loss_endpoint(
        country_code: str = Path(...),
    ) -> TDLossResponse:
        """Get T&D loss factor."""
        return service.get_td_loss_factor(country_code)

    @router.post("/compliance", response_model=ComplianceResponse)
    async def check_compliance_endpoint(request: Dict[str, Any]) -> ComplianceResponse:
        """Check regulatory compliance."""
        result = request.get("result")
        frameworks = request.get("frameworks")
        return service.check_compliance(result, frameworks)

    @router.get("/health")
    async def health_check_endpoint() -> Dict[str, Any]:
        """Service health check."""
        return service.health_check()

    @router.get("/stats")
    async def get_stats_endpoint() -> Dict[str, Any]:
        """Get service statistics."""
        return service.get_statistics()

    return router


# ===================================================================
# FastAPI app configuration
# ===================================================================


def configure_fuel_energy_activities(app: Any) -> None:
    """Configure a FastAPI app with fuel & energy activities endpoints.

    Registers the fuel & energy activities router with the provided
    FastAPI app instance.

    Args:
        app: FastAPI application instance.
    """
    if not FASTAPI_AVAILABLE:
        logger.warning(
            "FastAPI not available, cannot configure fuel & energy activities endpoints"
        )
        return

    router = get_router()
    if router is not None:
        app.include_router(router)
        logger.info("Fuel & Energy Activities router registered with FastAPI app")
    else:
        logger.warning("Failed to create Fuel & Energy Activities router")
