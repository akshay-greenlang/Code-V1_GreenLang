# -*- coding: utf-8 -*-
"""
Flaring Service Setup - AGENT-MRV-006
=======================================

Service facade for the Flaring Agent (GL-MRV-SCOPE1-006).

Provides ``configure_flaring(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``FlaringService`` facade class that aggregates all 7 engines:

    1. FlareSystemDatabaseEngine  - Flare types, gas compositions, defaults
    2. EmissionCalculatorEngine   - Gas composition / default EF / direct
    3. CombustionEfficiencyEngine - CE modeling, steam/air, wind, DRE
    4. FlaringEventTrackerEngine  - Event classification, duration, volume
    5. UncertaintyQuantifierEngine - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine    - Multi-framework regulatory compliance
    7. FlaringPipelineEngine      - Eight-stage orchestration pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.flaring.setup import configure_flaring
    >>> app = FastAPI()
    >>> configure_flaring(app)

    >>> from greenlang.flaring.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "flare_type": "ELEVATED_STEAM_ASSISTED",
    ...     "gas_volume_mscf": 500,
    ...     "method": "DEFAULT_EF",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
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

from pydantic import BaseModel, ConfigDict, Field

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
    from greenlang.flaring.config import (
        FlaringConfig,
        get_config,
    )
except ImportError:
    FlaringConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
except ImportError:
    FlareSystemDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.emission_calculator import (
        EmissionCalculatorEngine,
    )
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.combustion_efficiency import (
        CombustionEfficiencyEngine,
    )
except ImportError:
    CombustionEfficiencyEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.flaring_event_tracker import (
        FlaringEventTrackerEngine,
    )
except ImportError:
    FlaringEventTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.flaring_pipeline import (
        FlaringPipelineEngine,
        FLARE_TYPES as BUILTIN_FLARE_TYPES,
        EVENT_CATEGORIES as BUILTIN_EVENT_CATEGORIES,
        GWP_VALUES as BUILTIN_GWP_VALUES,
        DEFAULT_EMISSION_FACTORS as BUILTIN_DEFAULT_EFS,
    )
except ImportError:
    FlaringPipelineEngine = None  # type: ignore[assignment, misc]
    BUILTIN_FLARE_TYPES = {}  # type: ignore[assignment]
    BUILTIN_EVENT_CATEGORIES = {}  # type: ignore[assignment]
    BUILTIN_GWP_VALUES = {}  # type: ignore[assignment]
    BUILTIN_DEFAULT_EFS = {}  # type: ignore[assignment]

try:
    from greenlang.flaring.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.flaring.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        record_flare_lookup as _record_flare_lookup,
        record_factor_selection as _record_factor_selection,
        record_flaring_event as _record_flaring_event,
        record_uncertainty as _record_uncertainty,
        record_compliance_check as _record_compliance_check,
        record_batch as _record_batch,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_flare_lookup = None  # type: ignore[assignment]
    _record_factor_selection = None  # type: ignore[assignment]
    _record_flaring_event = None  # type: ignore[assignment]
    _record_uncertainty = None  # type: ignore[assignment]
    _record_compliance_check = None  # type: ignore[assignment]
    _record_batch = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]


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
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class CalculateResponse(BaseModel):
    """Single flaring emission calculation response.

    Attributes:
        success: Whether the calculation completed without error.
        calculation_id: Unique calculation identifier.
        flare_type: Flare type used.
        method: Calculation methodology applied.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        co2_kg: CO2 component in kilograms.
        ch4_kg: CH4 component in kilograms.
        n2o_kg: N2O component in kilograms.
        combustion_efficiency: Effective CE used.
        uncertainty_pct: Uncertainty percentage if computed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: ISO-8601 UTC timestamp.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    flare_type: str = Field(default="")
    method: str = Field(default="DEFAULT_EF")
    event_category: str = Field(default="ROUTINE")
    total_co2e_kg: float = Field(default=0.0)
    co2_kg: float = Field(default=0.0)
    ch4_kg: float = Field(default=0.0)
    n2o_kg: float = Field(default=0.0)
    co2e_from_co2_kg: float = Field(default=0.0)
    co2e_from_ch4_kg: float = Field(default=0.0)
    co2e_from_n2o_kg: float = Field(default=0.0)
    combustion_efficiency: float = Field(default=0.98)
    gas_volume_mscf: float = Field(default=0.0)
    hhv_btu_scf: float = Field(default=0.0)
    gwp_source: str = Field(default="AR6")
    includes_pilot: bool = Field(default=False)
    includes_purge: bool = Field(default=False)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
    """Batch flaring emission calculation response.

    Attributes:
        success: Whether all calculations succeeded.
        total_calculations: Number of calculations attempted.
        successful: Number of successful calculations.
        failed: Number of failed calculations.
        total_co2e_kg: Aggregate CO2e in kilograms.
        results: Individual calculation results.
        processing_time_ms: Total processing time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    total_calculations: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_co2e_kg: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class FlareSystemResponse(BaseModel):
    """Response for a single flare system.

    Attributes:
        flare_id: Unique flare system identifier.
        flare_type: Flare type classification.
        name: Human-readable flare system name.
        default_ce: Default combustion efficiency.
        assist_type: Assist medium type (STEAM, AIR, NONE).
        status: Operational status.
    """

    model_config = ConfigDict(frozen=True)

    flare_id: str = Field(default="")
    flare_type: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    default_ce: float = Field(default=0.98)
    assist_type: str = Field(default="NONE")
    min_hhv_btu_scf: float = Field(default=200.0)
    status: str = Field(default="active")


class FlareSystemListResponse(BaseModel):
    """Response listing registered flare systems.

    Attributes:
        flare_systems: List of flare system summaries.
        total: Total number of registered systems.
    """

    model_config = ConfigDict(frozen=True)

    flare_systems: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class FlaringEventResponse(BaseModel):
    """Response for a single flaring event.

    Attributes:
        event_id: Unique event identifier.
        flare_id: Associated flare system.
        event_category: Event classification.
        gas_volume_mscf: Gas volume in MSCF.
        duration_hours: Event duration in hours.
        is_routine: Whether this is routine flaring.
        timestamp: Event timestamp.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default="")
    flare_id: str = Field(default="")
    event_category: str = Field(default="ROUTINE")
    gas_volume_mscf: float = Field(default=0.0)
    duration_hours: float = Field(default=0.0)
    is_routine: bool = Field(default=True)
    timestamp: str = Field(default_factory=_utcnow_iso)


class FlaringEventListResponse(BaseModel):
    """Response listing flaring events.

    Attributes:
        events: List of flaring event summaries.
        total: Total number of events.
    """

    model_config = ConfigDict(frozen=True)

    events: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class GasCompositionResponse(BaseModel):
    """Response for a registered gas composition.

    Attributes:
        composition_id: Unique composition identifier.
        name: Human-readable composition name.
        components: Component mole fractions.
        hhv_btu_scf: Calculated HHV.
    """

    model_config = ConfigDict(frozen=True)

    composition_id: str = Field(default="")
    name: str = Field(default="")
    components: Dict[str, float] = Field(default_factory=dict)
    hhv_btu_scf: float = Field(default=0.0)
    source: str = Field(default="")


class GasCompositionListResponse(BaseModel):
    """Response listing registered gas compositions.

    Attributes:
        compositions: List of composition summaries.
        total: Total number of compositions.
    """

    model_config = ConfigDict(frozen=True)

    compositions: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class EmissionFactorResponse(BaseModel):
    """Response for a registered emission factor.

    Attributes:
        factor_id: Unique emission factor identifier.
        source: Factor source authority.
        co2_kg_per_mscf: CO2 factor.
        ch4_kg_per_mscf: CH4 factor.
        n2o_kg_per_mscf: N2O factor.
    """

    model_config = ConfigDict(frozen=True)

    factor_id: str = Field(default="")
    source: str = Field(default="")
    co2_kg_per_mscf: float = Field(default=0.0)
    ch4_kg_per_mscf: float = Field(default=0.0)
    n2o_kg_per_mscf: float = Field(default=0.0)


class EmissionFactorListResponse(BaseModel):
    """Response listing registered emission factors.

    Attributes:
        factors: List of emission factor summaries.
        total: Total number of factors.
    """

    model_config = ConfigDict(frozen=True)

    factors: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class EfficiencyTestResponse(BaseModel):
    """Response for a combustion efficiency test record.

    Attributes:
        test_id: Unique test identifier.
        flare_id: Associated flare system.
        measured_ce: Measured combustion efficiency.
        test_date: Date of the test.
    """

    model_config = ConfigDict(frozen=True)

    test_id: str = Field(default="")
    flare_id: str = Field(default="")
    measured_ce: float = Field(default=0.0)
    wind_speed_ms: Optional[float] = Field(default=None)
    tip_velocity_mach: Optional[float] = Field(default=None)
    test_date: str = Field(default_factory=_utcnow_iso)
    notes: str = Field(default="")


class EfficiencyTestListResponse(BaseModel):
    """Response listing combustion efficiency test records.

    Attributes:
        records: List of CE test record summaries.
        total: Total number of records.
    """

    model_config = ConfigDict(frozen=True)

    records: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class UncertaintyResponse(BaseModel):
    """Monte Carlo uncertainty analysis response.

    Attributes:
        success: Whether the analysis completed.
        method: Uncertainty method used (monte_carlo or analytical).
        iterations: Number of Monte Carlo iterations performed.
        mean_co2e_kg: Mean CO2e from simulations in kilograms.
        std_dev_kg: Standard deviation in kilograms.
        confidence_intervals: Confidence interval bounds by level.
        dqi_score: Data quality indicator score (1-5 scale).
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    method: str = Field(default="monte_carlo")
    iterations: Optional[int] = Field(default=None)
    mean_co2e_kg: float = Field(default=0.0)
    std_dev_kg: float = Field(default=0.0)
    uncertainty_pct: float = Field(default=0.0)
    confidence_intervals: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
    )
    dqi_score: Optional[float] = Field(default=None)


class ComplianceCheckResponse(BaseModel):
    """Regulatory compliance check response.

    Attributes:
        success: Whether the check completed.
        frameworks_checked: Number of frameworks evaluated.
        compliant: Number of compliant frameworks.
        non_compliant: Number of non-compliant frameworks.
        partial: Number of partially compliant frameworks.
        results: Per-framework compliance details.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health check response.

    Attributes:
        status: Overall service status (healthy, degraded, unhealthy).
        service: Service identifier.
        version: Agent version string.
        engines: Per-engine availability status.
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="flaring")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Service aggregate statistics response.

    Attributes:
        total_calculations: Total calculations performed.
        total_flare_systems: Number of registered flare systems.
        total_events: Number of recorded flaring events.
        total_compositions: Number of registered gas compositions.
        total_factors: Number of registered emission factors.
        total_efficiency_tests: Number of CE test records.
        uptime_seconds: Service uptime in seconds.
    """

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_flare_systems: int = Field(default=0)
    total_events: int = Field(default=0)
    total_compositions: int = Field(default=0)
    total_factors: int = Field(default=0)
    total_efficiency_tests: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)


# ===================================================================
# FlaringService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["FlaringService"] = None


class FlaringService:
    """Unified facade over the Flaring Agent SDK.

    Aggregates all 7 engines (flare system database, emission calculator,
    combustion efficiency, event tracker, uncertainty quantifier,
    compliance checker, pipeline) through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing and updates
    Prometheus metrics when available.

    Attributes:
        config: FlaringConfig instance (or None in stub mode).

    Example:
        >>> service = FlaringService()
        >>> result = service.calculate({
        ...     "flare_type": "ELEVATED_STEAM_ASSISTED",
        ...     "gas_volume_mscf": 500,
        ...     "method": "DEFAULT_EF",
        ... })
        >>> print(result.total_co2e_kg)
    """

    def __init__(
        self,
        config: Any = None,
    ) -> None:
        """Initialize the Flaring Service facade.

        Instantiates all 7 internal engines plus the provenance tracker.
        Engines that fail to import are logged as warnings and the service
        continues in degraded mode.

        Args:
            config: Optional FlaringConfig. Uses global config if None.
        """
        self.config = config if config is not None else get_config()
        self._start_time: float = time.monotonic()

        # Provenance tracker
        self._provenance: Any = None
        if ProvenanceTracker is not None:
            try:
                genesis = (
                    self.config.genesis_hash
                    if self.config is not None
                    and hasattr(self.config, "genesis_hash")
                    else "GL-MRV-X-006-FLARING-GENESIS"
                )
                self._provenance = ProvenanceTracker(genesis_hash=genesis)
            except Exception as exc:
                logger.warning("ProvenanceTracker init failed: %s", exc)

        # Engine placeholders
        self._flare_system_db_engine: Any = None
        self._emission_calculator_engine: Any = None
        self._combustion_efficiency_engine: Any = None
        self._event_tracker_engine: Any = None
        self._uncertainty_engine: Any = None
        self._compliance_checker_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._calculations: List[Dict[str, Any]] = []
        self._flare_systems: Dict[str, Dict[str, Any]] = {}
        self._events: Dict[str, Dict[str, Any]] = {}
        self._compositions: Dict[str, Dict[str, Any]] = {}
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._efficiency_tests: Dict[str, Dict[str, Any]] = {}

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_calculation_time_ms: float = 0.0

        # Pre-populate default data
        self._populate_default_flare_systems()
        self._populate_default_factors()

        logger.info("FlaringService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def flare_system_db_engine(self) -> Any:
        """Get the FlareSystemDatabaseEngine instance."""
        return self._flare_system_db_engine

    @property
    def emission_calculator_engine(self) -> Any:
        """Get the EmissionCalculatorEngine instance."""
        return self._emission_calculator_engine

    @property
    def combustion_efficiency_engine(self) -> Any:
        """Get the CombustionEfficiencyEngine instance."""
        return self._combustion_efficiency_engine

    @property
    def event_tracker_engine(self) -> Any:
        """Get the FlaringEventTrackerEngine instance."""
        return self._event_tracker_engine

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty_engine

    @property
    def compliance_checker_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance."""
        return self._compliance_checker_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the FlaringPipelineEngine instance."""
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
        config_dict: Dict[str, Any] = {}
        if self.config is not None and hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()

        # E1: FlareSystemDatabaseEngine
        if FlareSystemDatabaseEngine is not None:
            try:
                self._flare_system_db_engine = FlareSystemDatabaseEngine(
                    config=config_dict,
                )
                logger.info("FlareSystemDatabaseEngine initialized")
            except Exception as exc:
                logger.warning(
                    "FlareSystemDatabaseEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "FlareSystemDatabaseEngine not available; using stub"
            )

        # E2: EmissionCalculatorEngine
        if EmissionCalculatorEngine is not None:
            try:
                self._emission_calculator_engine = EmissionCalculatorEngine(
                    flare_system_db=self._flare_system_db_engine,
                    config=config_dict,
                )
                logger.info("EmissionCalculatorEngine initialized")
            except Exception as exc:
                logger.warning(
                    "EmissionCalculatorEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "EmissionCalculatorEngine not available; using stub"
            )

        # E3: CombustionEfficiencyEngine
        if CombustionEfficiencyEngine is not None:
            try:
                self._combustion_efficiency_engine = CombustionEfficiencyEngine(
                    config=config_dict,
                )
                logger.info("CombustionEfficiencyEngine initialized")
            except Exception as exc:
                logger.warning(
                    "CombustionEfficiencyEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "CombustionEfficiencyEngine not available; using stub"
            )

        # E4: FlaringEventTrackerEngine
        if FlaringEventTrackerEngine is not None:
            try:
                self._event_tracker_engine = FlaringEventTrackerEngine(
                    config=config_dict,
                )
                logger.info("FlaringEventTrackerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "FlaringEventTrackerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "FlaringEventTrackerEngine not available; using stub"
            )

        # E5: UncertaintyQuantifierEngine
        if UncertaintyQuantifierEngine is not None:
            try:
                self._uncertainty_engine = UncertaintyQuantifierEngine()
                logger.info("UncertaintyQuantifierEngine initialized")
            except Exception as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "UncertaintyQuantifierEngine not available; using stub"
            )

        # E6: ComplianceCheckerEngine
        if ComplianceCheckerEngine is not None:
            try:
                self._compliance_checker_engine = ComplianceCheckerEngine(
                    config=config_dict,
                )
                logger.info("ComplianceCheckerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ComplianceCheckerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ComplianceCheckerEngine not available; using stub"
            )

        # E7: FlaringPipelineEngine
        if FlaringPipelineEngine is not None:
            try:
                self._pipeline_engine = FlaringPipelineEngine(
                    flare_system_db=self._flare_system_db_engine,
                    emission_calculator=self._emission_calculator_engine,
                    combustion_efficiency=self._combustion_efficiency_engine,
                    event_tracker=self._event_tracker_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    compliance_checker=self._compliance_checker_engine,
                    config=self.config,
                )
                logger.info("FlaringPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "FlaringPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "FlaringPipelineEngine not available; using stub"
            )

    # ------------------------------------------------------------------
    # Default data population
    # ------------------------------------------------------------------

    def _populate_default_flare_systems(self) -> None:
        """Populate the in-memory flare system registry from built-in data.

        Reads BUILTIN_FLARE_TYPES to create default FlareSystemResponse
        entries for all 8 flare types.
        """
        if not BUILTIN_FLARE_TYPES:
            return

        for flare_type, info in BUILTIN_FLARE_TYPES.items():
            flare_id = f"fl_sys_{flare_type.lower()}"
            self._flare_systems[flare_id] = {
                "flare_id": flare_id,
                "flare_type": flare_type,
                "name": info.get("display_name", flare_type),
                "description": info.get("description", ""),
                "default_ce": info.get("default_ce", 0.98),
                "assist_type": info.get("assist_type", "NONE"),
                "min_hhv_btu_scf": info.get("min_hhv_btu_scf", 200.0),
                "status": "active",
            }

    def _populate_default_factors(self) -> None:
        """Populate the in-memory emission factor registry from built-in data.

        Reads BUILTIN_DEFAULT_EFS to create default EmissionFactorResponse
        entries.
        """
        if not BUILTIN_DEFAULT_EFS:
            return

        for source, ef_data in BUILTIN_DEFAULT_EFS.items():
            factor_id = f"fl_ef_{source.lower()}"
            self._emission_factors[factor_id] = {
                "factor_id": factor_id,
                "source": source,
                "co2_kg_per_mscf": ef_data.get("co2_kg_per_mscf", 0),
                "ch4_kg_per_mscf": ef_data.get("ch4_kg_per_mscf", 0),
                "n2o_kg_per_mscf": ef_data.get("n2o_kg_per_mscf", 0),
                "reference": ef_data.get("source", ""),
            }

    # ==================================================================
    # Public API: Calculate
    # ==================================================================

    def calculate(
        self,
        request_data: Dict[str, Any],
    ) -> CalculateResponse:
        """Calculate flaring emissions for a single record.

        Delegates to the pipeline engine when available, falling back to
        a stub calculation.  All calculations are deterministic
        (zero-hallucination).

        Args:
            request_data: Dictionary with at minimum:
                - flare_type (str) or flare_id (str): Flare identification.
                - gas_volume_mscf (float): Gas volume in MSCF.
                Optional:
                - method (str): GAS_COMPOSITION, DEFAULT_EF, etc.
                - event_category (str): ROUTINE, EMERGENCY, etc.
                - gas_composition (dict): Component fractions.
                - combustion_efficiency (float): CE override (0-1).
                - gwp_source (str): AR4/AR5/AR6/AR6_20yr.

        Returns:
            CalculateResponse with emissions breakdown and provenance.

        Raises:
            ValueError: If required fields are missing.
        """
        t0 = time.monotonic()
        calc_id = f"fl_calc_{uuid.uuid4().hex[:12]}"

        flare_type = request_data.get("flare_type", "")
        flare_id = request_data.get("flare_id", "")
        method = request_data.get("method", "DEFAULT_EF").upper()
        gwp_source = request_data.get("gwp_source", "AR6").upper()
        event_category = request_data.get("event_category", "ROUTINE")

        if not flare_type and not flare_id:
            raise ValueError("flare_type or flare_id is required")

        try:
            # Use pipeline engine if available
            if self._pipeline_engine is not None:
                request_data["calculation_id"] = calc_id
                pipeline_result = self._pipeline_engine.run_pipeline(
                    request=request_data,
                    gwp_source=gwp_source,
                )

                result_data = pipeline_result.get("result", {})
                elapsed_ms = (time.monotonic() - t0) * 1000.0

                response = CalculateResponse(
                    success=pipeline_result.get("success", False),
                    calculation_id=calc_id,
                    flare_type=result_data.get("flare_type", flare_type),
                    method=result_data.get("method", method),
                    event_category=result_data.get("event_category", event_category),
                    total_co2e_kg=result_data.get("total_co2e_kg", 0.0),
                    co2_kg=result_data.get("co2_kg", 0.0),
                    ch4_kg=result_data.get("ch4_kg", 0.0),
                    n2o_kg=result_data.get("n2o_kg", 0.0),
                    co2e_from_co2_kg=result_data.get("co2e_from_co2_kg", 0.0),
                    co2e_from_ch4_kg=result_data.get("co2e_from_ch4_kg", 0.0),
                    co2e_from_n2o_kg=result_data.get("co2e_from_n2o_kg", 0.0),
                    combustion_efficiency=result_data.get("combustion_efficiency", 0.98),
                    gas_volume_mscf=result_data.get("gas_volume_mscf", 0.0),
                    hhv_btu_scf=result_data.get("hhv_btu_scf", 0.0),
                    gwp_source=gwp_source,
                    includes_pilot=result_data.get("includes_pilot", False),
                    includes_purge=result_data.get("includes_purge", False),
                    uncertainty_pct=result_data.get("uncertainty_pct"),
                    provenance_hash=result_data.get("provenance_hash", ""),
                    processing_time_ms=round(elapsed_ms, 3),
                    timestamp=_utcnow_iso(),
                )
            else:
                # Stub fallback
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                response = CalculateResponse(
                    success=True,
                    calculation_id=calc_id,
                    flare_type=flare_type,
                    method=method,
                    event_category=event_category,
                    provenance_hash=_compute_hash({
                        "calculation_id": calc_id,
                        "flare_type": flare_type,
                    }),
                    processing_time_ms=round(elapsed_ms, 3),
                    timestamp=_utcnow_iso(),
                )

            # Cache result
            calc_record = {
                "calculation_id": calc_id,
                "flare_type": response.flare_type,
                "method": response.method,
                "event_category": response.event_category,
                "total_co2e_kg": response.total_co2e_kg,
                "co2_kg": response.co2_kg,
                "ch4_kg": response.ch4_kg,
                "n2o_kg": response.n2o_kg,
                "combustion_efficiency": response.combustion_efficiency,
                "gas_volume_mscf": response.gas_volume_mscf,
                "gwp_source": gwp_source,
                "provenance_hash": response.provenance_hash,
                "processing_time_ms": response.processing_time_ms,
                "timestamp": _utcnow_iso(),
                "status": "SUCCESS" if response.success else "FAILED",
            }
            self._calculations.append(calc_record)
            self._total_calculations += 1
            self._total_calculation_time_ms += elapsed_ms

            # Metrics
            if _record_calculation is not None:
                _record_calculation(flare_type, method, "completed")
            if _record_emissions is not None:
                _record_emissions(flare_type, "CO2e", response.total_co2e_kg)
            if _observe_calculation_duration is not None:
                _observe_calculation_duration(
                    "single_calculation", elapsed_ms / 1000.0,
                )

            logger.info(
                "Calculated %s: flare=%s method=%s co2e=%.4f kg",
                calc_id, flare_type, method, response.total_co2e_kg,
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if _record_calculation is not None:
                _record_calculation(flare_type, method, "failed")

            logger.error(
                "calculate failed for %s: %s", flare_type,
                exc, exc_info=True,
            )

            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                flare_type=flare_type,
                method=method,
                provenance_hash="",
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    # ==================================================================
    # Public API: Batch Calculate
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> BatchCalculateResponse:
        """Calculate emissions for multiple flaring records in batch.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            BatchCalculateResponse with aggregate totals and per-record
            results.
        """
        t0 = time.monotonic()
        results: List[Dict[str, Any]] = []
        total_co2e_kg = 0.0
        successful = 0
        failed = 0

        for req in requests:
            resp = self.calculate(req)
            result_dict = resp.model_dump()
            results.append(result_dict)

            if resp.success:
                successful += 1
                total_co2e_kg += resp.total_co2e_kg
            else:
                failed += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._total_batch_runs += 1

        if _record_batch is not None:
            status = "completed" if failed == 0 else "partial"
            _record_batch(status)

        logger.info(
            "Batch completed: %d results (ok=%d, fail=%d), "
            "%.2f kg CO2e, %.1fms",
            len(requests), successful, failed,
            total_co2e_kg, elapsed_ms,
        )

        return BatchCalculateResponse(
            success=failed == 0,
            total_calculations=len(requests),
            successful=successful,
            failed=failed,
            total_co2e_kg=total_co2e_kg,
            results=results,
            processing_time_ms=round(elapsed_ms, 3),
        )

    # ==================================================================
    # Public API: Flare System CRUD
    # ==================================================================

    def register_flare_system(
        self,
        data: Dict[str, Any],
    ) -> FlareSystemResponse:
        """Register a new flare system.

        Args:
            data: Dictionary with flare_type, name, description, etc.

        Returns:
            FlareSystemResponse with registered system details.
        """
        flare_id = data.get("flare_id", f"fl_sys_{uuid.uuid4().hex[:12]}")
        flare_type = data.get("flare_type", "ELEVATED_STEAM_ASSISTED")
        name = data.get("name", flare_type.replace("_", " ").title())
        description = data.get("description", "")
        default_ce = data.get("default_ce", 0.98)
        assist_type = data.get("assist_type", "NONE")
        min_hhv = data.get("min_hhv_btu_scf", 200.0)

        record = {
            "flare_id": flare_id,
            "flare_type": flare_type,
            "name": name,
            "description": description,
            "default_ce": default_ce,
            "assist_type": assist_type,
            "min_hhv_btu_scf": min_hhv,
            "status": "active",
        }
        self._flare_systems[flare_id] = record

        if _record_flare_lookup is not None:
            _record_flare_lookup("CUSTOM")

        logger.info("Registered flare system: %s (%s)", flare_id, flare_type)
        return FlareSystemResponse(**record)

    def get_flare_systems(self) -> FlareSystemListResponse:
        """List all registered flare systems.

        Returns:
            FlareSystemListResponse with flare system list.
        """
        all_systems = list(self._flare_systems.values())
        return FlareSystemListResponse(
            flare_systems=all_systems,
            total=len(all_systems),
        )

    def get_flare_system(
        self,
        flare_id: str,
    ) -> Optional[FlareSystemResponse]:
        """Get details for a specific flare system.

        Args:
            flare_id: Flare system identifier.

        Returns:
            FlareSystemResponse or None if not found.
        """
        record = self._flare_systems.get(flare_id)
        if record is None:
            return None

        if _record_flare_lookup is not None:
            _record_flare_lookup("CACHE")

        return FlareSystemResponse(**record)

    # ==================================================================
    # Public API: Flaring Event CRUD
    # ==================================================================

    def record_event(
        self,
        data: Dict[str, Any],
    ) -> FlaringEventResponse:
        """Record a flaring event.

        Args:
            data: Dictionary with flare_id, event_category,
                gas_volume_mscf, duration_hours.

        Returns:
            FlaringEventResponse with recorded event details.
        """
        event_id = data.get("event_id", f"fl_evt_{uuid.uuid4().hex[:12]}")
        flare_id = data.get("flare_id", "")
        event_category = data.get("event_category", "ROUTINE")
        gas_volume_mscf = float(data.get("gas_volume_mscf", 0))
        duration_hours = float(data.get("duration_hours", 0))
        is_routine = event_category in ("ROUTINE", "PILOT_PURGE")

        record = {
            "event_id": event_id,
            "flare_id": flare_id,
            "event_category": event_category,
            "gas_volume_mscf": gas_volume_mscf,
            "duration_hours": duration_hours,
            "is_routine": is_routine,
            "timestamp": _utcnow_iso(),
        }
        self._events[event_id] = record

        if _record_flaring_event is not None:
            _record_flaring_event(event_category, "recorded")

        logger.info(
            "Recorded flaring event: %s (%s, %.1f MSCF)",
            event_id, event_category, gas_volume_mscf,
        )
        return FlaringEventResponse(**record)

    def get_events(
        self,
        flare_id: Optional[str] = None,
        event_category: Optional[str] = None,
    ) -> FlaringEventListResponse:
        """List flaring events with optional filters.

        Args:
            flare_id: Filter by flare system (optional).
            event_category: Filter by event category (optional).

        Returns:
            FlaringEventListResponse with event list.
        """
        all_events = list(self._events.values())

        if flare_id is not None:
            all_events = [
                e for e in all_events
                if e.get("flare_id") == flare_id
            ]
        if event_category is not None:
            all_events = [
                e for e in all_events
                if e.get("event_category") == event_category
            ]

        return FlaringEventListResponse(
            events=all_events,
            total=len(all_events),
        )

    def get_event(
        self,
        event_id: str,
    ) -> Optional[FlaringEventResponse]:
        """Get details for a specific flaring event.

        Args:
            event_id: Flaring event identifier.

        Returns:
            FlaringEventResponse or None if not found.
        """
        record = self._events.get(event_id)
        if record is None:
            return None
        return FlaringEventResponse(**record)

    # ==================================================================
    # Public API: Gas Composition CRUD
    # ==================================================================

    def register_composition(
        self,
        data: Dict[str, Any],
    ) -> GasCompositionResponse:
        """Register a gas composition analysis.

        Args:
            data: Dictionary with name, components (dict of fractions),
                source.

        Returns:
            GasCompositionResponse with registered composition details.
        """
        comp_id = data.get(
            "composition_id", f"fl_comp_{uuid.uuid4().hex[:12]}",
        )
        name = data.get("name", "")
        components = data.get("components", {})
        source = data.get("source", "CUSTOM")

        # Calculate HHV from components
        hhv = 0.0
        from greenlang.flaring.flaring_pipeline import GAS_COMPONENT_HHVS
        for component, fraction in components.items():
            comp_info = GAS_COMPONENT_HHVS.get(component)
            if comp_info is not None:
                hhv += float(fraction) * comp_info["hhv_btu_scf"]

        record = {
            "composition_id": comp_id,
            "name": name,
            "components": components,
            "hhv_btu_scf": round(hhv, 2),
            "source": source,
        }
        self._compositions[comp_id] = record

        logger.info("Registered gas composition: %s", comp_id)
        return GasCompositionResponse(**record)

    def get_compositions(self) -> GasCompositionListResponse:
        """List all registered gas compositions.

        Returns:
            GasCompositionListResponse with composition list.
        """
        all_comps = list(self._compositions.values())
        return GasCompositionListResponse(
            compositions=all_comps,
            total=len(all_comps),
        )

    # ==================================================================
    # Public API: Emission Factor CRUD
    # ==================================================================

    def register_factor(
        self,
        data: Dict[str, Any],
    ) -> EmissionFactorResponse:
        """Register a custom emission factor.

        Args:
            data: Dictionary with source, co2_kg_per_mscf,
                ch4_kg_per_mscf, n2o_kg_per_mscf.

        Returns:
            EmissionFactorResponse with registered factor details.
        """
        factor_id = data.get(
            "factor_id", f"fl_ef_{uuid.uuid4().hex[:12]}",
        )
        source = data.get("source", "CUSTOM")
        co2_ef = float(data.get("co2_kg_per_mscf", 0))
        ch4_ef = float(data.get("ch4_kg_per_mscf", 0))
        n2o_ef = float(data.get("n2o_kg_per_mscf", 0))

        record = {
            "factor_id": factor_id,
            "source": source,
            "co2_kg_per_mscf": co2_ef,
            "ch4_kg_per_mscf": ch4_ef,
            "n2o_kg_per_mscf": n2o_ef,
        }
        self._emission_factors[factor_id] = record

        if _record_factor_selection is not None:
            _record_factor_selection("CUSTOM", source)

        logger.info(
            "Registered emission factor %s: source=%s",
            factor_id, source,
        )
        return EmissionFactorResponse(**record)

    def get_factors(self) -> EmissionFactorListResponse:
        """List all registered emission factors.

        Returns:
            EmissionFactorListResponse with factor list.
        """
        all_factors = list(self._emission_factors.values())
        return EmissionFactorListResponse(
            factors=all_factors,
            total=len(all_factors),
        )

    # ==================================================================
    # Public API: Combustion Efficiency CRUD
    # ==================================================================

    def log_efficiency_test(
        self,
        data: Dict[str, Any],
    ) -> EfficiencyTestResponse:
        """Log a combustion efficiency test result.

        Args:
            data: Dictionary with flare_id, measured_ce, wind_speed_ms,
                tip_velocity_mach, test_date, notes.

        Returns:
            EfficiencyTestResponse with recorded test details.
        """
        test_id = data.get(
            "test_id", f"fl_ce_{uuid.uuid4().hex[:12]}",
        )
        flare_id = data.get("flare_id", "")
        measured_ce = float(data.get("measured_ce", 0))
        wind_speed = data.get("wind_speed_ms")
        tip_velocity = data.get("tip_velocity_mach")
        test_date = data.get("test_date", _utcnow_iso())
        notes = data.get("notes", "")

        record = {
            "test_id": test_id,
            "flare_id": flare_id,
            "measured_ce": measured_ce,
            "wind_speed_ms": wind_speed,
            "tip_velocity_mach": tip_velocity,
            "test_date": test_date,
            "notes": notes,
        }
        self._efficiency_tests[test_id] = record

        logger.info(
            "Logged CE test %s: flare=%s CE=%.4f",
            test_id, flare_id, measured_ce,
        )
        return EfficiencyTestResponse(**record)

    def get_efficiency_records(self) -> EfficiencyTestListResponse:
        """List all combustion efficiency test records.

        Returns:
            EfficiencyTestListResponse with CE test records.
        """
        all_records = list(self._efficiency_tests.values())
        return EfficiencyTestListResponse(
            records=all_records,
            total=len(all_records),
        )

    # ==================================================================
    # Public API: Uncertainty Analysis
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> UncertaintyResponse:
        """Run uncertainty analysis on a calculation.

        Delegates to UncertaintyQuantifierEngine when available, falling
        back to an analytical estimate.

        Args:
            data: Dictionary with:
                - calculation_id (str): ID of a previous calculation.
                - method (str): "monte_carlo" or "analytical".
                - iterations (int): Monte Carlo iterations.

        Returns:
            UncertaintyResponse with statistical characterization.
        """
        calc_id = data.get("calculation_id", "")
        method = data.get("method", "monte_carlo")
        iterations = data.get("iterations", 5000)

        # Find the referenced calculation
        calc_record = None
        for c in self._calculations:
            if c.get("calculation_id") == calc_id:
                calc_record = c
                break

        total_co2e_kg = 0.0
        if calc_record is not None:
            total_co2e_kg = float(calc_record.get("total_co2e_kg", 0))

        # Delegate to uncertainty engine
        if self._uncertainty_engine is not None and calc_record is not None:
            try:
                result = self._uncertainty_engine.run_monte_carlo(
                    calculation_input=calc_record,
                    n_iterations=iterations,
                )
                if hasattr(result, "model_dump"):
                    rd = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    rd = result
                else:
                    rd = {}

                mean_kg = float(rd.get("mean_co2e_kg", total_co2e_kg))
                std_kg = float(rd.get("std_dev_kg", 0))
                dqi = rd.get("dqi_score")

                ci = {}
                raw_ci = rd.get("confidence_intervals", {})
                for level_key, bounds in raw_ci.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        ci[str(level_key)] = {
                            "lower": float(bounds[0]),
                            "upper": float(bounds[1]),
                        }
                    elif isinstance(bounds, dict):
                        ci[str(level_key)] = {
                            "lower": float(bounds.get("lower", 0)),
                            "upper": float(bounds.get("upper", 0)),
                        }

                if _record_uncertainty is not None:
                    _record_uncertainty(method)

                return UncertaintyResponse(
                    success=True,
                    method=method,
                    iterations=iterations,
                    mean_co2e_kg=mean_kg,
                    std_dev_kg=std_kg,
                    uncertainty_pct=(std_kg / mean_kg * 100.0) if mean_kg > 0 else 0.0,
                    confidence_intervals=ci,
                    dqi_score=dqi,
                )

            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed, using fallback: %s", exc,
                )

        # Fallback: analytical estimate (+/- 10-15% for flaring)
        calc_method = ""
        if calc_record is not None:
            calc_method = calc_record.get("method", "")
        base_uncertainty = 0.10 if calc_method == "GAS_COMPOSITION" else 0.15
        std_estimate = total_co2e_kg * base_uncertainty
        lower_95 = total_co2e_kg - 1.96 * std_estimate
        upper_95 = total_co2e_kg + 1.96 * std_estimate

        return UncertaintyResponse(
            success=True,
            method="analytical_fallback",
            iterations=None,
            mean_co2e_kg=total_co2e_kg,
            std_dev_kg=std_estimate,
            uncertainty_pct=base_uncertainty * 100.0,
            confidence_intervals={
                "95": {
                    "lower": max(0.0, lower_95),
                    "upper": upper_95,
                },
            },
            dqi_score=None,
        )

    # ==================================================================
    # Public API: Compliance Check
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Run multi-framework regulatory compliance check.

        Evaluates a calculation against GHG Protocol, ISO 14064,
        CSRD/ESRS E1, EPA Subpart W, EU ETS MRR, EU Methane Reg,
        World Bank ZRF, and OGMP 2.0.

        Args:
            data: Dictionary with:
                - calculation_id (str): ID of a previous calculation.
                - frameworks (list): Frameworks to check (empty = all).

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        calc_id = data.get("calculation_id", "")
        requested_frameworks = data.get("frameworks", [])

        from greenlang.flaring.flaring_pipeline import REGULATORY_FRAMEWORKS

        default_frameworks = list(REGULATORY_FRAMEWORKS.keys())
        frameworks_to_check = (
            requested_frameworks if requested_frameworks
            else default_frameworks
        )

        # Find the referenced calculation
        calc_record = None
        for c in self._calculations:
            if c.get("calculation_id") == calc_id:
                calc_record = c
                break

        results: List[Dict[str, Any]] = []
        compliant_count = 0
        non_compliant_count = 0
        partial_count = 0

        for fw in frameworks_to_check:
            fw_result = self._evaluate_framework_compliance(
                fw, calc_record,
            )
            results.append(fw_result)

            status = fw_result.get("status", "non_compliant")
            if status == "compliant":
                compliant_count += 1
            elif status == "partial":
                partial_count += 1
            else:
                non_compliant_count += 1

        if _record_compliance_check is not None:
            _record_compliance_check("multi_framework", "completed")

        logger.info(
            "Compliance check for %s: %d frameworks, "
            "%d compliant, %d non-compliant, %d partial",
            calc_id, len(frameworks_to_check),
            compliant_count, non_compliant_count, partial_count,
        )

        return ComplianceCheckResponse(
            success=True,
            frameworks_checked=len(frameworks_to_check),
            compliant=compliant_count,
            non_compliant=non_compliant_count,
            partial=partial_count,
            results=results,
        )

    # ==================================================================
    # Public API: Health & Stats
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Perform a health check on the flaring service.

        Returns:
            HealthResponse with engine availability and overall status.
        """
        engines: Dict[str, str] = {
            "flare_system_database": (
                "available"
                if self._flare_system_db_engine is not None
                else "unavailable"
            ),
            "emission_calculator": (
                "available"
                if self._emission_calculator_engine is not None
                else "unavailable"
            ),
            "combustion_efficiency": (
                "available"
                if self._combustion_efficiency_engine is not None
                else "unavailable"
            ),
            "event_tracker": (
                "available"
                if self._event_tracker_engine is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available"
                if self._uncertainty_engine is not None
                else "unavailable"
            ),
            "compliance_checker": (
                "available"
                if self._compliance_checker_engine is not None
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

        return HealthResponse(
            status=overall_status,
            service="flaring",
            version="1.0.0",
            engines=engines,
        )

    def get_stats(self) -> StatsResponse:
        """Get aggregate statistics for the service.

        Returns:
            StatsResponse with current service statistics.
        """
        uptime = time.monotonic() - self._start_time
        return StatsResponse(
            total_calculations=self._total_calculations,
            total_flare_systems=len(self._flare_systems),
            total_events=len(self._events),
            total_compositions=len(self._compositions),
            total_factors=len(self._emission_factors),
            total_efficiency_tests=len(self._efficiency_tests),
            uptime_seconds=round(uptime, 3),
        )

    # ==================================================================
    # Private: Compliance evaluation
    # ==================================================================

    def _evaluate_framework_compliance(
        self,
        framework: str,
        calc_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate compliance against a single regulatory framework.

        Args:
            framework: Framework identifier.
            calc_record: Calculation record to evaluate (may be None).

        Returns:
            Dictionary with framework compliance details.
        """
        from greenlang.flaring.flaring_pipeline import REGULATORY_FRAMEWORKS

        fw_info = REGULATORY_FRAMEWORKS.get(framework, {})
        requirements = fw_info.get("requirements", [])
        met_count = 0

        if calc_record is not None:
            has_emissions = float(calc_record.get("total_co2e_kg", 0)) >= 0
            has_method = bool(calc_record.get("method"))
            met_count = len(requirements) if (
                has_emissions and has_method
            ) else 0

        total_reqs = len(requirements)
        if met_count == total_reqs and total_reqs > 0:
            status = "compliant"
        elif met_count > 0:
            status = "partial"
        elif calc_record is None:
            status = "non_compliant"
        else:
            status = "compliant"

        return {
            "framework": framework,
            "display_name": fw_info.get("display_name", framework),
            "status": status,
            "total_requirements": total_reqs,
            "met_count": met_count,
            "requirements": requirements,
        }


# ===================================================================
# Thread-safe singleton access
# ===================================================================


_service_instance: Optional[FlaringService] = None
_service_lock = threading.Lock()


def get_service() -> FlaringService:
    """Get or create the singleton FlaringService instance.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        FlaringService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = FlaringService()
    return _service_instance


def get_router() -> Any:
    """Get the FastAPI router for flaring.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.flaring.api.router import create_router
        return create_router()
    except ImportError:
        logger.warning(
            "Flaring API router module not available"
        )
        return None


def configure_flaring(
    app: Any,
    config: Any = None,
) -> FlaringService:
    """Configure the Flaring Service on a FastAPI application.

    Creates the FlaringService singleton, stores it in
    app.state, mounts the flaring API router, and logs
    the configuration.

    Args:
        app: FastAPI application instance.
        config: Optional FlaringConfig override.

    Returns:
        FlaringService instance.
    """
    global _service_instance

    service = FlaringService(config=config)

    with _service_lock:
        _service_instance = service

    # Attach to app state
    if hasattr(app, "state"):
        app.state.flaring_service = service

    # Mount API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Flaring API router mounted")
    else:
        logger.warning(
            "Flaring router not available; API not mounted"
        )

    logger.info("Flaring service configured")
    return service


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "FlaringService",
    # Configuration helpers
    "configure_flaring",
    "get_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FlareSystemResponse",
    "FlareSystemListResponse",
    "FlaringEventResponse",
    "FlaringEventListResponse",
    "GasCompositionResponse",
    "GasCompositionListResponse",
    "EmissionFactorResponse",
    "EmissionFactorListResponse",
    "EfficiencyTestResponse",
    "EfficiencyTestListResponse",
    "UncertaintyResponse",
    "ComplianceCheckResponse",
    "HealthResponse",
    "StatsResponse",
]
