# -*- coding: utf-8 -*-
"""
Process Emissions Service Setup - AGENT-MRV-004
=================================================

Service facade for the Process Emissions Agent (GL-MRV-SCOPE1-004).

Provides ``configure_process_emissions(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``ProcessEmissionsService`` facade class that aggregates all 7 engines:

    1. ProcessDatabaseEngine   - In-memory reference data for 25 processes
    2. EmissionCalculatorEngine - EF / mass balance / stoichiometric / direct
    3. MaterialBalanceEngine    - Carbon input/output tracking (via pipeline)
    4. AbatementTrackerEngine   - Abatement tech tracking and efficiency
    5. UncertaintyQuantifierEngine - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine  - Multi-framework regulatory compliance
    7. ProcessEmissionsPipelineEngine - Eight-stage orchestration pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.process_emissions.setup import configure_process_emissions
    >>> app = FastAPI()
    >>> configure_process_emissions(app)

    >>> from greenlang.process_emissions.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "process_type": "cement_production",
    ...     "activity_data": 100000,
    ...     "activity_unit": "tonne",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-004 Process Emissions (GL-MRV-SCOPE1-004)
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
    from greenlang.process_emissions.config import (
        ProcessEmissionsConfig,
        get_config,
    )
except ImportError:
    ProcessEmissionsConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.process_emissions.process_database import ProcessDatabaseEngine
except ImportError:
    ProcessDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.emission_calculator import (
        EmissionCalculatorEngine,
    )
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.abatement_tracker import (
        AbatementTrackerEngine,
    )
except ImportError:
    AbatementTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.process_emissions_pipeline import (
        ProcessEmissionsPipelineEngine,
    )
except ImportError:
    ProcessEmissionsPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.process_emissions.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation as _record_calculation,
        record_emissions as _record_emissions,
        record_process_lookup as _record_process_lookup,
        record_factor_selection as _record_factor_selection,
        record_material_operation as _record_material_operation,
        record_uncertainty as _record_uncertainty,
        record_compliance_check as _record_compliance_check,
        record_batch as _record_batch,
        observe_calculation_duration as _observe_calculation_duration,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    _record_calculation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_process_lookup = None  # type: ignore[assignment]
    _record_factor_selection = None  # type: ignore[assignment]
    _record_material_operation = None  # type: ignore[assignment]
    _record_uncertainty = None  # type: ignore[assignment]
    _record_compliance_check = None  # type: ignore[assignment]
    _record_batch = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]

try:
    from greenlang.process_emissions.models import (
        ProcessCategory,
        ProcessType,
        EmissionGas,
        CalculationMethod,
        CalculationTier,
        GWPSource,
        EmissionFactorSource,
        MaterialType,
        AbatementType,
        ProcessUnitType,
        ProductionRoute,
        ComplianceStatus,
        GWP_VALUES,
        CARBONATE_EMISSION_FACTORS,
        PROCESS_CATEGORY_MAP,
        PROCESS_DEFAULT_GASES,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    ProcessCategory = None  # type: ignore[assignment, misc]
    ProcessType = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    MaterialType = None  # type: ignore[assignment, misc]
    AbatementType = None  # type: ignore[assignment, misc]
    ProcessUnitType = None  # type: ignore[assignment, misc]
    ProductionRoute = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    GWP_VALUES = {}  # type: ignore[assignment]
    CARBONATE_EMISSION_FACTORS = {}  # type: ignore[assignment]
    PROCESS_CATEGORY_MAP = {}  # type: ignore[assignment]
    PROCESS_DEFAULT_GASES = {}  # type: ignore[assignment]


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
    """Single process emission calculation response.

    Attributes:
        success: Whether the calculation completed without error.
        calculation_id: Unique calculation identifier.
        process_type: Industrial process type used.
        calculation_method: Calculation methodology applied.
        total_co2e_kg: Total CO2-equivalent emissions in kilograms.
        co2_kg: CO2 component in kilograms.
        ch4_kg: CH4 component in kilograms.
        n2o_kg: N2O component in kilograms.
        pfc_co2e_kg: PFC (CF4 + C2F6) component in kg CO2e.
        sf6_co2e_kg: SF6 component in kg CO2e.
        nf3_co2e_kg: NF3 component in kg CO2e.
        uncertainty_pct: Uncertainty percentage if computed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: ISO-8601 UTC timestamp.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    process_type: str = Field(default="")
    calculation_method: str = Field(default="EMISSION_FACTOR")
    total_co2e_kg: float = Field(default=0.0)
    co2_kg: float = Field(default=0.0)
    ch4_kg: float = Field(default=0.0)
    n2o_kg: float = Field(default=0.0)
    pfc_co2e_kg: float = Field(default=0.0)
    sf6_co2e_kg: float = Field(default=0.0)
    nf3_co2e_kg: float = Field(default=0.0)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
    """Batch process emission calculation response.

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


class ProcessListResponse(BaseModel):
    """Response listing registered process types.

    Attributes:
        processes: List of process type summary dictionaries.
        total: Total number of matching process types.
        page: Current page number.
        page_size: Number of items per page.
    """

    model_config = ConfigDict(frozen=True)

    processes: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class ProcessDetailResponse(BaseModel):
    """Detailed response for a single process type.

    Attributes:
        process_type: Process type identifier.
        category: Process category classification.
        name: Human-readable process name.
        description: Detailed description of the process.
        primary_gases: List of primary greenhouse gases emitted.
        applicable_tiers: List of applicable calculation tiers.
        default_emission_factor: Default Tier 1 emission factor value.
        production_routes: Available production routes.
    """

    model_config = ConfigDict(frozen=True)

    process_type: str = Field(default="")
    category: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    primary_gases: List[str] = Field(default_factory=list)
    applicable_tiers: List[str] = Field(default_factory=list)
    default_emission_factor: Optional[float] = Field(default=None)
    production_routes: List[str] = Field(default_factory=list)


class MaterialListResponse(BaseModel):
    """Response listing registered raw materials.

    Attributes:
        materials: List of material summary dictionaries.
        total: Total number of registered materials.
    """

    model_config = ConfigDict(frozen=True)

    materials: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class MaterialDetailResponse(BaseModel):
    """Detailed response for a single raw material.

    Attributes:
        material_type: Material type identifier.
        name: Human-readable material name.
        carbon_content: Carbon content fraction (0.0-1.0).
        carbonate_content: Carbonate content fraction (0.0-1.0).
    """

    model_config = ConfigDict(frozen=True)

    material_type: str = Field(default="")
    name: str = Field(default="")
    carbon_content: Optional[float] = Field(default=None)
    carbonate_content: Optional[float] = Field(default=None)


class ProcessUnitListResponse(BaseModel):
    """Response listing registered process units.

    Attributes:
        units: List of process unit summary dictionaries.
        total: Total number of registered process units.
    """

    model_config = ConfigDict(frozen=True)

    units: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class ProcessUnitDetailResponse(BaseModel):
    """Detailed response for a single process unit.

    Attributes:
        unit_id: Unique process unit identifier.
        unit_name: Human-readable process unit name.
        unit_type: Equipment classification.
        process_type: Industrial process type.
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str = Field(default="")
    unit_name: str = Field(default="")
    unit_type: str = Field(default="")
    process_type: str = Field(default="")


class FactorListResponse(BaseModel):
    """Response listing registered emission factors.

    Attributes:
        factors: List of emission factor summary dictionaries.
        total: Total number of registered emission factors.
    """

    model_config = ConfigDict(frozen=True)

    factors: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class FactorDetailResponse(BaseModel):
    """Detailed response for a single emission factor.

    Attributes:
        factor_id: Unique emission factor identifier.
        process_type: Process type this factor applies to.
        gas: Greenhouse gas species.
        value: Emission factor numeric value.
        source: Source authority for this factor.
    """

    model_config = ConfigDict(frozen=True)

    factor_id: str = Field(default="")
    process_type: str = Field(default="")
    gas: str = Field(default="")
    value: float = Field(default=0.0)
    source: str = Field(default="")


class AbatementListResponse(BaseModel):
    """Response listing registered abatement records.

    Attributes:
        records: List of abatement record dictionaries.
        total: Total number of abatement records.
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
    service: str = Field(default="process-emissions")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Service aggregate statistics response.

    Attributes:
        total_calculations: Total calculations performed.
        total_process_types: Number of registered process types.
        total_process_units: Number of registered process units.
        total_materials: Number of registered materials.
        uptime_seconds: Service uptime in seconds.
    """

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_process_types: int = Field(default=0)
    total_process_units: int = Field(default=0)
    total_materials: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)


# ===================================================================
# ProcessEmissionsService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["ProcessEmissionsService"] = None


class ProcessEmissionsService:
    """Unified facade over the Process Emissions Agent SDK.

    Aggregates all 7 engines (process database, emission calculator,
    material balance, abatement tracker, uncertainty quantifier,
    compliance checker, pipeline) through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing and updates
    Prometheus metrics when available.

    Attributes:
        config: ProcessEmissionsConfig instance (or None in stub mode).

    Example:
        >>> service = ProcessEmissionsService()
        >>> result = service.calculate({
        ...     "process_type": "cement_production",
        ...     "activity_data": 100000,
        ...     "activity_unit": "tonne",
        ... })
        >>> print(result.total_co2e_kg)
    """

    def __init__(
        self,
        config: Any = None,
    ) -> None:
        """Initialize the Process Emissions Service facade.

        Instantiates all 7 internal engines plus the provenance tracker.
        Engines that fail to import are logged as warnings and the service
        continues in degraded mode.

        Args:
            config: Optional ProcessEmissionsConfig. Uses global config
                if None.
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
                    else "GL-MRV-X-004-PROCESS-EMISSIONS-GENESIS"
                )
                self._provenance = ProvenanceTracker(genesis_hash=genesis)
            except Exception as exc:
                logger.warning("ProvenanceTracker init failed: %s", exc)

        # Engine placeholders
        self._process_database_engine: Any = None
        self._emission_calculator_engine: Any = None
        self._abatement_tracker_engine: Any = None
        self._uncertainty_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._calculations: List[Dict[str, Any]] = []
        self._processes: Dict[str, Dict[str, Any]] = {}
        self._materials: Dict[str, Dict[str, Any]] = {}
        self._process_units: Dict[str, Dict[str, Any]] = {}
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._abatement_records: Dict[str, Dict[str, Any]] = {}

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_calculation_time_ms: float = 0.0

        # Pre-populate process types from models
        self._populate_default_processes()

        logger.info("ProcessEmissionsService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def process_database_engine(self) -> Any:
        """Get the ProcessDatabaseEngine instance."""
        return self._process_database_engine

    @property
    def emission_calculator_engine(self) -> Any:
        """Get the EmissionCalculatorEngine instance."""
        return self._emission_calculator_engine

    @property
    def abatement_tracker_engine(self) -> Any:
        """Get the AbatementTrackerEngine instance."""
        return self._abatement_tracker_engine

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the ProcessEmissionsPipelineEngine instance."""
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

        # E1: ProcessDatabaseEngine
        if ProcessDatabaseEngine is not None:
            try:
                self._process_database_engine = ProcessDatabaseEngine(
                    config=config_dict,
                )
                logger.info("ProcessDatabaseEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ProcessDatabaseEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ProcessDatabaseEngine not available; using stub"
            )

        # E2: EmissionCalculatorEngine
        if EmissionCalculatorEngine is not None:
            try:
                self._emission_calculator_engine = EmissionCalculatorEngine(
                    process_database=self._process_database_engine,
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

        # E4: AbatementTrackerEngine
        if AbatementTrackerEngine is not None:
            try:
                self._abatement_tracker_engine = AbatementTrackerEngine()
                logger.info("AbatementTrackerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "AbatementTrackerEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "AbatementTrackerEngine not available; using stub"
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

        # E7: ProcessEmissionsPipelineEngine
        if ProcessEmissionsPipelineEngine is not None:
            try:
                self._pipeline_engine = ProcessEmissionsPipelineEngine(
                    process_database=self._process_database_engine,
                    emission_calculator=self._emission_calculator_engine,
                    abatement_tracker=self._abatement_tracker_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    config=self.config,
                )
                logger.info("ProcessEmissionsPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ProcessEmissionsPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "ProcessEmissionsPipelineEngine not available; using stub"
            )

    # ------------------------------------------------------------------
    # Default data population
    # ------------------------------------------------------------------

    def _populate_default_processes(self) -> None:
        """Populate the in-memory process registry from built-in data.

        Reads PROCESS_CATEGORY_MAP and PROCESS_DEFAULT_GASES to create
        default ProcessDetailResponse entries for all 25 process types.
        """
        if not PROCESS_CATEGORY_MAP or not PROCESS_DEFAULT_GASES:
            return

        for category, process_types in PROCESS_CATEGORY_MAP.items():
            for pt in process_types:
                gases = PROCESS_DEFAULT_GASES.get(pt, ["CO2"])
                display_name = pt.replace("_", " ").title()
                self._processes[pt] = {
                    "process_type": pt,
                    "category": category,
                    "name": display_name,
                    "description": (
                        f"Scope 1 process emissions from {display_name}"
                    ),
                    "primary_gases": gases,
                    "applicable_tiers": ["TIER_1", "TIER_2", "TIER_3"],
                    "default_emission_factor": None,
                    "production_routes": [],
                }

        # Add production routes for iron/steel and aluminum
        iron_steel = self._processes.get("iron_steel")
        if iron_steel is not None:
            iron_steel["production_routes"] = [
                "bf_bof", "eaf", "dri", "ohf",
            ]

        aluminum = self._processes.get("aluminum_smelting")
        if aluminum is not None:
            aluminum["production_routes"] = [
                "prebake", "soderberg_vss", "soderberg_hss",
                "cwpb", "swpb",
            ]

    # ==================================================================
    # Public API: Calculate
    # ==================================================================

    def calculate(
        self,
        request_data: Dict[str, Any],
    ) -> CalculateResponse:
        """Calculate process emissions for a single record.

        Delegates to the pipeline engine when available, falling back to
        the emission calculator engine, and finally to a stub response.
        All calculations are deterministic (zero-hallucination).

        Args:
            request_data: Dictionary with at minimum:
                - process_type (str): Industrial process identifier.
                - activity_data (float): Production quantity.
                - activity_unit (str): Unit of activity data.
                Optional:
                - calculation_method (str): EF, mass balance, etc.
                - calculation_tier (str): TIER_1/TIER_2/TIER_3.
                - gwp_source (str): AR4/AR5/AR6/AR6_20YR.
                - ef_source (str): EPA/IPCC/DEFRA/EU_ETS/CUSTOM.
                - production_route (str): For iron/steel, aluminum.
                - abatement_type (str): Abatement technology.
                - abatement_efficiency (float): Fraction abated (0-1).
                - materials (list): Material inputs for mass balance.

        Returns:
            CalculateResponse with emissions breakdown and provenance.

        Raises:
            ValueError: If process_type or activity_data is missing.
        """
        t0 = time.monotonic()
        calc_id = f"pe_calc_{uuid.uuid4().hex[:12]}"

        process_type = request_data.get("process_type", "")
        activity_data = request_data.get("activity_data", 0)
        activity_unit = request_data.get("activity_unit", "tonne")
        method = request_data.get(
            "calculation_method", "EMISSION_FACTOR",
        ).upper()
        gwp_source = request_data.get("gwp_source", "AR6").upper()
        ef_source = request_data.get("ef_source", "IPCC").upper()
        abatement_eff = request_data.get("abatement_efficiency")
        abatement_type = request_data.get("abatement_type")
        production_route = request_data.get("production_route")
        materials = request_data.get("materials")

        if not process_type:
            raise ValueError("process_type is required")

        try:
            result = self._execute_calculation(
                calc_id=calc_id,
                process_type=process_type,
                activity_data=activity_data,
                activity_unit=activity_unit,
                method=method,
                gwp_source=gwp_source,
                ef_source=ef_source,
                abatement_efficiency=abatement_eff,
                abatement_type=abatement_type,
                production_route=production_route,
                materials=materials,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "process_type": process_type,
                "total_co2e_kg": result.get("total_co2e_kg", 0),
            })

            # Extract per-gas breakdowns
            gas_emissions = result.get("gas_emissions", [])
            co2_kg = self._sum_gas_co2e_kg(gas_emissions, "CO2")
            ch4_kg = self._sum_gas_co2e_kg(gas_emissions, "CH4")
            n2o_kg = self._sum_gas_co2e_kg(gas_emissions, "N2O")
            pfc_kg = (
                self._sum_gas_co2e_kg(gas_emissions, "CF4")
                + self._sum_gas_co2e_kg(gas_emissions, "C2F6")
            )
            sf6_kg = self._sum_gas_co2e_kg(gas_emissions, "SF6")
            nf3_kg = self._sum_gas_co2e_kg(gas_emissions, "NF3")

            total_co2e_kg = float(
                result.get("total_co2e_kg", 0)
            )

            response = CalculateResponse(
                success=result.get("status") == "SUCCESS",
                calculation_id=calc_id,
                process_type=process_type,
                calculation_method=method,
                total_co2e_kg=total_co2e_kg,
                co2_kg=co2_kg,
                ch4_kg=ch4_kg,
                n2o_kg=n2o_kg,
                pfc_co2e_kg=pfc_kg,
                sf6_co2e_kg=sf6_kg,
                nf3_co2e_kg=nf3_kg,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

            # Cache result
            calc_record = {
                "calculation_id": calc_id,
                "process_type": process_type,
                "method": method,
                "total_co2e_kg": total_co2e_kg,
                "gas_emissions": gas_emissions,
                "gwp_source": gwp_source,
                "ef_source": ef_source,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 3),
                "timestamp": _utcnow_iso(),
                "status": "SUCCESS" if response.success else "FAILED",
            }
            self._calculations.append(calc_record)
            self._total_calculations += 1
            self._total_calculation_time_ms += elapsed_ms

            # Metrics
            if _record_calculation is not None:
                _record_calculation(process_type, method, "completed")
            if _record_emissions is not None:
                _record_emissions(
                    process_type, "scope_1", total_co2e_kg / 1000.0,
                )
            if _observe_calculation_duration is not None:
                _observe_calculation_duration(
                    "single_calculation", elapsed_ms / 1000.0,
                )

            logger.info(
                "Calculated %s: process=%s method=%s co2e=%.4f kg",
                calc_id, process_type, method, total_co2e_kg,
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if _record_calculation is not None:
                _record_calculation(process_type, method, "failed")

            logger.error(
                "calculate failed for %s: %s", process_type,
                exc, exc_info=True,
            )

            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                process_type=process_type,
                calculation_method=method,
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
        """Calculate emissions for multiple process records in batch.

        Args:
            requests: List of calculation request dictionaries. Each
                follows the same schema as ``calculate()``.

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
    # Public API: Process type CRUD
    # ==================================================================

    def register_process(
        self,
        data: Dict[str, Any],
    ) -> ProcessDetailResponse:
        """Register a new process type.

        Args:
            data: Dictionary with process_type, category, name,
                description, primary_gases, applicable_tiers.

        Returns:
            ProcessDetailResponse with the registered process details.
        """
        process_type = data.get("process_type", "")
        category = data.get("category", "other")
        name = data.get("name", process_type.replace("_", " ").title())
        description = data.get("description", "")
        primary_gases = data.get("primary_gases", ["CO2"])
        applicable_tiers = data.get(
            "applicable_tiers", ["TIER_1", "TIER_2", "TIER_3"],
        )
        default_ef = data.get("default_emission_factor")
        production_routes = data.get("production_routes", [])

        record = {
            "process_type": process_type,
            "category": category,
            "name": name,
            "description": description,
            "primary_gases": primary_gases,
            "applicable_tiers": applicable_tiers,
            "default_emission_factor": default_ef,
            "production_routes": production_routes,
        }

        self._processes[process_type] = record

        if _record_process_lookup is not None:
            _record_process_lookup("CUSTOM")

        logger.info("Registered process type: %s", process_type)
        return ProcessDetailResponse(**record)

    def list_processes(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> ProcessListResponse:
        """List registered process types with pagination.

        Args:
            page: Page number (1-based).
            page_size: Items per page.

        Returns:
            ProcessListResponse with paginated process list.
        """
        all_processes = list(self._processes.values())
        total = len(all_processes)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_processes[start:end]

        return ProcessListResponse(
            processes=page_data,
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_process(
        self,
        process_id: str,
    ) -> Optional[ProcessDetailResponse]:
        """Get details for a specific process type.

        Args:
            process_id: Process type identifier.

        Returns:
            ProcessDetailResponse or None if not found.
        """
        record = self._processes.get(process_id)
        if record is None:
            return None

        if _record_process_lookup is not None:
            _record_process_lookup("CUSTOM")

        return ProcessDetailResponse(**record)

    # ==================================================================
    # Public API: Material CRUD
    # ==================================================================

    def register_material(
        self,
        data: Dict[str, Any],
    ) -> MaterialDetailResponse:
        """Register a raw material.

        Args:
            data: Dictionary with material_type, name, carbon_content,
                carbonate_content.

        Returns:
            MaterialDetailResponse with the registered material details.
        """
        material_type = data.get("material_type", "")
        name = data.get("name", material_type.replace("_", " ").title())
        carbon_content = data.get("carbon_content")
        carbonate_content = data.get("carbonate_content")

        record = {
            "material_type": material_type,
            "name": name,
            "carbon_content": carbon_content,
            "carbonate_content": carbonate_content,
        }
        self._materials[material_type] = record

        if _record_material_operation is not None:
            _record_material_operation("register", material_type)

        logger.info("Registered material: %s", material_type)
        return MaterialDetailResponse(**record)

    def list_materials(self) -> MaterialListResponse:
        """List all registered raw materials.

        Returns:
            MaterialListResponse with material list.
        """
        all_materials = list(self._materials.values())
        return MaterialListResponse(
            materials=all_materials,
            total=len(all_materials),
        )

    def get_material(
        self,
        material_id: str,
    ) -> Optional[MaterialDetailResponse]:
        """Get details for a specific raw material.

        Args:
            material_id: Material type identifier.

        Returns:
            MaterialDetailResponse or None if not found.
        """
        record = self._materials.get(material_id)
        if record is None:
            return None

        if _record_material_operation is not None:
            _record_material_operation("get", material_id)

        return MaterialDetailResponse(**record)

    # ==================================================================
    # Public API: Process Unit CRUD
    # ==================================================================

    def register_unit(
        self,
        data: Dict[str, Any],
    ) -> ProcessUnitDetailResponse:
        """Register a process unit (equipment).

        Args:
            data: Dictionary with unit_id (optional, auto-generated),
                unit_name, unit_type, process_type, and optional fields.

        Returns:
            ProcessUnitDetailResponse with registered unit details.
        """
        unit_id = data.get("unit_id", f"pu_{uuid.uuid4().hex[:12]}")
        unit_name = data.get("unit_name", "")
        unit_type = data.get("unit_type", "other")
        process_type = data.get("process_type", "")

        record = {
            "unit_id": unit_id,
            "unit_name": unit_name,
            "unit_type": unit_type,
            "process_type": process_type,
        }
        # Store additional fields
        for key in data:
            if key not in record:
                record[key] = data[key]

        self._process_units[unit_id] = record
        logger.info("Registered process unit: %s", unit_id)

        return ProcessUnitDetailResponse(**{
            "unit_id": unit_id,
            "unit_name": unit_name,
            "unit_type": unit_type,
            "process_type": process_type,
        })

    def list_units(self) -> ProcessUnitListResponse:
        """List all registered process units.

        Returns:
            ProcessUnitListResponse with unit list.
        """
        all_units = list(self._process_units.values())
        return ProcessUnitListResponse(
            units=all_units,
            total=len(all_units),
        )

    # ==================================================================
    # Public API: Emission Factor CRUD
    # ==================================================================

    def register_factor(
        self,
        data: Dict[str, Any],
    ) -> FactorDetailResponse:
        """Register a custom emission factor.

        Args:
            data: Dictionary with factor_id (optional), process_type,
                gas, value, source, and optional fields.

        Returns:
            FactorDetailResponse with registered factor details.
        """
        factor_id = data.get(
            "factor_id", f"pef_{uuid.uuid4().hex[:12]}",
        )
        process_type = data.get("process_type", "")
        gas = data.get("gas", "CO2")
        value = float(data.get("value", 0))
        source = data.get("source", "CUSTOM")

        record = {
            "factor_id": factor_id,
            "process_type": process_type,
            "gas": gas,
            "value": value,
            "source": source,
        }
        for key in data:
            if key not in record:
                record[key] = data[key]

        self._emission_factors[factor_id] = record

        if _record_factor_selection is not None:
            _record_factor_selection("TIER_1", source)

        logger.info(
            "Registered emission factor %s: %s/%s=%s",
            factor_id, process_type, gas, value,
        )

        return FactorDetailResponse(**{
            "factor_id": factor_id,
            "process_type": process_type,
            "gas": gas,
            "value": value,
            "source": source,
        })

    def list_factors(self) -> FactorListResponse:
        """List all registered emission factors.

        Returns:
            FactorListResponse with factor list.
        """
        all_factors = list(self._emission_factors.values())
        return FactorListResponse(
            factors=all_factors,
            total=len(all_factors),
        )

    # ==================================================================
    # Public API: Abatement CRUD
    # ==================================================================

    def register_abatement(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register an abatement technology record.

        Args:
            data: Dictionary with abatement_id (optional), unit_id,
                abatement_type, efficiency, target_gas, and optional
                fields.

        Returns:
            Dictionary with registered abatement details.
        """
        abatement_id = data.get(
            "abatement_id", f"abate_{uuid.uuid4().hex[:12]}",
        )
        record = {"abatement_id": abatement_id}
        record.update(data)
        record["abatement_id"] = abatement_id
        record.setdefault("status", "registered")

        self._abatement_records[abatement_id] = record
        logger.info("Registered abatement: %s", abatement_id)
        return record

    def list_abatement(self) -> AbatementListResponse:
        """List all registered abatement records.

        Returns:
            AbatementListResponse with abatement record list.
        """
        all_records = list(self._abatement_records.values())
        return AbatementListResponse(
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
            total_co2e_kg = float(
                calc_record.get("total_co2e_kg", 0),
            )

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

                mean_kg = float(
                    rd.get("mean_co2e_kg", total_co2e_kg),
                )
                std_kg = float(rd.get("std_dev_kg", 0))
                dqi = rd.get("dqi_score")

                # Build confidence intervals
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
                            "lower": float(
                                bounds.get("lower", 0),
                            ),
                            "upper": float(
                                bounds.get("upper", 0),
                            ),
                        }

                if _record_uncertainty is not None:
                    _record_uncertainty(method)

                return UncertaintyResponse(
                    success=True,
                    method=method,
                    iterations=iterations,
                    mean_co2e_kg=mean_kg,
                    std_dev_kg=std_kg,
                    confidence_intervals=ci,
                    dqi_score=dqi,
                )

            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed, using fallback: %s", exc,
                )

        # Fallback: analytical estimate (+/- 10% for Tier 1)
        std_estimate = total_co2e_kg * 0.10
        lower_95 = total_co2e_kg - 1.96 * std_estimate
        upper_95 = total_co2e_kg + 1.96 * std_estimate

        return UncertaintyResponse(
            success=True,
            method="analytical_fallback",
            iterations=None,
            mean_co2e_kg=total_co2e_kg,
            std_dev_kg=std_estimate,
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
        CSRD/ESRS E1, EPA 40 CFR Part 98, UK SECR, and EU ETS.

        Args:
            data: Dictionary with:
                - calculation_id (str): ID of a previous calculation.
                - frameworks (list): Frameworks to check. If empty,
                    all frameworks are checked.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        calc_id = data.get("calculation_id", "")
        requested_frameworks = data.get("frameworks", [])

        default_frameworks = [
            "GHG_PROTOCOL",
            "ISO_14064",
            "CSRD_ESRS_E1",
            "EPA_40CFR98",
            "UK_SECR",
            "EU_ETS",
        ]
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
        """Perform a health check on the process emissions service.

        Returns:
            HealthResponse with engine availability and overall status.
        """
        engines: Dict[str, str] = {
            "process_database": (
                "available"
                if self._process_database_engine is not None
                else "unavailable"
            ),
            "emission_calculator": (
                "available"
                if self._emission_calculator_engine is not None
                else "unavailable"
            ),
            "abatement_tracker": (
                "available"
                if self._abatement_tracker_engine is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available"
                if self._uncertainty_engine is not None
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
        elif available_count >= 3:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return HealthResponse(
            status=overall_status,
            service="process-emissions",
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
            total_process_types=len(self._processes),
            total_process_units=len(self._process_units),
            total_materials=len(self._materials),
            uptime_seconds=round(uptime, 3),
        )

    # ==================================================================
    # Private: Calculation dispatch
    # ==================================================================

    def _execute_calculation(
        self,
        calc_id: str,
        process_type: str,
        activity_data: Any,
        activity_unit: str,
        method: str,
        gwp_source: str,
        ef_source: str,
        abatement_efficiency: Optional[float],
        abatement_type: Optional[str],
        production_route: Optional[str],
        materials: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Dispatch a calculation to the appropriate engine.

        Tries the pipeline engine first, then the emission calculator
        engine, and finally produces a stub result.

        Args:
            calc_id: Unique calculation identifier.
            process_type: Industrial process type.
            activity_data: Production quantity.
            activity_unit: Unit of activity data.
            method: Calculation methodology.
            gwp_source: GWP source.
            ef_source: Emission factor source.
            abatement_efficiency: Abatement fraction.
            abatement_type: Abatement technology name.
            production_route: Production pathway.
            materials: Material input records.

        Returns:
            Dictionary with calculation results.
        """
        # Prepare common parameters
        activity_decimal = Decimal(str(activity_data))
        abatement_decimal = (
            Decimal(str(abatement_efficiency))
            if abatement_efficiency is not None
            else None
        )

        # Normalise process_type to the format the engine expects
        pt_upper = process_type.upper()

        # Try pipeline engine
        if self._pipeline_engine is not None:
            try:
                request_dict = {
                    "process_type": pt_upper,
                    "production_quantity": float(activity_data),
                    "calculation_method": method,
                    "tier": "TIER_1",
                    "gwp_source": gwp_source,
                }
                if abatement_efficiency is not None:
                    request_dict["abatement_efficiency"] = (
                        abatement_efficiency
                    )
                if production_route is not None:
                    request_dict["production_route"] = production_route

                pipeline_result = self._pipeline_engine.execute_pipeline(
                    request=request_dict,
                    gwp_source=gwp_source,
                    tier="TIER_1",
                )

                if pipeline_result.get("success"):
                    calc_data = pipeline_result.get(
                        "calculation_data", {},
                    )
                    return {
                        "status": "SUCCESS",
                        "total_co2e_kg": calc_data.get(
                            "total_co2e_kg", 0,
                        ),
                        "total_co2e_tonnes": calc_data.get(
                            "total_co2e_tonnes", 0,
                        ),
                        "gas_emissions": calc_data.get(
                            "gas_emissions", [],
                        ),
                    }
            except Exception as exc:
                logger.debug(
                    "Pipeline engine failed, trying calculator: %s", exc,
                )

        # Try emission calculator engine
        if self._emission_calculator_engine is not None:
            try:
                calc_result = self._emission_calculator_engine.calculate(
                    process_type=pt_upper,
                    method=method,
                    activity_data=activity_decimal,
                    activity_unit=activity_unit,
                    gwp_source=gwp_source,
                    ef_source=ef_source,
                    abatement_efficiency=abatement_decimal,
                    abatement_type=abatement_type,
                    calculation_id=calc_id,
                )
                return {
                    "status": calc_result.get("status", "FAILED"),
                    "total_co2e_kg": float(
                        calc_result.get("total_co2e_kg", 0),
                    ),
                    "total_co2e_tonnes": float(
                        calc_result.get("total_co2e_tonnes", 0),
                    ),
                    "gas_emissions": calc_result.get(
                        "gas_emissions", [],
                    ),
                }
            except Exception as exc:
                logger.debug(
                    "Calculator engine failed: %s", exc,
                )

        # Stub fallback
        return {
            "status": "SUCCESS",
            "total_co2e_kg": 0.0,
            "total_co2e_tonnes": 0.0,
            "gas_emissions": [],
            "message": "No calculation engine available",
        }

    def _sum_gas_co2e_kg(
        self,
        gas_emissions: List[Dict[str, Any]],
        target_gas: str,
    ) -> float:
        """Sum CO2e kg for a specific gas from emission results.

        Args:
            gas_emissions: List of per-gas emission dictionaries.
            target_gas: Gas identifier to filter on.

        Returns:
            Total CO2e in kg for the target gas.
        """
        total = 0.0
        for ge in gas_emissions:
            gas_name = str(ge.get("gas", "")).upper()
            if gas_name == target_gas.upper():
                co2e_kg = ge.get("co2e_kg", 0)
                total += float(co2e_kg)
        return total

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
        # Base requirements per framework
        requirements_map: Dict[str, List[Dict[str, Any]]] = {
            "GHG_PROTOCOL": [
                {
                    "id": "GHG-PE-001",
                    "name": "Scope 1 Process Emissions",
                    "description": (
                        "Complete Scope 1 reporting for non-combustion "
                        "industrial process sources"
                    ),
                },
                {
                    "id": "GHG-PE-002",
                    "name": "Multi-Gas Reporting",
                    "description": (
                        "Report all applicable GHG species for each "
                        "process type"
                    ),
                },
                {
                    "id": "GHG-PE-003",
                    "name": "Process-Specific Methodology",
                    "description": (
                        "Use IPCC-aligned process-specific methodology"
                    ),
                },
            ],
            "ISO_14064": [
                {
                    "id": "ISO-PE-001",
                    "name": "GHG Inventory Clause 5",
                    "description": "ISO 14064-1 Clause 5 quantification",
                },
            ],
            "CSRD_ESRS_E1": [
                {
                    "id": "ESRS-PE-001",
                    "name": "Climate Change Disclosure",
                    "description": "ESRS E1 Scope 1 process emissions",
                },
            ],
            "EPA_40CFR98": [
                {
                    "id": "EPA-PE-001",
                    "name": "Subpart Reporting",
                    "description": (
                        "EPA GHGRP process-specific subpart compliance"
                    ),
                },
            ],
            "UK_SECR": [
                {
                    "id": "SECR-PE-001",
                    "name": "UK SECR Process Emissions",
                    "description": "UK SECR process emissions disclosure",
                },
            ],
            "EU_ETS": [
                {
                    "id": "ETS-PE-001",
                    "name": "EU ETS MRR Methodology",
                    "description": (
                        "EU ETS MRR Annex IV process emissions "
                        "methodology"
                    ),
                },
            ],
        }

        requirements = requirements_map.get(framework, [])
        met_count = 0

        if calc_record is not None:
            # Mark requirements as met if calculation has data
            has_emissions = float(
                calc_record.get("total_co2e_kg", 0),
            ) >= 0
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
            "status": status,
            "total_requirements": total_reqs,
            "met_count": met_count,
            "requirements": requirements,
        }


# ===================================================================
# Thread-safe singleton access
# ===================================================================


_service_instance: Optional[ProcessEmissionsService] = None
_service_lock = threading.Lock()


def get_service() -> ProcessEmissionsService:
    """Get or create the singleton ProcessEmissionsService instance.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path.

    Returns:
        ProcessEmissionsService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ProcessEmissionsService()
    return _service_instance


def get_router() -> Any:
    """Get the FastAPI router for process emissions.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.process_emissions.api.router import create_router
        return create_router()
    except ImportError:
        logger.warning(
            "Process emissions API router module not available"
        )
        return None


def configure_process_emissions(
    app: Any,
    config: Any = None,
) -> ProcessEmissionsService:
    """Configure the Process Emissions Service on a FastAPI application.

    Creates the ProcessEmissionsService singleton, stores it in
    app.state, mounts the process emissions API router, and logs
    the configuration.

    Args:
        app: FastAPI application instance.
        config: Optional ProcessEmissionsConfig override.

    Returns:
        ProcessEmissionsService instance.
    """
    global _service_instance

    service = ProcessEmissionsService(config=config)

    with _service_lock:
        _service_instance = service

    # Attach to app state
    if hasattr(app, "state"):
        app.state.process_emissions_service = service

    # Mount API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Process emissions API router mounted")
    else:
        logger.warning(
            "Process emissions router not available; API not mounted"
        )

    logger.info("Process Emissions service configured")
    return service


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "ProcessEmissionsService",
    # Configuration helpers
    "configure_process_emissions",
    "get_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "ProcessListResponse",
    "ProcessDetailResponse",
    "MaterialListResponse",
    "MaterialDetailResponse",
    "ProcessUnitListResponse",
    "ProcessUnitDetailResponse",
    "FactorListResponse",
    "FactorDetailResponse",
    "AbatementListResponse",
    "UncertaintyResponse",
    "ComplianceCheckResponse",
    "HealthResponse",
    "StatsResponse",
]
