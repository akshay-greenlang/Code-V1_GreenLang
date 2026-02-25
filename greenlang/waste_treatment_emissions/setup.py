# -*- coding: utf-8 -*-
"""
Waste Treatment Emissions Service Setup - AGENT-MRV-008
=======================================================

Service facade for the On-site Waste Treatment Emissions Agent
(GL-MRV-SCOPE1-008).

Provides ``configure_waste_treatment(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``WasteTreatmentEmissionsService`` facade class that aggregates all
7 engines:

    1. WasteTreatmentDatabaseEngine    - Waste types, EFs, DOC tables
    2. BiologicalTreatmentEngine       - Composting, AD, MBT, vermicomposting
    3. ThermalTreatmentEngine          - Incineration, pyrolysis, gasification
    4. WastewaterTreatmentEngine       - BOD/COD CH4, N2O from wastewater
    5. UncertaintyQuantifierEngine     - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine         - Multi-framework regulatory compliance
    7. WasteTreatmentPipelineEngine    - Orchestrated calculation pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.waste_treatment_emissions.setup import (
    ...     configure_waste_treatment,
    ... )
    >>> app = FastAPI()
    >>> configure_waste_treatment(app)

    >>> from greenlang.waste_treatment_emissions.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "facility_id": "fac-001",
    ...     "waste_stream_id": "ws-001",
    ...     "treatment_method": "incineration",
    ...     "waste_category": "municipal_solid_waste",
    ...     "waste_quantity_tonnes": 500.0,
    ...     "calculation_method": "IPCC_TIER_1",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 On-site Waste Treatment Emissions (GL-MRV-SCOPE1-008)
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
    from greenlang.waste_treatment_emissions.config import (
        WasteTreatmentConfig,
        get_config,
    )
except ImportError:
    WasteTreatmentConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.waste_treatment_emissions.waste_treatment_database import (
        WasteTreatmentDatabaseEngine,
    )
    _DATABASE_ENGINE_AVAILABLE = True
except ImportError:
    WasteTreatmentDatabaseEngine = None  # type: ignore[assignment, misc]
    _DATABASE_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.biological_treatment import (
        BiologicalTreatmentEngine,
    )
    _BIOLOGICAL_ENGINE_AVAILABLE = True
except ImportError:
    BiologicalTreatmentEngine = None  # type: ignore[assignment, misc]
    _BIOLOGICAL_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.thermal_treatment import (
        ThermalTreatmentEngine,
    )
    _THERMAL_ENGINE_AVAILABLE = True
except ImportError:
    ThermalTreatmentEngine = None  # type: ignore[assignment, misc]
    _THERMAL_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.wastewater_treatment import (
        WastewaterTreatmentEngine,
    )
    _WASTEWATER_ENGINE_AVAILABLE = True
except ImportError:
    WastewaterTreatmentEngine = None  # type: ignore[assignment, misc]
    _WASTEWATER_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
    _UNCERTAINTY_ENGINE_AVAILABLE = True
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]
    _UNCERTAINTY_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
    _COMPLIANCE_ENGINE_AVAILABLE = True
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]
    _COMPLIANCE_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.waste_treatment_pipeline import (
        WasteTreatmentPipelineEngine,
    )
    _PIPELINE_ENGINE_AVAILABLE = True
except ImportError:
    WasteTreatmentPipelineEngine = None  # type: ignore[assignment, misc]
    _PIPELINE_ENGINE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.provenance import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.waste_treatment_emissions.metrics import (
        MetricsCollector as _MetricsCollector,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _MetricsCollector = None  # type: ignore[assignment, misc]
    _METRICS_AVAILABLE = False


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


def _short_id(prefix: str) -> str:
    """Generate a short prefixed identifier using UUID4 hex truncation."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serialisable object or Pydantic model instance.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# Waste treatment method categories
# ===================================================================

TREATMENT_METHODS: List[str] = [
    "landfill",
    "incineration",
    "composting",
    "anaerobic_digestion",
    "mechanical_biological_treatment",
    "open_burning",
    "pyrolysis",
    "gasification",
    "wastewater_treatment",
    "recycling",
]

WASTE_CATEGORIES: List[str] = [
    "municipal_solid_waste",
    "industrial_waste",
    "clinical_waste",
    "construction_demolition",
    "hazardous_waste",
    "agricultural_waste",
    "sewage_sludge",
    "electronic_waste",
    "food_waste",
    "garden_waste",
]

COMPLIANCE_FRAMEWORKS: List[str] = [
    "GHG_PROTOCOL",
    "IPCC_2006",
    "CSRD_ESRS_E1",
    "EU_WASTE_DIRECTIVE",
    "UK_SECR",
    "EPA_GHGRP",
    "UNFCCC",
]


# ===================================================================
# Lightweight Pydantic response models (14 models)
# ===================================================================


class CalculateResponse(BaseModel):
    """Single waste treatment emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    facility_id: str = Field(default="")
    waste_stream_id: str = Field(default="")
    treatment_method: str = Field(default="")
    calculation_method: str = Field(default="IPCC_TIER_1")
    waste_category: str = Field(default="")
    waste_quantity_tonnes: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    co2_tonnes: float = Field(default=0.0)
    ch4_tonnes: float = Field(default=0.0)
    n2o_tonnes: float = Field(default=0.0)
    biogenic_co2_tonnes: float = Field(default=0.0)
    fossil_co2_tonnes: float = Field(default=0.0)
    emissions_by_gas: Dict[str, float] = Field(default_factory=dict)
    methane_recovered_tonnes: float = Field(default=0.0)
    energy_recovered_gj: float = Field(default=0.0)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
    """Batch waste treatment emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    batch_id: str = Field(default="")
    tenant_id: str = Field(default="default")
    total_calculations: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_ch4_tonnes: float = Field(default=0.0)
    total_n2o_tonnes: float = Field(default=0.0)
    total_biogenic_co2_tonnes: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class FacilityResponse(BaseModel):
    """Response for a single waste treatment facility."""

    model_config = ConfigDict(frozen=True)

    facility_id: str = Field(default="")
    name: str = Field(default="")
    facility_type: str = Field(default="")
    treatment_methods: List[str] = Field(default_factory=list)
    capacity_tonnes_per_year: float = Field(default=0.0)
    latitude: float = Field(default=0.0)
    longitude: float = Field(default=0.0)
    country_code: str = Field(default="")
    tenant_id: str = Field(default="")
    permit_number: str = Field(default="")
    operational_status: str = Field(default="active")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class FacilityListResponse(BaseModel):
    """Response listing waste treatment facilities."""

    model_config = ConfigDict(frozen=True)

    facilities: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class WasteStreamResponse(BaseModel):
    """Response for a single waste stream definition."""

    model_config = ConfigDict(frozen=True)

    stream_id: str = Field(default="")
    facility_id: str = Field(default="")
    name: str = Field(default="")
    waste_category: str = Field(default="")
    doc_content: float = Field(default=0.0)
    moisture_content: float = Field(default=0.0)
    fossil_carbon_fraction: float = Field(default=0.0)
    generation_rate_tonnes_per_year: float = Field(default=0.0)
    tenant_id: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class WasteStreamListResponse(BaseModel):
    """Response listing waste streams."""

    model_config = ConfigDict(frozen=True)

    waste_streams: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class TreatmentEventResponse(BaseModel):
    """Response for a treatment event record."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default="")
    facility_id: str = Field(default="")
    waste_stream_id: str = Field(default="")
    treatment_method: str = Field(default="")
    waste_quantity_tonnes: float = Field(default=0.0)
    event_date: str = Field(default="")
    operating_parameters: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")


class MethaneRecoveryResponse(BaseModel):
    """Response for a methane recovery record."""

    model_config = ConfigDict(frozen=True)

    recovery_id: str = Field(default="")
    facility_id: str = Field(default="")
    collection_efficiency: float = Field(default=0.0)
    destruction_efficiency: float = Field(default=0.0)
    ch4_captured_tonnes: float = Field(default=0.0)
    ch4_flared_tonnes: float = Field(default=0.0)
    ch4_utilized_tonnes: float = Field(default=0.0)
    ch4_vented_tonnes: float = Field(default=0.0)
    recovery_date: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")


class EnergyRecoveryResponse(BaseModel):
    """Response for an energy recovery record."""

    model_config = ConfigDict(frozen=True)

    recovery_id: str = Field(default="")
    facility_id: str = Field(default="")
    energy_type: str = Field(default="electricity")
    energy_generated_gj: float = Field(default=0.0)
    grid_displacement_co2e_tonnes: float = Field(default=0.0)
    efficiency: float = Field(default=0.0)
    recovery_date: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")


class ComplianceCheckResponse(BaseModel):
    """Regulatory compliance check response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    id: str = Field(default="")
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)


class UncertaintyResponse(BaseModel):
    """Monte Carlo uncertainty analysis response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="monte_carlo")
    iterations: int = Field(default=0)
    mean_co2e_tonnes: float = Field(default=0.0)
    std_dev_tonnes: float = Field(default=0.0)
    ci_lower: float = Field(default=0.0)
    ci_upper: float = Field(default=0.0)
    confidence_level: float = Field(default=95.0)
    coefficient_of_variation: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)


class AggregationResponse(BaseModel):
    """Aggregated waste treatment emissions response."""

    model_config = ConfigDict(frozen=True)

    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_co2e_tonnes: float = Field(default=0.0)
    total_ch4_tonnes: float = Field(default=0.0)
    total_n2o_tonnes: float = Field(default=0.0)
    total_biogenic_co2_tonnes: float = Field(default=0.0)
    waste_processed_tonnes: float = Field(default=0.0)
    calculation_count: int = Field(default=0)
    period: str = Field(default="annual")


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="waste-treatment-emissions")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utcnow_iso)


class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_facilities: int = Field(default=0)
    total_waste_streams: int = Field(default=0)
    total_treatment_events: int = Field(default=0)
    total_methane_recovery_records: int = Field(default=0)
    total_energy_recovery_records: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_waste_processed_tonnes: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)


# ===================================================================
# WasteTreatmentEmissionsService facade
# ===================================================================

_singleton_lock = threading.RLock()
_singleton_instance: Optional["WasteTreatmentEmissionsService"] = None


class WasteTreatmentEmissionsService:
    """Unified facade over the Waste Treatment Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing when the
    provenance tracker is available.

    Engines:
        1. WasteTreatmentDatabaseEngine    - Emission factors and DOC tables
        2. BiologicalTreatmentEngine       - Composting, AD, MBT
        3. ThermalTreatmentEngine          - Incineration, pyrolysis, gasification
        4. WastewaterTreatmentEngine       - BOD/COD CH4 and N2O
        5. UncertaintyQuantifierEngine     - Monte Carlo uncertainty
        6. ComplianceCheckerEngine         - Multi-framework compliance
        7. WasteTreatmentPipelineEngine    - Orchestrated pipeline

    Example:
        >>> service = WasteTreatmentEmissionsService()
        >>> result = service.calculate({
        ...     "facility_id": "fac-001",
        ...     "waste_stream_id": "ws-001",
        ...     "treatment_method": "incineration",
        ...     "waste_category": "municipal_solid_waste",
        ...     "waste_quantity_tonnes": 500.0,
        ... })
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Waste Treatment Emissions Service facade.

        Args:
            config: Optional WasteTreatmentConfig override.  When None,
                the singleton from ``get_config()`` is used.
        """
        self.config = config if config is not None else get_config()
        self._start_time: float = time.monotonic()

        # Engine placeholders
        self._database_engine: Any = None
        self._biological_engine: Any = None
        self._thermal_engine: Any = None
        self._wastewater_engine: Any = None
        self._uncertainty_engine: Any = None
        self._compliance_engine: Any = None
        self._pipeline_engine: Any = None

        # Provenance tracker
        self._provenance_tracker: Any = None
        self._init_provenance_tracker()

        # Metrics collector
        self._metrics_collector: Any = None
        self._init_metrics_collector()

        # Initialize engines with graceful degradation
        self._init_engines()

        # In-memory stores
        self._calculations: List[Dict[str, Any]] = []
        self._facilities: Dict[str, Dict[str, Any]] = {}
        self._waste_streams: Dict[str, Dict[str, Any]] = {}
        self._treatment_events: List[Dict[str, Any]] = []
        self._methane_recovery: List[Dict[str, Any]] = []
        self._energy_recovery: List[Dict[str, Any]] = []
        self._compliance_results: List[Dict[str, Any]] = []
        self._uncertainty_results: List[Dict[str, Any]] = []

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_co2e: float = 0.0
        self._total_waste_processed: float = 0.0

        logger.info("WasteTreatmentEmissionsService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def database_engine(self) -> Any:
        """Get the WasteTreatmentDatabaseEngine instance."""
        return self._database_engine

    @property
    def biological_engine(self) -> Any:
        """Get the BiologicalTreatmentEngine instance."""
        return self._biological_engine

    @property
    def thermal_engine(self) -> Any:
        """Get the ThermalTreatmentEngine instance."""
        return self._thermal_engine

    @property
    def wastewater_engine(self) -> Any:
        """Get the WastewaterTreatmentEngine instance."""
        return self._wastewater_engine

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty_engine

    @property
    def compliance_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance."""
        return self._compliance_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the WasteTreatmentPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Provenance tracker initialization
    # ------------------------------------------------------------------

    def _init_provenance_tracker(self) -> None:
        """Initialise the provenance tracker with graceful degradation."""
        if _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            try:
                self._provenance_tracker = ProvenanceTracker()
                logger.info("ProvenanceTracker initialized")
            except Exception as exc:
                logger.warning(
                    "ProvenanceTracker init failed: %s", exc,
                )
        else:
            logger.warning("ProvenanceTracker not available")

    # ------------------------------------------------------------------
    # Metrics collector initialization
    # ------------------------------------------------------------------

    def _init_metrics_collector(self) -> None:
        """Initialise the metrics collector with graceful degradation."""
        if _METRICS_AVAILABLE and _MetricsCollector is not None:
            try:
                self._metrics_collector = _MetricsCollector()
                logger.info("MetricsCollector initialized")
            except Exception as exc:
                logger.warning(
                    "MetricsCollector init failed: %s", exc,
                )
        else:
            logger.warning("MetricsCollector not available")

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines."""
        config_dict: Dict[str, Any] = {}
        if self.config is not None and hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()

        # E1: WasteTreatmentDatabaseEngine
        self._init_single_engine(
            "WasteTreatmentDatabaseEngine",
            WasteTreatmentDatabaseEngine,
            "_database_engine",
            config_dict,
        )

        # E2: BiologicalTreatmentEngine
        self._init_single_engine(
            "BiologicalTreatmentEngine",
            BiologicalTreatmentEngine,
            "_biological_engine",
            config_dict,
        )

        # E3: ThermalTreatmentEngine
        self._init_single_engine(
            "ThermalTreatmentEngine",
            ThermalTreatmentEngine,
            "_thermal_engine",
            config_dict,
        )

        # E4: WastewaterTreatmentEngine
        self._init_single_engine(
            "WastewaterTreatmentEngine",
            WastewaterTreatmentEngine,
            "_wastewater_engine",
            config_dict,
        )

        # E5: UncertaintyQuantifierEngine
        self._init_single_engine(
            "UncertaintyQuantifierEngine",
            UncertaintyQuantifierEngine,
            "_uncertainty_engine",
            config_dict,
        )

        # E6: ComplianceCheckerEngine
        self._init_single_engine(
            "ComplianceCheckerEngine",
            ComplianceCheckerEngine,
            "_compliance_engine",
            config_dict,
        )

        # E7: WasteTreatmentPipelineEngine (composite - receives engines)
        if _PIPELINE_ENGINE_AVAILABLE and WasteTreatmentPipelineEngine is not None:
            try:
                self._pipeline_engine = WasteTreatmentPipelineEngine(
                    database_engine=self._database_engine,
                    biological_engine=self._biological_engine,
                    thermal_engine=self._thermal_engine,
                    wastewater_engine=self._wastewater_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    compliance_engine=self._compliance_engine,
                    config=self.config,
                )
                logger.info("WasteTreatmentPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "WasteTreatmentPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("WasteTreatmentPipelineEngine not available")

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        attr_name: str,
        config_dict: Dict[str, Any],
    ) -> None:
        """Initialize a single engine with graceful degradation.

        Args:
            name: Human-readable engine name for logging.
            engine_class: Engine class or None if import failed.
            attr_name: Instance attribute name to set.
            config_dict: Configuration dictionary to pass.
        """
        if engine_class is not None:
            try:
                setattr(self, attr_name, engine_class(config=config_dict))
                logger.info("%s initialized", name)
            except Exception as exc:
                logger.warning("%s init failed: %s", name, exc)
        else:
            logger.warning("%s not available", name)

    # ------------------------------------------------------------------
    # Provenance recording helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data: Any = None,
    ) -> Optional[str]:
        """Record a provenance entry if the tracker is available.

        Args:
            entity_type: Type of entity (FACILITY, CALCULATION, etc.).
            entity_id: Unique entity identifier.
            action: Action performed (CREATE, CALCULATE, etc.).
            data: Optional data payload for hashing.

        Returns:
            The chain hash string, or None if provenance is unavailable.
        """
        if self._provenance_tracker is None:
            return None
        try:
            entry = self._provenance_tracker.record(
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                data=data,
            )
            return entry.hash_value
        except Exception as exc:
            logger.warning("Provenance recording failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Metrics recording helper
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        treatment_method: str = "",
        calculation_method: str = "",
        waste_category: str = "",
        duration_seconds: float = 0.0,
        co2e_tonnes: float = 0.0,
        gas: str = "",
    ) -> None:
        """Record metrics if the metrics collector is available.

        Args:
            treatment_method: Waste treatment method.
            calculation_method: Calculation methodology.
            waste_category: Waste category.
            duration_seconds: Calculation duration in seconds.
            co2e_tonnes: Emissions in tCO2e for counter.
            gas: Gas species for emissions metric.
        """
        if self._metrics_collector is None:
            return
        try:
            if treatment_method and calculation_method and waste_category:
                self._metrics_collector.record_calculation(
                    treatment_method, calculation_method, waste_category,
                )
            if duration_seconds > 0 and treatment_method:
                self._metrics_collector.observe_calculation_duration(
                    treatment_method,
                    calculation_method or "unknown",
                    duration_seconds,
                )
            if co2e_tonnes > 0 and gas and treatment_method:
                self._metrics_collector.record_emissions(
                    gas, treatment_method,
                    waste_category or "unknown",
                    co2e_tonnes,
                )
        except Exception as exc:
            logger.warning("Metrics recording failed: %s", exc)

    # ==================================================================
    # Public API [1/20]: Calculate
    # ==================================================================

    def calculate(
        self,
        request_data: Dict[str, Any],
    ) -> CalculateResponse:
        """Calculate waste treatment emissions for a single record.

        Computes GHG emissions (CO2, CH4, N2O) from a waste treatment
        operation using IPCC methodology (Tier 1/2/3), mass balance,
        first order decay, or direct measurement.

        Args:
            request_data: Calculation parameters including facility_id,
                waste_stream_id, treatment_method, waste_category,
                waste_quantity_tonnes, and optional overrides for
                calculation_method, gwp_source, emission_factors.

        Returns:
            CalculateResponse with per-gas breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("wt_calc")

        facility_id = request_data.get("facility_id", "")
        waste_stream_id = request_data.get("waste_stream_id", "")
        treatment_method = request_data.get("treatment_method", "")
        calc_method = request_data.get("calculation_method", "IPCC_TIER_1")
        waste_category = request_data.get("waste_category", "")
        waste_qty = float(request_data.get("waste_quantity_tonnes", 0))

        try:
            total_co2e = 0.0
            co2_tonnes = 0.0
            ch4_tonnes = 0.0
            n2o_tonnes = 0.0
            biogenic_co2 = 0.0
            fossil_co2 = 0.0
            emissions_by_gas: Dict[str, float] = {}
            ch4_recovered = 0.0
            energy_recovered = 0.0

            # Try pipeline engine first (full orchestration)
            if self._pipeline_engine is not None:
                pipe_result = self._execute_pipeline(request_data)
                if pipe_result is not None:
                    total_co2e = float(pipe_result.get("total_co2e_tonnes", 0))
                    co2_tonnes = float(pipe_result.get("co2_tonnes", 0))
                    ch4_tonnes = float(pipe_result.get("ch4_tonnes", 0))
                    n2o_tonnes = float(pipe_result.get("n2o_tonnes", 0))
                    biogenic_co2 = float(
                        pipe_result.get("biogenic_co2_tonnes", 0),
                    )
                    fossil_co2 = float(
                        pipe_result.get("fossil_co2_tonnes", 0),
                    )
                    emissions_by_gas = {
                        k: float(v)
                        for k, v in pipe_result.get(
                            "emissions_by_gas", {},
                        ).items()
                    }
                    ch4_recovered = float(
                        pipe_result.get("methane_recovered_tonnes", 0),
                    )
                    energy_recovered = float(
                        pipe_result.get("energy_recovered_gj", 0),
                    )

            # Fallback: route to treatment-specific engine
            elif total_co2e == 0.0:
                fallback = self._calculate_fallback(request_data)
                total_co2e = fallback.get("total_co2e_tonnes", 0.0)
                co2_tonnes = fallback.get("co2_tonnes", 0.0)
                ch4_tonnes = fallback.get("ch4_tonnes", 0.0)
                n2o_tonnes = fallback.get("n2o_tonnes", 0.0)
                biogenic_co2 = fallback.get("biogenic_co2_tonnes", 0.0)
                fossil_co2 = fallback.get("fossil_co2_tonnes", 0.0)
                emissions_by_gas = fallback.get("emissions_by_gas", {})

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "facility_id": facility_id,
                "treatment_method": treatment_method,
                "waste_quantity_tonnes": waste_qty,
                "total_co2e": total_co2e,
            })

            # Record provenance
            self._record_provenance(
                "CALCULATION", calc_id, "CALCULATE",
                data={
                    "total_co2e_tonnes": total_co2e,
                    "treatment_method": treatment_method,
                    "waste_quantity_tonnes": waste_qty,
                },
            )

            # Record metrics
            self._record_metrics(
                treatment_method=treatment_method,
                calculation_method=calc_method,
                waste_category=waste_category,
                duration_seconds=elapsed_ms / 1000.0,
                co2e_tonnes=total_co2e,
                gas="CO2e",
            )

            response = CalculateResponse(
                success=True,
                calculation_id=calc_id,
                facility_id=facility_id,
                waste_stream_id=waste_stream_id,
                treatment_method=treatment_method,
                calculation_method=calc_method,
                waste_category=waste_category,
                waste_quantity_tonnes=waste_qty,
                total_co2e_tonnes=round(total_co2e, 6),
                co2_tonnes=round(co2_tonnes, 6),
                ch4_tonnes=round(ch4_tonnes, 6),
                n2o_tonnes=round(n2o_tonnes, 6),
                biogenic_co2_tonnes=round(biogenic_co2, 6),
                fossil_co2_tonnes=round(fossil_co2, 6),
                emissions_by_gas=emissions_by_gas,
                methane_recovered_tonnes=round(ch4_recovered, 6),
                energy_recovered_gj=round(energy_recovered, 4),
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

            # Cache the calculation
            self._cache_calculation(
                calc_id, request_data, response, provenance_hash,
            )
            self._total_calculations += 1
            self._total_co2e += total_co2e
            self._total_waste_processed += waste_qty

            logger.info(
                "Calculated %s: method=%s treatment=%s "
                "co2e=%.4f tonnes, waste=%.2f tonnes",
                calc_id, calc_method, treatment_method,
                total_co2e, waste_qty,
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate failed: %s", exc, exc_info=True,
            )
            if self._metrics_collector is not None:
                try:
                    self._metrics_collector.record_calculation_error(
                        "calculation_error",
                    )
                except Exception:
                    pass
            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                facility_id=facility_id,
                waste_stream_id=waste_stream_id,
                treatment_method=treatment_method,
                calculation_method=calc_method,
                waste_category=waste_category,
                waste_quantity_tonnes=waste_qty,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _execute_pipeline(
        self,
        request_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Delegate calculation to the pipeline engine.

        Args:
            request_data: Raw calculation request.

        Returns:
            Pipeline result dictionary or None on failure.
        """
        try:
            result = self._pipeline_engine.execute_pipeline(
                request=request_data,
                gwp_source=request_data.get("gwp_source", "AR6"),
            )
            return result.get("calculation_data", result)
        except Exception as exc:
            logger.warning(
                "Pipeline engine execution failed: %s", exc,
            )
            return None

    def _calculate_fallback(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Route calculation to treatment-specific engine.

        Falls back to a zero-emission stub when no engine is available.

        Args:
            request_data: Raw calculation request.

        Returns:
            Dictionary with per-gas emission values.
        """
        treatment_method = request_data.get("treatment_method", "")

        # Biological treatment methods
        biological_methods = {
            "composting", "anaerobic_digestion",
            "mechanical_biological_treatment",
        }
        # Thermal treatment methods
        thermal_methods = {
            "incineration", "pyrolysis", "gasification", "open_burning",
        }

        result: Dict[str, float] = {
            "total_co2e_tonnes": 0.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes": 0.0,
            "n2o_tonnes": 0.0,
            "biogenic_co2_tonnes": 0.0,
            "fossil_co2_tonnes": 0.0,
        }

        try:
            if (treatment_method in biological_methods
                    and self._biological_engine is not None):
                engine_result = self._biological_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(self._extract_gas_values(engine_result))

            elif (treatment_method in thermal_methods
                    and self._thermal_engine is not None):
                engine_result = self._thermal_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(self._extract_gas_values(engine_result))

            elif (treatment_method == "wastewater_treatment"
                    and self._wastewater_engine is not None):
                engine_result = self._wastewater_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(self._extract_gas_values(engine_result))

            elif self._database_engine is not None:
                # Generic fallback via database emission factors
                engine_result = self._database_engine.calculate_emissions(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(self._extract_gas_values(engine_result))

        except Exception as exc:
            logger.warning(
                "Fallback calculation failed for %s: %s",
                treatment_method, exc,
            )

        return result

    def _extract_gas_values(
        self,
        engine_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract per-gas emission values from an engine result.

        Args:
            engine_result: Raw engine output dictionary.

        Returns:
            Dictionary with standardised per-gas keys.
        """
        emissions_by_gas: Dict[str, float] = {}
        raw_by_gas = engine_result.get("emissions_by_gas", {})
        for gas, val in raw_by_gas.items():
            emissions_by_gas[gas] = float(val)

        return {
            "total_co2e_tonnes": float(
                engine_result.get("total_co2e_tonnes", 0),
            ),
            "co2_tonnes": float(engine_result.get("co2_tonnes", 0)),
            "ch4_tonnes": float(engine_result.get("ch4_tonnes", 0)),
            "n2o_tonnes": float(engine_result.get("n2o_tonnes", 0)),
            "biogenic_co2_tonnes": float(
                engine_result.get("biogenic_co2_tonnes", 0),
            ),
            "fossil_co2_tonnes": float(
                engine_result.get("fossil_co2_tonnes", 0),
            ),
            "emissions_by_gas": emissions_by_gas,
        }

    def _cache_calculation(
        self,
        calc_id: str,
        request_data: Dict[str, Any],
        response: CalculateResponse,
        provenance_hash: str,
    ) -> None:
        """Cache a calculation result in the in-memory store.

        Args:
            calc_id: Unique calculation identifier.
            request_data: Original request data.
            response: The calculation response.
            provenance_hash: SHA-256 provenance hash.
        """
        self._calculations.append({
            "calculation_id": calc_id,
            "facility_id": response.facility_id,
            "waste_stream_id": response.waste_stream_id,
            "treatment_method": response.treatment_method,
            "calculation_method": response.calculation_method,
            "waste_category": response.waste_category,
            "waste_quantity_tonnes": response.waste_quantity_tonnes,
            "total_co2e_tonnes": response.total_co2e_tonnes,
            "co2_tonnes": response.co2_tonnes,
            "ch4_tonnes": response.ch4_tonnes,
            "n2o_tonnes": response.n2o_tonnes,
            "biogenic_co2_tonnes": response.biogenic_co2_tonnes,
            "fossil_co2_tonnes": response.fossil_co2_tonnes,
            "emissions_by_gas": response.emissions_by_gas,
            "methane_recovered_tonnes": response.methane_recovered_tonnes,
            "energy_recovered_gj": response.energy_recovered_gj,
            "provenance_hash": provenance_hash,
            "timestamp": response.timestamp,
            "status": "SUCCESS" if response.success else "FAILED",
            "tenant_id": request_data.get("tenant_id", ""),
        })

    # ==================================================================
    # Public API [2/20]: Batch Calculate
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
        tenant_id: str = "default",
    ) -> BatchCalculateResponse:
        """Batch calculate waste treatment emissions.

        Processes multiple waste treatment emission calculations in
        sequence, aggregating totals and collecting individual results.

        Args:
            requests: List of calculation request dictionaries.
            tenant_id: Tenant identifier applied to all calculations
                that do not specify their own tenant_id.

        Returns:
            BatchCalculateResponse with aggregate totals.
        """
        t0 = time.monotonic()
        batch_id = _short_id("wt_batch")
        results: List[Dict[str, Any]] = []
        total_co2e = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        total_biogenic = 0.0
        successful = 0
        failed = 0

        for req in requests:
            if "tenant_id" not in req:
                req["tenant_id"] = tenant_id

            resp = self.calculate(req)
            results.append(resp.model_dump())
            if resp.success:
                successful += 1
                total_co2e += resp.total_co2e_tonnes
                total_ch4 += resp.ch4_tonnes
                total_n2o += resp.n2o_tonnes
                total_biogenic += resp.biogenic_co2_tonnes
            else:
                failed += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._total_batch_runs += 1

        # Record provenance for the batch
        self._record_provenance(
            "BATCH", batch_id, "CALCULATE",
            data={
                "total_calculations": len(requests),
                "successful": successful,
                "failed": failed,
                "total_co2e_tonnes": total_co2e,
            },
        )

        return BatchCalculateResponse(
            success=failed == 0,
            batch_id=batch_id,
            tenant_id=tenant_id,
            total_calculations=len(requests),
            successful=successful,
            failed=failed,
            total_co2e_tonnes=round(total_co2e, 6),
            total_ch4_tonnes=round(total_ch4, 6),
            total_n2o_tonnes=round(total_n2o, 6),
            total_biogenic_co2_tonnes=round(total_biogenic, 6),
            results=results,
            processing_time_ms=round(elapsed_ms, 3),
        )

    # ==================================================================
    # Public API [3/20]: Get Calculation
    # ==================================================================

    def get_calculation(
        self,
        calculation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a calculation result by its identifier.

        Args:
            calculation_id: Unique calculation identifier.

        Returns:
            Calculation dictionary or None if not found.
        """
        for calc in self._calculations:
            if calc.get("calculation_id") == calculation_id:
                return calc
        return None

    # ==================================================================
    # Public API [4/20]: List Calculations
    # ==================================================================

    def list_calculations(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        treatment_method: Optional[str] = None,
        waste_category: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List calculation results with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            treatment_method: Optional filter by treatment method.
            waste_category: Optional filter by waste category.
            from_date: Optional ISO date string for lower bound.
            to_date: Optional ISO date string for upper bound.

        Returns:
            Dictionary with paginated calculation list.
        """
        filtered = list(self._calculations)

        if tenant_id is not None:
            filtered = [
                c for c in filtered
                if c.get("tenant_id") == tenant_id
            ]
        if treatment_method is not None:
            filtered = [
                c for c in filtered
                if c.get("treatment_method") == treatment_method
            ]
        if waste_category is not None:
            filtered = [
                c for c in filtered
                if c.get("waste_category") == waste_category
            ]
        if from_date is not None:
            filtered = [
                c for c in filtered
                if c.get("timestamp", "") >= from_date
            ]
        if to_date is not None:
            filtered = [
                c for c in filtered
                if c.get("timestamp", "") <= to_date
            ]

        # Sort by timestamp descending
        filtered.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True,
        )

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = filtered[start:end]

        return {
            "calculations": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API [5/20]: Delete Calculation
    # ==================================================================

    def delete_calculation(
        self,
        calculation_id: str,
    ) -> bool:
        """Delete a calculation result by its identifier.

        Args:
            calculation_id: Unique calculation identifier.

        Returns:
            True if the calculation was found and deleted, False otherwise.
        """
        for i, calc in enumerate(self._calculations):
            if calc.get("calculation_id") == calculation_id:
                removed = self._calculations.pop(i)
                self._record_provenance(
                    "CALCULATION", calculation_id, "DELETE",
                    data={"deleted_at": _utcnow_iso()},
                )
                logger.info("Deleted calculation %s", calculation_id)
                return True
        return False

    # ==================================================================
    # Public API [6/20]: Register Facility
    # ==================================================================

    def register_facility(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a new waste treatment facility.

        Args:
            data: Facility data including name, facility_type,
                treatment_methods, capacity_tonnes_per_year, latitude,
                longitude, country_code, tenant_id.

        Returns:
            Dictionary with the registered facility details.

        Raises:
            ValueError: If required fields are missing.
        """
        facility_id = _short_id("fac")

        # Validate required fields
        required_fields = [
            "name", "facility_type", "tenant_id",
        ]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()

        treatment_methods = data.get("treatment_methods", [])
        if isinstance(treatment_methods, str):
            treatment_methods = [treatment_methods]

        provenance_hash = _compute_hash({
            "facility_id": facility_id,
            "name": data.get("name"),
            "facility_type": data.get("facility_type"),
            "tenant_id": data.get("tenant_id"),
        })

        record: Dict[str, Any] = {
            "facility_id": facility_id,
            "name": data.get("name", ""),
            "facility_type": data.get("facility_type", ""),
            "treatment_methods": treatment_methods,
            "capacity_tonnes_per_year": float(
                data.get("capacity_tonnes_per_year", 0),
            ),
            "latitude": float(data.get("latitude", 0)),
            "longitude": float(data.get("longitude", 0)),
            "country_code": data.get("country_code", ""),
            "tenant_id": data.get("tenant_id", ""),
            "permit_number": data.get("permit_number", ""),
            "operational_status": data.get("operational_status", "active"),
            "climate_zone": data.get("climate_zone", "temperate"),
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._facilities[facility_id] = record
        self._record_provenance(
            "FACILITY", facility_id, "CREATE", data=record,
        )

        logger.info(
            "Registered facility %s: name=%s type=%s capacity=%.0f t/yr",
            facility_id, record["name"], record["facility_type"],
            record["capacity_tonnes_per_year"],
        )
        return record

    # ==================================================================
    # Public API [7/20]: List Facilities
    # ==================================================================

    def list_facilities(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        facility_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List registered waste treatment facilities with filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            facility_type: Optional filter by facility type.

        Returns:
            Dictionary with paginated facility list.
        """
        all_facilities = list(self._facilities.values())

        if tenant_id is not None:
            all_facilities = [
                f for f in all_facilities
                if f.get("tenant_id") == tenant_id
            ]
        if facility_type is not None:
            all_facilities = [
                f for f in all_facilities
                if f.get("facility_type") == facility_type
            ]

        total = len(all_facilities)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_facilities[start:end]

        return {
            "facilities": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API [8/20]: Update Facility
    # ==================================================================

    def update_facility(
        self,
        facility_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing waste treatment facility.

        Args:
            facility_id: Unique facility identifier.
            data: Dictionary of fields to update. Only non-None
                values will be applied.

        Returns:
            Updated facility dictionary.

        Raises:
            ValueError: If the facility is not found.
        """
        existing = self._facilities.get(facility_id)
        if existing is None:
            raise ValueError(f"Facility not found: {facility_id}")

        # Updatable fields whitelist
        updatable = {
            "name", "facility_type", "treatment_methods",
            "capacity_tonnes_per_year", "latitude", "longitude",
            "country_code", "permit_number", "operational_status",
            "climate_zone",
        }

        for key, value in data.items():
            if key in updatable and value is not None:
                if key == "capacity_tonnes_per_year":
                    value = float(value)
                elif key in ("latitude", "longitude"):
                    value = float(value)
                elif key == "treatment_methods" and isinstance(value, str):
                    value = [value]
                existing[key] = value

        existing["updated_at"] = _utcnow_iso()

        # Recompute provenance hash
        existing["provenance_hash"] = _compute_hash({
            "facility_id": facility_id,
            "name": existing.get("name"),
            "facility_type": existing.get("facility_type"),
            "updated_at": existing.get("updated_at"),
        })

        self._facilities[facility_id] = existing
        self._record_provenance(
            "FACILITY", facility_id, "UPDATE", data=existing,
        )

        logger.info("Updated facility %s", facility_id)
        return existing

    # ==================================================================
    # Public API [9/20]: Register Waste Stream
    # ==================================================================

    def register_waste_stream(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a new waste stream definition.

        Args:
            data: Waste stream data including facility_id, name,
                waste_category, doc_content, moisture_content,
                fossil_carbon_fraction, generation_rate_tonnes_per_year.

        Returns:
            Dictionary with the registered waste stream details.

        Raises:
            ValueError: If required fields are missing.
        """
        stream_id = _short_id("ws")

        required_fields = ["facility_id", "name", "waste_category"]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()

        provenance_hash = _compute_hash({
            "stream_id": stream_id,
            "facility_id": data.get("facility_id"),
            "name": data.get("name"),
            "waste_category": data.get("waste_category"),
        })

        record: Dict[str, Any] = {
            "stream_id": stream_id,
            "facility_id": data.get("facility_id", ""),
            "name": data.get("name", ""),
            "waste_category": data.get("waste_category", ""),
            "doc_content": float(data.get("doc_content", 0)),
            "moisture_content": float(data.get("moisture_content", 0)),
            "fossil_carbon_fraction": float(
                data.get("fossil_carbon_fraction", 0),
            ),
            "biogenic_carbon_fraction": float(
                data.get("biogenic_carbon_fraction", 0),
            ),
            "generation_rate_tonnes_per_year": float(
                data.get("generation_rate_tonnes_per_year", 0),
            ),
            "tenant_id": data.get("tenant_id", ""),
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._waste_streams[stream_id] = record
        self._record_provenance(
            "WASTE_STREAM", stream_id, "CREATE", data=record,
        )

        logger.info(
            "Registered waste stream %s: name=%s category=%s facility=%s",
            stream_id, record["name"], record["waste_category"],
            record["facility_id"],
        )
        return record

    # ==================================================================
    # Public API [10/20]: List Waste Streams
    # ==================================================================

    def list_waste_streams(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        waste_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List registered waste streams with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            facility_id: Optional filter by facility.
            waste_category: Optional filter by waste category.

        Returns:
            Dictionary with paginated waste stream list.
        """
        all_streams = list(self._waste_streams.values())

        if tenant_id is not None:
            all_streams = [
                s for s in all_streams
                if s.get("tenant_id") == tenant_id
            ]
        if facility_id is not None:
            all_streams = [
                s for s in all_streams
                if s.get("facility_id") == facility_id
            ]
        if waste_category is not None:
            all_streams = [
                s for s in all_streams
                if s.get("waste_category") == waste_category
            ]

        total = len(all_streams)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_streams[start:end]

        return {
            "waste_streams": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API [11/20]: Update Waste Stream
    # ==================================================================

    def update_waste_stream(
        self,
        stream_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an existing waste stream definition.

        Args:
            stream_id: Unique waste stream identifier.
            data: Dictionary of fields to update. Only non-None
                values will be applied.

        Returns:
            Updated waste stream dictionary.

        Raises:
            ValueError: If the waste stream is not found.
        """
        existing = self._waste_streams.get(stream_id)
        if existing is None:
            raise ValueError(f"Waste stream not found: {stream_id}")

        updatable = {
            "name", "waste_category", "doc_content", "moisture_content",
            "fossil_carbon_fraction", "biogenic_carbon_fraction",
            "generation_rate_tonnes_per_year",
        }

        for key, value in data.items():
            if key in updatable and value is not None:
                if key in (
                    "doc_content", "moisture_content",
                    "fossil_carbon_fraction", "biogenic_carbon_fraction",
                    "generation_rate_tonnes_per_year",
                ):
                    value = float(value)
                existing[key] = value

        existing["updated_at"] = _utcnow_iso()

        # Recompute provenance hash
        existing["provenance_hash"] = _compute_hash({
            "stream_id": stream_id,
            "name": existing.get("name"),
            "waste_category": existing.get("waste_category"),
            "updated_at": existing.get("updated_at"),
        })

        self._waste_streams[stream_id] = existing
        self._record_provenance(
            "WASTE_STREAM", stream_id, "UPDATE", data=existing,
        )

        logger.info("Updated waste stream %s", stream_id)
        return existing

    # ==================================================================
    # Public API [12/20]: Record Treatment Event
    # ==================================================================

    def record_treatment_event(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a waste treatment event.

        Args:
            data: Treatment event data including facility_id,
                waste_stream_id, treatment_method, waste_quantity_tonnes,
                event_date, and optional operating_parameters.

        Returns:
            Dictionary with the recorded treatment event details.
        """
        event_id = _short_id("te")
        facility_id = data.get("facility_id", "")
        waste_stream_id = data.get("waste_stream_id", "")
        treatment_method = data.get("treatment_method", "")
        waste_qty = float(data.get("waste_quantity_tonnes", 0))
        event_date = data.get("event_date", _utcnow_iso())
        operating_params = data.get("operating_parameters", {})

        provenance_hash = _compute_hash({
            "event_id": event_id,
            "facility_id": facility_id,
            "waste_stream_id": waste_stream_id,
            "treatment_method": treatment_method,
            "waste_quantity_tonnes": waste_qty,
            "event_date": event_date,
        })

        record: Dict[str, Any] = {
            "event_id": event_id,
            "facility_id": facility_id,
            "waste_stream_id": waste_stream_id,
            "treatment_method": treatment_method,
            "waste_quantity_tonnes": waste_qty,
            "event_date": event_date,
            "operating_parameters": operating_params,
            "tenant_id": data.get("tenant_id", ""),
            "provenance_hash": provenance_hash,
            "created_at": _utcnow_iso(),
        }
        self._treatment_events.append(record)
        self._record_provenance(
            "TREATMENT_EVENT", event_id, "TREAT", data=record,
        )

        # Record metrics for treatment type tracking
        if self._metrics_collector is not None:
            try:
                biological_methods = {
                    "composting", "anaerobic_digestion",
                    "mechanical_biological_treatment",
                }
                thermal_methods = {
                    "incineration", "pyrolysis", "gasification",
                }
                if treatment_method in biological_methods:
                    bio_type_map = {
                        "composting": "composting",
                        "anaerobic_digestion": "ad",
                        "mechanical_biological_treatment": "mbt",
                    }
                    self._metrics_collector.record_biological_treatment(
                        bio_type_map.get(treatment_method, treatment_method),
                    )
                elif treatment_method in thermal_methods:
                    self._metrics_collector.record_thermal_treatment(
                        treatment_method,
                    )
                self._metrics_collector.record_waste_processed(
                    treatment_method,
                    data.get("waste_category", "unknown"),
                    waste_qty,
                )
            except Exception:
                pass

        logger.info(
            "Recorded treatment event %s: facility=%s method=%s "
            "qty=%.2f tonnes",
            event_id, facility_id, treatment_method, waste_qty,
        )
        return record

    # ==================================================================
    # Public API [13/20]: List Treatment Events
    # ==================================================================

    def list_treatment_events(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        treatment_method: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List treatment event records with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            facility_id: Optional filter by facility.
            treatment_method: Optional filter by treatment method.
            from_date: Optional ISO date string for lower bound.
            to_date: Optional ISO date string for upper bound.

        Returns:
            Dictionary with paginated treatment event list.
        """
        filtered = list(self._treatment_events)

        if tenant_id is not None:
            filtered = [
                e for e in filtered
                if e.get("tenant_id") == tenant_id
            ]
        if facility_id is not None:
            filtered = [
                e for e in filtered
                if e.get("facility_id") == facility_id
            ]
        if treatment_method is not None:
            filtered = [
                e for e in filtered
                if e.get("treatment_method") == treatment_method
            ]
        if from_date is not None:
            filtered = [
                e for e in filtered
                if e.get("event_date", "") >= from_date
            ]
        if to_date is not None:
            filtered = [
                e for e in filtered
                if e.get("event_date", "") <= to_date
            ]

        # Sort by event_date descending
        filtered.sort(
            key=lambda x: x.get("event_date", ""),
            reverse=True,
        )

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = filtered[start:end]

        return {
            "treatment_events": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API [14/20]: Record Methane Recovery
    # ==================================================================

    def record_methane_recovery(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a methane recovery event.

        Tracks landfill gas or biogas capture, flaring, utilisation,
        and venting volumes for a facility.

        Args:
            data: Methane recovery data including facility_id,
                collection_efficiency, destruction_efficiency,
                ch4_captured_tonnes, ch4_flared_tonnes,
                ch4_utilized_tonnes, ch4_vented_tonnes, recovery_date.

        Returns:
            Dictionary with the recorded methane recovery details.
        """
        recovery_id = _short_id("mr")
        facility_id = data.get("facility_id", "")
        collection_eff = float(data.get("collection_efficiency", 0))
        destruction_eff = float(data.get("destruction_efficiency", 0))
        ch4_captured = float(data.get("ch4_captured_tonnes", 0))
        ch4_flared = float(data.get("ch4_flared_tonnes", 0))
        ch4_utilized = float(data.get("ch4_utilized_tonnes", 0))
        ch4_vented = float(data.get("ch4_vented_tonnes", 0))
        recovery_date = data.get("recovery_date", _utcnow_iso())

        provenance_hash = _compute_hash({
            "recovery_id": recovery_id,
            "facility_id": facility_id,
            "ch4_captured_tonnes": ch4_captured,
            "ch4_flared_tonnes": ch4_flared,
            "ch4_utilized_tonnes": ch4_utilized,
            "recovery_date": recovery_date,
        })

        record: Dict[str, Any] = {
            "recovery_id": recovery_id,
            "facility_id": facility_id,
            "collection_efficiency": collection_eff,
            "destruction_efficiency": destruction_eff,
            "ch4_captured_tonnes": ch4_captured,
            "ch4_flared_tonnes": ch4_flared,
            "ch4_utilized_tonnes": ch4_utilized,
            "ch4_vented_tonnes": ch4_vented,
            "recovery_date": recovery_date,
            "tenant_id": data.get("tenant_id", ""),
            "provenance_hash": provenance_hash,
            "created_at": _utcnow_iso(),
        }
        self._methane_recovery.append(record)
        self._record_provenance(
            "METHANE_RECOVERY", recovery_id, "CAPTURE", data=record,
        )

        # Record methane recovery metrics
        if self._metrics_collector is not None:
            try:
                if ch4_captured > 0:
                    self._metrics_collector.record_methane_recovery(
                        "captured", ch4_captured,
                    )
                if ch4_flared > 0:
                    self._metrics_collector.record_methane_recovery(
                        "flared", ch4_flared,
                    )
                if ch4_utilized > 0:
                    self._metrics_collector.record_methane_recovery(
                        "utilized", ch4_utilized,
                    )
                if ch4_vented > 0:
                    self._metrics_collector.record_methane_recovery(
                        "vented", ch4_vented,
                    )
            except Exception:
                pass

        logger.info(
            "Recorded methane recovery %s: facility=%s "
            "captured=%.2f flared=%.2f utilized=%.2f vented=%.2f tonnes",
            recovery_id, facility_id, ch4_captured,
            ch4_flared, ch4_utilized, ch4_vented,
        )
        return record

    # ==================================================================
    # Public API [15/20]: Get Methane Recovery
    # ==================================================================

    def get_methane_recovery(
        self,
        facility_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Get methane recovery history for a facility.

        Args:
            facility_id: Waste treatment facility identifier.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Dictionary with paginated methane recovery list.
        """
        filtered = [
            r for r in self._methane_recovery
            if r.get("facility_id") == facility_id
        ]

        # Sort by recovery_date descending
        filtered.sort(
            key=lambda x: x.get("recovery_date", ""),
            reverse=True,
        )

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = filtered[start:end]

        # Compute summary totals
        total_captured = sum(
            float(r.get("ch4_captured_tonnes", 0)) for r in filtered
        )
        total_flared = sum(
            float(r.get("ch4_flared_tonnes", 0)) for r in filtered
        )
        total_utilized = sum(
            float(r.get("ch4_utilized_tonnes", 0)) for r in filtered
        )
        total_vented = sum(
            float(r.get("ch4_vented_tonnes", 0)) for r in filtered
        )

        return {
            "facility_id": facility_id,
            "records": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
            "summary": {
                "total_captured_tonnes": round(total_captured, 6),
                "total_flared_tonnes": round(total_flared, 6),
                "total_utilized_tonnes": round(total_utilized, 6),
                "total_vented_tonnes": round(total_vented, 6),
            },
        }

    # ==================================================================
    # Public API [16/20]: Check Compliance
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run multi-framework compliance check.

        Checks waste treatment emission calculations against applicable
        regulatory frameworks (GHG Protocol, IPCC 2006, CSRD ESRS E1,
        EU Waste Directive, UK SECR, EPA GHGRP, UNFCCC).

        Args:
            data: Compliance check parameters including optional
                calculation_id and frameworks list.

        Returns:
            Dictionary with compliance check results.
        """
        compliance_id = _short_id("comp")
        calc_id = data.get("calculation_id", "")
        frameworks = data.get("frameworks", [])

        # Find referenced calculation
        calc_record = None
        for c in self._calculations:
            if c.get("calculation_id") == calc_id:
                calc_record = c
                break

        # Delegate to compliance engine
        if self._compliance_engine is not None:
            try:
                compliance_data = dict(calc_record) if calc_record else {}
                compliance_data.update(data)
                result = self._compliance_engine.check_compliance(
                    calculation_data=compliance_data,
                    frameworks=frameworks if frameworks else None,
                )
                return self._format_compliance_result(
                    compliance_id, result,
                )
            except Exception as exc:
                logger.warning("Compliance engine failed: %s", exc)

        # Fallback
        return self._compliance_fallback(
            compliance_id, frameworks,
        )

    def _format_compliance_result(
        self,
        compliance_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format engine compliance result into response dict.

        Args:
            compliance_id: Unique compliance check identifier.
            result: Raw engine result.

        Returns:
            Formatted compliance result dictionary.
        """
        fw_results = result.get("results", {})
        results_list: List[Dict[str, Any]] = []
        for fw, fw_result in fw_results.items():
            if isinstance(fw_result, dict):
                fw_result["framework"] = fw
                results_list.append(fw_result)

        compliant_count = int(result.get("compliant", 0))
        non_compliant_count = int(result.get("non_compliant", 0))
        partial_count = int(result.get("partial", 0))

        # Record compliance metrics
        if self._metrics_collector is not None:
            try:
                for fw_item in results_list:
                    fw_name = fw_item.get("framework", "unknown")
                    status = fw_item.get("status", "not_assessed")
                    self._metrics_collector.record_compliance_check(
                        fw_name, status,
                    )
            except Exception:
                pass

        formatted: Dict[str, Any] = {
            "id": compliance_id,
            "success": True,
            "frameworks_checked": int(
                result.get("frameworks_checked", len(results_list)),
            ),
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "partial": partial_count,
            "results": results_list,
            "timestamp": _utcnow_iso(),
        }
        self._compliance_results.append(formatted)

        self._record_provenance(
            "COMPLIANCE_CHECK", compliance_id, "CHECK",
            data=formatted,
        )

        return formatted

    def _compliance_fallback(
        self,
        compliance_id: str,
        frameworks: List[str],
    ) -> Dict[str, Any]:
        """Generate a fallback compliance result.

        Returns not-assessed status for all requested frameworks
        when the compliance engine is unavailable.

        Args:
            compliance_id: Unique compliance check identifier.
            frameworks: Requested frameworks to check.

        Returns:
            Fallback compliance result dictionary.
        """
        check_frameworks = (
            frameworks if frameworks else COMPLIANCE_FRAMEWORKS
        )

        results_list: List[Dict[str, Any]] = []
        for fw in check_frameworks:
            results_list.append({
                "framework": fw,
                "status": "not_assessed",
                "total_requirements": 0,
                "passed": 0,
                "failed": 0,
                "findings": [],
                "recommendations": [
                    "Connect compliance engine for full assessment",
                ],
            })

        result: Dict[str, Any] = {
            "id": compliance_id,
            "success": True,
            "frameworks_checked": len(check_frameworks),
            "compliant": 0,
            "non_compliant": 0,
            "partial": 0,
            "results": results_list,
            "timestamp": _utcnow_iso(),
        }
        self._compliance_results.append(result)

        self._record_provenance(
            "COMPLIANCE_CHECK", compliance_id, "CHECK",
            data=result,
        )

        return result

    # ==================================================================
    # Public API [17/20]: Get Compliance Result
    # ==================================================================

    def get_compliance_result(
        self,
        compliance_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a compliance check result by its identifier.

        Args:
            compliance_id: Unique compliance check identifier.

        Returns:
            Compliance result dictionary or None if not found.
        """
        for result in self._compliance_results:
            if result.get("id") == compliance_id:
                return result
        return None

    # ==================================================================
    # Public API [18/20]: Run Uncertainty
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation.

        Supports Monte Carlo simulation and analytical (error
        propagation) methods for quantifying confidence intervals
        around waste treatment emission estimates.

        Args:
            data: Uncertainty parameters including calculation_id,
                iterations, seed, and confidence_level.

        Returns:
            Dictionary with uncertainty analysis results.
        """
        calc_id = data.get("calculation_id", "")
        method = data.get("method", "monte_carlo")
        iterations = int(data.get("iterations", 5000))
        seed = int(data.get("seed", 42))
        confidence_level = float(data.get("confidence_level", 95.0))

        # Find the referenced calculation
        calc_record = None
        for c in self._calculations:
            if c.get("calculation_id") == calc_id:
                calc_record = c
                break

        total_co2e = 0.0
        if calc_record is not None:
            total_co2e = float(
                calc_record.get("total_co2e_tonnes", 0),
            )

        # Delegate to uncertainty engine
        if (self._uncertainty_engine is not None
                and calc_record is not None):
            try:
                result = self._uncertainty_engine.quantify_uncertainty(
                    calculation_input=calc_record,
                    method=method,
                    n_iterations=iterations,
                    seed=seed,
                )
                formatted = self._format_uncertainty_result(
                    calc_id, result, iterations, confidence_level,
                )
                # Record metrics
                if self._metrics_collector is not None:
                    try:
                        self._metrics_collector.record_uncertainty_run(
                            method,
                        )
                    except Exception:
                        pass
                return formatted
            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed: %s", exc,
                )

        # Fallback: analytical estimate
        fallback = self._uncertainty_fallback(
            calc_id, total_co2e, iterations, confidence_level,
        )

        # Record metrics for fallback
        if self._metrics_collector is not None:
            try:
                self._metrics_collector.record_uncertainty_run(
                    "analytical_fallback",
                )
            except Exception:
                pass

        return fallback

    def _format_uncertainty_result(
        self,
        calc_id: str,
        result: Dict[str, Any],
        iterations: int,
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Format engine uncertainty result into response dict.

        Args:
            calc_id: Calculation identifier.
            result: Raw engine result.
            iterations: Monte Carlo iterations.
            confidence_level: Confidence level percentage.

        Returns:
            Formatted uncertainty result dictionary.
        """
        mean_co2e = float(result.get("mean_co2e_tonnes", 0))
        std_dev = float(result.get("std_dev_tonnes", 0))
        ci_lower = float(result.get("ci_lower", 0))
        ci_upper = float(result.get("ci_upper", 0))
        cv = float(result.get("coefficient_of_variation", 0))

        provenance_hash = _compute_hash({
            "calculation_id": calc_id,
            "mean_co2e": mean_co2e,
            "std_dev": std_dev,
            "iterations": iterations,
        })

        formatted: Dict[str, Any] = {
            "success": True,
            "calculation_id": calc_id,
            "method": "monte_carlo",
            "iterations": iterations,
            "mean_co2e_tonnes": round(mean_co2e, 6),
            "std_dev_tonnes": round(std_dev, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "confidence_level": confidence_level,
            "coefficient_of_variation": round(cv, 4),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        }
        self._uncertainty_results.append(formatted)

        self._record_provenance(
            "UNCERTAINTY_RUN", calc_id, "ASSESS",
            data=formatted,
        )

        return formatted

    def _uncertainty_fallback(
        self,
        calc_id: str,
        total_co2e: float,
        iterations: int,
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Compute analytical uncertainty fallback.

        Uses a conservative 50% uncertainty estimate when the Monte
        Carlo engine is unavailable, consistent with IPCC Tier 1
        default uncertainty ranges for waste sector emissions.

        Args:
            calc_id: Calculation identifier.
            total_co2e: Total CO2e emission estimate.
            iterations: Requested iterations (not used in fallback).
            confidence_level: Confidence level percentage.

        Returns:
            Analytical uncertainty result dictionary.
        """
        std_estimate = abs(total_co2e) * 0.50
        z_score = 1.96  # 95% CI default
        if confidence_level >= 99.0:
            z_score = 2.576
        elif confidence_level >= 95.0:
            z_score = 1.96
        elif confidence_level >= 90.0:
            z_score = 1.645

        ci_lower = total_co2e - z_score * std_estimate
        ci_upper = total_co2e + z_score * std_estimate
        cv = (
            (std_estimate / abs(total_co2e) * 100.0)
            if total_co2e != 0 else 0.0
        )

        provenance_hash = _compute_hash({
            "calculation_id": calc_id,
            "mean_co2e": total_co2e,
            "std_dev": std_estimate,
            "method": "analytical_fallback",
        })

        result: Dict[str, Any] = {
            "success": True,
            "calculation_id": calc_id,
            "method": "analytical_fallback",
            "iterations": 0,
            "mean_co2e_tonnes": round(total_co2e, 6),
            "std_dev_tonnes": round(std_estimate, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "confidence_level": confidence_level,
            "coefficient_of_variation": round(cv, 4),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        }
        self._uncertainty_results.append(result)

        self._record_provenance(
            "UNCERTAINTY_RUN", calc_id, "ASSESS",
            data=result,
        )

        return result

    # ==================================================================
    # Public API [19/20]: Aggregate
    # ==================================================================

    def aggregate(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate waste treatment emission results.

        Groups calculation results by tenant, facility, treatment
        method, waste category, or time period and computes totals.

        Args:
            data: Aggregation parameters including tenant_id, period,
                group_by, date_from, date_to, and optional filters
                for treatment_method and waste_category.

        Returns:
            Dictionary with aggregated emission totals.
        """
        tenant_id = data.get("tenant_id", "")
        period = data.get("period", "annual")
        group_by = data.get("group_by", ["treatment_method"])
        date_from = data.get("date_from")
        date_to = data.get("date_to")
        filter_treatment = data.get("treatment_method")
        filter_category = data.get("waste_category")

        # Filter calculations by tenant
        filtered = [
            c for c in self._calculations
            if c.get("tenant_id") == tenant_id
            and c.get("status") == "SUCCESS"
        ]

        # Apply date filters
        if date_from is not None:
            filtered = [
                c for c in filtered
                if c.get("timestamp", "") >= date_from
            ]
        if date_to is not None:
            filtered = [
                c for c in filtered
                if c.get("timestamp", "") <= date_to
            ]

        # Apply treatment method filter
        if filter_treatment is not None:
            filtered = [
                c for c in filtered
                if c.get("treatment_method") == filter_treatment
            ]

        # Apply waste category filter
        if filter_category is not None:
            filtered = [
                c for c in filtered
                if c.get("waste_category") == filter_category
            ]

        # Group and aggregate
        groups: Dict[str, Dict[str, float]] = {}
        total_co2e = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        total_biogenic = 0.0
        total_waste = 0.0

        for calc in filtered:
            # Build group key
            key_parts = []
            for field_name in group_by:
                key_parts.append(str(calc.get(field_name, "unknown")))
            group_key = "|".join(key_parts) if key_parts else "all"

            if group_key not in groups:
                groups[group_key] = {
                    "total_co2e_tonnes": 0.0,
                    "co2_tonnes": 0.0,
                    "ch4_tonnes": 0.0,
                    "n2o_tonnes": 0.0,
                    "biogenic_co2_tonnes": 0.0,
                    "waste_processed_tonnes": 0.0,
                    "count": 0.0,
                }

            co2e = float(calc.get("total_co2e_tonnes", 0))
            ch4 = float(calc.get("ch4_tonnes", 0))
            n2o = float(calc.get("n2o_tonnes", 0))
            biogenic = float(calc.get("biogenic_co2_tonnes", 0))
            waste = float(calc.get("waste_quantity_tonnes", 0))

            groups[group_key]["total_co2e_tonnes"] += co2e
            groups[group_key]["co2_tonnes"] += float(
                calc.get("co2_tonnes", 0),
            )
            groups[group_key]["ch4_tonnes"] += ch4
            groups[group_key]["n2o_tonnes"] += n2o
            groups[group_key]["biogenic_co2_tonnes"] += biogenic
            groups[group_key]["waste_processed_tonnes"] += waste
            groups[group_key]["count"] += 1

            total_co2e += co2e
            total_ch4 += ch4
            total_n2o += n2o
            total_biogenic += biogenic
            total_waste += waste

        self._record_provenance(
            "AGGREGATION", _short_id("agg"), "AGGREGATE",
            data={
                "tenant_id": tenant_id,
                "total_co2e_tonnes": total_co2e,
                "calculation_count": len(filtered),
            },
        )

        return {
            "groups": groups,
            "total_co2e_tonnes": round(total_co2e, 6),
            "total_ch4_tonnes": round(total_ch4, 6),
            "total_n2o_tonnes": round(total_n2o, 6),
            "total_biogenic_co2_tonnes": round(total_biogenic, 6),
            "waste_processed_tonnes": round(total_waste, 4),
            "calculation_count": len(filtered),
            "period": period,
            "tenant_id": tenant_id,
            "timestamp": _utcnow_iso(),
        }

    # ==================================================================
    # Public API [20/20]: Health Check
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Service health check.

        Returns engine availability status and overall service health.

        Returns:
            HealthResponse with engine availability mapping.
        """
        engines: Dict[str, str] = {
            "waste_treatment_database": (
                "available"
                if self._database_engine is not None
                else "unavailable"
            ),
            "biological_treatment": (
                "available"
                if self._biological_engine is not None
                else "unavailable"
            ),
            "thermal_treatment": (
                "available"
                if self._thermal_engine is not None
                else "unavailable"
            ),
            "wastewater_treatment": (
                "available"
                if self._wastewater_engine is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available"
                if self._uncertainty_engine is not None
                else "unavailable"
            ),
            "compliance_checker": (
                "available"
                if self._compliance_engine is not None
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
        total = len(engines)

        if available_count == total:
            status = "healthy"
        elif available_count >= 3:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            service="waste-treatment-emissions",
            version="1.0.0",
            engines=engines,
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Additional API: Get Stats
    # ==================================================================

    def get_stats(self) -> StatsResponse:
        """Get service statistics.

        Returns aggregate counts for all tracked entities and
        cumulative emission totals.

        Returns:
            StatsResponse with aggregate counts and uptime.
        """
        uptime = time.monotonic() - self._start_time

        # Compute total emissions from cached calculations
        total_emissions = sum(
            float(c.get("total_co2e_tonnes", 0))
            for c in self._calculations
            if c.get("status") == "SUCCESS"
        )

        # Compute total waste processed
        total_waste = sum(
            float(c.get("waste_quantity_tonnes", 0))
            for c in self._calculations
            if c.get("status") == "SUCCESS"
        )

        return StatsResponse(
            total_calculations=self._total_calculations,
            total_facilities=len(self._facilities),
            total_waste_streams=len(self._waste_streams),
            total_treatment_events=len(self._treatment_events),
            total_methane_recovery_records=len(self._methane_recovery),
            total_energy_recovery_records=len(self._energy_recovery),
            total_compliance_checks=len(self._compliance_results),
            total_uncertainty_runs=len(self._uncertainty_results),
            total_co2e_tonnes=round(total_emissions, 6),
            total_waste_processed_tonnes=round(total_waste, 4),
            uptime_seconds=round(uptime, 3),
        )


# ===================================================================
# Thread-safe singleton access
# ===================================================================

_service_instance: Optional[WasteTreatmentEmissionsService] = None
_service_lock = threading.RLock()


def get_service() -> WasteTreatmentEmissionsService:
    """Get or create the singleton WasteTreatmentEmissionsService instance.

    Uses double-checked locking with a reentrant lock for thread
    safety with minimal contention on the hot path.

    Returns:
        WasteTreatmentEmissionsService singleton instance.

    Example:
        >>> svc_a = get_service()
        >>> svc_b = get_service()
        >>> assert svc_a is svc_b
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = WasteTreatmentEmissionsService()
    return _service_instance


def reset_service() -> None:
    """Reset the singleton service instance for test teardown.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for use in test fixtures to prevent state leakage.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("WasteTreatmentEmissionsService singleton reset")


# ===================================================================
# FastAPI Router factory
# ===================================================================


def get_router() -> Any:
    """Get the FastAPI router for waste treatment emissions.

    Attempts to import the dedicated router module. Falls back to
    None if the router module or FastAPI is not available.

    Returns:
        FastAPI APIRouter or None if unavailable.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.waste_treatment_emissions.api.router import (
            create_router,
        )
        return create_router()
    except ImportError:
        logger.warning(
            "Waste treatment emissions API router module not available"
        )
        return None


# ===================================================================
# FastAPI application integration
# ===================================================================


def configure_waste_treatment(
    app: Any,
    config: Any = None,
) -> WasteTreatmentEmissionsService:
    """Configure the Waste Treatment Emissions Service on a FastAPI app.

    Creates the WasteTreatmentEmissionsService singleton, stores it
    in ``app.state``, and mounts the API router.

    Args:
        app: FastAPI application instance.
        config: Optional WasteTreatmentConfig override.

    Returns:
        WasteTreatmentEmissionsService instance.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> svc = configure_waste_treatment(app)
        >>> assert hasattr(app.state, "waste_treatment_emissions_service")
    """
    global _service_instance

    service = WasteTreatmentEmissionsService(config=config)

    with _service_lock:
        _service_instance = service

    if hasattr(app, "state"):
        app.state.waste_treatment_emissions_service = service

    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Waste treatment emissions API router mounted")
    else:
        logger.warning(
            "Waste treatment emissions router not available; "
            "API not mounted"
        )

    logger.info("Waste Treatment Emissions service configured")
    return service


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "WasteTreatmentEmissionsService",
    # Integration helpers
    "configure_waste_treatment",
    "get_service",
    "reset_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FacilityResponse",
    "FacilityListResponse",
    "WasteStreamResponse",
    "WasteStreamListResponse",
    "TreatmentEventResponse",
    "MethaneRecoveryResponse",
    "EnergyRecoveryResponse",
    "ComplianceCheckResponse",
    "UncertaintyResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
    # Constants
    "TREATMENT_METHODS",
    "WASTE_CATEGORIES",
    "COMPLIANCE_FRAMEWORKS",
]
