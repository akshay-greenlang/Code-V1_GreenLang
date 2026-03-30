# -*- coding: utf-8 -*-
"""
Agricultural Emissions Service Setup - AGENT-MRV-008
=====================================================

Service facade for the Agricultural Emissions Agent
(GL-MRV-SCOPE1-008).

Provides ``configure_agricultural_emissions(app)``, ``get_service()``,
and ``get_router()`` for FastAPI integration.  Also exposes the
``AgriculturalEmissionsService`` facade class that aggregates all
7 engines:

    1. AgriculturalDatabaseEngine     - Livestock EFs, crop EFs, feed data
    2. EntericFermentationEngine      - CH4 from ruminant digestion
    3. ManureManagementEngine         - CH4/N2O from manure storage/treatment
    4. CroplandEmissionsEngine        - N2O from fertiliser, lime, urea, residues
    5. UncertaintyQuantifierEngine    - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine        - Multi-framework regulatory compliance
    7. AgriculturalPipelineEngine     - Orchestrated calculation pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.agricultural_emissions.setup import (
    ...     configure_agricultural_emissions,
    ... )
    >>> app = FastAPI()
    >>> configure_agricultural_emissions(app)

    >>> from greenlang.agents.mrv.agricultural_emissions.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "farm_id": "farm-001",
    ...     "source_category": "enteric_fermentation",
    ...     "livestock_type": "dairy_cattle",
    ...     "head_count": 200,
    ...     "calculation_method": "IPCC_TIER_1",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
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

from pydantic import ConfigDict, Field
from greenlang.schemas import GreenLangBase, utcnow

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
    from greenlang.agents.mrv.agricultural_emissions.config import (
        AgriculturalConfig,
        get_config,
    )
except ImportError:
    AgriculturalConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agents.mrv.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    _DATABASE_ENGINE_AVAILABLE = True
except ImportError:
    AgriculturalDatabaseEngine = None  # type: ignore[assignment, misc]
    _DATABASE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.enteric_fermentation import (
        EntericFermentationEngine,
    )
    _ENTERIC_ENGINE_AVAILABLE = True
except ImportError:
    EntericFermentationEngine = None  # type: ignore[assignment, misc]
    _ENTERIC_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.manure_management import (
        ManureManagementEngine,
    )
    _MANURE_ENGINE_AVAILABLE = True
except ImportError:
    ManureManagementEngine = None  # type: ignore[assignment, misc]
    _MANURE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.cropland_emissions import (
        CroplandEmissionsEngine,
    )
    _CROPLAND_ENGINE_AVAILABLE = True
except ImportError:
    CroplandEmissionsEngine = None  # type: ignore[assignment, misc]
    _CROPLAND_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
    _UNCERTAINTY_ENGINE_AVAILABLE = True
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]
    _UNCERTAINTY_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
    _COMPLIANCE_ENGINE_AVAILABLE = True
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]
    _COMPLIANCE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.agricultural_pipeline import (
        AgriculturalPipelineEngine,
    )
    _PIPELINE_ENGINE_AVAILABLE = True
except ImportError:
    AgriculturalPipelineEngine = None  # type: ignore[assignment, misc]
    _PIPELINE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.provenance import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    _PROVENANCE_AVAILABLE = False

try:
    from greenlang.agents.mrv.agricultural_emissions.metrics import (
        MetricsCollector as _MetricsCollector,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _MetricsCollector = None  # type: ignore[assignment, misc]
    _METRICS_AVAILABLE = False

# ===================================================================
# Utility helpers
# ===================================================================

def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return utcnow().isoformat()

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
# Agricultural source categories and reference constants
# ===================================================================

SOURCE_CATEGORIES: List[str] = [
    "enteric_fermentation",
    "manure_management",
    "rice_cultivation",
    "agricultural_soils",
    "field_burning",
    "liming",
    "urea_application",
]

LIVESTOCK_TYPES: List[str] = [
    "dairy_cattle",
    "non_dairy_cattle",
    "buffalo",
    "sheep",
    "goats",
    "camels",
    "horses",
    "mules_asses",
    "swine",
    "poultry_layers",
    "poultry_broilers",
    "turkeys",
    "ducks",
    "deer",
    "alpacas",
    "llamas",
]

MANURE_SYSTEMS: List[str] = [
    "lagoon",
    "liquid_slurry",
    "solid_storage",
    "dry_lot",
    "pasture_range_paddock",
    "daily_spread",
    "anaerobic_digester",
    "composting",
    "deep_bedding",
    "pit_storage",
    "burned_for_fuel",
    "other",
]

CROP_INPUT_TYPES: List[str] = [
    "synthetic_fertiliser",
    "organic_fertiliser",
    "crop_residue",
    "manure_applied",
    "urea",
    "limestone",
    "dolomite",
    "rice_paddy",
]

RICE_WATER_REGIMES: List[str] = [
    "continuously_flooded",
    "single_aeration",
    "multiple_aeration",
    "rainfed_regular",
    "rainfed_drought",
    "deep_water",
    "upland",
]

COMPLIANCE_FRAMEWORKS: List[str] = [
    "GHG_PROTOCOL",
    "IPCC_2006",
    "IPCC_2019",
    "CSRD_ESRS_E1",
    "US_EPA_GHGRP",
    "EU_EFFORT_SHARING",
    "UNFCCC",
    "SBTi_FLAG",
]

# ===================================================================
# Lightweight Pydantic response models (14 models)
# ===================================================================

class CalculateResponse(GreenLangBase):
    """Single agricultural emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    farm_id: str = Field(default="")
    source_category: str = Field(default="")
    livestock_type: str = Field(default="")
    calculation_method: str = Field(default="IPCC_TIER_1")
    head_count: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    co2_tonnes: float = Field(default=0.0)
    ch4_tonnes: float = Field(default=0.0)
    n2o_tonnes: float = Field(default=0.0)
    ch4_co2e_tonnes: float = Field(default=0.0)
    n2o_co2e_tonnes: float = Field(default=0.0)
    emissions_by_gas: Dict[str, float] = Field(default_factory=dict)
    emissions_by_source: Dict[str, float] = Field(default_factory=dict)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)

class BatchCalculateResponse(GreenLangBase):
    """Batch agricultural emission calculation response."""

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
    total_ch4_co2e_tonnes: float = Field(default=0.0)
    total_n2o_co2e_tonnes: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)

class FarmResponse(GreenLangBase):
    """Response for a single farm / agricultural facility."""

    model_config = ConfigDict(frozen=True)

    farm_id: str = Field(default="")
    name: str = Field(default="")
    farm_type: str = Field(default="")
    area_ha: float = Field(default=0.0)
    latitude: float = Field(default=0.0)
    longitude: float = Field(default=0.0)
    country_code: str = Field(default="")
    climate_zone: str = Field(default="")
    soil_type: str = Field(default="")
    tenant_id: str = Field(default="")
    operational_status: str = Field(default="active")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class FarmListResponse(GreenLangBase):
    """Response listing farms / agricultural facilities."""

    model_config = ConfigDict(frozen=True)

    farms: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)

class LivestockResponse(GreenLangBase):
    """Response for a single livestock population record."""

    model_config = ConfigDict(frozen=True)

    herd_id: str = Field(default="")
    farm_id: str = Field(default="")
    livestock_type: str = Field(default="")
    head_count: int = Field(default=0)
    average_weight_kg: float = Field(default=0.0)
    milk_yield_kg_per_day: float = Field(default=0.0)
    feed_digestibility_pct: float = Field(default=0.0)
    gross_energy_mj_per_day: float = Field(default=0.0)
    manure_system: str = Field(default="")
    grazing_fraction: float = Field(default=0.0)
    pregnancy_fraction: float = Field(default=0.0)
    reporting_year: int = Field(default=0)
    tenant_id: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class LivestockListResponse(GreenLangBase):
    """Response listing livestock populations."""

    model_config = ConfigDict(frozen=True)

    livestock: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)

class CroplandInputResponse(GreenLangBase):
    """Response for a single cropland input record (fertiliser/lime/urea)."""

    model_config = ConfigDict(frozen=True)

    input_id: str = Field(default="")
    farm_id: str = Field(default="")
    input_type: str = Field(default="")
    product_name: str = Field(default="")
    quantity_tonnes: float = Field(default=0.0)
    nitrogen_content_fraction: float = Field(default=0.0)
    carbon_content_fraction: float = Field(default=0.0)
    area_applied_ha: float = Field(default=0.0)
    application_date: str = Field(default="")
    reporting_year: int = Field(default=0)
    tenant_id: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class RiceFieldResponse(GreenLangBase):
    """Response for a single rice field registration."""

    model_config = ConfigDict(frozen=True)

    field_id: str = Field(default="")
    farm_id: str = Field(default="")
    area_ha: float = Field(default=0.0)
    water_regime: str = Field(default="continuously_flooded")
    organic_amendment: str = Field(default="none")
    organic_amendment_rate_tonnes_per_ha: float = Field(default=0.0)
    cultivation_period_days: int = Field(default=120)
    soil_type: str = Field(default="")
    previous_crop: str = Field(default="")
    reporting_year: int = Field(default=0)
    baseline_ef_kg_ch4_per_ha: float = Field(default=0.0)
    scaling_factor_water: float = Field(default=1.0)
    scaling_factor_amendment: float = Field(default=1.0)
    scaling_factor_preseason: float = Field(default=1.0)
    tenant_id: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

class FieldBurningResponse(GreenLangBase):
    """Response for a single field burning event."""

    model_config = ConfigDict(frozen=True)

    burning_id: str = Field(default="")
    farm_id: str = Field(default="")
    crop_type: str = Field(default="")
    area_burned_ha: float = Field(default=0.0)
    residue_dry_matter_tonnes: float = Field(default=0.0)
    combustion_factor: float = Field(default=0.0)
    ch4_ef_g_per_kg: float = Field(default=0.0)
    n2o_ef_g_per_kg: float = Field(default=0.0)
    ch4_tonnes: float = Field(default=0.0)
    n2o_tonnes: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    burning_date: str = Field(default="")
    provenance_hash: str = Field(default="")
    created_at: str = Field(default="")

class ComplianceCheckResponse(GreenLangBase):
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

class UncertaintyResponse(GreenLangBase):
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

class AggregationResponse(GreenLangBase):
    """Aggregated agricultural emissions response."""

    model_config = ConfigDict(frozen=True)

    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_co2e_tonnes: float = Field(default=0.0)
    total_ch4_tonnes: float = Field(default=0.0)
    total_n2o_tonnes: float = Field(default=0.0)
    total_ch4_co2e_tonnes: float = Field(default=0.0)
    total_n2o_co2e_tonnes: float = Field(default=0.0)
    total_head_count: int = Field(default=0)
    calculation_count: int = Field(default=0)
    period: str = Field(default="annual")

class HealthResponse(GreenLangBase):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="agricultural-emissions")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utcnow_iso)

class StatsResponse(GreenLangBase):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_farms: int = Field(default=0)
    total_livestock_herds: int = Field(default=0)
    total_cropland_inputs: int = Field(default=0)
    total_rice_fields: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_head_count: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)

# ===================================================================
# AgriculturalEmissionsService facade
# ===================================================================

_singleton_lock = threading.RLock()
_singleton_instance: Optional["AgriculturalEmissionsService"] = None

class AgriculturalEmissionsService:
    """Unified facade over the Agricultural Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing when the
    provenance tracker is available.

    Engines:
        1. AgriculturalDatabaseEngine     - Emission factors, feed data
        2. EntericFermentationEngine      - CH4 from ruminant digestion
        3. ManureManagementEngine         - CH4/N2O from manure
        4. CroplandEmissionsEngine        - N2O from soils, lime, urea
        5. UncertaintyQuantifierEngine    - Monte Carlo uncertainty
        6. ComplianceCheckerEngine        - Multi-framework compliance
        7. AgriculturalPipelineEngine     - Orchestrated pipeline

    Example:
        >>> service = AgriculturalEmissionsService()
        >>> result = service.calculate({
        ...     "farm_id": "farm-001",
        ...     "source_category": "enteric_fermentation",
        ...     "livestock_type": "dairy_cattle",
        ...     "head_count": 200,
        ... })
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Agricultural Emissions Service facade.

        Args:
            config: Optional AgriculturalConfig override.  When None,
                the singleton from ``get_config()`` is used.
        """
        self.config = config if config is not None else get_config()
        self._start_time: float = time.monotonic()

        # Engine placeholders
        self._database_engine: Any = None
        self._enteric_engine: Any = None
        self._manure_engine: Any = None
        self._cropland_engine: Any = None
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
        self._farms: Dict[str, Dict[str, Any]] = {}
        self._livestock: Dict[str, Dict[str, Any]] = {}
        self._manure_systems: Dict[str, Dict[str, Any]] = {}
        self._cropland_inputs: Dict[str, Dict[str, Any]] = {}
        self._rice_fields: Dict[str, Dict[str, Any]] = {}
        self._compliance_results: List[Dict[str, Any]] = []
        self._uncertainty_results: List[Dict[str, Any]] = []

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_co2e: float = 0.0

        logger.info("AgriculturalEmissionsService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def database_engine(self) -> Any:
        """Get the AgriculturalDatabaseEngine instance."""
        return self._database_engine

    @property
    def enteric_engine(self) -> Any:
        """Get the EntericFermentationEngine instance."""
        return self._enteric_engine

    @property
    def manure_engine(self) -> Any:
        """Get the ManureManagementEngine instance."""
        return self._manure_engine

    @property
    def cropland_engine(self) -> Any:
        """Get the CroplandEmissionsEngine instance."""
        return self._cropland_engine

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
        """Get the AgriculturalPipelineEngine instance."""
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

        # E1: AgriculturalDatabaseEngine
        self._init_single_engine(
            "AgriculturalDatabaseEngine",
            AgriculturalDatabaseEngine,
            "_database_engine",
            config_dict,
        )

        # E2: EntericFermentationEngine
        self._init_single_engine(
            "EntericFermentationEngine",
            EntericFermentationEngine,
            "_enteric_engine",
            config_dict,
        )

        # E3: ManureManagementEngine
        self._init_single_engine(
            "ManureManagementEngine",
            ManureManagementEngine,
            "_manure_engine",
            config_dict,
        )

        # E4: CroplandEmissionsEngine
        self._init_single_engine(
            "CroplandEmissionsEngine",
            CroplandEmissionsEngine,
            "_cropland_engine",
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

        # E7: AgriculturalPipelineEngine (composite - receives engines)
        if (
            _PIPELINE_ENGINE_AVAILABLE
            and AgriculturalPipelineEngine is not None
        ):
            try:
                self._pipeline_engine = AgriculturalPipelineEngine(
                    database_engine=self._database_engine,
                    enteric_engine=self._enteric_engine,
                    manure_engine=self._manure_engine,
                    cropland_engine=self._cropland_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    compliance_engine=self._compliance_engine,
                    config=self.config,
                )
                logger.info("AgriculturalPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "AgriculturalPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("AgriculturalPipelineEngine not available")

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
            entity_type: Type of entity (FARM, CALCULATION, etc.).
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
        source_category: str = "",
        calculation_method: str = "",
        livestock_type: str = "",
        duration_seconds: float = 0.0,
        co2e_tonnes: float = 0.0,
        gas: str = "",
    ) -> None:
        """Record metrics if the metrics collector is available.

        Args:
            source_category: Agricultural emission source category.
            calculation_method: Calculation methodology.
            livestock_type: Livestock type (for enteric/manure).
            duration_seconds: Calculation duration in seconds.
            co2e_tonnes: Emissions in tCO2e for counter.
            gas: Gas species for emissions metric.
        """
        if self._metrics_collector is None:
            return
        try:
            if source_category and calculation_method:
                self._metrics_collector.record_calculation(
                    source_category, calculation_method,
                    livestock_type or "n/a",
                )
            if duration_seconds > 0 and source_category:
                self._metrics_collector.observe_calculation_duration(
                    source_category,
                    calculation_method or "unknown",
                    duration_seconds,
                )
            if co2e_tonnes > 0 and gas and source_category:
                self._metrics_collector.record_emissions(
                    gas, source_category,
                    livestock_type or "unknown",
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
        """Calculate agricultural emissions for a single record.

        Computes GHG emissions (CH4, N2O, CO2) from agricultural
        activities using IPCC methodology (Tier 1/2/3).  Supports
        enteric fermentation, manure management, cropland N2O,
        rice cultivation, field burning, liming, and urea.

        Args:
            request_data: Calculation parameters including farm_id,
                source_category, livestock_type, head_count, and
                optional overrides for calculation_method, gwp_source,
                emission_factors.

        Returns:
            CalculateResponse with per-gas breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("ag_calc")

        farm_id = request_data.get("farm_id", "")
        source_category = request_data.get("source_category", "")
        livestock_type = request_data.get("livestock_type", "")
        calc_method = request_data.get(
            "calculation_method", "IPCC_TIER_1",
        )
        head_count = int(request_data.get("head_count", 0))

        try:
            total_co2e = 0.0
            co2_tonnes = 0.0
            ch4_tonnes = 0.0
            n2o_tonnes = 0.0
            ch4_co2e = 0.0
            n2o_co2e = 0.0
            emissions_by_gas: Dict[str, float] = {}
            emissions_by_source: Dict[str, float] = {}

            # Try pipeline engine first (full orchestration)
            if self._pipeline_engine is not None:
                pipe_result = self._execute_pipeline(request_data)
                if pipe_result is not None:
                    total_co2e = float(
                        pipe_result.get("total_co2e_tonnes", 0),
                    )
                    co2_tonnes = float(
                        pipe_result.get("co2_tonnes", 0),
                    )
                    ch4_tonnes = float(
                        pipe_result.get("ch4_tonnes", 0),
                    )
                    n2o_tonnes = float(
                        pipe_result.get("n2o_tonnes", 0),
                    )
                    ch4_co2e = float(
                        pipe_result.get("ch4_co2e_tonnes", 0),
                    )
                    n2o_co2e = float(
                        pipe_result.get("n2o_co2e_tonnes", 0),
                    )
                    emissions_by_gas = {
                        k: float(v)
                        for k, v in pipe_result.get(
                            "emissions_by_gas", {},
                        ).items()
                    }
                    emissions_by_source = {
                        k: float(v)
                        for k, v in pipe_result.get(
                            "emissions_by_source", {},
                        ).items()
                    }

            # Fallback: route to source-specific engine
            elif total_co2e == 0.0:
                fallback = self._calculate_fallback(request_data)
                total_co2e = fallback.get("total_co2e_tonnes", 0.0)
                co2_tonnes = fallback.get("co2_tonnes", 0.0)
                ch4_tonnes = fallback.get("ch4_tonnes", 0.0)
                n2o_tonnes = fallback.get("n2o_tonnes", 0.0)
                ch4_co2e = fallback.get("ch4_co2e_tonnes", 0.0)
                n2o_co2e = fallback.get("n2o_co2e_tonnes", 0.0)
                emissions_by_gas = fallback.get(
                    "emissions_by_gas", {},
                )
                emissions_by_source = fallback.get(
                    "emissions_by_source", {},
                )

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "farm_id": farm_id,
                "source_category": source_category,
                "livestock_type": livestock_type,
                "head_count": head_count,
                "total_co2e": total_co2e,
            })

            # Record provenance
            self._record_provenance(
                "CALCULATION", calc_id, "CALCULATE",
                data={
                    "total_co2e_tonnes": total_co2e,
                    "source_category": source_category,
                    "livestock_type": livestock_type,
                    "head_count": head_count,
                },
            )

            # Record metrics
            self._record_metrics(
                source_category=source_category,
                calculation_method=calc_method,
                livestock_type=livestock_type,
                duration_seconds=elapsed_ms / 1000.0,
                co2e_tonnes=total_co2e,
                gas="CO2e",
            )

            response = CalculateResponse(
                success=True,
                calculation_id=calc_id,
                farm_id=farm_id,
                source_category=source_category,
                livestock_type=livestock_type,
                calculation_method=calc_method,
                head_count=head_count,
                total_co2e_tonnes=round(total_co2e, 6),
                co2_tonnes=round(co2_tonnes, 6),
                ch4_tonnes=round(ch4_tonnes, 6),
                n2o_tonnes=round(n2o_tonnes, 6),
                ch4_co2e_tonnes=round(ch4_co2e, 6),
                n2o_co2e_tonnes=round(n2o_co2e, 6),
                emissions_by_gas=emissions_by_gas,
                emissions_by_source=emissions_by_source,
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

            logger.info(
                "Calculated %s: category=%s livestock=%s "
                "co2e=%.4f tonnes, head_count=%d",
                calc_id, source_category, livestock_type,
                total_co2e, head_count,
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
                farm_id=farm_id,
                source_category=source_category,
                livestock_type=livestock_type,
                calculation_method=calc_method,
                head_count=head_count,
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
        """Route calculation to source-specific engine.

        Falls back to a zero-emission stub when no engine is available.

        Args:
            request_data: Raw calculation request.

        Returns:
            Dictionary with per-gas emission values.
        """
        source_category = request_data.get("source_category", "")

        # Enteric fermentation categories
        enteric_sources = {"enteric_fermentation"}
        # Manure management categories
        manure_sources = {"manure_management"}
        # Cropland / soil categories
        cropland_sources = {
            "agricultural_soils", "liming", "urea_application",
            "rice_cultivation", "field_burning",
        }

        result: Dict[str, float] = {
            "total_co2e_tonnes": 0.0,
            "co2_tonnes": 0.0,
            "ch4_tonnes": 0.0,
            "n2o_tonnes": 0.0,
            "ch4_co2e_tonnes": 0.0,
            "n2o_co2e_tonnes": 0.0,
        }

        try:
            if (
                source_category in enteric_sources
                and self._enteric_engine is not None
            ):
                engine_result = self._enteric_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(
                        self._extract_gas_values(engine_result),
                    )

            elif (
                source_category in manure_sources
                and self._manure_engine is not None
            ):
                engine_result = self._manure_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(
                        self._extract_gas_values(engine_result),
                    )

            elif (
                source_category in cropland_sources
                and self._cropland_engine is not None
            ):
                engine_result = self._cropland_engine.calculate(
                    request_data,
                )
                if isinstance(engine_result, dict):
                    result.update(
                        self._extract_gas_values(engine_result),
                    )

            elif self._database_engine is not None:
                # Generic fallback via database emission factors
                engine_result = (
                    self._database_engine.calculate_emissions(
                        request_data,
                    )
                )
                if isinstance(engine_result, dict):
                    result.update(
                        self._extract_gas_values(engine_result),
                    )

        except Exception as exc:
            logger.warning(
                "Fallback calculation failed for %s: %s",
                source_category, exc,
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

        emissions_by_source: Dict[str, float] = {}
        raw_by_source = engine_result.get("emissions_by_source", {})
        for src, val in raw_by_source.items():
            emissions_by_source[src] = float(val)

        return {
            "total_co2e_tonnes": float(
                engine_result.get("total_co2e_tonnes", 0),
            ),
            "co2_tonnes": float(
                engine_result.get("co2_tonnes", 0),
            ),
            "ch4_tonnes": float(
                engine_result.get("ch4_tonnes", 0),
            ),
            "n2o_tonnes": float(
                engine_result.get("n2o_tonnes", 0),
            ),
            "ch4_co2e_tonnes": float(
                engine_result.get("ch4_co2e_tonnes", 0),
            ),
            "n2o_co2e_tonnes": float(
                engine_result.get("n2o_co2e_tonnes", 0),
            ),
            "emissions_by_gas": emissions_by_gas,
            "emissions_by_source": emissions_by_source,
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
            "farm_id": response.farm_id,
            "source_category": response.source_category,
            "livestock_type": response.livestock_type,
            "calculation_method": response.calculation_method,
            "head_count": response.head_count,
            "total_co2e_tonnes": response.total_co2e_tonnes,
            "co2_tonnes": response.co2_tonnes,
            "ch4_tonnes": response.ch4_tonnes,
            "n2o_tonnes": response.n2o_tonnes,
            "ch4_co2e_tonnes": response.ch4_co2e_tonnes,
            "n2o_co2e_tonnes": response.n2o_co2e_tonnes,
            "emissions_by_gas": response.emissions_by_gas,
            "emissions_by_source": response.emissions_by_source,
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
        """Batch calculate agricultural emissions.

        Processes multiple agricultural emission calculations in
        sequence, aggregating totals and collecting individual results.
        Supports up to 10,000 records per batch.

        Args:
            requests: List of calculation request dictionaries.
            tenant_id: Tenant identifier applied to all calculations
                that do not specify their own tenant_id.

        Returns:
            BatchCalculateResponse with aggregate totals.
        """
        t0 = time.monotonic()
        batch_id = _short_id("ag_batch")
        results: List[Dict[str, Any]] = []
        total_co2e = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        total_ch4_co2e = 0.0
        total_n2o_co2e = 0.0
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
                total_ch4_co2e += resp.ch4_co2e_tonnes
                total_n2o_co2e += resp.n2o_co2e_tonnes
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
            total_ch4_co2e_tonnes=round(total_ch4_co2e, 6),
            total_n2o_co2e_tonnes=round(total_n2o_co2e, 6),
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
        source_category: Optional[str] = None,
        livestock_type: Optional[str] = None,
        farm_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List calculation results with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            source_category: Optional filter by source category.
            livestock_type: Optional filter by livestock type.
            farm_id: Optional filter by farm.
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
        if source_category is not None:
            filtered = [
                c for c in filtered
                if c.get("source_category") == source_category
            ]
        if livestock_type is not None:
            filtered = [
                c for c in filtered
                if c.get("livestock_type") == livestock_type
            ]
        if farm_id is not None:
            filtered = [
                c for c in filtered
                if c.get("farm_id") == farm_id
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
    ) -> Dict[str, Any]:
        """Delete a calculation result by its identifier.

        Args:
            calculation_id: Unique calculation identifier.

        Returns:
            Dictionary with deletion status and metadata.
        """
        for i, calc in enumerate(self._calculations):
            if calc.get("calculation_id") == calculation_id:
                removed = self._calculations.pop(i)
                self._record_provenance(
                    "CALCULATION", calculation_id, "DELETE",
                    data={"deleted_at": _utcnow_iso()},
                )
                logger.info("Deleted calculation %s", calculation_id)
                return {
                    "deleted": True,
                    "calculation_id": calculation_id,
                    "timestamp": _utcnow_iso(),
                }
        return {
            "deleted": False,
            "calculation_id": calculation_id,
            "reason": "not_found",
            "timestamp": _utcnow_iso(),
        }

    # ==================================================================
    # Public API [6/20]: Register Farm
    # ==================================================================

    def register_farm(
        self,
        data: Dict[str, Any],
    ) -> FarmResponse:
        """Register a new farm / agricultural facility.

        Args:
            data: Farm data including name, farm_type, area_ha,
                latitude, longitude, country_code, climate_zone,
                soil_type, tenant_id.

        Returns:
            FarmResponse with the registered farm details.

        Raises:
            ValueError: If required fields are missing.
        """
        farm_id = _short_id("farm")

        # Validate required fields
        required_fields = ["name", "farm_type", "tenant_id"]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()

        provenance_hash = _compute_hash({
            "farm_id": farm_id,
            "name": data.get("name"),
            "farm_type": data.get("farm_type"),
            "tenant_id": data.get("tenant_id"),
        })

        record: Dict[str, Any] = {
            "farm_id": farm_id,
            "name": data.get("name", ""),
            "farm_type": data.get("farm_type", ""),
            "area_ha": float(data.get("area_ha", 0)),
            "latitude": float(data.get("latitude", 0)),
            "longitude": float(data.get("longitude", 0)),
            "country_code": data.get("country_code", ""),
            "climate_zone": data.get("climate_zone", ""),
            "soil_type": data.get("soil_type", ""),
            "tenant_id": data.get("tenant_id", ""),
            "operational_status": data.get(
                "operational_status", "active",
            ),
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._farms[farm_id] = record
        self._record_provenance(
            "FARM", farm_id, "CREATE", data=record,
        )

        logger.info(
            "Registered farm %s: name=%s type=%s area=%.1f ha",
            farm_id, record["name"], record["farm_type"],
            record["area_ha"],
        )

        return FarmResponse(
            farm_id=farm_id,
            name=record["name"],
            farm_type=record["farm_type"],
            area_ha=record["area_ha"],
            latitude=record["latitude"],
            longitude=record["longitude"],
            country_code=record["country_code"],
            climate_zone=record["climate_zone"],
            soil_type=record["soil_type"],
            tenant_id=record["tenant_id"],
            operational_status=record["operational_status"],
            provenance_hash=provenance_hash,
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ==================================================================
    # Public API [7/20]: List Farms
    # ==================================================================

    def list_farms(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        farm_type: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> FarmListResponse:
        """List registered farms with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            farm_type: Optional filter by farm type.
            country_code: Optional filter by country code.

        Returns:
            FarmListResponse with paginated farm list.
        """
        all_farms = list(self._farms.values())

        if tenant_id is not None:
            all_farms = [
                f for f in all_farms
                if f.get("tenant_id") == tenant_id
            ]
        if farm_type is not None:
            all_farms = [
                f for f in all_farms
                if f.get("farm_type") == farm_type
            ]
        if country_code is not None:
            all_farms = [
                f for f in all_farms
                if f.get("country_code") == country_code
            ]

        total = len(all_farms)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_farms[start:end]

        return FarmListResponse(
            farms=page_data,
            total=total,
            page=page,
            page_size=page_size,
        )

    # ==================================================================
    # Public API [8/20]: Update Farm
    # ==================================================================

    def update_farm(
        self,
        farm_id: str,
        data: Dict[str, Any],
    ) -> FarmResponse:
        """Update an existing farm / agricultural facility.

        Args:
            farm_id: Unique farm identifier.
            data: Dictionary of fields to update. Only non-None
                values will be applied.

        Returns:
            FarmResponse with updated farm details.

        Raises:
            ValueError: If the farm is not found.
        """
        existing = self._farms.get(farm_id)
        if existing is None:
            raise ValueError(f"Farm not found: {farm_id}")

        # Updatable fields whitelist
        updatable = {
            "name", "farm_type", "area_ha",
            "latitude", "longitude", "country_code",
            "climate_zone", "soil_type", "operational_status",
        }

        for key, value in data.items():
            if key in updatable and value is not None:
                if key == "area_ha":
                    value = float(value)
                elif key in ("latitude", "longitude"):
                    value = float(value)
                existing[key] = value

        existing["updated_at"] = _utcnow_iso()

        # Recompute provenance hash
        existing["provenance_hash"] = _compute_hash({
            "farm_id": farm_id,
            "name": existing.get("name"),
            "farm_type": existing.get("farm_type"),
            "updated_at": existing.get("updated_at"),
        })

        self._farms[farm_id] = existing
        self._record_provenance(
            "FARM", farm_id, "UPDATE", data=existing,
        )

        logger.info("Updated farm %s", farm_id)

        return FarmResponse(
            farm_id=farm_id,
            name=existing.get("name", ""),
            farm_type=existing.get("farm_type", ""),
            area_ha=existing.get("area_ha", 0.0),
            latitude=existing.get("latitude", 0.0),
            longitude=existing.get("longitude", 0.0),
            country_code=existing.get("country_code", ""),
            climate_zone=existing.get("climate_zone", ""),
            soil_type=existing.get("soil_type", ""),
            tenant_id=existing.get("tenant_id", ""),
            operational_status=existing.get(
                "operational_status", "active",
            ),
            provenance_hash=existing.get("provenance_hash", ""),
            created_at=existing.get("created_at", ""),
            updated_at=existing.get("updated_at", ""),
        )

    # ==================================================================
    # Public API [9/20]: Register Livestock
    # ==================================================================

    def register_livestock(
        self,
        data: Dict[str, Any],
    ) -> LivestockResponse:
        """Register a livestock population (herd/flock) record.

        Records animal population data for a farm, used as input to
        enteric fermentation and manure management calculations.

        Args:
            data: Livestock data including farm_id, livestock_type,
                head_count, average_weight_kg, milk_yield_kg_per_day,
                feed_digestibility_pct, manure_system, grazing_fraction,
                pregnancy_fraction, reporting_year, tenant_id.

        Returns:
            LivestockResponse with the registered livestock details.

        Raises:
            ValueError: If required fields are missing.
        """
        herd_id = _short_id("herd")

        required_fields = [
            "farm_id", "livestock_type", "head_count",
        ]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()
        farm_id = data.get("farm_id", "")
        livestock_type = data.get("livestock_type", "")
        head_count = int(data.get("head_count", 0))
        avg_weight = float(data.get("average_weight_kg", 0))
        milk_yield = float(data.get("milk_yield_kg_per_day", 0))
        feed_digest = float(data.get("feed_digestibility_pct", 0))
        gross_energy = float(data.get("gross_energy_mj_per_day", 0))
        manure_system = data.get("manure_system", "")
        grazing_frac = float(data.get("grazing_fraction", 0))
        pregnancy_frac = float(data.get("pregnancy_fraction", 0))
        reporting_year = int(data.get("reporting_year", 0))
        tenant_id = data.get("tenant_id", "")

        provenance_hash = _compute_hash({
            "herd_id": herd_id,
            "farm_id": farm_id,
            "livestock_type": livestock_type,
            "head_count": head_count,
        })

        record: Dict[str, Any] = {
            "herd_id": herd_id,
            "farm_id": farm_id,
            "livestock_type": livestock_type,
            "head_count": head_count,
            "average_weight_kg": avg_weight,
            "milk_yield_kg_per_day": milk_yield,
            "feed_digestibility_pct": feed_digest,
            "gross_energy_mj_per_day": gross_energy,
            "manure_system": manure_system,
            "grazing_fraction": grazing_frac,
            "pregnancy_fraction": pregnancy_frac,
            "reporting_year": reporting_year,
            "tenant_id": tenant_id,
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._livestock[herd_id] = record
        self._record_provenance(
            "LIVESTOCK", herd_id, "CREATE", data=record,
        )

        # Record metrics for livestock registration
        if self._metrics_collector is not None:
            try:
                self._metrics_collector.record_livestock_registered(
                    livestock_type, head_count,
                )
            except Exception:
                pass

        logger.info(
            "Registered livestock %s: farm=%s type=%s "
            "head_count=%d weight=%.1f kg",
            herd_id, farm_id, livestock_type,
            head_count, avg_weight,
        )

        return LivestockResponse(
            herd_id=herd_id,
            farm_id=farm_id,
            livestock_type=livestock_type,
            head_count=head_count,
            average_weight_kg=avg_weight,
            milk_yield_kg_per_day=milk_yield,
            feed_digestibility_pct=feed_digest,
            gross_energy_mj_per_day=gross_energy,
            manure_system=manure_system,
            grazing_fraction=grazing_frac,
            pregnancy_fraction=pregnancy_frac,
            reporting_year=reporting_year,
            tenant_id=tenant_id,
            provenance_hash=provenance_hash,
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ==================================================================
    # Public API [10/20]: List Livestock
    # ==================================================================

    def list_livestock(
        self,
        page: int = 1,
        page_size: int = 20,
        farm_id: Optional[str] = None,
        livestock_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> LivestockListResponse:
        """List registered livestock populations with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            farm_id: Optional filter by farm.
            livestock_type: Optional filter by livestock type.
            tenant_id: Optional filter by tenant.

        Returns:
            LivestockListResponse with paginated livestock list.
        """
        all_livestock = list(self._livestock.values())

        if farm_id is not None:
            all_livestock = [
                ls for ls in all_livestock
                if ls.get("farm_id") == farm_id
            ]
        if livestock_type is not None:
            all_livestock = [
                ls for ls in all_livestock
                if ls.get("livestock_type") == livestock_type
            ]
        if tenant_id is not None:
            all_livestock = [
                ls for ls in all_livestock
                if ls.get("tenant_id") == tenant_id
            ]

        total = len(all_livestock)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_livestock[start:end]

        return LivestockListResponse(
            livestock=page_data,
            total=total,
            page=page,
            page_size=page_size,
        )

    # ==================================================================
    # Public API [11/20]: Update Livestock
    # ==================================================================

    def update_livestock(
        self,
        herd_id: str,
        data: Dict[str, Any],
    ) -> LivestockResponse:
        """Update an existing livestock population record.

        Args:
            herd_id: Unique herd/flock identifier.
            data: Dictionary of fields to update. Only non-None
                values will be applied.

        Returns:
            LivestockResponse with updated livestock details.

        Raises:
            ValueError: If the livestock record is not found.
        """
        existing = self._livestock.get(herd_id)
        if existing is None:
            raise ValueError(f"Livestock not found: {herd_id}")

        # Updatable fields whitelist
        updatable = {
            "head_count", "average_weight_kg",
            "milk_yield_kg_per_day", "feed_digestibility_pct",
            "gross_energy_mj_per_day", "manure_system",
            "grazing_fraction", "pregnancy_fraction",
            "reporting_year",
        }

        for key, value in data.items():
            if key in updatable and value is not None:
                if key == "head_count":
                    value = int(value)
                elif key == "reporting_year":
                    value = int(value)
                elif key in (
                    "average_weight_kg", "milk_yield_kg_per_day",
                    "feed_digestibility_pct",
                    "gross_energy_mj_per_day",
                    "grazing_fraction", "pregnancy_fraction",
                ):
                    value = float(value)
                existing[key] = value

        existing["updated_at"] = _utcnow_iso()

        # Recompute provenance hash
        existing["provenance_hash"] = _compute_hash({
            "herd_id": herd_id,
            "livestock_type": existing.get("livestock_type"),
            "head_count": existing.get("head_count"),
            "updated_at": existing.get("updated_at"),
        })

        self._livestock[herd_id] = existing
        self._record_provenance(
            "LIVESTOCK", herd_id, "UPDATE", data=existing,
        )

        logger.info("Updated livestock %s", herd_id)

        return LivestockResponse(
            herd_id=herd_id,
            farm_id=existing.get("farm_id", ""),
            livestock_type=existing.get("livestock_type", ""),
            head_count=existing.get("head_count", 0),
            average_weight_kg=existing.get("average_weight_kg", 0.0),
            milk_yield_kg_per_day=existing.get(
                "milk_yield_kg_per_day", 0.0,
            ),
            feed_digestibility_pct=existing.get(
                "feed_digestibility_pct", 0.0,
            ),
            gross_energy_mj_per_day=existing.get(
                "gross_energy_mj_per_day", 0.0,
            ),
            manure_system=existing.get("manure_system", ""),
            grazing_fraction=existing.get("grazing_fraction", 0.0),
            pregnancy_fraction=existing.get("pregnancy_fraction", 0.0),
            reporting_year=existing.get("reporting_year", 0),
            tenant_id=existing.get("tenant_id", ""),
            provenance_hash=existing.get("provenance_hash", ""),
            created_at=existing.get("created_at", ""),
            updated_at=existing.get("updated_at", ""),
        )

    # ==================================================================
    # Public API [12/20]: Register Cropland Input
    # ==================================================================

    def register_cropland_input(
        self,
        data: Dict[str, Any],
    ) -> CroplandInputResponse:
        """Register a cropland input record (fertiliser, lime, urea).

        Tracks nitrogen and carbon inputs applied to agricultural
        soils, used as input to N2O and CO2 emission calculations.

        Args:
            data: Cropland input data including farm_id, input_type,
                product_name, quantity_tonnes, nitrogen_content_fraction,
                carbon_content_fraction, area_applied_ha,
                application_date, reporting_year, tenant_id.

        Returns:
            CroplandInputResponse with the registered input details.

        Raises:
            ValueError: If required fields are missing.
        """
        input_id = _short_id("ci")

        required_fields = [
            "farm_id", "input_type", "quantity_tonnes",
        ]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()
        farm_id = data.get("farm_id", "")
        input_type = data.get("input_type", "")
        product_name = data.get("product_name", "")
        quantity_tonnes = float(data.get("quantity_tonnes", 0))
        n_content = float(data.get("nitrogen_content_fraction", 0))
        c_content = float(data.get("carbon_content_fraction", 0))
        area_applied = float(data.get("area_applied_ha", 0))
        application_date = data.get(
            "application_date", _utcnow_iso(),
        )
        reporting_year = int(data.get("reporting_year", 0))
        tenant_id = data.get("tenant_id", "")

        provenance_hash = _compute_hash({
            "input_id": input_id,
            "farm_id": farm_id,
            "input_type": input_type,
            "quantity_tonnes": quantity_tonnes,
            "nitrogen_content_fraction": n_content,
        })

        record: Dict[str, Any] = {
            "input_id": input_id,
            "farm_id": farm_id,
            "input_type": input_type,
            "product_name": product_name,
            "quantity_tonnes": quantity_tonnes,
            "nitrogen_content_fraction": n_content,
            "carbon_content_fraction": c_content,
            "area_applied_ha": area_applied,
            "application_date": application_date,
            "reporting_year": reporting_year,
            "tenant_id": tenant_id,
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._cropland_inputs[input_id] = record
        self._record_provenance(
            "CROPLAND_INPUT", input_id, "CREATE", data=record,
        )

        # Record cropland metrics
        if self._metrics_collector is not None:
            try:
                self._metrics_collector.record_cropland_input(
                    input_type, quantity_tonnes,
                )
            except Exception:
                pass

        logger.info(
            "Registered cropland input %s: farm=%s type=%s "
            "qty=%.2f tonnes N_frac=%.3f",
            input_id, farm_id, input_type,
            quantity_tonnes, n_content,
        )

        return CroplandInputResponse(
            input_id=input_id,
            farm_id=farm_id,
            input_type=input_type,
            product_name=product_name,
            quantity_tonnes=quantity_tonnes,
            nitrogen_content_fraction=n_content,
            carbon_content_fraction=c_content,
            area_applied_ha=area_applied,
            application_date=application_date,
            reporting_year=reporting_year,
            tenant_id=tenant_id,
            provenance_hash=provenance_hash,
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ==================================================================
    # Public API [13/20]: List Cropland Inputs
    # ==================================================================

    def list_cropland_inputs(
        self,
        page: int = 1,
        page_size: int = 20,
        farm_id: Optional[str] = None,
        input_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List registered cropland input records with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            farm_id: Optional filter by farm.
            input_type: Optional filter by input type.
            tenant_id: Optional filter by tenant.

        Returns:
            Dictionary with paginated cropland input list.
        """
        all_inputs = list(self._cropland_inputs.values())

        if farm_id is not None:
            all_inputs = [
                ci for ci in all_inputs
                if ci.get("farm_id") == farm_id
            ]
        if input_type is not None:
            all_inputs = [
                ci for ci in all_inputs
                if ci.get("input_type") == input_type
            ]
        if tenant_id is not None:
            all_inputs = [
                ci for ci in all_inputs
                if ci.get("tenant_id") == tenant_id
            ]

        # Sort by application_date descending
        all_inputs.sort(
            key=lambda x: x.get("application_date", ""),
            reverse=True,
        )

        total = len(all_inputs)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_inputs[start:end]

        return {
            "cropland_inputs": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API [14/20]: Register Rice Field
    # ==================================================================

    def register_rice_field(
        self,
        data: Dict[str, Any],
    ) -> RiceFieldResponse:
        """Register a rice field for CH4 emission calculations.

        Tracks rice paddy characteristics required for IPCC Tier 1/2
        methane emission estimates, including water regime, organic
        amendments, cultivation period, and scaling factors.

        Args:
            data: Rice field data including farm_id, area_ha,
                water_regime, organic_amendment,
                organic_amendment_rate_tonnes_per_ha,
                cultivation_period_days, soil_type, previous_crop,
                reporting_year, baseline_ef_kg_ch4_per_ha,
                scaling_factor_water, scaling_factor_amendment,
                scaling_factor_preseason, tenant_id.

        Returns:
            RiceFieldResponse with the registered rice field details.

        Raises:
            ValueError: If required fields are missing.
        """
        field_id = _short_id("rice")

        required_fields = ["farm_id", "area_ha"]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()
        farm_id = data.get("farm_id", "")
        area_ha = float(data.get("area_ha", 0))
        water_regime = data.get(
            "water_regime", "continuously_flooded",
        )
        organic_amendment = data.get("organic_amendment", "none")
        amendment_rate = float(data.get(
            "organic_amendment_rate_tonnes_per_ha", 0,
        ))
        cultivation_days = int(data.get(
            "cultivation_period_days", 120,
        ))
        soil_type = data.get("soil_type", "")
        previous_crop = data.get("previous_crop", "")
        reporting_year = int(data.get("reporting_year", 0))
        baseline_ef = float(data.get(
            "baseline_ef_kg_ch4_per_ha", 0,
        ))
        sf_water = float(data.get("scaling_factor_water", 1.0))
        sf_amendment = float(data.get(
            "scaling_factor_amendment", 1.0,
        ))
        sf_preseason = float(data.get(
            "scaling_factor_preseason", 1.0,
        ))
        tenant_id = data.get("tenant_id", "")

        provenance_hash = _compute_hash({
            "field_id": field_id,
            "farm_id": farm_id,
            "area_ha": area_ha,
            "water_regime": water_regime,
        })

        record: Dict[str, Any] = {
            "field_id": field_id,
            "farm_id": farm_id,
            "area_ha": area_ha,
            "water_regime": water_regime,
            "organic_amendment": organic_amendment,
            "organic_amendment_rate_tonnes_per_ha": amendment_rate,
            "cultivation_period_days": cultivation_days,
            "soil_type": soil_type,
            "previous_crop": previous_crop,
            "reporting_year": reporting_year,
            "baseline_ef_kg_ch4_per_ha": baseline_ef,
            "scaling_factor_water": sf_water,
            "scaling_factor_amendment": sf_amendment,
            "scaling_factor_preseason": sf_preseason,
            "tenant_id": tenant_id,
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._rice_fields[field_id] = record
        self._record_provenance(
            "RICE_FIELD", field_id, "CREATE", data=record,
        )

        # Record rice field metrics
        if self._metrics_collector is not None:
            try:
                self._metrics_collector.record_rice_field_registered(
                    water_regime, area_ha,
                )
            except Exception:
                pass

        logger.info(
            "Registered rice field %s: farm=%s area=%.2f ha "
            "water_regime=%s",
            field_id, farm_id, area_ha, water_regime,
        )

        return RiceFieldResponse(
            field_id=field_id,
            farm_id=farm_id,
            area_ha=area_ha,
            water_regime=water_regime,
            organic_amendment=organic_amendment,
            organic_amendment_rate_tonnes_per_ha=amendment_rate,
            cultivation_period_days=cultivation_days,
            soil_type=soil_type,
            previous_crop=previous_crop,
            reporting_year=reporting_year,
            baseline_ef_kg_ch4_per_ha=baseline_ef,
            scaling_factor_water=sf_water,
            scaling_factor_amendment=sf_amendment,
            scaling_factor_preseason=sf_preseason,
            tenant_id=tenant_id,
            provenance_hash=provenance_hash,
            created_at=now_iso,
            updated_at=now_iso,
        )

    # ==================================================================
    # Public API [15/20]: List Rice Fields
    # ==================================================================

    def list_rice_fields(
        self,
        page: int = 1,
        page_size: int = 20,
        farm_id: Optional[str] = None,
        water_regime: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List registered rice fields with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            farm_id: Optional filter by farm.
            water_regime: Optional filter by water regime.
            tenant_id: Optional filter by tenant.

        Returns:
            Dictionary with paginated rice field list.
        """
        all_fields = list(self._rice_fields.values())

        if farm_id is not None:
            all_fields = [
                rf for rf in all_fields
                if rf.get("farm_id") == farm_id
            ]
        if water_regime is not None:
            all_fields = [
                rf for rf in all_fields
                if rf.get("water_regime") == water_regime
            ]
        if tenant_id is not None:
            all_fields = [
                rf for rf in all_fields
                if rf.get("tenant_id") == tenant_id
            ]

        total = len(all_fields)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_fields[start:end]

        # Compute summary
        total_area = sum(
            float(rf.get("area_ha", 0)) for rf in all_fields
        )

        return {
            "rice_fields": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
            "summary": {
                "total_area_ha": round(total_area, 4),
                "total_fields": total,
            },
        }

    # ==================================================================
    # Public API [16/20]: Check Compliance
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Run multi-framework compliance check.

        Checks agricultural emission calculations against applicable
        regulatory frameworks (GHG Protocol, IPCC 2006, IPCC 2019,
        CSRD ESRS E1, US EPA GHGRP, EU Effort Sharing, UNFCCC,
        SBTi FLAG).

        Args:
            data: Compliance check parameters including optional
                calculation_id and frameworks list.

        Returns:
            ComplianceCheckResponse with compliance check results.
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
                compliance_data = (
                    dict(calc_record) if calc_record else {}
                )
                compliance_data.update(data)
                result = self._compliance_engine.check_compliance(
                    calculation_data=compliance_data,
                    frameworks=(
                        frameworks if frameworks else None
                    ),
                )
                return self._format_compliance_result(
                    compliance_id, result,
                )
            except Exception as exc:
                logger.warning(
                    "Compliance engine failed: %s", exc,
                )

        # Fallback
        return self._compliance_fallback(
            compliance_id, frameworks,
        )

    def _format_compliance_result(
        self,
        compliance_id: str,
        result: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Format engine compliance result into response model.

        Args:
            compliance_id: Unique compliance check identifier.
            result: Raw engine result.

        Returns:
            ComplianceCheckResponse with formatted results.
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
                result.get(
                    "frameworks_checked", len(results_list),
                ),
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

        return ComplianceCheckResponse(
            success=True,
            id=compliance_id,
            frameworks_checked=formatted["frameworks_checked"],
            compliant=compliant_count,
            non_compliant=non_compliant_count,
            partial=partial_count,
            results=results_list,
            timestamp=formatted["timestamp"],
        )

    def _compliance_fallback(
        self,
        compliance_id: str,
        frameworks: List[str],
    ) -> ComplianceCheckResponse:
        """Generate a fallback compliance result.

        Returns not-assessed status for all requested frameworks
        when the compliance engine is unavailable.

        Args:
            compliance_id: Unique compliance check identifier.
            frameworks: Requested frameworks to check.

        Returns:
            ComplianceCheckResponse with fallback results.
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

        formatted: Dict[str, Any] = {
            "id": compliance_id,
            "success": True,
            "frameworks_checked": len(check_frameworks),
            "compliant": 0,
            "non_compliant": 0,
            "partial": 0,
            "results": results_list,
            "timestamp": _utcnow_iso(),
        }
        self._compliance_results.append(formatted)

        self._record_provenance(
            "COMPLIANCE_CHECK", compliance_id, "CHECK",
            data=formatted,
        )

        return ComplianceCheckResponse(
            success=True,
            id=compliance_id,
            frameworks_checked=len(check_frameworks),
            compliant=0,
            non_compliant=0,
            partial=0,
            results=results_list,
            timestamp=formatted["timestamp"],
        )

    # ==================================================================
    # Public API [17/20]: Get Compliance Result
    # ==================================================================

    def get_compliance_result(
        self,
        compliance_id: str,
    ) -> Optional[ComplianceCheckResponse]:
        """Get a compliance check result by its identifier.

        Args:
            compliance_id: Unique compliance check identifier.

        Returns:
            ComplianceCheckResponse or None if not found.
        """
        for result in self._compliance_results:
            if result.get("id") == compliance_id:
                return ComplianceCheckResponse(
                    success=result.get("success", True),
                    id=result.get("id", ""),
                    frameworks_checked=result.get(
                        "frameworks_checked", 0,
                    ),
                    compliant=result.get("compliant", 0),
                    non_compliant=result.get("non_compliant", 0),
                    partial=result.get("partial", 0),
                    results=result.get("results", []),
                    timestamp=result.get("timestamp", ""),
                )
        return None

    # ==================================================================
    # Public API [18/20]: Run Uncertainty
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> UncertaintyResponse:
        """Run uncertainty analysis on a calculation.

        Supports Monte Carlo simulation and analytical (error
        propagation) methods for quantifying confidence intervals
        around agricultural emission estimates.

        Args:
            data: Uncertainty parameters including calculation_id,
                method, iterations, seed, and confidence_level.

        Returns:
            UncertaintyResponse with uncertainty analysis results.
        """
        calc_id = data.get("calculation_id", "")
        method = data.get("method", "monte_carlo")
        iterations = int(data.get("iterations", 5000))
        seed = int(data.get("seed", 42))
        confidence_level = float(
            data.get("confidence_level", 95.0),
        )

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
        if (
            self._uncertainty_engine is not None
            and calc_record is not None
        ):
            try:
                result = (
                    self._uncertainty_engine.quantify_uncertainty(
                        calculation_input=calc_record,
                        method=method,
                        n_iterations=iterations,
                        seed=seed,
                    )
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
    ) -> UncertaintyResponse:
        """Format engine uncertainty result into response model.

        Args:
            calc_id: Calculation identifier.
            result: Raw engine result.
            iterations: Monte Carlo iterations.
            confidence_level: Confidence level percentage.

        Returns:
            UncertaintyResponse with formatted results.
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

        return UncertaintyResponse(
            success=True,
            calculation_id=calc_id,
            method="monte_carlo",
            iterations=iterations,
            mean_co2e_tonnes=round(mean_co2e, 6),
            std_dev_tonnes=round(std_dev, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_level=confidence_level,
            coefficient_of_variation=round(cv, 4),
            provenance_hash=provenance_hash,
            timestamp=formatted["timestamp"],
        )

    def _uncertainty_fallback(
        self,
        calc_id: str,
        total_co2e: float,
        iterations: int,
        confidence_level: float,
    ) -> UncertaintyResponse:
        """Compute analytical uncertainty fallback.

        Uses a conservative 30% uncertainty estimate when the Monte
        Carlo engine is unavailable, consistent with IPCC Tier 1
        default uncertainty ranges for agricultural emissions
        (enteric CH4: +/-20-50%, manure CH4: +/-30-50%).

        Args:
            calc_id: Calculation identifier.
            total_co2e: Total CO2e emission estimate.
            iterations: Requested iterations (not used in fallback).
            confidence_level: Confidence level percentage.

        Returns:
            UncertaintyResponse with analytical fallback results.
        """
        std_estimate = abs(total_co2e) * 0.30
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

        formatted: Dict[str, Any] = {
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
        self._uncertainty_results.append(formatted)

        self._record_provenance(
            "UNCERTAINTY_RUN", calc_id, "ASSESS",
            data=formatted,
        )

        return UncertaintyResponse(
            success=True,
            calculation_id=calc_id,
            method="analytical_fallback",
            iterations=0,
            mean_co2e_tonnes=round(total_co2e, 6),
            std_dev_tonnes=round(std_estimate, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_level=confidence_level,
            coefficient_of_variation=round(cv, 4),
            provenance_hash=provenance_hash,
            timestamp=formatted["timestamp"],
        )

    # ==================================================================
    # Public API [19/20]: Aggregate
    # ==================================================================

    def aggregate(
        self,
        data: Dict[str, Any],
    ) -> AggregationResponse:
        """Aggregate agricultural emission results.

        Groups calculation results by tenant, farm, source category,
        livestock type, or time period and computes totals.

        Args:
            data: Aggregation parameters including tenant_id, period,
                group_by, date_from, date_to, and optional filters
                for source_category and livestock_type.

        Returns:
            AggregationResponse with aggregated emission totals.
        """
        tenant_id = data.get("tenant_id", "")
        period = data.get("period", "annual")
        group_by = data.get(
            "group_by", ["source_category"],
        )
        date_from = data.get("date_from")
        date_to = data.get("date_to")
        filter_source = data.get("source_category")
        filter_livestock = data.get("livestock_type")
        filter_farm = data.get("farm_id")

        # Filter calculations by tenant and success status
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

        # Apply source category filter
        if filter_source is not None:
            filtered = [
                c for c in filtered
                if c.get("source_category") == filter_source
            ]

        # Apply livestock type filter
        if filter_livestock is not None:
            filtered = [
                c for c in filtered
                if c.get("livestock_type") == filter_livestock
            ]

        # Apply farm filter
        if filter_farm is not None:
            filtered = [
                c for c in filtered
                if c.get("farm_id") == filter_farm
            ]

        # Group and aggregate
        groups: Dict[str, Dict[str, float]] = {}
        total_co2e = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        total_ch4_co2e = 0.0
        total_n2o_co2e = 0.0
        total_head = 0

        for calc in filtered:
            # Build group key
            key_parts = []
            for field_name in group_by:
                key_parts.append(
                    str(calc.get(field_name, "unknown")),
                )
            group_key = (
                "|".join(key_parts) if key_parts else "all"
            )

            if group_key not in groups:
                groups[group_key] = {
                    "total_co2e_tonnes": 0.0,
                    "co2_tonnes": 0.0,
                    "ch4_tonnes": 0.0,
                    "n2o_tonnes": 0.0,
                    "ch4_co2e_tonnes": 0.0,
                    "n2o_co2e_tonnes": 0.0,
                    "head_count": 0.0,
                    "count": 0.0,
                }

            co2e = float(calc.get("total_co2e_tonnes", 0))
            ch4 = float(calc.get("ch4_tonnes", 0))
            n2o = float(calc.get("n2o_tonnes", 0))
            ch4_co2e = float(calc.get("ch4_co2e_tonnes", 0))
            n2o_co2e = float(calc.get("n2o_co2e_tonnes", 0))
            head = int(calc.get("head_count", 0))

            groups[group_key]["total_co2e_tonnes"] += co2e
            groups[group_key]["co2_tonnes"] += float(
                calc.get("co2_tonnes", 0),
            )
            groups[group_key]["ch4_tonnes"] += ch4
            groups[group_key]["n2o_tonnes"] += n2o
            groups[group_key]["ch4_co2e_tonnes"] += ch4_co2e
            groups[group_key]["n2o_co2e_tonnes"] += n2o_co2e
            groups[group_key]["head_count"] += head
            groups[group_key]["count"] += 1

            total_co2e += co2e
            total_ch4 += ch4
            total_n2o += n2o
            total_ch4_co2e += ch4_co2e
            total_n2o_co2e += n2o_co2e
            total_head += head

        self._record_provenance(
            "AGGREGATION", _short_id("agg"), "AGGREGATE",
            data={
                "tenant_id": tenant_id,
                "total_co2e_tonnes": total_co2e,
                "calculation_count": len(filtered),
            },
        )

        return AggregationResponse(
            groups=groups,
            total_co2e_tonnes=round(total_co2e, 6),
            total_ch4_tonnes=round(total_ch4, 6),
            total_n2o_tonnes=round(total_n2o, 6),
            total_ch4_co2e_tonnes=round(total_ch4_co2e, 6),
            total_n2o_co2e_tonnes=round(total_n2o_co2e, 6),
            total_head_count=total_head,
            calculation_count=len(filtered),
            period=period,
        )

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
            "agricultural_database": (
                "available"
                if self._database_engine is not None
                else "unavailable"
            ),
            "enteric_fermentation": (
                "available"
                if self._enteric_engine is not None
                else "unavailable"
            ),
            "manure_management": (
                "available"
                if self._manure_engine is not None
                else "unavailable"
            ),
            "cropland_emissions": (
                "available"
                if self._cropland_engine is not None
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
            service="agricultural-emissions",
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

        # Compute total head count from livestock records
        total_head = sum(
            int(ls.get("head_count", 0))
            for ls in self._livestock.values()
        )

        return StatsResponse(
            total_calculations=self._total_calculations,
            total_farms=len(self._farms),
            total_livestock_herds=len(self._livestock),
            total_cropland_inputs=len(self._cropland_inputs),
            total_rice_fields=len(self._rice_fields),
            total_compliance_checks=len(self._compliance_results),
            total_uncertainty_runs=len(self._uncertainty_results),
            total_batch_runs=self._total_batch_runs,
            total_co2e_tonnes=round(total_emissions, 6),
            total_head_count=total_head,
            uptime_seconds=round(uptime, 3),
        )

# ===================================================================
# Thread-safe singleton access
# ===================================================================

_service_instance: Optional[AgriculturalEmissionsService] = None
_service_lock = threading.RLock()

def get_service() -> AgriculturalEmissionsService:
    """Get or create the singleton AgriculturalEmissionsService instance.

    Uses double-checked locking with a reentrant lock for thread
    safety with minimal contention on the hot path.

    Returns:
        AgriculturalEmissionsService singleton instance.

    Example:
        >>> svc_a = get_service()
        >>> svc_b = get_service()
        >>> assert svc_a is svc_b
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = AgriculturalEmissionsService()
    return _service_instance

def set_service(
    service: AgriculturalEmissionsService,
) -> None:
    """Set the singleton to a pre-configured service instance.

    Useful for dependency injection in tests or when the service
    needs to be initialized with a specific configuration before
    any calls to ``get_service()``.

    Args:
        service: Pre-configured AgriculturalEmissionsService instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.debug(
        "AgriculturalEmissionsService singleton set explicitly"
    )

def reset_service() -> None:
    """Reset the singleton service instance for test teardown.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for use in test fixtures to prevent state leakage.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("AgriculturalEmissionsService singleton reset")

# ===================================================================
# FastAPI Router factory
# ===================================================================

def get_router() -> Any:
    """Get the FastAPI router for agricultural emissions.

    Attempts to import the dedicated router module. Falls back to
    None if the router module or FastAPI is not available.

    Returns:
        FastAPI APIRouter or None if unavailable.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.agents.mrv.agricultural_emissions.api.router import (
            create_router,
        )
        return create_router()
    except ImportError:
        logger.warning(
            "Agricultural emissions API router module not available"
        )
        return None

# ===================================================================
# FastAPI application integration
# ===================================================================

def configure_agricultural_emissions(
    app: Any,
    config: Any = None,
) -> AgriculturalEmissionsService:
    """Configure the Agricultural Emissions Service on a FastAPI app.

    Creates the AgriculturalEmissionsService singleton, stores it
    in ``app.state``, and mounts the API router.

    Args:
        app: FastAPI application instance.
        config: Optional AgriculturalConfig override.

    Returns:
        AgriculturalEmissionsService instance.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> svc = configure_agricultural_emissions(app)
        >>> assert hasattr(
        ...     app.state, "agricultural_emissions_service",
        ... )
    """
    global _service_instance

    service = AgriculturalEmissionsService(config=config)

    with _service_lock:
        _service_instance = service

    if hasattr(app, "state"):
        app.state.agricultural_emissions_service = service

    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Agricultural emissions API router mounted")
    else:
        logger.warning(
            "Agricultural emissions router not available; "
            "API not mounted"
        )

    logger.info("Agricultural Emissions service configured")
    return service

# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "AgriculturalEmissionsService",
    # Integration helpers
    "configure_agricultural_emissions",
    "get_service",
    "set_service",
    "reset_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "FarmResponse",
    "FarmListResponse",
    "LivestockResponse",
    "LivestockListResponse",
    "CroplandInputResponse",
    "RiceFieldResponse",
    "FieldBurningResponse",
    "ComplianceCheckResponse",
    "UncertaintyResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
    # Constants
    "SOURCE_CATEGORIES",
    "LIVESTOCK_TYPES",
    "MANURE_SYSTEMS",
    "CROP_INPUT_TYPES",
    "RICE_WATER_REGIMES",
    "COMPLIANCE_FRAMEWORKS",
]
