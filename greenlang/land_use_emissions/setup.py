# -*- coding: utf-8 -*-
"""
Land Use Emissions Service Setup - AGENT-MRV-006
=================================================

Service facade for the Land Use Emissions Agent (GL-MRV-SCOPE1-006).

Provides ``configure_land_use(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``LandUseEmissionsService`` facade class that aggregates all 7 engines:

    1. LandUseDatabaseEngine      - Land categories, EFs, climate zones
    2. CarbonStockCalculatorEngine - Stock-difference & gain-loss methods
    3. LandUseChangeTrackerEngine  - Transition tracking, 6x6 matrix
    4. SoilOrganicCarbonEngine     - SOC assessment (F_LU * F_MG * F_I)
    5. UncertaintyQuantifierEngine - Monte Carlo & analytical uncertainty
    6. ComplianceCheckerEngine     - Multi-framework regulatory compliance
    7. LandUsePipelineEngine       - Orchestrated calculation pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.land_use_emissions.setup import configure_land_use
    >>> app = FastAPI()
    >>> configure_land_use(app)

    >>> from greenlang.land_use_emissions.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "parcel_id": "parcel-001",
    ...     "from_category": "forest_land",
    ...     "to_category": "cropland",
    ...     "area_ha": 100.0,
    ...     "climate_zone": "tropical_moist",
    ...     "soil_type": "high_activity_clay",
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Land Use Emissions (GL-MRV-SCOPE1-006)
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
    from greenlang.land_use_emissions.config import (
        LandUseConfig,
        get_config,
    )
except ImportError:
    LandUseConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.land_use_emissions.land_use_database import (
        LandUseDatabaseEngine,
    )
except ImportError:
    LandUseDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.carbon_stock_calculator import (
        CarbonStockCalculatorEngine,
    )
except ImportError:
    CarbonStockCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.land_use_change_tracker import (
        LandUseChangeTrackerEngine,
    )
except ImportError:
    LandUseChangeTrackerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.soil_organic_carbon import (
        SoilOrganicCarbonEngine,
    )
except ImportError:
    SoilOrganicCarbonEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.land_use_pipeline import (
        LandUsePipelineEngine,
    )
except ImportError:
    LandUsePipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.land_use_emissions.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]


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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# IPCC land categories (used for transition matrix defaults)
# ===================================================================

IPCC_LAND_CATEGORIES: List[str] = [
    "forest_land",
    "cropland",
    "grassland",
    "wetland",
    "settlement",
    "other_land",
]


# ===================================================================
# Lightweight Pydantic response models (14 models)
# ===================================================================


class CalculateResponse(BaseModel):
    """Single land use emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    from_category: str = Field(default="")
    to_category: str = Field(default="")
    method: str = Field(default="stock_difference")
    tier: str = Field(default="tier_1")
    total_co2e_tonnes: float = Field(default=0.0)
    removals_co2e_tonnes: float = Field(default=0.0)
    net_co2e_tonnes: float = Field(default=0.0)
    emissions_by_pool: Dict[str, float] = Field(default_factory=dict)
    emissions_by_gas: Dict[str, float] = Field(default_factory=dict)
    area_ha: float = Field(default=0.0)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
    """Batch land use emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    batch_id: str = Field(default="")
    total_calculations: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_removals_tonnes: float = Field(default=0.0)
    net_co2e_tonnes: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class ParcelResponse(BaseModel):
    """Response for a single land parcel."""

    model_config = ConfigDict(frozen=True)

    parcel_id: str = Field(default="")
    name: str = Field(default="")
    area_ha: float = Field(default=0.0)
    land_category: str = Field(default="")
    climate_zone: str = Field(default="")
    soil_type: str = Field(default="")
    latitude: float = Field(default=0.0)
    longitude: float = Field(default=0.0)
    tenant_id: str = Field(default="")
    country_code: str = Field(default="")
    management_practice: str = Field(default="nominally_managed")
    input_level: str = Field(default="medium")
    peatland_status: Optional[str] = Field(default=None)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class ParcelListResponse(BaseModel):
    """Response listing land parcels."""

    model_config = ConfigDict(frozen=True)

    parcels: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class CarbonStockResponse(BaseModel):
    """Response for a carbon stock snapshot."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(default="")
    parcel_id: str = Field(default="")
    pool: str = Field(default="")
    stock_tc_ha: float = Field(default=0.0)
    measurement_date: str = Field(default="")
    tier: str = Field(default="tier_1")
    source: str = Field(default="IPCC_2006")
    provenance_hash: str = Field(default="")


class TransitionResponse(BaseModel):
    """Response for a land-use transition record."""

    model_config = ConfigDict(frozen=True)

    transition_id: str = Field(default="")
    parcel_id: str = Field(default="")
    from_category: str = Field(default="")
    to_category: str = Field(default="")
    transition_date: str = Field(default="")
    area_ha: float = Field(default=0.0)
    transition_type: str = Field(default="")
    disturbance_type: str = Field(default="none")


class TransitionMatrixResponse(BaseModel):
    """Response for the 6x6 transition matrix."""

    model_config = ConfigDict(frozen=True)

    categories: List[str] = Field(default_factory=list)
    matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_area_ha: float = Field(default=0.0)
    total_transitions: int = Field(default=0)


class SOCAssessmentResponse(BaseModel):
    """Response for a soil organic carbon assessment."""

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(default="")
    parcel_id: str = Field(default="")
    soc_ref: float = Field(default=0.0)
    f_lu: float = Field(default=1.0)
    f_mg: float = Field(default=1.0)
    f_i: float = Field(default=1.0)
    soc_current: float = Field(default=0.0)
    soc_previous: Optional[float] = Field(default=None)
    delta_soc_annual: float = Field(default=0.0)
    delta_soc_total: float = Field(default=0.0)
    depth_cm: int = Field(default=30)
    provenance_hash: str = Field(default="")


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


class AggregationResponse(BaseModel):
    """Aggregated land use emissions response."""

    model_config = ConfigDict(frozen=True)

    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_co2e_tonnes: float = Field(default=0.0)
    total_removals_tonnes: float = Field(default=0.0)
    net_co2e_tonnes: float = Field(default=0.0)
    area_ha: float = Field(default=0.0)
    calculation_count: int = Field(default=0)
    period: str = Field(default="annual")


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="land-use-emissions")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_parcels: int = Field(default=0)
    total_transitions: int = Field(default=0)
    total_carbon_stocks: int = Field(default=0)
    total_soc_assessments: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)


# ===================================================================
# LandUseEmissionsService facade
# ===================================================================

_singleton_lock = threading.Lock()
_singleton_instance: Optional["LandUseEmissionsService"] = None


class LandUseEmissionsService:
    """Unified facade over the Land Use Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing.

    Example:
        >>> service = LandUseEmissionsService()
        >>> result = service.calculate({
        ...     "parcel_id": "parcel-001",
        ...     "from_category": "forest_land",
        ...     "to_category": "cropland",
        ...     "area_ha": 100.0,
        ...     "climate_zone": "tropical_moist",
        ...     "soil_type": "high_activity_clay",
        ... })
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Land Use Emissions Service facade.

        Args:
            config: Optional configuration override.
        """
        self.config = config if config is not None else get_config()
        self._start_time: float = time.monotonic()

        # Engine placeholders
        self._database_engine: Any = None
        self._carbon_stock_engine: Any = None
        self._change_tracker_engine: Any = None
        self._soc_engine: Any = None
        self._uncertainty_engine: Any = None
        self._compliance_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores
        self._calculations: List[Dict[str, Any]] = []
        self._parcels: Dict[str, Dict[str, Any]] = {}
        self._carbon_stocks: List[Dict[str, Any]] = []
        self._transitions: List[Dict[str, Any]] = []
        self._soc_assessments: List[Dict[str, Any]] = []
        self._compliance_results: List[Dict[str, Any]] = []
        self._uncertainty_results: List[Dict[str, Any]] = []

        # Statistics
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0

        logger.info("LandUseEmissionsService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def database_engine(self) -> Any:
        """Get the LandUseDatabaseEngine instance."""
        return self._database_engine

    @property
    def carbon_stock_engine(self) -> Any:
        """Get the CarbonStockCalculatorEngine instance."""
        return self._carbon_stock_engine

    @property
    def change_tracker_engine(self) -> Any:
        """Get the LandUseChangeTrackerEngine instance."""
        return self._change_tracker_engine

    @property
    def soc_engine(self) -> Any:
        """Get the SoilOrganicCarbonEngine instance."""
        return self._soc_engine

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
        """Get the LandUsePipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines."""
        config_dict: Dict[str, Any] = {}
        if self.config is not None and hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()

        # E1: LandUseDatabaseEngine
        self._init_single_engine(
            "LandUseDatabaseEngine",
            LandUseDatabaseEngine,
            "_database_engine",
            config_dict,
        )

        # E2: CarbonStockCalculatorEngine
        self._init_single_engine(
            "CarbonStockCalculatorEngine",
            CarbonStockCalculatorEngine,
            "_carbon_stock_engine",
            config_dict,
        )

        # E3: LandUseChangeTrackerEngine
        self._init_single_engine(
            "LandUseChangeTrackerEngine",
            LandUseChangeTrackerEngine,
            "_change_tracker_engine",
            config_dict,
        )

        # E4: SoilOrganicCarbonEngine
        self._init_single_engine(
            "SoilOrganicCarbonEngine",
            SoilOrganicCarbonEngine,
            "_soc_engine",
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

        # E7: LandUsePipelineEngine
        if LandUsePipelineEngine is not None:
            try:
                self._pipeline_engine = LandUsePipelineEngine(
                    database_engine=self._database_engine,
                    carbon_stock_engine=self._carbon_stock_engine,
                    change_tracker_engine=self._change_tracker_engine,
                    soc_engine=self._soc_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    compliance_engine=self._compliance_engine,
                    config=self.config,
                )
                logger.info("LandUsePipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "LandUsePipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("LandUsePipelineEngine not available")

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

    # ==================================================================
    # Public API: Calculate
    # ==================================================================

    def calculate(
        self,
        request_data: Dict[str, Any],
    ) -> CalculateResponse:
        """Calculate land use emissions for a single record.

        Computes carbon stock changes and resulting GHG emissions for
        a land parcel or transition event using IPCC methodology.

        Args:
            request_data: Calculation parameters including parcel_id,
                from_category, to_category, area_ha, climate_zone,
                soil_type, and optional method/tier/pools overrides.

        Returns:
            CalculateResponse with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = f"lu_calc_{uuid.uuid4().hex[:12]}"

        from_cat = request_data.get("from_category", "forest_land")
        to_cat = request_data.get("to_category", "cropland")
        method = request_data.get("method", "stock_difference")
        tier = request_data.get("tier", "tier_1")
        area_ha = float(request_data.get("area_ha", 0))

        try:
            total_co2e = 0.0
            removals = 0.0
            emissions_by_pool: Dict[str, float] = {}
            emissions_by_gas: Dict[str, float] = {}

            # Try pipeline engine
            if self._pipeline_engine is not None:
                pipe_result = self._pipeline_engine.execute_pipeline(
                    request=request_data,
                    gwp_source=request_data.get("gwp_source", "AR6"),
                )
                calc_data = pipe_result.get("calculation_data", {})
                total_co2e = float(
                    calc_data.get("total_co2e_tonnes", 0),
                )
                removals = float(
                    calc_data.get("removals_co2e_tonnes", 0),
                )
                emissions_by_pool = {
                    k: float(v)
                    for k, v in calc_data.get(
                        "emissions_by_pool", {},
                    ).items()
                }
                emissions_by_gas = {
                    k: float(v)
                    for k, v in calc_data.get(
                        "emissions_by_gas", {},
                    ).items()
                }
            elif self._carbon_stock_engine is not None:
                # Fallback to carbon stock calculator
                result = self._carbon_stock_engine.calculate(
                    request_data,
                )
                if isinstance(result, dict):
                    total_co2e = float(
                        result.get("total_co2e_tonnes", 0),
                    )
                    removals = float(
                        result.get("removals_co2e_tonnes", 0),
                    )

            net_co2e = total_co2e - removals
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "from_category": from_cat,
                "to_category": to_cat,
                "area_ha": area_ha,
                "total_co2e": total_co2e,
            })

            response = CalculateResponse(
                success=True,
                calculation_id=calc_id,
                from_category=from_cat,
                to_category=to_cat,
                method=method,
                tier=tier,
                total_co2e_tonnes=total_co2e,
                removals_co2e_tonnes=removals,
                net_co2e_tonnes=net_co2e,
                emissions_by_pool=emissions_by_pool,
                emissions_by_gas=emissions_by_gas,
                area_ha=area_ha,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

            # Cache the calculation
            self._cache_calculation(
                calc_id, request_data, response, provenance_hash,
            )
            self._total_calculations += 1

            logger.info(
                "Calculated %s: %s->%s method=%s co2e=%.4f tonnes",
                calc_id, from_cat, to_cat, method, total_co2e,
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate failed: %s", exc, exc_info=True,
            )
            return CalculateResponse(
                success=False,
                calculation_id=calc_id,
                from_category=from_cat,
                to_category=to_cat,
                method=method,
                tier=tier,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

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
            "from_category": response.from_category,
            "to_category": response.to_category,
            "method": response.method,
            "tier": response.tier,
            "total_co2e_tonnes": response.total_co2e_tonnes,
            "removals_co2e_tonnes": response.removals_co2e_tonnes,
            "net_co2e_tonnes": response.net_co2e_tonnes,
            "emissions_by_pool": response.emissions_by_pool,
            "emissions_by_gas": response.emissions_by_gas,
            "area_ha": response.area_ha,
            "provenance_hash": provenance_hash,
            "timestamp": response.timestamp,
            "status": "SUCCESS" if response.success else "FAILED",
            "tenant_id": request_data.get("tenant_id", ""),
            "parcel_id": request_data.get("parcel_id", ""),
            "climate_zone": request_data.get("climate_zone", ""),
        })

    # ==================================================================
    # Public API: Batch Calculate
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> BatchCalculateResponse:
        """Batch calculate land use emissions.

        Args:
            requests: List of calculation request dictionaries.
            gwp_source: Optional GWP source applied to all calculations.
            tenant_id: Optional tenant identifier for all calculations.

        Returns:
            BatchCalculateResponse with aggregate totals.
        """
        t0 = time.monotonic()
        batch_id = f"lu_batch_{uuid.uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []
        total_co2e = 0.0
        total_removals = 0.0
        successful = 0
        failed = 0

        for req in requests:
            if gwp_source is not None and "gwp_source" not in req:
                req["gwp_source"] = gwp_source
            if tenant_id is not None and "tenant_id" not in req:
                req["tenant_id"] = tenant_id

            resp = self.calculate(req)
            results.append(resp.model_dump())
            if resp.success:
                successful += 1
                total_co2e += resp.total_co2e_tonnes
                total_removals += resp.removals_co2e_tonnes
            else:
                failed += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._total_batch_runs += 1
        net_co2e = total_co2e - total_removals

        return BatchCalculateResponse(
            success=failed == 0,
            batch_id=batch_id,
            total_calculations=len(requests),
            successful=successful,
            failed=failed,
            total_co2e_tonnes=total_co2e,
            total_removals_tonnes=total_removals,
            net_co2e_tonnes=net_co2e,
            results=results,
            processing_time_ms=round(elapsed_ms, 3),
        )

    # ==================================================================
    # Public API: Carbon Stock CRUD
    # ==================================================================

    def record_carbon_stock(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a point-in-time carbon stock measurement.

        Args:
            data: Carbon stock snapshot data including parcel_id,
                pool, stock_tc_ha, and measurement_date.

        Returns:
            Dictionary with the recorded snapshot details.
        """
        snapshot_id = f"cs_{uuid.uuid4().hex[:12]}"
        parcel_id = data.get("parcel_id", "")
        pool = data.get("pool", "above_ground_biomass")
        stock_tc_ha = float(data.get("stock_tc_ha", 0))
        measurement_date = data.get("measurement_date", _utcnow_iso())
        tier = data.get("tier", "tier_1")
        source = data.get("source", "IPCC_2006")
        uncertainty_pct = data.get("uncertainty_pct")
        notes = data.get("notes", "")

        provenance_hash = _compute_hash({
            "snapshot_id": snapshot_id,
            "parcel_id": parcel_id,
            "pool": pool,
            "stock_tc_ha": stock_tc_ha,
            "measurement_date": measurement_date,
        })

        record: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "parcel_id": parcel_id,
            "pool": pool,
            "stock_tc_ha": stock_tc_ha,
            "measurement_date": measurement_date,
            "tier": tier,
            "source": source,
            "uncertainty_pct": uncertainty_pct,
            "notes": notes,
            "provenance_hash": provenance_hash,
            "created_at": _utcnow_iso(),
        }
        self._carbon_stocks.append(record)

        logger.info(
            "Recorded carbon stock %s: parcel=%s pool=%s stock=%.2f tC/ha",
            snapshot_id, parcel_id, pool, stock_tc_ha,
        )
        return record

    def get_carbon_stocks(
        self,
        parcel_id: str,
        pool: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """Get carbon stock measurement history for a parcel.

        Args:
            parcel_id: Land parcel identifier.
            pool: Optional filter by carbon pool.
            page: Page number (1-indexed).
            page_size: Items per page.

        Returns:
            Dictionary with paginated snapshot list.
        """
        filtered = [
            s for s in self._carbon_stocks
            if s.get("parcel_id") == parcel_id
        ]
        if pool is not None:
            filtered = [
                s for s in filtered if s.get("pool") == pool
            ]

        # Sort by measurement_date descending
        filtered.sort(
            key=lambda x: x.get("measurement_date", ""),
            reverse=True,
        )

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = filtered[start:end]

        return {
            "parcel_id": parcel_id,
            "snapshots": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ==================================================================
    # Public API: Transition CRUD
    # ==================================================================

    def record_transition(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a land-use transition event.

        Args:
            data: Transition data including parcel_id, from_category,
                to_category, transition_date, area_ha, transition_type.

        Returns:
            Dictionary with the recorded transition details.
        """
        transition_id = f"tr_{uuid.uuid4().hex[:12]}"
        parcel_id = data.get("parcel_id", "")
        from_cat = data.get("from_category", "")
        to_cat = data.get("to_category", "")
        transition_date = data.get("transition_date", _utcnow_iso())
        area_ha = float(data.get("area_ha", 0))
        transition_type = data.get("transition_type", "remaining")
        disturbance_type = data.get("disturbance_type", "none")
        notes = data.get("notes", "")

        # Auto-detect transition type if from/to differ
        if from_cat != to_cat and transition_type == "remaining":
            transition_type = "conversion"

        provenance_hash = _compute_hash({
            "transition_id": transition_id,
            "parcel_id": parcel_id,
            "from_category": from_cat,
            "to_category": to_cat,
            "area_ha": area_ha,
        })

        record: Dict[str, Any] = {
            "transition_id": transition_id,
            "parcel_id": parcel_id,
            "from_category": from_cat,
            "to_category": to_cat,
            "transition_date": transition_date,
            "area_ha": area_ha,
            "transition_type": transition_type,
            "disturbance_type": disturbance_type,
            "notes": notes,
            "provenance_hash": provenance_hash,
            "created_at": _utcnow_iso(),
        }
        self._transitions.append(record)

        # Delegate to change tracker engine
        if self._change_tracker_engine is not None:
            try:
                self._change_tracker_engine.record_transition(data)
            except Exception as exc:
                logger.warning(
                    "Change tracker engine failed: %s", exc,
                )

        logger.info(
            "Recorded transition %s: %s->%s area=%.2f ha",
            transition_id, from_cat, to_cat, area_ha,
        )
        return record

    def get_transitions(
        self,
        page: int = 1,
        page_size: int = 20,
        parcel_id: Optional[str] = None,
        from_category: Optional[str] = None,
        to_category: Optional[str] = None,
        transition_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List land-use transition records with filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            parcel_id: Optional filter by parcel identifier.
            from_category: Optional filter by source land category.
            to_category: Optional filter by target land category.
            transition_type: Optional filter by transition type.

        Returns:
            Dictionary with paginated transition list.
        """
        filtered = list(self._transitions)

        if parcel_id is not None:
            filtered = [
                t for t in filtered
                if t.get("parcel_id") == parcel_id
            ]
        if from_category is not None:
            filtered = [
                t for t in filtered
                if t.get("from_category") == from_category
            ]
        if to_category is not None:
            filtered = [
                t for t in filtered
                if t.get("to_category") == to_category
            ]
        if transition_type is not None:
            filtered = [
                t for t in filtered
                if t.get("transition_type") == transition_type
            ]

        # Sort by transition_date descending
        filtered.sort(
            key=lambda x: x.get("transition_date", ""),
            reverse=True,
        )

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = filtered[start:end]

        return {
            "transitions": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def get_transition_matrix(
        self,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get the 6x6 IPCC land category transition matrix.

        Summarizes total area transitioned between each pair of
        land categories. Matrix[from][to] = total area in hectares.

        Args:
            tenant_id: Optional tenant filter.

        Returns:
            Dictionary with categories, matrix, and totals.
        """
        # Initialize empty 6x6 matrix
        matrix: Dict[str, Dict[str, float]] = {}
        for from_cat in IPCC_LAND_CATEGORIES:
            matrix[from_cat] = {}
            for to_cat in IPCC_LAND_CATEGORIES:
                matrix[from_cat][to_cat] = 0.0

        # Filter transitions
        filtered = list(self._transitions)
        if tenant_id is not None:
            # Resolve parcel tenant_ids
            filtered = [
                t for t in filtered
                if self._parcels.get(
                    t.get("parcel_id", ""), {},
                ).get("tenant_id") == tenant_id
            ]

        # Populate the matrix
        total_area = 0.0
        total_transitions = 0
        for trans in filtered:
            from_cat = trans.get("from_category", "")
            to_cat = trans.get("to_category", "")
            area = float(trans.get("area_ha", 0))

            if from_cat in matrix and to_cat in matrix.get(from_cat, {}):
                matrix[from_cat][to_cat] += area
                total_area += area
                total_transitions += 1

        return {
            "categories": IPCC_LAND_CATEGORIES,
            "matrix": matrix,
            "total_area_ha": round(total_area, 4),
            "total_transitions": total_transitions,
        }

    # ==================================================================
    # Public API: SOC Assessment
    # ==================================================================

    def assess_soc(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a soil organic carbon assessment.

        Uses IPCC Tier 1 approach: SOC = SOC_ref * F_LU * F_MG * F_I

        Args:
            data: SOC assessment parameters including parcel_id,
                climate_zone, soil_type, land_category.

        Returns:
            Dictionary with SOC assessment results.
        """
        assessment_id = f"soc_{uuid.uuid4().hex[:12]}"

        # Try the SOC engine first
        if self._soc_engine is not None:
            try:
                result = self._soc_engine.assess(data)
                if isinstance(result, dict):
                    result["assessment_id"] = assessment_id
                    result["parcel_id"] = data.get("parcel_id", "")
                    result["provenance_hash"] = _compute_hash(result)
                    self._soc_assessments.append(result)
                    logger.info(
                        "SOC assessment %s via engine: parcel=%s",
                        assessment_id, data.get("parcel_id"),
                    )
                    return result
            except Exception as exc:
                logger.warning(
                    "SOC engine failed, using fallback: %s", exc,
                )

        # Fallback: compute SOC using IPCC defaults from models
        return self._assess_soc_fallback(assessment_id, data)

    def _assess_soc_fallback(
        self,
        assessment_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback SOC assessment using IPCC default tables.

        Args:
            assessment_id: Unique assessment identifier.
            data: SOC assessment parameters.

        Returns:
            Dictionary with SOC assessment results.
        """
        parcel_id = data.get("parcel_id", "")
        climate_zone = data.get("climate_zone", "tropical_moist")
        soil_type = data.get("soil_type", "high_activity_clay")
        land_category = data.get("land_category", "forest_land")
        management = data.get(
            "management_practice", "nominally_managed",
        )
        input_level = data.get("input_level", "medium")
        depth_cm = int(data.get("depth_cm", 30))
        transition_years = int(data.get("transition_years", 20))

        # Import lookup tables
        try:
            from greenlang.land_use_emissions.models import (
                SOC_REFERENCE_STOCKS,
                SOC_LAND_USE_FACTORS,
                SOC_MANAGEMENT_FACTORS,
                SOC_INPUT_FACTORS,
                ClimateZone as CZ,
                SoilType as ST,
                LandCategory as LC,
                ManagementPractice as MP,
                InputLevel as IL,
            )

            cz_enum = CZ(climate_zone)
            st_enum = ST(soil_type)
            lc_enum = LC(land_category)
            mp_enum = MP(management)
            il_enum = IL(input_level)

            soc_ref = float(
                SOC_REFERENCE_STOCKS.get((cz_enum, st_enum), Decimal("50")),
            )
            f_lu = float(
                SOC_LAND_USE_FACTORS.get(
                    (lc_enum, cz_enum), Decimal("1.0"),
                ),
            )
            f_mg = float(
                SOC_MANAGEMENT_FACTORS.get(
                    (mp_enum, cz_enum), Decimal("1.0"),
                ),
            )
            f_i = float(
                SOC_INPUT_FACTORS.get(il_enum, Decimal("1.0")),
            )
        except (ImportError, ValueError) as exc:
            logger.warning(
                "SOC factor lookup failed: %s. Using defaults.", exc,
            )
            soc_ref = 50.0
            f_lu = 1.0
            f_mg = 1.0
            f_i = 1.0

        soc_current = soc_ref * f_lu * f_mg * f_i

        # Calculate previous SOC if transition info provided
        soc_previous = None
        delta_soc_annual = 0.0
        delta_soc_total = 0.0

        prev_lc = data.get("previous_land_category")
        if prev_lc is not None:
            try:
                from greenlang.land_use_emissions.models import (
                    LandCategory as LC2,
                    ManagementPractice as MP2,
                    InputLevel as IL2,
                )
                prev_lc_enum = LC2(prev_lc)
                prev_mgmt = data.get(
                    "previous_management", "nominally_managed",
                )
                prev_input = data.get("previous_input_level", "medium")
                prev_mp_enum = MP2(prev_mgmt)
                prev_il_enum = IL2(prev_input)

                prev_f_lu = float(
                    SOC_LAND_USE_FACTORS.get(
                        (prev_lc_enum, cz_enum), Decimal("1.0"),
                    ),
                )
                prev_f_mg = float(
                    SOC_MANAGEMENT_FACTORS.get(
                        (prev_mp_enum, cz_enum), Decimal("1.0"),
                    ),
                )
                prev_f_i = float(
                    SOC_INPUT_FACTORS.get(
                        prev_il_enum, Decimal("1.0"),
                    ),
                )
                soc_previous = soc_ref * prev_f_lu * prev_f_mg * prev_f_i
                delta_soc_total = soc_current - soc_previous
                if transition_years > 0:
                    delta_soc_annual = delta_soc_total / transition_years
            except (ImportError, ValueError) as exc:
                logger.warning(
                    "Previous SOC lookup failed: %s", exc,
                )

        provenance_hash = _compute_hash({
            "assessment_id": assessment_id,
            "parcel_id": parcel_id,
            "soc_ref": soc_ref,
            "f_lu": f_lu,
            "f_mg": f_mg,
            "f_i": f_i,
            "soc_current": soc_current,
        })

        result: Dict[str, Any] = {
            "assessment_id": assessment_id,
            "parcel_id": parcel_id,
            "soc_ref": round(soc_ref, 4),
            "f_lu": round(f_lu, 4),
            "f_mg": round(f_mg, 4),
            "f_i": round(f_i, 4),
            "soc_current": round(soc_current, 4),
            "soc_previous": (
                round(soc_previous, 4) if soc_previous is not None
                else None
            ),
            "delta_soc_annual": round(delta_soc_annual, 6),
            "delta_soc_total": round(delta_soc_total, 4),
            "depth_cm": depth_cm,
            "climate_zone": climate_zone,
            "soil_type": soil_type,
            "land_category": land_category,
            "management_practice": management,
            "input_level": input_level,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        }

        self._soc_assessments.append(result)
        logger.info(
            "SOC assessment %s (fallback): parcel=%s soc=%.2f tC/ha",
            assessment_id, parcel_id, soc_current,
        )
        return result

    # ==================================================================
    # Public API: Uncertainty Analysis
    # ==================================================================

    def run_uncertainty(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation.

        Args:
            data: Uncertainty parameters including calculation_id,
                iterations, seed, and confidence_level.

        Returns:
            Dictionary with uncertainty analysis results.
        """
        calc_id = data.get("calculation_id", "")
        method = "monte_carlo"
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
                return self._format_uncertainty_result(
                    calc_id, result, iterations, confidence_level,
                )
            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed: %s", exc,
                )

        # Fallback: analytical estimate
        return self._uncertainty_fallback(
            calc_id, total_co2e, iterations, confidence_level,
        )

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
        Carlo engine is unavailable.

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
        cv = (std_estimate / abs(total_co2e) * 100.0) if total_co2e != 0 else 0.0

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
        return result

    # ==================================================================
    # Public API: Compliance Check
    # ==================================================================

    def check_compliance(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run multi-framework compliance check.

        Args:
            data: Compliance check parameters including optional
                calculation_id and frameworks list.

        Returns:
            Dictionary with compliance check results.
        """
        compliance_id = f"comp_{uuid.uuid4().hex[:12]}"
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

        formatted: Dict[str, Any] = {
            "id": compliance_id,
            "success": True,
            "frameworks_checked": result.get("frameworks_checked", 0),
            "compliant": result.get("compliant", 0),
            "non_compliant": result.get("non_compliant", 0),
            "partial": result.get("partial", 0),
            "results": results_list,
            "timestamp": _utcnow_iso(),
        }
        self._compliance_results.append(formatted)
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
        default_frameworks = [
            "GHG_PROTOCOL", "IPCC", "CSRD",
            "EU_LULUCF", "UK_SECR", "UNFCCC",
        ]
        check_frameworks = frameworks if frameworks else default_frameworks

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
        return result

    # ==================================================================
    # Public API: Aggregation
    # ==================================================================

    def aggregate(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate land use emission results.

        Args:
            data: Aggregation parameters including tenant_id,
                period, group_by, date_from, date_to, and filters.

        Returns:
            Dictionary with aggregated emission totals.
        """
        tenant_id = data.get("tenant_id", "")
        period = data.get("period", "annual")
        group_by = data.get("group_by", ["land_category"])
        date_from = data.get("date_from")
        date_to = data.get("date_to")
        land_categories = data.get("land_categories")

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

        # Apply land category filter
        if land_categories is not None:
            filtered = [
                c for c in filtered
                if c.get("to_category") in land_categories
                or c.get("from_category") in land_categories
            ]

        # Group and aggregate
        groups: Dict[str, Dict[str, float]] = {}
        total_co2e = 0.0
        total_removals = 0.0
        total_area = 0.0

        for calc in filtered:
            # Build group key
            key_parts = []
            for field in group_by:
                key_parts.append(str(calc.get(field, "unknown")))
            group_key = "|".join(key_parts) if key_parts else "all"

            if group_key not in groups:
                groups[group_key] = {
                    "total_co2e_tonnes": 0.0,
                    "removals_co2e_tonnes": 0.0,
                    "net_co2e_tonnes": 0.0,
                    "area_ha": 0.0,
                    "count": 0.0,
                }

            co2e = float(calc.get("total_co2e_tonnes", 0))
            rem = float(calc.get("removals_co2e_tonnes", 0))
            area = float(calc.get("area_ha", 0))

            groups[group_key]["total_co2e_tonnes"] += co2e
            groups[group_key]["removals_co2e_tonnes"] += rem
            groups[group_key]["net_co2e_tonnes"] += (co2e - rem)
            groups[group_key]["area_ha"] += area
            groups[group_key]["count"] += 1

            total_co2e += co2e
            total_removals += rem
            total_area += area

        return {
            "groups": groups,
            "total_co2e_tonnes": round(total_co2e, 6),
            "total_removals_tonnes": round(total_removals, 6),
            "net_co2e_tonnes": round(total_co2e - total_removals, 6),
            "area_ha": round(total_area, 4),
            "calculation_count": len(filtered),
            "period": period,
            "tenant_id": tenant_id,
            "timestamp": _utcnow_iso(),
        }

    # ==================================================================
    # Public API: Parcel CRUD
    # ==================================================================

    def get_parcel(
        self,
        parcel_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a land parcel by its identifier.

        Args:
            parcel_id: Unique parcel identifier.

        Returns:
            Parcel dictionary or None if not found.
        """
        return self._parcels.get(parcel_id)

    def list_parcels(
        self,
        page: int = 1,
        page_size: int = 20,
        tenant_id: Optional[str] = None,
        land_category: Optional[str] = None,
        climate_zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List registered land parcels with pagination and filters.

        Args:
            page: Page number (1-indexed).
            page_size: Items per page.
            tenant_id: Optional filter by tenant.
            land_category: Optional filter by land category.
            climate_zone: Optional filter by climate zone.

        Returns:
            Dictionary with paginated parcel list.
        """
        all_parcels = list(self._parcels.values())

        if tenant_id is not None:
            all_parcels = [
                p for p in all_parcels
                if p.get("tenant_id") == tenant_id
            ]
        if land_category is not None:
            all_parcels = [
                p for p in all_parcels
                if p.get("land_category") == land_category
            ]
        if climate_zone is not None:
            all_parcels = [
                p for p in all_parcels
                if p.get("climate_zone") == climate_zone
            ]

        total = len(all_parcels)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = all_parcels[start:end]

        return {
            "parcels": page_data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def register_parcel(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a new land parcel for LULUCF tracking.

        Args:
            data: Parcel data including name, area_ha, land_category,
                climate_zone, soil_type, latitude, longitude, tenant_id.

        Returns:
            Dictionary with the registered parcel details.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        parcel_id = f"parcel_{uuid.uuid4().hex[:12]}"

        # Validate required fields
        required_fields = [
            "name", "area_ha", "land_category", "climate_zone",
            "soil_type", "latitude", "longitude", "tenant_id",
        ]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ValueError(
                f"Missing required fields: {', '.join(missing)}"
            )

        now_iso = _utcnow_iso()
        provenance_hash = _compute_hash({
            "parcel_id": parcel_id,
            "name": data.get("name"),
            "area_ha": data.get("area_ha"),
            "land_category": data.get("land_category"),
            "tenant_id": data.get("tenant_id"),
        })

        record: Dict[str, Any] = {
            "parcel_id": parcel_id,
            "name": data.get("name", ""),
            "area_ha": float(data.get("area_ha", 0)),
            "land_category": data.get("land_category", ""),
            "climate_zone": data.get("climate_zone", ""),
            "soil_type": data.get("soil_type", ""),
            "latitude": float(data.get("latitude", 0)),
            "longitude": float(data.get("longitude", 0)),
            "tenant_id": data.get("tenant_id", ""),
            "country_code": data.get("country_code", ""),
            "management_practice": data.get(
                "management_practice", "nominally_managed",
            ),
            "input_level": data.get("input_level", "medium"),
            "peatland_status": data.get("peatland_status"),
            "provenance_hash": provenance_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        self._parcels[parcel_id] = record
        logger.info(
            "Registered parcel %s: name=%s category=%s area=%.2f ha",
            parcel_id, record["name"], record["land_category"],
            record["area_ha"],
        )
        return record

    def update_parcel(
        self,
        parcel_id: str,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update an existing land parcel's attributes.

        Args:
            parcel_id: Unique parcel identifier.
            data: Dictionary of fields to update. Only non-None
                values will be applied.

        Returns:
            Updated parcel dictionary or None if not found.
        """
        existing = self._parcels.get(parcel_id)
        if existing is None:
            return None

        # Updatable fields whitelist
        updatable = {
            "name", "area_ha", "land_category", "climate_zone",
            "soil_type", "management_practice", "input_level",
            "peatland_status", "country_code", "latitude", "longitude",
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
            "parcel_id": parcel_id,
            "name": existing.get("name"),
            "area_ha": existing.get("area_ha"),
            "land_category": existing.get("land_category"),
            "updated_at": existing.get("updated_at"),
        })

        self._parcels[parcel_id] = existing
        logger.info("Updated parcel %s", parcel_id)
        return existing

    # ==================================================================
    # Public API: Health & Stats
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Service health check.

        Returns:
            HealthResponse with engine availability status.
        """
        engines: Dict[str, str] = {
            "land_use_database": (
                "available"
                if self._database_engine is not None
                else "unavailable"
            ),
            "carbon_stock_calculator": (
                "available"
                if self._carbon_stock_engine is not None
                else "unavailable"
            ),
            "land_use_change_tracker": (
                "available"
                if self._change_tracker_engine is not None
                else "unavailable"
            ),
            "soil_organic_carbon": (
                "available"
                if self._soc_engine is not None
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
            service="land-use-emissions",
            version="1.0.0",
            engines=engines,
        )

    def get_stats(self) -> StatsResponse:
        """Get service statistics.

        Returns:
            StatsResponse with aggregate counts and uptime.
        """
        uptime = time.monotonic() - self._start_time
        return StatsResponse(
            total_calculations=self._total_calculations,
            total_parcels=len(self._parcels),
            total_transitions=len(self._transitions),
            total_carbon_stocks=len(self._carbon_stocks),
            total_soc_assessments=len(self._soc_assessments),
            total_compliance_checks=len(self._compliance_results),
            uptime_seconds=round(uptime, 3),
        )


# ===================================================================
# Thread-safe singleton access
# ===================================================================

_service_instance: Optional[LandUseEmissionsService] = None
_service_lock = threading.Lock()


def get_service() -> LandUseEmissionsService:
    """Get or create the singleton LandUseEmissionsService instance.

    Returns:
        LandUseEmissionsService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = LandUseEmissionsService()
    return _service_instance


def get_router() -> Any:
    """Get the FastAPI router for land use emissions.

    Returns:
        FastAPI APIRouter or None if FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.land_use_emissions.api.router import create_router
        return create_router()
    except ImportError:
        logger.warning(
            "Land use emissions API router module not available"
        )
        return None


def configure_land_use(
    app: Any,
    config: Any = None,
) -> LandUseEmissionsService:
    """Configure the Land Use Emissions Service on a FastAPI application.

    Creates the LandUseEmissionsService singleton, stores it in
    app.state, and mounts the API router.

    Args:
        app: FastAPI application instance.
        config: Optional configuration override.

    Returns:
        LandUseEmissionsService instance.
    """
    global _service_instance

    service = LandUseEmissionsService(config=config)

    with _service_lock:
        _service_instance = service

    if hasattr(app, "state"):
        app.state.land_use_emissions_service = service

    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Land use emissions API router mounted")
    else:
        logger.warning(
            "Land use emissions router not available; API not mounted"
        )

    logger.info("Land Use Emissions service configured")
    return service


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "LandUseEmissionsService",
    "configure_land_use",
    "get_service",
    "get_router",
    "CalculateResponse",
    "BatchCalculateResponse",
    "ParcelResponse",
    "ParcelListResponse",
    "CarbonStockResponse",
    "TransitionResponse",
    "TransitionMatrixResponse",
    "SOCAssessmentResponse",
    "UncertaintyResponse",
    "ComplianceCheckResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
]
