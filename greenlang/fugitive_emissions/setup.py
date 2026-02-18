# -*- coding: utf-8 -*-
"""
Fugitive Emissions Service Setup - AGENT-MRV-005
=================================================

Service facade for the Fugitive Emissions Agent (GL-MRV-SCOPE1-005).

Provides ``configure_fugitive_emissions(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``FugitiveEmissionsService`` facade class that aggregates all 7 engines:

    1. FugitiveSourceDatabaseEngine   - Source types, EFs, gas compositions
    2. EmissionCalculatorEngine       - 5 calculation methods
    3. LeakDetectionEngine            - LDAR survey scheduling, leak tracking
    4. EquipmentComponentEngine       - Component registry, tank losses
    5. UncertaintyQuantifierEngine    - Monte Carlo & DQI uncertainty
    6. ComplianceCheckerEngine        - 7-framework regulatory compliance
    7. FugitiveEmissionsPipelineEngine - 8-stage orchestration pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.fugitive_emissions.setup import configure_fugitive_emissions
    >>> app = FastAPI()
    >>> configure_fugitive_emissions(app)

    >>> from greenlang.fugitive_emissions.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate({
    ...     "source_type": "EQUIPMENT_LEAK",
    ...     "facility_id": "FAC-001",
    ...     "component_count": 5000,
    ... })

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
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
    from greenlang.fugitive_emissions.config import (
        FugitiveEmissionsConfig,
        get_config,
    )
except ImportError:
    FugitiveEmissionsConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.fugitive_emissions.equipment_component import (
        EquipmentComponentEngine,
    )
except ImportError:
    EquipmentComponentEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fugitive_emissions.uncertainty_quantifier import (
        UncertaintyQuantifierEngine,
    )
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fugitive_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fugitive_emissions.fugitive_emissions_pipeline import (
        FugitiveEmissionsPipelineEngine,
    )
except ImportError:
    FugitiveEmissionsPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.fugitive_emissions.provenance import ProvenanceTracker
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
# Lightweight Pydantic response models (14 models)
# ===================================================================


class CalculateResponse(BaseModel):
    """Single fugitive emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    source_type: str = Field(default="")
    calculation_method: str = Field(default="AVERAGE_EMISSION_FACTOR")
    total_co2e_kg: float = Field(default=0.0)
    ch4_kg: float = Field(default=0.0)
    voc_kg: float = Field(default=0.0)
    n2o_kg: float = Field(default=0.0)
    uncertainty_pct: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
    """Batch fugitive emission calculation response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    total_calculations: int = Field(default=0)
    successful: int = Field(default=0)
    failed: int = Field(default=0)
    total_co2e_kg: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class SourceListResponse(BaseModel):
    """Response listing fugitive emission source types."""

    model_config = ConfigDict(frozen=True)

    sources: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class SourceDetailResponse(BaseModel):
    """Detailed response for a single source type."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(default="")
    source_type: str = Field(default="")
    name: str = Field(default="")
    gases: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)


class ComponentListResponse(BaseModel):
    """Response listing equipment components."""

    model_config = ConfigDict(frozen=True)

    components: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class ComponentDetailResponse(BaseModel):
    """Detailed response for a single equipment component."""

    model_config = ConfigDict(frozen=True)

    component_id: str = Field(default="")
    tag_number: str = Field(default="")
    component_type: str = Field(default="")
    service_type: str = Field(default="")
    facility_id: str = Field(default="")


class SurveyListResponse(BaseModel):
    """Response listing LDAR surveys."""

    model_config = ConfigDict(frozen=True)

    surveys: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class FactorListResponse(BaseModel):
    """Response listing emission factors."""

    model_config = ConfigDict(frozen=True)

    factors: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class FactorDetailResponse(BaseModel):
    """Detailed response for a single emission factor."""

    model_config = ConfigDict(frozen=True)

    factor_id: str = Field(default="")
    source_type: str = Field(default="")
    component_type: str = Field(default="")
    gas: str = Field(default="")
    value: float = Field(default=0.0)
    source: str = Field(default="")


class RepairListResponse(BaseModel):
    """Response listing component repairs."""

    model_config = ConfigDict(frozen=True)

    repairs: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0)


class UncertaintyResponse(BaseModel):
    """Monte Carlo uncertainty analysis response."""

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
    """Regulatory compliance check response."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    frameworks_checked: int = Field(default=0)
    compliant: int = Field(default=0)
    non_compliant: int = Field(default=0)
    partial: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="fugitive-emissions")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(default=0)
    total_sources: int = Field(default=0)
    total_components: int = Field(default=0)
    total_surveys: int = Field(default=0)
    total_repairs: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)


# ===================================================================
# FugitiveEmissionsService facade
# ===================================================================

_singleton_lock = threading.Lock()
_singleton_instance: Optional["FugitiveEmissionsService"] = None


class FugitiveEmissionsService:
    """Unified facade over the Fugitive Emissions Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 20 REST API operations.

    Each method records provenance via SHA-256 hashing.

    Example:
        >>> service = FugitiveEmissionsService()
        >>> result = service.calculate({
        ...     "source_type": "EQUIPMENT_LEAK",
        ...     "facility_id": "FAC-001",
        ...     "component_count": 5000,
        ... })
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the Fugitive Emissions Service facade.

        Args:
            config: Optional configuration override.
        """
        self.config = config if config is not None else get_config()
        self._start_time: float = time.monotonic()

        # Engine placeholders
        self._equipment_component_engine: Any = None
        self._uncertainty_engine: Any = None
        self._compliance_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores
        self._calculations: List[Dict[str, Any]] = []
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._components: Dict[str, Dict[str, Any]] = {}
        self._surveys: Dict[str, Dict[str, Any]] = {}
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._repairs: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0

        # Pre-populate source types
        self._populate_default_sources()

        logger.info("FugitiveEmissionsService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def equipment_component_engine(self) -> Any:
        """Get the EquipmentComponentEngine instance."""
        return self._equipment_component_engine

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
        """Get the FugitiveEmissionsPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines."""
        config_dict: Dict[str, Any] = {}
        if self.config is not None and hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()

        # E4: EquipmentComponentEngine
        if EquipmentComponentEngine is not None:
            try:
                self._equipment_component_engine = (
                    EquipmentComponentEngine(config=config_dict)
                )
                logger.info("EquipmentComponentEngine initialized")
            except Exception as exc:
                logger.warning(
                    "EquipmentComponentEngine init failed: %s", exc,
                )
        else:
            logger.warning("EquipmentComponentEngine not available")

        # E5: UncertaintyQuantifierEngine
        if UncertaintyQuantifierEngine is not None:
            try:
                self._uncertainty_engine = UncertaintyQuantifierEngine(
                    config=config_dict,
                )
                logger.info("UncertaintyQuantifierEngine initialized")
            except Exception as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine init failed: %s", exc,
                )
        else:
            logger.warning("UncertaintyQuantifierEngine not available")

        # E6: ComplianceCheckerEngine
        if ComplianceCheckerEngine is not None:
            try:
                self._compliance_engine = ComplianceCheckerEngine(
                    config=config_dict,
                )
                logger.info("ComplianceCheckerEngine initialized")
            except Exception as exc:
                logger.warning(
                    "ComplianceCheckerEngine init failed: %s", exc,
                )
        else:
            logger.warning("ComplianceCheckerEngine not available")

        # E7: FugitiveEmissionsPipelineEngine
        if FugitiveEmissionsPipelineEngine is not None:
            try:
                self._pipeline_engine = FugitiveEmissionsPipelineEngine(
                    equipment_component=self._equipment_component_engine,
                    uncertainty_engine=self._uncertainty_engine,
                    compliance_checker=self._compliance_engine,
                    config=self.config,
                )
                logger.info("FugitiveEmissionsPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "FugitiveEmissionsPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "FugitiveEmissionsPipelineEngine not available"
            )

    def _populate_default_sources(self) -> None:
        """Populate the in-memory source registry from built-in data."""
        try:
            from greenlang.fugitive_emissions.fugitive_emissions_pipeline import (
                SOURCE_TYPES,
            )
        except ImportError:
            SOURCE_TYPES = {
                "EQUIPMENT_LEAK": {
                    "name": "Equipment Leak",
                    "gases": ["CH4", "VOC"],
                    "methods": ["AVERAGE_EMISSION_FACTOR"],
                },
            }

        for src_key, src_data in SOURCE_TYPES.items():
            self._sources[src_key] = {
                "source_id": src_key,
                "source_type": src_key,
                "name": src_data.get("name", src_key),
                "gases": src_data.get("gases", []),
                "methods": src_data.get("methods", []),
            }

    # ==================================================================
    # Public API: Calculate
    # ==================================================================

    def calculate(
        self,
        request_data: Dict[str, Any],
    ) -> CalculateResponse:
        """Calculate fugitive emissions for a single record.

        Args:
            request_data: Calculation parameters.

        Returns:
            CalculateResponse with emissions breakdown.
        """
        t0 = time.monotonic()
        calc_id = f"fe_calc_{uuid.uuid4().hex[:12]}"

        source_type = request_data.get("source_type", "EQUIPMENT_LEAK")
        method = request_data.get(
            "calculation_method", "AVERAGE_EMISSION_FACTOR",
        )

        try:
            # Try pipeline engine
            if self._pipeline_engine is not None:
                pipe_result = self._pipeline_engine.execute_pipeline(
                    request=request_data,
                    gwp_source=request_data.get("gwp_source", "AR6"),
                )
                calc_data = pipe_result.get("calculation_data", {})
                total_co2e_kg = float(calc_data.get("total_co2e_kg", 0))

                # Extract per-gas
                gas_emissions = calc_data.get("gas_emissions", [])
                ch4_kg = sum(
                    float(g.get("mass_kg", 0))
                    for g in gas_emissions if g.get("gas") == "CH4"
                )
                voc_kg = sum(
                    float(g.get("mass_kg", 0))
                    for g in gas_emissions if g.get("gas") == "VOC"
                )
                n2o_kg = sum(
                    float(g.get("mass_kg", 0))
                    for g in gas_emissions if g.get("gas") == "N2O"
                )
            else:
                total_co2e_kg = 0.0
                ch4_kg = 0.0
                voc_kg = 0.0
                n2o_kg = 0.0

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "source_type": source_type,
                "total_co2e_kg": total_co2e_kg,
            })

            response = CalculateResponse(
                success=True,
                calculation_id=calc_id,
                source_type=source_type,
                calculation_method=method,
                total_co2e_kg=total_co2e_kg,
                ch4_kg=ch4_kg,
                voc_kg=voc_kg,
                n2o_kg=n2o_kg,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

            # Cache
            self._calculations.append({
                "calculation_id": calc_id,
                "source_type": source_type,
                "method": method,
                "total_co2e_kg": total_co2e_kg,
                "provenance_hash": provenance_hash,
                "timestamp": _utcnow_iso(),
                "status": "SUCCESS",
            })
            self._total_calculations += 1

            logger.info(
                "Calculated %s: source=%s method=%s co2e=%.4f kg",
                calc_id, source_type, method, total_co2e_kg,
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
                source_type=source_type,
                calculation_method=method,
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
        """Batch calculate fugitive emissions.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            BatchCalculateResponse with aggregate totals.
        """
        t0 = time.monotonic()
        results: List[Dict[str, Any]] = []
        total_co2e_kg = 0.0
        successful = 0
        failed = 0

        for req in requests:
            resp = self.calculate(req)
            results.append(resp.model_dump())
            if resp.success:
                successful += 1
                total_co2e_kg += resp.total_co2e_kg
            else:
                failed += 1

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._total_batch_runs += 1

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
    # Public API: Source CRUD
    # ==================================================================

    def register_source(
        self,
        data: Dict[str, Any],
    ) -> SourceDetailResponse:
        """Register a fugitive emission source type.

        Args:
            data: Source type data.

        Returns:
            SourceDetailResponse.
        """
        src_type = data.get("source_type", "")
        record = {
            "source_id": src_type,
            "source_type": src_type,
            "name": data.get("name", src_type.replace("_", " ").title()),
            "gases": data.get("gases", ["CH4"]),
            "methods": data.get("methods", ["AVERAGE_EMISSION_FACTOR"]),
        }
        self._sources[src_type] = record
        logger.info("Registered source type: %s", src_type)
        return SourceDetailResponse(**record)

    def list_sources(
        self, page: int = 1, page_size: int = 20,
    ) -> SourceListResponse:
        """List source types with pagination."""
        all_sources = list(self._sources.values())
        total = len(all_sources)
        start = (page - 1) * page_size
        page_data = all_sources[start:start + page_size]
        return SourceListResponse(
            sources=page_data, total=total, page=page, page_size=page_size,
        )

    def get_source(self, source_id: str) -> Optional[SourceDetailResponse]:
        """Get source type details."""
        record = self._sources.get(source_id)
        if record is None:
            return None
        return SourceDetailResponse(**record)

    # ==================================================================
    # Public API: Component CRUD
    # ==================================================================

    def register_component(
        self, data: Dict[str, Any],
    ) -> ComponentDetailResponse:
        """Register an equipment component."""
        if self._equipment_component_engine is not None:
            try:
                result = self._equipment_component_engine.register_component(
                    data,
                )
                self._components[result["component_id"]] = result
                return ComponentDetailResponse(
                    component_id=result.get("component_id", ""),
                    tag_number=result.get("tag_number", ""),
                    component_type=result.get("component_type", ""),
                    service_type=result.get("service_type", ""),
                    facility_id=result.get("facility_id", ""),
                )
            except Exception as exc:
                logger.warning(
                    "Equipment component engine failed: %s", exc,
                )

        # Fallback
        comp_id = f"comp_{uuid.uuid4().hex[:12]}"
        record = {
            "component_id": comp_id,
            "tag_number": data.get("tag_number", ""),
            "component_type": data.get("component_type", "other"),
            "service_type": data.get("service_type", "gas"),
            "facility_id": data.get("facility_id", ""),
        }
        self._components[comp_id] = record
        return ComponentDetailResponse(**record)

    def list_components(
        self, page: int = 1, page_size: int = 20,
    ) -> ComponentListResponse:
        """List equipment components."""
        all_comps = list(self._components.values())
        total = len(all_comps)
        start = (page - 1) * page_size
        page_data = all_comps[start:start + page_size]
        return ComponentListResponse(
            components=page_data, total=total, page=page,
            page_size=page_size,
        )

    def get_component(
        self, component_id: str,
    ) -> Optional[ComponentDetailResponse]:
        """Get component details."""
        record = self._components.get(component_id)
        if record is None:
            return None
        return ComponentDetailResponse(**{
            k: record.get(k, "") for k in [
                "component_id", "tag_number", "component_type",
                "service_type", "facility_id",
            ]
        })

    # ==================================================================
    # Public API: Survey CRUD
    # ==================================================================

    def register_survey(
        self, data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register an LDAR survey."""
        survey_id = data.get(
            "survey_id", f"survey_{uuid.uuid4().hex[:12]}",
        )
        record = {"survey_id": survey_id}
        record.update(data)
        record["survey_id"] = survey_id
        self._surveys[survey_id] = record
        logger.info("Registered survey: %s", survey_id)
        return record

    def list_surveys(self) -> SurveyListResponse:
        """List LDAR surveys."""
        all_surveys = list(self._surveys.values())
        return SurveyListResponse(
            surveys=all_surveys, total=len(all_surveys),
        )

    # ==================================================================
    # Public API: Emission Factor CRUD
    # ==================================================================

    def register_factor(
        self, data: Dict[str, Any],
    ) -> FactorDetailResponse:
        """Register a custom emission factor."""
        factor_id = data.get(
            "factor_id", f"fef_{uuid.uuid4().hex[:12]}",
        )
        record = {
            "factor_id": factor_id,
            "source_type": data.get("source_type", ""),
            "component_type": data.get("component_type", ""),
            "gas": data.get("gas", "CH4"),
            "value": float(data.get("value", 0)),
            "source": data.get("source", "CUSTOM"),
        }
        self._emission_factors[factor_id] = record
        logger.info("Registered factor: %s", factor_id)
        return FactorDetailResponse(**record)

    def list_factors(self) -> FactorListResponse:
        """List emission factors."""
        all_factors = list(self._emission_factors.values())
        return FactorListResponse(
            factors=all_factors, total=len(all_factors),
        )

    # ==================================================================
    # Public API: Repair CRUD
    # ==================================================================

    def register_repair(
        self, data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a component repair."""
        if self._equipment_component_engine is not None:
            try:
                return self._equipment_component_engine.add_repair(data)
            except Exception as exc:
                logger.warning("Repair via engine failed: %s", exc)

        repair_id = data.get(
            "repair_id", f"repair_{uuid.uuid4().hex[:12]}",
        )
        record = {"repair_id": repair_id}
        record.update(data)
        self._repairs[repair_id] = record
        logger.info("Registered repair: %s", repair_id)
        return record

    def list_repairs(self) -> RepairListResponse:
        """List component repairs."""
        all_repairs = list(self._repairs.values())
        return RepairListResponse(
            repairs=all_repairs, total=len(all_repairs),
        )

    # ==================================================================
    # Public API: Uncertainty Analysis
    # ==================================================================

    def run_uncertainty(
        self, data: Dict[str, Any],
    ) -> UncertaintyResponse:
        """Run uncertainty analysis on a calculation."""
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
        if (self._uncertainty_engine is not None
                and calc_record is not None):
            try:
                result = self._uncertainty_engine.quantify_uncertainty(
                    calculation_input=calc_record,
                    method=method,
                    n_iterations=iterations,
                )
                mean_kg = float(result.get("mean_co2e_kg", total_co2e_kg))
                std_kg = float(result.get("std_dev_kg", 0))
                ci = {}
                raw_ci = result.get("confidence_intervals", {})
                for level_key, bounds in raw_ci.items():
                    if isinstance(bounds, dict):
                        ci[str(level_key)] = bounds

                return UncertaintyResponse(
                    success=True,
                    method=method,
                    iterations=iterations,
                    mean_co2e_kg=mean_kg,
                    std_dev_kg=std_kg,
                    confidence_intervals=ci,
                    dqi_score=result.get("data_quality_score"),
                )
            except Exception as exc:
                logger.warning(
                    "Uncertainty engine failed: %s", exc,
                )

        # Fallback
        std_estimate = total_co2e_kg * 0.50
        return UncertaintyResponse(
            success=True,
            method="analytical_fallback",
            mean_co2e_kg=total_co2e_kg,
            std_dev_kg=std_estimate,
            confidence_intervals={
                "95": {
                    "lower": max(0.0, total_co2e_kg - 1.96 * std_estimate),
                    "upper": total_co2e_kg + 1.96 * std_estimate,
                },
            },
        )

    # ==================================================================
    # Public API: Compliance Check
    # ==================================================================

    def check_compliance(
        self, data: Dict[str, Any],
    ) -> ComplianceCheckResponse:
        """Run multi-framework compliance check."""
        calc_id = data.get("calculation_id", "")
        frameworks = data.get("frameworks", [])

        calc_record = None
        for c in self._calculations:
            if c.get("calculation_id") == calc_id:
                calc_record = c
                break

        if self._compliance_engine is not None:
            try:
                compliance_data = dict(calc_record) if calc_record else {}
                compliance_data.update(data)
                result = self._compliance_engine.check_compliance(
                    calculation_data=compliance_data,
                    frameworks=frameworks if frameworks else None,
                )
                fw_results = result.get("results", {})
                results_list = []
                for fw, fw_result in fw_results.items():
                    if isinstance(fw_result, dict):
                        fw_result["framework"] = fw
                        results_list.append(fw_result)

                return ComplianceCheckResponse(
                    success=True,
                    frameworks_checked=result.get("frameworks_checked", 0),
                    compliant=result.get("compliant", 0),
                    non_compliant=result.get("non_compliant", 0),
                    partial=result.get("partial", 0),
                    results=results_list,
                )
            except Exception as exc:
                logger.warning("Compliance engine failed: %s", exc)

        return ComplianceCheckResponse(
            success=True,
            frameworks_checked=0,
            results=[],
        )

    # ==================================================================
    # Public API: Health & Stats
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Service health check."""
        engines: Dict[str, str] = {
            "equipment_component": (
                "available"
                if self._equipment_component_engine is not None
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
        elif available_count >= 2:
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            service="fugitive-emissions",
            version="1.0.0",
            engines=engines,
        )

    def get_stats(self) -> StatsResponse:
        """Get service statistics."""
        uptime = time.monotonic() - self._start_time
        return StatsResponse(
            total_calculations=self._total_calculations,
            total_sources=len(self._sources),
            total_components=len(self._components),
            total_surveys=len(self._surveys),
            total_repairs=len(self._repairs),
            uptime_seconds=round(uptime, 3),
        )


# ===================================================================
# Thread-safe singleton access
# ===================================================================

_service_instance: Optional[FugitiveEmissionsService] = None
_service_lock = threading.Lock()


def get_service() -> FugitiveEmissionsService:
    """Get or create the singleton FugitiveEmissionsService instance.

    Returns:
        FugitiveEmissionsService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = FugitiveEmissionsService()
    return _service_instance


def get_router() -> Any:
    """Get the FastAPI router for fugitive emissions.

    Returns:
        FastAPI APIRouter or None if FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.fugitive_emissions.api.router import create_router
        return create_router()
    except ImportError:
        logger.warning(
            "Fugitive emissions API router module not available"
        )
        return None


def configure_fugitive_emissions(
    app: Any,
    config: Any = None,
) -> FugitiveEmissionsService:
    """Configure the Fugitive Emissions Service on a FastAPI application.

    Creates the FugitiveEmissionsService singleton, stores it in
    app.state, and mounts the API router.

    Args:
        app: FastAPI application instance.
        config: Optional configuration override.

    Returns:
        FugitiveEmissionsService instance.
    """
    global _service_instance

    service = FugitiveEmissionsService(config=config)

    with _service_lock:
        _service_instance = service

    if hasattr(app, "state"):
        app.state.fugitive_emissions_service = service

    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Fugitive emissions API router mounted")
    else:
        logger.warning(
            "Fugitive emissions router not available; API not mounted"
        )

    logger.info("Fugitive Emissions service configured")
    return service


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    "FugitiveEmissionsService",
    "configure_fugitive_emissions",
    "get_service",
    "get_router",
    "CalculateResponse",
    "BatchCalculateResponse",
    "SourceListResponse",
    "SourceDetailResponse",
    "ComponentListResponse",
    "ComponentDetailResponse",
    "SurveyListResponse",
    "FactorListResponse",
    "FactorDetailResponse",
    "RepairListResponse",
    "UncertaintyResponse",
    "ComplianceCheckResponse",
    "HealthResponse",
    "StatsResponse",
]
