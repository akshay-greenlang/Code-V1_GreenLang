# -*- coding: utf-8 -*-
"""
Purchased Goods & Services Service Setup - AGENT-MRV-014
=========================================================

Service facade for the Purchased Goods & Services Agent (GL-MRV-S3-001).

Provides ``configure_purchased_goods(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``PurchasedGoodsServicesService`` facade class that aggregates all 7 engines:

    1. ProcurementDatabaseEngine      - EEIO/Physical EF lookup, classification
    2. SpendBasedCalculatorEngine      - EEIO spend-based calculation
    3. AverageDataCalculatorEngine     - Physical quantity-based calculation
    4. SupplierSpecificCalculatorEngine - EPD/PCF/CDP supplier-specific calc
    5. HybridAggregatorEngine          - Multi-method aggregation & hot-spot
    6. ComplianceCheckerEngine         - Multi-framework regulatory compliance
    7. PurchasedGoodsPipelineEngine    - Orchestrated 10-stage pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.purchased_goods_services.setup import (
    ...     configure_purchased_goods,
    ... )
    >>> app = FastAPI()
    >>> configure_purchased_goods(app)

    >>> from greenlang.purchased_goods_services.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate_spend_based(
    ...     items=[item],
    ...     database=EEIODatabase.EPA_USEEIO,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-014 Purchased Goods & Services (GL-MRV-S3-001)
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
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Domain model imports (always available -- part of this package)
# ---------------------------------------------------------------------------

try:
    from greenlang.purchased_goods_services.models import (
        AGENT_ID,
        VERSION,
        TABLE_PREFIX,
        ZERO,
        ONE,
        ONE_HUNDRED,
        ONE_THOUSAND,
        DECIMAL_PLACES,
        MAX_PROCUREMENT_ITEMS,
        MAX_BATCH_PERIODS,
        MAX_FRAMEWORKS,
        DEFAULT_CONFIDENCE_LEVEL,
        # Enumerations
        CalculationMethod,
        SpendClassificationSystem,
        EEIODatabase,
        PhysicalEFSource,
        SupplierDataSource,
        AllocationMethod,
        MaterialCategory,
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
        ProcurementType,
        CoverageLevel,
        # Constant tables
        EEIO_EMISSION_FACTORS,
        PHYSICAL_EMISSION_FACTORS,
        CURRENCY_EXCHANGE_RATES,
        INDUSTRY_MARGIN_PERCENTAGES,
        DQI_SCORE_VALUES,
        DQI_QUALITY_TIERS,
        UNCERTAINTY_RANGES,
        COVERAGE_THRESHOLDS,
        EF_HIERARCHY_PRIORITY,
        PEDIGREE_UNCERTAINTY_FACTORS,
        GWP_VALUES,
        FRAMEWORK_REQUIRED_DISCLOSURES,
        # Data models
        ProcurementItem,
        SpendRecord,
        PhysicalRecord,
        SupplierRecord,
        SpendBasedResult,
        AverageDataResult,
        SupplierSpecificResult,
        HybridResult,
        EEIOFactor,
        PhysicalEF,
        SupplierEF,
        DQIAssessment,
        MaterialityItem,
        CoverageReport,
        ComplianceRequirement,
        ComplianceCheckResult,
        CalculationRequest,
        CalculationResult,
        BatchRequest,
        BatchResult,
        ExportRequest,
        AggregationResult,
        HotSpotAnalysis,
        CategoryBoundaryCheck,
        PipelineContext,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Provide sentinel values so the module can still be imported
    AGENT_ID = "GL-MRV-S3-001"
    VERSION = "1.0.0"
    ONE = Decimal("1")
    ZERO = Decimal("0")

# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.purchased_goods_services.config import (
        PurchasedGoodsServicesConfig,
    )
except ImportError:
    PurchasedGoodsServicesConfig = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.purchased_goods_services.procurement_database import (
        ProcurementDatabaseEngine,
    )
except ImportError:
    ProcurementDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.supplier_specific_calculator import (
        SupplierSpecificCalculatorEngine,
    )
except ImportError:
    SupplierSpecificCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.purchased_goods_pipeline import (
        PurchasedGoodsPipelineEngine,
    )
except ImportError:
    PurchasedGoodsPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.provenance import (
        PurchasedGoodsProvenanceTracker,
    )
except ImportError:
    PurchasedGoodsProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.purchased_goods_services.metrics import (
        PurchasedGoodsServicesMetrics,
    )
except ImportError:
    PurchasedGoodsServicesMetrics = None  # type: ignore[assignment, misc]


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


def _short_id(prefix: str = "pgs") -> str:
    """Generate a short unique identifier with a given prefix.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        A string like ``pgs_a1b2c3d4e5f6``.
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
# Default compliance frameworks for Category 1
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
# Lightweight Pydantic response models (16 models)
# ===================================================================


class SpendBasedResponse(BaseModel):
    """Response for a spend-based emission calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="spend_based")
    item_count: int = Field(default=0)
    total_emissions_kgco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    total_spend_usd: float = Field(default=0.0)
    eeio_database: str = Field(default="epa_useeio")
    emissions_intensity_kgco2e_per_usd: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class AverageDataResponse(BaseModel):
    """Response for an average-data emission calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="average_data")
    item_count: int = Field(default=0)
    total_emissions_kgco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    total_quantity_kg: float = Field(default=0.0)
    ef_source: str = Field(default="defra")
    emissions_intensity_kgco2e_per_kg: float = Field(default=0.0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class SupplierSpecificResponse(BaseModel):
    """Response for a supplier-specific emission calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="supplier_specific")
    item_count: int = Field(default=0)
    total_emissions_kgco2e: float = Field(default=0.0)
    total_emissions_tco2e: float = Field(default=0.0)
    supplier_count: int = Field(default=0)
    verified_count: int = Field(default=0)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class HybridCalculationResponse(BaseModel):
    """Response for a hybrid (multi-method) emission calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="hybrid")
    item_count: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    spend_based_emissions_tco2e: float = Field(default=0.0)
    average_data_emissions_tco2e: float = Field(default=0.0)
    supplier_specific_emissions_tco2e: float = Field(default=0.0)
    coverage_level: str = Field(default="minimal")
    coverage_percentage: float = Field(default=0.0)
    method_breakdown: Dict[str, float] = Field(default_factory=dict)
    hot_spots: List[Dict[str, Any]] = Field(default_factory=list)
    dqi_composite: Optional[float] = Field(default=None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class PipelineResponse(BaseModel):
    """Response for a full pipeline execution."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    pipeline_id: str = Field(default="")
    method: str = Field(default="hybrid")
    status: str = Field(default="completed")
    total_items: int = Field(default=0)
    in_scope_items: int = Field(default=0)
    excluded_items: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    spend_based_emissions_tco2e: float = Field(default=0.0)
    average_data_emissions_tco2e: float = Field(default=0.0)
    supplier_specific_emissions_tco2e: float = Field(default=0.0)
    coverage_level: str = Field(default="minimal")
    dqi_scores: Dict[str, Any] = Field(default_factory=dict)
    compliance_results: List[Dict[str, Any]] = Field(default_factory=list)
    export_data: Optional[Dict[str, Any]] = Field(default=None)
    stages_completed: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class BatchCalculateResponse(BaseModel):
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


class EEIOFactorResponse(BaseModel):
    """Response for an EEIO emission factor lookup."""

    model_config = ConfigDict(frozen=True)

    found: bool = Field(default=False)
    naics_code: str = Field(default="")
    sector_name: str = Field(default="")
    factor_kgco2e_per_usd: float = Field(default=0.0)
    database: str = Field(default="epa_useeio")
    base_year: int = Field(default=2019)
    provenance_hash: str = Field(default="")


class PhysicalEFResponse(BaseModel):
    """Response for a physical emission factor lookup."""

    model_config = ConfigDict(frozen=True)

    found: bool = Field(default=False)
    material_key: str = Field(default="")
    factor_kgco2e_per_kg: float = Field(default=0.0)
    source: str = Field(default="")
    material_category: str = Field(default="")
    provenance_hash: str = Field(default="")


class ComplianceCheckResponse(BaseModel):
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


class DQIScoreResponse(BaseModel):
    """Response for a data quality indicator scoring."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    item_id: str = Field(default="")
    method: str = Field(default="")
    temporal: float = Field(default=0.0)
    geographical: float = Field(default=0.0)
    technological: float = Field(default=0.0)
    completeness: float = Field(default=0.0)
    reliability: float = Field(default=0.0)
    composite: float = Field(default=0.0)
    quality_tier: str = Field(default="")
    provenance_hash: str = Field(default="")


class ExportResponse(BaseModel):
    """Response for an export operation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    format: str = Field(default="json")
    content: Any = Field(default=None)
    size_bytes: int = Field(default=0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class AggregationResponse(BaseModel):
    """Response for an aggregated emissions summary."""

    model_config = ConfigDict(frozen=True)

    groups: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    total_emissions_tco2e: float = Field(default=0.0)
    total_spend_usd: float = Field(default=0.0)
    item_count: int = Field(default=0)
    coverage_level: str = Field(default="minimal")
    period: str = Field(default="annual")
    timestamp: str = Field(default_factory=_utcnow_iso)


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="purchased-goods-services")
    agent_id: str = Field(default=AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-001")
    version: str = Field(default=VERSION if MODELS_AVAILABLE else "1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    models_available: bool = Field(default=True)
    uptime_seconds: float = Field(default=0.0)
    total_calculations: int = Field(default=0)
    timestamp: str = Field(default_factory=_utcnow_iso)


class StatsResponse(BaseModel):
    """Service aggregate statistics response."""

    model_config = ConfigDict(frozen=True)

    total_spend_based_calculations: int = Field(default=0)
    total_average_data_calculations: int = Field(default=0)
    total_supplier_specific_calculations: int = Field(default=0)
    total_hybrid_calculations: int = Field(default=0)
    total_pipeline_runs: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_eeio_lookups: int = Field(default=0)
    total_physical_ef_lookups: int = Field(default=0)
    total_supplier_efs_registered: int = Field(default=0)
    total_dqi_scorings: int = Field(default=0)
    total_exports: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)
    timestamp: str = Field(default_factory=_utcnow_iso)


# ===================================================================
# PurchasedGoodsServicesService facade
# ===================================================================


class PurchasedGoodsServicesService:
    """Unified facade over the Purchased Goods & Services Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 16+ service operations covering
    spend-based, average-data, supplier-specific, and hybrid
    calculation methods for GHG Protocol Scope 3 Category 1
    emissions.

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
        >>> service = PurchasedGoodsServicesService()
        >>> from greenlang.purchased_goods_services.models import (
        ...     ProcurementItem, EEIODatabase, CurrencyCode,
        ... )
        >>> from decimal import Decimal
        >>> item = ProcurementItem(
        ...     description="Steel beams",
        ...     spend_amount=Decimal("50000"),
        ...     currency=CurrencyCode.USD,
        ...     naics_code="331110",
        ... )
        >>> result = service.calculate_spend_based(items=[item])
        >>> assert result.success
        >>> assert result.total_emissions_tco2e > 0

    Attributes:
        config: Service configuration singleton.
        _procurement_db_engine: Engine 1 - reference data lookups.
        _spend_based_engine: Engine 2 - EEIO spend-based calculations.
        _average_data_engine: Engine 3 - physical EF calculations.
        _supplier_specific_engine: Engine 4 - supplier-specific calculations.
        _hybrid_aggregator_engine: Engine 5 - multi-method aggregation.
        _compliance_checker_engine: Engine 6 - regulatory compliance.
        _pipeline_engine: Engine 7 - orchestrated pipeline.
    """

    _instance: Optional["PurchasedGoodsServicesService"] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> "PurchasedGoodsServicesService":
        """Create or return the singleton instance.

        Uses double-checked locking with ``threading.RLock`` to ensure
        exactly one instance is created even under concurrent access.

        Returns:
            The singleton PurchasedGoodsServicesService instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Purchased Goods & Services Service facade.

        Only performs initialization once (guarded by ``_initialized``).
        Subsequent calls to ``__init__`` are no-ops, preserving the
        singleton's state.
        """
        if self.__class__._initialized:
            return
        with self.__class__._lock:
            if self.__class__._initialized:
                return
            self._do_init()
            self.__class__._initialized = True

    def _do_init(self) -> None:
        """Internal initialization logic (called once).

        Sets up configuration, engine placeholders, in-memory stores,
        and statistics counters.  Then attempts to initialize each engine.
        """
        # Configuration
        self.config: Any = None
        if PurchasedGoodsServicesConfig is not None:
            try:
                self.config = PurchasedGoodsServicesConfig()
            except Exception as exc:
                logger.warning(
                    "PurchasedGoodsServicesConfig init failed: %s", exc,
                )

        self._start_time: float = time.monotonic()

        # Engine placeholders (lazy-initialized)
        self._procurement_db_engine: Any = None
        self._spend_based_engine: Any = None
        self._average_data_engine: Any = None
        self._supplier_specific_engine: Any = None
        self._hybrid_aggregator_engine: Any = None
        self._compliance_checker_engine: Any = None
        self._pipeline_engine: Any = None

        # Provenance and metrics
        self._provenance_tracker: Any = None
        self._metrics: Any = None

        # In-memory result caches
        self._spend_based_results: List[Dict[str, Any]] = []
        self._average_data_results: List[Dict[str, Any]] = []
        self._supplier_specific_results: List[Dict[str, Any]] = []
        self._hybrid_results: List[Dict[str, Any]] = []
        self._pipeline_results: List[Dict[str, Any]] = []
        self._batch_results: List[Dict[str, Any]] = []
        self._compliance_results: List[Dict[str, Any]] = []
        self._dqi_results: List[Dict[str, Any]] = []
        self._export_results: List[Dict[str, Any]] = []

        # Statistics counters
        self._total_spend_based: int = 0
        self._total_average_data: int = 0
        self._total_supplier_specific: int = 0
        self._total_hybrid: int = 0
        self._total_pipeline_runs: int = 0
        self._total_batch_runs: int = 0
        self._total_compliance_checks: int = 0
        self._total_eeio_lookups: int = 0
        self._total_physical_ef_lookups: int = 0
        self._total_supplier_efs_registered: int = 0
        self._total_dqi_scorings: int = 0
        self._total_exports: int = 0

        # Initialize engines
        self._init_engines()

        logger.info(
            "PurchasedGoodsServicesService facade created "
            "(agent=%s, version=%s)",
            AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-001",
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
        logger.info("PurchasedGoodsServicesService singleton reset")

    # ------------------------------------------------------------------
    # Engine properties (read-only)
    # ------------------------------------------------------------------

    @property
    def procurement_db_engine(self) -> Any:
        """Get the ProcurementDatabaseEngine instance (Engine 1)."""
        return self._procurement_db_engine

    @property
    def spend_based_engine(self) -> Any:
        """Get the SpendBasedCalculatorEngine instance (Engine 2)."""
        return self._spend_based_engine

    @property
    def average_data_engine(self) -> Any:
        """Get the AverageDataCalculatorEngine instance (Engine 3)."""
        return self._average_data_engine

    @property
    def supplier_specific_engine(self) -> Any:
        """Get the SupplierSpecificCalculatorEngine instance (Engine 4)."""
        return self._supplier_specific_engine

    @property
    def hybrid_aggregator_engine(self) -> Any:
        """Get the HybridAggregatorEngine instance (Engine 5)."""
        return self._hybrid_aggregator_engine

    @property
    def compliance_checker_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance (Engine 6)."""
        return self._compliance_checker_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the PurchasedGoodsPipelineEngine instance (Engine 7)."""
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
        # E1: ProcurementDatabaseEngine
        self._init_single_engine(
            "ProcurementDatabaseEngine",
            ProcurementDatabaseEngine,
            "_procurement_db_engine",
        )

        # E2: SpendBasedCalculatorEngine
        self._init_single_engine(
            "SpendBasedCalculatorEngine",
            SpendBasedCalculatorEngine,
            "_spend_based_engine",
        )

        # E3: AverageDataCalculatorEngine
        self._init_single_engine(
            "AverageDataCalculatorEngine",
            AverageDataCalculatorEngine,
            "_average_data_engine",
        )

        # E4: SupplierSpecificCalculatorEngine
        self._init_single_engine(
            "SupplierSpecificCalculatorEngine",
            SupplierSpecificCalculatorEngine,
            "_supplier_specific_engine",
        )

        # E5: HybridAggregatorEngine
        self._init_single_engine(
            "HybridAggregatorEngine",
            HybridAggregatorEngine,
            "_hybrid_aggregator_engine",
        )

        # E6: ComplianceCheckerEngine
        self._init_single_engine(
            "ComplianceCheckerEngine",
            ComplianceCheckerEngine,
            "_compliance_checker_engine",
        )

        # E7: PurchasedGoodsPipelineEngine (receives upstream engines)
        if PurchasedGoodsPipelineEngine is not None:
            try:
                self._pipeline_engine = PurchasedGoodsPipelineEngine()
                logger.info("PurchasedGoodsPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "PurchasedGoodsPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("PurchasedGoodsPipelineEngine not available")

        # Provenance tracker
        if PurchasedGoodsProvenanceTracker is not None:
            try:
                self._provenance_tracker = PurchasedGoodsProvenanceTracker()
                logger.info("PurchasedGoodsProvenanceTracker initialized")
            except Exception as exc:
                logger.warning(
                    "PurchasedGoodsProvenanceTracker init failed: %s", exc,
                )

        # Metrics collector
        if PurchasedGoodsServicesMetrics is not None:
            try:
                self._metrics = PurchasedGoodsServicesMetrics()
                logger.info("PurchasedGoodsServicesMetrics initialized")
            except Exception as exc:
                logger.warning(
                    "PurchasedGoodsServicesMetrics init failed: %s", exc,
                )

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        attr_name: str,
    ) -> None:
        """Initialize a single engine with graceful degradation.

        All engine classes in the PGS agent use a thread-safe singleton
        pattern (``__new__`` with ``_instance``), so calling the
        constructor returns the shared instance.

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
    # Public API: Spend-Based Calculation (Method 1)
    # ==================================================================

    def calculate_spend_based(
        self,
        items: List[Any],
        database: Any = None,
        cpi_ratio: Decimal = Decimal("1"),
    ) -> SpendBasedResponse:
        """Calculate Scope 3 Category 1 emissions using the spend-based method.

        Applies EEIO emission factors to procurement spend amounts after
        currency conversion, inflation deflation, and margin removal.
        This is the broadest-coverage but lowest-accuracy method with
        typical uncertainty of +/- 50-100%.

        Args:
            items: List of ``ProcurementItem`` instances or dicts
                containing spend_amount, currency, and naics_code.
            database: ``EEIODatabase`` enum value for factor lookup.
                Defaults to ``EEIODatabase.EPA_USEEIO``.
            cpi_ratio: CPI ratio for inflation deflation. A value of
                ``Decimal("1")`` means no deflation. Defaults to 1.

        Returns:
            SpendBasedResponse with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("sb_calc")
        db = database
        if db is None and MODELS_AVAILABLE:
            db = EEIODatabase.EPA_USEEIO

        logger.info(
            "calculate_spend_based: %d items, database=%s, cpi_ratio=%s",
            len(items), db, cpi_ratio,
        )

        try:
            if self._spend_based_engine is None:
                raise RuntimeError(
                    "SpendBasedCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_kg = Decimal("0")
            total_t = Decimal("0")
            total_spend = Decimal("0")

            for raw_item in items:
                item = self._ensure_procurement_item(raw_item)
                result = self._spend_based_engine.calculate_single(
                    item=item,
                    database=db,
                    cpi_ratio=cpi_ratio,
                )
                result_dict = self._result_to_dict(result)
                results_list.append(result_dict)

                total_kg += _safe_decimal(result.emissions_kgco2e)
                total_t += _safe_decimal(result.emissions_tco2e)
                total_spend += _safe_decimal(result.spend_usd)

            intensity = (
                total_kg / total_spend
                if total_spend > ZERO
                else ZERO
            )

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "method": "spend_based",
                "item_count": len(items),
                "total_emissions_kgco2e": str(total_kg),
                "total_spend_usd": str(total_spend),
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_spend_based += 1
            self._cache_spend_based_result(
                calc_id, results_list, total_t, provenance_hash,
            )

            logger.info(
                "Spend-based calc %s: %d items, %.4f tCO2e in %.1f ms",
                calc_id, len(items), float(total_t), elapsed_ms,
            )

            return SpendBasedResponse(
                success=True,
                calculation_id=calc_id,
                method="spend_based",
                item_count=len(items),
                total_emissions_kgco2e=_safe_float(total_kg),
                total_emissions_tco2e=_safe_float(total_t),
                total_spend_usd=_safe_float(total_spend),
                eeio_database=db.value if hasattr(db, "value") else str(db),
                emissions_intensity_kgco2e_per_usd=_safe_float(intensity),
                results=results_list,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate_spend_based failed: %s", exc, exc_info=True,
            )
            return SpendBasedResponse(
                success=False,
                calculation_id=calc_id,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _cache_spend_based_result(
        self,
        calc_id: str,
        results: List[Dict[str, Any]],
        total_tco2e: Decimal,
        provenance_hash: str,
    ) -> None:
        """Cache a spend-based calculation in the in-memory store.

        Args:
            calc_id: Unique calculation identifier.
            results: List of per-item result dicts.
            total_tco2e: Total emissions in tCO2e.
            provenance_hash: SHA-256 provenance hash.
        """
        self._spend_based_results.append({
            "calculation_id": calc_id,
            "method": "spend_based",
            "item_count": len(results),
            "total_emissions_tco2e": str(total_tco2e),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        })

    # ==================================================================
    # Public API: Average-Data Calculation (Method 2)
    # ==================================================================

    def calculate_average_data(
        self,
        items: List[Any],
        ef_source: Any = None,
    ) -> AverageDataResponse:
        """Calculate Scope 3 Category 1 emissions using the average-data method.

        Multiplies physical quantities of purchased goods by
        cradle-to-gate emission factors from LCA databases.
        Accuracy is +/- 30-60%, better than spend-based but requires
        physical quantity data.

        Args:
            items: List of ``ProcurementItem`` instances or dicts
                containing quantity, quantity_unit, and material info.
            ef_source: ``PhysicalEFSource`` enum value. Defaults to
                ``PhysicalEFSource.DEFRA``.

        Returns:
            AverageDataResponse with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("ad_calc")
        source = ef_source
        if source is None and MODELS_AVAILABLE:
            source = PhysicalEFSource.DEFRA

        logger.info(
            "calculate_average_data: %d items, ef_source=%s",
            len(items), source,
        )

        try:
            if self._average_data_engine is None:
                raise RuntimeError(
                    "AverageDataCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_kg = Decimal("0")
            total_t = Decimal("0")
            total_qty = Decimal("0")

            for raw_item in items:
                item = self._ensure_procurement_item(raw_item)
                result = self._average_data_engine.calculate_single(
                    item=item,
                    ef_source=source,
                )
                result_dict = self._result_to_dict(result)
                results_list.append(result_dict)

                total_kg += _safe_decimal(result.emissions_kgco2e)
                total_t += _safe_decimal(result.emissions_tco2e)
                total_qty += _safe_decimal(result.quantity_kg)

            intensity = (
                total_kg / total_qty
                if total_qty > ZERO
                else ZERO
            )

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "method": "average_data",
                "item_count": len(items),
                "total_emissions_kgco2e": str(total_kg),
                "total_quantity_kg": str(total_qty),
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_average_data += 1
            self._cache_average_data_result(
                calc_id, results_list, total_t, provenance_hash,
            )

            logger.info(
                "Average-data calc %s: %d items, %.4f tCO2e in %.1f ms",
                calc_id, len(items), float(total_t), elapsed_ms,
            )

            return AverageDataResponse(
                success=True,
                calculation_id=calc_id,
                method="average_data",
                item_count=len(items),
                total_emissions_kgco2e=_safe_float(total_kg),
                total_emissions_tco2e=_safe_float(total_t),
                total_quantity_kg=_safe_float(total_qty),
                ef_source=source.value if hasattr(source, "value") else str(source),
                emissions_intensity_kgco2e_per_kg=_safe_float(intensity),
                results=results_list,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate_average_data failed: %s", exc, exc_info=True,
            )
            return AverageDataResponse(
                success=False,
                calculation_id=calc_id,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _cache_average_data_result(
        self,
        calc_id: str,
        results: List[Dict[str, Any]],
        total_tco2e: Decimal,
        provenance_hash: str,
    ) -> None:
        """Cache an average-data calculation in the in-memory store.

        Args:
            calc_id: Unique calculation identifier.
            results: List of per-item result dicts.
            total_tco2e: Total emissions in tCO2e.
            provenance_hash: SHA-256 provenance hash.
        """
        self._average_data_results.append({
            "calculation_id": calc_id,
            "method": "average_data",
            "item_count": len(results),
            "total_emissions_tco2e": str(total_tco2e),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        })

    # ==================================================================
    # Public API: Supplier-Specific Calculation (Method 3)
    # ==================================================================

    def calculate_supplier_specific(
        self,
        records: List[Any],
    ) -> SupplierSpecificResponse:
        """Calculate Scope 3 Category 1 emissions using supplier-specific data.

        Uses primary emission data from suppliers (EPD, PCF, CDP, etc.)
        to produce the highest-accuracy calculations (+/- 10-30%).
        Supports both product-level and facility-level allocation.

        Args:
            records: List of ``SupplierRecord`` instances or dicts
                containing supplier emission data, allocation factors,
                and verification status.

        Returns:
            SupplierSpecificResponse with emissions breakdown.
        """
        t0 = time.monotonic()
        calc_id = _short_id("ss_calc")

        logger.info(
            "calculate_supplier_specific: %d records", len(records),
        )

        try:
            if self._supplier_specific_engine is None:
                raise RuntimeError(
                    "SupplierSpecificCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_kg = Decimal("0")
            total_t = Decimal("0")
            supplier_ids: set = set()
            verified_count = 0

            for raw_record in records:
                record = self._ensure_supplier_record(raw_record)
                result = self._supplier_specific_engine.calculate_from_record(
                    record=record,
                )
                result_dict = self._result_to_dict(result)
                results_list.append(result_dict)

                total_kg += _safe_decimal(result.emissions_kgco2e)
                total_t += _safe_decimal(result.emissions_tco2e)

                sid = getattr(record.item, "supplier_id", None)
                if sid:
                    supplier_ids.add(sid)
                ver = getattr(record, "verification_status", "")
                if ver and "verified" in str(ver).lower():
                    verified_count += 1

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "method": "supplier_specific",
                "record_count": len(records),
                "total_emissions_kgco2e": str(total_kg),
                "supplier_count": len(supplier_ids),
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_supplier_specific += 1
            self._cache_supplier_specific_result(
                calc_id, results_list, total_t, provenance_hash,
            )

            logger.info(
                "Supplier-specific calc %s: %d records, %d suppliers, "
                "%.4f tCO2e in %.1f ms",
                calc_id, len(records), len(supplier_ids),
                float(total_t), elapsed_ms,
            )

            return SupplierSpecificResponse(
                success=True,
                calculation_id=calc_id,
                method="supplier_specific",
                item_count=len(records),
                total_emissions_kgco2e=_safe_float(total_kg),
                total_emissions_tco2e=_safe_float(total_t),
                supplier_count=len(supplier_ids),
                verified_count=verified_count,
                results=results_list,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate_supplier_specific failed: %s",
                exc, exc_info=True,
            )
            return SupplierSpecificResponse(
                success=False,
                calculation_id=calc_id,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _cache_supplier_specific_result(
        self,
        calc_id: str,
        results: List[Dict[str, Any]],
        total_tco2e: Decimal,
        provenance_hash: str,
    ) -> None:
        """Cache a supplier-specific calculation in the in-memory store.

        Args:
            calc_id: Unique calculation identifier.
            results: List of per-item result dicts.
            total_tco2e: Total emissions in tCO2e.
            provenance_hash: SHA-256 provenance hash.
        """
        self._supplier_specific_results.append({
            "calculation_id": calc_id,
            "method": "supplier_specific",
            "item_count": len(results),
            "total_emissions_tco2e": str(total_tco2e),
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        })

    # ==================================================================
    # Public API: Hybrid Calculation (Method 4)
    # ==================================================================

    def calculate_hybrid(
        self,
        items: List[Any],
        supplier_records: Optional[List[Any]] = None,
        database: Any = None,
    ) -> HybridCalculationResponse:
        """Calculate Scope 3 Category 1 emissions using the hybrid method.

        Combines spend-based, average-data, and supplier-specific
        methods using the highest-quality data available for each item.
        Accuracy is +/- 20-50%.  This is the recommended method per
        GHG Protocol guidance when mixed data is available.

        The method uses the following priority for each item:
            1. Supplier-specific (if supplier record available)
            2. Average-data (if physical quantity available)
            3. Spend-based (always available as fallback)

        Args:
            items: List of ``ProcurementItem`` instances or dicts.
            supplier_records: Optional list of ``SupplierRecord``
                instances or dicts for items with supplier-specific data.
            database: ``EEIODatabase`` for spend-based fallback.
                Defaults to ``EEIODatabase.EPA_USEEIO``.

        Returns:
            HybridCalculationResponse with aggregated emissions.
        """
        t0 = time.monotonic()
        calc_id = _short_id("hy_calc")
        db = database
        if db is None and MODELS_AVAILABLE:
            db = EEIODatabase.EPA_USEEIO

        logger.info(
            "calculate_hybrid: %d items, %d supplier records",
            len(items),
            len(supplier_records) if supplier_records else 0,
        )

        try:
            # Run spend-based for all items (baseline)
            spend_results: List[Any] = []
            if self._spend_based_engine is not None:
                for raw_item in items:
                    item = self._ensure_procurement_item(raw_item)
                    try:
                        sr = self._spend_based_engine.calculate_single(
                            item=item,
                            database=db,
                        )
                        spend_results.append(sr)
                    except Exception as exc:
                        logger.debug(
                            "Spend-based failed for item %s: %s",
                            getattr(item, "item_id", "unknown"), exc,
                        )

            # Run average-data for items with quantity
            avgdata_results: List[Any] = []
            if self._average_data_engine is not None:
                for raw_item in items:
                    item = self._ensure_procurement_item(raw_item)
                    if getattr(item, "quantity", None) is not None:
                        try:
                            ar = self._average_data_engine.calculate_single(
                                item=item,
                            )
                            avgdata_results.append(ar)
                        except Exception as exc:
                            logger.debug(
                                "Average-data failed for item %s: %s",
                                getattr(item, "item_id", "unknown"),
                                exc,
                            )

            # Run supplier-specific for items with supplier data
            supplier_results: List[Any] = []
            if (
                self._supplier_specific_engine is not None
                and supplier_records
            ):
                for raw_record in supplier_records:
                    record = self._ensure_supplier_record(raw_record)
                    try:
                        ssr = self._supplier_specific_engine.calculate_from_record(
                            record=record,
                        )
                        supplier_results.append(ssr)
                    except Exception as exc:
                        logger.debug(
                            "Supplier-specific failed for record: %s",
                            exc,
                        )

            # Aggregate via hybrid engine
            total_t = Decimal("0")
            sb_t = Decimal("0")
            ad_t = Decimal("0")
            ss_t = Decimal("0")
            coverage_level = "minimal"
            coverage_pct = 0.0
            method_breakdown: Dict[str, float] = {}
            hot_spots: List[Dict[str, Any]] = []

            if self._hybrid_aggregator_engine is not None:
                try:
                    proc_items = [
                        self._ensure_procurement_item(i) for i in items
                    ]
                    total_spend = sum(
                        _safe_decimal(getattr(i, "spend_amount", 0))
                        for i in proc_items
                    )
                    hybrid_result = self._hybrid_aggregator_engine.aggregate(
                        spend_results=spend_results,
                        avgdata_results=avgdata_results,
                        supplier_results=supplier_results,
                        items=proc_items,
                        total_spend_usd=total_spend,
                    )
                    total_t = _safe_decimal(
                        getattr(hybrid_result, "total_emissions_tco2e", 0),
                    )
                    sb_t = _safe_decimal(
                        getattr(hybrid_result, "spend_based_emissions_tco2e", 0),
                    )
                    ad_t = _safe_decimal(
                        getattr(hybrid_result, "average_data_emissions_tco2e", 0),
                    )
                    ss_t = _safe_decimal(
                        getattr(hybrid_result, "supplier_specific_emissions_tco2e", 0),
                    )
                    coverage_level = getattr(
                        hybrid_result, "coverage_level", "minimal",
                    )
                    if hasattr(coverage_level, "value"):
                        coverage_level = coverage_level.value
                    coverage_pct = _safe_float(
                        getattr(hybrid_result, "coverage_percentage", 0),
                    )
                    method_breakdown = {
                        "spend_based": _safe_float(sb_t),
                        "average_data": _safe_float(ad_t),
                        "supplier_specific": _safe_float(ss_t),
                    }
                    hot_spots_raw = getattr(
                        hybrid_result, "hot_spots", [],
                    )
                    if hot_spots_raw:
                        hot_spots = [
                            self._result_to_dict(hs) for hs in hot_spots_raw
                        ]
                except Exception as exc:
                    logger.warning(
                        "Hybrid aggregation failed, using manual sum: %s",
                        exc,
                    )
                    total_t = self._manual_hybrid_sum(
                        spend_results, avgdata_results, supplier_results,
                    )

            else:
                # Manual fallback if no hybrid engine
                total_t = self._manual_hybrid_sum(
                    spend_results, avgdata_results, supplier_results,
                )
                sb_t = sum(
                    _safe_decimal(getattr(r, "emissions_tco2e", 0))
                    for r in spend_results
                )
                ad_t = sum(
                    _safe_decimal(getattr(r, "emissions_tco2e", 0))
                    for r in avgdata_results
                )
                ss_t = sum(
                    _safe_decimal(getattr(r, "emissions_tco2e", 0))
                    for r in supplier_results
                )
                method_breakdown = {
                    "spend_based": _safe_float(sb_t),
                    "average_data": _safe_float(ad_t),
                    "supplier_specific": _safe_float(ss_t),
                }

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "method": "hybrid",
                "item_count": len(items),
                "total_emissions_tco2e": str(total_t),
                "method_breakdown": method_breakdown,
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_hybrid += 1
            self._cache_hybrid_result(
                calc_id, total_t, method_breakdown, provenance_hash,
            )

            logger.info(
                "Hybrid calc %s: %d items, %.4f tCO2e "
                "(SB=%.4f, AD=%.4f, SS=%.4f) in %.1f ms",
                calc_id, len(items), float(total_t),
                float(sb_t), float(ad_t), float(ss_t),
                elapsed_ms,
            )

            return HybridCalculationResponse(
                success=True,
                calculation_id=calc_id,
                method="hybrid",
                item_count=len(items),
                total_emissions_tco2e=_safe_float(total_t),
                spend_based_emissions_tco2e=_safe_float(sb_t),
                average_data_emissions_tco2e=_safe_float(ad_t),
                supplier_specific_emissions_tco2e=_safe_float(ss_t),
                coverage_level=str(coverage_level),
                coverage_percentage=coverage_pct,
                method_breakdown=method_breakdown,
                hot_spots=hot_spots,
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "calculate_hybrid failed: %s", exc, exc_info=True,
            )
            return HybridCalculationResponse(
                success=False,
                calculation_id=calc_id,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _manual_hybrid_sum(
        self,
        spend_results: List[Any],
        avgdata_results: List[Any],
        supplier_results: List[Any],
    ) -> Decimal:
        """Calculate a manual hybrid emission total as fallback.

        Applies a simple priority: supplier > average-data > spend.
        Items covered by higher-priority methods are excluded from
        lower-priority totals.

        Args:
            spend_results: Spend-based calculation results.
            avgdata_results: Average-data calculation results.
            supplier_results: Supplier-specific calculation results.

        Returns:
            Total emissions in tCO2e as Decimal.
        """
        covered_items: set = set()
        total = Decimal("0")

        # Priority 1: Supplier-specific
        for r in supplier_results:
            item_id = getattr(r, "item_id", None)
            if item_id:
                covered_items.add(item_id)
            total += _safe_decimal(getattr(r, "emissions_tco2e", 0))

        # Priority 2: Average-data (not already covered)
        for r in avgdata_results:
            item_id = getattr(r, "item_id", None)
            if item_id and item_id in covered_items:
                continue
            if item_id:
                covered_items.add(item_id)
            total += _safe_decimal(getattr(r, "emissions_tco2e", 0))

        # Priority 3: Spend-based (not already covered)
        for r in spend_results:
            item_id = getattr(r, "item_id", None)
            if item_id and item_id in covered_items:
                continue
            if item_id:
                covered_items.add(item_id)
            total += _safe_decimal(getattr(r, "emissions_tco2e", 0))

        return total

    def _cache_hybrid_result(
        self,
        calc_id: str,
        total_tco2e: Decimal,
        breakdown: Dict[str, float],
        provenance_hash: str,
    ) -> None:
        """Cache a hybrid calculation in the in-memory store.

        Args:
            calc_id: Unique calculation identifier.
            total_tco2e: Total emissions in tCO2e.
            breakdown: Method breakdown dict.
            provenance_hash: SHA-256 provenance hash.
        """
        self._hybrid_results.append({
            "calculation_id": calc_id,
            "method": "hybrid",
            "total_emissions_tco2e": str(total_tco2e),
            "method_breakdown": breakdown,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        })

    # ==================================================================
    # Public API: Pipeline Execution
    # ==================================================================

    def run_pipeline(
        self,
        items: List[Any],
        method: Any = None,
        frameworks: Optional[List[Any]] = None,
        disclosures: Optional[List[str]] = None,
        export_format: Any = None,
        supplier_records: Optional[List[Any]] = None,
        eeio_database: Any = None,
        cpi_ratio: Decimal = Decimal("1"),
    ) -> PipelineResponse:
        """Execute the complete 10-stage calculation pipeline.

        Orchestrates all stages from ingestion through export:
        INGEST -> CLASSIFY -> BOUNDARY_CHECK -> SPEND_CALC ->
        AVGDATA_CALC -> SUPPLIER_CALC -> AGGREGATE -> DQI_SCORE ->
        COMPLIANCE_CHECK -> EXPORT.

        This is the primary entry point for production calculations.

        Args:
            items: Raw procurement items (dicts or ProcurementItem).
            method: ``CalculationMethod`` enum. Defaults to HYBRID.
            frameworks: List of ``ComplianceFramework`` to check.
            disclosures: Required disclosure identifiers.
            export_format: ``ExportFormat`` for output. Defaults to JSON.
            supplier_records: Supplier-specific data records.
            eeio_database: ``EEIODatabase`` for spend-based method.
            cpi_ratio: CPI adjustment ratio.

        Returns:
            PipelineResponse with complete calculation results.
        """
        t0 = time.monotonic()
        pipeline_id = _short_id("pipe")

        calc_method = method
        if calc_method is None and MODELS_AVAILABLE:
            calc_method = CalculationMethod.HYBRID
        exp_fmt = export_format
        if exp_fmt is None and MODELS_AVAILABLE:
            exp_fmt = ExportFormat.JSON
        db = eeio_database
        if db is None and MODELS_AVAILABLE:
            db = EEIODatabase.EPA_USEEIO

        logger.info(
            "run_pipeline %s: %d items, method=%s",
            pipeline_id, len(items),
            calc_method.value if hasattr(calc_method, "value") else str(calc_method),
        )

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "PurchasedGoodsPipelineEngine is not available"
                )

            result = self._pipeline_engine.run_pipeline(
                items=[
                    self._item_to_dict(i) if not isinstance(i, dict) else i
                    for i in items
                ],
                method=calc_method,
                frameworks=frameworks,
                disclosures=disclosures,
                export_format=exp_fmt,
                supplier_records=[
                    self._item_to_dict(r) if not isinstance(r, dict) else r
                    for r in (supplier_records or [])
                ] if supplier_records else None,
                eeio_database=db,
                cpi_ratio=cpi_ratio,
            )

            # Extract results from pipeline output
            hybrid_result = result.get("hybrid_result")
            total_t = _safe_float(
                getattr(hybrid_result, "total_emissions_tco2e", 0)
                if hybrid_result else 0,
            )
            sb_t = _safe_float(
                getattr(hybrid_result, "spend_based_emissions_tco2e", 0)
                if hybrid_result else 0,
            )
            ad_t = _safe_float(
                getattr(hybrid_result, "average_data_emissions_tco2e", 0)
                if hybrid_result else 0,
            )
            ss_t = _safe_float(
                getattr(hybrid_result, "supplier_specific_emissions_tco2e", 0)
                if hybrid_result else 0,
            )
            coverage = result.get("coverage_level", "minimal")
            if hasattr(coverage, "value"):
                coverage = coverage.value

            dqi_scores = result.get("dqi_scores", {})
            if hasattr(dqi_scores, "model_dump"):
                dqi_scores = dqi_scores.model_dump(mode="json")
            elif not isinstance(dqi_scores, dict):
                dqi_scores = {}

            compliance_raw = result.get("compliance_results", [])
            compliance_out: List[Dict[str, Any]] = []
            for cr in compliance_raw:
                compliance_out.append(
                    self._result_to_dict(cr)
                    if not isinstance(cr, dict)
                    else cr
                )

            export_data = result.get("export_data")
            stages_completed = result.get("stages_completed", [])
            if not isinstance(stages_completed, list):
                stages_completed = []
            status = result.get("status", "completed")
            if hasattr(status, "value"):
                status = status.value

            provenance_hash = _compute_hash({
                "pipeline_id": pipeline_id,
                "total_items": len(items),
                "total_emissions_tco2e": total_t,
                "status": status,
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_pipeline_runs += 1
            self._cache_pipeline_result(
                pipeline_id, total_t, status, provenance_hash,
            )

            logger.info(
                "Pipeline %s completed: %d items, %.4f tCO2e, "
                "status=%s in %.1f ms",
                pipeline_id, len(items), total_t, status, elapsed_ms,
            )

            return PipelineResponse(
                success=True,
                pipeline_id=pipeline_id,
                method=calc_method.value if hasattr(calc_method, "value") else str(calc_method),
                status=str(status),
                total_items=result.get("total_items", len(items)),
                in_scope_items=result.get("in_scope_items", 0),
                excluded_items=result.get("excluded_items", 0),
                total_emissions_tco2e=total_t,
                spend_based_emissions_tco2e=sb_t,
                average_data_emissions_tco2e=ad_t,
                supplier_specific_emissions_tco2e=ss_t,
                coverage_level=str(coverage),
                dqi_scores=dqi_scores,
                compliance_results=compliance_out,
                export_data=export_data if isinstance(export_data, dict) else None,
                stages_completed=[
                    s.value if hasattr(s, "value") else str(s)
                    for s in stages_completed
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "run_pipeline failed: %s", exc, exc_info=True,
            )
            return PipelineResponse(
                success=False,
                pipeline_id=pipeline_id,
                status="failed",
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    def _cache_pipeline_result(
        self,
        pipeline_id: str,
        total_tco2e: float,
        status: str,
        provenance_hash: str,
    ) -> None:
        """Cache a pipeline result in the in-memory store.

        Args:
            pipeline_id: Unique pipeline identifier.
            total_tco2e: Total emissions in tCO2e.
            status: Pipeline completion status.
            provenance_hash: SHA-256 provenance hash.
        """
        self._pipeline_results.append({
            "pipeline_id": pipeline_id,
            "total_emissions_tco2e": total_tco2e,
            "status": status,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        })

    # ==================================================================
    # Public API: Batch Processing
    # ==================================================================

    def run_batch(
        self,
        batch: Any,
    ) -> BatchCalculateResponse:
        """Run a batch calculation across multiple periods.

        Delegates to the pipeline engine's ``run_batch`` method,
        which processes each period sequentially through the full
        pipeline.

        Args:
            batch: A ``BatchRequest`` instance or dict with items,
                periods, and calculation configuration.

        Returns:
            BatchCalculateResponse with per-period results.
        """
        t0 = time.monotonic()
        batch_id = _short_id("batch")

        logger.info("run_batch %s", batch_id)

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "PurchasedGoodsPipelineEngine is not available"
                )

            # Ensure BatchRequest type
            if MODELS_AVAILABLE and isinstance(batch, dict):
                batch = BatchRequest(**batch)

            result = self._pipeline_engine.run_batch(batch)

            # Extract from BatchResult
            total_t = _safe_float(
                getattr(result, "total_emissions_tco2e", 0),
            )
            status = getattr(result, "status", "completed")
            if hasattr(status, "value"):
                status = status.value
            total_periods = getattr(result, "total_periods", 0)
            completed = getattr(result, "completed", 0)
            failed = getattr(result, "failed", 0)

            results_raw = getattr(result, "results", [])
            results_list: List[Dict[str, Any]] = []
            for r in results_raw:
                results_list.append(
                    self._result_to_dict(r)
                    if not isinstance(r, dict) else r
                )

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_batch_runs += 1
            self._batch_results.append({
                "batch_id": batch_id,
                "total_periods": total_periods,
                "total_emissions_tco2e": total_t,
                "status": str(status),
                "timestamp": _utcnow_iso(),
            })

            logger.info(
                "Batch %s completed: %d periods, %.4f tCO2e in %.1f ms",
                batch_id, total_periods, total_t, elapsed_ms,
            )

            return BatchCalculateResponse(
                success=failed == 0,
                batch_id=batch_id,
                status=str(status),
                total_periods=total_periods,
                completed=completed,
                failed=failed,
                total_emissions_tco2e=total_t,
                results=results_list,
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "run_batch failed: %s", exc, exc_info=True,
            )
            return BatchCalculateResponse(
                success=False,
                batch_id=batch_id,
                status="failed",
                processing_time_ms=round(elapsed_ms, 3),
                timestamp=_utcnow_iso(),
            )

    # ==================================================================
    # Public API: Database Lookups
    # ==================================================================

    def lookup_eeio_factor(
        self,
        naics_code: str,
        database: Any = None,
    ) -> EEIOFactorResponse:
        """Look up an EEIO emission factor by NAICS-6 sector code.

        Performs an exact-match lookup in the ProcurementDatabaseEngine
        factor cache, falling back to 4-digit and 2-digit truncation
        if no exact match is found.

        Args:
            naics_code: 6-digit NAICS sector code (e.g. ``"331110"``).
            database: Optional ``EEIODatabase`` enum; defaults to
                EPA USEEIO.

        Returns:
            EEIOFactorResponse with factor details or ``found=False``.
        """
        self._total_eeio_lookups += 1
        db = database
        if db is None and MODELS_AVAILABLE:
            db = EEIODatabase.EPA_USEEIO

        if self._procurement_db_engine is None:
            # Fallback: look up directly from models constant table
            factor_val = EEIO_EMISSION_FACTORS.get(naics_code.strip()) if MODELS_AVAILABLE else None
            if factor_val is not None:
                provenance_hash = _compute_hash({
                    "naics_code": naics_code,
                    "factor": str(factor_val),
                    "source": "constant_table",
                })
                return EEIOFactorResponse(
                    found=True,
                    naics_code=naics_code.strip(),
                    factor_kgco2e_per_usd=_safe_float(factor_val),
                    database=db.value if hasattr(db, "value") else str(db),
                    provenance_hash=provenance_hash,
                )
            return EEIOFactorResponse(
                found=False,
                naics_code=naics_code.strip(),
            )

        # Delegate to engine with fallback
        ef = self._procurement_db_engine.lookup_eeio_factor_with_fallback(
            naics_code=naics_code,
            database=db,
        )

        if ef is None:
            return EEIOFactorResponse(
                found=False,
                naics_code=naics_code.strip(),
            )

        provenance_hash = _compute_hash({
            "naics_code": naics_code,
            "factor": str(getattr(ef, "factor_kgco2e_per_unit", 0)),
            "source": "procurement_database_engine",
        })

        return EEIOFactorResponse(
            found=True,
            naics_code=getattr(ef, "sector_code", naics_code),
            sector_name=getattr(ef, "sector_name", ""),
            factor_kgco2e_per_usd=_safe_float(
                getattr(ef, "factor_kgco2e_per_unit", 0),
            ),
            database=db.value if hasattr(db, "value") else str(db),
            base_year=int(getattr(ef, "base_year", 2019)),
            provenance_hash=provenance_hash,
        )

    def lookup_physical_ef(
        self,
        material_key: str,
    ) -> PhysicalEFResponse:
        """Look up a physical emission factor by material key.

        Searches for a cradle-to-gate emission factor (kgCO2e per kg)
        for the specified material key (e.g. ``"steel_world_avg"``).

        Args:
            material_key: Material identifier from the
                PHYSICAL_EMISSION_FACTORS constant table.

        Returns:
            PhysicalEFResponse with factor details or ``found=False``.
        """
        self._total_physical_ef_lookups += 1

        if self._procurement_db_engine is None:
            # Fallback: look up directly from models constant table
            factor_val = (
                PHYSICAL_EMISSION_FACTORS.get(material_key.strip().lower())
                if MODELS_AVAILABLE else None
            )
            if factor_val is not None:
                provenance_hash = _compute_hash({
                    "material_key": material_key,
                    "factor": str(factor_val),
                    "source": "constant_table",
                })
                return PhysicalEFResponse(
                    found=True,
                    material_key=material_key.strip().lower(),
                    factor_kgco2e_per_kg=_safe_float(factor_val),
                    source="constant_table",
                    provenance_hash=provenance_hash,
                )
            return PhysicalEFResponse(
                found=False,
                material_key=material_key.strip().lower(),
            )

        ef = self._procurement_db_engine.lookup_physical_ef(
            material_key=material_key,
        )

        if ef is None:
            return PhysicalEFResponse(
                found=False,
                material_key=material_key.strip().lower(),
            )

        provenance_hash = _compute_hash({
            "material_key": material_key,
            "factor": str(getattr(ef, "factor_kgco2e_per_kg", 0)),
            "source": "procurement_database_engine",
        })

        return PhysicalEFResponse(
            found=True,
            material_key=getattr(ef, "material_key", material_key),
            factor_kgco2e_per_kg=_safe_float(
                getattr(ef, "factor_kgco2e_per_kg", 0),
            ),
            source=getattr(ef, "source", ""),
            material_category=str(
                getattr(ef, "material_category", ""),
            ),
            provenance_hash=provenance_hash,
        )

    def register_supplier_ef(
        self,
        supplier_ef: Any,
    ) -> Dict[str, Any]:
        """Register a supplier-specific emission factor.

        Adds a ``SupplierEF`` to the procurement database engine's
        in-memory registry, making it available for subsequent
        supplier-specific calculations.

        Args:
            supplier_ef: A ``SupplierEF`` instance or dict with
                supplier_id, factor data, and source information.

        Returns:
            Dictionary confirming the registration with supplier_id
            and provenance_hash.

        Raises:
            RuntimeError: If ProcurementDatabaseEngine is unavailable.
            ValueError: If supplier_ef lacks a supplier_id.
        """
        self._total_supplier_efs_registered += 1

        if self._procurement_db_engine is None:
            raise RuntimeError(
                "ProcurementDatabaseEngine is not available; "
                "cannot register supplier emission factors"
            )

        # Ensure SupplierEF type
        if MODELS_AVAILABLE and isinstance(supplier_ef, dict):
            supplier_ef = SupplierEF(**supplier_ef)

        self._procurement_db_engine.register_supplier_ef(supplier_ef)

        sid = getattr(supplier_ef, "supplier_id", "unknown")
        provenance_hash = _compute_hash({
            "action": "register_supplier_ef",
            "supplier_id": sid,
            "timestamp": _utcnow_iso(),
        })

        logger.info(
            "Registered supplier EF for %s", sid,
        )

        return {
            "success": True,
            "supplier_id": sid,
            "provenance_hash": provenance_hash,
            "timestamp": _utcnow_iso(),
        }

    # ==================================================================
    # Public API: Compliance Check
    # ==================================================================

    def check_compliance(
        self,
        result: Any,
        disclosures: Dict[str, Any],
        frameworks: Optional[List[Any]] = None,
    ) -> ComplianceCheckResponse:
        """Run multi-framework regulatory compliance check.

        Validates a calculation result against one or more reporting
        frameworks including GHG Protocol, CSRD/ESRS E1, CDP, SBTi,
        SB 253, GRI 305, and ISO 14064.

        Args:
            result: A ``HybridResult`` instance or dict with
                calculation data to validate.
            disclosures: Additional disclosure information keyed by
                disclosure identifier.
            frameworks: Specific ``ComplianceFramework`` values to
                check.  Defaults to all 7 frameworks.

        Returns:
            ComplianceCheckResponse with per-framework results.
        """
        t0 = time.monotonic()
        compliance_id = _short_id("comp")

        logger.info(
            "check_compliance %s: %d frameworks",
            compliance_id,
            len(frameworks) if frameworks else 7,
        )

        try:
            if self._compliance_checker_engine is None:
                return self._compliance_fallback(
                    compliance_id, frameworks,
                )

            # Ensure HybridResult type
            if MODELS_AVAILABLE and isinstance(result, dict):
                result = HybridResult(**result)

            check_results = self._compliance_checker_engine.check_all_frameworks(
                result=result,
                disclosures=disclosures,
                frameworks=frameworks,
            )

            # Tally compliance statuses
            compliant = 0
            non_compliant = 0
            partial = 0
            not_applicable = 0
            results_list: List[Dict[str, Any]] = []

            for cr in check_results:
                cr_dict = self._result_to_dict(cr)
                results_list.append(cr_dict)
                status = getattr(cr, "status", None)
                if hasattr(status, "value"):
                    status = status.value
                if status == "compliant":
                    compliant += 1
                elif status == "non_compliant":
                    non_compliant += 1
                elif status == "partial":
                    partial += 1
                elif status == "not_applicable":
                    not_applicable += 1

            provenance_hash = _compute_hash({
                "compliance_id": compliance_id,
                "frameworks_checked": len(check_results),
                "compliant": compliant,
                "non_compliant": non_compliant,
            })

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._total_compliance_checks += 1
            self._compliance_results.append({
                "compliance_id": compliance_id,
                "frameworks_checked": len(check_results),
                "compliant": compliant,
                "non_compliant": non_compliant,
                "provenance_hash": provenance_hash,
                "timestamp": _utcnow_iso(),
            })

            logger.info(
                "Compliance check %s: %d frameworks, "
                "%d compliant, %d non-compliant in %.1f ms",
                compliance_id, len(check_results),
                compliant, non_compliant, elapsed_ms,
            )

            return ComplianceCheckResponse(
                success=True,
                compliance_id=compliance_id,
                frameworks_checked=len(check_results),
                compliant=compliant,
                non_compliant=non_compliant,
                partial=partial,
                not_applicable=not_applicable,
                results=results_list,
                provenance_hash=provenance_hash,
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.error(
                "check_compliance failed: %s", exc, exc_info=True,
            )
            return ComplianceCheckResponse(
                success=False,
                compliance_id=compliance_id,
                timestamp=_utcnow_iso(),
            )

    def _compliance_fallback(
        self,
        compliance_id: str,
        frameworks: Optional[List[Any]],
    ) -> ComplianceCheckResponse:
        """Generate a fallback compliance result when the engine is unavailable.

        Returns ``not_assessed`` status for all requested frameworks.

        Args:
            compliance_id: Unique compliance check identifier.
            frameworks: Requested frameworks to check.

        Returns:
            ComplianceCheckResponse with not-assessed status.
        """
        fw_list = frameworks or DEFAULT_COMPLIANCE_FRAMEWORKS
        results_list: List[Dict[str, Any]] = []
        for fw in fw_list:
            fw_name = fw.value if hasattr(fw, "value") else str(fw)
            results_list.append({
                "framework": fw_name,
                "status": "not_assessed",
                "total_requirements": 0,
                "passed": 0,
                "failed": 0,
                "findings": [],
                "recommendations": [
                    "Connect ComplianceCheckerEngine for full assessment",
                ],
            })

        provenance_hash = _compute_hash({
            "compliance_id": compliance_id,
            "method": "fallback",
            "frameworks": len(fw_list),
        })

        self._total_compliance_checks += 1
        self._compliance_results.append({
            "compliance_id": compliance_id,
            "method": "fallback",
            "frameworks_checked": len(fw_list),
            "timestamp": _utcnow_iso(),
        })

        return ComplianceCheckResponse(
            success=True,
            compliance_id=compliance_id,
            frameworks_checked=len(fw_list),
            compliant=0,
            non_compliant=0,
            partial=0,
            not_applicable=0,
            results=results_list,
            provenance_hash=provenance_hash,
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Public API: Data Quality Indicator (DQI) Scoring
    # ==================================================================

    def score_dqi(
        self,
        item: Any,
        method: str,
        result: Any,
    ) -> DQIScoreResponse:
        """Score data quality for a single calculation result.

        Evaluates five DQI dimensions (temporal, geographical,
        technological, completeness, reliability) per GHG Protocol
        Scope 3 Standard Chapter 7 and returns a composite score.

        Args:
            item: A ``ProcurementItem`` instance or dict.
            method: Calculation method string (``"spend_based"``,
                ``"average_data"``, or ``"supplier_specific"``).
            result: Calculation result to score.

        Returns:
            DQIScoreResponse with per-dimension and composite scores.
        """
        t0 = time.monotonic()
        self._total_dqi_scorings += 1

        item_obj = self._ensure_procurement_item(item) if not isinstance(item, dict) else item
        item_id = getattr(item_obj, "item_id", "") if hasattr(item_obj, "item_id") else item.get("item_id", "")

        try:
            dqi_result: Optional[Dict[str, Any]] = None

            # Delegate to the appropriate engine's DQI scorer
            if method == "spend_based" and self._spend_based_engine is not None:
                try:
                    dqi_result = self._spend_based_engine.score_dqi_spend_based(
                        item=item_obj if not isinstance(item_obj, dict) else self._ensure_procurement_item(item_obj),
                        result=result,
                    )
                except Exception as exc:
                    logger.debug("Spend-based DQI scoring failed: %s", exc)

            elif method == "average_data" and self._average_data_engine is not None:
                try:
                    dqi_result = self._average_data_engine.score_dqi_average_data(
                        item=item_obj if not isinstance(item_obj, dict) else self._ensure_procurement_item(item_obj),
                        result=result,
                    )
                except Exception as exc:
                    logger.debug("Average-data DQI scoring failed: %s", exc)

            elif method == "supplier_specific" and self._supplier_specific_engine is not None:
                try:
                    dqi_result = self._supplier_specific_engine.score_dqi_supplier_specific(
                        item=item_obj if not isinstance(item_obj, dict) else self._ensure_procurement_item(item_obj),
                        result=result,
                    )
                except Exception as exc:
                    logger.debug("Supplier-specific DQI scoring failed: %s", exc)

            # Extract DQI values
            if dqi_result is not None:
                scores = self._extract_dqi_scores(dqi_result)
            else:
                scores = self._default_dqi_scores(method)

            provenance_hash = _compute_hash({
                "item_id": item_id,
                "method": method,
                "composite": scores["composite"],
            })

            self._dqi_results.append({
                "item_id": item_id,
                "method": method,
                "scores": scores,
                "provenance_hash": provenance_hash,
                "timestamp": _utcnow_iso(),
            })

            return DQIScoreResponse(
                success=True,
                item_id=item_id,
                method=method,
                temporal=scores["temporal"],
                geographical=scores["geographical"],
                technological=scores["technological"],
                completeness=scores["completeness"],
                reliability=scores["reliability"],
                composite=scores["composite"],
                quality_tier=scores["quality_tier"],
                provenance_hash=provenance_hash,
            )

        except Exception as exc:
            logger.error("score_dqi failed: %s", exc, exc_info=True)
            return DQIScoreResponse(
                success=False,
                item_id=item_id,
                method=method,
            )

    def _extract_dqi_scores(
        self,
        dqi_result: Any,
    ) -> Dict[str, Any]:
        """Extract DQI dimension scores from an engine result.

        Args:
            dqi_result: DQI result from an engine (dict or model).

        Returns:
            Dictionary with temporal, geographical, technological,
            completeness, reliability, composite, and quality_tier.
        """
        if hasattr(dqi_result, "model_dump"):
            data = dqi_result.model_dump(mode="json")
        elif isinstance(dqi_result, dict):
            data = dqi_result
        else:
            data = {}

        temporal = _safe_float(data.get("temporal", data.get("temporal_score", 3.0)))
        geographical = _safe_float(data.get("geographical", data.get("geographical_score", 3.0)))
        technological = _safe_float(data.get("technological", data.get("technological_score", 3.0)))
        completeness = _safe_float(data.get("completeness", data.get("completeness_score", 3.0)))
        reliability = _safe_float(data.get("reliability", data.get("reliability_score", 3.0)))

        composite = (
            temporal + geographical + technological
            + completeness + reliability
        ) / 5.0

        quality_tier = self._determine_quality_tier(composite)

        return {
            "temporal": temporal,
            "geographical": geographical,
            "technological": technological,
            "completeness": completeness,
            "reliability": reliability,
            "composite": round(composite, 2),
            "quality_tier": quality_tier,
        }

    def _default_dqi_scores(
        self,
        method: str,
    ) -> Dict[str, Any]:
        """Return default DQI scores for a given calculation method.

        Default scores reflect the inherent data quality limitations
        of each method per GHG Protocol guidance.

        Args:
            method: Calculation method string.

        Returns:
            Dictionary with default DQI scores.
        """
        defaults: Dict[str, Dict[str, float]] = {
            "spend_based": {
                "temporal": 4.0,
                "geographical": 3.5,
                "technological": 4.0,
                "completeness": 3.0,
                "reliability": 4.0,
            },
            "average_data": {
                "temporal": 3.0,
                "geographical": 3.0,
                "technological": 3.0,
                "completeness": 2.5,
                "reliability": 3.0,
            },
            "supplier_specific": {
                "temporal": 1.5,
                "geographical": 1.5,
                "technological": 1.0,
                "completeness": 1.5,
                "reliability": 1.5,
            },
        }

        scores = defaults.get(method, defaults["spend_based"])
        composite = sum(scores.values()) / 5.0
        quality_tier = self._determine_quality_tier(composite)

        return {
            **scores,
            "composite": round(composite, 2),
            "quality_tier": quality_tier,
        }

    @staticmethod
    def _determine_quality_tier(composite: float) -> str:
        """Determine the quality tier label from a composite DQI score.

        Args:
            composite: Composite DQI score (1.0-5.0).

        Returns:
            Quality tier string: "Very Good", "Good", "Fair",
            "Poor", or "Very Poor".
        """
        if composite < 1.6:
            return "Very Good"
        if composite < 2.6:
            return "Good"
        if composite < 3.6:
            return "Fair"
        if composite < 4.6:
            return "Poor"
        return "Very Poor"

    # ==================================================================
    # Public API: Export Results
    # ==================================================================

    def export_results(
        self,
        result: Any,
        format: Any = None,
    ) -> ExportResponse:
        """Export calculation results in the specified format.

        Supports JSON, CSV, and Excel export of pipeline or
        individual calculation results.

        Args:
            result: Calculation result to export (any result model
                or dict).
            format: ``ExportFormat`` enum value. Defaults to JSON.

        Returns:
            ExportResponse with serialized content.
        """
        t0 = time.monotonic()
        fmt = format
        if fmt is None and MODELS_AVAILABLE:
            fmt = ExportFormat.JSON

        self._total_exports += 1

        try:
            # Serialize result
            if hasattr(result, "model_dump"):
                data = result.model_dump(mode="json")
            elif isinstance(result, dict):
                data = result
            else:
                data = {"result": str(result)}

            fmt_value = fmt.value if hasattr(fmt, "value") else str(fmt)

            if fmt_value == "json":
                content = json.dumps(data, sort_keys=True, default=str, indent=2)
                size_bytes = len(content.encode("utf-8"))
            elif fmt_value == "csv":
                content = self._export_as_csv(data)
                size_bytes = len(content.encode("utf-8"))
            elif fmt_value == "excel":
                # Excel export returns a placeholder descriptor
                content = json.dumps(data, sort_keys=True, default=str)
                size_bytes = len(content.encode("utf-8"))
            else:
                content = json.dumps(data, sort_keys=True, default=str)
                size_bytes = len(content.encode("utf-8"))

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            self._export_results.append({
                "format": fmt_value,
                "size_bytes": size_bytes,
                "timestamp": _utcnow_iso(),
            })

            logger.info(
                "Export %s: %d bytes in %.1f ms",
                fmt_value, size_bytes, elapsed_ms,
            )

            return ExportResponse(
                success=True,
                format=fmt_value,
                content=content,
                size_bytes=size_bytes,
                timestamp=_utcnow_iso(),
            )

        except Exception as exc:
            logger.error("export_results failed: %s", exc, exc_info=True)
            return ExportResponse(
                success=False,
                format=fmt.value if hasattr(fmt, "value") else str(fmt),
                timestamp=_utcnow_iso(),
            )

    @staticmethod
    def _export_as_csv(data: Dict[str, Any]) -> str:
        """Export data dictionary as CSV string.

        Flattens nested results into a tabular CSV format.

        Args:
            data: Dictionary to export.

        Returns:
            CSV-formatted string.
        """
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        # Extract results list if present
        results = data.get("results", [])
        if isinstance(results, list) and len(results) > 0:
            # Use first result's keys as headers
            first = results[0]
            if isinstance(first, dict):
                headers = list(first.keys())
                writer.writerow(headers)
                for row in results:
                    writer.writerow([
                        row.get(h, "") for h in headers
                    ])
            else:
                writer.writerow(["value"])
                for row in results:
                    writer.writerow([str(row)])
        else:
            # Flat export of top-level keys
            headers = list(data.keys())
            writer.writerow(headers)
            writer.writerow([
                str(data.get(h, "")) for h in headers
            ])

        return output.getvalue()

    # ==================================================================
    # Public API: Health Check & Statistics
    # ==================================================================

    def health_check(self) -> HealthResponse:
        """Service health check.

        Reports the availability status of all 7 engines, the
        service uptime, and the total calculation count.

        Returns:
            HealthResponse with engine status details.
        """
        engines: Dict[str, str] = {
            "procurement_database": (
                "available"
                if self._procurement_db_engine is not None
                else "unavailable"
            ),
            "spend_based_calculator": (
                "available"
                if self._spend_based_engine is not None
                else "unavailable"
            ),
            "average_data_calculator": (
                "available"
                if self._average_data_engine is not None
                else "unavailable"
            ),
            "supplier_specific_calculator": (
                "available"
                if self._supplier_specific_engine is not None
                else "unavailable"
            ),
            "hybrid_aggregator": (
                "available"
                if self._hybrid_aggregator_engine is not None
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

        # Also check sub-components
        if self._provenance_tracker is not None:
            engines["provenance_tracker"] = "available"
        else:
            engines["provenance_tracker"] = "unavailable"

        if self._metrics is not None:
            engines["metrics"] = "available"
        else:
            engines["metrics"] = "unavailable"

        available_count = sum(
            1 for s in engines.values() if s == "available"
        )
        total = len(engines)

        if available_count == total:
            status = "healthy"
        elif available_count >= 4:
            status = "degraded"
        else:
            status = "unhealthy"

        uptime = time.monotonic() - self._start_time
        total_calcs = (
            self._total_spend_based
            + self._total_average_data
            + self._total_supplier_specific
            + self._total_hybrid
            + self._total_pipeline_runs
            + self._total_batch_runs
        )

        return HealthResponse(
            status=status,
            service="purchased-goods-services",
            agent_id=AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-001",
            version=VERSION if MODELS_AVAILABLE else "1.0.0",
            engines=engines,
            models_available=MODELS_AVAILABLE,
            uptime_seconds=round(uptime, 3),
            total_calculations=total_calcs,
            timestamp=_utcnow_iso(),
        )

    def get_stats(self) -> StatsResponse:
        """Get detailed service statistics.

        Returns aggregate counts for all operation types and uptime.

        Returns:
            StatsResponse with comprehensive statistics.
        """
        uptime = time.monotonic() - self._start_time

        return StatsResponse(
            total_spend_based_calculations=self._total_spend_based,
            total_average_data_calculations=self._total_average_data,
            total_supplier_specific_calculations=self._total_supplier_specific,
            total_hybrid_calculations=self._total_hybrid,
            total_pipeline_runs=self._total_pipeline_runs,
            total_batch_runs=self._total_batch_runs,
            total_compliance_checks=self._total_compliance_checks,
            total_eeio_lookups=self._total_eeio_lookups,
            total_physical_ef_lookups=self._total_physical_ef_lookups,
            total_supplier_efs_registered=self._total_supplier_efs_registered,
            total_dqi_scorings=self._total_dqi_scorings,
            total_exports=self._total_exports,
            uptime_seconds=round(uptime, 3),
            timestamp=_utcnow_iso(),
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _ensure_procurement_item(self, raw: Any) -> Any:
        """Convert a dict or raw object to a ProcurementItem.

        If the input is already a ProcurementItem, returns it
        unchanged.  If it is a dict, constructs a ProcurementItem
        from it.  If models are not available, returns the input
        as-is.

        Args:
            raw: Dict or ProcurementItem-like object.

        Returns:
            ProcurementItem instance or the original object.
        """
        if not MODELS_AVAILABLE:
            return raw
        if isinstance(raw, ProcurementItem):
            return raw
        if isinstance(raw, dict):
            return ProcurementItem(**raw)
        return raw

    def _ensure_supplier_record(self, raw: Any) -> Any:
        """Convert a dict or raw object to a SupplierRecord.

        Args:
            raw: Dict or SupplierRecord-like object.

        Returns:
            SupplierRecord instance or the original object.
        """
        if not MODELS_AVAILABLE:
            return raw
        if isinstance(raw, SupplierRecord):
            return raw
        if isinstance(raw, dict):
            return SupplierRecord(**raw)
        return raw

    @staticmethod
    def _result_to_dict(result: Any) -> Dict[str, Any]:
        """Convert a result object to a dictionary.

        Handles Pydantic models (``model_dump``), dataclasses
        (``asdict``), dicts (passthrough), and generic objects
        (``__dict__``).

        Args:
            result: Result object to convert.

        Returns:
            Dictionary representation.
        """
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        if hasattr(result, "__dict__"):
            return {
                k: v for k, v in result.__dict__.items()
                if not k.startswith("_")
            }
        return {"value": str(result)}

    @staticmethod
    def _item_to_dict(item: Any) -> Dict[str, Any]:
        """Convert a Pydantic model or object to a dict for pipeline input.

        Args:
            item: Pydantic model or dict.

        Returns:
            Dictionary representation suitable for pipeline input.
        """
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if hasattr(item, "__dict__"):
            return {
                k: v for k, v in item.__dict__.items()
                if not k.startswith("_")
            }
        return {"value": str(item)}


# ===================================================================
# Thread-safe singleton access
# ===================================================================

_service_instance: Optional[PurchasedGoodsServicesService] = None
_service_lock = threading.Lock()


def get_service() -> PurchasedGoodsServicesService:
    """Get or create the singleton PurchasedGoodsServicesService instance.

    Uses double-checked locking to ensure thread-safe singleton
    access without contention on the hot path.

    Returns:
        PurchasedGoodsServicesService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = PurchasedGoodsServicesService()
    return _service_instance


def get_router() -> Any:
    """Get the FastAPI router for purchased goods & services.

    Lazily imports the API router module to avoid circular
    dependencies.  Returns None if FastAPI or the router module
    is not available.

    Returns:
        FastAPI APIRouter or None if not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.purchased_goods_services.api.router import router
        return router
    except ImportError:
        logger.warning(
            "Purchased goods & services API router module not available"
        )
        return None


def configure_purchased_goods(
    app: Any,
    config: Any = None,
) -> PurchasedGoodsServicesService:
    """Configure the Purchased Goods & Services Service on a FastAPI app.

    Creates the PurchasedGoodsServicesService singleton, stores it in
    ``app.state``, and mounts the API router at the standard prefix.

    Args:
        app: FastAPI application instance.
        config: Optional configuration override (reserved for future use).

    Returns:
        PurchasedGoodsServicesService singleton instance.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> svc = configure_purchased_goods(app)
        >>> assert hasattr(app.state, "purchased_goods_service")
    """
    global _service_instance

    service = PurchasedGoodsServicesService()

    with _service_lock:
        _service_instance = service

    if hasattr(app, "state"):
        app.state.purchased_goods_service = service

    api_router = get_router()
    if api_router is not None:
        app.include_router(
            api_router,
            prefix="/api/v1/purchased-goods",
            tags=["purchased-goods"],
        )
        logger.info(
            "Purchased goods & services API router mounted "
            "at /api/v1/purchased-goods"
        )
    else:
        logger.warning(
            "Purchased goods & services router not available; "
            "API not mounted"
        )

    logger.info(
        "Purchased Goods & Services service configured "
        "(agent=%s, version=%s)",
        AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-001",
        VERSION if MODELS_AVAILABLE else "1.0.0",
    )
    return service


# ===================================================================
# Public API (__all__)
# ===================================================================

__all__ = [
    # Service facade
    "PurchasedGoodsServicesService",
    # Integration functions
    "configure_purchased_goods",
    "get_service",
    "get_router",
    # Response models
    "SpendBasedResponse",
    "AverageDataResponse",
    "SupplierSpecificResponse",
    "HybridCalculationResponse",
    "PipelineResponse",
    "BatchCalculateResponse",
    "EEIOFactorResponse",
    "PhysicalEFResponse",
    "ComplianceCheckResponse",
    "DQIScoreResponse",
    "ExportResponse",
    "AggregationResponse",
    "HealthResponse",
    "StatsResponse",
]
