# -*- coding: utf-8 -*-
"""
Capital Goods Service Setup - AGENT-MRV-015
============================================

Service facade for the Capital Goods Agent (GL-MRV-S3-002).

Provides ``configure_capital_goods(app)``, ``get_service()``, and
``get_router()`` for FastAPI integration.  Also exposes the
``CapitalGoodsService`` facade class that aggregates all 7 engines:

    1. CapitalAssetDatabaseEngine       - EEIO/Physical EF lookup, classification
    2. SpendBasedCalculatorEngine       - EEIO spend-based calculation
    3. AverageDataCalculatorEngine      - Physical quantity-based calculation
    4. SupplierSpecificCalculatorEngine - EPD/PCF/CDP supplier-specific calc
    5. HybridAggregatorEngine           - Multi-method aggregation & hot-spot
    6. ComplianceCheckerEngine          - Multi-framework regulatory compliance
    7. CapitalGoodsPipelineEngine       - Orchestrated 10-stage pipeline

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.agents.mrv.capital_goods.setup import configure_capital_goods
    >>> app = FastAPI()
    >>> configure_capital_goods(app)

    >>> from greenlang.agents.mrv.capital_goods.setup import get_service
    >>> svc = get_service()
    >>> result = svc.calculate_spend_based(
    ...     records=[record],
    ...     database=EEIODatabase.EPA_USEEIO,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-015 Capital Goods (GL-MRV-S3-002)
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
from greenlang.schemas import utcnow

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
    from greenlang.agents.mrv.capital_goods.models import (
        AGENT_ID,
        VERSION,
        TABLE_PREFIX,
        ZERO,
        ONE,
        ONE_HUNDRED,
        ONE_THOUSAND,
        DECIMAL_PLACES,
        MAX_ASSET_RECORDS,
        MAX_BATCH_PERIODS,
        MAX_FRAMEWORKS,
        DEFAULT_CONFIDENCE_LEVEL,
        # Enumerations
        CalculationMethod,
        AssetCategory,
        AssetSubCategory,
        SpendClassificationSystem,
        EEIODatabase,
        PhysicalEFSource,
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
        CapitalizationPolicy,
        # Data models
        CapitalAssetRecord,
        CapExSpendRecord,
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
        AssetClassification,
        CapitalizationThreshold,
        UsefulLifeRange,
        DepreciationContext,
        MaterialityItem,
        CoverageReport,
        ComplianceRequirement,
        ComplianceCheckResult,
        CalculationRequest,
        BatchRequest,
        CalculationResult,
        AggregationResult,
        HotSpotAnalysis,
        CategoryBoundaryCheck,
        PipelineContext,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Provide sentinel values so the module can still be imported
    AGENT_ID = "GL-MRV-S3-002"
    VERSION = "1.0.0"
    ONE = Decimal("1")
    ZERO = Decimal("0")

# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.capital_goods.config import CapitalGoodsConfig
except ImportError:
    CapitalGoodsConfig = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.capital_goods.capital_asset_database import (
        CapitalAssetDatabaseEngine,
    )
except ImportError:
    CapitalAssetDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.supplier_specific_calculator import (
        SupplierSpecificCalculatorEngine,
    )
except ImportError:
    SupplierSpecificCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.capital_goods_pipeline import (
        CapitalGoodsPipelineEngine,
    )
except ImportError:
    CapitalGoodsPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.provenance import (
        CapitalGoodsProvenanceTracker,
    )
except ImportError:
    CapitalGoodsProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.agents.mrv.capital_goods.metrics import CapitalGoodsMetrics
except ImportError:
    CapitalGoodsMetrics = None  # type: ignore[assignment, misc]

# ===================================================================
# Utility helpers
# ===================================================================

def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return utcnow().isoformat()

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _short_id(prefix: str = "cg") -> str:
    """Generate a short unique identifier with a given prefix.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        A string like ``cg_a1b2c3d4e5f6``.
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
# Default compliance frameworks for Category 2
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

class CalculateResponse(BaseModel):
    """Response for a full pipeline calculation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    calculation_id: str = Field(default="")
    method: str = Field(default="hybrid")
    total_items: int = Field(default=0)
    in_scope_items: int = Field(default=0)
    excluded_items: int = Field(default=0)
    total_emissions_tco2e: float = Field(default=0.0)
    spend_based_emissions_tco2e: float = Field(default=0.0)
    average_data_emissions_tco2e: float = Field(default=0.0)
    supplier_specific_emissions_tco2e: float = Field(default=0.0)
    coverage_level: str = Field(default="minimal")
    dqi_composite: Optional[float] = Field(default=None)
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

class CalculationListResponse(BaseModel):
    """Response for listing stored calculations."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(default=0)
    limit: int = Field(default=100)
    offset: int = Field(default=0)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)

class CalculationDetailResponse(BaseModel):
    """Response for retrieving a single calculation by ID."""

    model_config = ConfigDict(frozen=True)

    found: bool = Field(default=False)
    calculation_id: str = Field(default="")
    data: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: str = Field(default_factory=_utcnow_iso)

class AssetRegisterResponse(BaseModel):
    """Response for registering a capital asset."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    asset_id: str = Field(default="")
    category: str = Field(default="")
    acquisition_cost: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: str = Field(default_factory=_utcnow_iso)

class AssetListResponse(BaseModel):
    """Response for listing registered assets."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(default=0)
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)

class EFListResponse(BaseModel):
    """Response for listing emission factors."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(default=0)
    source: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    factors: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)

class ClassifyResponse(BaseModel):
    """Response for asset classification."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    classified_count: int = Field(default=0)
    classifications: List[Dict[str, Any]] = Field(default_factory=list)
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

class UncertaintyResponse(BaseModel):
    """Response for uncertainty analysis."""

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
    total_capex_usd: float = Field(default=0.0)
    asset_count: int = Field(default=0)
    coverage_level: str = Field(default="minimal")
    period: str = Field(default="annual")
    timestamp: str = Field(default_factory=_utcnow_iso)

class HotSpotResponse(BaseModel):
    """Response for hot-spot analysis."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    top_assets: List[Dict[str, Any]] = Field(default_factory=list)
    top_categories: List[Dict[str, Any]] = Field(default_factory=list)
    top_suppliers: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: str = Field(default_factory=_utcnow_iso)

class ExportResponse(BaseModel):
    """Response for an export operation."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(default=True)
    format: str = Field(default="json")
    content: Any = Field(default=None)
    size_bytes: int = Field(default=0)
    timestamp: str = Field(default_factory=_utcnow_iso)

class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy")
    service: str = Field(default="capital-goods")
    agent_id: str = Field(default=AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-002")
    version: str = Field(default=VERSION if MODELS_AVAILABLE else "1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    models_available: bool = Field(default=True)
    uptime_seconds: float = Field(default=0.0)
    total_calculations: int = Field(default=0)
    timestamp: str = Field(default_factory=_utcnow_iso)

# ===================================================================
# CapitalGoodsService facade
# ===================================================================

class CapitalGoodsService:
    """Unified facade over the Capital Goods Agent SDK.

    Aggregates all 7 engines through a single entry point with
    convenience methods for the 18+ service operations covering
    spend-based, average-data, supplier-specific, and hybrid
    calculation methods for GHG Protocol Scope 3 Category 2
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
        >>> service = CapitalGoodsService()
        >>> from greenlang.agents.mrv.capital_goods.models import (
        ...     CapitalAssetRecord, AssetCategory, CurrencyCode,
        ... )
        >>> from decimal import Decimal
        >>> asset = CapitalAssetRecord(
        ...     description="CNC Machine",
        ...     acquisition_cost=Decimal("250000"),
        ...     currency=CurrencyCode.USD,
        ...     category=AssetCategory.MACHINERY,
        ... )
        >>> result = service.register_asset(record=asset)
        >>> assert result["success"]

    Attributes:
        config: Service configuration singleton.
        _asset_db_engine: Engine 1 - reference data lookups.
        _spend_based_engine: Engine 2 - EEIO spend-based calculations.
        _average_data_engine: Engine 3 - physical EF calculations.
        _supplier_specific_engine: Engine 4 - supplier-specific calculations.
        _hybrid_aggregator_engine: Engine 5 - multi-method aggregation.
        _compliance_checker_engine: Engine 6 - regulatory compliance.
        _pipeline_engine: Engine 7 - orchestrated pipeline.
    """

    _instance: Optional["CapitalGoodsService"] = None
    _lock: threading.RLock = threading.RLock()
    _initialized: bool = False

    def __new__(cls) -> "CapitalGoodsService":
        """Create or return the singleton instance.

        Uses double-checked locking with ``threading.RLock`` to ensure
        exactly one instance is created even under concurrent access.

        Returns:
            The singleton CapitalGoodsService instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the Capital Goods Service facade.

        Only performs initialization once (guarded by ``_initialized``).
        Subsequent calls to ``__init__`` are no-ops, preserving the
        singleton's state.

        Args:
            config: Optional CapitalGoodsConfig instance. If None, a default
                config is created if CapitalGoodsConfig is available.
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
        if self.config is None and CapitalGoodsConfig is not None:
            try:
                self.config = CapitalGoodsConfig()
            except Exception as exc:
                logger.warning(
                    "CapitalGoodsConfig init failed: %s", exc,
                )

        self._start_time: float = time.monotonic()

        # Engine placeholders (lazy-initialized)
        self._asset_db_engine: Any = None
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
        self._calculation_results: List[Dict[str, Any]] = []
        self._batch_results: List[Dict[str, Any]] = []
        self._asset_registry: List[Dict[str, Any]] = []
        self._compliance_results: List[Dict[str, Any]] = []
        self._export_results: List[Dict[str, Any]] = []

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_spend_based: int = 0
        self._total_average_data: int = 0
        self._total_supplier_specific: int = 0
        self._total_hybrid: int = 0
        self._total_compliance_checks: int = 0
        self._total_asset_registrations: int = 0
        self._total_classifications: int = 0
        self._total_exports: int = 0

        # Initialize engines
        self._init_engines()

        logger.info(
            "CapitalGoodsService facade created "
            "(agent=%s, version=%s)",
            AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-002",
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
        logger.info("CapitalGoodsService singleton reset")

    # ------------------------------------------------------------------
    # Engine properties (read-only)
    # ------------------------------------------------------------------

    @property
    def asset_db_engine(self) -> Any:
        """Get the CapitalAssetDatabaseEngine instance (Engine 1)."""
        return self._asset_db_engine

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
        """Get the CapitalGoodsPipelineEngine instance (Engine 7)."""
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
        # E1: CapitalAssetDatabaseEngine
        self._init_single_engine(
            "CapitalAssetDatabaseEngine",
            CapitalAssetDatabaseEngine,
            "_asset_db_engine",
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

        # E7: CapitalGoodsPipelineEngine (receives upstream engines)
        if CapitalGoodsPipelineEngine is not None:
            try:
                self._pipeline_engine = CapitalGoodsPipelineEngine()
                logger.info("CapitalGoodsPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "CapitalGoodsPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning("CapitalGoodsPipelineEngine not available")

        # Provenance tracker
        if CapitalGoodsProvenanceTracker is not None:
            try:
                self._provenance_tracker = CapitalGoodsProvenanceTracker()
                logger.info("CapitalGoodsProvenanceTracker initialized")
            except Exception as exc:
                logger.warning(
                    "CapitalGoodsProvenanceTracker init failed: %s", exc,
                )

        # Metrics collector
        if CapitalGoodsMetrics is not None:
            try:
                self._metrics = CapitalGoodsMetrics()
                logger.info("CapitalGoodsMetrics initialized")
            except Exception as exc:
                logger.warning(
                    "CapitalGoodsMetrics init failed: %s", exc,
                )

    def _init_single_engine(
        self,
        name: str,
        engine_class: Any,
        attr_name: str,
    ) -> None:
        """Initialize a single engine with graceful degradation.

        All engine classes in the Capital Goods agent use a thread-safe
        singleton pattern (``__new__`` with ``_instance``), so calling
        the constructor returns the shared instance.

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
    # Public API: Main calculation methods (21 methods)
    # ==================================================================

    def calculate(
        self,
        request: Any,
    ) -> CalculateResponse:
        """Execute full pipeline calculation for capital goods emissions.

        Runs the complete 10-stage pipeline including boundary checks,
        data quality scoring, method selection, calculation, aggregation,
        compliance checking, and export preparation.

        Args:
            request: CalculationRequest instance containing asset records,
                calculation method, and configuration.

        Returns:
            CalculateResponse with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("cg_calc")

        logger.info(
            "calculate: method=%s, assets=%d",
            request.method if hasattr(request, "method") else "hybrid",
            len(request.assets) if hasattr(request, "assets") else 0,
        )

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "CapitalGoodsPipelineEngine is not available"
                )

            result = self._pipeline_engine.run(request)

            with self.__class__._lock:
                self._total_calculations += 1
                self._calculation_results.append({
                    "calculation_id": calc_id,
                    "result": result,
                    "timestamp": _utcnow_iso(),
                })

            provenance_hash = _compute_hash({
                "calculation_id": calc_id,
                "request": request,
                "result": result,
            })

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return CalculateResponse(
                success=True,
                calculation_id=calc_id,
                method=str(result.method) if hasattr(result, "method") else "hybrid",
                total_items=result.total_items if hasattr(result, "total_items") else 0,
                in_scope_items=result.in_scope_items if hasattr(result, "in_scope_items") else 0,
                excluded_items=result.excluded_items if hasattr(result, "excluded_items") else 0,
                total_emissions_tco2e=_safe_float(result.total_emissions_tco2e) if hasattr(result, "total_emissions_tco2e") else 0.0,
                spend_based_emissions_tco2e=_safe_float(result.spend_based_emissions_tco2e) if hasattr(result, "spend_based_emissions_tco2e") else 0.0,
                average_data_emissions_tco2e=_safe_float(result.average_data_emissions_tco2e) if hasattr(result, "average_data_emissions_tco2e") else 0.0,
                supplier_specific_emissions_tco2e=_safe_float(result.supplier_specific_emissions_tco2e) if hasattr(result, "supplier_specific_emissions_tco2e") else 0.0,
                coverage_level=str(result.coverage_level) if hasattr(result, "coverage_level") else "minimal",
                dqi_composite=_safe_float(result.dqi_composite) if hasattr(result, "dqi_composite") else None,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

        except Exception as exc:
            logger.error("calculate failed: %s", exc, exc_info=True)
            raise

    def calculate_batch(
        self,
        batch: Any,
    ) -> BatchCalculateResponse:
        """Execute batch calculation across multiple periods.

        Processes multiple calculation requests (typically monthly or
        quarterly periods) in sequence, collecting results and
        aggregating totals.

        Args:
            batch: BatchRequest instance containing multiple period requests.

        Returns:
            BatchCalculateResponse with per-period results and totals.
        """
        t0 = time.monotonic()
        batch_id = _short_id("cg_batch")

        logger.info(
            "calculate_batch: batch_id=%s, periods=%d",
            batch_id,
            len(batch.periods) if hasattr(batch, "periods") else 0,
        )

        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "CapitalGoodsPipelineEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")
            completed_count = 0
            failed_count = 0

            periods = batch.periods if hasattr(batch, "periods") else []
            for period_request in periods:
                try:
                    result = self._pipeline_engine.run(period_request)
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

            return BatchCalculateResponse(
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

    def calculate_spend_based(
        self,
        records: List[Any],
        database: Any = None,
        cpi_ratio: Decimal = Decimal("1"),
    ) -> Dict[str, Any]:
        """Calculate Category 2 emissions using the spend-based method.

        Applies EEIO emission factors to capital expenditure amounts after
        currency conversion, inflation deflation, and margin removal.

        Args:
            records: List of CapExSpendRecord instances or dicts.
            database: EEIODatabase enum value. Defaults to EPA_USEEIO.
            cpi_ratio: CPI ratio for inflation deflation. Defaults to 1.

        Returns:
            Dict with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("cg_sb")

        logger.info(
            "calculate_spend_based: records=%d, database=%s",
            len(records),
            database,
        )

        try:
            if self._spend_based_engine is None:
                raise RuntimeError(
                    "SpendBasedCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")

            for record in records:
                result = self._spend_based_engine.calculate_single(
                    record=record,
                    database=database,
                    cpi_ratio=cpi_ratio,
                )
                results_list.append(self._result_to_dict(result))
                total_emissions += _safe_decimal(result.emissions_tco2e) if hasattr(result, "emissions_tco2e") else ZERO

            with self.__class__._lock:
                self._total_spend_based += 1

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return {
                "success": True,
                "calculation_id": calc_id,
                "method": "spend_based",
                "record_count": len(records),
                "total_emissions_tco2e": _safe_float(total_emissions),
                "results": results_list,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as exc:
            logger.error("calculate_spend_based failed: %s", exc, exc_info=True)
            raise

    def calculate_average_data(
        self,
        records: List[Any],
        ef_source: Any = None,
    ) -> Dict[str, Any]:
        """Calculate Category 2 emissions using the average-data method.

        Applies physical emission factors to asset quantities (mass, area, units).

        Args:
            records: List of PhysicalRecord instances or dicts.
            ef_source: PhysicalEFSource enum value. Defaults to DEFRA.

        Returns:
            Dict with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("cg_ad")

        logger.info(
            "calculate_average_data: records=%d, ef_source=%s",
            len(records),
            ef_source,
        )

        try:
            if self._average_data_engine is None:
                raise RuntimeError(
                    "AverageDataCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")

            for record in records:
                result = self._average_data_engine.calculate_single(
                    record=record,
                    ef_source=ef_source,
                )
                results_list.append(self._result_to_dict(result))
                total_emissions += _safe_decimal(result.emissions_tco2e) if hasattr(result, "emissions_tco2e") else ZERO

            with self.__class__._lock:
                self._total_average_data += 1

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return {
                "success": True,
                "calculation_id": calc_id,
                "method": "average_data",
                "record_count": len(records),
                "total_emissions_tco2e": _safe_float(total_emissions),
                "results": results_list,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as exc:
            logger.error("calculate_average_data failed: %s", exc, exc_info=True)
            raise

    def calculate_supplier_specific(
        self,
        records: List[Any],
    ) -> Dict[str, Any]:
        """Calculate Category 2 emissions using supplier-specific data.

        Uses primary data from suppliers (EPDs, PCFs, CDP disclosures).

        Args:
            records: List of SupplierRecord instances or dicts.

        Returns:
            Dict with emissions breakdown and provenance.
        """
        t0 = time.monotonic()
        calc_id = _short_id("cg_ss")

        logger.info(
            "calculate_supplier_specific: records=%d",
            len(records),
        )

        try:
            if self._supplier_specific_engine is None:
                raise RuntimeError(
                    "SupplierSpecificCalculatorEngine is not available"
                )

            results_list: List[Dict[str, Any]] = []
            total_emissions = Decimal("0")

            for record in records:
                result = self._supplier_specific_engine.calculate_single(
                    record=record,
                )
                results_list.append(self._result_to_dict(result))
                total_emissions += _safe_decimal(result.emissions_tco2e) if hasattr(result, "emissions_tco2e") else ZERO

            with self.__class__._lock:
                self._total_supplier_specific += 1

            t1 = time.monotonic()
            processing_time_ms = (t1 - t0) * 1000.0

            return {
                "success": True,
                "calculation_id": calc_id,
                "method": "supplier_specific",
                "record_count": len(records),
                "total_emissions_tco2e": _safe_float(total_emissions),
                "results": results_list,
                "processing_time_ms": processing_time_ms,
            }

        except Exception as exc:
            logger.error("calculate_supplier_specific failed: %s", exc, exc_info=True)
            raise

    def list_calculations(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> CalculationListResponse:
        """List stored calculations with optional filtering.

        Args:
            filters: Optional filter dict (e.g., {"method": "hybrid"}).
            limit: Maximum number of results to return. Defaults to 100.
            offset: Number of results to skip. Defaults to 0.

        Returns:
            CalculationListResponse with matching calculations.
        """
        with self.__class__._lock:
            all_results = self._calculation_results.copy()

        # Apply filters if provided
        if filters:
            filtered = []
            for item in all_results:
                match = True
                for key, value in filters.items():
                    if item.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            all_results = filtered

        # Apply pagination
        paginated = all_results[offset:offset + limit]

        return CalculationListResponse(
            total=len(all_results),
            limit=limit,
            offset=offset,
            items=paginated,
        )

    def get_calculation(
        self,
        calc_id: str,
    ) -> CalculationDetailResponse:
        """Retrieve a single calculation by ID.

        Args:
            calc_id: Calculation identifier.

        Returns:
            CalculationDetailResponse with calculation data if found.
        """
        with self.__class__._lock:
            for item in self._calculation_results:
                if item.get("calculation_id") == calc_id:
                    return CalculationDetailResponse(
                        found=True,
                        calculation_id=calc_id,
                        data=item,
                    )

        return CalculationDetailResponse(
            found=False,
            calculation_id=calc_id,
        )

    def delete_calculation(
        self,
        calc_id: str,
    ) -> bool:
        """Delete a calculation by ID.

        Args:
            calc_id: Calculation identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self.__class__._lock:
            for idx, item in enumerate(self._calculation_results):
                if item.get("calculation_id") == calc_id:
                    del self._calculation_results[idx]
                    logger.info("Deleted calculation: %s", calc_id)
                    return True

        logger.warning("Calculation not found for deletion: %s", calc_id)
        return False

    def register_asset(
        self,
        record: Any,
    ) -> AssetRegisterResponse:
        """Register a capital asset in the database.

        Args:
            record: CapitalAssetRecord instance.

        Returns:
            AssetRegisterResponse with asset ID and provenance.
        """
        t0 = time.monotonic()
        asset_id = _short_id("asset")

        logger.info(
            "register_asset: category=%s, cost=%s",
            record.category if hasattr(record, "category") else "unknown",
            record.acquisition_cost if hasattr(record, "acquisition_cost") else 0,
        )

        try:
            if self._asset_db_engine is None:
                raise RuntimeError(
                    "CapitalAssetDatabaseEngine is not available"
                )

            # Store in asset registry
            asset_data = {
                "asset_id": asset_id,
                "record": record,
                "timestamp": _utcnow_iso(),
            }

            with self.__class__._lock:
                self._asset_registry.append(asset_data)
                self._total_asset_registrations += 1

            provenance_hash = _compute_hash(record)

            return AssetRegisterResponse(
                success=True,
                asset_id=asset_id,
                category=str(record.category) if hasattr(record, "category") else "",
                acquisition_cost=_safe_float(record.acquisition_cost) if hasattr(record, "acquisition_cost") else 0.0,
                provenance_hash=provenance_hash,
            )

        except Exception as exc:
            logger.error("register_asset failed: %s", exc, exc_info=True)
            raise

    def list_assets(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AssetListResponse:
        """List registered assets with optional filtering.

        Args:
            filters: Optional filter dict (e.g., {"category": "machinery"}).

        Returns:
            AssetListResponse with matching assets.
        """
        with self.__class__._lock:
            all_assets = self._asset_registry.copy()

        # Apply filters if provided
        if filters:
            filtered = []
            for item in all_assets:
                match = True
                record = item.get("record")
                for key, value in filters.items():
                    if hasattr(record, key) and getattr(record, key) != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            all_assets = filtered

        return AssetListResponse(
            total=len(all_assets),
            assets=all_assets,
        )

    def update_asset(
        self,
        asset_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update an asset's attributes.

        Args:
            asset_id: Asset identifier.
            updates: Dict of attribute updates.

        Returns:
            Dict with updated asset data.
        """
        with self.__class__._lock:
            for item in self._asset_registry:
                if item.get("asset_id") == asset_id:
                    record = item.get("record")
                    for key, value in updates.items():
                        if hasattr(record, key):
                            setattr(record, key, value)
                    logger.info("Updated asset: %s", asset_id)
                    return {
                        "success": True,
                        "asset_id": asset_id,
                        "updated_fields": list(updates.keys()),
                    }

        logger.warning("Asset not found for update: %s", asset_id)
        return {
            "success": False,
            "asset_id": asset_id,
            "error": "Asset not found",
        }

    def get_emission_factors(
        self,
        source: Optional[str] = None,
        category: Optional[str] = None,
    ) -> EFListResponse:
        """Get emission factors from the database.

        Args:
            source: Optional EF source filter (e.g., "epa_useeio").
            category: Optional asset category filter (e.g., "machinery").

        Returns:
            EFListResponse with matching emission factors.
        """
        try:
            if self._asset_db_engine is None:
                raise RuntimeError(
                    "CapitalAssetDatabaseEngine is not available"
                )

            factors = self._asset_db_engine.get_emission_factors(
                source=source,
                category=category,
            )

            return EFListResponse(
                total=len(factors),
                source=source,
                category=category,
                factors=factors,
            )

        except Exception as exc:
            logger.error("get_emission_factors failed: %s", exc, exc_info=True)
            raise

    def register_custom_ef(
        self,
        factor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom emission factor.

        Args:
            factor_data: Dict containing EF data (category, factor, source, etc.).

        Returns:
            Dict with registration confirmation.
        """
        try:
            if self._asset_db_engine is None:
                raise RuntimeError(
                    "CapitalAssetDatabaseEngine is not available"
                )

            ef_id = _short_id("ef")
            result = self._asset_db_engine.register_custom_ef(
                ef_id=ef_id,
                factor_data=factor_data,
            )

            return {
                "success": True,
                "ef_id": ef_id,
                "factor_data": factor_data,
            }

        except Exception as exc:
            logger.error("register_custom_ef failed: %s", exc, exc_info=True)
            raise

    def classify_assets(
        self,
        records: List[Any],
    ) -> ClassifyResponse:
        """Classify assets into categories and subcategories.

        Args:
            records: List of CapitalAssetRecord instances.

        Returns:
            ClassifyResponse with classification results.
        """
        try:
            if self._asset_db_engine is None:
                raise RuntimeError(
                    "CapitalAssetDatabaseEngine is not available"
                )

            classifications: List[Dict[str, Any]] = []

            for record in records:
                classification = self._asset_db_engine.classify_asset(record)
                classifications.append(self._result_to_dict(classification))

            with self.__class__._lock:
                self._total_classifications += 1

            return ClassifyResponse(
                success=True,
                classified_count=len(classifications),
                classifications=classifications,
            )

        except Exception as exc:
            logger.error("classify_assets failed: %s", exc, exc_info=True)
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
        compliance_id = _short_id("cg_comp")

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

    def run_uncertainty(
        self,
        result: Any,
        method: Optional[str] = None,
        confidence_level: float = 95.0,
    ) -> UncertaintyResponse:
        """Run uncertainty quantification on a calculation result.

        Args:
            result: CalculationResult instance.
            method: Optional uncertainty method. Defaults to "pedigree".
            confidence_level: Confidence level percentage. Defaults to 95.0.

        Returns:
            UncertaintyResponse with uncertainty bounds.
        """
        try:
            if self._hybrid_aggregator_engine is None:
                raise RuntimeError(
                    "HybridAggregatorEngine is not available"
                )

            uncertainty = self._hybrid_aggregator_engine.quantify_uncertainty(
                result=result,
                method=method or "pedigree",
                confidence_level=confidence_level,
            )

            return UncertaintyResponse(
                success=True,
                method=method or "pedigree",
                confidence_level=confidence_level,
                lower_bound_tco2e=_safe_float(uncertainty.get("lower_bound_tco2e", 0.0)),
                upper_bound_tco2e=_safe_float(uncertainty.get("upper_bound_tco2e", 0.0)),
                uncertainty_percentage=_safe_float(uncertainty.get("uncertainty_percentage", 0.0)),
            )

        except Exception as exc:
            logger.error("run_uncertainty failed: %s", exc, exc_info=True)
            raise

    def get_aggregations(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AggregationResponse:
        """Get aggregated emissions summary.

        Args:
            filters: Optional filter dict for aggregation scope.

        Returns:
            AggregationResponse with grouped emissions totals.
        """
        try:
            if self._hybrid_aggregator_engine is None:
                raise RuntimeError(
                    "HybridAggregatorEngine is not available"
                )

            with self.__class__._lock:
                all_results = self._calculation_results.copy()

            aggregations = self._hybrid_aggregator_engine.aggregate(
                results=all_results,
                filters=filters,
            )

            return AggregationResponse(
                groups=aggregations.get("groups", {}),
                total_emissions_tco2e=_safe_float(aggregations.get("total_emissions_tco2e", 0.0)),
                total_capex_usd=_safe_float(aggregations.get("total_capex_usd", 0.0)),
                asset_count=aggregations.get("asset_count", 0),
                coverage_level=aggregations.get("coverage_level", "minimal"),
                period=aggregations.get("period", "annual"),
            )

        except Exception as exc:
            logger.error("get_aggregations failed: %s", exc, exc_info=True)
            raise

    def get_hot_spots(
        self,
        result: Any,
    ) -> HotSpotResponse:
        """Identify emission hot-spots in a calculation result.

        Args:
            result: CalculationResult instance.

        Returns:
            HotSpotResponse with top contributors.
        """
        try:
            if self._hybrid_aggregator_engine is None:
                raise RuntimeError(
                    "HybridAggregatorEngine is not available"
                )

            hot_spots = self._hybrid_aggregator_engine.identify_hot_spots(result)

            return HotSpotResponse(
                success=True,
                top_assets=hot_spots.get("top_assets", []),
                top_categories=hot_spots.get("top_categories", []),
                top_suppliers=hot_spots.get("top_suppliers", []),
            )

        except Exception as exc:
            logger.error("get_hot_spots failed: %s", exc, exc_info=True)
            raise

    def export_report(
        self,
        result: Any,
        format: str = "json",
    ) -> ExportResponse:
        """Export calculation result to specified format.

        Args:
            result: CalculationResult instance.
            format: Export format ("json", "csv", "xlsx", "pdf"). Defaults to "json".

        Returns:
            ExportResponse with exported content.
        """
        try:
            if self._pipeline_engine is None:
                raise RuntimeError(
                    "CapitalGoodsPipelineEngine is not available"
                )

            exported = self._pipeline_engine.export(result, format)

            with self.__class__._lock:
                self._total_exports += 1
                self._export_results.append({
                    "format": format,
                    "timestamp": _utcnow_iso(),
                })

            size_bytes = len(str(exported)) if exported else 0

            return ExportResponse(
                success=True,
                format=format,
                content=exported,
                size_bytes=size_bytes,
            )

        except Exception as exc:
            logger.error("export_report failed: %s", exc, exc_info=True)
            raise

    def health_check(self) -> HealthResponse:
        """Service health check.

        Returns:
            HealthResponse with service status and engine availability.
        """
        engines = {
            "asset_db": "available" if self._asset_db_engine is not None else "unavailable",
            "spend_based": "available" if self._spend_based_engine is not None else "unavailable",
            "average_data": "available" if self._average_data_engine is not None else "unavailable",
            "supplier_specific": "available" if self._supplier_specific_engine is not None else "unavailable",
            "hybrid_aggregator": "available" if self._hybrid_aggregator_engine is not None else "unavailable",
            "compliance_checker": "available" if self._compliance_checker_engine is not None else "unavailable",
            "pipeline": "available" if self._pipeline_engine is not None else "unavailable",
        }

        uptime = time.monotonic() - self._start_time

        with self.__class__._lock:
            total_calcs = self._total_calculations

        return HealthResponse(
            status="healthy",
            service="capital-goods",
            agent_id=AGENT_ID if MODELS_AVAILABLE else "GL-MRV-S3-002",
            version=VERSION if MODELS_AVAILABLE else "1.0.0",
            engines=engines,
            models_available=MODELS_AVAILABLE,
            uptime_seconds=uptime,
            total_calculations=total_calcs,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dict with aggregate statistics.
        """
        uptime = time.monotonic() - self._start_time

        with self.__class__._lock:
            stats = {
                "total_calculations": self._total_calculations,
                "total_batch_runs": self._total_batch_runs,
                "total_spend_based": self._total_spend_based,
                "total_average_data": self._total_average_data,
                "total_supplier_specific": self._total_supplier_specific,
                "total_hybrid": self._total_hybrid,
                "total_compliance_checks": self._total_compliance_checks,
                "total_asset_registrations": self._total_asset_registrations,
                "total_classifications": self._total_classifications,
                "total_exports": self._total_exports,
                "uptime_seconds": uptime,
                "timestamp": _utcnow_iso(),
            }

        return stats

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

_SERVICE_INSTANCE: Optional[CapitalGoodsService] = None

def get_service(config: Optional[Any] = None) -> CapitalGoodsService:
    """Get the singleton CapitalGoodsService instance.

    Args:
        config: Optional CapitalGoodsConfig instance.

    Returns:
        The singleton CapitalGoodsService instance.
    """
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = CapitalGoodsService(config)
    return _SERVICE_INSTANCE

# ===================================================================
# FastAPI router configuration
# ===================================================================

def get_router() -> Any:
    """Create and configure a FastAPI APIRouter for capital goods endpoints.

    Returns:
        APIRouter instance with all endpoints configured, or None if
        FastAPI is not available.
    """
    if not FASTAPI_AVAILABLE or APIRouter is None:
        logger.warning("FastAPI not available, cannot create router")
        return None

    router = APIRouter(
        prefix="/api/v1/capital-goods",
        tags=["capital-goods"],
    )

    service = get_service()

    @router.post("/calculate", response_model=CalculateResponse)
    async def calculate_endpoint(request: Dict[str, Any]) -> CalculateResponse:
        """Execute full pipeline calculation."""
        # Convert dict to CalculationRequest if models available
        if MODELS_AVAILABLE:
            calc_request = CalculationRequest(**request)
        else:
            calc_request = request
        return service.calculate(calc_request)

    @router.post("/calculate/batch", response_model=BatchCalculateResponse)
    async def calculate_batch_endpoint(batch: Dict[str, Any]) -> BatchCalculateResponse:
        """Execute batch calculation."""
        if MODELS_AVAILABLE:
            batch_request = BatchRequest(**batch)
        else:
            batch_request = batch
        return service.calculate_batch(batch_request)

    @router.get("/calculations", response_model=CalculationListResponse)
    async def list_calculations_endpoint(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> CalculationListResponse:
        """List stored calculations."""
        return service.list_calculations(limit=limit, offset=offset)

    @router.get("/calculations/{calc_id}", response_model=CalculationDetailResponse)
    async def get_calculation_endpoint(
        calc_id: str = Path(...),
    ) -> CalculationDetailResponse:
        """Get a single calculation by ID."""
        return service.get_calculation(calc_id)

    @router.delete("/calculations/{calc_id}")
    async def delete_calculation_endpoint(calc_id: str = Path(...)) -> Dict[str, Any]:
        """Delete a calculation by ID."""
        deleted = service.delete_calculation(calc_id)
        return {"success": deleted, "calculation_id": calc_id}

    @router.post("/assets", response_model=AssetRegisterResponse)
    async def register_asset_endpoint(record: Dict[str, Any]) -> AssetRegisterResponse:
        """Register a capital asset."""
        if MODELS_AVAILABLE:
            asset_record = CapitalAssetRecord(**record)
        else:
            asset_record = record
        return service.register_asset(asset_record)

    @router.get("/assets", response_model=AssetListResponse)
    async def list_assets_endpoint() -> AssetListResponse:
        """List registered assets."""
        return service.list_assets()

    @router.get("/emission-factors", response_model=EFListResponse)
    async def get_emission_factors_endpoint(
        source: Optional[str] = Query(default=None),
        category: Optional[str] = Query(default=None),
    ) -> EFListResponse:
        """Get emission factors."""
        return service.get_emission_factors(source=source, category=category)

    @router.post("/compliance", response_model=ComplianceResponse)
    async def check_compliance_endpoint(request: Dict[str, Any]) -> ComplianceResponse:
        """Check regulatory compliance."""
        result = request.get("result")
        frameworks = request.get("frameworks")
        return service.check_compliance(result, frameworks)

    @router.get("/health", response_model=HealthResponse)
    async def health_check_endpoint() -> HealthResponse:
        """Service health check."""
        return service.health_check()

    @router.get("/stats")
    async def get_stats_endpoint() -> Dict[str, Any]:
        """Get service statistics."""
        return service.get_stats()

    return router

# ===================================================================
# FastAPI app configuration
# ===================================================================

def configure_capital_goods(app: Any) -> None:
    """Configure a FastAPI app with capital goods endpoints.

    Registers the capital goods router with the provided FastAPI app
    instance.

    Args:
        app: FastAPI application instance.
    """
    if not FASTAPI_AVAILABLE:
        logger.warning(
            "FastAPI not available, cannot configure capital goods endpoints"
        )
        return

    router = get_router()
    if router is not None:
        app.include_router(router)
        logger.info("Capital Goods router registered with FastAPI app")
    else:
        logger.warning("Failed to create Capital Goods router")
