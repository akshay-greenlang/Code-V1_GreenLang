# -*- coding: utf-8 -*-
"""
GL-DATA-X-017: GreenLang Time Series Gap Filler Agent SDK
=========================================================

This package provides intelligent time series gap detection, frequency
analysis, and gap filling for GreenLang sustainability datasets. It
supports:

- Gap detection against expected frequency grids with gap characterization
  (short_gap, long_gap, block_gap, periodic, systematic)
- Automatic frequency inference (9 levels: sub_minute through yearly)
  with regularity scoring and mixed-frequency detection
- Linear, cubic spline, polynomial, piecewise cubic Hermite, Akima,
  and nearest-neighbor interpolation with time-weighting
- Seasonal decomposition filling with multi-period pattern matching,
  calendar-aware business day/holiday handling, and Holt-Winters
- OLS trend extrapolation, exponential smoothing (single/double/triple),
  moving average extrapolation, regime-aware extrapolation
- Cross-series correlation analysis (Pearson, Spearman, Kendall) with
  regression-based fill, donor series matching, multi-series consensus
- End-to-end pipeline orchestration: detect -> characterize -> select
  strategy -> fill -> validate -> document
- Calendar-aware gap filling with business days, holidays, fiscal
  periods, and custom reporting windows
- Per-fill confidence scoring and fill quality validation with
  distribution preservation checks
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- 20 REST API endpoints

Key Components:
    - config: TimeSeriesGapFillerConfig with GL_TSGF_ env prefix
    - gap_detector: Gap detection engine
    - frequency_analyzer: Frequency analysis engine
    - interpolation_engine: Numeric interpolation engine
    - seasonal_filler: Seasonal decomposition fill engine
    - trend_extrapolator: Trend extrapolation engine
    - cross_series_filler: Cross-series correlation fill engine
    - gap_filler_pipeline: End-to-end pipeline engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - setup: Service facade and FastAPI integration

Example:
    >>> from greenlang.time_series_gap_filler import TimeSeriesGapFillerService
    >>> service = TimeSeriesGapFillerService()
    >>> result = service.detect_gaps(
    ...     series=[1.0, None, None, 4.0, 5.0],
    ...     timestamps=[0, 1, 2, 3, 4],
    ... )
    >>> print(result.total_gaps)

Agent ID: GL-DATA-X-017
Agent Name: Time Series Gap Filler Agent
Internal Label: AGENT-DATA-014
"""

# ---------------------------------------------------------------------------
# Agent metadata constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-DATA-X-017"
AGENT_NAME = "Time Series Gap Filler Agent"
AGENT_VERSION = "1.0.0"
AGENT_CATEGORY = "Layer 2 - Data Quality Agents"
AGENT_LABEL = "AGENT-DATA-014"

__version__ = AGENT_VERSION
__agent_id__ = AGENT_ID
__agent_name__ = AGENT_NAME

# SDK availability flag
TIME_SERIES_GAP_FILLER_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration (4 items)
# ---------------------------------------------------------------------------
from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Provenance (2 items)
# ---------------------------------------------------------------------------
from greenlang.time_series_gap_filler.provenance import (
    ProvenanceTracker,
    ProvenanceEntry,
)

# ---------------------------------------------------------------------------
# Metrics (25 items)
# ---------------------------------------------------------------------------
from greenlang.time_series_gap_filler.metrics import (
    PROMETHEUS_AVAILABLE,
    tsgf_jobs_processed_total,
    tsgf_gaps_detected_total,
    tsgf_gaps_filled_total,
    tsgf_validations_passed_total,
    tsgf_frequencies_detected_total,
    tsgf_strategies_selected_total,
    tsgf_fill_confidence,
    tsgf_processing_duration_seconds,
    tsgf_gap_duration_seconds,
    tsgf_active_jobs,
    tsgf_total_gaps_open,
    tsgf_processing_errors_total,
    inc_jobs_processed,
    inc_gaps_detected,
    inc_gaps_filled,
    inc_validations,
    inc_frequencies,
    inc_strategies,
    observe_confidence,
    observe_duration,
    observe_gap_duration,
    set_active_jobs,
    set_gaps_open,
    inc_errors,
)

# ---------------------------------------------------------------------------
# Core engines (Layer 2 SDK) - optional imports with graceful fallback (7)
# ---------------------------------------------------------------------------
try:
    from greenlang.time_series_gap_filler.gap_detector import (
        GapDetector,
    )
except ImportError:
    GapDetector = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.frequency_analyzer import (
        FrequencyAnalyzer,
    )
except ImportError:
    FrequencyAnalyzer = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.interpolation_engine import (
        InterpolationEngine,
    )
except ImportError:
    InterpolationEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.seasonal_filler import (
        SeasonalFiller,
    )
except ImportError:
    SeasonalFiller = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.trend_extrapolator import (
        TrendExtrapolator,
    )
except ImportError:
    TrendExtrapolator = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.cross_series_filler import (
        CrossSeriesFiller,
    )
except ImportError:
    CrossSeriesFiller = None  # type: ignore[assignment, misc]

try:
    from greenlang.time_series_gap_filler.gap_filler_pipeline import (
        GapFillerPipeline,
    )
except ImportError:
    GapFillerPipeline = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Layer 1 re-exports from missing_value_imputer (3 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.missing_value_imputer.time_series_imputer import (
        TimeSeriesImputerEngine as L1TimeSeriesImputerEngine,
    )
    TimeSeriesImputerEngine = L1TimeSeriesImputerEngine
except ImportError:
    TimeSeriesImputerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.missing_value_imputer.models import (
        ImputedValue as L1ImputedValue,
        ConfidenceLevel as L1ConfidenceLevel,
    )
    ImputedValue = L1ImputedValue
    ConfidenceLevel = L1ConfidenceLevel
except ImportError:
    ImputedValue = None  # type: ignore[assignment, misc]
    ConfidenceLevel = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.timeliness_tracker import (
        TimelinessTracker as L1TimelinessTracker,
    )
    TimelinessTracker = L1TimelinessTracker
except ImportError:
    TimelinessTracker = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Service setup facade (4 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.time_series_gap_filler.setup import (
        TimeSeriesGapFillerService,
        configure_gap_filler,
        get_gap_filler,
        get_router,
    )
except ImportError:
    TimeSeriesGapFillerService = None  # type: ignore[assignment, misc]
    configure_gap_filler = None  # type: ignore[assignment, misc]
    get_gap_filler = None  # type: ignore[assignment, misc]
    get_router = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Response/result models from setup (8 items)
# ---------------------------------------------------------------------------
try:
    from greenlang.time_series_gap_filler.setup import (
        GapDetectionResult,
        BatchDetectionResult,
        FrequencyResult,
        FillResult,
        ValidationResult,
        CorrelationResult,
        CalendarDefinition,
        GapStatistics,
    )
except ImportError:
    GapDetectionResult = None  # type: ignore[assignment, misc]
    BatchDetectionResult = None  # type: ignore[assignment, misc]
    FrequencyResult = None  # type: ignore[assignment, misc]
    FillResult = None  # type: ignore[assignment, misc]
    ValidationResult = None  # type: ignore[assignment, misc]
    CorrelationResult = None  # type: ignore[assignment, misc]
    CalendarDefinition = None  # type: ignore[assignment, misc]
    GapStatistics = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Router (1 item)
# ---------------------------------------------------------------------------
try:
    from greenlang.time_series_gap_filler.api.router import router
except ImportError:
    router = None  # type: ignore[assignment]


__all__ = [
    # -------------------------------------------------------------------------
    # Agent metadata (5)
    # -------------------------------------------------------------------------
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_VERSION",
    "AGENT_CATEGORY",
    "AGENT_LABEL",
    # -------------------------------------------------------------------------
    # Version and identity (3)
    # -------------------------------------------------------------------------
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # -------------------------------------------------------------------------
    # SDK flag (1)
    # -------------------------------------------------------------------------
    "TIME_SERIES_GAP_FILLER_SDK_AVAILABLE",
    # -------------------------------------------------------------------------
    # Configuration (4)
    # -------------------------------------------------------------------------
    "TimeSeriesGapFillerConfig",
    "get_config",
    "set_config",
    "reset_config",
    # -------------------------------------------------------------------------
    # Provenance (2)
    # -------------------------------------------------------------------------
    "ProvenanceTracker",
    "ProvenanceEntry",
    # -------------------------------------------------------------------------
    # Metrics flag (1)
    # -------------------------------------------------------------------------
    "PROMETHEUS_AVAILABLE",
    # -------------------------------------------------------------------------
    # Metric objects (12)
    # -------------------------------------------------------------------------
    "tsgf_jobs_processed_total",
    "tsgf_gaps_detected_total",
    "tsgf_gaps_filled_total",
    "tsgf_validations_passed_total",
    "tsgf_frequencies_detected_total",
    "tsgf_strategies_selected_total",
    "tsgf_fill_confidence",
    "tsgf_processing_duration_seconds",
    "tsgf_gap_duration_seconds",
    "tsgf_active_jobs",
    "tsgf_total_gaps_open",
    "tsgf_processing_errors_total",
    # -------------------------------------------------------------------------
    # Metric helper functions (12)
    # -------------------------------------------------------------------------
    "inc_jobs_processed",
    "inc_gaps_detected",
    "inc_gaps_filled",
    "inc_validations",
    "inc_frequencies",
    "inc_strategies",
    "observe_confidence",
    "observe_duration",
    "observe_gap_duration",
    "set_active_jobs",
    "set_gaps_open",
    "inc_errors",
    # -------------------------------------------------------------------------
    # Core engines (Layer 2) (7)
    # -------------------------------------------------------------------------
    "GapDetector",
    "FrequencyAnalyzer",
    "InterpolationEngine",
    "SeasonalFiller",
    "TrendExtrapolator",
    "CrossSeriesFiller",
    "GapFillerPipeline",
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (4)
    # -------------------------------------------------------------------------
    "TimeSeriesImputerEngine",
    "ImputedValue",
    "ConfidenceLevel",
    "TimelinessTracker",
    # -------------------------------------------------------------------------
    # Service setup facade (4)
    # -------------------------------------------------------------------------
    "TimeSeriesGapFillerService",
    "configure_gap_filler",
    "get_gap_filler",
    "get_router",
    # -------------------------------------------------------------------------
    # Response/result models (8)
    # -------------------------------------------------------------------------
    "GapDetectionResult",
    "BatchDetectionResult",
    "FrequencyResult",
    "FillResult",
    "ValidationResult",
    "CorrelationResult",
    "CalendarDefinition",
    "GapStatistics",
    # -------------------------------------------------------------------------
    # Router (1)
    # -------------------------------------------------------------------------
    "router",
]
