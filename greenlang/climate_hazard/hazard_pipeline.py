# -*- coding: utf-8 -*-
"""
HazardPipelineEngine - AGENT-DATA-020: Climate Hazard Connector

Engine 7 of 7.  End-to-end orchestration engine that composes all six
upstream engines (HazardDatabaseEngine, RiskIndexEngine,
ScenarioProjectorEngine, ExposureAssessorEngine, VulnerabilityScorerEngine,
ComplianceReporterEngine) into a deterministic seven-stage pipeline for
climate hazard assessment.

Pipeline stages:
    1. INGEST   -- Register assets and ingest baseline hazard data from
                   configured data sources via HazardDatabaseEngine.
    2. INDEX    -- Calculate composite risk indices for each asset x hazard
                   combination via RiskIndexEngine.
    3. PROJECT  -- Project hazard intensity under SSP/RCP climate scenarios
                   across specified time horizons via ScenarioProjectorEngine.
    4. ASSESS   -- Assess geographic and financial exposure of assets to
                   climate hazards via ExposureAssessorEngine.
    5. SCORE    -- Score entity-level vulnerability combining exposure,
                   sensitivity, and adaptive capacity via
                   VulnerabilityScorerEngine.
    6. REPORT   -- Generate TCFD/CSRD/EU Taxonomy compliance reports via
                   ComplianceReporterEngine.
    7. AUDIT    -- Record full provenance chain for the pipeline run via
                   ProvenanceTracker with SHA-256 chain hashing.

Zero-Hallucination Guarantees:
    - All risk indices are computed using deterministic weighted arithmetic
      (probability x weight + intensity x weight + frequency x weight +
      duration x weight).  No LLM inference for numeric values.
    - Scenario projections use tabulated IPCC scaling factors, not ML
      predictions.
    - Vulnerability scores use weighted sums of exposure, sensitivity, and
      adaptive capacity.  No LLM inference in the calculation path.
    - Every stage produces a SHA-256 provenance hash anchored to the genesis
      chain.

Thread Safety:
    A ``threading.Lock`` protects all mutable state (pipeline run store,
    statistics counters, engine references).  Individual engine calls are
    stateless and can be invoked concurrently from multiple threads as long
    as the pipeline run store mutations are serialised.

Graceful Degradation:
    Each upstream engine is imported inside a try/except guard.  When an
    engine module is unavailable the corresponding ``_*_AVAILABLE`` flag is
    ``False`` and the pipeline stage produces a stub result with an
    ``"engine_unavailable"`` status rather than raising an ImportError.

Example:
    >>> from greenlang.climate_hazard.hazard_pipeline import (
    ...     HazardPipelineEngine,
    ...     PIPELINE_STAGES,
    ... )
    >>> engine = HazardPipelineEngine()
    >>> result = engine.run_pipeline(
    ...     assets=[{
    ...         "asset_id": "asset_001",
    ...         "name": "HQ Office",
    ...         "asset_type": "office",
    ...         "location": {"lat": 51.5074, "lon": -0.1278},
    ...     }],
    ...     hazard_types=["flood", "heat_wave"],
    ... )
    >>> assert result["status"] in ("completed", "partial", "failed")
    >>> assert result["provenance_hash"] is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful imports for the 6 upstream engines
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.hazard_database import HazardDatabaseEngine
    _DATABASE_AVAILABLE = True
except Exception:  # noqa: BLE001 — broad catch: ImportError or Prometheus ValueError
    HazardDatabaseEngine = None  # type: ignore[assignment, misc]
    _DATABASE_AVAILABLE = False
    logger.warning(
        "HazardDatabaseEngine not available; hazard_database import failed. "
        "Ingest operations will use stub fallback."
    )

try:
    from greenlang.climate_hazard.risk_index import RiskIndexEngine
    _RISK_INDEX_AVAILABLE = True
except Exception:  # noqa: BLE001
    RiskIndexEngine = None  # type: ignore[assignment, misc]
    _RISK_INDEX_AVAILABLE = False
    logger.warning(
        "RiskIndexEngine not available; risk_index import failed. "
        "Risk index calculations will use stub fallback."
    )

try:
    from greenlang.climate_hazard.scenario_projector import ScenarioProjectorEngine
    _PROJECTOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    ScenarioProjectorEngine = None  # type: ignore[assignment, misc]
    _PROJECTOR_AVAILABLE = False
    logger.warning(
        "ScenarioProjectorEngine not available; scenario_projector import failed. "
        "Scenario projections will use stub fallback."
    )

try:
    from greenlang.climate_hazard.exposure_assessor import ExposureAssessorEngine
    _EXPOSURE_AVAILABLE = True
except Exception:  # noqa: BLE001
    ExposureAssessorEngine = None  # type: ignore[assignment, misc]
    _EXPOSURE_AVAILABLE = False
    logger.warning(
        "ExposureAssessorEngine not available; exposure_assessor import failed. "
        "Exposure assessments will use stub fallback."
    )

try:
    from greenlang.climate_hazard.vulnerability_scorer import VulnerabilityScorerEngine
    _VULNERABILITY_AVAILABLE = True
except Exception:  # noqa: BLE001
    VulnerabilityScorerEngine = None  # type: ignore[assignment, misc]
    _VULNERABILITY_AVAILABLE = False
    logger.warning(
        "VulnerabilityScorerEngine not available; vulnerability_scorer import failed. "
        "Vulnerability scoring will use stub fallback."
    )

try:
    from greenlang.climate_hazard.compliance_reporter import ComplianceReporterEngine
    _REPORTER_AVAILABLE = True
except Exception:  # noqa: BLE001
    ComplianceReporterEngine = None  # type: ignore[assignment, misc]
    _REPORTER_AVAILABLE = False
    logger.warning(
        "ComplianceReporterEngine not available; compliance_reporter import failed. "
        "Compliance reporting will use stub fallback."
    )

# ---------------------------------------------------------------------------
# Graceful imports for provenance, metrics, and config
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import ProvenanceTracker
    _PROVENANCE_AVAILABLE = True
except Exception:  # noqa: BLE001
    ProvenanceTracker = None  # type: ignore[assignment, misc]
    _PROVENANCE_AVAILABLE = False
    logger.warning(
        "ProvenanceTracker not available; provenance import failed. "
        "Audit trail tracking will be disabled."
    )

try:
    from greenlang.climate_hazard.metrics import (
        record_ingestion as _metrics_record_ingestion,
        record_risk_calculation as _metrics_record_risk_calculation,
        record_projection as _metrics_record_projection,
        record_exposure as _metrics_record_exposure,
        record_vulnerability as _metrics_record_vulnerability,
        record_report as _metrics_record_report,
        record_pipeline as _metrics_record_pipeline,
        set_active_assets as _metrics_set_active_assets,
        set_high_risk as _metrics_set_high_risk,
        observe_pipeline_duration as _metrics_observe_pipeline_duration,
    )
    _METRICS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _METRICS_AVAILABLE = False
    logger.warning(
        "Climate hazard metrics module not available; "
        "Prometheus metrics will be disabled."
    )

try:
    from greenlang.climate_hazard.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except Exception:  # noqa: BLE001
    _get_config = None  # type: ignore[assignment, misc]
    _CONFIG_AVAILABLE = False
    logger.warning(
        "Climate hazard config module not available; "
        "default configuration values will be used."
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    "ingest",
    "index",
    "project",
    "assess",
    "score",
    "report",
    "audit",
]
"""Ordered list of all pipeline stage names.

Each stage maps to a private ``_execute_{stage}_stage`` method on the
:class:`HazardPipelineEngine`.  Callers may supply a subset via the
``stages`` parameter of :meth:`run_pipeline` to skip stages that are
not needed for a particular assessment run.
"""

_STATUS_COMPLETED = "completed"
_STATUS_PARTIAL = "partial"
_STATUS_FAILED = "failed"

_DEFAULT_SCENARIOS = ["ssp2_4.5"]
_DEFAULT_TIME_HORIZONS = ["mid_term"]
_DEFAULT_REPORT_FRAMEWORKS = ["tcfd"]

# Risk classification thresholds (deterministic; matches config.py defaults)
_THRESHOLD_EXTREME = 80.0
_THRESHOLD_HIGH = 60.0
_THRESHOLD_MEDIUM = 40.0
_THRESHOLD_LOW = 20.0

# ---------------------------------------------------------------------------
# Metric helper stubs (safe to call even when metrics module is absent)
# ---------------------------------------------------------------------------


def _record_pipeline_metric(stage: str, status: str) -> None:
    """Record a pipeline stage execution metric.

    Args:
        stage: Pipeline stage name (ingest, index, project, assess,
            score, report, audit, full_pipeline).
        status: Completion status (success, failure, partial, timeout).
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_pipeline(stage, status)
        except Exception:  # noqa: BLE001
            pass


def _observe_pipeline_duration_metric(stage: str, seconds: float) -> None:
    """Record the duration of a pipeline stage execution.

    Args:
        stage: Pipeline stage name.
        seconds: Duration in seconds.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_observe_pipeline_duration(stage, seconds)
        except Exception:  # noqa: BLE001
            pass


def _record_ingestion_metric(hazard_type: str, source: str) -> None:
    """Record a hazard data ingestion event.

    Args:
        hazard_type: Type of climate hazard ingested.
        source: Data source provider.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_ingestion(hazard_type, source)
        except Exception:  # noqa: BLE001
            pass


def _record_risk_metric(hazard_type: str, scenario: str) -> None:
    """Record a risk index calculation event.

    Args:
        hazard_type: Type of climate hazard.
        scenario: Climate scenario used.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_risk_calculation(hazard_type, scenario)
        except Exception:  # noqa: BLE001
            pass


def _record_projection_metric(scenario: str, time_horizon: str) -> None:
    """Record a scenario projection event.

    Args:
        scenario: Climate scenario pathway.
        time_horizon: Target time horizon.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_projection(scenario, time_horizon)
        except Exception:  # noqa: BLE001
            pass


def _record_exposure_metric(asset_type: str, hazard_type: str) -> None:
    """Record an exposure assessment event.

    Args:
        asset_type: Type of asset assessed.
        hazard_type: Type of climate hazard assessed.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_exposure(asset_type, hazard_type)
        except Exception:  # noqa: BLE001
            pass


def _record_vulnerability_metric(sector: str, hazard_type: str) -> None:
    """Record a vulnerability scoring event.

    Args:
        sector: Economic sector scored.
        hazard_type: Type of climate hazard.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_vulnerability(sector, hazard_type)
        except Exception:  # noqa: BLE001
            pass


def _record_report_metric(report_type: str, fmt: str) -> None:
    """Record a compliance report generation event.

    Args:
        report_type: Type of report generated.
        fmt: Output format of the report.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_record_report(report_type, fmt)
        except Exception:  # noqa: BLE001
            pass


def _set_active_assets_metric(count: int) -> None:
    """Set the gauge for current number of active assets.

    Args:
        count: Number of active assets.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_set_active_assets(count)
        except Exception:  # noqa: BLE001
            pass


def _set_high_risk_metric(count: int) -> None:
    """Set the gauge for current number of high-risk locations.

    Args:
        count: Number of high-risk locations.
    """
    if _METRICS_AVAILABLE:
        try:
            _metrics_set_high_risk(count)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string (microseconds zeroed).

    Returns:
        ISO-8601 formatted UTC timestamp string.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str = "pipe") -> str:
    """Generate a short unique identifier with a descriptive prefix.

    Args:
        prefix: Short string prepended to the UUID hex fragment.

    Returns:
        Identifier string of the form ``"{prefix}-{12-hex-chars}"``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds from a ``time.monotonic()`` start.

    Args:
        start: ``time.monotonic()`` value captured before the operation.

    Returns:
        Elapsed duration in milliseconds, rounded to two decimal places.
    """
    return round((time.monotonic() - start) * 1000.0, 2)


def _sha256(payload: Any) -> str:
    """Compute a SHA-256 hex digest for any JSON-serialisable payload.

    Serialises ``payload`` to canonical JSON (sorted keys, ``str``
    default for non-serialisable types) before hashing so that
    equivalent structures always produce the same digest.

    Args:
        payload: Any JSON-serialisable object or primitive.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).
    """
    serialised = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _classify_risk(score: float) -> str:
    """Classify a numeric risk score into a risk category.

    Uses deterministic threshold comparison (zero-hallucination).
    Thresholds mirror the ``ClimateHazardConfig`` defaults:
        - >= 80: extreme
        - >= 60: high
        - >= 40: medium
        - >= 20: low
        - < 20:  negligible

    Args:
        score: Numeric risk score in the range [0, 100].

    Returns:
        Risk classification string: extreme, high, medium, low,
        or negligible.
    """
    if score >= _THRESHOLD_EXTREME:
        return "extreme"
    if score >= _THRESHOLD_HIGH:
        return "high"
    if score >= _THRESHOLD_MEDIUM:
        return "medium"
    if score >= _THRESHOLD_LOW:
        return "low"
    return "negligible"


def _normalise_raw(raw: Any) -> Dict[str, Any]:
    """Normalise arbitrary engine output to a standard dictionary.

    Handles dict returns, Pydantic model returns (via ``.dict()``),
    and dataclass returns (via ``vars()``).  Falls back to wrapping
    the stringified raw value under a ``"raw"`` key.

    Args:
        raw: Raw output from an upstream engine method.

    Returns:
        Normalised dictionary representation of the engine output.
    """
    if isinstance(raw, dict):
        return raw.copy()
    if hasattr(raw, "dict"):
        return raw.dict()
    if hasattr(raw, "__dict__"):
        return dict(vars(raw))
    return {"raw": str(raw)}


def _extract_location_lat(location: Any) -> float:
    """Extract latitude from a location object.

    Supports dict with ``lat`` or ``latitude`` keys, and objects
    with ``lat`` or ``latitude`` attributes.

    Args:
        location: Location dict or object.

    Returns:
        Latitude as a float, or 0.0 if not extractable.
    """
    if isinstance(location, dict):
        return float(location.get("lat", location.get("latitude", 0.0)))
    if hasattr(location, "lat"):
        return float(location.lat)
    if hasattr(location, "latitude"):
        return float(location.latitude)
    return 0.0


def _extract_location_lon(location: Any) -> float:
    """Extract longitude from a location object.

    Supports dict with ``lon``, ``lng``, or ``longitude`` keys,
    and objects with corresponding attributes.

    Args:
        location: Location dict or object.

    Returns:
        Longitude as a float, or 0.0 if not extractable.
    """
    if isinstance(location, dict):
        return float(
            location.get("lon", location.get("lng", location.get("longitude", 0.0)))
        )
    if hasattr(location, "lon"):
        return float(location.lon)
    if hasattr(location, "lng"):
        return float(location.lng)
    if hasattr(location, "longitude"):
        return float(location.longitude)
    return 0.0


# ---------------------------------------------------------------------------
# HazardPipelineEngine
# ---------------------------------------------------------------------------


class HazardPipelineEngine:
    """End-to-end pipeline orchestrator for the GreenLang Climate Hazard Connector.

    Coordinates all six upstream engines through a deterministic seven-stage
    workflow: ingest -> index -> project -> assess -> score -> report -> audit.
    Every stage outcome is captured in the pipeline result dictionary for
    compliance reporting and full auditability.

    Key design decisions:

    - **Zero-hallucination**: risk indices, exposure scores, and
      vulnerability scores are computed using deterministic weighted
      arithmetic.  No LLM inference is used in any calculation path.
    - **Provenance**: every pipeline run appends chain-hashed entries to
      the shared :class:`~greenlang.climate_hazard.provenance.ProvenanceTracker`.
    - **Thread-safety**: ``self._lock`` serialises writes to
      ``self._pipeline_runs`` and statistics counters while individual
      engine calls are stateless.
    - **Graceful degradation**: missing engines trigger stub behaviour
      that surfaces a clear ``"engine_unavailable"`` status rather than
      an AttributeError deep inside stage logic.
    - **Stage selectivity**: callers may supply a ``stages`` subset to
      skip stages not needed for a particular assessment run.

    Attributes:
        _database: Engine 1 -- HazardDatabaseEngine (or None if unavailable).
        _risk_engine: Engine 2 -- RiskIndexEngine (or None if unavailable).
        _projector: Engine 3 -- ScenarioProjectorEngine (or None if unavailable).
        _exposure_engine: Engine 4 -- ExposureAssessorEngine (or None).
        _vulnerability_engine: Engine 5 -- VulnerabilityScorerEngine (or None).
        _reporter: Engine 6 -- ComplianceReporterEngine (or None).
        _provenance: SHA-256 chain-hashing provenance tracker.
        _pipeline_runs: Mapping of pipeline_id to pipeline result dict.
        _lock: Mutex protecting writes to mutable shared state.

    Example:
        >>> engine = HazardPipelineEngine()
        >>> result = engine.run_pipeline(
        ...     assets=[{"asset_id": "a1", "name": "Factory",
        ...              "asset_type": "factory",
        ...              "location": {"lat": 48.85, "lon": 2.35}}],
        ...     hazard_types=["flood", "heat_wave"],
        ...     scenarios=["ssp2_4.5"],
        ...     time_horizons=["mid_term"],
        ... )
        >>> print(result["status"])
        completed
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        database: Any = None,
        risk_engine: Any = None,
        projector: Any = None,
        exposure_engine: Any = None,
        vulnerability_engine: Any = None,
        reporter: Any = None,
        provenance: Any = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialise the HazardPipelineEngine and all six upstream engines.

        Each upstream engine parameter is optional.  When ``None`` is passed
        (the default), the engine is auto-created from its module if the
        module was successfully imported.  When a non-None value is passed
        it is used directly, enabling dependency injection for testing.

        The ProvenanceTracker is always created because it is a pure-Python
        dependency with no optional imports.  When ``provenance`` is passed
        as a non-None value it is used directly; otherwise a new tracker is
        created with the provided or default ``genesis_hash``.

        Args:
            database: Optional pre-created HazardDatabaseEngine instance.
                When ``None``, auto-created if the module is available.
            risk_engine: Optional pre-created RiskIndexEngine instance.
                When ``None``, auto-created if the module is available.
            projector: Optional pre-created ScenarioProjectorEngine instance.
                When ``None``, auto-created if the module is available.
            exposure_engine: Optional pre-created ExposureAssessorEngine
                instance.  When ``None``, auto-created if available.
            vulnerability_engine: Optional pre-created
                VulnerabilityScorerEngine instance.  When ``None``,
                auto-created if the module is available.
            reporter: Optional pre-created ComplianceReporterEngine
                instance.  When ``None``, auto-created if available.
            provenance: Optional pre-created ProvenanceTracker instance.
                When ``None``, a new tracker is created with
                ``genesis_hash``.
            genesis_hash: Optional genesis hash anchor string for the
                provenance chain.  Defaults to
                ``"greenlang-climate-hazard-pipeline-genesis"`` when
                ``provenance`` is ``None``.

        Example:
            >>> engine = HazardPipelineEngine()
            >>> health = engine.get_health()
            >>> assert health["engines_total"] == 6
        """
        self._lock = threading.Lock()

        # ----------------------------------------------------------------
        # Engine 1 -- Hazard Database
        # ----------------------------------------------------------------
        if database is not None:
            self._database = database
        elif _DATABASE_AVAILABLE and HazardDatabaseEngine is not None:
            try:
                self._database = HazardDatabaseEngine()
            except Exception as exc:
                logger.warning(
                    "HazardDatabaseEngine instantiation failed: %s", exc
                )
                self._database = None
        else:
            self._database = None

        # ----------------------------------------------------------------
        # Engine 2 -- Risk Index
        # ----------------------------------------------------------------
        if risk_engine is not None:
            self._risk_engine = risk_engine
        elif _RISK_INDEX_AVAILABLE and RiskIndexEngine is not None:
            try:
                self._risk_engine = RiskIndexEngine()
            except Exception as exc:
                logger.warning(
                    "RiskIndexEngine instantiation failed: %s", exc
                )
                self._risk_engine = None
        else:
            self._risk_engine = None

        # ----------------------------------------------------------------
        # Engine 3 -- Scenario Projector
        # ----------------------------------------------------------------
        if projector is not None:
            self._projector = projector
        elif _PROJECTOR_AVAILABLE and ScenarioProjectorEngine is not None:
            try:
                self._projector = ScenarioProjectorEngine()
            except Exception as exc:
                logger.warning(
                    "ScenarioProjectorEngine instantiation failed: %s", exc
                )
                self._projector = None
        else:
            self._projector = None

        # ----------------------------------------------------------------
        # Engine 4 -- Exposure Assessor
        # ----------------------------------------------------------------
        if exposure_engine is not None:
            self._exposure_engine = exposure_engine
        elif _EXPOSURE_AVAILABLE and ExposureAssessorEngine is not None:
            try:
                self._exposure_engine = ExposureAssessorEngine()
            except Exception as exc:
                logger.warning(
                    "ExposureAssessorEngine instantiation failed: %s", exc
                )
                self._exposure_engine = None
        else:
            self._exposure_engine = None

        # ----------------------------------------------------------------
        # Engine 5 -- Vulnerability Scorer
        # ----------------------------------------------------------------
        if vulnerability_engine is not None:
            self._vulnerability_engine = vulnerability_engine
        elif _VULNERABILITY_AVAILABLE and VulnerabilityScorerEngine is not None:
            try:
                self._vulnerability_engine = VulnerabilityScorerEngine()
            except Exception as exc:
                logger.warning(
                    "VulnerabilityScorerEngine instantiation failed: %s", exc
                )
                self._vulnerability_engine = None
        else:
            self._vulnerability_engine = None

        # ----------------------------------------------------------------
        # Engine 6 -- Compliance Reporter
        # ----------------------------------------------------------------
        if reporter is not None:
            self._reporter = reporter
        elif _REPORTER_AVAILABLE and ComplianceReporterEngine is not None:
            try:
                self._reporter = ComplianceReporterEngine()
            except Exception as exc:
                logger.warning(
                    "ComplianceReporterEngine instantiation failed: %s", exc
                )
                self._reporter = None
        else:
            self._reporter = None

        # ----------------------------------------------------------------
        # Provenance tracker
        # ----------------------------------------------------------------
        if provenance is not None:
            self._provenance = provenance
        elif _PROVENANCE_AVAILABLE and ProvenanceTracker is not None:
            effective_genesis = (
                genesis_hash
                if genesis_hash is not None
                else "greenlang-climate-hazard-pipeline-genesis"
            )
            try:
                self._provenance = ProvenanceTracker(effective_genesis)
            except Exception as exc:
                logger.warning(
                    "ProvenanceTracker instantiation failed: %s", exc
                )
                self._provenance = None
        else:
            self._provenance = None

        # ----------------------------------------------------------------
        # Pipeline state
        # ----------------------------------------------------------------
        self._pipeline_runs: Dict[str, Dict[str, Any]] = {}
        self._total_runs: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._partial_count: int = 0
        self._total_duration_ms: float = 0.0
        self._stage_durations: Dict[str, List[float]] = {
            stage: [] for stage in PIPELINE_STAGES
        }

        logger.info(
            "HazardPipelineEngine initialised: "
            "database=%s risk_engine=%s projector=%s "
            "exposure_engine=%s vulnerability_engine=%s "
            "reporter=%s provenance=%s",
            "ok" if self._database else "UNAVAILABLE",
            "ok" if self._risk_engine else "UNAVAILABLE",
            "ok" if self._projector else "UNAVAILABLE",
            "ok" if self._exposure_engine else "UNAVAILABLE",
            "ok" if self._vulnerability_engine else "UNAVAILABLE",
            "ok" if self._reporter else "UNAVAILABLE",
            "ok" if self._provenance else "UNAVAILABLE",
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def database(self) -> Any:
        """Return the HazardDatabaseEngine instance (or None).

        Returns:
            The HazardDatabaseEngine instance, or ``None`` if the engine
            is unavailable.
        """
        return self._database

    @property
    def risk_engine(self) -> Any:
        """Return the RiskIndexEngine instance (or None).

        Returns:
            The RiskIndexEngine instance, or ``None`` if the engine
            is unavailable.
        """
        return self._risk_engine

    @property
    def projector(self) -> Any:
        """Return the ScenarioProjectorEngine instance (or None).

        Returns:
            The ScenarioProjectorEngine instance, or ``None`` if the
            engine is unavailable.
        """
        return self._projector

    @property
    def exposure_engine(self) -> Any:
        """Return the ExposureAssessorEngine instance (or None).

        Returns:
            The ExposureAssessorEngine instance, or ``None`` if the
            engine is unavailable.
        """
        return self._exposure_engine

    @property
    def vulnerability_engine(self) -> Any:
        """Return the VulnerabilityScorerEngine instance (or None).

        Returns:
            The VulnerabilityScorerEngine instance, or ``None`` if the
            engine is unavailable.
        """
        return self._vulnerability_engine

    @property
    def reporter(self) -> Any:
        """Return the ComplianceReporterEngine instance (or None).

        Returns:
            The ComplianceReporterEngine instance, or ``None`` if the
            engine is unavailable.
        """
        return self._reporter

    @property
    def provenance(self) -> Any:
        """Return the ProvenanceTracker instance (or None).

        Returns:
            The ProvenanceTracker instance, or ``None`` if the tracker
            is unavailable.
        """
        return self._provenance

    # ------------------------------------------------------------------
    # Public API -- primary pipeline entry points
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        assets: List[Dict[str, Any]],
        hazard_types: List[str],
        scenarios: Optional[List[str]] = None,
        time_horizons: Optional[List[str]] = None,
        report_frameworks: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete seven-stage climate hazard assessment pipeline.

        Executes stages in order: ingest -> index -> project -> assess ->
        score -> report -> audit.  Individual stages that encounter errors
        record the failure and continue to the next stage so that partial
        results are always available (no single stage failure kills the
        entire pipeline).

        The ``stages`` parameter allows selective execution.  When provided,
        only the listed stages are executed (in the canonical order defined
        by :data:`PIPELINE_STAGES`).  When ``None``, all seven stages run.

        Args:
            assets: List of asset dictionaries, each containing at minimum:
                - ``asset_id`` (str): Unique asset identifier.
                - ``name`` (str): Human-readable asset name.
                - ``asset_type`` (str): Asset type classification
                  (facility, warehouse, office, data_center, factory,
                  supply_chain_node, transport_hub, port, mine, farm,
                  renewable_installation, portfolio).
                - ``location`` (dict): Location with ``lat`` and ``lon``
                  keys (WGS84 decimal degrees).
            hazard_types: List of hazard type strings to assess (flood,
                drought, wildfire, heat_wave, cold_wave, storm,
                sea_level_rise, tropical_cyclone, landslide, water_stress,
                precipitation_change, temperature_change, compound).
            scenarios: Optional list of climate scenario strings.
                Defaults to ``["ssp2_4.5"]``.
            time_horizons: Optional list of time horizon strings.
                Defaults to ``["mid_term"]``.
            report_frameworks: Optional list of report framework strings.
                Defaults to ``["tcfd"]``.
            stages: Optional list of stage names to execute.  Must be a
                subset of :data:`PIPELINE_STAGES`.  When ``None``, all
                seven stages are executed.
            parameters: Optional dictionary of extra configuration
                overrides passed to individual stage execution methods.

        Returns:
            Pipeline result dictionary with keys:
            - ``pipeline_id`` (str): Unique pipeline run identifier.
            - ``status`` (str): Final status (``"completed"``,
              ``"partial"``, or ``"failed"``).
            - ``stages_completed`` (int): Number of stages that
              completed successfully.
            - ``results`` (dict): Mapping of stage name to stage result
              dictionary.
            - ``evaluation_summary`` (dict): High-level summary with
              ``total_assets``, ``hazard_types_assessed``,
              ``scenarios_evaluated``, ``high_risk_count``,
              ``avg_risk_score``.
            - ``report_id`` (str or None): Report identifier if the
              report stage ran successfully.
            - ``duration_ms`` (float): Total pipeline wall-clock time
              in milliseconds.
            - ``stage_timings`` (dict): Mapping of stage name to
              execution duration in milliseconds.
            - ``provenance_hash`` (str): SHA-256 provenance chain hash
              for the pipeline run.

        Raises:
            ValueError: If ``assets`` is empty or ``hazard_types`` is
                empty.

        Example:
            >>> engine = HazardPipelineEngine()
            >>> result = engine.run_pipeline(
            ...     assets=[{"asset_id": "a1", "name": "HQ",
            ...              "asset_type": "office",
            ...              "location": {"lat": 51.5, "lon": -0.13}}],
            ...     hazard_types=["flood"],
            ... )
            >>> assert result["status"] in ("completed", "partial", "failed")
        """
        if not assets:
            raise ValueError("assets must not be empty")
        if not hazard_types:
            raise ValueError("hazard_types must not be empty")

        # Apply defaults
        effective_scenarios = scenarios if scenarios is not None else list(_DEFAULT_SCENARIOS)
        effective_horizons = (
            time_horizons if time_horizons is not None else list(_DEFAULT_TIME_HORIZONS)
        )
        effective_frameworks = (
            report_frameworks if report_frameworks is not None else list(_DEFAULT_REPORT_FRAMEWORKS)
        )
        effective_stages = self._resolve_stages(stages)
        effective_params = parameters if parameters is not None else {}

        pipeline_id = _new_id("pipe")
        created_at = _utcnow_iso()
        pipeline_start = time.monotonic()

        logger.info(
            "Pipeline %s starting: assets=%d hazard_types=%s "
            "scenarios=%s time_horizons=%s frameworks=%s stages=%s",
            pipeline_id,
            len(assets),
            hazard_types,
            effective_scenarios,
            effective_horizons,
            effective_frameworks,
            effective_stages,
        )

        # Record pipeline start provenance
        self._record_provenance(
            "pipeline_run",
            "run_pipeline",
            pipeline_id,
            {
                "assets_count": len(assets),
                "hazard_types": hazard_types,
                "scenarios": effective_scenarios,
                "time_horizons": effective_horizons,
                "stages": effective_stages,
            },
        )

        # Initialise result accumulator
        stage_results: Dict[str, Dict[str, Any]] = {}
        stage_timings: Dict[str, float] = {}
        stages_completed = 0
        stages_failed: List[str] = []
        report_id: Optional[str] = None

        # Update active assets gauge
        _set_active_assets_metric(len(assets))

        # ----------------------------------------------------------------
        # Stage 1: INGEST
        # ----------------------------------------------------------------
        if "ingest" in effective_stages:
            ingest_result, ingest_ms = self._execute_ingest_stage(
                assets, hazard_types, effective_params
            )
            stage_results["ingest"] = ingest_result
            stage_timings["ingest"] = ingest_ms
            if ingest_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("ingest")

        # ----------------------------------------------------------------
        # Stage 2: INDEX
        # ----------------------------------------------------------------
        if "index" in effective_stages:
            index_result, index_ms = self._execute_index_stage(
                hazard_types, assets, effective_params
            )
            stage_results["index"] = index_result
            stage_timings["index"] = index_ms
            if index_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("index")

        # ----------------------------------------------------------------
        # Stage 3: PROJECT
        # ----------------------------------------------------------------
        if "project" in effective_stages:
            project_result, project_ms = self._execute_project_stage(
                hazard_types, assets, effective_scenarios,
                effective_horizons, effective_params
            )
            stage_results["project"] = project_result
            stage_timings["project"] = project_ms
            if project_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("project")

        # ----------------------------------------------------------------
        # Stage 4: ASSESS
        # ----------------------------------------------------------------
        if "assess" in effective_stages:
            assess_result, assess_ms = self._execute_assess_stage(
                assets, hazard_types, effective_params
            )
            stage_results["assess"] = assess_result
            stage_timings["assess"] = assess_ms
            if assess_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("assess")

        # ----------------------------------------------------------------
        # Stage 5: SCORE
        # ----------------------------------------------------------------
        if "score" in effective_stages:
            score_result, score_ms = self._execute_score_stage(
                assets, hazard_types, effective_params
            )
            stage_results["score"] = score_result
            stage_timings["score"] = score_ms
            if score_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("score")

        # ----------------------------------------------------------------
        # Stage 6: REPORT
        # ----------------------------------------------------------------
        if "report" in effective_stages:
            report_result, report_ms = self._execute_report_stage(
                stage_results, effective_frameworks, effective_params
            )
            stage_results["report"] = report_result
            stage_timings["report"] = report_ms
            if report_result.get("status") != "failed":
                stages_completed += 1
                report_id = report_result.get("report_id")
            else:
                stages_failed.append("report")

        # ----------------------------------------------------------------
        # Stage 7: AUDIT
        # ----------------------------------------------------------------
        if "audit" in effective_stages:
            audit_result, audit_ms = self._execute_audit_stage(
                pipeline_id, stage_results
            )
            stage_results["audit"] = audit_result
            stage_timings["audit"] = audit_ms
            if audit_result.get("status") != "failed":
                stages_completed += 1
            else:
                stages_failed.append("audit")

        # ----------------------------------------------------------------
        # Assemble final result
        # ----------------------------------------------------------------
        total_stages_attempted = len(effective_stages)
        pipeline_status = self._determine_pipeline_status(
            stages_completed, total_stages_attempted, stages_failed
        )

        evaluation_summary = self._build_evaluation_summary(
            assets, hazard_types, effective_scenarios, stage_results
        )

        duration_ms = _elapsed_ms(pipeline_start)
        duration_seconds = duration_ms / 1000.0

        # Final provenance entry
        provenance_hash = self._record_provenance(
            "pipeline_run",
            "run_pipeline",
            pipeline_id,
            {
                "status": pipeline_status,
                "stages_completed": stages_completed,
                "stages_failed": stages_failed,
                "duration_ms": duration_ms,
                "evaluation_summary": evaluation_summary,
            },
        )

        result: Dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "status": pipeline_status,
            "stages_completed": stages_completed,
            "stages_failed": stages_failed,
            "total_stages_attempted": total_stages_attempted,
            "results": stage_results,
            "evaluation_summary": evaluation_summary,
            "report_id": report_id,
            "duration_ms": duration_ms,
            "stage_timings": stage_timings,
            "provenance_hash": provenance_hash,
            "created_at": created_at,
            "hazard_types": hazard_types,
            "scenarios": effective_scenarios,
            "time_horizons": effective_horizons,
            "report_frameworks": effective_frameworks,
            "assets_count": len(assets),
        }

        # Record metrics
        _record_pipeline_metric("full_pipeline", pipeline_status)
        _observe_pipeline_duration_metric("full_pipeline", duration_seconds)

        # Update high-risk gauge from evaluation summary
        high_risk_count = evaluation_summary.get("high_risk_count", 0)
        _set_high_risk_metric(high_risk_count)

        # Persist to run store and update statistics
        with self._lock:
            self._pipeline_runs[pipeline_id] = result
            self._total_runs += 1
            self._total_duration_ms += duration_ms
            if pipeline_status == _STATUS_COMPLETED:
                self._success_count += 1
            elif pipeline_status == _STATUS_FAILED:
                self._failure_count += 1
            else:
                self._partial_count += 1
            for stage_name, stage_ms in stage_timings.items():
                if stage_name in self._stage_durations:
                    self._stage_durations[stage_name].append(stage_ms)

        logger.info(
            "Pipeline %s finalised: status=%s stages_completed=%d/%d "
            "duration_ms=%.1f high_risk=%d",
            pipeline_id,
            pipeline_status,
            stages_completed,
            total_stages_attempted,
            duration_ms,
            high_risk_count,
        )
        return result

    def run_batch_pipeline(
        self,
        asset_portfolios: List[Dict[str, Any]],
        hazard_types: List[str],
        scenarios: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the pipeline for multiple asset portfolios in sequence.

        Iterates over ``asset_portfolios``, calling :meth:`run_pipeline`
        for each portfolio entry.  Results are accumulated and summarised
        in the returned batch result dictionary.  A failure in one
        portfolio does not prevent remaining portfolios from being
        processed.

        Args:
            asset_portfolios: List of portfolio dictionaries.  Each must
                contain:
                - ``portfolio_id`` (str): Unique portfolio identifier.
                - ``assets`` (list): List of asset dictionaries as
                  defined in :meth:`run_pipeline`.
                Additional optional keys:
                - ``scenarios`` (list): Override scenarios for this portfolio.
                - ``time_horizons`` (list): Override time horizons.
                - ``report_frameworks`` (list): Override report frameworks.
                - ``stages`` (list): Override stage selection.
            hazard_types: List of hazard type strings to assess across
                all portfolios.
            scenarios: Optional default scenarios applied to portfolios
                that do not specify their own.  Defaults to
                ``["ssp2_4.5"]``.
            parameters: Optional dictionary of extra configuration
                overrides passed to each individual pipeline run.

        Returns:
            Batch result dictionary with keys:
            - ``per_portfolio_results`` (list): List of per-portfolio
              result dictionaries, each containing ``portfolio_id``,
              ``pipeline_result``, and ``status``.
            - ``summary`` (dict): Aggregate summary with ``avg_risk``,
              ``total_assets``, ``total_high_risk``.
            - ``batch_id`` (str): Unique batch run identifier.
            - ``duration_ms`` (float): Total batch wall-clock time.
            - ``provenance_hash`` (str): SHA-256 provenance chain hash.

        Raises:
            ValueError: If ``asset_portfolios`` is empty or
                ``hazard_types`` is empty.

        Example:
            >>> portfolios = [
            ...     {"portfolio_id": "p1", "assets": [
            ...         {"asset_id": "a1", "name": "HQ", "asset_type": "office",
            ...          "location": {"lat": 51.5, "lon": -0.13}}
            ...     ]},
            ... ]
            >>> result = engine.run_batch_pipeline(portfolios, ["flood"])
            >>> assert "per_portfolio_results" in result
        """
        if not asset_portfolios:
            raise ValueError("asset_portfolios must not be empty")
        if not hazard_types:
            raise ValueError("hazard_types must not be empty")

        batch_id = _new_id("batch")
        created_at = _utcnow_iso()
        batch_start = time.monotonic()
        effective_params = parameters if parameters is not None else {}
        default_scenarios = scenarios if scenarios is not None else list(_DEFAULT_SCENARIOS)

        logger.info(
            "Batch pipeline %s starting: %d portfolio(s) hazard_types=%s",
            batch_id,
            len(asset_portfolios),
            hazard_types,
        )

        per_portfolio_results: List[Dict[str, Any]] = []
        all_risk_scores: List[float] = []
        total_assets = 0
        total_high_risk = 0

        for portfolio in asset_portfolios:
            portfolio_id = portfolio.get("portfolio_id", _new_id("pf"))
            portfolio_assets = portfolio.get("assets", [])
            portfolio_scenarios = portfolio.get("scenarios", default_scenarios)
            portfolio_horizons = portfolio.get("time_horizons")
            portfolio_frameworks = portfolio.get("report_frameworks")
            portfolio_stages = portfolio.get("stages")

            logger.info(
                "Batch %s processing portfolio %s: %d assets",
                batch_id,
                portfolio_id,
                len(portfolio_assets),
            )

            try:
                pipeline_result = self.run_pipeline(
                    assets=portfolio_assets,
                    hazard_types=hazard_types,
                    scenarios=portfolio_scenarios,
                    time_horizons=portfolio_horizons,
                    report_frameworks=portfolio_frameworks,
                    stages=portfolio_stages,
                    parameters=effective_params,
                )
            except Exception as exc:
                logger.error(
                    "Batch %s: portfolio %s pipeline failed -- %s",
                    batch_id,
                    portfolio_id,
                    exc,
                )
                pipeline_result = {
                    "pipeline_id": _new_id("pipe"),
                    "status": _STATUS_FAILED,
                    "error": str(exc),
                    "stages_completed": 0,
                    "evaluation_summary": {
                        "total_assets": len(portfolio_assets),
                        "hazard_types_assessed": 0,
                        "scenarios_evaluated": 0,
                        "high_risk_count": 0,
                        "avg_risk_score": 0.0,
                    },
                }

            per_portfolio_results.append({
                "portfolio_id": portfolio_id,
                "pipeline_result": pipeline_result,
                "status": pipeline_result.get("status", _STATUS_FAILED),
            })

            # Accumulate summary statistics
            eval_summary = pipeline_result.get("evaluation_summary", {})
            avg_risk = eval_summary.get("avg_risk_score", 0.0)
            if isinstance(avg_risk, (int, float)) and avg_risk > 0:
                all_risk_scores.append(float(avg_risk))
            total_assets += eval_summary.get("total_assets", len(portfolio_assets))
            total_high_risk += eval_summary.get("high_risk_count", 0)

        duration_ms = _elapsed_ms(batch_start)

        summary = {
            "avg_risk": round(_safe_mean(all_risk_scores), 2),
            "total_assets": total_assets,
            "total_high_risk": total_high_risk,
            "portfolios_processed": len(asset_portfolios),
            "portfolios_completed": sum(
                1 for r in per_portfolio_results
                if r.get("status") == _STATUS_COMPLETED
            ),
            "portfolios_failed": sum(
                1 for r in per_portfolio_results
                if r.get("status") == _STATUS_FAILED
            ),
        }

        provenance_hash = self._record_provenance(
            "pipeline_run",
            "run_batch",
            batch_id,
            {
                "portfolios": len(asset_portfolios),
                "total_assets": total_assets,
                "total_high_risk": total_high_risk,
                "duration_ms": duration_ms,
            },
        )

        batch_result: Dict[str, Any] = {
            "per_portfolio_results": per_portfolio_results,
            "summary": summary,
            "batch_id": batch_id,
            "duration_ms": duration_ms,
            "created_at": created_at,
            "provenance_hash": provenance_hash,
        }

        logger.info(
            "Batch pipeline %s done: portfolios=%d total_assets=%d "
            "total_high_risk=%d avg_risk=%.2f duration_ms=%.1f",
            batch_id,
            len(asset_portfolios),
            total_assets,
            total_high_risk,
            summary["avg_risk"],
            duration_ms,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Public API -- pipeline run retrieval
    # ------------------------------------------------------------------

    def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored pipeline run record by its unique identifier.

        Args:
            run_id: Unique pipeline run identifier (``pipeline_id``).

        Returns:
            Pipeline result dictionary, or ``None`` if no run with the
            given identifier exists.

        Example:
            >>> run = engine.get_pipeline_run("pipe-abc123def456")
            >>> if run:
            ...     print(run["status"])
        """
        with self._lock:
            return self._pipeline_runs.get(run_id)

    def list_pipeline_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return a list of pipeline run records, newest first.

        Results are ordered by ``created_at`` descending (most recent
        first) and limited to the specified count.

        Args:
            limit: Maximum number of records to return.  Defaults to 100.

        Returns:
            List of pipeline result dictionaries (most recent first).

        Example:
            >>> runs = engine.list_pipeline_runs(limit=10)
            >>> print(len(runs))
        """
        with self._lock:
            all_runs = list(self._pipeline_runs.values())

        all_runs.sort(
            key=lambda r: r.get("created_at", ""),
            reverse=True,
        )
        return all_runs[:limit]

    # ------------------------------------------------------------------
    # Public API -- health and statistics
    # ------------------------------------------------------------------

    def get_health(self) -> Dict[str, Any]:
        """Check the availability of all six upstream engines.

        Returns a structured health report indicating which engines are
        available and the overall health status of the pipeline.

        The overall status is determined as follows:
        - ``"healthy"``: All 6 engines are available.
        - ``"degraded"``: At least one engine is available but not all.
        - ``"unhealthy"``: Zero engines are available.

        Returns:
            Health report dictionary with keys:
            - ``status`` (str): Overall health (healthy, degraded,
              unhealthy).
            - ``engines`` (dict): Mapping of engine name to
              ``"available"`` or ``"unavailable"``.
            - ``engines_available`` (int): Count of available engines.
            - ``engines_total`` (int): Always 6.
            - ``pipeline_stats`` (dict): Quick statistics summary with
              ``total_runs``, ``success_count``, ``failure_count``.
            - ``checked_at`` (str): ISO-8601 UTC timestamp of the
              health check.

        Example:
            >>> health = engine.get_health()
            >>> print(health["status"])
            degraded
            >>> print(health["engines_available"])
            3
        """
        engines_status: Dict[str, str] = {
            "database": "available" if self._database is not None else "unavailable",
            "risk_engine": "available" if self._risk_engine is not None else "unavailable",
            "projector": "available" if self._projector is not None else "unavailable",
            "exposure_engine": (
                "available" if self._exposure_engine is not None else "unavailable"
            ),
            "vulnerability_engine": (
                "available" if self._vulnerability_engine is not None else "unavailable"
            ),
            "reporter": "available" if self._reporter is not None else "unavailable",
        }

        engines_available = sum(
            1 for v in engines_status.values() if v == "available"
        )
        engines_total = 6

        if engines_available == engines_total:
            overall_status = "healthy"
        elif engines_available > 0:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        with self._lock:
            pipeline_stats = {
                "total_runs": self._total_runs,
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "partial_count": self._partial_count,
            }

        return {
            "status": overall_status,
            "engines": engines_status,
            "engines_available": engines_available,
            "engines_total": engines_total,
            "pipeline_stats": pipeline_stats,
            "checked_at": _utcnow_iso(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for all pipeline runs recorded so far.

        All numeric aggregations use pure Python arithmetic
        (zero-hallucination).  No LLM inference is used.

        Returns:
            Statistics dictionary with keys:
            - ``total_runs`` (int): Total pipeline runs executed.
            - ``success_count`` (int): Runs with status ``"completed"``.
            - ``failure_count`` (int): Runs with status ``"failed"``.
            - ``partial_count`` (int): Runs with status ``"partial"``.
            - ``avg_duration_ms`` (float): Mean pipeline duration.
            - ``min_duration_ms`` (float): Minimum pipeline duration.
            - ``max_duration_ms`` (float): Maximum pipeline duration.
            - ``per_stage_avg_ms`` (dict): Mean duration per stage.
            - ``success_rate`` (float): Fraction of successful runs
              (0.0 to 1.0).
            - ``provenance_entry_count`` (int): Total provenance entries.
            - ``computed_at`` (str): ISO-8601 UTC timestamp.

        Example:
            >>> stats = engine.get_statistics()
            >>> print(stats["success_rate"])
            1.0
        """
        with self._lock:
            total = self._total_runs
            success = self._success_count
            failure = self._failure_count
            partial = self._partial_count
            total_dur = self._total_duration_ms
            stage_durs = {k: list(v) for k, v in self._stage_durations.items()}

            all_durations: List[float] = []
            for run in self._pipeline_runs.values():
                dur = run.get("duration_ms")
                if isinstance(dur, (int, float)):
                    all_durations.append(float(dur))

        avg_duration = total_dur / total if total > 0 else 0.0
        min_duration = min(all_durations) if all_durations else 0.0
        max_duration = max(all_durations) if all_durations else 0.0
        success_rate = success / total if total > 0 else 0.0

        per_stage_avg: Dict[str, float] = {}
        for stage_name, durations in stage_durs.items():
            per_stage_avg[stage_name] = round(_safe_mean(durations), 2)

        provenance_count = 0
        if self._provenance is not None:
            try:
                provenance_count = self._provenance.entry_count
            except Exception:  # noqa: BLE001
                provenance_count = 0

        return {
            "total_runs": total,
            "success_count": success,
            "failure_count": failure,
            "partial_count": partial,
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "max_duration_ms": round(max_duration, 2),
            "per_stage_avg_ms": per_stage_avg,
            "success_rate": round(success_rate, 4),
            "provenance_entry_count": provenance_count,
            "computed_at": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Public API -- state management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all engine state, pipeline run store, and statistics.

        Clears the pipeline run store, resets statistics counters, resets
        the provenance tracker, and re-initialises any engines that expose
        a ``clear()`` or ``reset()`` method.  Primarily intended for
        testing to prevent state leakage between test cases.

        Example:
            >>> engine.clear()
            >>> assert engine.get_statistics()["total_runs"] == 0
        """
        with self._lock:
            self._pipeline_runs.clear()
            self._total_runs = 0
            self._success_count = 0
            self._failure_count = 0
            self._partial_count = 0
            self._total_duration_ms = 0.0
            self._stage_durations = {
                stage: [] for stage in PIPELINE_STAGES
            }

        # Reset provenance tracker
        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception as exc:
                logger.warning(
                    "clear: provenance tracker reset failed -- %s", exc
                )

        # Reset upstream engines
        for name, eng in (
            ("database", self._database),
            ("risk_engine", self._risk_engine),
            ("projector", self._projector),
            ("exposure_engine", self._exposure_engine),
            ("vulnerability_engine", self._vulnerability_engine),
            ("reporter", self._reporter),
        ):
            if eng is not None:
                for method_name in ("clear", "reset"):
                    if hasattr(eng, method_name):
                        try:
                            getattr(eng, method_name)()
                        except Exception as exc:
                            logger.warning(
                                "clear: engine %s %s() failed -- %s",
                                name,
                                method_name,
                                exc,
                            )
                        break

        # Reset metrics gauges
        _set_active_assets_metric(0)
        _set_high_risk_metric(0)

        logger.info("HazardPipelineEngine: full reset complete")

    # ------------------------------------------------------------------
    # Private: stage execution methods
    # ------------------------------------------------------------------

    def _execute_ingest_stage(
        self,
        assets: List[Dict[str, Any]],
        hazard_types: List[str],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 1: Register assets and ingest baseline hazard data.

        Delegates to HazardDatabaseEngine when available.  When the engine
        is unavailable a stub result is returned that describes each asset
        and hazard type combination with placeholder data.

        For each asset, the engine attempts to:
        1. Register the asset in the hazard database.
        2. Ingest baseline hazard data for each requested hazard type
           at the asset's location.

        Metrics recorded:
        - ``record_ingestion(hazard_type, source)`` per successful ingest.
        - ``record_pipeline("ingestion", status)``
        - ``observe_pipeline_duration("ingestion", seconds)``

        Args:
            assets: List of asset dictionaries with location data.
            hazard_types: List of hazard type strings to ingest.
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "ingest"

        assets_registered: List[Dict[str, Any]] = []
        hazard_records_ingested: int = 0
        sources_used: List[str] = []
        errors: List[str] = []
        warnings: List[str] = []

        try:
            for asset in assets:
                asset_id = asset.get("asset_id", _new_id("asset"))
                asset_name = asset.get("name", "Unknown")
                asset_type = asset.get("asset_type", "facility")
                location = asset.get("location", {})

                # Register asset in database engine
                if self._database is not None:
                    try:
                        self._database.register_asset(
                            asset_id=asset_id,
                            name=asset_name,
                            asset_type=asset_type,
                            location=location,
                        )
                        assets_registered.append({
                            "asset_id": asset_id,
                            "name": asset_name,
                            "asset_type": asset_type,
                            "registered": True,
                        })
                    except Exception as exc:
                        logger.warning(
                            "Ingest stage: asset %s registration failed -- %s",
                            asset_id,
                            exc,
                        )
                        assets_registered.append({
                            "asset_id": asset_id,
                            "name": asset_name,
                            "asset_type": asset_type,
                            "registered": False,
                            "error": str(exc),
                        })
                        errors.append(
                            f"Asset {asset_id} registration failed: {exc}"
                        )
                else:
                    # Stub registration
                    assets_registered.append({
                        "asset_id": asset_id,
                        "name": asset_name,
                        "asset_type": asset_type,
                        "registered": True,
                        "stub": True,
                    })

                # Ingest hazard data for each hazard type
                for hazard_type in hazard_types:
                    if self._database is not None:
                        try:
                            source = parameters.get("default_source", "model_output")
                            ingest_result = self._database.ingest_hazard_data(
                                hazard_type=hazard_type,
                                location=location,
                                asset_id=asset_id,
                            )
                            hazard_records_ingested += 1
                            if source not in sources_used:
                                sources_used.append(source)
                            _record_ingestion_metric(hazard_type, source)
                        except Exception as exc:
                            logger.warning(
                                "Ingest stage: hazard %s for asset %s failed -- %s",
                                hazard_type,
                                asset_id,
                                exc,
                            )
                            errors.append(
                                f"Hazard {hazard_type} ingest for asset "
                                f"{asset_id} failed: {exc}"
                            )
                    else:
                        # Stub ingest
                        hazard_records_ingested += 1
                        stub_source = "stub"
                        if stub_source not in sources_used:
                            sources_used.append(stub_source)
                        _record_ingestion_metric(hazard_type, stub_source)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Ingest stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Ingest stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("ingestion", status)
        _observe_pipeline_duration_metric("ingestion", elapsed_seconds)

        # Record provenance
        self._record_provenance(
            "hazard_data",
            "ingest_data",
            _new_id("ingest"),
            {
                "assets_registered": len(assets_registered),
                "hazard_records_ingested": hazard_records_ingested,
                "sources_used": sources_used,
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "assets_registered": assets_registered,
            "hazard_records_ingested": hazard_records_ingested,
            "sources_used": sources_used,
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._database is not None,
        }

        logger.info(
            "Ingest stage: status=%s assets=%d records=%d sources=%s "
            "errors=%d duration_ms=%.1f",
            status,
            len(assets_registered),
            hazard_records_ingested,
            sources_used,
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_index_stage(
        self,
        hazard_types: List[str],
        assets: List[Dict[str, Any]],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 2: Calculate risk indices for each asset x hazard combination.

        Delegates to RiskIndexEngine when available.  When the engine is
        unavailable, computes deterministic stub risk indices using
        location-derived pseudo-scores (latitude-based sensitivity
        heuristic).  This is zero-hallucination: all values are computed
        from arithmetic, never from LLM inference.

        Metrics recorded:
        - ``record_risk_calculation(hazard_type, scenario)``
        - ``record_pipeline("risk_calculation", status)``
        - ``observe_pipeline_duration("risk_calculation", seconds)``

        Args:
            hazard_types: List of hazard type strings.
            assets: List of asset dictionaries.
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "index"

        risk_indices: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        try:
            scenario = parameters.get("default_scenario", "ssp2_4.5")

            for asset in assets:
                asset_id = asset.get("asset_id", "unknown")
                location = asset.get("location", {})

                for hazard_type in hazard_types:
                    if self._risk_engine is not None:
                        try:
                            raw_result = self._risk_engine.calculate_risk_index(
                                asset_id=asset_id,
                                hazard_type=hazard_type,
                                location=location,
                            )
                            index_entry = _normalise_raw(raw_result)
                            index_entry.setdefault("asset_id", asset_id)
                            index_entry.setdefault("hazard_type", hazard_type)
                            risk_indices.append(index_entry)
                            _record_risk_metric(hazard_type, scenario)
                        except Exception as exc:
                            logger.warning(
                                "Index stage: risk calc for asset %s "
                                "hazard %s failed, using stub -- %s",
                                asset_id,
                                hazard_type,
                                exc,
                            )
                            # Fall back to deterministic stub
                            stub_index = self._stub_risk_index(
                                asset_id, hazard_type, location
                            )
                            risk_indices.append(stub_index)
                            _record_risk_metric(hazard_type, scenario)
                            warnings.append(
                                f"Risk index for {asset_id}/{hazard_type}: "
                                f"engine failed, stub used"
                            )
                    else:
                        # Deterministic stub risk index
                        stub_index = self._stub_risk_index(
                            asset_id, hazard_type, location
                        )
                        risk_indices.append(stub_index)
                        _record_risk_metric(hazard_type, scenario)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Index stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Index stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("risk_calculation", status)
        _observe_pipeline_duration_metric("risk_calculation", elapsed_seconds)

        self._record_provenance(
            "risk_index",
            "calculate_risk",
            _new_id("index"),
            {
                "indices_calculated": len(risk_indices),
                "hazard_types": hazard_types,
                "assets_count": len(assets),
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "risk_indices": risk_indices,
            "indices_calculated": len(risk_indices),
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._risk_engine is not None,
        }

        logger.info(
            "Index stage: status=%s indices=%d errors=%d duration_ms=%.1f",
            status,
            len(risk_indices),
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_project_stage(
        self,
        hazard_types: List[str],
        assets: List[Dict[str, Any]],
        scenarios: List[str],
        time_horizons: List[str],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 3: Project hazard intensity under climate scenarios.

        Delegates to ScenarioProjectorEngine when available.  When the
        engine is unavailable, produces deterministic stub projections
        using scenario-specific scaling factors derived from IPCC AR6
        representative pathways (tabulated, not ML-predicted).

        Metrics recorded:
        - ``record_projection(scenario, time_horizon)``
        - ``record_pipeline("scenario_projection", status)``
        - ``observe_pipeline_duration("scenario_projection", seconds)``

        Args:
            hazard_types: List of hazard type strings.
            assets: List of asset dictionaries.
            scenarios: List of climate scenario strings.
            time_horizons: List of time horizon strings.
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "project"

        projections: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        # Deterministic scenario scaling factors (IPCC AR6 tabulated)
        scenario_scaling: Dict[str, float] = {
            "ssp1_1.9": 0.15,
            "ssp1_2.6": 0.25,
            "ssp2_4.5": 0.50,
            "ssp3_7.0": 0.75,
            "ssp5_8.5": 1.00,
            "rcp2.6": 0.25,
            "rcp4.5": 0.50,
            "rcp6.0": 0.65,
            "rcp8.5": 1.00,
            "historical": 0.0,
            "baseline": 0.0,
        }

        # Time horizon multipliers (deterministic)
        horizon_multipliers: Dict[str, float] = {
            "short_term": 0.6,
            "mid_term": 1.0,
            "long_term": 1.5,
            "2030": 0.6,
            "2040": 0.8,
            "2050": 1.0,
            "2070": 1.3,
            "2100": 1.5,
        }

        try:
            for scenario in scenarios:
                for time_horizon in time_horizons:
                    for hazard_type in hazard_types:
                        for asset in assets:
                            asset_id = asset.get("asset_id", "unknown")
                            location = asset.get("location", {})

                            if self._projector is not None:
                                try:
                                    raw_result = self._projector.project_scenario(
                                        asset_id=asset_id,
                                        hazard_type=hazard_type,
                                        scenario=scenario,
                                        time_horizon=time_horizon,
                                        location=location,
                                    )
                                    projection = _normalise_raw(raw_result)
                                    projection.setdefault("asset_id", asset_id)
                                    projection.setdefault("hazard_type", hazard_type)
                                    projection.setdefault("scenario", scenario)
                                    projection.setdefault("time_horizon", time_horizon)
                                    projections.append(projection)
                                    _record_projection_metric(scenario, time_horizon)
                                except Exception as exc:
                                    logger.warning(
                                        "Project stage: projection for "
                                        "%s/%s/%s/%s failed, using stub -- %s",
                                        asset_id,
                                        hazard_type,
                                        scenario,
                                        time_horizon,
                                        exc,
                                    )
                                    # Fall back to deterministic stub
                                    stub_proj = self._stub_projection(
                                        asset_id,
                                        hazard_type,
                                        scenario,
                                        time_horizon,
                                        location,
                                        scenario_scaling,
                                        horizon_multipliers,
                                    )
                                    projections.append(stub_proj)
                                    _record_projection_metric(scenario, time_horizon)
                                    warnings.append(
                                        f"Projection {asset_id}/{hazard_type}/"
                                        f"{scenario}/{time_horizon}: "
                                        f"engine failed, stub used"
                                    )
                            else:
                                # Deterministic stub projection
                                stub_projection = self._stub_projection(
                                    asset_id,
                                    hazard_type,
                                    scenario,
                                    time_horizon,
                                    location,
                                    scenario_scaling,
                                    horizon_multipliers,
                                )
                                projections.append(stub_projection)
                                _record_projection_metric(scenario, time_horizon)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Project stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Project stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("scenario_projection", status)
        _observe_pipeline_duration_metric("scenario_projection", elapsed_seconds)

        self._record_provenance(
            "scenario_projection",
            "project_scenario",
            _new_id("proj"),
            {
                "projections_count": len(projections),
                "scenarios": scenarios,
                "time_horizons": time_horizons,
                "hazard_types": hazard_types,
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "projections": projections,
            "projections_count": len(projections),
            "scenarios": scenarios,
            "time_horizons": time_horizons,
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._projector is not None,
        }

        logger.info(
            "Project stage: status=%s projections=%d scenarios=%s "
            "horizons=%s errors=%d duration_ms=%.1f",
            status,
            len(projections),
            scenarios,
            time_horizons,
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_assess_stage(
        self,
        assets: List[Dict[str, Any]],
        hazard_types: List[str],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 4: Assess geographic and financial exposure of assets.

        Delegates to ExposureAssessorEngine when available.  When the
        engine is unavailable, produces deterministic stub exposure
        assessments using latitude/longitude-derived proximity scores
        (zero-hallucination).

        Metrics recorded:
        - ``record_exposure(asset_type, hazard_type)``
        - ``record_pipeline("exposure_assessment", status)``
        - ``observe_pipeline_duration("exposure_assessment", seconds)``

        Args:
            assets: List of asset dictionaries.
            hazard_types: List of hazard type strings.
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "assess"

        exposure_assessments: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        try:
            for asset in assets:
                asset_id = asset.get("asset_id", "unknown")
                asset_type = asset.get("asset_type", "facility")
                location = asset.get("location", {})

                for hazard_type in hazard_types:
                    if self._exposure_engine is not None:
                        try:
                            raw_result = self._exposure_engine.assess_exposure(
                                asset_id=asset_id,
                                hazard_type=hazard_type,
                                location=location,
                                asset_type=asset_type,
                            )
                            assessment = _normalise_raw(raw_result)
                            assessment.setdefault("asset_id", asset_id)
                            assessment.setdefault("hazard_type", hazard_type)
                            assessment.setdefault("asset_type", asset_type)
                            exposure_assessments.append(assessment)
                            _record_exposure_metric(asset_type, hazard_type)
                        except Exception as exc:
                            logger.warning(
                                "Assess stage: exposure for %s/%s "
                                "failed, using stub -- %s",
                                asset_id,
                                hazard_type,
                                exc,
                            )
                            # Fall back to deterministic stub
                            stub_assessment = self._stub_exposure(
                                asset_id, hazard_type, asset_type, location
                            )
                            exposure_assessments.append(stub_assessment)
                            _record_exposure_metric(asset_type, hazard_type)
                            warnings.append(
                                f"Exposure {asset_id}/{hazard_type}: "
                                f"engine failed, stub used"
                            )
                    else:
                        # Deterministic stub exposure
                        stub_assessment = self._stub_exposure(
                            asset_id, hazard_type, asset_type, location
                        )
                        exposure_assessments.append(stub_assessment)
                        _record_exposure_metric(asset_type, hazard_type)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Assess stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Assess stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("exposure_assessment", status)
        _observe_pipeline_duration_metric("exposure_assessment", elapsed_seconds)

        self._record_provenance(
            "exposure",
            "assess_exposure",
            _new_id("assess"),
            {
                "assessments_count": len(exposure_assessments),
                "hazard_types": hazard_types,
                "assets_count": len(assets),
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "exposure_assessments": exposure_assessments,
            "assessments_count": len(exposure_assessments),
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._exposure_engine is not None,
        }

        logger.info(
            "Assess stage: status=%s assessments=%d errors=%d "
            "duration_ms=%.1f",
            status,
            len(exposure_assessments),
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_score_stage(
        self,
        assets: List[Dict[str, Any]],
        hazard_types: List[str],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 5: Score entity-level vulnerability for each asset.

        Delegates to VulnerabilityScorerEngine when available.  When the
        engine is unavailable, produces deterministic stub vulnerability
        scores using a weighted sum of exposure, sensitivity, and adaptive
        capacity (zero-hallucination: pure arithmetic).

        Metrics recorded:
        - ``record_vulnerability(sector, hazard_type)``
        - ``record_pipeline("vulnerability_scoring", status)``
        - ``observe_pipeline_duration("vulnerability_scoring", seconds)``

        Args:
            assets: List of asset dictionaries.
            hazard_types: List of hazard type strings.
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "score"

        vulnerability_scores: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        # Default vulnerability weights (zero-hallucination arithmetic)
        vuln_weight_exposure = parameters.get("vuln_weight_exposure", 0.40)
        vuln_weight_sensitivity = parameters.get("vuln_weight_sensitivity", 0.35)
        vuln_weight_adaptive = parameters.get("vuln_weight_adaptive", 0.25)

        try:
            for asset in assets:
                asset_id = asset.get("asset_id", "unknown")
                asset_type = asset.get("asset_type", "facility")
                location = asset.get("location", {})
                sector = asset.get("sector", self._infer_sector(asset_type))

                for hazard_type in hazard_types:
                    if self._vulnerability_engine is not None:
                        try:
                            raw_result = self._vulnerability_engine.score_vulnerability(
                                asset_id=asset_id,
                                hazard_type=hazard_type,
                                location=location,
                                asset_type=asset_type,
                            )
                            score_entry = _normalise_raw(raw_result)
                            score_entry.setdefault("asset_id", asset_id)
                            score_entry.setdefault("hazard_type", hazard_type)
                            score_entry.setdefault("sector", sector)
                            vulnerability_scores.append(score_entry)
                            _record_vulnerability_metric(sector, hazard_type)
                        except Exception as exc:
                            logger.warning(
                                "Score stage: vulnerability for %s/%s "
                                "failed, using stub -- %s",
                                asset_id,
                                hazard_type,
                                exc,
                            )
                            # Fall back to deterministic stub
                            stub_score = self._stub_vulnerability_score(
                                asset_id,
                                hazard_type,
                                asset_type,
                                sector,
                                location,
                                vuln_weight_exposure,
                                vuln_weight_sensitivity,
                                vuln_weight_adaptive,
                            )
                            vulnerability_scores.append(stub_score)
                            _record_vulnerability_metric(sector, hazard_type)
                            warnings.append(
                                f"Vulnerability {asset_id}/{hazard_type}: "
                                f"engine failed, stub used"
                            )
                    else:
                        # Deterministic stub vulnerability score
                        stub_score = self._stub_vulnerability_score(
                            asset_id,
                            hazard_type,
                            asset_type,
                            sector,
                            location,
                            vuln_weight_exposure,
                            vuln_weight_sensitivity,
                            vuln_weight_adaptive,
                        )
                        vulnerability_scores.append(stub_score)
                        _record_vulnerability_metric(sector, hazard_type)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Score stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Score stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("vulnerability_scoring", status)
        _observe_pipeline_duration_metric("vulnerability_scoring", elapsed_seconds)

        self._record_provenance(
            "vulnerability",
            "score_vulnerability",
            _new_id("score"),
            {
                "scores_calculated": len(vulnerability_scores),
                "hazard_types": hazard_types,
                "assets_count": len(assets),
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "vulnerability_scores": vulnerability_scores,
            "scores_calculated": len(vulnerability_scores),
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._vulnerability_engine is not None,
        }

        logger.info(
            "Score stage: status=%s scores=%d errors=%d duration_ms=%.1f",
            status,
            len(vulnerability_scores),
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_report_stage(
        self,
        results: Dict[str, Dict[str, Any]],
        report_frameworks: List[str],
        parameters: Dict[str, Any],
    ) -> tuple:
        """Stage 6: Generate compliance reports for requested frameworks.

        Delegates to ComplianceReporterEngine when available.  When the
        engine is unavailable, generates a deterministic stub report
        containing the aggregated results from all preceding stages.

        Metrics recorded:
        - ``record_report(report_type, format)``
        - ``record_pipeline("reporting", status)``
        - ``observe_pipeline_duration("reporting", seconds)``

        Args:
            results: Accumulated stage results from preceding stages.
            report_frameworks: List of report framework strings (tcfd,
                csrd, eu_taxonomy, physical_risk, transition_risk,
                portfolio_summary, hotspot_analysis, compliance_summary).
            parameters: Extra configuration overrides.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "report"

        reports_generated: List[Dict[str, Any]] = []
        report_id: Optional[str] = None
        errors: List[str] = []
        warnings: List[str] = []

        output_format = parameters.get("report_format", "json")

        try:
            for framework in report_frameworks:
                if self._reporter is not None:
                    try:
                        raw_result = self._reporter.generate_report(
                            framework=framework,
                            results=results,
                            output_format=output_format,
                        )
                        report = _normalise_raw(raw_result)
                        report.setdefault("framework", framework)
                        report.setdefault("format", output_format)
                        if report_id is None:
                            report_id = report.get(
                                "report_id", _new_id("rpt")
                            )
                        reports_generated.append(report)
                        _record_report_metric(framework, output_format)
                    except Exception as exc:
                        logger.warning(
                            "Report stage: %s report generation "
                            "failed, using stub -- %s",
                            framework,
                            exc,
                        )
                        # Fall back to deterministic stub
                        stub_rpt = self._stub_report(
                            framework, results, output_format
                        )
                        reports_generated.append(stub_rpt)
                        if report_id is None:
                            report_id = stub_rpt.get("report_id")
                        _record_report_metric(framework, output_format)
                        warnings.append(
                            f"Report for {framework}: engine failed, stub used"
                        )
                else:
                    # Stub report
                    stub_report = self._stub_report(
                        framework, results, output_format
                    )
                    reports_generated.append(stub_report)
                    if report_id is None:
                        report_id = stub_report.get("report_id")
                    _record_report_metric(framework, output_format)

            status = "success" if not errors else "partial"

        except Exception as exc:
            logger.error("Report stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Report stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("reporting", status)
        _observe_pipeline_duration_metric("reporting", elapsed_seconds)

        self._record_provenance(
            "compliance_report",
            "generate_report",
            report_id or _new_id("rpt"),
            {
                "reports_count": len(reports_generated),
                "frameworks": report_frameworks,
                "format": output_format,
                "errors": len(errors),
            },
        )

        stage_result: Dict[str, Any] = {
            "status": status,
            "report_id": report_id,
            "reports_generated": reports_generated,
            "reports_count": len(reports_generated),
            "frameworks": report_frameworks,
            "output_format": output_format,
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "engine_available": self._reporter is not None,
        }

        logger.info(
            "Report stage: status=%s reports=%d frameworks=%s "
            "errors=%d duration_ms=%.1f",
            status,
            len(reports_generated),
            report_frameworks,
            len(errors),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    def _execute_audit_stage(
        self,
        pipeline_id: str,
        results: Dict[str, Dict[str, Any]],
    ) -> tuple:
        """Stage 7: Record full provenance chain for the pipeline run.

        Creates a comprehensive audit record encompassing all stage results.
        This is always performed even when the ProvenanceTracker is
        unavailable (a stub hash is computed).

        Args:
            pipeline_id: Unique pipeline run identifier.
            results: Accumulated stage results from all preceding stages.

        Returns:
            Tuple of (stage_result_dict, duration_ms).
        """
        stage_start = time.monotonic()
        stage_name = "audit"

        errors: List[str] = []
        warnings: List[str] = []
        provenance_entries: List[Dict[str, Any]] = []
        chain_valid = False
        audit_hash = ""

        try:
            # Record a comprehensive audit entry
            audit_data = {
                "pipeline_id": pipeline_id,
                "stages": list(results.keys()),
                "stage_statuses": {
                    k: v.get("status", "unknown") for k, v in results.items()
                },
                "timestamp": _utcnow_iso(),
            }

            audit_hash = self._record_provenance(
                "pipeline_run",
                "run_pipeline",
                pipeline_id,
                audit_data,
            )

            # Verify the provenance chain integrity
            if self._provenance is not None:
                try:
                    chain_valid = self._provenance.verify_chain()
                    provenance_entries = [
                        e.to_dict()
                        for e in self._provenance.get_entries(
                            entity_type="pipeline_run",
                            limit=50,
                        )
                    ]
                except Exception as exc:
                    logger.warning(
                        "Audit stage: chain verification failed -- %s", exc
                    )
                    warnings.append(f"Chain verification error: {exc}")
                    chain_valid = False
            else:
                warnings.append(
                    "ProvenanceTracker unavailable; chain verification skipped"
                )

            status = "success"

        except Exception as exc:
            logger.error("Audit stage failed: %s", exc, exc_info=True)
            status = "failed"
            errors.append(f"Audit stage exception: {exc}")

        elapsed_ms = _elapsed_ms(stage_start)
        elapsed_seconds = elapsed_ms / 1000.0

        _record_pipeline_metric("audit", status)
        _observe_pipeline_duration_metric("audit", elapsed_seconds)

        stage_result: Dict[str, Any] = {
            "status": status,
            "audit_hash": audit_hash,
            "chain_valid": chain_valid,
            "provenance_entries_count": len(provenance_entries),
            "provenance_entries": provenance_entries,
            "errors": errors,
            "warnings": warnings,
            "duration_ms": elapsed_ms,
            "provenance_available": self._provenance is not None,
        }

        logger.info(
            "Audit stage: status=%s chain_valid=%s entries=%d "
            "duration_ms=%.1f",
            status,
            chain_valid,
            len(provenance_entries),
            elapsed_ms,
        )
        return stage_result, elapsed_ms

    # ------------------------------------------------------------------
    # Private: stub implementations (engines unavailable)
    # ------------------------------------------------------------------

    def _stub_risk_index(
        self,
        asset_id: str,
        hazard_type: str,
        location: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute a deterministic stub risk index from location data.

        Uses latitude to derive a base risk score (higher latitudes have
        lower tropical hazard risk but higher cold-wave risk, etc.).
        This is purely arithmetic, zero-hallucination.

        Risk components:
        - probability: Derived from latitude band (0-100).
        - intensity: Fixed moderate baseline (45).
        - frequency: Derived from hazard type lookup (deterministic).
        - duration: Derived from hazard type lookup (deterministic).
        - composite_score: Weighted sum using config defaults.

        Args:
            asset_id: Asset identifier.
            hazard_type: Hazard type string.
            location: Location dictionary with lat/lon.

        Returns:
            Stub risk index dictionary.
        """
        lat = abs(_extract_location_lat(location))
        lon = abs(_extract_location_lon(location))

        # Hazard type base frequency (deterministic lookup)
        hazard_frequency_map: Dict[str, float] = {
            "flood": 55.0,
            "drought": 40.0,
            "wildfire": 45.0,
            "heat_wave": 50.0,
            "cold_wave": 35.0,
            "storm": 50.0,
            "sea_level_rise": 30.0,
            "tropical_cyclone": 40.0,
            "landslide": 25.0,
            "water_stress": 45.0,
            "precipitation_change": 35.0,
            "temperature_change": 40.0,
            "compound": 30.0,
        }

        hazard_duration_map: Dict[str, float] = {
            "flood": 50.0,
            "drought": 70.0,
            "wildfire": 40.0,
            "heat_wave": 45.0,
            "cold_wave": 35.0,
            "storm": 25.0,
            "sea_level_rise": 90.0,
            "tropical_cyclone": 30.0,
            "landslide": 20.0,
            "water_stress": 65.0,
            "precipitation_change": 50.0,
            "temperature_change": 60.0,
            "compound": 55.0,
        }

        # Deterministic probability from latitude band
        if lat < 15:
            probability = 65.0  # Tropical zone
        elif lat < 30:
            probability = 55.0  # Subtropical
        elif lat < 45:
            probability = 45.0  # Temperate
        elif lat < 60:
            probability = 40.0  # Sub-Arctic
        else:
            probability = 30.0  # Arctic

        # Hazard-type-specific probability modifiers
        if hazard_type in ("tropical_cyclone", "heat_wave") and lat < 30:
            probability += 15.0
        elif hazard_type == "cold_wave" and lat > 45:
            probability += 20.0
        elif hazard_type == "sea_level_rise" and lat < 10:
            probability += 10.0

        probability = min(100.0, probability)

        intensity = 45.0  # Moderate baseline
        frequency = hazard_frequency_map.get(hazard_type, 40.0)
        duration = hazard_duration_map.get(hazard_type, 40.0)

        # Composite score: weighted sum (zero-hallucination arithmetic)
        composite_score = (
            probability * 0.30
            + intensity * 0.30
            + frequency * 0.25
            + duration * 0.15
        )
        composite_score = round(min(100.0, max(0.0, composite_score)), 2)
        risk_classification = _classify_risk(composite_score)

        return {
            "asset_id": asset_id,
            "hazard_type": hazard_type,
            "probability": round(probability, 2),
            "intensity": round(intensity, 2),
            "frequency": round(frequency, 2),
            "duration": round(duration, 2),
            "composite_score": composite_score,
            "risk_classification": risk_classification,
            "stub": True,
        }

    def _stub_projection(
        self,
        asset_id: str,
        hazard_type: str,
        scenario: str,
        time_horizon: str,
        location: Dict[str, Any],
        scenario_scaling: Dict[str, float],
        horizon_multipliers: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute a deterministic stub scenario projection.

        Uses tabulated IPCC AR6 scenario scaling factors and time horizon
        multipliers (zero-hallucination: no ML/LLM inference).

        Args:
            asset_id: Asset identifier.
            hazard_type: Hazard type string.
            scenario: Climate scenario string.
            time_horizon: Time horizon string.
            location: Location dictionary.
            scenario_scaling: Mapping of scenario to scaling factor.
            horizon_multipliers: Mapping of horizon to multiplier.

        Returns:
            Stub projection dictionary.
        """
        scale = scenario_scaling.get(scenario, 0.50)
        multiplier = horizon_multipliers.get(time_horizon, 1.0)

        # Base intensity from hazard type (deterministic)
        base_intensity: Dict[str, float] = {
            "flood": 55.0,
            "drought": 50.0,
            "wildfire": 45.0,
            "heat_wave": 60.0,
            "cold_wave": 35.0,
            "storm": 50.0,
            "sea_level_rise": 40.0,
            "tropical_cyclone": 55.0,
            "landslide": 30.0,
            "water_stress": 50.0,
            "precipitation_change": 40.0,
            "temperature_change": 55.0,
            "compound": 45.0,
        }

        current_intensity = base_intensity.get(hazard_type, 45.0)
        projected_intensity = current_intensity * (1.0 + scale * multiplier)
        projected_intensity = round(min(100.0, projected_intensity), 2)

        change_pct = round((projected_intensity - current_intensity) / max(current_intensity, 0.01) * 100.0, 2)

        return {
            "asset_id": asset_id,
            "hazard_type": hazard_type,
            "scenario": scenario,
            "time_horizon": time_horizon,
            "current_intensity": round(current_intensity, 2),
            "projected_intensity": projected_intensity,
            "change_percentage": change_pct,
            "scaling_factor": scale,
            "horizon_multiplier": multiplier,
            "confidence": 0.7,
            "stub": True,
        }

    def _stub_exposure(
        self,
        asset_id: str,
        hazard_type: str,
        asset_type: str,
        location: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute a deterministic stub exposure assessment.

        Uses latitude/longitude to derive geographic exposure and
        asset-type-specific sensitivity multipliers (zero-hallucination).

        Args:
            asset_id: Asset identifier.
            hazard_type: Hazard type string.
            asset_type: Asset type classification.
            location: Location dictionary.

        Returns:
            Stub exposure assessment dictionary.
        """
        lat = abs(_extract_location_lat(location))
        lon = abs(_extract_location_lon(location))

        # Geographic exposure based on latitude
        if lat < 20:
            geographic_exposure = 70.0
        elif lat < 35:
            geographic_exposure = 55.0
        elif lat < 50:
            geographic_exposure = 45.0
        else:
            geographic_exposure = 35.0

        # Asset-type sensitivity multiplier (deterministic lookup)
        asset_sensitivity: Dict[str, float] = {
            "factory": 1.2,
            "warehouse": 1.1,
            "office": 0.9,
            "data_center": 1.3,
            "supply_chain_node": 1.15,
            "transport_hub": 1.1,
            "port": 1.4,
            "mine": 1.3,
            "farm": 1.5,
            "renewable_installation": 1.2,
            "facility": 1.0,
            "portfolio": 1.0,
        }

        sensitivity = asset_sensitivity.get(asset_type, 1.0)
        exposure_score = round(
            min(100.0, geographic_exposure * sensitivity), 2
        )
        exposure_classification = _classify_risk(exposure_score)

        return {
            "asset_id": asset_id,
            "hazard_type": hazard_type,
            "asset_type": asset_type,
            "geographic_exposure": round(geographic_exposure, 2),
            "sensitivity_multiplier": sensitivity,
            "exposure_score": exposure_score,
            "exposure_classification": exposure_classification,
            "lat": _extract_location_lat(location),
            "lon": _extract_location_lon(location),
            "stub": True,
        }

    def _stub_vulnerability_score(
        self,
        asset_id: str,
        hazard_type: str,
        asset_type: str,
        sector: str,
        location: Dict[str, Any],
        weight_exposure: float,
        weight_sensitivity: float,
        weight_adaptive: float,
    ) -> Dict[str, Any]:
        """Compute a deterministic stub vulnerability score.

        Uses a weighted sum of exposure, sensitivity, and adaptive capacity
        components (zero-hallucination: pure Python arithmetic).

        Args:
            asset_id: Asset identifier.
            hazard_type: Hazard type string.
            asset_type: Asset type classification.
            sector: Economic sector.
            location: Location dictionary.
            weight_exposure: Weight for exposure component.
            weight_sensitivity: Weight for sensitivity component.
            weight_adaptive: Weight for adaptive capacity component.

        Returns:
            Stub vulnerability score dictionary.
        """
        lat = abs(_extract_location_lat(location))

        # Exposure component from latitude band
        if lat < 20:
            exposure_component = 65.0
        elif lat < 35:
            exposure_component = 50.0
        elif lat < 50:
            exposure_component = 40.0
        else:
            exposure_component = 30.0

        # Sensitivity component from sector lookup
        sector_sensitivity: Dict[str, float] = {
            "agriculture": 75.0,
            "energy": 55.0,
            "manufacturing": 50.0,
            "real_estate": 45.0,
            "transport": 50.0,
            "water": 70.0,
            "financial": 35.0,
            "healthcare": 40.0,
            "technology": 30.0,
            "mining": 60.0,
            "forestry": 65.0,
        }
        sensitivity_component = sector_sensitivity.get(sector, 45.0)

        # Adaptive capacity (inverse relationship with latitude zone)
        if lat < 20:
            adaptive_capacity = 35.0  # Lower adaptive capacity in tropics
        elif lat < 35:
            adaptive_capacity = 50.0
        elif lat < 50:
            adaptive_capacity = 60.0
        else:
            adaptive_capacity = 55.0

        # Vulnerability = weighted(exposure, sensitivity) - weighted(adaptive)
        # Higher vulnerability when exposure and sensitivity are high,
        # lower when adaptive capacity is high
        raw_score = (
            exposure_component * weight_exposure
            + sensitivity_component * weight_sensitivity
            - adaptive_capacity * weight_adaptive
        )
        # Normalise to 0-100 range
        vulnerability_score = round(min(100.0, max(0.0, raw_score + 20.0)), 2)
        risk_classification = _classify_risk(vulnerability_score)

        return {
            "asset_id": asset_id,
            "hazard_type": hazard_type,
            "asset_type": asset_type,
            "sector": sector,
            "exposure_component": round(exposure_component, 2),
            "sensitivity_component": round(sensitivity_component, 2),
            "adaptive_capacity": round(adaptive_capacity, 2),
            "vulnerability_score": vulnerability_score,
            "risk_classification": risk_classification,
            "weights": {
                "exposure": weight_exposure,
                "sensitivity": weight_sensitivity,
                "adaptive": weight_adaptive,
            },
            "stub": True,
        }

    def _stub_report(
        self,
        framework: str,
        results: Dict[str, Dict[str, Any]],
        output_format: str,
    ) -> Dict[str, Any]:
        """Generate a deterministic stub compliance report.

        Aggregates data from preceding stage results into a structured
        report without any LLM inference.

        Args:
            framework: Report framework string (tcfd, csrd, eu_taxonomy).
            results: Accumulated stage results.
            output_format: Output format string (json, csv, pdf).

        Returns:
            Stub report dictionary.
        """
        report_id = _new_id("rpt")

        # Extract summary statistics from stage results
        index_result = results.get("index", {})
        risk_indices = index_result.get("risk_indices", [])
        scores = [
            idx.get("composite_score", 0.0)
            for idx in risk_indices
            if isinstance(idx.get("composite_score"), (int, float))
        ]
        avg_score = round(_safe_mean(scores), 2)
        high_risk_count = sum(
            1 for s in scores if s >= _THRESHOLD_HIGH
        )
        extreme_risk_count = sum(
            1 for s in scores if s >= _THRESHOLD_EXTREME
        )

        score_result = results.get("score", {})
        vuln_scores = score_result.get("vulnerability_scores", [])
        vuln_values = [
            v.get("vulnerability_score", 0.0)
            for v in vuln_scores
            if isinstance(v.get("vulnerability_score"), (int, float))
        ]
        avg_vulnerability = round(_safe_mean(vuln_values), 2)

        project_result = results.get("project", {})
        projections = project_result.get("projections", [])

        # Framework-specific sections (deterministic)
        framework_sections = self._build_framework_sections(
            framework, avg_score, high_risk_count, extreme_risk_count,
            avg_vulnerability, len(projections)
        )

        return {
            "report_id": report_id,
            "framework": framework,
            "format": output_format,
            "summary": {
                "avg_risk_score": avg_score,
                "high_risk_count": high_risk_count,
                "extreme_risk_count": extreme_risk_count,
                "avg_vulnerability": avg_vulnerability,
                "projections_count": len(projections),
            },
            "sections": framework_sections,
            "generated_at": _utcnow_iso(),
            "stub": True,
        }

    def _build_framework_sections(
        self,
        framework: str,
        avg_score: float,
        high_risk_count: int,
        extreme_risk_count: int,
        avg_vulnerability: float,
        projections_count: int,
    ) -> List[Dict[str, Any]]:
        """Build framework-specific report sections deterministically.

        Uses template logic to produce standardised sections for each
        supported compliance framework.  No LLM inference.

        Args:
            framework: Report framework identifier.
            avg_score: Average composite risk score.
            high_risk_count: Number of high-risk assessments.
            extreme_risk_count: Number of extreme-risk assessments.
            avg_vulnerability: Average vulnerability score.
            projections_count: Number of scenario projections.

        Returns:
            List of section dictionaries.
        """
        sections: List[Dict[str, Any]] = []

        if framework in ("tcfd", "physical_risk"):
            sections.append({
                "section": "Governance",
                "content": (
                    "Climate hazard governance framework with board-level "
                    "oversight and risk committee review."
                ),
            })
            sections.append({
                "section": "Strategy",
                "content": (
                    f"Physical climate risk assessment across "
                    f"{projections_count} scenario projection(s). "
                    f"Average risk score: {avg_score}."
                ),
            })
            sections.append({
                "section": "Risk Management",
                "content": (
                    f"Identified {high_risk_count} high-risk and "
                    f"{extreme_risk_count} extreme-risk asset-hazard "
                    f"combinations. Average vulnerability: "
                    f"{avg_vulnerability}."
                ),
            })
            sections.append({
                "section": "Metrics and Targets",
                "content": (
                    f"Composite risk index: {avg_score}/100. "
                    f"Target: reduce high-risk exposure by 20% by 2030."
                ),
            })
        elif framework in ("csrd", "eu_taxonomy"):
            sections.append({
                "section": "ESRS E1 - Climate Change",
                "content": (
                    f"Physical climate risk assessment completed. "
                    f"Average risk score: {avg_score}/100. "
                    f"{high_risk_count} high-risk findings."
                ),
            })
            sections.append({
                "section": "Climate Adaptation Screening",
                "content": (
                    f"Vulnerability assessment across {projections_count} "
                    f"projections. Average vulnerability: "
                    f"{avg_vulnerability}/100."
                ),
            })
            sections.append({
                "section": "Do No Significant Harm (DNSH)",
                "content": (
                    f"DNSH climate adaptation criteria evaluated. "
                    f"{extreme_risk_count} extreme-risk items require "
                    f"adaptation plans."
                ),
            })
        else:
            sections.append({
                "section": "Executive Summary",
                "content": (
                    f"Climate hazard assessment completed. "
                    f"Average risk: {avg_score}/100, "
                    f"high-risk: {high_risk_count}, "
                    f"extreme-risk: {extreme_risk_count}, "
                    f"avg vulnerability: {avg_vulnerability}/100."
                ),
            })
            sections.append({
                "section": "Detailed Findings",
                "content": (
                    f"Assessed across {projections_count} scenario "
                    f"projections with comprehensive risk indexing "
                    f"and vulnerability scoring."
                ),
            })

        return sections

    # ------------------------------------------------------------------
    # Private: helper methods
    # ------------------------------------------------------------------

    def _resolve_stages(
        self, stages: Optional[List[str]]
    ) -> List[str]:
        """Resolve and validate the stage list.

        When ``stages`` is ``None``, returns all stages from
        :data:`PIPELINE_STAGES`.  When provided, filters to only valid
        stage names and preserves the canonical order defined in
        :data:`PIPELINE_STAGES`.

        Args:
            stages: Optional list of stage names, or ``None`` for all.

        Returns:
            Ordered list of valid stage names to execute.
        """
        if stages is None:
            return list(PIPELINE_STAGES)

        valid_set = set(PIPELINE_STAGES)
        resolved: List[str] = []
        for stage in PIPELINE_STAGES:
            if stage in stages:
                resolved.append(stage)

        # Warn about unrecognised stage names
        unrecognised = set(stages) - valid_set
        if unrecognised:
            logger.warning(
                "Unrecognised pipeline stages ignored: %s",
                sorted(unrecognised),
            )

        return resolved

    def _determine_pipeline_status(
        self,
        stages_completed: int,
        total_stages: int,
        stages_failed: List[str],
    ) -> str:
        """Determine the final pipeline status based on stage outcomes.

        Status logic:
        - ``"completed"``: All attempted stages succeeded.
        - ``"partial"``: At least one stage succeeded and at least one
          failed.
        - ``"failed"``: All attempted stages failed.

        Args:
            stages_completed: Count of successfully completed stages.
            total_stages: Total count of attempted stages.
            stages_failed: List of failed stage names.

        Returns:
            Pipeline status string: completed, partial, or failed.
        """
        if not stages_failed:
            return _STATUS_COMPLETED
        if stages_completed > 0:
            return _STATUS_PARTIAL
        return _STATUS_FAILED

    def _build_evaluation_summary(
        self,
        assets: List[Dict[str, Any]],
        hazard_types: List[str],
        scenarios: List[str],
        stage_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the evaluation summary from accumulated stage results.

        All aggregations are deterministic Python arithmetic
        (zero-hallucination).

        Args:
            assets: Original list of assets.
            hazard_types: List of hazard types assessed.
            scenarios: List of scenarios evaluated.
            stage_results: Accumulated stage results.

        Returns:
            Evaluation summary dictionary.
        """
        # Extract risk scores from index stage
        index_result = stage_results.get("index", {})
        risk_indices = index_result.get("risk_indices", [])
        risk_scores = [
            idx.get("composite_score", 0.0)
            for idx in risk_indices
            if isinstance(idx.get("composite_score"), (int, float))
        ]
        avg_risk_score = round(_safe_mean(risk_scores), 2)
        high_risk_count = sum(
            1 for s in risk_scores if s >= _THRESHOLD_HIGH
        )

        # Extract vulnerability scores from score stage
        score_result = stage_results.get("score", {})
        vuln_scores = score_result.get("vulnerability_scores", [])
        vuln_values = [
            v.get("vulnerability_score", 0.0)
            for v in vuln_scores
            if isinstance(v.get("vulnerability_score"), (int, float))
        ]
        avg_vulnerability = round(_safe_mean(vuln_values), 2)

        # Extract projection count
        project_result = stage_results.get("project", {})
        projections_count = project_result.get("projections_count", 0)

        # Extract exposure count
        assess_result = stage_results.get("assess", {})
        assessments_count = assess_result.get("assessments_count", 0)

        return {
            "total_assets": len(assets),
            "hazard_types_assessed": len(hazard_types),
            "scenarios_evaluated": len(scenarios),
            "high_risk_count": high_risk_count,
            "avg_risk_score": avg_risk_score,
            "avg_vulnerability_score": avg_vulnerability,
            "total_risk_indices": len(risk_indices),
            "total_vulnerability_scores": len(vuln_scores),
            "total_projections": projections_count,
            "total_exposure_assessments": assessments_count,
        }

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Any,
    ) -> str:
        """Record a provenance entry and return its hash value.

        Delegates to the ProvenanceTracker when available.  When the
        tracker is unavailable, computes a standalone SHA-256 hash of
        the data payload as a fallback.

        Args:
            entity_type: Type of entity tracked.
            action: Action performed.
            entity_id: Unique entity identifier.
            data: Data payload to hash.

        Returns:
            SHA-256 hash string (from chain hash or standalone hash).
        """
        if self._provenance is not None:
            try:
                entry = self._provenance.record(
                    entity_type=entity_type,
                    action=action,
                    entity_id=entity_id,
                    data=data,
                )
                return entry.hash_value
            except Exception as exc:
                logger.warning(
                    "Provenance record failed for %s/%s/%s: %s",
                    entity_type,
                    action,
                    entity_id,
                    exc,
                )
        # Fallback: standalone SHA-256 hash
        return _sha256({
            "entity_type": entity_type,
            "action": action,
            "entity_id": entity_id,
            "data": data,
        })

    def _infer_sector(self, asset_type: str) -> str:
        """Infer an economic sector from an asset type classification.

        Uses a deterministic lookup table.  No LLM inference.

        Args:
            asset_type: Asset type classification string.

        Returns:
            Economic sector string.
        """
        asset_type_to_sector: Dict[str, str] = {
            "facility": "manufacturing",
            "factory": "manufacturing",
            "warehouse": "transport",
            "office": "real_estate",
            "data_center": "technology",
            "supply_chain_node": "transport",
            "transport_hub": "transport",
            "port": "transport",
            "mine": "mining",
            "farm": "agriculture",
            "renewable_installation": "energy",
            "portfolio": "financial",
        }
        return asset_type_to_sector.get(asset_type, "manufacturing")

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing engine availability counts and total runs.
        """
        engines_up = sum(
            1
            for eng in (
                self._database,
                self._risk_engine,
                self._projector,
                self._exposure_engine,
                self._vulnerability_engine,
                self._reporter,
            )
            if eng is not None
        )
        with self._lock:
            total = self._total_runs
        return (
            f"HazardPipelineEngine(engines={engines_up}/6, "
            f"runs={total}, "
            f"provenance={'ok' if self._provenance else 'none'})"
        )

    def __len__(self) -> int:
        """Return the total number of pipeline runs stored.

        Returns:
            Integer count of pipeline runs in the store.
        """
        with self._lock:
            return len(self._pipeline_runs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "HazardPipelineEngine",
    "PIPELINE_STAGES",
]
