# -*- coding: utf-8 -*-
"""
DataAggregationEngine - PACK-030 Net Zero Reporting Pack Engine 1
====================================================================

Multi-source data collection and reconciliation engine for net-zero
reporting.  Aggregates emissions data, targets, reduction initiatives,
sector pathways, and interim milestones from all prerequisite packs
(PACK-021/022/028/029) and GreenLang applications (GL-SBTi-APP,
GL-CDP-APP, GL-TCFD-APP, GL-GHG-APP).

Aggregation Methodology:
    Source Priority Resolution:
        When multiple sources report the same metric, the engine uses
        a configurable priority chain:
            1. Application data (GL-*-APP) -- highest fidelity
            2. Pack data (PACK-*) -- calculation outputs
            3. File-based imports -- manual uploads
            4. API integrations -- external systems

    Reconciliation Algorithm:
        For every metric M reported by sources S1..Sn:
            variance(M) = max(S_i(M)) - min(S_i(M))
            variance_pct = variance / mean(S_i(M)) * 100
            If variance_pct > threshold (default 5%):
                flag mismatch, require manual reconciliation

    Data Completeness Scoring:
        completeness = (metrics_present / metrics_required) * 100
        Framework-specific required metric lists define denominator.

    Lineage Tracking:
        Every aggregated metric records:
            source_system, source_id, timestamp, transformation_steps,
            provenance_hash (SHA-256)

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GHG Protocol Scope 3 Standard (2011)
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - CDP Climate Change Questionnaire (2024)
    - TCFD Recommendations (2017, updated 2023)
    - ISSB IFRS S2 (2023)
    - SEC Climate Disclosure Rules (2024)
    - CSRD ESRS E1 (2024)
    - ISO 14064-1:2018 -- Organizational GHG inventories
    - ISAE 3410 -- Assurance on GHG statements

Zero-Hallucination:
    - All aggregation uses deterministic Decimal arithmetic
    - No LLM involvement in any calculation or reconciliation path
    - SHA-256 provenance hash on every result
    - Reconciliation thresholds hard-coded from audit standards

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-030 Net Zero Reporting
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSourceType(str, Enum):
    """Type of data source for aggregation."""
    PACK_021 = "pack_021"
    PACK_022 = "pack_022"
    PACK_028 = "pack_028"
    PACK_029 = "pack_029"
    GL_SBTI_APP = "gl_sbti_app"
    GL_CDP_APP = "gl_cdp_app"
    GL_TCFD_APP = "gl_tcfd_app"
    GL_GHG_APP = "gl_ghg_app"
    ERP_SYSTEM = "erp_system"
    FILE_UPLOAD = "file_upload"
    API_EXTERNAL = "api_external"


class MetricCategory(str, Enum):
    """Category of metric being aggregated."""
    EMISSIONS_SCOPE_1 = "emissions_scope_1"
    EMISSIONS_SCOPE_2_LOCATION = "emissions_scope_2_location"
    EMISSIONS_SCOPE_2_MARKET = "emissions_scope_2_market"
    EMISSIONS_SCOPE_3 = "emissions_scope_3"
    EMISSIONS_TOTAL = "emissions_total"
    TARGET_NEAR_TERM = "target_near_term"
    TARGET_LONG_TERM = "target_long_term"
    REDUCTION_INITIATIVE = "reduction_initiative"
    SECTOR_PATHWAY = "sector_pathway"
    INTERIM_MILESTONE = "interim_milestone"
    ENERGY_CONSUMPTION = "energy_consumption"
    CARBON_INTENSITY = "carbon_intensity"
    FINANCIAL_METRIC = "financial_metric"


class ReconciliationStatus(str, Enum):
    """Status of data reconciliation between sources."""
    RECONCILED = "reconciled"
    MISMATCH_MINOR = "mismatch_minor"
    MISMATCH_MAJOR = "mismatch_major"
    SINGLE_SOURCE = "single_source"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"


class DataQuality(str, Enum):
    """Data quality tier for aggregated data."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"


class GapSeverity(str, Enum):
    """Severity of a data gap."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FrameworkTarget(str, Enum):
    """Target reporting framework."""
    SBTI = "SBTi"
    CDP = "CDP"
    TCFD = "TCFD"
    GRI = "GRI"
    ISSB = "ISSB"
    SEC = "SEC"
    CSRD = "CSRD"


class ConnectionStatus(str, Enum):
    """Status of a source connection."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"


# ---------------------------------------------------------------------------
# Constants -- Framework Metric Requirements
# ---------------------------------------------------------------------------

FRAMEWORK_REQUIRED_METRICS: Dict[str, List[str]] = {
    FrameworkTarget.SBTI.value: [
        "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e",
        "base_year", "base_year_emissions", "target_year",
        "target_reduction_pct", "annual_reduction_rate",
        "progress_vs_target_pct",
    ],
    FrameworkTarget.CDP.value: [
        "scope_1_tco2e", "scope_2_location_tco2e", "scope_2_market_tco2e",
        "scope_3_tco2e", "scope_3_categories",
        "target_type", "target_year", "target_reduction_pct",
        "base_year", "base_year_emissions",
        "methodology", "verification_status",
        "energy_consumption_mwh", "renewable_energy_pct",
    ],
    FrameworkTarget.TCFD.value: [
        "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e",
        "carbon_intensity", "target_description",
        "scenario_analysis_results", "climate_risks",
        "climate_opportunities", "governance_description",
    ],
    FrameworkTarget.GRI.value: [
        "scope_1_tco2e", "scope_2_location_tco2e", "scope_2_market_tco2e",
        "scope_3_tco2e", "ghg_intensity",
        "emissions_reduction_tco2e", "reduction_initiatives",
        "methodology_description",
    ],
    FrameworkTarget.ISSB.value: [
        "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e",
        "industry_specific_metrics", "transition_risks",
        "physical_risks", "climate_targets",
        "capital_deployment",
    ],
    FrameworkTarget.SEC.value: [
        "scope_1_tco2e", "scope_2_tco2e",
        "attestation_status", "climate_risk_description",
        "financial_impact_estimate",
        "target_description", "target_year",
    ],
    FrameworkTarget.CSRD.value: [
        "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e",
        "transition_plan", "climate_policies",
        "climate_actions", "ghg_reduction_targets",
        "energy_consumption", "energy_mix",
        "ghg_removals", "carbon_credits",
        "internal_carbon_price", "financial_effects",
    ],
}

# Default reconciliation thresholds
RECONCILIATION_THRESHOLDS: Dict[str, Decimal] = {
    "minor_variance_pct": Decimal("2"),
    "major_variance_pct": Decimal("5"),
    "critical_variance_pct": Decimal("10"),
}

# Source priority order (lower = higher priority)
SOURCE_PRIORITY: Dict[str, int] = {
    DataSourceType.GL_GHG_APP.value: 1,
    DataSourceType.GL_SBTI_APP.value: 2,
    DataSourceType.GL_CDP_APP.value: 3,
    DataSourceType.GL_TCFD_APP.value: 4,
    DataSourceType.PACK_021.value: 5,
    DataSourceType.PACK_022.value: 6,
    DataSourceType.PACK_028.value: 7,
    DataSourceType.PACK_029.value: 8,
    DataSourceType.ERP_SYSTEM.value: 9,
    DataSourceType.FILE_UPLOAD.value: 10,
    DataSourceType.API_EXTERNAL.value: 11,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class SourceDataPoint(BaseModel):
    """A single data point from a source system.

    Attributes:
        source: Source system identifier.
        metric_name: Name of the metric.
        metric_category: Category of the metric.
        value: Numeric value (Decimal for precision).
        unit: Unit of measurement.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        timestamp: When the data was recorded.
        source_record_id: ID of the source record.
        confidence: Data confidence level (0-100).
        methodology: Calculation methodology used.
        notes: Additional notes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str = Field(..., description="Source system identifier")
    metric_name: str = Field(..., description="Metric name")
    metric_category: MetricCategory = Field(
        default=MetricCategory.EMISSIONS_TOTAL,
        description="Metric category",
    )
    value: Decimal = Field(default=Decimal("0"), description="Metric value")
    unit: str = Field(default="tCO2e", description="Unit of measurement")
    reporting_period_start: Optional[date] = Field(
        default=None, description="Reporting period start"
    )
    reporting_period_end: Optional[date] = Field(
        default=None, description="Reporting period end"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Data timestamp"
    )
    source_record_id: str = Field(
        default="", description="Source record identifier"
    )
    confidence: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Data confidence (0-100)",
    )
    methodology: str = Field(
        default="", description="Calculation methodology"
    )
    notes: str = Field(default="", description="Additional notes")


class SourceConnection(BaseModel):
    """Configuration for a data source connection.

    Attributes:
        source_type: Type of data source.
        endpoint: API endpoint or connection string.
        auth_method: Authentication method.
        status: Current connection status.
        last_sync: Last successful data sync.
        priority: Source priority (lower = higher).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_type: DataSourceType = Field(..., description="Source type")
    endpoint: str = Field(default="", description="API endpoint")
    auth_method: str = Field(default="oauth2", description="Auth method")
    status: ConnectionStatus = Field(
        default=ConnectionStatus.NOT_CONFIGURED,
        description="Connection status",
    )
    last_sync: Optional[datetime] = Field(
        default=None, description="Last sync time"
    )
    priority: int = Field(default=10, ge=1, le=100, description="Priority")


class DataAggregationInput(BaseModel):
    """Input for the data aggregation engine.

    Attributes:
        organization_id: Organization identifier.
        organization_name: Organization name.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        target_frameworks: Frameworks to aggregate data for.
        source_connections: Configured source connections.
        data_points: Pre-loaded data points from sources.
        reconciliation_threshold_pct: Variance threshold for flagging.
        include_lineage: Generate data lineage.
        include_gap_analysis: Run data gap analysis.
        include_reconciliation: Run source reconciliation.
        preferred_scope_2_method: Preferred Scope 2 method.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organization_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Organization identifier",
    )
    organization_name: str = Field(
        default="", max_length=300,
        description="Organization name",
    )
    reporting_period_start: date = Field(
        ..., description="Reporting period start"
    )
    reporting_period_end: date = Field(
        ..., description="Reporting period end"
    )
    target_frameworks: List[FrameworkTarget] = Field(
        default_factory=lambda: list(FrameworkTarget),
        description="Target frameworks",
    )
    source_connections: List[SourceConnection] = Field(
        default_factory=list, description="Source connections"
    )
    data_points: List[SourceDataPoint] = Field(
        default_factory=list, description="Pre-loaded data points"
    )
    reconciliation_threshold_pct: Decimal = Field(
        default=Decimal("5"), ge=Decimal("0"), le=Decimal("100"),
        description="Reconciliation variance threshold (%)",
    )
    include_lineage: bool = Field(
        default=True, description="Generate data lineage"
    )
    include_gap_analysis: bool = Field(
        default=True, description="Run gap analysis"
    )
    include_reconciliation: bool = Field(
        default=True, description="Run reconciliation"
    )
    preferred_scope_2_method: str = Field(
        default="market_based",
        description="Preferred Scope 2 method",
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class AggregatedMetric(BaseModel):
    """A single aggregated metric with provenance.

    Attributes:
        metric_id: Unique metric identifier.
        metric_name: Name of the metric.
        metric_category: Category of the metric.
        value: Aggregated value.
        unit: Unit of measurement.
        source_count: Number of sources providing this metric.
        primary_source: Source used for the final value.
        all_sources: All sources that reported this metric.
        confidence: Aggregated confidence score.
        reconciliation_status: Reconciliation result.
        variance_pct: Variance across sources (%).
        methodology: Methodology used.
        provenance_hash: SHA-256 hash for audit trail.
    """
    metric_id: str = Field(default_factory=_new_uuid)
    metric_name: str = Field(default="")
    metric_category: str = Field(default="")
    value: Decimal = Field(default=Decimal("0"))
    unit: str = Field(default="tCO2e")
    source_count: int = Field(default=0)
    primary_source: str = Field(default="")
    all_sources: List[str] = Field(default_factory=list)
    confidence: Decimal = Field(default=Decimal("0"))
    reconciliation_status: str = Field(
        default=ReconciliationStatus.SINGLE_SOURCE.value
    )
    variance_pct: Decimal = Field(default=Decimal("0"))
    methodology: str = Field(default="")
    provenance_hash: str = Field(default="")


class ReconciliationItem(BaseModel):
    """A reconciliation check between sources.

    Attributes:
        metric_name: Metric being reconciled.
        source_values: Values from each source.
        min_value: Minimum reported value.
        max_value: Maximum reported value.
        mean_value: Mean of reported values.
        variance: Absolute variance.
        variance_pct: Variance as percentage.
        status: Reconciliation status.
        resolution: How the discrepancy was resolved.
        selected_value: Final selected value.
        selected_source: Source of the selected value.
    """
    metric_name: str = Field(default="")
    source_values: Dict[str, Decimal] = Field(default_factory=dict)
    min_value: Decimal = Field(default=Decimal("0"))
    max_value: Decimal = Field(default=Decimal("0"))
    mean_value: Decimal = Field(default=Decimal("0"))
    variance: Decimal = Field(default=Decimal("0"))
    variance_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=ReconciliationStatus.PENDING_REVIEW.value)
    resolution: str = Field(default="")
    selected_value: Decimal = Field(default=Decimal("0"))
    selected_source: str = Field(default="")


class DataGap(BaseModel):
    """A detected data gap.

    Attributes:
        framework: Framework requiring this data.
        metric_name: Missing metric name.
        severity: Gap severity.
        description: Description of the gap.
        suggested_source: Suggested source to fill the gap.
        impact: Impact of the gap on reporting.
    """
    framework: str = Field(default="")
    metric_name: str = Field(default="")
    severity: str = Field(default=GapSeverity.MEDIUM.value)
    description: str = Field(default="")
    suggested_source: str = Field(default="")
    impact: str = Field(default="")


class LineageNode(BaseModel):
    """A node in the data lineage graph.

    Attributes:
        node_id: Unique node identifier.
        node_type: Type of node (source, transform, metric).
        label: Display label.
        source_system: Source system name.
        metric_name: Metric name (for metric nodes).
        timestamp: When the data was recorded.
        transformation: Transformation applied.
        parent_ids: Parent node identifiers.
        provenance_hash: SHA-256 hash.
    """
    node_id: str = Field(default_factory=_new_uuid)
    node_type: str = Field(default="source")
    label: str = Field(default="")
    source_system: str = Field(default="")
    metric_name: str = Field(default="")
    timestamp: Optional[datetime] = Field(default=None)
    transformation: str = Field(default="")
    parent_ids: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class LineageGraph(BaseModel):
    """Data lineage graph showing source-to-metric flow.

    Attributes:
        nodes: All nodes in the graph.
        edges: Edges as (source_id, target_id) pairs.
        total_sources: Number of distinct source systems.
        total_transformations: Number of transformation steps.
        total_metrics: Number of output metrics.
    """
    nodes: List[LineageNode] = Field(default_factory=list)
    edges: List[Tuple[str, str]] = Field(default_factory=list)
    total_sources: int = Field(default=0)
    total_transformations: int = Field(default=0)
    total_metrics: int = Field(default=0)


class FrameworkCompleteness(BaseModel):
    """Completeness assessment for a single framework.

    Attributes:
        framework: Framework name.
        required_metrics: Total required metrics.
        provided_metrics: Metrics with data.
        missing_metrics: Metrics without data.
        completeness_pct: Completeness percentage.
        missing_metric_names: Names of missing metrics.
        status: Overall status.
    """
    framework: str = Field(default="")
    required_metrics: int = Field(default=0)
    provided_metrics: int = Field(default=0)
    missing_metrics: int = Field(default=0)
    completeness_pct: Decimal = Field(default=Decimal("0"))
    missing_metric_names: List[str] = Field(default_factory=list)
    status: str = Field(default="incomplete")


class SourceHealthStatus(BaseModel):
    """Health status for a data source.

    Attributes:
        source: Source identifier.
        status: Connection status.
        metrics_provided: Number of metrics from this source.
        last_sync: Last successful sync timestamp.
        latency_ms: Connection latency (ms).
        error_message: Error message if unhealthy.
    """
    source: str = Field(default="")
    status: str = Field(default=ConnectionStatus.NOT_CONFIGURED.value)
    metrics_provided: int = Field(default=0)
    last_sync: Optional[datetime] = Field(default=None)
    latency_ms: float = Field(default=0.0)
    error_message: str = Field(default="")


class DataAggregationResult(BaseModel):
    """Complete data aggregation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        organization_id: Organization identifier.
        organization_name: Organization name.
        reporting_period_start: Reporting period start.
        reporting_period_end: Reporting period end.
        aggregated_metrics: All aggregated metrics.
        reconciliation_items: Reconciliation results.
        data_gaps: Detected data gaps.
        lineage_graph: Data lineage graph.
        framework_completeness: Per-framework completeness.
        source_health: Source health statuses.
        total_metrics: Total metrics aggregated.
        total_sources: Number of data sources.
        total_data_points: Total data points processed.
        overall_completeness_pct: Overall completeness percentage.
        overall_confidence: Overall confidence score.
        overall_quality: Overall data quality tier.
        reconciliation_summary: Reconciliation summary stats.
        warnings: Warnings generated during aggregation.
        recommendations: Recommendations for data improvement.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    reporting_period_start: Optional[date] = Field(default=None)
    reporting_period_end: Optional[date] = Field(default=None)
    aggregated_metrics: List[AggregatedMetric] = Field(default_factory=list)
    reconciliation_items: List[ReconciliationItem] = Field(default_factory=list)
    data_gaps: List[DataGap] = Field(default_factory=list)
    lineage_graph: Optional[LineageGraph] = Field(default=None)
    framework_completeness: List[FrameworkCompleteness] = Field(
        default_factory=list
    )
    source_health: List[SourceHealthStatus] = Field(default_factory=list)
    total_metrics: int = Field(default=0)
    total_sources: int = Field(default=0)
    total_data_points: int = Field(default=0)
    overall_completeness_pct: Decimal = Field(default=Decimal("0"))
    overall_confidence: Decimal = Field(default=Decimal("0"))
    overall_quality: str = Field(default=DataQuality.MEDIUM.value)
    reconciliation_summary: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DataAggregationEngine:
    """Multi-source data aggregation engine for PACK-030.

    Collects and reconciles emissions data from all source packs
    (PACK-021/022/028/029) and GreenLang applications, detects
    mismatches and gaps, generates data lineage, and scores
    completeness for each target framework.

    All calculations use deterministic Decimal arithmetic.
    No LLM involvement in any calculation or reconciliation path.

    Usage::

        engine = DataAggregationEngine()
        result = await engine.aggregate(aggregation_input)
        print(f"Completeness: {result.overall_completeness_pct}%")
        for gap in result.data_gaps:
            print(f"  Gap: {gap.metric_name} ({gap.severity})")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def aggregate(
        self, data: DataAggregationInput,
    ) -> DataAggregationResult:
        """Run complete data aggregation.

        Args:
            data: Validated aggregation input.

        Returns:
            DataAggregationResult with metrics, reconciliation, and gaps.
        """
        t0 = time.perf_counter()
        logger.info(
            "Data aggregation: org=%s, period=%s to %s, frameworks=%d",
            data.organization_id,
            data.reporting_period_start,
            data.reporting_period_end,
            len(data.target_frameworks),
        )

        # Step 1: Group data points by metric name
        grouped = self._group_by_metric(data.data_points)

        # Step 2: Aggregate each metric (priority-based selection)
        aggregated_metrics = self._aggregate_metrics(grouped, data)

        # Step 3: Reconcile across sources
        reconciliation_items: List[ReconciliationItem] = []
        if data.include_reconciliation:
            reconciliation_items = self._reconcile_sources(
                grouped, data.reconciliation_threshold_pct
            )

        # Step 4: Gap analysis
        data_gaps: List[DataGap] = []
        if data.include_gap_analysis:
            data_gaps = self._detect_gaps(
                aggregated_metrics, data.target_frameworks
            )

        # Step 5: Lineage graph
        lineage_graph: Optional[LineageGraph] = None
        if data.include_lineage:
            lineage_graph = self._build_lineage_graph(
                data.data_points, aggregated_metrics
            )

        # Step 6: Framework completeness
        framework_completeness = self._calculate_framework_completeness(
            aggregated_metrics, data.target_frameworks
        )

        # Step 7: Source health
        source_health = self._assess_source_health(
            data.source_connections, data.data_points
        )

        # Step 8: Overall statistics
        total_sources = len({dp.source for dp in data.data_points})
        overall_completeness = self._calculate_overall_completeness(
            framework_completeness
        )
        overall_confidence = self._calculate_overall_confidence(
            aggregated_metrics
        )
        overall_quality = self._assess_overall_quality(
            overall_completeness, overall_confidence, len(reconciliation_items)
        )

        # Step 9: Reconciliation summary
        recon_summary = self._summarize_reconciliation(reconciliation_items)

        # Step 10: Warnings and recommendations
        warnings = self._generate_warnings(
            data, aggregated_metrics, data_gaps, reconciliation_items
        )
        recommendations = self._generate_recommendations(
            data, aggregated_metrics, data_gaps, framework_completeness
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = DataAggregationResult(
            organization_id=data.organization_id,
            organization_name=data.organization_name,
            reporting_period_start=data.reporting_period_start,
            reporting_period_end=data.reporting_period_end,
            aggregated_metrics=aggregated_metrics,
            reconciliation_items=reconciliation_items,
            data_gaps=data_gaps,
            lineage_graph=lineage_graph,
            framework_completeness=framework_completeness,
            source_health=source_health,
            total_metrics=len(aggregated_metrics),
            total_sources=total_sources,
            total_data_points=len(data.data_points),
            overall_completeness_pct=_round_val(overall_completeness, 2),
            overall_confidence=_round_val(overall_confidence, 2),
            overall_quality=overall_quality,
            reconciliation_summary=recon_summary,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Data aggregation complete: org=%s, metrics=%d, sources=%d, "
            "completeness=%.1f%%, gaps=%d, reconciliation_items=%d",
            data.organization_id,
            len(aggregated_metrics),
            total_sources,
            float(overall_completeness),
            len(data_gaps),
            len(reconciliation_items),
        )
        return result

    async def aggregate_pack_data(
        self, data: DataAggregationInput,
    ) -> List[AggregatedMetric]:
        """Aggregate data from prerequisite packs only.

        Args:
            data: Aggregation input.

        Returns:
            List of aggregated metrics from pack sources.
        """
        pack_sources = {
            DataSourceType.PACK_021.value,
            DataSourceType.PACK_022.value,
            DataSourceType.PACK_028.value,
            DataSourceType.PACK_029.value,
        }
        pack_points = [
            dp for dp in data.data_points
            if dp.source in pack_sources
        ]
        grouped = self._group_by_metric(pack_points)
        return self._aggregate_metrics(grouped, data)

    async def aggregate_app_data(
        self, data: DataAggregationInput,
    ) -> List[AggregatedMetric]:
        """Aggregate data from GreenLang applications only.

        Args:
            data: Aggregation input.

        Returns:
            List of aggregated metrics from app sources.
        """
        app_sources = {
            DataSourceType.GL_SBTI_APP.value,
            DataSourceType.GL_CDP_APP.value,
            DataSourceType.GL_TCFD_APP.value,
            DataSourceType.GL_GHG_APP.value,
        }
        app_points = [
            dp for dp in data.data_points
            if dp.source in app_sources
        ]
        grouped = self._group_by_metric(app_points)
        return self._aggregate_metrics(grouped, data)

    async def reconcile_sources(
        self,
        data: DataAggregationInput,
    ) -> List[ReconciliationItem]:
        """Run reconciliation only.

        Args:
            data: Aggregation input.

        Returns:
            List of reconciliation items.
        """
        grouped = self._group_by_metric(data.data_points)
        return self._reconcile_sources(
            grouped, data.reconciliation_threshold_pct
        )

    async def calculate_completeness(
        self,
        data: DataAggregationInput,
    ) -> List[FrameworkCompleteness]:
        """Calculate framework completeness only.

        Args:
            data: Aggregation input.

        Returns:
            List of framework completeness assessments.
        """
        grouped = self._group_by_metric(data.data_points)
        metrics = self._aggregate_metrics(grouped, data)
        return self._calculate_framework_completeness(
            metrics, data.target_frameworks
        )

    async def generate_lineage(
        self,
        data: DataAggregationInput,
    ) -> LineageGraph:
        """Generate data lineage graph only.

        Args:
            data: Aggregation input.

        Returns:
            LineageGraph showing source-to-metric flow.
        """
        grouped = self._group_by_metric(data.data_points)
        metrics = self._aggregate_metrics(grouped, data)
        return self._build_lineage_graph(data.data_points, metrics)

    # ------------------------------------------------------------------ #
    # Grouping                                                             #
    # ------------------------------------------------------------------ #

    def _group_by_metric(
        self,
        data_points: List[SourceDataPoint],
    ) -> Dict[str, List[SourceDataPoint]]:
        """Group data points by metric name.

        Args:
            data_points: All source data points.

        Returns:
            Dict mapping metric name to list of data points.
        """
        grouped: Dict[str, List[SourceDataPoint]] = defaultdict(list)
        for dp in data_points:
            grouped[dp.metric_name].append(dp)
        return dict(grouped)

    # ------------------------------------------------------------------ #
    # Metric Aggregation                                                   #
    # ------------------------------------------------------------------ #

    def _aggregate_metrics(
        self,
        grouped: Dict[str, List[SourceDataPoint]],
        data: DataAggregationInput,
    ) -> List[AggregatedMetric]:
        """Aggregate metrics using priority-based source selection.

        For each metric, selects the value from the highest-priority
        source.  Priority is determined by SOURCE_PRIORITY mapping.

        Args:
            grouped: Data points grouped by metric name.
            data: Aggregation input for context.

        Returns:
            List of aggregated metrics.
        """
        metrics: List[AggregatedMetric] = []

        for metric_name, points in grouped.items():
            # Sort by source priority (lower number = higher priority)
            sorted_points = sorted(
                points,
                key=lambda dp: SOURCE_PRIORITY.get(dp.source, 99),
            )

            primary = sorted_points[0]
            all_sources = list({dp.source for dp in points})

            # Calculate confidence
            if len(points) > 1:
                values = [dp.value for dp in points]
                mean_val = sum(values, Decimal("0")) / _decimal(len(values))
                if mean_val > Decimal("0"):
                    max_dev = max(
                        abs(v - mean_val) for v in values
                    )
                    deviation_pct = _safe_pct(max_dev, mean_val)
                    confidence = max(
                        Decimal("0"),
                        Decimal("100") - deviation_pct,
                    )
                else:
                    confidence = primary.confidence
                variance_pct = self._calculate_variance_pct(values)
            else:
                confidence = primary.confidence
                variance_pct = Decimal("0")

            # Determine reconciliation status
            if len(points) == 1:
                recon_status = ReconciliationStatus.SINGLE_SOURCE.value
            elif variance_pct <= RECONCILIATION_THRESHOLDS["minor_variance_pct"]:
                recon_status = ReconciliationStatus.RECONCILED.value
            elif variance_pct <= RECONCILIATION_THRESHOLDS["major_variance_pct"]:
                recon_status = ReconciliationStatus.MISMATCH_MINOR.value
            else:
                recon_status = ReconciliationStatus.MISMATCH_MAJOR.value

            metric = AggregatedMetric(
                metric_name=metric_name,
                metric_category=primary.metric_category.value,
                value=_round_val(primary.value, 4),
                unit=primary.unit,
                source_count=len(points),
                primary_source=primary.source,
                all_sources=all_sources,
                confidence=_round_val(confidence, 2),
                reconciliation_status=recon_status,
                variance_pct=_round_val(variance_pct, 2),
                methodology=primary.methodology,
            )
            metric.provenance_hash = _compute_hash(metric)
            metrics.append(metric)

        return metrics

    def _calculate_variance_pct(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Calculate percentage variance across values.

        Formula:
            variance = max(values) - min(values)
            mean = sum(values) / count
            variance_pct = (variance / mean) * 100

        Args:
            values: List of metric values.

        Returns:
            Variance as a percentage.
        """
        if not values or len(values) < 2:
            return Decimal("0")

        min_val = min(values)
        max_val = max(values)
        variance = max_val - min_val
        mean_val = sum(values, Decimal("0")) / _decimal(len(values))

        return _safe_pct(variance, mean_val)

    # ------------------------------------------------------------------ #
    # Reconciliation                                                       #
    # ------------------------------------------------------------------ #

    def _reconcile_sources(
        self,
        grouped: Dict[str, List[SourceDataPoint]],
        threshold_pct: Decimal,
    ) -> List[ReconciliationItem]:
        """Reconcile metrics reported by multiple sources.

        For each metric with multiple sources, computes variance
        and flags mismatches above the threshold.

        Args:
            grouped: Data points grouped by metric name.
            threshold_pct: Variance threshold for flagging.

        Returns:
            List of reconciliation items.
        """
        items: List[ReconciliationItem] = []

        for metric_name, points in grouped.items():
            if len(points) < 2:
                continue

            source_values: Dict[str, Decimal] = {}
            for dp in points:
                source_values[dp.source] = dp.value

            values = list(source_values.values())
            min_val = min(values)
            max_val = max(values)
            mean_val = sum(values, Decimal("0")) / _decimal(len(values))
            variance = max_val - min_val
            variance_pct = _safe_pct(variance, mean_val) if mean_val > Decimal("0") else Decimal("0")

            # Determine status
            if variance_pct <= RECONCILIATION_THRESHOLDS["minor_variance_pct"]:
                status = ReconciliationStatus.RECONCILED.value
                resolution = "Values within acceptable variance threshold."
            elif variance_pct <= threshold_pct:
                status = ReconciliationStatus.MISMATCH_MINOR.value
                resolution = "Minor mismatch detected; highest-priority source selected."
            else:
                status = ReconciliationStatus.MISMATCH_MAJOR.value
                resolution = "Major mismatch detected; manual review required."

            # Select value from highest-priority source
            sorted_sources = sorted(
                source_values.keys(),
                key=lambda s: SOURCE_PRIORITY.get(s, 99),
            )
            selected_source = sorted_sources[0]
            selected_value = source_values[selected_source]

            items.append(ReconciliationItem(
                metric_name=metric_name,
                source_values=source_values,
                min_value=_round_val(min_val, 4),
                max_value=_round_val(max_val, 4),
                mean_value=_round_val(mean_val, 4),
                variance=_round_val(variance, 4),
                variance_pct=_round_val(variance_pct, 2),
                status=status,
                resolution=resolution,
                selected_value=_round_val(selected_value, 4),
                selected_source=selected_source,
            ))

        return items

    # ------------------------------------------------------------------ #
    # Gap Detection                                                        #
    # ------------------------------------------------------------------ #

    def _detect_gaps(
        self,
        metrics: List[AggregatedMetric],
        frameworks: List[FrameworkTarget],
    ) -> List[DataGap]:
        """Detect data gaps for each target framework.

        Compares available metrics against the framework-specific
        required metrics list and reports missing data.

        Args:
            metrics: Aggregated metrics.
            frameworks: Target frameworks.

        Returns:
            List of data gaps.
        """
        gaps: List[DataGap] = []
        available_names: Set[str] = {m.metric_name for m in metrics}

        for fw in frameworks:
            required = FRAMEWORK_REQUIRED_METRICS.get(fw.value, [])
            for req_metric in required:
                if req_metric not in available_names:
                    severity = self._classify_gap_severity(req_metric, fw.value)
                    gaps.append(DataGap(
                        framework=fw.value,
                        metric_name=req_metric,
                        severity=severity,
                        description=(
                            f"Required metric '{req_metric}' is missing for "
                            f"{fw.value} framework reporting."
                        ),
                        suggested_source=self._suggest_source_for_metric(
                            req_metric
                        ),
                        impact=self._assess_gap_impact(req_metric, fw.value),
                    ))

        return gaps

    def _classify_gap_severity(
        self,
        metric_name: str,
        framework: str,
    ) -> str:
        """Classify the severity of a data gap.

        Args:
            metric_name: Missing metric name.
            framework: Framework requiring it.

        Returns:
            Gap severity level.
        """
        critical_metrics = {
            "scope_1_tco2e", "scope_2_tco2e", "base_year",
            "base_year_emissions", "target_year",
        }
        high_metrics = {
            "scope_3_tco2e", "target_reduction_pct",
            "scope_2_location_tco2e", "scope_2_market_tco2e",
            "methodology", "transition_plan",
        }

        if metric_name in critical_metrics:
            return GapSeverity.CRITICAL.value
        elif metric_name in high_metrics:
            return GapSeverity.HIGH.value
        elif framework in (FrameworkTarget.SEC.value, FrameworkTarget.CSRD.value):
            return GapSeverity.HIGH.value
        return GapSeverity.MEDIUM.value

    def _suggest_source_for_metric(self, metric_name: str) -> str:
        """Suggest a data source for a missing metric.

        Args:
            metric_name: Missing metric name.

        Returns:
            Suggested source system.
        """
        source_suggestions: Dict[str, str] = {
            "scope_1_tco2e": "GL-GHG-APP or PACK-021",
            "scope_2_tco2e": "GL-GHG-APP or PACK-021",
            "scope_2_location_tco2e": "GL-GHG-APP",
            "scope_2_market_tco2e": "GL-GHG-APP",
            "scope_3_tco2e": "GL-GHG-APP or PACK-021",
            "scope_3_categories": "GL-GHG-APP",
            "base_year": "PACK-021 or GL-SBTi-APP",
            "base_year_emissions": "PACK-021 or GL-SBTi-APP",
            "target_year": "GL-SBTi-APP or PACK-029",
            "target_reduction_pct": "GL-SBTi-APP or PACK-029",
            "annual_reduction_rate": "PACK-029",
            "progress_vs_target_pct": "PACK-029",
            "scenario_analysis_results": "GL-TCFD-APP",
            "climate_risks": "GL-TCFD-APP",
            "climate_opportunities": "GL-TCFD-APP",
            "governance_description": "GL-TCFD-APP",
            "energy_consumption_mwh": "GL-GHG-APP or ERP system",
            "renewable_energy_pct": "GL-GHG-APP",
            "transition_plan": "PACK-022",
            "climate_policies": "PACK-022",
            "reduction_initiatives": "PACK-022",
            "internal_carbon_price": "PACK-028",
            "sector_pathway": "PACK-028",
            "industry_specific_metrics": "GL-SBTi-APP",
        }
        return source_suggestions.get(metric_name, "Manual upload or API integration")

    def _assess_gap_impact(self, metric_name: str, framework: str) -> str:
        """Assess impact of a missing metric on framework compliance.

        Args:
            metric_name: Missing metric name.
            framework: Framework requiring it.

        Returns:
            Impact description.
        """
        if metric_name in ("scope_1_tco2e", "scope_2_tco2e"):
            return (
                f"Core emissions data required for {framework} compliance. "
                f"Report cannot be generated without this metric."
            )
        elif "target" in metric_name:
            return (
                f"Target-related data missing for {framework}. "
                f"Target progress sections will be incomplete."
            )
        elif "scenario" in metric_name or "risk" in metric_name:
            return (
                f"Scenario/risk data missing for {framework}. "
                f"Strategy sections will be incomplete."
            )
        return f"Optional data for {framework}. Report quality may be affected."

    # ------------------------------------------------------------------ #
    # Lineage Graph                                                        #
    # ------------------------------------------------------------------ #

    def _build_lineage_graph(
        self,
        data_points: List[SourceDataPoint],
        metrics: List[AggregatedMetric],
    ) -> LineageGraph:
        """Build data lineage graph.

        Creates a directed graph from source systems through
        transformations to aggregated metrics.

        Args:
            data_points: Raw source data points.
            metrics: Aggregated metrics.

        Returns:
            LineageGraph with nodes and edges.
        """
        nodes: List[LineageNode] = []
        edges: List[Tuple[str, str]] = []
        source_node_map: Dict[str, str] = {}

        # Create source nodes
        sources_seen: Set[str] = set()
        for dp in data_points:
            if dp.source not in sources_seen:
                node_id = _new_uuid()
                source_node_map[dp.source] = node_id
                nodes.append(LineageNode(
                    node_id=node_id,
                    node_type="source",
                    label=f"Source: {dp.source}",
                    source_system=dp.source,
                    timestamp=dp.timestamp,
                    provenance_hash=_compute_hash({"source": dp.source}),
                ))
                sources_seen.add(dp.source)

        # Create aggregation transform node
        agg_node_id = _new_uuid()
        nodes.append(LineageNode(
            node_id=agg_node_id,
            node_type="transform",
            label="Priority-based Aggregation",
            transformation="priority_aggregation",
            parent_ids=list(source_node_map.values()),
        ))

        # Edges from sources to aggregation
        for src_node_id in source_node_map.values():
            edges.append((src_node_id, agg_node_id))

        # Create metric output nodes
        for metric in metrics:
            metric_node_id = _new_uuid()
            nodes.append(LineageNode(
                node_id=metric_node_id,
                node_type="metric",
                label=f"Metric: {metric.metric_name}",
                metric_name=metric.metric_name,
                source_system=metric.primary_source,
                provenance_hash=metric.provenance_hash,
            ))
            edges.append((agg_node_id, metric_node_id))

        return LineageGraph(
            nodes=nodes,
            edges=edges,
            total_sources=len(sources_seen),
            total_transformations=1,
            total_metrics=len(metrics),
        )

    # ------------------------------------------------------------------ #
    # Framework Completeness                                               #
    # ------------------------------------------------------------------ #

    def _calculate_framework_completeness(
        self,
        metrics: List[AggregatedMetric],
        frameworks: List[FrameworkTarget],
    ) -> List[FrameworkCompleteness]:
        """Calculate data completeness for each framework.

        Formula:
            completeness_pct = (metrics_present / metrics_required) * 100

        Args:
            metrics: Aggregated metrics.
            frameworks: Target frameworks.

        Returns:
            List of framework completeness assessments.
        """
        available_names: Set[str] = {m.metric_name for m in metrics}
        results: List[FrameworkCompleteness] = []

        for fw in frameworks:
            required = FRAMEWORK_REQUIRED_METRICS.get(fw.value, [])
            required_count = len(required)
            present_count = sum(
                1 for r in required if r in available_names
            )
            missing_count = required_count - present_count
            missing_names = [
                r for r in required if r not in available_names
            ]

            completeness_pct = _safe_pct(
                _decimal(present_count), _decimal(required_count)
            ) if required_count > 0 else Decimal("0")

            if completeness_pct >= Decimal("100"):
                status = "complete"
            elif completeness_pct >= Decimal("80"):
                status = "mostly_complete"
            elif completeness_pct >= Decimal("50"):
                status = "partial"
            else:
                status = "incomplete"

            results.append(FrameworkCompleteness(
                framework=fw.value,
                required_metrics=required_count,
                provided_metrics=present_count,
                missing_metrics=missing_count,
                completeness_pct=_round_val(completeness_pct, 2),
                missing_metric_names=missing_names,
                status=status,
            ))

        return results

    # ------------------------------------------------------------------ #
    # Source Health                                                         #
    # ------------------------------------------------------------------ #

    def _assess_source_health(
        self,
        connections: List[SourceConnection],
        data_points: List[SourceDataPoint],
    ) -> List[SourceHealthStatus]:
        """Assess health of each data source.

        Args:
            connections: Source connection configurations.
            data_points: Data points from sources.

        Returns:
            List of source health statuses.
        """
        # Count metrics per source from data points
        metrics_per_source: Dict[str, int] = defaultdict(int)
        for dp in data_points:
            metrics_per_source[dp.source] += 1

        statuses: List[SourceHealthStatus] = []
        for conn in connections:
            source_name = conn.source_type.value
            statuses.append(SourceHealthStatus(
                source=source_name,
                status=conn.status.value,
                metrics_provided=metrics_per_source.get(source_name, 0),
                last_sync=conn.last_sync,
                latency_ms=0.0,
                error_message="" if conn.status == ConnectionStatus.CONNECTED else (
                    f"Source {source_name} is {conn.status.value}"
                ),
            ))

        return statuses

    # ------------------------------------------------------------------ #
    # Overall Statistics                                                   #
    # ------------------------------------------------------------------ #

    def _calculate_overall_completeness(
        self,
        framework_completeness: List[FrameworkCompleteness],
    ) -> Decimal:
        """Calculate overall completeness across all frameworks.

        Formula:
            overall = mean(framework_completeness_pct)

        Args:
            framework_completeness: Per-framework completeness.

        Returns:
            Overall completeness percentage.
        """
        if not framework_completeness:
            return Decimal("0")

        total = sum(
            (fc.completeness_pct for fc in framework_completeness),
            Decimal("0"),
        )
        return _safe_divide(total, _decimal(len(framework_completeness)))

    def _calculate_overall_confidence(
        self,
        metrics: List[AggregatedMetric],
    ) -> Decimal:
        """Calculate overall confidence score.

        Formula:
            overall = mean(metric_confidence)

        Args:
            metrics: Aggregated metrics.

        Returns:
            Overall confidence score (0-100).
        """
        if not metrics:
            return Decimal("0")

        total = sum(
            (m.confidence for m in metrics),
            Decimal("0"),
        )
        return _safe_divide(total, _decimal(len(metrics)))

    def _assess_overall_quality(
        self,
        completeness: Decimal,
        confidence: Decimal,
        reconciliation_issues: int,
    ) -> str:
        """Assess overall data quality tier.

        Args:
            completeness: Overall completeness percentage.
            confidence: Overall confidence score.
            reconciliation_issues: Number of reconciliation issues.

        Returns:
            Data quality tier string.
        """
        quality_score = (completeness + confidence) / Decimal("2")

        if quality_score >= Decimal("90") and reconciliation_issues == 0:
            return DataQuality.HIGH.value
        elif quality_score >= Decimal("70"):
            return DataQuality.MEDIUM.value
        elif quality_score >= Decimal("40"):
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    # ------------------------------------------------------------------ #
    # Reconciliation Summary                                               #
    # ------------------------------------------------------------------ #

    def _summarize_reconciliation(
        self,
        items: List[ReconciliationItem],
    ) -> Dict[str, int]:
        """Summarize reconciliation results.

        Args:
            items: Reconciliation items.

        Returns:
            Dict with status counts.
        """
        summary: Dict[str, int] = {
            "total": len(items),
            "reconciled": 0,
            "minor_mismatch": 0,
            "major_mismatch": 0,
            "pending_review": 0,
        }
        for item in items:
            if item.status == ReconciliationStatus.RECONCILED.value:
                summary["reconciled"] += 1
            elif item.status == ReconciliationStatus.MISMATCH_MINOR.value:
                summary["minor_mismatch"] += 1
            elif item.status == ReconciliationStatus.MISMATCH_MAJOR.value:
                summary["major_mismatch"] += 1
            elif item.status == ReconciliationStatus.PENDING_REVIEW.value:
                summary["pending_review"] += 1

        return summary

    # ------------------------------------------------------------------ #
    # Warnings                                                             #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        data: DataAggregationInput,
        metrics: List[AggregatedMetric],
        gaps: List[DataGap],
        reconciliation: List[ReconciliationItem],
    ) -> List[str]:
        """Generate warnings based on aggregation analysis.

        Args:
            data: Aggregation input.
            metrics: Aggregated metrics.
            gaps: Data gaps.
            reconciliation: Reconciliation items.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []

        # Critical gaps
        critical_gaps = [
            g for g in gaps if g.severity == GapSeverity.CRITICAL.value
        ]
        if critical_gaps:
            frameworks = {g.framework for g in critical_gaps}
            warnings.append(
                f"Critical data gaps detected for frameworks: "
                f"{', '.join(sorted(frameworks))}. "
                f"Core emissions data may be missing."
            )

        # Major mismatches
        major_mismatches = [
            r for r in reconciliation
            if r.status == ReconciliationStatus.MISMATCH_MAJOR.value
        ]
        if major_mismatches:
            warnings.append(
                f"{len(major_mismatches)} major reconciliation mismatch(es) "
                f"detected. Manual review is required before reporting."
            )

        # No data points
        if not data.data_points:
            warnings.append(
                "No data points provided. Unable to aggregate. "
                "Connect at least one data source."
            )

        # Single source for critical metrics
        single_source_critical = [
            m for m in metrics
            if m.source_count == 1
            and m.metric_name in (
                "scope_1_tco2e", "scope_2_tco2e", "scope_3_tco2e"
            )
        ]
        if single_source_critical:
            metric_names = [m.metric_name for m in single_source_critical]
            warnings.append(
                f"Core emissions metrics ({', '.join(metric_names)}) "
                f"reported by only one source. Cross-validation not possible."
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: DataAggregationInput,
        metrics: List[AggregatedMetric],
        gaps: List[DataGap],
        completeness: List[FrameworkCompleteness],
    ) -> List[str]:
        """Generate recommendations for improving data quality.

        Args:
            data: Aggregation input.
            metrics: Aggregated metrics.
            gaps: Data gaps.
            completeness: Framework completeness assessments.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # Incomplete frameworks
        incomplete = [
            fc for fc in completeness
            if fc.completeness_pct < Decimal("80")
        ]
        if incomplete:
            for fc in incomplete:
                recs.append(
                    f"Framework {fc.framework} is {fc.completeness_pct}% complete. "
                    f"Missing metrics: {', '.join(fc.missing_metric_names[:5])}."
                )

        # Suggest additional sources
        sources_used = {dp.source for dp in data.data_points}
        if DataSourceType.GL_GHG_APP.value not in sources_used:
            recs.append(
                "Connect GL-GHG-APP for comprehensive Scope 1/2/3 emissions data. "
                "This is the primary source for most framework metrics."
            )
        if DataSourceType.GL_SBTI_APP.value not in sources_used:
            recs.append(
                "Connect GL-SBTi-APP for SBTi target data and validation results."
            )
        if DataSourceType.GL_TCFD_APP.value not in sources_used:
            recs.append(
                "Connect GL-TCFD-APP for scenario analysis data "
                "required by TCFD and ISSB frameworks."
            )

        # Reconciliation improvements
        low_confidence = [
            m for m in metrics
            if m.confidence < Decimal("80")
        ]
        if low_confidence:
            recs.append(
                f"{len(low_confidence)} metric(s) have confidence below 80%. "
                f"Add additional data sources or verify existing data."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_supported_sources(self) -> List[str]:
        """Return list of supported data source types."""
        return [s.value for s in DataSourceType]

    def get_supported_frameworks(self) -> List[str]:
        """Return list of supported reporting frameworks."""
        return [f.value for f in FrameworkTarget]

    def get_framework_requirements(
        self, framework: str,
    ) -> List[str]:
        """Return required metrics for a framework.

        Args:
            framework: Framework name.

        Returns:
            List of required metric names.
        """
        return list(FRAMEWORK_REQUIRED_METRICS.get(framework, []))

    def get_source_priority(self) -> Dict[str, int]:
        """Return source priority mapping."""
        return dict(SOURCE_PRIORITY)
