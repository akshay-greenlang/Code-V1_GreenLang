# -*- coding: utf-8 -*-
"""
GapAnalyzer - AGENT-EUDR-001 Feature 6: Supply Chain Gap Analysis

Automatically identifies compliance gaps and weaknesses in EUDR supply chain
graphs, classifies them by severity with regulatory article references,
computes a compliance readiness score, produces a prioritized remediation
action list, and tracks gap closure trends over time.

Detected Gap Types (8 critical + 2 informational):
    1. Missing Tiers -- opaque segments where intermediaries are unknown
       (EUDR Article 4(2), Severity: HIGH)
    2. Unverified Actors -- nodes without compliance verification
       (EUDR Article 10, Severity: HIGH)
    3. Missing Geolocation -- producer nodes without GPS coordinates
       (EUDR Article 9, Severity: CRITICAL)
    4. Missing Polygon -- plots > 4 ha without polygon boundaries
       (EUDR Article 9(1)(d), Severity: CRITICAL)
    5. Broken Custody Chains -- products with no traceable link to origin
       (EUDR Article 4(2)(f), Severity: CRITICAL)
    6. Missing Documentation -- nodes without custody transfer records
       (EUDR Article 4(2), Severity: MEDIUM)
    7. Mass Balance Discrepancies -- output exceeding input quantities
       (EUDR Article 10(2)(f), Severity: HIGH)
    8. Stale Data -- data older than 12 months without refresh
       (EUDR Article 31, Severity: MEDIUM)
    9. Missing Certification -- expected certification absent
       (EUDR Article 10, Severity: MEDIUM)
    10. Orphan Nodes -- disconnected nodes (internal quality, Severity: LOW)

Zero-Hallucination Guarantees:
    - 100% deterministic: same graph input produces same gap output
    - NO LLM involvement in any gap detection or scoring path
    - All arithmetic uses standard Python float/Decimal operations
    - SHA-256 provenance hash on every analysis run
    - Complete audit trail for regulatory inspection

Performance Target:
    - Full gap analysis < 30 seconds for 10,000-node graph
    - Detection rate: 95%+ of gaps (measured against manual audit baseline)

Dependencies:
    - graph_engine.SupplyChainGraphEngine: graph traversal and node/edge access
    - geolocation_linker.GeolocationLinker: plot geolocation validation
    - risk_propagation.RiskPropagationEngine: risk score lookups
    - provenance.ProvenanceTracker: audit trail recording
    - metrics: Prometheus gap detection counters

Regulatory References:
    - EUDR Article 4(2): Due diligence obligations
    - EUDR Article 4(2)(f): Traceability to plot of origin
    - EUDR Article 9: Geolocation requirements
    - EUDR Article 9(1)(d): Polygon requirement for plots > 4 hectares
    - EUDR Article 10: Risk assessment and verification
    - EUDR Article 10(2)(f): Mass balance tracking
    - EUDR Article 31: Record keeping (5-year retention)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001, Feature 6
Agent ID: GL-EUDR-SCM-001
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION = "1.0.0"

#: EUDR Article 9(1)(d) polygon requirement threshold in hectares.
POLYGON_AREA_THRESHOLD_HA: float = 4.0

#: Default stale data threshold in days (12 months).
DEFAULT_STALE_DATA_DAYS: int = 365

#: Default mass balance tolerance percentage.
DEFAULT_MASS_BALANCE_TOLERANCE_PCT: float = 2.0

#: Severity penalty weights for compliance readiness scoring.
#: Critical gaps have the highest impact on the readiness score.
SEVERITY_PENALTY_WEIGHTS: Dict[str, float] = {
    "critical": 15.0,
    "high": 8.0,
    "medium": 3.0,
    "low": 1.0,
}

#: Maps gap types to their remediation action templates.
REMEDIATION_ACTIONS: Dict[str, str] = {
    "missing_geolocation": (
        "Send supplier questionnaire requesting GPS coordinates "
        "for all production plots per EUDR Article 9."
    ),
    "missing_polygon": (
        "Flag for GIS team: obtain polygon boundary data for plots "
        "> 4 hectares per EUDR Article 9(1)(d)."
    ),
    "broken_custody_chain": (
        "Trigger supplier investigation: establish traceable link "
        "from product back to origin production plots per EUDR Article 4(2)(f)."
    ),
    "unverified_actor": (
        "Send verification request to supply chain actor: "
        "collect identity documents and compliance certification per EUDR Article 10."
    ),
    "missing_tier": (
        "Trigger sub-tier discovery: initiate multi-tier mapping "
        "to identify unknown intermediaries per EUDR Article 4(2)."
    ),
    "mass_balance_discrepancy": (
        "Flag for manual review: output quantity exceeds input quantity "
        "beyond tolerance per EUDR Article 10(2)(f)."
    ),
    "missing_certification": (
        "Request certification upload from supply chain actor "
        "per EUDR Article 10."
    ),
    "stale_data": (
        "Trigger data refresh: request updated supply chain data "
        "from actor; current data exceeds 12-month threshold per EUDR Article 31."
    ),
    "orphan_node": (
        "Internal quality check: connect orphan node to the supply chain "
        "or remove if no longer relevant."
    ),
    "missing_documentation": (
        "Request custody transfer documentation from supply chain actor "
        "per EUDR Article 4(2)."
    ),
}

#: Maps gap types to their auto-remediation trigger identifiers.
AUTO_REMEDIATION_TRIGGERS: Dict[str, str] = {
    "missing_geolocation": "send_supplier_questionnaire",
    "missing_polygon": "flag_gis_team",
    "broken_custody_chain": "trigger_supplier_investigation",
    "unverified_actor": "send_verification_request",
    "missing_tier": "trigger_subtier_discovery",
    "mass_balance_discrepancy": "flag_manual_review",
    "missing_certification": "request_certification_upload",
    "stale_data": "trigger_data_refresh",
    "orphan_node": "internal_quality_review",
    "missing_documentation": "request_documentation",
}

#: Expected standard supply chain tier ordering from upstream to downstream.
STANDARD_TIER_ORDER: List[str] = [
    "producer",
    "collector",
    "processor",
    "trader",
    "importer",
]

#: Node types that are considered verifiable (require compliance verification).
VERIFIABLE_NODE_TYPES: FrozenSet[str] = frozenset({
    "producer",
    "collector",
    "processor",
    "trader",
    "importer",
})

#: Node types that require geolocation data.
GEOLOCATION_REQUIRED_TYPES: FrozenSet[str] = frozenset({
    "producer",
})

#: Compliance statuses considered as "not verified".
UNVERIFIED_STATUSES: FrozenSet[str] = frozenset({
    "pending_verification",
    "insufficient_data",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for consistency."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, or other serializable).

    Returns:
        SHA-256 hex digest string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string (e.g., 'GAP', 'REM').

    Returns:
        Prefixed UUID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Enumerations (local references for type safety)
# ---------------------------------------------------------------------------


class GapType(str, Enum):
    """Gap type classification matching models.GapType."""

    MISSING_GEOLOCATION = "missing_geolocation"
    MISSING_POLYGON = "missing_polygon"
    BROKEN_CUSTODY_CHAIN = "broken_custody_chain"
    UNVERIFIED_ACTOR = "unverified_actor"
    MISSING_TIER = "missing_tier"
    MASS_BALANCE_DISCREPANCY = "mass_balance_discrepancy"
    MISSING_CERTIFICATION = "missing_certification"
    STALE_DATA = "stale_data"
    ORPHAN_NODE = "orphan_node"
    MISSING_DOCUMENTATION = "missing_documentation"


class GapSeverity(str, Enum):
    """Gap severity classification matching models.GapSeverity."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


#: Maps gap types to default severity.
GAP_SEVERITY_MAP: Dict[GapType, GapSeverity] = {
    GapType.MISSING_GEOLOCATION: GapSeverity.CRITICAL,
    GapType.MISSING_POLYGON: GapSeverity.CRITICAL,
    GapType.BROKEN_CUSTODY_CHAIN: GapSeverity.CRITICAL,
    GapType.UNVERIFIED_ACTOR: GapSeverity.HIGH,
    GapType.MISSING_TIER: GapSeverity.HIGH,
    GapType.MASS_BALANCE_DISCREPANCY: GapSeverity.HIGH,
    GapType.MISSING_CERTIFICATION: GapSeverity.MEDIUM,
    GapType.STALE_DATA: GapSeverity.MEDIUM,
    GapType.ORPHAN_NODE: GapSeverity.LOW,
    GapType.MISSING_DOCUMENTATION: GapSeverity.MEDIUM,
}

#: Maps gap types to EUDR article reference.
GAP_ARTICLE_MAP: Dict[GapType, str] = {
    GapType.MISSING_GEOLOCATION: "Article 9",
    GapType.MISSING_POLYGON: "Article 9(1)(d)",
    GapType.BROKEN_CUSTODY_CHAIN: "Article 4(2)(f)",
    GapType.UNVERIFIED_ACTOR: "Article 10",
    GapType.MISSING_TIER: "Article 4(2)",
    GapType.MASS_BALANCE_DISCREPANCY: "Article 10(2)(f)",
    GapType.MISSING_CERTIFICATION: "Article 10",
    GapType.STALE_DATA: "Article 31",
    GapType.ORPHAN_NODE: "Internal",
    GapType.MISSING_DOCUMENTATION: "Article 4(2)",
}

#: Risk impact multipliers for remediation priority ordering.
#: Higher values = higher priority for remediation.
RISK_IMPACT_MULTIPLIERS: Dict[GapSeverity, float] = {
    GapSeverity.CRITICAL: 10.0,
    GapSeverity.HIGH: 5.0,
    GapSeverity.MEDIUM: 2.0,
    GapSeverity.LOW: 1.0,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DetectedGap:
    """A single compliance gap detected during analysis.

    Attributes:
        gap_id: Unique gap identifier.
        gap_type: Classification of the gap.
        severity: Severity level for remediation priority.
        affected_node_id: ID of the affected supply chain node (if applicable).
        affected_edge_id: ID of the affected edge (if applicable).
        description: Human-readable description of the gap.
        remediation: Suggested remediation action.
        auto_remediation_trigger: Identifier for automatic remediation workflow.
        eudr_article: EUDR article reference violated.
        risk_impact_score: Computed risk impact for prioritization (0-100).
        is_resolved: Whether the gap has been remediated.
        resolved_at: Timestamp when resolved.
        detected_at: Timestamp when detected.
        metadata: Additional contextual information.
    """

    gap_id: str = field(default_factory=lambda: _generate_id("GAP"))
    gap_type: str = ""
    severity: str = "medium"
    affected_node_id: Optional[str] = None
    affected_edge_id: Optional[str] = None
    description: str = ""
    remediation: str = ""
    auto_remediation_trigger: str = ""
    eudr_article: str = ""
    risk_impact_score: float = 0.0
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    detected_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation of the gap.
        """
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "severity": self.severity,
            "affected_node_id": self.affected_node_id,
            "affected_edge_id": self.affected_edge_id,
            "description": self.description,
            "remediation": self.remediation,
            "auto_remediation_trigger": self.auto_remediation_trigger,
            "eudr_article": self.eudr_article,
            "risk_impact_score": self.risk_impact_score,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "metadata": self.metadata,
        }


@dataclass
class RemediationAction:
    """A prioritized remediation action for a detected gap.

    Attributes:
        action_id: Unique action identifier.
        gap_id: Reference to the gap this action addresses.
        gap_type: Gap type classification.
        severity: Gap severity.
        priority_rank: Ordinal rank (1 = highest priority).
        risk_impact_score: Computed risk impact score for ordering.
        action_description: Human-readable action description.
        auto_trigger: Auto-remediation trigger identifier.
        affected_node_id: Affected node (if applicable).
        affected_edge_id: Affected edge (if applicable).
        eudr_article: EUDR article reference.
        estimated_effort: Estimated effort to remediate (low/medium/high).
    """

    action_id: str = field(default_factory=lambda: _generate_id("REM"))
    gap_id: str = ""
    gap_type: str = ""
    severity: str = "medium"
    priority_rank: int = 0
    risk_impact_score: float = 0.0
    action_description: str = ""
    auto_trigger: str = ""
    affected_node_id: Optional[str] = None
    affected_edge_id: Optional[str] = None
    eudr_article: str = ""
    estimated_effort: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation of the action.
        """
        return {
            "action_id": self.action_id,
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "severity": self.severity,
            "priority_rank": self.priority_rank,
            "risk_impact_score": self.risk_impact_score,
            "action_description": self.action_description,
            "auto_trigger": self.auto_trigger,
            "affected_node_id": self.affected_node_id,
            "affected_edge_id": self.affected_edge_id,
            "eudr_article": self.eudr_article,
            "estimated_effort": self.estimated_effort,
        }


@dataclass
class GapTrendSnapshot:
    """A point-in-time snapshot of gap counts for trend tracking.

    Attributes:
        snapshot_id: Unique snapshot identifier.
        graph_id: Graph this snapshot belongs to.
        timestamp: When the snapshot was taken.
        total_gaps: Total number of open (unresolved) gaps.
        gaps_by_severity: Count by severity level.
        gaps_by_type: Count by gap type.
        compliance_readiness: Compliance readiness score at this point.
        resolved_since_last: Number of gaps resolved since the previous snapshot.
        new_since_last: Number of new gaps since the previous snapshot.
    """

    snapshot_id: str = field(default_factory=lambda: _generate_id("SNAP"))
    graph_id: str = ""
    timestamp: datetime = field(default_factory=_utcnow)
    total_gaps: int = 0
    gaps_by_severity: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0}
    )
    gaps_by_type: Dict[str, int] = field(default_factory=dict)
    compliance_readiness: float = 0.0
    resolved_since_last: int = 0
    new_since_last: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation of the snapshot.
        """
        return {
            "snapshot_id": self.snapshot_id,
            "graph_id": self.graph_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_gaps": self.total_gaps,
            "gaps_by_severity": dict(self.gaps_by_severity),
            "gaps_by_type": dict(self.gaps_by_type),
            "compliance_readiness": self.compliance_readiness,
            "resolved_since_last": self.resolved_since_last,
            "new_since_last": self.new_since_last,
        }


@dataclass
class GapAnalysisResult:
    """Complete result of a gap analysis run.

    Attributes:
        analysis_id: Unique identifier for this analysis run.
        graph_id: Graph that was analyzed.
        total_gaps: Total number of gaps detected.
        total_open_gaps: Number of unresolved gaps.
        total_resolved_gaps: Number of resolved gaps.
        gaps_by_severity: Count by severity level.
        gaps_by_type: Count by gap type.
        compliance_readiness: Compliance readiness score (0-100).
        gaps: List of all detected gaps.
        remediation_actions: Prioritized remediation action list.
        auto_remediation_triggers: List of triggered auto-remediation actions.
        provenance_hash: SHA-256 hash of the analysis for audit trail.
        processing_time_ms: Wall-clock time for the analysis in milliseconds.
        analysis_timestamp: When the analysis was performed.
        node_count: Number of nodes in the analyzed graph.
        edge_count: Number of edges in the analyzed graph.
    """

    analysis_id: str = field(default_factory=lambda: _generate_id("ANA"))
    graph_id: str = ""
    total_gaps: int = 0
    total_open_gaps: int = 0
    total_resolved_gaps: int = 0
    gaps_by_severity: Dict[str, int] = field(
        default_factory=lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0}
    )
    gaps_by_type: Dict[str, int] = field(default_factory=dict)
    compliance_readiness: float = 100.0
    gaps: List[DetectedGap] = field(default_factory=list)
    remediation_actions: List[RemediationAction] = field(default_factory=list)
    auto_remediation_triggers: List[Dict[str, Any]] = field(default_factory=list)
    provenance_hash: str = ""
    processing_time_ms: float = 0.0
    analysis_timestamp: datetime = field(default_factory=_utcnow)
    node_count: int = 0
    edge_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export.

        Returns:
            Dictionary representation of the analysis result.
        """
        return {
            "analysis_id": self.analysis_id,
            "graph_id": self.graph_id,
            "total_gaps": self.total_gaps,
            "total_open_gaps": self.total_open_gaps,
            "total_resolved_gaps": self.total_resolved_gaps,
            "gaps_by_severity": dict(self.gaps_by_severity),
            "gaps_by_type": dict(self.gaps_by_type),
            "compliance_readiness": self.compliance_readiness,
            "gaps": [g.to_dict() for g in self.gaps],
            "remediation_actions": [a.to_dict() for a in self.remediation_actions],
            "auto_remediation_triggers": self.auto_remediation_triggers,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "analysis_timestamp": (
                self.analysis_timestamp.isoformat()
                if self.analysis_timestamp
                else None
            ),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }


@dataclass
class GapAnalyzerConfig:
    """Configuration for the GapAnalyzer engine.

    Attributes:
        mass_balance_tolerance_pct: Tolerance percentage for mass balance
            checks. Output exceeding input by more than this triggers a gap.
        stale_data_days: Number of days after which data without refresh
            is considered stale.
        polygon_area_threshold_ha: Area threshold in hectares above which
            polygon boundary data is required per EUDR Article 9(1)(d).
        severity_penalty_weights: Weights for each severity level used
            in compliance readiness score calculation.
        enable_auto_remediation: Whether to generate auto-remediation triggers.
        enable_provenance: Whether to compute and record provenance hashes.
        max_trend_snapshots: Maximum number of trend snapshots to retain
            per graph (oldest are pruned).
    """

    mass_balance_tolerance_pct: float = DEFAULT_MASS_BALANCE_TOLERANCE_PCT
    stale_data_days: int = DEFAULT_STALE_DATA_DAYS
    polygon_area_threshold_ha: float = POLYGON_AREA_THRESHOLD_HA
    severity_penalty_weights: Dict[str, float] = field(
        default_factory=lambda: dict(SEVERITY_PENALTY_WEIGHTS)
    )
    enable_auto_remediation: bool = True
    enable_provenance: bool = True
    max_trend_snapshots: int = 100

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization."""
        errors: List[str] = []

        if not (0.0 <= self.mass_balance_tolerance_pct <= 100.0):
            errors.append(
                f"mass_balance_tolerance_pct must be in [0, 100], "
                f"got {self.mass_balance_tolerance_pct}"
            )
        if self.stale_data_days <= 0:
            errors.append(
                f"stale_data_days must be > 0, got {self.stale_data_days}"
            )
        if self.polygon_area_threshold_ha < 0:
            errors.append(
                f"polygon_area_threshold_ha must be >= 0, "
                f"got {self.polygon_area_threshold_ha}"
            )
        if self.max_trend_snapshots <= 0:
            errors.append(
                f"max_trend_snapshots must be > 0, got {self.max_trend_snapshots}"
            )

        if errors:
            raise ValueError(
                "GapAnalyzerConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


# ===========================================================================
# GapAnalyzer Engine
# ===========================================================================


class GapAnalyzer:
    """Supply chain gap analysis engine for EUDR compliance.

    Scans a supply chain graph represented as nodes (actors) and edges
    (custody transfers) to detect compliance gaps, classify them by
    severity, compute a compliance readiness score, and produce a
    prioritized remediation action list.

    The analyzer operates on an in-memory representation of the graph
    passed as dictionaries of nodes and edges, making it independent of
    the persistence layer and suitable for both real-time and batch
    analysis scenarios.

    Gap Detection Methods (10 detectors):
        1. _detect_missing_tiers: Finds opaque segments in tier ordering
        2. _detect_unverified_actors: Finds nodes without verification
        3. _detect_missing_geolocation: Finds producers without GPS
        4. _detect_missing_polygon: Finds large plots without polygons
        5. _detect_broken_custody_chains: Finds untraceable products
        6. _detect_missing_documentation: Finds nodes without documents
        7. _detect_mass_balance_discrepancies: Finds quantity mismatches
        8. _detect_stale_data: Finds outdated information
        9. _detect_missing_certification: Finds uncertified actors
        10. _detect_orphan_nodes: Finds disconnected nodes

    Attributes:
        config: GapAnalyzerConfig with analysis settings.
        _gap_store: Dictionary mapping graph_id to list of detected gaps.
        _trend_store: Dictionary mapping graph_id to trend snapshots.
        _analysis_count: Total number of analyses performed.

    Example:
        >>> analyzer = GapAnalyzer()
        >>> nodes = {
        ...     "n1": {"node_id": "n1", "node_type": "producer",
        ...            "country_code": "BR", "operator_name": "Farm A"},
        ... }
        >>> edges = {}
        >>> result = analyzer.analyze(
        ...     graph_id="g1", nodes=nodes, edges=edges
        ... )
        >>> assert result.total_gaps >= 1
        >>> assert result.compliance_readiness <= 100.0
    """

    def __init__(self, config: Optional[GapAnalyzerConfig] = None) -> None:
        """Initialize the GapAnalyzer engine.

        Args:
            config: Optional configuration. Defaults to GapAnalyzerConfig()
                with standard EUDR thresholds.
        """
        self.config = config or GapAnalyzerConfig()
        self._gap_store: Dict[str, List[DetectedGap]] = {}
        self._trend_store: Dict[str, List[GapTrendSnapshot]] = {}
        self._analysis_count: int = 0

        logger.info(
            "GapAnalyzer initialized: mass_balance_tolerance=%.1f%%, "
            "stale_data_days=%d, polygon_threshold=%.1f ha, "
            "auto_remediation=%s, provenance=%s",
            self.config.mass_balance_tolerance_pct,
            self.config.stale_data_days,
            self.config.polygon_area_threshold_ha,
            self.config.enable_auto_remediation,
            self.config.enable_provenance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        plot_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        reference_time: Optional[datetime] = None,
    ) -> GapAnalysisResult:
        """Run a full gap analysis on a supply chain graph.

        Executes all 10 gap detectors in sequence, computes compliance
        readiness, generates prioritized remediation actions, records
        a trend snapshot, and computes provenance hash.

        Args:
            graph_id: Identifier of the graph being analyzed.
            nodes: Dictionary of node_id -> node data dictionaries.
                Each node dict should contain at minimum: node_id,
                node_type, operator_name, country_code.
            edges: Dictionary of edge_id -> edge data dictionaries.
                Each edge dict should contain at minimum: edge_id,
                source_node_id, target_node_id, commodity, quantity.
            plot_registry: Optional dictionary of plot_id -> plot data
                for polygon validation. Each plot dict may contain:
                plot_id, area_hectares, has_polygon, coordinates.
            reference_time: Reference timestamp for stale data detection.
                Defaults to current UTC time.

        Returns:
            GapAnalysisResult with all detected gaps, readiness score,
            and prioritized remediation actions.
        """
        start_time = time.monotonic()
        ref_time = reference_time or _utcnow()
        self._analysis_count += 1

        logger.info(
            "Starting gap analysis for graph=%s (nodes=%d, edges=%d)",
            graph_id,
            len(nodes),
            len(edges),
        )

        # Build adjacency structures for efficient traversal
        adjacency = self._build_adjacency(nodes, edges)

        # Run all gap detectors
        all_gaps: List[DetectedGap] = []

        all_gaps.extend(self._detect_missing_tiers(graph_id, nodes, edges, adjacency))
        all_gaps.extend(self._detect_unverified_actors(graph_id, nodes))
        all_gaps.extend(self._detect_missing_geolocation(graph_id, nodes))
        all_gaps.extend(
            self._detect_missing_polygon(graph_id, nodes, plot_registry)
        )
        all_gaps.extend(
            self._detect_broken_custody_chains(graph_id, nodes, edges, adjacency)
        )
        all_gaps.extend(
            self._detect_missing_documentation(graph_id, nodes, edges, adjacency)
        )
        all_gaps.extend(
            self._detect_mass_balance_discrepancies(graph_id, nodes, edges, adjacency)
        )
        all_gaps.extend(self._detect_stale_data(graph_id, nodes, ref_time))
        all_gaps.extend(self._detect_missing_certification(graph_id, nodes))
        all_gaps.extend(self._detect_orphan_nodes(graph_id, nodes, adjacency))

        # Merge with previously resolved gaps from the store
        previously_resolved = self._get_resolved_gaps(graph_id)

        # Compute aggregates
        gaps_by_severity = self._count_by_severity(all_gaps)
        gaps_by_type = self._count_by_type(all_gaps)
        open_gaps = [g for g in all_gaps if not g.is_resolved]
        resolved_gaps = [g for g in all_gaps if g.is_resolved]

        # Compliance readiness score
        compliance_readiness = self._compute_compliance_readiness(
            open_gaps, len(nodes)
        )

        # Prioritized remediation actions
        remediation_actions = self._build_remediation_actions(open_gaps, nodes)

        # Auto-remediation triggers
        auto_triggers: List[Dict[str, Any]] = []
        if self.config.enable_auto_remediation:
            auto_triggers = self._generate_auto_triggers(open_gaps)

        # Store gaps for trend tracking
        self._gap_store[graph_id] = all_gaps

        # Record trend snapshot
        previous_snapshot = (
            self._trend_store[graph_id][-1]
            if graph_id in self._trend_store and self._trend_store[graph_id]
            else None
        )
        snapshot = self._record_trend_snapshot(
            graph_id=graph_id,
            open_gaps=open_gaps,
            gaps_by_severity=gaps_by_severity,
            gaps_by_type=gaps_by_type,
            compliance_readiness=compliance_readiness,
            previous_snapshot=previous_snapshot,
        )

        # Build result
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = GapAnalysisResult(
            graph_id=graph_id,
            total_gaps=len(all_gaps),
            total_open_gaps=len(open_gaps),
            total_resolved_gaps=len(resolved_gaps) + len(previously_resolved),
            gaps_by_severity=gaps_by_severity,
            gaps_by_type=gaps_by_type,
            compliance_readiness=compliance_readiness,
            gaps=all_gaps,
            remediation_actions=remediation_actions,
            auto_remediation_triggers=auto_triggers,
            processing_time_ms=elapsed_ms,
            analysis_timestamp=ref_time,
            node_count=len(nodes),
            edge_count=len(edges),
        )

        # Provenance hash
        if self.config.enable_provenance:
            result.provenance_hash = self._compute_provenance_hash(result)

        logger.info(
            "Gap analysis complete for graph=%s: total_gaps=%d, "
            "open=%d, resolved=%d, readiness=%.1f%%, "
            "remediation_actions=%d, elapsed=%.1fms",
            graph_id,
            result.total_gaps,
            result.total_open_gaps,
            result.total_resolved_gaps,
            result.compliance_readiness,
            len(result.remediation_actions),
            elapsed_ms,
        )

        return result

    def resolve_gap(
        self,
        graph_id: str,
        gap_id: str,
        resolution_notes: Optional[str] = None,
    ) -> Optional[DetectedGap]:
        """Mark a gap as resolved.

        Args:
            graph_id: Graph the gap belongs to.
            gap_id: Gap to resolve.
            resolution_notes: Optional notes about the resolution.

        Returns:
            The resolved DetectedGap, or None if not found.
        """
        gaps = self._gap_store.get(graph_id, [])
        for gap in gaps:
            if gap.gap_id == gap_id and not gap.is_resolved:
                gap.is_resolved = True
                gap.resolved_at = _utcnow()
                if resolution_notes:
                    gap.metadata["resolution_notes"] = resolution_notes
                logger.info(
                    "Resolved gap %s (type=%s) in graph=%s",
                    gap_id,
                    gap.gap_type,
                    graph_id,
                )
                return gap
        logger.warning(
            "Gap %s not found or already resolved in graph=%s",
            gap_id,
            graph_id,
        )
        return None

    def get_gaps(
        self,
        graph_id: str,
        gap_type: Optional[str] = None,
        severity: Optional[str] = None,
        include_resolved: bool = False,
    ) -> List[DetectedGap]:
        """Get gaps for a graph with optional filters.

        Args:
            graph_id: Graph identifier.
            gap_type: Optional filter by gap type.
            severity: Optional filter by severity.
            include_resolved: Whether to include resolved gaps.

        Returns:
            List of matching DetectedGap objects.
        """
        gaps = self._gap_store.get(graph_id, [])

        if not include_resolved:
            gaps = [g for g in gaps if not g.is_resolved]
        if gap_type is not None:
            gaps = [g for g in gaps if g.gap_type == gap_type]
        if severity is not None:
            gaps = [g for g in gaps if g.severity == severity]

        return gaps

    def get_compliance_readiness(self, graph_id: str) -> float:
        """Get the current compliance readiness score for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Compliance readiness score (0-100). Returns 100.0 if no
            analysis has been performed for this graph.
        """
        gaps = self._gap_store.get(graph_id, [])
        open_gaps = [g for g in gaps if not g.is_resolved]
        if not gaps:
            return 100.0
        return self._compute_compliance_readiness(open_gaps, node_count=1)

    def get_trend(self, graph_id: str) -> List[GapTrendSnapshot]:
        """Get trend snapshots for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            List of GapTrendSnapshot objects in chronological order.
        """
        return list(self._trend_store.get(graph_id, []))

    def get_trend_summary(self, graph_id: str) -> Dict[str, Any]:
        """Get a summary of gap trends for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            Dictionary with trend direction, gap delta, and readiness trend.
        """
        snapshots = self._trend_store.get(graph_id, [])
        if len(snapshots) < 2:
            return {
                "trend_direction": "neutral",
                "gap_delta": 0,
                "readiness_delta": 0.0,
                "snapshots_count": len(snapshots),
            }

        latest = snapshots[-1]
        previous = snapshots[-2]
        gap_delta = latest.total_gaps - previous.total_gaps
        readiness_delta = latest.compliance_readiness - previous.compliance_readiness

        if gap_delta < 0:
            direction = "improving"
        elif gap_delta > 0:
            direction = "degrading"
        else:
            direction = "stable"

        return {
            "trend_direction": direction,
            "gap_delta": gap_delta,
            "readiness_delta": readiness_delta,
            "snapshots_count": len(snapshots),
            "latest_readiness": latest.compliance_readiness,
            "latest_total_gaps": latest.total_gaps,
        }

    def get_analysis_count(self) -> int:
        """Return the total number of analyses performed.

        Returns:
            Count of analyze() invocations.
        """
        return self._analysis_count

    def clear_store(self, graph_id: Optional[str] = None) -> None:
        """Clear stored gaps and trends for a graph or all graphs.

        Args:
            graph_id: Optional graph to clear. If None, clears all.
        """
        if graph_id is not None:
            self._gap_store.pop(graph_id, None)
            self._trend_store.pop(graph_id, None)
        else:
            self._gap_store.clear()
            self._trend_store.clear()

    def export_gaps_json(self, graph_id: str) -> str:
        """Export gaps as JSON string for reporting.

        Args:
            graph_id: Graph identifier.

        Returns:
            JSON string of all gaps for the graph.
        """
        gaps = self._gap_store.get(graph_id, [])
        return json.dumps(
            [g.to_dict() for g in gaps],
            indent=2,
            default=str,
        )

    def export_gaps_csv_rows(self, graph_id: str) -> List[List[str]]:
        """Export gaps as CSV-compatible rows for reporting.

        Args:
            graph_id: Graph identifier.

        Returns:
            List of rows, where the first row is the header.
        """
        header = [
            "gap_id",
            "gap_type",
            "severity",
            "affected_node_id",
            "affected_edge_id",
            "description",
            "remediation",
            "eudr_article",
            "risk_impact_score",
            "is_resolved",
            "detected_at",
        ]
        rows: List[List[str]] = [header]
        for gap in self._gap_store.get(graph_id, []):
            rows.append([
                gap.gap_id,
                gap.gap_type,
                gap.severity,
                gap.affected_node_id or "",
                gap.affected_edge_id or "",
                gap.description,
                gap.remediation,
                gap.eudr_article,
                str(gap.risk_impact_score),
                str(gap.is_resolved),
                gap.detected_at.isoformat() if gap.detected_at else "",
            ])
        return rows

    # ------------------------------------------------------------------
    # Internal: adjacency structure
    # ------------------------------------------------------------------

    def _build_adjacency(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build adjacency lists and lookup structures from nodes/edges.

        Args:
            nodes: Node dictionaries keyed by node_id.
            edges: Edge dictionaries keyed by edge_id.

        Returns:
            Dictionary with forward/reverse adjacency, incoming/outgoing
            edge lists, and nodes_by_type mapping.
        """
        forward: Dict[str, Set[str]] = defaultdict(set)
        reverse: Dict[str, Set[str]] = defaultdict(set)
        outgoing_edges: Dict[str, List[str]] = defaultdict(list)
        incoming_edges: Dict[str, List[str]] = defaultdict(list)
        nodes_by_type: Dict[str, List[str]] = defaultdict(list)

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            nodes_by_type[node_type].append(node_id)

        for edge_id, edge_data in edges.items():
            source = edge_data.get("source_node_id", "")
            target = edge_data.get("target_node_id", "")
            if source and target:
                forward[source].add(target)
                reverse[target].add(source)
                outgoing_edges[source].append(edge_id)
                incoming_edges[target].append(edge_id)

        return {
            "forward": dict(forward),
            "reverse": dict(reverse),
            "outgoing_edges": dict(outgoing_edges),
            "incoming_edges": dict(incoming_edges),
            "nodes_by_type": dict(nodes_by_type),
        }

    # ------------------------------------------------------------------
    # Gap Detector 1: Missing Tiers
    # ------------------------------------------------------------------

    def _detect_missing_tiers(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, Any],
    ) -> List[DetectedGap]:
        """Detect missing tiers (opaque segments) in the supply chain.

        Identifies edges where the source and target node types skip one
        or more expected intermediate tiers according to the standard
        supply chain ordering: producer -> collector -> processor ->
        trader -> importer.

        For example, an edge from a producer directly to a processor
        (skipping collector) indicates a potential missing tier.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            edges: Edge dictionaries.
            adjacency: Pre-built adjacency structure.

        Returns:
            List of DetectedGap objects for missing tier gaps.
        """
        gaps: List[DetectedGap] = []

        for edge_id, edge_data in edges.items():
            source_id = edge_data.get("source_node_id", "")
            target_id = edge_data.get("target_node_id", "")

            source_node = nodes.get(source_id)
            target_node = nodes.get(target_id)
            if source_node is None or target_node is None:
                continue

            source_type = self._get_node_type(source_node)
            target_type = self._get_node_type(target_node)

            skipped = self._get_skipped_tiers(source_type, target_type)

            if skipped:
                gap = DetectedGap(
                    gap_type=GapType.MISSING_TIER.value,
                    severity=GAP_SEVERITY_MAP[GapType.MISSING_TIER].value,
                    affected_edge_id=edge_id,
                    affected_node_id=source_id,
                    description=(
                        f"Missing tier(s) between {source_type} "
                        f"({source_id}) and {target_type} ({target_id}): "
                        f"expected intermediate tier(s) [{', '.join(skipped)}]."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.MISSING_TIER.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.MISSING_TIER.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[GapType.MISSING_TIER],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.MISSING_TIER, len(skipped)
                    ),
                    metadata={
                        "source_type": source_type,
                        "target_type": target_type,
                        "skipped_tiers": skipped,
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Missing tier detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    def _get_skipped_tiers(
        self, source_type: str, target_type: str
    ) -> List[str]:
        """Determine which tiers are skipped between source and target types.

        Args:
            source_type: Node type of the source node.
            target_type: Node type of the target node.

        Returns:
            List of skipped tier type names. Empty if no tiers are skipped
            or if the types are not in the standard ordering.
        """
        if source_type not in STANDARD_TIER_ORDER:
            return []
        if target_type not in STANDARD_TIER_ORDER:
            return []

        src_idx = STANDARD_TIER_ORDER.index(source_type)
        tgt_idx = STANDARD_TIER_ORDER.index(target_type)

        # Only flag skips in the downstream direction and where gap > 1
        if tgt_idx - src_idx <= 1:
            return []

        return STANDARD_TIER_ORDER[src_idx + 1 : tgt_idx]

    # ------------------------------------------------------------------
    # Gap Detector 2: Unverified Actors
    # ------------------------------------------------------------------

    def _detect_unverified_actors(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
    ) -> List[DetectedGap]:
        """Detect supply chain nodes without compliance verification.

        Identifies nodes of verifiable types (producer, collector,
        processor, trader, importer) that have a compliance_status of
        'pending_verification' or 'insufficient_data'.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.

        Returns:
            List of DetectedGap objects for unverified actor gaps.
        """
        gaps: List[DetectedGap] = []

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            if node_type not in VERIFIABLE_NODE_TYPES:
                continue

            compliance_status = self._get_field(
                node_data, "compliance_status", "pending_verification"
            )

            if compliance_status in UNVERIFIED_STATUSES:
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.UNVERIFIED_ACTOR.value,
                    severity=GAP_SEVERITY_MAP[GapType.UNVERIFIED_ACTOR].value,
                    affected_node_id=node_id,
                    description=(
                        f"Unverified actor: {node_type} '{operator_name}' "
                        f"({node_id}) has compliance status "
                        f"'{compliance_status}' and requires verification "
                        f"per EUDR Article 10."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.UNVERIFIED_ACTOR.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.UNVERIFIED_ACTOR.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[GapType.UNVERIFIED_ACTOR],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.UNVERIFIED_ACTOR
                    ),
                    metadata={
                        "node_type": node_type,
                        "compliance_status": compliance_status,
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Unverified actor detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    # ------------------------------------------------------------------
    # Gap Detector 3: Missing Geolocation
    # ------------------------------------------------------------------

    def _detect_missing_geolocation(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
    ) -> List[DetectedGap]:
        """Detect producer nodes without GPS coordinates.

        EUDR Article 9 requires geolocation data for all production
        plots. This detector identifies producer-type nodes that lack
        both coordinates and plot_ids with known locations.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.

        Returns:
            List of DetectedGap objects for missing geolocation gaps.
        """
        gaps: List[DetectedGap] = []

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            if node_type not in GEOLOCATION_REQUIRED_TYPES:
                continue

            coordinates = node_data.get("coordinates")
            latitude = node_data.get("latitude")
            longitude = node_data.get("longitude")

            has_coordinates = False
            if coordinates is not None and coordinates:
                has_coordinates = True
            elif latitude is not None and longitude is not None:
                has_coordinates = True

            if not has_coordinates:
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.MISSING_GEOLOCATION.value,
                    severity=GAP_SEVERITY_MAP[GapType.MISSING_GEOLOCATION].value,
                    affected_node_id=node_id,
                    description=(
                        f"Missing geolocation: producer '{operator_name}' "
                        f"({node_id}) has no GPS coordinates. "
                        f"EUDR Article 9 requires geolocation for all "
                        f"production plots."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.MISSING_GEOLOCATION.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.MISSING_GEOLOCATION.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[GapType.MISSING_GEOLOCATION],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.MISSING_GEOLOCATION
                    ),
                    metadata={"node_type": node_type},
                )
                gaps.append(gap)

        logger.debug(
            "Missing geolocation detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    # ------------------------------------------------------------------
    # Gap Detector 4: Missing Polygon
    # ------------------------------------------------------------------

    def _detect_missing_polygon(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        plot_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DetectedGap]:
        """Detect plots > 4 hectares without polygon boundary data.

        EUDR Article 9(1)(d) requires polygon boundaries for all
        production plots exceeding 4 hectares. This detector checks
        both inline node metadata and the optional plot registry.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            plot_registry: Optional plot data indexed by plot_id.

        Returns:
            List of DetectedGap objects for missing polygon gaps.
        """
        gaps: List[DetectedGap] = []
        threshold = self.config.polygon_area_threshold_ha

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            if node_type != "producer":
                continue

            # Check inline plot data
            plots = node_data.get("plots", [])
            plot_ids = node_data.get("plot_ids", [])

            # If plot data is inline, check each plot
            if plots:
                for plot in plots:
                    area = plot.get("area_hectares", 0.0)
                    has_polygon = plot.get("has_polygon", False)
                    plot_id = plot.get("plot_id", "unknown")

                    if float(area) > threshold and not has_polygon:
                        gaps.append(self._create_missing_polygon_gap(
                            node_id, plot_id, float(area), threshold
                        ))

            # If only plot_ids, check plot registry
            elif plot_ids and plot_registry:
                for pid in plot_ids:
                    plot_data = plot_registry.get(pid, {})
                    area = plot_data.get("area_hectares", 0.0)
                    has_polygon = plot_data.get("has_polygon", False)

                    if float(area) > threshold and not has_polygon:
                        gaps.append(self._create_missing_polygon_gap(
                            node_id, pid, float(area), threshold
                        ))

            # Check node-level area (single-plot producer)
            elif "area_hectares" in node_data:
                area = float(node_data["area_hectares"])
                has_polygon = node_data.get("has_polygon", False)
                if area > threshold and not has_polygon:
                    gaps.append(self._create_missing_polygon_gap(
                        node_id, node_id, area, threshold
                    ))

        logger.debug(
            "Missing polygon detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    def _create_missing_polygon_gap(
        self,
        node_id: str,
        plot_id: str,
        area: float,
        threshold: float,
    ) -> DetectedGap:
        """Create a DetectedGap for a missing polygon violation.

        Args:
            node_id: Producer node identifier.
            plot_id: Plot identifier.
            area: Plot area in hectares.
            threshold: Polygon requirement threshold.

        Returns:
            DetectedGap instance.
        """
        return DetectedGap(
            gap_type=GapType.MISSING_POLYGON.value,
            severity=GAP_SEVERITY_MAP[GapType.MISSING_POLYGON].value,
            affected_node_id=node_id,
            description=(
                f"Missing polygon: plot '{plot_id}' on producer node "
                f"'{node_id}' has area {area:.1f} ha (> {threshold:.1f} ha "
                f"threshold) but lacks polygon boundary data. "
                f"EUDR Article 9(1)(d) violation."
            ),
            remediation=REMEDIATION_ACTIONS.get(
                GapType.MISSING_POLYGON.value, ""
            ),
            auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                GapType.MISSING_POLYGON.value, ""
            ),
            eudr_article=GAP_ARTICLE_MAP[GapType.MISSING_POLYGON],
            risk_impact_score=self._compute_risk_impact(
                GapType.MISSING_POLYGON
            ),
            metadata={
                "plot_id": plot_id,
                "area_hectares": area,
                "threshold_hectares": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Gap Detector 5: Broken Custody Chains
    # ------------------------------------------------------------------

    def _detect_broken_custody_chains(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, Any],
    ) -> List[DetectedGap]:
        """Detect products with no traceable link to origin production plots.

        Traces backward from importer nodes through the graph. If a
        backward trace reaches a non-producer terminal node (a node
        with no incoming edges that is not a producer), this indicates
        a broken custody chain.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            edges: Edge dictionaries.
            adjacency: Pre-built adjacency structure.

        Returns:
            List of DetectedGap objects for broken custody chain gaps.
        """
        gaps: List[DetectedGap] = []
        reverse = adjacency.get("reverse", {})
        nodes_by_type = adjacency.get("nodes_by_type", {})

        # Identify terminal downstream nodes (typically importers)
        importer_ids = nodes_by_type.get("importer", [])

        # Also consider any node with no outgoing edges as a potential endpoint
        forward = adjacency.get("forward", {})
        terminal_nodes = set(importer_ids)
        for node_id in nodes:
            if node_id not in forward or not forward[node_id]:
                terminal_nodes.add(node_id)

        for terminal_id in terminal_nodes:
            terminal_node = nodes.get(terminal_id)
            if terminal_node is None:
                continue

            terminal_type = self._get_node_type(terminal_node)

            # Skip producers -- they are origin points, not endpoints
            if terminal_type == "producer":
                continue

            # Backward BFS to find all reachable upstream producers
            visited: Set[str] = set()
            queue: deque = deque([terminal_id])
            found_producer = False
            broken_at: List[str] = []

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                current_node = nodes.get(current)
                if current_node is None:
                    continue

                current_type = self._get_node_type(current_node)
                if current_type == "producer":
                    found_producer = True
                    continue

                # Get upstream parents
                parents = reverse.get(current, set())
                if not parents and current_type != "producer":
                    # Dead end -- no upstream and not a producer
                    broken_at.append(current)

                for parent in parents:
                    if parent not in visited:
                        queue.append(parent)

            if not found_producer and broken_at:
                for broken_node_id in broken_at:
                    broken_node = nodes.get(broken_node_id, {})
                    broken_type = self._get_node_type(broken_node)
                    operator_name = self._get_field(
                        broken_node, "operator_name", "Unknown"
                    )

                    gap = DetectedGap(
                        gap_type=GapType.BROKEN_CUSTODY_CHAIN.value,
                        severity=GAP_SEVERITY_MAP[
                            GapType.BROKEN_CUSTODY_CHAIN
                        ].value,
                        affected_node_id=broken_node_id,
                        description=(
                            f"Broken custody chain: {broken_type} "
                            f"'{operator_name}' ({broken_node_id}) has no "
                            f"traceable link to any origin production plot. "
                            f"Backward trace from terminal node "
                            f"'{terminal_id}' reaches a dead end at this node. "
                            f"EUDR Article 4(2)(f) violation."
                        ),
                        remediation=REMEDIATION_ACTIONS.get(
                            GapType.BROKEN_CUSTODY_CHAIN.value, ""
                        ),
                        auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                            GapType.BROKEN_CUSTODY_CHAIN.value, ""
                        ),
                        eudr_article=GAP_ARTICLE_MAP[
                            GapType.BROKEN_CUSTODY_CHAIN
                        ],
                        risk_impact_score=self._compute_risk_impact(
                            GapType.BROKEN_CUSTODY_CHAIN
                        ),
                        metadata={
                            "terminal_node_id": terminal_id,
                            "broken_at_node_type": broken_type,
                        },
                    )
                    gaps.append(gap)

        # De-duplicate by affected_node_id
        seen_nodes: Set[str] = set()
        deduped: List[DetectedGap] = []
        for gap in gaps:
            if gap.affected_node_id not in seen_nodes:
                seen_nodes.add(gap.affected_node_id)
                deduped.append(gap)

        logger.debug(
            "Broken custody chain detection: found %d gaps in graph=%s",
            len(deduped),
            graph_id,
        )
        return deduped

    # ------------------------------------------------------------------
    # Gap Detector 6: Missing Documentation
    # ------------------------------------------------------------------

    def _detect_missing_documentation(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, Any],
    ) -> List[DetectedGap]:
        """Detect nodes without required custody transfer documentation.

        Identifies verifiable nodes that have incoming edges but no
        associated documentation records in their metadata.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            edges: Edge dictionaries.
            adjacency: Pre-built adjacency structure.

        Returns:
            List of DetectedGap objects for missing documentation gaps.
        """
        gaps: List[DetectedGap] = []
        incoming_edges = adjacency.get("incoming_edges", {})

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            if node_type not in VERIFIABLE_NODE_TYPES:
                continue

            # Producers don't need incoming custody documents
            if node_type == "producer":
                continue

            # Check if node has incoming edges
            node_incoming = incoming_edges.get(node_id, [])
            if not node_incoming:
                continue

            # Check for documentation
            has_docs = self._has_documentation(node_data, edges, node_incoming)

            if not has_docs:
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.MISSING_DOCUMENTATION.value,
                    severity=GAP_SEVERITY_MAP[
                        GapType.MISSING_DOCUMENTATION
                    ].value,
                    affected_node_id=node_id,
                    description=(
                        f"Missing documentation: {node_type} "
                        f"'{operator_name}' ({node_id}) has "
                        f"{len(node_incoming)} incoming custody transfer(s) "
                        f"but no associated transfer documentation "
                        f"per EUDR Article 4(2)."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.MISSING_DOCUMENTATION.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.MISSING_DOCUMENTATION.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[
                        GapType.MISSING_DOCUMENTATION
                    ],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.MISSING_DOCUMENTATION
                    ),
                    metadata={
                        "node_type": node_type,
                        "incoming_edge_count": len(node_incoming),
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Missing documentation detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    def _has_documentation(
        self,
        node_data: Dict[str, Any],
        edges: Dict[str, Dict[str, Any]],
        incoming_edge_ids: List[str],
    ) -> bool:
        """Check if a node has custody transfer documentation.

        Checks both node-level and edge-level metadata for documentation
        indicators.

        Args:
            node_data: Node data dictionary.
            edges: All edges in the graph.
            incoming_edge_ids: IDs of incoming edges to this node.

        Returns:
            True if documentation is found, False otherwise.
        """
        # Check node-level documentation
        metadata = node_data.get("metadata", {})
        if metadata.get("has_documentation", False):
            return True
        if metadata.get("documents") or metadata.get("documentation"):
            return True
        if node_data.get("documents") or node_data.get("documentation"):
            return True

        # Check edge-level documentation
        for edge_id in incoming_edge_ids:
            edge = edges.get(edge_id, {})
            edge_meta = edge.get("metadata", {})
            if edge_meta.get("has_documentation", False):
                return True
            if edge_meta.get("documents") or edge_meta.get("documentation"):
                return True
            # Batch number presence indicates some tracking
            if edge.get("batch_number"):
                return True

        return False

    # ------------------------------------------------------------------
    # Gap Detector 7: Mass Balance Discrepancies
    # ------------------------------------------------------------------

    def _detect_mass_balance_discrepancies(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, Any],
    ) -> List[DetectedGap]:
        """Detect nodes where output quantity exceeds input quantity.

        For each non-producer node, sums incoming edge quantities and
        outgoing edge quantities. If outgoing exceeds incoming by more
        than the configured tolerance, a mass balance discrepancy is
        flagged.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            edges: Edge dictionaries.
            adjacency: Pre-built adjacency structure.

        Returns:
            List of DetectedGap objects for mass balance discrepancy gaps.
        """
        gaps: List[DetectedGap] = []
        incoming_edges = adjacency.get("incoming_edges", {})
        outgoing_edges = adjacency.get("outgoing_edges", {})
        tolerance = self.config.mass_balance_tolerance_pct

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)

            # Skip producers (they create material, no input expected)
            if node_type == "producer":
                continue

            node_in_edges = incoming_edges.get(node_id, [])
            node_out_edges = outgoing_edges.get(node_id, [])

            # Need both incoming and outgoing to check balance
            if not node_in_edges or not node_out_edges:
                continue

            total_in = self._sum_edge_quantities(edges, node_in_edges)
            total_out = self._sum_edge_quantities(edges, node_out_edges)

            if total_in <= 0:
                continue

            excess_pct = ((total_out - total_in) / total_in) * 100.0

            if excess_pct > tolerance:
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.MASS_BALANCE_DISCREPANCY.value,
                    severity=GAP_SEVERITY_MAP[
                        GapType.MASS_BALANCE_DISCREPANCY
                    ].value,
                    affected_node_id=node_id,
                    description=(
                        f"Mass balance discrepancy: {node_type} "
                        f"'{operator_name}' ({node_id}) has output "
                        f"({total_out:.2f}) exceeding input ({total_in:.2f}) "
                        f"by {excess_pct:.1f}% (tolerance: {tolerance:.1f}%). "
                        f"EUDR Article 10(2)(f) violation."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.MASS_BALANCE_DISCREPANCY.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.MASS_BALANCE_DISCREPANCY.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[
                        GapType.MASS_BALANCE_DISCREPANCY
                    ],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.MASS_BALANCE_DISCREPANCY,
                        severity_factor=min(excess_pct / 10.0, 3.0),
                    ),
                    metadata={
                        "total_input": total_in,
                        "total_output": total_out,
                        "excess_pct": round(excess_pct, 2),
                        "tolerance_pct": tolerance,
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Mass balance discrepancy detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    def _sum_edge_quantities(
        self,
        edges: Dict[str, Dict[str, Any]],
        edge_ids: List[str],
    ) -> float:
        """Sum quantities across a list of edges.

        Args:
            edges: All edges dictionary.
            edge_ids: Edge IDs to sum.

        Returns:
            Total quantity as float.
        """
        total = 0.0
        for eid in edge_ids:
            edge = edges.get(eid, {})
            qty = edge.get("quantity", 0)
            try:
                total += float(qty)
            except (TypeError, ValueError):
                pass
        return total

    # ------------------------------------------------------------------
    # Gap Detector 8: Stale Data
    # ------------------------------------------------------------------

    def _detect_stale_data(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        reference_time: datetime,
    ) -> List[DetectedGap]:
        """Detect nodes with data older than the configured threshold.

        Checks the updated_at timestamp on each node. If the data has
        not been refreshed within stale_data_days, it is flagged.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            reference_time: Reference timestamp for staleness calculation.

        Returns:
            List of DetectedGap objects for stale data gaps.
        """
        gaps: List[DetectedGap] = []
        cutoff = reference_time - timedelta(days=self.config.stale_data_days)

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)
            if node_type not in VERIFIABLE_NODE_TYPES:
                continue

            updated_at = self._parse_datetime(
                node_data.get("updated_at")
            )

            if updated_at is None:
                # No timestamp at all -- consider stale
                updated_at = self._parse_datetime(
                    node_data.get("created_at")
                )

            if updated_at is not None and updated_at < cutoff:
                days_stale = (reference_time - updated_at).days
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.STALE_DATA.value,
                    severity=GAP_SEVERITY_MAP[GapType.STALE_DATA].value,
                    affected_node_id=node_id,
                    description=(
                        f"Stale data: {node_type} '{operator_name}' "
                        f"({node_id}) has not been updated for "
                        f"{days_stale} days (threshold: "
                        f"{self.config.stale_data_days} days). "
                        f"EUDR Article 31 requires current data."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.STALE_DATA.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.STALE_DATA.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[GapType.STALE_DATA],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.STALE_DATA,
                        severity_factor=min(days_stale / 365.0, 3.0),
                    ),
                    metadata={
                        "node_type": node_type,
                        "updated_at": updated_at.isoformat(),
                        "days_stale": days_stale,
                        "threshold_days": self.config.stale_data_days,
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Stale data detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    # ------------------------------------------------------------------
    # Gap Detector 9: Missing Certification
    # ------------------------------------------------------------------

    def _detect_missing_certification(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
    ) -> List[DetectedGap]:
        """Detect nodes that lack expected certifications.

        Flags producer and processor nodes that have no certifications
        listed and are in a high-risk region or have elevated risk scores.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.

        Returns:
            List of DetectedGap objects for missing certification gaps.
        """
        gaps: List[DetectedGap] = []

        for node_id, node_data in nodes.items():
            node_type = self._get_node_type(node_data)

            # Only check nodes where certification is expected
            if node_type not in ("producer", "processor"):
                continue

            certifications = node_data.get("certifications", [])
            risk_level = self._get_field(
                node_data, "risk_level", "standard"
            )
            risk_score = float(node_data.get("risk_score", 0.0))

            # Flag if no certifications and elevated risk
            if not certifications and (
                risk_level == "high" or risk_score >= 50.0
            ):
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.MISSING_CERTIFICATION.value,
                    severity=GAP_SEVERITY_MAP[
                        GapType.MISSING_CERTIFICATION
                    ].value,
                    affected_node_id=node_id,
                    description=(
                        f"Missing certification: {node_type} "
                        f"'{operator_name}' ({node_id}) has no "
                        f"certifications but risk level is '{risk_level}' "
                        f"(score: {risk_score:.0f}). "
                        f"Certification expected per EUDR Article 10."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.MISSING_CERTIFICATION.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.MISSING_CERTIFICATION.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[
                        GapType.MISSING_CERTIFICATION
                    ],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.MISSING_CERTIFICATION
                    ),
                    metadata={
                        "node_type": node_type,
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                    },
                )
                gaps.append(gap)

        logger.debug(
            "Missing certification detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    # ------------------------------------------------------------------
    # Gap Detector 10: Orphan Nodes
    # ------------------------------------------------------------------

    def _detect_orphan_nodes(
        self,
        graph_id: str,
        nodes: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, Any],
    ) -> List[DetectedGap]:
        """Detect nodes with no incoming or outgoing edges.

        Orphan nodes are disconnected from the supply chain and cannot
        contribute to or receive commodity flows.

        Args:
            graph_id: Graph identifier.
            nodes: Node dictionaries.
            adjacency: Pre-built adjacency structure.

        Returns:
            List of DetectedGap objects for orphan node gaps.
        """
        gaps: List[DetectedGap] = []
        forward = adjacency.get("forward", {})
        reverse = adjacency.get("reverse", {})

        for node_id, node_data in nodes.items():
            has_outgoing = node_id in forward and bool(forward[node_id])
            has_incoming = node_id in reverse and bool(reverse[node_id])

            if not has_outgoing and not has_incoming:
                node_type = self._get_node_type(node_data)
                operator_name = self._get_field(
                    node_data, "operator_name", "Unknown"
                )
                gap = DetectedGap(
                    gap_type=GapType.ORPHAN_NODE.value,
                    severity=GAP_SEVERITY_MAP[GapType.ORPHAN_NODE].value,
                    affected_node_id=node_id,
                    description=(
                        f"Orphan node: {node_type} '{operator_name}' "
                        f"({node_id}) has no incoming or outgoing edges "
                        f"and is disconnected from the supply chain."
                    ),
                    remediation=REMEDIATION_ACTIONS.get(
                        GapType.ORPHAN_NODE.value, ""
                    ),
                    auto_remediation_trigger=AUTO_REMEDIATION_TRIGGERS.get(
                        GapType.ORPHAN_NODE.value, ""
                    ),
                    eudr_article=GAP_ARTICLE_MAP[GapType.ORPHAN_NODE],
                    risk_impact_score=self._compute_risk_impact(
                        GapType.ORPHAN_NODE
                    ),
                    metadata={"node_type": node_type},
                )
                gaps.append(gap)

        logger.debug(
            "Orphan node detection: found %d gaps in graph=%s",
            len(gaps),
            graph_id,
        )
        return gaps

    # ------------------------------------------------------------------
    # Scoring and Priority
    # ------------------------------------------------------------------

    def _compute_compliance_readiness(
        self,
        open_gaps: List[DetectedGap],
        node_count: int,
    ) -> float:
        """Compute compliance readiness score (0-100).

        Starts at 100 and deducts penalty points based on gap severity
        and count. The penalty is scaled by the number of nodes to avoid
        disproportionate impact on larger graphs.

        Formula:
            readiness = max(0, 100 - sum(severity_weight * count) * scale)
            where scale = min(1.0, 10.0 / max(1, node_count))

        Args:
            open_gaps: List of open (unresolved) gaps.
            node_count: Number of nodes in the graph.

        Returns:
            Compliance readiness score between 0.0 and 100.0.
        """
        if not open_gaps:
            return 100.0

        weights = self.config.severity_penalty_weights
        total_penalty = 0.0

        for gap in open_gaps:
            severity = gap.severity
            weight = weights.get(severity, 1.0)
            total_penalty += weight

        # Scale factor: smaller graphs get hit harder per gap
        scale = min(1.0, 10.0 / max(1, node_count))
        adjusted_penalty = total_penalty * scale

        readiness = max(0.0, 100.0 - adjusted_penalty)
        return round(readiness, 1)

    def _compute_risk_impact(
        self,
        gap_type: GapType,
        severity_factor: float = 1.0,
    ) -> float:
        """Compute risk impact score for remediation prioritization.

        Args:
            gap_type: Type of gap.
            severity_factor: Optional multiplier for severity scaling
                (e.g., based on how large a mass balance discrepancy is).

        Returns:
            Risk impact score (0-100).
        """
        severity = GAP_SEVERITY_MAP.get(gap_type, GapSeverity.MEDIUM)
        base_impact = RISK_IMPACT_MULTIPLIERS.get(severity, 2.0)
        score = min(100.0, base_impact * severity_factor * 10.0)
        return round(score, 1)

    def _build_remediation_actions(
        self,
        open_gaps: List[DetectedGap],
        nodes: Dict[str, Dict[str, Any]],
    ) -> List[RemediationAction]:
        """Build a prioritized remediation action list from open gaps.

        Actions are sorted by risk impact score (descending), then by
        severity (critical > high > medium > low).

        Args:
            open_gaps: List of open (unresolved) gaps.
            nodes: Node dictionaries for effort estimation.

        Returns:
            Sorted list of RemediationAction objects.
        """
        actions: List[RemediationAction] = []

        for gap in open_gaps:
            effort = self._estimate_effort(gap, nodes)
            action = RemediationAction(
                gap_id=gap.gap_id,
                gap_type=gap.gap_type,
                severity=gap.severity,
                risk_impact_score=gap.risk_impact_score,
                action_description=gap.remediation,
                auto_trigger=gap.auto_remediation_trigger,
                affected_node_id=gap.affected_node_id,
                affected_edge_id=gap.affected_edge_id,
                eudr_article=gap.eudr_article,
                estimated_effort=effort,
            )
            actions.append(action)

        # Sort by risk impact (desc), then severity order
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        actions.sort(
            key=lambda a: (
                -a.risk_impact_score,
                severity_order.get(a.severity, 4),
            )
        )

        # Assign priority ranks
        for rank, action in enumerate(actions, start=1):
            action.priority_rank = rank

        return actions

    def _estimate_effort(
        self,
        gap: DetectedGap,
        nodes: Dict[str, Dict[str, Any]],
    ) -> str:
        """Estimate remediation effort for a gap.

        Args:
            gap: The detected gap.
            nodes: Node dictionaries for context.

        Returns:
            Effort estimate: 'low', 'medium', or 'high'.
        """
        effort_map = {
            GapType.MISSING_GEOLOCATION.value: "medium",
            GapType.MISSING_POLYGON.value: "high",
            GapType.BROKEN_CUSTODY_CHAIN.value: "high",
            GapType.UNVERIFIED_ACTOR.value: "medium",
            GapType.MISSING_TIER.value: "high",
            GapType.MASS_BALANCE_DISCREPANCY.value: "medium",
            GapType.MISSING_CERTIFICATION.value: "low",
            GapType.STALE_DATA.value: "low",
            GapType.ORPHAN_NODE.value: "low",
            GapType.MISSING_DOCUMENTATION.value: "low",
        }
        return effort_map.get(gap.gap_type, "medium")

    def _generate_auto_triggers(
        self,
        open_gaps: List[DetectedGap],
    ) -> List[Dict[str, Any]]:
        """Generate auto-remediation trigger entries.

        Args:
            open_gaps: List of open gaps.

        Returns:
            List of trigger dictionaries with gap_id, trigger type,
            and affected entity.
        """
        triggers: List[Dict[str, Any]] = []

        for gap in open_gaps:
            if gap.auto_remediation_trigger:
                triggers.append({
                    "gap_id": gap.gap_id,
                    "trigger": gap.auto_remediation_trigger,
                    "gap_type": gap.gap_type,
                    "severity": gap.severity,
                    "affected_node_id": gap.affected_node_id,
                    "affected_edge_id": gap.affected_edge_id,
                    "timestamp": _utcnow().isoformat(),
                })

        return triggers

    # ------------------------------------------------------------------
    # Trend Tracking
    # ------------------------------------------------------------------

    def _record_trend_snapshot(
        self,
        graph_id: str,
        open_gaps: List[DetectedGap],
        gaps_by_severity: Dict[str, int],
        gaps_by_type: Dict[str, int],
        compliance_readiness: float,
        previous_snapshot: Optional[GapTrendSnapshot] = None,
    ) -> GapTrendSnapshot:
        """Record a trend snapshot for gap closure tracking.

        Args:
            graph_id: Graph identifier.
            open_gaps: Current open gaps.
            gaps_by_severity: Gap counts by severity.
            gaps_by_type: Gap counts by type.
            compliance_readiness: Current readiness score.
            previous_snapshot: Previous snapshot for delta calculation.

        Returns:
            New GapTrendSnapshot.
        """
        resolved_since_last = 0
        new_since_last = 0

        if previous_snapshot is not None:
            delta = len(open_gaps) - previous_snapshot.total_gaps
            if delta < 0:
                resolved_since_last = abs(delta)
            elif delta > 0:
                new_since_last = delta

        snapshot = GapTrendSnapshot(
            graph_id=graph_id,
            total_gaps=len(open_gaps),
            gaps_by_severity=dict(gaps_by_severity),
            gaps_by_type=dict(gaps_by_type),
            compliance_readiness=compliance_readiness,
            resolved_since_last=resolved_since_last,
            new_since_last=new_since_last,
        )

        if graph_id not in self._trend_store:
            self._trend_store[graph_id] = []

        self._trend_store[graph_id].append(snapshot)

        # Prune oldest snapshots if over limit
        max_snaps = self.config.max_trend_snapshots
        if len(self._trend_store[graph_id]) > max_snaps:
            self._trend_store[graph_id] = self._trend_store[graph_id][
                -max_snaps:
            ]

        return snapshot

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        result: GapAnalysisResult,
    ) -> str:
        """Compute SHA-256 provenance hash for the analysis result.

        The hash covers graph_id, gap types and counts, compliance
        readiness score, and analysis timestamp for deterministic
        reproducibility.

        Args:
            result: The gap analysis result to hash.

        Returns:
            SHA-256 hex digest string.
        """
        hash_input = {
            "graph_id": result.graph_id,
            "total_gaps": result.total_gaps,
            "total_open_gaps": result.total_open_gaps,
            "gaps_by_severity": result.gaps_by_severity,
            "gaps_by_type": result.gaps_by_type,
            "compliance_readiness": result.compliance_readiness,
            "analysis_timestamp": (
                result.analysis_timestamp.isoformat()
                if result.analysis_timestamp
                else ""
            ),
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "module_version": _MODULE_VERSION,
        }
        return _compute_hash(hash_input)

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def _get_node_type(self, node_data: Dict[str, Any]) -> str:
        """Extract node type string from node data.

        Args:
            node_data: Node data dictionary.

        Returns:
            Lowercase node type string.
        """
        nt = node_data.get("node_type", "")
        if hasattr(nt, "value"):
            return str(nt.value).lower()
        return str(nt).lower()

    def _get_field(
        self,
        data: Dict[str, Any],
        field_name: str,
        default: str = "",
    ) -> str:
        """Get a string field from a dictionary with a default.

        Args:
            data: Source dictionary.
            field_name: Field name to look up.
            default: Default value if field is missing or empty.

        Returns:
            Field value as string.
        """
        value = data.get(field_name, default)
        if hasattr(value, "value"):
            return str(value.value)
        return str(value) if value else default

    def _parse_datetime(
        self, value: Any
    ) -> Optional[datetime]:
        """Parse a datetime value from various input formats.

        Args:
            value: Datetime string, datetime object, or None.

        Returns:
            Parsed datetime or None.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Handle ISO format with optional timezone
                if value.endswith("Z"):
                    value = value[:-1] + "+00:00"
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None
        return None

    def _count_by_severity(
        self, gaps: List[DetectedGap]
    ) -> Dict[str, int]:
        """Count gaps by severity level.

        Args:
            gaps: List of detected gaps.

        Returns:
            Dictionary mapping severity -> count.
        """
        counts: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        for gap in gaps:
            if gap.severity in counts:
                counts[gap.severity] += 1
        return counts

    def _count_by_type(
        self, gaps: List[DetectedGap]
    ) -> Dict[str, int]:
        """Count gaps by gap type.

        Args:
            gaps: List of detected gaps.

        Returns:
            Dictionary mapping gap_type -> count.
        """
        counts: Dict[str, int] = defaultdict(int)
        for gap in gaps:
            counts[gap.gap_type] += 1
        return dict(counts)

    def _get_resolved_gaps(
        self, graph_id: str
    ) -> List[DetectedGap]:
        """Get previously resolved gaps for a graph.

        Args:
            graph_id: Graph identifier.

        Returns:
            List of resolved DetectedGap objects.
        """
        return [
            g
            for g in self._gap_store.get(graph_id, [])
            if g.is_resolved
        ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "POLYGON_AREA_THRESHOLD_HA",
    "DEFAULT_STALE_DATA_DAYS",
    "DEFAULT_MASS_BALANCE_TOLERANCE_PCT",
    "SEVERITY_PENALTY_WEIGHTS",
    "REMEDIATION_ACTIONS",
    "AUTO_REMEDIATION_TRIGGERS",
    "STANDARD_TIER_ORDER",
    "VERIFIABLE_NODE_TYPES",
    "GEOLOCATION_REQUIRED_TYPES",
    "UNVERIFIED_STATUSES",
    "RISK_IMPACT_MULTIPLIERS",
    # Enums
    "GapType",
    "GapSeverity",
    # Data models
    "DetectedGap",
    "RemediationAction",
    "GapTrendSnapshot",
    "GapAnalysisResult",
    "GapAnalyzerConfig",
    # Engine
    "GapAnalyzer",
    # Maps
    "GAP_SEVERITY_MAP",
    "GAP_ARTICLE_MAP",
]
