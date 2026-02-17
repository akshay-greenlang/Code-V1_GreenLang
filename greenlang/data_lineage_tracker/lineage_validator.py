# -*- coding: utf-8 -*-
"""
Lineage Validator Engine - AGENT-DATA-018

Validates lineage graph completeness and consistency by detecting orphan
nodes, broken edges, cycles, stale metadata, and insufficient source
coverage.  Produces combined validation reports with pass/warn/fail
outcomes, actionable remediation recommendations, and SHA-256
provenance-tracked audit records for every validation run.

Engine 5 of 7 in the Data Lineage Tracker Agent SDK.

Zero-Hallucination Guarantees:
    - All scoring uses deterministic Python arithmetic (weighted sums,
      min/max clamping, linear penalties).  No LLM calls for numeric
      computations or coverage scoring.
    - Completeness score is a weighted combination of four sub-scores:
      orphan penalty (0.3), broken-edge penalty (0.3), source coverage
      (0.2), and cycle penalty (0.2).
    - Freshness scoring uses elapsed-time comparison against configurable
      ``freshness_max_age_hours`` threshold.
    - Cycle detection delegates to ``LineageGraphEngine.detect_cycles()``
      which implements Tarjan/DFS internally.
    - Coverage and completeness thresholds are read from
      ``DataLineageTrackerConfig`` (``coverage_warn_threshold`` and
      ``coverage_fail_threshold``).

Validation Report Structure:
    {
        "id":                 str,      # VAL-<uuid_hex[:12]>
        "scope":              str,      # "full" | "pipeline" | "targeted"
        "orphan_nodes":       int,      # count of orphan nodes detected
        "broken_edges":       int,      # count of broken edges detected
        "cycles_detected":    int,      # count of cycles found
        "source_coverage":    float,    # 0.0-1.0 report-level coverage
        "completeness_score": float,    # 0.0-1.0 weighted composite
        "freshness_score":    float,    # 0.0-1.0 metadata freshness
        "issues":             list,     # detailed issue records
        "recommendations":    list,     # actionable remediation strings
        "result":             str,      # "pass" | "warn" | "fail"
        "validated_at":       str       # ISO-8601 UTC timestamp
    }

Example:
    >>> from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    >>> from greenlang.data_lineage_tracker.lineage_validator import LineageValidatorEngine
    >>> graph = LineageGraphEngine()
    >>> validator = LineageValidatorEngine(graph)
    >>> report = validator.validate(scope="full")
    >>> assert report["result"] in ("pass", "warn", "fail")
    >>> assert 0.0 <= report["completeness_score"] <= 1.0

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
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

from greenlang.data_lineage_tracker.config import get_config
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
from greenlang.data_lineage_tracker.metrics import (
    record_validation,
    observe_processing_duration,
    PROMETHEUS_AVAILABLE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["LineageValidatorEngine"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHT_ORPHAN = 0.3
"""Completeness score weight for orphan-node penalty."""

_WEIGHT_BROKEN_EDGE = 0.3
"""Completeness score weight for broken-edge penalty."""

_WEIGHT_COVERAGE = 0.2
"""Completeness score weight for source coverage."""

_WEIGHT_CYCLE = 0.2
"""Completeness score weight for cycle penalty."""

_RESULT_PASS = "pass"
_RESULT_WARN = "warn"
_RESULT_FAIL = "fail"

_SCOPE_FULL = "full"
_SCOPE_PIPELINE = "pipeline"
_SCOPE_TARGETED = "targeted"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for stability."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_validation_id() -> str:
    """Generate a unique validation identifier.

    Format: ``VAL-<12-char hex from uuid4>``.

    Returns:
        Validation ID string.
    """
    return f"VAL-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# LineageValidatorEngine
# ---------------------------------------------------------------------------


class LineageValidatorEngine:
    """Validates lineage graph completeness and consistency.

    Runs structural integrity checks against the in-memory lineage graph
    maintained by :class:`LineageGraphEngine`, covering:

    * **Orphan nodes** -- nodes with zero incoming *and* zero outgoing edges.
    * **Broken edges** -- edges whose ``source`` or ``target`` reference a
      node ID that does not exist in the graph.
    * **Cycles** -- strongly-connected components that indicate circular
      data flow (delegated to ``graph.detect_cycles()``).
    * **Source coverage** -- whether every ``report``-type node traces back
      to at least one authoritative source node.
    * **Freshness** -- whether asset and transformation metadata has been
      updated within the configured ``freshness_max_age_hours`` window.

    The engine computes a composite **completeness score** (0.0 to 1.0)
    from the sub-checks using fixed weights and maps the score to a
    ``pass`` / ``warn`` / ``fail`` result via the config-level coverage
    thresholds.

    Every validation run is recorded in the provenance chain and emitted
    as a Prometheus ``gl_dlt_validations_total`` counter increment.

    Attributes:
        _graph: Reference to the lineage graph engine under inspection.
        _validations: In-memory store of validation reports keyed by ID.
        _lock: Threading lock for safe concurrent access.
        _provenance: ProvenanceTracker for SHA-256 audit trail recording.

    Example:
        >>> graph = LineageGraphEngine()
        >>> validator = LineageValidatorEngine(graph)
        >>> report = validator.validate()
        >>> assert report["result"] in ("pass", "warn", "fail")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph: LineageGraphEngine,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the LineageValidatorEngine.

        Args:
            graph: A :class:`LineageGraphEngine` instance whose nodes
                and edges will be validated.
            provenance: Optional :class:`ProvenanceTracker` for audit
                trail recording.  When ``None`` a fresh tracker is
                created internally.

        Raises:
            TypeError: If *graph* is not a :class:`LineageGraphEngine`.
        """
        if not isinstance(graph, LineageGraphEngine):
            raise TypeError(
                "graph must be a LineageGraphEngine instance, "
                f"got {type(graph).__name__}"
            )
        self._graph: LineageGraphEngine = graph
        self._validations: Dict[str, dict] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = provenance if provenance is not None else ProvenanceTracker()
        logger.info(
            "LineageValidatorEngine initialized (graph id=%s)",
            id(graph),
        )

    # ------------------------------------------------------------------
    # Primary validation entry-point
    # ------------------------------------------------------------------

    def validate(
        self,
        scope: str = _SCOPE_FULL,
        include_freshness: bool = True,
        include_coverage: bool = True,
    ) -> dict:
        """Run all validation checks and return a combined report.

        Executes orphan detection, broken-edge detection, cycle detection,
        source-coverage analysis, freshness checking (optional), and
        completeness scoring.  The report is persisted in-memory and
        tracked via the provenance chain.

        Args:
            scope: Validation scope label.  One of ``"full"``,
                ``"pipeline"``, or ``"targeted"``.  Defaults to
                ``"full"``.
            include_freshness: When ``True`` (default) the freshness
                sub-check is included in the report and factored into
                recommendations.
            include_coverage: When ``True`` (default) the source-coverage
                sub-check is included.

        Returns:
            A dictionary containing the complete validation report::

                {
                    "id": "VAL-...",
                    "scope": "full",
                    "orphan_nodes": 2,
                    "broken_edges": 0,
                    "cycles_detected": 0,
                    "source_coverage": 0.85,
                    "completeness_score": 0.92,
                    "freshness_score": 1.0,
                    "issues": [...],
                    "recommendations": [...],
                    "result": "pass",
                    "validated_at": "2026-02-17T12:00:00+00:00"
                }

        Raises:
            RuntimeError: If graph introspection fails unexpectedly.
        """
        start = time.monotonic()
        validation_id = _generate_validation_id()
        cfg = get_config()
        issues: List[dict] = []
        logger.info(
            "Starting lineage validation id=%s scope=%s",
            validation_id,
            scope,
        )

        # -- Step 1: Orphan nodes ----------------------------------------
        orphans = self.detect_orphan_nodes()
        for orphan in orphans:
            issues.append({
                "type": "orphan_node",
                "severity": "warning",
                "node_id": orphan.get("node_id", ""),
                "node_type": orphan.get("node_type", "unknown"),
                "detail": (
                    f"Node '{orphan.get('node_id', '')}' has no incoming "
                    f"or outgoing edges"
                ),
            })

        # -- Step 2: Broken edges ----------------------------------------
        broken = self.detect_broken_edges()
        for edge_info in broken:
            issues.append({
                "type": "broken_edge",
                "severity": "error",
                "edge_id": edge_info.get("edge_id", ""),
                "detail": edge_info.get("issue", "Broken edge detected"),
            })

        # -- Step 3: Cycles ----------------------------------------------
        cycles = self.detect_cycles()
        for cycle in cycles:
            issues.append({
                "type": "cycle",
                "severity": "error",
                "nodes": cycle,
                "detail": (
                    f"Cycle detected involving {len(cycle)} node(s): "
                    f"{' -> '.join(cycle[:5])}"
                    + ("..." if len(cycle) > 5 else "")
                ),
            })

        # -- Step 4: Source coverage -------------------------------------
        coverage_result = {"coverage_score": 1.0}
        if include_coverage:
            coverage_result = self.compute_source_coverage()
            if coverage_result["coverage_score"] < cfg.coverage_warn_threshold:
                issues.append({
                    "type": "low_coverage",
                    "severity": "warning",
                    "coverage_score": coverage_result["coverage_score"],
                    "detail": (
                        f"Source coverage {coverage_result['coverage_score']:.2f} "
                        f"is below warning threshold "
                        f"{cfg.coverage_warn_threshold:.2f}"
                    ),
                })

        # -- Step 5: Freshness -------------------------------------------
        freshness_result = {"freshness_score": 1.0}
        if include_freshness:
            freshness_result = self.check_freshness()
            if freshness_result["freshness_score"] < 1.0:
                issues.append({
                    "type": "stale_metadata",
                    "severity": "warning",
                    "freshness_score": freshness_result["freshness_score"],
                    "stale_assets": freshness_result.get("stale_assets", 0),
                    "detail": (
                        f"Freshness score {freshness_result['freshness_score']:.2f}, "
                        f"{freshness_result.get('stale_assets', 0)} stale asset(s)"
                    ),
                })

        # -- Step 6: Completeness score ----------------------------------
        completeness = self.compute_completeness_score()

        # -- Step 7: Recommendations -------------------------------------
        recommendations = self.generate_recommendations(issues)

        # -- Step 8: Result determination --------------------------------
        result = self._determine_result(completeness, cfg)

        # -- Step 9: Assemble report -------------------------------------
        validated_at = _utcnow().isoformat()
        report: dict = {
            "id": validation_id,
            "scope": scope,
            "orphan_nodes": len(orphans),
            "broken_edges": len(broken),
            "cycles_detected": len(cycles),
            "source_coverage": round(
                coverage_result["coverage_score"], 4
            ),
            "completeness_score": round(completeness, 4),
            "freshness_score": round(
                freshness_result["freshness_score"], 4
            ),
            "issues": issues,
            "recommendations": recommendations,
            "result": result,
            "validated_at": validated_at,
        }

        # -- Step 10: Persist and track ----------------------------------
        with self._lock:
            self._validations[validation_id] = report

        self._provenance.record(
            entity_type="validation",
            entity_id=validation_id,
            action="validation_completed",
            metadata={
                "scope": scope,
                "result": result,
                "completeness_score": completeness,
                "orphan_nodes": len(orphans),
                "broken_edges": len(broken),
                "cycles_detected": len(cycles),
            },
        )

        elapsed = time.monotonic() - start
        observe_processing_duration("validate", elapsed)
        record_validation(result)

        logger.info(
            "Lineage validation id=%s completed in %.3fs result=%s "
            "completeness=%.4f orphans=%d broken=%d cycles=%d",
            validation_id,
            elapsed,
            result,
            completeness,
            len(orphans),
            len(broken),
            len(cycles),
        )
        return report

    # ------------------------------------------------------------------
    # Orphan detection
    # ------------------------------------------------------------------

    def detect_orphan_nodes(self) -> List[dict]:
        """Find nodes with no incoming AND no outgoing edges.

        A node is considered orphaned if it is entirely disconnected
        from the rest of the lineage graph.  Orphans indicate assets
        that were registered but never linked to any transformation or
        data flow.

        Returns:
            List of dictionaries describing each orphan node::

                [
                    {
                        "node_id": "asset-001",
                        "node_type": "dataset",
                        "name": "...",
                        "detected_at": "2026-02-17T..."
                    },
                    ...
                ]
        """
        start = time.monotonic()
        orphans: List[dict] = []

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()

        # Build sets of node IDs that participate in at least one edge
        connected_ids: set = set()
        for edge in edges:
            source_id = (
                edge.get("source_id")
                or edge.get("source")
                or edge.get("from_id", "")
            )
            target_id = (
                edge.get("target_id")
                or edge.get("target")
                or edge.get("to_id", "")
            )
            if source_id:
                connected_ids.add(source_id)
            if target_id:
                connected_ids.add(target_id)

        detected_at = _utcnow().isoformat()
        for node in nodes:
            node_id = node.get("id") or node.get("node_id", "")
            if node_id and node_id not in connected_ids:
                orphans.append({
                    "node_id": node_id,
                    "node_type": node.get("type", node.get("asset_type", "unknown")),
                    "name": node.get("name", ""),
                    "detected_at": detected_at,
                })

        elapsed = time.monotonic() - start
        observe_processing_duration("detect_orphan_nodes", elapsed)
        logger.debug(
            "Orphan detection completed in %.3fs: %d orphan(s) out of %d node(s)",
            elapsed,
            len(orphans),
            len(nodes),
        )
        return orphans

    # ------------------------------------------------------------------
    # Broken edge detection
    # ------------------------------------------------------------------

    def detect_broken_edges(self) -> List[dict]:
        """Find edges referencing non-existent source or target nodes.

        A broken edge is an edge whose ``source_id`` or ``target_id``
        does not correspond to any node currently registered in the
        lineage graph.  This typically occurs when assets are removed
        without cleaning up their lineage relationships.

        Returns:
            List of dictionaries describing each broken edge::

                [
                    {
                        "edge_id": "edge-123",
                        "source_id": "asset-001",
                        "target_id": "asset-999",
                        "issue": "Target node 'asset-999' does not exist"
                    },
                    ...
                ]
        """
        start = time.monotonic()
        broken: List[dict] = []

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()

        # Build lookup of valid node IDs
        valid_ids: set = set()
        for node in nodes:
            node_id = node.get("id") or node.get("node_id", "")
            if node_id:
                valid_ids.add(node_id)

        for edge in edges:
            edge_id = edge.get("id") or edge.get("edge_id", "")
            source_id = (
                edge.get("source_id")
                or edge.get("source")
                or edge.get("from_id", "")
            )
            target_id = (
                edge.get("target_id")
                or edge.get("target")
                or edge.get("to_id", "")
            )

            issues_found: List[str] = []

            if source_id and source_id not in valid_ids:
                issues_found.append(
                    f"Source node '{source_id}' does not exist"
                )
            if target_id and target_id not in valid_ids:
                issues_found.append(
                    f"Target node '{target_id}' does not exist"
                )
            if not source_id:
                issues_found.append("Edge has no source node reference")
            if not target_id:
                issues_found.append("Edge has no target node reference")

            if issues_found:
                broken.append({
                    "edge_id": edge_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "issue": "; ".join(issues_found),
                })

        elapsed = time.monotonic() - start
        observe_processing_duration("detect_broken_edges", elapsed)
        logger.debug(
            "Broken edge detection completed in %.3fs: %d broken out of %d edge(s)",
            elapsed,
            len(broken),
            len(edges),
        )
        return broken

    # ------------------------------------------------------------------
    # Cycle detection (delegated)
    # ------------------------------------------------------------------

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the lineage graph.

        Delegates to :meth:`LineageGraphEngine.detect_cycles` which
        implements DFS-based cycle detection internally.

        Returns:
            List of cycles, where each cycle is a list of node ID
            strings forming the circular path.  An empty list
            indicates a cycle-free (DAG) graph.

        Example:
            >>> cycles = validator.detect_cycles()
            >>> if cycles:
            ...     print(f"Found {len(cycles)} cycle(s)")
        """
        start = time.monotonic()

        cycles = self._graph.detect_cycles()

        elapsed = time.monotonic() - start
        observe_processing_duration("detect_cycles", elapsed)
        logger.debug(
            "Cycle detection completed in %.3fs: %d cycle(s) found",
            elapsed,
            len(cycles),
        )
        return cycles

    # ------------------------------------------------------------------
    # Source coverage
    # ------------------------------------------------------------------

    def compute_source_coverage(self) -> dict:
        """Compute source coverage for report-type nodes.

        For each node with ``type == "report"`` in the lineage graph,
        this method checks whether the node has at least one upstream
        path leading to an authoritative source node (type
        ``"source"`` or ``"raw_data"`` or ``"external_source"``).

        Coverage is calculated as the ratio of reports with at least one
        traced authoritative source to the total number of report nodes.
        If there are no report nodes the coverage is 1.0 (vacuously true).

        Returns:
            Dictionary with coverage details::

                {
                    "coverage_score": 0.85,
                    "covered_reports": 17,
                    "total_reports": 20,
                    "uncovered_fields": [
                        {"report_id": "rpt-003", "name": "..."}
                    ]
                }
        """
        start = time.monotonic()

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()

        # Identify report nodes and source nodes
        report_nodes: List[dict] = []
        source_node_ids: set = set()
        _AUTHORITATIVE_TYPES = frozenset({
            "source", "raw_data", "external_source",
        })

        for node in nodes:
            node_type = node.get("type", node.get("asset_type", "")).lower()
            node_id = node.get("id") or node.get("node_id", "")
            if node_type == "report":
                report_nodes.append(node)
            if node_type in _AUTHORITATIVE_TYPES:
                source_node_ids.add(node_id)

        # If no reports exist, coverage is vacuously complete
        if not report_nodes:
            elapsed = time.monotonic() - start
            observe_processing_duration("compute_source_coverage", elapsed)
            return {
                "coverage_score": 1.0,
                "covered_reports": 0,
                "total_reports": 0,
                "uncovered_fields": [],
            }

        # Build reverse adjacency list for upstream traversal
        reverse_adj: Dict[str, List[str]] = {}
        for edge in edges:
            target_id = (
                edge.get("target_id")
                or edge.get("target")
                or edge.get("to_id", "")
            )
            source_id = (
                edge.get("source_id")
                or edge.get("source")
                or edge.get("from_id", "")
            )
            if target_id and source_id:
                if target_id not in reverse_adj:
                    reverse_adj[target_id] = []
                reverse_adj[target_id].append(source_id)

        covered_count = 0
        uncovered: List[dict] = []

        for rpt in report_nodes:
            rpt_id = rpt.get("id") or rpt.get("node_id", "")
            if self._has_upstream_source(rpt_id, reverse_adj, source_node_ids):
                covered_count += 1
            else:
                uncovered.append({
                    "report_id": rpt_id,
                    "name": rpt.get("name", ""),
                })

        total = len(report_nodes)
        score = covered_count / total if total > 0 else 1.0

        elapsed = time.monotonic() - start
        observe_processing_duration("compute_source_coverage", elapsed)
        logger.debug(
            "Source coverage computed in %.3fs: %.4f (%d/%d reports covered)",
            elapsed,
            score,
            covered_count,
            total,
        )

        return {
            "coverage_score": round(score, 4),
            "covered_reports": covered_count,
            "total_reports": total,
            "uncovered_fields": uncovered,
        }

    # ------------------------------------------------------------------
    # Completeness score
    # ------------------------------------------------------------------

    def compute_completeness_score(self) -> float:
        """Compute a weighted completeness score for the lineage graph.

        The score is composed of four weighted sub-scores:

        * **Orphan penalty (30%):** ``1.0 - min(orphan_count / total_nodes, 1.0)``
          A graph where every node is orphaned scores 0.0 on this dimension.
        * **Broken-edge penalty (30%):** ``1.0 - min(broken_count / total_edges, 1.0)``
          A graph where every edge is broken scores 0.0.
        * **Source coverage (20%):** The ``coverage_score`` from
          :meth:`compute_source_coverage`.
        * **Cycle penalty (20%):** ``1.0`` if no cycles, ``0.0`` if any
          cycle exists.

        Returns:
            Float between 0.0 (completely broken) and 1.0 (perfect).

        Example:
            >>> score = validator.compute_completeness_score()
            >>> assert 0.0 <= score <= 1.0
        """
        start = time.monotonic()

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()
        total_nodes = len(nodes)
        total_edges = len(edges)

        # Sub-score 1: Orphan penalty
        orphans = self.detect_orphan_nodes()
        if total_nodes > 0:
            orphan_sub = 1.0 - min(len(orphans) / total_nodes, 1.0)
        else:
            orphan_sub = 1.0  # No nodes means no orphan issues

        # Sub-score 2: Broken-edge penalty
        broken = self.detect_broken_edges()
        if total_edges > 0:
            broken_sub = 1.0 - min(len(broken) / total_edges, 1.0)
        else:
            broken_sub = 1.0  # No edges means no broken-edge issues

        # Sub-score 3: Source coverage
        coverage_result = self.compute_source_coverage()
        coverage_sub = coverage_result["coverage_score"]

        # Sub-score 4: Cycle penalty (binary)
        cycles = self.detect_cycles()
        cycle_sub = 0.0 if cycles else 1.0

        # Weighted combination
        completeness = (
            _WEIGHT_ORPHAN * orphan_sub
            + _WEIGHT_BROKEN_EDGE * broken_sub
            + _WEIGHT_COVERAGE * coverage_sub
            + _WEIGHT_CYCLE * cycle_sub
        )

        # Clamp to [0.0, 1.0] for safety
        completeness = max(0.0, min(1.0, completeness))

        elapsed = time.monotonic() - start
        observe_processing_duration("compute_completeness_score", elapsed)
        logger.debug(
            "Completeness score computed in %.3fs: %.4f "
            "(orphan=%.2f broken=%.2f coverage=%.2f cycle=%.2f)",
            elapsed,
            completeness,
            orphan_sub,
            broken_sub,
            coverage_sub,
            cycle_sub,
        )
        return round(completeness, 4)

    # ------------------------------------------------------------------
    # Freshness check
    # ------------------------------------------------------------------

    def check_freshness(
        self,
        max_age_hours: Optional[int] = None,
    ) -> dict:
        """Check if lineage metadata is stale.

        Inspects every node in the graph for a ``last_updated`` (or
        ``updated_at``) timestamp.  Nodes whose metadata is older than
        ``max_age_hours`` are flagged as stale.

        The freshness score is computed as::

            freshness = 1.0 - (stale_count / total_count)

        If there are no nodes the freshness is 1.0.

        Args:
            max_age_hours: Maximum acceptable age in hours before a
                node is considered stale.  Defaults to the config value
                ``freshness_max_age_hours``.

        Returns:
            Dictionary with freshness details::

                {
                    "freshness_score": 0.92,
                    "stale_assets": 3,
                    "total_assets": 40,
                    "stale_list": [
                        {
                            "node_id": "asset-007",
                            "name": "...",
                            "last_updated": "2026-02-10T...",
                            "age_hours": 168.5
                        },
                        ...
                    ]
                }
        """
        start = time.monotonic()
        cfg = get_config()
        threshold_hours = (
            max_age_hours
            if max_age_hours is not None
            else cfg.freshness_max_age_hours
        )

        nodes = self._graph.get_all_nodes()
        now = _utcnow()
        stale_list: List[dict] = []

        for node in nodes:
            node_id = node.get("id") or node.get("node_id", "")
            last_updated_raw = (
                node.get("last_updated")
                or node.get("updated_at")
                or node.get("last_refreshed_at")
            )
            if not last_updated_raw:
                # Nodes without a timestamp are considered stale
                stale_list.append({
                    "node_id": node_id,
                    "name": node.get("name", ""),
                    "last_updated": None,
                    "age_hours": float("inf"),
                })
                continue

            last_updated = self._parse_timestamp(last_updated_raw)
            if last_updated is None:
                stale_list.append({
                    "node_id": node_id,
                    "name": node.get("name", ""),
                    "last_updated": str(last_updated_raw),
                    "age_hours": float("inf"),
                })
                continue

            delta = now - last_updated
            age_hours = delta.total_seconds() / 3600.0

            if age_hours > threshold_hours:
                stale_list.append({
                    "node_id": node_id,
                    "name": node.get("name", ""),
                    "last_updated": last_updated.isoformat(),
                    "age_hours": round(age_hours, 2),
                })

        total = len(nodes)
        stale_count = len(stale_list)
        freshness_score = 1.0 - (stale_count / total) if total > 0 else 1.0
        freshness_score = max(0.0, min(1.0, freshness_score))

        elapsed = time.monotonic() - start
        observe_processing_duration("check_freshness", elapsed)
        logger.debug(
            "Freshness check completed in %.3fs: score=%.4f "
            "stale=%d/%d threshold=%dh",
            elapsed,
            freshness_score,
            stale_count,
            total,
            threshold_hours,
        )

        return {
            "freshness_score": round(freshness_score, 4),
            "stale_assets": stale_count,
            "total_assets": total,
            "stale_list": stale_list,
        }

    # ------------------------------------------------------------------
    # Pipeline coverage
    # ------------------------------------------------------------------

    def check_pipeline_coverage(self, pipeline_id: str) -> dict:
        """Check lineage coverage for a specific pipeline.

        Identifies all nodes that belong to the given pipeline (via the
        node's ``pipeline_id`` metadata field) and determines how many
        of those nodes are properly connected within the lineage graph
        (i.e., have at least one incoming or outgoing edge).

        Nodes with zero edges are reported as coverage gaps.

        Args:
            pipeline_id: Identifier of the pipeline to check.

        Returns:
            Dictionary with pipeline coverage details::

                {
                    "pipeline_id": "pipeline-abc",
                    "assets_traced": 8,
                    "total_assets": 10,
                    "coverage": 0.8,
                    "gaps": [
                        {"node_id": "asset-005", "name": "..."},
                        {"node_id": "asset-009", "name": "..."}
                    ]
                }

        Raises:
            ValueError: If *pipeline_id* is empty.
        """
        if not pipeline_id:
            raise ValueError("pipeline_id must not be empty")

        start = time.monotonic()

        nodes = self._graph.get_all_nodes()
        edges = self._graph.get_all_edges()

        # Filter nodes belonging to this pipeline
        pipeline_nodes: List[dict] = []
        for node in nodes:
            if node.get("pipeline_id") == pipeline_id:
                pipeline_nodes.append(node)

        if not pipeline_nodes:
            elapsed = time.monotonic() - start
            observe_processing_duration("check_pipeline_coverage", elapsed)
            logger.debug(
                "Pipeline coverage: no nodes found for pipeline_id=%s",
                pipeline_id,
            )
            return {
                "pipeline_id": pipeline_id,
                "assets_traced": 0,
                "total_assets": 0,
                "coverage": 1.0,
                "gaps": [],
            }

        # Build set of connected node IDs
        connected_ids: set = set()
        for edge in edges:
            source_id = (
                edge.get("source_id")
                or edge.get("source")
                or edge.get("from_id", "")
            )
            target_id = (
                edge.get("target_id")
                or edge.get("target")
                or edge.get("to_id", "")
            )
            if source_id:
                connected_ids.add(source_id)
            if target_id:
                connected_ids.add(target_id)

        traced = 0
        gaps: List[dict] = []
        for node in pipeline_nodes:
            node_id = node.get("id") or node.get("node_id", "")
            if node_id in connected_ids:
                traced += 1
            else:
                gaps.append({
                    "node_id": node_id,
                    "name": node.get("name", ""),
                })

        total = len(pipeline_nodes)
        coverage = traced / total if total > 0 else 1.0

        elapsed = time.monotonic() - start
        observe_processing_duration("check_pipeline_coverage", elapsed)
        logger.debug(
            "Pipeline coverage for '%s' computed in %.3fs: %.4f (%d/%d)",
            pipeline_id,
            elapsed,
            coverage,
            traced,
            total,
        )

        return {
            "pipeline_id": pipeline_id,
            "assets_traced": traced,
            "total_assets": total,
            "coverage": round(coverage, 4),
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Recommendation generation
    # ------------------------------------------------------------------

    def generate_recommendations(self, issues: List[dict]) -> List[str]:
        """Generate actionable remediation recommendations for issues.

        Maps each issue type to a human-readable suggestion that
        describes the corrective action a data engineer should take.

        Args:
            issues: List of issue dictionaries as produced by
                :meth:`validate`.  Each issue must have a ``"type"``
                key.

        Returns:
            De-duplicated list of recommendation strings.

        Example:
            >>> recs = validator.generate_recommendations([
            ...     {"type": "orphan_node", "node_id": "asset-001"}
            ... ])
            >>> assert len(recs) >= 1
        """
        recommendations: List[str] = []
        seen_types: set = set()

        for issue in issues:
            issue_type = issue.get("type", "")

            if issue_type == "orphan_node":
                node_id = issue.get("node_id", "<unknown>")
                rec = (
                    f"Register lineage edges for asset '{node_id}' to "
                    f"connect it to its upstream sources or downstream "
                    f"consumers and resolve the orphan node."
                )
                recommendations.append(rec)

            elif issue_type == "broken_edge":
                edge_id = issue.get("edge_id", "<unknown>")
                rec = (
                    f"Review edge '{edge_id}' and either register the "
                    f"missing source/target node or remove the stale "
                    f"edge to resolve the broken reference."
                )
                recommendations.append(rec)

            elif issue_type == "cycle":
                cycle_nodes = issue.get("nodes", [])
                preview = ", ".join(cycle_nodes[:3])
                rec = (
                    f"Investigate the circular dependency involving "
                    f"nodes [{preview}] and refactor the data flow to "
                    f"eliminate the cycle."
                )
                recommendations.append(rec)

            elif issue_type == "low_coverage":
                score = issue.get("coverage_score", 0.0)
                rec = (
                    f"Source coverage is {score:.0%}.  Trace uncovered "
                    f"report nodes back to their authoritative data "
                    f"sources and add the missing lineage edges."
                )
                if "low_coverage" not in seen_types:
                    recommendations.append(rec)
                    seen_types.add("low_coverage")

            elif issue_type == "stale_metadata":
                stale_count = issue.get("stale_assets", 0)
                rec = (
                    f"{stale_count} asset(s) have stale lineage metadata.  "
                    f"Trigger a lineage refresh for affected assets or "
                    f"verify that upstream pipelines are running on schedule."
                )
                if "stale_metadata" not in seen_types:
                    recommendations.append(rec)
                    seen_types.add("stale_metadata")

            else:
                # Generic recommendation for unknown issue types
                detail = issue.get("detail", "Unknown issue detected")
                rec = (
                    f"Review issue: {detail}.  Consult the lineage "
                    f"graph documentation for remediation steps."
                )
                recommendations.append(rec)

        if not recommendations:
            recommendations.append(
                "No issues detected.  The lineage graph is complete "
                "and consistent."
            )

        logger.debug(
            "Generated %d recommendation(s) for %d issue(s)",
            len(recommendations),
            len(issues),
        )
        return recommendations

    # ------------------------------------------------------------------
    # Validation retrieval
    # ------------------------------------------------------------------

    def get_validation(self, validation_id: str) -> Optional[dict]:
        """Retrieve a stored validation report by its ID.

        Args:
            validation_id: The validation report identifier
                (e.g., ``"VAL-abc123def456"``).

        Returns:
            The validation report dictionary, or ``None`` if not found.
        """
        with self._lock:
            return self._validations.get(validation_id)

    def list_validations(
        self,
        scope: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """List stored validation reports with optional filtering.

        Reports are returned in reverse chronological order (newest
        first) up to the specified limit.

        Args:
            scope: Optional filter by validation scope (e.g.,
                ``"full"``, ``"pipeline"``).
            result: Optional filter by validation result (``"pass"``,
                ``"warn"``, ``"fail"``).
            limit: Maximum number of reports to return.  Defaults to
                100.

        Returns:
            List of validation report dictionaries, newest first.
        """
        with self._lock:
            reports = list(self._validations.values())

        # Apply filters
        if scope is not None:
            reports = [r for r in reports if r.get("scope") == scope]
        if result is not None:
            reports = [r for r in reports if r.get("result") == result]

        # Sort by validated_at descending (newest first)
        reports.sort(
            key=lambda r: r.get("validated_at", ""),
            reverse=True,
        )

        return reports[:limit]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return summary statistics for the validator engine.

        Provides counts of total validations performed, breakdowns by
        result type, average completeness score, and the most recent
        validation timestamp.

        Returns:
            Dictionary with validator statistics::

                {
                    "total_validations": 15,
                    "pass_count": 10,
                    "warn_count": 3,
                    "fail_count": 2,
                    "average_completeness": 0.87,
                    "average_freshness": 0.93,
                    "last_validated_at": "2026-02-17T...",
                    "provenance_entries": 15
                }
        """
        with self._lock:
            all_reports = list(self._validations.values())

        total = len(all_reports)
        pass_count = sum(1 for r in all_reports if r.get("result") == _RESULT_PASS)
        warn_count = sum(1 for r in all_reports if r.get("result") == _RESULT_WARN)
        fail_count = sum(1 for r in all_reports if r.get("result") == _RESULT_FAIL)

        completeness_scores = [
            r.get("completeness_score", 0.0) for r in all_reports
        ]
        freshness_scores = [
            r.get("freshness_score", 0.0) for r in all_reports
        ]

        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores
            else 0.0
        )
        avg_freshness = (
            sum(freshness_scores) / len(freshness_scores)
            if freshness_scores
            else 0.0
        )

        # Most recent validation timestamp
        last_validated = ""
        if all_reports:
            sorted_reports = sorted(
                all_reports,
                key=lambda r: r.get("validated_at", ""),
                reverse=True,
            )
            last_validated = sorted_reports[0].get("validated_at", "")

        return {
            "total_validations": total,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "average_completeness": round(avg_completeness, 4),
            "average_freshness": round(avg_freshness, 4),
            "last_validated_at": last_validated,
            "provenance_entries": self._provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Clear / reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all stored validation results.

        Removes all in-memory validation reports.  The provenance chain
        is **not** cleared by this method (use
        ``provenance.reset()`` separately if needed).
        """
        with self._lock:
            count = len(self._validations)
            self._validations.clear()
        logger.info(
            "LineageValidatorEngine cleared: %d validation report(s) removed",
            count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_result(self, completeness: float, cfg: Any) -> str:
        """Map a completeness score to a pass/warn/fail result.

        Uses the configuration thresholds:

        * ``completeness < coverage_fail_threshold`` -> ``"fail"``
        * ``completeness < coverage_warn_threshold`` -> ``"warn"``
        * otherwise -> ``"pass"``

        Args:
            completeness: Completeness score in [0.0, 1.0].
            cfg: Configuration object with threshold attributes.

        Returns:
            One of ``"pass"``, ``"warn"``, or ``"fail"``.
        """
        if completeness < cfg.coverage_fail_threshold:
            return _RESULT_FAIL
        if completeness < cfg.coverage_warn_threshold:
            return _RESULT_WARN
        return _RESULT_PASS

    def _has_upstream_source(
        self,
        node_id: str,
        reverse_adj: Dict[str, List[str]],
        source_ids: set,
    ) -> bool:
        """Check if a node has at least one upstream authoritative source.

        Performs a BFS/DFS walk through the reverse adjacency list to
        find whether any ancestor of ``node_id`` is in ``source_ids``.

        Args:
            node_id: Starting node to trace upstream from.
            reverse_adj: Reverse adjacency list mapping target IDs to
                their source IDs.
            source_ids: Set of node IDs considered authoritative sources.

        Returns:
            ``True`` if at least one upstream source is reachable,
            ``False`` otherwise.
        """
        if node_id in source_ids:
            return True

        visited: set = set()
        stack: List[str] = [node_id]
        cfg = get_config()
        max_depth = cfg.max_graph_depth

        depth = 0
        while stack and depth < max_depth:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current in source_ids:
                return True

            parents = reverse_adj.get(current, [])
            for parent_id in parents:
                if parent_id not in visited:
                    stack.append(parent_id)

            depth += 1

        return False

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        """Parse a timestamp value into a timezone-aware datetime.

        Accepts ISO-8601 strings and ``datetime`` objects.  Naive
        datetimes are assumed to be UTC.

        Args:
            value: Timestamp to parse (str or datetime).

        Returns:
            A timezone-aware ``datetime``, or ``None`` if parsing fails.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if isinstance(value, str):
            # Try ISO-8601 parsing
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

            # Fallback: try fromisoformat (Python 3.7+)
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                pass

        logger.warning(
            "Failed to parse timestamp value: %r",
            value,
        )
        return None

    def _build_provenance_hash(self, data: Any) -> str:
        """Compute a SHA-256 hash for arbitrary data.

        Serialises the payload to canonical JSON with sorted keys before
        hashing so that equivalent structures produce identical digests.

        Args:
            data: JSON-serialisable data payload.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        if data is None:
            serialized = "null"
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
