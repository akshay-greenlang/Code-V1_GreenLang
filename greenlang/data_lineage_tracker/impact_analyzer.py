# -*- coding: utf-8 -*-
"""
ImpactAnalyzerEngine - AGENT-DATA-018: Data Lineage Tracker (GL-DATA-X-021)

Engine 4 of 7 in the Data Lineage Tracker Agent pipeline. Performs forward
and backward lineage traversal with impact scoring to answer two fundamental
data governance questions:

    1. **Backward (provenance):** "Where did this data come from?"
       Traces any asset upstream to all its source datasets, fields, and
       external sources, recording the transformation chain at each hop.

    2. **Forward (impact analysis):** "What breaks if this data changes?"
       Traces any asset downstream to all its consumers (reports, metrics,
       dashboards), scoring each by impact severity and freshness sensitivity.

Capabilities:
    - Backward lineage traversal with configurable depth limiting
    - Forward lineage traversal with blast radius calculation
    - Impact severity scoring (critical/high/medium/low) based on consumer
      asset type, distance, and transformation complexity
    - Dependency matrix generation for sets of assets
    - Root cause analysis (find all terminal source nodes)
    - Critical path discovery between arbitrary source/target pairs
    - Combined full analysis (forward + backward in a single report)
    - Thread-safe via threading.Lock for concurrent API access
    - SHA-256 provenance tracking on every analysis invocation

Zero-Hallucination Guarantees:
    - All traversal logic uses deterministic BFS via collections.deque
    - Impact severity is computed from explicit scoring rules, not ML/LLM
    - Blast radius is a simple ratio (affected / total), no estimation
    - Path finding uses exhaustive DFS with cycle detection
    - SHA-256 provenance recorded on every analysis invocation
    - No LLM calls anywhere in the analysis path

Example:
    >>> from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    >>> from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine
    >>> graph = LineageGraphEngine()
    >>> # ... register assets and edges ...
    >>> analyzer = ImpactAnalyzerEngine(graph)
    >>> backward = analyzer.analyze_backward("asset_report_001")
    >>> print(backward["source_count"])
    5
    >>> forward = analyzer.analyze_forward("asset_raw_001")
    >>> print(forward["blast_radius"])
    0.35
    >>> full = analyzer.analyze_full("asset_mid_001")
    >>> print(full["backward"]["source_count"], full["forward"]["affected_count"])
    3 7

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
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.data_lineage_tracker.config import get_config
from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
from greenlang.data_lineage_tracker.metrics import (
    PROMETHEUS_AVAILABLE,
    observe_processing_duration,
    record_impact_analysis,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SEVERITY_LEVELS: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

SEVERITY_THRESHOLDS: Dict[str, float] = {
    "critical": 0.9,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.0,
}

# Asset type importance weights for impact severity scoring.
# Higher weight means the consumer is more business-critical.
_ASSET_TYPE_WEIGHTS: Dict[str, float] = {
    "report": 1.0,
    "metric": 0.85,
    "dashboard": 0.80,
    "pipeline": 0.65,
    "dataset": 0.50,
    "agent": 0.45,
    "field": 0.40,
    "external_source": 0.30,
}

# Default weight for asset types not in the lookup table.
_DEFAULT_ASSET_TYPE_WEIGHT: float = 0.35

# Maximum paths returned by find_critical_paths to prevent combinatorial blow-up.
_MAX_CRITICAL_PATHS: int = 50


# ---------------------------------------------------------------------------
# Helper: UTC timestamp
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# ImpactAnalyzerEngine
# ---------------------------------------------------------------------------


class ImpactAnalyzerEngine:
    """Performs forward and backward lineage traversal with impact scoring.

    This engine operates on a ``LineageGraphEngine`` instance to answer
    provenance and impact analysis queries. Every analysis result is
    stored in-memory with a unique ID and full provenance hash so that
    audit trails can be reconstructed.

    Thread Safety:
        All public methods acquire ``self._lock`` before mutating shared
        state (``self._analyses``). Read-only graph traversals do not
        require the lock unless they store results.

    Attributes:
        _graph: The lineage graph engine to traverse.
        _analyses: In-memory store of completed analysis results keyed
            by analysis_id (UUID string).
        _lock: Threading lock for thread-safe mutation of shared state.
        _provenance: Provenance tracker for SHA-256 audit chain.

    Example:
        >>> graph = LineageGraphEngine()
        >>> analyzer = ImpactAnalyzerEngine(graph)
        >>> result = analyzer.analyze_backward("asset_001")
        >>> assert result["direction"] == "backward"
    """

    def __init__(
        self,
        graph: LineageGraphEngine,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize the ImpactAnalyzerEngine.

        Args:
            graph: The lineage graph engine providing node/edge access.
            provenance: Optional provenance tracker. A new instance is
                created if not supplied.
        """
        self._graph: LineageGraphEngine = graph
        self._analyses: Dict[str, dict] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = provenance if provenance is not None else ProvenanceTracker()
        logger.info("ImpactAnalyzerEngine initialized")

    # ------------------------------------------------------------------
    # Backward analysis (upstream provenance)
    # ------------------------------------------------------------------

    def analyze_backward(
        self,
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """Trace an asset upstream to all its source data.

        Performs a breadth-first traversal of incoming (upstream) edges
        from the root asset, recording every ancestor, its distance,
        the path taken, and the transformations applied.

        Args:
            asset_id: The root asset whose upstream lineage is queried.
            max_depth: Maximum traversal depth. Defaults to the config
                value ``default_traversal_depth`` when ``None``.

        Returns:
            Dictionary containing:
                - analysis_id (str): Unique ID for this analysis run.
                - root_asset_id (str): The queried asset.
                - direction (str): ``"backward"``.
                - depth (int): Maximum depth reached during traversal.
                - affected_assets (list[dict]): Each upstream asset with
                  ``asset_id``, ``distance``, ``path``, ``source_type``,
                  ``transformations_applied``.
                - affected_count (int): Total upstream assets found.
                - source_count (int): Count of terminal source nodes
                  (assets with no further upstream).
                - transformation_count (int): Total transformations
                  encountered across all paths.
                - unique_agents (list[str]): Distinct agent IDs involved.
                - provenance_hash (str): SHA-256 hash of the result.
                - created_at (str): ISO-8601 UTC timestamp.

        Raises:
            ValueError: If ``asset_id`` is empty or not found in graph.
        """
        start_time = time.monotonic()
        self._validate_asset_id(asset_id)

        effective_depth = self._resolve_depth(max_depth)
        affected_assets: List[dict] = []
        visited: Set[str] = {asset_id}
        max_depth_reached: int = 0
        all_transformations: List[str] = []
        all_agents: Set[str] = set()

        # BFS queue: (current_asset_id, distance, path_so_far)
        queue: deque = deque()
        queue.append((asset_id, 0, [asset_id]))

        while queue:
            current_id, distance, path = queue.popleft()
            if distance >= effective_depth:
                continue

            upstream_edges = self._graph.get_incoming_edges(current_id)
            for edge in upstream_edges:
                source_id = edge.get("source_asset_id", "")
                if not source_id or source_id in visited:
                    continue

                visited.add(source_id)
                new_distance = distance + 1
                new_path = path + [source_id]
                max_depth_reached = max(max_depth_reached, new_distance)

                # Collect transformation info from edge metadata
                transformation_type = edge.get("transformation_type", "unknown")
                agent_id = edge.get("agent_id", "")
                all_transformations.append(transformation_type)
                if agent_id:
                    all_agents.add(agent_id)

                # Determine source type from graph node metadata
                source_type = self._get_asset_type(source_id)

                # Build path from source to root (reversed for readability)
                path_to_root = list(reversed(new_path))

                # Collect transformations along this specific path
                path_transformations = self._collect_path_transformations(
                    new_path
                )

                affected_assets.append({
                    "asset_id": source_id,
                    "distance": new_distance,
                    "path": path_to_root,
                    "source_type": source_type,
                    "transformations_applied": path_transformations,
                })

                queue.append((source_id, new_distance, new_path))

        # Count terminal sources (no upstream edges of their own)
        source_count = self._count_terminal_sources(affected_assets)

        analysis_id = str(uuid.uuid4())
        created_at = _utcnow().isoformat()

        result: dict = {
            "analysis_id": analysis_id,
            "root_asset_id": asset_id,
            "direction": "backward",
            "depth": max_depth_reached,
            "affected_assets": affected_assets,
            "affected_count": len(affected_assets),
            "source_count": source_count,
            "transformation_count": len(all_transformations),
            "unique_agents": sorted(all_agents),
            "provenance_hash": "",
            "created_at": created_at,
        }

        # Provenance
        result["provenance_hash"] = self._compute_provenance_hash(result)
        self._record_provenance(analysis_id, "backward", result)

        # Store result
        with self._lock:
            self._analyses[analysis_id] = result

        # Metrics
        elapsed = time.monotonic() - start_time
        self._record_metrics("backward", result, elapsed)

        logger.info(
            "Backward analysis complete: asset=%s depth=%d affected=%d "
            "sources=%d transformations=%d elapsed=%.3fs",
            asset_id,
            max_depth_reached,
            len(affected_assets),
            source_count,
            len(all_transformations),
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Forward analysis (downstream impact)
    # ------------------------------------------------------------------

    def analyze_forward(
        self,
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """Trace an asset downstream to all its consumers with impact scoring.

        Performs a breadth-first traversal of outgoing (downstream) edges
        from the root asset, scoring each consumer by impact severity and
        freshness sensitivity.

        Args:
            asset_id: The root asset whose downstream impact is queried.
            max_depth: Maximum traversal depth. Defaults to the config
                value ``default_traversal_depth`` when ``None``.

        Returns:
            Dictionary containing:
                - analysis_id (str): Unique ID for this analysis run.
                - root_asset_id (str): The queried asset.
                - direction (str): ``"forward"``.
                - depth (int): Maximum depth reached during traversal.
                - affected_assets (list[dict]): Each downstream asset with
                  ``asset_id``, ``distance``, ``path``, ``asset_type``,
                  ``impact_severity``, ``freshness_sensitivity``,
                  ``transformations_applied``.
                - affected_count (int): Total downstream assets found.
                - blast_radius (float): Ratio of affected to total nodes.
                - severity_summary (dict): Counts per severity level.
                - unique_agents (list[str]): Distinct agent IDs involved.
                - provenance_hash (str): SHA-256 hash of the result.
                - created_at (str): ISO-8601 UTC timestamp.

        Raises:
            ValueError: If ``asset_id`` is empty or not found in graph.
        """
        start_time = time.monotonic()
        self._validate_asset_id(asset_id)

        effective_depth = self._resolve_depth(max_depth)
        affected_assets: List[dict] = []
        visited: Set[str] = {asset_id}
        max_depth_reached: int = 0
        all_agents: Set[str] = set()
        severity_counts: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        # BFS queue: (current_asset_id, distance, path_so_far)
        queue: deque = deque()
        queue.append((asset_id, 0, [asset_id]))

        while queue:
            current_id, distance, path = queue.popleft()
            if distance >= effective_depth:
                continue

            downstream_edges = self._graph.get_outgoing_edges(current_id)
            for edge in downstream_edges:
                target_id = edge.get("target_asset_id", "")
                if not target_id or target_id in visited:
                    continue

                visited.add(target_id)
                new_distance = distance + 1
                new_path = path + [target_id]
                max_depth_reached = max(max_depth_reached, new_distance)

                # Collect transformation info
                agent_id = edge.get("agent_id", "")
                if agent_id:
                    all_agents.add(agent_id)

                asset_type = self._get_asset_type(target_id)

                # Compute impact severity and freshness sensitivity
                affected_entry = {
                    "asset_id": target_id,
                    "distance": new_distance,
                    "path": list(new_path),
                    "asset_type": asset_type,
                    "impact_severity": "",
                    "freshness_sensitivity": 0.0,
                    "transformations_applied": self._collect_path_transformations(
                        new_path
                    ),
                }

                severity = self.compute_impact_severity(
                    asset_id, affected_entry
                )
                affected_entry["impact_severity"] = severity
                affected_entry["freshness_sensitivity"] = (
                    self._compute_freshness_sensitivity(new_distance)
                )

                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                affected_assets.append(affected_entry)

                queue.append((target_id, new_distance, new_path))

        blast_radius = self._compute_blast_radius_value(len(affected_assets))

        analysis_id = str(uuid.uuid4())
        created_at = _utcnow().isoformat()

        # Determine highest severity for metrics reporting
        highest_severity = self._highest_severity(severity_counts)

        result: dict = {
            "analysis_id": analysis_id,
            "root_asset_id": asset_id,
            "direction": "forward",
            "depth": max_depth_reached,
            "affected_assets": affected_assets,
            "affected_count": len(affected_assets),
            "blast_radius": blast_radius,
            "severity_summary": severity_counts,
            "unique_agents": sorted(all_agents),
            "provenance_hash": "",
            "created_at": created_at,
        }

        # Provenance
        result["provenance_hash"] = self._compute_provenance_hash(result)
        self._record_provenance(analysis_id, "forward", result)

        # Store result
        with self._lock:
            self._analyses[analysis_id] = result

        # Metrics
        elapsed = time.monotonic() - start_time
        self._record_metrics("forward", result, elapsed)

        logger.info(
            "Forward analysis complete: asset=%s depth=%d affected=%d "
            "blast_radius=%.4f severity=%s elapsed=%.3fs",
            asset_id,
            max_depth_reached,
            len(affected_assets),
            blast_radius,
            highest_severity,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Blast radius calculation
    # ------------------------------------------------------------------

    def compute_blast_radius(self, asset_id: str) -> float:
        """Compute the blast radius for an asset.

        The blast radius is the ratio of downstream affected assets to
        the total number of nodes in the lineage graph. A value of 0.0
        means no downstream consumers; 1.0 means every node in the graph
        is affected.

        This method performs a full forward traversal with no depth limit
        (constrained only by ``max_graph_depth`` from config) to count
        all reachable downstream nodes.

        Args:
            asset_id: The asset whose blast radius is computed.

        Returns:
            Float between 0.0 and 1.0 inclusive.

        Raises:
            ValueError: If ``asset_id`` is empty or not found in graph.
        """
        self._validate_asset_id(asset_id)

        # Count all reachable downstream nodes via BFS
        config = get_config()
        max_depth = config.max_graph_depth
        visited: Set[str] = {asset_id}
        queue: deque = deque()
        queue.append((asset_id, 0))

        while queue:
            current_id, distance = queue.popleft()
            if distance >= max_depth:
                continue
            downstream_edges = self._graph.get_outgoing_edges(current_id)
            for edge in downstream_edges:
                target_id = edge.get("target_asset_id", "")
                if target_id and target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, distance + 1))

        # Subtract 1 because visited includes the root asset itself
        affected_count = len(visited) - 1
        return self._compute_blast_radius_value(affected_count)

    # ------------------------------------------------------------------
    # Impact severity scoring
    # ------------------------------------------------------------------

    def compute_impact_severity(
        self,
        asset_id: str,
        affected_asset: dict,
    ) -> str:
        """Compute impact severity for a single affected downstream asset.

        The severity score is a weighted combination of three factors:
            1. **Asset type importance** (40%): Reports and metrics are
               more critical than intermediate datasets.
            2. **Distance penalty** (30%): Closer consumers are more
               directly affected (inverse of distance).
            3. **Transformation complexity** (30%): More transformations
               in the path indicate more complex (and fragile) lineage.

        The composite score is mapped to severity levels via thresholds:
            - >= 0.9 : critical
            - >= 0.7 : high
            - >= 0.4 : medium
            - <  0.4 : low

        Args:
            asset_id: The root asset being analyzed (for context).
            affected_asset: Dictionary with keys ``asset_type``,
                ``distance``, and ``transformations_applied``.

        Returns:
            One of ``"critical"``, ``"high"``, ``"medium"``, ``"low"``.
        """
        asset_type = affected_asset.get("asset_type", "")
        distance = affected_asset.get("distance", 1)
        transformations = affected_asset.get("transformations_applied", [])

        # Factor 1: Asset type importance (0.0 to 1.0)
        type_weight = _ASSET_TYPE_WEIGHTS.get(
            asset_type, _DEFAULT_ASSET_TYPE_WEIGHT
        )

        # Factor 2: Distance penalty (closer = higher impact)
        # Inverse distance capped between 0.0 and 1.0
        distance_score = 1.0 / max(distance, 1)

        # Factor 3: Transformation complexity
        # Fewer transformations = more direct, higher impact
        # Score decays with transformation count
        tx_count = len(transformations) if transformations else 0
        if tx_count == 0:
            complexity_score = 1.0
        else:
            complexity_score = 1.0 / (1.0 + 0.2 * tx_count)

        # Weighted composite
        composite = (
            0.40 * type_weight
            + 0.30 * distance_score
            + 0.30 * complexity_score
        )

        return self._score_to_severity(composite)

    # ------------------------------------------------------------------
    # Dependency matrix
    # ------------------------------------------------------------------

    def get_dependency_matrix(
        self, asset_ids: List[str],
    ) -> Dict[str, Dict[str, bool]]:
        """Compute a dependency matrix for a set of assets.

        For each pair ``(A, B)`` in the provided asset IDs, determines
        whether ``A`` depends on ``B`` (i.e., ``B`` is reachable by
        following upstream edges from ``A``).

        This method uses BFS backward traversal from each asset and
        checks membership in the provided set.

        Args:
            asset_ids: List of asset IDs to include in the matrix.

        Returns:
            Nested dictionary where ``result[A][B]`` is ``True`` if
            asset ``A`` depends on asset ``B``, ``False`` otherwise.
            Diagonal entries (``result[A][A]``) are always ``False``.
        """
        start_time = time.monotonic()
        config = get_config()
        max_depth = config.max_graph_depth
        asset_set = set(asset_ids)

        matrix: Dict[str, Dict[str, bool]] = {}

        for asset_id in asset_ids:
            matrix[asset_id] = {}
            # BFS backward to find all reachable upstream nodes
            reachable_upstream: Set[str] = set()
            visited: Set[str] = {asset_id}
            queue: deque = deque()
            queue.append((asset_id, 0))

            while queue:
                current_id, distance = queue.popleft()
                if distance >= max_depth:
                    continue
                upstream_edges = self._graph.get_incoming_edges(current_id)
                for edge in upstream_edges:
                    source_id = edge.get("source_asset_id", "")
                    if source_id and source_id not in visited:
                        visited.add(source_id)
                        reachable_upstream.add(source_id)
                        queue.append((source_id, distance + 1))

            # Populate row: A depends on B if B is reachable upstream from A
            for other_id in asset_ids:
                if other_id == asset_id:
                    matrix[asset_id][other_id] = False
                else:
                    matrix[asset_id][other_id] = other_id in reachable_upstream

        elapsed = time.monotonic() - start_time
        observe_processing_duration("dependency_matrix", elapsed)
        logger.info(
            "Dependency matrix computed for %d assets in %.3fs",
            len(asset_ids),
            elapsed,
        )
        return matrix

    # ------------------------------------------------------------------
    # Root cause analysis
    # ------------------------------------------------------------------

    def find_root_causes(self, asset_id: str) -> List[dict]:
        """Find all terminal root sources for a given asset.

        A root source is a node that has no incoming (upstream) edges
        in the lineage graph -- it represents an original data source
        from which the queried asset ultimately derives.

        Args:
            asset_id: The asset whose root causes are sought.

        Returns:
            List of dictionaries, each containing:
                - asset_id (str): The root source asset ID.
                - asset_type (str): The type of the root source.
                - distance (int): Shortest distance from the queried
                  asset to this root source.
                - path (list[str]): Ordered path from root source to
                  the queried asset.
                - transformations (list[str]): Transformation types
                  along the path.

        Raises:
            ValueError: If ``asset_id`` is empty or not found in graph.
        """
        start_time = time.monotonic()
        self._validate_asset_id(asset_id)

        config = get_config()
        max_depth = config.max_graph_depth

        root_causes: List[dict] = []
        visited: Set[str] = {asset_id}
        # BFS: (current_id, distance, path_so_far)
        queue: deque = deque()
        queue.append((asset_id, 0, [asset_id]))

        while queue:
            current_id, distance, path = queue.popleft()
            if distance >= max_depth:
                continue

            upstream_edges = self._graph.get_incoming_edges(current_id)

            # If no upstream edges, current node is a root source
            if not upstream_edges and current_id != asset_id:
                root_path = list(reversed(path))
                root_causes.append({
                    "asset_id": current_id,
                    "asset_type": self._get_asset_type(current_id),
                    "distance": distance,
                    "path": root_path,
                    "transformations": self._collect_path_transformations(
                        path
                    ),
                })
                continue

            for edge in upstream_edges:
                source_id = edge.get("source_asset_id", "")
                if source_id and source_id not in visited:
                    visited.add(source_id)
                    new_path = path + [source_id]
                    queue.append((source_id, distance + 1, new_path))

        elapsed = time.monotonic() - start_time
        observe_processing_duration("root_cause_analysis", elapsed)
        logger.info(
            "Root cause analysis for asset=%s found %d root sources in %.3fs",
            asset_id,
            len(root_causes),
            elapsed,
        )
        return root_causes

    # ------------------------------------------------------------------
    # Critical path discovery
    # ------------------------------------------------------------------

    def find_critical_paths(
        self,
        source_id: str,
        target_id: str,
    ) -> List[List[str]]:
        """Find all paths between a source and target asset.

        Uses depth-first search with cycle detection to enumerate paths,
        capped at ``_MAX_CRITICAL_PATHS`` to prevent combinatorial
        blow-up. Paths are sorted by length (shortest first), reflecting
        the most direct (and therefore most critical) data flow.

        Args:
            source_id: Starting asset for path discovery.
            target_id: Destination asset for path discovery.

        Returns:
            List of paths, where each path is a list of asset IDs from
            ``source_id`` to ``target_id``. Sorted by length ascending
            (shortest = most critical first). Empty list if no path
            exists.

        Raises:
            ValueError: If either ID is empty or not found in graph.
        """
        start_time = time.monotonic()
        self._validate_asset_id(source_id)
        self._validate_asset_id(target_id)

        if source_id == target_id:
            return [[source_id]]

        config = get_config()
        max_depth = config.max_graph_depth
        all_paths: List[List[str]] = []

        # DFS with explicit stack: (current_id, path_so_far, visited_set)
        stack: List[Tuple[str, List[str], Set[str]]] = [
            (source_id, [source_id], {source_id})
        ]

        while stack and len(all_paths) < _MAX_CRITICAL_PATHS:
            current_id, path, visited_in_path = stack.pop()

            if len(path) > max_depth:
                continue

            downstream_edges = self._graph.get_outgoing_edges(current_id)
            for edge in downstream_edges:
                next_id = edge.get("target_asset_id", "")
                if not next_id or next_id in visited_in_path:
                    continue

                new_path = path + [next_id]

                if next_id == target_id:
                    all_paths.append(new_path)
                    if len(all_paths) >= _MAX_CRITICAL_PATHS:
                        break
                else:
                    new_visited = visited_in_path | {next_id}
                    stack.append((next_id, new_path, new_visited))

        # Sort by path length (shortest = most critical)
        all_paths.sort(key=len)

        elapsed = time.monotonic() - start_time
        observe_processing_duration("critical_path_discovery", elapsed)
        logger.info(
            "Critical path discovery: source=%s target=%s found %d paths "
            "in %.3fs",
            source_id,
            target_id,
            len(all_paths),
            elapsed,
        )
        return all_paths

    # ------------------------------------------------------------------
    # Full analysis (combined forward + backward)
    # ------------------------------------------------------------------

    def analyze_full(
        self,
        asset_id: str,
        max_depth: Optional[int] = None,
    ) -> dict:
        """Run combined forward and backward analysis for an asset.

        Produces a unified impact report containing both upstream
        provenance and downstream impact data in a single invocation.

        Args:
            asset_id: The asset to analyze in both directions.
            max_depth: Maximum traversal depth for both directions.
                Defaults to the config value when ``None``.

        Returns:
            Dictionary containing:
                - analysis_id (str): Unique ID for this combined analysis.
                - root_asset_id (str): The queried asset.
                - direction (str): ``"bidirectional"``.
                - backward (dict): Full backward analysis result.
                - forward (dict): Full forward analysis result.
                - root_causes (list[dict]): Terminal source nodes.
                - blast_radius (float): Forward blast radius.
                - total_affected_count (int): Combined upstream +
                  downstream affected count (deduplicated).
                - provenance_hash (str): SHA-256 hash of the result.
                - created_at (str): ISO-8601 UTC timestamp.

        Raises:
            ValueError: If ``asset_id`` is empty or not found in graph.
        """
        start_time = time.monotonic()
        self._validate_asset_id(asset_id)

        backward = self.analyze_backward(asset_id, max_depth)
        forward = self.analyze_forward(asset_id, max_depth)
        root_causes = self.find_root_causes(asset_id)

        # Deduplicate affected assets across both directions
        backward_ids = {
            a["asset_id"] for a in backward.get("affected_assets", [])
        }
        forward_ids = {
            a["asset_id"] for a in forward.get("affected_assets", [])
        }
        total_unique = len(backward_ids | forward_ids)

        analysis_id = str(uuid.uuid4())
        created_at = _utcnow().isoformat()

        result: dict = {
            "analysis_id": analysis_id,
            "root_asset_id": asset_id,
            "direction": "bidirectional",
            "backward": backward,
            "forward": forward,
            "root_causes": root_causes,
            "blast_radius": forward.get("blast_radius", 0.0),
            "total_affected_count": total_unique,
            "provenance_hash": "",
            "created_at": created_at,
        }

        # Provenance
        result["provenance_hash"] = self._compute_provenance_hash(result)
        self._record_provenance(analysis_id, "bidirectional", result)

        # Store result
        with self._lock:
            self._analyses[analysis_id] = result

        # Metrics
        elapsed = time.monotonic() - start_time
        highest = self._highest_severity(
            forward.get("severity_summary", {})
        )
        record_impact_analysis("bidirectional", highest)
        observe_processing_duration("impact_analyze", elapsed)

        logger.info(
            "Full analysis complete: asset=%s upstream=%d downstream=%d "
            "total_unique=%d blast_radius=%.4f elapsed=%.3fs",
            asset_id,
            backward.get("affected_count", 0),
            forward.get("affected_count", 0),
            total_unique,
            forward.get("blast_radius", 0.0),
            elapsed,
        )
        return result

    def analyze_all(
        self,
        max_depth: Optional[int] = None,
    ) -> dict:
        """Run broad impact analysis across all root nodes in the graph.

        Discovers all root nodes (sources with no incoming edges) and
        performs forward impact analysis from each. This method is used
        by the pipeline's ``analyze`` stage to produce a graph-wide
        impact summary without requiring a specific asset_id.

        Args:
            max_depth: Maximum traversal depth for each root analysis.
                Defaults to the config value when ``None``.

        Returns:
            Dictionary containing:
                - analyses (list[dict]): Per-root forward analysis results.
                - root_count (int): Number of root nodes analyzed.
                - total_affected (int): Total unique downstream assets.
                - max_blast_radius (float): Highest blast radius found.
                - status (str): ``"completed"``.
        """
        start_time = time.monotonic()

        roots = self._graph.get_roots() if hasattr(self._graph, "get_roots") else []
        analyses = []
        all_affected: Set[str] = set()
        max_blast = 0.0

        for root_id in roots:
            try:
                fwd = self.analyze_forward(root_id, max_depth)
                analyses.append(fwd)
                for a in fwd.get("affected_assets", []):
                    all_affected.add(a.get("asset_id", ""))
                max_blast = max(max_blast, fwd.get("blast_radius", 0.0))
            except Exception as exc:
                logger.warning(
                    "analyze_all: skipping root %s: %s", root_id, exc,
                )

        elapsed = time.monotonic() - start_time
        observe_processing_duration("impact_analyze_all", elapsed)

        logger.info(
            "analyze_all complete: roots=%d affected=%d max_blast=%.4f "
            "elapsed=%.3fs",
            len(roots),
            len(all_affected),
            max_blast,
            elapsed,
        )

        return {
            "analyses": analyses,
            "root_count": len(roots),
            "total_affected": len(all_affected),
            "max_blast_radius": max_blast,
            "status": "completed",
        }

    # ------------------------------------------------------------------
    # Analysis retrieval
    # ------------------------------------------------------------------

    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        """Retrieve a stored analysis result by its unique ID.

        Args:
            analysis_id: The UUID of the analysis to retrieve.

        Returns:
            The analysis result dictionary, or ``None`` if not found.
        """
        with self._lock:
            return self._analyses.get(analysis_id)

    def list_analyses(
        self,
        asset_id: Optional[str] = None,
        direction: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """List stored analysis results with optional filtering.

        Args:
            asset_id: Filter by root asset ID. When ``None``, all
                analyses are returned.
            direction: Filter by direction (``"forward"``,
                ``"backward"``, ``"bidirectional"``). When ``None``,
                all directions are included.
            limit: Maximum number of results to return. Defaults to 100.

        Returns:
            List of analysis result dictionaries, ordered by creation
            time descending (most recent first), capped at ``limit``.
        """
        with self._lock:
            candidates = list(self._analyses.values())

        # Apply filters
        if asset_id is not None:
            candidates = [
                a for a in candidates
                if a.get("root_asset_id") == asset_id
            ]
        if direction is not None:
            candidates = [
                a for a in candidates
                if a.get("direction") == direction
            ]

        # Sort by created_at descending (most recent first)
        candidates.sort(
            key=lambda a: a.get("created_at", ""),
            reverse=True,
        )

        return candidates[:limit]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return summary statistics for all stored analyses.

        Returns:
            Dictionary containing:
                - total_analyses (int): Total number of stored analyses.
                - backward_count (int): Number of backward analyses.
                - forward_count (int): Number of forward analyses.
                - bidirectional_count (int): Number of full analyses.
                - unique_assets_analyzed (int): Distinct root assets.
                - avg_affected_count (float): Mean affected assets
                  per analysis.
                - max_blast_radius (float): Highest blast radius seen.
                - severity_distribution (dict): Aggregate severity
                  counts across all forward analyses.
        """
        with self._lock:
            analyses = list(self._analyses.values())

        if not analyses:
            return {
                "total_analyses": 0,
                "backward_count": 0,
                "forward_count": 0,
                "bidirectional_count": 0,
                "unique_assets_analyzed": 0,
                "avg_affected_count": 0.0,
                "max_blast_radius": 0.0,
                "severity_distribution": {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0,
                },
            }

        backward_count = sum(
            1 for a in analyses if a.get("direction") == "backward"
        )
        forward_count = sum(
            1 for a in analyses if a.get("direction") == "forward"
        )
        bidirectional_count = sum(
            1 for a in analyses if a.get("direction") == "bidirectional"
        )
        unique_assets = len({a.get("root_asset_id") for a in analyses})

        total_affected = sum(
            a.get("affected_count", a.get("total_affected_count", 0))
            for a in analyses
        )
        avg_affected = total_affected / len(analyses) if analyses else 0.0

        max_blast = max(
            (a.get("blast_radius", 0.0) for a in analyses),
            default=0.0,
        )

        # Aggregate severity distribution from forward analyses
        agg_severity: Dict[str, int] = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }
        for a in analyses:
            summary = a.get("severity_summary", {})
            for level in agg_severity:
                agg_severity[level] += summary.get(level, 0)

        return {
            "total_analyses": len(analyses),
            "backward_count": backward_count,
            "forward_count": forward_count,
            "bidirectional_count": bidirectional_count,
            "unique_assets_analyzed": unique_assets,
            "avg_affected_count": round(avg_affected, 2),
            "max_blast_radius": round(max_blast, 4),
            "severity_distribution": agg_severity,
        }

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all stored analysis results.

        Removes all in-memory analysis data. Intended for testing and
        administrative reset scenarios. Does not affect the underlying
        lineage graph or provenance tracker.
        """
        with self._lock:
            count = len(self._analyses)
            self._analyses.clear()
        logger.info("ImpactAnalyzerEngine cleared %d stored analyses", count)

    # ------------------------------------------------------------------
    # Internal: validation helpers
    # ------------------------------------------------------------------

    def _validate_asset_id(self, asset_id: str) -> None:
        """Validate that an asset ID is non-empty and exists in the graph.

        Args:
            asset_id: The asset ID to validate.

        Raises:
            ValueError: If the ID is empty or not found in the graph.
        """
        if not asset_id:
            raise ValueError("asset_id must not be empty")
        if not self._graph.has_node(asset_id):
            raise ValueError(
                f"Asset '{asset_id}' not found in lineage graph"
            )

    def _resolve_depth(self, max_depth: Optional[int]) -> int:
        """Resolve the effective traversal depth.

        Uses the provided value if not ``None``, otherwise falls back
        to the config ``default_traversal_depth``. Caps at the config
        ``max_graph_depth`` to prevent runaway traversals.

        Args:
            max_depth: Caller-supplied depth, or ``None`` for default.

        Returns:
            The effective depth limit as a positive integer.
        """
        config = get_config()
        if max_depth is not None:
            return min(max(max_depth, 1), config.max_graph_depth)
        return min(config.default_traversal_depth, config.max_graph_depth)

    # ------------------------------------------------------------------
    # Internal: graph access helpers
    # ------------------------------------------------------------------

    def _get_asset_type(self, asset_id: str) -> str:
        """Retrieve the asset_type for a node from the graph.

        Falls back to ``"unknown"`` if the node has no type metadata.

        Args:
            asset_id: The asset whose type is queried.

        Returns:
            Asset type string (e.g., ``"dataset"``, ``"report"``).
        """
        node = self._graph.get_node(asset_id)
        if node is None:
            return "unknown"
        return node.get("asset_type", "unknown")

    def _collect_path_transformations(
        self, path: List[str],
    ) -> List[str]:
        """Collect transformation types along a sequence of asset IDs.

        For each consecutive pair ``(path[i], path[i+1])``, looks up
        the edge between them and extracts the transformation type.

        Args:
            path: Ordered list of asset IDs forming a lineage path.

        Returns:
            List of transformation type strings along the path.
        """
        transformations: List[str] = []
        for i in range(len(path) - 1):
            # Check both directions: the path may be forward or backward
            edges = self._graph.get_edges_between(path[i], path[i + 1])
            if not edges:
                # Try reversed direction (backward path stored in
                # traversal order, not edge direction)
                edges = self._graph.get_edges_between(path[i + 1], path[i])
            for edge in edges:
                tx_type = edge.get("transformation_type", "")
                if tx_type:
                    transformations.append(tx_type)
        return transformations

    def _count_terminal_sources(
        self, affected_assets: List[dict],
    ) -> int:
        """Count terminal source nodes among affected assets.

        A terminal source is an asset with no upstream (incoming) edges
        in the lineage graph.

        Args:
            affected_assets: List of upstream asset dictionaries.

        Returns:
            Count of terminal sources.
        """
        count = 0
        for asset in affected_assets:
            aid = asset.get("asset_id", "")
            if aid:
                upstream = self._graph.get_incoming_edges(aid)
                if not upstream:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Internal: scoring helpers
    # ------------------------------------------------------------------

    def _score_to_severity(self, score: float) -> str:
        """Map a numeric composite score to a severity level string.

        Uses ``SEVERITY_THRESHOLDS`` to classify: the first threshold
        that the score meets or exceeds (checked from highest to lowest)
        determines the severity.

        Args:
            score: Composite score between 0.0 and 1.0.

        Returns:
            One of ``"critical"``, ``"high"``, ``"medium"``, ``"low"``.
        """
        if score >= SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        if score >= SEVERITY_THRESHOLDS["high"]:
            return "high"
        if score >= SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    def _compute_freshness_sensitivity(self, distance: int) -> float:
        """Compute freshness sensitivity based on hop distance.

        Assets closer to the root asset are more sensitive to freshness
        changes. Sensitivity is inversely proportional to distance.

        Args:
            distance: Number of hops from the root asset.

        Returns:
            Sensitivity score between 0.0 and 1.0. Distance 1 yields
            1.0 (maximum sensitivity); higher distances yield lower
            scores that asymptotically approach 0.0.
        """
        if distance <= 0:
            return 1.0
        return 1.0 / distance

    def _compute_blast_radius_value(self, affected_count: int) -> float:
        """Compute blast radius as a ratio of affected to total nodes.

        Args:
            affected_count: Number of downstream affected nodes
                (excluding the root node itself).

        Returns:
            Float between 0.0 and 1.0 inclusive. Returns 0.0 if the
            graph is empty or has only the root node.
        """
        total_nodes = self._graph.node_count()
        if total_nodes <= 1:
            return 0.0
        # Denominator is (total - 1) because the root asset is excluded
        return min(affected_count / (total_nodes - 1), 1.0)

    def _highest_severity(self, severity_counts: Dict[str, int]) -> str:
        """Determine the highest severity level with a non-zero count.

        Args:
            severity_counts: Dictionary mapping severity levels to
                their occurrence counts.

        Returns:
            The highest severity level that has count > 0, or
            ``"none"`` if all counts are zero.
        """
        for level in ("critical", "high", "medium", "low"):
            if severity_counts.get(level, 0) > 0:
                return level
        return "none"

    # ------------------------------------------------------------------
    # Internal: provenance helpers
    # ------------------------------------------------------------------

    def _compute_provenance_hash(self, result: dict) -> str:
        """Compute a SHA-256 hash for an analysis result.

        The hash covers the full result (excluding the provenance_hash
        field itself to avoid circular dependency) using canonical JSON
        serialization with sorted keys.

        Args:
            result: The analysis result dictionary.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Shallow copy to exclude provenance_hash from hash input
        hashable = {
            k: v for k, v in result.items() if k != "provenance_hash"
        }
        serialized = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_provenance(
        self,
        analysis_id: str,
        direction: str,
        result: dict,
    ) -> None:
        """Record a provenance entry for an analysis operation.

        Args:
            analysis_id: Unique analysis identifier.
            direction: Traversal direction (backward/forward/bidirectional).
            result: The analysis result being recorded.
        """
        self._provenance.record(
            entity_type="impact_analysis",
            entity_id=analysis_id,
            action="impact_analyzed",
            metadata={
                "direction": direction,
                "root_asset_id": result.get("root_asset_id", ""),
                "affected_count": result.get(
                    "affected_count",
                    result.get("total_affected_count", 0),
                ),
                "provenance_hash": result.get("provenance_hash", ""),
            },
        )

    # ------------------------------------------------------------------
    # Internal: metrics helpers
    # ------------------------------------------------------------------

    def _record_metrics(
        self,
        direction: str,
        result: dict,
        elapsed: float,
    ) -> None:
        """Record Prometheus metrics for an analysis operation.

        Args:
            direction: Traversal direction for the metric label.
            result: The analysis result for extracting severity.
            elapsed: Wall-clock time of the operation in seconds.
        """
        if direction == "forward":
            severity = self._highest_severity(
                result.get("severity_summary", {})
            )
        else:
            severity = "none"

        record_impact_analysis(direction, severity)
        observe_processing_duration("impact_analyze", elapsed)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["ImpactAnalyzerEngine"]
