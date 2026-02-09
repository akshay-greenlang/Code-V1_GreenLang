# -*- coding: utf-8 -*-
"""
Cluster Resolver Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Groups matched record pairs into duplicate clusters using Union-Find
(disjoint set with path compression and union by rank) or Connected
Components (BFS-based graph traversal). Provides cluster quality
metrics, representative selection, and cluster manipulation.

Zero-Hallucination Guarantees:
    - Union-Find uses path compression and union by rank (O(alpha(n)))
    - Connected Components uses deterministic BFS traversal
    - Quality metrics use deterministic arithmetic
    - No ML/LLM calls in clustering path
    - Provenance recorded for every clustering operation

Supported Algorithms:
    UNION_FIND:             Disjoint-set with path compression + union by rank
    CONNECTED_COMPONENTS:   BFS-based connected component discovery

Example:
    >>> from greenlang.duplicate_detector.cluster_resolver import ClusterResolver
    >>> resolver = ClusterResolver()
    >>> clusters = resolver.form_clusters(
    ...     match_results=matches,
    ...     algorithm=ClusterAlgorithm.UNION_FIND,
    ... )

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.duplicate_detector.models import (
    ClusterAlgorithm,
    DuplicateCluster,
    MatchClassification,
    MatchResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ClusterResolver",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a clustering operation."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_QUALITY: float = 0.5


# =============================================================================
# UnionFind (internal data structure)
# =============================================================================


class _UnionFind:
    """Disjoint-set (Union-Find) with path compression and union by rank.

    Provides near O(1) amortized find and union operations via
    path compression (flattening trees during find) and union by
    rank (attaching smaller trees under larger ones).

    Attributes:
        _parent: Mapping of element to its parent.
        _rank: Mapping of element to its rank (tree depth bound).
    """

    def __init__(self) -> None:
        """Initialize empty union-find structure."""
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def make_set(self, x: str) -> None:
        """Create a singleton set for element x.

        Args:
            x: Element to create a set for.
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        """Find the representative of the set containing x.

        Uses path compression to flatten the tree structure.

        Args:
            x: Element to find the representative for.

        Returns:
            Representative element of the set.
        """
        if x not in self._parent:
            self.make_set(x)

        # Path compression: iterative approach
        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        # Flatten the path
        current = x
        while self._parent[current] != root:
            next_parent = self._parent[current]
            self._parent[current] = root
            current = next_parent

        return root

    def union(self, x: str, y: str) -> bool:
        """Merge the sets containing x and y.

        Uses union by rank to keep trees balanced.

        Args:
            x: First element.
            y: Second element.

        Returns:
            True if sets were merged, False if already in same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
        elif self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1

        return True

    def get_components(self) -> Dict[str, List[str]]:
        """Get all connected components as groups.

        Returns:
            Dictionary mapping representative to list of members.
        """
        components: Dict[str, List[str]] = defaultdict(list)
        for element in self._parent:
            root = self.find(element)
            components[root].append(element)
        return dict(components)


# =============================================================================
# ClusterResolver
# =============================================================================


class ClusterResolver:
    """Cluster resolution engine for duplicate detection.

    Groups matched record pairs into duplicate clusters using
    Union-Find or Connected Components algorithms. Provides
    cluster quality assessment, representative selection, and
    cluster manipulation (split/merge).

    This engine follows GreenLang's zero-hallucination principle:
    all clustering uses deterministic graph algorithms with no ML.

    Attributes:
        _stats_lock: Threading lock for stats updates.
        _invocations: Total invocation count.
        _successes: Total successful invocations.
        _failures: Total failed invocations.
        _total_duration_ms: Cumulative processing time.

    Example:
        >>> resolver = ClusterResolver()
        >>> clusters = resolver.form_clusters(matches)
        >>> for c in clusters:
        ...     print(c.cluster_id, c.member_count, c.cluster_quality)
    """

    def __init__(self) -> None:
        """Initialize ClusterResolver with empty statistics."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None
        logger.info("ClusterResolver initialized")

    # ------------------------------------------------------------------
    # Public API - Cluster formation
    # ------------------------------------------------------------------

    def form_clusters(
        self,
        match_results: List[MatchResult],
        algorithm: ClusterAlgorithm = ClusterAlgorithm.UNION_FIND,
        min_quality: float = _DEFAULT_MIN_QUALITY,
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DuplicateCluster]:
        """Form duplicate clusters from match results.

        Args:
            match_results: List of MATCH-classified record pairs.
            algorithm: Clustering algorithm to use.
            min_quality: Minimum quality score to accept a cluster.
            records: Optional full records for representative selection.

        Returns:
            List of DuplicateCluster instances.

        Raises:
            ValueError: If match_results is empty.
        """
        start_time = time.monotonic()
        try:
            if not match_results:
                raise ValueError("match_results must not be empty")

            # Filter to MATCH-classified pairs only
            matches = [
                r for r in match_results
                if r.classification == MatchClassification.MATCH
            ]

            if not matches:
                logger.info("No MATCH pairs found, returning empty clusters")
                self._record_success(time.monotonic() - start_time)
                return []

            if algorithm == ClusterAlgorithm.UNION_FIND:
                clusters = self.union_find_clustering(matches, records)
            elif algorithm == ClusterAlgorithm.CONNECTED_COMPONENTS:
                clusters = self.connected_components(matches, records)
            else:
                raise ValueError(f"Unknown cluster algorithm: {algorithm}")

            # Filter by quality
            filtered = [c for c in clusters if c.cluster_quality >= min_quality]

            self._record_success(time.monotonic() - start_time)
            logger.info(
                "Formed %d clusters from %d matches using %s "
                "(%d passed quality threshold %.2f)",
                len(clusters), len(matches), algorithm.value,
                len(filtered), min_quality,
            )
            return filtered

        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error("Cluster formation failed: %s", e)
            raise

    def union_find_clustering(
        self,
        match_results: List[MatchResult],
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DuplicateCluster]:
        """Cluster matched pairs using Union-Find algorithm.

        Uses disjoint-set with path compression and union by rank
        for O(alpha(n)) amortized per-operation complexity.

        Args:
            match_results: List of MATCH-classified pairs.
            records: Optional full records for representative selection.

        Returns:
            List of DuplicateCluster instances.
        """
        uf = _UnionFind()

        # Build pair scores index for quality computation
        pair_scores: Dict[Tuple[str, str], float] = {}

        for mr in match_results:
            uf.make_set(mr.record_a_id)
            uf.make_set(mr.record_b_id)
            uf.union(mr.record_a_id, mr.record_b_id)

            pair_key = tuple(sorted([mr.record_a_id, mr.record_b_id]))
            pair_scores[pair_key] = mr.overall_score

        components = uf.get_components()
        clusters: List[DuplicateCluster] = []

        for _root, members in components.items():
            if len(members) < 2:
                continue

            cluster = self._build_cluster(
                members, pair_scores, records,
            )
            clusters.append(cluster)

        return clusters

    def connected_components(
        self,
        match_results: List[MatchResult],
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[DuplicateCluster]:
        """Cluster matched pairs using BFS connected components.

        Builds an adjacency list from match pairs and discovers
        connected components via breadth-first search.

        Args:
            match_results: List of MATCH-classified pairs.
            records: Optional full records for representative selection.

        Returns:
            List of DuplicateCluster instances.
        """
        # Build adjacency list
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        pair_scores: Dict[Tuple[str, str], float] = {}

        for mr in match_results:
            adjacency[mr.record_a_id].add(mr.record_b_id)
            adjacency[mr.record_b_id].add(mr.record_a_id)

            pair_key = tuple(sorted([mr.record_a_id, mr.record_b_id]))
            pair_scores[pair_key] = mr.overall_score

        visited: Set[str] = set()
        clusters: List[DuplicateCluster] = []

        for node in adjacency:
            if node in visited:
                continue

            # BFS
            component: List[str] = []
            queue: deque[str] = deque([node])
            visited.add(node)

            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= 2:
                cluster = self._build_cluster(
                    component, pair_scores, records,
                )
                clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------------
    # Public API - Cluster quality metrics
    # ------------------------------------------------------------------

    def compute_cluster_quality(
        self,
        member_ids: List[str],
        pair_scores: Dict[Tuple[str, str], float],
    ) -> float:
        """Compute overall quality score for a cluster.

        Quality is the average pairwise similarity among all
        member pairs. Only pairs that were actually compared
        (present in pair_scores) are included.

        Args:
            member_ids: List of record identifiers in the cluster.
            pair_scores: Mapping of (sorted pair) to similarity score.

        Returns:
            Average pairwise similarity (0.0 to 1.0).
        """
        if len(member_ids) < 2:
            return 0.0

        total_score = 0.0
        pair_count = 0

        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                pair_key = tuple(sorted([member_ids[i], member_ids[j]]))
                if pair_key in pair_scores:
                    total_score += pair_scores[pair_key]
                    pair_count += 1

        if pair_count == 0:
            return 0.0

        return round(total_score / pair_count, 6)

    def compute_cluster_density(
        self,
        member_ids: List[str],
        pair_scores: Dict[Tuple[str, str], float],
    ) -> float:
        """Compute density (edge ratio) of a cluster.

        Density = actual_edges / possible_edges where possible_edges
        is n*(n-1)/2 for n members.

        Args:
            member_ids: List of record identifiers in the cluster.
            pair_scores: Mapping of (sorted pair) to similarity score.

        Returns:
            Density ratio (0.0 to 1.0).
        """
        n = len(member_ids)
        if n < 2:
            return 0.0

        possible_edges = n * (n - 1) / 2
        actual_edges = 0

        for i in range(n):
            for j in range(i + 1, n):
                pair_key = tuple(sorted([member_ids[i], member_ids[j]]))
                if pair_key in pair_scores:
                    actual_edges += 1

        return round(actual_edges / possible_edges, 6)

    def compute_cluster_diameter(
        self,
        member_ids: List[str],
        pair_scores: Dict[Tuple[str, str], float],
    ) -> float:
        """Compute diameter (max pairwise distance) of a cluster.

        Diameter = max(1 - similarity) across all compared pairs.

        Args:
            member_ids: List of record identifiers in the cluster.
            pair_scores: Mapping of (sorted pair) to similarity score.

        Returns:
            Maximum distance (0.0 to 1.0).
        """
        if len(member_ids) < 2:
            return 0.0

        max_distance = 0.0
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                pair_key = tuple(sorted([member_ids[i], member_ids[j]]))
                if pair_key in pair_scores:
                    distance = 1.0 - pair_scores[pair_key]
                    max_distance = max(max_distance, distance)

        return round(max_distance, 6)

    # ------------------------------------------------------------------
    # Public API - Representative selection
    # ------------------------------------------------------------------

    def select_representative(
        self,
        member_ids: List[str],
        records: Optional[Dict[str, Dict[str, Any]]] = None,
        pair_scores: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> str:
        """Select the representative record for a cluster.

        Selection priority:
        1. If records provided: most complete record (fewest None values)
        2. If pair_scores provided: record with highest average similarity
        3. Otherwise: first record (alphabetically)

        Args:
            member_ids: List of record identifiers in the cluster.
            records: Optional full records for completeness scoring.
            pair_scores: Optional pair scores for centrality scoring.

        Returns:
            Identifier of the selected representative record.
        """
        if not member_ids:
            return ""
        if len(member_ids) == 1:
            return member_ids[0]

        # Strategy 1: Most complete record
        if records:
            best_id = member_ids[0]
            best_completeness = -1
            for rid in member_ids:
                rec = records.get(rid, {})
                completeness = sum(
                    1 for v in rec.values() if v is not None and str(v).strip()
                )
                if completeness > best_completeness:
                    best_completeness = completeness
                    best_id = rid
            return best_id

        # Strategy 2: Highest average similarity
        if pair_scores:
            best_id = member_ids[0]
            best_avg = -1.0
            for rid in member_ids:
                scores_for_rid: List[float] = []
                for other_id in member_ids:
                    if other_id == rid:
                        continue
                    pair_key = tuple(sorted([rid, other_id]))
                    if pair_key in pair_scores:
                        scores_for_rid.append(pair_scores[pair_key])
                avg = (
                    sum(scores_for_rid) / len(scores_for_rid)
                    if scores_for_rid else 0.0
                )
                if avg > best_avg:
                    best_avg = avg
                    best_id = rid
            return best_id

        # Strategy 3: First alphabetically
        return sorted(member_ids)[0]

    # ------------------------------------------------------------------
    # Public API - Cluster manipulation
    # ------------------------------------------------------------------

    def split_cluster(
        self,
        cluster: DuplicateCluster,
        record_ids_group_a: List[str],
        record_ids_group_b: List[str],
        pair_scores: Optional[Dict[Tuple[str, str], float]] = None,
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[DuplicateCluster, DuplicateCluster]:
        """Split a cluster into two sub-clusters.

        Args:
            cluster: The cluster to split.
            record_ids_group_a: Record IDs for the first sub-cluster.
            record_ids_group_b: Record IDs for the second sub-cluster.
            pair_scores: Optional pair scores for quality computation.
            records: Optional full records for representative selection.

        Returns:
            Tuple of two new DuplicateCluster instances.

        Raises:
            ValueError: If groups don't partition the cluster.
        """
        all_ids = set(cluster.member_record_ids)
        group_a_set = set(record_ids_group_a)
        group_b_set = set(record_ids_group_b)

        if group_a_set & group_b_set:
            raise ValueError("Groups must not overlap")
        if group_a_set | group_b_set != all_ids:
            raise ValueError(
                "Groups must include all cluster members exactly once"
            )
        if len(record_ids_group_a) < 2 or len(record_ids_group_b) < 2:
            raise ValueError("Each sub-cluster must have at least 2 members")

        scores = pair_scores or {}

        cluster_a = self._build_cluster(
            record_ids_group_a, scores, records,
        )
        cluster_b = self._build_cluster(
            record_ids_group_b, scores, records,
        )

        logger.info(
            "Split cluster %s into two sub-clusters: %d and %d members",
            cluster.cluster_id, cluster_a.member_count, cluster_b.member_count,
        )
        return (cluster_a, cluster_b)

    def merge_clusters(
        self,
        cluster_a: DuplicateCluster,
        cluster_b: DuplicateCluster,
        pair_scores: Optional[Dict[Tuple[str, str], float]] = None,
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DuplicateCluster:
        """Merge two clusters into a single cluster.

        Args:
            cluster_a: First cluster.
            cluster_b: Second cluster.
            pair_scores: Optional pair scores for quality computation.
            records: Optional full records for representative selection.

        Returns:
            New merged DuplicateCluster.

        Raises:
            ValueError: If clusters share members.
        """
        set_a = set(cluster_a.member_record_ids)
        set_b = set(cluster_b.member_record_ids)

        if set_a & set_b:
            raise ValueError("Clusters must not share members")

        combined = list(set_a | set_b)
        scores = pair_scores or {}

        merged = self._build_cluster(combined, scores, records)

        logger.info(
            "Merged clusters %s and %s into %s (%d members)",
            cluster_a.cluster_id, cluster_b.cluster_id,
            merged.cluster_id, merged.member_count,
        )
        return merged

    # ------------------------------------------------------------------
    # Public API - Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics."""
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations
            return {
                "engine_name": "ClusterResolver",
                "invocations": self._invocations,
                "successes": self._successes,
                "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": (
                    self._last_invoked_at.isoformat()
                    if self._last_invoked_at else None
                ),
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_cluster(
        self,
        member_ids: List[str],
        pair_scores: Dict[Tuple[str, str], float],
        records: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DuplicateCluster:
        """Build a DuplicateCluster from member IDs and scores.

        Args:
            member_ids: List of record identifiers.
            pair_scores: Mapping of sorted pairs to similarity scores.
            records: Optional full records for representative selection.

        Returns:
            Populated DuplicateCluster instance.
        """
        sorted_ids = sorted(member_ids)
        quality = self.compute_cluster_quality(sorted_ids, pair_scores)
        density = self.compute_cluster_density(sorted_ids, pair_scores)
        diameter = self.compute_cluster_diameter(sorted_ids, pair_scores)
        representative = self.select_representative(
            sorted_ids, records, pair_scores,
        )

        provenance = _compute_provenance(
            "build_cluster",
            f"{len(sorted_ids)}:{quality}:{representative}",
        )

        return DuplicateCluster(
            cluster_id=str(uuid.uuid4()),
            member_record_ids=sorted_ids,
            representative_id=representative,
            cluster_quality=quality,
            density=density,
            diameter=diameter,
            member_count=len(sorted_ids),
            provenance_hash=provenance,
        )

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()
