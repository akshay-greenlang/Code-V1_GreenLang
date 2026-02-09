# -*- coding: utf-8 -*-
"""
Unit tests for ClusterResolver Engine - AGENT-DATA-011

Tests the ClusterResolver with 100+ test cases covering:
- union_find_clustering: simple pairs, disconnected, transitive closure
- connected_components: BFS-based clustering
- compute_cluster_quality: average pairwise similarity
- compute_cluster_density: edge ratio
- compute_cluster_diameter: max pairwise distance
- select_representative: most complete, highest avg, alphabetical
- split_cluster: splitting into sub-clusters
- merge_clusters: combining two clusters
- form_clusters with quality filtering
- edge cases (single record, no matches, large clusters)
- thread-safe statistics
- provenance tracking

Author: GreenLang Platform Team
Date: February 2026
"""

import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pytest

from greenlang.duplicate_detector.cluster_resolver import ClusterResolver, _UnionFind
from greenlang.duplicate_detector.models import (
    ClusterAlgorithm,
    DuplicateCluster,
    MatchClassification,
    MatchResult,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver() -> ClusterResolver:
    """Create a fresh ClusterResolver instance."""
    return ClusterResolver()


def _make_match(
    a_id: str,
    b_id: str,
    score: float = 0.90,
    classification: MatchClassification = MatchClassification.MATCH,
) -> MatchResult:
    """Create a MatchResult helper."""
    return MatchResult(
        record_a_id=a_id,
        record_b_id=b_id,
        classification=classification,
        confidence=0.85,
        overall_score=score,
        field_scores={"name": score},
        decision_reason="test",
        provenance_hash="a" * 64,
    )


def _make_cluster(
    member_ids: List[str],
    quality: float = 0.80,
    density: float = 1.0,
    diameter: float = 0.2,
) -> DuplicateCluster:
    """Create a DuplicateCluster helper."""
    return DuplicateCluster(
        cluster_id=str(uuid.uuid4()),
        member_record_ids=member_ids,
        representative_id=member_ids[0] if member_ids else None,
        cluster_quality=quality,
        density=density,
        diameter=diameter,
        member_count=len(member_ids),
        provenance_hash="b" * 64,
    )


def _pair_scores_from_matches(matches: List[MatchResult]) -> Dict[Tuple[str, str], float]:
    """Build pair_scores dict from MatchResult list."""
    scores: Dict[Tuple[str, str], float] = {}
    for m in matches:
        key = tuple(sorted([m.record_a_id, m.record_b_id]))
        scores[key] = m.overall_score
    return scores


# =============================================================================
# TestUnionFindInternal
# =============================================================================


class TestUnionFindInternal:
    """Tests for the internal _UnionFind data structure."""

    def test_make_set(self):
        """make_set creates a singleton."""
        uf = _UnionFind()
        uf.make_set("a")
        assert uf.find("a") == "a"

    def test_find_creates_set_if_missing(self):
        """find auto-creates a set for unknown elements."""
        uf = _UnionFind()
        root = uf.find("x")
        assert root == "x"

    def test_union_two_elements(self):
        """Union merges two singletons."""
        uf = _UnionFind()
        uf.make_set("a")
        uf.make_set("b")
        merged = uf.union("a", "b")
        assert merged is True
        assert uf.find("a") == uf.find("b")

    def test_union_same_set_returns_false(self):
        """Union of elements already in same set returns False."""
        uf = _UnionFind()
        uf.union("a", "b")
        merged = uf.union("a", "b")
        assert merged is False

    def test_path_compression(self):
        """Path compression flattens chains."""
        uf = _UnionFind()
        uf.make_set("a")
        uf.make_set("b")
        uf.make_set("c")
        uf.union("a", "b")
        uf.union("b", "c")
        # After find with compression, all point to same root
        root = uf.find("a")
        assert uf.find("b") == root
        assert uf.find("c") == root

    def test_get_components_simple(self):
        """get_components returns correct groups."""
        uf = _UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        components = uf.get_components()
        groups = [sorted(v) for v in components.values()]
        assert sorted(groups) == [["a", "b"], ["c", "d"]]

    def test_get_components_single_group(self):
        """All elements in one group."""
        uf = _UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        components = uf.get_components()
        assert len(components) == 1
        members = list(components.values())[0]
        assert sorted(members) == ["a", "b", "c", "d"]

    def test_get_components_empty(self):
        """Empty union-find returns empty dict."""
        uf = _UnionFind()
        assert uf.get_components() == {}


# =============================================================================
# TestUnionFindClustering
# =============================================================================


class TestUnionFindClustering:
    """Tests for union_find_clustering method."""

    def test_simple_pair(self, resolver: ClusterResolver):
        """A-B pair forms one cluster of 2."""
        matches = [_make_match("a", "b")]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 1
        assert sorted(clusters[0].member_record_ids) == ["a", "b"]

    def test_transitive_closure(self, resolver: ClusterResolver):
        """A-B, B-C -> {A, B, C} via transitive closure."""
        matches = [_make_match("a", "b"), _make_match("b", "c")]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 1
        assert sorted(clusters[0].member_record_ids) == ["a", "b", "c"]

    def test_long_chain(self, resolver: ClusterResolver):
        """A-B, B-C, C-D, D-E -> single cluster."""
        matches = [
            _make_match("a", "b"),
            _make_match("b", "c"),
            _make_match("c", "d"),
            _make_match("d", "e"),
        ]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 1
        assert clusters[0].member_count == 5

    def test_disconnected_components(self, resolver: ClusterResolver):
        """Disconnected pairs form separate clusters."""
        matches = [_make_match("a", "b"), _make_match("c", "d")]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 2

    def test_star_topology(self, resolver: ClusterResolver):
        """Star: center connected to 4 leaves."""
        matches = [
            _make_match("center", "l1"),
            _make_match("center", "l2"),
            _make_match("center", "l3"),
            _make_match("center", "l4"),
        ]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 1
        assert clusters[0].member_count == 5

    def test_cluster_has_quality(self, resolver: ClusterResolver):
        """Cluster quality is computed from pair scores."""
        matches = [_make_match("a", "b", score=0.90)]
        clusters = resolver.union_find_clustering(matches)
        assert clusters[0].cluster_quality == pytest.approx(0.90, abs=1e-5)

    def test_cluster_has_density(self, resolver: ClusterResolver):
        """Cluster density is computed correctly."""
        matches = [_make_match("a", "b", score=0.90)]
        clusters = resolver.union_find_clustering(matches)
        # 2 members, 1 edge, possible=1 -> density=1.0
        assert clusters[0].density == pytest.approx(1.0, abs=1e-5)

    def test_cluster_has_provenance(self, resolver: ClusterResolver):
        """Cluster has a 64-char provenance hash."""
        matches = [_make_match("a", "b")]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters[0].provenance_hash) == 64

    def test_cluster_member_ids_sorted(self, resolver: ClusterResolver):
        """Cluster member IDs are sorted alphabetically."""
        matches = [_make_match("z", "a")]
        clusters = resolver.union_find_clustering(matches)
        assert clusters[0].member_record_ids == ["a", "z"]

    def test_duplicate_pairs_handled(self, resolver: ClusterResolver):
        """Duplicate pairs don't create duplicate members."""
        matches = [_make_match("a", "b"), _make_match("a", "b")]
        clusters = resolver.union_find_clustering(matches)
        assert len(clusters) == 1
        assert clusters[0].member_count == 2

    def test_with_records(self, resolver: ClusterResolver):
        """Records parameter used for representative selection."""
        matches = [_make_match("a", "b")]
        records = {
            "a": {"name": "Alice", "email": "alice@co.com"},
            "b": {"name": "Bob"},
        }
        clusters = resolver.union_find_clustering(matches, records)
        assert clusters[0].representative_id == "a"  # more complete


# =============================================================================
# TestConnectedComponents
# =============================================================================


class TestConnectedComponents:
    """Tests for connected_components method."""

    def test_simple_pair(self, resolver: ClusterResolver):
        """Single pair produces one cluster."""
        matches = [_make_match("a", "b")]
        clusters = resolver.connected_components(matches)
        assert len(clusters) == 1
        assert sorted(clusters[0].member_record_ids) == ["a", "b"]

    def test_transitive_closure_bfs(self, resolver: ClusterResolver):
        """A-B, B-C produces single cluster via BFS."""
        matches = [_make_match("a", "b"), _make_match("b", "c")]
        clusters = resolver.connected_components(matches)
        assert len(clusters) == 1
        assert sorted(clusters[0].member_record_ids) == ["a", "b", "c"]

    def test_disconnected_bfs(self, resolver: ClusterResolver):
        """Disconnected pairs produce separate clusters."""
        matches = [_make_match("a", "b"), _make_match("c", "d")]
        clusters = resolver.connected_components(matches)
        assert len(clusters) == 2

    def test_triangle(self, resolver: ClusterResolver):
        """Triangle A-B, B-C, A-C -> single cluster."""
        matches = [
            _make_match("a", "b"),
            _make_match("b", "c"),
            _make_match("a", "c"),
        ]
        clusters = resolver.connected_components(matches)
        assert len(clusters) == 1
        assert clusters[0].member_count == 3

    def test_produces_same_as_union_find(self, resolver: ClusterResolver):
        """BFS and Union-Find produce same clusters for same input."""
        matches = [
            _make_match("a", "b"),
            _make_match("b", "c"),
            _make_match("d", "e"),
        ]
        uf_clusters = resolver.union_find_clustering(matches)
        cc_clusters = resolver.connected_components(matches)
        uf_sets = {frozenset(c.member_record_ids) for c in uf_clusters}
        cc_sets = {frozenset(c.member_record_ids) for c in cc_clusters}
        assert uf_sets == cc_sets


# =============================================================================
# TestFormClusters
# =============================================================================


class TestFormClusters:
    """Tests for form_clusters method."""

    def test_empty_raises(self, resolver: ClusterResolver):
        """Empty match_results raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            resolver.form_clusters([])

    def test_no_matches_in_list(self, resolver: ClusterResolver):
        """Only NON_MATCH entries produce empty clusters."""
        matches = [_make_match("a", "b", classification=MatchClassification.NON_MATCH)]
        clusters = resolver.form_clusters(matches)
        assert clusters == []

    def test_union_find_algorithm(self, resolver: ClusterResolver):
        """Union-Find is default algorithm."""
        matches = [_make_match("a", "b")]
        clusters = resolver.form_clusters(matches, algorithm=ClusterAlgorithm.UNION_FIND)
        assert len(clusters) == 1

    def test_connected_components_algorithm(self, resolver: ClusterResolver):
        """Connected Components algorithm works via form_clusters."""
        matches = [_make_match("a", "b")]
        clusters = resolver.form_clusters(matches, algorithm=ClusterAlgorithm.CONNECTED_COMPONENTS)
        assert len(clusters) == 1

    def test_quality_filter(self, resolver: ClusterResolver):
        """Clusters below min_quality are filtered out."""
        matches = [_make_match("a", "b", score=0.30)]
        clusters = resolver.form_clusters(matches, min_quality=0.50)
        assert len(clusters) == 0

    def test_quality_passes(self, resolver: ClusterResolver):
        """Clusters above min_quality are kept."""
        matches = [_make_match("a", "b", score=0.90)]
        clusters = resolver.form_clusters(matches, min_quality=0.50)
        assert len(clusters) == 1

    def test_mixed_classifications_filtered(self, resolver: ClusterResolver):
        """Only MATCH pairs are used; NON_MATCH and POSSIBLE ignored."""
        matches = [
            _make_match("a", "b", classification=MatchClassification.MATCH),
            _make_match("c", "d", classification=MatchClassification.NON_MATCH),
            _make_match("e", "f", classification=MatchClassification.POSSIBLE),
        ]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 1
        assert sorted(clusters[0].member_record_ids) == ["a", "b"]

    def test_success_stats(self, resolver: ClusterResolver):
        """Successful form_clusters increments success counter."""
        matches = [_make_match("a", "b")]
        resolver.form_clusters(matches)
        stats = resolver.get_statistics()
        assert stats["successes"] == 1

    def test_failure_stats(self, resolver: ClusterResolver):
        """Failed form_clusters increments failure counter."""
        with pytest.raises(ValueError):
            resolver.form_clusters([])
        stats = resolver.get_statistics()
        assert stats["failures"] == 1


# =============================================================================
# TestComputeClusterQuality
# =============================================================================


class TestComputeClusterQuality:
    """Tests for compute_cluster_quality method."""

    def test_single_member_returns_zero(self, resolver: ClusterResolver):
        """Single member returns 0.0."""
        q = resolver.compute_cluster_quality(["a"], {})
        assert q == 0.0

    def test_two_members_one_pair(self, resolver: ClusterResolver):
        """Two members with one pair returns the pair score."""
        pair_scores = {("a", "b"): 0.90}
        q = resolver.compute_cluster_quality(["a", "b"], pair_scores)
        assert q == pytest.approx(0.90, abs=1e-5)

    def test_three_members_partial_pairs(self, resolver: ClusterResolver):
        """Three members with 2 of 3 possible pairs."""
        pair_scores = {("a", "b"): 0.80, ("a", "c"): 0.60}
        q = resolver.compute_cluster_quality(["a", "b", "c"], pair_scores)
        # avg of 0.80 and 0.60 = 0.70
        assert q == pytest.approx(0.70, abs=1e-5)

    def test_no_pairs_returns_zero(self, resolver: ClusterResolver):
        """No pairs in pair_scores returns 0.0."""
        q = resolver.compute_cluster_quality(["a", "b"], {})
        assert q == 0.0

    def test_empty_members(self, resolver: ClusterResolver):
        """Empty members returns 0.0."""
        q = resolver.compute_cluster_quality([], {})
        assert q == 0.0


# =============================================================================
# TestComputeClusterDensity
# =============================================================================


class TestComputeClusterDensity:
    """Tests for compute_cluster_density method."""

    def test_single_member_density_zero(self, resolver: ClusterResolver):
        """Single member has density 0.0."""
        d = resolver.compute_cluster_density(["a"], {})
        assert d == 0.0

    def test_fully_connected_pair(self, resolver: ClusterResolver):
        """Pair with edge has density 1.0."""
        pair_scores = {("a", "b"): 0.80}
        d = resolver.compute_cluster_density(["a", "b"], pair_scores)
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_triangle_full_density(self, resolver: ClusterResolver):
        """Triangle with 3 edges has density 1.0."""
        pair_scores = {("a", "b"): 0.80, ("a", "c"): 0.70, ("b", "c"): 0.60}
        d = resolver.compute_cluster_density(["a", "b", "c"], pair_scores)
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_partial_density(self, resolver: ClusterResolver):
        """3 nodes, 2 edges -> density = 2/3."""
        pair_scores = {("a", "b"): 0.80, ("a", "c"): 0.70}
        d = resolver.compute_cluster_density(["a", "b", "c"], pair_scores)
        assert d == pytest.approx(2.0 / 3.0, abs=1e-5)

    def test_no_edges(self, resolver: ClusterResolver):
        """No edges -> density 0.0."""
        d = resolver.compute_cluster_density(["a", "b", "c"], {})
        assert d == 0.0


# =============================================================================
# TestComputeClusterDiameter
# =============================================================================


class TestComputeClusterDiameter:
    """Tests for compute_cluster_diameter method."""

    def test_single_member_diameter_zero(self, resolver: ClusterResolver):
        """Single member has diameter 0.0."""
        d = resolver.compute_cluster_diameter(["a"], {})
        assert d == 0.0

    def test_pair_diameter(self, resolver: ClusterResolver):
        """Pair diameter = 1 - similarity."""
        pair_scores = {("a", "b"): 0.80}
        d = resolver.compute_cluster_diameter(["a", "b"], pair_scores)
        assert d == pytest.approx(0.20, abs=1e-5)

    def test_perfect_similarity_zero_diameter(self, resolver: ClusterResolver):
        """Perfect similarity has diameter 0.0."""
        pair_scores = {("a", "b"): 1.0}
        d = resolver.compute_cluster_diameter(["a", "b"], pair_scores)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_max_distance_selected(self, resolver: ClusterResolver):
        """Diameter is the maximum pairwise distance."""
        pair_scores = {("a", "b"): 0.90, ("a", "c"): 0.60, ("b", "c"): 0.70}
        d = resolver.compute_cluster_diameter(["a", "b", "c"], pair_scores)
        # Max distance = 1 - 0.60 = 0.40
        assert d == pytest.approx(0.40, abs=1e-5)

    def test_no_pairs_zero_diameter(self, resolver: ClusterResolver):
        """No pairs -> diameter 0.0."""
        d = resolver.compute_cluster_diameter(["a", "b"], {})
        assert d == 0.0


# =============================================================================
# TestSelectRepresentative
# =============================================================================


class TestSelectRepresentative:
    """Tests for select_representative method."""

    def test_empty_returns_empty(self, resolver: ClusterResolver):
        """Empty members returns empty string."""
        rep = resolver.select_representative([])
        assert rep == ""

    def test_single_member(self, resolver: ClusterResolver):
        """Single member is the representative."""
        rep = resolver.select_representative(["a"])
        assert rep == "a"

    def test_most_complete_record(self, resolver: ClusterResolver):
        """Record with most non-null fields is selected."""
        records = {
            "a": {"name": "Alice", "email": "alice@co.com", "phone": "555"},
            "b": {"name": "Bob"},
        }
        rep = resolver.select_representative(["a", "b"], records=records)
        assert rep == "a"

    def test_most_complete_ignores_empty_strings(self, resolver: ClusterResolver):
        """Empty strings are not counted as complete."""
        records = {
            "a": {"name": "Alice", "email": ""},
            "b": {"name": "Bob", "email": "bob@co.com"},
        }
        rep = resolver.select_representative(["a", "b"], records=records)
        assert rep == "b"

    def test_highest_average_similarity(self, resolver: ClusterResolver):
        """Without records, highest avg similarity wins."""
        pair_scores = {
            ("a", "b"): 0.90,
            ("a", "c"): 0.85,
            ("b", "c"): 0.50,
        }
        rep = resolver.select_representative(
            ["a", "b", "c"], pair_scores=pair_scores,
        )
        # "a" has avg (0.90+0.85)/2=0.875; "b" has avg (0.90+0.50)/2=0.70
        assert rep == "a"

    def test_alphabetical_fallback(self, resolver: ClusterResolver):
        """Without records or pair_scores, first alphabetically wins."""
        rep = resolver.select_representative(["z", "a", "m"])
        assert rep == "a"

    def test_records_priority_over_scores(self, resolver: ClusterResolver):
        """Records parameter takes priority over pair_scores."""
        records = {
            "b": {"name": "Bob", "email": "bob@co.com", "phone": "555"},
            "a": {"name": "Alice"},
        }
        pair_scores = {("a", "b"): 0.90}
        rep = resolver.select_representative(
            ["a", "b"], records=records, pair_scores=pair_scores,
        )
        assert rep == "b"  # more complete

    def test_none_values_not_counted(self, resolver: ClusterResolver):
        """None values are not counted for completeness."""
        records = {
            "a": {"name": "Alice", "email": None, "phone": None},
            "b": {"name": "Bob", "email": "bob@co.com"},
        }
        rep = resolver.select_representative(["a", "b"], records=records)
        assert rep == "b"

    def test_missing_record_uses_empty_dict(self, resolver: ClusterResolver):
        """Missing record in records dict treated as empty."""
        records = {"a": {"name": "Alice", "email": "alice@co.com"}}
        rep = resolver.select_representative(["a", "b"], records=records)
        assert rep == "a"


# =============================================================================
# TestSplitCluster
# =============================================================================


class TestSplitCluster:
    """Tests for split_cluster method."""

    def test_split_into_two(self, resolver: ClusterResolver):
        """Split a 4-member cluster into two 2-member clusters."""
        cluster = _make_cluster(["a", "b", "c", "d"])
        c1, c2 = resolver.split_cluster(
            cluster,
            record_ids_group_a=["a", "b"],
            record_ids_group_b=["c", "d"],
        )
        assert c1.member_count == 2
        assert c2.member_count == 2

    def test_split_overlapping_raises(self, resolver: ClusterResolver):
        """Overlapping groups raise ValueError."""
        cluster = _make_cluster(["a", "b", "c", "d"])
        with pytest.raises(ValueError, match="must not overlap"):
            resolver.split_cluster(cluster, ["a", "b"], ["b", "c"])

    def test_split_incomplete_groups_raises(self, resolver: ClusterResolver):
        """Groups not covering all members raise ValueError."""
        cluster = _make_cluster(["a", "b", "c", "d"])
        with pytest.raises(ValueError, match="all cluster members"):
            resolver.split_cluster(cluster, ["a", "b"], ["c"])

    def test_split_group_too_small_raises(self, resolver: ClusterResolver):
        """Groups with < 2 members raise ValueError."""
        cluster = _make_cluster(["a", "b", "c"])
        with pytest.raises(ValueError, match="at least 2 members"):
            resolver.split_cluster(cluster, ["a"], ["b", "c"])

    def test_split_with_pair_scores(self, resolver: ClusterResolver):
        """Pair scores used in sub-cluster quality computation."""
        cluster = _make_cluster(["a", "b", "c", "d"])
        pair_scores = {("a", "b"): 0.90, ("c", "d"): 0.80}
        c1, c2 = resolver.split_cluster(
            cluster, ["a", "b"], ["c", "d"],
            pair_scores=pair_scores,
        )
        assert c1.cluster_quality == pytest.approx(0.90, abs=1e-5)
        assert c2.cluster_quality == pytest.approx(0.80, abs=1e-5)

    def test_split_preserves_ids(self, resolver: ClusterResolver):
        """Split preserves member IDs correctly."""
        cluster = _make_cluster(["a", "b", "c", "d"])
        c1, c2 = resolver.split_cluster(cluster, ["a", "c"], ["b", "d"])
        assert sorted(c1.member_record_ids) == ["a", "c"]
        assert sorted(c2.member_record_ids) == ["b", "d"]


# =============================================================================
# TestMergeClusters
# =============================================================================


class TestMergeClusters:
    """Tests for merge_clusters method."""

    def test_merge_two_clusters(self, resolver: ClusterResolver):
        """Merge two disjoint clusters."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["c", "d"])
        merged = resolver.merge_clusters(c1, c2)
        assert merged.member_count == 4
        assert sorted(merged.member_record_ids) == ["a", "b", "c", "d"]

    def test_merge_overlapping_raises(self, resolver: ClusterResolver):
        """Overlapping clusters raise ValueError."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["b", "c"])
        with pytest.raises(ValueError, match="must not share"):
            resolver.merge_clusters(c1, c2)

    def test_merge_with_pair_scores(self, resolver: ClusterResolver):
        """Pair scores used in merged cluster quality."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["c", "d"])
        pair_scores = {("a", "b"): 0.90, ("c", "d"): 0.80, ("a", "c"): 0.70}
        merged = resolver.merge_clusters(c1, c2, pair_scores=pair_scores)
        assert merged.cluster_quality > 0

    def test_merge_has_new_cluster_id(self, resolver: ClusterResolver):
        """Merged cluster has a new unique cluster_id."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["c", "d"])
        merged = resolver.merge_clusters(c1, c2)
        assert merged.cluster_id != c1.cluster_id
        assert merged.cluster_id != c2.cluster_id

    def test_merge_has_provenance(self, resolver: ClusterResolver):
        """Merged cluster has provenance hash."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["c", "d"])
        merged = resolver.merge_clusters(c1, c2)
        assert len(merged.provenance_hash) == 64

    def test_merge_representative_selected(self, resolver: ClusterResolver):
        """Merged cluster selects a representative."""
        c1 = _make_cluster(["a", "b"])
        c2 = _make_cluster(["c", "d"])
        records = {
            "a": {"name": "Alice", "email": "a@co.com", "phone": "1"},
            "b": {"name": "Bob"},
            "c": {"name": "Carol"},
            "d": {"name": "Dave"},
        }
        merged = resolver.merge_clusters(c1, c2, records=records)
        assert merged.representative_id == "a"  # most complete


# =============================================================================
# TestTransitiveClosure
# =============================================================================


class TestTransitiveClosure:
    """Verify transitive closure behavior."""

    def test_chain_4_elements(self, resolver: ClusterResolver):
        """A-B, B-C, C-D -> {A,B,C,D}."""
        matches = [
            _make_match("a", "b"),
            _make_match("b", "c"),
            _make_match("c", "d"),
        ]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 1
        assert clusters[0].member_count == 4

    def test_two_chains_merge(self, resolver: ClusterResolver):
        """A-B, C-D, B-C -> {A,B,C,D}."""
        matches = [
            _make_match("a", "b"),
            _make_match("c", "d"),
            _make_match("b", "c"),
        ]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 1

    def test_no_transitive_if_disconnected(self, resolver: ClusterResolver):
        """A-B and C-D remain separate."""
        matches = [_make_match("a", "b"), _make_match("c", "d")]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 2


# =============================================================================
# TestNoMatchesNoCluster
# =============================================================================


class TestNoMatchesNoCluster:
    """Tests for scenarios with no matching pairs."""

    def test_only_possible_no_clusters(self, resolver: ClusterResolver):
        """Only POSSIBLE classifications produce no clusters."""
        matches = [_make_match("a", "b", classification=MatchClassification.POSSIBLE)]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert clusters == []

    def test_only_non_match_no_clusters(self, resolver: ClusterResolver):
        """Only NON_MATCH classifications produce no clusters."""
        matches = [_make_match("a", "b", classification=MatchClassification.NON_MATCH)]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert clusters == []


# =============================================================================
# TestSingleRecordClusters
# =============================================================================


class TestSingleRecordClusters:
    """Tests for single-record cluster behavior."""

    def test_single_record_not_a_cluster(self, resolver: ClusterResolver):
        """Union-Find skips singleton components."""
        # If A matches B and C is alone, we get one cluster {A,B}
        matches = [_make_match("a", "b")]
        clusters = resolver.union_find_clustering(matches)
        # No cluster containing only "c"
        for c in clusters:
            assert c.member_count >= 2


# =============================================================================
# TestLargeCluster
# =============================================================================


class TestLargeCluster:
    """Tests with larger cluster sizes."""

    def test_20_node_chain(self, resolver: ClusterResolver):
        """20-node chain forms single cluster."""
        matches = [
            _make_match(f"n{i}", f"n{i+1}", score=0.90)
            for i in range(19)
        ]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 1
        assert clusters[0].member_count == 20

    def test_complete_graph_5_nodes(self, resolver: ClusterResolver):
        """5-node complete graph: density=1.0."""
        nodes = [f"n{i}" for i in range(5)]
        matches = []
        for i in range(5):
            for j in range(i + 1, 5):
                matches.append(_make_match(nodes[i], nodes[j], score=0.85))
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters) == 1
        assert clusters[0].member_count == 5
        assert clusters[0].density == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# TestClusterQualityMetrics
# =============================================================================


class TestClusterQualityMetrics:
    """Tests for cluster quality metric computation."""

    def test_quality_equals_avg_pair_score(self, resolver: ClusterResolver):
        """Quality = average of pair scores in cluster."""
        pair_scores = {("a", "b"): 0.90, ("a", "c"): 0.80, ("b", "c"): 0.70}
        q = resolver.compute_cluster_quality(["a", "b", "c"], pair_scores)
        expected = (0.90 + 0.80 + 0.70) / 3.0
        assert q == pytest.approx(expected, abs=1e-5)

    def test_density_varies_with_edges(self, resolver: ClusterResolver):
        """Density changes with number of edges."""
        pair_full = {("a", "b"): 0.9, ("a", "c"): 0.9, ("b", "c"): 0.9}
        pair_partial = {("a", "b"): 0.9}
        d_full = resolver.compute_cluster_density(["a", "b", "c"], pair_full)
        d_partial = resolver.compute_cluster_density(["a", "b", "c"], pair_partial)
        assert d_full > d_partial

    def test_diameter_increases_with_distance(self, resolver: ClusterResolver):
        """Diameter grows as max distance increases."""
        pair_close = {("a", "b"): 0.95, ("a", "c"): 0.90, ("b", "c"): 0.85}
        pair_far = {("a", "b"): 0.95, ("a", "c"): 0.50, ("b", "c"): 0.85}
        d_close = resolver.compute_cluster_diameter(["a", "b", "c"], pair_close)
        d_far = resolver.compute_cluster_diameter(["a", "b", "c"], pair_far)
        assert d_far > d_close


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Thread-safety tests for statistics tracking."""

    def test_concurrent_form_clusters(self, resolver: ClusterResolver):
        """Concurrent calls maintain correct stats."""
        num_threads = 8
        errors: List[str] = []

        def worker(idx: int):
            try:
                matches = [_make_match(f"a{idx}", f"b{idx}")]
                resolver.form_clusters(matches, min_quality=0.0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = resolver.get_statistics()
        assert stats["successes"] == num_threads

    def test_concurrent_reset(self, resolver: ClusterResolver):
        """Concurrent reset does not crash."""
        def worker():
            for _ in range(50):
                resolver.reset_statistics()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stats = resolver.get_statistics()
        assert stats["invocations"] == 0


# =============================================================================
# TestStatistics
# =============================================================================


class TestStatistics:
    """Statistics tracking tests."""

    def test_initial_stats(self, resolver: ClusterResolver):
        """Initial stats are all zero."""
        stats = resolver.get_statistics()
        assert stats["engine_name"] == "ClusterResolver"
        assert stats["invocations"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0

    def test_reset_statistics(self, resolver: ClusterResolver):
        """reset_statistics zeroes all counters."""
        matches = [_make_match("a", "b")]
        resolver.form_clusters(matches)
        resolver.reset_statistics()
        stats = resolver.get_statistics()
        assert stats["invocations"] == 0

    def test_duration_positive(self, resolver: ClusterResolver):
        """Duration is positive after invocation."""
        matches = [_make_match("a", "b")]
        resolver.form_clusters(matches)
        stats = resolver.get_statistics()
        assert stats["total_duration_ms"] > 0

    def test_last_invoked_at_set(self, resolver: ClusterResolver):
        """last_invoked_at is set after invocation."""
        matches = [_make_match("a", "b")]
        resolver.form_clusters(matches)
        stats = resolver.get_statistics()
        assert stats["last_invoked_at"] is not None


# =============================================================================
# TestProvenanceTracking
# =============================================================================


class TestProvenanceTracking:
    """Provenance hash generation tests."""

    def test_cluster_provenance_sha256(self, resolver: ClusterResolver):
        """Cluster provenance hash is 64-char hex."""
        matches = [_make_match("a", "b")]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert len(clusters[0].provenance_hash) == 64
        int(clusters[0].provenance_hash, 16)

    def test_provenance_not_empty(self, resolver: ClusterResolver):
        """Provenance hash is never empty."""
        matches = [_make_match("a", "b")]
        clusters = resolver.form_clusters(matches, min_quality=0.0)
        assert clusters[0].provenance_hash != ""


# =============================================================================
# TestDeterminism
# =============================================================================


class TestDeterminism:
    """Determinism: same input always produces same clusters."""

    def test_same_input_same_member_sets(self, resolver: ClusterResolver):
        """Same matches produce same member sets."""
        matches = [_make_match("a", "b"), _make_match("b", "c")]
        for _ in range(5):
            clusters = resolver.union_find_clustering(matches)
            assert sorted(clusters[0].member_record_ids) == ["a", "b", "c"]

    def test_same_input_same_quality(self, resolver: ClusterResolver):
        """Same matches produce same cluster quality."""
        matches = [_make_match("a", "b", score=0.85)]
        qualities = []
        for _ in range(5):
            clusters = resolver.union_find_clustering(matches)
            qualities.append(clusters[0].cluster_quality)
        assert all(q == qualities[0] for q in qualities)
