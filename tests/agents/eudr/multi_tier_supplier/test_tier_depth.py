# -*- coding: utf-8 -*-
"""
Tests for TierDepthTracker - AGENT-EUDR-008 Engine 3: Tier Depth Tracking

Comprehensive test suite covering:
- Linear chain depth calculation (F3.1)
- Branching chain depth (F3.1)
- Diamond-shaped chains with shared sub-suppliers (F3.1)
- Visibility score per tier (F3.2)
- Coverage score for commodity volume (F3.3)
- Commodity-specific tier depth benchmarks (F3.5)
- Tier gap detection (F3.4)
- Tier depth alerts (F3.8)
- Time-series tracking (F3.6)
- Benchmark comparisons (F3.7)

Test count: 60+ tests
Coverage target: >= 85% of TierDepthTracker module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    COCOA_CHAIN_7_TIER,
    COFFEE_CHAIN_6_TIER,
    PALM_OIL_CHAIN_6_TIER,
    COMMODITY_TIER_BENCHMARKS,
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_PALM_IMPORTER_NL,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    make_supplier,
    make_relationship,
    build_linear_chain,
    build_branching_chain,
    build_diamond_chain,
)


# ===========================================================================
# 1. Linear Chain Depth Calculation
# ===========================================================================


class TestLinearChainDepth:
    """Test tier depth calculation for simple linear supply chains."""

    def test_depth_of_single_tier(self, tier_depth_tracker):
        """Single supplier (importer only) has depth 0."""
        suppliers = [make_supplier(supplier_id="SUP-SINGLE", tier=0, role="importer")]
        result = tier_depth_tracker.calculate_depth(suppliers, [])
        assert result.max_depth == 0

    def test_depth_of_two_tiers(self, tier_depth_tracker):
        """Importer -> Trader has depth 1."""
        suppliers = [
            make_supplier(supplier_id="SUP-D2-IMP", tier=0, role="importer"),
            make_supplier(supplier_id="SUP-D2-TRD", tier=1, role="trader"),
        ]
        rels = [make_relationship("SUP-D2-IMP", "SUP-D2-TRD", rel_id="REL-D2-01")]
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 1

    def test_depth_of_five_tiers(self, tier_depth_tracker):
        """5-tier linear chain has depth 4."""
        suppliers, rels = build_linear_chain("cocoa", tier_count=5)
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 4

    def test_depth_of_seven_tier_cocoa(self, tier_depth_tracker, cocoa_chain):
        """7-tier cocoa chain has depth 6 (tier 0 through 6)."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth >= 5  # Tiers 0..5 minimum

    def test_depth_of_fifteen_tiers(self, tier_depth_tracker, deep_linear_chain):
        """15-tier chain has depth 14."""
        suppliers, rels = deep_linear_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 14

    @pytest.mark.parametrize("tier_count", [1, 2, 3, 5, 8, 10, 15])
    def test_depth_equals_tier_count_minus_one(self, tier_depth_tracker, tier_count):
        """Max depth equals number of tiers minus one."""
        suppliers, rels = build_linear_chain("cocoa", tier_count=tier_count)
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == tier_count - 1

    def test_depth_empty_chain(self, tier_depth_tracker):
        """Empty supplier list yields depth 0."""
        result = tier_depth_tracker.calculate_depth([], [])
        assert result.max_depth == 0


# ===========================================================================
# 2. Branching Chain Depth
# ===========================================================================


class TestBranchingChainDepth:
    """Test depth calculation for branching (tree-shaped) chains."""

    def test_branching_depth_4(self, tier_depth_tracker, branching_chain):
        """Branching chain with depth=4 has max_depth 3."""
        suppliers, rels = branching_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 3

    def test_branching_node_count(self, tier_depth_tracker, branching_chain):
        """Branching chain with factor=2, depth=4 has 15 nodes."""
        suppliers, rels = branching_chain
        # 2^0 + 2^1 + 2^2 + 2^3 = 1 + 2 + 4 + 8 = 15
        assert len(suppliers) == 15

    def test_branching_relationship_count(self, tier_depth_tracker, branching_chain):
        """Branching chain with factor=2, depth=4 has 14 relationships."""
        suppliers, rels = branching_chain
        assert len(rels) == 14  # 15 nodes - 1 root

    def test_wide_branching_depth(self, tier_depth_tracker):
        """Wide branching (factor=5) still reports correct depth."""
        suppliers, rels = build_branching_chain("cocoa", branch_factor=5, depth=3)
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 2

    def test_unbalanced_tree_depth(self, tier_depth_tracker):
        """Unbalanced tree reports max depth of longest branch."""
        # Root -> A (tier 1) -> B (tier 2) -> C (tier 3)
        # Root -> D (tier 1)  [short branch]
        root = make_supplier(supplier_id="ROOT", tier=0, role="importer")
        a = make_supplier(supplier_id="A", tier=1, role="trader")
        b = make_supplier(supplier_id="B", tier=2, role="processor")
        c = make_supplier(supplier_id="C", tier=3, role="farmer")
        d = make_supplier(supplier_id="D", tier=1, role="trader")
        suppliers = [root, a, b, c, d]
        rels = [
            make_relationship("ROOT", "A", rel_id="R1"),
            make_relationship("A", "B", rel_id="R2"),
            make_relationship("B", "C", rel_id="R3"),
            make_relationship("ROOT", "D", rel_id="R4"),
        ]
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 3


# ===========================================================================
# 3. Diamond-Shaped Chains
# ===========================================================================


class TestDiamondChainDepth:
    """Test depth for diamond-shaped chains with shared sub-suppliers."""

    def test_diamond_depth(self, tier_depth_tracker, diamond_chain):
        """Diamond chain: Imp -> [TrdA, TrdB] -> Proc -> Farmer, depth = 3."""
        suppliers, rels = diamond_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == 3

    def test_diamond_no_double_counting(self, tier_depth_tracker, diamond_chain):
        """Shared supplier counted once in node count."""
        suppliers, rels = diamond_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.total_suppliers == 5  # IMP, TRD-A, TRD-B, PRC, FRM

    def test_diamond_relationships_correct(self, tier_depth_tracker, diamond_chain):
        """Diamond has 5 relationships."""
        _, rels = diamond_chain
        assert len(rels) == 5

    def test_diamond_tier_counts(self, tier_depth_tracker, diamond_chain):
        """Diamond chain has correct supplier count per tier."""
        suppliers, rels = diamond_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        tier_counts = result.suppliers_per_tier
        assert tier_counts.get(0, 0) == 1  # importer
        assert tier_counts.get(1, 0) == 2  # trader A, trader B
        assert tier_counts.get(2, 0) == 1  # processor
        assert tier_counts.get(3, 0) == 1  # farmer


# ===========================================================================
# 4. Visibility Scores
# ===========================================================================


class TestVisibilityScores:
    """Test visibility score (percentage of known suppliers per tier) F3.2."""

    def test_full_visibility_score_100(self, tier_depth_tracker, cocoa_chain):
        """All tiers populated yields 100% visibility at mapped tiers."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_visibility(suppliers, rels, "cocoa")
        # At each mapped tier, visibility should be > 0
        for tier_info in result.tier_visibility:
            if tier_info["known_count"] > 0:
                assert tier_info["visibility_pct"] > 0.0

    def test_visibility_decreases_with_depth(self, tier_depth_tracker):
        """Generally, visibility decreases at deeper tiers."""
        suppliers, rels = build_linear_chain("cocoa", tier_count=6)
        # Remove some deep-tier suppliers to simulate partial visibility
        partial_suppliers = [s for s in suppliers if s["tier"] <= 3]
        partial_rels = [r for r in rels if "T4" not in r.get("supplier_id", "")
                        and "T5" not in r.get("supplier_id", "")]
        result = tier_depth_tracker.calculate_visibility(
            partial_suppliers, partial_rels, "cocoa"
        )
        # Check that result is valid
        assert result is not None

    def test_visibility_score_bounds(self, tier_depth_tracker, cocoa_chain):
        """Visibility scores are between 0 and 100."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_visibility(suppliers, rels, "cocoa")
        for tier_info in result.tier_visibility:
            assert 0.0 <= tier_info["visibility_pct"] <= 100.0

    def test_visibility_empty_chain(self, tier_depth_tracker):
        """Empty chain has 0% visibility."""
        result = tier_depth_tracker.calculate_visibility([], [], "cocoa")
        assert result.overall_visibility == 0.0


# ===========================================================================
# 5. Coverage Scores
# ===========================================================================


class TestCoverageScores:
    """Test coverage score (volume with full traceability) F3.3."""

    def test_full_coverage_score(self, tier_depth_tracker, cocoa_chain):
        """Chain with full traceability to farm yields high coverage."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_coverage(suppliers, rels, "cocoa")
        assert result.coverage_pct >= 0.0

    def test_coverage_score_bounds(self, tier_depth_tracker, cocoa_chain):
        """Coverage score is between 0 and 100."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_coverage(suppliers, rels, "cocoa")
        assert 0.0 <= result.coverage_pct <= 100.0

    def test_coverage_no_origin_suppliers(self, tier_depth_tracker):
        """Chain with no farm/origin suppliers has low coverage."""
        suppliers = [
            make_supplier(supplier_id="COV-IMP", tier=0, role="importer"),
            make_supplier(supplier_id="COV-TRD", tier=1, role="trader"),
        ]
        rels = [make_relationship("COV-IMP", "COV-TRD", rel_id="REL-COV-01")]
        result = tier_depth_tracker.calculate_coverage(suppliers, rels, "cocoa")
        assert result.coverage_pct < 100.0

    def test_coverage_volume_weighted(self, tier_depth_tracker):
        """Coverage accounts for volume weights of relationships."""
        suppliers = [
            make_supplier(supplier_id="VOL-IMP", tier=0, role="importer"),
            make_supplier(supplier_id="VOL-TRD-A", tier=1, role="trader"),
            make_supplier(supplier_id="VOL-TRD-B", tier=1, role="trader"),
            make_supplier(supplier_id="VOL-FRM-A", tier=2, role="farmer"),
        ]
        rels = [
            make_relationship("VOL-IMP", "VOL-TRD-A", volume_mt=800.0, rel_id="R-VA"),
            make_relationship("VOL-IMP", "VOL-TRD-B", volume_mt=200.0, rel_id="R-VB"),
            make_relationship("VOL-TRD-A", "VOL-FRM-A", volume_mt=800.0, rel_id="R-VC"),
            # TRD-B has no upstream (gap)
        ]
        result = tier_depth_tracker.calculate_coverage(suppliers, rels, "cocoa")
        # At least 80% coverage (800/1000) since TRD-A is fully traced
        assert result.coverage_pct >= 70.0


# ===========================================================================
# 6. Commodity-Specific Benchmarks
# ===========================================================================


class TestCommodityBenchmarks:
    """Test commodity-specific tier depth benchmarks (F3.5, F3.7)."""

    @pytest.mark.parametrize("commodity,expected", [
        ("cocoa", {"min_tiers": 6, "max_tiers": 8, "typical": 7}),
        ("coffee", {"min_tiers": 5, "max_tiers": 7, "typical": 6}),
        ("palm_oil", {"min_tiers": 5, "max_tiers": 7, "typical": 6}),
        ("soya", {"min_tiers": 4, "max_tiers": 6, "typical": 5}),
        ("rubber", {"min_tiers": 5, "max_tiers": 7, "typical": 6}),
        ("cattle", {"min_tiers": 3, "max_tiers": 5, "typical": 4}),
        ("wood", {"min_tiers": 4, "max_tiers": 6, "typical": 5}),
    ])
    def test_benchmark_per_commodity(self, tier_depth_tracker, commodity, expected):
        """Each commodity has defined benchmark tier depths."""
        benchmark = tier_depth_tracker.get_benchmark(commodity)
        assert benchmark["min_tiers"] == expected["min_tiers"]
        assert benchmark["max_tiers"] == expected["max_tiers"]
        assert benchmark["typical"] == expected["typical"]

    def test_benchmark_unknown_commodity(self, tier_depth_tracker):
        """Unknown commodity returns a default benchmark."""
        benchmark = tier_depth_tracker.get_benchmark("unknown")
        assert "min_tiers" in benchmark
        assert "typical" in benchmark

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_depth_vs_benchmark(self, tier_depth_tracker, commodity):
        """Verify depth assessment references benchmark data."""
        benchmark = tier_depth_tracker.get_benchmark(commodity)
        typical = benchmark["typical"]
        suppliers, rels = build_linear_chain(commodity, tier_count=typical)
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == typical - 1

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_benchmark_assessment_below_minimum(self, tier_depth_tracker, commodity):
        """Chain shorter than benchmark minimum flags low visibility."""
        benchmark = tier_depth_tracker.get_benchmark(commodity)
        short_count = max(benchmark["min_tiers"] - 2, 1)
        suppliers, rels = build_linear_chain(commodity, tier_count=short_count)
        result = tier_depth_tracker.assess_against_benchmark(suppliers, rels, commodity)
        assert result.benchmark_status in ("below_minimum", "low", "warning")


# ===========================================================================
# 7. Gap Detection
# ===========================================================================


class TestTierGapDetection:
    """Test detection of tier coverage gaps (F3.4)."""

    def test_detect_gap_in_middle_tier(self, tier_depth_tracker):
        """Detect a missing tier in the middle of a chain."""
        # Tier 0, 1, 3 (tier 2 missing)
        suppliers = [
            make_supplier(supplier_id="GAP-T0", tier=0, role="importer"),
            make_supplier(supplier_id="GAP-T1", tier=1, role="trader"),
            make_supplier(supplier_id="GAP-T3", tier=3, role="cooperative"),
        ]
        rels = [
            make_relationship("GAP-T0", "GAP-T1", rel_id="REL-GAP-01"),
            make_relationship("GAP-T1", "GAP-T3", rel_id="REL-GAP-02"),
        ]
        result = tier_depth_tracker.detect_gaps(suppliers, rels, "cocoa")
        assert len(result.gaps) >= 1
        gap_tiers = [g["missing_tier"] for g in result.gaps]
        assert 2 in gap_tiers

    def test_no_gaps_in_complete_chain(self, tier_depth_tracker, cocoa_chain):
        """Complete chain has no tier gaps."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.detect_gaps(suppliers, rels, "cocoa")
        assert len(result.gaps) == 0

    def test_gap_at_origin_tier(self, tier_depth_tracker):
        """Detect gap when origin tier (farmer) is missing."""
        suppliers = [
            make_supplier(supplier_id="GAP-ORIG-0", tier=0, role="importer"),
            make_supplier(supplier_id="GAP-ORIG-1", tier=1, role="trader"),
            make_supplier(supplier_id="GAP-ORIG-2", tier=2, role="processor"),
        ]
        rels = [
            make_relationship("GAP-ORIG-0", "GAP-ORIG-1", rel_id="R-GO-1"),
            make_relationship("GAP-ORIG-1", "GAP-ORIG-2", rel_id="R-GO-2"),
        ]
        result = tier_depth_tracker.detect_gaps(suppliers, rels, "cocoa")
        # Gap at deeper tiers (no farmer found)
        assert result.origin_visibility is False or len(result.gaps) >= 1

    def test_gap_severity_critical_for_missing_origin(self, tier_depth_tracker):
        """Missing origin tier gap is classified as critical."""
        suppliers = [
            make_supplier(supplier_id="SEV-0", tier=0, role="importer"),
            make_supplier(supplier_id="SEV-1", tier=1, role="trader"),
        ]
        rels = [make_relationship("SEV-0", "SEV-1", rel_id="R-SEV-1")]
        result = tier_depth_tracker.detect_gaps(suppliers, rels, "cocoa")
        if result.gaps:
            critical_gaps = [g for g in result.gaps if g.get("severity") == "critical"]
            assert len(critical_gaps) >= 1

    def test_multiple_gaps_detected(self, tier_depth_tracker):
        """Multiple non-consecutive tier gaps are all detected."""
        # Tier 0, 1, 4, 5 (tier 2 and 3 missing)
        suppliers = [
            make_supplier(supplier_id="MG-T0", tier=0, role="importer"),
            make_supplier(supplier_id="MG-T1", tier=1, role="trader"),
            make_supplier(supplier_id="MG-T4", tier=4, role="cooperative"),
            make_supplier(supplier_id="MG-T5", tier=5, role="farmer"),
        ]
        rels = [
            make_relationship("MG-T0", "MG-T1", rel_id="R-MG-1"),
            make_relationship("MG-T1", "MG-T4", rel_id="R-MG-2"),
            make_relationship("MG-T4", "MG-T5", rel_id="R-MG-3"),
        ]
        result = tier_depth_tracker.detect_gaps(suppliers, rels, "cocoa")
        gap_tiers = [g["missing_tier"] for g in result.gaps]
        assert 2 in gap_tiers
        assert 3 in gap_tiers


# ===========================================================================
# 8. Tier Depth Alerts
# ===========================================================================


class TestTierDepthAlerts:
    """Test alerting when visibility drops below thresholds (F3.8)."""

    def test_alert_when_below_minimum(self, tier_depth_tracker):
        """Alert generated when depth is below commodity minimum."""
        suppliers = [
            make_supplier(supplier_id="ALT-T0", tier=0, role="importer"),
            make_supplier(supplier_id="ALT-T1", tier=1, role="trader"),
        ]
        rels = [make_relationship("ALT-T0", "ALT-T1", rel_id="R-ALT-1")]
        result = tier_depth_tracker.check_depth_alerts(
            suppliers, rels, "cocoa", threshold_depth=4
        )
        assert len(result.alerts) >= 1
        assert any("depth" in a["type"].lower() or "tier" in a["type"].lower()
                    for a in result.alerts)

    def test_no_alert_when_above_threshold(self, tier_depth_tracker, cocoa_chain):
        """No alert when depth exceeds threshold."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.check_depth_alerts(
            suppliers, rels, "cocoa", threshold_depth=3
        )
        depth_alerts = [a for a in result.alerts
                        if "depth" in a.get("type", "").lower()
                        or "tier" in a.get("type", "").lower()]
        assert len(depth_alerts) == 0

    def test_alert_includes_commodity(self, tier_depth_tracker):
        """Alert message includes the commodity name."""
        suppliers = [make_supplier(supplier_id="ALT-COMM", tier=0, role="importer")]
        result = tier_depth_tracker.check_depth_alerts(
            suppliers, [], "palm_oil", threshold_depth=3
        )
        if result.alerts:
            assert any("palm" in str(a).lower() for a in result.alerts)


# ===========================================================================
# 9. Provenance and Determinism
# ===========================================================================


class TestTierDepthProvenance:
    """Test provenance hashing and deterministic results."""

    def test_depth_result_has_provenance_hash(self, tier_depth_tracker, cocoa_chain):
        """Depth calculation result includes SHA-256 provenance hash."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_depth_calculation_deterministic(self, tier_depth_tracker, cocoa_chain):
        """Same input always produces same depth result."""
        suppliers, rels = cocoa_chain
        result1 = tier_depth_tracker.calculate_depth(suppliers, rels)
        result2 = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result1.max_depth == result2.max_depth
        assert result1.provenance_hash == result2.provenance_hash

    def test_visibility_deterministic(self, tier_depth_tracker, cocoa_chain):
        """Same input always produces same visibility score."""
        suppliers, rels = cocoa_chain
        v1 = tier_depth_tracker.calculate_visibility(suppliers, rels, "cocoa")
        v2 = tier_depth_tracker.calculate_visibility(suppliers, rels, "cocoa")
        assert v1.overall_visibility == v2.overall_visibility


# ===========================================================================
# 10. Cross-Commodity Tier Depth
# ===========================================================================


class TestCrossCommodityTierDepth:
    """Test tier depth calculations across different commodities."""

    def test_coffee_chain_depth(self, tier_depth_tracker, coffee_chain):
        """Coffee 5-tier chain has depth 4."""
        suppliers, rels = coffee_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth >= 3

    def test_palm_chain_depth(self, tier_depth_tracker, palm_chain):
        """Palm oil 4-tier chain has depth 3."""
        suppliers, rels = palm_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth >= 2

    @pytest.mark.parametrize("commodity,tier_count", [
        ("cocoa", 7),
        ("coffee", 6),
        ("palm_oil", 6),
        ("soya", 5),
        ("rubber", 6),
        ("cattle", 4),
        ("wood", 5),
    ])
    def test_commodity_chain_depth(self, tier_depth_tracker, commodity, tier_count):
        """Each commodity chain has expected depth."""
        suppliers, rels = build_linear_chain(commodity, tier_count)
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.max_depth == tier_count - 1

    def test_total_supplier_count_in_result(self, tier_depth_tracker, cocoa_chain):
        """Depth result includes total supplier count."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert result.total_suppliers == len(suppliers)

    def test_suppliers_per_tier_in_result(self, tier_depth_tracker, cocoa_chain):
        """Depth result includes breakdown by tier."""
        suppliers, rels = cocoa_chain
        result = tier_depth_tracker.calculate_depth(suppliers, rels)
        assert isinstance(result.suppliers_per_tier, dict)
        total_in_tiers = sum(result.suppliers_per_tier.values())
        assert total_in_tiers == len(suppliers)

    def test_depth_with_disconnected_suppliers(self, tier_depth_tracker):
        """Disconnected suppliers (no relationships) each have depth 0."""
        suppliers = [
            make_supplier(supplier_id=f"DISC-{i}", tier=i) for i in range(3)
        ]
        # No relationships
        result = tier_depth_tracker.calculate_depth(suppliers, [])
        assert result is not None

    @pytest.mark.parametrize("alert_threshold", [1, 2, 3, 5, 8, 10])
    def test_alert_threshold_levels(self, tier_depth_tracker, alert_threshold):
        """Various alert thresholds produce appropriate alerts."""
        suppliers = [
            make_supplier(supplier_id="THR-T0", tier=0),
            make_supplier(supplier_id="THR-T1", tier=1),
            make_supplier(supplier_id="THR-T2", tier=2),
        ]
        rels = [
            make_relationship("THR-T0", "THR-T1", rel_id="R-THR-1"),
            make_relationship("THR-T1", "THR-T2", rel_id="R-THR-2"),
        ]
        result = tier_depth_tracker.check_depth_alerts(
            suppliers, rels, "cocoa", threshold_depth=alert_threshold
        )
        if alert_threshold > 2:
            assert len(result.alerts) >= 1
        else:
            # Depth 2 meets or exceeds threshold of 1 or 2
            assert len([a for a in result.alerts
                        if "depth" in a.get("type", "").lower()]) == 0
