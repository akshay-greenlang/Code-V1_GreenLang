# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 7: SiteComparisonEngine

Covers peer group building, KPI calculation, statistics (mean, median,
percentiles), site ranking, best practice identification, improvement
potential, trend calculation, and peer group minimum size.
Target: ~50 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.site_comparison_engine import (
        SiteComparisonEngine,
        PeerGroup,
        SiteKPI,
        ComparisonStatistics,
        ComparisonResult,
        SiteRanking,
        ImprovementPotential,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return SiteComparisonEngine()


@pytest.fixture
def peer_sites():
    """Five manufacturing sites with KPI data."""
    return [
        {
            "site_id": "site-001",
            "site_name": "Chicago Plant",
            "facility_type": "MANUFACTURING",
            "floor_area_m2": Decimal("25000"),
            "headcount": 350,
            "total_tco2e": Decimal("18000"),
            "revenue": Decimal("50000000"),
        },
        {
            "site_id": "site-A",
            "site_name": "Dallas Plant",
            "facility_type": "MANUFACTURING",
            "floor_area_m2": Decimal("20000"),
            "headcount": 280,
            "total_tco2e": Decimal("12000"),
            "revenue": Decimal("35000000"),
        },
        {
            "site_id": "site-B",
            "site_name": "Detroit Plant",
            "facility_type": "MANUFACTURING",
            "floor_area_m2": Decimal("30000"),
            "headcount": 400,
            "total_tco2e": Decimal("25000"),
            "revenue": Decimal("60000000"),
        },
        {
            "site_id": "site-C",
            "site_name": "Phoenix Plant",
            "facility_type": "MANUFACTURING",
            "floor_area_m2": Decimal("15000"),
            "headcount": 200,
            "total_tco2e": Decimal("10000"),
            "revenue": Decimal("25000000"),
        },
        {
            "site_id": "site-D",
            "site_name": "Portland Plant",
            "facility_type": "MANUFACTURING",
            "floor_area_m2": Decimal("18000"),
            "headcount": 250,
            "total_tco2e": Decimal("8000"),
            "revenue": Decimal("30000000"),
        },
    ]


# ============================================================================
# Peer Group Tests
# ============================================================================

class TestPeerGroup:

    def test_build_peer_group(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="Manufacturing Peers",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        assert group is not None
        assert group.group_name == "Manufacturing Peers"
        assert len(group.sites) == 5

    def test_build_peer_group_min_size(self, engine):
        small_group = [
            {
                "site_id": "s1", "site_name": "S1",
                "facility_type": "OFFICE",
                "floor_area_m2": Decimal("1000"),
                "headcount": 50,
                "total_tco2e": Decimal("100"),
            },
        ]
        with pytest.raises((ValueError, Exception)):
            engine.build_peer_group(
                group_name="Too Small",
                facility_type="OFFICE",
                sites=small_group,
                min_size=3,
            )


# ============================================================================
# KPI Calculation Tests
# ============================================================================

class TestKPICalculation:

    def test_calculate_site_kpis(self, engine, peer_sites):
        kpis = engine.calculate_kpis(peer_sites[0])
        assert kpis is not None
        # emissions_per_m2 = 18000 / 25000 = 0.72
        assert kpis.emissions_per_m2 == Decimal("0.72") or \
               abs(kpis.emissions_per_m2 - Decimal("0.72")) < Decimal("0.01")

    def test_calculate_kpis_per_fte(self, engine, peer_sites):
        kpis = engine.calculate_kpis(peer_sites[0])
        # 18000 / 350 = 51.43
        assert abs(kpis.emissions_per_fte - Decimal("51.43")) < Decimal("1")

    def test_calculate_kpis_all_sites(self, engine, peer_sites):
        for site in peer_sites:
            kpis = engine.calculate_kpis(site)
            assert kpis.emissions_per_m2 > Decimal("0")
            assert kpis.emissions_per_fte > Decimal("0")


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:

    def test_calculate_statistics_mean_median(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        stats = engine.calculate_statistics(group, kpi="EMISSIONS_PER_M2")
        assert stats.mean > Decimal("0")
        assert stats.median > Decimal("0")
        assert stats.min <= stats.median <= stats.max

    def test_statistics_percentiles(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        stats = engine.calculate_statistics(group, kpi="EMISSIONS_PER_M2")
        assert stats.p25 <= stats.median
        assert stats.median <= stats.p75

    def test_statistics_std_dev(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        stats = engine.calculate_statistics(group, kpi="EMISSIONS_PER_M2")
        assert stats.std_dev >= Decimal("0")


# ============================================================================
# Ranking Tests
# ============================================================================

class TestSiteRanking:

    def test_rank_sites(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        rankings = engine.rank_sites(group, kpi="EMISSIONS_PER_M2")
        assert len(rankings) == 5
        # Best site (lowest emissions per m2) should be rank 1
        assert rankings[0].rank == 1
        assert rankings[0].kpi_value <= rankings[-1].kpi_value

    def test_rank_sites_order(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        rankings = engine.rank_sites(group, kpi="EMISSIONS_PER_M2")
        for i in range(len(rankings) - 1):
            assert rankings[i].kpi_value <= rankings[i + 1].kpi_value


# ============================================================================
# Best Practice and Improvement Tests
# ============================================================================

class TestBestPractice:

    def test_identify_best_practices(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        best = engine.identify_best_practice(group, kpi="EMISSIONS_PER_M2")
        assert best is not None
        assert best.site_id is not None

    def test_improvement_potential(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="MFG",
            facility_type="MANUFACTURING",
            sites=peer_sites,
        )
        potentials = engine.calculate_improvement_potential(
            group, kpi="EMISSIONS_PER_M2",
        )
        assert len(potentials) == 5
        # Best performer should have zero or minimal improvement potential
        best_potential = min(potentials, key=lambda p: p.potential_tco2e)
        assert best_potential.potential_tco2e >= Decimal("0")


# ============================================================================
# Comparison Tests
# ============================================================================

class TestSiteComparison:

    def test_compare_sites(self, engine, peer_sites):
        result = engine.compare_sites(
            site_a=peer_sites[0],
            site_b=peer_sites[4],
            kpi="EMISSIONS_PER_M2",
        )
        assert result is not None
        assert result.difference is not None

    def test_compare_sites_same(self, engine, peer_sites):
        result = engine.compare_sites(
            site_a=peer_sites[0],
            site_b=peer_sites[0],
            kpi="EMISSIONS_PER_M2",
        )
        assert result.difference == Decimal("0") or abs(result.difference) < Decimal("0.01")


# ============================================================================
# Trend Tests
# ============================================================================

class TestTrend:

    def test_trend_calculation(self, engine, peer_sites):
        historical = {
            "site-001": {
                2024: Decimal("20000"),
                2025: Decimal("19000"),
                2026: Decimal("18000"),
            },
        }
        trend = engine.calculate_trend(
            site_id="site-001",
            historical_kpi=historical["site-001"],
        )
        assert trend is not None
        assert trend.direction in ("DECREASING", "IMPROVING", "DOWN")

    def test_trend_stable(self, engine):
        historical = {
            2024: Decimal("18000"),
            2025: Decimal("18000"),
            2026: Decimal("18000"),
        }
        trend = engine.calculate_trend(
            site_id="site-001",
            historical_kpi=historical,
        )
        assert trend.direction in ("STABLE", "FLAT", "UNCHANGED") or \
               abs(trend.change_pct) < Decimal("1")


# ============================================================================
# Peer Group Minimum Size Tests
# ============================================================================

class TestPeerGroupMinSize:

    def test_peer_group_min_size(self, engine, peer_sites):
        # Should work with 5 sites and min_size=3
        group = engine.build_peer_group(
            group_name="OK Size",
            facility_type="MANUFACTURING",
            sites=peer_sites,
            min_size=3,
        )
        assert len(group.sites) == 5

    def test_peer_group_exactly_min(self, engine, peer_sites):
        group = engine.build_peer_group(
            group_name="Min Size",
            facility_type="MANUFACTURING",
            sites=peer_sites[:3],
            min_size=3,
        )
        assert len(group.sites) == 3
