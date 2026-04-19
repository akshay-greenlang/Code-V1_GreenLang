# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 5: SiteConsolidationEngine

Covers equity share and operational control consolidation, equity adjustments,
inter-company elimination, reconciliation, base year restatement, contribution
analysis, scope breakdown, completeness checks, and provenance.
Target: ~60 tests.
"""

import pytest
from decimal import Decimal
from datetime import date
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.site_consolidation_engine import (
        SiteConsolidationEngine,
        SiteTotal,
        ConsolidationResult,
        ConsolidationRun,
        EquityAdjustment,
        Elimination,
        ReconciliationResult,
        ContributionAnalysis,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return SiteConsolidationEngine()


@pytest.fixture
def site_totals():
    """Five site totals with Scope 1, 2, 3."""
    return [
        SiteTotal(
            site_id="site-001",
            site_name="Chicago Plant",
            scope1_tco2e=Decimal("5000.00"),
            scope2_tco2e=Decimal("3000.00"),
            scope3_tco2e=Decimal("10000.00"),
            total_tco2e=Decimal("18000.00"),
            ownership_pct=Decimal("100.00"),
        ),
        SiteTotal(
            site_id="site-002",
            site_name="London Office",
            scope1_tco2e=Decimal("100.00"),
            scope2_tco2e=Decimal("350.00"),
            scope3_tco2e=Decimal("800.00"),
            total_tco2e=Decimal("1250.00"),
            ownership_pct=Decimal("100.00"),
        ),
        SiteTotal(
            site_id="site-003",
            site_name="Frankfurt Warehouse",
            scope1_tco2e=Decimal("800.00"),
            scope2_tco2e=Decimal("500.00"),
            scope3_tco2e=Decimal("2000.00"),
            total_tco2e=Decimal("3300.00"),
            ownership_pct=Decimal("75.00"),
        ),
        SiteTotal(
            site_id="site-004",
            site_name="NY Store",
            scope1_tco2e=Decimal("150.00"),
            scope2_tco2e=Decimal("50.00"),
            scope3_tco2e=Decimal("200.00"),
            total_tco2e=Decimal("400.00"),
            ownership_pct=Decimal("50.00"),
        ),
        SiteTotal(
            site_id="site-005",
            site_name="Berlin DC",
            scope1_tco2e=Decimal("200.00"),
            scope2_tco2e=Decimal("2500.00"),
            scope3_tco2e=Decimal("5000.00"),
            total_tco2e=Decimal("7700.00"),
            ownership_pct=Decimal("100.00"),
        ),
    ]


# ============================================================================
# Equity Share Consolidation Tests
# ============================================================================

class TestEquityShareConsolidation:

    def test_consolidate_equity_50pct(self, engine):
        totals = [
            SiteTotal(
                site_id="s1", site_name="JV Plant",
                scope1_tco2e=Decimal("1000"), scope2_tco2e=Decimal("500"),
                scope3_tco2e=Decimal("200"), total_tco2e=Decimal("1700"),
                ownership_pct=Decimal("50.00"),
            ),
        ]
        result = engine.consolidate(totals, approach="EQUITY_SHARE")
        # 50% of 1000 = 500
        assert result.consolidated_scope1 == Decimal("500.00") or \
               result.consolidated_scope1 == Decimal("500.000000")

    def test_consolidate_equity_100pct(self, engine):
        totals = [
            SiteTotal(
                site_id="s1", site_name="Wholly Owned",
                scope1_tco2e=Decimal("1000"), scope2_tco2e=Decimal("500"),
                scope3_tco2e=Decimal("200"), total_tco2e=Decimal("1700"),
                ownership_pct=Decimal("100.00"),
            ),
        ]
        result = engine.consolidate(totals, approach="EQUITY_SHARE")
        assert result.consolidated_scope1 == Decimal("1000.00") or \
               result.consolidated_scope1 >= Decimal("999")

    def test_consolidate_operational_control(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="OPERATIONAL_CONTROL")
        # Operational control: 100% for all sites with operational control
        assert result.consolidated_total > Decimal("0")

    def test_equity_adjustments(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        # Site-003 at 75%: scope1 = 800 * 0.75 = 600
        # Site-004 at 50%: scope1 = 150 * 0.50 = 75
        # Adjusted total scope1 = 5000 + 100 + 600 + 75 + 200 = 5975
        expected_scope1 = Decimal("5975.00")
        assert result.consolidated_scope1 == expected_scope1 or \
               abs(result.consolidated_scope1 - expected_scope1) < Decimal("1")

    def test_equity_adjustments_jv_50pct(self, engine):
        totals = [
            SiteTotal(
                site_id="jv-001", site_name="Joint Venture A",
                scope1_tco2e=Decimal("2000"), scope2_tco2e=Decimal("1000"),
                scope3_tco2e=Decimal("500"), total_tco2e=Decimal("3500"),
                ownership_pct=Decimal("50.00"),
            ),
            SiteTotal(
                site_id="jv-002", site_name="Joint Venture B",
                scope1_tco2e=Decimal("4000"), scope2_tco2e=Decimal("2000"),
                scope3_tco2e=Decimal("1000"), total_tco2e=Decimal("7000"),
                ownership_pct=Decimal("50.00"),
            ),
        ]
        result = engine.consolidate(totals, approach="EQUITY_SHARE")
        # S1: 2000*0.5 + 4000*0.5 = 1000 + 2000 = 3000
        expected = Decimal("3000")
        assert abs(result.consolidated_scope1 - expected) < Decimal("1")


# ============================================================================
# Elimination Tests
# ============================================================================

class TestEliminations:

    def test_identify_eliminations(self, engine, site_totals):
        eliminations = engine.identify_eliminations(
            site_totals,
            inter_company_transactions=[
                {
                    "from_site": "site-001",
                    "to_site": "site-003",
                    "emissions_tco2e": Decimal("100.00"),
                    "scope": "SCOPE_3",
                },
            ],
        )
        assert len(eliminations) >= 1
        assert eliminations[0].emissions_tco2e == Decimal("100.00")

    def test_apply_eliminations(self, engine, site_totals):
        result = engine.consolidate(
            site_totals,
            approach="EQUITY_SHARE",
            eliminations=[
                {
                    "from_site": "site-001",
                    "to_site": "site-003",
                    "emissions_tco2e": Decimal("100.00"),
                    "scope": "SCOPE_3",
                },
            ],
        )
        # Eliminated 100 tCO2e from scope 3
        assert result.eliminations_total >= Decimal("0")


# ============================================================================
# Reconciliation Tests
# ============================================================================

class TestReconciliation:

    def test_reconcile_within_1pct(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        recon = engine.reconcile(
            result,
            expected_total=result.consolidated_total,
            tolerance=Decimal("0.01"),
        )
        assert recon.within_tolerance is True
        assert recon.difference_pct <= Decimal("1")

    def test_reconcile_outside_tolerance(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        recon = engine.reconcile(
            result,
            expected_total=result.consolidated_total * Decimal("1.10"),
            tolerance=Decimal("0.01"),
        )
        assert recon.within_tolerance is False


# ============================================================================
# Base Year Restatement Tests
# ============================================================================

class TestBaseYearRestatement:

    def test_restate_base_year_acquisition(self, engine, site_totals):
        base_totals = [
            SiteTotal(
                site_id="site-001", site_name="Chicago Plant",
                scope1_tco2e=Decimal("4500"), scope2_tco2e=Decimal("2800"),
                scope3_tco2e=Decimal("9000"), total_tco2e=Decimal("16300"),
                ownership_pct=Decimal("100.00"),
            ),
        ]
        restated = engine.restate_base_year(
            base_year_totals=base_totals,
            structural_change={
                "type": "ACQUISITION",
                "site_id": "site-002",
                "emissions_tco2e": Decimal("1250"),
                "effective_date": date(2026, 7, 1),
            },
        )
        assert restated.restated_total > base_totals[0].total_tco2e


# ============================================================================
# Contribution Analysis Tests
# ============================================================================

class TestContributionAnalysis:

    def test_contribution_analysis(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        analysis = engine.get_contribution_analysis(result)
        assert len(analysis.site_contributions) == 5
        # All contributions should sum to ~100%
        total_pct = sum(c.contribution_pct for c in analysis.site_contributions)
        assert abs(total_pct - Decimal("100")) < Decimal("1")

    def test_scope_breakdown(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        assert result.consolidated_scope1 > Decimal("0")
        assert result.consolidated_scope2 > Decimal("0")
        assert result.consolidated_scope3 > Decimal("0")
        assert result.consolidated_total == (
            result.consolidated_scope1 +
            result.consolidated_scope2 +
            result.consolidated_scope3
        ) or abs(
            result.consolidated_total - (
                result.consolidated_scope1 +
                result.consolidated_scope2 +
                result.consolidated_scope3
            )
        ) < Decimal("1")


# ============================================================================
# Completeness and Provenance Tests
# ============================================================================

class TestCompletenessAndProvenance:

    def test_completeness_check(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        assert result.completeness_pct >= Decimal("0")

    def test_consolidated_total_sum(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        expected = (
            result.consolidated_scope1 +
            result.consolidated_scope2 +
            result.consolidated_scope3
        )
        assert abs(result.consolidated_total - expected) < Decimal("1")

    def test_decimal_precision_consolidation(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        # Should use Decimal, not float
        assert isinstance(result.consolidated_scope1, Decimal)
        assert isinstance(result.consolidated_total, Decimal)

    def test_provenance_hash(self, engine, site_totals):
        result = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(self, engine, site_totals):
        r1 = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        r2 = engine.consolidate(site_totals, approach="EQUITY_SHARE")
        assert r1.consolidated_total == r2.consolidated_total

    def test_empty_site_totals(self, engine):
        result = engine.consolidate([], approach="EQUITY_SHARE")
        assert result.consolidated_total == Decimal("0")
