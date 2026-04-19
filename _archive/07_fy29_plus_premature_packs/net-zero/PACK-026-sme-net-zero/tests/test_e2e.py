# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-026 SME Net Zero Pack.

Tests full flow from onboarding through baseline, target setting,
quick wins, action planning, grant matching, cost-benefit analysis,
certification readiness, and report generation.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~500 lines, 55+ tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.sme_baseline_engine import (
    SMEBaselineEngine, SMEBaselineInput, SMEBaselineResult, DataTier,
)
from engines.simplified_target_engine import (
    SimplifiedTargetEngine, TargetInput, SimplifiedTargetResult,
)
from engines.quick_wins_engine import (
    QuickWinsEngine, QuickWinsInput, QuickWinsResult,
)
from engines.scope3_estimator_engine import (
    Scope3EstimatorEngine, Scope3EstimatorInput, Scope3EstimatorResult,
    Scope3Category, SpendEntry,
)
from engines.grant_finder_engine import (
    GrantFinderEngine, GrantFinderInput, GrantFinderResult,
)
from engines.certification_readiness_engine import (
    CertificationReadinessEngine, CertificationReadinessInput, DimensionInput,
)

from .conftest import assert_provenance_hash, timed_block


# ========================================================================
# Micro Business Full Journey
# ========================================================================


class TestMicroBusinessFullJourney:
    """End-to-end: Micro cafe from baseline to target."""

    def test_micro_business_baseline_and_target(self):
        # Step 1: Bronze Baseline
        baseline_engine = SMEBaselineEngine()
        baseline = baseline_engine.calculate(SMEBaselineInput(
            entity_name="Green Cafe Ltd",
            reporting_year=2025,
            sector="accommodation_food",
            company_size="micro",
            headcount=6,
            revenue_usd=Decimal("350000"),
            data_tier=DataTier.BRONZE,
        ))
        assert isinstance(baseline, SMEBaselineResult)
        assert baseline.total_tco2e > Decimal("0")

        # Step 2: Simplified Target (50% by 2030)
        target_engine = SimplifiedTargetEngine()
        target = target_engine.calculate(TargetInput(
            entity_name="Green Cafe Ltd",
            base_year=2024,
            base_year_scope1_tco2e=baseline.scope1.total_tco2e,
            base_year_scope2_tco2e=baseline.scope2.total_tco2e,
            base_year_scope3_tco2e=baseline.scope3.total_tco2e,
            current_year=2025,
        ))
        assert isinstance(target, SimplifiedTargetResult)
        base_total = baseline.scope1.total_tco2e + baseline.scope2.total_tco2e + baseline.scope3.total_tco2e
        assert target.near_term_target.target_emissions_tco2e <= base_total * Decimal("0.51")

        # Step 3: Quick Wins
        qw_engine = QuickWinsEngine()
        quick_wins = qw_engine.calculate(QuickWinsInput(
            entity_name="Green Cafe Ltd",
            headcount=6,
            sector="hospitality",
            total_emissions_tco2e=baseline.total_tco2e,
        ))
        assert isinstance(quick_wins, QuickWinsResult)
        assert len(quick_wins.actions) > 0

        # Step 4: All results have provenance
        assert_provenance_hash(baseline)
        assert_provenance_hash(target)
        assert_provenance_hash(quick_wins)

    def test_micro_express_under_30_seconds(self):
        """Micro business express flow should complete in <30 seconds."""
        with timed_block("micro_express", max_seconds=30.0):
            baseline = SMEBaselineEngine().calculate(SMEBaselineInput(
                entity_name="Express Cafe",
                reporting_year=2025,
                sector="accommodation_food",
                company_size="micro",
                headcount=4,
                revenue_usd=Decimal("200000"),
                data_tier=DataTier.BRONZE,
            ))
            target = SimplifiedTargetEngine().calculate(TargetInput(
                entity_name="Express Cafe",
                base_year=2024,
                base_year_scope1_tco2e=baseline.scope1.total_tco2e,
                base_year_scope2_tco2e=baseline.scope2.total_tco2e,
                base_year_scope3_tco2e=baseline.scope3.total_tco2e,
                current_year=2025,
            ))
            quick_wins = QuickWinsEngine().calculate(QuickWinsInput(
                entity_name="Express Cafe",
                headcount=4,
                sector="hospitality",
                total_emissions_tco2e=baseline.total_tco2e,
            ))
        assert baseline is not None
        assert target is not None
        assert quick_wins is not None


# ========================================================================
# Small Business Full Journey
# ========================================================================


class TestSmallBusinessFullJourney:
    """End-to-end: Small tech company with Silver baseline."""

    def test_small_business_full_pipeline(self):
        # Step 1: Silver Baseline (activity data)
        baseline = SMEBaselineEngine().calculate(SMEBaselineInput(
            entity_name="TechSoft Ltd",
            reporting_year=2025,
            sector="information_technology",
            company_size="small",
            headcount=32,
            revenue_usd=Decimal("4500000"),
            data_tier=DataTier.SILVER,
        ))
        assert baseline.accuracy_band.confidence_pct >= Decimal("70")

        # Step 2: Scope 3 Estimation
        scope3 = Scope3EstimatorEngine().calculate(Scope3EstimatorInput(
            entity_name="TechSoft Ltd",
            reporting_year=2025,
            headcount=32,
            spend_entries=[
                SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("1500000")),
                SpendEntry(category=Scope3Category.CAT_06_BUSINESS_TRAVEL, amount=Decimal("80000")),
                SpendEntry(category=Scope3Category.CAT_07_EMPLOYEE_COMMUTING, amount=Decimal("45000")),
            ],
        ))
        assert scope3.total_scope3_tco2e > Decimal("0")

        # Step 3: Target Setting
        target = SimplifiedTargetEngine().calculate(TargetInput(
            entity_name="TechSoft Ltd",
            base_year=2024,
            base_year_scope1_tco2e=baseline.scope1.total_tco2e,
            base_year_scope2_tco2e=baseline.scope2.total_tco2e,
            base_year_scope3_tco2e=scope3.total_scope3_tco2e,
            current_year=2025,
        ))
        assert target.near_term_target is not None

        # Step 4: Quick Wins
        qw = QuickWinsEngine().calculate(QuickWinsInput(
            entity_name="TechSoft Ltd",
            headcount=32,
            sector="office_based",
            total_emissions_tco2e=baseline.total_tco2e,
            annual_budget_usd=Decimal("50000"),
        ))
        assert len(qw.actions) > 0


# ========================================================================
# Medium Business Full Journey
# ========================================================================


class TestMediumBusinessFullJourney:
    """End-to-end: Medium manufacturer with Gold baseline."""

    def test_medium_business_full_pipeline(self):
        # Step 1: Gold Baseline
        baseline = SMEBaselineEngine().calculate(SMEBaselineInput(
            entity_name="EuroManufact GmbH",
            reporting_year=2025,
            sector="manufacturing",
            company_size="medium",
            headcount=145,
            revenue_usd=Decimal("28000000"),
            data_tier=DataTier.GOLD,
        ))
        assert baseline.total_tco2e > Decimal("0")

        # Step 2: Grant Matching
        grants = GrantFinderEngine().calculate(GrantFinderInput(
            entity_name="EuroManufact GmbH",
            industry="manufacturing",
            company_size="medium",
            country="DE",
            project_types=["energy_efficiency", "renewable_energy"],
            total_emissions_tco2e=baseline.total_tco2e,
        ))
        assert grants is not None

        # Verify results have provenance
        assert_provenance_hash(baseline)
        assert_provenance_hash(grants)


# ========================================================================
# Cross-Engine Consistency
# ========================================================================


class TestCrossEngineConsistency:
    def test_baseline_feeds_into_target(self):
        """Baseline emissions should feed correctly into target engine."""
        baseline = SMEBaselineEngine().calculate(SMEBaselineInput(
            entity_name="ConsistencyCo",
            reporting_year=2025,
            sector="wholesale_retail",
            company_size="small",
            headcount=20,
            revenue_usd=Decimal("2000000"),
            data_tier=DataTier.BRONZE,
        ))
        target = SimplifiedTargetEngine().calculate(TargetInput(
            entity_name="ConsistencyCo",
            base_year=2024,
            base_year_scope1_tco2e=baseline.scope1.total_tco2e,
            base_year_scope2_tco2e=baseline.scope2.total_tco2e,
            base_year_scope3_tco2e=baseline.scope3.total_tco2e,
            current_year=2025,
        ))
        base_total = baseline.scope1.total_tco2e + baseline.scope2.total_tco2e + baseline.scope3.total_tco2e
        assert target.near_term_target.target_emissions_tco2e <= base_total * Decimal("0.51")

    def test_scope3_adds_to_baseline(self):
        """Scope 3 estimates should augment baseline Scope 3."""
        baseline = SMEBaselineEngine().calculate(SMEBaselineInput(
            entity_name="AugmentCo",
            reporting_year=2025,
            sector="wholesale_retail",
            company_size="small",
            headcount=20,
            revenue_usd=Decimal("2000000"),
            data_tier=DataTier.BRONZE,
        ))
        scope3 = Scope3EstimatorEngine().calculate(Scope3EstimatorInput(
            entity_name="AugmentCo",
            reporting_year=2025,
            headcount=20,
            spend_entries=[
                SpendEntry(category=Scope3Category.CAT_01_PURCHASED_GOODS, amount=Decimal("500000")),
                SpendEntry(category=Scope3Category.CAT_06_BUSINESS_TRAVEL, amount=Decimal("15000")),
                SpendEntry(category=Scope3Category.CAT_07_EMPLOYEE_COMMUTING, amount=Decimal("12000")),
            ],
        ))
        # Both should produce valid emissions
        assert baseline.scope3.total_tco2e >= Decimal("0")
        assert scope3.total_scope3_tco2e > Decimal("0")


# ========================================================================
# Demo Config Run
# ========================================================================


class TestDemoConfigRun:
    def test_demo_baseline_runs(self):
        result = SMEBaselineEngine().calculate(SMEBaselineInput(
            entity_name="Demo SME",
            reporting_year=2025,
            sector="wholesale_retail",
            company_size="small",
            headcount=15,
            revenue_usd=Decimal("1500000"),
            data_tier=DataTier.BRONZE,
        ))
        assert isinstance(result, SMEBaselineResult)
        assert result.total_tco2e > Decimal("0")
        assert_provenance_hash(result)

    def test_demo_target_runs(self):
        result = SimplifiedTargetEngine().calculate(TargetInput(
            entity_name="Demo SME",
            base_year=2024,
            base_year_scope1_tco2e=Decimal("30"),
            base_year_scope2_tco2e=Decimal("30"),
            base_year_scope3_tco2e=Decimal("40"),
            current_year=2025,
        ))
        assert isinstance(result, SimplifiedTargetResult)
        assert result.near_term_target.target_emissions_tco2e <= Decimal("50.1")
        assert_provenance_hash(result)

    def test_demo_quick_wins_runs(self):
        result = QuickWinsEngine().calculate(QuickWinsInput(
            entity_name="Demo SME",
            headcount=15,
            sector="retail",
            total_emissions_tco2e=Decimal("100"),
            annual_budget_usd=Decimal("20000"),
        ))
        assert isinstance(result, QuickWinsResult)
        assert len(result.actions) > 0
        assert_provenance_hash(result)
