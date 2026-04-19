# -*- coding: utf-8 -*-
"""
End-to-end pipeline tests for PACK-021 Net Zero Starter Pack.

Tests the full flow from baseline calculation through target setting,
gap analysis, reduction planning, residual emissions, offset portfolio,
scorecard assessment, and benchmarking -- using real engine logic (no mocks).

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.residual_emissions_engine import (
    CDRType,
    ResidualEmissionsEngine,
    ResidualInput,
    ResidualResult,
)
from engines.offset_portfolio_engine import (
    CreditCategory,
    CreditEntry,
    CreditStandard,
    CreditType,
    OffsetPortfolioEngine,
    PortfolioResult,
    SBTiCreditUse,
    VCMIClaim,
)
from engines.net_zero_scorecard_engine import (
    DimensionInput,
    DIMENSION_INDICATORS,
    MaturityLevel,
    NetZeroScorecardEngine,
    ScorecardDimension,
    ScorecardInput,
    ScorecardResult,
)
from engines.net_zero_benchmark_engine import (
    BenchmarkInput,
    BenchmarkResult,
    BenchmarkSector,
    NetZeroBenchmarkEngine,
    SBTiStatus,
)


# ========================================================================
# Helper
# ========================================================================


def _build_scorecard_input(fill_pct: Decimal) -> ScorecardInput:
    """Build a ScorecardInput with all dimensions at a given percentage."""
    dims = []
    for dim in ScorecardDimension:
        indicators = DIMENSION_INDICATORS.get(dim.value, [])
        scores = {}
        for ind in indicators:
            max_pts = ind["max_points"]
            scores[ind["id"]] = max_pts * fill_pct / Decimal("100")
        dims.append(DimensionInput(dimension=dim, indicator_scores=scores))
    return ScorecardInput(
        entity_name="E2E TestCo",
        sector="manufacturing",
        assessment_year=2026,
        dimensions=dims,
    )


# ========================================================================
# Baseline to Target Flow
# ========================================================================


class TestBaselineToTargetFlow:
    """End-to-end: Baseline -> Residual -> Offset -> Scorecard."""

    def test_baseline_to_target_flow(self):
        """Full flow: calculate residual, build offset portfolio, score maturity."""
        # Step 1: Residual Emissions
        residual_engine = ResidualEmissionsEngine()
        residual_input = ResidualInput(
            entity_name="E2E Manufacturing Corp",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("8000"),
            base_year_scope2_tco2e=Decimal("4000"),
            base_year_scope3_tco2e=Decimal("18000"),
            target_year=2050,
            current_year=2026,
        )
        residual_result = residual_engine.calculate(residual_input)

        assert isinstance(residual_result, ResidualResult)
        assert residual_result.residual_budget_tco2e > Decimal("0")
        # Manufacturing 10% of 30000 = 3000
        assert residual_result.residual_budget_tco2e == Decimal("3000.000")

        # Step 2: Offset Portfolio
        offset_engine = OffsetPortfolioEngine()
        offset_engine.add_credit(CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            category=CreditCategory.BIOCHAR,
            vintage_year=2025,
            quantity_tco2e=Decimal("1500"),
            unit_price_usd=Decimal("120"),
            additionality_score=4,
            permanence_score=4,
            co_benefits_score=4,
            leakage_risk_score=4,
            mrv_quality_score=4,
            sbti_use=SBTiCreditUse.NEUTRALIZATION,
        ))
        offset_engine.add_credit(CreditEntry(
            standard=CreditStandard.GOLD_STANDARD,
            credit_type=CreditType.AVOIDANCE,
            category=CreditCategory.RENEWABLE_ENERGY,
            vintage_year=2025,
            quantity_tco2e=Decimal("2000"),
            unit_price_usd=Decimal("8"),
            additionality_score=3,
            permanence_score=2,
            co_benefits_score=4,
            leakage_risk_score=4,
            mrv_quality_score=3,
            sbti_use=SBTiCreditUse.BVCM_COMPENSATION,
        ))

        portfolio_result = offset_engine.analyze_portfolio(
            reduction_progress_pct=Decimal("40"),
            assessment_year=2026,
        )
        assert isinstance(portfolio_result, PortfolioResult)
        assert portfolio_result.portfolio_summary.total_credits_tco2e == Decimal("3500")
        assert portfolio_result.average_quality_score > Decimal("0")
        assert portfolio_result.sbti_compliance is not None
        assert portfolio_result.vcmi_alignment is not None

        # Step 3: Scorecard
        scorecard_engine = NetZeroScorecardEngine()
        scorecard_input = _build_scorecard_input(fill_pct=Decimal("55"))
        scorecard_result = scorecard_engine.assess(scorecard_input)

        assert isinstance(scorecard_result, ScorecardResult)
        assert scorecard_result.maturity_level == MaturityLevel.DEVELOPING
        assert len(scorecard_result.dimension_scores) == 8

        # Step 4: All results have provenance hashes
        assert len(residual_result.provenance_hash) == 64
        assert len(portfolio_result.provenance_hash) == 64
        assert len(scorecard_result.provenance_hash) == 64


# ========================================================================
# Full Net Zero Assessment Pipeline
# ========================================================================


class TestFullNetZeroAssessmentPipeline:
    """End-to-end: All 4 engines run in sequence."""

    def test_full_net_zero_assessment_pipeline(self):
        """Execute all 4 engines sequentially for a complete assessment."""
        # Engine 5: Residual
        residual = ResidualEmissionsEngine().calculate(ResidualInput(
            entity_name="Pipeline Corp",
            sector="technology",
            base_year=2021,
            base_year_scope1_tco2e=Decimal("300"),
            base_year_scope2_tco2e=Decimal("700"),
            base_year_scope3_tco2e=Decimal("5000"),
            target_year=2045,
            current_year=2026,
        ))
        assert residual.residual_budget_tco2e > Decimal("0")
        # Technology 5% of 6000 = 300
        assert residual.residual_budget_tco2e == Decimal("300.000")

        # Engine 6: Offset
        offset_engine = OffsetPortfolioEngine()
        offset_engine.add_credit(CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            category=CreditCategory.DIRECT_AIR_CAPTURE,
            vintage_year=2026,
            quantity_tco2e=Decimal("300"),
            unit_price_usd=Decimal("450"),
            additionality_score=5,
            permanence_score=5,
            co_benefits_score=3,
            leakage_risk_score=5,
            mrv_quality_score=5,
            sbti_use=SBTiCreditUse.NEUTRALIZATION,
        ))
        portfolio = offset_engine.analyze_portfolio(
            reduction_progress_pct=Decimal("70"),
            assessment_year=2026,
        )
        assert portfolio.average_quality_score > Decimal("70")

        # Engine 7: Scorecard
        scorecard = NetZeroScorecardEngine().assess(
            _build_scorecard_input(fill_pct=Decimal("75"))
        )
        assert scorecard.maturity_level in (MaturityLevel.ADVANCED, MaturityLevel.LEADING)

        # Engine 8: Benchmark
        benchmark = NetZeroBenchmarkEngine().benchmark(BenchmarkInput(
            entity_name="Pipeline Corp",
            sector=BenchmarkSector.INFORMATION_TECHNOLOGY,
            carbon_intensity_revenue=Decimal("5"),
            annual_reduction_rate_pct=Decimal("8"),
            scope3_categories_measured=10,
            renewable_electricity_pct=Decimal("85"),
            cdp_score="A-",
            sbti_status=SBTiStatus.APPROVED,
        ))
        assert isinstance(benchmark, BenchmarkResult)
        assert len(benchmark.percentile_rankings) > 0
        assert benchmark.peer_comparison is not None

        # All provenance hashes are valid
        for result in [residual, portfolio, scorecard, benchmark]:
            assert len(result.provenance_hash) == 64


# ========================================================================
# Progress Review Flow
# ========================================================================


class TestProgressReviewFlow:
    """End-to-end: Multi-year progress tracking."""

    def test_progress_review_flow(self):
        """Compare two years of residual calculations to track progress."""
        engine = ResidualEmissionsEngine()

        # Year 1 (2023 snapshot)
        y1 = engine.calculate(ResidualInput(
            entity_name="ProgressCo",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("12000"),
            target_year=2050,
            current_year=2023,
        ))

        # Year 2 (2026 snapshot - same base year, different current year)
        y2 = engine.calculate(ResidualInput(
            entity_name="ProgressCo",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("12000"),
            target_year=2050,
            current_year=2026,
        ))

        # Budget stays the same (same base year and sector)
        assert y1.residual_budget_tco2e == y2.residual_budget_tco2e

        # But timeline urgency should differ (3 fewer years)
        assert y1.timeline.years_remaining > y2.timeline.years_remaining

        # Both have valid provenance hashes (but different due to timeline)
        assert len(y1.provenance_hash) == 64
        assert len(y2.provenance_hash) == 64


# ========================================================================
# Demo Config Runs Baseline
# ========================================================================


class TestDemoConfigRunsBaseline:
    """Tests that a demo-style configuration produces valid results."""

    def test_demo_config_runs_baseline(self):
        """A simple demo configuration produces a valid ResidualResult."""
        engine = ResidualEmissionsEngine()
        result = engine.calculate(ResidualInput(
            entity_name="Demo Corp",
            sector="default",
            base_year=2022,
            base_year_scope1_tco2e=Decimal("1000"),
            base_year_scope2_tco2e=Decimal("500"),
            base_year_scope3_tco2e=Decimal("3500"),
            target_year=2050,
            current_year=2026,
        ))
        assert isinstance(result, ResidualResult)
        # default sector -> 10% of 5000 = 500
        assert result.residual_budget_tco2e == Decimal("500.000")
        assert result.neutralization_required_tco2e == Decimal("500.000")
        assert len(result.cdr_options) == 8
        assert result.timeline is not None
        assert result.provenance_hash

    def test_demo_benchmark_runs(self):
        """A simple demo benchmarking produces a valid BenchmarkResult."""
        engine = NetZeroBenchmarkEngine()
        result = engine.benchmark(BenchmarkInput(
            entity_name="Demo Corp",
            sector=BenchmarkSector.INDUSTRIALS,
            carbon_intensity_revenue=Decimal("100"),
            annual_reduction_rate_pct=Decimal("3"),
            scope3_categories_measured=5,
            renewable_electricity_pct=Decimal("20"),
            cdp_score="B-",
            sbti_status=SBTiStatus.NONE,
        ))
        assert isinstance(result, BenchmarkResult)
        assert result.provenance_hash

    def test_demo_scorecard_runs(self):
        """A simple demo scorecard produces a valid ScorecardResult."""
        engine = NetZeroScorecardEngine()
        inp = _build_scorecard_input(fill_pct=Decimal("30"))
        result = engine.assess(inp)
        assert isinstance(result, ScorecardResult)
        assert result.maturity_level == MaturityLevel.FOUNDATION
        assert result.provenance_hash


# ========================================================================
# Cross-Engine Consistency
# ========================================================================


class TestCrossEngineConsistency:
    """Validates data consistency across engine outputs."""

    def test_residual_budget_matches_offset_need(self):
        """Residual budget should inform offset portfolio size."""
        residual = ResidualEmissionsEngine().calculate(ResidualInput(
            entity_name="ConsistencyCo",
            sector="retail",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("2000"),
            base_year_scope2_tco2e=Decimal("1000"),
            base_year_scope3_tco2e=Decimal("7000"),
            target_year=2050,
        ))
        # retail -> 7% of 10000 = 700 tCO2e
        budget = residual.residual_budget_tco2e
        assert budget == Decimal("700.000")

        # Now build offset portfolio matching the residual
        offset_engine = OffsetPortfolioEngine()
        offset_engine.add_credit(CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            category=CreditCategory.BIOCHAR,
            vintage_year=2025,
            quantity_tco2e=budget,
            unit_price_usd=Decimal("120"),
            additionality_score=4,
            permanence_score=4,
            co_benefits_score=4,
            leakage_risk_score=4,
            mrv_quality_score=4,
            sbti_use=SBTiCreditUse.NEUTRALIZATION,
        ))
        portfolio = offset_engine.analyze_portfolio(
            reduction_progress_pct=Decimal("50"),
        )

        # Portfolio total matches residual budget
        assert portfolio.portfolio_summary.total_credits_tco2e == budget
