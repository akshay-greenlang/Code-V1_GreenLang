# -*- coding: utf-8 -*-
"""
Tests for SupplierEngagementEngine - PACK-022 Engine 3

4-tier supplier engagement cascade with maturity scoring, engagement
program design, progress tracking, and Scope 3 impact estimation.

Coverage targets: 85%+ of SupplierEngagementEngine methods.
"""

import pytest
from decimal import Decimal

from engines.supplier_engagement_engine import (
    SupplierEngagementEngine,
    EngagementInput,
    EngagementResult,
    SupplierEntry,
    MaturityScores,
    TieredSupplier,
    EngagementPlan,
    ProgressSummary,
    CoverageMetrics,
    Scope3ImpactEstimate,
    SupplierTier,
    EngagementLevel,
    SupplierMaturity,
    ProgressStatus,
    TIER_THRESHOLDS,
    MATURITY_THRESHOLDS,
    EXPECTED_REDUCTION_RATES,
    SBTI_SET_COVERAGE_TARGET_PCT,
    RESOURCE_ESTIMATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_supplier(
    sid: str,
    name: str,
    emissions: Decimal,
    has_sbti: bool = False,
    maturity: MaturityScores = None,
    spend: Decimal = Decimal("0"),
    milestones_total: int = 0,
    milestones_completed: int = 0,
    engagement_start: str = None,
) -> SupplierEntry:
    """Create a SupplierEntry with defaults."""
    return SupplierEntry(
        supplier_id=sid,
        supplier_name=name,
        emissions_tco2e=emissions,
        spend_usd=spend,
        has_sbti_target=has_sbti,
        maturity_scores=maturity or MaturityScores(),
        milestones_total=milestones_total,
        milestones_completed=milestones_completed,
        engagement_start_date=engagement_start,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a SupplierEngagementEngine instance."""
    return SupplierEngagementEngine()


@pytest.fixture
def basic_suppliers():
    """Create a list of 5 suppliers with varied emissions."""
    return [
        _make_supplier("S1", "MajorSteel", Decimal("25000"), has_sbti=True),
        _make_supplier("S2", "BigChem", Decimal("15000")),
        _make_supplier("S3", "MidLogistics", Decimal("8000")),
        _make_supplier("S4", "SmallPaper", Decimal("3000")),
        _make_supplier("S5", "TinySupply", Decimal("1000")),
    ]


@pytest.fixture
def basic_input(basic_suppliers):
    """Basic engagement input with 5 suppliers."""
    return EngagementInput(
        entity_name="TestCorp",
        total_scope3_tco2e=Decimal("52000"),
        suppliers=basic_suppliers,
        target_coverage_pct=Decimal("67"),
        engagement_horizon_years=5,
        annual_budget_usd=Decimal("200000"),
        include_resource_plan=True,
    )


@pytest.fixture
def many_suppliers():
    """Create 30 suppliers with varied emissions for tier testing."""
    suppliers = []
    for i in range(30):
        emissions = Decimal(str(30000 - i * 900))
        suppliers.append(
            _make_supplier(
                f"S{i:03d}",
                f"Supplier_{i:03d}",
                max(emissions, Decimal("100")),
                has_sbti=(i < 5),  # top 5 have SBTi
            )
        )
    return suppliers


@pytest.fixture
def large_input(many_suppliers):
    """Engagement input with 30 suppliers."""
    total = sum(s.emissions_tco2e for s in many_suppliers)
    return EngagementInput(
        entity_name="LargeCorp",
        total_scope3_tco2e=total,
        suppliers=many_suppliers,
        target_coverage_pct=Decimal("67"),
        engagement_horizon_years=5,
        annual_budget_usd=Decimal("500000"),
    )


# ---------------------------------------------------------------------------
# TestEngagementBasic
# ---------------------------------------------------------------------------


class TestEngagementBasic:
    """Basic functionality tests for SupplierEngagementEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated."""
        engine = SupplierEngagementEngine()
        assert engine.engine_version == "1.0.0"

    def test_calculate_returns_result(self, engine, basic_input):
        """calculate() returns an EngagementResult."""
        result = engine.calculate(basic_input)
        assert isinstance(result, EngagementResult)

    def test_result_entity_name(self, engine, basic_input):
        """Result carries entity name."""
        result = engine.calculate(basic_input)
        assert result.entity_name == "TestCorp"

    def test_result_total_scope3(self, engine, basic_input):
        """Result carries total Scope 3 emissions."""
        result = engine.calculate(basic_input)
        assert float(result.total_scope3_tco2e) == pytest.approx(52000.0, rel=1e-3)

    def test_result_provenance_hash(self, engine, basic_input):
        """Result has 64-char hex provenance hash."""
        result = engine.calculate(basic_input)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_result_processing_time(self, engine, basic_input):
        """Result has positive processing time."""
        result = engine.calculate(basic_input)
        assert result.processing_time_ms > 0.0


# ---------------------------------------------------------------------------
# TestTierAssignment
# ---------------------------------------------------------------------------


class TestTierAssignment:
    """Tests for supplier tier assignment."""

    def test_all_suppliers_tiered(self, engine, basic_input):
        """All suppliers receive a tier assignment."""
        result = engine.calculate(basic_input)
        assert len(result.tiered_suppliers) == 5

    def test_suppliers_sorted_by_emissions(self, engine, basic_input):
        """Suppliers are sorted by emissions descending."""
        result = engine.calculate(basic_input)
        emissions = [float(s.emissions_tco2e) for s in result.tiered_suppliers]
        assert emissions == sorted(emissions, reverse=True)

    def test_tier_values_are_valid(self, engine, basic_input):
        """All tier values are valid enum values."""
        valid_tiers = {t.value for t in SupplierTier}
        result = engine.calculate(basic_input)
        for s in result.tiered_suppliers:
            assert s.tier in valid_tiers

    def test_top_supplier_is_critical(self, engine, basic_input):
        """Largest supplier by emissions is in CRITICAL tier."""
        result = engine.calculate(basic_input)
        top = result.tiered_suppliers[0]
        assert top.supplier_name == "MajorSteel"
        assert top.tier == SupplierTier.CRITICAL.value

    def test_emissions_pct_sums_to_100(self, engine, basic_input):
        """Emission percentages sum to approximately 100%."""
        result = engine.calculate(basic_input)
        total_pct = sum(float(s.emissions_pct_of_scope3) for s in result.tiered_suppliers)
        assert total_pct == pytest.approx(100.0, rel=0.05)

    def test_cumulative_pct_non_decreasing(self, engine, basic_input):
        """Cumulative emission percentage is non-decreasing."""
        result = engine.calculate(basic_input)
        prev = 0.0
        for s in result.tiered_suppliers:
            cum = float(s.cumulative_emissions_pct)
            assert cum >= prev - 0.01
            prev = cum

    def test_large_input_tiers_distributed(self, engine, large_input):
        """With 30 suppliers, multiple tiers are used."""
        result = engine.calculate(large_input)
        tiers = {s.tier for s in result.tiered_suppliers}
        assert len(tiers) >= 2


# ---------------------------------------------------------------------------
# TestTierDistribution
# ---------------------------------------------------------------------------


class TestTierDistribution:
    """Tests for tier distribution in result."""

    def test_tier_distribution_populated(self, engine, basic_input):
        """Tier distribution dict is populated."""
        result = engine.calculate(basic_input)
        assert len(result.tier_distribution) > 0

    def test_tier_distribution_sums_to_total(self, engine, basic_input):
        """Tier distribution sums to total supplier count."""
        result = engine.calculate(basic_input)
        total = sum(result.tier_distribution.values())
        assert total == len(result.tiered_suppliers)


# ---------------------------------------------------------------------------
# TestMaturityAssessment
# ---------------------------------------------------------------------------


class TestMaturityAssessment:
    """Tests for maturity scoring and classification."""

    def test_calculate_maturity_score_all_ones(self, engine):
        """All 1s scores yield minimum maturity score."""
        scores = MaturityScores(governance=1, data=1, targets=1, actions=1, disclosure=1)
        result = engine._calculate_maturity_score(scores)
        # (1+1+1+1+1)/25 * 100 = 20
        assert float(result) == pytest.approx(20.0, rel=1e-2)

    def test_calculate_maturity_score_all_fives(self, engine):
        """All 5s scores yield maximum maturity score."""
        scores = MaturityScores(governance=5, data=5, targets=5, actions=5, disclosure=5)
        result = engine._calculate_maturity_score(scores)
        # (5+5+5+5+5)/25 * 100 = 100
        assert float(result) == pytest.approx(100.0, rel=1e-2)

    def test_calculate_maturity_score_mixed(self, engine):
        """Mixed scores produce expected result."""
        scores = MaturityScores(governance=3, data=2, targets=4, actions=3, disclosure=3)
        result = engine._calculate_maturity_score(scores)
        # (3+2+4+3+3)/25 * 100 = 60
        assert float(result) == pytest.approx(60.0, rel=1e-2)

    def test_classify_maturity_leader(self, engine):
        """Score >= 80 is classified as leader."""
        assert engine._classify_maturity(Decimal("85")) == SupplierMaturity.LEADER.value

    def test_classify_maturity_advanced(self, engine):
        """Score >= 60 and < 80 is classified as advanced."""
        assert engine._classify_maturity(Decimal("65")) == SupplierMaturity.ADVANCED.value

    def test_classify_maturity_developing(self, engine):
        """Score >= 40 and < 60 is classified as developing."""
        assert engine._classify_maturity(Decimal("45")) == SupplierMaturity.DEVELOPING.value

    def test_classify_maturity_beginning(self, engine):
        """Score >= 20 and < 40 is classified as beginning."""
        assert engine._classify_maturity(Decimal("25")) == SupplierMaturity.BEGINNING.value

    def test_classify_maturity_unaware(self, engine):
        """Score < 20 is classified as unaware."""
        assert engine._classify_maturity(Decimal("10")) == SupplierMaturity.UNAWARE.value


# ---------------------------------------------------------------------------
# TestProgressAssessment
# ---------------------------------------------------------------------------


class TestProgressAssessment:
    """Tests for RAG status determination."""

    def test_rag_green(self, engine):
        """80%+ milestones is GREEN."""
        status = engine._determine_rag_status(8, 10, True)
        assert status == ProgressStatus.GREEN.value

    def test_rag_amber(self, engine):
        """50-79% milestones is AMBER."""
        status = engine._determine_rag_status(6, 10, True)
        assert status == ProgressStatus.AMBER.value

    def test_rag_red(self, engine):
        """<50% milestones is RED."""
        status = engine._determine_rag_status(3, 10, True)
        assert status == ProgressStatus.RED.value

    def test_rag_not_started_no_engagement(self, engine):
        """No engagement returns NOT_STARTED."""
        status = engine._determine_rag_status(0, 0, False)
        assert status == ProgressStatus.NOT_STARTED.value

    def test_rag_not_started_zero_milestones(self, engine):
        """Zero total milestones returns NOT_STARTED."""
        status = engine._determine_rag_status(0, 0, True)
        assert status == ProgressStatus.NOT_STARTED.value


# ---------------------------------------------------------------------------
# TestEngagementPlans
# ---------------------------------------------------------------------------


class TestEngagementPlans:
    """Tests for engagement plan generation."""

    def test_plans_generated(self, engine, basic_input):
        """Engagement plans are generated."""
        result = engine.calculate(basic_input)
        assert len(result.engagement_plans) > 0

    def test_plans_have_milestones(self, engine, basic_input):
        """Each plan has milestones."""
        result = engine.calculate(basic_input)
        for plan in result.engagement_plans:
            assert len(plan.milestones) >= 2  # at least base milestones

    def test_critical_tier_has_most_milestones(self, engine, basic_input):
        """Critical tier plan has the most milestones."""
        result = engine.calculate(basic_input)
        critical_plan = [p for p in result.engagement_plans if p.tier == "critical"]
        if critical_plan:
            basic_plan = [p for p in result.engagement_plans if p.tier == "basic"]
            if basic_plan:
                assert len(critical_plan[0].milestones) > len(basic_plan[0].milestones)

    def test_plans_resource_hours_positive(self, engine, basic_input):
        """Plans have positive resource hours."""
        result = engine.calculate(basic_input)
        for plan in result.engagement_plans:
            assert float(plan.resource_hours_per_year) > 0

    def test_plans_cost_positive(self, engine, basic_input):
        """Plans have positive estimated cost."""
        result = engine.calculate(basic_input)
        for plan in result.engagement_plans:
            assert float(plan.estimated_cost_per_year_usd) > 0


# ---------------------------------------------------------------------------
# TestCoverageMetrics
# ---------------------------------------------------------------------------


class TestCoverageMetrics:
    """Tests for SBTi SET coverage metrics."""

    def test_coverage_metrics_present(self, engine, basic_input):
        """Coverage metrics are populated."""
        result = engine.calculate(basic_input)
        assert isinstance(result.coverage_metrics, CoverageMetrics)

    def test_coverage_reflects_sbti_suppliers(self, engine, basic_input):
        """Coverage counts suppliers with SBTi targets."""
        result = engine.calculate(basic_input)
        # S1 has SBTi target, 25000/52000 ~ 48.08%
        assert float(result.coverage_metrics.current_coverage_by_emissions_pct) > 0

    def test_coverage_on_track_false_when_below_target(self, engine, basic_input):
        """Coverage is not on track when below target."""
        result = engine.calculate(basic_input)
        # 48% < 67%
        assert result.coverage_metrics.on_track is False

    def test_coverage_on_track_when_all_sbti(self, engine):
        """Coverage is on track when all suppliers have SBTi targets."""
        suppliers = [
            _make_supplier("S1", "Sup1", Decimal("5000"), has_sbti=True),
            _make_supplier("S2", "Sup2", Decimal("3000"), has_sbti=True),
        ]
        inp = EngagementInput(
            entity_name="AllSBTi",
            total_scope3_tco2e=Decimal("8000"),
            suppliers=suppliers,
        )
        result = engine.calculate(inp)
        assert result.coverage_metrics.on_track is True

    def test_gap_pct_correct(self, engine, basic_input):
        """Coverage gap is correctly computed."""
        result = engine.calculate(basic_input)
        gap = float(result.coverage_metrics.gap_pct)
        coverage = float(result.coverage_metrics.current_coverage_by_emissions_pct)
        target = float(result.coverage_metrics.target_coverage_pct)
        assert gap == pytest.approx(max(0, target - coverage), abs=0.5)


# ---------------------------------------------------------------------------
# TestScope3Impact
# ---------------------------------------------------------------------------


class TestScope3Impact:
    """Tests for Scope 3 impact estimation."""

    def test_impact_present(self, engine, basic_input):
        """Scope 3 impact is populated."""
        result = engine.calculate(basic_input)
        assert isinstance(result.scope3_impact, Scope3ImpactEstimate)

    def test_annual_reduction_non_negative(self, engine, basic_input):
        """Annual reduction is non-negative."""
        result = engine.calculate(basic_input)
        assert float(result.scope3_impact.annual_reduction_tco2e) >= 0

    def test_five_year_cumulative_positive(self, engine, basic_input):
        """Five-year cumulative reduction is positive."""
        result = engine.calculate(basic_input)
        assert float(result.scope3_impact.five_year_cumulative_tco2e) >= 0

    def test_reduction_by_tier_populated(self, engine, basic_input):
        """Reduction by tier has entries."""
        result = engine.calculate(basic_input)
        assert len(result.scope3_impact.reduction_by_tier) > 0


# ---------------------------------------------------------------------------
# TestRecommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_generated(self, engine, basic_input):
        """Recommendations are generated."""
        result = engine.calculate(basic_input)
        assert len(result.recommendations) > 0

    def test_coverage_gap_recommendation(self, engine, basic_input):
        """Coverage gap produces a recommendation."""
        result = engine.calculate(basic_input)
        assert any("coverage" in r.lower() or "set" in r.lower() for r in result.recommendations)


# ---------------------------------------------------------------------------
# TestUtilityMethods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for engine utility methods."""

    def test_assess_supplier_maturity(self, engine):
        """assess_supplier_maturity returns correct structure."""
        scores = MaturityScores(governance=4, data=3, targets=5, actions=4, disclosure=3)
        result = engine.assess_supplier_maturity(scores)
        assert "score" in result
        assert "classification" in result
        assert "recommendations" in result
        assert "provenance_hash" in result

    def test_assess_supplier_maturity_leader(self, engine):
        """High scores yield leader classification."""
        scores = MaturityScores(governance=5, data=5, targets=5, actions=5, disclosure=5)
        result = engine.assess_supplier_maturity(scores)
        assert result["classification"] == SupplierMaturity.LEADER.value

    def test_assess_supplier_maturity_recommendations_for_low(self, engine):
        """Low dimension scores produce recommendations."""
        scores = MaturityScores(governance=1, data=1, targets=1, actions=1, disclosure=1)
        result = engine.assess_supplier_maturity(scores)
        assert len(result["recommendations"]) >= 5

    def test_get_tier_requirements(self, engine):
        """get_tier_requirements returns all 4 tiers."""
        reqs = engine.get_tier_requirements()
        assert len(reqs) == 4
        assert SupplierTier.CRITICAL.value in reqs

    def test_get_summary(self, engine, basic_input):
        """get_summary returns expected structure."""
        result = engine.calculate(basic_input)
        summary = engine.get_summary(result)
        assert summary["entity_name"] == "TestCorp"
        assert "coverage_pct" in summary
        assert "provenance_hash" in summary


# ---------------------------------------------------------------------------
# TestProgressSummary
# ---------------------------------------------------------------------------


class TestProgressSummary:
    """Tests for progress summary."""

    def test_progress_summary_present(self, engine, basic_input):
        """Progress summary is populated."""
        result = engine.calculate(basic_input)
        assert isinstance(result.progress_summary, ProgressSummary)

    def test_total_suppliers_correct(self, engine, basic_input):
        """Total suppliers count matches input."""
        result = engine.calculate(basic_input)
        assert result.progress_summary.total_suppliers == 5

    def test_sbti_count_correct(self, engine, basic_input):
        """SBTi count matches suppliers with targets."""
        result = engine.calculate(basic_input)
        assert result.progress_summary.suppliers_with_sbti == 1


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_single_supplier(self, engine):
        """Works with a single supplier."""
        suppliers = [_make_supplier("S1", "OnlyOne", Decimal("10000"))]
        inp = EngagementInput(
            entity_name="Solo",
            total_scope3_tco2e=Decimal("10000"),
            suppliers=suppliers,
        )
        result = engine.calculate(inp)
        assert len(result.tiered_suppliers) == 1

    def test_all_zero_emissions_suppliers(self, engine):
        """Suppliers with zero emissions are handled."""
        suppliers = [
            _make_supplier("S1", "Zero1", Decimal("0")),
            _make_supplier("S2", "Zero2", Decimal("0")),
        ]
        inp = EngagementInput(
            entity_name="ZeroCo",
            total_scope3_tco2e=Decimal("1"),
            suppliers=suppliers,
        )
        result = engine.calculate(inp)
        assert len(result.tiered_suppliers) == 2

    def test_budget_warning_recommendation(self, engine):
        """Budget shortfall generates a recommendation."""
        suppliers = [
            _make_supplier("S1", "BigOne", Decimal("50000")),
            _make_supplier("S2", "MedOne", Decimal("30000")),
            _make_supplier("S3", "SmlOne", Decimal("20000")),
        ]
        inp = EngagementInput(
            entity_name="BudgetCo",
            total_scope3_tco2e=Decimal("100000"),
            suppliers=suppliers,
            annual_budget_usd=Decimal("100"),  # very low budget
        )
        result = engine.calculate(inp)
        assert any("budget" in r.lower() or "cost" in r.lower() for r in result.recommendations)


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum definitions."""

    def test_supplier_tier_values(self):
        """SupplierTier has 4 values."""
        assert len(SupplierTier) == 4
        assert SupplierTier.CRITICAL.value == "critical"
        assert SupplierTier.BASIC.value == "basic"

    def test_engagement_level_values(self):
        """EngagementLevel has 4 values."""
        assert len(EngagementLevel) == 4
        assert EngagementLevel.COLLABORATE.value == "collaborate"
        assert EngagementLevel.INFORM.value == "inform"

    def test_supplier_maturity_values(self):
        """SupplierMaturity has 5 values."""
        assert len(SupplierMaturity) == 5

    def test_progress_status_values(self):
        """ProgressStatus has 4 values."""
        assert len(ProgressStatus) == 4
        assert ProgressStatus.GREEN.value == "green"
        assert ProgressStatus.NOT_STARTED.value == "not_started"


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for Pydantic input validation."""

    def test_empty_suppliers_rejected(self):
        """Empty suppliers list is rejected."""
        with pytest.raises(Exception):
            EngagementInput(
                entity_name="Bad",
                total_scope3_tco2e=Decimal("10000"),
                suppliers=[],
            )

    def test_zero_scope3_rejected(self):
        """Zero total_scope3_tco2e is rejected."""
        with pytest.raises(Exception):
            EngagementInput(
                entity_name="Bad",
                total_scope3_tco2e=Decimal("0"),
                suppliers=[_make_supplier("S1", "X", Decimal("100"))],
            )

    def test_empty_entity_name_rejected(self):
        """Empty entity name is rejected."""
        with pytest.raises(Exception):
            EngagementInput(
                entity_name="",
                total_scope3_tco2e=Decimal("10000"),
                suppliers=[_make_supplier("S1", "X", Decimal("100"))],
            )
