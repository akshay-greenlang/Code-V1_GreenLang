# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Financial Materiality Engine Tests
==================================================================================

Unit tests for FinancialMaterialityEngine (Engine 2) covering financial score
calculation, batch assessment, ranking, KPI mapping, risk/opportunity
classification, aggregate exposure, provenance hashing, and edge cases.

ESRS 1 Para 49-51: financial_score = magnitude_w * likelihood_w * time_horizon_w

Target: 45+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the financial_materiality engine module."""
    return _load_engine("financial_materiality")


@pytest.fixture
def engine(mod):
    """Create a fresh FinancialMaterialityEngine instance."""
    return mod.FinancialMaterialityEngine()


@pytest.fixture
def sample_financial_impact(mod):
    """Create a sample FinancialImpact for testing."""
    return mod.FinancialImpact(
        matter_id="FM-001",
        matter_name="Carbon Pricing Risk",
        esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        magnitude=4,
        likelihood=4,
        time_horizon=mod.TimeHorizon.SHORT_TERM,
        risk_or_opportunity=mod.RiskOrOpportunity.RISK,
        affected_resources=[mod.AffectedResource.COST, mod.AffectedResource.CAPITAL],
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestFinancialMaterialityEnums:
    """Tests for financial materiality enums."""

    def test_time_horizon_values(self, mod):
        """TimeHorizon has 3 values."""
        assert len(mod.TimeHorizon) == 3

    def test_financial_magnitude_values(self, mod):
        """FinancialMagnitude has 5 levels (1-5)."""
        assert len(mod.FinancialMagnitude) == 5
        int_values = {m.value for m in mod.FinancialMagnitude}
        assert int_values == {1, 2, 3, 4, 5}

    def test_financial_likelihood_values(self, mod):
        """FinancialLikelihood has 5 levels (1-5)."""
        assert len(mod.FinancialLikelihood) == 5

    def test_affected_resource_values(self, mod):
        """AffectedResource has 6 values."""
        assert len(mod.AffectedResource) == 6
        values = {m.value for m in mod.AffectedResource}
        expected = {
            "revenue", "cost", "assets", "liabilities",
            "capital", "access_to_finance",
        }
        assert values == expected

    def test_risk_or_opportunity_values(self, mod):
        """RiskOrOpportunity has 3 values: RISK, OPPORTUNITY, BOTH."""
        assert len(mod.RiskOrOpportunity) == 3
        values = {m.value for m in mod.RiskOrOpportunity}
        assert values == {"risk", "opportunity", "both"}

    def test_esrs_topic_values(self, mod):
        """ESRSTopic has 10 topics."""
        assert len(mod.ESRSTopic) == 10


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestFinancialMaterialityConstants:
    """Tests for financial materiality constants."""

    def test_time_horizon_weights_financial(self, mod):
        """TIME_HORIZON_WEIGHTS: short=1.00, medium=0.80, long=0.60."""
        assert mod.TIME_HORIZON_WEIGHTS["short_term"] == Decimal("1.00")
        assert mod.TIME_HORIZON_WEIGHTS["medium_term"] == Decimal("0.80")
        assert mod.TIME_HORIZON_WEIGHTS["long_term"] == Decimal("0.60")

    def test_magnitude_weights(self, mod):
        """MAGNITUDE_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.MAGNITUDE_WEIGHTS[1] == Decimal("0.20")
        assert mod.MAGNITUDE_WEIGHTS[3] == Decimal("0.60")
        assert mod.MAGNITUDE_WEIGHTS[5] == Decimal("1.00")

    def test_likelihood_weights(self, mod):
        """LIKELIHOOD_WEIGHTS maps 1-5 to Decimal 0.20-1.00."""
        assert mod.LIKELIHOOD_WEIGHTS[1] == Decimal("0.20")
        assert mod.LIKELIHOOD_WEIGHTS[5] == Decimal("1.00")

    def test_default_financial_threshold(self, mod):
        """Default financial threshold is 0.40."""
        assert mod.DEFAULT_FINANCIAL_THRESHOLD == Decimal("0.40")

    def test_financial_kpi_map_has_10_topics(self, mod):
        """FINANCIAL_KPI_MAP has entries for all 10 ESRS topics."""
        assert len(mod.FINANCIAL_KPI_MAP) == 10
        assert "e1_climate" in mod.FINANCIAL_KPI_MAP
        assert "g1_business_conduct" in mod.FINANCIAL_KPI_MAP

    def test_financial_kpi_map_non_empty(self, mod):
        """Each topic in FINANCIAL_KPI_MAP has at least one KPI."""
        for topic, kpis in mod.FINANCIAL_KPI_MAP.items():
            assert len(kpis) >= 1, f"Topic {topic} has no KPIs"


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestFinancialImpactModel:
    """Tests for FinancialImpact Pydantic model."""

    def test_create_valid_financial_impact(self, mod):
        """Create a valid FinancialImpact."""
        fi = mod.FinancialImpact(
            matter_id="FI-001",
            magnitude=3,
            likelihood=3,
        )
        assert fi.magnitude == 3
        assert fi.likelihood == 3
        assert fi.time_horizon == mod.TimeHorizon.SHORT_TERM

    def test_magnitude_out_of_range(self, mod):
        """Magnitude outside 1-5 is rejected."""
        with pytest.raises(Exception):
            mod.FinancialImpact(
                matter_id="FI-001", magnitude=0, likelihood=3,
            )
        with pytest.raises(Exception):
            mod.FinancialImpact(
                matter_id="FI-001", magnitude=6, likelihood=3,
            )

    def test_likelihood_out_of_range(self, mod):
        """Likelihood outside 1-5 is rejected."""
        with pytest.raises(Exception):
            mod.FinancialImpact(
                matter_id="FI-001", magnitude=3, likelihood=0,
            )

    def test_financial_impact_matter_id_required(self, mod):
        """FinancialImpact requires matter_id."""
        with pytest.raises(Exception):
            mod.FinancialImpact(
                matter_id="", magnitude=3, likelihood=3,
            )


# ===========================================================================
# Financial Score Calculation Tests
# ===========================================================================


class TestFinancialScoreCalculation:
    """Tests for calculate_financial_score method."""

    def test_score_all_fives_short_term(self, engine, mod):
        """All 5s short term: 1.0 * 1.0 * 1.0 = 1.0."""
        score = engine.calculate_financial_score(
            magnitude=5, likelihood=5,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        assert isinstance(score, Decimal)
        assert score == pytest.approx(Decimal("1.00"), abs=Decimal("0.01"))

    def test_score_all_ones_short_term(self, engine, mod):
        """All 1s short term: 0.20 * 0.20 * 1.0 = 0.04."""
        score = engine.calculate_financial_score(
            magnitude=1, likelihood=1,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        assert score == pytest.approx(Decimal("0.04"), abs=Decimal("0.01"))

    def test_score_medium_term_discount(self, engine, mod):
        """Medium term applies 0.80 discount."""
        score = engine.calculate_financial_score(
            magnitude=5, likelihood=5,
            time_horizon=mod.TimeHorizon.MEDIUM_TERM,
        )
        # 1.0 * 1.0 * 0.80 = 0.80
        assert score == pytest.approx(Decimal("0.80"), abs=Decimal("0.01"))

    def test_score_long_term_discount(self, engine, mod):
        """Long term applies 0.60 discount."""
        score = engine.calculate_financial_score(
            magnitude=5, likelihood=5,
            time_horizon=mod.TimeHorizon.LONG_TERM,
        )
        # 1.0 * 1.0 * 0.60 = 0.60
        assert score == pytest.approx(Decimal("0.60"), abs=Decimal("0.01"))

    def test_score_deterministic(self, engine, mod):
        """Same inputs produce identical scores."""
        s1 = engine.calculate_financial_score(
            magnitude=3, likelihood=4,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        s2 = engine.calculate_financial_score(
            magnitude=3, likelihood=4,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        assert s1 == s2

    @pytest.mark.parametrize("mag,lik,expected_approx", [
        (1, 1, Decimal("0.040")),
        (2, 2, Decimal("0.160")),
        (3, 3, Decimal("0.360")),
        (4, 4, Decimal("0.640")),
        (5, 5, Decimal("1.000")),
    ])
    def test_score_parametric_short_term(self, engine, mod, mag, lik, expected_approx):
        """Parametric test for financial scores at short term."""
        score = engine.calculate_financial_score(
            magnitude=mag, likelihood=lik,
            time_horizon=mod.TimeHorizon.SHORT_TERM,
        )
        assert score == pytest.approx(expected_approx, abs=Decimal("0.01"))


# ===========================================================================
# Assess Financial Impact Tests
# ===========================================================================


class TestAssessFinancialImpact:
    """Tests for assess_financial_impact method."""

    def test_assess_basic(self, engine, sample_financial_impact):
        """Basic assessment returns valid FinancialMaterialityResult."""
        result = engine.assess_financial_impact(sample_financial_impact)
        assert result.matter_id == "FM-001"
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_assess_materiality_determined(self, engine, sample_financial_impact):
        """High magnitude/likelihood results in is_material=True."""
        result = engine.assess_financial_impact(sample_financial_impact)
        # mag=4, lik=4, short => 0.80*0.80*1.00 = 0.64 > 0.40
        assert result.is_material is True

    def test_assess_low_score_not_material(self, engine, mod):
        """Low magnitude/likelihood results in is_material=False."""
        fi = mod.FinancialImpact(
            matter_id="FM-LOW",
            magnitude=1, likelihood=1,
            time_horizon=mod.TimeHorizon.LONG_TERM,
        )
        result = engine.assess_financial_impact(fi)
        # 0.20 * 0.20 * 0.60 = 0.024 < 0.40
        assert result.is_material is False

    def test_assess_custom_threshold(self, engine, sample_financial_impact):
        """Custom threshold changes materiality determination."""
        result = engine.assess_financial_impact(
            sample_financial_impact, threshold=Decimal("0.90")
        )
        # 0.64 < 0.90
        assert result.is_material is False

    def test_assess_processing_time(self, engine, sample_financial_impact):
        """Processing time is recorded."""
        result = engine.assess_financial_impact(sample_financial_impact)
        assert result.processing_time_ms >= 0.0


# ===========================================================================
# Batch Assessment Tests
# ===========================================================================


class TestBatchFinancialAssessment:
    """Tests for batch_assess method."""

    def test_batch_assess_basic(self, engine, mod):
        """Batch assess processes multiple impacts."""
        impacts = [
            mod.FinancialImpact(
                matter_id=f"BF-{i}", magnitude=i + 1, likelihood=i + 1,
            )
            for i in range(3)
        ]
        result = engine.batch_assess(impacts)
        assert len(result.results) == 3

    def test_batch_assess_empty_raises(self, engine):
        """Empty impacts list raises ValueError."""
        with pytest.raises(ValueError):
            engine.batch_assess([])

    def test_batch_assess_ranking(self, engine, mod):
        """Results are ranked by score."""
        impacts = [
            mod.FinancialImpact(
                matter_id="BF-LOW", magnitude=1, likelihood=1,
            ),
            mod.FinancialImpact(
                matter_id="BF-HIGH", magnitude=5, likelihood=5,
            ),
        ]
        result = engine.batch_assess(impacts)
        rankings = [r.ranking for r in result.results]
        assert 1 in rankings


# ===========================================================================
# KPI and Classification Tests
# ===========================================================================


class TestKPIAndClassification:
    """Tests for KPI mapping and risk/opportunity classification."""

    def test_get_affected_kpis_climate(self, engine, mod):
        """Get KPIs for E1 climate topic."""
        kpis = engine.get_affected_kpis(mod.ESRSTopic.E1_CLIMATE)
        assert isinstance(kpis, list)
        assert len(kpis) >= 3
        assert "carbon_pricing_costs" in kpis

    def test_get_affected_kpis_all_topics(self, engine, mod):
        """All ESRS topics return non-empty KPI lists."""
        for topic in mod.ESRSTopic:
            kpis = engine.get_affected_kpis(topic)
            assert len(kpis) >= 1, f"Topic {topic} has no KPIs"

    def test_classify_risk_opportunity(self, engine, mod):
        """classify_risk_opportunity groups by RISK/OPPORTUNITY/BOTH."""
        # classify_risk_opportunity takes List[FinancialMaterialityResult]
        impacts = [
            mod.FinancialImpact(
                matter_id="C-1", magnitude=3, likelihood=3,
                risk_or_opportunity=mod.RiskOrOpportunity.RISK,
            ),
            mod.FinancialImpact(
                matter_id="C-2", magnitude=3, likelihood=3,
                risk_or_opportunity=mod.RiskOrOpportunity.OPPORTUNITY,
            ),
            mod.FinancialImpact(
                matter_id="C-3", magnitude=3, likelihood=3,
                risk_or_opportunity=mod.RiskOrOpportunity.BOTH,
            ),
        ]
        # First assess to get FinancialMaterialityResult objects
        results = [engine.assess_financial_impact(imp) for imp in impacts]
        classified = engine.classify_risk_opportunity(results)
        assert isinstance(classified, dict)
        assert "risks" in classified
        assert "opportunities" in classified
        assert "both" in classified


# ===========================================================================
# Aggregate Exposure Tests
# ===========================================================================


class TestAggregateExposure:
    """Tests for calculate_aggregate_exposure."""

    def test_aggregate_exposure_basic(self, engine, mod):
        """Aggregate exposure returns a numeric summary."""
        # calculate_aggregate_exposure takes List[FinancialMaterialityResult]
        impacts = [
            mod.FinancialImpact(
                matter_id=f"AE-{i}", magnitude=3, likelihood=3,
                estimated_financial_range_low_eur=100000.0,
                estimated_financial_range_high_eur=500000.0,
            )
            for i in range(3)
        ]
        # First assess to get results
        results = [engine.assess_financial_impact(imp) for imp in impacts]
        exposure = engine.calculate_aggregate_exposure(results)
        assert exposure is not None
        assert isinstance(exposure, dict)
        assert "material_matters_count" in exposure


# ===========================================================================
# Interpret Score Tests
# ===========================================================================


class TestFinancialInterpretScore:
    """Tests for financial score interpretation."""

    def test_interpret_high_score(self, engine):
        """High score produces meaningful label."""
        label = engine.interpret_score(Decimal("0.90"))
        assert isinstance(label, str)
        assert len(label) > 0

    def test_interpret_low_score(self, engine):
        """Low score produces meaningful label."""
        label = engine.interpret_score(Decimal("0.10"))
        assert isinstance(label, str)
        assert len(label) > 0

    @pytest.mark.parametrize("score", [
        Decimal("0.05"), Decimal("0.25"), Decimal("0.45"),
        Decimal("0.65"), Decimal("0.85"), Decimal("1.00"),
    ])
    def test_interpret_score_returns_string(self, engine, score):
        """All valid scores return non-empty strings."""
        label = engine.interpret_score(score)
        assert isinstance(label, str)
        assert len(label) > 0


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestFinancialProvenanceHash:
    """Tests for financial materiality provenance hashing."""

    def test_hash_is_64_chars(self, engine, sample_financial_impact):
        """Provenance hash is a 64-character SHA-256 hex string."""
        result = engine.assess_financial_impact(sample_financial_impact)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Verify valid hex

    def test_hash_score_determinism(self, engine, mod):
        """Same inputs produce identical financial scores."""
        fi = mod.FinancialImpact(
            matter_id="DET-FM", magnitude=4, likelihood=3,
        )
        r1 = engine.assess_financial_impact(fi)
        r2 = engine.assess_financial_impact(fi)
        assert r1.financial_score == r2.financial_score


# ===========================================================================
# Score Breakdown Tests
# ===========================================================================


class TestFinancialScoreBreakdown:
    """Tests for get_score_breakdown method."""

    def test_score_breakdown_returns_dict(self, engine, sample_financial_impact):
        """get_score_breakdown returns a dictionary."""
        result = engine.assess_financial_impact(sample_financial_impact)
        breakdown = engine.get_score_breakdown(result)
        assert isinstance(breakdown, dict)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestFinancialEdgeCases:
    """Edge case tests for financial materiality engine."""

    def test_borderline_materiality(self, engine, mod):
        """Score near threshold boundary is handled correctly."""
        # Find a combination that gives score close to 0.40
        fi = mod.FinancialImpact(
            matter_id="BORDER-1", magnitude=2, likelihood=3,
            time_horizon=mod.TimeHorizon.MEDIUM_TERM,
        )
        result = engine.assess_financial_impact(fi)
        # 0.40 * 0.60 * 0.80 = 0.192 < 0.40
        assert result.financial_score < Decimal("0.40")
        assert result.is_material is False
