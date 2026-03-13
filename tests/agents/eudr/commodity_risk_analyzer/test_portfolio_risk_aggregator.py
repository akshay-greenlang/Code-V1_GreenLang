# -*- coding: utf-8 -*-
"""
Unit tests for PortfolioRiskAggregator (AGENT-EUDR-018 Engine 8).

Tests portfolio analysis, HHI concentration index, diversification scoring,
total risk exposure, VaR calculation, scenario simulation, correlation
matrix, diversification recommendations, and portfolio comparison for
EUDR commodity portfolios.

Coverage target: 85%+
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.engines.portfolio_risk_aggregator import (
    PortfolioRiskAggregator,
    EUDR_COMMODITIES,
    COMMODITY_RISK_PARAMETERS,
    RISK_CORRELATIONS,
    SCENARIO_IMPACTS,
    HHI_UNCONCENTRATED,
    HHI_MODERATE,
    HHI_MAX,
    VALID_SCENARIO_TYPES,
    VAR_Z_SCORES,
    CommodityPosition,
    PortfolioSummary,
)

SEVEN_COMMODITIES = sorted(EUDR_COMMODITIES)


# ---------------------------------------------------------------------------
# Helpers / sample data
# ---------------------------------------------------------------------------


def _balanced_portfolio():
    """Return a balanced 3-commodity portfolio."""
    return [
        {"commodity": "soya", "weight": 0.34, "exposure_value": 340000, "supplier_count": 5, "origin_countries": ["BR"]},
        {"commodity": "cocoa", "weight": 0.33, "exposure_value": 330000, "supplier_count": 4, "origin_countries": ["CI"]},
        {"commodity": "coffee", "weight": 0.33, "exposure_value": 330000, "supplier_count": 3, "origin_countries": ["CO"]},
    ]


def _concentrated_portfolio():
    """Return a single-commodity portfolio (maximum HHI)."""
    return [
        {"commodity": "oil_palm", "weight": 1.0, "exposure_value": 1000000, "supplier_count": 2, "origin_countries": ["ID"]},
    ]


def _diverse_portfolio():
    """Return an equally-weighted 7-commodity portfolio."""
    w = round(1.0 / 7, 4)
    return [
        {"commodity": c, "weight": w, "exposure_value": 100000, "supplier_count": 2, "origin_countries": ["BR"]}
        for c in SEVEN_COMMODITIES
    ]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for PortfolioRiskAggregator initialization."""

    @pytest.mark.unit
    def test_init_empty_cache(self):
        """Aggregator initializes with empty portfolio cache."""
        agg = PortfolioRiskAggregator()
        assert agg._portfolio_cache == {}

    @pytest.mark.unit
    def test_init_creates_lock(self):
        """Aggregator creates a reentrant lock."""
        agg = PortfolioRiskAggregator()
        assert agg._lock is not None


# ---------------------------------------------------------------------------
# TestAnalyzePortfolio
# ---------------------------------------------------------------------------

class TestAnalyzePortfolio:
    """Tests for analyze_portfolio method."""

    @pytest.mark.unit
    def test_analyze_balanced(self, portfolio_risk_aggregator):
        """Balanced portfolio analysis returns all expected fields."""
        result = portfolio_risk_aggregator.analyze_portfolio(_balanced_portfolio())
        assert "hhi" in result
        assert "diversification_score" in result
        assert "weighted_risk_score" in result
        assert "var_95" in result
        assert result["commodity_count"] == 3

    @pytest.mark.unit
    def test_analyze_single_commodity(self, portfolio_risk_aggregator):
        """Single commodity yields HHI = 10000."""
        result = portfolio_risk_aggregator.analyze_portfolio(_concentrated_portfolio())
        assert Decimal(result["hhi"]) == Decimal("10000.00")
        assert result["concentration_level"] == "HIGHLY_CONCENTRATED"

    @pytest.mark.unit
    def test_analyze_empty_raises(self, portfolio_risk_aggregator):
        """Empty positions list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            portfolio_risk_aggregator.analyze_portfolio([])

    @pytest.mark.unit
    def test_analyze_invalid_commodity_raises(self, portfolio_risk_aggregator):
        """Invalid commodity in position raises ValueError."""
        positions = [{"commodity": "banana", "weight": 1.0, "exposure_value": 100}]
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            portfolio_risk_aggregator.analyze_portfolio(positions)


# ---------------------------------------------------------------------------
# TestConcentrationIndex
# ---------------------------------------------------------------------------

class TestConcentrationIndex:
    """Tests for calculate_concentration_index (HHI)."""

    @pytest.mark.unit
    def test_single_commodity_hhi_10000(self, portfolio_risk_aggregator):
        """Single commodity -> HHI = 10000."""
        hhi = portfolio_risk_aggregator.calculate_concentration_index(_concentrated_portfolio())
        assert hhi == Decimal("10000.00")

    @pytest.mark.unit
    def test_balanced_three_lower_hhi(self, portfolio_risk_aggregator):
        """Three balanced commodities -> HHI around 3333."""
        hhi = portfolio_risk_aggregator.calculate_concentration_index(_balanced_portfolio())
        # ~33.33% each -> 33.33^2 * 3 ~ 3333
        assert hhi < Decimal("4000")
        assert hhi > Decimal("3000")

    @pytest.mark.unit
    def test_seven_equal_lowest_hhi(self, portfolio_risk_aggregator):
        """Seven equal commodities -> HHI ~ 1428 (unconcentrated)."""
        hhi = portfolio_risk_aggregator.calculate_concentration_index(_diverse_portfolio())
        assert hhi < HHI_UNCONCENTRATED


# ---------------------------------------------------------------------------
# TestDiversificationScore
# ---------------------------------------------------------------------------

class TestDiversificationScore:
    """Tests for calculate_diversification_score."""

    @pytest.mark.unit
    def test_single_commodity_zero_diversification(self, portfolio_risk_aggregator):
        """Single commodity -> diversification score 0."""
        score = portfolio_risk_aggregator.calculate_diversification_score(
            _concentrated_portfolio(),
        )
        assert score == Decimal("0")

    @pytest.mark.unit
    def test_diverse_portfolio_high_score(self, portfolio_risk_aggregator):
        """Seven equal commodities -> high diversification score."""
        score = portfolio_risk_aggregator.calculate_diversification_score(
            _diverse_portfolio(),
        )
        assert score > Decimal("80")

    @pytest.mark.unit
    def test_diversification_in_range(self, portfolio_risk_aggregator):
        """Score is between 0 and 100."""
        score = portfolio_risk_aggregator.calculate_diversification_score(
            _balanced_portfolio(),
        )
        assert Decimal("0") <= score <= Decimal("100")


# ---------------------------------------------------------------------------
# TestTotalRiskExposure
# ---------------------------------------------------------------------------

class TestTotalRiskExposure:
    """Tests for calculate_total_risk_exposure."""

    @pytest.mark.unit
    def test_exposure_positive(self, portfolio_risk_aggregator):
        """Total risk exposure is positive for a portfolio with exposure."""
        exposure = portfolio_risk_aggregator.calculate_total_risk_exposure(
            _balanced_portfolio(),
        )
        assert exposure > Decimal("0")

    @pytest.mark.unit
    def test_exposure_formula(self, portfolio_risk_aggregator):
        """Exposure = sum(exposure_value * risk_score / 100)."""
        positions = [
            {"commodity": "soya", "weight": 1.0, "exposure_value": 100000, "risk_score": 50},
        ]
        exposure = portfolio_risk_aggregator.calculate_total_risk_exposure(positions)
        # 100000 * 50 / 100 = 50000
        assert exposure == Decimal("50000.00")


# ---------------------------------------------------------------------------
# TestPortfolioSummary
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    """Tests for get_portfolio_summary."""

    @pytest.mark.unit
    def test_no_cache_returns_status(self, portfolio_risk_aggregator):
        """No cached analysis returns no_portfolios_analyzed status."""
        result = portfolio_risk_aggregator.get_portfolio_summary()
        assert result["status"] == "no_portfolios_analyzed"

    @pytest.mark.unit
    def test_summary_after_analysis(self, portfolio_risk_aggregator):
        """Summary returns the most recent analysis after analyze_portfolio."""
        portfolio_risk_aggregator.analyze_portfolio(_balanced_portfolio())
        result = portfolio_risk_aggregator.get_portfolio_summary()
        assert "hhi" in result

    @pytest.mark.unit
    def test_summary_not_found_name(self, portfolio_risk_aggregator):
        """Non-existent portfolio name returns not_found status."""
        result = portfolio_risk_aggregator.get_portfolio_summary(portfolio_name="nonexistent")
        assert result["status"] == "portfolio_not_found"


# ---------------------------------------------------------------------------
# TestScenarioSimulation
# ---------------------------------------------------------------------------

class TestScenarioSimulation:
    """Tests for simulate_scenario."""

    @pytest.mark.unit
    def test_price_shock(self, portfolio_risk_aggregator):
        """Price shock increases risk."""
        result = portfolio_risk_aggregator.simulate_scenario(
            _balanced_portfolio(),
            "price_shock",
            {"magnitude": "-0.20"},
        )
        assert result["scenario_type"] == "price_shock"
        assert Decimal(result["delta"]["risk_change"]) > Decimal("0")

    @pytest.mark.unit
    def test_supply_disruption(self, portfolio_risk_aggregator):
        """Supply disruption on specific commodity increases exposure."""
        result = portfolio_risk_aggregator.simulate_scenario(
            _balanced_portfolio(),
            "supply_disruption",
            {"affected_commodity": "cocoa", "severity": "high"},
        )
        assert Decimal(result["delta"]["exposure_change"]) > Decimal("0")

    @pytest.mark.unit
    def test_regulatory_change(self, portfolio_risk_aggregator):
        """Regulatory change scenario runs without error."""
        result = portfolio_risk_aggregator.simulate_scenario(
            _balanced_portfolio(),
            "regulatory_change",
            {"affected_commodity": "all", "severity": "medium"},
        )
        assert "baseline" in result
        assert "scenario" in result

    @pytest.mark.unit
    def test_climate_event(self, portfolio_risk_aggregator):
        """Climate event scenario increases VaR."""
        result = portfolio_risk_aggregator.simulate_scenario(
            _balanced_portfolio(),
            "climate_event",
            {"severity": "high"},
        )
        assert Decimal(result["delta"]["var_change"]) > Decimal("0")

    @pytest.mark.unit
    def test_invalid_scenario_raises(self, portfolio_risk_aggregator):
        """Invalid scenario type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scenario_type"):
            portfolio_risk_aggregator.simulate_scenario(
                _balanced_portfolio(),
                "earthquake",
                {},
            )


# ---------------------------------------------------------------------------
# TestCorrelationMatrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Tests for calculate_correlation_matrix."""

    @pytest.mark.unit
    def test_diagonal_is_one(self, portfolio_risk_aggregator):
        """Diagonal entries are 1.0."""
        result = portfolio_risk_aggregator.calculate_correlation_matrix(
            ["soya", "cattle"],
        )
        assert result["matrix"]["soya"]["soya"] == str(Decimal("1.0"))
        assert result["matrix"]["cattle"]["cattle"] == str(Decimal("1.0"))

    @pytest.mark.unit
    def test_symmetric_matrix(self, portfolio_risk_aggregator):
        """Matrix is symmetric: corr(A,B) == corr(B,A)."""
        result = portfolio_risk_aggregator.calculate_correlation_matrix(
            ["oil_palm", "rubber"],
        )
        assert result["matrix"]["oil_palm"]["rubber"] == result["matrix"]["rubber"]["oil_palm"]

    @pytest.mark.unit
    def test_empty_list_raises(self, portfolio_risk_aggregator):
        """Empty commodity list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            portfolio_risk_aggregator.calculate_correlation_matrix([])

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, portfolio_risk_aggregator):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            portfolio_risk_aggregator.calculate_correlation_matrix(["banana"])

    @pytest.mark.unit
    def test_deduplication(self, portfolio_risk_aggregator):
        """Duplicate commodities are deduplicated."""
        result = portfolio_risk_aggregator.calculate_correlation_matrix(
            ["soya", "soya", "cocoa"],
        )
        assert result["size"] == 2


# ---------------------------------------------------------------------------
# TestDiversificationRecommendations
# ---------------------------------------------------------------------------

class TestDiversificationRecommendations:
    """Tests for recommend_diversification."""

    @pytest.mark.unit
    def test_concentrated_gets_recommendations(self, portfolio_risk_aggregator):
        """Highly concentrated portfolio receives recommendations."""
        result = portfolio_risk_aggregator.recommend_diversification(
            _concentrated_portfolio(),
        )
        assert result["recommendation_count"] > 0
        assert len(result["missing_commodities"]) == 6  # 7 - 1

    @pytest.mark.unit
    def test_diverse_portfolio_fewer_recommendations(self, portfolio_risk_aggregator):
        """Diverse portfolio has fewer (or zero) recommendations about adding commodities."""
        result = portfolio_risk_aggregator.recommend_diversification(
            _diverse_portfolio(),
        )
        assert len(result["missing_commodities"]) == 0

    @pytest.mark.unit
    def test_overweight_detected(self, portfolio_risk_aggregator):
        """Overweight positions are identified."""
        positions = [
            {"commodity": "oil_palm", "weight": 0.80, "exposure_value": 800000},
            {"commodity": "soya", "weight": 0.10, "exposure_value": 100000},
            {"commodity": "cocoa", "weight": 0.10, "exposure_value": 100000},
        ]
        result = portfolio_risk_aggregator.recommend_diversification(positions)
        assert "oil_palm" in result["overweight_commodities"]

    @pytest.mark.unit
    def test_projected_equal_weight_hhi(self, portfolio_risk_aggregator):
        """Projected equal-weight metrics are included."""
        result = portfolio_risk_aggregator.recommend_diversification(
            _balanced_portfolio(),
        )
        assert "projected_metrics_equal_weight" in result
        assert "hhi" in result["projected_metrics_equal_weight"]


# ---------------------------------------------------------------------------
# TestVaR
# ---------------------------------------------------------------------------

class TestVaR:
    """Tests for calculate_var."""

    @pytest.mark.unit
    def test_var_95_positive(self, portfolio_risk_aggregator):
        """VaR at 95% confidence is positive."""
        var = portfolio_risk_aggregator.calculate_var(_balanced_portfolio(), 0.95)
        assert var > Decimal("0")

    @pytest.mark.unit
    def test_var_99_greater_than_95(self, portfolio_risk_aggregator):
        """VaR at 99% is greater than VaR at 95%."""
        var_95 = portfolio_risk_aggregator.calculate_var(_balanced_portfolio(), 0.95)
        var_99 = portfolio_risk_aggregator.calculate_var(_balanced_portfolio(), 0.99)
        assert var_99 > var_95

    @pytest.mark.unit
    def test_var_invalid_confidence_raises(self, portfolio_risk_aggregator):
        """Unsupported confidence level raises ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            portfolio_risk_aggregator.calculate_var(_balanced_portfolio(), 0.80)

    @pytest.mark.unit
    def test_var_single_commodity(self, portfolio_risk_aggregator):
        """VaR for single-commodity portfolio is calculable."""
        var = portfolio_risk_aggregator.calculate_var(_concentrated_portfolio(), 0.95)
        assert var >= Decimal("0")


# ---------------------------------------------------------------------------
# TestComparePortfolios
# ---------------------------------------------------------------------------

class TestComparePortfolios:
    """Tests for compare_portfolios."""

    @pytest.mark.unit
    def test_compare_returns_both_metrics(self, portfolio_risk_aggregator):
        """Comparison includes metrics for both portfolios."""
        result = portfolio_risk_aggregator.compare_portfolios(
            _balanced_portfolio(),
            _concentrated_portfolio(),
        )
        assert "portfolio_a" in result
        assert "portfolio_b" in result
        assert "delta" in result
        assert "recommendation" in result

    @pytest.mark.unit
    def test_diverse_vs_concentrated_recommendation(self, portfolio_risk_aggregator):
        """Diverse portfolio is recommended over concentrated one."""
        result = portfolio_risk_aggregator.compare_portfolios(
            _diverse_portfolio(),
            _concentrated_portfolio(),
        )
        assert "Portfolio A" in result["recommendation"] or "diversified" in result["recommendation"].lower()

    @pytest.mark.unit
    def test_delta_values_present(self, portfolio_risk_aggregator):
        """Delta section includes all diff metrics."""
        result = portfolio_risk_aggregator.compare_portfolios(
            _balanced_portfolio(),
            _diverse_portfolio(),
        )
        assert "hhi_diff" in result["delta"]
        assert "diversification_diff" in result["delta"]
        assert "risk_diff" in result["delta"]

    @pytest.mark.unit
    def test_compare_empty_a_raises(self, portfolio_risk_aggregator):
        """Empty portfolio A raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            portfolio_risk_aggregator.compare_portfolios([], _balanced_portfolio())


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance hash integrity."""

    @pytest.mark.unit
    def test_portfolio_analysis_provenance(self, portfolio_risk_aggregator):
        """Portfolio analysis has a 64-char SHA-256 provenance hash."""
        result = portfolio_risk_aggregator.analyze_portfolio(_balanced_portfolio())
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_correlation_provenance(self, portfolio_risk_aggregator):
        """Correlation matrix has provenance hash."""
        result = portfolio_risk_aggregator.calculate_correlation_matrix(["soya", "cocoa"])
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_scenario_provenance(self, portfolio_risk_aggregator):
        """Scenario simulation has provenance hash."""
        result = portfolio_risk_aggregator.simulate_scenario(
            _balanced_portfolio(), "price_shock", {"magnitude": "0.1"},
        )
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_comparison_provenance(self, portfolio_risk_aggregator):
        """Portfolio comparison has provenance hash."""
        result = portfolio_risk_aggregator.compare_portfolios(
            _balanced_portfolio(), _diverse_portfolio(),
        )
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for boundary conditions and error handling."""

    @pytest.mark.unit
    def test_non_list_input_raises(self, portfolio_risk_aggregator):
        """Non-list positions input raises ValueError."""
        with pytest.raises(ValueError, match="non-empty list"):
            portfolio_risk_aggregator.analyze_portfolio("not_a_list")

    @pytest.mark.unit
    def test_exceeds_max_positions_raises(self, portfolio_risk_aggregator):
        """Portfolio exceeding 500 positions raises ValueError."""
        positions = [
            {"commodity": "soya", "weight": 0.01, "exposure_value": 1000}
            for _ in range(501)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            portfolio_risk_aggregator.analyze_portfolio(positions)

    @pytest.mark.unit
    def test_zero_weight_gets_equal_weight(self, portfolio_risk_aggregator):
        """Positions with zero weights get equal-weight fallback."""
        positions = [
            {"commodity": "soya", "weight": 0, "exposure_value": 100000},
            {"commodity": "cocoa", "weight": 0, "exposure_value": 100000},
        ]
        result = portfolio_risk_aggregator.analyze_portfolio(positions)
        # After normalization both should have weight 0.5
        for breakdown in result["commodity_breakdown"]:
            assert Decimal(breakdown["weight"]) == Decimal("0.5000")

    @pytest.mark.unit
    def test_processing_time_positive(self, portfolio_risk_aggregator):
        """Processing time is a positive float."""
        result = portfolio_risk_aggregator.analyze_portfolio(_balanced_portfolio())
        assert result["processing_time_ms"] >= 0

    @pytest.mark.unit
    def test_recommendation_provenance(self, portfolio_risk_aggregator):
        """Diversification recommendation has provenance hash."""
        result = portfolio_risk_aggregator.recommend_diversification(_balanced_portfolio())
        assert len(result["provenance_hash"]) == 64
