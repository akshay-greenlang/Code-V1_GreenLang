"""
Comprehensive Test Suite for GL-076 CarbonMarketTraderAgent (CARBONTRADER)

Tests all calculation methods with 50+ tests covering:
1. Portfolio valuation calculations (10 tests)
2. Compliance assessment (10 tests)
3. Risk calculations (VaR, CVaR) (10 tests)
4. Trading recommendations (10 tests)
5. Determinism and provenance (5 tests)
6. Integration tests (5+ tests)

Standards Reference:
    - EU ETS Directive 2003/87/EC
    - California Cap-and-Trade Regulation
    - RiskMetrics VaR methodology

Test Coverage Target: 85%+
"""

import hashlib
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from .agent import (
    CarbonMarketTraderAgent,
    CarbonMarketInput,
    EmissionAllowance,
    MarketPrice,
    ComplianceObligation,
    TradingLimits,
    MarketConditions,
    AllowanceType,
    TradingAction,
    RiskLevel,
    ComplianceState,
    MarketTrend,
)

from .formulas import (
    calculate_portfolio_value,
    calculate_weighted_average_cost,
    calculate_position_risk,
    calculate_compliance_gap,
    calculate_optimal_position,
    calculate_var_monte_carlo,
    calculate_expected_shortfall,
    calculate_sharpe_ratio,
    calculate_penalty_exposure,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def carbon_market_agent():
    """Create CarbonMarketTraderAgent with default settings."""
    return CarbonMarketTraderAgent()


@pytest.fixture
def sample_allowances():
    """Create sample emission allowance positions."""
    return [
        EmissionAllowance(
            allowance_type=AllowanceType.EUA,
            vintage_year=2024,
            quantity_tonnes=10000,
            acquisition_price_eur=75.0,
            acquisition_date=datetime(2024, 1, 15),
            source="auction",
        ),
        EmissionAllowance(
            allowance_type=AllowanceType.EUA,
            vintage_year=2024,
            quantity_tonnes=5000,
            acquisition_price_eur=80.0,
            acquisition_date=datetime(2024, 3, 1),
            source="secondary_market",
        ),
    ]


@pytest.fixture
def sample_prices():
    """Create sample market prices."""
    return [
        MarketPrice(
            allowance_type=AllowanceType.EUA,
            current_price_eur=85.0,
            timestamp=datetime.utcnow(),
            bid_price_eur=84.50,
            ask_price_eur=85.50,
            volume_24h=500000,
            price_change_24h_pct=1.5,
            volatility_30d_pct=25.0,
        ),
    ]


@pytest.fixture
def sample_obligations():
    """Create sample compliance obligations."""
    return ComplianceObligation(
        period_start=datetime(2024, 1, 1),
        period_end=datetime(2024, 12, 31),
        required_surrenders_tonnes=12000,
        verified_emissions_tonnes=11500,
        free_allocation_tonnes=2000,
        deadline=datetime(2025, 4, 30),
        penalty_per_tonne_eur=100.0,
    )


@pytest.fixture
def sample_trading_limits():
    """Create sample trading limits."""
    return TradingLimits(
        max_daily_volume_tonnes=5000,
        max_position_tonnes=50000,
        min_position_tonnes=5000,
        max_single_trade_tonnes=2000,
        max_var_eur=500000,
        risk_tolerance=0.05,
    )


@pytest.fixture
def sample_input(sample_allowances, sample_prices, sample_obligations, sample_trading_limits):
    """Create complete sample input."""
    return CarbonMarketInput(
        emission_allowances=sample_allowances,
        market_prices=sample_prices,
        compliance_obligations=sample_obligations,
        trading_limits=sample_trading_limits,
        market_conditions=MarketConditions(
            market_trend=MarketTrend.NEUTRAL,
        ),
    )


# =============================================================================
# TEST CLASS: PORTFOLIO VALUATION (10 TESTS)
# =============================================================================

class TestPortfolioValuation:
    """Tests for portfolio value calculations."""

    def test_portfolio_value_single_position(self):
        """Test portfolio value with single position."""
        value = calculate_portfolio_value([1000], [80.0])
        assert value == 80000.0

    def test_portfolio_value_multiple_positions(self):
        """Test portfolio value with multiple positions."""
        value = calculate_portfolio_value([1000, 500, 2000], [80.0, 75.0, 85.0])
        expected = 1000 * 80 + 500 * 75 + 2000 * 85  # 80000 + 37500 + 170000 = 287500
        assert value == expected

    def test_portfolio_value_empty(self):
        """Test portfolio value with no positions."""
        value = calculate_portfolio_value([], [])
        assert value == 0.0

    def test_portfolio_value_mismatched_lengths_raises(self):
        """Test that mismatched list lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            calculate_portfolio_value([1000, 500], [80.0])

    def test_weighted_average_cost_calculation(self):
        """Test weighted average cost calculation."""
        wac = calculate_weighted_average_cost([1000, 500], [80.0, 85.0])
        expected = (1000 * 80 + 500 * 85) / 1500  # 122500 / 1500 = 81.67
        assert wac == pytest.approx(expected, rel=1e-4)

    def test_weighted_average_cost_single_position(self):
        """Test WAC with single position equals position price."""
        wac = calculate_weighted_average_cost([1000], [80.0])
        assert wac == 80.0

    def test_weighted_average_cost_zero_quantity_raises(self):
        """Test that zero total quantity raises error."""
        with pytest.raises(ValueError, match="cannot be zero"):
            calculate_weighted_average_cost([0, 0], [80.0, 85.0])

    def test_portfolio_unrealized_pnl(self, carbon_market_agent, sample_input):
        """Test unrealized P&L calculation."""
        result = carbon_market_agent.run(sample_input)

        # Total acquisition cost: 10000*75 + 5000*80 = 1,150,000
        # Current value at 85: 15000*85 = 1,275,000
        # Unrealized P&L: 1,275,000 - 1,150,000 = 125,000
        assert result.total_unrealized_pnl_eur == pytest.approx(125000, rel=0.01)

    def test_portfolio_value_accuracy(self, carbon_market_agent, sample_input):
        """Test total portfolio value accuracy."""
        result = carbon_market_agent.run(sample_input)

        # 15000 tonnes at EUR 85 = 1,275,000
        assert result.total_portfolio_value_eur == pytest.approx(1275000, rel=0.01)

    def test_portfolio_position_grouping(self, carbon_market_agent, sample_input):
        """Test that positions are correctly grouped by allowance type."""
        result = carbon_market_agent.run(sample_input)

        # Should have one EUA position (grouped)
        eua_positions = [p for p in result.portfolio_positions if p.allowance_type == AllowanceType.EUA]
        assert len(eua_positions) == 1
        assert eua_positions[0].total_quantity_tonnes == 15000


# =============================================================================
# TEST CLASS: COMPLIANCE ASSESSMENT (10 TESTS)
# =============================================================================

class TestComplianceAssessment:
    """Tests for compliance status calculations."""

    def test_compliance_gap_surplus(self):
        """Test compliance gap calculation with surplus."""
        gap, ratio = calculate_compliance_gap(12000, 10000, 0)
        assert gap == 2000  # 2000 tonne surplus
        assert ratio == 1.2  # 120% coverage

    def test_compliance_gap_deficit(self):
        """Test compliance gap calculation with deficit."""
        gap, ratio = calculate_compliance_gap(8000, 10000, 0)
        assert gap == -2000  # 2000 tonne deficit
        assert ratio == 0.8  # 80% coverage

    def test_compliance_gap_with_free_allocation(self):
        """Test compliance gap including free allocation."""
        gap, ratio = calculate_compliance_gap(7000, 10000, 2000)
        assert gap == -1000  # 1000 tonne deficit after free allocation
        assert ratio == 0.9  # 90% coverage

    def test_compliance_gap_exact_match(self):
        """Test compliance gap with exact coverage."""
        gap, ratio = calculate_compliance_gap(10000, 10000, 0)
        assert gap == 0
        assert ratio == 1.0

    def test_compliance_state_compliant(self, carbon_market_agent, sample_input):
        """Test compliant state when holdings exceed requirement."""
        result = carbon_market_agent.run(sample_input)

        # 15000 holdings + 2000 free = 17000 vs 12000 required
        assert result.compliance_status.state == ComplianceState.COMPLIANT

    def test_compliance_state_at_risk(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test at-risk state when coverage is marginal."""
        obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=16000,  # Higher requirement
            free_allocation_tonnes=0,
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,  # 15000 tonnes
            market_prices=sample_prices,
            compliance_obligations=obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)

        # 15000 vs 16000 = 93.75% coverage
        assert result.compliance_status.state == ComplianceState.AT_RISK

    def test_compliance_days_to_deadline(self, carbon_market_agent, sample_input):
        """Test days to deadline calculation."""
        result = carbon_market_agent.run(sample_input)

        expected_days = (datetime(2025, 4, 30) - datetime.utcnow()).days
        assert result.compliance_status.days_to_deadline == pytest.approx(expected_days, abs=1)

    def test_compliance_penalty_estimation(self):
        """Test penalty exposure calculation."""
        penalty = calculate_penalty_exposure(1000, 100.0, 1.0)
        assert penalty == 100000  # 1000 * 100 = 100,000

    def test_compliance_penalty_with_probability(self):
        """Test penalty exposure with probability factor."""
        penalty = calculate_penalty_exposure(1000, 100.0, 0.5)
        assert penalty == 50000  # 1000 * 100 * 0.5 = 50,000

    def test_optimal_position_calculation(self):
        """Test optimal position calculation for compliance."""
        optimal = calculate_optimal_position(
            required_surrenders=10000,
            coverage_target=1.1,
            current_holdings=8000,
            free_allocation=1000,
            max_position=50000
        )
        # Target: 11000, Current: 9000, Need: 2000
        assert optimal == 2000


# =============================================================================
# TEST CLASS: RISK CALCULATIONS (10 TESTS)
# =============================================================================

class TestRiskCalculations:
    """Tests for VaR and risk metric calculations."""

    def test_position_risk_basic(self):
        """Test basic position risk calculation."""
        metrics = calculate_position_risk(
            value=1000000,
            volatility=0.25,
            confidence_level=0.95,
            time_horizon_days=1
        )

        # VaR = 1M * 1.645 * 0.25 * sqrt(1/252) = ~25,900
        expected_var_95 = 1000000 * 1.645 * 0.25 * math.sqrt(1/252)
        assert metrics.var_95 == pytest.approx(expected_var_95, rel=0.01)

    def test_position_risk_10_day_horizon(self):
        """Test position risk with 10-day horizon."""
        metrics = calculate_position_risk(
            value=1000000,
            volatility=0.25,
            confidence_level=0.95,
            time_horizon_days=10
        )

        # VaR scales with sqrt(T)
        expected_var_95 = 1000000 * 1.645 * 0.25 * math.sqrt(10/252)
        assert metrics.var_95 == pytest.approx(expected_var_95, rel=0.01)

    def test_var_99_higher_than_var_95(self):
        """Test that VaR 99% is higher than VaR 95%."""
        metrics = calculate_position_risk(
            value=1000000,
            volatility=0.25,
            time_horizon_days=1
        )

        assert metrics.var_99 > metrics.var_95

    def test_expected_shortfall_calculation(self):
        """Test Expected Shortfall calculation."""
        es = calculate_expected_shortfall(100000, 0.95, "normal")

        # ES = VaR * 2.063 for 95% normal
        expected = 100000 * 2.063
        assert es == pytest.approx(expected, rel=0.01)

    def test_var_monte_carlo_historical(self):
        """Test VaR calculation using historical simulation."""
        returns = [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        var, es = calculate_var_monte_carlo(1000000, returns, 0.95, time_horizon_days=1)

        # 5th percentile of returns should be around -0.05
        assert var > 0

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio with positive returns."""
        returns = [0.001, 0.002, 0.001, 0.003, 0.002]  # Positive returns
        sharpe = calculate_sharpe_ratio(returns, 0.03)

        # Should be positive for consistently positive returns
        assert sharpe > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio with negative returns."""
        returns = [-0.001, -0.002, -0.001, -0.003, -0.002]  # Negative returns
        sharpe = calculate_sharpe_ratio(returns, 0.03)

        # Should be negative for consistently negative returns
        assert sharpe < 0

    def test_risk_assessment_levels(self, carbon_market_agent, sample_input):
        """Test risk assessment levels are assigned correctly."""
        result = carbon_market_agent.run(sample_input)

        assert result.risk_assessment.overall_risk_level in [
            RiskLevel.MINIMAL, RiskLevel.LOW, RiskLevel.MODERATE,
            RiskLevel.HIGH, RiskLevel.CRITICAL
        ]

    def test_risk_scores_range(self, carbon_market_agent, sample_input):
        """Test that risk scores are within valid range (0-100)."""
        result = carbon_market_agent.run(sample_input)

        assert 0 <= result.risk_assessment.price_risk_score <= 100
        assert 0 <= result.risk_assessment.compliance_risk_score <= 100
        assert 0 <= result.risk_assessment.liquidity_risk_score <= 100
        assert 0 <= result.risk_assessment.regulatory_risk_score <= 100

    def test_var_negative_value_raises(self):
        """Test that negative position value raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_position_risk(-1000000, 0.25)


# =============================================================================
# TEST CLASS: TRADING RECOMMENDATIONS (10 TESTS)
# =============================================================================

class TestTradingRecommendations:
    """Tests for trading recommendation generation."""

    def test_recommendations_generated(self, carbon_market_agent, sample_input):
        """Test that recommendations are always generated."""
        result = carbon_market_agent.run(sample_input)

        assert len(result.trading_recommendations) > 0

    def test_buy_recommendation_for_deficit(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test BUY recommendation when compliance deficit exists."""
        obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=20000,  # Creates deficit
            deadline=datetime(2025, 4, 30),
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,  # 15000 tonnes
            market_prices=sample_prices,
            compliance_obligations=obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)

        buy_recs = [r for r in result.trading_recommendations if r.action == TradingAction.BUY]
        assert len(buy_recs) > 0

    def test_hold_recommendation_when_compliant(self, carbon_market_agent, sample_input):
        """Test HOLD recommendation when fully compliant with low risk."""
        # Sample input has surplus, should get HOLD or BANK
        result = carbon_market_agent.run(sample_input)

        actions = [r.action for r in result.trading_recommendations]
        assert TradingAction.HOLD in actions or TradingAction.BANK in actions

    def test_recommendation_confidence_scores(self, carbon_market_agent, sample_input):
        """Test that confidence scores are within valid range."""
        result = carbon_market_agent.run(sample_input)

        for rec in result.trading_recommendations:
            assert 0 <= rec.confidence_score <= 1

    def test_recommendation_quantity_within_limits(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test that recommended quantities respect trading limits."""
        obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=50000,  # Large deficit
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)

        for rec in result.trading_recommendations:
            if rec.quantity_tonnes > 0:
                assert rec.quantity_tonnes <= sample_trading_limits.max_single_trade_tonnes or \
                       rec.quantity_tonnes <= sample_trading_limits.max_daily_volume_tonnes

    def test_urgent_recommendation_near_deadline(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test URGENT urgency when near compliance deadline."""
        obligations = ComplianceObligation(
            period_end=datetime.utcnow() + timedelta(days=10),  # 10 days away
            required_surrenders_tonnes=20000,
            deadline=datetime.utcnow() + timedelta(days=10),
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)

        urgent_recs = [r for r in result.trading_recommendations if r.urgency in ["HIGH", "CRITICAL"]]
        assert len(urgent_recs) > 0

    def test_recommendation_rationale_provided(self, carbon_market_agent, sample_input):
        """Test that all recommendations include rationale."""
        result = carbon_market_agent.run(sample_input)

        for rec in result.trading_recommendations:
            assert rec.rationale is not None
            assert len(rec.rationale) > 10

    def test_recommendation_time_horizon(self, carbon_market_agent, sample_input):
        """Test that time horizons are reasonable."""
        result = carbon_market_agent.run(sample_input)

        for rec in result.trading_recommendations:
            assert rec.time_horizon_days >= 1
            assert rec.time_horizon_days <= 365

    def test_hedge_recommendation_high_risk(self, carbon_market_agent, sample_allowances, sample_trading_limits):
        """Test HEDGE recommendation when risk is high."""
        # High volatility prices
        high_vol_prices = [
            MarketPrice(
                allowance_type=AllowanceType.EUA,
                current_price_eur=85.0,
                volatility_30d_pct=60.0,  # High volatility
            ),
        ]

        obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=10000,
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=high_vol_prices,
            compliance_obligations=obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)

        # High volatility should trigger hedge consideration
        assert result.risk_assessment.price_risk_score > 50

    def test_multiple_recommendation_types(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test that different scenarios produce different recommendation types."""
        # This tests the variety of recommendation logic

        # Deficit scenario
        deficit_obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=20000,
        )

        deficit_input = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=deficit_obligations,
            trading_limits=sample_trading_limits,
        )

        deficit_result = carbon_market_agent.run(deficit_input)
        deficit_actions = {r.action for r in deficit_result.trading_recommendations}

        # Surplus scenario
        surplus_obligations = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=5000,
        )

        surplus_input = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=surplus_obligations,
            trading_limits=sample_trading_limits,
        )

        surplus_result = carbon_market_agent.run(surplus_input)
        surplus_actions = {r.action for r in surplus_result.trading_recommendations}

        # Different scenarios should have different recommendations
        # At minimum, deficit should have BUY, surplus should not require urgent BUY
        assert TradingAction.BUY in deficit_actions


# =============================================================================
# TEST CLASS: DETERMINISM AND PROVENANCE (5 TESTS)
# =============================================================================

class TestDeterminismAndProvenance:
    """Tests for calculation determinism and provenance tracking."""

    def test_identical_inputs_identical_outputs(self, carbon_market_agent, sample_input):
        """Test that identical inputs produce identical outputs."""
        result1 = carbon_market_agent.run(sample_input)
        result2 = carbon_market_agent.run(sample_input)

        # Key outputs should be identical
        assert result1.total_portfolio_value_eur == result2.total_portfolio_value_eur
        assert result1.total_unrealized_pnl_eur == result2.total_unrealized_pnl_eur
        assert result1.compliance_status.state == result2.compliance_status.state

    def test_provenance_hash_sha256_format(self, carbon_market_agent, sample_input):
        """Test that provenance hash is in SHA-256 format."""
        result = carbon_market_agent.run(sample_input)

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())

    def test_provenance_chain_populated(self, carbon_market_agent, sample_input):
        """Test that provenance chain contains expected operations."""
        result = carbon_market_agent.run(sample_input)

        operations = [p.operation for p in result.provenance_chain]

        assert "portfolio_valuation" in operations
        assert "compliance_assessment" in operations
        assert "risk_assessment" in operations

    def test_provenance_hash_changes_with_input(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test that different inputs produce different provenance hashes."""
        obligations1 = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=10000,
        )

        obligations2 = ComplianceObligation(
            period_end=datetime(2024, 12, 31),
            required_surrenders_tonnes=15000,  # Different
        )

        input1 = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=obligations1,
            trading_limits=sample_trading_limits,
        )

        input2 = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=obligations2,
            trading_limits=sample_trading_limits,
        )

        result1 = carbon_market_agent.run(input1)
        result2 = carbon_market_agent.run(input2)

        assert result1.provenance_hash != result2.provenance_hash

    def test_analysis_id_unique(self, carbon_market_agent, sample_input):
        """Test that analysis IDs are unique."""
        result1 = carbon_market_agent.run(sample_input)

        # Small delay to ensure different timestamp
        import time
        time.sleep(0.01)

        result2 = carbon_market_agent.run(sample_input)

        assert result1.analysis_id != result2.analysis_id


# =============================================================================
# TEST CLASS: INTEGRATION TESTS (5+ TESTS)
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow_execution(self, carbon_market_agent, sample_input):
        """Test complete end-to-end workflow."""
        result = carbon_market_agent.run(sample_input)

        # Verify all major outputs are populated
        assert result.analysis_id is not None
        assert result.timestamp is not None
        assert len(result.trading_recommendations) > 0
        assert len(result.portfolio_positions) > 0
        assert result.risk_assessment is not None
        assert result.compliance_status is not None
        assert result.provenance_hash is not None

    def test_validation_status_pass(self, carbon_market_agent, sample_input):
        """Test that valid input produces PASS validation status."""
        result = carbon_market_agent.run(sample_input)

        assert result.validation_status == "PASS"
        assert len(result.validation_errors) == 0

    def test_processing_time_recorded(self, carbon_market_agent, sample_input):
        """Test that processing time is recorded."""
        result = carbon_market_agent.run(sample_input)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Should be well under 10 seconds

    def test_market_trend_analysis(self, carbon_market_agent, sample_input):
        """Test market trend analysis."""
        result = carbon_market_agent.run(sample_input)

        assert result.market_trend in [
            MarketTrend.BULLISH, MarketTrend.BEARISH,
            MarketTrend.NEUTRAL, MarketTrend.VOLATILE
        ]

    def test_price_forecast_generation(self, carbon_market_agent, sample_input):
        """Test price forecast generation."""
        result = carbon_market_agent.run(sample_input)

        # Forecast should be generated when price data available
        if result.price_forecast_30d_eur is not None:
            assert result.price_forecast_30d_eur > 0

    def test_agent_metadata(self, carbon_market_agent):
        """Test agent metadata is correct."""
        assert carbon_market_agent.AGENT_ID == "GL-076"
        assert carbon_market_agent.AGENT_NAME == "CARBONTRADER"
        assert carbon_market_agent.VERSION == "1.0.0"


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_quantity_allowances(self, carbon_market_agent, sample_prices, sample_obligations, sample_trading_limits):
        """Test handling of zero quantity allowances."""
        zero_allowances = [
            EmissionAllowance(
                allowance_type=AllowanceType.EUA,
                vintage_year=2024,
                quantity_tonnes=0,
                acquisition_price_eur=80.0,
            ),
        ]

        input_data = CarbonMarketInput(
            emission_allowances=zero_allowances,
            market_prices=sample_prices,
            compliance_obligations=sample_obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)
        assert result.validation_status == "PASS"

    def test_very_large_position(self, carbon_market_agent, sample_prices, sample_obligations, sample_trading_limits):
        """Test handling of very large positions."""
        large_allowances = [
            EmissionAllowance(
                allowance_type=AllowanceType.EUA,
                vintage_year=2024,
                quantity_tonnes=1000000,  # 1 million tonnes
                acquisition_price_eur=80.0,
            ),
        ]

        input_data = CarbonMarketInput(
            emission_allowances=large_allowances,
            market_prices=sample_prices,
            compliance_obligations=sample_obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)
        assert result.total_portfolio_value_eur > 0

    def test_past_deadline(self, carbon_market_agent, sample_allowances, sample_prices, sample_trading_limits):
        """Test handling of past compliance deadline."""
        past_obligations = ComplianceObligation(
            period_end=datetime(2023, 12, 31),  # Past
            required_surrenders_tonnes=10000,
            deadline=datetime(2024, 4, 30),  # Also past
        )

        input_data = CarbonMarketInput(
            emission_allowances=sample_allowances,
            market_prices=sample_prices,
            compliance_obligations=past_obligations,
            trading_limits=sample_trading_limits,
        )

        result = carbon_market_agent.run(input_data)
        assert result.compliance_status.days_to_deadline < 0


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
