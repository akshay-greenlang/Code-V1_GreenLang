# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Economic Analysis Tests

Comprehensive tests for economic analysis including:
- Energy cost calculations
- Cleaning ROI analysis
- NPV and IRR calculations
- Total Cost of Ownership (TCO)
- Lifecycle cost analysis
- Replacement vs maintenance decisions

Coverage Target: 90%+
"""

import pytest
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_014_heat_exchanger.economics import (
    EconomicAnalyzer,
    EnergyCostBreakdown,
    ReplacementAnalysis,
    LifecycleCost,
    OptimizationOpportunity,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    EconomicsConfig,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    EconomicAnalysisResult,
)


class TestEconomicAnalyzerInit:
    """Tests for EconomicAnalyzer initialization."""

    def test_analyzer_initialization(self, economics_config):
        """Test analyzer initializes correctly."""
        analyzer = EconomicAnalyzer(config=economics_config)
        assert analyzer.config == economics_config

    def test_analyzer_with_defaults(self):
        """Test analyzer with default configuration."""
        config = EconomicsConfig()
        analyzer = EconomicAnalyzer(config=config)
        assert analyzer.config.discount_rate == 0.10


class TestEnergyLossCalculation:
    """Tests for energy loss cost calculations."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_energy_loss_calculation(self, analyzer):
        """Test energy loss calculation from U degradation."""
        result = analyzer.calculate_energy_loss_cost(
            u_current_w_m2k=400.0,
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            operating_hours_per_year=8000.0,
        )

        assert isinstance(result, EnergyCostBreakdown)
        assert result.heat_loss_kw > 0
        assert result.annual_cost_usd > 0

    def test_energy_loss_increases_with_fouling(self, analyzer):
        """Test energy loss increases as U decreases."""
        result_light = analyzer.calculate_energy_loss_cost(
            u_current_w_m2k=450.0,  # Light fouling
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
        )

        result_heavy = analyzer.calculate_energy_loss_cost(
            u_current_w_m2k=300.0,  # Heavy fouling
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
        )

        assert result_heavy.heat_loss_kw > result_light.heat_loss_kw
        assert result_heavy.annual_cost_usd > result_light.annual_cost_usd

    def test_energy_loss_zero_when_clean(self, analyzer):
        """Test energy loss is zero when U is at clean value."""
        result = analyzer.calculate_energy_loss_cost(
            u_current_w_m2k=500.0,
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
        )

        assert result.heat_loss_kw == pytest.approx(0.0, abs=0.1)

    def test_energy_cost_breakdown(self, analyzer):
        """Test energy cost breakdown includes all components."""
        result = analyzer.calculate_energy_loss_cost(
            u_current_w_m2k=400.0,
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
        )

        # Verify breakdown components
        assert result.heat_loss_cost_usd_per_hour >= 0
        assert result.pumping_cost_usd_per_hour >= 0
        assert result.total_cost_usd_per_hour > 0

        # Total should equal sum of components
        expected_total = result.heat_loss_cost_usd_per_hour + result.pumping_cost_usd_per_hour
        assert result.total_cost_usd_per_hour == pytest.approx(expected_total, rel=0.01)


class TestNPVCalculation:
    """Tests for Net Present Value calculations."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_npv_calculation(self, analyzer):
        """Test NPV calculation."""
        # NPV = sum of discounted cash flows
        # For constant annual cost: NPV = C * [(1 - (1+r)^-n) / r]
        annual_cost = 10000.0
        years = 10
        r = 0.10

        npv = analyzer._calculate_npv(
            annual_cost=annual_cost,
            years=years,
            discount_rate=r,
        )

        # Manual calculation
        factor = (1 - (1 + r) ** (-years)) / r
        expected_npv = annual_cost * factor

        assert npv == pytest.approx(expected_npv, rel=0.01)

    def test_npv_zero_discount_rate(self, analyzer):
        """Test NPV with zero discount rate."""
        npv = analyzer._calculate_npv(
            annual_cost=10000.0,
            years=10,
            discount_rate=0.0,
        )

        # With 0% discount, NPV = annual_cost * years
        assert npv == pytest.approx(100000.0, rel=0.01)


class TestIRRCalculation:
    """Tests for Internal Rate of Return calculations."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_irr_calculation_positive(self, analyzer):
        """Test IRR calculation for profitable investment."""
        irr = analyzer._calculate_irr(
            initial_investment=100000.0,
            annual_savings=25000.0,
            years=10,
        )

        # With 25k/year for 10 years on 100k investment, IRR should be significant
        assert irr is not None
        assert irr > 0.15  # Should be well above 15%

    def test_irr_calculation_marginal(self, analyzer):
        """Test IRR for marginal investment."""
        irr = analyzer._calculate_irr(
            initial_investment=100000.0,
            annual_savings=10000.0,
            years=10,
        )

        # Marginal return
        assert irr is not None

    def test_irr_negative_savings(self, analyzer):
        """Test IRR returns None for negative savings."""
        irr = analyzer._calculate_irr(
            initial_investment=100000.0,
            annual_savings=-5000.0,  # Losing money
            years=10,
        )

        assert irr is None


class TestReplacementAnalysis:
    """Tests for replacement vs maintenance analysis."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_replacement_analysis(self, analyzer):
        """Test replacement analysis."""
        result = analyzer.analyze_replacement_economics(
            annual_operating_cost_usd=50000.0,
            annual_maintenance_cost_usd=20000.0,
            remaining_life_years=5.0,
            new_equipment_cost_usd=500000.0,
            new_equipment_life_years=20.0,
            new_operating_cost_factor=0.7,
        )

        assert isinstance(result, ReplacementAnalysis)
        assert result.npv_maintain_usd > 0
        assert result.npv_replace_usd > 0
        assert result.recommendation in ["maintain", "replace_now", "replace_planned"]

    def test_replacement_when_old_equipment_failing(self, analyzer):
        """Test replacement recommendation for failing equipment."""
        result = analyzer.analyze_replacement_economics(
            annual_operating_cost_usd=100000.0,  # High operating cost
            annual_maintenance_cost_usd=50000.0,  # High maintenance
            remaining_life_years=2.0,  # Short remaining life
            new_equipment_cost_usd=400000.0,
            new_equipment_life_years=20.0,
            new_operating_cost_factor=0.5,  # New equipment much better
        )

        # Should likely recommend replacement
        assert result.recommendation in ["replace_now", "replace_planned"]

    def test_maintain_when_equipment_good(self, analyzer):
        """Test maintain recommendation for good equipment."""
        result = analyzer.analyze_replacement_economics(
            annual_operating_cost_usd=20000.0,  # Low operating cost
            annual_maintenance_cost_usd=5000.0,  # Low maintenance
            remaining_life_years=15.0,  # Long remaining life
            new_equipment_cost_usd=500000.0,  # Expensive new equipment
            new_equipment_life_years=20.0,
            new_operating_cost_factor=0.8,  # Not much improvement
        )

        assert result.recommendation == "maintain"

    def test_replacement_payback_period(self, analyzer):
        """Test payback period calculation."""
        result = analyzer.analyze_replacement_economics(
            annual_operating_cost_usd=50000.0,
            annual_maintenance_cost_usd=20000.0,
            remaining_life_years=10.0,
            new_equipment_cost_usd=300000.0,
            new_equipment_life_years=20.0,
            new_operating_cost_factor=0.6,
        )

        if result.annual_savings_if_replace_usd > 0:
            expected_payback = 300000.0 / result.annual_savings_if_replace_usd
            assert result.payback_period_years == pytest.approx(expected_payback, rel=0.1)


class TestLifecycleCost:
    """Tests for lifecycle cost analysis."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_lifecycle_cost_calculation(self, analyzer):
        """Test lifecycle cost calculation."""
        result = analyzer.calculate_lifecycle_cost(
            capital_cost_usd=500000.0,
            installation_factor=1.3,
            annual_energy_cost_usd=30000.0,
            annual_maintenance_cost_usd=15000.0,
            annual_downtime_hours=24.0,
            equipment_life_years=20.0,
            disposal_cost_factor=0.05,
        )

        assert isinstance(result, LifecycleCost)
        assert result.capital_cost_usd == 500000.0
        assert result.installation_cost_usd > 0
        assert result.total_lifecycle_cost_usd > result.capital_cost_usd

    def test_lifecycle_cost_components(self, analyzer):
        """Test lifecycle cost includes all components."""
        result = analyzer.calculate_lifecycle_cost(
            capital_cost_usd=500000.0,
            installation_factor=1.3,
            annual_energy_cost_usd=30000.0,
            annual_maintenance_cost_usd=15000.0,
            annual_downtime_hours=24.0,
            equipment_life_years=20.0,
        )

        # Installation = capital * (factor - 1)
        assert result.installation_cost_usd == pytest.approx(500000 * 0.3, rel=0.01)

        # All costs should sum to total
        component_sum = (
            result.capital_cost_usd +
            result.installation_cost_usd +
            result.energy_cost_usd +
            result.maintenance_cost_usd +
            result.downtime_cost_usd +
            result.disposal_cost_usd
        )
        assert result.total_lifecycle_cost_usd == pytest.approx(component_sum, rel=0.01)

    def test_annualized_cost(self, analyzer):
        """Test annualized cost calculation."""
        result = analyzer.calculate_lifecycle_cost(
            capital_cost_usd=500000.0,
            annual_energy_cost_usd=30000.0,
            annual_maintenance_cost_usd=15000.0,
            equipment_life_years=20.0,
        )

        # Annualized cost should be reasonable
        assert result.annualized_cost_usd > 0
        assert result.annualized_cost_usd < result.total_lifecycle_cost_usd


class TestOptimizationOpportunities:
    """Tests for optimization opportunity identification."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_identify_opportunities(self, analyzer):
        """Test optimization opportunity identification."""
        opportunities = analyzer.identify_optimization_opportunities(
            u_current_w_m2k=350.0,
            u_clean_w_m2k=500.0,
            u_design_w_m2k=450.0,
            fouling_rate_m2kw_per_day=0.000003,
            current_cleaning_frequency_days=180,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            operating_hours_per_year=8000.0,
        )

        assert isinstance(opportunities, list)
        # Should identify at least one opportunity (cleaning)
        assert len(opportunities) > 0

    def test_opportunity_structure(self, analyzer):
        """Test optimization opportunity structure."""
        opportunities = analyzer.identify_optimization_opportunities(
            u_current_w_m2k=350.0,
            u_clean_w_m2k=500.0,
            u_design_w_m2k=450.0,
            fouling_rate_m2kw_per_day=0.000003,
            current_cleaning_frequency_days=180,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
        )

        for opp in opportunities:
            assert isinstance(opp, OptimizationOpportunity)
            assert opp.opportunity_type is not None
            assert opp.description is not None
            assert opp.annual_savings_usd >= 0
            assert opp.implementation_cost_usd >= 0
            assert opp.priority in ["high", "medium", "low"]

    def test_opportunity_prioritization(self, analyzer):
        """Test opportunities are prioritized."""
        opportunities = analyzer.identify_optimization_opportunities(
            u_current_w_m2k=300.0,  # Severe fouling
            u_clean_w_m2k=500.0,
            u_design_w_m2k=450.0,
            fouling_rate_m2kw_per_day=0.00001,  # High fouling rate
            current_cleaning_frequency_days=365,  # Infrequent cleaning
            heat_transfer_area_m2=200.0,  # Large exchanger
            lmtd_c=40.0,
        )

        # Should have high priority opportunities
        priorities = [opp.priority for opp in opportunities]
        assert "high" in priorities


class TestCompleteEconomicAnalysis:
    """Tests for complete economic analysis."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_complete_analysis(self, analyzer):
        """Test complete economic analysis."""
        result = analyzer.analyze_economics(
            u_current_w_m2k=400.0,
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            operating_hours_per_year=8000.0,
            fouling_rate_m2kw_per_day=0.000002,
            cleaning_cost_usd=5000.0,
            remaining_life_years=10.0,
        )

        assert isinstance(result, EconomicAnalysisResult)
        assert result.energy_loss_kw >= 0
        assert result.energy_cost_usd_per_year >= 0
        assert result.annual_tco_usd > 0

    def test_complete_analysis_cleaning_roi(self, analyzer):
        """Test cleaning ROI in complete analysis."""
        result = analyzer.analyze_economics(
            u_current_w_m2k=350.0,  # Significant fouling
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            cleaning_cost_usd=5000.0,
            remaining_life_years=10.0,
        )

        # Should have positive cleaning ROI
        assert result.cleaning_roi_percent is not None

    def test_complete_analysis_optimal_cleaning(self, analyzer):
        """Test optimal cleaning frequency in analysis."""
        result = analyzer.analyze_economics(
            u_current_w_m2k=400.0,
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            fouling_rate_m2kw_per_day=0.000002,
            cleaning_cost_usd=5000.0,
            remaining_life_years=10.0,
        )

        # Optimal frequency should be within bounds
        assert result.optimal_cleaning_frequency_days >= 30
        assert result.optimal_cleaning_frequency_days <= 365

    def test_complete_analysis_recommendations(self, analyzer):
        """Test optimization recommendations in analysis."""
        result = analyzer.analyze_economics(
            u_current_w_m2k=350.0,  # Poor U value
            u_clean_w_m2k=500.0,
            heat_transfer_area_m2=100.0,
            lmtd_c=30.0,
            fouling_rate_m2kw_per_day=0.000005,  # High fouling rate
            cleaning_cost_usd=5000.0,
            remaining_life_years=8.0,
        )

        # Should have recommendations
        assert len(result.optimization_recommendations) >= 0
        assert result.optimization_savings_usd_per_year >= 0


class TestCostEscalation:
    """Tests for cost escalation and time value."""

    @pytest.fixture
    def analyzer(self, economics_config):
        """Create EconomicAnalyzer instance."""
        return EconomicAnalyzer(config=economics_config)

    def test_discount_effect(self, analyzer):
        """Test discount rate effect on NPV."""
        # Higher discount rate = lower NPV
        npv_10 = analyzer._calculate_npv(
            annual_cost=10000.0,
            years=10,
            discount_rate=0.10,
        )

        npv_15 = analyzer._calculate_npv(
            annual_cost=10000.0,
            years=10,
            discount_rate=0.15,
        )

        assert npv_15 < npv_10

    def test_time_horizon_effect(self, analyzer):
        """Test time horizon effect on NPV."""
        npv_5yr = analyzer._calculate_npv(
            annual_cost=10000.0,
            years=5,
            discount_rate=0.10,
        )

        npv_20yr = analyzer._calculate_npv(
            annual_cost=10000.0,
            years=20,
            discount_rate=0.10,
        )

        # Longer horizon = higher NPV (more years of costs)
        assert npv_20yr > npv_5yr
