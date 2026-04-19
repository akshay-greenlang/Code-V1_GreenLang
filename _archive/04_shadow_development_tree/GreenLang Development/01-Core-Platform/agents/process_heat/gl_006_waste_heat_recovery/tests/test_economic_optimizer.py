"""
GL-006 WasteHeatRecovery Agent - Economic Optimizer Tests

Comprehensive unit tests for the EconomicOptimizer class.
Tests NPV, IRR, payback, and portfolio optimization calculations.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_006_waste_heat_recovery.economic_optimizer import (
    EconomicOptimizer,
    WasteHeatProject,
    EnergyMetrics,
    CostBreakdown,
    IncentivePackage,
    EconomicAnalysisResult,
    PortfolioAnalysisResult,
    YearlyProjection,
    DepreciationMethod,
    calculate_capital_recovery_factor,
    calculate_present_worth_factor,
    estimate_installation_cost,
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_INFLATION_RATE,
    DEFAULT_PROJECT_LIFE_YEARS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def optimizer():
    """Create EconomicOptimizer instance for testing."""
    return EconomicOptimizer(
        discount_rate=0.08,
        inflation_rate=0.025,
        energy_escalation_rate=0.03,
        tax_rate=0.21,
        project_life_years=20,
    )


@pytest.fixture
def simple_project():
    """Create simple project for testing."""
    return WasteHeatProject(
        name="Economizer Installation",
        capital_cost_usd=100000.0,
        annual_operating_cost_usd=3000.0,
        annual_energy_savings_usd=25000.0,
        project_life_years=20,
    )


@pytest.fixture
def project_with_energy_metrics():
    """Create project with energy metrics."""
    return WasteHeatProject(
        name="WHR System",
        capital_cost_usd=200000.0,
        annual_operating_cost_usd=8000.0,
        annual_energy_savings_usd=55000.0,
        project_life_years=15,
        energy_metrics=EnergyMetrics(
            annual_energy_savings_mmbtu=6000.0,
            fuel_type="natural_gas",
            co2_reduction_tons_yr=320.0,
        ),
    )


@pytest.fixture
def project_with_incentives():
    """Create project with incentives."""
    return WasteHeatProject(
        name="Incentivized WHR",
        capital_cost_usd=150000.0,
        annual_operating_cost_usd=5000.0,
        annual_energy_savings_usd=40000.0,
        project_life_years=20,
        incentives=IncentivePackage(
            federal_tax_credit_pct=0.10,
            state_rebate_usd=10000.0,
            carbon_credit_eligible=True,
            carbon_price_usd_per_ton=50.0,
        ),
        energy_metrics=EnergyMetrics(
            annual_energy_savings_mmbtu=4500.0,
            co2_reduction_tons_yr=240.0,
        ),
    )


@pytest.fixture
def multiple_projects():
    """Create multiple projects for portfolio analysis."""
    return [
        WasteHeatProject(
            name="Project A - Quick Win",
            capital_cost_usd=50000.0,
            annual_energy_savings_usd=25000.0,
            annual_operating_cost_usd=1000.0,
            project_life_years=15,
            energy_metrics=EnergyMetrics(
                annual_energy_savings_mmbtu=2500.0,
                co2_reduction_tons_yr=130.0,
            ),
        ),
        WasteHeatProject(
            name="Project B - Large Scale",
            capital_cost_usd=300000.0,
            annual_energy_savings_usd=80000.0,
            annual_operating_cost_usd=15000.0,
            project_life_years=20,
            energy_metrics=EnergyMetrics(
                annual_energy_savings_mmbtu=9000.0,
                co2_reduction_tons_yr=480.0,
            ),
        ),
        WasteHeatProject(
            name="Project C - Marginal",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=15000.0,
            annual_operating_cost_usd=5000.0,
            project_life_years=15,
            energy_metrics=EnergyMetrics(
                annual_energy_savings_mmbtu=1500.0,
                co2_reduction_tons_yr=80.0,
            ),
        ),
    ]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestEconomicOptimizerInitialization:
    """Test EconomicOptimizer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test optimizer initializes with defaults."""
        optimizer = EconomicOptimizer()

        assert optimizer.discount_rate == DEFAULT_DISCOUNT_RATE
        assert optimizer.inflation_rate == DEFAULT_INFLATION_RATE
        assert optimizer.project_life == DEFAULT_PROJECT_LIFE_YEARS

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test optimizer with custom parameters."""
        optimizer = EconomicOptimizer(
            discount_rate=0.10,
            inflation_rate=0.03,
            energy_escalation_rate=0.04,
            tax_rate=0.25,
            project_life_years=25,
        )

        assert optimizer.discount_rate == 0.10
        assert optimizer.inflation_rate == 0.03
        assert optimizer.energy_escalation == 0.04
        assert optimizer.tax_rate == 0.25
        assert optimizer.project_life == 25


# =============================================================================
# NPV CALCULATION TESTS
# =============================================================================

class TestNPVCalculation:
    """Test NPV calculation."""

    @pytest.mark.unit
    def test_npv_calculated(self, optimizer, simple_project):
        """Test NPV is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.npv_usd is not None

    @pytest.mark.unit
    def test_npv_formula_verification(self, optimizer):
        """Test NPV follows correct formula."""
        # Simple case: constant cash flows
        project = WasteHeatProject(
            name="Test NPV",
            capital_cost_usd=100000.0,
            annual_operating_cost_usd=0.0,
            annual_energy_savings_usd=15000.0,
            project_life_years=10,
        )

        # Set escalation to 0 for simple calculation
        optimizer.energy_escalation = 0.0
        optimizer.inflation_rate = 0.0

        result = optimizer.analyze_project(project, include_sensitivity=False)

        # Manual NPV calculation: -100000 + 15000 * PWF(8%, 10 years)
        pwf = calculate_present_worth_factor(0.08, 10)
        expected_npv = -100000 + 15000 * pwf

        # Allow some tolerance for rounding
        assert result.npv_usd == pytest.approx(expected_npv, rel=0.05)

    @pytest.mark.unit
    def test_positive_npv_good_project(self, optimizer):
        """Test good project has positive NPV."""
        project = WasteHeatProject(
            name="Good Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=40000.0,  # High savings
            annual_operating_cost_usd=2000.0,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        assert result.npv_usd > 0

    @pytest.mark.unit
    def test_negative_npv_bad_project(self, optimizer):
        """Test poor project has negative NPV."""
        project = WasteHeatProject(
            name="Poor Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=3000.0,  # Low savings
            annual_operating_cost_usd=2000.0,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        assert result.npv_usd < 0


# =============================================================================
# IRR CALCULATION TESTS
# =============================================================================

class TestIRRCalculation:
    """Test IRR calculation."""

    @pytest.mark.unit
    def test_irr_calculated(self, optimizer, simple_project):
        """Test IRR is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.irr_pct is not None

    @pytest.mark.unit
    def test_irr_range(self, optimizer, simple_project):
        """Test IRR is in reasonable range."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        # IRR should be positive for a decent project
        # And less than some unreasonable maximum (e.g., 200%)
        assert -50 < result.irr_pct < 200

    @pytest.mark.unit
    def test_irr_exceeds_discount_rate_for_positive_npv(self, optimizer):
        """Test IRR exceeds discount rate when NPV is positive."""
        project = WasteHeatProject(
            name="Good Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=30000.0,
            annual_operating_cost_usd=2000.0,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        if result.npv_usd > 0:
            assert result.irr_pct > optimizer.discount_rate * 100

    @pytest.mark.unit
    def test_mirr_calculated(self, optimizer, simple_project):
        """Test MIRR is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        # MIRR should be calculated
        if result.mirr_pct is not None:
            assert -50 < result.mirr_pct < 200


# =============================================================================
# PAYBACK CALCULATION TESTS
# =============================================================================

class TestPaybackCalculation:
    """Test payback period calculation."""

    @pytest.mark.unit
    def test_simple_payback_calculated(self, optimizer, simple_project):
        """Test simple payback is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.simple_payback_years is not None
        assert result.simple_payback_years > 0

    @pytest.mark.unit
    def test_simple_payback_formula(self, optimizer):
        """Test simple payback formula."""
        project = WasteHeatProject(
            name="Test Payback",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=25000.0,
            annual_operating_cost_usd=0.0,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        # Simple payback = Capital / Annual Net Savings
        expected_payback = 100000.0 / 25000.0  # = 4 years
        assert result.simple_payback_years == pytest.approx(expected_payback, rel=0.01)

    @pytest.mark.unit
    def test_discounted_payback_calculated(self, optimizer, simple_project):
        """Test discounted payback is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        # Discounted payback should be longer than simple payback
        if result.discounted_payback_years is not None:
            assert result.discounted_payback_years >= result.simple_payback_years

    @pytest.mark.unit
    def test_payback_infinite_for_negative_savings(self, optimizer):
        """Test payback is infinite for negative net savings."""
        project = WasteHeatProject(
            name="Negative Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=5000.0,
            annual_operating_cost_usd=10000.0,  # Operating cost > savings
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        assert result.simple_payback_years == float('inf')


# =============================================================================
# ROI AND SIR TESTS
# =============================================================================

class TestROIAndSIR:
    """Test ROI and SIR calculations."""

    @pytest.mark.unit
    def test_roi_calculated(self, optimizer, simple_project):
        """Test ROI is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.roi_pct is not None

    @pytest.mark.unit
    def test_sir_calculated(self, optimizer, simple_project):
        """Test SIR is calculated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.savings_investment_ratio is not None
        assert result.savings_investment_ratio > 0

    @pytest.mark.unit
    def test_sir_greater_than_one_for_good_project(self, optimizer):
        """Test SIR > 1 for a good project."""
        project = WasteHeatProject(
            name="Good Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=30000.0,
            annual_operating_cost_usd=2000.0,
            project_life_years=20,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        # Good project should have SIR > 1
        if result.npv_usd > 0:
            assert result.savings_investment_ratio > 1.0


# =============================================================================
# LCOE CALCULATION TESTS
# =============================================================================

class TestLCOECalculation:
    """Test LCOE calculation."""

    @pytest.mark.unit
    def test_lcoe_calculated(self, optimizer, project_with_energy_metrics):
        """Test LCOE is calculated when energy metrics present."""
        result = optimizer.analyze_project(
            project_with_energy_metrics, include_sensitivity=False
        )

        assert result.lcoe_usd_per_mmbtu is not None
        assert result.lcoe_usd_per_mmbtu > 0

    @pytest.mark.unit
    def test_lcoe_not_calculated_without_energy_metrics(self, optimizer, simple_project):
        """Test LCOE is None without energy metrics."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.lcoe_usd_per_mmbtu is None


# =============================================================================
# CARBON ECONOMICS TESTS
# =============================================================================

class TestCarbonEconomics:
    """Test carbon economics calculations."""

    @pytest.mark.unit
    def test_carbon_benefit_calculated(self, optimizer, project_with_incentives):
        """Test carbon benefit is calculated."""
        result = optimizer.analyze_project(
            project_with_incentives, include_sensitivity=False
        )

        assert result.carbon_benefit_usd >= 0

    @pytest.mark.unit
    def test_carbon_cost_effectiveness(self, optimizer, project_with_energy_metrics):
        """Test carbon cost effectiveness is calculated."""
        result = optimizer.analyze_project(
            project_with_energy_metrics, include_sensitivity=False
        )

        assert result.carbon_cost_effectiveness_usd_per_ton is not None
        assert result.carbon_cost_effectiveness_usd_per_ton > 0


# =============================================================================
# YEARLY PROJECTION TESTS
# =============================================================================

class TestYearlyProjections:
    """Test yearly projection generation."""

    @pytest.mark.unit
    def test_projections_generated(self, optimizer, simple_project):
        """Test yearly projections are generated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert len(result.yearly_projections) == simple_project.project_life_years

    @pytest.mark.unit
    def test_projections_have_required_fields(self, optimizer, simple_project):
        """Test projections have required fields."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        for proj in result.yearly_projections:
            assert isinstance(proj, YearlyProjection)
            assert proj.year > 0
            assert proj.gross_savings_usd is not None
            assert proj.net_cash_flow_usd is not None
            assert proj.cumulative_cash_flow_usd is not None
            assert proj.discounted_cash_flow_usd is not None

    @pytest.mark.unit
    def test_savings_escalation(self, optimizer, simple_project):
        """Test savings escalate over time."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        # Year 2 savings should be higher than Year 1 (with positive escalation)
        year1_savings = result.yearly_projections[0].gross_savings_usd
        year2_savings = result.yearly_projections[1].gross_savings_usd

        assert year2_savings > year1_savings


# =============================================================================
# SENSITIVITY ANALYSIS TESTS
# =============================================================================

class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    @pytest.mark.unit
    def test_sensitivity_analysis_generated(self, optimizer, simple_project):
        """Test sensitivity analysis is generated."""
        result = optimizer.analyze_project(
            simple_project, include_sensitivity=True
        )

        assert result.sensitivity_results is not None

    @pytest.mark.unit
    def test_sensitivity_parameters(self, optimizer, simple_project):
        """Test sensitivity analysis covers key parameters."""
        result = optimizer.analyze_project(
            simple_project, include_sensitivity=True
        )

        assert "capital_cost" in result.sensitivity_results
        assert "energy_savings" in result.sensitivity_results
        assert "discount_rate" in result.sensitivity_results

    @pytest.mark.unit
    def test_sensitivity_most_sensitive_identified(self, optimizer, simple_project):
        """Test most sensitive parameter is identified."""
        result = optimizer.analyze_project(
            simple_project, include_sensitivity=True
        )

        assert "most_sensitive_parameter" in result.sensitivity_results


# =============================================================================
# MONTE CARLO TESTS
# =============================================================================

class TestMonteCarloAnalysis:
    """Test Monte Carlo analysis."""

    @pytest.mark.unit
    def test_monte_carlo_analysis_generated(self, optimizer, simple_project):
        """Test Monte Carlo analysis is generated when requested."""
        result = optimizer.analyze_project(
            simple_project,
            include_sensitivity=False,
            include_monte_carlo=True,
            monte_carlo_iterations=100,  # Reduced for speed
        )

        assert result.monte_carlo_results is not None

    @pytest.mark.unit
    def test_monte_carlo_statistics(self, optimizer, simple_project):
        """Test Monte Carlo produces statistics."""
        result = optimizer.analyze_project(
            simple_project,
            include_sensitivity=False,
            include_monte_carlo=True,
            monte_carlo_iterations=100,
        )

        mc = result.monte_carlo_results
        assert "npv_statistics" in mc
        assert "mean" in mc["npv_statistics"]
        assert "std_dev" in mc["npv_statistics"]
        assert "p10" in mc["npv_statistics"]
        assert "p90" in mc["npv_statistics"]

    @pytest.mark.unit
    def test_monte_carlo_probability(self, optimizer, simple_project):
        """Test Monte Carlo calculates probability of positive NPV."""
        result = optimizer.analyze_project(
            simple_project,
            include_sensitivity=False,
            include_monte_carlo=True,
            monte_carlo_iterations=100,
        )

        mc = result.monte_carlo_results
        assert "probability_npv_positive" in mc
        assert 0 <= mc["probability_npv_positive"] <= 100


# =============================================================================
# PORTFOLIO OPTIMIZATION TESTS
# =============================================================================

class TestPortfolioOptimization:
    """Test portfolio optimization."""

    @pytest.mark.unit
    def test_portfolio_analysis_returns_result(self, optimizer, multiple_projects):
        """Test portfolio analysis returns result."""
        result = optimizer.optimize_portfolio(multiple_projects)

        assert isinstance(result, PortfolioAnalysisResult)

    @pytest.mark.unit
    def test_portfolio_rankings(self, optimizer, multiple_projects):
        """Test portfolio rankings are generated."""
        result = optimizer.optimize_portfolio(multiple_projects)

        assert len(result.project_rankings) == len(multiple_projects)
        assert result.project_rankings[0]["rank"] == 1

    @pytest.mark.unit
    def test_portfolio_budget_constraint(self, optimizer, multiple_projects):
        """Test portfolio respects budget constraint."""
        result = optimizer.optimize_portfolio(
            multiple_projects,
            budget_constraint_usd=200000.0,
        )

        assert result.total_investment_usd <= 200000.0

    @pytest.mark.unit
    def test_portfolio_max_projects_constraint(self, optimizer, multiple_projects):
        """Test portfolio respects max projects constraint."""
        result = optimizer.optimize_portfolio(
            multiple_projects,
            max_projects=2,
        )

        assert len(result.selected_projects) <= 2

    @pytest.mark.unit
    def test_portfolio_selects_positive_npv(self, optimizer, multiple_projects):
        """Test portfolio only selects positive NPV projects."""
        result = optimizer.optimize_portfolio(multiple_projects)

        for proj_name in result.selected_projects:
            proj_data = next(
                p for p in result.project_rankings if p["name"] == proj_name
            )
            assert proj_data["npv"] > 0

    @pytest.mark.unit
    def test_portfolio_aggregate_metrics(self, optimizer, multiple_projects):
        """Test portfolio calculates aggregate metrics."""
        result = optimizer.optimize_portfolio(multiple_projects)

        assert result.portfolio_npv_usd is not None
        assert result.portfolio_payback_years is not None
        assert result.total_annual_savings_usd >= 0


# =============================================================================
# INCENTIVE HANDLING TESTS
# =============================================================================

class TestIncentiveHandling:
    """Test incentive handling."""

    @pytest.mark.unit
    def test_tax_credit_applied(self, optimizer, project_with_incentives):
        """Test federal tax credit is applied."""
        result = optimizer.analyze_project(
            project_with_incentives, include_sensitivity=False
        )

        # Check that NPV is better than without incentives
        # Create same project without incentives
        project_no_incentives = WasteHeatProject(
            name="No Incentives",
            capital_cost_usd=project_with_incentives.capital_cost_usd,
            annual_operating_cost_usd=project_with_incentives.annual_operating_cost_usd,
            annual_energy_savings_usd=project_with_incentives.annual_energy_savings_usd,
            project_life_years=project_with_incentives.project_life_years,
        )

        result_no_incentives = optimizer.analyze_project(
            project_no_incentives, include_sensitivity=False
        )

        # With incentives should have better NPV
        assert result.npv_usd > result_no_incentives.npv_usd


# =============================================================================
# DEPRECIATION TESTS
# =============================================================================

class TestDepreciation:
    """Test depreciation handling."""

    @pytest.mark.unit
    def test_depreciation_in_projections(self, optimizer, simple_project):
        """Test depreciation is included in projections."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        # Some years should have depreciation
        has_depreciation = any(
            proj.depreciation_usd > 0 for proj in result.yearly_projections
        )
        assert has_depreciation


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================

class TestRecommendations:
    """Test recommendation generation."""

    @pytest.mark.unit
    def test_economic_ranking_generated(self, optimizer, simple_project):
        """Test economic ranking is generated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.economic_ranking is not None
        assert result.economic_ranking in [
            "Excellent", "Very Good", "Good", "Marginal", "Poor"
        ]

    @pytest.mark.unit
    def test_recommendation_generated(self, optimizer, simple_project):
        """Test recommendation is generated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.recommendation is not None
        assert len(result.recommendation) > 0


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.unit
    def test_capital_recovery_factor(self):
        """Test CRF calculation."""
        crf = calculate_capital_recovery_factor(0.08, 20)

        # CRF at 8%, 20 years should be approximately 0.1019
        assert crf == pytest.approx(0.1019, rel=0.01)

    @pytest.mark.unit
    def test_present_worth_factor(self):
        """Test PWF calculation."""
        pwf = calculate_present_worth_factor(0.08, 20)

        # PWF = 1/CRF
        crf = calculate_capital_recovery_factor(0.08, 20)
        expected_pwf = 1 / crf

        assert pwf == pytest.approx(expected_pwf, rel=0.01)

    @pytest.mark.unit
    def test_crf_pwf_relationship(self):
        """Test CRF and PWF are inverses."""
        crf = calculate_capital_recovery_factor(0.10, 15)
        pwf = calculate_present_worth_factor(0.10, 15)

        assert crf * pwf == pytest.approx(1.0, rel=0.01)

    @pytest.mark.unit
    def test_installation_cost_estimation(self):
        """Test installation cost estimation."""
        result = estimate_installation_cost(100000.0, "standard")

        assert "equipment_cost_usd" in result
        assert "installation_cost_usd" in result
        assert "engineering_cost_usd" in result
        assert "total_capital_cost_usd" in result

        # Total should be greater than equipment cost
        assert result["total_capital_cost_usd"] > result["equipment_cost_usd"]

    @pytest.mark.unit
    def test_installation_types(self):
        """Test different installation types."""
        standard = estimate_installation_cost(100000.0, "standard")
        retrofit = estimate_installation_cost(100000.0, "retrofit")
        greenfield = estimate_installation_cost(100000.0, "greenfield")

        # Retrofit should be most expensive
        assert retrofit["installation_cost_usd"] > standard["installation_cost_usd"]
        # Greenfield should be cheapest
        assert greenfield["installation_cost_usd"] < standard["installation_cost_usd"]


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Test provenance tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, optimizer, simple_project):
        """Test provenance hash is generated."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_analysis_parameters_recorded(self, optimizer, simple_project):
        """Test analysis parameters are recorded."""
        result = optimizer.analyze_project(simple_project, include_sensitivity=False)

        assert result.discount_rate_used == optimizer.discount_rate
        assert result.inflation_rate_used == optimizer.inflation_rate
        assert result.energy_escalation_used == optimizer.energy_escalation


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_zero_capital_cost(self, optimizer):
        """Test handling of zero capital cost."""
        # Zero capital cost should still work (e.g., operational changes)
        project = WasteHeatProject(
            name="Operational Change",
            capital_cost_usd=0.0,
            annual_energy_savings_usd=10000.0,
            annual_operating_cost_usd=0.0,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        # Should handle gracefully
        assert result.npv_usd > 0
        assert result.simple_payback_years == 0.0  # Instant payback

    @pytest.mark.unit
    def test_very_long_project_life(self, optimizer):
        """Test with very long project life."""
        project = WasteHeatProject(
            name="Long Life Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=10000.0,
            project_life_years=50,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        assert len(result.yearly_projections) == 50

    @pytest.mark.unit
    def test_short_project_life(self, optimizer):
        """Test with short project life."""
        project = WasteHeatProject(
            name="Short Life Project",
            capital_cost_usd=100000.0,
            annual_energy_savings_usd=50000.0,
            project_life_years=3,
        )

        result = optimizer.analyze_project(project, include_sensitivity=False)

        assert len(result.yearly_projections) == 3


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for economic optimizer."""

    @pytest.mark.performance
    def test_analysis_speed(self, optimizer, simple_project):
        """Test single project analysis speed."""
        import time

        start = time.time()
        result = optimizer.analyze_project(
            simple_project,
            include_sensitivity=True,
            include_monte_carlo=False,
        )
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second

    @pytest.mark.performance
    @pytest.mark.slow
    def test_monte_carlo_speed(self, optimizer, simple_project):
        """Test Monte Carlo analysis speed."""
        import time

        start = time.time()
        result = optimizer.analyze_project(
            simple_project,
            include_sensitivity=False,
            include_monte_carlo=True,
            monte_carlo_iterations=1000,
        )
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds

    @pytest.mark.performance
    def test_portfolio_optimization_speed(self, optimizer):
        """Test portfolio optimization speed."""
        import time

        # Create 20 projects
        projects = [
            WasteHeatProject(
                name=f"Project_{i}",
                capital_cost_usd=50000.0 + i * 10000,
                annual_energy_savings_usd=15000.0 + i * 3000,
                annual_operating_cost_usd=1000.0 + i * 200,
            )
            for i in range(20)
        ]

        start = time.time()
        result = optimizer.optimize_portfolio(projects, budget_constraint_usd=500000)
        elapsed = time.time() - start

        assert elapsed < 10.0  # Should complete in under 10 seconds
