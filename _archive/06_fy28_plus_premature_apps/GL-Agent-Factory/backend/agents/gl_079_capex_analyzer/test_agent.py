"""
GL-079: CAPEX Analyzer Agent - Test Suite

Comprehensive test coverage for CapexAnalyzerAgent.

Test Coverage Target: 85%+
"""

import pytest
from datetime import datetime
from typing import Dict, List

from .agent import (
    CapexAnalyzerAgent,
    CapexAnalyzerInput,
    CapexAnalyzerOutput,
    EquipmentCost,
    InstallationCost,
    SoftCost,
    FundingSource,
    CapexBreakdown,
    CostComparison,
    SensitivityAnalysis,
    EquipmentType,
    CostCategory,
    ProjectPhase,
    FundingType,
)
from .formulas import (
    calculate_total_capex,
    calculate_cost_per_unit,
    calculate_contingency,
    calculate_installed_cost,
    calculate_installation_labor,
    calculate_soft_costs,
    run_sensitivity_analysis,
    calculate_escalation,
    calculate_benchmark_percentile,
    calculate_financing_costs,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create CapexAnalyzerAgent instance."""
    return CapexAnalyzerAgent()


@pytest.fixture
def basic_equipment():
    """Create basic equipment cost list."""
    return [
        EquipmentCost(
            equipment_type=EquipmentType.SOLAR_PV,
            description="Solar PV Modules",
            quantity=100,
            unit_cost_usd=300,
            capacity_per_unit=0.4,
            capacity_unit="kW",
        ),
        EquipmentCost(
            equipment_type=EquipmentType.BATTERY_STORAGE,
            description="Battery Storage System",
            quantity=1,
            unit_cost_usd=50000,
            capacity_per_unit=100,
            capacity_unit="kWh",
        ),
    ]


@pytest.fixture
def basic_installation():
    """Create basic installation costs."""
    return [
        InstallationCost(
            category=CostCategory.ELECTRICAL,
            description="Electrical Installation",
            labor_hours=160,
            labor_rate_usd=75,
            material_cost_usd=5000,
        ),
        InstallationCost(
            category=CostCategory.STRUCTURAL,
            description="Mounting System",
            labor_hours=80,
            labor_rate_usd=65,
            material_cost_usd=8000,
        ),
    ]


@pytest.fixture
def basic_soft_costs():
    """Create basic soft costs."""
    return [
        SoftCost(
            category=CostCategory.ENGINEERING,
            description="Engineering Design",
            cost_usd=0,
            is_percentage=True,
            percentage=8.0,
        ),
        SoftCost(
            category=CostCategory.PERMITTING,
            description="Permits and Fees",
            cost_usd=5000,
        ),
    ]


@pytest.fixture
def basic_input(basic_equipment, basic_installation, basic_soft_costs):
    """Create basic input for testing."""
    return CapexAnalyzerInput(
        project_name="Solar + Storage Project",
        project_location="California",
        equipment_costs=basic_equipment,
        installation_costs=basic_installation,
        soft_costs=basic_soft_costs,
        contingency_percent=10.0,
        project_size_capacity=40.0,
        capacity_unit="kW",
    )


# =============================================================================
# AGENT TESTS
# =============================================================================

class TestAgentInitialization:
    """Test agent initialization."""

    def test_agent_creates_successfully(self, agent):
        """Test agent creates with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "GL-079"
        assert agent.AGENT_NAME == "CAPEXANALYZER"

    def test_agent_has_benchmarks(self, agent):
        """Test agent has cost benchmarks."""
        assert len(agent.benchmarks) > 0


class TestRunMethod:
    """Test the main run method."""

    def test_run_returns_output(self, agent, basic_input):
        """Test run returns CapexAnalyzerOutput."""
        result = agent.run(basic_input)
        assert isinstance(result, CapexAnalyzerOutput)

    def test_run_calculates_equipment_total(self, agent, basic_input):
        """Test equipment cost calculation."""
        result = agent.run(basic_input)
        # 100 * $300 + 1 * $50,000 = $80,000
        assert result.total_equipment_cost_usd == 80000.0

    def test_run_calculates_installation_total(self, agent, basic_input):
        """Test installation cost calculation."""
        result = agent.run(basic_input)
        # 160 * $75 + $5000 + 80 * $65 + $8000 = $30,200
        assert result.total_installation_cost_usd == 30200.0

    def test_run_calculates_contingency(self, agent, basic_input):
        """Test contingency calculation."""
        result = agent.run(basic_input)
        assert result.contingency_usd > 0

    def test_run_has_cost_breakdown(self, agent, basic_input):
        """Test cost breakdown is generated."""
        result = agent.run(basic_input)
        assert len(result.cost_breakdown) > 0

    def test_run_has_provenance(self, agent, basic_input):
        """Test provenance tracking."""
        result = agent.run(basic_input)
        assert len(result.provenance_chain) > 0
        assert len(result.provenance_hash) == 64


# =============================================================================
# FORMULA TESTS
# =============================================================================

class TestCapexFormulas:
    """Test CAPEX calculation formulas."""

    def test_total_capex_calculation(self):
        """Test total CAPEX calculation."""
        result = calculate_total_capex(
            equipment_cost=100000,
            installation_cost=50000,
            soft_cost=20000,
            contingency_percent=10,
        )
        # (100000 + 50000 + 20000) * 1.10 = 187,000
        assert result.total_capex == 187000.0

    def test_cost_per_unit_calculation(self):
        """Test cost per unit calculation."""
        cost, unit = calculate_cost_per_unit(100000, 50, "kW")
        assert cost == 2000.0
        assert unit == "$/kW"

    def test_contingency_by_complexity(self):
        """Test contingency varies by complexity."""
        low_cont, low_pct = calculate_contingency(100000, "LOW", "MATURE")
        high_cont, high_pct = calculate_contingency(100000, "HIGH", "EMERGING")

        assert high_pct > low_pct

    def test_installed_cost_factor(self):
        """Test installed cost calculation."""
        installed = calculate_installed_cost(100000, 1.5)
        assert installed == 150000.0

    def test_installation_labor(self):
        """Test labor cost and hours calculation."""
        cost, hours = calculate_installation_labor(100000, 35, 75)
        assert cost == 35000.0
        assert hours == 466.7  # 35000 / 75

    def test_soft_costs_calculation(self):
        """Test soft cost percentages."""
        result = calculate_soft_costs(
            hard_costs=100000,
            engineering_percent=8,
            permitting_percent=3,
            project_management_percent=5,
        )
        assert result["engineering"] == 8000.0
        assert result["permitting"] == 3000.0
        assert result["total_soft_costs"] == 16000.0


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_sensitivity_returns_results(self):
        """Test sensitivity analysis returns results."""
        base_costs = {
            "equipment": 100000,
            "installation": 50000,
            "soft_costs": 20000,
        }
        results = run_sensitivity_analysis(base_costs, 20.0)
        assert len(results) == 3

    def test_sensitivity_ranking(self):
        """Test sensitivity results are ranked."""
        base_costs = {
            "equipment": 100000,
            "installation": 10000,
        }
        results = run_sensitivity_analysis(base_costs, 20.0)
        # Equipment should be ranked #1 (higher impact)
        assert results[0].parameter_name == "equipment"
        assert results[0].sensitivity_rank == 1


class TestEscalation:
    """Test cost escalation."""

    def test_escalation_calculation(self):
        """Test escalation over time."""
        escalated, amount = calculate_escalation(100000, 3, 3.0)
        # 100000 * (1.03)^3 = 109,272.70
        assert escalated == 109272.7

    def test_zero_escalation(self):
        """Test zero escalation rate."""
        escalated, amount = calculate_escalation(100000, 5, 0.0)
        assert escalated == 100000.0
        assert amount == 0.0


class TestBenchmarking:
    """Test benchmarking calculations."""

    def test_below_benchmark(self):
        """Test project below benchmark."""
        percentile, status = calculate_benchmark_percentile(80, 100, 150, 200)
        assert status == "BELOW"
        assert percentile < 25

    def test_within_benchmark(self):
        """Test project within benchmark range."""
        percentile, status = calculate_benchmark_percentile(125, 100, 150, 200)
        assert status == "LOW"
        assert 25 <= percentile <= 50

    def test_above_benchmark(self):
        """Test project above benchmark."""
        percentile, status = calculate_benchmark_percentile(250, 100, 150, 200)
        assert status == "ABOVE"
        assert percentile > 75


class TestFinancing:
    """Test financing calculations."""

    def test_financing_costs(self):
        """Test financing cost calculation."""
        result = calculate_financing_costs(
            principal=100000,
            interest_rate=5.0,
            term_years=10,
            loan_fees_percent=2.0,
        )
        assert result["loan_fees"] == 2000.0
        assert result["total_interest"] > 0
        assert result["monthly_payment"] > 0

    def test_zero_interest_financing(self):
        """Test zero interest loan."""
        result = calculate_financing_costs(
            principal=100000,
            interest_rate=0.0,
            term_years=10,
        )
        assert result["total_interest"] == 0.0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_equipment_only(self, agent):
        """Test with equipment costs only."""
        input_data = CapexAnalyzerInput(
            project_name="Equipment Only",
            equipment_costs=[
                EquipmentCost(
                    equipment_type=EquipmentType.HVAC,
                    description="HVAC Unit",
                    quantity=1,
                    unit_cost_usd=10000,
                ),
            ],
        )
        result = agent.run(input_data)
        assert result.total_equipment_cost_usd == 10000.0
        assert result.total_installation_cost_usd == 0.0

    def test_no_contingency(self, agent, basic_equipment):
        """Test with zero contingency."""
        input_data = CapexAnalyzerInput(
            project_name="No Contingency",
            equipment_costs=basic_equipment,
            contingency_percent=0.0,
        )
        result = agent.run(input_data)
        assert result.contingency_usd == 0.0


# =============================================================================
# PACK_SPEC TESTS
# =============================================================================

class TestPackSpec:
    """Test PACK_SPEC configuration."""

    def test_pack_spec_exists(self):
        """Test PACK_SPEC is defined."""
        from .agent import PACK_SPEC
        assert PACK_SPEC is not None
        assert PACK_SPEC["id"] == "GL-079"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
