"""GL-080: OPEX Optimizer Agent - Test Suite"""

import pytest
from .agent import (
    OpexOptimizerAgent,
    OpexOptimizerInput,
    EnergyCost,
    MaintenanceCost,
    LaborCost,
    EnergyType,
    MaintenanceType,
)
from .formulas import (
    calculate_annual_opex,
    calculate_energy_cost,
    calculate_labor_optimization,
    project_opex_savings,
)


@pytest.fixture
def agent():
    return OpexOptimizerAgent()


@pytest.fixture
def basic_input():
    return OpexOptimizerInput(
        facility_name="Test Facility",
        energy_costs=[
            EnergyCost(
                energy_type=EnergyType.ELECTRICITY,
                annual_consumption=1000000,
                rate_per_unit=0.10,
                demand_charge_annual=12000,
            ),
        ],
        maintenance_costs=[
            MaintenanceCost(
                equipment_name="HVAC System",
                annual_cost_usd=25000,
            ),
        ],
        labor_costs=[
            LaborCost(
                role="Facility Manager",
                fte_count=1,
                annual_salary_usd=75000,
            ),
        ],
    )


class TestAgentInitialization:
    def test_agent_creates(self, agent):
        assert agent.AGENT_ID == "GL-080"

    def test_run_returns_output(self, agent, basic_input):
        result = agent.run(basic_input)
        assert result.total_annual_opex_usd > 0


class TestFormulas:
    def test_annual_opex(self):
        result = calculate_annual_opex(
            {"electricity": 100000},
            {"hvac": 25000},
            {"manager": 100000},
            {"insurance": 10000},
        )
        assert result.total_opex == 235000

    def test_energy_cost(self):
        result = calculate_energy_cost(
            consumption_kwh=100000,
            energy_rate=0.10,
            demand_kw=100,
            demand_rate=15,
        )
        assert result["energy_cost"] == 10000
        assert result["demand_cost"] == 1500

    def test_labor_optimization(self):
        result = calculate_labor_optimization(
            current_fte=5,
            annual_salary=60000,
            benefits_percent=30,
            automation_potential_percent=20,
            automation_cost=100000,
        )
        assert result["annual_savings"] > 0

    def test_project_savings(self):
        result = project_opex_savings(
            baseline_opex=500000,
            annual_savings=50000,
            years=10,
        )
        assert result["nominal_savings"] > 500000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
