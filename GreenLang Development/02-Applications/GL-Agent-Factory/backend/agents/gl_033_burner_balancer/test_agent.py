"""GL-033 Burner Balancer Agent - Golden Tests"""

import pytest
from datetime import datetime

from .agent import (
    BurnerBalancerAgent,
    BurnerBalancerInput,
    BurnerData,
)
from .models import BurnerType, FuelType, BurnerStatus, BalancingObjective
from .formulas import (
    calculate_stoichiometric_air_fuel_ratio,
    calculate_excess_air_percent,
    calculate_combustion_efficiency,
    distribute_load_to_burners,
)


class TestCombustionFormulas:
    """Tests for combustion calculations."""

    def test_stoichiometric_ratio_natural_gas(self):
        ratio = calculate_stoichiometric_air_fuel_ratio("NATURAL_GAS")
        assert ratio == pytest.approx(9.52, rel=0.01)

    def test_excess_air_calculation(self):
        excess = calculate_excess_air_percent(
            actual_air_flow=1100,
            fuel_flow=100,
            stoich_ratio=10.0
        )
        # (1100/100)/10 - 1 = 0.1 = 10%
        assert excess == pytest.approx(10.0, rel=0.01)

    def test_combustion_efficiency(self):
        eff = calculate_combustion_efficiency(
            flue_gas_temp_c=200,
            ambient_temp_c=25,
            o2_percent=3.0,
            fuel_type="NATURAL_GAS"
        )
        assert 80 <= eff <= 95


class TestLoadDistribution:
    """Tests for load distribution."""

    def test_uniform_distribution(self):
        rates = distribute_load_to_burners(
            total_load_percent=60,
            burner_capacities=[10, 10, 10],
            burner_efficiencies=[85, 85, 85],
            burner_status=["MODULATING", "MODULATING", "MODULATING"],
            objective="UNIFORM_HEATING"
        )
        # Should distribute evenly
        assert all(55 <= r <= 65 for r in rates)

    def test_efficiency_prioritization(self):
        rates = distribute_load_to_burners(
            total_load_percent=30,
            burner_capacities=[10, 10, 10],
            burner_efficiencies=[95, 85, 75],
            burner_status=["MODULATING", "MODULATING", "MODULATING"],
            objective="EFFICIENCY"
        )
        # Most efficient burner should run higher
        assert rates[0] >= rates[1] >= rates[2]


class TestBurnerBalancerAgent:
    """Integration tests."""

    @pytest.fixture
    def agent(self):
        return BurnerBalancerAgent()

    @pytest.fixture
    def valid_input(self):
        return BurnerBalancerInput(
            system_id="FURNACE-001",
            burner_data=[
                BurnerData(
                    burner_id="B1",
                    burner_type=BurnerType.LOW_NOX,
                    capacity_mmbtu_hr=10.0,
                    current_firing_rate=70,
                    fuel_flow_scfh=10000,
                    air_flow_scfh=100000,
                    o2_percent=4.0,
                    efficiency_rating=88
                ),
                BurnerData(
                    burner_id="B2",
                    burner_type=BurnerType.LOW_NOX,
                    capacity_mmbtu_hr=10.0,
                    current_firing_rate=70,
                    fuel_flow_scfh=10000,
                    air_flow_scfh=100000,
                    o2_percent=4.0,
                    efficiency_rating=86
                )
            ],
            fuel_type=FuelType.NATURAL_GAS,
            load_demand_percent=70,
            optimization_objective=BalancingObjective.BALANCED
        )

    def test_agent_run(self, agent, valid_input):
        result = agent.run(valid_input)
        assert result.system_id == "FURNACE-001"
        assert result.validation_status == "PASS"
        assert len(result.optimal_firing_rates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
