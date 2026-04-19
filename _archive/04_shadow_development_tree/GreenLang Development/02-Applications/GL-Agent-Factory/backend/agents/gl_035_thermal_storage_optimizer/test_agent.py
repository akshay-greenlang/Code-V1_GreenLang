"""GL-035 Thermal Storage Optimizer - Golden Tests"""

import pytest
from .agent import (
    ThermalStorageOptimizerAgent,
    ThermalStorageOptimizerInput,
    DemandProfile,
    EnergyPrice,
)


class TestThermalStorageOptimizer:
    @pytest.fixture
    def agent(self):
        return ThermalStorageOptimizerAgent()

    @pytest.fixture
    def valid_input(self):
        # Create 24-hour demand and price profiles
        demand_profile = [DemandProfile(hour=h, demand_kw=500 + 300 * (1 if 8 <= h <= 18 else 0)) for h in range(24)]
        energy_prices = [EnergyPrice(hour=h, price_per_kwh=0.05 + 0.10 * (1 if 12 <= h <= 18 else 0)) for h in range(24)]

        return ThermalStorageOptimizerInput(
            system_id="TES-001",
            storage_capacity_kwh=2000,
            max_charge_rate_kw=500,
            max_discharge_rate_kw=500,
            round_trip_efficiency=0.90,
            current_state_of_charge=0.3,
            demand_profile=demand_profile,
            energy_prices=energy_prices,
            demand_charge_per_kw=15.0
        )

    def test_agent_run(self, agent, valid_input):
        result = agent.run(valid_input)
        assert result.system_id == "TES-001"
        assert result.validation_status == "PASS"
        assert len(result.optimal_schedule) == 24

    def test_cost_savings(self, agent, valid_input):
        result = agent.run(valid_input)
        # With TOU pricing and peak shaving, should have some savings
        assert result.cost_savings >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
