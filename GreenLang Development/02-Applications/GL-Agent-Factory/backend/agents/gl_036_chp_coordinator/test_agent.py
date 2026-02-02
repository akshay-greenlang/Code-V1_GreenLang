"""GL-036 CHP Coordinator Agent - Golden Tests"""

import pytest
from .agent import CHPCoordinatorAgent, CHPCoordinatorInput


class TestCHPCoordinator:
    @pytest.fixture
    def agent(self):
        return CHPCoordinatorAgent()

    def test_thermal_led_dispatch(self, agent):
        result = agent.run(CHPCoordinatorInput(
            system_id="CHP-001",
            electrical_demand_kw=800,
            thermal_demand_kw=1200,
            fuel_price_per_mmbtu=5.0,
            grid_price_per_kwh=0.10
        ))
        assert result.chp_thermal_output_kw > 0
        assert result.validation_status == "PASS"

    def test_cost_savings(self, agent):
        result = agent.run(CHPCoordinatorInput(
            system_id="CHP-002",
            electrical_demand_kw=1000,
            thermal_demand_kw=1500,
            chp_electrical_efficiency=0.35,
            chp_thermal_efficiency=0.45
        ))
        # CHP should provide cost savings in most scenarios
        assert result.cost_savings_per_hour >= 0 or result.chp_electrical_output_kw == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
