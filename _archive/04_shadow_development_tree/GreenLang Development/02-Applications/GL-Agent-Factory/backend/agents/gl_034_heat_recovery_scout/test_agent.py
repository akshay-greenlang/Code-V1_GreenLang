"""GL-034 Heat Recovery Scout Agent - Golden Tests"""

import pytest
from .agent import (
    HeatRecoveryScoutAgent,
    HeatRecoveryScoutInput,
    ExhaustStream,
    HeatDemand,
    UtilityCosts,
)


class TestHeatRecoveryScoutAgent:
    @pytest.fixture
    def agent(self):
        return HeatRecoveryScoutAgent()

    @pytest.fixture
    def valid_input(self):
        return HeatRecoveryScoutInput(
            facility_id="PLANT-001",
            exhaust_streams=[
                ExhaustStream(
                    stream_id="EX-001",
                    source_equipment="Furnace-1",
                    temperature_celsius=400,
                    flow_rate_kg_hr=5000,
                    specific_heat_kj_kg_k=1.1
                ),
                ExhaustStream(
                    stream_id="EX-002",
                    source_equipment="Boiler-1",
                    temperature_celsius=200,
                    flow_rate_kg_hr=8000,
                    specific_heat_kj_kg_k=1.0
                )
            ],
            process_heat_demands=[
                HeatDemand(
                    demand_id="HD-001",
                    process_name="Preheating",
                    required_temp_celsius=150,
                    heat_load_kw=500
                ),
                HeatDemand(
                    demand_id="HD-002",
                    process_name="Drying",
                    required_temp_celsius=80,
                    heat_load_kw=300
                )
            ],
            utility_costs=UtilityCosts(natural_gas_per_mmbtu=6.0)
        )

    def test_agent_run(self, agent, valid_input):
        result = agent.run(valid_input)
        assert result.facility_id == "PLANT-001"
        assert result.validation_status == "PASS"
        assert len(result.recovery_opportunities) > 0

    def test_opportunities_ranked(self, agent, valid_input):
        result = agent.run(valid_input)
        # Check ranking is sequential
        for i, opp in enumerate(result.recovery_opportunities):
            assert opp.priority_rank == i + 1

    def test_npv_calculation(self, agent, valid_input):
        result = agent.run(valid_input)
        # NPV should be calculated for all opportunities
        for opp in result.recovery_opportunities:
            # Positive NPV means profitable
            if opp.simple_payback_years < 10:
                assert opp.npv_10yr > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
