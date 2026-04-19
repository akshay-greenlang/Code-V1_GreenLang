"""GL-045 Carbon Intensity Tracker - Golden Tests"""
import pytest
from .agent import CarbonIntensityTrackerAgent, CarbonIntensityTrackerInput, FuelConsumption

class TestCarbonIntensityTracker:
    def test_basic_run(self):
        agent = CarbonIntensityTrackerAgent()
        result = agent.run(CarbonIntensityTrackerInput(
            facility_id="PLANT-001",
            fuel_consumption=[FuelConsumption(fuel_type="natural_gas", quantity=100, unit="mmbtu")],
            electricity_kwh=50000,
            production_output=100
        ))
        assert result.validation_status == "PASS"
        assert result.scope1_emissions_kg > 0
        assert result.scope2_emissions_kg > 0

    def test_carbon_intensity_calculation(self):
        agent = CarbonIntensityTrackerAgent()
        result = agent.run(CarbonIntensityTrackerInput(
            facility_id="PLANT-002",
            fuel_consumption=[FuelConsumption(fuel_type="natural_gas", quantity=53.06, unit="mmbtu")],
            electricity_kwh=0,
            production_output=1
        ))
        # 53.06 MMBtu * 53.06 kg/MMBtu = ~2815 kg CO2 for 1 unit
        assert result.carbon_intensity_kg_per_unit == pytest.approx(2815.4, rel=0.01)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
