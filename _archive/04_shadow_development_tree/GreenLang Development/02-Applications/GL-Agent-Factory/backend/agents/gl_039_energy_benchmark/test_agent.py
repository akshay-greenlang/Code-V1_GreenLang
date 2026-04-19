"""GL-039 Energy Benchmark - Golden Tests"""
import pytest
from .agent import EnergyBenchmarkAgent, EnergyBenchmarkInput

class TestEnergyBenchmark:
    def test_basic_run(self):
        agent = EnergyBenchmarkAgent()
        result = agent.run(EnergyBenchmarkInput(
            facility_id="PLANT-001",
            industry_sector="steel",
            production_quantity=10000,
            total_energy_kwh=6000000
        ))
        assert result.validation_status == "PASS"
        assert result.specific_energy_kwh_per_unit == 600

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
