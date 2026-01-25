"""GL-044 Water Treatment Advisor - Golden Tests"""
import pytest
from .agent import WaterTreatmentAdvisorAgent, WaterTreatmentAdvisorInput, WaterAnalysis

class TestWaterTreatmentAdvisor:
    def test_basic_run(self):
        agent = WaterTreatmentAdvisorAgent()
        result = agent.run(WaterTreatmentAdvisorInput(
            system_id="BOILER-001",
            water_analysis=WaterAnalysis(sample_point="feedwater", ph=9.5, conductivity_umho=800)
        ))
        assert result.validation_status == "PASS"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
