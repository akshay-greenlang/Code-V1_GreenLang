"""GL-038 Insulation Auditor - Golden Tests"""
import pytest
from .agent import InsulationAuditorAgent, InsulationAuditorInput, PipeSpec

class TestInsulationAuditor:
    def test_basic_run(self):
        agent = InsulationAuditorAgent()
        result = agent.run(InsulationAuditorInput(
            facility_id="PLANT-001",
            pipe_specs=[
                PipeSpec(pipe_id="P1", nominal_diameter_inches=6, length_meters=100, process_temp_celsius=300, current_insulation_thickness_mm=25),
                PipeSpec(pipe_id="P2", nominal_diameter_inches=4, length_meters=50, process_temp_celsius=200, current_insulation_thickness_mm=0)
            ]
        ))
        assert result.validation_status == "PASS"
        assert len(result.pipe_analyses) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
