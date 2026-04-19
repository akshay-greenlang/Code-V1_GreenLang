"""GL-042 PressureMaster - Golden Tests"""
import pytest
from .agent import PressureMasterAgent, PressureMasterInput, HeaderPressure

class TestPressureMaster:
    def test_basic_run(self):
        agent = PressureMasterAgent()
        result = agent.run(PressureMasterInput(
            system_id="STEAM-001",
            header_pressures=[
                HeaderPressure(header_id="HP", current_pressure_psig=600, setpoint_psig=600, min_pressure_psig=550, max_pressure_psig=650)
            ]
        ))
        assert result.validation_status == "PASS"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
