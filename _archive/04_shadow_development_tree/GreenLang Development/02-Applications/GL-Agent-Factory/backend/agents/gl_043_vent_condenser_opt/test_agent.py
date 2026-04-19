"""GL-043 Vent Condenser Optimizer - Golden Tests"""
import pytest
from .agent import VentCondenserOptAgent, VentCondenserOptInput

class TestVentCondenserOpt:
    def test_basic_run(self):
        agent = VentCondenserOptAgent()
        result = agent.run(VentCondenserOptInput(condenser_id="VC-001", vent_flow_lb_hr=1000))
        assert result.validation_status == "PASS"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
