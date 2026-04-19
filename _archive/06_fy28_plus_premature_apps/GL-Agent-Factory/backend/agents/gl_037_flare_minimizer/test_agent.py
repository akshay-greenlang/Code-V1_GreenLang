"""GL-037 Flare Minimizer - Golden Tests"""
import pytest
from .agent import FlareMinimzerAgent, FlareMinimzerInput

class TestFlareMinimizer:
    def test_basic_run(self):
        agent = FlareMinimzerAgent()
        result = agent.run(FlareMinimzerInput(flare_id="FL-001", flare_flow_scfh=5000))
        assert result.validation_status == "PASS"
        assert result.recoverable_volume_scfh > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
