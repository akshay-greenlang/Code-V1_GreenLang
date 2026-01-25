"""GL-041 Startup Optimizer - Golden Tests"""
import pytest
from .agent import StartupOptimizerAgent, StartupOptimizerInput, EquipmentItem

class TestStartupOptimizer:
    def test_basic_run(self):
        agent = StartupOptimizerAgent()
        result = agent.run(StartupOptimizerInput(
            system_id="BOILER-001",
            equipment_list=[
                EquipmentItem(equipment_id="FAN", equipment_type="ID Fan", min_startup_time_minutes=5),
                EquipmentItem(equipment_id="BURNER", equipment_type="Burner", min_startup_time_minutes=10, dependencies=["FAN"]),
                EquipmentItem(equipment_id="HEATER", equipment_type="Air Preheater", min_startup_time_minutes=15, dependencies=["BURNER"])
            ]
        ))
        assert result.validation_status == "PASS"
        assert len(result.optimal_sequence) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
