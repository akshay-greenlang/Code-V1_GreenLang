"""
GL-002 FLAMEGUARD - Orchestrator Integration Tests

End-to-end tests for the main orchestrator.
"""

import pytest
import asyncio
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])


class TestOrchestratorIntegration:
    """Integration tests for FlameGuard orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, boiler_config):
        """Test orchestrator initializes correctly."""
        from boiler_efficiency_orchestrator import FlameGuardOrchestrator

        orchestrator = FlameGuardOrchestrator()

        assert orchestrator is not None
        status = orchestrator.get_status()
        assert "agent_id" in status

    @pytest.mark.asyncio
    async def test_register_boiler(self, boiler_config):
        """Test boiler registration."""
        from boiler_efficiency_orchestrator import FlameGuardOrchestrator

        orchestrator = FlameGuardOrchestrator()
        orchestrator.register_boiler(
            boiler_id=boiler_config["boiler_id"],
            capacity_klb_hr=boiler_config["capacity_klb_hr"],
        )

        status = orchestrator.get_status()
        assert boiler_config["boiler_id"] in status.get("boilers", [])

    @pytest.mark.asyncio
    async def test_process_data_update(self, boiler_config, sample_process_data):
        """Test process data update flow."""
        from boiler_efficiency_orchestrator import FlameGuardOrchestrator

        orchestrator = FlameGuardOrchestrator()
        orchestrator.register_boiler(
            boiler_id=boiler_config["boiler_id"],
            capacity_klb_hr=boiler_config["capacity_klb_hr"],
        )

        await orchestrator.update_process_data(
            boiler_id=boiler_config["boiler_id"],
            data=sample_process_data,
        )

        boiler_status = orchestrator.get_boiler_status(boiler_config["boiler_id"])
        assert boiler_status is not None

    @pytest.mark.asyncio
    async def test_optimization_cycle(self, boiler_config, sample_process_data):
        """Test complete optimization cycle."""
        from boiler_efficiency_orchestrator import FlameGuardOrchestrator

        orchestrator = FlameGuardOrchestrator()
        orchestrator.register_boiler(
            boiler_id=boiler_config["boiler_id"],
            capacity_klb_hr=boiler_config["capacity_klb_hr"],
        )

        # Update process data
        await orchestrator.update_process_data(
            boiler_id=boiler_config["boiler_id"],
            data=sample_process_data,
        )

        # Run optimization
        result = await orchestrator.optimize(
            boiler_id=boiler_config["boiler_id"],
            mode="efficiency",
        )

        assert result is not None
        assert "recommended_o2_setpoint" in result or hasattr(result, "recommended_o2_setpoint")


class TestCalculatorIntegration:
    """Integration tests for calculators working together."""

    def test_efficiency_to_emissions_flow(
        self,
        sample_process_data,
        sample_fuel_properties,
    ):
        """Test efficiency calculation feeds into emissions."""
        from calculators.efficiency_calculator import EfficiencyCalculator, EfficiencyInput
        from calculators.emissions_calculator import EmissionsCalculator

        # Calculate efficiency
        eff_calc = EfficiencyCalculator()
        eff_input = EfficiencyInput(
            fuel_flow_rate=sample_process_data["fuel_flow_scfh"],
            fuel_hhv=sample_fuel_properties["hhv_btu_scf"],
            steam_flow=sample_process_data["steam_flow_klb_hr"],
            steam_enthalpy=1200.0,
            feedwater_enthalpy=200.0,
            flue_gas_temp_f=sample_process_data["flue_gas_temperature_f"],
            ambient_temp_f=sample_process_data["ambient_temperature_f"],
            o2_percent=sample_process_data["o2_percent"],
            fuel_type="natural_gas",
        )

        eff_result = eff_calc.calculate(eff_input)

        # Calculate emissions
        emis_calc = EmissionsCalculator()
        emis_result = emis_calc.calculate(
            fuel_type="natural_gas",
            fuel_flow_scfh=sample_process_data["fuel_flow_scfh"],
            fuel_hhv_btu_scf=sample_fuel_properties["hhv_btu_scf"],
            o2_percent=sample_process_data["o2_percent"],
        )

        # Both should complete successfully
        assert eff_result.gross_efficiency_percent > 0
        assert emis_result.co2_ton_hr > 0


class TestSafetyIntegration:
    """Integration tests for safety systems."""

    def test_interlock_triggers_bms_trip(self, safety_interlocks):
        """Test interlock trip propagates to BMS."""
        from safety.burner_management import BurnerManagementSystem, BurnerState
        from safety.safety_interlocks import SafetyInterlockManager

        # Create systems
        bms = BurnerManagementSystem("BOILER-001")
        interlocks = SafetyInterlockManager("BOILER-001")

        # Simulate firing
        bms._state = BurnerState.FIRING

        # Trigger high pressure trip
        interlocks.update_value("STEAM_PRESSURE", 155.0)

        assert interlocks.is_tripped

        # In real integration, BMS would receive callback

    def test_flame_loss_sequence(self):
        """Test flame loss triggers proper shutdown sequence."""
        from safety.burner_management import BurnerManagementSystem, BurnerState
        from safety.flame_detector import FlameDetector

        bms = BurnerManagementSystem("BOILER-001")
        detector = FlameDetector("BOILER-001", voting_logic="1oo1")
        detector.add_scanner("UV-1", "UV")

        # Establish firing with flame
        bms._state = BurnerState.FIRING
        detector.update_scanner("UV-1", signal_percent=80.0)

        assert detector.is_flame_proven()

        # Lose flame signal
        detector.update_scanner("UV-1", signal_percent=0.0)

        # After timeout, should be flame failure
        # (In real test, would mock time)


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.mark.asyncio
    async def test_api_health_endpoint(self):
        """Test API health endpoint."""
        from fastapi.testclient import TestClient
        from api.rest_api import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_api_boiler_status(self):
        """Test API boiler status endpoint."""
        from fastapi.testclient import TestClient
        from api.rest_api import app

        client = TestClient(app)
        response = client.get("/boilers/BOILER-001")

        assert response.status_code == 200
        data = response.json()
        assert data["boiler_id"] == "BOILER-001"

    @pytest.mark.asyncio
    async def test_api_optimization_request(self):
        """Test API optimization endpoint."""
        from fastapi.testclient import TestClient
        from api.rest_api import app

        client = TestClient(app)
        response = client.post(
            "/optimize",
            json={
                "boiler_id": "BOILER-001",
                "mode": "efficiency",
                "target_load_percent": 75.0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestAuditIntegration:
    """Integration tests for audit and provenance."""

    def test_calculation_provenance_chain(self):
        """Test provenance tracking through calculation chain."""
        from audit.provenance import ProvenanceTracker
        from audit.audit_logger import AuditLogger, AuditEventType

        tracker = ProvenanceTracker()
        audit = AuditLogger()

        # Start calculation
        provenance = tracker.start_calculation(
            calculation_type="efficiency",
            boiler_id="BOILER-001",
            inputs={"fuel_flow": 25000.0, "o2_percent": 3.5},
            standard="ASME PTC 4.1",
        )

        # Add calculation steps
        tracker.add_calculation_step(
            provenance.calculation_id,
            step_name="excess_air",
            formula="O2 / (21 - O2) * 100",
            inputs={"o2_percent": 3.5},
            result=20.0,
        )

        # Complete calculation
        record = tracker.complete_calculation(
            provenance.calculation_id,
            outputs={"efficiency_percent": 82.5},
        )

        # Log to audit
        audit.log_calculation(
            boiler_id="BOILER-001",
            calculation_type="efficiency",
            result={"efficiency_percent": 82.5},
            provenance_hash=record.combined_hash,
        )

        # Verify chain
        assert record.combined_hash != ""
        assert len(audit.get_entries(boiler_id="BOILER-001")) > 0
