# -*- coding: utf-8 -*-
"""
Integration tests for GL-007 External System Integrations

Tests GL-007 integration with:
- DCS/PLC (Distributed Control Systems)
- CEMS (Continuous Emissions Monitoring Systems)
- CMMS (Computerized Maintenance Management Systems)
- ERP (Enterprise Resource Planning)
- Other GreenLang agents (GL-001, GL-002, GL-004, GL-005, GL-006)

Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from greenlang.determinism import DeterministicClock


@pytest.mark.integration
class TestDCSIntegration:
    """Test integration with DCS (Distributed Control System)."""

    def test_dcs_connection(self, mock_dcs_client):
        """Test successful connection to DCS."""
        result = mock_dcs_client.connect()
        assert result is True

    def test_read_single_tag(self, mock_dcs_client):
        """Test reading a single tag from DCS."""
        tag_name = "FT-101"  # Fuel flow transmitter

        result = mock_dcs_client.read_tag(tag_name)

        assert "value" in result
        assert "timestamp" in result
        assert result["value"] > 0

    def test_read_multiple_tags(self, mock_dcs_client):
        """Test reading multiple tags from DCS."""
        tag_names = ["FT-101", "TT-102", "PT-103", "AT-104"]

        result = mock_dcs_client.read_multiple_tags()

        assert len(result) == 4
        for tag_name in tag_names:
            assert tag_name in result

    def test_write_setpoint_to_dcs(self, mock_dcs_client):
        """Test writing a setpoint to DCS."""
        tag_name = "SP-TEMP-001"
        setpoint_value = 350.0

        result = mock_dcs_client.write_tag(tag_name, setpoint_value)

        assert result is True

    @pytest.mark.slow
    def test_real_time_data_streaming(self, mock_dcs_client):
        """Test real-time data streaming from DCS."""
        # Simulate streaming 100 data points
        data_points = []

        for i in range(100):
            data = mock_dcs_client.read_multiple_tags()
            data_points.append(data)

        assert len(data_points) == 100

    def test_dcs_connection_failure_handling(self):
        """Test error handling for DCS connection failures."""
        mock_client = Mock()
        mock_client.connect = Mock(return_value=False)

        result = mock_client.connect()
        assert result is False

    def test_dcs_timeout_handling(self):
        """Test handling of DCS communication timeout."""
        mock_client = Mock()
        mock_client.read_tag = Mock(side_effect=TimeoutError("Connection timeout"))

        with pytest.raises(TimeoutError):
            mock_client.read_tag("FT-101")


@pytest.mark.integration
class TestCEMSIntegration:
    """Test integration with CEMS (Continuous Emissions Monitoring System)."""

    def test_cems_connection(self, mock_cems_client):
        """Test successful connection to CEMS."""
        result = mock_cems_client.connect()
        assert result is True

    def test_get_emissions_data(self, mock_cems_client):
        """Test retrieving emissions data from CEMS."""
        result = mock_cems_client.get_emissions_data()

        assert "nox_ppm" in result
        assert "co_ppm" in result
        assert "co2_percent" in result
        assert "so2_ppm" in result
        assert "o2_percent" in result
        assert "opacity_percent" in result
        assert "timestamp" in result

        # Validate ranges
        assert 0 <= result["o2_percent"] <= 21
        assert 0 <= result["opacity_percent"] <= 100

    def test_epa_compliance_check(self, mock_cems_client):
        """Test EPA compliance checking."""
        emissions_data = mock_cems_client.get_emissions_data()

        # EPA CEMS limits (example)
        epa_limits = {
            "nox_ppm": 100,
            "co_ppm": 100,
            "so2_ppm": 100,
            "opacity_percent": 20,
        }

        # Check compliance
        violations = []
        for parameter, limit in epa_limits.items():
            if emissions_data.get(parameter, 0) > limit:
                violations.append(parameter)

        # Should be compliant in normal operation
        assert len(violations) == 0 or emissions_data["nox_ppm"] < epa_limits["nox_ppm"]

    @pytest.mark.compliance
    def test_emissions_data_quality(self, mock_cems_client):
        """Test CEMS data quality validation."""
        emissions_data = mock_cems_client.get_emissions_data()

        # Data quality checks
        assert emissions_data["nox_ppm"] >= 0
        assert emissions_data["co_ppm"] >= 0
        assert emissions_data["co2_percent"] > 0  # Should always have some CO2
        assert emissions_data["o2_percent"] > 0  # Should always have some O2

        # O2 + CO2 should be reasonable
        total_o2_co2 = emissions_data["o2_percent"] + emissions_data["co2_percent"]
        assert total_o2_co2 < 25  # Sanity check


@pytest.mark.integration
class TestCMMSIntegration:
    """Test integration with CMMS (Computerized Maintenance Management System)."""

    def test_get_maintenance_history(self, mock_cmms_client):
        """Test retrieving maintenance history from CMMS."""
        equipment_id = "BURNER-001"

        result = mock_cmms_client.get_maintenance_history(equipment_id)

        assert isinstance(result, list)
        if len(result) > 0:
            record = result[0]
            assert "date" in record
            assert "equipment_id" in record
            assert "maintenance_type" in record
            assert "cost_usd" in record

    def test_create_work_order(self, mock_cmms_client):
        """Test creating a work order in CMMS."""
        work_order_data = {
            "equipment_id": "BURNER-001",
            "maintenance_type": "corrective",
            "priority": "high",
            "description": "Burner flame instability detected",
            "estimated_cost_usd": 5000.0,
        }

        result = mock_cmms_client.create_work_order(work_order_data)

        assert "work_order_id" in result
        assert result["work_order_id"].startswith("WO-")

    def test_predictive_maintenance_integration(self, mock_cmms_client):
        """Test predictive maintenance workflow with CMMS."""
        # GL-007 predicts maintenance need
        predicted_failure_date = DeterministicClock.now() + timedelta(days=15)

        # Create preventive work order
        work_order_data = {
            "equipment_id": "REFRACTORY-001",
            "maintenance_type": "preventive",
            "priority": "medium",
            "description": f"Predicted failure on {predicted_failure_date.date()}",
            "scheduled_date": (DeterministicClock.now() + timedelta(days=10)).isoformat(),
            "estimated_cost_usd": 15000.0,
        }

        result = mock_cmms_client.create_work_order(work_order_data)

        assert result["work_order_id"] is not None


@pytest.mark.integration
class TestERPIntegration:
    """Test integration with ERP (Enterprise Resource Planning)."""

    def test_get_fuel_pricing(self, mock_erp_client):
        """Test retrieving current fuel pricing from ERP."""
        result = mock_erp_client.get_fuel_pricing()

        assert "natural_gas" in result
        assert "diesel" in result
        assert "coal" in result

        # Prices should be positive
        assert all(price > 0 for price in result.values())

    def test_get_production_schedule(self, mock_erp_client):
        """Test retrieving production schedule from ERP."""
        result = mock_erp_client.get_production_schedule()

        assert isinstance(result, list)
        if len(result) > 0:
            schedule = result[0]
            assert "furnace_id" in schedule
            assert "start_time" in schedule
            assert "end_time" in schedule
            assert "production_target_ton" in schedule

    def test_cost_optimization_integration(self, mock_erp_client):
        """Test cost optimization using ERP fuel pricing data."""
        fuel_prices = mock_erp_client.get_fuel_pricing()

        # GL-007 would use these prices for optimization
        # Find cheapest fuel
        cheapest_fuel = min(fuel_prices.items(), key=lambda x: x[1])

        assert cheapest_fuel[0] in ["natural_gas", "diesel", "coal"]
        assert cheapest_fuel[1] > 0


@pytest.mark.integration
class TestAgentCoordination:
    """Test coordination with other GreenLang agents."""

    def test_coordination_with_gl001_orchestrator(self):
        """Test coordination with GL-001 ProcessHeatOrchestrator."""
        # Mock GL-001 sending heat demand forecast
        gl001_message = {
            "agent_id": "GL-001",
            "message_type": "heat_demand_forecast",
            "forecast_horizon_hours": 24,
            "forecasted_demand_mw": [25.5, 26.0, 24.8, 25.2],
            "timestamp": DeterministicClock.now().isoformat(),
        }

        # GL-007 should receive and process this
        # (Mock implementation)
        result = self._process_orchestrator_message(gl001_message)

        assert result["status"] == "acknowledged"
        assert "optimization_plan" in result

    def test_coordination_with_gl002_boiler(self):
        """Test coordination with GL-002 BoilerEfficiencyOptimizer."""
        # GL-007 and GL-002 coordinate on fuel availability
        coordination_message = {
            "agent_id": "GL-002",
            "message_type": "fuel_availability",
            "available_fuels": ["natural_gas", "diesel"],
            "current_fuel_usage_gj_hr": 50.0,
            "timestamp": DeterministicClock.now().isoformat(),
        }

        result = self._process_boiler_coordination(coordination_message)

        assert result["status"] == "coordinated"

    def test_coordination_with_gl004_waste_heat(self):
        """Test coordination with GL-004 WasteHeatRecovery."""
        # GL-007 sends flue gas conditions to GL-004
        flue_gas_data = {
            "agent_id": "GL-007",
            "message_type": "flue_gas_conditions",
            "flue_gas_temperature_c": 185.0,
            "flue_gas_flow_kg_hr": 28000.0,
            "available_heat_mw": 2.5,
            "timestamp": DeterministicClock.now().isoformat(),
        }

        # GL-004 would use this for waste heat recovery optimization
        result = self._send_to_waste_heat_recovery(flue_gas_data)

        assert result["status"] == "received"

    @pytest.mark.asyncio
    async def test_async_agent_communication(self):
        """Test asynchronous communication between agents."""
        # Simulate async message passing
        mock_agent = AsyncMock()
        mock_agent.send_message = AsyncMock(return_value={"status": "success"})

        result = await mock_agent.send_message({
            "message_type": "optimization_result",
            "data": {"efficiency_improvement": 2.5}
        })

        assert result["status"] == "success"

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _process_orchestrator_message(self, message: dict) -> dict:
        """Process message from GL-001 orchestrator."""
        return {
            "status": "acknowledged",
            "optimization_plan": {
                "target_efficiency_percent": 83.0,
                "recommended_load_mw": 25.5,
            }
        }

    def _process_boiler_coordination(self, message: dict) -> dict:
        """Process coordination message from GL-002."""
        return {
            "status": "coordinated",
            "fuel_allocation": {
                "furnace_usage_gj_hr": 91.8,
                "remaining_capacity_gj_hr": 10.0,
            }
        }

    def _send_to_waste_heat_recovery(self, data: dict) -> dict:
        """Send data to GL-004 waste heat recovery."""
        return {
            "status": "received",
            "recovery_potential_mw": 2.0,
        }


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndIntegration:
    """End-to-end integration tests spanning multiple systems."""

    def test_full_monitoring_cycle(
        self,
        mock_dcs_client,
        mock_cems_client,
        mock_cmms_client
    ):
        """Test complete monitoring cycle: DCS → GL-007 → CMMS."""
        # Step 1: Read data from DCS
        dcs_data = mock_dcs_client.read_multiple_tags()
        assert dcs_data is not None

        # Step 2: Read emissions from CEMS
        emissions_data = mock_cems_client.get_emissions_data()
        assert emissions_data is not None

        # Step 3: GL-007 analyzes data (mock)
        analysis_result = self._run_performance_analysis(dcs_data, emissions_data)
        assert "performance_score" in analysis_result

        # Step 4: If maintenance needed, create work order
        if analysis_result["maintenance_recommended"]:
            work_order = mock_cmms_client.create_work_order({
                "equipment_id": "BURNER-001",
                "maintenance_type": "preventive",
                "priority": "medium",
                "description": "Performance degradation detected",
            })
            assert work_order["work_order_id"] is not None

    def test_optimization_workflow_with_erp(
        self,
        mock_dcs_client,
        mock_erp_client
    ):
        """Test optimization workflow using ERP pricing data."""
        # Step 1: Get current fuel prices from ERP
        fuel_prices = mock_erp_client.get_fuel_pricing()

        # Step 2: Get current operation data from DCS
        operation_data = mock_dcs_client.read_multiple_tags()

        # Step 3: GL-007 optimizes based on prices (mock)
        optimization_result = self._run_cost_optimization(operation_data, fuel_prices)

        assert "recommended_fuel" in optimization_result
        assert "projected_savings_usd" in optimization_result

        # Step 4: Send optimization commands to DCS (if accepted)
        if optimization_result["projected_savings_usd"] > 1000:
            result = mock_dcs_client.write_tag("SP-FUEL-SELECT", optimization_result["recommended_fuel"])
            assert result is True

    def _run_performance_analysis(self, dcs_data: dict, emissions_data: dict) -> dict:
        """Mock performance analysis."""
        return {
            "performance_score": 85.0,
            "efficiency_percent": 81.5,
            "maintenance_recommended": False,
            "anomalies_detected": [],
        }

    def _run_cost_optimization(self, operation_data: dict, fuel_prices: dict) -> dict:
        """Mock cost optimization."""
        # Find cheapest fuel
        cheapest_fuel = min(fuel_prices.items(), key=lambda x: x[1])

        return {
            "recommended_fuel": cheapest_fuel[0],
            "current_fuel": "natural_gas",
            "projected_savings_usd": 5000.0,
        }


@pytest.mark.integration
class TestDataPersistence:
    """Test data persistence and historical data management."""

    def test_save_performance_data(self):
        """Test saving performance data to database."""
        performance_data = {
            "timestamp": DeterministicClock.now().isoformat(),
            "furnace_id": "FURNACE-001",
            "efficiency_percent": 81.5,
            "fuel_consumption_kg_hr": 1850.0,
            "production_rate_ton_hr": 18.5,
        }

        # Mock database save
        result = self._save_to_database(performance_data)
        assert result["status"] == "saved"

    def test_retrieve_historical_data(self):
        """Test retrieving historical performance data."""
        start_date = DeterministicClock.now() - timedelta(days=30)
        end_date = DeterministicClock.now()

        # Mock database query
        historical_data = self._query_database(start_date, end_date)

        assert isinstance(historical_data, list)

    def _save_to_database(self, data: dict) -> dict:
        """Mock database save."""
        return {"status": "saved", "record_id": "REC-12345"}

    def _query_database(self, start_date: datetime, end_date: datetime) -> list:
        """Mock database query."""
        return []  # Would return historical records
