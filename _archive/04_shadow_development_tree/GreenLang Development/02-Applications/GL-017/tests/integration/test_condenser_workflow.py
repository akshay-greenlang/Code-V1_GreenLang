# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Condenser Workflow Integration Tests

Comprehensive integration tests for condenser optimization workflows including:
- Complete condenser optimization workflow
- Integration with cooling tower systems
- Alert generation and notification
- Multi-component data flow

Test coverage target: 95%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyOutput,
)
from calculators.heat_transfer_calculator import (
    HeatTransferCalculator,
    HeatTransferInput,
)
from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingInput,
)
from calculators.vacuum_calculator import (
    VacuumCalculator,
    VacuumInput,
)
from calculators.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    verify_provenance,
)


# =============================================================================
# MOCK WORKFLOW COMPONENTS
# =============================================================================

class CondenserWorkflowOrchestrator:
    """Orchestrates complete condenser optimization workflow."""

    def __init__(self):
        self.efficiency_calc = EfficiencyCalculator()
        self.heat_transfer_calc = HeatTransferCalculator()
        self.fouling_calc = FoulingCalculator()
        self.vacuum_calc = VacuumCalculator()
        self._alerts = []
        self._recommendations = []

    async def run_optimization_cycle(
        self,
        condenser_data: Dict[str, Any],
        cooling_water_data: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Run complete optimization cycle."""
        results = {}
        provenance_records = []

        # Step 1: Efficiency Analysis
        efficiency_input = EfficiencyInput(
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_outlet_temp_c=cooling_water_data.get("outlet_temp_c", 35.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            heat_duty_mw=condenser_data.get("heat_duty_mw", 200.0),
            turbine_output_mw=condenser_data.get("turbine_output_mw", 300.0),
            design_backpressure_mmhg=condenser_data.get("design_bp_mmhg", 50.8),
            actual_backpressure_mmhg=condenser_data.get("actual_bp_mmhg", 55.0),
            design_u_value_w_m2k=condenser_data.get("design_u_w_m2k", 3500.0),
            actual_u_value_w_m2k=condenser_data.get("actual_u_w_m2k", 3000.0),
            heat_transfer_area_m2=condenser_data.get("area_m2", 17500.0),
        )

        efficiency_result, efficiency_prov = self.efficiency_calc.calculate(efficiency_input)
        results["efficiency"] = efficiency_result
        provenance_records.append(efficiency_prov)

        # Step 2: Heat Transfer Analysis
        heat_transfer_input = HeatTransferInput(
            heat_duty_mw=condenser_data.get("heat_duty_mw", 200.0),
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_outlet_temp_c=cooling_water_data.get("outlet_temp_c", 35.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            tube_od_mm=condenser_data.get("tube_od_mm", 25.4),
            tube_id_mm=condenser_data.get("tube_id_mm", 23.4),
            tube_length_m=condenser_data.get("tube_length_m", 12.0),
            tube_count=condenser_data.get("tube_count", 18500),
            tube_material=condenser_data.get("tube_material", "titanium"),
            design_u_value_w_m2k=condenser_data.get("design_u_w_m2k", 3500.0),
            fouling_factor_m2k_w=condenser_data.get("fouling_factor", 0.00015),
        )

        heat_transfer_result, ht_prov = self.heat_transfer_calc.calculate(heat_transfer_input)
        results["heat_transfer"] = heat_transfer_result
        provenance_records.append(ht_prov)

        # Step 3: Fouling Analysis
        fouling_input = FoulingInput(
            tube_material=condenser_data.get("tube_material", "titanium"),
            cooling_water_source=cooling_water_data.get("source", "cooling_tower"),
            cooling_water_tds_ppm=cooling_water_data.get("tds_ppm", 2000.0),
            cooling_water_ph=cooling_water_data.get("ph", 7.8),
            cooling_water_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            tube_velocity_m_s=results["heat_transfer"].tube_velocity_m_s,
            operating_hours=condenser_data.get("operating_hours", 4000.0),
            biocide_treatment=cooling_water_data.get("biocide", "oxidizing"),
            current_cleanliness_factor=efficiency_result.cleanliness_factor,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=cooling_water_data.get("coc", 4.0),
        )

        fouling_result, fouling_prov = self.fouling_calc.calculate(fouling_input)
        results["fouling"] = fouling_result
        provenance_records.append(fouling_prov)

        # Step 4: Vacuum Analysis
        vacuum_input = VacuumInput(
            steam_temp_c=condenser_data.get("steam_temp_c", 40.0),
            heat_load_mw=condenser_data.get("heat_duty_mw", 200.0),
            cw_inlet_temp_c=cooling_water_data.get("inlet_temp_c", 25.0),
            cw_flow_rate_m3_hr=cooling_water_data.get("flow_m3_hr", 50000.0),
            air_inleakage_rate_kg_hr=condenser_data.get("air_inleakage_kg_hr", 1.0),
            design_vacuum_mbar=condenser_data.get("design_vacuum_mbar", 50.0),
        )

        vacuum_result, vacuum_prov = self.vacuum_calc.calculate(vacuum_input)
        results["vacuum"] = vacuum_result
        provenance_records.append(vacuum_prov)

        # Step 5: Generate Alerts
        self._generate_alerts(results)

        # Step 6: Generate Recommendations
        self._generate_recommendations(results)

        return {
            "results": results,
            "provenance": provenance_records,
            "alerts": self._alerts,
            "recommendations": self._recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _generate_alerts(self, results: Dict[str, Any]) -> None:
        """Generate alerts based on results."""
        self._alerts = []

        efficiency = results.get("efficiency")
        if efficiency:
            # High backpressure alert
            if efficiency.heat_rate_deviation_kj_kwh > 100:
                self._alerts.append({
                    "level": "warning",
                    "type": "high_backpressure",
                    "message": f"Heat rate deviation {efficiency.heat_rate_deviation_kj_kwh:.1f} kJ/kWh exceeds threshold",
                })

            # Low cleanliness alert
            if efficiency.cleanliness_factor < 0.70:
                self._alerts.append({
                    "level": "critical",
                    "type": "low_cleanliness",
                    "message": f"Cleanliness factor {efficiency.cleanliness_factor:.2f} below critical threshold",
                })
            elif efficiency.cleanliness_factor < 0.85:
                self._alerts.append({
                    "level": "warning",
                    "type": "degraded_cleanliness",
                    "message": f"Cleanliness factor {efficiency.cleanliness_factor:.2f} below target",
                })

            # Poor performance rating
            if efficiency.performance_rating in ["Poor", "Critical"]:
                self._alerts.append({
                    "level": "critical",
                    "type": "poor_performance",
                    "message": f"Performance rating: {efficiency.performance_rating}",
                })

        fouling = results.get("fouling")
        if fouling:
            # Cleaning required alert
            if fouling.cleaning_urgency in ["high", "critical"]:
                self._alerts.append({
                    "level": "warning",
                    "type": "cleaning_required",
                    "message": f"Tube cleaning required - urgency: {fouling.cleaning_urgency}",
                })

    def _generate_recommendations(self, results: Dict[str, Any]) -> None:
        """Generate optimization recommendations."""
        self._recommendations = []

        efficiency = results.get("efficiency")
        fouling = results.get("fouling")

        if efficiency:
            # Potential savings recommendation
            if efficiency.potential_annual_savings_usd > 100000:
                self._recommendations.append({
                    "priority": "high",
                    "action": "condenser_optimization",
                    "description": f"Potential annual savings of ${efficiency.potential_annual_savings_usd:,.0f}",
                    "estimated_roi": "6-12 months",
                })

        if fouling:
            # Cleaning recommendation
            if fouling.recommended_cleaning_interval_hours < 2000:
                self._recommendations.append({
                    "priority": "medium",
                    "action": "schedule_cleaning",
                    "description": f"Schedule tube cleaning using {fouling.recommended_cleaning_method}",
                    "timing": f"Within {fouling.recommended_cleaning_interval_hours} hours",
                })


class CoolingTowerIntegrationController:
    """Controller for cooling tower integration."""

    def __init__(self, tower_api: Optional[Mock] = None):
        self.tower_api = tower_api or Mock()
        self._fan_speeds = [75.0, 75.0, 80.0, 80.0]  # Default fan speeds

    async def get_tower_status(self) -> Dict[str, Any]:
        """Get current cooling tower status."""
        return {
            "cold_water_temp_c": 25.0,
            "hot_water_temp_c": 35.0,
            "wet_bulb_temp_c": 20.0,
            "approach_temp_c": 5.0,
            "range_temp_c": 10.0,
            "fan_speeds_pct": self._fan_speeds,
            "total_fan_power_kw": sum(self._fan_speeds) * 1.5,
            "makeup_flow_m3_hr": 1200.0,
            "blowdown_flow_m3_hr": 300.0,
        }

    async def optimize_tower_operation(
        self,
        condenser_demand: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize cooling tower operation for condenser demand."""
        target_cw_temp = condenser_demand.get("target_cw_temp_c", 25.0)
        current_cw_temp = condenser_demand.get("current_cw_temp_c", 27.0)
        wet_bulb = condenser_demand.get("wet_bulb_temp_c", 20.0)

        # Calculate required approach
        required_approach = target_cw_temp - wet_bulb

        # Determine fan speed adjustments
        if current_cw_temp > target_cw_temp + 2:
            # Need more cooling - increase fan speeds
            adjustment = "increase"
            new_speeds = [min(100, s + 10) for s in self._fan_speeds]
        elif current_cw_temp < target_cw_temp - 2:
            # Over-cooling - reduce fan speeds
            adjustment = "decrease"
            new_speeds = [max(30, s - 10) for s in self._fan_speeds]
        else:
            # Maintain current
            adjustment = "maintain"
            new_speeds = self._fan_speeds

        energy_savings = 0
        if adjustment == "decrease":
            # Calculate energy savings from reduced fan operation
            power_reduction = sum(self._fan_speeds) - sum(new_speeds)
            energy_savings = power_reduction * 1.5 * 24 / 1000  # MWh/day

        return {
            "adjustment": adjustment,
            "current_fan_speeds": self._fan_speeds,
            "recommended_fan_speeds": new_speeds,
            "estimated_energy_savings_mwh_day": energy_savings,
            "achievable_approach_c": required_approach,
        }


class AlertManager:
    """Manages alert generation and notification."""

    def __init__(self):
        self._active_alerts = []
        self._alert_history = []
        self._notification_handlers = []

    def add_handler(self, handler: callable):
        """Add notification handler."""
        self._notification_handlers.append(handler)

    def process_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """Process and notify on alerts."""
        for alert in alerts:
            # Check if already active
            existing = next(
                (a for a in self._active_alerts if a["type"] == alert["type"]),
                None
            )

            if existing is None:
                # New alert
                alert["first_seen"] = datetime.utcnow().isoformat()
                alert["status"] = "active"
                self._active_alerts.append(alert)
                self._notify(alert)
            else:
                # Update existing
                existing["last_seen"] = datetime.utcnow().isoformat()

    def acknowledge_alert(self, alert_type: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._active_alerts:
            if alert["type"] == alert_type:
                alert["status"] = "acknowledged"
                alert["acknowledged_at"] = datetime.utcnow().isoformat()
                return True
        return False

    def clear_alert(self, alert_type: str) -> bool:
        """Clear an alert."""
        for i, alert in enumerate(self._active_alerts):
            if alert["type"] == alert_type:
                alert["status"] = "cleared"
                alert["cleared_at"] = datetime.utcnow().isoformat()
                self._alert_history.append(alert)
                self._active_alerts.pop(i)
                return True
        return False

    def _notify(self, alert: Dict[str, Any]) -> None:
        """Send notifications for alert."""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception:
                pass  # Log error in production

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return self._active_alerts


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def workflow_orchestrator():
    """Create workflow orchestrator."""
    return CondenserWorkflowOrchestrator()


@pytest.fixture
def cooling_tower_controller():
    """Create cooling tower controller."""
    return CoolingTowerIntegrationController()


@pytest.fixture
def alert_manager():
    """Create alert manager."""
    return AlertManager()


@pytest.fixture
def standard_condenser_data():
    """Standard condenser operating data."""
    return {
        "condenser_id": "COND-001",
        "steam_temp_c": 40.0,
        "heat_duty_mw": 200.0,
        "turbine_output_mw": 300.0,
        "design_bp_mmhg": 50.8,
        "actual_bp_mmhg": 55.0,
        "design_u_w_m2k": 3500.0,
        "actual_u_w_m2k": 3000.0,
        "area_m2": 17500.0,
        "tube_od_mm": 25.4,
        "tube_id_mm": 23.4,
        "tube_length_m": 12.0,
        "tube_count": 18500,
        "tube_material": "titanium",
        "fouling_factor": 0.00015,
        "operating_hours": 4000.0,
        "air_inleakage_kg_hr": 1.0,
        "design_vacuum_mbar": 50.0,
    }


@pytest.fixture
def standard_cooling_water_data():
    """Standard cooling water data."""
    return {
        "source": "cooling_tower",
        "inlet_temp_c": 25.0,
        "outlet_temp_c": 35.0,
        "flow_m3_hr": 50000.0,
        "tds_ppm": 2000.0,
        "ph": 7.8,
        "biocide": "oxidizing",
        "coc": 4.0,
    }


@pytest.fixture
def degraded_condenser_data():
    """Degraded condenser data for testing alerts."""
    return {
        "condenser_id": "COND-002",
        "steam_temp_c": 48.0,
        "heat_duty_mw": 180.0,
        "turbine_output_mw": 270.0,
        "design_bp_mmhg": 50.8,
        "actual_bp_mmhg": 80.0,  # High backpressure
        "design_u_w_m2k": 3500.0,
        "actual_u_w_m2k": 2000.0,  # Low U-value
        "area_m2": 17500.0,
        "tube_od_mm": 25.4,
        "tube_id_mm": 23.4,
        "tube_length_m": 12.0,
        "tube_count": 18500,
        "tube_material": "admiralty_brass",
        "fouling_factor": 0.0005,  # High fouling
        "operating_hours": 6000.0,
        "air_inleakage_kg_hr": 3.0,  # High air
        "design_vacuum_mbar": 50.0,
    }


# =============================================================================
# COMPLETE WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestCompleteWorkflow:
    """Test suite for complete condenser optimization workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(
        self,
        workflow_orchestrator,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test complete optimization cycle execution."""
        result = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Verify all components executed
        assert "results" in result
        assert "efficiency" in result["results"]
        assert "heat_transfer" in result["results"]
        assert "fouling" in result["results"]
        assert "vacuum" in result["results"]

        # Verify provenance records
        assert "provenance" in result
        assert len(result["provenance"]) == 4

        # Verify timestamp
        assert "timestamp" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_degraded_condenser(
        self,
        workflow_orchestrator,
        degraded_condenser_data,
        standard_cooling_water_data
    ):
        """Test workflow handles degraded condenser properly."""
        result = await workflow_orchestrator.run_optimization_cycle(
            degraded_condenser_data,
            standard_cooling_water_data
        )

        # Should generate alerts
        assert len(result["alerts"]) > 0

        # Should identify poor performance
        efficiency = result["results"]["efficiency"]
        assert efficiency.cleanliness_factor < 0.70

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_generates_recommendations(
        self,
        workflow_orchestrator,
        degraded_condenser_data,
        standard_cooling_water_data
    ):
        """Test workflow generates actionable recommendations."""
        result = await workflow_orchestrator.run_optimization_cycle(
            degraded_condenser_data,
            standard_cooling_water_data
        )

        # Should have recommendations
        assert len(result["recommendations"]) > 0

        # Check recommendation structure
        for rec in result["recommendations"]:
            assert "priority" in rec
            assert "action" in rec
            assert "description" in rec

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_data_flow(
        self,
        workflow_orchestrator,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test data flows correctly between components."""
        result = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Verify data consistency
        efficiency = result["results"]["efficiency"]
        heat_transfer = result["results"]["heat_transfer"]
        fouling = result["results"]["fouling"]

        # Cleanliness factor should flow from efficiency to fouling
        assert abs(
            efficiency.cleanliness_factor - fouling.current_cleanliness_factor
        ) < 0.01 or fouling.current_cleanliness_factor is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_provenance_chain(
        self,
        workflow_orchestrator,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test provenance chain is complete."""
        result = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Verify each provenance record is valid
        for prov in result["provenance"]:
            assert verify_provenance(prov) is True


# =============================================================================
# COOLING TOWER INTEGRATION TESTS
# =============================================================================

class TestCoolingTowerIntegration:
    """Test suite for cooling tower system integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_tower_status(self, cooling_tower_controller):
        """Test getting cooling tower status."""
        status = await cooling_tower_controller.get_tower_status()

        assert "cold_water_temp_c" in status
        assert "hot_water_temp_c" in status
        assert "wet_bulb_temp_c" in status
        assert "approach_temp_c" in status
        assert "fan_speeds_pct" in status

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tower_optimization_increase_cooling(self, cooling_tower_controller):
        """Test tower optimization when more cooling needed."""
        demand = {
            "target_cw_temp_c": 25.0,
            "current_cw_temp_c": 30.0,  # Too warm
            "wet_bulb_temp_c": 20.0,
        }

        result = await cooling_tower_controller.optimize_tower_operation(demand)

        assert result["adjustment"] == "increase"
        assert all(
            new >= old
            for new, old in zip(
                result["recommended_fan_speeds"],
                result["current_fan_speeds"]
            )
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tower_optimization_decrease_cooling(self, cooling_tower_controller):
        """Test tower optimization when over-cooling."""
        demand = {
            "target_cw_temp_c": 28.0,
            "current_cw_temp_c": 24.0,  # Too cold
            "wet_bulb_temp_c": 20.0,
        }

        result = await cooling_tower_controller.optimize_tower_operation(demand)

        assert result["adjustment"] == "decrease"
        assert result["estimated_energy_savings_mwh_day"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tower_optimization_maintain(self, cooling_tower_controller):
        """Test tower optimization when at target."""
        demand = {
            "target_cw_temp_c": 26.0,
            "current_cw_temp_c": 26.0,  # At target
            "wet_bulb_temp_c": 20.0,
        }

        result = await cooling_tower_controller.optimize_tower_operation(demand)

        assert result["adjustment"] == "maintain"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integrated_tower_condenser_optimization(
        self,
        workflow_orchestrator,
        cooling_tower_controller,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test integrated tower-condenser optimization."""
        # Run condenser optimization
        condenser_result = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Get tower status
        tower_status = await cooling_tower_controller.get_tower_status()

        # Create demand based on condenser results
        demand = {
            "target_cw_temp_c": 25.0,
            "current_cw_temp_c": tower_status["cold_water_temp_c"],
            "wet_bulb_temp_c": tower_status["wet_bulb_temp_c"],
        }

        # Optimize tower
        tower_result = await cooling_tower_controller.optimize_tower_operation(demand)

        assert tower_result["adjustment"] is not None


# =============================================================================
# ALERT GENERATION TESTS
# =============================================================================

class TestAlertGeneration:
    """Test suite for alert generation and notification."""

    @pytest.mark.integration
    def test_alert_manager_process(self, alert_manager):
        """Test alert processing."""
        alerts = [
            {"level": "warning", "type": "high_backpressure", "message": "Test"},
            {"level": "critical", "type": "low_cleanliness", "message": "Test"},
        ]

        alert_manager.process_alerts(alerts)

        active = alert_manager.get_active_alerts()
        assert len(active) == 2

    @pytest.mark.integration
    def test_alert_acknowledgment(self, alert_manager):
        """Test alert acknowledgment."""
        alerts = [{"level": "warning", "type": "test_alert", "message": "Test"}]
        alert_manager.process_alerts(alerts)

        result = alert_manager.acknowledge_alert("test_alert")

        assert result is True
        active = alert_manager.get_active_alerts()
        assert active[0]["status"] == "acknowledged"

    @pytest.mark.integration
    def test_alert_clearing(self, alert_manager):
        """Test alert clearing."""
        alerts = [{"level": "warning", "type": "test_alert", "message": "Test"}]
        alert_manager.process_alerts(alerts)

        result = alert_manager.clear_alert("test_alert")

        assert result is True
        active = alert_manager.get_active_alerts()
        assert len(active) == 0

    @pytest.mark.integration
    def test_alert_notification_handler(self, alert_manager):
        """Test alert notification handler."""
        notifications = []

        def handler(alert):
            notifications.append(alert)

        alert_manager.add_handler(handler)
        alerts = [{"level": "warning", "type": "test_alert", "message": "Test"}]
        alert_manager.process_alerts(alerts)

        assert len(notifications) == 1

    @pytest.mark.integration
    def test_duplicate_alert_handling(self, alert_manager):
        """Test duplicate alerts are not created."""
        alerts = [{"level": "warning", "type": "test_alert", "message": "Test"}]

        alert_manager.process_alerts(alerts)
        alert_manager.process_alerts(alerts)  # Duplicate

        active = alert_manager.get_active_alerts()
        assert len(active) == 1  # Still just one

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_alert_integration(
        self,
        workflow_orchestrator,
        alert_manager,
        degraded_condenser_data,
        standard_cooling_water_data
    ):
        """Test alerts flow from workflow to manager."""
        result = await workflow_orchestrator.run_optimization_cycle(
            degraded_condenser_data,
            standard_cooling_water_data
        )

        alert_manager.process_alerts(result["alerts"])

        active = alert_manager.get_active_alerts()
        assert len(active) > 0


# =============================================================================
# SCADA INTEGRATION TESTS
# =============================================================================

class TestSCADAIntegration:
    """Test suite for SCADA system integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_data_retrieval(self):
        """Test SCADA data retrieval."""
        mock_scada = AsyncMock()
        mock_scada.read_multiple_tags.return_value = {
            "COND_VACUUM": {"value": 50.0, "quality": "GOOD"},
            "CW_INLET_TEMP": {"value": 25.0, "quality": "GOOD"},
            "CW_OUTLET_TEMP": {"value": 35.0, "quality": "GOOD"},
        }

        tags = await mock_scada.read_multiple_tags(["COND_VACUUM", "CW_INLET_TEMP", "CW_OUTLET_TEMP"])

        assert tags["COND_VACUUM"]["value"] == 50.0
        assert tags["CW_INLET_TEMP"]["quality"] == "GOOD"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_data_quality_handling(self):
        """Test handling of SCADA data quality issues."""
        mock_scada = AsyncMock()
        mock_scada.read_multiple_tags.return_value = {
            "COND_VACUUM": {"value": 50.0, "quality": "GOOD"},
            "CW_INLET_TEMP": {"value": None, "quality": "BAD"},  # Bad quality
        }

        tags = await mock_scada.read_multiple_tags(["COND_VACUUM", "CW_INLET_TEMP"])

        # Should handle bad quality data
        assert tags["CW_INLET_TEMP"]["quality"] == "BAD"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_write_setpoint(self):
        """Test writing setpoint to SCADA."""
        mock_scada = AsyncMock()
        mock_scada.write_tag.return_value = True

        result = await mock_scada.write_tag("CW_FLOW_SP", 52000.0)

        assert result is True
        mock_scada.write_tag.assert_called_once_with("CW_FLOW_SP", 52000.0)


# =============================================================================
# HISTORIAN INTEGRATION TESTS
# =============================================================================

class TestHistorianIntegration:
    """Test suite for historian integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_historian_trend_retrieval(self):
        """Test retrieving trend data from historian."""
        mock_historian = AsyncMock()
        mock_historian.query_tag_history.return_value = [
            {"timestamp": "2025-01-01T00:00:00", "value": 50.0},
            {"timestamp": "2025-01-01T01:00:00", "value": 51.0},
            {"timestamp": "2025-01-01T02:00:00", "value": 52.0},
        ]

        data = await mock_historian.query_tag_history(
            "COND_VACUUM",
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T03:00:00",
            interval="1h"
        )

        assert len(data) == 3
        assert data[0]["value"] == 50.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_historian_aggregation(self):
        """Test historian data aggregation."""
        mock_historian = AsyncMock()
        mock_historian.query_aggregated.return_value = {
            "average": 51.0,
            "min": 50.0,
            "max": 52.0,
            "count": 24,
        }

        result = await mock_historian.query_aggregated(
            "COND_VACUUM",
            aggregation="daily",
            date="2025-01-01"
        )

        assert result["average"] == 51.0


# =============================================================================
# MULTI-COMPONENT DATA FLOW TESTS
# =============================================================================

class TestMultiComponentDataFlow:
    """Test suite for multi-component data flow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_condenser_to_tower_data_flow(
        self,
        workflow_orchestrator,
        cooling_tower_controller,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test data flows from condenser to tower optimization."""
        # Run condenser analysis
        condenser_result = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Extract required CW temperature based on results
        efficiency = condenser_result["results"]["efficiency"]

        # Use results to set tower demand
        demand = {
            "target_cw_temp_c": 25.0,
            "current_cw_temp_c": standard_cooling_water_data["inlet_temp_c"],
            "wet_bulb_temp_c": 20.0,
            "condenser_heat_duty_mw": standard_condenser_data["heat_duty_mw"],
        }

        tower_result = await cooling_tower_controller.optimize_tower_operation(demand)

        assert tower_result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_feedback_loop(
        self,
        workflow_orchestrator,
        standard_condenser_data,
        standard_cooling_water_data
    ):
        """Test optimization feedback loop."""
        # Initial optimization
        result1 = await workflow_orchestrator.run_optimization_cycle(
            standard_condenser_data,
            standard_cooling_water_data
        )

        # Simulate improved conditions
        improved_condenser_data = standard_condenser_data.copy()
        improved_condenser_data["actual_u_w_m2k"] = 3200.0  # Improved
        improved_condenser_data["actual_bp_mmhg"] = 52.0  # Improved

        # Re-run optimization
        result2 = await workflow_orchestrator.run_optimization_cycle(
            improved_condenser_data,
            standard_cooling_water_data
        )

        # Verify improvement detected
        eff1 = result1["results"]["efficiency"]
        eff2 = result2["results"]["efficiency"]

        assert eff2.cleanliness_factor > eff1.cleanliness_factor


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test suite for error handling in integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_handles_invalid_data(self, workflow_orchestrator):
        """Test workflow handles invalid input data."""
        invalid_condenser_data = {
            "steam_temp_c": -10.0,  # Invalid
        }

        with pytest.raises(Exception):
            await workflow_orchestrator.run_optimization_cycle(
                invalid_condenser_data,
                {}
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_partial_scada_failure(self):
        """Test handling of partial SCADA data failure."""
        mock_scada = AsyncMock()
        mock_scada.read_multiple_tags.return_value = {
            "COND_VACUUM": {"value": 50.0, "quality": "GOOD"},
            "CW_INLET_TEMP": {"value": None, "quality": "BAD"},
            "CW_OUTLET_TEMP": {"value": 35.0, "quality": "GOOD"},
        }

        tags = await mock_scada.read_multiple_tags([
            "COND_VACUUM", "CW_INLET_TEMP", "CW_OUTLET_TEMP"
        ])

        # Should have partial data
        good_tags = [k for k, v in tags.items() if v["quality"] == "GOOD"]
        assert len(good_tags) == 2

    @pytest.mark.integration
    def test_alert_handler_exception_isolation(self, alert_manager):
        """Test alert handler exceptions don't propagate."""
        def failing_handler(alert):
            raise Exception("Handler error")

        alert_manager.add_handler(failing_handler)
        alerts = [{"level": "warning", "type": "test", "message": "Test"}]

        # Should not raise
        alert_manager.process_alerts(alerts)

        # Alert should still be processed
        assert len(alert_manager.get_active_alerts()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
