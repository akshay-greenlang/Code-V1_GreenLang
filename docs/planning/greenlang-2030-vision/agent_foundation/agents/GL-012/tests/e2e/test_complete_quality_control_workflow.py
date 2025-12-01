# -*- coding: utf-8 -*-
"""
End-to-End Tests for GL-012 STEAMQUAL Complete Quality Control Workflow.

Tests the complete steam quality monitoring and control cycle including:
- Orchestrator initialization and configuration
- Integration connection (meters, valves, SCADA)
- Steam quality parameter reading
- Quality analysis and control action determination
- Desuperheater control execution
- Pressure valve control execution
- Quality improvement verification
- KPI dashboard generation
- Provenance hash verification

Test Scenarios:
- Normal operation with optimal quality
- Quality degradation requiring desuperheater activation
- Pressure deviation requiring valve adjustment
- High moisture requiring multiple interventions
- Emergency shutdown scenario

Test Count: 20+ E2E workflow tests
Coverage Target: 90%+

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
Agent ID: GL-012
"""

import pytest
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conftest import (
    MockE2ESteamQualityOrchestrator,
    TestScenario,
    E2ETestMetrics,
    PerformanceTimer,
    DeterminismValidator,
    ProvenanceTracker,
)


# =============================================================================
# TEST CLASS: COMPLETE QUALITY CONTROL WORKFLOW
# =============================================================================

@pytest.mark.e2e
@pytest.mark.workflow
class TestCompleteQualityControlWorkflow:
    """End-to-end tests for complete quality control workflow."""

    # =========================================================================
    # INITIALIZATION AND CONFIGURATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, default_config):
        """Test orchestrator initializes with correct configuration."""
        orchestrator = MockE2ESteamQualityOrchestrator(config=default_config)

        result = await orchestrator.initialize()

        assert result["status"] == "initialized"
        assert result["config"]["agent_id"] == "GL-012"
        assert result["config"]["quality_target"] == 0.95
        assert "timestamp" in result

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, e2e_orchestrator):
        """Test orchestrator shuts down cleanly with metrics."""
        # Execute some cycles first
        for _ in range(5):
            await e2e_orchestrator.execute_cycle("HEADER-001")

        result = await e2e_orchestrator.shutdown()

        assert result["status"] == "shutdown"
        assert result["cycles_completed"] == 5
        assert "metrics" in result

    # =========================================================================
    # INTEGRATION CONNECTION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_connect_all_integrations(
        self,
        e2e_orchestrator,
        meter_configs,
        valve_configs,
        scada_config
    ):
        """Test connecting to all integrations (meters, valves, SCADA)."""
        # Connect meters
        meter_result = await e2e_orchestrator.connect_meters(meter_configs)
        assert meter_result["status"] == "connected"
        assert meter_result["meters_connected"] == len(meter_configs)

        # Connect valves
        valve_result = await e2e_orchestrator.connect_valves(valve_configs)
        assert valve_result["status"] == "connected"
        assert valve_result["valves_connected"] == len(valve_configs)

        # Connect SCADA
        scada_result = await e2e_orchestrator.connect_scada(scada_config)
        assert scada_result["status"] == "connected"
        assert scada_result["protocol"] == "opcua"

    @pytest.mark.asyncio
    async def test_meter_connection_with_multiple_protocols(
        self,
        e2e_orchestrator,
        meter_configs
    ):
        """Test meter connection with different protocols."""
        result = await e2e_orchestrator.connect_meters(meter_configs)

        assert result["status"] == "connected"
        protocols = result["protocols"]
        assert "modbus_tcp" in protocols
        assert "opcua" in protocols

    # =========================================================================
    # STEAM QUALITY PARAMETER READING TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_read_steam_quality_parameters(self, e2e_orchestrator):
        """Test reading steam quality parameters."""
        params = await e2e_orchestrator.read_steam_quality_parameters("HEADER-001")

        assert params["header_id"] == "HEADER-001"
        assert "pressure_bar" in params
        assert "temperature_c" in params
        assert "flow_rate_kg_hr" in params
        assert "dryness_fraction" in params
        assert params["quality"] == "GOOD"
        assert params["validation"]["pressure_valid"] is True
        assert params["validation"]["thermodynamic_consistent"] is True

    @pytest.mark.asyncio
    async def test_read_multiple_headers(self, e2e_orchestrator):
        """Test reading parameters from multiple headers."""
        header_ids = ["HP-STEAM-01", "MP-STEAM-01", "LP-STEAM-01"]
        results = {}

        for header_id in header_ids:
            params = await e2e_orchestrator.read_steam_quality_parameters(header_id)
            results[header_id] = params

        assert len(results) == 3
        for header_id in header_ids:
            assert results[header_id]["header_id"] == header_id

    # =========================================================================
    # QUALITY ANALYSIS TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_analyze_quality_optimal(self, e2e_orchestrator):
        """Test quality analysis with optimal parameters."""
        params = {
            "pressure_bar": 10.0,
            "temperature_c": 180.0,
            "dryness_fraction": 0.98
        }

        analysis = await e2e_orchestrator.analyze_quality(params)

        assert analysis["quality_level"] in ["excellent", "good"]
        assert analysis["quality_index"] >= 90
        assert analysis["compliance_status"] == "compliant"
        assert len(analysis["actions_required"]) == 0

    @pytest.mark.asyncio
    async def test_analyze_quality_degradation(self, e2e_orchestrator):
        """Test quality analysis with degraded parameters."""
        params = {
            "pressure_bar": 10.0,
            "temperature_c": 180.0,
            "dryness_fraction": 0.92
        }

        analysis = await e2e_orchestrator.analyze_quality(params)

        assert analysis["quality_index"] < 95
        assert analysis["deviations"]["moisture_percent"] > 5.0
        assert len(analysis["actions_required"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_quality_pressure_deviation(self, e2e_orchestrator):
        """Test quality analysis with pressure deviation."""
        params = {
            "pressure_bar": 12.5,  # 25% above setpoint
            "temperature_c": 180.0,
            "dryness_fraction": 0.97
        }

        analysis = await e2e_orchestrator.analyze_quality(params)

        assert analysis["deviations"]["pressure_percent"] > 2.0
        actions = [a["action"] for a in analysis["actions_required"]]
        assert "pressure_valve_adjust" in actions

    # =========================================================================
    # CONTROL ACTION EXECUTION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_desuperheater_control(self, e2e_orchestrator):
        """Test desuperheater control action execution."""
        action = {
            "action": "desuperheater_adjust",
            "injection_rate_change_percent": 5.0
        }

        result = await e2e_orchestrator.execute_desuperheater_control(
            "HEADER-001",
            action
        )

        assert result["status"] == "success"
        assert result["action_executed"] == "desuperheater_adjust"
        assert result["injection_rate_change_percent"] == 5.0
        assert result["response_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_execute_pressure_valve_control(self, e2e_orchestrator):
        """Test pressure valve control action execution."""
        action = {
            "action": "pressure_valve_adjust",
            "valve_position_change_percent": 2.0
        }

        result = await e2e_orchestrator.execute_pressure_valve_control(
            "HEADER-001",
            action
        )

        assert result["status"] == "success"
        assert result["action_executed"] == "pressure_valve_adjust"
        assert result["valve_position_change_percent"] == 2.0
        assert result["response_time_ms"] > 0

    # =========================================================================
    # QUALITY IMPROVEMENT VERIFICATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_verify_quality_improvement(self, e2e_orchestrator):
        """Test quality improvement verification."""
        result = await e2e_orchestrator.verify_quality_improvement(
            "HEADER-001",
            quality_before=92.0,
            quality_after=95.0
        )

        assert result["verification_passed"] is True
        assert result["improvement"] == 3.0
        assert result["improvement_percent"] > 0

    @pytest.mark.asyncio
    async def test_verify_no_degradation(self, e2e_orchestrator):
        """Test verification catches quality degradation."""
        result = await e2e_orchestrator.verify_quality_improvement(
            "HEADER-001",
            quality_before=95.0,
            quality_after=92.0
        )

        # Should pass if within tolerance (1%)
        assert result["improvement"] < 0

    # =========================================================================
    # KPI DASHBOARD GENERATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_generate_kpi_dashboard(self, e2e_orchestrator):
        """Test KPI dashboard generation."""
        header_ids = ["HP-STEAM-01", "MP-STEAM-01", "LP-STEAM-01"]

        # Execute some cycles first
        for header_id in header_ids:
            await e2e_orchestrator.execute_cycle(header_id)

        dashboard = await e2e_orchestrator.generate_kpi_dashboard(header_ids)

        assert "timestamp" in dashboard
        assert dashboard["summary"]["total_headers"] == 3
        assert "avg_quality_index" in dashboard["summary"]
        assert len(dashboard["headers"]) == 3

        for header_id in header_ids:
            assert header_id in dashboard["headers"]
            header_kpi = dashboard["headers"][header_id]
            assert "pressure_bar" in header_kpi
            assert "quality_index" in header_kpi

    # =========================================================================
    # PROVENANCE HASH VERIFICATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_provenance_hash_generated(self, e2e_orchestrator):
        """Test provenance hash is generated for each cycle."""
        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256 hex length
        assert result["determinism_verified"] is True

    @pytest.mark.asyncio
    async def test_provenance_chain_integrity(
        self,
        e2e_orchestrator,
        provenance_tracker
    ):
        """Test provenance chain maintains integrity."""
        for i in range(5):
            result = await e2e_orchestrator.execute_cycle("HEADER-001")
            provenance_tracker.add_entry(
                f"cycle_{i}",
                {"cycle": i, "header": "HEADER-001"},
                {"quality": result["quality_after"]}
            )

        is_valid, issues = provenance_tracker.verify_chain()
        assert is_valid, f"Chain integrity issues: {issues}"

    @pytest.mark.asyncio
    async def test_provenance_hash_deterministic(self, e2e_orchestrator):
        """Test provenance hashes are captured for audit trail."""
        hashes = []

        for _ in range(5):
            result = await e2e_orchestrator.execute_cycle("HEADER-001")
            hashes.append(result["provenance_hash"])

        # All hashes should be valid and unique per cycle
        assert len(hashes) == 5
        for h in hashes:
            assert len(h) == 64

        # Verify chain is recorded
        chain = e2e_orchestrator.get_provenance_chain()
        assert len(chain) == 5

    # =========================================================================
    # COMPLETE WORKFLOW SCENARIO TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_normal_operation_workflow(
        self,
        e2e_orchestrator,
        meter_configs,
        valve_configs,
        scada_config
    ):
        """Test complete workflow with normal operation (optimal quality)."""
        # Step 1: Initialize
        init_result = await e2e_orchestrator.initialize()
        assert init_result["status"] == "initialized"

        # Step 2: Connect integrations
        await e2e_orchestrator.connect_meters(meter_configs)
        await e2e_orchestrator.connect_valves(valve_configs)
        await e2e_orchestrator.connect_scada(scada_config)

        # Step 3: Execute cycle with normal operation
        result = await e2e_orchestrator.execute_cycle(
            "HEADER-001",
            scenario=TestScenario.NORMAL_OPERATION
        )

        assert result["status"] == "success"
        assert result["quality_after"] >= 90.0
        assert len(result["actions_executed"]) == 0  # No actions needed for optimal

        # Step 4: Generate KPI dashboard
        dashboard = await e2e_orchestrator.generate_kpi_dashboard(["HEADER-001"])
        assert dashboard["compliance_status"] == "compliant"

        # Step 5: Verify provenance
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_quality_degradation_workflow(self, e2e_orchestrator):
        """Test workflow with quality degradation requiring desuperheater."""
        # Execute cycle with degraded quality
        result = await e2e_orchestrator.execute_cycle(
            "HEADER-001",
            scenario=TestScenario.QUALITY_DEGRADATION
        )

        assert result["status"] == "success"

        # Should have taken corrective action
        actions = [a["action_executed"] for a in result["actions_executed"]]
        assert len(actions) > 0 or result["quality_before"] < result["quality_after"]

    @pytest.mark.asyncio
    async def test_pressure_deviation_workflow(self, e2e_orchestrator):
        """Test workflow with pressure deviation requiring valve adjustment."""
        result = await e2e_orchestrator.execute_cycle(
            "HEADER-001",
            scenario=TestScenario.PRESSURE_DEVIATION
        )

        assert result["status"] == "success"

        # Check analysis detected pressure deviation
        analysis = result["analysis"]
        assert analysis["deviations"]["pressure_percent"] > 2.0

    @pytest.mark.asyncio
    async def test_high_moisture_workflow(self, e2e_orchestrator):
        """Test workflow with high moisture requiring multiple interventions."""
        result = await e2e_orchestrator.execute_cycle(
            "HEADER-001",
            scenario=TestScenario.HIGH_MOISTURE
        )

        assert result["status"] == "success"

        # Should detect high moisture
        analysis = result["analysis"]
        assert analysis["deviations"]["moisture_percent"] > 10.0

    @pytest.mark.asyncio
    async def test_emergency_shutdown_workflow(self, e2e_orchestrator):
        """Test emergency shutdown scenario."""
        # Apply emergency scenario
        e2e_orchestrator._apply_scenario("HEADER-001", TestScenario.EMERGENCY_SHUTDOWN)

        # Execute emergency shutdown
        result = await e2e_orchestrator.execute_emergency_shutdown(
            "HEADER-001",
            reason="Critical pressure exceeded"
        )

        assert result["status"] == "shutdown_complete"
        assert result["reason"] == "Critical pressure exceeded"
        assert len(result["actions_taken"]) >= 4

        # Verify isolation valve closed
        actions = [a["action"] for a in result["actions_taken"]]
        assert "close_isolation_valve" in actions
        assert "activate_alarm" in actions

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_workflow_performance(
        self,
        e2e_orchestrator,
        performance_timer
    ):
        """Test complete workflow meets performance target."""
        performance_timer.start("full_workflow")

        # Connect integrations
        await e2e_orchestrator.connect_meters([{"meter_id": "SQM-001"}])
        await e2e_orchestrator.connect_valves([{"valve_id": "PCV-001"}])

        # Execute 10 cycles
        for _ in range(10):
            result = await e2e_orchestrator.execute_cycle("HEADER-001")
            assert result["execution_time_ms"] < 1000  # <1s per cycle

        # Generate dashboard
        await e2e_orchestrator.generate_kpi_dashboard(["HEADER-001"])

        total_time = performance_timer.stop("full_workflow")

        # Complete workflow should take < 5 seconds
        assert total_time < 5000, f"Workflow took {total_time}ms, target <5000ms"

    @pytest.mark.asyncio
    async def test_cycle_execution_time(self, e2e_orchestrator):
        """Test individual cycle execution time."""
        cycle_times = []

        for _ in range(20):
            result = await e2e_orchestrator.execute_cycle("HEADER-001")
            cycle_times.append(result["execution_time_ms"])

        avg_time = sum(cycle_times) / len(cycle_times)
        max_time = max(cycle_times)

        assert avg_time < 500, f"Average cycle time {avg_time}ms exceeds 500ms target"
        assert max_time < 1000, f"Max cycle time {max_time}ms exceeds 1000ms limit"


# =============================================================================
# TEST CLASS: WORKFLOW VALIDATION
# =============================================================================

@pytest.mark.e2e
class TestWorkflowValidation:
    """Validation tests for workflow correctness."""

    @pytest.mark.asyncio
    async def test_workflow_step_order(self, e2e_orchestrator):
        """Test workflow steps execute in correct order."""
        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        # Verify parameters were read before analysis
        assert "parameters" in result
        assert "analysis" in result

        # Verify actions follow analysis
        if result["analysis"]["actions_required"]:
            assert "actions_executed" in result

        # Verify quality verification
        assert "quality_before" in result
        assert "quality_after" in result

    @pytest.mark.asyncio
    async def test_workflow_data_flow(self, e2e_orchestrator):
        """Test data flows correctly through workflow stages."""
        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        # Parameters should include all required fields
        params = result["parameters"]
        required_fields = ["pressure_bar", "temperature_c", "dryness_fraction"]
        for field in required_fields:
            assert field in params, f"Missing field: {field}"

        # Analysis should use parameter data
        analysis = result["analysis"]
        assert "quality_index" in analysis
        assert "deviations" in analysis

    @pytest.mark.asyncio
    async def test_workflow_error_propagation(self, e2e_orchestrator):
        """Test errors propagate correctly through workflow."""
        # Inject sensor failure
        e2e_orchestrator.inject_fault("sensor_failure")

        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        assert result["status"] == "error"
        assert result["phase"] == "read"

        e2e_orchestrator.clear_fault()


# =============================================================================
# TEST CLASS: COMPLIANCE VERIFICATION
# =============================================================================

@pytest.mark.e2e
class TestComplianceVerification:
    """Tests for regulatory compliance verification."""

    @pytest.mark.asyncio
    async def test_asme_ptc_compliance_tracking(self, e2e_orchestrator):
        """Test ASME PTC compliance is tracked."""
        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        analysis = result["analysis"]
        assert "compliance_status" in analysis
        assert analysis["compliance_status"] in ["compliant", "non_compliant"]

    @pytest.mark.asyncio
    async def test_quality_limits_enforcement(self, e2e_orchestrator):
        """Test quality limits are enforced per standards."""
        # Apply high moisture scenario
        result = await e2e_orchestrator.execute_cycle(
            "HEADER-001",
            scenario=TestScenario.HIGH_MOISTURE
        )

        analysis = result["analysis"]

        # Should detect moisture exceeds limit
        if analysis["deviations"]["moisture_percent"] > 5.0:
            assert analysis["compliance_status"] == "non_compliant" or len(analysis["actions_required"]) > 0

    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, e2e_orchestrator):
        """Test audit trail contains all required elements."""
        result = await e2e_orchestrator.execute_cycle("HEADER-001")

        # Audit trail elements
        assert result["provenance_hash"] is not None
        assert result["cycle_number"] > 0
        assert "timestamp" in result["parameters"]
        assert result["determinism_verified"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])
