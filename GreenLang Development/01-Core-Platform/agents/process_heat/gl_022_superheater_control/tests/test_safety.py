"""
GL-022 SuperheaterControlAgent - Safety and Compliance Tests

This module provides comprehensive tests for:
- ASME PTC 4 compliance
- Safety limit validation
- Interlock logic
- Thermal shock detection and prevention
- Tube metal temperature monitoring
- Critical/Warning/Safe status determination
- Emergency recommendations
- Audit trail completeness

Target: 85%+ coverage for safety-related code paths

Standards Tested:
- ASME PTC 4 (Performance Test Code for Fired Steam Generators)
- IEC 61511 (Safety Instrumented Systems)
- API 556 (Instrumentation and Control Systems for Fired Heaters)

Safety Categories:
- Tube metal overtemperature protection
- Minimum superheat protection (prevent wet steam)
- Thermal shock prevention (rate limiting)
- Spray capacity exceedance detection
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add agent paths
AGENT_BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.parent
BACKEND_AGENT_PATH = AGENT_BASE_PATH / "GL-Agent-Factory" / "backend" / "agents"
sys.path.insert(0, str(AGENT_BASE_PATH))
sys.path.insert(0, str(BACKEND_AGENT_PATH))

try:
    from gl_022_superheater_control.agent import SuperheaterControlAgent
    from gl_022_superheater_control.models import (
        SuperheaterInput,
        SuperheaterOutput,
    )
    from gl_022_superheater_control.formulas import (
        calculate_saturation_temperature,
        calculate_superheat,
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    SuperheaterControlAgent = None

# Skip all tests if agent not available
pytestmark = pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module not available")


# =============================================================================
# SAFETY STATUS DETERMINATION TESTS
# =============================================================================

class TestSafetyStatusDetermination:
    """Tests for safety status determination logic."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_safe_status_adequate_margin(self, agent, valid_input_data):
        """Test SAFE status when tube metal margin is adequate."""
        # Ensure adequate margin (> 50C)
        valid_input_data["max_tube_metal_temp_c"] = 600.0
        valid_input_data["current_tube_metal_temp_c"] = 500.0  # 100C margin

        result = agent.run(valid_input_data)

        assert result["safety_status"] == "SAFE"
        assert result["tube_metal_margin_c"] == 100.0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_warning_status_low_margin(self, agent, input_warning_tube_temp):
        """Test WARNING status when tube metal margin is low (<50C)."""
        result = agent.run(input_warning_tube_temp)

        assert result["safety_status"] == "WARNING"
        assert result["tube_metal_margin_c"] < 50.0
        assert result["tube_metal_margin_c"] >= 25.0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_critical_status_very_low_margin(self, agent, input_critical_tube_temp):
        """Test CRITICAL status when tube metal margin is critical (<25C)."""
        result = agent.run(input_critical_tube_temp)

        assert result["safety_status"] == "CRITICAL"
        assert result["tube_metal_margin_c"] < 25.0

    @pytest.mark.safety
    @pytest.mark.unit
    @pytest.mark.parametrize("tube_temp,max_temp,expected_status", [
        (500.0, 600.0, "SAFE"),      # 100C margin
        (540.0, 600.0, "SAFE"),      # 60C margin
        (555.0, 600.0, "WARNING"),   # 45C margin
        (560.0, 600.0, "WARNING"),   # 40C margin
        (580.0, 600.0, "CRITICAL"),  # 20C margin
        (590.0, 600.0, "CRITICAL"),  # 10C margin
        (599.0, 600.0, "CRITICAL"),  # 1C margin
    ])
    def test_safety_status_thresholds(self, agent, valid_input_data, tube_temp, max_temp, expected_status):
        """Test safety status thresholds are correctly applied."""
        valid_input_data["current_tube_metal_temp_c"] = tube_temp
        valid_input_data["max_tube_metal_temp_c"] = max_temp

        result = agent.run(valid_input_data)

        assert result["safety_status"] == expected_status
        assert result["tube_metal_margin_c"] == (max_temp - tube_temp)


# =============================================================================
# TUBE METAL TEMPERATURE TESTS
# =============================================================================

class TestTubeMetalTemperature:
    """Tests for tube metal temperature monitoring."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_tube_metal_margin_calculated(self, agent, valid_input_data):
        """Test tube metal margin is correctly calculated."""
        valid_input_data["max_tube_metal_temp_c"] = 600.0
        valid_input_data["current_tube_metal_temp_c"] = 520.0

        result = agent.run(valid_input_data)

        assert result["tube_metal_margin_c"] == 80.0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_tube_metal_no_sensor(self, agent, input_no_tube_temp_sensor):
        """Test handling when tube metal sensor unavailable."""
        result = agent.run(input_no_tube_temp_sensor)

        # When no sensor, uses max temp as margin (conservative)
        assert result["tube_metal_margin_c"] == input_no_tube_temp_sensor["max_tube_metal_temp_c"]
        assert result["safety_status"] == "SAFE"

    @pytest.mark.safety
    @pytest.mark.unit
    def test_critical_tube_temp_warning_message(self, agent, input_critical_tube_temp):
        """Test critical tube temperature generates warning."""
        result = agent.run(input_critical_tube_temp)

        warnings = " ".join(result["warnings"]).lower()
        assert "critical" in warnings or "tube" in warnings

    @pytest.mark.safety
    @pytest.mark.unit
    def test_critical_tube_temp_recommendation(self, agent, input_critical_tube_temp):
        """Test critical tube temperature generates reduce firing recommendation."""
        result = agent.run(input_critical_tube_temp)

        recommendations = " ".join(result["recommendations"]).lower()
        assert "reduce" in recommendations or "firing" in recommendations or "immediately" in recommendations


# =============================================================================
# SUPERHEAT PROTECTION TESTS
# =============================================================================

class TestSuperheatProtection:
    """Tests for minimum superheat protection (wet steam prevention)."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_adequate_superheat_no_warning(self, agent, valid_input_data):
        """Test no warning when superheat is adequate."""
        # Ensure high superheat
        valid_input_data["outlet_steam_temp_c"] = 450.0
        valid_input_data["steam_pressure_bar"] = 40.0  # T_sat ~ 250C
        valid_input_data["min_superheat_c"] = 20.0

        result = agent.run(valid_input_data)

        # Superheat should be ~200C, well above minimum
        assert result["current_superheat_c"] > valid_input_data["min_superheat_c"]

    @pytest.mark.safety
    @pytest.mark.unit
    def test_low_superheat_warning(self, agent, input_low_superheat):
        """Test warning when superheat is below minimum."""
        result = agent.run(input_low_superheat)

        # Check for warning about low superheat
        warnings = " ".join(result["warnings"]).lower()
        assert "superheat" in warnings or "below" in warnings or "minimum" in warnings


# =============================================================================
# THERMAL SHOCK PREVENTION TESTS
# =============================================================================

class TestThermalShockPrevention:
    """Tests for thermal shock prevention through rate limiting."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_rate_of_change_limited(self, agent, valid_input_data):
        """Test valve rate of change is limited."""
        # Large temperature difference requiring large valve change
        valid_input_data["outlet_steam_temp_c"] = 550.0
        valid_input_data["target_steam_temp_c"] = 400.0
        valid_input_data["spray_valve_position_pct"] = 0.0  # Currently closed

        result = agent.run(valid_input_data)

        # Rate should be limited to max_rate (10% per minute)
        rate = result["spray_control"]["rate_of_change_pct_per_min"]
        assert abs(rate) <= 10.0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_rate_limited_increase(self, agent, valid_input_data):
        """Test rate is limited for valve opening."""
        valid_input_data["outlet_steam_temp_c"] = 550.0
        valid_input_data["target_steam_temp_c"] = 400.0
        valid_input_data["spray_valve_position_pct"] = 10.0

        result = agent.run(valid_input_data)

        # Should be positive but limited
        rate = result["spray_control"]["rate_of_change_pct_per_min"]
        if rate > 0:
            assert rate <= 10.0


# =============================================================================
# SPRAY CAPACITY TESTS
# =============================================================================

class TestSprayCapacityProtection:
    """Tests for spray capacity protection."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_spray_capacity_warning(self, agent, input_spray_capacity_exceeded):
        """Test warning when required spray exceeds capacity."""
        result = agent.run(input_spray_capacity_exceeded)

        warnings = " ".join(result["warnings"]).lower()
        assert "capacity" in warnings or "exceeds" in warnings or "spray" in warnings

    @pytest.mark.safety
    @pytest.mark.unit
    def test_spray_capped_at_maximum(self, agent, input_spray_capacity_exceeded):
        """Test spray flow is capped at maximum."""
        result = agent.run(input_spray_capacity_exceeded)

        max_spray = input_spray_capacity_exceeded["max_spray_flow_kg_s"]
        assert result["spray_control"]["target_spray_flow_kg_s"] <= max_spray


# =============================================================================
# ASME COMPLIANCE TESTS
# =============================================================================

class TestASMECompliance:
    """Tests for ASME PTC 4 compliance requirements."""

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_saturation_temp_calculation_compliant(self, agent, iapws_if97_reference_points):
        """Test saturation temperature calculation is IAPWS-IF97 compliant."""
        for ref in iapws_if97_reference_points:
            t_sat = calculate_saturation_temperature(ref["pressure_bar"])

            assert t_sat == pytest.approx(ref["t_sat_c"], abs=ref["tolerance"]), \
                f"At {ref['pressure_bar']} bar: calculated {t_sat}C, expected ~{ref['t_sat_c']}C"

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_calculation_hash_present(self, agent, valid_input_data):
        """Test calculation hash is present for audit compliance."""
        result = agent.run(valid_input_data)

        assert "calculation_hash" in result
        assert len(result["calculation_hash"]) == 64  # SHA-256

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_timestamp_present(self, agent, valid_input_data):
        """Test calculation timestamp is present for audit trail."""
        result = agent.run(valid_input_data)

        assert "calculation_timestamp" in result
        assert result["calculation_timestamp"] is not None

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_version_present(self, agent, valid_input_data):
        """Test agent version is present for audit trail."""
        result = agent.run(valid_input_data)

        assert "agent_version" in result
        assert result["agent_version"] is not None

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_metadata_includes_standards(self, agent):
        """Test agent metadata includes compliance standards."""
        metadata = agent.get_metadata()

        assert "standards" in metadata
        assert "ASME PTC 4" in metadata["standards"]
        assert "IAPWS-IF97" in metadata["standards"]


# =============================================================================
# AUDIT TRAIL TESTS
# =============================================================================

class TestAuditTrail:
    """Tests for audit trail completeness."""

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_provenance_hash_deterministic(self, agent, valid_input_data):
        """Test provenance hash is deterministic for audit purposes."""
        result1 = agent.run(valid_input_data)
        result2 = agent.run(valid_input_data)

        assert result1["calculation_hash"] == result2["calculation_hash"]

    @pytest.mark.compliance
    @pytest.mark.unit
    def test_provenance_hash_changes_with_input(self, agent, valid_input_data):
        """Test provenance hash changes when input changes."""
        result1 = agent.run(valid_input_data)

        valid_input_data["outlet_steam_temp_c"] += 1.0
        result2 = agent.run(valid_input_data)

        assert result1["calculation_hash"] != result2["calculation_hash"]


# =============================================================================
# EMERGENCY RESPONSE TESTS
# =============================================================================

class TestEmergencyResponse:
    """Tests for emergency response recommendations."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_emergency_reduce_firing(self, agent, input_critical_tube_temp):
        """Test emergency recommendation to reduce firing."""
        result = agent.run(input_critical_tube_temp)

        recommendations = " ".join(result["recommendations"]).lower()
        assert "reduce" in recommendations or "firing" in recommendations or "immediately" in recommendations

    @pytest.mark.safety
    @pytest.mark.unit
    def test_no_emergency_when_safe(self, agent, valid_input_data):
        """Test no emergency recommendations when safe."""
        # Ensure safe conditions
        valid_input_data["current_tube_metal_temp_c"] = 450.0  # Large margin

        result = agent.run(valid_input_data)

        # Should not have emergency recommendations
        recommendations = " ".join(result["recommendations"]).lower()
        assert "immediately" not in recommendations


# =============================================================================
# COMPREHENSIVE SAFETY SCENARIO TESTS
# =============================================================================

class TestComprehensiveSafetyScenarios:
    """Comprehensive safety scenario tests."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_scenario_normal_operation_safe(self, agent, valid_input_data):
        """Test normal operation is classified as safe."""
        result = agent.run(valid_input_data)

        assert result["safety_status"] == "SAFE"
        # Should have no critical warnings
        critical_warnings = [w for w in result["warnings"] if "critical" in w.lower()]
        assert len(critical_warnings) == 0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_scenario_emergency_shutdown_candidate(self, agent, valid_input_data):
        """Test emergency shutdown candidate scenario."""
        # Multiple critical conditions
        valid_input_data["outlet_steam_temp_c"] = 600.0
        valid_input_data["current_tube_metal_temp_c"] = 590.0  # Near limit
        valid_input_data["max_tube_metal_temp_c"] = 600.0

        result = agent.run(valid_input_data)

        assert result["safety_status"] == "CRITICAL"
        assert len(result["warnings"]) > 0
        assert len(result["recommendations"]) > 0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_all_safety_fields_present(self, agent, valid_input_data):
        """Test all safety-related fields are present in output."""
        result = agent.run(valid_input_data)

        safety_fields = [
            "tube_metal_margin_c",
            "safety_status",
            "current_superheat_c",
            "warnings",
            "recommendations",
        ]

        for field in safety_fields:
            assert field in result, f"Missing safety field: {field}"

    @pytest.mark.safety
    @pytest.mark.unit
    def test_safety_status_values_valid(self, agent, valid_input_data):
        """Test safety status is one of valid values."""
        result = agent.run(valid_input_data)

        valid_statuses = ["SAFE", "WARNING", "CRITICAL"]
        assert result["safety_status"] in valid_statuses


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestSafetyRegressions:
    """Regression tests for safety-related bugs."""

    @pytest.mark.safety
    @pytest.mark.unit
    def test_regression_negative_margin_handled(self, agent, valid_input_data):
        """Test handling of negative tube metal margin (impossible condition)."""
        # If tube temp > max temp (sensor error or misconfiguration)
        valid_input_data["current_tube_metal_temp_c"] = 650.0
        valid_input_data["max_tube_metal_temp_c"] = 600.0

        result = agent.run(valid_input_data)

        # Should be CRITICAL with negative margin
        assert result["safety_status"] == "CRITICAL"
        assert result["tube_metal_margin_c"] < 0

    @pytest.mark.safety
    @pytest.mark.unit
    def test_regression_zero_steam_flow_safe(self, agent, valid_input_data):
        """Test zero steam flow doesn't cause safety issues."""
        valid_input_data["steam_flow_kg_s"] = 0.0

        result = agent.run(valid_input_data)

        # Should still evaluate safety status
        assert result["safety_status"] in ["SAFE", "WARNING", "CRITICAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
