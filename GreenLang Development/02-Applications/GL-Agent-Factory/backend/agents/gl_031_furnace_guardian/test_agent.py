"""
GL-031 Furnace Guardian Agent - Golden Tests

This module provides comprehensive tests for the FurnaceGuardianAgent
including unit tests for formulas and integration tests for the agent.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from .agent import (
    FurnaceGuardianAgent,
    FurnaceGuardianInput,
    FurnaceGuardianOutput,
    TemperatureReading,
    PressureReading,
    FlameStatus,
    InterlockStatus,
    PurgeData,
)
from .models import (
    InterlockType,
    FlameDetectorType,
    FurnaceType,
    ComplianceStandard,
    PurgeStatus,
    RiskLevel,
    ViolationSeverity,
)
from .formulas import (
    calculate_purge_volume_changes,
    verify_purge_complete,
    calculate_flame_signal_quality,
    calculate_safety_score,
    check_temperature_limits,
    check_pressure_limits,
)


# =============================================================================
# FORMULA UNIT TESTS
# =============================================================================

class TestPurgeCalculations:
    """Tests for purge cycle calculations."""

    def test_purge_volume_changes_typical(self):
        """Test typical purge volume change calculation."""
        # 5000 CFM airflow, 2000 ft3 furnace, 2 minute purge
        changes = calculate_purge_volume_changes(
            airflow_cfm=5000,
            furnace_volume_cubic_feet=2000,
            purge_time_seconds=120
        )
        # 5000 * 2 / 2000 = 5 volume changes
        assert changes == pytest.approx(5.0, rel=0.01)

    def test_purge_volume_changes_minimum(self):
        """Test minimum purge for Class A furnace."""
        # Calculate time needed for exactly 4 volume changes
        changes = calculate_purge_volume_changes(
            airflow_cfm=1000,
            furnace_volume_cubic_feet=1000,
            purge_time_seconds=240  # 4 minutes
        )
        assert changes == pytest.approx(4.0, rel=0.01)

    def test_verify_purge_complete_class_a_pass(self):
        """Test successful Class A purge verification."""
        result = verify_purge_complete(
            airflow_cfm=5000,
            furnace_volume_cubic_feet=1000,
            purge_time_seconds=60,
            furnace_class="A",
            minimum_purge_time_seconds=30
        )
        assert result.is_valid is True
        assert result.actual_volume_changes >= 4.0

    def test_verify_purge_complete_class_b_fail(self):
        """Test failed Class B purge (needs 8 volume changes)."""
        result = verify_purge_complete(
            airflow_cfm=1000,
            furnace_volume_cubic_feet=1000,
            purge_time_seconds=60,  # Only 1 volume change
            furnace_class="B",
            minimum_purge_time_seconds=30
        )
        assert result.is_valid is False
        assert result.required_volume_changes == 8.0

    def test_verify_purge_minimum_time_fail(self):
        """Test purge fails due to insufficient time even with enough volume changes."""
        result = verify_purge_complete(
            airflow_cfm=10000,
            furnace_volume_cubic_feet=1000,
            purge_time_seconds=25,  # Less than 30 second minimum
            furnace_class="A",
            minimum_purge_time_seconds=30
        )
        assert result.is_valid is False
        assert "time" in result.message.lower()


class TestFlameSignalQuality:
    """Tests for flame signal quality calculations."""

    def test_flame_quality_excellent(self):
        """Test excellent flame signal quality."""
        quality, is_acceptable = calculate_flame_signal_quality(
            signal_strength=90,
            noise_level=5
        )
        assert quality == "EXCELLENT"
        assert is_acceptable is True

    def test_flame_quality_good(self):
        """Test good flame signal quality."""
        quality, is_acceptable = calculate_flame_signal_quality(
            signal_strength=60,
            noise_level=10
        )
        assert quality == "GOOD"
        assert is_acceptable is True

    def test_flame_quality_marginal(self):
        """Test marginal flame signal quality."""
        quality, is_acceptable = calculate_flame_signal_quality(
            signal_strength=15,
            noise_level=10
        )
        assert quality == "MARGINAL"
        assert is_acceptable is False

    def test_flame_quality_poor(self):
        """Test poor flame signal quality."""
        quality, is_acceptable = calculate_flame_signal_quality(
            signal_strength=5,
            noise_level=5
        )
        assert quality == "POOR"
        assert is_acceptable is False


class TestTemperatureLimits:
    """Tests for temperature limit checking."""

    def test_temperature_normal(self):
        """Test temperature within normal range."""
        status, is_safe = check_temperature_limits(
            temperature_celsius=500,
            low_limit=200,
            high_limit=800
        )
        assert status == "NORMAL"
        assert is_safe is True

    def test_temperature_high(self):
        """Test high temperature alarm."""
        status, is_safe = check_temperature_limits(
            temperature_celsius=850,
            low_limit=200,
            high_limit=800
        )
        assert status == "HIGH"
        assert is_safe is False

    def test_temperature_low(self):
        """Test low temperature alarm."""
        status, is_safe = check_temperature_limits(
            temperature_celsius=150,
            low_limit=200,
            high_limit=800
        )
        assert status == "LOW"
        assert is_safe is False

    def test_temperature_trip(self):
        """Test high-high temperature trip."""
        status, is_safe = check_temperature_limits(
            temperature_celsius=950,
            low_limit=200,
            high_limit=800,
            high_high_limit=900
        )
        assert status == "TRIP"
        assert is_safe is False


class TestPressureLimits:
    """Tests for pressure limit checking."""

    def test_pressure_normal(self):
        """Test pressure within normal range."""
        status, is_safe = check_pressure_limits(
            pressure_kpa=101.3,
            low_limit=90,
            high_limit=120
        )
        assert status == "NORMAL"
        assert is_safe is True

    def test_pressure_high_high_trip(self):
        """Test high-high pressure trip."""
        status, is_safe = check_pressure_limits(
            pressure_kpa=150,
            low_limit=90,
            high_limit=120,
            high_high_limit=140
        )
        assert status == "HIGH_HIGH_TRIP"
        assert is_safe is False


class TestSafetyScore:
    """Tests for overall safety score calculation."""

    def test_safety_score_perfect(self):
        """Test perfect safety score."""
        result = calculate_safety_score(
            interlocks_ok=10,
            interlocks_total=10,
            purge_valid=True,
            flame_detected=True,
            flame_signal_quality="EXCELLENT",
            temps_in_range=5,
            temps_total=5,
            pressures_in_range=3,
            pressures_total=3
        )
        assert result.score >= 95
        assert result.risk_level == "NONE"

    def test_safety_score_critical(self):
        """Test critical safety score."""
        result = calculate_safety_score(
            interlocks_ok=2,
            interlocks_total=10,
            purge_valid=False,
            flame_detected=False,
            flame_signal_quality="POOR",
            temps_in_range=1,
            temps_total=5,
            pressures_in_range=1,
            pressures_total=3
        )
        assert result.score < 50
        assert result.risk_level == "CRITICAL"


# =============================================================================
# AGENT INTEGRATION TESTS
# =============================================================================

class TestFurnaceGuardianAgent:
    """Integration tests for FurnaceGuardianAgent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for tests."""
        return FurnaceGuardianAgent()

    @pytest.fixture
    def valid_input(self):
        """Create valid input data for tests."""
        return FurnaceGuardianInput(
            furnace_id="FRN-TEST-001",
            furnace_type=FurnaceType.PROCESS_FURNACE,
            temps=[
                TemperatureReading(
                    sensor_id="T1",
                    value_celsius=500,
                    low_limit=200,
                    high_limit=800
                ),
                TemperatureReading(
                    sensor_id="T2",
                    value_celsius=520,
                    low_limit=200,
                    high_limit=800
                )
            ],
            pressures=[
                PressureReading(
                    sensor_id="P1",
                    value_kpa=101.3,
                    low_limit=90,
                    high_limit=120
                )
            ],
            flame_status=FlameStatus(
                is_detected=True,
                signal_strength=85,
                noise_level=5,
                detector_type=FlameDetectorType.UV_IR_COMBINED
            ),
            interlocks=[
                InterlockStatus(
                    interlock_type=InterlockType.FLAME_FAILURE,
                    is_ok=True
                ),
                InterlockStatus(
                    interlock_type=InterlockType.HIGH_TEMPERATURE,
                    is_ok=True
                ),
                InterlockStatus(
                    interlock_type=InterlockType.LOW_COMBUSTION_AIR,
                    is_ok=True
                )
            ],
            purge_data=PurgeData(
                status=PurgeStatus.COMPLETE,
                airflow_cfm=5000,
                furnace_volume_cubic_feet=1000,
                purge_time_seconds=120,
                furnace_class="A"
            ),
            compliance_standards=[ComplianceStandard.NFPA_86]
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-031"
        assert agent.AGENT_NAME == "FURNACE-GUARDIAN"
        assert agent.VERSION == "1.0.0"

    def test_agent_run_valid_input(self, agent, valid_input):
        """Test agent runs successfully with valid input."""
        result = agent.run(valid_input)

        assert isinstance(result, FurnaceGuardianOutput)
        assert result.furnace_id == "FRN-TEST-001"
        assert result.safety_score >= 0
        assert result.safety_score <= 100
        assert result.validation_status == "PASS"
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_agent_high_safety_score(self, agent, valid_input):
        """Test agent produces high safety score for good conditions."""
        result = agent.run(valid_input)

        assert result.safety_score >= 80
        assert result.risk_level in [RiskLevel.NONE, RiskLevel.LOW]

    def test_agent_detects_interlock_bypass(self, agent, valid_input):
        """Test agent detects bypassed interlocks."""
        valid_input.interlocks[0].is_bypassed = True
        valid_input.interlocks[0].bypass_reason = "Maintenance"

        result = agent.run(valid_input)

        # Should have a bypass violation
        bypass_violations = [
            v for v in result.violations
            if v.category == "INTERLOCK_BYPASS"
        ]
        assert len(bypass_violations) > 0

    def test_agent_detects_temperature_violation(self, agent, valid_input):
        """Test agent detects temperature limit violation."""
        valid_input.temps[0].value_celsius = 850  # Above high limit of 800

        result = agent.run(valid_input)

        temp_violations = [
            v for v in result.violations
            if v.category == "TEMPERATURE_LIMIT"
        ]
        assert len(temp_violations) > 0

    def test_agent_detects_purge_bypass(self, agent, valid_input):
        """Test agent detects purge bypass as critical violation."""
        valid_input.purge_data.status = PurgeStatus.BYPASSED

        result = agent.run(valid_input)

        purge_violations = [
            v for v in result.violations
            if v.category == "PURGE_BYPASS"
        ]
        assert len(purge_violations) > 0
        assert purge_violations[0].severity == ViolationSeverity.EMERGENCY

    def test_agent_generates_corrective_actions(self, agent, valid_input):
        """Test agent generates corrective actions for violations."""
        valid_input.temps[0].value_celsius = 850  # Violation

        result = agent.run(valid_input)

        assert len(result.corrective_actions) > 0
        assert all(a.action_id for a in result.corrective_actions)

    def test_agent_provenance_chain(self, agent, valid_input):
        """Test agent creates complete provenance chain."""
        result = agent.run(valid_input)

        assert len(result.provenance_chain) >= 5  # At least 5 calculation steps
        assert all(p.operation for p in result.provenance_chain)
        assert all(p.input_hash for p in result.provenance_chain)
        assert all(p.output_hash for p in result.provenance_chain)

    def test_agent_compliance_status(self, agent, valid_input):
        """Test agent reports compliance status."""
        result = agent.run(valid_input)

        assert len(result.compliance_status) > 0
        nfpa_status = next(
            (c for c in result.compliance_status if c.standard == ComplianceStandard.NFPA_86),
            None
        )
        assert nfpa_status is not None


# =============================================================================
# GOLDEN TEST CASES
# =============================================================================

class TestGoldenCases:
    """Golden test cases with known expected outputs."""

    @pytest.fixture
    def agent(self):
        return FurnaceGuardianAgent()

    def test_golden_case_1_normal_operation(self, agent):
        """Golden case: Normal furnace operation."""
        input_data = FurnaceGuardianInput(
            furnace_id="GOLDEN-001",
            temps=[
                TemperatureReading(sensor_id="T1", value_celsius=650, low_limit=300, high_limit=900)
            ],
            pressures=[
                PressureReading(sensor_id="P1", value_kpa=105, low_limit=95, high_limit=115)
            ],
            flame_status=FlameStatus(is_detected=True, signal_strength=90, noise_level=3),
            interlocks=[
                InterlockStatus(interlock_type=InterlockType.FLAME_FAILURE, is_ok=True),
                InterlockStatus(interlock_type=InterlockType.HIGH_TEMPERATURE, is_ok=True)
            ],
            purge_data=PurgeData(
                status=PurgeStatus.COMPLETE,
                airflow_cfm=6000,
                furnace_volume_cubic_feet=1000,
                purge_time_seconds=60
            )
        )

        result = agent.run(input_data)

        # Expected: High safety score, no violations
        assert result.safety_score >= 90
        assert result.risk_level in [RiskLevel.NONE, RiskLevel.LOW]
        assert len(result.violations) == 0

    def test_golden_case_2_critical_failure(self, agent):
        """Golden case: Critical safety failures."""
        input_data = FurnaceGuardianInput(
            furnace_id="GOLDEN-002",
            temps=[
                TemperatureReading(
                    sensor_id="T1",
                    value_celsius=950,
                    low_limit=300,
                    high_limit=800,
                    high_high_limit=900
                )
            ],
            pressures=[
                PressureReading(sensor_id="P1", value_kpa=150, low_limit=95, high_limit=115)
            ],
            flame_status=FlameStatus(is_detected=False, signal_strength=0, noise_level=10),
            interlocks=[
                InterlockStatus(interlock_type=InterlockType.FLAME_FAILURE, is_ok=False),
                InterlockStatus(interlock_type=InterlockType.HIGH_TEMPERATURE, is_ok=False)
            ],
            purge_data=PurgeData(status=PurgeStatus.FAILED)
        )

        result = agent.run(input_data)

        # Expected: Critical risk level, multiple violations
        assert result.risk_level == RiskLevel.CRITICAL
        assert len(result.violations) >= 3
        assert len(result.corrective_actions) >= 3

    def test_golden_case_3_marginal_operation(self, agent):
        """Golden case: Marginal operation requiring attention."""
        input_data = FurnaceGuardianInput(
            furnace_id="GOLDEN-003",
            temps=[
                TemperatureReading(sensor_id="T1", value_celsius=780, low_limit=300, high_limit=800)
            ],
            pressures=[
                PressureReading(sensor_id="P1", value_kpa=100, low_limit=95, high_limit=115)
            ],
            flame_status=FlameStatus(is_detected=True, signal_strength=25, noise_level=8),
            interlocks=[
                InterlockStatus(interlock_type=InterlockType.FLAME_FAILURE, is_ok=True),
                InterlockStatus(
                    interlock_type=InterlockType.HIGH_TEMPERATURE,
                    is_ok=True,
                    is_bypassed=True,
                    bypass_reason="Calibration"
                )
            ]
        )

        result = agent.run(input_data)

        # Expected: Moderate risk, some warnings
        assert result.risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH]
        assert len(result.violations) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
