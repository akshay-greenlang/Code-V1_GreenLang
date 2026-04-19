"""
Unit tests for NFPA 86 Furnace Compliance Checker (TASK-195)

Tests all compliance checking functionality including:
- Class A/B/C/D furnace validation
- Purge cycle calculation and validation
- Safety interlock configuration
- Flame failure response timing
- LEL monitoring
- Trial for ignition timing
"""

import pytest
import logging
from datetime import datetime
from greenlang.safety.nfpa_86_furnace import (
    NFPA86ComplianceChecker,
    FurnaceClass,
    AtmosphereType,
    ComplianceLevel,
    PurgeConfiguration,
    SafetyInterlockConfig,
    FurnaceConfiguration,
    NFPA86TimingRequirements,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def checker():
    """Create NFPA86ComplianceChecker instance."""
    return NFPA86ComplianceChecker()


@pytest.fixture
def base_furnace_config():
    """Create base furnace configuration for testing."""
    return FurnaceConfiguration(
        equipment_id="FURN-TEST-001",
        classification=FurnaceClass.CLASS_A,
        atmosphere_type=AtmosphereType.AIR,
        maximum_temperature_deg_f=1000.0,
        furnace_volume_cubic_feet=500.0,
        burner_input_btuh=500000.0,
        has_flame_supervision=True,
        has_combustion_safeguards=True,
        has_lel_monitoring=True,
        has_emergency_shutdown=True,
        has_temperature_monitoring=True,
        interlocks=[
            SafetyInterlockConfig(
                name="Low Fuel Pressure",
                setpoint=5.0,
                unit="psig",
                is_operational=True,
            )
        ],
    )


@pytest.fixture
def purge_config():
    """Create purge configuration for testing."""
    return PurgeConfiguration(
        furnace_volume_cubic_feet=500.0,
        airflow_cfm=1000.0,
        purge_air_quality="clean_dry_air",
        minimum_airflow_percent=25.0,
    )


@pytest.fixture
def safety_interlock_config():
    """Create safety interlock configuration."""
    return SafetyInterlockConfig(
        name="Low Fuel Pressure",
        setpoint=5.0,
        unit="psig",
        is_high_trip=False,
        response_time_limit_seconds=3.0,
        is_operational=True,
    )


# =============================================================================
# Test NFPA 86 Timing Requirements
# =============================================================================

class TestNFPA86TimingRequirements:
    """Test NFPA 86 timing constant values."""

    def test_prepurge_min_volume_changes(self):
        """Test prepurge minimum volume changes requirement."""
        assert NFPA86TimingRequirements.PREPURGE_MIN_VOLUME_CHANGES == 4

    def test_prepurge_min_airflow_percent(self):
        """Test prepurge minimum airflow percentage."""
        assert NFPA86TimingRequirements.PREPURGE_MIN_AIRFLOW_PERCENT == 25.0

    def test_flame_failure_response_max(self):
        """Test flame failure response time maximum."""
        assert NFPA86TimingRequirements.FLAME_FAILURE_RESPONSE_MAX_SECONDS == 4.0

    def test_pilot_trial_max(self):
        """Test pilot trial maximum time."""
        assert NFPA86TimingRequirements.PILOT_TRIAL_MAX_SECONDS == 10.0

    def test_main_trial_max(self):
        """Test main flame trial maximum time."""
        assert NFPA86TimingRequirements.MAIN_TRIAL_MAX_SECONDS == 10.0

    def test_lel_alarm_threshold(self):
        """Test LEL alarm threshold."""
        assert NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT == 25.0

    def test_lel_shutdown_threshold(self):
        """Test LEL shutdown threshold."""
        assert NFPA86TimingRequirements.LEL_SHUTDOWN_THRESHOLD_PERCENT == 50.0


# =============================================================================
# Test Purge Cycle Validation
# =============================================================================

class TestPurgeCycleValidation:
    """Test purge cycle calculation and validation."""

    def test_purge_time_calculation(self, checker, purge_config):
        """Test purge time calculation formula."""
        # Time = (4 × 500) / 1000 × 60 = 120 seconds
        required_time = purge_config.calculate_required_prepurge_time()
        assert required_time == pytest.approx(120.0, rel=0.01)

    def test_purge_time_minimum(self, checker):
        """Test minimum prepurge time enforced."""
        purge = PurgeConfiguration(
            furnace_volume_cubic_feet=100.0,
            airflow_cfm=2000.0,  # High flow, short time
        )
        required_time = purge.calculate_required_prepurge_time()
        # Even with high flow, minimum time should be applied
        assert required_time >= NFPA86TimingRequirements.PREPURGE_MIN_TIME_SECONDS

    def test_volume_changes_calculation(self, purge_config):
        """Test volume changes calculation."""
        volume_changes = purge_config.calculate_volume_changes(120.0)
        # (1000 CFM × 120 sec / 60) / 500 cu ft = 4 volume changes
        assert volume_changes == pytest.approx(4.0, rel=0.01)

    def test_validate_purge_cycle_valid(self, checker):
        """Test valid purge cycle validation."""
        is_valid, message, time_sec = checker.validate_purge_cycle(
            AtmosphereType.ENDOTHERMIC,
            500.0,  # volume
            1000.0  # flow rate
        )
        assert is_valid is True
        assert time_sec == pytest.approx(120.0, rel=0.01)
        assert "Purge time" in message

    def test_validate_purge_cycle_insufficient_flow(self, checker):
        """Test purge validation with insufficient flow rate."""
        is_valid, message, time_sec = checker.validate_purge_cycle(
            AtmosphereType.ENDOTHERMIC,
            500.0,
            100.0  # Very low flow
        )
        # Should still be valid but with longer time
        assert time_sec > 120.0

    def test_validate_purge_cycle_zero_volume(self, checker):
        """Test purge validation with zero volume."""
        is_valid, message, time_sec = checker.validate_purge_cycle(
            AtmosphereType.AIR,
            0.0,  # Zero volume
            1000.0
        )
        assert is_valid is False
        assert "positive" in message.lower()

    def test_validate_purge_cycle_zero_flow(self, checker):
        """Test purge validation with zero flow rate."""
        is_valid, message, time_sec = checker.validate_purge_cycle(
            AtmosphereType.AIR,
            500.0,
            0.0  # Zero flow
        )
        assert is_valid is False
        assert "positive" in message.lower()


# =============================================================================
# Test Class A Furnace Compliance
# =============================================================================

class TestClassAFurnaceCompliance:
    """Test Class A furnace compliance checking."""

    def test_class_a_compliant(self, checker, base_furnace_config):
        """Test fully compliant Class A furnace."""
        base_furnace_config.classification = FurnaceClass.CLASS_A
        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.equipment_id == "FURN-TEST-001"
        assert result.classification == FurnaceClass.CLASS_A
        assert result.compliance_level == ComplianceLevel.COMPLIANT
        assert result.requirements_failed == 0

    def test_class_a_missing_flame_supervision(self, checker, base_furnace_config):
        """Test Class A without flame supervision."""
        base_furnace_config.has_flame_supervision = False
        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT
        assert any("8.5" in str(f.get("section", "")) for f in result.findings)

    def test_class_a_missing_lel_monitoring(self, checker, base_furnace_config):
        """Test Class A without LEL monitoring."""
        base_furnace_config.has_lel_monitoring = False
        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT
        assert any("8.8" in str(f.get("section", "")) for f in result.findings)

    def test_class_a_missing_emergency_shutdown(self, checker, base_furnace_config):
        """Test Class A without emergency shutdown."""
        base_furnace_config.has_emergency_shutdown = False
        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT
        assert any("8.11" in str(f.get("section", "")) for f in result.findings)


# =============================================================================
# Test Class B Furnace Compliance
# =============================================================================

class TestClassBFurnaceCompliance:
    """Test Class B furnace compliance checking."""

    def test_class_b_compliant(self, checker, base_furnace_config):
        """Test fully compliant Class B furnace."""
        base_furnace_config.classification = FurnaceClass.CLASS_B
        result = checker.check_class_b_furnace(base_furnace_config)

        assert result.classification == FurnaceClass.CLASS_B
        assert result.compliance_level == ComplianceLevel.COMPLIANT

    def test_class_b_missing_combustion_safeguards(self, checker, base_furnace_config):
        """Test Class B without combustion safeguards."""
        base_furnace_config.has_combustion_safeguards = False
        result = checker.check_class_b_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT


# =============================================================================
# Test Class C Furnace Compliance
# =============================================================================

class TestClassCFurnaceCompliance:
    """Test Class C furnace (special atmosphere) compliance checking."""

    def test_class_c_compliant(self, checker, base_furnace_config):
        """Test fully compliant Class C furnace."""
        base_furnace_config.classification = FurnaceClass.CLASS_C
        base_furnace_config.atmosphere_type = AtmosphereType.ENDOTHERMIC
        base_furnace_config.has_purge_capability = True
        result = checker.check_class_c_furnace(base_furnace_config)

        assert result.classification == FurnaceClass.CLASS_C
        assert result.compliance_level == ComplianceLevel.COMPLIANT

    def test_class_c_missing_purge_capability(self, checker, base_furnace_config):
        """Test Class C without purge capability."""
        base_furnace_config.has_purge_capability = False
        result = checker.check_class_c_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT
        assert any("8.7" in str(f.get("section", "")) for f in result.findings)

    def test_class_c_endothermic_atmosphere(self, checker, base_furnace_config):
        """Test Class C with endothermic atmosphere."""
        base_furnace_config.atmosphere_type = AtmosphereType.ENDOTHERMIC
        base_furnace_config.has_purge_capability = True
        result = checker.check_class_c_furnace(base_furnace_config)

        assert result.purge_time_calculated_seconds > 0
        logger.info(f"Purge time for endothermic: {result.purge_time_calculated_seconds:.1f}s")

    def test_class_c_hydrogen_atmosphere(self, checker, base_furnace_config):
        """Test Class C with hydrogen atmosphere."""
        base_furnace_config.atmosphere_type = AtmosphereType.HYDROGEN
        base_furnace_config.has_purge_capability = True
        result = checker.check_class_c_furnace(base_furnace_config)

        assert result.purge_time_calculated_seconds > 0


# =============================================================================
# Test Class D Furnace Compliance
# =============================================================================

class TestClassDFurnaceCompliance:
    """Test Class D furnace (vacuum) compliance checking."""

    def test_class_d_compliant(self, checker, base_furnace_config):
        """Test fully compliant Class D furnace."""
        base_furnace_config.classification = FurnaceClass.CLASS_D
        base_furnace_config.atmosphere_type = AtmosphereType.VACUUM
        base_furnace_config.interlocks = [
            SafetyInterlockConfig(
                name="Vacuum Pressure",
                setpoint=100.0,
                unit="mtorr",
                is_operational=True,
            )
        ]
        result = checker.check_class_d_furnace(base_furnace_config)

        assert result.classification == FurnaceClass.CLASS_D
        assert result.compliance_level == ComplianceLevel.COMPLIANT

    def test_class_d_wrong_atmosphere(self, checker, base_furnace_config):
        """Test Class D with non-vacuum atmosphere."""
        base_furnace_config.atmosphere_type = AtmosphereType.AIR
        result = checker.check_class_d_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT
        assert any("8.12" in str(f.get("section", "")) for f in result.findings)

    def test_class_d_no_interlocks(self, checker, base_furnace_config):
        """Test Class D without safety interlocks."""
        base_furnace_config.atmosphere_type = AtmosphereType.VACUUM
        base_furnace_config.interlocks = []
        result = checker.check_class_d_furnace(base_furnace_config)

        assert result.compliance_level != ComplianceLevel.COMPLIANT


# =============================================================================
# Test Safety Interlock Validation
# =============================================================================

class TestSafetyInterlockValidation:
    """Test safety interlock configuration validation."""

    def test_validate_interlocks_operational(self, checker, base_furnace_config):
        """Test validation of operational interlocks."""
        base_furnace_config.interlocks = [
            SafetyInterlockConfig(
                name="Low Pressure",
                setpoint=5.0,
                unit="psig",
                is_operational=True,
            ),
            SafetyInterlockConfig(
                name="High Temperature",
                setpoint=1000.0,
                unit="°F",
                is_operational=True,
            )
        ]
        is_valid, message = checker.validate_safety_interlocks(base_furnace_config)

        assert is_valid is True
        assert "2" in message  # Should mention 2 operational interlocks

    def test_validate_interlocks_none_configured(self, checker, base_furnace_config):
        """Test validation with no interlocks."""
        base_furnace_config.interlocks = []
        is_valid, message = checker.validate_safety_interlocks(base_furnace_config)

        assert is_valid is False
        assert "No safety interlocks" in message

    def test_validate_interlocks_operational_with_failed(self, checker, base_furnace_config):
        """Test validation with mix of operational and failed interlocks."""
        base_furnace_config.interlocks = [
            SafetyInterlockConfig(
                name="Operational",
                is_operational=True,
            ),
            SafetyInterlockConfig(
                name="Failed",
                is_operational=False,
            ),
        ]
        is_valid, message = checker.validate_safety_interlocks(base_furnace_config)

        assert is_valid is True
        assert "1 operational" in message
        assert "Failed" in message


# =============================================================================
# Test Flame Failure Response Validation
# =============================================================================

class TestFlameFailureResponse:
    """Test flame failure response time validation."""

    def test_flame_failure_response_compliant(self, checker):
        """Test compliant flame failure response."""
        is_compliant, response_ms, message = checker.calculate_flame_failure_response(
            detection_time_ms=1.0,
            fuel_shutoff_time_ms=2.0
        )

        assert is_compliant is True
        assert response_ms == pytest.approx(3.0)
        assert "PASS" in message

    def test_flame_failure_response_limit(self, checker):
        """Test response at the limit (4 seconds)."""
        is_compliant, response_ms, message = checker.calculate_flame_failure_response(
            detection_time_ms=2.0,
            fuel_shutoff_time_ms=2.0
        )

        assert is_compliant is True
        assert response_ms == pytest.approx(4.0)  # Total response time in milliseconds

    def test_flame_failure_response_violation(self, checker):
        """Test response exceeding limit."""
        is_compliant, response_ms, message = checker.calculate_flame_failure_response(
            detection_time_ms=2500.0,
            fuel_shutoff_time_ms=2500.0
        )

        assert is_compliant is False
        assert response_ms == pytest.approx(5000.0)
        assert "FAIL" in message


# =============================================================================
# Test Trial for Ignition Validation
# =============================================================================

class TestTrialForIgnition:
    """Test trial for ignition timing validation."""

    def test_trial_valid(self, checker):
        """Test valid trial for ignition timing."""
        is_valid, message = checker.validate_trial_for_ignition(
            pilot_trial_seconds=8.0,
            main_trial_seconds=5.0
        )

        assert is_valid is True
        assert "valid" in message.lower()

    def test_trial_pilot_timeout(self, checker):
        """Test pilot trial exceeding time limit."""
        is_valid, message = checker.validate_trial_for_ignition(
            pilot_trial_seconds=12.0,
            main_trial_seconds=5.0
        )

        assert is_valid is False
        assert "12.0" in message
        assert "Pilot" in message

    def test_trial_main_timeout(self, checker):
        """Test main flame trial exceeding time limit."""
        is_valid, message = checker.validate_trial_for_ignition(
            pilot_trial_seconds=8.0,
            main_trial_seconds=12.0
        )

        assert is_valid is False
        assert "12.0" in message
        assert "Main" in message

    def test_trial_total_timeout(self, checker):
        """Test total trial time exceeding limit."""
        is_valid, message = checker.validate_trial_for_ignition(
            pilot_trial_seconds=8.0,
            main_trial_seconds=8.0  # Total = 16 seconds (exceeds 15 limit)
        )

        assert is_valid is False
        assert "16.0" in message


# =============================================================================
# Test LEL Monitoring
# =============================================================================

class TestLELMonitoring:
    """Test LEL (Lower Explosive Limit) monitoring validation."""

    def test_lel_safe_level(self, checker):
        """Test LEL at safe level below alarm."""
        level, message = checker.validate_lel_monitoring(10.0)

        assert level == ComplianceLevel.COMPLIANT
        assert "below alarm" in message.lower()

    def test_lel_alarm_level(self, checker):
        """Test LEL in alarm range."""
        level, message = checker.validate_lel_monitoring(35.0)

        assert level == ComplianceLevel.CONDITIONAL
        assert "alarm range" in message.lower()

    def test_lel_shutdown_level(self, checker):
        """Test LEL exceeding shutdown threshold."""
        level, message = checker.validate_lel_monitoring(75.0)

        assert level == ComplianceLevel.NON_COMPLIANT
        assert "EMERGENCY SHUTDOWN" in message

    def test_lel_at_alarm_threshold(self, checker):
        """Test LEL exactly at alarm threshold."""
        level, message = checker.validate_lel_monitoring(
            NFPA86TimingRequirements.LEL_ALARM_THRESHOLD_PERCENT
        )

        assert level == ComplianceLevel.CONDITIONAL

    def test_lel_at_shutdown_threshold(self, checker):
        """Test LEL exactly at shutdown threshold."""
        level, message = checker.validate_lel_monitoring(
            NFPA86TimingRequirements.LEL_SHUTDOWN_THRESHOLD_PERCENT
        )

        assert level == ComplianceLevel.NON_COMPLIANT


# =============================================================================
# Test Compliance Reporting
# =============================================================================

class TestComplianceReporting:
    """Test compliance reporting functionality."""

    def test_compliance_report_generation(self, checker, base_furnace_config):
        """Test generating compliance report."""
        checker.check_class_a_furnace(base_furnace_config)
        report = checker.get_compliance_report()

        assert "NFPA86" in report["report_id"]
        assert report["standard"] == "NFPA 86-2023"
        assert report["total_checks_performed"] == 1
        assert "class_a" in report["checks_by_class"]

    def test_compliance_report_summary(self, checker, base_furnace_config):
        """Test compliance report summary section."""
        checker.check_class_a_furnace(base_furnace_config)
        report = checker.get_compliance_report()

        assert "compliance_summary" in report
        assert "compliant" in report["compliance_summary"]
        assert "timing_requirements" in report

    def test_provenance_hash_generation(self, checker, base_furnace_config):
        """Test provenance hash generation."""
        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256 hex


# =============================================================================
# Test Check History
# =============================================================================

class TestCheckHistory:
    """Test compliance check history tracking."""

    def test_check_history_accumulation(self, checker, base_furnace_config):
        """Test that checks are recorded in history."""
        assert len(checker.check_history) == 0

        checker.check_class_a_furnace(base_furnace_config)
        assert len(checker.check_history) == 1

        base_furnace_config.equipment_id = "FURN-002"
        base_furnace_config.classification = FurnaceClass.CLASS_B
        checker.check_class_b_furnace(base_furnace_config)
        assert len(checker.check_history) == 2

    def test_check_history_preserves_results(self, checker, base_furnace_config):
        """Test that history preserves result details."""
        result1 = checker.check_class_a_furnace(base_furnace_config)
        result2 = checker.check_class_a_furnace(base_furnace_config)

        assert checker.check_history[0].check_id != checker.check_history[1].check_id
        assert checker.check_history[0].requirements_failed == 0
        assert checker.check_history[1].requirements_failed == 0


# =============================================================================
# Test Furnace Configuration Validation
# =============================================================================

class TestFurnaceConfigurationValidation:
    """Test furnace configuration edge cases."""

    def test_class_a_with_all_features(self, checker, base_furnace_config):
        """Test Class A with all optional features."""
        base_furnace_config.has_flame_supervision = True
        base_furnace_config.has_combustion_safeguards = True
        base_furnace_config.has_lel_monitoring = True
        base_furnace_config.has_emergency_shutdown = True
        base_furnace_config.has_temperature_monitoring = True

        result = checker.check_class_a_furnace(base_furnace_config)

        assert result.compliance_level == ComplianceLevel.COMPLIANT
        assert result.requirements_failed == 0

    def test_class_a_minimal_config(self, checker):
        """Test Class A with minimal required configuration."""
        config = FurnaceConfiguration(
            equipment_id="FURN-MIN",
            classification=FurnaceClass.CLASS_A,
            atmosphere_type=AtmosphereType.AIR,
            maximum_temperature_deg_f=800.0,
            furnace_volume_cubic_feet=300.0,
            burner_input_btuh=300000.0,
            has_flame_supervision=True,
            has_lel_monitoring=True,
            has_emergency_shutdown=True,
            has_temperature_monitoring=True,
            has_combustion_safeguards=True,
            interlocks=[
                SafetyInterlockConfig(
                    name="Test Interlock",
                    is_operational=True,
                )
            ],
        )

        result = checker.check_class_a_furnace(config)

        assert result.compliance_level == ComplianceLevel.COMPLIANT


# =============================================================================
# Integration Tests
# =============================================================================

class TestComplianceIntegration:
    """Integration tests for compliance checking workflow."""

    def test_complete_class_a_workflow(self, checker):
        """Test complete Class A furnace compliance workflow."""
        config = FurnaceConfiguration(
            equipment_id="FURN-COMPLETE",
            classification=FurnaceClass.CLASS_A,
            atmosphere_type=AtmosphereType.AIR,
            maximum_temperature_deg_f=1000.0,
            furnace_volume_cubic_feet=500.0,
            burner_input_btuh=500000.0,
            has_flame_supervision=True,
            has_combustion_safeguards=True,
            has_lel_monitoring=True,
            has_emergency_shutdown=True,
            has_temperature_monitoring=True,
            interlocks=[
                SafetyInterlockConfig(
                    name="Test Interlock",
                    is_operational=True,
                )
            ],
        )

        # Check compliance
        result = checker.check_class_a_furnace(config)

        # Validate flame failure response
        flame_ok, response_ms, msg = checker.calculate_flame_failure_response(1.0, 2.0)

        # Validate trial for ignition
        trial_ok, trial_msg = checker.validate_trial_for_ignition(8.0, 7.0)

        # Validate LEL monitoring
        lel_level, lel_msg = checker.validate_lel_monitoring(15.0)

        # Validate purge cycle
        purge_ok, purge_msg, purge_time = checker.validate_purge_cycle(
            AtmosphereType.AIR,
            config.furnace_volume_cubic_feet,
            1000.0
        )

        assert result.compliance_level == ComplianceLevel.COMPLIANT
        assert flame_ok is True
        assert trial_ok is True
        assert lel_level == ComplianceLevel.COMPLIANT
        assert purge_ok is True
        assert purge_time > 0

    def test_multiple_furnace_classes(self, checker, base_furnace_config):
        """Test checking multiple furnace classes."""
        classes = [
            (FurnaceClass.CLASS_A, checker.check_class_a_furnace),
            (FurnaceClass.CLASS_B, checker.check_class_b_furnace),
            (FurnaceClass.CLASS_C, checker.check_class_c_furnace),
            (FurnaceClass.CLASS_D, checker.check_class_d_furnace),
        ]

        for furnace_class, check_method in classes:
            base_furnace_config.classification = furnace_class
            base_furnace_config.equipment_id = f"FURN-{furnace_class.value}"

            # Adjust config for class requirements
            if furnace_class == FurnaceClass.CLASS_C:
                base_furnace_config.atmosphere_type = AtmosphereType.ENDOTHERMIC
                base_furnace_config.has_purge_capability = True
            elif furnace_class == FurnaceClass.CLASS_D:
                base_furnace_config.atmosphere_type = AtmosphereType.VACUUM
                base_furnace_config.interlocks = [
                    SafetyInterlockConfig(name="Vacuum", is_operational=True)
                ]

            result = check_method(base_furnace_config)

            assert result.classification == furnace_class
            assert result.check_id.startswith("F86-")
            assert len(result.provenance_hash) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
