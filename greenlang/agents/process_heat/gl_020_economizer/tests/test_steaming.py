"""
Unit tests for GL-020 ECONOPULSE Steaming Economizer Detector

Tests steaming detection, approach temperature monitoring, and prevention logic.
Target coverage: 85%+

Standards Reference:
    - ASME PTC 4.1 Steam Generating Units
    - ASME Boiler and Pressure Vessel Code

Zero-Hallucination: All calculations use deterministic formulas.
"""

import pytest
import math
from datetime import datetime, timezone

from ..steaming import (
    SteamingDetector,
    SteamingConfig,
    SteamingInput,
    SteamingResult,
    create_steaming_detector,
    MIN_SAFE_APPROACH_F,
    APPROACH_WARNING_F,
    APPROACH_ALARM_F,
    APPROACH_CRITICAL_F,
    SAT_TEMP_COEF_A,
    SAT_TEMP_COEF_B,
    SAT_TEMP_COEF_C,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default steaming detection configuration."""
    return SteamingConfig()


@pytest.fixture
def detector(default_config):
    """Create default steaming detector."""
    return SteamingDetector(default_config)


@pytest.fixture
def config_with_recirculation():
    """Configuration with recirculation enabled."""
    return SteamingConfig(
        recirculation_enabled=True,
        recirculation_trigger_approach_f=15.0,
        recirculation_flow_pct=20.0,
    )


@pytest.fixture
def normal_input():
    """Normal operating condition (safe)."""
    return SteamingInput(
        timestamp=datetime.now(timezone.utc),
        water_outlet_temp_f=420.0,
        water_outlet_pressure_psig=500.0,
        current_load_pct=75.0,
        water_flow_lb_hr=80000.0,
        design_water_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
    )


@pytest.fixture
def warning_input():
    """Warning condition (approaching steaming)."""
    return SteamingInput(
        timestamp=datetime.now(timezone.utc),
        water_outlet_temp_f=455.0,  # Close to saturation
        water_outlet_pressure_psig=500.0,
        current_load_pct=50.0,
        water_flow_lb_hr=60000.0,
        design_water_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
    )


@pytest.fixture
def critical_input():
    """Critical condition (steaming imminent)."""
    return SteamingInput(
        timestamp=datetime.now(timezone.utc),
        water_outlet_temp_f=465.0,  # Very close to saturation
        water_outlet_pressure_psig=500.0,
        current_load_pct=35.0,
        water_flow_lb_hr=40000.0,
        design_water_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
    )


@pytest.fixture
def steaming_input():
    """Actual steaming condition."""
    return SteamingInput(
        timestamp=datetime.now(timezone.utc),
        water_outlet_temp_f=475.0,  # Above saturation
        water_outlet_pressure_psig=500.0,
        current_load_pct=25.0,
        water_flow_lb_hr=30000.0,
        design_water_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
    )


@pytest.fixture
def fluctuating_input():
    """Input with DP and temperature fluctuations."""
    return SteamingInput(
        timestamp=datetime.now(timezone.utc),
        water_outlet_temp_f=450.0,
        water_outlet_pressure_psig=500.0,
        current_load_pct=60.0,
        water_flow_lb_hr=65000.0,
        design_water_flow_lb_hr=100000.0,
        gas_inlet_temp_f=600.0,
        recent_dp_values=[5.0, 5.5, 4.8, 5.8, 4.5, 5.2, 6.0, 4.3],  # Fluctuating
        recent_temp_values=[448.0, 452.0, 447.0, 455.0, 445.0, 453.0],  # Fluctuating
    )


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestSteamingConfig:
    """Test steaming detection configuration."""

    def test_default_values(self, default_config):
        """Test default configuration values."""
        assert default_config.design_approach_temp_f == 30.0
        assert default_config.design_subcooling_f == 20.0
        assert default_config.design_outlet_pressure_psig == 500.0
        assert default_config.approach_warning_f == APPROACH_WARNING_F
        assert default_config.approach_alarm_f == APPROACH_ALARM_F
        assert default_config.approach_critical_f == APPROACH_CRITICAL_F
        assert default_config.steaming_detection_enabled is True
        assert default_config.recirculation_enabled is False

    def test_recirculation_config(self, config_with_recirculation):
        """Test recirculation configuration."""
        assert config_with_recirculation.recirculation_enabled is True
        assert config_with_recirculation.recirculation_trigger_approach_f == 15.0
        assert config_with_recirculation.recirculation_flow_pct == 20.0


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestSteamingDetectorInit:
    """Test detector initialization."""

    def test_default_initialization(self, detector, default_config):
        """Test default initialization."""
        assert detector.config == default_config

    def test_factory_function(self):
        """Test factory function."""
        detector = create_steaming_detector()
        assert isinstance(detector, SteamingDetector)

    def test_factory_with_config(self, config_with_recirculation):
        """Test factory with custom config."""
        detector = create_steaming_detector(config_with_recirculation)
        assert detector.config.recirculation_enabled is True


# =============================================================================
# SATURATION TEMPERATURE TESTS
# =============================================================================

class TestSaturationTemperature:
    """Test saturation temperature calculation."""

    def test_atmospheric_pressure(self, detector):
        """Test saturation temperature at atmospheric pressure."""
        t_sat = detector.calculate_saturation_temperature(0.0)

        # At 14.7 psia (0 psig), T_sat = 212F
        assert t_sat == pytest.approx(212.0, rel=0.02)

    def test_500_psig(self, detector):
        """Test saturation temperature at 500 psig."""
        t_sat = detector.calculate_saturation_temperature(500.0)

        # At 514.7 psia, T_sat should be approximately 467-470F
        assert 460.0 < t_sat < 480.0

    def test_1000_psig(self, detector):
        """Test saturation temperature at 1000 psig."""
        t_sat = detector.calculate_saturation_temperature(1000.0)

        # At 1014.7 psia, T_sat should be approximately 544-548F
        assert 540.0 < t_sat < 560.0

    def test_temperature_increases_with_pressure(self, detector):
        """Test that saturation temperature increases with pressure."""
        pressures = [100, 300, 500, 700, 1000]
        temps = [detector.calculate_saturation_temperature(p) for p in pressures]

        for i in range(1, len(temps)):
            assert temps[i] > temps[i - 1]

    def test_negative_pressure_handling(self, detector):
        """Test handling of negative pressure (vacuum)."""
        t_sat = detector.calculate_saturation_temperature(-5.0)

        # Should return reasonable value
        assert t_sat > 100.0
        assert t_sat < 220.0


# =============================================================================
# APPROACH TEMPERATURE TESTS
# =============================================================================

class TestApproachTemperature:
    """Test approach temperature calculation."""

    def test_safe_approach(self, detector):
        """Test safe approach temperature."""
        approach, subcooling = detector.calculate_approach_temperature(
            water_outlet_temp_f=420.0,
            saturation_temp_f=470.0,
        )

        assert approach == pytest.approx(50.0, rel=0.01)
        assert subcooling == pytest.approx(50.0, rel=0.01)

    def test_warning_approach(self, detector):
        """Test approach at warning level."""
        approach, subcooling = detector.calculate_approach_temperature(
            water_outlet_temp_f=455.0,
            saturation_temp_f=470.0,
        )

        assert approach == pytest.approx(15.0, rel=0.01)
        assert approach <= APPROACH_WARNING_F

    def test_critical_approach(self, detector):
        """Test approach at critical level."""
        approach, subcooling = detector.calculate_approach_temperature(
            water_outlet_temp_f=467.0,
            saturation_temp_f=470.0,
        )

        assert approach == pytest.approx(3.0, rel=0.1)
        assert approach < APPROACH_CRITICAL_F

    def test_negative_approach_steaming(self, detector):
        """Test negative approach (steaming condition)."""
        approach, subcooling = detector.calculate_approach_temperature(
            water_outlet_temp_f=475.0,
            saturation_temp_f=470.0,
        )

        assert approach < 0
        assert subcooling < 0


# =============================================================================
# RISK ASSESSMENT TESTS
# =============================================================================

class TestApproachRiskAssessment:
    """Test approach temperature risk assessment."""

    def test_low_risk(self, detector):
        """Test low risk with large approach."""
        risk_level, risk_score = detector.assess_approach_risk(40.0)

        assert risk_level == "low"
        assert risk_score < 30

    def test_moderate_risk(self, detector):
        """Test moderate risk at warning level."""
        risk_level, risk_score = detector.assess_approach_risk(12.0)

        assert risk_level == "moderate"
        assert 50 <= risk_score < 80

    def test_high_risk(self, detector):
        """Test high risk at alarm level."""
        risk_level, risk_score = detector.assess_approach_risk(7.0)

        assert risk_level == "high"
        assert 75 <= risk_score < 100

    def test_critical_risk(self, detector):
        """Test critical risk below critical threshold."""
        risk_level, risk_score = detector.assess_approach_risk(3.0)

        assert risk_level == "critical"
        assert risk_score == 100.0

    def test_risk_score_increases_with_decreasing_approach(self, detector):
        """Test risk score increases as approach decreases."""
        approaches = [40.0, 20.0, 12.0, 7.0, 3.0]
        scores = [detector.assess_approach_risk(a)[1] for a in approaches]

        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]


# =============================================================================
# FLUCTUATION DETECTION TESTS
# =============================================================================

class TestFluctuationDetection:
    """Test DP and temperature fluctuation detection."""

    def test_no_fluctuation_stable(self, detector):
        """Test no fluctuation detected with stable values."""
        values = [5.0, 5.1, 5.0, 5.1, 5.0, 5.1]

        detected, magnitude = detector.detect_fluctuations(
            values, threshold=10.0, is_percentage=True
        )

        assert detected is False
        assert magnitude < 10.0

    def test_fluctuation_detected(self, detector):
        """Test fluctuation detected with varying values."""
        values = [5.0, 6.0, 4.5, 6.5, 4.0, 5.5]  # ~20% variation

        detected, magnitude = detector.detect_fluctuations(
            values, threshold=10.0, is_percentage=True
        )

        assert detected is True
        assert magnitude > 10.0

    def test_temperature_fluctuation_absolute(self, detector):
        """Test temperature fluctuation with absolute threshold."""
        values = [450.0, 455.0, 447.0, 458.0, 445.0]  # ~7F std dev

        detected, magnitude = detector.detect_fluctuations(
            values, threshold=5.0, is_percentage=False
        )

        assert detected is True
        assert magnitude > 5.0

    def test_insufficient_data(self, detector):
        """Test handling of insufficient data."""
        values = [5.0, 5.1]  # Only 2 points

        detected, magnitude = detector.detect_fluctuations(
            values, threshold=10.0, is_percentage=True
        )

        assert detected is False
        assert magnitude == 0.0


# =============================================================================
# LOW LOAD RISK TESTS
# =============================================================================

class TestLowLoadRisk:
    """Test low load risk assessment."""

    def test_normal_load_safe(self, detector):
        """Test normal load is safe."""
        low_load_risk, min_safe = detector.assess_low_load_risk(
            current_load_pct=75.0,
            water_flow_pct=80.0,
        )

        assert low_load_risk is False

    def test_low_load_risky(self, detector):
        """Test low load is risky."""
        low_load_risk, min_safe = detector.assess_low_load_risk(
            current_load_pct=25.0,
            water_flow_pct=30.0,
        )

        assert low_load_risk is True

    def test_low_flow_increases_risk(self, detector):
        """Test low flow increases minimum safe load."""
        _, min_safe_normal = detector.assess_low_load_risk(
            current_load_pct=50.0,
            water_flow_pct=80.0,
        )
        _, min_safe_low_flow = detector.assess_low_load_risk(
            current_load_pct=50.0,
            water_flow_pct=20.0,
        )

        assert min_safe_low_flow > min_safe_normal


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================

class TestRecommendations:
    """Test action recommendations."""

    def test_no_action_low_risk(self, detector):
        """Test no action needed at low risk."""
        recs = detector.determine_recommendations(
            risk_level="low",
            risk_score=20.0,
            approach_temp_f=40.0,
            low_load_risk=False,
            dp_fluctuation=False,
            temp_fluctuation=False,
            water_flow_pct=80.0,
            recirculation_active=False,
        )

        assert recs["action_required"] is False
        assert recs["reduce_heat_input"] is False

    def test_action_required_high_risk(self, detector):
        """Test action required at high risk."""
        recs = detector.determine_recommendations(
            risk_level="high",
            risk_score=85.0,
            approach_temp_f=7.0,
            low_load_risk=False,
            dp_fluctuation=False,
            temp_fluctuation=False,
            water_flow_pct=70.0,
            recirculation_active=False,
        )

        assert recs["action_required"] is True
        assert recs["increase_water_flow"] is True

    def test_critical_action(self, detector):
        """Test critical action recommendations."""
        recs = detector.determine_recommendations(
            risk_level="critical",
            risk_score=100.0,
            approach_temp_f=3.0,
            low_load_risk=True,
            dp_fluctuation=True,
            temp_fluctuation=True,
            water_flow_pct=40.0,
            recirculation_active=False,
        )

        assert recs["action_required"] is True
        assert recs["reduce_heat_input"] is True
        assert "CRITICAL" in recs["recommended_action"]

    def test_recirculation_recommendation(self):
        """Test recirculation recommendation."""
        config = SteamingConfig(
            recirculation_enabled=True,
            recirculation_trigger_approach_f=15.0,
        )
        detector = SteamingDetector(config)

        recs = detector.determine_recommendations(
            risk_level="moderate",
            risk_score=60.0,
            approach_temp_f=12.0,
            low_load_risk=False,
            dp_fluctuation=False,
            temp_fluctuation=False,
            water_flow_pct=70.0,
            recirculation_active=False,
        )

        assert recs["activate_recirculation"] is True


# =============================================================================
# COMPLETE DETECTION TESTS
# =============================================================================

class TestCompleteDetection:
    """Test complete steaming detection."""

    def test_normal_operation(self, detector, normal_input):
        """Test normal operation detection."""
        result = detector.detect(normal_input)

        assert isinstance(result, SteamingResult)
        assert result.steaming_detected is False
        assert result.steaming_risk == "low"
        assert result.action_required is False

    def test_warning_condition(self, detector, warning_input):
        """Test warning condition detection."""
        result = detector.detect(warning_input)

        assert result.steaming_risk in ["moderate", "high"]
        assert result.approach_temp_f < 20.0

    def test_critical_condition(self, detector, critical_input):
        """Test critical condition detection."""
        result = detector.detect(critical_input)

        assert result.steaming_risk in ["high", "critical"]
        assert result.action_required is True

    def test_steaming_detected(self, detector, steaming_input):
        """Test actual steaming detection."""
        result = detector.detect(steaming_input)

        assert result.steaming_detected is True
        assert result.steaming_risk == "critical"
        assert result.steaming_risk_score >= 90

    def test_fluctuation_detection_in_result(self, detector, fluctuating_input):
        """Test fluctuation detection in complete result."""
        result = detector.detect(fluctuating_input)

        # Should detect fluctuations
        assert result.dp_fluctuation_detected is True or result.temp_fluctuation_detected is True

    def test_saturation_temp_provided(self, detector):
        """Test with provided saturation temperature."""
        input_data = SteamingInput(
            timestamp=datetime.now(timezone.utc),
            water_outlet_temp_f=420.0,
            water_outlet_pressure_psig=500.0,
            current_load_pct=75.0,
            water_flow_lb_hr=80000.0,
            design_water_flow_lb_hr=100000.0,
            gas_inlet_temp_f=600.0,
            saturation_temp_f=470.0,  # Explicitly provided
        )

        result = detector.detect(input_data)

        assert result.saturation_temp_f == 470.0

    def test_drum_pressure_used(self, detector):
        """Test drum pressure used for saturation calculation."""
        input_data = SteamingInput(
            timestamp=datetime.now(timezone.utc),
            water_outlet_temp_f=420.0,
            water_outlet_pressure_psig=500.0,
            current_load_pct=75.0,
            water_flow_lb_hr=80000.0,
            design_water_flow_lb_hr=100000.0,
            gas_inlet_temp_f=600.0,
            drum_pressure_psig=600.0,  # Higher drum pressure
        )

        result = detector.detect(input_data)

        # Saturation temp should be based on drum pressure
        expected_sat = detector.calculate_saturation_temperature(600.0)
        assert result.saturation_temp_f == pytest.approx(expected_sat, rel=0.01)

    def test_provenance_hash_included(self, detector, normal_input):
        """Test provenance hash in result."""
        result = detector.detect(normal_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    def test_calculation_method_recorded(self, detector, normal_input):
        """Test calculation method is recorded."""
        result = detector.detect(normal_input)

        assert result.calculation_method == "APPROACH_TEMP_MONITORING"


# =============================================================================
# PARAMETERIZED TESTS
# =============================================================================

class TestParameterized:
    """Parameterized tests for various conditions."""

    @pytest.mark.parametrize("approach,expected_risk", [
        (40.0, "low"),
        (20.0, "low"),
        (12.0, "moderate"),
        (7.0, "high"),
        (3.0, "critical"),
    ])
    def test_risk_levels_by_approach(self, detector, approach, expected_risk):
        """Test risk level by approach temperature."""
        risk_level, _ = detector.assess_approach_risk(approach)
        assert risk_level == expected_risk

    @pytest.mark.parametrize("pressure_psig,expected_sat_range", [
        (0, (210, 214)),
        (100, (325, 345)),
        (300, (410, 430)),
        (500, (460, 480)),
        (1000, (540, 560)),
    ])
    def test_saturation_temp_at_pressures(self, detector, pressure_psig, expected_sat_range):
        """Test saturation temperature at various pressures."""
        t_sat = detector.calculate_saturation_temperature(pressure_psig)
        assert expected_sat_range[0] <= t_sat <= expected_sat_range[1]

    @pytest.mark.parametrize("load_pct,flow_pct,expected_low_load_risk", [
        (80.0, 85.0, False),
        (50.0, 55.0, False),
        (35.0, 40.0, False),
        (25.0, 30.0, True),
        (20.0, 20.0, True),
    ])
    def test_low_load_risk_conditions(self, detector, load_pct, flow_pct, expected_low_load_risk):
        """Test low load risk at various conditions."""
        low_load_risk, _ = detector.assess_low_load_risk(load_pct, flow_pct)
        assert low_load_risk == expected_low_load_risk
