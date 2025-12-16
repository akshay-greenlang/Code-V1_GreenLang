"""
GL-003 Unified Steam System Optimizer - Configuration Tests

Unit tests for configuration module with comprehensive validation testing.
Target: 85%+ coverage of config.py

Tests:
    - SteamHeaderConfig validation
    - PRVConfig validation with ASME B31.1 constraints
    - QualityMonitoringConfig ASME limits
    - FlashRecoveryConfig thermodynamic constraints
    - ExergyOptimizationConfig weight validation
    - UnifiedSteamConfig complete validation
"""

import pytest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    UnifiedSteamConfig,
    SteamHeaderConfig,
    SteamHeaderLevel,
    PRVConfig,
    PRVSizingMethod,
    DesuperheaterConfig,
    DesuperheaterType,
    QualityMonitoringConfig,
    SteamQualityStandard,
    CondensateConfig,
    CondensateFlashMethod,
    FlashRecoveryConfig,
    SteamTrapSurveyConfig,
    ExergyOptimizationConfig,
    create_default_config,
)


# =============================================================================
# STEAM HEADER CONFIG TESTS
# =============================================================================

class TestSteamHeaderConfig:
    """Test suite for SteamHeaderConfig."""

    def test_valid_header_config(self, hp_header_config):
        """Test valid header configuration."""
        assert hp_header_config.name == "HP-MAIN"
        assert hp_header_config.level == SteamHeaderLevel.HIGH_PRESSURE
        assert hp_header_config.design_pressure_psig == 600.0

    def test_pressure_range_validation(self):
        """Test min/max pressure validation."""
        with pytest.raises(ValidationError) as exc_info:
            SteamHeaderConfig(
                name="TEST",
                design_pressure_psig=600.0,
                min_pressure_psig=620.0,  # Greater than max
                max_pressure_psig=580.0,
            )
        assert "max_pressure_psig must be >= min_pressure_psig" in str(exc_info.value)

    def test_design_pressure_in_range(self):
        """Test design pressure must be within operating range."""
        with pytest.raises(ValidationError) as exc_info:
            SteamHeaderConfig(
                name="TEST",
                design_pressure_psig=700.0,  # Above max
                min_pressure_psig=580.0,
                max_pressure_psig=620.0,
            )
        assert "design_pressure_psig must be <= max_pressure_psig" in str(exc_info.value)

    def test_design_pressure_below_min(self):
        """Test design pressure below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            SteamHeaderConfig(
                name="TEST",
                design_pressure_psig=500.0,  # Below min
                min_pressure_psig=580.0,
                max_pressure_psig=620.0,
            )
        assert "design_pressure_psig must be >= min_pressure_psig" in str(exc_info.value)

    def test_negative_pressure_rejection(self):
        """Test negative pressure values are rejected."""
        with pytest.raises(ValidationError):
            SteamHeaderConfig(
                name="TEST",
                design_pressure_psig=-10.0,
                min_pressure_psig=-20.0,
                max_pressure_psig=0.0,
            )

    def test_max_pressure_limit(self):
        """Test maximum pressure limit (1500 psig)."""
        with pytest.raises(ValidationError):
            SteamHeaderConfig(
                name="TEST",
                design_pressure_psig=1600.0,
                min_pressure_psig=1550.0,
                max_pressure_psig=1650.0,
            )

    def test_header_level_assignment(self):
        """Test header level is correctly assigned."""
        hp = SteamHeaderConfig(
            name="HP",
            level=SteamHeaderLevel.HIGH_PRESSURE,
            design_pressure_psig=600.0,
            min_pressure_psig=580.0,
            max_pressure_psig=620.0,
        )
        assert hp.level == SteamHeaderLevel.HIGH_PRESSURE

    def test_default_values(self):
        """Test default values are applied."""
        config = SteamHeaderConfig(
            name="TEST",
            design_pressure_psig=150.0,
            min_pressure_psig=140.0,
            max_pressure_psig=160.0,
        )
        assert config.design_flow_lb_hr == 0.0
        assert config.exergy_reference_temp_f == 77.0
        assert config.level == SteamHeaderLevel.MEDIUM_PRESSURE

    @pytest.mark.parametrize("level,expected_value", [
        (SteamHeaderLevel.HIGH_PRESSURE, "HP"),
        (SteamHeaderLevel.MEDIUM_PRESSURE, "MP"),
        (SteamHeaderLevel.LOW_PRESSURE, "LP"),
        (SteamHeaderLevel.VERY_LOW_PRESSURE, "VLP"),
    ])
    def test_header_level_enum_values(self, level, expected_value):
        """Test header level enum values."""
        assert level.value == expected_value


# =============================================================================
# PRV CONFIG TESTS
# =============================================================================

class TestPRVConfig:
    """Test suite for PRVConfig."""

    def test_valid_prv_config(self, prv_config):
        """Test valid PRV configuration."""
        assert prv_config.prv_id == "PRV-HP-MP"
        assert prv_config.inlet_pressure_psig == 600.0
        assert prv_config.outlet_pressure_psig == 150.0

    def test_outlet_less_than_inlet(self):
        """Test outlet pressure must be less than inlet."""
        with pytest.raises(ValidationError) as exc_info:
            PRVConfig(
                prv_id="TEST",
                inlet_pressure_psig=150.0,
                outlet_pressure_psig=600.0,  # Greater than inlet
                design_flow_lb_hr=30000.0,
                max_flow_lb_hr=40000.0,
                cv_rated=150.0,
            )
        assert "outlet_pressure_psig must be < inlet_pressure_psig" in str(exc_info.value)

    def test_opening_range_validation(self):
        """Test opening percentage range validation."""
        with pytest.raises(ValidationError):
            PRVConfig(
                prv_id="TEST",
                inlet_pressure_psig=600.0,
                outlet_pressure_psig=150.0,
                design_flow_lb_hr=30000.0,
                max_flow_lb_hr=40000.0,
                cv_rated=150.0,
                target_opening_min_pct=70.0,
                target_opening_max_pct=50.0,  # Less than min
            )

    def test_asme_b31_1_opening_targets(self, asme_prv_opening_targets):
        """Test ASME B31.1 opening targets (50-70%)."""
        config = PRVConfig(
            prv_id="TEST",
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
            design_flow_lb_hr=30000.0,
            max_flow_lb_hr=40000.0,
            cv_rated=150.0,
            target_opening_min_pct=50.0,
            target_opening_max_pct=70.0,
        )
        assert config.target_opening_min_pct == asme_prv_opening_targets["minimum"]
        assert config.target_opening_max_pct == asme_prv_opening_targets["maximum"]

    def test_sizing_method_enum(self):
        """Test PRV sizing method enum values."""
        assert PRVSizingMethod.ASME_B31_1.value == "asme_b31_1"
        assert PRVSizingMethod.API_520.value == "api_520"
        assert PRVSizingMethod.MANUFACTURER.value == "manufacturer"

    def test_desuperheater_configuration(self):
        """Test desuperheater configuration options."""
        config = PRVConfig(
            prv_id="TEST",
            inlet_pressure_psig=600.0,
            outlet_pressure_psig=150.0,
            design_flow_lb_hr=30000.0,
            max_flow_lb_hr=40000.0,
            cv_rated=150.0,
            desuperheater_enabled=True,
            desuperheater_type=DesuperheaterType.WATER_SPRAY,
            target_superheat_f=50.0,
        )
        assert config.desuperheater_enabled is True
        assert config.desuperheater_type == DesuperheaterType.WATER_SPRAY


# =============================================================================
# QUALITY MONITORING CONFIG TESTS
# =============================================================================

class TestQualityMonitoringConfig:
    """Test suite for QualityMonitoringConfig."""

    def test_valid_quality_config(self, quality_config):
        """Test valid quality monitoring configuration."""
        assert quality_config.standard == SteamQualityStandard.ASME
        assert quality_config.min_dryness_fraction == 0.95

    def test_asme_quality_limits(self, quality_config, asme_quality_limits):
        """Test ASME quality limits are correct."""
        assert quality_config.min_dryness_fraction == asme_quality_limits["min_dryness_fraction"]
        assert quality_config.max_tds_ppm_lp == asme_quality_limits["max_tds_lp"]
        assert quality_config.max_cation_conductivity_us_cm == asme_quality_limits["max_cation_conductivity"]

    def test_dryness_fraction_bounds(self):
        """Test dryness fraction bounds (0.80-1.0)."""
        with pytest.raises(ValidationError):
            QualityMonitoringConfig(min_dryness_fraction=0.5)  # Below 0.80

        with pytest.raises(ValidationError):
            QualityMonitoringConfig(min_dryness_fraction=1.1)  # Above 1.0

    def test_threshold_percentages(self):
        """Test threshold percentage bounds."""
        with pytest.raises(ValidationError):
            QualityMonitoringConfig(warning_threshold_pct=40.0)  # Below 50

        with pytest.raises(ValidationError):
            QualityMonitoringConfig(critical_threshold_pct=110.0)  # Above 100

    @pytest.mark.parametrize("tds_lp,tds_mp,tds_hp", [
        (3500.0, 3000.0, 2500.0),
        (2000.0, 1500.0, 1000.0),
        (4000.0, 3500.0, 3000.0),
    ])
    def test_tds_limits_by_pressure(self, tds_lp, tds_mp, tds_hp):
        """Test TDS limits vary correctly by pressure range."""
        config = QualityMonitoringConfig(
            max_tds_ppm_lp=tds_lp,
            max_tds_ppm_mp=tds_mp,
            max_tds_ppm_hp=tds_hp,
        )
        # LP limit should be highest (lower pressure = higher TDS tolerance)
        assert config.max_tds_ppm_lp >= config.max_tds_ppm_mp >= config.max_tds_ppm_hp


# =============================================================================
# CONDENSATE CONFIG TESTS
# =============================================================================

class TestCondensateConfig:
    """Test suite for CondensateConfig."""

    def test_valid_condensate_config(self, condensate_config):
        """Test valid condensate configuration."""
        assert condensate_config.target_return_rate_pct == 85.0
        assert condensate_config.flash_recovery_enabled is True

    def test_return_rate_bounds(self):
        """Test return rate percentage bounds (0-100)."""
        with pytest.raises(ValidationError):
            CondensateConfig(target_return_rate_pct=110.0)

        with pytest.raises(ValidationError):
            CondensateConfig(min_acceptable_return_pct=-10.0)

    def test_temperature_bounds(self):
        """Test temperature bounds."""
        # Temperature should be within reasonable range
        config = CondensateConfig(
            target_return_temp_f=180.0,
            min_return_temp_f=140.0,
        )
        assert config.target_return_temp_f == 180.0

    def test_flash_method_enum(self):
        """Test condensate flash method enum."""
        assert CondensateFlashMethod.FLASH_TANK.value == "flash_tank"
        assert CondensateFlashMethod.THERMODYNAMIC.value == "thermodynamic"


# =============================================================================
# FLASH RECOVERY CONFIG TESTS
# =============================================================================

class TestFlashRecoveryConfig:
    """Test suite for FlashRecoveryConfig."""

    def test_valid_flash_config(self, flash_recovery_config):
        """Test valid flash recovery configuration."""
        assert flash_recovery_config.condensate_pressure_psig == 150.0
        assert flash_recovery_config.flash_pressure_psig == 15.0

    def test_flash_pressure_less_than_condensate(self):
        """Test flash pressure must be less than condensate pressure."""
        with pytest.raises(ValidationError) as exc_info:
            FlashRecoveryConfig(
                condensate_pressure_psig=15.0,
                flash_pressure_psig=150.0,  # Greater than condensate
            )
        assert "flash_pressure_psig must be < condensate_pressure_psig" in str(exc_info.value)

    def test_operating_hours_bounds(self):
        """Test operating hours bounds (1000-8760)."""
        with pytest.raises(ValidationError):
            FlashRecoveryConfig(
                condensate_pressure_psig=150.0,
                flash_pressure_psig=15.0,
                operating_hours_per_year=100,  # Below 1000
            )

        with pytest.raises(ValidationError):
            FlashRecoveryConfig(
                condensate_pressure_psig=150.0,
                flash_pressure_psig=15.0,
                operating_hours_per_year=10000,  # Above 8760
            )

    def test_efficiency_bounds(self):
        """Test recovery efficiency bounds (50-100%)."""
        with pytest.raises(ValidationError):
            FlashRecoveryConfig(
                condensate_pressure_psig=150.0,
                flash_pressure_psig=15.0,
                min_recovery_efficiency_pct=40.0,  # Below 50
            )


# =============================================================================
# EXERGY OPTIMIZATION CONFIG TESTS
# =============================================================================

class TestExergyOptimizationConfig:
    """Test suite for ExergyOptimizationConfig."""

    def test_valid_exergy_config(self, exergy_config):
        """Test valid exergy configuration."""
        assert exergy_config.enabled is True
        assert exergy_config.reference_temperature_f == 77.0

    def test_weights_sum_to_one(self):
        """Test optimization weights must sum to 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ExergyOptimizationConfig(
                exergy_weight=0.5,
                cost_weight=0.3,
                reliability_weight=0.3,  # Sum = 1.1
            )
        assert "weights must sum to 1.0" in str(exc_info.value)

    def test_valid_weight_combinations(self):
        """Test valid weight combinations."""
        config = ExergyOptimizationConfig(
            exergy_weight=0.6,
            cost_weight=0.3,
            reliability_weight=0.1,
        )
        total = config.exergy_weight + config.cost_weight + config.reliability_weight
        assert abs(total - 1.0) < 0.001

    def test_weight_bounds(self):
        """Test weight bounds (0-1)."""
        with pytest.raises(ValidationError):
            ExergyOptimizationConfig(
                exergy_weight=1.5,  # Above 1.0
                cost_weight=0.0,
                reliability_weight=0.0,
            )


# =============================================================================
# UNIFIED STEAM CONFIG TESTS
# =============================================================================

class TestUnifiedSteamConfig:
    """Test suite for UnifiedSteamConfig."""

    def test_create_default_config(self):
        """Test default configuration creation."""
        config = create_default_config()

        assert config.agent_id == "GL-003-DEFAULT"
        assert len(config.headers) == 3
        assert len(config.prvs) == 2
        assert len(config.flash_recovery) == 1

    def test_default_config_headers(self):
        """Test default configuration has correct headers."""
        config = create_default_config()

        header_names = [h.name for h in config.headers]
        assert "HP-MAIN" in header_names
        assert "MP-MAIN" in header_names
        assert "LP-MAIN" in header_names

    def test_default_config_prvs(self):
        """Test default configuration has correct PRVs."""
        config = create_default_config()

        prv_ids = [p.prv_id for p in config.prvs]
        assert "PRV-HP-MP" in prv_ids
        assert "PRV-MP-LP" in prv_ids

    def test_config_with_custom_headers(self, hp_header_config, mp_header_config):
        """Test configuration with custom headers."""
        config = UnifiedSteamConfig(
            agent_id="TEST-001",
            headers=[hp_header_config, mp_header_config],
        )
        assert len(config.headers) == 2
        assert config.headers[0].name == "HP-MAIN"

    def test_config_provenance_enabled_default(self):
        """Test provenance is enabled by default."""
        config = UnifiedSteamConfig()
        assert config.provenance_enabled is True

    def test_config_calculation_precision(self):
        """Test calculation precision bounds (2-8)."""
        with pytest.raises(ValidationError):
            UnifiedSteamConfig(calculation_precision=1)  # Below 2

        with pytest.raises(ValidationError):
            UnifiedSteamConfig(calculation_precision=10)  # Above 8

    def test_config_serialization(self):
        """Test configuration can be serialized to dict."""
        config = create_default_config()
        config_dict = config.dict()

        assert "agent_id" in config_dict
        assert "headers" in config_dict
        assert "quality" in config_dict

    def test_config_json_serialization(self):
        """Test configuration can be serialized to JSON."""
        config = create_default_config()
        json_str = config.json()

        assert "GL-003-DEFAULT" in json_str
        assert "HP-MAIN" in json_str


# =============================================================================
# DESUPERHEATER CONFIG TESTS
# =============================================================================

class TestDesuperheaterConfig:
    """Test suite for DesuperheaterConfig."""

    def test_valid_desuperheater_config(self):
        """Test valid desuperheater configuration."""
        config = DesuperheaterConfig(
            desuperheater_id="DSH-001",
            type=DesuperheaterType.WATER_SPRAY,
            target_outlet_temp_f=400.0,
            min_approach_temp_f=20.0,
            spray_water_temp_f=200.0,
            max_spray_rate_lb_hr=5000.0,
            spray_water_pressure_psig=200.0,
        )
        assert config.desuperheater_id == "DSH-001"
        assert config.type == DesuperheaterType.WATER_SPRAY

    def test_desuperheater_types(self):
        """Test desuperheater type enum values."""
        assert DesuperheaterType.WATER_SPRAY.value == "water_spray"
        assert DesuperheaterType.STEAM_ATOMIZING.value == "steam_atomizing"
        assert DesuperheaterType.SURFACE_CONTACT.value == "surface_contact"
        assert DesuperheaterType.VENTURI.value == "venturi"

    def test_approach_temperature_bounds(self):
        """Test minimum approach temperature bounds (5-50F)."""
        with pytest.raises(ValidationError):
            DesuperheaterConfig(
                desuperheater_id="TEST",
                target_outlet_temp_f=400.0,
                min_approach_temp_f=2.0,  # Below 5
                spray_water_temp_f=200.0,
                max_spray_rate_lb_hr=5000.0,
                spray_water_pressure_psig=200.0,
            )


# =============================================================================
# STEAM TRAP SURVEY CONFIG TESTS
# =============================================================================

class TestSteamTrapSurveyConfig:
    """Test suite for SteamTrapSurveyConfig."""

    def test_valid_trap_survey_config(self, trap_survey_config):
        """Test valid trap survey configuration."""
        assert trap_survey_config.survey_enabled is True
        assert trap_survey_config.survey_frequency_days == 90

    def test_survey_frequency_bounds(self):
        """Test survey frequency bounds (30-365 days)."""
        with pytest.raises(ValidationError):
            SteamTrapSurveyConfig(survey_frequency_days=10)  # Below 30

        with pytest.raises(ValidationError):
            SteamTrapSurveyConfig(survey_frequency_days=400)  # Above 365

    def test_failure_threshold_bounds(self):
        """Test failure threshold bounds (0-50%)."""
        with pytest.raises(ValidationError):
            SteamTrapSurveyConfig(failed_open_threshold_pct=60.0)  # Above 50

    def test_default_trap_types(self):
        """Test default tracked trap types."""
        config = SteamTrapSurveyConfig()
        assert "thermodynamic" in config.tracked_trap_types
        assert "thermostatic" in config.tracked_trap_types
        assert "float_thermostatic" in config.tracked_trap_types
        assert "inverted_bucket" in config.tracked_trap_types
