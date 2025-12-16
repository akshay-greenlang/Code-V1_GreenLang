"""
GL-015 INSULSCAN - Configuration Tests

Unit tests for configuration models including EconomicConfig, SafetyConfig,
IRSurveyConfig, CondensationConfig, and InsulationAnalysisConfig.

Coverage target: 85%+
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_015_insulation_analysis.config import (
    EconomicConfig,
    SafetyConfig,
    IRSurveyConfig,
    CondensationConfig,
    InsulationAnalysisConfig,
    TemperatureUnit,
    LengthUnit,
    CurrencyCode,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_economic_config():
    """Create default economic configuration."""
    return EconomicConfig()


@pytest.fixture
def default_safety_config():
    """Create default safety configuration."""
    return SafetyConfig()


@pytest.fixture
def default_ir_config():
    """Create default IR survey configuration."""
    return IRSurveyConfig()


@pytest.fixture
def default_condensation_config():
    """Create default condensation configuration."""
    return CondensationConfig()


@pytest.fixture
def full_analysis_config():
    """Create complete analysis configuration."""
    return InsulationAnalysisConfig(facility_id="TEST-FACILITY-001")


# =============================================================================
# ECONOMIC CONFIG TESTS
# =============================================================================

class TestEconomicConfig:
    """Tests for EconomicConfig."""

    def test_default_values(self, default_economic_config):
        """Test default economic configuration values."""
        config = default_economic_config

        assert config.energy_cost_per_mmbtu == 8.50
        assert config.electricity_cost_per_kwh == 0.12
        assert config.currency == CurrencyCode.USD.value
        assert config.operating_hours_per_year == 8760
        assert config.plant_lifetime_years == 20
        assert config.discount_rate_pct == 10.0
        assert config.inflation_rate_pct == 2.5
        assert config.labor_rate_per_hour == 85.0
        assert config.minimum_payback_years == 2.0
        assert config.target_roi_pct == 25.0

    def test_insulation_cost_lookup(self, default_economic_config):
        """Test insulation cost lookup table."""
        config = default_economic_config

        assert "calcium_silicate" in config.insulation_cost_per_sqft
        assert "mineral_wool" in config.insulation_cost_per_sqft
        assert "aerogel" in config.insulation_cost_per_sqft
        assert config.insulation_cost_per_sqft["aerogel"] == 85.00
        assert config.insulation_cost_per_sqft["mineral_wool"] == 8.75

    def test_jacketing_cost_lookup(self, default_economic_config):
        """Test jacketing cost lookup table."""
        config = default_economic_config

        assert "aluminum" in config.jacketing_cost_per_sqft
        assert "stainless_steel" in config.jacketing_cost_per_sqft
        assert config.jacketing_cost_per_sqft["stainless_steel"] == 12.00
        assert config.jacketing_cost_per_sqft["pvc"] == 2.50

    def test_custom_economic_config(self):
        """Test custom economic configuration."""
        config = EconomicConfig(
            energy_cost_per_mmbtu=12.00,
            operating_hours_per_year=8000,
            discount_rate_pct=8.0,
        )

        assert config.energy_cost_per_mmbtu == 12.00
        assert config.operating_hours_per_year == 8000
        assert config.discount_rate_pct == 8.0

    def test_energy_cost_validation(self):
        """Test energy cost must be positive."""
        with pytest.raises(ValidationError):
            EconomicConfig(energy_cost_per_mmbtu=0)

        with pytest.raises(ValidationError):
            EconomicConfig(energy_cost_per_mmbtu=-5.0)

    def test_operating_hours_validation(self):
        """Test operating hours must be within range."""
        with pytest.raises(ValidationError):
            EconomicConfig(operating_hours_per_year=500)  # Below 1000

        with pytest.raises(ValidationError):
            EconomicConfig(operating_hours_per_year=9000)  # Above 8784

    def test_discount_rate_validation(self):
        """Test discount rate must be within range."""
        with pytest.raises(ValidationError):
            EconomicConfig(discount_rate_pct=-5.0)

        with pytest.raises(ValidationError):
            EconomicConfig(discount_rate_pct=35.0)  # Above 30

    def test_scaffolding_multiplier_validation(self):
        """Test scaffolding multiplier range."""
        with pytest.raises(ValidationError):
            EconomicConfig(scaffolding_cost_multiplier=0.5)  # Below 1.0

        with pytest.raises(ValidationError):
            EconomicConfig(scaffolding_cost_multiplier=4.0)  # Above 3.0


# =============================================================================
# SAFETY CONFIG TESTS
# =============================================================================

class TestSafetyConfig:
    """Tests for SafetyConfig."""

    def test_default_values(self, default_safety_config):
        """Test default safety configuration values."""
        config = default_safety_config

        assert config.max_touch_temperature_c == 60.0
        assert config.max_touch_temperature_f == 140.0
        assert config.warning_threshold_c == 50.0
        assert config.alarm_threshold_c == 55.0
        assert config.sil_level == 2
        assert config.personnel_protection_zone_ft == 3.0

    def test_burn_threshold_map(self, default_safety_config):
        """Test burn threshold lookup tables."""
        config = default_safety_config

        assert "metal" in config.burn_threshold_map
        assert "non_metal" in config.burn_threshold_map
        assert config.burn_threshold_map["metal"]["60C"] == 1.0
        assert config.burn_threshold_map["non_metal"]["60C"] == 10.0

    def test_custom_safety_config(self):
        """Test custom safety configuration."""
        config = SafetyConfig(
            max_touch_temperature_c=55.0,
            max_touch_temperature_f=131.0,
            sil_level=3,
        )

        assert config.max_touch_temperature_c == 55.0
        assert config.sil_level == 3

    def test_temperature_c_validation(self):
        """Test Celsius temperature limits."""
        with pytest.raises(ValidationError):
            SafetyConfig(max_touch_temperature_c=35.0)  # Below 40

        with pytest.raises(ValidationError):
            SafetyConfig(max_touch_temperature_c=85.0)  # Above 80

    def test_sil_level_validation(self):
        """Test SIL level range."""
        # Valid SIL levels
        for sil in [0, 1, 2, 3, 4]:
            config = SafetyConfig(sil_level=sil)
            assert config.sil_level == sil

        with pytest.raises(ValidationError):
            SafetyConfig(sil_level=5)


# =============================================================================
# IR SURVEY CONFIG TESTS
# =============================================================================

class TestIRSurveyConfig:
    """Tests for IRSurveyConfig."""

    def test_default_values(self, default_ir_config):
        """Test default IR survey configuration values."""
        config = default_ir_config

        assert config.emissivity_default == 0.95
        assert config.reflected_temperature_f == 70.0
        assert config.distance_ft == 6.0
        assert config.hot_spot_threshold_delta_f == 15.0
        assert config.damaged_insulation_threshold_pct == 25.0
        assert config.missing_insulation_threshold_pct == 100.0
        assert config.ambient_temperature_correction is True
        assert config.wind_speed_correction is True
        assert config.solar_loading_correction is True

    def test_emissivity_validation(self):
        """Test emissivity range validation."""
        with pytest.raises(ValidationError):
            IRSurveyConfig(emissivity_default=0.05)  # Below 0.1

        with pytest.raises(ValidationError):
            IRSurveyConfig(emissivity_default=1.5)  # Above 1.0

    def test_distance_validation(self):
        """Test camera distance validation."""
        with pytest.raises(ValidationError):
            IRSurveyConfig(distance_ft=0)

        with pytest.raises(ValidationError):
            IRSurveyConfig(distance_ft=150)  # Above 100

    def test_custom_ir_config(self):
        """Test custom IR configuration."""
        config = IRSurveyConfig(
            camera_model="FLIR T1040",
            emissivity_default=0.90,
            hot_spot_threshold_delta_f=20.0,
        )

        assert config.camera_model == "FLIR T1040"
        assert config.emissivity_default == 0.90
        assert config.hot_spot_threshold_delta_f == 20.0


# =============================================================================
# CONDENSATION CONFIG TESTS
# =============================================================================

class TestCondensationConfig:
    """Tests for CondensationConfig."""

    def test_default_values(self, default_condensation_config):
        """Test default condensation configuration values."""
        config = default_condensation_config

        assert config.design_ambient_temp_f == 95.0
        assert config.design_relative_humidity_pct == 90.0
        assert config.design_dew_point_margin_f == 5.0
        assert config.vapor_barrier_required is True
        assert config.vapor_barrier_perm_rating == 0.02
        assert config.cold_service_threshold_f == 60.0
        assert config.cryogenic_threshold_f == -100.0

    def test_humidity_validation(self):
        """Test humidity percentage validation."""
        with pytest.raises(ValidationError):
            CondensationConfig(design_relative_humidity_pct=-5.0)

        with pytest.raises(ValidationError):
            CondensationConfig(design_relative_humidity_pct=110.0)

    def test_custom_condensation_config(self):
        """Test custom condensation configuration."""
        config = CondensationConfig(
            design_ambient_temp_f=85.0,
            design_relative_humidity_pct=80.0,
            cryogenic_threshold_f=-200.0,
        )

        assert config.design_ambient_temp_f == 85.0
        assert config.design_relative_humidity_pct == 80.0
        assert config.cryogenic_threshold_f == -200.0


# =============================================================================
# INSULATION ANALYSIS CONFIG TESTS
# =============================================================================

class TestInsulationAnalysisConfig:
    """Tests for InsulationAnalysisConfig."""

    def test_required_facility_id(self):
        """Test facility_id is required."""
        with pytest.raises(ValidationError):
            InsulationAnalysisConfig()

    def test_default_values(self, full_analysis_config):
        """Test default analysis configuration values."""
        config = full_analysis_config

        assert config.facility_id == "TEST-FACILITY-001"
        assert config.config_name == "Default Insulation Analysis Config"
        assert config.version == "1.0.0"
        assert config.temperature_unit == TemperatureUnit.FAHRENHEIT.value
        assert config.length_unit == LengthUnit.INCHES.value
        assert config.default_ambient_temp_f == 77.0
        assert config.default_wind_speed_mph == 0.0
        assert config.convergence_tolerance == 0.001
        assert config.max_iterations == 100
        assert config.include_radiation is True
        assert config.include_convection is True
        assert config.audit_enabled is True
        assert config.provenance_tracking is True

    def test_sub_configurations(self, full_analysis_config):
        """Test sub-configuration objects are created."""
        config = full_analysis_config

        assert isinstance(config.economic, EconomicConfig)
        assert isinstance(config.safety, SafetyConfig)
        assert isinstance(config.ir_survey, IRSurveyConfig)
        assert isinstance(config.condensation, CondensationConfig)

    def test_safety_temperature_consistency_validation(self):
        """Test safety temperature consistency validation."""
        # The validator should auto-correct F/C mismatch
        config = InsulationAnalysisConfig(
            facility_id="TEST",
            safety=SafetyConfig(
                max_touch_temperature_c=60.0,
                max_touch_temperature_f=140.0,  # Correct conversion
            )
        )

        # Should be consistent (60C = 140F)
        expected_f = config.safety.max_touch_temperature_c * 9/5 + 32
        assert abs(config.safety.max_touch_temperature_f - expected_f) <= 1.0

    def test_config_id_auto_generated(self):
        """Test config_id is auto-generated as UUID."""
        config = InsulationAnalysisConfig(facility_id="TEST")

        assert config.config_id is not None
        assert len(config.config_id) == 36  # UUID format

    def test_created_at_timestamp(self):
        """Test created_at timestamp is set."""
        config = InsulationAnalysisConfig(facility_id="TEST")

        assert config.created_at is not None
        assert isinstance(config.created_at, datetime)
        assert config.created_at.tzinfo is not None  # Should be timezone-aware

    def test_convergence_tolerance_validation(self):
        """Test convergence tolerance range."""
        with pytest.raises(ValidationError):
            InsulationAnalysisConfig(
                facility_id="TEST",
                convergence_tolerance=0.00001  # Below 0.0001
            )

        with pytest.raises(ValidationError):
            InsulationAnalysisConfig(
                facility_id="TEST",
                convergence_tolerance=0.5  # Above 0.1
            )

    def test_max_iterations_validation(self):
        """Test max iterations range."""
        with pytest.raises(ValidationError):
            InsulationAnalysisConfig(
                facility_id="TEST",
                max_iterations=5  # Below 10
            )

        with pytest.raises(ValidationError):
            InsulationAnalysisConfig(
                facility_id="TEST",
                max_iterations=5000  # Above 1000
            )

    def test_custom_sub_configurations(self):
        """Test custom sub-configurations are applied."""
        config = InsulationAnalysisConfig(
            facility_id="TEST",
            economic=EconomicConfig(
                energy_cost_per_mmbtu=15.00,
                operating_hours_per_year=7500,
            ),
            safety=SafetyConfig(
                max_touch_temperature_c=55.0,
                sil_level=3,
            ),
        )

        assert config.economic.energy_cost_per_mmbtu == 15.00
        assert config.economic.operating_hours_per_year == 7500
        assert config.safety.max_touch_temperature_c == 55.0
        assert config.safety.sil_level == 3

    def test_metadata_field(self):
        """Test metadata field accepts custom data."""
        config = InsulationAnalysisConfig(
            facility_id="TEST",
            metadata={
                "project": "Phase 1",
                "auditor": "John Doe",
                "custom_field": 123,
            }
        )

        assert config.metadata["project"] == "Phase 1"
        assert config.metadata["auditor"] == "John Doe"
        assert config.metadata["custom_field"] == 123


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestEnums:
    """Tests for configuration enums."""

    def test_temperature_unit_values(self):
        """Test temperature unit enum values."""
        assert TemperatureUnit.FAHRENHEIT.value == "F"
        assert TemperatureUnit.CELSIUS.value == "C"
        assert TemperatureUnit.KELVIN.value == "K"

    def test_length_unit_values(self):
        """Test length unit enum values."""
        assert LengthUnit.INCHES.value == "in"
        assert LengthUnit.FEET.value == "ft"
        assert LengthUnit.METERS.value == "m"
        assert LengthUnit.MILLIMETERS.value == "mm"

    def test_currency_code_values(self):
        """Test currency code enum values."""
        assert CurrencyCode.USD.value == "USD"
        assert CurrencyCode.EUR.value == "EUR"
        assert CurrencyCode.GBP.value == "GBP"
        assert CurrencyCode.CAD.value == "CAD"
