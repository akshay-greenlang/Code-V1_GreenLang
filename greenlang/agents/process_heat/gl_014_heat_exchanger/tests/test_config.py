# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Configuration Tests

Comprehensive tests for all configuration classes including validation,
defaults, and TEMA type verification.

Coverage Target: 90%+
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    HeatExchangerConfig,
    TubeGeometryConfig,
    ShellGeometryConfig,
    PlateGeometryConfig,
    AirCooledGeometryConfig,
    FoulingConfig,
    CleaningConfig,
    TubeIntegrityConfig,
    OperatingLimitsConfig,
    EconomicsConfig,
    MLConfig,
    TEMAFoulingFactors,
    ExchangerType,
    TEMAFrontEnd,
    TEMAShell,
    TEMARearEnd,
    TEMAClass,
    FlowArrangement,
    FoulingCategory,
    CleaningMethod,
    TubeLayout,
    TubeMaterial,
    AlertSeverity,
    FailureMode,
)


class TestExchangerTypeEnum:
    """Tests for ExchangerType enumeration."""

    def test_all_exchanger_types_defined(self):
        """Verify all expected exchanger types exist."""
        expected_types = [
            "shell_tube", "plate", "plate_fin", "air_cooled",
            "double_pipe", "spiral", "scraped_surface", "reboiler", "condenser"
        ]
        for etype in expected_types:
            assert hasattr(ExchangerType, etype.upper())

    def test_exchanger_type_values(self):
        """Test exchanger type string values."""
        assert ExchangerType.SHELL_TUBE.value == "shell_tube"
        assert ExchangerType.PLATE.value == "plate"
        assert ExchangerType.AIR_COOLED.value == "air_cooled"


class TestTEMAEnums:
    """Tests for TEMA classification enumerations."""

    def test_tema_front_end_types(self):
        """Test TEMA front end head types."""
        assert TEMAFrontEnd.A.value == "A"  # Channel and removable cover
        assert TEMAFrontEnd.B.value == "B"  # Bonnet
        assert TEMAFrontEnd.N.value == "N"  # Non-removable cover

    def test_tema_shell_types(self):
        """Test TEMA shell types."""
        assert TEMAShell.E.value == "E"  # One pass shell
        assert TEMAShell.F.value == "F"  # Two pass with longitudinal baffle
        assert TEMAShell.K.value == "K"  # Kettle reboiler
        assert TEMAShell.X.value == "X"  # Cross flow

    def test_tema_rear_end_types(self):
        """Test TEMA rear end head types."""
        assert TEMARearEnd.S.value == "S"  # Floating head with backing device
        assert TEMARearEnd.U.value == "U"  # U-tube bundle
        assert TEMARearEnd.T.value == "T"  # Pull through floating head

    def test_tema_class(self):
        """Test TEMA mechanical standards classes."""
        assert TEMAClass.R.value == "R"  # Severe requirements
        assert TEMAClass.C.value == "C"  # Moderate requirements
        assert TEMAClass.B.value == "B"  # Chemical service


class TestTubeGeometryConfig:
    """Tests for TubeGeometryConfig."""

    def test_default_values(self):
        """Test default tube geometry values."""
        config = TubeGeometryConfig()
        assert config.outer_diameter_mm == 25.4
        assert config.wall_thickness_mm == 2.11
        assert config.tube_length_m == 6.096
        assert config.tube_count == 100
        assert config.tube_passes == 2

    def test_inner_diameter_calculation(self):
        """Test tube inner diameter property calculation."""
        config = TubeGeometryConfig(
            outer_diameter_mm=25.4,
            wall_thickness_mm=2.11,
        )
        expected_id = 25.4 - 2 * 2.11
        assert config.inner_diameter_mm == pytest.approx(expected_id, rel=1e-6)

    def test_tube_area_calculation(self):
        """Test total tube heat transfer area calculation."""
        import math
        config = TubeGeometryConfig(
            outer_diameter_mm=25.4,
            tube_length_m=6.096,
            tube_count=100,
        )
        expected_area = math.pi * 0.0254 * 6.096 * 100
        assert config.tube_area_m2 == pytest.approx(expected_area, rel=1e-6)

    def test_pitch_ratio_calculation(self):
        """Test pitch to diameter ratio calculation."""
        config = TubeGeometryConfig(
            outer_diameter_mm=25.4,
            tube_pitch_mm=31.75,
        )
        expected_ratio = 31.75 / 25.4
        assert config.pitch_ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_validation_positive_values(self):
        """Test validation requires positive values."""
        with pytest.raises(PydanticValidationError):
            TubeGeometryConfig(outer_diameter_mm=-25.4)

        with pytest.raises(PydanticValidationError):
            TubeGeometryConfig(tube_count=0)

    def test_tube_passes_limits(self):
        """Test tube passes must be between 1 and 16."""
        with pytest.raises(PydanticValidationError):
            TubeGeometryConfig(tube_passes=0)

        with pytest.raises(PydanticValidationError):
            TubeGeometryConfig(tube_passes=17)

    @pytest.mark.parametrize("layout", list(TubeLayout))
    def test_all_tube_layouts(self, layout):
        """Test all tube layout patterns are valid."""
        config = TubeGeometryConfig(tube_layout=layout)
        assert config.tube_layout == layout

    @pytest.mark.parametrize("material", list(TubeMaterial))
    def test_all_tube_materials(self, material):
        """Test all tube materials are valid."""
        config = TubeGeometryConfig(tube_material=material)
        assert config.tube_material == material


class TestShellGeometryConfig:
    """Tests for ShellGeometryConfig."""

    def test_default_values(self):
        """Test default shell geometry values."""
        config = ShellGeometryConfig()
        assert config.inner_diameter_mm == 610.0
        assert config.shell_passes == 1
        assert config.baffle_cut_percent == 25.0
        assert config.baffle_spacing_mm == 300.0

    def test_baffle_cut_limits(self):
        """Test baffle cut percentage must be 15-45%."""
        # Valid range
        config = ShellGeometryConfig(baffle_cut_percent=25.0)
        assert config.baffle_cut_percent == 25.0

        # Too low
        with pytest.raises(PydanticValidationError):
            ShellGeometryConfig(baffle_cut_percent=10.0)

        # Too high
        with pytest.raises(PydanticValidationError):
            ShellGeometryConfig(baffle_cut_percent=50.0)

    def test_shell_passes_limits(self):
        """Test shell passes must be between 1 and 8."""
        config = ShellGeometryConfig(shell_passes=4)
        assert config.shell_passes == 4

        with pytest.raises(PydanticValidationError):
            ShellGeometryConfig(shell_passes=0)


class TestPlateGeometryConfig:
    """Tests for PlateGeometryConfig."""

    def test_default_values(self):
        """Test default plate geometry values."""
        config = PlateGeometryConfig()
        assert config.plate_count == 50
        assert config.chevron_angle_deg == 60.0

    def test_heat_transfer_area_calculation(self):
        """Test plate heat transfer area calculation."""
        config = PlateGeometryConfig(
            plate_count=50,
            plate_length_mm=1000.0,
            plate_width_mm=400.0,
        )
        # Area = (N-2) * L * W
        expected_area = 48 * 1.0 * 0.4
        assert config.heat_transfer_area_m2 == pytest.approx(expected_area, rel=1e-6)

    def test_chevron_angle_limits(self):
        """Test chevron angle must be 25-70 degrees."""
        with pytest.raises(PydanticValidationError):
            PlateGeometryConfig(chevron_angle_deg=20.0)

        with pytest.raises(PydanticValidationError):
            PlateGeometryConfig(chevron_angle_deg=75.0)

    def test_minimum_plate_count(self):
        """Test minimum plate count is 3."""
        with pytest.raises(PydanticValidationError):
            PlateGeometryConfig(plate_count=2)


class TestFoulingConfig:
    """Tests for FoulingConfig."""

    def test_default_fouling_values(self):
        """Test default TEMA fouling resistance values."""
        config = FoulingConfig()
        assert config.shell_side_fouling_m2kw == 0.00017
        assert config.tube_side_fouling_m2kw == 0.00017

    def test_design_fouling_factor_limits(self):
        """Test design fouling factor must be 1.0-2.0."""
        config = FoulingConfig(design_fouling_factor=1.5)
        assert config.design_fouling_factor == 1.5

        with pytest.raises(PydanticValidationError):
            FoulingConfig(design_fouling_factor=0.5)

        with pytest.raises(PydanticValidationError):
            FoulingConfig(design_fouling_factor=2.5)

    @pytest.mark.parametrize("category", list(FoulingCategory))
    def test_all_fouling_categories(self, category):
        """Test all fouling categories are valid."""
        config = FoulingConfig(fouling_category=category)
        assert config.fouling_category == category


class TestCleaningConfig:
    """Tests for CleaningConfig."""

    def test_default_cleaning_values(self):
        """Test default cleaning configuration values."""
        config = CleaningConfig()
        assert config.minimum_interval_days == 30
        assert config.maximum_interval_days == 365
        assert config.effectiveness_threshold == 0.70

    def test_effectiveness_threshold_limits(self):
        """Test effectiveness threshold must be 0.5-1.0."""
        with pytest.raises(PydanticValidationError):
            CleaningConfig(effectiveness_threshold=0.4)

    @pytest.mark.parametrize("method", list(CleaningMethod))
    def test_all_cleaning_methods(self, method):
        """Test all cleaning methods are valid."""
        config = CleaningConfig(preferred_methods=[method])
        assert method in config.preferred_methods


class TestTubeIntegrityConfig:
    """Tests for TubeIntegrityConfig."""

    def test_default_integrity_values(self):
        """Test default tube integrity values."""
        config = TubeIntegrityConfig()
        assert config.design_life_years == 20.0
        assert config.minimum_wall_thickness_mm == 1.25
        assert config.expected_corrosion_rate_mm_year == 0.1

    def test_inspection_interval_limits(self):
        """Test inspection interval must be 6-120 months."""
        config = TubeIntegrityConfig(inspection_interval_months=24)
        assert config.inspection_interval_months == 24

        with pytest.raises(PydanticValidationError):
            TubeIntegrityConfig(inspection_interval_months=3)

        with pytest.raises(PydanticValidationError):
            TubeIntegrityConfig(inspection_interval_months=150)


class TestOperatingLimitsConfig:
    """Tests for OperatingLimitsConfig."""

    def test_default_operating_limits(self):
        """Test default operating limit values."""
        config = OperatingLimitsConfig()
        assert config.max_shell_inlet_temp_c == 300.0
        assert config.max_tube_velocity_m_s == 3.0
        assert config.min_tube_velocity_m_s == 0.5

    def test_effectiveness_alarm_threshold(self):
        """Test effectiveness alarm threshold."""
        config = OperatingLimitsConfig(alarm_effectiveness=0.65)
        assert config.alarm_effectiveness == 0.65


class TestEconomicsConfig:
    """Tests for EconomicsConfig."""

    def test_default_economics_values(self):
        """Test default economics configuration."""
        config = EconomicsConfig()
        assert config.energy_cost_usd_per_kwh == 0.10
        assert config.discount_rate == 0.10
        assert config.replacement_cost_usd == 500000.0

    def test_discount_rate_limits(self):
        """Test discount rate must be 0-30%."""
        config = EconomicsConfig(discount_rate=0.15)
        assert config.discount_rate == 0.15

        with pytest.raises(PydanticValidationError):
            EconomicsConfig(discount_rate=0.35)


class TestMLConfig:
    """Tests for MLConfig."""

    def test_default_ml_config(self):
        """Test default ML configuration."""
        config = MLConfig()
        assert config.enabled == True
        assert config.fouling_prediction_enabled == True
        assert config.confidence_threshold == 0.80

    def test_confidence_threshold_limits(self):
        """Test confidence threshold must be 0.5-0.99."""
        with pytest.raises(PydanticValidationError):
            MLConfig(confidence_threshold=0.4)

        with pytest.raises(PydanticValidationError):
            MLConfig(confidence_threshold=1.0)


class TestTEMAFoulingFactors:
    """Tests for TEMAFoulingFactors (TEMA RGP-T2.4)."""

    def test_tema_water_fouling_factors(self):
        """Test TEMA water service fouling factors."""
        factors = TEMAFoulingFactors()
        assert factors.cooling_tower_water == 0.00035
        assert factors.sea_water == 0.00017
        assert factors.boiler_feedwater == 0.00009
        assert factors.river_water == 0.00035

    def test_tema_hydrocarbon_fouling_factors(self):
        """Test TEMA hydrocarbon service fouling factors."""
        factors = TEMAFoulingFactors()
        assert factors.fuel_oil == 0.00088
        assert factors.crude_oil_dry == 0.00035
        assert factors.crude_oil_wet == 0.00053
        assert factors.gasoline == 0.00018
        assert factors.naphtha == 0.00018

    def test_tema_process_fouling_factors(self):
        """Test TEMA process stream fouling factors."""
        factors = TEMAFoulingFactors()
        assert factors.steam == 0.00009
        assert factors.process_gas == 0.00018
        assert factors.organic_solvents == 0.00018


class TestHeatExchangerConfig:
    """Tests for main HeatExchangerConfig."""

    def test_tema_type_validation_valid(self):
        """Test TEMA type validation with valid types."""
        valid_types = ["AES", "BEM", "AEU", "AKT", "AJW"]
        for tema_type in valid_types:
            config = HeatExchangerConfig(
                exchanger_id="E-001",
                exchanger_type=ExchangerType.SHELL_TUBE,
                tema_type=tema_type,
            )
            assert config.tema_type == tema_type

    def test_tema_type_validation_invalid_front_end(self):
        """Test TEMA type validation rejects invalid front end."""
        with pytest.raises(PydanticValidationError):
            HeatExchangerConfig(
                exchanger_id="E-001",
                exchanger_type=ExchangerType.SHELL_TUBE,
                tema_type="XES",  # X is not a valid front end
            )

    def test_tema_type_validation_invalid_shell(self):
        """Test TEMA type validation rejects invalid shell."""
        with pytest.raises(PydanticValidationError):
            HeatExchangerConfig(
                exchanger_id="E-001",
                exchanger_type=ExchangerType.SHELL_TUBE,
                tema_type="AAS",  # A is not a valid shell type
            )

    def test_tema_type_validation_invalid_rear_end(self):
        """Test TEMA type validation rejects invalid rear end."""
        with pytest.raises(PydanticValidationError):
            HeatExchangerConfig(
                exchanger_id="E-001",
                exchanger_type=ExchangerType.SHELL_TUBE,
                tema_type="AEA",  # A is not a valid rear end
            )

    def test_tema_type_uppercase_conversion(self):
        """Test TEMA type is converted to uppercase."""
        config = HeatExchangerConfig(
            exchanger_id="E-001",
            exchanger_type=ExchangerType.SHELL_TUBE,
            tema_type="aes",
        )
        assert config.tema_type == "AES"

    def test_shell_tube_auto_geometry(self):
        """Test shell-tube exchanger creates default geometry."""
        config = HeatExchangerConfig(
            exchanger_id="E-001",
            exchanger_type=ExchangerType.SHELL_TUBE,
        )
        assert config.tube_geometry is not None
        assert config.shell_geometry is not None

    def test_plate_exchanger_auto_geometry(self):
        """Test plate exchanger creates default plate geometry."""
        config = HeatExchangerConfig(
            exchanger_id="PHE-001",
            exchanger_type=ExchangerType.PLATE,
        )
        assert config.plate_geometry is not None

    def test_air_cooled_auto_geometry(self):
        """Test air-cooled exchanger creates default geometry."""
        config = HeatExchangerConfig(
            exchanger_id="AC-001",
            exchanger_type=ExchangerType.AIR_COOLED,
        )
        assert config.air_cooled_geometry is not None

    def test_complete_config_creation(self, shell_tube_config):
        """Test complete shell-tube configuration creation."""
        config = shell_tube_config
        assert config.exchanger_id == "E-1001"
        assert config.exchanger_type == ExchangerType.SHELL_TUBE
        assert config.tema_type == "AES"
        assert config.design_duty_kw == 1000.0
        assert config.design_u_w_m2k == 500.0

    def test_config_json_serialization(self, shell_tube_config):
        """Test configuration can be serialized to JSON."""
        config = shell_tube_config
        json_str = config.json()
        assert "E-1001" in json_str
        assert "shell_tube" in json_str

    def test_config_dict_conversion(self, shell_tube_config):
        """Test configuration can be converted to dict."""
        config = shell_tube_config
        config_dict = config.dict()
        assert config_dict["exchanger_id"] == "E-1001"
        assert config_dict["exchanger_type"] == "shell_tube"


class TestEnumValues:
    """Tests for enum string values."""

    def test_flow_arrangement_values(self):
        """Test flow arrangement enum values."""
        assert FlowArrangement.COUNTER_FLOW.value == "counter_flow"
        assert FlowArrangement.PARALLEL_FLOW.value == "parallel_flow"
        assert FlowArrangement.CROSS_FLOW_MIXED.value == "cross_flow_mixed"

    def test_alert_severity_values(self):
        """Test alert severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ALARM.value == "alarm"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_failure_mode_values(self):
        """Test failure mode enum values."""
        assert FailureMode.TUBE_LEAK.value == "tube_leak"
        assert FailureMode.TUBE_RUPTURE.value == "tube_rupture"
        assert FailureMode.FOULING_CRITICAL.value == "fouling_critical"
