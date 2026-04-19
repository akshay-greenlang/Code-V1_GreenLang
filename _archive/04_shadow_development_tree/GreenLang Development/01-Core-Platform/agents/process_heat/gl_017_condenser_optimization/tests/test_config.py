"""
GL-017 CONDENSYNC Agent - Configuration Tests

Unit tests for all configuration schemas and validation logic.
Tests cover CondenserOptimizationConfig and all sub-configurations.

Coverage targets:
    - All Pydantic validators
    - Field constraints (min/max/default)
    - Enum conversions
    - Nested configuration handling
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CondenserOptimizationConfig,
    CoolingTowerConfig,
    TubeFoulingConfig,
    VacuumSystemConfig,
    AirIngresConfig,
    CleanlinessConfig,
    PerformanceConfig,
    CondenserType,
    TubeMaterial,
    CoolingWaterSource,
    VacuumEquipmentType,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default condenser configuration."""
    return CondenserOptimizationConfig(condenser_id="TEST-C-001")


@pytest.fixture
def cooling_tower_config():
    """Create default cooling tower configuration."""
    return CoolingTowerConfig()


@pytest.fixture
def tube_fouling_config():
    """Create default tube fouling configuration."""
    return TubeFoulingConfig()


@pytest.fixture
def vacuum_system_config():
    """Create default vacuum system configuration."""
    return VacuumSystemConfig()


@pytest.fixture
def air_ingress_config():
    """Create default air ingress configuration."""
    return AirIngresConfig()


@pytest.fixture
def cleanliness_config():
    """Create default cleanliness configuration."""
    return CleanlinessConfig()


@pytest.fixture
def performance_config():
    """Create default performance configuration."""
    return PerformanceConfig()


# =============================================================================
# ENUMS TESTS
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_condenser_type_values(self):
        """Test CondenserType enum has all expected values."""
        expected = {
            "single_pass", "two_pass", "divided_waterbox",
            "single_shell", "multi_shell", "air_cooled"
        }
        actual = {e.value for e in CondenserType}
        assert actual == expected

    def test_tube_material_values(self):
        """Test TubeMaterial enum has all expected values."""
        expected = {
            "admiralty_brass", "aluminum_brass", "copper_nickel_90_10",
            "copper_nickel_70_30", "stainless_304", "stainless_316",
            "titanium", "duplex_2205"
        }
        actual = {e.value for e in TubeMaterial}
        assert actual == expected

    def test_cooling_water_source_values(self):
        """Test CoolingWaterSource enum values."""
        expected = {
            "once_through_fresh", "once_through_seawater",
            "cooling_tower_mechanical", "cooling_tower_natural",
            "hybrid_cooling", "dry_cooling"
        }
        actual = {e.value for e in CoolingWaterSource}
        assert actual == expected

    def test_vacuum_equipment_type_values(self):
        """Test VacuumEquipmentType enum values."""
        expected = {
            "steam_jet_ejector", "liquid_ring_pump",
            "dry_vacuum_pump", "hybrid_system"
        }
        actual = {e.value for e in VacuumEquipmentType}
        assert actual == expected


# =============================================================================
# COOLING TOWER CONFIG TESTS
# =============================================================================

class TestCoolingTowerConfig:
    """Test CoolingTowerConfig validation."""

    def test_default_values(self, cooling_tower_config):
        """Test default configuration values."""
        assert cooling_tower_config.tower_type == "mechanical_draft"
        assert cooling_tower_config.design_capacity_gpm == 50000.0
        assert cooling_tower_config.design_range_f == 15.0
        assert cooling_tower_config.design_approach_f == 8.0
        assert cooling_tower_config.target_cycles_concentration == 5.0

    def test_cycles_validation(self):
        """Test cycles of concentration validation."""
        # Valid range
        config = CoolingTowerConfig(target_cycles_concentration=6.0)
        assert config.target_cycles_concentration == 6.0

        # Below minimum
        with pytest.raises(ValidationError):
            CoolingTowerConfig(target_cycles_concentration=1.0)

        # Above maximum
        with pytest.raises(ValidationError):
            CoolingTowerConfig(target_cycles_concentration=15.0)

    def test_chemistry_limits(self, cooling_tower_config):
        """Test chemistry limit defaults."""
        assert cooling_tower_config.max_silica_ppm == 150.0
        assert cooling_tower_config.max_calcium_ppm == 800.0
        assert cooling_tower_config.max_chlorides_ppm == 500.0
        assert cooling_tower_config.max_conductivity_umhos == 3000.0
        assert cooling_tower_config.target_ph == 8.0

    def test_design_range_validation(self):
        """Test design range constraints."""
        # Valid
        config = CoolingTowerConfig(design_range_f=20.0)
        assert config.design_range_f == 20.0

        # Invalid - exceeds max
        with pytest.raises(ValidationError):
            CoolingTowerConfig(design_range_f=50.0)

    @pytest.mark.parametrize("approach,expected_valid", [
        (5.0, True),
        (15.0, True),
        (0.0, False),
        (25.0, False),
    ])
    def test_design_approach_validation(self, approach, expected_valid):
        """Test design approach validation."""
        if expected_valid:
            config = CoolingTowerConfig(design_approach_f=approach)
            assert config.design_approach_f == approach
        else:
            with pytest.raises(ValidationError):
                CoolingTowerConfig(design_approach_f=approach)


# =============================================================================
# TUBE FOULING CONFIG TESTS
# =============================================================================

class TestTubeFoulingConfig:
    """Test TubeFoulingConfig validation."""

    def test_default_values(self, tube_fouling_config):
        """Test default configuration values."""
        assert tube_fouling_config.design_cleanliness_factor == 0.85
        assert tube_fouling_config.tube_material == TubeMaterial.STAINLESS_316
        assert tube_fouling_config.tube_od_in == 0.875
        assert tube_fouling_config.tube_gauge == 18
        assert tube_fouling_config.tube_count == 15000

    def test_cleanliness_thresholds(self, tube_fouling_config):
        """Test cleanliness threshold ordering."""
        assert tube_fouling_config.cleanliness_warning_threshold > tube_fouling_config.cleanliness_alarm_threshold
        assert tube_fouling_config.cleanliness_alarm_threshold > tube_fouling_config.cleaning_trigger_threshold

    def test_backpressure_thresholds(self, tube_fouling_config):
        """Test backpressure deviation thresholds."""
        assert tube_fouling_config.backpressure_deviation_warning_inhg < tube_fouling_config.backpressure_deviation_alarm_inhg

    @pytest.mark.parametrize("cleanliness,valid", [
        (0.85, True),
        (0.5, True),
        (1.0, True),
        (0.4, False),
        (1.1, False),
    ])
    def test_design_cleanliness_validation(self, cleanliness, valid):
        """Test design cleanliness factor validation."""
        if valid:
            config = TubeFoulingConfig(design_cleanliness_factor=cleanliness)
            assert config.design_cleanliness_factor == cleanliness
        else:
            with pytest.raises(ValidationError):
                TubeFoulingConfig(design_cleanliness_factor=cleanliness)

    def test_tube_gauge_validation(self):
        """Test tube gauge (BWG) validation."""
        # Valid gauges
        for gauge in [14, 16, 18, 20, 22, 24]:
            config = TubeFoulingConfig(tube_gauge=gauge)
            assert config.tube_gauge == gauge

        # Invalid gauges
        with pytest.raises(ValidationError):
            TubeFoulingConfig(tube_gauge=10)

        with pytest.raises(ValidationError):
            TubeFoulingConfig(tube_gauge=30)


# =============================================================================
# VACUUM SYSTEM CONFIG TESTS
# =============================================================================

class TestVacuumSystemConfig:
    """Test VacuumSystemConfig validation."""

    def test_default_values(self, vacuum_system_config):
        """Test default configuration values."""
        assert vacuum_system_config.primary_equipment == VacuumEquipmentType.STEAM_JET_EJECTOR
        assert vacuum_system_config.ejector_stages == 2
        assert vacuum_system_config.design_vacuum_inhga == 1.5
        assert vacuum_system_config.motive_steam_pressure_psig == 150.0

    def test_vacuum_limits(self, vacuum_system_config):
        """Test vacuum limit relationships."""
        # Min vacuum (worst) > design vacuum > max vacuum (best)
        assert vacuum_system_config.min_vacuum_inhga > vacuum_system_config.design_vacuum_inhga
        assert vacuum_system_config.design_vacuum_inhga > vacuum_system_config.max_vacuum_inhga

    def test_ejector_stages_validation(self):
        """Test ejector stages validation."""
        for stages in [1, 2, 3, 4]:
            config = VacuumSystemConfig(ejector_stages=stages)
            assert config.ejector_stages == stages

        with pytest.raises(ValidationError):
            VacuumSystemConfig(ejector_stages=0)

        with pytest.raises(ValidationError):
            VacuumSystemConfig(ejector_stages=5)

    def test_backup_equipment_optional(self):
        """Test backup equipment is optional."""
        config = VacuumSystemConfig()
        assert config.backup_equipment is None

        config = VacuumSystemConfig(backup_equipment=VacuumEquipmentType.LIQUID_RING_PUMP)
        assert config.backup_equipment == VacuumEquipmentType.LIQUID_RING_PUMP


# =============================================================================
# AIR INGRESS CONFIG TESTS
# =============================================================================

class TestAirIngresConfig:
    """Test AirIngresConfig validation."""

    def test_default_values(self, air_ingress_config):
        """Test default configuration values."""
        assert air_ingress_config.max_air_ingress_scfm == 10.0
        assert air_ingress_config.warning_air_ingress_scfm == 5.0
        assert air_ingress_config.dissolved_oxygen_monitoring is True
        assert air_ingress_config.tracer_gas_testing is True

    def test_air_ingress_thresholds(self, air_ingress_config):
        """Test air ingress threshold relationships."""
        assert air_ingress_config.warning_air_ingress_scfm < air_ingress_config.max_air_ingress_scfm

    def test_do_thresholds(self, air_ingress_config):
        """Test dissolved oxygen threshold relationships."""
        assert air_ingress_config.do_warning_ppb < air_ingress_config.do_alarm_ppb

    def test_subcooling_thresholds(self, air_ingress_config):
        """Test subcooling threshold relationships."""
        assert air_ingress_config.subcooling_warning_f < air_ingress_config.subcooling_alarm_f


# =============================================================================
# CLEANLINESS CONFIG TESTS
# =============================================================================

class TestCleanlinessConfig:
    """Test CleanlinessConfig validation."""

    def test_default_values(self, cleanliness_config):
        """Test default configuration values."""
        assert cleanliness_config.hei_edition == "12th"
        assert cleanliness_config.heat_transfer_coefficient_method == "hei_standard"
        assert cleanliness_config.reference_inlet_temp_f == 70.0
        assert cleanliness_config.reference_velocity_fps == 7.0

    def test_correction_factors(self, cleanliness_config):
        """Test correction factor defaults."""
        assert cleanliness_config.tube_material_factor == 1.0
        assert cleanliness_config.inlet_water_factor == 0.85
        assert cleanliness_config.include_velocity_correction is True
        assert cleanliness_config.include_temperature_correction is True

    @pytest.mark.parametrize("velocity,valid", [
        (7.0, True),
        (3.0, True),
        (12.0, True),
        (2.0, False),
        (15.0, False),
    ])
    def test_reference_velocity_validation(self, velocity, valid):
        """Test reference velocity validation."""
        if valid:
            config = CleanlinessConfig(reference_velocity_fps=velocity)
            assert config.reference_velocity_fps == velocity
        else:
            with pytest.raises(ValidationError):
                CleanlinessConfig(reference_velocity_fps=velocity)


# =============================================================================
# PERFORMANCE CONFIG TESTS
# =============================================================================

class TestPerformanceConfig:
    """Test PerformanceConfig validation."""

    def test_default_values(self, performance_config):
        """Test default configuration values."""
        assert performance_config.design_duty_btu_hr == 500_000_000.0
        assert performance_config.design_steam_flow_lb_hr == 500000.0
        assert performance_config.design_backpressure_inhga == 1.5
        assert performance_config.design_inlet_temp_f == 70.0
        assert performance_config.design_outlet_temp_f == 95.0

    def test_deviation_thresholds(self, performance_config):
        """Test deviation threshold relationships."""
        assert performance_config.backpressure_deviation_warning_pct < performance_config.backpressure_deviation_alarm_pct

    def test_load_range(self, performance_config):
        """Test load range configuration."""
        assert performance_config.load_range_min_pct < performance_config.load_range_max_pct
        assert performance_config.curve_points >= 5


# =============================================================================
# MAIN CONFIG TESTS
# =============================================================================

class TestCondenserOptimizationConfig:
    """Test CondenserOptimizationConfig validation."""

    def test_required_condenser_id(self):
        """Test condenser_id is required."""
        with pytest.raises(ValidationError):
            CondenserOptimizationConfig()

    def test_default_name_from_id(self):
        """Test default name is set from condenser_id."""
        config = CondenserOptimizationConfig(condenser_id="C-001")
        assert config.name == "Condenser C-001"

    def test_explicit_name(self):
        """Test explicit name is preserved."""
        config = CondenserOptimizationConfig(
            condenser_id="C-001",
            name="Main Condenser"
        )
        assert config.name == "Main Condenser"

    def test_default_sub_configs(self, default_config):
        """Test default sub-configurations are created."""
        assert isinstance(default_config.cooling_tower, CoolingTowerConfig)
        assert isinstance(default_config.tube_fouling, TubeFoulingConfig)
        assert isinstance(default_config.vacuum_system, VacuumSystemConfig)
        assert isinstance(default_config.air_ingress, AirIngresConfig)
        assert isinstance(default_config.cleanliness, CleanlinessConfig)
        assert isinstance(default_config.performance, PerformanceConfig)

    def test_enum_values(self, default_config):
        """Test enum default values."""
        assert default_config.condenser_type == CondenserType.TWO_PASS
        assert default_config.cooling_source == CoolingWaterSource.COOLING_TOWER_MECHANICAL

    def test_safety_settings(self, default_config):
        """Test safety configuration defaults."""
        assert default_config.sil_level == 2
        assert default_config.low_vacuum_trip_inhga == 5.0
        assert default_config.high_hotwell_level_trip_pct == 90.0

    def test_optimization_settings(self, default_config):
        """Test optimization configuration defaults."""
        assert default_config.optimization_enabled is True
        assert default_config.optimization_interval_s == 300

    def test_data_collection_settings(self, default_config):
        """Test data collection configuration."""
        assert default_config.data_collection_interval_s >= 1
        assert default_config.data_collection_interval_s <= 60

    @pytest.mark.parametrize("sil_level,valid", [
        (1, True),
        (2, True),
        (3, True),
        (0, False),
        (4, False),
    ])
    def test_sil_level_validation(self, sil_level, valid):
        """Test SIL level validation."""
        if valid:
            config = CondenserOptimizationConfig(
                condenser_id="C-001",
                sil_level=sil_level
            )
            assert config.sil_level == sil_level
        else:
            with pytest.raises(ValidationError):
                CondenserOptimizationConfig(
                    condenser_id="C-001",
                    sil_level=sil_level
                )

    def test_custom_sub_config(self):
        """Test custom sub-configuration."""
        custom_cooling = CoolingTowerConfig(
            design_range_f=20.0,
            target_cycles_concentration=6.0
        )
        config = CondenserOptimizationConfig(
            condenser_id="C-001",
            cooling_tower=custom_cooling
        )
        assert config.cooling_tower.design_range_f == 20.0
        assert config.cooling_tower.target_cycles_concentration == 6.0

    def test_design_surface_area(self, default_config):
        """Test design surface area default."""
        assert default_config.design_surface_area_ft2 == 150000.0

    def test_shell_and_passes(self, default_config):
        """Test shell count and passes defaults."""
        assert default_config.shell_count == 1
        assert default_config.passes == 2

    def test_serialization(self, default_config):
        """Test config can be serialized to dict."""
        config_dict = default_config.dict()
        assert "condenser_id" in config_dict
        assert "cooling_tower" in config_dict
        assert "tube_fouling" in config_dict

    def test_json_serialization(self, default_config):
        """Test config can be serialized to JSON."""
        json_str = default_config.json()
        assert "TEST-C-001" in json_str


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_boundary_values_cooling_tower(self):
        """Test boundary values for cooling tower config."""
        # Minimum valid values
        config = CoolingTowerConfig(
            design_capacity_gpm=0.1,
            design_range_f=0.1,
            design_approach_f=0.1,
            target_cycles_concentration=1.5,
        )
        assert config.design_capacity_gpm == 0.1

    def test_boundary_values_vacuum(self):
        """Test boundary values for vacuum config."""
        config = VacuumSystemConfig(
            design_vacuum_inhga=0.5,
            ejector_stages=1,
            air_removal_capacity_scfm=0.1,
        )
        assert config.design_vacuum_inhga == 0.5

    def test_whitespace_condenser_id(self):
        """Test condenser_id with whitespace."""
        config = CondenserOptimizationConfig(condenser_id="  C-001  ")
        assert config.condenser_id == "  C-001  "  # Not stripped by default

    def test_empty_historian_prefix(self):
        """Test empty historian tag prefix."""
        config = CondenserOptimizationConfig(
            condenser_id="C-001",
            historian_tag_prefix=""
        )
        assert config.historian_tag_prefix == ""
