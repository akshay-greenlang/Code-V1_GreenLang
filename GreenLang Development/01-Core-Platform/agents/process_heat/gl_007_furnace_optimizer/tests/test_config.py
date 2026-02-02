# -*- coding: utf-8 -*-
"""
GL-007 Configuration Tests
==========================

Unit tests for GL-007 FurnaceOptimizer/CoolingTowerOptimizer configuration.
Tests configuration schemas, validation, and defaults per NFPA 86, ASHRAE, CTI.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_007_furnace_optimizer.config import (
    GL007Config,
    FurnaceOptimizerConfig,
    CoolingTowerConfig,
    CombustionConfig,
    HeatTransferConfig,
    BurnerConfig,
    NFPA86Config,
    ASHRAEConfig,
    ExplainabilityConfig,
    ProvenanceConfig,
    FurnaceType,
    FuelType,
    CoolingTowerType,
    FillType,
    ControlMode,
    SafetyIntegrityLevel,
    create_default_config,
    create_high_efficiency_config,
)


class TestBurnerConfig:
    """Tests for burner configuration."""

    def test_default_burner(self, default_burner_config):
        """Test default burner configuration."""
        config = default_burner_config
        assert config.burner_id == "BNR-001"
        assert config.capacity_mmbtu_hr == 10.0

    def test_turndown_ratio_bounds(self):
        """Test turndown ratio bounds."""
        with pytest.raises(ValidationError):
            BurnerConfig(turndown_ratio=1.0)  # Must be >= 2

    def test_capacity_bounds(self):
        """Test capacity bounds."""
        with pytest.raises(ValidationError):
            BurnerConfig(capacity_mmbtu_hr=0)  # Must be > 0


class TestCombustionConfig:
    """Tests for combustion configuration."""

    def test_default_config(self, default_combustion_config):
        """Test default combustion configuration."""
        config = default_combustion_config
        assert config.fuel_type == FuelType.NATURAL_GAS
        assert config.target_o2_pct == 3.0

    def test_o2_target_bounds(self):
        """Test O2 target bounds."""
        with pytest.raises(ValidationError):
            CombustionConfig(target_o2_pct=0.5)  # Must be >= 1.0

    def test_excess_air_bounds(self):
        """Test excess air bounds."""
        with pytest.raises(ValidationError):
            CombustionConfig(target_excess_air_pct=2.0)  # Must be >= 5


class TestHeatTransferConfig:
    """Tests for heat transfer configuration."""

    def test_default_config(self, default_heat_transfer_config):
        """Test default heat transfer configuration."""
        config = default_heat_transfer_config
        assert config.radiant_surface_area_ft2 == 500.0
        assert config.convective_surface_area_ft2 == 1000.0

    def test_total_area_calculation(self, default_heat_transfer_config):
        """Test total surface area auto-calculation."""
        config = default_heat_transfer_config
        assert config.total_surface_area_ft2 == 1500.0

    def test_positive_areas(self):
        """Test positive surface area requirement."""
        with pytest.raises(ValidationError):
            HeatTransferConfig(radiant_surface_area_ft2=-100)


class TestFurnaceOptimizerConfig:
    """Tests for furnace optimizer configuration."""

    def test_required_furnace_id(self):
        """Test that furnace_id is required."""
        with pytest.raises(ValidationError):
            FurnaceOptimizerConfig()

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = FurnaceOptimizerConfig(furnace_id="FUR-001")
        assert config.furnace_id == "FUR-001"
        assert config.furnace_type == FurnaceType.DIRECT_FIRED

    def test_full_config(self, default_furnace_config):
        """Test complete configuration."""
        config = default_furnace_config
        assert config.design_temp_f == 1800.0
        assert config.design_efficiency_pct == 85.0

    def test_temperature_bounds(self):
        """Test temperature bounds."""
        with pytest.raises(ValidationError):
            FurnaceOptimizerConfig(
                furnace_id="FUR-001",
                design_temp_f=100.0,  # Must be >= 200
            )

    def test_tmt_trip_validation(self):
        """Test TMT trip > alarm validation."""
        with pytest.raises(ValidationError):
            FurnaceOptimizerConfig(
                furnace_id="FUR-001",
                tmt_alarm_f=1500.0,
                tmt_trip_f=1400.0,  # Must be >= alarm
            )

    def test_max_temp_validation(self):
        """Test max temp >= design temp validation."""
        with pytest.raises(ValidationError):
            FurnaceOptimizerConfig(
                furnace_id="FUR-001",
                design_temp_f=2000.0,
                max_operating_temp_f=1800.0,  # Must be >= design
            )


class TestCoolingTowerConfig:
    """Tests for cooling tower configuration."""

    def test_required_tower_id(self):
        """Test that tower_id is required."""
        with pytest.raises(ValidationError):
            CoolingTowerConfig()

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = CoolingTowerConfig(tower_id="CT-001")
        assert config.tower_id == "CT-001"
        assert config.tower_type == CoolingTowerType.MECHANICAL_INDUCED

    def test_full_config(self, default_cooling_tower_config):
        """Test complete configuration."""
        config = default_cooling_tower_config
        assert config.design_wet_bulb_f == 78.0
        assert config.design_range_f == 20.0

    def test_range_calculation(self):
        """Test range auto-calculation from temps."""
        config = CoolingTowerConfig(
            tower_id="CT-001",
            design_hot_water_temp_f=105.0,
            design_cold_water_temp_f=85.0,
        )
        # Range should be calculated as hot - cold
        assert config.design_range_f == 20.0

    def test_heat_rejection_tons_calculation(self):
        """Test heat rejection tons auto-calculation."""
        config = CoolingTowerConfig(
            tower_id="CT-001",
            design_heat_rejection_mmbtu_hr=25.0,
        )
        # 25 MMBtu/hr / 0.012 = ~2083 tons
        assert abs(config.design_heat_rejection_tons - 2083.33) < 1.0

    def test_wet_bulb_bounds(self):
        """Test wet bulb temperature bounds."""
        with pytest.raises(ValidationError):
            CoolingTowerConfig(
                tower_id="CT-001",
                design_wet_bulb_f=100.0,  # Must be <= 90
            )


class TestNFPA86Config:
    """Tests for NFPA 86 safety configuration."""

    def test_default_config(self, default_nfpa86_config):
        """Test default NFPA 86 configuration."""
        config = default_nfpa86_config
        assert config.nfpa_86_compliance is True
        assert config.min_purge_time_sec == 60

    def test_purge_time_minimum(self):
        """Test minimum purge time per NFPA 86."""
        with pytest.raises(ValidationError):
            NFPA86Config(min_purge_time_sec=10)  # Must be >= 15

    def test_flame_failure_response(self):
        """Test flame failure response time bounds."""
        with pytest.raises(ValidationError):
            NFPA86Config(max_flame_failure_response_sec=0.5)  # Must be >= 1


class TestASHRAEConfig:
    """Tests for ASHRAE configuration."""

    def test_default_config(self, default_ashrae_config):
        """Test default ASHRAE configuration."""
        config = default_ashrae_config
        assert config.ashrae_90_1_compliance is True
        assert config.climate_zone == "4A"

    def test_tower_efficiency_bounds(self):
        """Test tower efficiency bounds per ASHRAE 90.1."""
        with pytest.raises(ValidationError):
            ASHRAEConfig(min_tower_efficiency_gpm_hp=10.0)  # Must be >= 20


class TestGL007Config:
    """Tests for complete GL-007 configuration."""

    def test_default_creates_furnace(self):
        """Test that default creates furnace if neither specified."""
        config = GL007Config()
        assert config.furnace is not None
        assert config.furnace.furnace_id == "FUR-001"

    def test_full_config(self, gl007_config):
        """Test complete configuration."""
        config = gl007_config
        assert config.agent_id == "GL-007-TEST"
        assert config.furnace is not None
        assert config.cooling_tower is not None

    def test_default_sub_configs(self):
        """Test default sub-configurations."""
        config = GL007Config()
        assert config.nfpa86 is not None
        assert config.ashrae is not None
        assert config.explainability is not None
        assert config.provenance is not None


class TestFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_create_default_config(self):
        """Test default configuration factory."""
        config = create_default_config()
        assert config.furnace is not None
        assert config.cooling_tower is not None
        assert config.furnace.furnace_id == "FUR-001"
        assert config.cooling_tower.tower_id == "CT-001"

    def test_create_default_config_custom_ids(self):
        """Test default configuration with custom IDs."""
        config = create_default_config(
            furnace_id="FURNACE-A",
            tower_id="TOWER-B",
        )
        assert config.furnace.furnace_id == "FURNACE-A"
        assert config.cooling_tower.tower_id == "TOWER-B"

    def test_create_high_efficiency_config(self):
        """Test high efficiency configuration factory."""
        config = create_high_efficiency_config()
        assert config.furnace is not None
        assert config.cooling_tower is not None
        # High efficiency should have recuperative furnace
        assert config.furnace.furnace_type == FurnaceType.RECUPERATIVE
        # High efficiency should have air preheat enabled
        assert config.furnace.combustion.air_preheat_enabled is True


class TestEnums:
    """Tests for configuration enums."""

    def test_furnace_type_values(self):
        """Test furnace type enum values."""
        assert FurnaceType.DIRECT_FIRED.value == "direct_fired"
        assert FurnaceType.INDIRECT_FIRED.value == "indirect_fired"
        assert FurnaceType.RECUPERATIVE.value == "recuperative"

    def test_fuel_type_values(self):
        """Test fuel type enum values."""
        assert FuelType.NATURAL_GAS.value == "natural_gas"
        assert FuelType.PROPANE.value == "propane"
        assert FuelType.HYDROGEN.value == "hydrogen"

    def test_cooling_tower_type_values(self):
        """Test cooling tower type enum values."""
        assert CoolingTowerType.MECHANICAL_INDUCED.value == "mechanical_induced"
        assert CoolingTowerType.COUNTERFLOW.value == "counterflow"

    def test_sil_values(self):
        """Test safety integrity level enum values."""
        assert SafetyIntegrityLevel.SIL_1.value == "sil_1"
        assert SafetyIntegrityLevel.SIL_2.value == "sil_2"
        assert SafetyIntegrityLevel.SIL_3.value == "sil_3"
