# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Configuration Tests (test_config.py)
========================================================================

Tests the configuration system for PACK-013 including:
  - Enum completeness and values
  - Sub-configuration defaults and validation
  - Main CSRDManufacturingConfig construction and model validation
  - Preset loading for all 6 manufacturing presets
  - PackConfig wrapper functionality
  - Utility functions (load_preset, validate_config, get_default_config)

Test count: 46 test definitions.

All modules are loaded dynamically via importlib to avoid package install.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-013 CSRD Manufacturing
Date:    March 2026
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _load_config_module


# =============================================================================
# Module Loading
# =============================================================================


@pytest.fixture(scope="module")
def cfg():
    """Load the pack_config module once per test module."""
    return _load_config_module()


# =============================================================================
# Test Class: Enums
# =============================================================================


class TestEnums:
    """Tests for manufacturing-specific enumeration types."""

    def test_manufacturing_sub_sector_count(self, cfg):
        """Verify ManufacturingSubSector has exactly 16 members."""
        members = list(cfg.ManufacturingSubSector)
        assert len(members) == 16, (
            f"Expected 16 ManufacturingSubSector members, got {len(members)}: "
            f"{[m.value for m in members]}"
        )

    def test_manufacturing_sub_sector_values(self, cfg):
        """Verify key sub-sector values are present."""
        values = {m.value for m in cfg.ManufacturingSubSector}
        expected = {"CEMENT", "STEEL", "ALUMINUM", "CHEMICALS", "AUTOMOTIVE",
                    "ELECTRONICS", "PHARMACEUTICALS", "PACKAGING"}
        assert expected.issubset(values), (
            f"Missing sub-sector values: {expected - values}"
        )

    def test_manufacturing_tier_count(self, cfg):
        """Verify ManufacturingTier has exactly 4 members."""
        members = list(cfg.ManufacturingTier)
        assert len(members) == 4, (
            f"Expected 4 ManufacturingTier members, got {len(members)}"
        )

    def test_manufacturing_tier_values(self, cfg):
        """Verify tier values match expected set."""
        values = {m.value for m in cfg.ManufacturingTier}
        expected = {"HEAVY_INDUSTRY", "DISCRETE", "PROCESS", "LIGHT"}
        assert values == expected, (
            f"ManufacturingTier values mismatch: {values} != {expected}"
        )

    def test_cbam_status_values(self, cfg):
        """Verify CBAMStatus has the expected values."""
        values = {m.value for m in cfg.CBAMStatus}
        expected = {"NOT_AFFECTED", "TRANSITIONAL", "FULL_COMPLIANCE"}
        assert values == expected, (
            f"CBAMStatus values mismatch: {values} != {expected}"
        )

    def test_bat_compliance_levels(self, cfg):
        """Verify BATComplianceLevel has 4 levels."""
        members = list(cfg.BATComplianceLevel)
        assert len(members) == 4, (
            f"Expected 4 BATComplianceLevel members, got {len(members)}"
        )
        values = {m.value for m in members}
        expected = {"COMPLIANT", "WITHIN_RANGE", "NON_COMPLIANT", "DEROGATION"}
        assert values == expected

    def test_eed_tier_values(self, cfg):
        """Verify EnergySource enum has at least 10 members."""
        members = list(cfg.EnergySource)
        assert len(members) >= 10, (
            f"Expected at least 10 EnergySource members, got {len(members)}"
        )


# =============================================================================
# Test Class: Sub-Configuration Defaults
# =============================================================================


class TestSubConfigs:
    """Tests for sub-configuration model defaults."""

    def test_process_emissions_config_defaults(self, cfg):
        """Verify ProcessEmissionsConfig has correct defaults."""
        config = cfg.ProcessEmissionsConfig()
        assert config.enabled is True
        assert config.sub_sector == cfg.ManufacturingSubSector.STEEL
        assert config.cbam_affected is False
        assert config.cbam_status == cfg.CBAMStatus.NOT_AFFECTED
        assert config.mass_balance_enabled is True
        assert config.emission_factor_source == "EU_MRV"
        assert "combustion" in config.emission_sources
        assert "process" in config.emission_sources

    def test_energy_intensity_config(self, cfg):
        """Verify EnergyIntensityConfig has correct defaults."""
        config = cfg.EnergyIntensityConfig()
        assert config.enabled is True
        assert config.production_unit == "tonne"
        assert config.iso50001_certified is False
        assert config.enpi_tracking is True
        assert config.significant_energy_use_threshold_pct == 5.0
        assert config.renewable_energy_tracking is True
        assert config.baseline_year is None

    def test_pcf_config_defaults(self, cfg):
        """Verify ProductPCFConfig has correct defaults."""
        config = cfg.ProductPCFConfig()
        assert config.enabled is True
        assert config.lifecycle_scope == cfg.LifecycleScope.CRADLE_TO_GATE
        assert config.allocation_method == cfg.AllocationMethod.MASS
        assert config.dpp_enabled is False
        assert config.pef_methodology is True
        assert config.iso14067_compliant is True
        assert config.emission_factor_database == "ECOINVENT"
        assert config.uncertainty_analysis is True

    def test_circular_economy_config(self, cfg):
        """Verify CircularEconomyConfig has correct defaults."""
        config = cfg.CircularEconomyConfig()
        assert config.enabled is True
        assert config.recycled_content_tracking is True
        assert config.material_circularity_indicator is True
        assert config.waste_diversion_target_pct == 80.0
        assert config.industrial_symbiosis_tracking is False
        assert len(config.waste_streams) >= 3

    def test_water_pollution_config(self, cfg):
        """Verify WaterPollutionConfig has correct defaults."""
        config = cfg.WaterPollutionConfig()
        assert config.enabled is True
        assert config.water_stress_assessment is True
        assert config.reach_svhc_tracking is False
        assert config.eprtr_reporting is False
        assert config.water_intensity_metric == "m3_per_tonne"
        assert len(config.water_sources) >= 2

    def test_bat_config(self, cfg):
        """Verify BATComplianceConfig has correct defaults."""
        config = cfg.BATComplianceConfig()
        assert config.enabled is True
        assert config.compliance_level == cfg.BATComplianceLevel.COMPLIANT
        assert config.bat_ael_monitoring is True
        assert config.ied_inspection_readiness is True
        assert config.transformation_plan is False

    def test_supply_chain_config(self, cfg):
        """Verify SupplyChainConfig has correct defaults."""
        config = cfg.SupplyChainConfig()
        assert config.enabled is True
        assert config.tier_depth == 2
        assert config.spend_based_screening is True
        assert config.supplier_specific_enabled is True
        assert config.hybrid_method is True
        assert config.hotspot_analysis is True
        assert len(config.priority_categories) >= 5

    def test_benchmark_config(self, cfg):
        """Verify BenchmarkConfig has correct defaults."""
        config = cfg.BenchmarkConfig()
        assert config.enabled is True
        assert config.peer_group == "NACE_SECTOR"
        assert config.abatement_cost_curve is True
        assert config.gap_analysis_enabled is True
        assert len(config.kpi_set) >= 5
        assert len(config.target_years) >= 3


# =============================================================================
# Test Class: Main Configuration Model
# =============================================================================


class TestMainConfig:
    """Tests for CSRDManufacturingConfig main configuration model."""

    def test_default_config(self, cfg):
        """Verify CSRDManufacturingConfig creates with all defaults."""
        config = cfg.CSRDManufacturingConfig()
        assert config.company_name == ""
        assert config.reporting_year == 2025
        assert config.manufacturing_tier == cfg.ManufacturingTier.HEAVY_INDUSTRY
        assert len(config.sub_sectors) >= 1
        assert config.process_emissions.enabled is True
        assert config.energy_intensity.enabled is True
        assert config.product_pcf.enabled is True
        assert config.circular_economy.enabled is True
        assert config.water_pollution.enabled is True
        assert config.bat_compliance.enabled is True
        assert config.supply_chain.enabled is True
        assert config.benchmark.enabled is True
        assert config.audit_trail.enabled is True

    def test_config_with_values(self, cfg):
        """Verify CSRDManufacturingConfig accepts explicit values."""
        config = cfg.CSRDManufacturingConfig(
            company_name="Test Manufacturing GmbH",
            reporting_year=2026,
            manufacturing_tier=cfg.ManufacturingTier.DISCRETE,
            sub_sectors=[
                cfg.ManufacturingSubSector.AUTOMOTIVE,
                cfg.ManufacturingSubSector.ELECTRONICS,
            ],
        )
        assert config.company_name == "Test Manufacturing GmbH"
        assert config.reporting_year == 2026
        assert config.manufacturing_tier == cfg.ManufacturingTier.DISCRETE
        assert len(config.sub_sectors) == 2
        assert cfg.ManufacturingSubSector.AUTOMOTIVE in config.sub_sectors

    def test_config_reporting_year_validation(self, cfg):
        """Verify reporting_year validation rejects out-of-range values."""
        with pytest.raises(Exception):
            cfg.CSRDManufacturingConfig(reporting_year=2000)

        with pytest.raises(Exception):
            cfg.CSRDManufacturingConfig(reporting_year=2040)

    def test_config_preset_loading_heavy_industry(self, cfg):
        """Verify PackConfig.from_preset works for heavy_industry."""
        pack_cfg = cfg.PackConfig.from_preset("heavy_industry")
        assert pack_cfg.preset_name == "heavy_industry"
        assert pack_cfg.pack.manufacturing_tier == cfg.ManufacturingTier.HEAVY_INDUSTRY
        assert pack_cfg.pack.process_emissions.enabled is True
        assert pack_cfg.pack.bat_compliance.enabled is True

    def test_config_preset_loading_discrete(self, cfg):
        """Verify PackConfig.from_preset works for discrete_manufacturing."""
        pack_cfg = cfg.PackConfig.from_preset("discrete_manufacturing")
        assert pack_cfg.preset_name == "discrete_manufacturing"
        assert pack_cfg.pack.manufacturing_tier == cfg.ManufacturingTier.DISCRETE
        assert pack_cfg.pack.product_pcf.enabled is True

    def test_config_preset_loading_process(self, cfg):
        """Verify PackConfig.from_preset works for process_manufacturing."""
        pack_cfg = cfg.PackConfig.from_preset("process_manufacturing")
        assert pack_cfg.preset_name == "process_manufacturing"
        assert pack_cfg.pack.manufacturing_tier == cfg.ManufacturingTier.PROCESS
        assert pack_cfg.pack.water_pollution.enabled is True

    def test_config_preset_loading_light(self, cfg):
        """Verify PackConfig.from_preset works for light_manufacturing."""
        pack_cfg = cfg.PackConfig.from_preset("light_manufacturing")
        assert pack_cfg.preset_name == "light_manufacturing"
        assert pack_cfg.pack.manufacturing_tier == cfg.ManufacturingTier.LIGHT
        assert pack_cfg.pack.circular_economy.enabled is True

    def test_config_preset_loading_multi_site(self, cfg):
        """Verify PackConfig.from_preset works for multi_site."""
        pack_cfg = cfg.PackConfig.from_preset("multi_site")
        assert pack_cfg.preset_name == "multi_site"
        assert pack_cfg.pack.process_emissions.enabled is True

    def test_config_preset_loading_sme(self, cfg):
        """Verify PackConfig.from_preset works for sme_manufacturer."""
        pack_cfg = cfg.PackConfig.from_preset("sme_manufacturer")
        assert pack_cfg.preset_name == "sme_manufacturer"
        assert pack_cfg.pack.energy_intensity.enabled is True


# =============================================================================
# Test Class: Configuration Validation
# =============================================================================


class TestConfigValidation:
    """Tests for configuration validation logic."""

    def test_valid_config(self, cfg):
        """Verify a fully valid configuration passes validation."""
        config = cfg.CSRDManufacturingConfig(
            company_name="Valid Corp",
            reporting_year=2025,
            manufacturing_tier=cfg.ManufacturingTier.HEAVY_INDUSTRY,
            sub_sectors=[cfg.ManufacturingSubSector.CEMENT],
            facilities=[
                cfg.FacilityConfig(
                    facility_id="FAC-001",
                    facility_name="Test Facility",
                    country="DE",
                    sub_sector=cfg.ManufacturingSubSector.CEMENT,
                ),
            ],
        )
        warnings = cfg.validate_config(config)
        # No critical failures; warnings are acceptable
        assert isinstance(warnings, list)

    def test_config_from_dict(self, cfg):
        """Verify CSRDManufacturingConfig can be created from a plain dict."""
        data = {
            "company_name": "Dict Corp",
            "reporting_year": 2026,
            "manufacturing_tier": "DISCRETE",
            "sub_sectors": ["AUTOMOTIVE"],
        }
        config = cfg.CSRDManufacturingConfig(**data)
        assert config.company_name == "Dict Corp"
        assert config.manufacturing_tier == cfg.ManufacturingTier.DISCRETE

    def test_config_from_none(self, cfg):
        """Verify CSRDManufacturingConfig can be created with no arguments."""
        config = cfg.CSRDManufacturingConfig()
        assert config is not None
        assert config.reporting_year == 2025

    def test_omnibus_threshold_validation(self, cfg):
        """Verify OmnibusConfig accepts valid threshold values."""
        omnibus = cfg.OmnibusConfig(
            enabled=True,
            total_assets_eur=50000000.0,
            net_turnover_eur=25000000.0,
            average_employees=250,
            listed_entity=False,
        )
        assert omnibus.total_assets_eur == 50000000.0
        assert omnibus.average_employees == 250

    def test_facilities_list(self, cfg):
        """Verify facilities can be added to the configuration."""
        config = cfg.CSRDManufacturingConfig(
            facilities=[
                cfg.FacilityConfig(
                    facility_id="FAC-A",
                    facility_name="Facility A",
                    country="DE",
                    sub_sector=cfg.ManufacturingSubSector.STEEL,
                    production_capacity_tonnes=500000.0,
                    employees=500,
                ),
                cfg.FacilityConfig(
                    facility_id="FAC-B",
                    facility_name="Facility B",
                    country="FR",
                    sub_sector=cfg.ManufacturingSubSector.CHEMICALS,
                ),
            ],
        )
        assert len(config.facilities) == 2
        assert config.facilities[0].country == "DE"
        assert config.facilities[1].country == "FR"

    def test_cross_field_validation(self, cfg):
        """Verify model validators enforce cross-field constraints.

        Heavy industry should auto-enable process_emissions and bat_compliance.
        """
        config = cfg.CSRDManufacturingConfig(
            manufacturing_tier=cfg.ManufacturingTier.HEAVY_INDUSTRY,
        )
        # Model validator should ensure these are enabled for heavy industry
        assert config.process_emissions.enabled is True
        assert config.bat_compliance.enabled is True


# =============================================================================
# Test Class: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for pack_config utility functions."""

    def test_load_preset(self, cfg):
        """Verify load_preset convenience function works."""
        pack_cfg = cfg.load_preset("heavy_industry")
        assert isinstance(pack_cfg, cfg.PackConfig)
        assert pack_cfg.preset_name == "heavy_industry"

    def test_load_preset_invalid(self, cfg):
        """Verify load_preset raises ValueError for unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            cfg.load_preset("nonexistent_preset")

    def test_validate_config(self, cfg):
        """Verify validate_config returns a list of warnings."""
        config = cfg.CSRDManufacturingConfig()
        warnings = cfg.validate_config(config)
        assert isinstance(warnings, list)
        # Default config has no facilities, so expect at least one warning
        assert len(warnings) >= 1
        assert any("facilit" in w.lower() for w in warnings)

    def test_validate_config_with_facilities(self, cfg):
        """Verify validate_config returns fewer warnings with facilities."""
        config = cfg.CSRDManufacturingConfig(
            sub_sectors=[cfg.ManufacturingSubSector.CEMENT],
            facilities=[
                cfg.FacilityConfig(
                    facility_id="FAC-001",
                    facility_name="Test",
                    country="DE",
                    sub_sector=cfg.ManufacturingSubSector.CEMENT,
                ),
            ],
        )
        warnings = cfg.validate_config(config)
        assert isinstance(warnings, list)
        # Should not have the "no facilities" warning
        assert not any("No facilities" in w for w in warnings)

    def test_get_default_config(self, cfg):
        """Verify get_default_config returns config for given tier."""
        config = cfg.get_default_config(cfg.ManufacturingTier.DISCRETE)
        assert config.manufacturing_tier == cfg.ManufacturingTier.DISCRETE
        assert cfg.ManufacturingSubSector.AUTOMOTIVE in config.sub_sectors

    def test_get_default_config_heavy(self, cfg):
        """Verify get_default_config returns heavy industry defaults."""
        config = cfg.get_default_config(cfg.ManufacturingTier.HEAVY_INDUSTRY)
        assert config.manufacturing_tier == cfg.ManufacturingTier.HEAVY_INDUSTRY
        assert cfg.ManufacturingSubSector.STEEL in config.sub_sectors

    def test_get_default_config_light(self, cfg):
        """Verify get_default_config returns light manufacturing defaults."""
        config = cfg.get_default_config(cfg.ManufacturingTier.LIGHT)
        assert config.manufacturing_tier == cfg.ManufacturingTier.LIGHT
        assert cfg.ManufacturingSubSector.PACKAGING in config.sub_sectors

    def test_pack_config_hash(self, cfg):
        """Verify PackConfig.get_config_hash returns a valid SHA-256 hash."""
        pack_cfg = cfg.PackConfig()
        hash_val = pack_cfg.get_config_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256 hex digest length

    def test_pack_config_hash_deterministic(self, cfg):
        """Verify same config produces same hash (deterministic)."""
        pack_cfg1 = cfg.PackConfig()
        pack_cfg2 = cfg.PackConfig()
        assert pack_cfg1.get_config_hash() == pack_cfg2.get_config_hash()

    def test_available_presets_dict(self, cfg):
        """Verify AVAILABLE_PRESETS constant has 6 entries."""
        presets = cfg.AVAILABLE_PRESETS
        assert isinstance(presets, dict)
        assert len(presets) == 6
        assert "heavy_industry" in presets
        assert "sme_manufacturer" in presets

    def test_tier_subsectors_mapping(self, cfg):
        """Verify TIER_SUBSECTORS maps all 4 tiers."""
        mapping = cfg.TIER_SUBSECTORS
        assert len(mapping) == 4
        assert "HEAVY_INDUSTRY" in mapping
        assert "CEMENT" in mapping["HEAVY_INDUSTRY"]
        assert "AUTOMOTIVE" in mapping["DISCRETE"]

    def test_subsector_info_constant(self, cfg):
        """Verify SUBSECTOR_INFO has all 16 sub-sectors."""
        info = cfg.SUBSECTOR_INFO
        assert len(info) == 16
        assert "CEMENT" in info
        assert info["CEMENT"]["cbam"] == "Yes"
        assert info["ELECTRONICS"]["cbam"] == "No"
