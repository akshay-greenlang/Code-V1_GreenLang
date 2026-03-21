# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Configuration Tests (test_config.py)
=============================================================================

Tests configuration completeness and correctness:
  - Enum completeness tests (IndustrySector, FacilityTier, EnergyCarrier, etc.)
  - Sub-config default value tests
  - Main IndustrialEnergyAuditConfig construction tests
  - 8 preset loading tests
  - Model validator tests (data_center disables steam, etc.)
  - Utility function tests

Test Count Target: ~150 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-031 Industrial Energy Audit
Date:    March 2026
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    _load_config_module,
    _load_engine,
    PRESETS_DIR,
    PRESET_NAMES,
    CONFIG_DIR,
)


# =============================================================================
# 1. Enum Completeness Tests
# =============================================================================


class TestConfigEnums:
    """Test all configuration enums for completeness."""

    def test_industry_sector_count(self, config_module):
        """IndustrySector enum has exactly 15 members."""
        members = list(config_module.IndustrySector)
        assert len(members) == 15, f"Expected 15, got {len(members)}: {[m.value for m in members]}"

    def test_industry_sector_values(self, config_module):
        """IndustrySector contains expected values."""
        expected = {
            "MANUFACTURING", "PROCESS_INDUSTRY", "FOOD_BEVERAGE",
            "DATA_CENTER", "WAREHOUSE_LOGISTICS", "AUTOMOTIVE",
            "STEEL_METALS", "PHARMACEUTICAL", "CHEMICAL",
            "CEMENT", "GLASS", "PAPER_PULP", "TEXTILE", "PLASTICS", "OTHER",
        }
        actual = {m.value for m in config_module.IndustrySector}
        assert actual == expected

    def test_facility_tier_count(self, config_module):
        """FacilityTier enum has exactly 3 members."""
        assert len(list(config_module.FacilityTier)) == 3

    def test_facility_tier_values(self, config_module):
        """FacilityTier contains LARGE_ENTERPRISE, MID_MARKET, SME."""
        expected = {"LARGE_ENTERPRISE", "MID_MARKET", "SME"}
        actual = {m.value for m in config_module.FacilityTier}
        assert actual == expected

    def test_energy_carrier_count(self, config_module):
        """EnergyCarrier enum has exactly 12 members."""
        members = list(config_module.EnergyCarrier)
        assert len(members) == 12, f"Got {len(members)}: {[m.value for m in members]}"

    def test_energy_carrier_includes_electricity(self, config_module):
        """EnergyCarrier includes ELECTRICITY."""
        values = {m.value for m in config_module.EnergyCarrier}
        assert "ELECTRICITY" in values

    def test_energy_carrier_includes_compressed_air(self, config_module):
        """EnergyCarrier includes COMPRESSED_AIR."""
        values = {m.value for m in config_module.EnergyCarrier}
        assert "COMPRESSED_AIR" in values

    def test_audit_level_count(self, config_module):
        """AuditLevel enum has exactly 3 members."""
        assert len(list(config_module.AuditLevel)) == 3

    def test_audit_level_values(self, config_module):
        """AuditLevel contains WALK_THROUGH, DETAILED, INVESTMENT_GRADE."""
        expected = {"WALK_THROUGH", "DETAILED", "INVESTMENT_GRADE"}
        actual = {m.value for m in config_module.AuditLevel}
        assert actual == expected

    def test_motor_efficiency_class_count(self, config_module):
        """MotorEfficiencyClass enum has exactly 5 members."""
        assert len(list(config_module.MotorEfficiencyClass)) == 5

    def test_motor_efficiency_class_values(self, config_module):
        """MotorEfficiencyClass contains IE1 through IE5."""
        expected = {"IE1", "IE2", "IE3", "IE4", "IE5"}
        actual = {m.value for m in config_module.MotorEfficiencyClass}
        assert actual == expected

    def test_normalization_method_count(self, config_module):
        """NormalizationMethod enum has exactly 4 members."""
        assert len(list(config_module.NormalizationMethod)) == 4

    def test_reporting_frequency_count(self, config_module):
        """ReportingFrequency enum has exactly 7 members."""
        assert len(list(config_module.ReportingFrequency)) == 7

    def test_compliance_status_count(self, config_module):
        """ComplianceStatus enum has exactly 5 members."""
        assert len(list(config_module.ComplianceStatus)) == 5

    def test_output_format_count(self, config_module):
        """OutputFormat enum has exactly 4 members."""
        assert len(list(config_module.OutputFormat)) == 4

    def test_enpi_type_count(self, config_module):
        """EnPIType enum has at least 7 members."""
        assert len(list(config_module.EnPIType)) >= 7

    def test_enpi_type_includes_sec(self, config_module):
        """EnPIType includes SEC."""
        values = {m.value for m in config_module.EnPIType}
        assert "SEC" in values

    def test_enpi_type_includes_pue(self, config_module):
        """EnPIType includes PUE for data centers."""
        values = {m.value for m in config_module.EnPIType}
        assert "PUE" in values


# =============================================================================
# 2. Sub-Config Default Tests
# =============================================================================


class TestSubConfigDefaults:
    """Test sub-configuration models instantiate with correct defaults."""

    def test_baseline_config_defaults(self, config_module):
        """BaselineConfig instantiates with sensible defaults."""
        bc = config_module.BaselineConfig()
        assert bc.min_months == 12
        assert bc.r_squared_threshold == pytest.approx(0.75)
        assert bc.cv_rmse_threshold == pytest.approx(0.25)
        assert bc.normalization_method == config_module.NormalizationMethod.PRODUCTION_VOLUME
        assert bc.ipmvp_option == "C"

    def test_audit_config_defaults(self, config_module):
        """AuditConfig has EN 16247 compliance by default."""
        ac = config_module.AuditConfig()
        assert ac.default_audit_level == config_module.AuditLevel.DETAILED
        assert ac.en16247_compliance is True
        assert ac.eed_article_8 is True
        assert ac.schedule_months == 48
        assert ac.minimum_coverage_pct == pytest.approx(90.0)

    def test_equipment_config_defaults(self, config_module):
        """EquipmentConfig has enabled=True, motor IE3 minimum."""
        ec = config_module.EquipmentConfig()
        assert ec.enabled is True
        assert ec.motor_efficiency_min_class == config_module.MotorEfficiencyClass.IE3
        assert ec.vsd_retrofit_analysis is True

    def test_steam_config_defaults(self, config_module):
        """SteamConfig has enabled=True, 87% boiler target."""
        sc = config_module.SteamConfig()
        assert sc.enabled is True
        assert sc.boiler_efficiency_target_pct == pytest.approx(87.0)
        assert sc.condensate_return_target_pct == pytest.approx(85.0)
        assert sc.blowdown_target_pct == pytest.approx(5.0)

    def test_compressed_air_config_defaults(self, config_module):
        """CompressedAirConfig has enabled=True, 6.5 kW specific power target."""
        ca = config_module.CompressedAirConfig()
        assert ca.enabled is True
        assert ca.specific_power_target_kw_per_m3_min == pytest.approx(6.5)
        assert ca.leak_rate_target_pct == pytest.approx(10.0)
        assert ca.pressure_optimization is True

    def test_waste_heat_config_defaults(self, config_module):
        """WasteHeatConfig has enabled=True, 60C minimum temperature."""
        wh = config_module.WasteHeatConfig()
        assert wh.enabled is True
        assert wh.min_temperature_c == pytest.approx(60.0)
        assert wh.pinch_analysis_enabled is False  # Opt-in
        assert wh.flue_gas_recovery is True

    def test_lighting_config_defaults(self, config_module):
        """LightingConfig has enabled=True, EN 12464-1 standard."""
        lc = config_module.LightingConfig()
        assert lc.enabled is True
        assert lc.lpd_standard == "EN_12464_1"
        assert lc.led_retrofit_analysis is True

    def test_hvac_config_defaults(self, config_module):
        """HVACConfig has enabled=True, VSD and economizer analysis."""
        hc = config_module.HVACConfig()
        assert hc.enabled is True
        assert hc.vsd_analysis is True
        assert hc.economizer_analysis is True
        assert hc.heat_recovery is True

    def test_benchmark_config_defaults(self, config_module):
        """BenchmarkConfig has enabled=True, BAT-AEL comparison."""
        bc = config_module.BenchmarkConfig()
        assert bc.enabled is True
        assert bc.bat_ael_comparison is True
        assert bc.percentile_tracking is True

    def test_iso50001_config_defaults(self, config_module):
        """ISO50001Config has enabled=False by default."""
        ic = config_module.ISO50001Config()
        assert ic.enabled is False
        assert ic.eed_exemption_tracking is True

    def test_eed_config_defaults(self, config_module):
        """EEDConfig has enabled=True, 250 employees threshold."""
        ec = config_module.EEDConfig()
        assert ec.enabled is True
        assert ec.large_enterprise_threshold_employees == 250
        assert ec.large_enterprise_threshold_revenue_eur == pytest.approx(50_000_000.0)
        assert ec.mandatory_audit is True
        assert ec.audit_interval_months == 48

    def test_performance_config_defaults(self, config_module):
        """PerformanceConfig has sensible defaults."""
        pc = config_module.PerformanceConfig()
        assert pc.max_facilities == 50
        assert pc.batch_size == 1000

    def test_security_config_defaults(self, config_module):
        """SecurityConfig has CONFIDENTIAL classification by default."""
        sc = config_module.SecurityConfig()
        assert sc.data_classification == "CONFIDENTIAL"
        assert sc.audit_logging is True

    def test_audit_trail_config_defaults(self, config_module):
        """AuditTrailConfig has SHA-256 provenance by default."""
        at = config_module.AuditTrailConfig()
        assert at.enabled is True
        assert at.sha256_provenance is True
        assert at.retention_years == 7


# =============================================================================
# 3. Main Config Construction Tests
# =============================================================================


class TestIndustrialEnergyAuditConfig:
    """Test IndustrialEnergyAuditConfig construction and field validation."""

    def test_default_construction(self, config_module):
        """IndustrialEnergyAuditConfig can be created with all defaults."""
        cfg = config_module.IndustrialEnergyAuditConfig()
        assert cfg.reporting_year >= 2020
        assert cfg.facility_tier == config_module.FacilityTier.LARGE_ENTERPRISE
        assert cfg.industry_sector == config_module.IndustrySector.MANUFACTURING

    def test_custom_facility_name(self, config_module):
        """Facility name can be set."""
        cfg = config_module.IndustrialEnergyAuditConfig(facility_name="TestPlant")
        assert cfg.facility_name == "TestPlant"

    def test_custom_sector(self, config_module):
        """Industry sector can be set."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            industry_sector=config_module.IndustrySector.DATA_CENTER
        )
        assert cfg.industry_sector == config_module.IndustrySector.DATA_CENTER

    def test_all_sub_configs_present(self, config_module):
        """Config has all expected sub-config attributes."""
        cfg = config_module.IndustrialEnergyAuditConfig()
        for attr in [
            "baseline", "audit", "equipment", "steam", "compressed_air",
            "waste_heat", "lighting", "hvac", "benchmark", "iso50001", "eed",
            "performance", "security", "audit_trail", "reporting",
        ]:
            assert hasattr(cfg, attr), f"Missing sub-config: {attr}"

    def test_energy_carriers_default(self, config_module):
        """Default energy carriers include ELECTRICITY and NATURAL_GAS."""
        cfg = config_module.IndustrialEnergyAuditConfig()
        carrier_values = [c.value for c in cfg.energy_carriers]
        assert "ELECTRICITY" in carrier_values
        assert "NATURAL_GAS" in carrier_values


# =============================================================================
# 4. Preset Loading Tests
# =============================================================================


class TestPresetLoading:
    """Test loading of all 8 presets."""

    def test_manufacturing_preset_loads(self, config_module):
        """Manufacturing preset YAML exists and loads."""
        path = PRESETS_DIR / "manufacturing_plant.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            assert data is not None
            assert isinstance(data, dict)
            assert data.get("industry_sector") == "MANUFACTURING"

    def test_manufacturing_preset_via_pack_config(self, config_module):
        """Manufacturing preset loads via PackConfig.from_preset()."""
        path = PRESETS_DIR / "manufacturing_plant.yaml"
        if path.exists():
            cfg = config_module.PackConfig.from_preset("manufacturing_plant")
            assert cfg is not None
            assert cfg.preset_name == "manufacturing_plant"
            assert cfg.pack_id == "PACK-031-industrial-energy-audit"

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_each_preset_has_valid_sector(self, config_module, preset_name):
        """Each preset specifies a valid industry sector."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            sector = data.get("industry_sector", "MANUFACTURING")
            valid_sectors = {m.value for m in config_module.IndustrySector}
            assert sector in valid_sectors, f"Preset {preset_name} has invalid sector: {sector}"

    def test_available_presets_list(self, config_module):
        """list_available_presets returns all 8 presets."""
        presets = config_module.list_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 8


# =============================================================================
# 5. Model Validator Tests
# =============================================================================


class TestModelValidators:
    """Test model validators that enforce cross-field constraints."""

    def test_data_center_disables_steam(self, config_module):
        """Data center sector auto-disables steam analysis."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            industry_sector=config_module.IndustrySector.DATA_CENTER,
            steam=config_module.SteamConfig(enabled=True),
        )
        assert cfg.steam.enabled is False

    def test_data_center_disables_compressed_air(self, config_module):
        """Data center sector auto-disables compressed air analysis."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            industry_sector=config_module.IndustrySector.DATA_CENTER,
            compressed_air=config_module.CompressedAirConfig(enabled=True),
        )
        assert cfg.compressed_air.enabled is False

    def test_process_industry_enables_waste_heat(self, config_module):
        """Process industry sector auto-enables waste heat recovery."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            industry_sector=config_module.IndustrySector.PROCESS_INDUSTRY,
            waste_heat=config_module.WasteHeatConfig(enabled=False),
        )
        assert cfg.waste_heat.enabled is True

    def test_steel_metals_enables_waste_heat(self, config_module):
        """Steel/metals sector auto-enables waste heat recovery."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            industry_sector=config_module.IndustrySector.STEEL_METALS,
            waste_heat=config_module.WasteHeatConfig(enabled=False),
        )
        assert cfg.waste_heat.enabled is True

    def test_large_enterprise_enables_eed(self, config_module):
        """Large enterprise auto-enables EED compliance."""
        cfg = config_module.IndustrialEnergyAuditConfig(
            facility_tier=config_module.FacilityTier.LARGE_ENTERPRISE,
            eed=config_module.EEDConfig(enabled=False),
        )
        assert cfg.eed.enabled is True

    def test_ipmvp_option_validation(self, config_module):
        """BaselineConfig rejects invalid IPMVP options."""
        with pytest.raises(Exception):
            config_module.BaselineConfig(ipmvp_option="Z")

    def test_ipmvp_option_case_normalization(self, config_module):
        """BaselineConfig normalizes IPMVP option to uppercase."""
        bc = config_module.BaselineConfig(ipmvp_option="c")
        assert bc.ipmvp_option == "C"

    def test_en16247_parts_validation(self, config_module):
        """AuditConfig rejects invalid EN 16247 part numbers."""
        with pytest.raises(Exception):
            config_module.AuditConfig(en16247_parts=[0, 6])

    def test_en16247_parts_deduplication(self, config_module):
        """AuditConfig deduplicates and sorts EN 16247 parts."""
        ac = config_module.AuditConfig(en16247_parts=[3, 1, 3, 2, 1])
        assert ac.en16247_parts == [1, 2, 3]

    def test_config_merge_order(self, config_module):
        """PackConfig.merge applies overrides correctly."""
        base = config_module.PackConfig()
        merged = config_module.PackConfig.merge(
            base, {"facility_name": "OverriddenPlant"}
        )
        assert merged.pack.facility_name == "OverriddenPlant"

    def test_config_hash_deterministic(self, config_module):
        """PackConfig.get_config_hash is deterministic."""
        cfg1 = config_module.PackConfig()
        cfg2 = config_module.PackConfig()
        assert cfg1.get_config_hash() == cfg2.get_config_hash()

    def test_config_hash_64_chars(self, config_module):
        """Config hash is 64 characters (SHA-256)."""
        cfg = config_module.PackConfig()
        assert len(cfg.get_config_hash()) == 64


# =============================================================================
# 6. Reference Data Tests
# =============================================================================


class TestReferenceData:
    """Test module-level reference data constants."""

    def test_sector_info_has_15_entries(self, config_module):
        """SECTOR_INFO has entries for all 15 sectors."""
        assert len(config_module.SECTOR_INFO) >= 15

    def test_sector_info_manufacturing(self, config_module):
        """SECTOR_INFO has MANUFACTURING entry with NACE code."""
        info = config_module.SECTOR_INFO["MANUFACTURING"]
        assert "nace" in info
        assert "key_systems" in info

    def test_compressed_air_benchmarks(self, config_module):
        """COMPRESSED_AIR_BENCHMARKS has best_practice, good, average, poor."""
        benchmarks = config_module.COMPRESSED_AIR_BENCHMARKS
        assert benchmarks["best_practice"] == pytest.approx(5.5)
        assert benchmarks["good"] == pytest.approx(6.5)

    def test_steam_benchmarks(self, config_module):
        """STEAM_BENCHMARKS has boiler efficiency targets."""
        benchmarks = config_module.STEAM_BENCHMARKS
        assert benchmarks["boiler_efficiency_best"] == pytest.approx(0.92)
        assert benchmarks["condensate_return_target_pct"] == pytest.approx(85.0)

    def test_lpd_standards(self, config_module):
        """LPD_STANDARDS has office and warehouse entries."""
        lpd = config_module.LPD_STANDARDS
        assert "office" in lpd
        assert "warehouse_low_bay" in lpd
        assert lpd["office"] == pytest.approx(8.0)

    def test_pue_benchmarks(self, config_module):
        """PUE_BENCHMARKS has world_class through inefficient."""
        pue = config_module.PUE_BENCHMARKS
        assert pue["world_class"] == pytest.approx(1.1)
        assert pue["average"] == pytest.approx(1.6)

    def test_get_sector_info(self, config_module):
        """get_sector_info returns info for known sectors."""
        info = config_module.get_sector_info("MANUFACTURING")
        assert info is not None
        assert "name" in info

    def test_get_compressed_air_benchmark(self, config_module):
        """get_compressed_air_benchmark returns correct values."""
        assert config_module.get_compressed_air_benchmark("best_practice") == pytest.approx(5.5)
        assert config_module.get_compressed_air_benchmark("good") == pytest.approx(6.5)

    def test_get_lpd_standard(self, config_module):
        """get_lpd_standard returns correct LPD target."""
        assert config_module.get_lpd_standard("office") == pytest.approx(8.0)

    def test_get_pue_benchmark(self, config_module):
        """get_pue_benchmark returns correct PUE target."""
        assert config_module.get_pue_benchmark("efficient") == pytest.approx(1.3)

    def test_validate_config_warnings(self, config_module):
        """validate_config returns warnings for incomplete config."""
        cfg = config_module.IndustrialEnergyAuditConfig()
        warnings = config_module.validate_config(cfg)
        assert isinstance(warnings, list)
        # Default config has no facility_name -> should warn
        assert any("facility_name" in w for w in warnings)

    def test_validate_config_no_production(self, config_module):
        """validate_config warns when no production data set."""
        cfg = config_module.IndustrialEnergyAuditConfig()
        warnings = config_module.validate_config(cfg)
        assert any("production" in w.lower() for w in warnings)
