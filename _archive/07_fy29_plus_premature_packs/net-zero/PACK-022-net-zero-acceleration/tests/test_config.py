# -*- coding: utf-8 -*-
"""
Unit tests for PACK-022 Net Zero Acceleration Pack - Configuration Manager.

Tests PackConfig creation, NetZeroAccelerationConfig defaults, sub-config
models, preset loading, validation warnings, sector defaults, merge logic,
env overrides, constants, enums, and utility functions.
"""

import os
import sys
import pytest
from decimal import Decimal
from pathlib import Path

# Ensure pack root is on the Python path
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import (
    PackConfig,
    NetZeroAccelerationConfig,
    ScenarioConfig,
    PathwayConfig,
    SupplierConfig,
    Scope3Config,
    FinanceConfig,
    TemperatureConfig,
    DecompositionConfig,
    MultiEntityConfig,
    VCMIConfig,
    AssuranceConfig,
    ScenarioType,
    PathwayMethodology,
    SupplierTier,
    ScopeCategory,
    FinanceInstrument,
    TemperatureTarget,
    DecompositionMethod,
    EntityScope,
    VCMITier,
    AssuranceLevel,
    SectorClassification,
    SUPPORTED_PRESETS,
    SDA_SECTORS,
    SDA_INTENSITY_METRICS,
    TEMPERATURE_REGRESSION,
    VCMI_TIER_THRESHOLDS,
    IPCC_AR6_GWP100,
    SBTI_REDUCTION_RATES,
    SBTI_COVERAGE_THRESHOLDS,
    SECTOR_SCOPE3_PRIORITY,
    PRIORITY_SCOPE3_BY_SECTOR,
    SECTOR_INFO,
    DEFAULT_BASE_YEAR,
    DEFAULT_NEAR_TERM_YEAR,
    DEFAULT_LONG_TERM_YEAR,
    DEFAULT_MONTE_CARLO_RUNS,
    DEFAULT_MAX_SUPPLIERS,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_SCOPE3_CATEGORIES,
    load_config,
    load_preset,
    get_sector_defaults,
    merge_config,
    get_env_overrides,
    validate_config,
    get_sector_info,
    get_sda_intensity_metric,
    get_vcmi_tier_thresholds,
    get_temperature_regression,
    get_sbti_reduction_rate,
    get_gwp100,
    list_available_presets,
    list_sda_sectors,
)


# ---------------------------------------------------------------------------
# PackConfig Default Creation
# ---------------------------------------------------------------------------


class TestPackConfigDefaults:

    def test_default_pack_config(self):
        pc = PackConfig()
        assert pc.pack_id == "PACK-022-net-zero-acceleration"
        assert pc.config_version == "1.0.0"
        assert pc.preset_name is None
        assert isinstance(pc.pack, NetZeroAccelerationConfig)

    def test_default_acceleration_config(self):
        cfg = NetZeroAccelerationConfig()
        assert cfg.sector == SectorClassification.MANUFACTURING
        assert cfg.region == "EU"
        assert cfg.country == "DE"
        assert cfg.reporting_year == 2025
        assert cfg.base_year == DEFAULT_BASE_YEAR
        assert cfg.pack_version == "1.0.0"
        assert cfg.organization_name == ""

    def test_sub_config_instances(self):
        cfg = NetZeroAccelerationConfig()
        assert isinstance(cfg.scenario, ScenarioConfig)
        assert isinstance(cfg.pathway, PathwayConfig)
        assert isinstance(cfg.supplier, SupplierConfig)
        assert isinstance(cfg.scope3, Scope3Config)
        assert isinstance(cfg.finance, FinanceConfig)
        assert isinstance(cfg.temperature, TemperatureConfig)
        assert isinstance(cfg.decomposition, DecompositionConfig)
        assert isinstance(cfg.multi_entity, MultiEntityConfig)
        assert isinstance(cfg.vcmi, VCMIConfig)
        assert isinstance(cfg.assurance, AssuranceConfig)


# ---------------------------------------------------------------------------
# Sub-Config Defaults
# ---------------------------------------------------------------------------


class TestSubConfigDefaults:

    def test_scenario_defaults(self):
        sc = ScenarioConfig()
        assert sc.monte_carlo_runs == DEFAULT_MONTE_CARLO_RUNS
        assert sc.random_seed == 42
        assert sc.confidence_interval_pct == 95.0
        assert sc.sensitivity_top_n == 10
        assert ScenarioType.BAU in sc.scenario_types
        assert ScenarioType.MODERATE in sc.scenario_types
        assert ScenarioType.AMBITIOUS in sc.scenario_types

    def test_pathway_defaults(self):
        pc = PathwayConfig()
        assert pc.methodology == PathwayMethodology.ACA
        assert pc.sda_enabled is False
        assert pc.sda_sectors == []
        assert pc.ambition_level == TemperatureTarget.CELSIUS_1_5
        assert pc.near_term_target_year == DEFAULT_NEAR_TERM_YEAR
        assert pc.long_term_target_year == DEFAULT_LONG_TERM_YEAR
        assert pc.coverage_scope3_pct == 67.0

    def test_supplier_defaults(self):
        sc = SupplierConfig()
        assert sc.enabled is True
        assert sc.max_suppliers == DEFAULT_MAX_SUPPLIERS
        assert sc.top_supplier_count == 50
        assert sc.data_collection_frequency == "annual"
        assert sc.batch_size == 2000
        assert len(sc.engagement_scoring_dimensions) == 5

    def test_scope3_defaults(self):
        s3 = Scope3Config()
        assert s3.categories == DEFAULT_SCOPE3_CATEGORIES
        assert s3.activity_based_categories == [1, 3, 4, 5]
        assert s3.materiality_threshold_pct == 1.0
        assert s3.target_dqis_score == 2
        assert s3.pcaf_enabled is False

    def test_finance_defaults(self):
        fc = FinanceConfig()
        assert fc.enabled is True
        assert fc.discount_rate_pct == 8.0
        assert fc.carbon_price_eur_per_tco2e == 85.0
        assert fc.reporting_currency == "EUR"
        assert fc.planning_horizon_years == 10
        assert fc.taxonomy_alignment is True

    def test_temperature_defaults(self):
        tc = TemperatureConfig()
        assert tc.enabled is True
        assert tc.default_score_celsius == 3.2
        assert "WATS" in tc.aggregation_methods
        assert tc.portfolio_scoring is False

    def test_decomposition_defaults(self):
        dc = DecompositionConfig()
        assert dc.enabled is True
        assert dc.method == DecompositionMethod.LMDI
        assert dc.forecast_periods == 4
        assert dc.alert_threshold_pct == 10.0

    def test_multi_entity_defaults(self):
        mc = MultiEntityConfig()
        assert mc.enabled is False
        assert mc.max_entities == DEFAULT_MAX_ENTITIES
        assert mc.consolidation_method == EntityScope.OPERATIONAL_CONTROL
        assert mc.intercompany_elimination is True

    def test_vcmi_defaults(self):
        vc = VCMIConfig()
        assert vc.enabled is True
        assert vc.target_tier == VCMITier.GOLD
        assert vc.criteria_count == 15
        assert vc.credit_quality_min_score == 70
        assert "verra" in vc.preferred_registries

    def test_assurance_defaults(self):
        ac = AssuranceConfig()
        assert ac.enabled is True
        assert ac.assurance_level == AssuranceLevel.LIMITED
        assert ac.target_assurance_level == AssuranceLevel.REASONABLE
        assert ac.sha256_provenance is True
        assert ac.retention_years == 10


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestConfigValidation:

    def test_base_year_after_reporting_year_raises(self):
        with pytest.raises(ValueError, match="base_year"):
            NetZeroAccelerationConfig(base_year=2030, reporting_year=2025)

    def test_long_term_before_near_term_raises(self):
        with pytest.raises(ValueError):
            PathwayConfig(near_term_target_year=2035, long_term_target_year=2030)

    def test_invalid_sda_sector_raises(self):
        with pytest.raises(ValueError, match="Invalid SDA sectors"):
            PathwayConfig(sda_sectors=["INVALID_SECTOR"])

    def test_invalid_scope3_categories_raises(self):
        with pytest.raises(ValueError, match="Invalid Scope 3 categories"):
            Scope3Config(categories=[0, 16])

    def test_invalid_aggregation_method_raises(self):
        with pytest.raises(ValueError, match="Invalid aggregation methods"):
            TemperatureConfig(aggregation_methods=["INVALID"])

    def test_invalid_workpaper_format_raises(self):
        with pytest.raises(ValueError, match="Invalid workpaper format"):
            AssuranceConfig(workpaper_format="INVALID")

    def test_invalid_currency_raises(self):
        with pytest.raises(ValueError, match="3-letter ISO"):
            FinanceConfig(reporting_currency="TOOLONG")

    def test_invalid_supplier_frequency_raises(self):
        with pytest.raises(ValueError, match="Invalid frequency"):
            SupplierConfig(data_collection_frequency="daily")

    def test_invalid_portfolio_weighting_raises(self):
        with pytest.raises(ValueError, match="Invalid weighting"):
            TemperatureConfig(portfolio_weighting="invalid_method")


# ---------------------------------------------------------------------------
# Validate Config Warnings
# ---------------------------------------------------------------------------


class TestValidateConfigWarnings:

    def test_empty_org_name_warning(self):
        cfg = NetZeroAccelerationConfig()
        warnings = validate_config(cfg)
        assert any("organization_name" in w.lower() or "organization name" in w.lower() for w in warnings)

    def test_heavy_industry_without_sda_warning(self):
        cfg = NetZeroAccelerationConfig(
            sector=SectorClassification.HEAVY_INDUSTRY,
            organization_name="TestCo",
        )
        warnings = validate_config(cfg)
        assert any("SDA" in w for w in warnings)

    def test_financial_services_without_pcaf_warning(self):
        cfg = NetZeroAccelerationConfig(
            sector=SectorClassification.FINANCIAL_SERVICES,
            organization_name="FinCo",
        )
        warnings = validate_config(cfg)
        assert any("PCAF" in w for w in warnings)

    def test_few_scope3_categories_warning(self):
        cfg = NetZeroAccelerationConfig(
            organization_name="TestCo",
            scope3=Scope3Config(categories=[1, 2, 3]),
        )
        warnings = validate_config(cfg)
        assert any("Fewer than 5" in w for w in warnings)

    def test_zero_random_seed_warning(self):
        cfg = NetZeroAccelerationConfig(
            organization_name="TestCo",
            scenario=ScenarioConfig(random_seed=0),
        )
        warnings = validate_config(cfg)
        assert any("seed" in w.lower() for w in warnings)

    def test_no_assurance_progression_warning(self):
        cfg = NetZeroAccelerationConfig(
            organization_name="TestCo",
            assurance=AssuranceConfig(
                assurance_level=AssuranceLevel.LIMITED,
                target_assurance_level=AssuranceLevel.LIMITED,
            ),
        )
        warnings = validate_config(cfg)
        assert any("progression" in w.lower() or "reasonable" in w.lower() for w in warnings)

    def test_pack_config_validate_config_method(self):
        pc = PackConfig()
        warnings = pc.validate_config()
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# Enabled Engines
# ---------------------------------------------------------------------------


class TestEnabledEngines:

    def test_default_enabled_engines(self):
        cfg = NetZeroAccelerationConfig()
        engines = cfg.get_enabled_engines()
        assert "scenario_modeling" in engines
        assert "scope3_activity" in engines
        assert "temperature_scoring" in engines
        assert "variance_decomposition" in engines
        assert "vcmi_validation" in engines
        assert "assurance_workpaper" in engines
        # multi_entity disabled by default
        assert "multi_entity" not in engines

    def test_sda_enabled_adds_engine(self):
        cfg = NetZeroAccelerationConfig(
            pathway=PathwayConfig(sda_enabled=True, sda_sectors=["CEMENT"]),
        )
        engines = cfg.get_enabled_engines()
        assert "sda_pathway" in engines

    def test_multi_entity_enabled_adds_engine(self):
        cfg = NetZeroAccelerationConfig(
            multi_entity=MultiEntityConfig(enabled=True),
        )
        engines = cfg.get_enabled_engines()
        assert "multi_entity" in engines

    def test_engines_are_sorted(self):
        cfg = NetZeroAccelerationConfig()
        engines = cfg.get_enabled_engines()
        assert engines == sorted(engines)


# ---------------------------------------------------------------------------
# Config Hash
# ---------------------------------------------------------------------------


class TestConfigHash:

    def test_hash_is_64_hex(self):
        pc = PackConfig()
        h = pc.get_config_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_config_same_hash(self):
        pc1 = PackConfig()
        pc2 = PackConfig()
        assert pc1.get_config_hash() == pc2.get_config_hash()

    def test_different_config_different_hash(self):
        pc1 = PackConfig()
        pc2 = PackConfig(pack=NetZeroAccelerationConfig(reporting_year=2026))
        assert pc1.get_config_hash() != pc2.get_config_hash()


# ---------------------------------------------------------------------------
# Merge Config
# ---------------------------------------------------------------------------


class TestMergeConfig:

    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_config(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"top": {"a": 1, "b": 2}}
        override = {"top": {"b": 3, "c": 4}}
        result = merge_config(base, override)
        assert result["top"] == {"a": 1, "b": 3, "c": 4}

    def test_base_unchanged(self):
        base = {"a": 1}
        override = {"a": 2}
        merge_config(base, override)
        assert base["a"] == 1


# ---------------------------------------------------------------------------
# Environment Overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:

    def test_parses_simple_key(self):
        os.environ["NET_ZERO_ACCEL_REPORTING_YEAR"] = "2026"
        try:
            overrides = get_env_overrides("NET_ZERO_ACCEL_")
            assert overrides["reporting_year"] == 2026
        finally:
            del os.environ["NET_ZERO_ACCEL_REPORTING_YEAR"]

    def test_parses_nested_key(self):
        os.environ["NET_ZERO_ACCEL_PATHWAY__METHODOLOGY"] = "SDA"
        try:
            overrides = get_env_overrides("NET_ZERO_ACCEL_")
            assert overrides["pathway"]["methodology"] == "SDA"
        finally:
            del os.environ["NET_ZERO_ACCEL_PATHWAY__METHODOLOGY"]

    def test_parses_boolean_true(self):
        os.environ["NET_ZERO_ACCEL_TEST_BOOL"] = "true"
        try:
            overrides = get_env_overrides("NET_ZERO_ACCEL_")
            assert overrides["test_bool"] is True
        finally:
            del os.environ["NET_ZERO_ACCEL_TEST_BOOL"]

    def test_parses_boolean_false(self):
        os.environ["NET_ZERO_ACCEL_TEST_BOOL"] = "false"
        try:
            overrides = get_env_overrides("NET_ZERO_ACCEL_")
            assert overrides["test_bool"] is False
        finally:
            del os.environ["NET_ZERO_ACCEL_TEST_BOOL"]


# ---------------------------------------------------------------------------
# Sector Defaults
# ---------------------------------------------------------------------------


class TestSectorDefaults:

    def test_heavy_industry_defaults(self):
        cfg = get_sector_defaults(SectorClassification.HEAVY_INDUSTRY)
        assert cfg.sector == SectorClassification.HEAVY_INDUSTRY
        assert cfg.pathway.methodology == PathwayMethodology.SDA
        assert cfg.pathway.sda_enabled is True
        assert "CEMENT" in cfg.pathway.sda_sectors or "STEEL" in cfg.pathway.sda_sectors

    def test_financial_services_defaults(self):
        cfg = get_sector_defaults(SectorClassification.FINANCIAL_SERVICES)
        assert cfg.sector == SectorClassification.FINANCIAL_SERVICES
        assert cfg.pathway.methodology == PathwayMethodology.ACA
        assert cfg.scope3.pcaf_enabled is True
        assert cfg.multi_entity.enabled is True

    def test_technology_defaults(self):
        cfg = get_sector_defaults(SectorClassification.TECHNOLOGY)
        assert cfg.sector == SectorClassification.TECHNOLOGY
        assert cfg.pathway.methodology == PathwayMethodology.ACA

    def test_string_sector_input(self):
        cfg = get_sector_defaults("MANUFACTURING")
        assert cfg.sector == SectorClassification.MANUFACTURING

    def test_all_sectors_produce_valid_config(self):
        for sector in SectorClassification:
            cfg = get_sector_defaults(sector)
            assert isinstance(cfg, NetZeroAccelerationConfig)
            assert cfg.sector == sector


# ---------------------------------------------------------------------------
# Preset Loading
# ---------------------------------------------------------------------------


class TestPresetLoading:

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_load_preset_function_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent_preset")

    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PackConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_load_config_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert len(presets) == 8
        assert "heavy_industry" in presets
        assert "technology" in presets
        assert "financial_services" in presets

    def test_list_sda_sectors(self):
        sectors = list_sda_sectors()
        assert len(sectors) == 12
        assert "POWER" in sectors
        assert "CEMENT" in sectors
        assert "STEEL" in sectors

    def test_get_sector_info_enum(self):
        info = get_sector_info(SectorClassification.HEAVY_INDUSTRY)
        assert "name" in info
        assert "recommended_pathway" in info
        assert "sda_sectors" in info
        assert "key_levers" in info

    def test_get_sector_info_string(self):
        info = get_sector_info("HEAVY_INDUSTRY")
        assert info["recommended_pathway"] == "SDA"

    def test_get_sector_info_unknown(self):
        info = get_sector_info("UNKNOWN_SECTOR")
        assert "name" in info

    def test_get_sda_intensity_metric(self):
        assert get_sda_intensity_metric("POWER") == "tCO2e/MWh"
        assert get_sda_intensity_metric("CEMENT") == "tCO2e/tonne clinker"
        assert get_sda_intensity_metric("UNKNOWN") == "tCO2e/unit"

    def test_get_vcmi_tier_thresholds(self):
        silver = get_vcmi_tier_thresholds(VCMITier.SILVER)
        assert silver["foundation_min"] == 60.0
        gold = get_vcmi_tier_thresholds("GOLD")
        assert gold["overall_min"] == 72.0

    def test_get_temperature_regression(self):
        s12 = get_temperature_regression("scope_1_2")
        assert s12["intercept"] == 3.2
        s3 = get_temperature_regression("scope_3")
        assert s3["intercept"] == 3.5
        unknown = get_temperature_regression("unknown")
        assert unknown["intercept"] == 3.2  # falls back to scope_1_2

    def test_get_sbti_reduction_rate_enum(self):
        rates = get_sbti_reduction_rate(TemperatureTarget.CELSIUS_1_5)
        assert rates["scope_1_2_linear_annual"] == 4.2
        assert rates["scope_3_linear_annual"] == 2.5

    def test_get_sbti_reduction_rate_string(self):
        rates = get_sbti_reduction_rate("WELL_BELOW_2")
        assert rates["scope_1_2_linear_annual"] == 2.5

    def test_get_gwp100_known_gases(self):
        assert get_gwp100("CO2") == 1
        assert get_gwp100("CH4") == 27
        assert get_gwp100("N2O") == 273
        assert get_gwp100("SF6") == 25200

    def test_get_gwp100_unknown_gas(self):
        assert get_gwp100("UNKNOWN_GAS") == 0

    def test_get_gwp100_case_insensitive(self):
        assert get_gwp100("co2") == 1
        assert get_gwp100("ch4") == 27


# ---------------------------------------------------------------------------
# Constants Verification
# ---------------------------------------------------------------------------


class TestConstants:

    def test_default_constants(self):
        assert DEFAULT_BASE_YEAR == 2021
        assert DEFAULT_NEAR_TERM_YEAR == 2030
        assert DEFAULT_LONG_TERM_YEAR == 2050
        assert DEFAULT_MONTE_CARLO_RUNS == 1000
        assert DEFAULT_MAX_SUPPLIERS == 50000
        assert DEFAULT_MAX_ENTITIES == 50
        assert DEFAULT_SCOPE3_CATEGORIES == list(range(1, 16))

    def test_supported_presets_count(self):
        assert len(SUPPORTED_PRESETS) == 8

    def test_sda_sectors_count(self):
        assert len(SDA_SECTORS) == 12

    def test_sda_intensity_metrics_match_sectors(self):
        assert set(SDA_INTENSITY_METRICS.keys()) == set(SDA_SECTORS.keys())

    def test_ipcc_gwp100_count(self):
        assert len(IPCC_AR6_GWP100) == 11
        assert IPCC_AR6_GWP100["CO2"] == 1

    def test_sbti_reduction_rates_count(self):
        assert len(SBTI_REDUCTION_RATES) == 3
        assert "CELSIUS_1_5" in SBTI_REDUCTION_RATES

    def test_sbti_coverage_thresholds(self):
        assert SBTI_COVERAGE_THRESHOLDS["scope_1_near_term_pct"] == 95.0
        assert SBTI_COVERAGE_THRESHOLDS["scope_3_near_term_pct"] == 67.0

    def test_sector_scope3_priority_sectors(self):
        assert len(SECTOR_SCOPE3_PRIORITY) == 8
        for sector_key, cats in SECTOR_SCOPE3_PRIORITY.items():
            assert len(cats) == 15

    def test_priority_scope3_by_sector(self):
        assert len(PRIORITY_SCOPE3_BY_SECTOR) == 8
        assert 1 in PRIORITY_SCOPE3_BY_SECTOR["HEAVY_INDUSTRY"]

    def test_sector_info_count(self):
        assert len(SECTOR_INFO) == 8
        for key, info in SECTOR_INFO.items():
            assert "name" in info
            assert "recommended_pathway" in info

    def test_vcmi_tier_thresholds(self):
        assert len(VCMI_TIER_THRESHOLDS) == 3
        assert VCMI_TIER_THRESHOLDS["SILVER"]["overall_min"] == 55.0
        assert VCMI_TIER_THRESHOLDS["PLATINUM"]["overall_min"] == 86.0

    def test_temperature_regression(self):
        assert len(TEMPERATURE_REGRESSION) == 2
        assert TEMPERATURE_REGRESSION["scope_1_2"]["intercept"] == 3.2


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:

    def test_scenario_type_values(self):
        assert ScenarioType.BAU.value == "BAU"
        assert ScenarioType.MODERATE.value == "MODERATE"
        assert ScenarioType.AMBITIOUS.value == "AMBITIOUS"
        assert ScenarioType.CUSTOM.value == "CUSTOM"

    def test_pathway_methodology_values(self):
        assert PathwayMethodology.ACA.value == "ACA"
        assert PathwayMethodology.SDA.value == "SDA"
        assert PathwayMethodology.FLAG.value == "FLAG"

    def test_supplier_tier_values(self):
        assert SupplierTier.INFORM.value == "INFORM"
        assert SupplierTier.COLLABORATE.value == "COLLABORATE"

    def test_scope_category_values(self):
        assert ScopeCategory.SCOPE_1.value == "SCOPE_1"
        assert ScopeCategory.SCOPE_3.value == "SCOPE_3"

    def test_finance_instrument_values(self):
        assert FinanceInstrument.GREEN_BOND.value == "GREEN_BOND"
        assert len(FinanceInstrument) == 6

    def test_temperature_target_values(self):
        assert TemperatureTarget.CELSIUS_1_5.value == "CELSIUS_1_5"
        assert TemperatureTarget.WELL_BELOW_2.value == "WELL_BELOW_2"
        assert TemperatureTarget.CELSIUS_2.value == "CELSIUS_2"

    def test_decomposition_method_values(self):
        assert DecompositionMethod.LMDI.value == "LMDI"
        assert DecompositionMethod.SDA_DECOMP.value == "SDA_DECOMP"

    def test_entity_scope_values(self):
        assert EntityScope.EQUITY_SHARE.value == "EQUITY_SHARE"
        assert EntityScope.OPERATIONAL_CONTROL.value == "OPERATIONAL_CONTROL"

    def test_vcmi_tier_values(self):
        assert VCMITier.SILVER.value == "SILVER"
        assert VCMITier.GOLD.value == "GOLD"
        assert VCMITier.PLATINUM.value == "PLATINUM"

    def test_assurance_level_values(self):
        assert AssuranceLevel.LIMITED.value == "LIMITED"
        assert AssuranceLevel.REASONABLE.value == "REASONABLE"

    def test_sector_classification_values(self):
        assert len(SectorClassification) == 8
        assert SectorClassification.HEAVY_INDUSTRY.value == "HEAVY_INDUSTRY"
        assert SectorClassification.TECHNOLOGY.value == "TECHNOLOGY"
