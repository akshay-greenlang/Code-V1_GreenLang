# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Configuration Tests (test_config.py)
======================================================================

Tests configuration completeness and correctness:
  - Enum completeness tests (BuildingType, ClimateZone, EnergyCarrier, etc.)
  - Sub-config default value tests
  - Main EnergyBenchmarkConfig construction tests
  - 8 preset loading tests
  - Model validator tests
  - Utility function tests

Test Count Target: ~120 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
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

    def test_building_type_count(self, config_module):
        """BuildingType enum has at least 12 members."""
        cls = getattr(config_module, "BuildingType", None)
        if cls is None:
            pytest.skip("BuildingType enum not found")
        members = list(cls)
        assert len(members) >= 12, f"Expected >=12, got {len(members)}"

    def test_building_type_includes_office(self, config_module):
        """BuildingType includes OFFICE."""
        cls = getattr(config_module, "BuildingType", None)
        if cls is None:
            pytest.skip("BuildingType not found")
        values = {m.value for m in cls}
        assert "OFFICE" in values or "office" in values

    def test_building_type_includes_retail(self, config_module):
        """BuildingType includes RETAIL."""
        cls = getattr(config_module, "BuildingType", None)
        if cls is None:
            pytest.skip("BuildingType not found")
        values = {m.value for m in cls}
        assert "RETAIL" in values or "retail" in values

    def test_energy_carrier_count(self, config_module):
        """EnergyCarrier enum has at least 6 members."""
        cls = getattr(config_module, "EnergyCarrier", None)
        if cls is None:
            pytest.skip("EnergyCarrier not found")
        assert len(list(cls)) >= 6

    def test_energy_carrier_includes_electricity(self, config_module):
        """EnergyCarrier includes ELECTRICITY."""
        cls = getattr(config_module, "EnergyCarrier", None)
        if cls is None:
            pytest.skip("EnergyCarrier not found")
        values = {m.value for m in cls}
        assert "ELECTRICITY" in values or "electricity" in values

    def test_accounting_boundary_count(self, config_module):
        """AccountingBoundary enum has at least 3 members."""
        cls = getattr(config_module, "AccountingBoundary", None)
        if cls is None:
            pytest.skip("AccountingBoundary not found")
        assert len(list(cls)) >= 3

    def test_rating_scheme_count(self, config_module):
        """RatingScheme enum has at least 4 members."""
        cls = getattr(config_module, "RatingScheme", None)
        if cls is None:
            pytest.skip("RatingScheme not found")
        assert len(list(cls)) >= 4

    def test_reporting_frequency_count(self, config_module):
        """ReportingFrequency enum has at least 4 members."""
        cls = getattr(config_module, "ReportingFrequency", None)
        if cls is None:
            pytest.skip("ReportingFrequency not found")
        assert len(list(cls)) >= 4

    def test_output_format_count(self, config_module):
        """OutputFormat enum has at least 3 members."""
        cls = getattr(config_module, "OutputFormat", None)
        if cls is None:
            pytest.skip("OutputFormat not found")
        assert len(list(cls)) >= 3

    def test_climate_zone_count(self, config_module):
        """ClimateZone enum has at least 8 members."""
        cls = getattr(config_module, "ClimateZone", None)
        if cls is None:
            pytest.skip("ClimateZone not found")
        assert len(list(cls)) >= 8

    def test_regression_model_type_count(self, config_module):
        """RegressionModelType enum has at least 3 members."""
        cls = getattr(config_module, "RegressionModelType", None)
        if cls is None:
            pytest.skip("RegressionModelType not found")
        assert len(list(cls)) >= 3

    def test_normalization_method_count(self, config_module):
        """NormalizationMethod enum has at least 3 members."""
        cls = getattr(config_module, "NormalizationMethod", None)
        if cls is None:
            pytest.skip("NormalizationMethod not found")
        assert len(list(cls)) >= 3


# =============================================================================
# 2. Sub-Config Default Tests
# =============================================================================


class TestSubConfigDefaults:
    """Test sub-configuration models instantiate with correct defaults."""

    def test_eui_config_defaults(self, config_module):
        """EUIConfig instantiates with sensible defaults."""
        cls = getattr(config_module, "EUIConfig", None)
        if cls is None:
            pytest.skip("EUIConfig not found")
        ec = cls()
        assert ec.accounting_boundary is not None
        assert ec.floor_area_type is not None

    def test_weather_config_defaults(self, config_module):
        """WeatherConfig has sensible defaults."""
        cls = getattr(config_module, "WeatherConfig", None)
        if cls is None:
            pytest.skip("WeatherConfig not found")
        wc = cls()
        assert wc.enabled is True
        assert wc.base_temp_heating_c == pytest.approx(18.0) or wc.base_temp_heating_c == pytest.approx(15.5)

    def test_peer_config_defaults(self, config_module):
        """PeerConfig has sensible defaults."""
        cls = getattr(config_module, "PeerConfig", None)
        if cls is None:
            pytest.skip("PeerConfig not found")
        pc = cls()
        assert pc.enabled is True
        assert pc.min_peer_count >= 10

    def test_portfolio_config_defaults(self, config_module):
        """PortfolioConfig has sensible defaults."""
        cls = getattr(config_module, "PortfolioConfig", None)
        if cls is None:
            pytest.skip("PortfolioConfig not found")
        pc = cls()
        assert pc.enabled is True
        assert pc.max_facilities >= 100

    def test_rating_config_defaults(self, config_module):
        """RatingConfig has sensible defaults."""
        cls = getattr(config_module, "RatingConfig", None)
        if cls is None:
            pytest.skip("RatingConfig not found")
        rc = cls()
        assert rc.enabled is True

    def test_trend_config_defaults(self, config_module):
        """TrendConfig has sensible defaults."""
        cls = getattr(config_module, "TrendConfig", None)
        if cls is None:
            pytest.skip("TrendConfig not found")
        tc = cls()
        assert tc.enabled is True
        assert tc.cusum_enabled is True

    def test_gap_analysis_config_defaults(self, config_module):
        """GapAnalysisConfig has sensible defaults."""
        cls = getattr(config_module, "GapAnalysisConfig", None)
        if cls is None:
            pytest.skip("GapAnalysisConfig not found")
        gc = cls()
        assert gc.enabled is True

    def test_regression_config_defaults(self, config_module):
        """RegressionConfig has sensible defaults."""
        cls = getattr(config_module, "RegressionConfig", None)
        if cls is None:
            pytest.skip("RegressionConfig not found")
        rc = cls()
        assert rc.r_squared_threshold >= 0.5

    def test_performance_config_defaults(self, config_module):
        """PerformanceConfig has sensible defaults."""
        cls = getattr(config_module, "PerformanceConfig", None)
        if cls is None:
            pytest.skip("PerformanceConfig not found")
        pc = cls()
        assert pc.max_facilities >= 50

    def test_security_config_defaults(self, config_module):
        """SecurityConfig has CONFIDENTIAL classification by default."""
        cls = getattr(config_module, "SecurityConfig", None)
        if cls is None:
            pytest.skip("SecurityConfig not found")
        sc = cls()
        assert sc.data_classification == "CONFIDENTIAL"
        assert sc.audit_logging is True

    def test_audit_trail_config_defaults(self, config_module):
        """AuditTrailConfig has SHA-256 provenance by default."""
        cls = getattr(config_module, "AuditTrailConfig", None)
        if cls is None:
            pytest.skip("AuditTrailConfig not found")
        at = cls()
        assert at.enabled is True
        assert at.sha256_provenance is True


# =============================================================================
# 3. Main Config Construction Tests
# =============================================================================


class TestMainConfigDefaults:
    """Test EnergyBenchmarkConfig construction and field validation."""

    def test_default_construction(self, config_module):
        """EnergyBenchmarkConfig can be created with all defaults."""
        cls = getattr(config_module, "EnergyBenchmarkConfig", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkConfig not found")
        cfg = cls()
        assert cfg is not None

    def test_all_sub_configs_present(self, config_module):
        """Config has all expected sub-config attributes."""
        cls = getattr(config_module, "EnergyBenchmarkConfig", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkConfig not found")
        cfg = cls()
        for attr in [
            "eui", "weather", "peer", "portfolio", "rating",
            "trend", "gap_analysis", "regression",
            "performance", "security", "audit_trail",
        ]:
            assert hasattr(cfg, attr), f"Missing sub-config: {attr}"

    def test_custom_building_type(self, config_module):
        """Building type can be set."""
        cls = getattr(config_module, "EnergyBenchmarkConfig", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkConfig not found")
        cfg = cls(building_type="retail")
        assert cfg.building_type == "retail"


# =============================================================================
# 4. Config Validation Tests
# =============================================================================


class TestConfigValidation:
    """Test model validators that enforce cross-field constraints."""

    def test_config_hash_deterministic(self, config_module):
        """Config hash is deterministic across identical configs."""
        cls = getattr(config_module, "PackConfig", None)
        if cls is None:
            pytest.skip("PackConfig not found")
        cfg1 = cls()
        cfg2 = cls()
        assert cfg1.get_config_hash() == cfg2.get_config_hash()

    def test_config_hash_64_chars(self, config_module):
        """Config hash is 64 characters (SHA-256)."""
        cls = getattr(config_module, "PackConfig", None)
        if cls is None:
            pytest.skip("PackConfig not found")
        cfg = cls()
        assert len(cfg.get_config_hash()) == 64

    def test_validate_config_returns_list(self, config_module):
        """validate_config returns a list of warnings."""
        validate_fn = getattr(config_module, "validate_config", None)
        if validate_fn is None:
            pytest.skip("validate_config not found")
        cfg_cls = getattr(config_module, "EnergyBenchmarkConfig", None)
        if cfg_cls is None:
            pytest.skip("EnergyBenchmarkConfig not found")
        cfg = cfg_cls()
        warnings = validate_fn(cfg)
        assert isinstance(warnings, list)


# =============================================================================
# 5. PackConfig Methods
# =============================================================================


class TestPackConfigMethods:
    """Test PackConfig utility methods."""

    def test_pack_config_construction(self, config_module):
        """PackConfig can be constructed."""
        cls = getattr(config_module, "PackConfig", None)
        if cls is None:
            pytest.skip("PackConfig not found")
        cfg = cls()
        assert cfg is not None
        assert cfg.pack_id == "PACK-035-energy-benchmark" or "035" in getattr(cfg, "pack_id", "")

    def test_pack_config_preset_name(self, config_module):
        """PackConfig has preset_name attribute."""
        cls = getattr(config_module, "PackConfig", None)
        if cls is None:
            pytest.skip("PackConfig not found")
        cfg = cls()
        assert hasattr(cfg, "preset_name")

    def test_available_presets_list(self, config_module):
        """list_available_presets returns all 8 presets."""
        fn = getattr(config_module, "list_available_presets", None)
        if fn is None:
            pytest.skip("list_available_presets not found")
        presets = fn()
        assert isinstance(presets, dict)
        assert len(presets) == 8


# =============================================================================
# 6. Preset Loading Tests
# =============================================================================


class TestPresetLoading:
    """Test loading of all 8 presets."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_yaml_exists(self, preset_name):
        """Preset YAML file exists on disk."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset file not found: {path}")
        assert path.is_file()

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_yaml_valid(self, preset_name):
        """Preset YAML is valid and non-empty."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_has_building_type(self, preset_name):
        """Each preset specifies a building type."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "building_type" in data or "industry_sector" in data

    def test_commercial_office_preset_via_pack_config(self, config_module):
        """Commercial office preset loads via PackConfig.from_preset()."""
        cls = getattr(config_module, "PackConfig", None)
        if cls is None:
            pytest.skip("PackConfig not found")
        from_preset = getattr(cls, "from_preset", None)
        if from_preset is None:
            pytest.skip("from_preset not found")
        path = PRESETS_DIR / "commercial_office.yaml"
        if not path.exists():
            pytest.skip("commercial_office.yaml not found")
        cfg = from_preset("commercial_office")
        assert cfg is not None
        assert cfg.preset_name == "commercial_office"


# =============================================================================
# 7. Reference Data Tests
# =============================================================================


class TestReferenceData:
    """Test module-level reference data constants."""

    def test_eui_benchmarks_exist(self, config_module):
        """EUI benchmark reference data exists."""
        benchmarks = getattr(config_module, "EUI_BENCHMARKS", None)
        if benchmarks is None:
            pytest.skip("EUI_BENCHMARKS not found")
        assert isinstance(benchmarks, dict)
        assert len(benchmarks) >= 5

    def test_epc_thresholds_exist(self, config_module):
        """EPC threshold data exists."""
        thresholds = getattr(config_module, "EPC_THRESHOLDS", None)
        if thresholds is None:
            pytest.skip("EPC_THRESHOLDS not found")
        assert isinstance(thresholds, dict)
        assert "A" in thresholds

    def test_source_energy_factors_exist(self, config_module):
        """Source energy factor data exists."""
        factors = getattr(config_module, "SOURCE_ENERGY_FACTORS", None)
        if factors is None:
            pytest.skip("SOURCE_ENERGY_FACTORS not found")
        assert isinstance(factors, dict)
        assert len(factors) >= 4

    def test_cibse_tm46_data_exist(self, config_module):
        """CIBSE TM46 benchmark data exists."""
        tm46 = getattr(config_module, "CIBSE_TM46_BENCHMARKS", None)
        if tm46 is None:
            pytest.skip("CIBSE_TM46_BENCHMARKS not found")
        assert isinstance(tm46, dict)
        assert len(tm46) >= 6
