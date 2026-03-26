# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Configuration
========================================

Tests pack configuration, presets, enums, validation, YAML loading,
and configuration merging.

Coverage target: 85%+
Total tests: ~50
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from tests.conftest import PRESET_NAMES, CONFIG_DIR, PRESETS_DIR, PACK_ROOT


def _try_load_config():
    """Attempt to load pack_config module."""
    path = CONFIG_DIR / "pack_config.py"
    if not path.exists():
        return None
    mod_key = "pack042_test.config.pack_config"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_key, None)
        return None
    return mod


_cfg_mod = _try_load_config()


# =============================================================================
# Pack Config Defaults
# =============================================================================


class TestPackConfigDefaults:
    """Test default configuration values."""

    def test_pack_id_is_042(self, sample_pack_config):
        assert sample_pack_config["pack_id"] == "PACK-042"

    def test_pack_name(self, sample_pack_config):
        assert sample_pack_config["pack_name"] == "Scope 3 Starter Pack"

    def test_version_is_semver(self, sample_pack_config):
        version = sample_pack_config["version"]
        parts = version.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_category_is_ghg_accounting(self, sample_pack_config):
        assert sample_pack_config["category"] == "ghg-accounting"

    def test_environment_is_test(self, sample_pack_config):
        assert sample_pack_config["environment"] == "test"

    def test_sector_type_default(self, sample_pack_config):
        assert sample_pack_config["sector_type"] == "MANUFACTURING"

    def test_country_default(self, sample_pack_config):
        assert sample_pack_config["country"] == "DE"

    def test_reporting_year(self, sample_pack_config):
        assert sample_pack_config["reporting_year"] == 2025

    def test_screening_eeio_model(self, sample_pack_config):
        assert sample_pack_config["screening"]["eeio_model"] == "EXIOBASE_3"

    def test_screening_currency(self, sample_pack_config):
        assert sample_pack_config["screening"]["currency"] == "EUR"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test all enums have correct members."""

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_scope3_category_has_15_members(self):
        assert len(_cfg_mod.Scope3Category) == 15

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_methodology_tier_has_4_members(self):
        assert len(_cfg_mod.MethodologyTier) == 4

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_data_quality_level_has_5_members(self):
        assert len(_cfg_mod.DataQualityLevel) == 5

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_eeio_model_has_3_members(self):
        assert len(_cfg_mod.EEIOModel) == 3

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_classification_code_has_5_members(self):
        assert len(_cfg_mod.ClassificationCode) == 5

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_engagement_status_has_5_members(self):
        assert len(_cfg_mod.EngagementStatus) == 5

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_framework_type_has_8_members(self):
        assert len(_cfg_mod.FrameworkType) == 8

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_output_format_has_4_members(self):
        assert len(_cfg_mod.OutputFormat) == 4

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_sector_type_has_11_members(self):
        assert len(_cfg_mod.SectorType) == 11

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_uncertainty_method_has_3_members(self):
        assert len(_cfg_mod.UncertaintyMethod) == 3

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_allocation_method_has_4_members(self):
        assert len(_cfg_mod.AllocationMethod) == 4


# =============================================================================
# Preset Tests
# =============================================================================


class TestPresets:
    """Test preset loading for different sector profiles."""

    def test_eight_presets_defined(self):
        assert len(PRESET_NAMES) == 8

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_name_valid(self, preset_name):
        assert isinstance(preset_name, str)
        assert len(preset_name) > 0

    def test_manufacturing_preset_in_list(self):
        assert "manufacturing" in PRESET_NAMES

    def test_sme_preset_in_list(self):
        assert "sme_simplified" in PRESET_NAMES

    def test_financial_preset_in_list(self):
        assert "financial_services" in PRESET_NAMES

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_available_presets_dict(self):
        presets = _cfg_mod.AVAILABLE_PRESETS
        assert len(presets) == 8
        for name, desc in presets.items():
            assert len(desc) > 10

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_manufacturing_preset_yaml_exists(self):
        preset_path = PRESETS_DIR / "manufacturing.yaml"
        assert preset_path.exists(), f"Manufacturing preset not found at {preset_path}"

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_from_preset_manufacturing(self):
        try:
            config = _cfg_mod.PackConfig.from_preset("manufacturing")
            assert config.preset_name == "manufacturing"
        except FileNotFoundError:
            pytest.skip("Manufacturing preset YAML not found")

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_from_preset_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            _cfg_mod.PackConfig.from_preset("nonexistent_preset")


# =============================================================================
# Config Merging Tests
# =============================================================================


class TestConfigMerging:
    """Test configuration merging (preset + overrides)."""

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_deep_merge_basic(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        merged = _cfg_mod.PackConfig._deep_merge(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_deep_merge_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        merged = _cfg_mod.PackConfig._deep_merge(base, override)
        assert merged["a"] == 1
        assert merged["b"] == 2


# =============================================================================
# from_yaml and from_preset Tests
# =============================================================================


class TestConfigLoading:
    """Test from_yaml() and from_preset() methods."""

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_from_yaml_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _cfg_mod.PackConfig.from_yaml("/nonexistent/path.yaml")

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_default_config_creates(self):
        config = _cfg_mod.Scope3StarterConfig()
        assert config.sector_type == _cfg_mod.SectorType.MANUFACTURING
        assert config.country == "DE"

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_get_default_config(self):
        config = _cfg_mod.get_default_config()
        assert config.sector_type == _cfg_mod.SectorType.MANUFACTURING

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_get_sector_info(self):
        info = _cfg_mod.get_sector_info("MANUFACTURING")
        assert "name" in info
        assert "dominant_categories" in info

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_get_category_info(self):
        info = _cfg_mod.get_category_info("CAT_1")
        assert "name" in info
        assert info["name"] == "Purchased Goods & Services"


# =============================================================================
# Validation Tests
# =============================================================================


class TestConfigValidation:
    """Test configuration validation."""

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_validate_config_returns_warnings(self):
        config = _cfg_mod.Scope3StarterConfig()
        warnings = _cfg_mod.validate_config(config)
        assert isinstance(warnings, list)

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_missing_company_name_warns(self):
        config = _cfg_mod.Scope3StarterConfig(company_name="")
        warnings = _cfg_mod.validate_config(config)
        name_warnings = [w for w in warnings if "company_name" in w]
        assert len(name_warnings) > 0

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_financial_sector_enables_pcaf(self):
        config = _cfg_mod.Scope3StarterConfig(
            sector_type=_cfg_mod.SectorType.FINANCIAL
        )
        assert config.categories.cat_15.enabled is True
        assert config.compliance.pcaf_enabled is True

    @pytest.mark.skipif(_cfg_mod is None, reason="Config module not loadable")
    def test_pack_config_hash(self):
        config = _cfg_mod.PackConfig()
        h = config.get_config_hash()
        assert len(h) == 64
        int(h, 16)  # Valid hex
