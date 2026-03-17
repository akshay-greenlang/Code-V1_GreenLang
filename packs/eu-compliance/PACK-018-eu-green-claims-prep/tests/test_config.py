# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Configuration Tests
==========================================================

Tests for PackConfig, ClaimScope enum, CommunicationChannel enum,
preset file existence, YAML loading, validation, engine config.

Target: ~30 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import (
    _load_config_module,
    CONFIG_DIR,
    PRESETS_DIR,
    PRESET_NAMES,
    PACK_ROOT,
)


# ---------------------------------------------------------------------------
# Module-scoped config loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the pack_config module."""
    return _load_config_module()


# ===========================================================================
# Config Module Loading Tests
# ===========================================================================


class TestConfigModuleLoading:
    """Tests for config module loading."""

    def test_config_module_loads(self):
        """pack_config module loads successfully."""
        mod = _load_config_module()
        assert mod is not None

    def test_config_module_has_pack_config(self, mod):
        """Module exports PackConfig class."""
        assert hasattr(mod, "PackConfig")

    def test_config_module_has_all_engines(self, mod):
        """Module exports ALL_ENGINES list."""
        assert hasattr(mod, "ALL_ENGINES")
        assert isinstance(mod.ALL_ENGINES, list)
        assert len(mod.ALL_ENGINES) >= 6

    def test_config_module_has_all_workflows(self, mod):
        """Module exports ALL_WORKFLOWS list."""
        assert hasattr(mod, "ALL_WORKFLOWS")
        assert isinstance(mod.ALL_WORKFLOWS, list)
        assert len(mod.ALL_WORKFLOWS) >= 5

    def test_config_module_has_available_presets(self, mod):
        """Module exports AVAILABLE_PRESETS dict."""
        assert hasattr(mod, "AVAILABLE_PRESETS")
        assert isinstance(mod.AVAILABLE_PRESETS, dict)
        assert len(mod.AVAILABLE_PRESETS) >= 4


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestConfigEnums:
    """Tests for configuration enums."""

    def test_claim_scope_enum_exists(self, mod):
        """ClaimScope enum exists in config module."""
        assert hasattr(mod, "ClaimScope")

    def test_claim_scope_product(self, mod):
        """ClaimScope includes PRODUCT."""
        assert mod.ClaimScope.PRODUCT.value == "PRODUCT"

    def test_claim_scope_corporate(self, mod):
        """ClaimScope includes CORPORATE."""
        assert mod.ClaimScope.CORPORATE.value == "CORPORATE"

    def test_claim_scope_both(self, mod):
        """ClaimScope includes BOTH."""
        assert mod.ClaimScope.BOTH.value == "BOTH"

    def test_claim_scope_count(self, mod):
        """ClaimScope has exactly 3 values."""
        assert len(mod.ClaimScope) == 3

    def test_communication_channel_enum_exists(self, mod):
        """CommunicationChannel enum exists in config module."""
        assert hasattr(mod, "CommunicationChannel")

    def test_communication_channel_count(self, mod):
        """CommunicationChannel has exactly 7 values."""
        assert len(mod.CommunicationChannel) == 7

    def test_communication_channel_packaging(self, mod):
        """CommunicationChannel includes PACKAGING."""
        assert mod.CommunicationChannel.PACKAGING.value == "PACKAGING"

    def test_communication_channel_website(self, mod):
        """CommunicationChannel includes WEBSITE."""
        assert mod.CommunicationChannel.WEBSITE.value == "WEBSITE"


# ===========================================================================
# PackConfig Model Tests
# ===========================================================================


class TestPackConfigModel:
    """Tests for PackConfig Pydantic model."""

    def test_create_default_pack_config(self, mod):
        """PackConfig can be created with defaults."""
        config = mod.PackConfig()
        assert config.pack_name == "PACK-018-eu-green-claims-prep"
        assert config.version == "1.0.0"

    def test_pack_config_default_sector(self, mod):
        """PackConfig defaults sector to MANUFACTURING."""
        config = mod.PackConfig()
        assert config.sector == "MANUFACTURING"

    def test_pack_config_default_retention_years(self, mod):
        """PackConfig defaults evidence_retention_years to 5."""
        config = mod.PackConfig()
        assert config.evidence_retention_years == 5

    def test_pack_config_default_claim_scope(self, mod):
        """PackConfig defaults claim_scope to BOTH."""
        config = mod.PackConfig()
        assert config.claim_scope == mod.ClaimScope.BOTH

    def test_pack_config_has_config_hash(self, mod):
        """PackConfig has config_hash property."""
        config = mod.PackConfig()
        assert len(config.config_hash) == 64

    def test_pack_config_validate_method(self, mod):
        """PackConfig validate method returns list of warnings."""
        config = mod.PackConfig()
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_pack_config_get_engine_config(self, mod):
        """PackConfig get_engine_config returns dict."""
        config = mod.PackConfig()
        engine_cfg = config.get_engine_config("claim_substantiation")
        assert isinstance(engine_cfg, dict)
        assert "engine" in engine_cfg
        assert engine_cfg["engine"] == "claim_substantiation"

    def test_pack_config_get_workflow_config(self, mod):
        """PackConfig get_workflow_config returns dict."""
        config = mod.PackConfig()
        wf_cfg = config.get_workflow_config()
        assert isinstance(wf_cfg, dict)
        assert "workflows" in wf_cfg

    def test_pack_config_to_dict(self, mod):
        """PackConfig to_dict returns dict with config_hash."""
        config = mod.PackConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "config_hash" in d

    def test_pack_config_greenwashing_threshold(self, mod):
        """PackConfig default greenwashing_risk_threshold is 50."""
        config = mod.PackConfig()
        assert config.greenwashing_risk_threshold == Decimal("50")

    def test_pack_config_from_preset(self, mod):
        """PackConfig can load from preset if file exists."""
        preset_path = PRESETS_DIR / "manufacturing.yaml"
        if not preset_path.exists():
            pytest.skip("manufacturing.yaml preset not found")
        config = mod.PackConfig.from_preset("manufacturing")
        assert config is not None


# ===========================================================================
# Preset File Existence Tests
# ===========================================================================


class TestPresetFiles:
    """Tests for preset YAML file existence."""

    @pytest.mark.parametrize("preset_name", [
        "manufacturing", "retail", "financial_services", "energy",
    ])
    def test_preset_yaml_exists(self, preset_name):
        """Preset YAML file exists on disk."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        assert path.exists(), f"Preset file missing: {preset_name}.yaml"

    def test_preset_names_list(self):
        """PRESET_NAMES has at least 4 entries."""
        assert len(PRESET_NAMES) >= 4

    def test_preset_names_includes_manufacturing(self):
        """PRESET_NAMES includes 'manufacturing'."""
        assert "manufacturing" in PRESET_NAMES

    def test_preset_names_includes_retail(self):
        """PRESET_NAMES includes 'retail'."""
        assert "retail" in PRESET_NAMES

    def test_preset_names_includes_sme(self):
        """PRESET_NAMES includes 'sme'."""
        assert "sme" in PRESET_NAMES


# ===========================================================================
# Config File Existence Tests
# ===========================================================================


class TestConfigFileExistence:
    """Tests for configuration file existence."""

    def test_config_dir_exists(self):
        """Config directory exists."""
        assert CONFIG_DIR.exists()

    def test_pack_config_py_exists(self):
        """pack_config.py exists in config directory."""
        assert (CONFIG_DIR / "pack_config.py").exists()

    def test_presets_dir_exists(self):
        """Presets directory exists."""
        assert PRESETS_DIR.exists()
