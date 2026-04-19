# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Presets.

Tests all 6 presets loading, tier-specific validation, sector
configurations, and override merging.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~400 lines, 55+ tests
"""

import sys
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from config import (
    PackConfig,
    SUPPORTED_PRESETS,
    load_preset,
    list_available_presets,
    validate_config,
)

PRESET_DIR = _PACK_ROOT / "config" / "presets"

PRESET_NAMES = [
    "micro_business",
    "small_business",
    "medium_business",
    "service_sme",
    "manufacturing_sme",
    "retail_sme",
]


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(params=PRESET_NAMES, ids=PRESET_NAMES)
def preset_name(request) -> str:
    return request.param


# ========================================================================
# Individual Preset Loading
# ========================================================================


class TestMicroBusinessPreset:
    def test_micro_business_loads(self):
        config = load_preset("micro_business")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_micro_business_tier(self):
        config = load_preset("micro_business")
        assert config.pack.organization.sme_size.value == "MICRO"

    def test_micro_business_config_has_org(self):
        config = load_preset("micro_business")
        assert config.pack.organization is not None


class TestSmallBusinessPreset:
    def test_small_business_loads(self):
        config = load_preset("small_business")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_small_business_tier(self):
        config = load_preset("small_business")
        assert config.pack.organization.sme_size.value == "SMALL"


class TestMediumBusinessPreset:
    def test_medium_business_loads(self):
        config = load_preset("medium_business")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_medium_business_tier(self):
        config = load_preset("medium_business")
        assert config.pack.organization.sme_size.value == "MEDIUM"


class TestServiceSMEPreset:
    def test_service_sme_loads(self):
        config = load_preset("service_sme")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_service_sme_sector(self):
        config = load_preset("service_sme")
        sector = config.pack.organization.sector.value
        assert sector in ("SERVICES", "TECHNOLOGY")


class TestManufacturingSMEPreset:
    def test_manufacturing_sme_loads(self):
        config = load_preset("manufacturing_sme")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_manufacturing_sme_sector(self):
        config = load_preset("manufacturing_sme")
        assert config.pack.organization.sector.value == "MANUFACTURING"


class TestRetailSMEPreset:
    def test_retail_sme_loads(self):
        config = load_preset("retail_sme")
        assert config is not None
        assert isinstance(config, PackConfig)

    def test_retail_sme_sector(self):
        config = load_preset("retail_sme")
        assert config.pack.organization.sector.value == "RETAIL"


# ========================================================================
# All Presets Valid YAML
# ========================================================================


class TestAllPresetsValidYAML:
    def test_preset_yaml_exists(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        assert yaml_path.exists(), f"Missing preset file: {yaml_path}"

    def test_preset_yaml_parses(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_preset_yaml_has_organization(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "organization" in data

    def test_preset_yaml_has_sme_size(self, preset_name):
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        org = data.get("organization", {})
        assert "sme_size" in org


# ========================================================================
# Preset Override Merge
# ========================================================================


class TestPresetOverrideMerge:
    def test_override_organization_name(self):
        config = load_preset(
            "micro_business",
            overrides={"organization": {"name": "OverrideCo"}},
        )
        assert config.pack.organization.name == "OverrideCo"

    def test_override_reporting_year(self):
        config = load_preset(
            "small_business",
            overrides={"reporting_year": 2024},
        )
        assert config.pack.reporting_year == 2024

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            load_preset("nonexistent_preset")

    def test_override_target_reduction(self):
        config = load_preset(
            "micro_business",
            overrides={"target": {"near_term_reduction_pct": 50.0}},
        )
        assert config.pack.target.near_term_reduction_pct == 50.0


# ========================================================================
# SUPPORTED_PRESETS Constant
# ========================================================================


class TestSupportedPresetsConstant:
    def test_supported_presets_has_6_entries(self):
        assert len(SUPPORTED_PRESETS) == 6

    def test_all_preset_names_present(self):
        for name in PRESET_NAMES:
            assert name in SUPPORTED_PRESETS

    def test_supported_presets_have_descriptions(self):
        for name, desc in SUPPORTED_PRESETS.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


# ========================================================================
# Utility Functions
# ========================================================================


class TestUtilityFunctions:
    def test_list_available_presets(self):
        presets = list_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 6

    def test_validate_config_on_preset(self, preset_name):
        config = load_preset(preset_name)
        result = validate_config(config.pack)
        assert isinstance(result, list)


# ========================================================================
# PackConfig.from_preset
# ========================================================================


class TestPackConfigFromPreset:
    def test_from_preset_returns_pack_config(self, preset_name):
        config = PackConfig.from_preset(preset_name)
        assert isinstance(config, PackConfig)

    def test_from_preset_with_overrides(self):
        config = PackConfig.from_preset(
            "small_business",
            overrides={"organization": {"name": "TechOverride"}},
        )
        assert config.pack.organization.name == "TechOverride"


# ========================================================================
# Tier-Specific Validation
# ========================================================================


class TestTierSpecificValidation:
    @pytest.mark.parametrize("preset,expected_tier", [
        ("micro_business", "MICRO"),
        ("small_business", "SMALL"),
        ("medium_business", "MEDIUM"),
    ])
    def test_preset_tier_matches(self, preset, expected_tier):
        config = load_preset(preset)
        assert config.pack.organization.sme_size.value == expected_tier

    @pytest.mark.parametrize("preset", PRESET_NAMES)
    def test_preset_has_target_config(self, preset):
        config = load_preset(preset)
        assert config.pack.target is not None

    @pytest.mark.parametrize("preset", PRESET_NAMES)
    def test_preset_has_data_quality_config(self, preset):
        config = load_preset(preset)
        assert config.pack.data_quality is not None

    @pytest.mark.parametrize("preset", PRESET_NAMES)
    def test_preset_has_scope_config(self, preset):
        config = load_preset(preset)
        assert config.pack.scope is not None

    @pytest.mark.parametrize("preset", PRESET_NAMES)
    def test_preset_has_reduction_config(self, preset):
        config = load_preset(preset)
        assert config.pack.reduction is not None
