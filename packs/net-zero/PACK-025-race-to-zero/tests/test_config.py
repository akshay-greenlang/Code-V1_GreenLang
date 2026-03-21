# -*- coding: utf-8 -*-
"""
Tests for PACK-025 Race to Zero Pack configuration module.

Validates the config package structure, preset loading infrastructure,
and actor-type mapping correctness.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from config.presets import (
    AVAILABLE_PRESETS,
    ACTOR_TYPE_PRESET_MAP,
    DEFAULT_PRESET,
    get_preset_path,
    get_preset_for_actor_type,
)


# ========================================================================
# Config Package Structure
# ========================================================================


class TestConfigPackageStructure:
    """Tests for config package file structure."""

    def test_config_init_exists(self):
        config_init = _PACK_ROOT / "config" / "__init__.py"
        assert config_init.exists()

    def test_presets_init_exists(self):
        presets_init = _PACK_ROOT / "config" / "presets" / "__init__.py"
        assert presets_init.exists()

    def test_presets_dir_has_yaml_files(self):
        presets_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(presets_dir.glob("*.yaml"))
        assert len(yaml_files) == 8


# ========================================================================
# Preset Path Resolution
# ========================================================================


class TestPresetPathResolution:
    """Tests for preset path resolution functions."""

    def test_get_preset_path_all_presets(self):
        for name in AVAILABLE_PRESETS:
            path = get_preset_path(name)
            assert Path(path).exists()
            assert path.endswith(".yaml")

    def test_get_preset_path_returns_absolute(self):
        path = get_preset_path("corporate_commitment")
        assert Path(path).is_absolute()

    def test_get_preset_path_invalid_raises(self):
        with pytest.raises(KeyError):
            get_preset_path("invalid_preset_name")

    def test_get_preset_path_empty_string_raises(self):
        with pytest.raises(KeyError):
            get_preset_path("")


# ========================================================================
# Actor Type Mapping
# ========================================================================


class TestActorTypeMapping:
    """Tests for actor type to preset mapping."""

    def test_corporate_maps_to_corporate_commitment(self):
        assert get_preset_for_actor_type("CORPORATE") == "corporate_commitment"

    def test_financial_institution_maps_correctly(self):
        assert get_preset_for_actor_type("FINANCIAL_INSTITUTION") == "financial_institution"

    def test_city_maps_correctly(self):
        assert get_preset_for_actor_type("CITY") == "city_municipality"

    def test_region_maps_correctly(self):
        assert get_preset_for_actor_type("REGION") == "region_state"

    def test_sme_maps_correctly(self):
        assert get_preset_for_actor_type("SME") == "sme_business"

    def test_heavy_industry_maps_correctly(self):
        assert get_preset_for_actor_type("HEAVY_INDUSTRY") == "high_emitter"

    def test_services_maps_correctly(self):
        assert get_preset_for_actor_type("SERVICES") == "service_sector"

    def test_manufacturing_maps_correctly(self):
        assert get_preset_for_actor_type("MANUFACTURING") == "manufacturing_sector"

    def test_case_insensitive_lookup(self):
        assert get_preset_for_actor_type("corporate") == "corporate_commitment"
        assert get_preset_for_actor_type("Corporate") == "corporate_commitment"
        assert get_preset_for_actor_type("CORPORATE") == "corporate_commitment"

    def test_unknown_actor_type_raises(self):
        with pytest.raises(KeyError):
            get_preset_for_actor_type("UNKNOWN")


# ========================================================================
# Default Preset
# ========================================================================


class TestDefaultPreset:
    """Tests for default preset value."""

    def test_default_is_corporate_commitment(self):
        assert DEFAULT_PRESET == "corporate_commitment"

    def test_default_is_in_available_presets(self):
        assert DEFAULT_PRESET in AVAILABLE_PRESETS

    def test_default_preset_file_exists(self):
        path = get_preset_path(DEFAULT_PRESET)
        assert Path(path).exists()


# ========================================================================
# Consistency Checks
# ========================================================================


class TestConfigConsistency:
    """Tests for consistency between presets and mappings."""

    def test_all_mapped_presets_are_available(self):
        """Every preset in ACTOR_TYPE_PRESET_MAP exists in AVAILABLE_PRESETS."""
        for actor_type, preset_name in ACTOR_TYPE_PRESET_MAP.items():
            assert preset_name in AVAILABLE_PRESETS, (
                f"Actor type {actor_type} maps to {preset_name} "
                f"which is not in AVAILABLE_PRESETS"
            )

    def test_all_available_presets_are_mapped(self):
        """Every preset in AVAILABLE_PRESETS has at least one actor type mapping."""
        mapped_presets = set(ACTOR_TYPE_PRESET_MAP.values())
        for preset_name in AVAILABLE_PRESETS:
            assert preset_name in mapped_presets, (
                f"Preset {preset_name} has no actor type mapping"
            )

    def test_no_duplicate_preset_mappings(self):
        """No two actor types map to the same preset."""
        presets = list(ACTOR_TYPE_PRESET_MAP.values())
        assert len(presets) == len(set(presets)), "Duplicate preset mappings found"
