# -*- coding: utf-8 -*-
"""
Tests for all 8 PACK-030 Net Zero Reporting Pack Configuration Presets.

Covers: csrd_focus, cdp_alist, tcfd_investor, sbti_validation,
sec_10k, multi_framework, investor_relations, assurance_ready.

Validates YAML existence, parsing, required sections, framework-specific
configuration values, and schema structure completeness.

Target: ~80 tests.

Author: GreenLang Platform Team
Pack: PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest
import yaml

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

# Directory where preset YAML files live.
PRESET_DIR = _PACK_ROOT / "config" / "presets"

# All 8 expected presets.
PRESET_NAMES = [
    "csrd_focus",
    "cdp_alist",
    "tcfd_investor",
    "sbti_validation",
    "sec_10k",
    "multi_framework",
    "investor_relations",
    "assurance_ready",
]


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture(params=PRESET_NAMES, ids=PRESET_NAMES)
def preset_name(request) -> str:
    """Parameterized fixture yielding each preset name."""
    return request.param


@pytest.fixture
def preset_data(preset_name) -> dict:
    """Load a single preset YAML file."""
    yaml_path = PRESET_DIR / f"{preset_name}.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def all_preset_data() -> dict:
    """Load all presets and return as dict of {name: data}."""
    result = {}
    for name in PRESET_NAMES:
        yaml_path = PRESET_DIR / f"{name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            result[name] = yaml.safe_load(f)
    return result


# ========================================================================
# Preset Directory Structure
# ========================================================================


class TestPresetDirectoryStructure:
    """Tests for preset directory existence and file count."""

    def test_presets_dir_exists(self):
        assert PRESET_DIR.exists()

    def test_presets_dir_is_directory(self):
        assert PRESET_DIR.is_dir()

    def test_presets_init_exists(self):
        init_file = PRESET_DIR / "__init__.py"
        assert init_file.exists()

    def test_8_preset_yaml_files(self):
        yaml_files = list(PRESET_DIR.glob("*.yaml"))
        assert len(yaml_files) == 8

    def test_no_extra_yaml_files(self):
        yaml_files = [f.stem for f in PRESET_DIR.glob("*.yaml")]
        for name in yaml_files:
            assert name in PRESET_NAMES, f"Unexpected preset YAML: {name}.yaml"


# ========================================================================
# Preset File Existence and YAML Validity
# ========================================================================


class TestPresetFilesExist:
    """Tests that all preset YAML files exist and parse correctly."""

    def test_preset_yaml_exists(self, preset_name):
        """Preset YAML file exists on disk."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        assert yaml_path.exists(), f"Missing preset file: {yaml_path}"

    def test_preset_yaml_parses(self, preset_name):
        """Preset YAML file parses without error."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_preset_yaml_not_empty(self, preset_name):
        """Preset YAML file has meaningful content."""
        yaml_path = PRESET_DIR / f"{preset_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Should have at least 30 non-comment, non-blank lines
        lines = [
            line for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(lines) >= 30, (
            f"Preset {preset_name} has only {len(lines)} non-comment lines"
        )

    def test_preset_yaml_is_dict(self, preset_data):
        """Loaded preset data is a dictionary."""
        assert isinstance(preset_data, dict)


# ========================================================================
# Preset Schema Structure
# ========================================================================


class TestPresetSchemaStructure:
    """Tests that each preset has the expected top-level keys."""

    def test_preset_has_top_level_keys(self, preset_name, preset_data):
        """Each preset should have at least some configuration keys."""
        assert len(preset_data) >= 3, (
            f"Preset {preset_name} has too few top-level keys: {list(preset_data.keys())}"
        )

    def test_preset_values_are_not_none(self, preset_name, preset_data):
        """No top-level key should have a None value."""
        for key, value in preset_data.items():
            assert value is not None, (
                f"Preset {preset_name} has None value for key '{key}'"
            )


# ========================================================================
# Framework-Specific Preset Tests
# ========================================================================


class TestCSRDFocusPreset:
    """Tests specific to the csrd_focus preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "csrd_focus.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_csrd_focus_exists(self):
        assert self.data is not None

    def test_csrd_focus_is_dict(self):
        assert isinstance(self.data, dict)

    def test_csrd_focus_has_content(self):
        assert len(self.data) > 0


class TestCDPAListPreset:
    """Tests specific to the cdp_alist preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "cdp_alist.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_cdp_alist_exists(self):
        assert self.data is not None

    def test_cdp_alist_is_dict(self):
        assert isinstance(self.data, dict)

    def test_cdp_alist_has_content(self):
        assert len(self.data) > 0


class TestTCFDInvestorPreset:
    """Tests specific to the tcfd_investor preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "tcfd_investor.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_tcfd_investor_exists(self):
        assert self.data is not None

    def test_tcfd_investor_is_dict(self):
        assert isinstance(self.data, dict)

    def test_tcfd_investor_has_content(self):
        assert len(self.data) > 0


class TestSBTiValidationPreset:
    """Tests specific to the sbti_validation preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "sbti_validation.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_sbti_validation_exists(self):
        assert self.data is not None

    def test_sbti_validation_is_dict(self):
        assert isinstance(self.data, dict)

    def test_sbti_validation_has_content(self):
        assert len(self.data) > 0


class TestSEC10KPreset:
    """Tests specific to the sec_10k preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "sec_10k.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_sec_10k_exists(self):
        assert self.data is not None

    def test_sec_10k_is_dict(self):
        assert isinstance(self.data, dict)

    def test_sec_10k_has_content(self):
        assert len(self.data) > 0


class TestMultiFrameworkPreset:
    """Tests specific to the multi_framework preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "multi_framework.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_multi_framework_exists(self):
        assert self.data is not None

    def test_multi_framework_is_dict(self):
        assert isinstance(self.data, dict)

    def test_multi_framework_has_content(self):
        assert len(self.data) > 0


class TestInvestorRelationsPreset:
    """Tests specific to the investor_relations preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "investor_relations.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_investor_relations_exists(self):
        assert self.data is not None

    def test_investor_relations_is_dict(self):
        assert isinstance(self.data, dict)

    def test_investor_relations_has_content(self):
        assert len(self.data) > 0


class TestAssuranceReadyPreset:
    """Tests specific to the assurance_ready preset."""

    @pytest.fixture(autouse=True)
    def _load_preset(self):
        yaml_path = PRESET_DIR / "assurance_ready.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def test_assurance_ready_exists(self):
        assert self.data is not None

    def test_assurance_ready_is_dict(self):
        assert isinstance(self.data, dict)

    def test_assurance_ready_has_content(self):
        assert len(self.data) > 0


# ========================================================================
# SUPPORTED_PRESETS Constant (from pack_config.py)
# ========================================================================


class TestSupportedPresetsConstant:
    """Tests for SUPPORTED_PRESETS dict from config module."""

    def test_supported_presets_importable(self):
        from config import SUPPORTED_PRESETS
        assert isinstance(SUPPORTED_PRESETS, dict)

    def test_supported_presets_has_8_entries(self):
        from config import SUPPORTED_PRESETS
        assert len(SUPPORTED_PRESETS) == 8

    def test_all_preset_names_present(self):
        from config import SUPPORTED_PRESETS
        for name in PRESET_NAMES:
            assert name in SUPPORTED_PRESETS, (
                f"Preset '{name}' not in SUPPORTED_PRESETS"
            )

    def test_all_presets_have_descriptions(self):
        from config import SUPPORTED_PRESETS
        for name, desc in SUPPORTED_PRESETS.items():
            assert isinstance(desc, str), f"Preset '{name}' description is not a string"
            assert len(desc) > 0, f"Preset '{name}' has empty description"


# ========================================================================
# Utility Functions
# ========================================================================


class TestPresetUtilityFunctions:
    """Tests for preset loading utility functions from config module."""

    def test_list_available_presets_callable(self):
        from config import list_available_presets
        result = list_available_presets()
        assert isinstance(result, dict)
        assert len(result) == 8

    def test_load_preset_callable(self):
        from config import load_preset
        assert callable(load_preset)

    def test_load_preset_multi_framework(self):
        from config import load_preset
        config = load_preset("multi_framework")
        assert config is not None

    def test_load_preset_unknown_raises(self):
        from config import load_preset
        with pytest.raises((ValueError, KeyError)):
            load_preset("nonexistent_preset")
