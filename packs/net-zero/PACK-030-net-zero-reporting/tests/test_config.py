# -*- coding: utf-8 -*-
"""
Tests for PACK-030 Net Zero Reporting Pack configuration module.

Validates the config package structure, pack_config.py exports,
default settings, enum definitions, sub-config models, constants,
and utility functions for multi-framework report configuration.

Target: ~65 tests.

Author: GreenLang Platform Team
Pack: PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ========================================================================
# Config Package Structure
# ========================================================================


class TestConfigPackageStructure:
    """Tests for config package file structure."""

    def test_config_init_exists(self):
        config_init = _PACK_ROOT / "config" / "__init__.py"
        assert config_init.exists()

    def test_pack_config_exists(self):
        pack_config = _PACK_ROOT / "config" / "pack_config.py"
        assert pack_config.exists()

    def test_presets_init_exists(self):
        presets_init = _PACK_ROOT / "config" / "presets" / "__init__.py"
        assert presets_init.exists()

    def test_presets_dir_has_yaml_files(self):
        presets_dir = _PACK_ROOT / "config" / "presets"
        yaml_files = list(presets_dir.glob("*.yaml"))
        assert len(yaml_files) == 8


# ========================================================================
# Enum Imports
# ========================================================================


class TestConfigEnums:
    """Tests for configuration enum definitions."""

    def test_reporting_framework_enum(self):
        from config import ReportingFramework
        assert ReportingFramework is not None

    def test_output_format_enum(self):
        from config import OutputFormat
        assert OutputFormat is not None

    def test_assurance_level_enum(self):
        from config import AssuranceLevel
        assert AssuranceLevel is not None

    def test_branding_style_enum(self):
        from config import BrandingStyle
        assert BrandingStyle is not None

    def test_narrative_quality_enum(self):
        from config import NarrativeQuality
        assert NarrativeQuality is not None

    def test_report_status_enum(self):
        from config import ReportStatus
        assert ReportStatus is not None

    def test_stakeholder_view_type_enum(self):
        from config import StakeholderViewType
        assert StakeholderViewType is not None

    def test_translation_service_enum(self):
        from config import TranslationService
        assert TranslationService is not None

    def test_consistency_strictness_enum(self):
        from config import ConsistencyStrictness
        assert ConsistencyStrictness is not None

    def test_data_source_requirement_enum(self):
        from config import DataSourceRequirement
        assert DataSourceRequirement is not None


# ========================================================================
# Sub-Config Models
# ========================================================================


class TestSubConfigModels:
    """Tests for sub-configuration Pydantic models."""

    def test_framework_config_importable(self):
        from config import FrameworkConfig
        assert FrameworkConfig is not None

    def test_data_aggregation_config_importable(self):
        from config import DataAggregationConfig
        assert DataAggregationConfig is not None

    def test_narrative_config_importable(self):
        from config import NarrativeConfig
        assert NarrativeConfig is not None

    def test_xbrl_config_importable(self):
        from config import XBRLConfig
        assert XBRLConfig is not None

    def test_dashboard_config_importable(self):
        from config import DashboardConfig
        assert DashboardConfig is not None

    def test_assurance_config_importable(self):
        from config import AssuranceConfig
        assert AssuranceConfig is not None

    def test_validation_config_importable(self):
        from config import ValidationConfig
        assert ValidationConfig is not None

    def test_translation_config_importable(self):
        from config import TranslationConfig
        assert TranslationConfig is not None

    def test_branding_config_importable(self):
        from config import BrandingConfig
        assert BrandingConfig is not None

    def test_performance_config_importable(self):
        from config import PerformanceConfig
        assert PerformanceConfig is not None

    def test_notification_config_importable(self):
        from config import NotificationConfig
        assert NotificationConfig is not None

    def test_framework_output_config_importable(self):
        from config import FrameworkOutputConfig
        assert FrameworkOutputConfig is not None


# ========================================================================
# Main Config Models
# ========================================================================


class TestMainConfigModels:
    """Tests for main configuration Pydantic models."""

    def test_net_zero_reporting_config_importable(self):
        from config import NetZeroReportingConfig
        assert NetZeroReportingConfig is not None

    def test_pack_config_importable(self):
        from config import PackConfig
        assert PackConfig is not None

    def test_pack_config_has_from_preset(self):
        from config import PackConfig
        assert hasattr(PackConfig, "from_preset")


# ========================================================================
# Constants
# ========================================================================


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_supported_frameworks_constant(self):
        from config import SUPPORTED_FRAMEWORKS
        assert isinstance(SUPPORTED_FRAMEWORKS, (dict, list))

    def test_supported_languages_constant(self):
        from config import SUPPORTED_LANGUAGES
        assert isinstance(SUPPORTED_LANGUAGES, (dict, list))

    def test_supported_presets_constant(self):
        from config import SUPPORTED_PRESETS
        assert isinstance(SUPPORTED_PRESETS, dict)
        assert len(SUPPORTED_PRESETS) == 8

    def test_config_dir_constant(self):
        from config import CONFIG_DIR
        assert isinstance(CONFIG_DIR, Path)
        assert CONFIG_DIR.exists()

    def test_pack_base_dir_constant(self):
        from config import PACK_BASE_DIR
        assert isinstance(PACK_BASE_DIR, Path)
        assert PACK_BASE_DIR.exists()

    def test_default_net_zero_year(self):
        from config import DEFAULT_NET_ZERO_YEAR
        assert DEFAULT_NET_ZERO_YEAR == 2050

    def test_default_reporting_year(self):
        from config import DEFAULT_REPORTING_YEAR
        assert isinstance(DEFAULT_REPORTING_YEAR, int)
        assert DEFAULT_REPORTING_YEAR >= 2024

    def test_default_baseline_year(self):
        from config import DEFAULT_BASELINE_YEAR
        assert isinstance(DEFAULT_BASELINE_YEAR, int)

    def test_xbrl_taxonomy_specs(self):
        from config import XBRL_TAXONOMY_SPECS
        assert isinstance(XBRL_TAXONOMY_SPECS, dict)

    def test_assurance_standards(self):
        from config import ASSURANCE_STANDARDS
        assert isinstance(ASSURANCE_STANDARDS, dict)

    def test_output_format_specs(self):
        from config import OUTPUT_FORMAT_SPECS
        assert isinstance(OUTPUT_FORMAT_SPECS, dict)

    def test_data_source_packs(self):
        from config import DATA_SOURCE_PACKS
        assert isinstance(DATA_SOURCE_PACKS, (dict, list))

    def test_data_source_apps(self):
        from config import DATA_SOURCE_APPS
        assert isinstance(DATA_SOURCE_APPS, (dict, list))


# ========================================================================
# Utility Functions
# ========================================================================


class TestConfigUtilityFunctions:
    """Tests for configuration utility functions."""

    def test_list_available_presets(self):
        from config import list_available_presets
        result = list_available_presets()
        assert isinstance(result, dict)
        assert len(result) == 8

    def test_list_supported_frameworks(self):
        from config import list_supported_frameworks
        result = list_supported_frameworks()
        assert isinstance(result, (dict, list))

    def test_list_supported_languages(self):
        from config import list_supported_languages
        result = list_supported_languages()
        assert isinstance(result, (dict, list))

    def test_list_output_formats(self):
        from config import list_output_formats
        result = list_output_formats()
        assert isinstance(result, (dict, list))

    def test_load_config_callable(self):
        from config import load_config
        assert callable(load_config)

    def test_validate_config_callable(self):
        from config import validate_config
        assert callable(validate_config)

    def test_merge_config_callable(self):
        from config import merge_config
        assert callable(merge_config)

    def test_get_env_overrides_callable(self):
        from config import get_env_overrides
        result = get_env_overrides()
        assert isinstance(result, dict)

    def test_get_framework_info_callable(self):
        from config import get_framework_info
        assert callable(get_framework_info)

    def test_get_output_format_info_callable(self):
        from config import get_output_format_info
        assert callable(get_output_format_info)

    def test_get_xbrl_taxonomy_info_callable(self):
        from config import get_xbrl_taxonomy_info
        assert callable(get_xbrl_taxonomy_info)

    def test_get_assurance_standard_info_callable(self):
        from config import get_assurance_standard_info
        assert callable(get_assurance_standard_info)

    def test_list_branding_styles_callable(self):
        from config import list_branding_styles
        result = list_branding_styles()
        assert isinstance(result, (dict, list))

    def test_list_stakeholder_views_callable(self):
        from config import list_stakeholder_views
        result = list_stakeholder_views()
        assert isinstance(result, (dict, list))

    def test_list_consistency_rules_callable(self):
        from config import list_consistency_rules
        result = list_consistency_rules()
        assert isinstance(result, (dict, list))

    def test_list_evidence_bundle_components_callable(self):
        from config import list_evidence_bundle_components
        result = list_evidence_bundle_components()
        assert isinstance(result, (dict, list))


# ========================================================================
# Config __all__ Exports
# ========================================================================


class TestConfigAllExports:
    """Tests for config __init__.py __all__ list."""

    def test_config_has_all(self):
        from config import __all__
        assert isinstance(__all__, list)
        assert len(__all__) >= 40

    def test_config_all_contains_enums(self):
        from config import __all__
        enum_names = [
            "ReportingFramework", "OutputFormat", "AssuranceLevel",
            "BrandingStyle", "NarrativeQuality", "ReportStatus",
        ]
        for name in enum_names:
            assert name in __all__, f"Missing enum in config __all__: {name}"

    def test_config_all_contains_main_models(self):
        from config import __all__
        assert "NetZeroReportingConfig" in __all__
        assert "PackConfig" in __all__

    def test_config_all_contains_utility_functions(self):
        from config import __all__
        utility_names = [
            "load_config", "load_preset", "validate_config",
            "merge_config", "list_available_presets",
        ]
        for name in utility_names:
            assert name in __all__, f"Missing utility in config __all__: {name}"


# ========================================================================
# Default Settings
# ========================================================================


class TestDefaultSettings:
    """Tests for default configuration values."""

    def test_default_max_concurrent_reports(self):
        from config import DEFAULT_MAX_CONCURRENT_REPORTS
        assert isinstance(DEFAULT_MAX_CONCURRENT_REPORTS, int)
        assert DEFAULT_MAX_CONCURRENT_REPORTS >= 1

    def test_default_report_generation_timeout(self):
        from config import DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS
        assert isinstance(DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS, (int, float))
        assert DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS > 0

    def test_default_retention_years(self):
        from config import DEFAULT_RETENTION_YEARS
        assert isinstance(DEFAULT_RETENTION_YEARS, int)
        assert DEFAULT_RETENTION_YEARS >= 1

    def test_default_api_response_timeout(self):
        from config import DEFAULT_API_RESPONSE_TIMEOUT_MS
        assert isinstance(DEFAULT_API_RESPONSE_TIMEOUT_MS, (int, float))
        assert DEFAULT_API_RESPONSE_TIMEOUT_MS > 0
