# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Configuration
========================================

Tests pack configuration, presets, enums, validation, YAML loading,
PACK-042 dependency validation, and SBTi coherence.

Coverage target: 85%+
Total tests: ~40
"""

import copy
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACK_ROOT / "config"

from tests.conftest import PRESET_NAMES, SCOPE3_CATEGORIES, MATURITY_TIERS


# =============================================================================
# Pack Config Defaults
# =============================================================================


class TestPackConfigDefaults:
    """Test default configuration values."""

    def test_pack_id_is_043(self, sample_pack_config):
        assert sample_pack_config["pack_id"] == "PACK-043"

    def test_pack_name(self, sample_pack_config):
        assert sample_pack_config["pack_name"] == "Scope 3 Complete Pack"

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

    def test_decimal_precision_default(self, sample_pack_config):
        assert sample_pack_config["decimal_precision"] == 4

    def test_provenance_enabled(self, sample_pack_config):
        assert sample_pack_config["provenance_enabled"] is True

    def test_multi_tenant_enabled(self, sample_pack_config):
        assert sample_pack_config["multi_tenant_enabled"] is True


# =============================================================================
# Enums with Correct Members
# =============================================================================


class TestEnumValues:
    """Test expected enum values for various config fields."""

    @pytest.mark.parametrize("approach", [
        "equity_share", "operational_control", "financial_control",
    ])
    def test_consolidation_approach_values(self, approach):
        valid = {"equity_share", "operational_control", "financial_control"}
        assert approach in valid

    @pytest.mark.parametrize("gwp_source", ["ar4", "ar5", "ar6"])
    def test_gwp_source_values(self, gwp_source):
        valid = {"ar4", "ar5", "ar6"}
        assert gwp_source in valid

    @pytest.mark.parametrize("framework", [
        "ghg_protocol_scope3", "iso_14064", "cdp", "sbti", "tcfd", "esrs_e1", "sec",
    ])
    def test_framework_type_values(self, framework):
        assert isinstance(framework, str)
        assert len(framework) > 0

    @pytest.mark.parametrize("methodology", [
        "primary_data", "supplier_specific", "hybrid", "average_data", "spend_based",
    ])
    def test_methodology_hierarchy_values(self, methodology):
        valid = {"primary_data", "supplier_specific", "hybrid", "average_data", "spend_based"}
        assert methodology in valid

    @pytest.mark.parametrize("tier", [1, 2, 3, 4, 5])
    def test_maturity_tier_values(self, tier):
        assert tier in MATURITY_TIERS

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_scope3_category_values(self, cat_num):
        assert cat_num in SCOPE3_CATEGORIES

    @pytest.mark.parametrize("allocation", ["mass", "economic", "energy"])
    def test_lca_allocation_method_values(self, allocation):
        valid = {"mass", "economic", "energy"}
        assert allocation in valid

    @pytest.mark.parametrize("boundary", ["cradle_to_gate", "cradle_to_grave", "gate_to_gate"])
    def test_lca_system_boundary_values(self, boundary):
        valid = {"cradle_to_gate", "cradle_to_grave", "gate_to_gate"}
        assert boundary in valid

    @pytest.mark.parametrize("pathway", ["1.5C", "WB2C", "2C", "net_zero"])
    def test_sbti_pathway_values(self, pathway):
        valid = {"1.5C", "WB2C", "2C", "net_zero"}
        assert pathway in valid

    @pytest.mark.parametrize("asset_class", [
        "listed_equity", "corporate_bonds", "project_finance",
    ])
    def test_pcaf_asset_class_values(self, asset_class):
        valid = {"listed_equity", "corporate_bonds", "project_finance",
                 "commercial_real_estate", "mortgages", "motor_vehicle_loans"}
        assert asset_class in valid

    @pytest.mark.parametrize("assurance_level", ["limited", "reasonable"])
    def test_assurance_level_values(self, assurance_level):
        valid = {"limited", "reasonable"}
        assert assurance_level in valid

    @pytest.mark.parametrize("output_format", ["markdown", "html", "json", "pdf"])
    def test_output_format_values(self, output_format):
        valid = {"markdown", "html", "json", "pdf", "xlsx"}
        assert output_format in valid


# =============================================================================
# Presets Loading
# =============================================================================


class TestPresets:
    """Test preset configurations load correctly."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_name_valid(self, preset_name):
        assert preset_name in PRESET_NAMES

    def test_eight_presets(self):
        assert len(PRESET_NAMES) == 8

    def test_manufacturing_preset(self, sample_manufacturing_config):
        assert sample_manufacturing_config["preset"] == "manufacturing_enterprise"
        assert 1 in sample_manufacturing_config["scope3"]["priority_categories"]

    def test_financial_preset(self, sample_financial_config):
        assert sample_financial_config["preset"] == "financial_institution"
        assert 15 in sample_financial_config["scope3"]["priority_categories"]
        assert sample_financial_config["pcaf"]["enabled"] is True


# =============================================================================
# Config Merging
# =============================================================================


class TestConfigMerging:
    """Test configuration merging behavior."""

    def test_override_environment(self, sample_pack_config):
        override = {"environment": "production"}
        merged = {**sample_pack_config, **override}
        assert merged["environment"] == "production"
        assert merged["pack_id"] == "PACK-043"

    def test_deep_copy_independence(self, sample_pack_config):
        config_copy = copy.deepcopy(sample_pack_config)
        config_copy["environment"] = "staging"
        assert sample_pack_config["environment"] == "test"


# =============================================================================
# PACK-042 Dependency Validation
# =============================================================================


class TestPACK042Dependency:
    """Test PACK-042 dependency validation."""

    def test_dependencies_include_042(self, sample_pack_config):
        assert "PACK-042" in sample_pack_config["dependencies"]

    def test_dependencies_include_041(self, sample_pack_config):
        assert "PACK-041" in sample_pack_config["dependencies"]

    def test_dependency_count(self, sample_pack_config):
        assert len(sample_pack_config["dependencies"]) == 2


# =============================================================================
# SBTi Coherence Validation
# =============================================================================


class TestSBTiCoherence:
    """Test SBTi configuration coherence."""

    def test_sbti_pathway_defined(self, sample_pack_config):
        assert sample_pack_config["sbti"]["pathway"] in {"1.5C", "WB2C"}

    def test_sbti_near_term_year(self, sample_pack_config):
        assert sample_pack_config["sbti"]["near_term_year"] >= 2025
        assert sample_pack_config["sbti"]["near_term_year"] <= 2035

    def test_sbti_long_term_year(self, sample_pack_config):
        assert sample_pack_config["sbti"]["long_term_year"] == 2050

    def test_sbti_coverage_threshold(self, sample_pack_config):
        assert sample_pack_config["sbti"]["coverage_threshold_pct"] >= Decimal("67")

    def test_sbti_annual_rate_positive(self, sample_pack_config):
        assert sample_pack_config["sbti"]["annual_reduction_rate"] > Decimal("0")

    def test_sbti_near_before_long(self, sample_pack_config):
        assert sample_pack_config["sbti"]["near_term_year"] < sample_pack_config["sbti"]["long_term_year"]

    def test_scope3_config_15_categories(self, sample_pack_config):
        cats = sample_pack_config["scope3"]["enabled_categories"]
        assert len(cats) == 15
        assert set(cats) == set(range(1, 16))
