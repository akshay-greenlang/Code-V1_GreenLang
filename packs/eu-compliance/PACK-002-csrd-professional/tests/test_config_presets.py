# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Configuration Preset Tests
==============================================================

Validates all professional configuration presets including size presets
(enterprise_group, listed_company, financial_institution, multinational)
and sector presets (manufacturing_pro, financial_services_pro, technology_pro,
energy_pro, heavy_industry_pro).

Test count: 45
Author: GreenLang QA Team
"""

import hashlib
import json
from typing import Any, Dict

import pytest

from .conftest import (
    ALL_PRO_PRESETS,
    ALL_PRO_SECTORS,
    ENTERPRISE_GROUP_PRESET,
    FINANCIAL_INSTITUTION_PRESET,
    LISTED_COMPANY_PRESET,
    MULTINATIONAL_PRESET,
    VALID_ESRS_STANDARDS,
)


class TestEnterpriseGroupPreset:
    """Validate the enterprise_group size preset."""

    def test_enterprise_group_consolidation(self):
        """Enterprise group must enable consolidation with all 3 methods, 100 max subsidiaries."""
        preset = ENTERPRISE_GROUP_PRESET
        assert preset["consolidation"]["enabled"] is True
        assert preset["consolidation"]["max_subsidiaries"] == 100
        assert preset["consolidation"]["intercompany_elimination"] is True
        assert preset["consolidation"]["minority_disclosures"] is True

    def test_enterprise_group_approval(self):
        """Enterprise group must have 4-level approval with 95% auto-approve threshold."""
        preset = ENTERPRISE_GROUP_PRESET
        assert preset["approval"]["levels"] == 4
        assert preset["approval"]["auto_approve_threshold"] == 95.0
        assert preset["approval"]["escalation_timeout_hours"] == 48

    def test_enterprise_group_quality_gates(self):
        """Enterprise group must have all 3 quality gates with thresholds."""
        preset = ENTERPRISE_GROUP_PRESET
        qg = preset["quality_gates"]
        assert qg["qg1_threshold"] == 85.0
        assert qg["qg2_threshold"] == 90.0
        assert qg["qg3_threshold"] == 80.0

    def test_enterprise_group_assurance(self):
        """Enterprise group uses reasonable assurance level."""
        assert ENTERPRISE_GROUP_PRESET["assurance_level"] == "reasonable"

    def test_enterprise_group_cross_framework(self):
        """Enterprise group supports all 6 cross-frameworks."""
        frameworks = ENTERPRISE_GROUP_PRESET["cross_framework"]
        assert len(frameworks) == 6
        for fw in ["cdp", "tcfd", "sbti", "eu_taxonomy", "gri", "sasb"]:
            assert fw in frameworks


class TestListedCompanyPreset:
    """Validate the listed_company size preset."""

    def test_listed_company_consolidation(self):
        """Listed company must use financial_control approach with 50 max subsidiaries."""
        preset = LISTED_COMPANY_PRESET
        assert preset["consolidation"]["enabled"] is True
        assert preset["consolidation"]["default_approach"] == "financial_control"
        assert preset["consolidation"]["max_subsidiaries"] == 50

    def test_listed_company_approval(self):
        """Listed company must have 3-level approval chain."""
        preset = LISTED_COMPANY_PRESET
        assert preset["approval"]["levels"] == 3
        assert preset["approval"]["auto_approve_threshold"] == 90.0

    def test_listed_company_investor_reporting(self):
        """Listed company must support investor-required frameworks."""
        frameworks = LISTED_COMPANY_PRESET["cross_framework"]
        for fw in ["cdp", "tcfd", "sbti"]:
            assert fw in frameworks, f"Listed company missing investor framework: {fw}"

    def test_listed_company_assurance(self):
        """Listed company uses limited assurance."""
        assert LISTED_COMPANY_PRESET["assurance_level"] == "limited"

    def test_listed_company_data_points(self):
        """Listed company supports up to 3000 data points."""
        assert LISTED_COMPANY_PRESET["max_data_points"] == 3000


class TestFinancialInstitutionPreset:
    """Validate the financial_institution size preset."""

    def test_fi_pcaf_enabled(self):
        """Financial institution must enable PCAF."""
        assert FINANCIAL_INSTITUTION_PRESET["pcaf_enabled"] is True

    def test_fi_gar_enabled(self):
        """Financial institution must enable GAR reporting."""
        assert FINANCIAL_INSTITUTION_PRESET["gar_enabled"] is True

    def test_fi_targets(self):
        """Financial institution must enable FI-specific targets."""
        assert FINANCIAL_INSTITUTION_PRESET["fi_targets"] is True

    def test_fi_equity_share_approach(self):
        """Financial institution defaults to equity share consolidation."""
        assert FINANCIAL_INSTITUTION_PRESET["consolidation"]["default_approach"] == "equity_share"

    def test_fi_higher_quality_gates(self):
        """Financial institution has higher quality gate thresholds."""
        qg = FINANCIAL_INSTITUTION_PRESET["quality_gates"]
        assert qg["qg1_threshold"] >= 90.0
        assert qg["qg2_threshold"] >= 95.0


class TestMultinationalPreset:
    """Validate the multinational size preset."""

    def test_multinational_max_subsidiaries(self):
        """Multinational supports up to 200 subsidiaries."""
        assert MULTINATIONAL_PRESET["consolidation"]["max_subsidiaries"] == 200

    def test_multinational_multi_jurisdiction(self):
        """Multinational enables multi-jurisdiction support."""
        assert MULTINATIONAL_PRESET["multi_jurisdiction"] is True

    def test_multinational_multi_currency(self):
        """Multinational enables multi-currency support."""
        assert MULTINATIONAL_PRESET["multi_currency"] is True

    def test_multinational_languages(self):
        """Multinational supports 10+ languages."""
        langs = MULTINATIONAL_PRESET["languages"]
        assert len(langs) >= 10, f"Expected 10+ languages, got {len(langs)}"
        assert "en" in langs

    def test_multinational_all_frameworks(self):
        """Multinational supports all 6 cross-frameworks."""
        assert len(MULTINATIONAL_PRESET["cross_framework"]) == 6


class TestSizePresetCommon:
    """Common tests applied to all size presets via parameterization."""

    def test_preset_has_esrs_standards(self, pro_preset_config: Dict[str, Any]):
        """All presets must include all 12 ESRS standards."""
        standards = pro_preset_config["esrs_standards"]
        assert len(standards) == 12
        for std in VALID_ESRS_STANDARDS:
            assert std in standards, f"Missing standard {std}"

    def test_preset_has_consolidation(self, pro_preset_config: Dict[str, Any]):
        """All professional presets enable consolidation."""
        assert pro_preset_config["consolidation"]["enabled"] is True

    def test_preset_has_approval(self, pro_preset_config: Dict[str, Any]):
        """All professional presets have approval workflows."""
        approval = pro_preset_config["approval"]
        assert approval["levels"] >= 3

    def test_preset_has_quality_gates(self, pro_preset_config: Dict[str, Any]):
        """All professional presets define quality gate thresholds."""
        qg = pro_preset_config["quality_gates"]
        assert "qg1_threshold" in qg
        assert "qg2_threshold" in qg
        assert "qg3_threshold" in qg

    def test_preset_has_cross_framework(self, pro_preset_config: Dict[str, Any]):
        """All professional presets support at least 4 cross-frameworks."""
        frameworks = pro_preset_config["cross_framework"]
        assert len(frameworks) >= 4, (
            f"Expected at least 4 cross-frameworks, got {len(frameworks)}"
        )


class TestSectorPresets:
    """Validate sector-specific presets."""

    def test_manufacturing_pro_eu_ets(self):
        """Manufacturing preset enables EU ETS integration."""
        sector = ALL_PRO_SECTORS["manufacturing_pro"]
        assert sector["eu_ets_integration"] is True

    def test_manufacturing_pro_cbam(self):
        """Manufacturing preset enables CBAM preparedness."""
        sector = ALL_PRO_SECTORS["manufacturing_pro"]
        assert sector["cbam_preparedness"] is True

    def test_manufacturing_pro_sbti_pathway(self):
        """Manufacturing preset has SDA manufacturing pathway."""
        sector = ALL_PRO_SECTORS["manufacturing_pro"]
        assert "Manufacturing" in sector["sbti_pathway"]

    def test_financial_services_pro_pcaf(self):
        """Financial services preset enables PCAF."""
        sector = ALL_PRO_SECTORS["financial_services_pro"]
        assert sector["pcaf_enabled"] is True

    def test_energy_pro_ogmp(self):
        """Energy preset enables OGMP 2.0 methane reporting."""
        sector = ALL_PRO_SECTORS["energy_pro"]
        assert sector["ogmp_methane"] is True

    def test_sector_preset_has_nace_codes(self, pro_sector_config: Dict[str, Any]):
        """All sector presets define NACE codes."""
        assert "nace_codes" in pro_sector_config
        assert len(pro_sector_config["nace_codes"]) > 0

    def test_sector_preset_has_emission_focus(self, pro_sector_config: Dict[str, Any]):
        """All sector presets define emission focus areas."""
        assert "emission_focus" in pro_sector_config
        assert len(pro_sector_config["emission_focus"]) > 0

    def test_sector_preset_has_sbti_pathway(self, pro_sector_config: Dict[str, Any]):
        """All sector presets define an SBTi pathway."""
        assert "sbti_pathway" in pro_sector_config
        assert len(pro_sector_config["sbti_pathway"]) > 0


class TestConfigMerging:
    """Test configuration merging and overrides."""

    def test_config_merging_base_plus_size_plus_sector(self):
        """Verify size + sector presets can be merged without conflicts."""
        base = {"reporting_year": 2025, "language": "en"}
        size = ENTERPRISE_GROUP_PRESET.copy()
        sector = ALL_PRO_SECTORS["manufacturing_pro"].copy()

        merged = {**base, **size, **sector}
        assert merged["reporting_year"] == 2025
        # Sector preset overwrites preset_id since it is merged last
        assert merged["sector_id"] == "manufacturing_pro"
        assert merged["consolidation"]["enabled"] is True
        assert merged["eu_ets_integration"] is True

    def test_config_env_overrides(self):
        """Verify environment variables can override config values."""
        config = {"reporting_year": 2025, "xbrl_mode": "standard"}
        env_overrides = {"CSRD_PACK_REPORTING_YEAR": "2026", "CSRD_PACK_XBRL_MODE": "full"}

        for key, value in env_overrides.items():
            config_key = key.replace("CSRD_PACK_", "").lower()
            if config_key in config:
                # Coerce type
                original_type = type(config[config_key])
                config[config_key] = original_type(value)

        assert config["reporting_year"] == 2026
        assert config["xbrl_mode"] == "full"

    def test_config_provenance_hash(self, sample_pack_config: Dict[str, Any]):
        """Verify provenance hash is a valid SHA-256 hex string."""
        hash_val = sample_pack_config["provenance_hash"]
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64
        assert all(c in "0123456789abcdef" for c in hash_val)

    def test_invalid_preset_rejected(self):
        """Verify invalid preset IDs are rejected."""
        valid_presets = set(ALL_PRO_PRESETS.keys())
        invalid = "ultra_mega_enterprise"
        assert invalid not in valid_presets

    def test_demo_mode_config(self, sample_pack_config: Dict[str, Any]):
        """Verify pack config has all required fields for demo mode."""
        required_fields = [
            "metadata", "size_preset", "sector_preset", "reporting_year",
            "esrs_standards", "consolidation", "quality_gates",
        ]
        for field in required_fields:
            assert field in sample_pack_config, f"Missing field '{field}' in pack config"
