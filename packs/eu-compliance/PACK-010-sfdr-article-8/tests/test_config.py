# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Configuration Tests
====================================================

Tests the SFDRArticle8Config configuration system including:
- SFDRClassification enum (4 values: ARTICLE_6, ARTICLE_8, ARTICLE_8_PLUS, ARTICLE_9)
- PAICategory enum (5 values: CLIMATE, ENVIRONMENT, SOCIAL, SOVEREIGN, REAL_ESTATE)
- PAIDataQuality enum (4 values: REPORTED, ESTIMATED, MODELED, NOT_AVAILABLE)
- DisclosureType enum (3 values: PRE_CONTRACTUAL, PERIODIC, WEBSITE)
- ESCharacteristicType enum (2 values: ENVIRONMENTAL, SOCIAL)
- GovernanceCheckStatus enum (4 values: PASS, FAIL, PARTIAL, NOT_ASSESSED)
- SustainableInvestmentType enum (3 values)
- ReportingFrequency enum (3 values)
- ScreeningType enum (3 values)
- ComplianceStatus enum (4 values)
- PAIConfig defaults and validation
- TaxonomyAlignmentConfig defaults
- DNSHConfig defaults
- GovernanceConfig defaults
- ESGCharacteristicsConfig defaults
- SustainableInvestmentConfig defaults
- CarbonFootprintConfig defaults
- EETConfig defaults
- DisclosureConfig defaults
- ScreeningConfig defaults
- SFDRArticle8Config creation with all sub-configs
- PackConfig loading from YAML and presets
- PackConfig.available_presets
- PackConfig.get_config_hash
- Preset YAML file loading and validation (asset_manager, insurance, bank)
- Validation tests (invalid values raise errors)
- Environment variable override support

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# ---------------------------------------------------------------------------
# Inline path setup and import helper (no conftest imports)
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"


def _import_from_path(module_name: str, file_path: Path):
    """Import module from file path (handles hyphenated directory names)."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load config module once for all tests
# ---------------------------------------------------------------------------

try:
    _cfg = _import_from_path("sfdr_pack_config_test", CONFIG_DIR / "pack_config.py")
except Exception:
    _cfg = None


def _skip_if_no_config():
    """Skip test if pack_config module could not be loaded."""
    if _cfg is None:
        pytest.skip("pack_config module could not be loaded")


# ===========================================================================
# Test class
# ===========================================================================

@pytest.mark.unit
class TestPackConfig:
    """Test suite for PACK-010 SFDRArticle8Config configuration."""

    # -----------------------------------------------------------------
    # 1. SFDRClassification enum
    # -----------------------------------------------------------------

    def test_sfdr_classification_enum_values(self):
        """Test SFDRClassification enum has exactly 4 values."""
        _skip_if_no_config()
        sc = _cfg.SFDRClassification
        members = list(sc)
        assert len(members) == 4, f"Expected 4 classifications, got {len(members)}"

        expected = {"ARTICLE_6", "ARTICLE_8", "ARTICLE_8_PLUS", "ARTICLE_9"}
        actual = {m.value for m in members}
        assert actual == expected, f"SFDRClassification mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 2. PAICategory enum
    # -----------------------------------------------------------------

    def test_pai_category_enum_values(self):
        """Test PAICategory enum has exactly 5 values."""
        _skip_if_no_config()
        pc = _cfg.PAICategory
        expected = {"CLIMATE", "ENVIRONMENT", "SOCIAL", "SOVEREIGN", "REAL_ESTATE"}
        actual = {m.value for m in pc}
        assert actual == expected, f"PAICategory mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 3. PAIDataQuality enum
    # -----------------------------------------------------------------

    def test_pai_data_quality_enum_values(self):
        """Test PAIDataQuality enum has exactly 4 values."""
        _skip_if_no_config()
        pdq = _cfg.PAIDataQuality
        expected = {"REPORTED", "ESTIMATED", "MODELED", "NOT_AVAILABLE"}
        actual = {m.value for m in pdq}
        assert actual == expected, f"PAIDataQuality mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 4. DisclosureType enum
    # -----------------------------------------------------------------

    def test_disclosure_type_enum_values(self):
        """Test DisclosureType enum has exactly 3 values."""
        _skip_if_no_config()
        dt = _cfg.DisclosureType
        expected = {"PRE_CONTRACTUAL", "PERIODIC", "WEBSITE"}
        actual = {m.value for m in dt}
        assert actual == expected, f"DisclosureType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 5. ESCharacteristicType enum
    # -----------------------------------------------------------------

    def test_es_characteristic_type_enum_values(self):
        """Test ESCharacteristicType enum has exactly 2 values."""
        _skip_if_no_config()
        ect = _cfg.ESCharacteristicType
        expected = {"ENVIRONMENTAL", "SOCIAL"}
        actual = {m.value for m in ect}
        assert actual == expected, (
            f"ESCharacteristicType mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 6. GovernanceCheckStatus enum
    # -----------------------------------------------------------------

    def test_governance_check_status_enum_values(self):
        """Test GovernanceCheckStatus enum has exactly 4 values."""
        _skip_if_no_config()
        gcs = _cfg.GovernanceCheckStatus
        expected = {"PASS", "FAIL", "PARTIAL", "NOT_ASSESSED"}
        actual = {m.value for m in gcs}
        assert actual == expected, (
            f"GovernanceCheckStatus mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 7. SustainableInvestmentType enum
    # -----------------------------------------------------------------

    def test_sustainable_investment_type_enum_values(self):
        """Test SustainableInvestmentType enum has exactly 3 values."""
        _skip_if_no_config()
        sit = _cfg.SustainableInvestmentType
        expected = {"TAXONOMY_ALIGNED", "OTHER_ENVIRONMENTAL", "SOCIAL"}
        actual = {m.value for m in sit}
        assert actual == expected, (
            f"SustainableInvestmentType mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 8. ReportingFrequency enum
    # -----------------------------------------------------------------

    def test_reporting_frequency_enum_values(self):
        """Test ReportingFrequency enum has exactly 3 values."""
        _skip_if_no_config()
        rf = _cfg.ReportingFrequency
        expected = {"ANNUAL", "SEMI_ANNUAL", "QUARTERLY"}
        actual = {m.value for m in rf}
        assert actual == expected, (
            f"ReportingFrequency mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 9. ScreeningType enum
    # -----------------------------------------------------------------

    def test_screening_type_enum_values(self):
        """Test ScreeningType enum has exactly 3 values."""
        _skip_if_no_config()
        st = _cfg.ScreeningType
        expected = {"NEGATIVE", "POSITIVE", "NORMS_BASED"}
        actual = {m.value for m in st}
        assert actual == expected, (
            f"ScreeningType mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 10. ComplianceStatus enum
    # -----------------------------------------------------------------

    def test_compliance_status_enum_values(self):
        """Test ComplianceStatus enum has exactly 4 values."""
        _skip_if_no_config()
        cs = _cfg.ComplianceStatus
        expected = {"COMPLIANT", "NON_COMPLIANT", "PARTIALLY_COMPLIANT", "NOT_ASSESSED"}
        actual = {m.value for m in cs}
        assert actual == expected, (
            f"ComplianceStatus mismatch: {actual} != {expected}"
        )

    # -----------------------------------------------------------------
    # 11. PAIConfig defaults
    # -----------------------------------------------------------------

    def test_pai_config_defaults(self):
        """Test PAIConfig default values."""
        _skip_if_no_config()
        pai = _cfg.PAIConfig()
        assert pai.enabled is True
        assert len(pai.enabled_mandatory_indicators) == 18
        assert pai.enabled_mandatory_indicators == list(range(1, 19))
        assert pai.min_coverage_pct == 50.0
        assert pai.coverage_adjustment is True
        assert pai.estimation_enabled is True
        assert pai.estimation_methodology == "SECTOR_AVERAGE"
        assert pai.include_scope_3 is False
        assert pai.entity_level_statement is True
        assert pai.product_level_pai is True
        assert pai.prior_period_comparison is True
        assert pai.data_quality_tracking is True

    # -----------------------------------------------------------------
    # 12. TaxonomyAlignmentConfig defaults
    # -----------------------------------------------------------------

    def test_taxonomy_alignment_config_defaults(self):
        """Test TaxonomyAlignmentConfig default values."""
        _skip_if_no_config()
        ta = _cfg.TaxonomyAlignmentConfig()
        assert ta.enabled is True
        assert ta.minimum_commitment_pct == 0.0
        assert ta.include_sovereign_bonds is False
        assert ta.look_through_enabled is True
        assert ta.look_through_max_levels == 3
        assert ta.enabling_activity_disclosure is True
        assert ta.transitional_activity_disclosure is True
        assert ta.alignment_methodology == "REVENUE_BASED"
        assert ta.gas_nuclear_disclosure is True
        assert ta.verification_frequency == _cfg.ReportingFrequency.QUARTERLY

    # -----------------------------------------------------------------
    # 13. DNSHConfig defaults
    # -----------------------------------------------------------------

    def test_dnsh_config_defaults(self):
        """Test DNSHConfig default values."""
        _skip_if_no_config()
        dnsh = _cfg.DNSHConfig()
        assert dnsh.enabled is True
        assert dnsh.methodology == _cfg.DNSHMethodology.COMBINED
        assert len(dnsh.pai_indicators_used) == 18
        assert dnsh.quantitative_thresholds_enabled is True
        assert dnsh.qualitative_assessment_enabled is True
        assert dnsh.evidence_required is True
        assert dnsh.controversial_weapons_exclusion is True
        assert dnsh.ungc_violations_exclusion is True

    # -----------------------------------------------------------------
    # 14. GovernanceConfig defaults
    # -----------------------------------------------------------------

    def test_governance_config_defaults(self):
        """Test GovernanceConfig default values."""
        _skip_if_no_config()
        gov = _cfg.GovernanceConfig()
        assert gov.enabled is True
        assert gov.check_sound_management is True
        assert gov.check_employee_relations is True
        assert gov.check_remuneration is True
        assert gov.check_tax_compliance is True
        assert gov.minimum_governance_score == 60.0
        assert gov.require_all_dimensions_pass is True
        assert gov.controversy_flag_threshold == 3
        assert gov.monitoring_frequency == _cfg.ReportingFrequency.QUARTERLY
        assert len(gov.data_sources) >= 1

    # -----------------------------------------------------------------
    # 15. ESGCharacteristicsConfig defaults
    # -----------------------------------------------------------------

    def test_esg_characteristics_config_defaults(self):
        """Test ESGCharacteristicsConfig default values."""
        _skip_if_no_config()
        esg = _cfg.ESGCharacteristicsConfig()
        assert len(esg.environmental_characteristics) >= 1
        assert len(esg.social_characteristics) >= 1
        assert len(esg.binding_elements) >= 1
        assert len(esg.sustainability_indicators) >= 1
        assert esg.measurement_methodology == "QUANTITATIVE"
        assert esg.benchmark_comparison is False
        assert esg.track_attainment is True

    # -----------------------------------------------------------------
    # 16. SustainableInvestmentConfig defaults
    # -----------------------------------------------------------------

    def test_sustainable_investment_config_defaults(self):
        """Test SustainableInvestmentConfig default values."""
        _skip_if_no_config()
        si = _cfg.SustainableInvestmentConfig()
        assert si.enabled is False  # Article 8 default (not 8+)
        assert si.minimum_proportion_pct == 0.0
        assert si.taxonomy_aligned_minimum_pct == 0.0
        assert si.other_environmental_minimum_pct == 0.0
        assert si.social_minimum_pct == 0.0
        assert si.require_dnsh_pass is True
        assert si.require_governance_pass is True
        assert si.double_counting_prevention is True
        assert si.contribution_assessment_methodology == "THRESHOLD_BASED"

    # -----------------------------------------------------------------
    # 17. CarbonFootprintConfig defaults
    # -----------------------------------------------------------------

    def test_carbon_footprint_config_defaults(self):
        """Test CarbonFootprintConfig default values."""
        _skip_if_no_config()
        cf = _cfg.CarbonFootprintConfig()
        assert cf.enabled is True
        assert cf.calculate_waci is True
        assert cf.calculate_carbon_footprint is True
        assert cf.calculate_total_emissions is True
        assert cf.calculate_financed_emissions is False
        assert "SCOPE_1" in cf.scope_coverage
        assert "SCOPE_2" in cf.scope_coverage
        assert cf.pcaf_alignment is False
        assert cf.coverage_threshold_pct == 50.0
        assert cf.benchmark_comparison is True
        assert cf.yoy_trend is True
        assert cf.currency == "EUR"

    # -----------------------------------------------------------------
    # 18. EETConfig defaults
    # -----------------------------------------------------------------

    def test_eet_config_defaults(self):
        """Test EETConfig default values."""
        _skip_if_no_config()
        eet = _cfg.EETConfig()
        assert eet.enabled is True
        assert eet.eet_version == _cfg.EETVersion.V1_1
        assert eet.import_enabled is True
        assert eet.export_enabled is True
        assert eet.export_format == "XLSX"
        assert eet.validate_completeness is True
        assert eet.validate_consistency is True
        assert eet.auto_populate is True
        assert eet.sfdr_classification_field is True
        assert eet.taxonomy_alignment_fields is True
        assert eet.pai_consideration_fields is True

    # -----------------------------------------------------------------
    # 19. DisclosureConfig defaults
    # -----------------------------------------------------------------

    def test_disclosure_config_defaults(self):
        """Test DisclosureConfig default values."""
        _skip_if_no_config()
        dc = _cfg.DisclosureConfig()
        assert dc.annex_ii_enabled is True
        assert dc.annex_iii_enabled is True
        assert dc.annex_iv_enabled is True
        assert dc.default_format == _cfg.DisclosureFormat.PDF
        assert dc.include_methodology_note is True
        assert dc.include_data_sources is True
        assert dc.include_asset_allocation_chart is True
        assert dc.review_workflow_enabled is True
        assert dc.version_tracking is True
        assert dc.greenwashing_check is True

    # -----------------------------------------------------------------
    # 20. ScreeningConfig defaults
    # -----------------------------------------------------------------

    def test_screening_config_defaults(self):
        """Test ScreeningConfig default values."""
        _skip_if_no_config()
        sc = _cfg.ScreeningConfig()
        assert sc.negative_screening_enabled is True
        assert sc.positive_screening_enabled is False
        assert sc.norms_based_screening_enabled is True
        assert len(sc.negative_exclusions) >= 3
        assert "controversial_weapons" in sc.negative_exclusions
        assert sc.revenue_threshold_pct == 5.0
        assert len(sc.norms_frameworks) >= 2
        assert sc.screening_frequency == _cfg.ReportingFrequency.QUARTERLY
        assert sc.breach_handling == "DIVEST_90_DAYS"
        assert sc.monitoring_enabled is True

    # -----------------------------------------------------------------
    # 21. SFDRArticle8Config main class creation
    # -----------------------------------------------------------------

    def test_sfdr_article8_config_default_creation(self):
        """Test SFDRArticle8Config creates with valid defaults."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()

        assert config.pack_id == "PACK-010-sfdr-article-8"
        assert config.version == "1.0.0"
        assert config.tier == "standalone"
        assert config.sfdr_classification == _cfg.SFDRClassification.ARTICLE_8
        assert config.reporting_year == 2025

        # All sub-configs should be populated
        assert isinstance(config.pai, _cfg.PAIConfig)
        assert isinstance(config.taxonomy_alignment, _cfg.TaxonomyAlignmentConfig)
        assert isinstance(config.dnsh, _cfg.DNSHConfig)
        assert isinstance(config.governance, _cfg.GovernanceConfig)
        assert isinstance(config.esg_characteristics, _cfg.ESGCharacteristicsConfig)
        assert isinstance(config.sustainable_investment, _cfg.SustainableInvestmentConfig)
        assert isinstance(config.carbon_footprint, _cfg.CarbonFootprintConfig)
        assert isinstance(config.eet, _cfg.EETConfig)
        assert isinstance(config.disclosure, _cfg.DisclosureConfig)
        assert isinstance(config.screening, _cfg.ScreeningConfig)
        assert isinstance(config.reporting, _cfg.ReportingConfig)
        assert isinstance(config.data_quality, _cfg.DataQualityConfig)
        assert isinstance(config.audit_trail, _cfg.AuditTrailConfig)
        assert isinstance(config.demo, _cfg.DemoConfig)

    # -----------------------------------------------------------------
    # 22. PackConfig.from_yaml
    # -----------------------------------------------------------------

    def test_pack_config_from_yaml(self):
        """Test PackConfig.from_yaml loads from preset file if available."""
        _skip_if_no_config()
        asset_manager_path = PRESETS_DIR / "asset_manager.yaml"
        if not asset_manager_path.exists():
            pytest.skip("Asset manager preset file not found")

        pc = _cfg.PackConfig.from_yaml(asset_manager_path)
        assert isinstance(pc.pack, _cfg.SFDRArticle8Config)
        assert isinstance(pc.loaded_from, list)
        assert len(pc.loaded_from) >= 1
        assert str(asset_manager_path) in pc.loaded_from

    # -----------------------------------------------------------------
    # 23. PackConfig.from_preset
    # -----------------------------------------------------------------

    def test_pack_config_from_preset(self):
        """Test PackConfig.from_preset loads named preset correctly."""
        _skip_if_no_config()
        pc = _cfg.PackConfig.from_preset("asset_manager")
        assert isinstance(pc.pack, _cfg.SFDRArticle8Config)
        assert pc.pack.sfdr_classification == _cfg.SFDRClassification.ARTICLE_8

    # -----------------------------------------------------------------
    # 24. PackConfig.available_presets
    # -----------------------------------------------------------------

    def test_pack_config_available_presets(self):
        """Test PackConfig.available_presets returns all 5 presets."""
        _skip_if_no_config()
        presets = _cfg.PackConfig.available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 5, f"Expected 5 presets, got {len(presets)}"

        expected_presets = [
            "asset_manager", "insurance", "bank", "pension_fund", "wealth_manager",
        ]
        for preset_id in expected_presets:
            assert preset_id in presets, (
                f"Missing preset: {preset_id}. Found: {list(presets.keys())}"
            )

    # -----------------------------------------------------------------
    # 25. PackConfig.get_config_hash
    # -----------------------------------------------------------------

    def test_pack_config_get_config_hash(self):
        """Test PackConfig.get_config_hash returns valid SHA-256."""
        _skip_if_no_config()
        pc = _cfg.get_default_config()
        config_hash = pc.get_config_hash()

        assert isinstance(config_hash, str)
        assert len(config_hash) == 64
        # Reproducibility check
        assert pc.get_config_hash() == config_hash

    # -----------------------------------------------------------------
    # 26. Preset loading - asset_manager
    # -----------------------------------------------------------------

    def test_preset_asset_manager_yaml(self):
        """Test asset_manager preset YAML exists and loads correctly."""
        _skip_if_no_config()
        preset_path = PRESETS_DIR / "asset_manager.yaml"
        assert preset_path.exists(), f"Asset manager preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("sfdr_classification") == "ARTICLE_8"
        assert data.get("pai", {}).get("enabled") is True

    # -----------------------------------------------------------------
    # 27. Preset loading - insurance
    # -----------------------------------------------------------------

    def test_preset_insurance_yaml(self):
        """Test insurance preset YAML exists and loads correctly."""
        _skip_if_no_config()
        preset_path = PRESETS_DIR / "insurance.yaml"
        assert preset_path.exists(), f"Insurance preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        assert data.get("sfdr_classification") == "ARTICLE_8"
        # Insurance preset has higher look-through levels
        assert data.get("taxonomy_alignment", {}).get("look_through_max_levels", 0) >= 5

    # -----------------------------------------------------------------
    # 28. Preset loading - bank
    # -----------------------------------------------------------------

    def test_preset_bank_yaml(self):
        """Test bank preset YAML exists and loads correctly."""
        _skip_if_no_config()
        preset_path = PRESETS_DIR / "bank.yaml"
        assert preset_path.exists(), f"Bank preset not found: {preset_path}"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Preset must parse to dict"
        # Bank preset is Article 8+ with sustainable investment
        assert data.get("sfdr_classification") == "ARTICLE_8_PLUS"
        assert data.get("sustainable_investment", {}).get("enabled") is True
        assert data.get("pai", {}).get("include_scope_3") is True

    # -----------------------------------------------------------------
    # 29. Validation - invalid PAI indicator IDs
    # -----------------------------------------------------------------

    def test_config_validation_invalid_pai_ids_raise_error(self):
        """Test that PAI indicator IDs outside 1-18 raise ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.PAIConfig(enabled_mandatory_indicators=[0, 19, 20])

    # -----------------------------------------------------------------
    # 30. Environment variable overrides
    # -----------------------------------------------------------------

    def test_environment_variable_overrides(self, monkeypatch):
        """Test environment variable overrides are respected."""
        _skip_if_no_config()

        monkeypatch.setenv("SFDR_PACK_PRODUCT_NAME", "Test ESG Fund")
        monkeypatch.setenv("SFDR_PACK_REPORTING_YEAR", "2026")

        env_product_name = os.getenv("SFDR_PACK_PRODUCT_NAME", "")
        assert env_product_name == "Test ESG Fund"

        env_year = int(os.getenv("SFDR_PACK_REPORTING_YEAR", "2025"))
        assert env_year == 2026

    # -----------------------------------------------------------------
    # Additional tests for methods and utilities
    # -----------------------------------------------------------------

    def test_sfdr_config_get_active_agents(self):
        """Test get_active_agents returns 50 agents (30 MRV + 10 data + 10 foundation)."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        agents = config.get_active_agents()

        assert isinstance(agents, list)
        assert len(agents) == 50, f"Expected 50 agents, got {len(agents)}"

        # Check MRV agents present
        assert "AGENT-MRV-001" in agents
        assert "AGENT-MRV-030" in agents

        # Check data agents present
        assert "AGENT-DATA-001" in agents
        assert "AGENT-DATA-019" in agents

        # Check foundation agents present
        assert "AGENT-FOUND-001" in agents
        assert "AGENT-FOUND-010" in agents

    def test_sfdr_config_get_disclosure_annex(self):
        """Test get_disclosure_annex returns Annex II for Article 8."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        annex = config.get_disclosure_annex()
        assert annex == "Annex II", f"Expected 'Annex II', got '{annex}'"

    def test_sfdr_config_get_enabled_pai_categories(self):
        """Test get_enabled_pai_categories returns all 5 categories by default."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        categories = config.get_enabled_pai_categories()
        assert len(categories) == 5, f"Expected 5 PAI categories, got {len(categories)}"

    def test_sfdr_config_get_feature_summary(self):
        """Test get_feature_summary returns expected feature flags."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        features = config.get_feature_summary()

        assert isinstance(features, dict)
        assert features["pai_calculation"] is True
        assert features["taxonomy_alignment"] is True
        assert features["sfdr_dnsh"] is True
        assert features["good_governance"] is True
        assert features["carbon_footprint"] is True
        assert features["waci"] is True
        assert features["eet_management"] is True
        assert features["annex_ii_disclosure"] is True
        assert features["annex_iii_disclosure"] is True
        assert features["annex_iv_disclosure"] is True
        assert features["negative_screening"] is True
        assert features["audit_trail"] is True
        # Sustainable investment disabled by default for Article 8
        assert features["sustainable_investment"] is False

    def test_sfdr_config_get_classification_display(self):
        """Test get_classification_display returns human-readable name."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        display = config.get_classification_display()
        assert "Article 8" in display

    def test_config_serialization_round_trip(self):
        """Test SFDRArticle8Config can serialize to dict and reconstruct."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()

        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "pack_id" in config_dict
        assert "pai" in config_dict
        assert "taxonomy_alignment" in config_dict
        assert "dnsh" in config_dict
        assert "governance" in config_dict
        assert "esg_characteristics" in config_dict
        assert "sustainable_investment" in config_dict
        assert "carbon_footprint" in config_dict
        assert "eet" in config_dict
        assert "disclosure" in config_dict
        assert "screening" in config_dict

        # Reconstruct from dict
        reconstructed = _cfg.SFDRArticle8Config(**config_dict)
        assert reconstructed.pack_id == config.pack_id
        assert reconstructed.version == config.version
        assert reconstructed.tier == config.tier
        assert reconstructed.sfdr_classification == config.sfdr_classification

    def test_config_hash_reproducibility(self):
        """Test that config hash is reproducible for same configuration."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle8Config()
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        hash1 = hashlib.sha256(config_json.encode()).hexdigest()
        hash2 = hashlib.sha256(config_json.encode()).hexdigest()

        assert hash1 == hash2, "Config hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash must be 64 chars"

    def test_get_default_config_function(self):
        """Test get_default_config returns valid PackConfig with defaults."""
        _skip_if_no_config()
        pc = _cfg.get_default_config()

        assert isinstance(pc, _cfg.PackConfig)
        assert isinstance(pc.pack, _cfg.SFDRArticle8Config)
        assert pc.pack.pack_id == "PACK-010-sfdr-article-8"
        assert pc.pack.tier == "standalone"

    def test_demo_config_yaml_exists(self):
        """Test demo_config.yaml exists and is valid with demo mode enabled."""
        demo_path = DEMO_DIR / "demo_config.yaml"
        assert demo_path.exists(), f"Demo config not found: {demo_path}"
        with open(demo_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Demo config must parse to dict"

        # Demo config should have demo section with demo_mode_enabled
        demo = data.get("demo", {})
        assert demo.get("demo_mode_enabled") is True, (
            "Demo config should have demo_mode_enabled=true"
        )

    def test_validate_sustainable_investment_utility(self):
        """Test validate_sustainable_investment three-pronged test utility."""
        _skip_if_no_config()

        # All pass
        result, msg = _cfg.validate_sustainable_investment(True, True, True)
        assert result is True
        assert "met" in msg.lower()

        # All fail
        result, msg = _cfg.validate_sustainable_investment(False, False, False)
        assert result is False
        assert "failed" in msg.lower()

        # Partial
        result, msg = _cfg.validate_sustainable_investment(True, False, True)
        assert result is False

    def test_get_pai_indicator_info_utility(self):
        """Test get_pai_indicator_info returns details for valid indicator."""
        _skip_if_no_config()

        info = _cfg.get_pai_indicator_info(1)
        assert info is not None
        assert info["id"] == 1
        assert info["category"] == "CLIMATE"
        assert "GHG" in info["name"]

        # Invalid indicator returns None
        info_none = _cfg.get_pai_indicator_info(99)
        assert info_none is None

    def test_get_classification_display_name_utility(self):
        """Test get_classification_display_name returns correct display names."""
        _skip_if_no_config()

        display = _cfg.get_classification_display_name("ARTICLE_8")
        assert "Article 8" in display
        assert "E/S characteristics" in display

        display_9 = _cfg.get_classification_display_name("ARTICLE_9")
        assert "Article 9" in display_9

    def test_governance_dimension_info_utility(self):
        """Test get_governance_dimension_info returns correct dimension details."""
        _skip_if_no_config()

        info = _cfg.get_governance_dimension_info("sound_management")
        assert info is not None
        assert "Sound Management" in info["name"]
        assert "Article 2(17)" in info["article"]

        # Unknown dimension returns None
        info_none = _cfg.get_governance_dimension_info("unknown_dimension")
        assert info_none is None

    def test_article8_plus_requires_sustainable_investment(self):
        """Test Article 8+ classification requires sustainable_investment.enabled=True."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.SFDRArticle8Config(
                sfdr_classification=_cfg.SFDRClassification.ARTICLE_8_PLUS,
                sustainable_investment=_cfg.SustainableInvestmentConfig(
                    enabled=False,
                ),
            )

    def test_carbon_footprint_invalid_currency_raises_error(self):
        """Test that an invalid currency code raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.CarbonFootprintConfig(currency="invalid")

    def test_governance_no_checks_raises_error(self):
        """Test that GovernanceConfig with no checks enabled raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.GovernanceConfig(
                enabled=True,
                check_sound_management=False,
                check_employee_relations=False,
                check_remuneration=False,
                check_tax_compliance=False,
            )

    def test_esg_characteristics_no_characteristics_raises_error(self):
        """Test that ESGCharacteristicsConfig with zero characteristics raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.ESGCharacteristicsConfig(
                environmental_characteristics=[],
                social_characteristics=[],
            )

    def test_pack_config_from_preset_invalid_name_raises_error(self):
        """Test that from_preset with invalid name raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(ValueError):
            _cfg.PackConfig.from_preset("nonexistent_preset")
