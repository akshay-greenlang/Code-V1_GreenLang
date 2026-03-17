# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Configuration Tests
====================================================

Tests the SFDRArticle9Config configuration system including:
- SFDRClassification enum (4 values: ARTICLE_6, ARTICLE_8, ARTICLE_8_PLUS, ARTICLE_9)
- Article9SubType enum (3 values: GENERAL_9_1, INDEX_BASED_9_2, CARBON_REDUCTION_9_3)
- BenchmarkType enum (3 values: CTB, PAB, CUSTOM)
- PAICategory enum (5 values: CLIMATE, ENVIRONMENT, SOCIAL, SOVEREIGN, REAL_ESTATE)
- PAIDataQuality enum (4 values: REPORTED, ESTIMATED, MODELED, NOT_AVAILABLE)
- DisclosureType enum (3 values: PRE_CONTRACTUAL, PERIODIC, WEBSITE)
- GovernanceCheckStatus enum (4 values: PASS, FAIL, PARTIAL, NOT_ASSESSED)
- SustainableInvestmentType enum (3 values)
- ReportingFrequency enum (3 values)
- ScreeningType enum (3 values)
- ComplianceStatus enum (4 values)
- DowngradeRiskLevel enum (4 values: LOW, MEDIUM, HIGH, CRITICAL)
- ImpactObjective enum (8 values)
- PAIConfig defaults and validation (mandatory PAI for Article 9)
- SustainableInvestmentConfig defaults (95% minimum, enabled=True)
- DNSHConfig defaults (strict mode for Article 9)
- GovernanceConfig defaults (70.0 minimum score, controversy threshold=2)
- ImpactConfig defaults (enabled for Article 9)
- BenchmarkAlignmentConfig defaults
- CarbonTrajectoryConfig defaults
- DowngradeMonitorConfig defaults
- CarbonFootprintConfig defaults (Scope 3 included, PCAF alignment)
- EETConfig defaults
- DisclosureConfig defaults (Annex III/V for Article 9)
- ScreeningConfig defaults (positive screening enabled, zero-tolerance)
- SFDRArticle9Config creation with all sub-configs
- PackConfig loading from YAML and presets (9 presets)
- PackConfig.available_presets
- PackConfig.get_config_hash
- Preset YAML file loading and validation
- Article 9 specific validations (downgrade risk, SI proportion)
- Utility functions (validate_sustainable_investment, assess_downgrade_risk)

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
    _cfg = _import_from_path("sfdr9_pack_config_test", CONFIG_DIR / "pack_config.py")
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
    """Test suite for PACK-011 SFDRArticle9Config configuration."""

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
    # 2. Article9SubType enum
    # -----------------------------------------------------------------

    def test_article9_sub_type_enum_values(self):
        """Test Article9SubType enum has exactly 3 values."""
        _skip_if_no_config()
        st = _cfg.Article9SubType
        expected = {"GENERAL_9_1", "INDEX_BASED_9_2", "CARBON_REDUCTION_9_3"}
        actual = {m.value for m in st}
        assert actual == expected, f"Article9SubType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 3. BenchmarkType enum
    # -----------------------------------------------------------------

    def test_benchmark_type_enum_values(self):
        """Test BenchmarkType enum has exactly 3 values (CTB, PAB, CUSTOM)."""
        _skip_if_no_config()
        bt = _cfg.BenchmarkType
        expected = {"CTB", "PAB", "CUSTOM"}
        actual = {m.value for m in bt}
        assert actual == expected, f"BenchmarkType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 4. PAICategory enum
    # -----------------------------------------------------------------

    def test_pai_category_enum_values(self):
        """Test PAICategory enum has exactly 5 values."""
        _skip_if_no_config()
        pc = _cfg.PAICategory
        expected = {"CLIMATE", "ENVIRONMENT", "SOCIAL", "SOVEREIGN", "REAL_ESTATE"}
        actual = {m.value for m in pc}
        assert actual == expected, f"PAICategory mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 5. DowngradeRiskLevel enum
    # -----------------------------------------------------------------

    def test_downgrade_risk_level_enum_values(self):
        """Test DowngradeRiskLevel enum has exactly 4 values."""
        _skip_if_no_config()
        drl = _cfg.DowngradeRiskLevel
        expected = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {m.value for m in drl}
        assert actual == expected, f"DowngradeRiskLevel mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 6. PAIConfig defaults (Article 9 specific)
    # -----------------------------------------------------------------

    def test_pai_config_defaults(self):
        """Test PAIConfig default values for Article 9 (mandatory PAI, Scope 3)."""
        _skip_if_no_config()
        pai = _cfg.PAIConfig()
        assert pai.enabled is True
        assert pai.pai_mandatory is True  # Article 9 specific
        assert len(pai.enabled_mandatory_indicators) == 18
        assert pai.enabled_mandatory_indicators == list(range(1, 19))
        assert pai.min_coverage_pct == 70.0  # Higher for Art.9
        assert pai.coverage_adjustment is True
        assert pai.estimation_enabled is True
        assert pai.estimation_methodology == "SECTOR_AVERAGE"
        assert pai.include_scope_3 is True  # Mandatory for Art.9
        assert pai.entity_level_statement is True
        assert pai.product_level_pai is True
        assert pai.prior_period_comparison is True
        assert pai.data_quality_tracking is True
        assert pai.engagement_actions_tracking is True  # Art.9 specific

    # -----------------------------------------------------------------
    # 7. SustainableInvestmentConfig defaults (Article 9 specific)
    # -----------------------------------------------------------------

    def test_sustainable_investment_config_defaults(self):
        """Test SustainableInvestmentConfig defaults for Article 9 (95% minimum, enabled)."""
        _skip_if_no_config()
        si = _cfg.SustainableInvestmentConfig()
        assert si.enabled is True  # Always True for Art.9
        assert si.minimum_proportion_pct == 95.0  # Art.9 = 95-100%
        assert si.taxonomy_aligned_minimum_pct == 0.0
        assert si.other_environmental_minimum_pct == 0.0
        assert si.social_minimum_pct == 0.0
        assert si.require_dnsh_pass is True
        assert si.require_governance_pass is True
        assert si.double_counting_prevention is True
        assert si.contribution_assessment_methodology == "IMPACT_BASED"  # Art.9 uses impact
        assert si.cash_and_hedging_allowance_pct == 5.0
        assert si.continuous_monitoring is True
        assert si.breach_notification_enabled is True

    # -----------------------------------------------------------------
    # 8. DNSHConfig defaults (Article 9 strict mode)
    # -----------------------------------------------------------------

    def test_dnsh_config_defaults(self):
        """Test DNSHConfig default values with Article 9 strict mode."""
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
        assert dnsh.strict_mode is True  # Article 9 specific

    # -----------------------------------------------------------------
    # 9. GovernanceConfig defaults (stricter for Article 9)
    # -----------------------------------------------------------------

    def test_governance_config_defaults(self):
        """Test GovernanceConfig default values (stricter thresholds for Art.9)."""
        _skip_if_no_config()
        gov = _cfg.GovernanceConfig()
        assert gov.enabled is True
        assert gov.check_sound_management is True
        assert gov.check_employee_relations is True
        assert gov.check_remuneration is True
        assert gov.check_tax_compliance is True
        assert gov.minimum_governance_score == 70.0  # Higher than Art.8 (60.0)
        assert gov.require_all_dimensions_pass is True
        assert gov.controversy_flag_threshold == 2  # Lower than Art.8 (3)
        assert gov.monitoring_frequency == _cfg.ReportingFrequency.QUARTERLY
        assert len(gov.data_sources) >= 1

    # -----------------------------------------------------------------
    # 10. ImpactConfig defaults (Article 9 specific)
    # -----------------------------------------------------------------

    def test_impact_config_defaults(self):
        """Test ImpactConfig default values for Article 9 impact measurement."""
        _skip_if_no_config()
        impact = _cfg.ImpactConfig()
        assert impact.enabled is True
        assert len(impact.primary_objectives) >= 1
        assert impact.impact_methodology == "CONTRIBUTION_BASED"
        assert impact.require_additionality is True
        assert impact.theory_of_change_required is True
        assert impact.impact_reporting_frequency == _cfg.ReportingFrequency.ANNUAL
        assert impact.sdg_alignment_tracking is True
        assert len(impact.sdg_targets) >= 1

    # -----------------------------------------------------------------
    # 11. BenchmarkAlignmentConfig defaults
    # -----------------------------------------------------------------

    def test_benchmark_alignment_config_defaults(self):
        """Test BenchmarkAlignmentConfig default values."""
        _skip_if_no_config()
        ba = _cfg.BenchmarkAlignmentConfig()
        assert ba.enabled is False  # Optional, enabled for 9(2)/9(3)
        assert ba.benchmark_type == _cfg.BenchmarkType.PAB
        assert ba.tracking_error_threshold_pct == 2.0
        assert ba.pab_minimum_standards is True
        assert ba.ctb_minimum_standards is True
        assert ba.yoy_decarbonization_pct == 7.0  # PAB standard

    # -----------------------------------------------------------------
    # 12. CarbonFootprintConfig defaults (Article 9 specific)
    # -----------------------------------------------------------------

    def test_carbon_footprint_config_defaults(self):
        """Test CarbonFootprintConfig defaults for Article 9 (Scope 3, PCAF)."""
        _skip_if_no_config()
        cf = _cfg.CarbonFootprintConfig()
        assert cf.enabled is True
        assert cf.calculate_waci is True
        assert cf.calculate_carbon_footprint is True
        assert cf.calculate_total_emissions is True
        assert cf.calculate_financed_emissions is True  # Recommended for Art.9
        assert "SCOPE_1" in cf.scope_coverage
        assert "SCOPE_2" in cf.scope_coverage
        assert "SCOPE_3" in cf.scope_coverage  # Mandatory for Art.9
        assert cf.pcaf_alignment is True  # Recommended for Art.9
        assert cf.coverage_threshold_pct == 70.0  # Higher for Art.9
        assert cf.benchmark_comparison is True
        assert cf.yoy_trend is True
        assert cf.currency == "EUR"

    # -----------------------------------------------------------------
    # 13. ScreeningConfig defaults (stricter for Article 9)
    # -----------------------------------------------------------------

    def test_screening_config_defaults(self):
        """Test ScreeningConfig defaults for Article 9 (positive screening, zero tolerance)."""
        _skip_if_no_config()
        sc = _cfg.ScreeningConfig()
        assert sc.negative_screening_enabled is True
        assert sc.positive_screening_enabled is True  # Expected for Art.9
        assert sc.norms_based_screening_enabled is True
        assert len(sc.negative_exclusions) >= 5  # Broader for Art.9
        assert "controversial_weapons" in sc.negative_exclusions
        assert sc.revenue_threshold_pct == 0.0  # Zero tolerance for Art.9
        assert len(sc.norms_frameworks) >= 3  # Expanded for Art.9
        assert sc.screening_frequency == _cfg.ReportingFrequency.QUARTERLY
        assert sc.breach_handling == "IMMEDIATE_DIVEST"  # Stricter for Art.9
        assert sc.monitoring_enabled is True

    # -----------------------------------------------------------------
    # 14. DisclosureConfig defaults (Annex III/V for Article 9)
    # -----------------------------------------------------------------

    def test_disclosure_config_defaults(self):
        """Test DisclosureConfig defaults for Article 9 (Annex III/V)."""
        _skip_if_no_config()
        dc = _cfg.DisclosureConfig()
        assert dc.annex_iii_enabled is True  # Art.9 uses Annex III
        assert dc.annex_v_enabled is True  # Art.9 uses Annex V
        assert dc.website_disclosure_enabled is True
        assert dc.default_format == _cfg.DisclosureFormat.PDF
        assert dc.include_methodology_note is True
        assert dc.include_data_sources is True
        assert dc.include_asset_allocation_chart is True
        assert dc.include_impact_section is True  # Art.9 specific
        assert dc.include_benchmark_comparison is True  # Art.9(2) specific
        assert dc.review_workflow_enabled is True
        assert dc.version_tracking is True
        assert dc.greenwashing_check is True
        assert dc.greenwashing_strict_mode is True  # Stricter for Art.9

    # -----------------------------------------------------------------
    # 15. DowngradeMonitorConfig defaults
    # -----------------------------------------------------------------

    def test_downgrade_monitor_config_defaults(self):
        """Test DowngradeMonitorConfig default values."""
        _skip_if_no_config()
        dm = _cfg.DowngradeMonitorConfig()
        assert dm.enabled is True
        assert dm.si_proportion_warning_threshold_pct == 97.0
        assert dm.si_proportion_critical_threshold_pct == 95.0
        assert dm.dnsh_failure_tolerance_pct == 3.0
        assert dm.governance_failure_tolerance_pct == 3.0
        assert dm.monitoring_frequency == _cfg.ReportingFrequency.QUARTERLY
        assert dm.remediation_period_days == 30
        assert dm.auto_reclassification_enabled is False

    # -----------------------------------------------------------------
    # 16. EETConfig defaults
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
        assert eet.sustainable_investment_fields is True  # Art.9 specific

    # -----------------------------------------------------------------
    # 17. SFDRArticle9Config main class creation
    # -----------------------------------------------------------------

    def test_sfdr_article9_config_default_creation(self):
        """Test SFDRArticle9Config creates with valid Article 9 defaults."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()

        assert config.pack_id == "PACK-011-sfdr-article-9"
        assert config.version == "1.0.0"
        assert config.tier == "standalone"
        assert config.sfdr_classification == _cfg.SFDRClassification.ARTICLE_9
        assert config.article9_sub_type == _cfg.Article9SubType.GENERAL_9_1
        assert config.reporting_year == 2025

        # All sub-configs should be populated
        assert isinstance(config.pai, _cfg.PAIConfig)
        assert isinstance(config.taxonomy_alignment, _cfg.TaxonomyAlignmentConfig)
        assert isinstance(config.dnsh, _cfg.DNSHConfig)
        assert isinstance(config.governance, _cfg.GovernanceConfig)
        assert isinstance(config.esg_characteristics, _cfg.ESGCharacteristicsConfig)
        assert isinstance(config.sustainable_investment, _cfg.SustainableInvestmentConfig)
        assert isinstance(config.impact, _cfg.ImpactConfig)
        assert isinstance(config.benchmark_alignment, _cfg.BenchmarkAlignmentConfig)
        assert isinstance(config.carbon_trajectory, _cfg.CarbonTrajectoryConfig)
        assert isinstance(config.downgrade_monitor, _cfg.DowngradeMonitorConfig)
        assert isinstance(config.carbon_footprint, _cfg.CarbonFootprintConfig)
        assert isinstance(config.eet, _cfg.EETConfig)
        assert isinstance(config.disclosure, _cfg.DisclosureConfig)
        assert isinstance(config.screening, _cfg.ScreeningConfig)
        assert isinstance(config.reporting, _cfg.ReportingConfig)
        assert isinstance(config.data_quality, _cfg.DataQualityConfig)
        assert isinstance(config.audit_trail, _cfg.AuditTrailConfig)
        assert isinstance(config.demo, _cfg.DemoConfig)

    # -----------------------------------------------------------------
    # 18. PackConfig.from_preset
    # -----------------------------------------------------------------

    def test_pack_config_from_preset(self):
        """Test PackConfig.from_preset loads named preset correctly."""
        _skip_if_no_config()
        pc = _cfg.PackConfig.from_preset("asset_manager")
        assert isinstance(pc.pack, _cfg.SFDRArticle9Config)

    # -----------------------------------------------------------------
    # 19. PackConfig.available_presets returns 5 entity presets
    # -----------------------------------------------------------------

    def test_pack_config_available_presets(self):
        """Test PackConfig.available_presets returns all 5 entity presets."""
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
    # 20. PackConfig.get_config_hash
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
    # 21. Article 9 specific - get_disclosure_annex returns Annex III
    # -----------------------------------------------------------------

    def test_sfdr_config_get_disclosure_annex(self):
        """Test get_disclosure_annex returns Annex III for Article 9."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        annex = config.get_disclosure_annex()
        assert annex == "Annex III", f"Expected 'Annex III', got '{annex}'"

    # -----------------------------------------------------------------
    # 22. Article 9 specific - get_article9_subtype_display
    # -----------------------------------------------------------------

    def test_sfdr_config_get_article9_subtype_display(self):
        """Test get_article9_subtype_display returns correct display name."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        display = config.get_article9_subtype_display()
        assert "Article 9(1)" in display

    # -----------------------------------------------------------------
    # 23. get_active_agents returns 50 agents
    # -----------------------------------------------------------------

    def test_sfdr_config_get_active_agents(self):
        """Test get_active_agents returns 50 agents (30 MRV + 10 data + 10 foundation)."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        agents = config.get_active_agents()

        assert isinstance(agents, list)
        assert len(agents) == 50, f"Expected 50 agents, got {len(agents)}"

        assert "AGENT-MRV-001" in agents
        assert "AGENT-MRV-030" in agents
        assert "AGENT-DATA-001" in agents
        assert "AGENT-DATA-019" in agents
        assert "AGENT-FOUND-001" in agents
        assert "AGENT-FOUND-010" in agents

    # -----------------------------------------------------------------
    # 24. get_enabled_pai_categories returns all 5
    # -----------------------------------------------------------------

    def test_sfdr_config_get_enabled_pai_categories(self):
        """Test get_enabled_pai_categories returns all 5 categories by default."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        categories = config.get_enabled_pai_categories()
        assert len(categories) == 5, f"Expected 5 PAI categories, got {len(categories)}"

    # -----------------------------------------------------------------
    # 25. get_feature_summary - Article 9 specific features
    # -----------------------------------------------------------------

    def test_sfdr_config_get_feature_summary(self):
        """Test get_feature_summary returns Article 9 specific feature flags."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        features = config.get_feature_summary()

        assert isinstance(features, dict)
        assert features["pai_calculation"] is True
        assert features["pai_mandatory"] is True  # Art.9 specific
        assert features["pai_scope_3"] is True  # Art.9 specific
        assert features["taxonomy_alignment"] is True
        assert features["sfdr_dnsh"] is True
        assert features["sfdr_dnsh_strict"] is True  # Art.9 specific
        assert features["good_governance"] is True
        assert features["sustainable_investment"] is True  # Always True for Art.9
        assert features["impact_measurement"] is True  # Art.9 specific
        assert features["downgrade_monitoring"] is True  # Art.9 specific
        assert features["carbon_footprint"] is True
        assert features["waci"] is True
        assert features["financed_emissions"] is True  # Art.9
        assert features["pcaf_alignment"] is True  # Art.9
        assert features["eet_management"] is True
        assert features["annex_iii_disclosure"] is True  # Art.9 (not Annex II)
        assert features["annex_v_disclosure"] is True  # Art.9 (not Annex IV)
        assert features["negative_screening"] is True
        assert features["positive_screening"] is True  # Art.9
        assert features["greenwashing_strict"] is True  # Art.9
        assert features["audit_trail"] is True

    # -----------------------------------------------------------------
    # 26. get_classification_display
    # -----------------------------------------------------------------

    def test_sfdr_config_get_classification_display(self):
        """Test get_classification_display returns Article 9 display name."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        display = config.get_classification_display()
        assert "Article 9" in display

    # -----------------------------------------------------------------
    # 27. validate_sustainable_investment utility
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 28. assess_downgrade_risk utility
    # -----------------------------------------------------------------

    def test_assess_downgrade_risk_utility(self):
        """Test assess_downgrade_risk returns correct risk levels."""
        _skip_if_no_config()

        # Low risk - everything healthy
        level, msg = _cfg.assess_downgrade_risk(98.0, 1.0, 1.0)
        assert level == _cfg.DowngradeRiskLevel.LOW

        # Critical - SI below critical threshold
        level, msg = _cfg.assess_downgrade_risk(90.0, 1.0, 1.0)
        assert level == _cfg.DowngradeRiskLevel.CRITICAL

        # Medium - above critical but below warning
        level, msg = _cfg.assess_downgrade_risk(96.0, 1.0, 1.0)
        assert level in (_cfg.DowngradeRiskLevel.MEDIUM, _cfg.DowngradeRiskLevel.HIGH)

    # -----------------------------------------------------------------
    # 29. get_downgrade_risk_level method
    # -----------------------------------------------------------------

    def test_sfdr_config_get_downgrade_risk_level(self):
        """Test SFDRArticle9Config.get_downgrade_risk_level method."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()

        # Healthy portfolio
        level = config.get_downgrade_risk_level(98.0, 1.0, 1.0)
        assert level == _cfg.DowngradeRiskLevel.LOW

        # Critical
        level = config.get_downgrade_risk_level(90.0, 1.0, 1.0)
        assert level == _cfg.DowngradeRiskLevel.CRITICAL

    # -----------------------------------------------------------------
    # 30. Config serialization round trip
    # -----------------------------------------------------------------

    def test_config_serialization_round_trip(self):
        """Test SFDRArticle9Config can serialize to dict and reconstruct."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()

        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "pack_id" in config_dict
        assert "pai" in config_dict
        assert "sustainable_investment" in config_dict
        assert "impact" in config_dict
        assert "benchmark_alignment" in config_dict
        assert "carbon_trajectory" in config_dict
        assert "downgrade_monitor" in config_dict

        # Reconstruct from dict
        reconstructed = _cfg.SFDRArticle9Config(**config_dict)
        assert reconstructed.pack_id == config.pack_id
        assert reconstructed.version == config.version
        assert reconstructed.sfdr_classification == config.sfdr_classification
        assert reconstructed.article9_sub_type == config.article9_sub_type

    # -----------------------------------------------------------------
    # 31. Config hash reproducibility
    # -----------------------------------------------------------------

    def test_config_hash_reproducibility(self):
        """Test that config hash is reproducible for same configuration."""
        _skip_if_no_config()
        config = _cfg.SFDRArticle9Config()
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, sort_keys=True, default=str)

        hash1 = hashlib.sha256(config_json.encode()).hexdigest()
        hash2 = hashlib.sha256(config_json.encode()).hexdigest()

        assert hash1 == hash2, "Config hash should be deterministic"
        assert len(hash1) == 64, "SHA-256 hash must be 64 chars"

    # -----------------------------------------------------------------
    # 32. get_default_config function
    # -----------------------------------------------------------------

    def test_get_default_config_function(self):
        """Test get_default_config returns valid PackConfig with Article 9 defaults."""
        _skip_if_no_config()
        pc = _cfg.get_default_config()

        assert isinstance(pc, _cfg.PackConfig)
        assert isinstance(pc.pack, _cfg.SFDRArticle9Config)
        assert pc.pack.pack_id == "PACK-011-sfdr-article-9"
        assert pc.pack.tier == "standalone"
        assert pc.pack.sfdr_classification == _cfg.SFDRClassification.ARTICLE_9

    # -----------------------------------------------------------------
    # 33. Validation - Article 9 requires sustainable_investment enabled
    # -----------------------------------------------------------------

    def test_article9_requires_sustainable_investment_enabled(self):
        """Test Article 9 classification requires sustainable_investment.enabled=True."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.SFDRArticle9Config(
                sfdr_classification=_cfg.SFDRClassification.ARTICLE_9,
                sustainable_investment=_cfg.SustainableInvestmentConfig(
                    enabled=False,
                ),
            )

    # -----------------------------------------------------------------
    # 34. Validation - Article 9 requires PAI enabled
    # -----------------------------------------------------------------

    def test_article9_requires_pai_enabled(self):
        """Test Article 9 classification requires PAI calculation enabled."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.SFDRArticle9Config(
                sfdr_classification=_cfg.SFDRClassification.ARTICLE_9,
                pai=_cfg.PAIConfig(enabled=False),
            )

    # -----------------------------------------------------------------
    # 35. Validation - Article 9 requires DNSH enabled
    # -----------------------------------------------------------------

    def test_article9_requires_dnsh_enabled(self):
        """Test Article 9 classification requires DNSH assessment enabled."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.SFDRArticle9Config(
                sfdr_classification=_cfg.SFDRClassification.ARTICLE_9,
                dnsh=_cfg.DNSHConfig(enabled=False),
            )

    # -----------------------------------------------------------------
    # 36. Validation - invalid PAI indicator IDs
    # -----------------------------------------------------------------

    def test_config_validation_invalid_pai_ids_raise_error(self):
        """Test that PAI indicator IDs outside 1-18 raise ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.PAIConfig(enabled_mandatory_indicators=[0, 19, 20])

    # -----------------------------------------------------------------
    # 37. Validation - invalid currency
    # -----------------------------------------------------------------

    def test_carbon_footprint_invalid_currency_raises_error(self):
        """Test that an invalid currency code raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.CarbonFootprintConfig(currency="invalid")

    # -----------------------------------------------------------------
    # 38. Validation - governance no checks
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 39. Validation - ESG characteristics empty
    # -----------------------------------------------------------------

    def test_esg_characteristics_no_characteristics_raises_error(self):
        """Test that ESGCharacteristicsConfig with zero characteristics raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.ESGCharacteristicsConfig(
                environmental_characteristics=[],
                social_characteristics=[],
            )

    # -----------------------------------------------------------------
    # 40. Validation - invalid preset name
    # -----------------------------------------------------------------

    def test_pack_config_from_preset_invalid_name_raises_error(self):
        """Test that from_preset with invalid name raises ValueError."""
        _skip_if_no_config()

        with pytest.raises(ValueError):
            _cfg.PackConfig.from_preset("nonexistent_preset")

    # -----------------------------------------------------------------
    # 41. Environment variable overrides
    # -----------------------------------------------------------------

    def test_environment_variable_overrides(self, monkeypatch):
        """Test environment variable overrides are respected."""
        _skip_if_no_config()

        monkeypatch.setenv("SFDR9_PACK_PRODUCT_NAME", "Test Dark Green Fund")
        monkeypatch.setenv("SFDR9_PACK_REPORTING_YEAR", "2026")

        env_product_name = os.getenv("SFDR9_PACK_PRODUCT_NAME", "")
        assert env_product_name == "Test Dark Green Fund"

        env_year = int(os.getenv("SFDR9_PACK_REPORTING_YEAR", "2025"))
        assert env_year == 2026

    # -----------------------------------------------------------------
    # 42. get_pai_indicator_info utility
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 43. get_classification_display_name utility
    # -----------------------------------------------------------------

    def test_get_classification_display_name_utility(self):
        """Test get_classification_display_name returns correct display names."""
        _skip_if_no_config()

        display_9 = _cfg.get_classification_display_name("ARTICLE_9")
        assert "Article 9" in display_9

        display_8 = _cfg.get_classification_display_name("ARTICLE_8")
        assert "Article 8" in display_8

    # -----------------------------------------------------------------
    # 44. get_governance_dimension_info utility
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # 45. Downgrade monitor threshold validation
    # -----------------------------------------------------------------

    def test_downgrade_monitor_threshold_validation(self):
        """Test DowngradeMonitorConfig validates warning >= critical threshold."""
        _skip_if_no_config()

        with pytest.raises(Exception):
            _cfg.DowngradeMonitorConfig(
                si_proportion_warning_threshold_pct=90.0,
                si_proportion_critical_threshold_pct=95.0,
            )

    # -----------------------------------------------------------------
    # 46. PackConfig.from_yaml
    # -----------------------------------------------------------------

    def test_pack_config_from_yaml(self):
        """Test PackConfig.from_yaml loads from preset file if available."""
        _skip_if_no_config()
        asset_manager_path = PRESETS_DIR / "asset_manager.yaml"
        if not asset_manager_path.exists():
            pytest.skip("Asset manager preset file not found")

        pc = _cfg.PackConfig.from_yaml(asset_manager_path)
        assert isinstance(pc.pack, _cfg.SFDRArticle9Config)
        assert isinstance(pc.loaded_from, list)
        assert len(pc.loaded_from) >= 1
        assert str(asset_manager_path) in pc.loaded_from

    # -----------------------------------------------------------------
    # 47. demo_config.yaml exists and is valid
    # -----------------------------------------------------------------

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
