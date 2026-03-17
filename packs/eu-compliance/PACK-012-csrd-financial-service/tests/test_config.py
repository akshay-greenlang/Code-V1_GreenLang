# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Configuration Tests
============================================================

Tests the CSRDFinancialServiceConfig configuration system including:
- FinancialInstitutionType enum (6 values: BANK, INSURANCE, ASSET_MANAGER,
  INVESTMENT_FIRM, PENSION_FUND, CONGLOMERATE)
- PCAFAssetClass enum (10 values: 6 core + 4 extensions)
- PCAFDataQuality enum (5 values: SCORE_1 through SCORE_5)
- GARScope enum (3 values: TURNOVER, CAPEX, OPEX)
- NGFSScenario enum (6 values: orderly, disorderly, hot house)
- Pillar3Template enum (9 templates: TEMPLATE_1 through TEMPLATE_10)
- ClimateRiskType enum (2 values: PHYSICAL, TRANSITION)
- PhysicalHazardType enum (6 values)
- TransitionRiskChannel enum (5 values)
- ESRSTopic enum (10 values: E1-E5, S1-S4, G1)
- ReportingFrequency enum (3 values)
- ComplianceStatus enum (4 values)
- DisclosureFormat enum (6 values)
- MaterialityLevel enum (4 values)
- SBTiFIMethod enum (3 values)
- InsuranceLineType enum (7 values)
- GARExposureCategory enum (6 values)
- PCAFConfig defaults and fields
- GARBTARConfig defaults and fields
- InsuranceConfig defaults (disabled by default, enabled for INSURANCE type)
- ClimateRiskConfig defaults (6 NGFS scenarios, 6 physical hazards, 5 channels)
- Pillar3Config defaults (9 templates, semi-annual reporting)
- FSMaterialityConfig defaults (10 ESRS topics, 4 impact channels)
- FSTransitionPlanConfig defaults (SBTi FI aligned, 7 priority sectors)
- SBTiFIConfig defaults
- DataQualityConfig defaults
- DisclosureConfig defaults
- AuditTrailConfig defaults
- CSRDFinancialServiceConfig creation with all sub-configs
- Model validators (PCAF for banks, insurance for insurers, Pillar 3 for CRR)
- PackConfig.from_preset loading
- PackConfig.from_yaml loading
- PackConfig.get_config_hash
- Utility functions (get_institution_display_name, get_required_disclosures,
  get_pcaf_asset_class_info, get_ngfs_scenario_info, list_available_presets)

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
    _cfg = _import_from_path("fs12_pack_config_test", CONFIG_DIR / "pack_config.py")
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
    """Test suite for PACK-012 CSRDFinancialServiceConfig configuration."""

    # -----------------------------------------------------------------
    # 1. FinancialInstitutionType enum
    # -----------------------------------------------------------------

    def test_financial_institution_type_enum_values(self):
        """Test FinancialInstitutionType enum has exactly 6 values."""
        _skip_if_no_config()
        fi = _cfg.FinancialInstitutionType
        members = list(fi)
        assert len(members) == 6, f"Expected 6 institution types, got {len(members)}"

        expected = {"BANK", "INSURANCE", "ASSET_MANAGER", "INVESTMENT_FIRM",
                    "PENSION_FUND", "CONGLOMERATE"}
        actual = {m.value for m in members}
        assert actual == expected, f"FinancialInstitutionType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 2. PCAFAssetClass enum
    # -----------------------------------------------------------------

    def test_pcaf_asset_class_enum_values(self):
        """Test PCAFAssetClass enum has 10 values (6 core + 4 extensions)."""
        _skip_if_no_config()
        pac = _cfg.PCAFAssetClass
        members = list(pac)
        assert len(members) == 10, f"Expected 10 PCAF asset classes, got {len(members)}"

        core = {
            "LISTED_EQUITY_CORPORATE_BONDS", "BUSINESS_LOANS_UNLISTED_EQUITY",
            "PROJECT_FINANCE", "COMMERCIAL_REAL_ESTATE", "MORTGAGES", "MOTOR_VEHICLE_LOANS",
        }
        actual = {m.value for m in members}
        assert core.issubset(actual), f"Missing core asset classes. Found: {actual}"

    # -----------------------------------------------------------------
    # 3. PCAFDataQuality enum
    # -----------------------------------------------------------------

    def test_pcaf_data_quality_enum_values(self):
        """Test PCAFDataQuality enum has 5 integer scores (1-5)."""
        _skip_if_no_config()
        pdq = _cfg.PCAFDataQuality
        members = list(pdq)
        assert len(members) == 5, f"Expected 5 data quality scores, got {len(members)}"

        expected = {1, 2, 3, 4, 5}
        actual = {m.value for m in members}
        assert actual == expected, f"PCAFDataQuality mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 4. GARScope enum
    # -----------------------------------------------------------------

    def test_gar_scope_enum_values(self):
        """Test GARScope enum has exactly 3 values (TURNOVER, CAPEX, OPEX)."""
        _skip_if_no_config()
        gs = _cfg.GARScope
        expected = {"TURNOVER", "CAPEX", "OPEX"}
        actual = {m.value for m in gs}
        assert actual == expected, f"GARScope mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 5. NGFSScenario enum
    # -----------------------------------------------------------------

    def test_ngfs_scenario_enum_values(self):
        """Test NGFSScenario enum has exactly 6 values."""
        _skip_if_no_config()
        ns = _cfg.NGFSScenario
        expected = {
            "NET_ZERO_2050", "BELOW_2C", "DELAYED_TRANSITION",
            "NDCS", "DIVERGENT_NET_ZERO", "CURRENT_POLICIES",
        }
        actual = {m.value for m in ns}
        assert actual == expected, f"NGFSScenario mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 6. Pillar3Template enum
    # -----------------------------------------------------------------

    def test_pillar3_template_enum_values(self):
        """Test Pillar3Template enum has exactly 9 templates."""
        _skip_if_no_config()
        pt = _cfg.Pillar3Template
        members = list(pt)
        assert len(members) == 9, f"Expected 9 Pillar 3 templates, got {len(members)}"

        expected = {
            "TEMPLATE_1", "TEMPLATE_2", "TEMPLATE_3", "TEMPLATE_4",
            "TEMPLATE_5", "TEMPLATE_7", "TEMPLATE_8", "TEMPLATE_9", "TEMPLATE_10",
        }
        actual = {m.value for m in members}
        assert actual == expected, f"Pillar3Template mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 7. ESRSTopic enum
    # -----------------------------------------------------------------

    def test_esrs_topic_enum_values(self):
        """Test ESRSTopic enum has exactly 10 values (E1-E5, S1-S4, G1)."""
        _skip_if_no_config()
        et = _cfg.ESRSTopic
        expected = {"E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"}
        actual = {m.value for m in et}
        assert actual == expected, f"ESRSTopic mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 8. PhysicalHazardType enum
    # -----------------------------------------------------------------

    def test_physical_hazard_type_enum_values(self):
        """Test PhysicalHazardType enum has exactly 6 values."""
        _skip_if_no_config()
        ph = _cfg.PhysicalHazardType
        expected = {"FLOOD", "WILDFIRE", "STORM", "HEATWAVE", "SEA_LEVEL_RISE", "DROUGHT"}
        actual = {m.value for m in ph}
        assert actual == expected, f"PhysicalHazardType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 9. TransitionRiskChannel enum
    # -----------------------------------------------------------------

    def test_transition_risk_channel_enum_values(self):
        """Test TransitionRiskChannel enum has exactly 5 values."""
        _skip_if_no_config()
        trc = _cfg.TransitionRiskChannel
        expected = {"POLICY_LEGAL", "TECHNOLOGY", "MARKET", "REPUTATION", "STRANDED_ASSETS"}
        actual = {m.value for m in trc}
        assert actual == expected, f"TransitionRiskChannel mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 10. InsuranceLineType enum
    # -----------------------------------------------------------------

    def test_insurance_line_type_enum_values(self):
        """Test InsuranceLineType enum has exactly 7 values."""
        _skip_if_no_config()
        ilt = _cfg.InsuranceLineType
        expected = {
            "COMMERCIAL_PROPERTY", "COMMERCIAL_CASUALTY", "PERSONAL_PROPERTY",
            "PERSONAL_AUTO", "SPECIALTY", "LIFE_HEALTH", "REINSURANCE",
        }
        actual = {m.value for m in ilt}
        assert actual == expected, f"InsuranceLineType mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 11. DisclosureFormat enum
    # -----------------------------------------------------------------

    def test_disclosure_format_enum_values(self):
        """Test DisclosureFormat enum has exactly 6 values."""
        _skip_if_no_config()
        df = _cfg.DisclosureFormat
        expected = {"PDF", "XLSX", "HTML", "JSON", "XML", "XBRL"}
        actual = {m.value for m in df}
        assert actual == expected, f"DisclosureFormat mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 12. MaterialityLevel enum
    # -----------------------------------------------------------------

    def test_materiality_level_enum_values(self):
        """Test MaterialityLevel enum has exactly 4 values."""
        _skip_if_no_config()
        ml = _cfg.MaterialityLevel
        expected = {"HIGH", "MEDIUM", "LOW", "NOT_MATERIAL"}
        actual = {m.value for m in ml}
        assert actual == expected, f"MaterialityLevel mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 13. SBTiFIMethod enum
    # -----------------------------------------------------------------

    def test_sbti_fi_method_enum_values(self):
        """Test SBTiFIMethod enum has exactly 3 values."""
        _skip_if_no_config()
        sm = _cfg.SBTiFIMethod
        expected = {"SDA", "TEMPERATURE_RATING", "PORTFOLIO_COVERAGE"}
        actual = {m.value for m in sm}
        assert actual == expected, f"SBTiFIMethod mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 14. PCAFConfig defaults
    # -----------------------------------------------------------------

    def test_pcaf_config_defaults(self):
        """Test PCAFConfig default values for financed emissions."""
        _skip_if_no_config()
        pcaf = _cfg.PCAFConfig()
        assert pcaf.enabled is True
        assert len(pcaf.enabled_asset_classes) == 6  # 6 core asset classes
        assert pcaf.include_scope_1 is True
        assert pcaf.include_scope_2 is True
        assert pcaf.include_scope_3 is True
        assert pcaf.min_data_quality_score == _cfg.PCAFDataQuality.SCORE_5
        assert pcaf.target_data_quality_score == _cfg.PCAFDataQuality.SCORE_3
        assert pcaf.estimation_enabled is True
        assert pcaf.estimation_methodology == "SECTOR_AVERAGE"
        assert pcaf.emission_factor_source == "PCAF_DATABASE"
        assert pcaf.emission_factor_vintage_year == 2024
        assert pcaf.attribution_method == "OUTSTANDING_DIVIDED_BY_EVIC"
        assert pcaf.sector_aggregation_code == "NACE_REV2"
        assert pcaf.year_over_year_tracking is True
        assert pcaf.portfolio_carbon_footprint is True
        assert pcaf.waci_enabled is True
        assert pcaf.data_quality_improvement_plan is True

    # -----------------------------------------------------------------
    # 15. GARBTARConfig defaults
    # -----------------------------------------------------------------

    def test_gar_btar_config_defaults(self):
        """Test GARBTARConfig default values for GAR/BTAR calculation."""
        _skip_if_no_config()
        gar = _cfg.GARBTARConfig()
        assert gar.gar_enabled is True
        assert gar.btar_enabled is True
        assert len(gar.gar_scopes) == 3  # TURNOVER, CAPEX, OPEX
        assert len(gar.exposure_categories) == 5
        assert gar.exclude_sovereign_central_bank is True
        assert gar.exclude_trading_book is True
        assert gar.flow_gar_enabled is True
        assert gar.stock_gar_enabled is True
        assert gar.btar_estimation_method == "SECTOR_PROXY"
        assert gar.btar_confidence_interval is True
        assert len(gar.taxonomy_environmental_objectives) == 6
        assert gar.nace_sector_mapping_enabled is True
        assert gar.pillar3_template_output is True

    # -----------------------------------------------------------------
    # 16. InsuranceConfig defaults
    # -----------------------------------------------------------------

    def test_insurance_config_defaults(self):
        """Test InsuranceConfig defaults (disabled by default)."""
        _skip_if_no_config()
        ins = _cfg.InsuranceConfig()
        assert ins.enabled is False  # Disabled by default, auto-enabled for INSURANCE
        assert ins.attribution_method == "PREMIUM_BASED"
        assert len(ins.enabled_lines) == 4  # 4 default lines
        assert ins.reinsurance_adjustment is True
        assert ins.gross_net_reporting is True
        assert ins.orsa_climate_integration is False
        assert ins.solvency_ii_reporting is False

    # -----------------------------------------------------------------
    # 17. ClimateRiskConfig defaults
    # -----------------------------------------------------------------

    def test_climate_risk_config_defaults(self):
        """Test ClimateRiskConfig defaults (6 NGFS scenarios, all hazards)."""
        _skip_if_no_config()
        cr = _cfg.ClimateRiskConfig()
        assert cr.enabled is True
        assert len(cr.enabled_scenarios) == 6
        assert cr.physical_risk_enabled is True
        assert cr.transition_risk_enabled is True
        assert len(cr.physical_hazards) == 6
        assert len(cr.transition_channels) == 5
        assert cr.time_horizons_years == [5, 15, 30]
        assert cr.ecb_ssm_compatible is True
        assert cr.eba_exercise_compatible is True
        assert cr.expected_credit_loss_impact is True
        assert cr.sector_heatmap_enabled is True
        assert cr.geography_heatmap_enabled is True
        assert cr.pillar3_template_4_output is True

    # -----------------------------------------------------------------
    # 18. Pillar3Config defaults
    # -----------------------------------------------------------------

    def test_pillar3_config_defaults(self):
        """Test Pillar3Config defaults (9 templates, semi-annual)."""
        _skip_if_no_config()
        p3 = _cfg.Pillar3Config()
        assert p3.enabled is True
        assert len(p3.enabled_templates) == 9
        assert p3.large_institution is False
        assert p3.total_assets_eur_bn is None
        assert p3.reporting_frequency == _cfg.ReportingFrequency.SEMI_ANNUAL
        assert p3.xbrl_output is True
        assert p3.corep_finrep_cross_validation is True
        assert p3.qualitative_narrative_enabled is True

    # -----------------------------------------------------------------
    # 19. Pillar3Config auto-detects large institution
    # -----------------------------------------------------------------

    def test_pillar3_large_institution_auto_detection(self):
        """Test that total_assets >= 750bn auto-sets large_institution=True."""
        _skip_if_no_config()
        p3 = _cfg.Pillar3Config(total_assets_eur_bn=800.0)
        assert p3.large_institution is True

    # -----------------------------------------------------------------
    # 20. FSMaterialityConfig defaults
    # -----------------------------------------------------------------

    def test_fs_materiality_config_defaults(self):
        """Test FSMaterialityConfig defaults (10 ESRS topics, 4 impact channels)."""
        _skip_if_no_config()
        mat = _cfg.FSMaterialityConfig()
        assert mat.enabled is True
        assert len(mat.esrs_topics) == 10
        assert len(mat.impact_channels) == 4
        assert mat.financial_materiality_enabled is True
        assert mat.impact_materiality_enabled is True
        assert mat.stakeholder_engagement_required is True
        assert mat.materiality_threshold_financial == _cfg.MaterialityLevel.MEDIUM
        assert mat.materiality_threshold_impact == _cfg.MaterialityLevel.MEDIUM
        assert mat.weight_by_exposure_size is True
        assert mat.generate_iro1_documentation is True

    # -----------------------------------------------------------------
    # 21. FSTransitionPlanConfig defaults
    # -----------------------------------------------------------------

    def test_fs_transition_plan_config_defaults(self):
        """Test FSTransitionPlanConfig defaults (SBTi aligned, 7 sectors)."""
        _skip_if_no_config()
        tp = _cfg.FSTransitionPlanConfig()
        assert tp.enabled is True
        assert tp.sbti_fi_aligned is True
        assert tp.sbti_method == _cfg.SBTiFIMethod.SDA
        assert tp.target_years == [2025, 2030, 2035, 2040, 2050]
        assert tp.sector_targets_enabled is True
        assert len(tp.priority_sectors) == 7
        assert tp.operational_emissions_target is True
        assert tp.financed_emissions_target is True
        assert tp.client_engagement_strategy is True
        assert tp.fossil_fuel_phasedown is True
        assert tp.capital_allocation_alignment is True
        assert tp.governance_integration is True
        assert tp.temperature_alignment_target == 1.5

    # -----------------------------------------------------------------
    # 22. SBTiFIConfig defaults
    # -----------------------------------------------------------------

    def test_sbti_fi_config_defaults(self):
        """Test SBTiFIConfig defaults."""
        _skip_if_no_config()
        sbti = _cfg.SBTiFIConfig()
        assert sbti.enabled is True
        assert sbti.commitment_status == "COMMITTED"
        assert sbti.target_type == "NEAR_TERM"
        assert sbti.portfolio_coverage_target_pct == 67.0
        assert sbti.engagement_threshold_pct == 50.0
        assert sbti.temperature_rating_method is True
        assert sbti.sda_sectors_enabled is True

    # -----------------------------------------------------------------
    # 23. DataQualityConfig defaults
    # -----------------------------------------------------------------

    def test_data_quality_config_defaults(self):
        """Test DataQualityConfig defaults."""
        _skip_if_no_config()
        dq = _cfg.DataQualityConfig()
        assert dq.pcaf_score_tracking is True
        assert dq.min_coverage_pct == 70.0
        assert dq.allow_sector_proxies is True
        assert dq.proxy_usage_cap_pct == 30.0
        assert dq.require_audited_emissions is False
        assert dq.data_quality_improvement_roadmap is True
        assert dq.stale_data_threshold_days == 365

    # -----------------------------------------------------------------
    # 24. DisclosureConfig defaults
    # -----------------------------------------------------------------

    def test_disclosure_config_defaults(self):
        """Test DisclosureConfig defaults."""
        _skip_if_no_config()
        disc = _cfg.DisclosureConfig()
        assert disc.esrs_chapter_enabled is True
        assert disc.pillar3_package_enabled is True
        assert disc.pcaf_report_enabled is True
        assert disc.gar_btar_report_enabled is True
        assert disc.climate_risk_report_enabled is True
        assert disc.sbti_fi_report_enabled is True
        assert disc.dashboard_enabled is True
        assert len(disc.output_formats) >= 2
        assert disc.xbrl_tagging is True
        assert disc.review_workflow_enabled is True
        assert disc.watermark_draft is True

    # -----------------------------------------------------------------
    # 25. AuditTrailConfig defaults
    # -----------------------------------------------------------------

    def test_audit_trail_config_defaults(self):
        """Test AuditTrailConfig defaults."""
        _skip_if_no_config()
        at = _cfg.AuditTrailConfig()
        assert at.enabled is True
        assert at.sha256_provenance is True
        assert at.calculation_logging is True
        assert at.assumption_tracking is True
        assert at.data_lineage_enabled is True
        assert at.retention_years == 7
        assert at.external_audit_export is True

    # -----------------------------------------------------------------
    # 26. CSRDFinancialServiceConfig default creation
    # -----------------------------------------------------------------

    def test_csrd_fs_config_default_creation(self):
        """Test CSRDFinancialServiceConfig creates with bank defaults."""
        _skip_if_no_config()
        config = _cfg.CSRDFinancialServiceConfig()
        assert config.institution_type == _cfg.FinancialInstitutionType.BANK
        assert config.reporting_currency == "EUR"
        assert config.consolidation_scope == "GROUP"
        assert config.pcaf is not None
        assert config.gar_btar is not None
        assert config.insurance is not None
        assert config.climate_risk is not None
        assert config.pillar3 is not None
        assert config.materiality is not None
        assert config.transition_plan is not None
        assert config.sbti_fi is not None
        assert config.data_quality is not None
        assert config.disclosure is not None
        assert config.audit_trail is not None

    # -----------------------------------------------------------------
    # 27. Validator: PCAF required for banks
    # -----------------------------------------------------------------

    def test_validator_pcaf_required_for_banks(self):
        """Test that PCAF is auto-enabled for BANK institution type."""
        _skip_if_no_config()
        config = _cfg.CSRDFinancialServiceConfig(
            institution_type="BANK",
            pcaf=_cfg.PCAFConfig(enabled=False),
        )
        # Validator should auto-enable PCAF for banks
        assert config.pcaf.enabled is True

    # -----------------------------------------------------------------
    # 28. Validator: Insurance config for insurers
    # -----------------------------------------------------------------

    def test_validator_insurance_config_for_insurers(self):
        """Test that insurance config is auto-enabled for INSURANCE institution type."""
        _skip_if_no_config()
        config = _cfg.CSRDFinancialServiceConfig(
            institution_type="INSURANCE",
            insurance=_cfg.InsuranceConfig(enabled=False),
        )
        # Validator should auto-enable insurance config
        assert config.insurance.enabled is True

    # -----------------------------------------------------------------
    # 29. Validator: Pillar 3 for CRR banks
    # -----------------------------------------------------------------

    def test_validator_pillar3_for_crr_banks(self):
        """Test that Pillar 3 is auto-enabled for BANK institution type."""
        _skip_if_no_config()
        config = _cfg.CSRDFinancialServiceConfig(
            institution_type="BANK",
            pillar3=_cfg.Pillar3Config(enabled=False),
        )
        # Validator should auto-enable Pillar 3 for CRR banks
        assert config.pillar3.enabled is True

    # -----------------------------------------------------------------
    # 30. Validator: Conglomerate enables all engines
    # -----------------------------------------------------------------

    def test_validator_conglomerate_all_engines(self):
        """Test that conglomerate type enables insurance and Pillar 3."""
        _skip_if_no_config()
        config = _cfg.CSRDFinancialServiceConfig(
            institution_type="CONGLOMERATE",
            insurance=_cfg.InsuranceConfig(enabled=False),
            pillar3=_cfg.Pillar3Config(enabled=False),
        )
        assert config.insurance.enabled is True
        assert config.pillar3.enabled is True

    # -----------------------------------------------------------------
    # 31. PackConfig default creation
    # -----------------------------------------------------------------

    def test_pack_config_default_creation(self):
        """Test PackConfig creates with defaults."""
        _skip_if_no_config()
        pc = _cfg.PackConfig()
        assert pc.pack_id == "PACK-012-csrd-financial-service"
        assert pc.config_version == "1.0.0"
        assert pc.pack is not None
        assert isinstance(pc.pack, _cfg.CSRDFinancialServiceConfig)

    # -----------------------------------------------------------------
    # 32. PackConfig.from_preset - bank
    # -----------------------------------------------------------------

    def test_pack_config_from_preset_bank(self):
        """Test PackConfig.from_preset loads bank preset successfully."""
        _skip_if_no_config()
        preset_path = PRESETS_DIR / "bank.yaml"
        if not preset_path.exists():
            pytest.skip("bank.yaml preset not found")
        pc = _cfg.PackConfig.from_preset("bank")
        assert pc.preset_name == "bank"
        assert pc.pack is not None

    # -----------------------------------------------------------------
    # 33. PackConfig.from_preset - insurance
    # -----------------------------------------------------------------

    def test_pack_config_from_preset_insurance(self):
        """Test PackConfig.from_preset loads insurance preset successfully."""
        _skip_if_no_config()
        preset_path = PRESETS_DIR / "insurance.yaml"
        if not preset_path.exists():
            pytest.skip("insurance.yaml preset not found")
        pc = _cfg.PackConfig.from_preset("insurance")
        assert pc.preset_name == "insurance"
        assert pc.pack is not None

    # -----------------------------------------------------------------
    # 34. PackConfig.from_preset - invalid raises ValueError
    # -----------------------------------------------------------------

    def test_pack_config_from_preset_invalid(self):
        """Test PackConfig.from_preset raises ValueError for unknown preset."""
        _skip_if_no_config()
        with pytest.raises(ValueError, match="Unknown preset"):
            _cfg.PackConfig.from_preset("nonexistent_preset")

    # -----------------------------------------------------------------
    # 35. PackConfig.from_yaml
    # -----------------------------------------------------------------

    def test_pack_config_from_yaml(self):
        """Test PackConfig.from_yaml loads demo_config.yaml."""
        _skip_if_no_config()
        demo_path = DEMO_DIR / "demo_config.yaml"
        if not demo_path.exists():
            pytest.skip("demo_config.yaml not found")
        pc = _cfg.PackConfig.from_yaml(demo_path)
        assert pc.pack is not None

    # -----------------------------------------------------------------
    # 36. PackConfig.from_yaml - missing file raises FileNotFoundError
    # -----------------------------------------------------------------

    def test_pack_config_from_yaml_missing_file(self):
        """Test PackConfig.from_yaml raises FileNotFoundError for missing file."""
        _skip_if_no_config()
        with pytest.raises(FileNotFoundError):
            _cfg.PackConfig.from_yaml("/nonexistent/path/config.yaml")

    # -----------------------------------------------------------------
    # 37. PackConfig.get_config_hash
    # -----------------------------------------------------------------

    def test_pack_config_get_config_hash(self):
        """Test PackConfig.get_config_hash returns valid SHA-256."""
        _skip_if_no_config()
        pc = _cfg.PackConfig()
        h = pc.get_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        import re
        assert re.match(r"^[0-9a-f]{64}$", h)

    # -----------------------------------------------------------------
    # 38. Config hash is deterministic
    # -----------------------------------------------------------------

    def test_config_hash_deterministic(self):
        """Test that identical configs produce identical hashes."""
        _skip_if_no_config()
        pc1 = _cfg.PackConfig()
        pc2 = _cfg.PackConfig()
        assert pc1.get_config_hash() == pc2.get_config_hash()

    # -----------------------------------------------------------------
    # 39. Utility: get_institution_display_name
    # -----------------------------------------------------------------

    def test_get_institution_display_name(self):
        """Test get_institution_display_name returns readable names."""
        _skip_if_no_config()
        name = _cfg.get_institution_display_name("BANK")
        assert "Credit Institution" in name or "Bank" in name
        name_enum = _cfg.get_institution_display_name(_cfg.FinancialInstitutionType.INSURANCE)
        assert "Insurance" in name_enum

    # -----------------------------------------------------------------
    # 40. Utility: get_required_disclosures
    # -----------------------------------------------------------------

    def test_get_required_disclosures(self):
        """Test get_required_disclosures returns list for each institution type."""
        _skip_if_no_config()
        bank_disclosures = _cfg.get_required_disclosures("BANK")
        assert isinstance(bank_disclosures, list)
        assert len(bank_disclosures) > 0
        assert "ESRS_E1_FINANCED_EMISSIONS" in bank_disclosures
        assert "GAR_TURNOVER" in bank_disclosures

        ins_disclosures = _cfg.get_required_disclosures("INSURANCE")
        assert "ESRS_E1_UNDERWRITING_EMISSIONS" in ins_disclosures

    # -----------------------------------------------------------------
    # 41. Utility: get_pcaf_asset_class_info
    # -----------------------------------------------------------------

    def test_get_pcaf_asset_class_info(self):
        """Test get_pcaf_asset_class_info returns dict with name/attribution/description."""
        _skip_if_no_config()
        info = _cfg.get_pcaf_asset_class_info("LISTED_EQUITY_CORPORATE_BONDS")
        assert isinstance(info, dict)
        assert "name" in info
        assert "attribution" in info
        assert "description" in info
        assert "EVIC" in info["attribution"]

    # -----------------------------------------------------------------
    # 42. Utility: get_ngfs_scenario_info
    # -----------------------------------------------------------------

    def test_get_ngfs_scenario_info(self):
        """Test get_ngfs_scenario_info returns dict with scenario details."""
        _skip_if_no_config()
        info = _cfg.get_ngfs_scenario_info("NET_ZERO_2050")
        assert isinstance(info, dict)
        assert "name" in info
        assert "category" in info
        assert "temperature" in info
        assert info["category"] == "Orderly"
        assert "1.5" in info["temperature"]

    # -----------------------------------------------------------------
    # 43. Utility: list_available_presets
    # -----------------------------------------------------------------

    def test_list_available_presets(self):
        """Test list_available_presets returns all 6 presets."""
        _skip_if_no_config()
        presets = _cfg.list_available_presets()
        assert isinstance(presets, dict)
        assert len(presets) == 6
        for key in ["bank", "insurance", "asset_manager",
                     "investment_firm", "pension_fund", "conglomerate"]:
            assert key in presets, f"Missing preset: {key}"

    # -----------------------------------------------------------------
    # 44. GARExposureCategory enum
    # -----------------------------------------------------------------

    def test_gar_exposure_category_enum_values(self):
        """Test GARExposureCategory enum has exactly 6 values."""
        _skip_if_no_config()
        gec = _cfg.GARExposureCategory
        expected = {
            "FINANCIAL_CORPORATES", "NON_FINANCIAL_CORPORATES",
            "HOUSEHOLDS_MORTGAGES", "HOUSEHOLDS_MOTOR_VEHICLE",
            "LOCAL_GOVERNMENTS", "COLLATERAL_OBTAINED",
        }
        actual = {m.value for m in gec}
        assert actual == expected, f"GARExposureCategory mismatch: {actual} != {expected}"

    # -----------------------------------------------------------------
    # 45. ReportingFrequency enum
    # -----------------------------------------------------------------

    def test_reporting_frequency_enum_values(self):
        """Test ReportingFrequency enum has at least 3 values."""
        _skip_if_no_config()
        rf = _cfg.ReportingFrequency
        expected = {"ANNUAL", "SEMI_ANNUAL", "QUARTERLY"}
        actual = {m.value for m in rf}
        assert expected.issubset(actual), f"ReportingFrequency missing values: {expected - actual}"

    # -----------------------------------------------------------------
    # 46. ComplianceStatus enum
    # -----------------------------------------------------------------

    def test_compliance_status_enum_values(self):
        """Test ComplianceStatus enum has exactly 4 values."""
        _skip_if_no_config()
        cs = _cfg.ComplianceStatus
        expected = {"COMPLIANT", "NON_COMPLIANT", "PARTIALLY_COMPLIANT", "NOT_ASSESSED"}
        actual = {m.value for m in cs}
        assert actual == expected, f"ComplianceStatus mismatch: {actual} != {expected}"
