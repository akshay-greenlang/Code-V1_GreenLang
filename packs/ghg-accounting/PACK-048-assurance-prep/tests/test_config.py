"""
Unit tests for PACK-048 Configuration (pack_config.py).

Tests all 18 enums, 25 standard controls, 12 jurisdiction requirements,
10 ISAE 3410 categories, cost model parameters, materiality defaults,
8 presets, 15 sub-configs, main AssurancePackConfig, PackConfig,
and utility functions.

70+ tests covering:
  - Enum member counts and values
  - Reference data integrity (controls, jurisdictions, ISAE categories)
  - Sub-config defaults and validation
  - AssurancePackConfig creation and field validation
  - PackConfig from_preset, from_yaml, merge, get_config_hash
  - Utility functions (get_default_config, list_available_presets, etc.)
  - Environment variable overrides
  - Edge cases and validation errors

Author: GreenLang QA Team
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import (
    AVAILABLE_PRESETS,
    COST_MODEL_PARAMS,
    ISAE_3410_CATEGORIES,
    JURISDICTION_REQUIREMENTS,
    MATERIALITY_DEFAULTS,
    STANDARD_CONTROLS,
    AlertConfig,
    AssuranceLevel,
    AssurancePackConfig,
    AssuranceStandard,
    CompanySize,
    ControlCategory,
    ControlConfig,
    ControlEffectiveness,
    ControlMaturity,
    ControlType,
    CostTimelineConfig,
    EngagementConfig,
    EngagementPhase,
    EvidenceCategory,
    EvidenceConfig,
    EvidenceQuality,
    FindingSeverity,
    FindingType,
    Jurisdiction,
    MaterialityConfig,
    MaterialityType,
    PackConfig,
    PerformanceConfig,
    ProvenanceConfig,
    QueryPriority,
    QueryStatus,
    ReadinessConfig,
    RegulatoryConfig,
    ReportFormat,
    ReportingConfig,
    SamplingConfig,
    SamplingMethod,
    SecurityConfig,
    VerifierConfig,
    get_cost_estimate,
    get_default_config,
    get_isae3410_categories,
    get_jurisdiction_requirements,
    get_materiality_defaults,
    get_standard_controls,
    list_available_presets,
    validate_config,
)

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Enum Counts Tests (18 enums)
# ---------------------------------------------------------------------------


class TestEnumCounts:
    """Tests for enum member counts."""

    def test_assurance_standard_6_members(self):
        """Test AssuranceStandard has 6 members."""
        assert len(AssuranceStandard) == 6

    def test_assurance_level_6_members(self):
        """Test AssuranceLevel has 6 members."""
        assert len(AssuranceLevel) == 6

    def test_evidence_category_10_members(self):
        """Test EvidenceCategory has 10 members."""
        assert len(EvidenceCategory) == 10

    def test_evidence_quality_5_members(self):
        """Test EvidenceQuality has 5 members."""
        assert len(EvidenceQuality) == 5

    def test_control_category_5_members(self):
        """Test ControlCategory has 5 members."""
        assert len(ControlCategory) == 5

    def test_control_type_3_members(self):
        """Test ControlType has 3 members."""
        assert len(ControlType) == 3

    def test_control_effectiveness_4_members(self):
        """Test ControlEffectiveness has 4 members."""
        assert len(ControlEffectiveness) == 4

    def test_control_maturity_5_members(self):
        """Test ControlMaturity has 5 CMMI levels."""
        assert len(ControlMaturity) == 5

    def test_finding_severity_4_members(self):
        """Test FindingSeverity has 4 levels."""
        assert len(FindingSeverity) == 4

    def test_finding_type_5_members(self):
        """Test FindingType has 5 types."""
        assert len(FindingType) == 5

    def test_query_priority_4_members(self):
        """Test QueryPriority has 4 levels."""
        assert len(QueryPriority) == 4

    def test_query_status_6_members(self):
        """Test QueryStatus has 6 lifecycle statuses."""
        assert len(QueryStatus) == 6

    def test_materiality_type_5_members(self):
        """Test MaterialityType has 5 types."""
        assert len(MaterialityType) == 5

    def test_sampling_method_5_members(self):
        """Test SamplingMethod has 5 methods."""
        assert len(SamplingMethod) == 5

    def test_jurisdiction_12_members(self):
        """Test Jurisdiction has 12 jurisdictions."""
        assert len(Jurisdiction) == 12

    def test_company_size_7_members(self):
        """Test CompanySize has 7 classifications."""
        assert len(CompanySize) == 7

    def test_engagement_phase_5_members(self):
        """Test EngagementPhase has 5 phases."""
        assert len(EngagementPhase) == 5

    def test_report_format_6_members(self):
        """Test ReportFormat has 6 output formats."""
        assert len(ReportFormat) == 6


# ---------------------------------------------------------------------------
# Reference Data Tests
# ---------------------------------------------------------------------------


class TestStandardControls:
    """Tests for STANDARD_CONTROLS reference data."""

    def test_25_controls_defined(self):
        """Test 25 standard controls are defined."""
        assert len(STANDARD_CONTROLS) == 25

    def test_5_dc_controls(self):
        """Test 5 data collection controls (DC-01 to DC-05)."""
        dc_controls = [k for k in STANDARD_CONTROLS if k.startswith("DC-")]
        assert len(dc_controls) == 5

    def test_5_ca_controls(self):
        """Test 5 calculation controls (CA-01 to CA-05)."""
        ca_controls = [k for k in STANDARD_CONTROLS if k.startswith("CA-")]
        assert len(ca_controls) == 5

    def test_5_rv_controls(self):
        """Test 5 review controls (RV-01 to RV-05)."""
        rv_controls = [k for k in STANDARD_CONTROLS if k.startswith("RV-")]
        assert len(rv_controls) == 5

    def test_5_re_controls(self):
        """Test 5 reporting controls (RE-01 to RE-05)."""
        re_controls = [k for k in STANDARD_CONTROLS if k.startswith("RE-")]
        assert len(re_controls) == 5

    def test_5_it_controls(self):
        """Test 5 IT general controls (IT-01 to IT-05)."""
        it_controls = [k for k in STANDARD_CONTROLS if k.startswith("IT-")]
        assert len(it_controls) == 5

    def test_every_control_has_required_fields(self):
        """Test every control has name, category, type, and description."""
        required = {"name", "category", "type", "description"}
        for control_id, data in STANDARD_CONTROLS.items():
            for field in required:
                assert field in data, f"Control {control_id} missing field '{field}'"

    def test_control_categories_cover_5_types(self):
        """Test controls cover all 5 categories."""
        categories = set(c["category"] for c in STANDARD_CONTROLS.values())
        expected = {
            ControlCategory.DATA_COLLECTION.value,
            ControlCategory.CALCULATION.value,
            ControlCategory.REVIEW.value,
            ControlCategory.REPORTING.value,
            ControlCategory.IT_GENERAL.value,
        }
        assert categories == expected


class TestJurisdictionRequirements:
    """Tests for JURISDICTION_REQUIREMENTS reference data."""

    def test_12_jurisdictions_defined(self):
        """Test 12 jurisdictions are defined."""
        assert len(JURISDICTION_REQUIREMENTS) == 12

    def test_eu_csrd_exists(self):
        """Test EU CSRD jurisdiction exists with correct attributes."""
        eu = JURISDICTION_REQUIREMENTS["EU_CSRD"]
        assert eu["assurance_level"] == AssuranceLevel.LIMITED.value
        assert eu["standard"] == AssuranceStandard.ISAE_3410.value
        assert "SCOPE_3" in eu["scopes_required"]

    def test_us_sec_exists(self):
        """Test US SEC jurisdiction exists with SSAE 18 standard."""
        us = JURISDICTION_REQUIREMENTS["US_SEC"]
        assert us["standard"] == AssuranceStandard.SSAE_18.value
        assert "SCOPE_3" not in us["scopes_required"]

    def test_california_sb253_exists(self):
        """Test California SB 253 exists with ISO 14064-3."""
        ca = JURISDICTION_REQUIREMENTS["CALIFORNIA_SB253"]
        assert ca["standard"] == AssuranceStandard.ISO_14064_3.value
        assert "SCOPE_3" in ca["scopes_required"]

    def test_every_jurisdiction_has_required_fields(self):
        """Test every jurisdiction has all required fields."""
        required = {
            "jurisdiction_name", "assurance_level", "scopes_required",
            "effective_date", "standard", "company_threshold",
        }
        for jur_id, data in JURISDICTION_REQUIREMENTS.items():
            for field in required:
                assert field in data, f"Jurisdiction {jur_id} missing field '{field}'"


class TestISAE3410Categories:
    """Tests for ISAE_3410_CATEGORIES reference data."""

    def test_10_categories_defined(self):
        """Test 10 ISAE 3410 categories are defined."""
        assert len(ISAE_3410_CATEGORIES) == 10

    def test_category_weights_sum_to_1(self):
        """Test category weights sum to 1.0."""
        total = sum(c["weight"] for c in ISAE_3410_CATEGORIES.values())
        assert_decimal_equal(total, Decimal("1.0"), tolerance=Decimal("0.001"))

    def test_total_item_count_is_100(self):
        """Test total ISAE 3410 items across all categories is 100."""
        total = sum(c["item_count"] for c in ISAE_3410_CATEGORIES.values())
        assert total == 100

    def test_every_category_has_required_fields(self):
        """Test every category has name, weight, item_count, description."""
        required = {"category_name", "weight", "item_count", "description"}
        for cat_key, data in ISAE_3410_CATEGORIES.items():
            for field in required:
                assert field in data, f"Category '{cat_key}' missing field '{field}'"


class TestCostModelParams:
    """Tests for COST_MODEL_PARAMS reference data."""

    def test_7_size_tiers_in_base_costs(self):
        """Test 7 company size tiers are defined in base costs."""
        assert len(COST_MODEL_PARAMS["base_costs_by_size"]) == 7

    def test_9_multipliers_defined(self):
        """Test 9 cost multipliers are defined."""
        assert len(COST_MODEL_PARAMS["multipliers"]) == 9

    def test_5_hourly_rates_defined(self):
        """Test 5 hourly rate tiers are defined."""
        assert len(COST_MODEL_PARAMS["hourly_rates"]) == 5

    def test_reasonable_higher_than_limited(self):
        """Test reasonable assurance costs are higher than limited for all sizes."""
        for size, costs in COST_MODEL_PARAMS["base_costs_by_size"].items():
            assert costs["reasonable_eur"] > costs["limited_eur"], (
                f"Size {size}: reasonable ({costs['reasonable_eur']}) not > "
                f"limited ({costs['limited_eur']})"
            )

    def test_scope_3_multiplier_gt_1(self):
        """Test Scope 3 multiplier is greater than 1.0."""
        assert COST_MODEL_PARAMS["multipliers"]["scope_3_included"] > Decimal("1.0")

    def test_partner_rate_highest(self):
        """Test partner hourly rate is the highest."""
        rates = COST_MODEL_PARAMS["hourly_rates"]
        assert rates["partner_eur"] > rates["senior_manager_eur"]
        assert rates["senior_manager_eur"] > rates["manager_eur"]
        assert rates["manager_eur"] > rates["senior_associate_eur"]
        assert rates["senior_associate_eur"] > rates["associate_eur"]


class TestMaterialityDefaults:
    """Tests for MATERIALITY_DEFAULTS reference data."""

    def test_3_default_thresholds(self):
        """Test 3 materiality default thresholds defined."""
        assert len(MATERIALITY_DEFAULTS) == 3

    def test_overall_pct_is_5(self):
        """Test default overall materiality is 5%."""
        assert_decimal_equal(MATERIALITY_DEFAULTS["overall_pct"], Decimal("5.0"))

    def test_performance_pct_is_65(self):
        """Test default performance materiality is 65%."""
        assert_decimal_equal(MATERIALITY_DEFAULTS["performance_pct"], Decimal("65.0"))

    def test_trivial_pct_is_5(self):
        """Test default clearly trivial threshold is 5%."""
        assert_decimal_equal(MATERIALITY_DEFAULTS["trivial_pct"], Decimal("5.0"))


class TestAvailablePresets:
    """Tests for AVAILABLE_PRESETS reference data."""

    def test_8_presets_available(self):
        """Test 8 assurance presets available."""
        assert len(AVAILABLE_PRESETS) == 8

    def test_corporate_general_preset(self):
        """Test corporate_general preset exists."""
        assert "corporate_general" in AVAILABLE_PRESETS

    def test_csrd_limited_preset(self):
        """Test csrd_limited preset exists."""
        assert "csrd_limited" in AVAILABLE_PRESETS

    def test_csrd_reasonable_preset(self):
        """Test csrd_reasonable preset exists."""
        assert "csrd_reasonable" in AVAILABLE_PRESETS

    def test_sec_attestation_preset(self):
        """Test sec_attestation preset exists."""
        assert "sec_attestation" in AVAILABLE_PRESETS

    def test_california_sb253_preset(self):
        """Test california_sb253 preset exists."""
        assert "california_sb253" in AVAILABLE_PRESETS

    def test_multi_jurisdiction_preset(self):
        """Test multi_jurisdiction preset exists."""
        assert "multi_jurisdiction" in AVAILABLE_PRESETS

    def test_financial_services_preset(self):
        """Test financial_services preset exists."""
        assert "financial_services" in AVAILABLE_PRESETS

    def test_first_time_assurance_preset(self):
        """Test first_time_assurance preset exists."""
        assert "first_time_assurance" in AVAILABLE_PRESETS

    def test_all_presets_have_descriptions(self):
        """Test all presets have non-empty descriptions."""
        for name, desc in AVAILABLE_PRESETS.items():
            assert len(desc) > 20, f"Preset '{name}' description is too short: '{desc}'"


# ---------------------------------------------------------------------------
# Sub-Config Model Tests
# ---------------------------------------------------------------------------


class TestEvidenceConfig:
    """Tests for EvidenceConfig sub-config."""

    def test_default_10_categories(self):
        """Test default evidence config has all 10 categories."""
        config = EvidenceConfig()
        assert len(config.categories) == 10

    def test_default_quality_is_adequate(self):
        """Test default minimum quality is ADEQUATE."""
        config = EvidenceConfig()
        assert config.minimum_quality == EvidenceQuality.ADEQUATE

    def test_default_retention_7_years(self):
        """Test default evidence retention is 7 years."""
        config = EvidenceConfig()
        assert config.retention_years == 7

    def test_require_provenance_hash_default_true(self):
        """Test provenance hash is required by default."""
        config = EvidenceConfig()
        assert config.require_provenance_hash is True

    def test_invalid_format_raises(self):
        """Test invalid evidence_format raises ValueError."""
        with pytest.raises(ValueError, match="evidence_format"):
            EvidenceConfig(evidence_format="INVALID")


class TestReadinessConfig:
    """Tests for ReadinessConfig sub-config."""

    def test_default_standard_isae_3410(self):
        """Test default standard is ISAE 3410."""
        config = ReadinessConfig()
        assert config.standard == AssuranceStandard.ISAE_3410

    def test_default_target_level_limited(self):
        """Test default target level is LIMITED."""
        config = ReadinessConfig()
        assert config.target_level == AssuranceLevel.LIMITED

    def test_default_minimum_readiness_70(self):
        """Test default minimum readiness score is 70%."""
        config = ReadinessConfig()
        assert_decimal_equal(config.minimum_readiness_score_pct, Decimal("70.0"))


class TestProvenanceConfig:
    """Tests for ProvenanceConfig sub-config."""

    def test_default_hash_algorithm_sha256(self):
        """Test default hash algorithm is SHA-256."""
        config = ProvenanceConfig()
        assert config.hash_algorithm == "SHA-256"

    def test_invalid_hash_algorithm_raises(self):
        """Test non-SHA-256 algorithm raises ValueError."""
        with pytest.raises(ValueError, match="SHA-256"):
            ProvenanceConfig(hash_algorithm="MD5")

    def test_default_chain_depth_limit(self):
        """Test default chain depth limit is 100."""
        config = ProvenanceConfig()
        assert config.chain_depth_limit == 100


class TestControlConfig:
    """Tests for ControlConfig sub-config."""

    def test_default_5_categories(self):
        """Test default includes all 5 control categories."""
        config = ControlConfig()
        assert len(config.control_categories) == 5

    def test_default_target_maturity_level_3(self):
        """Test default target maturity is Level 3 Defined."""
        config = ControlConfig()
        assert config.target_maturity == ControlMaturity.LEVEL_3_DEFINED

    def test_default_sample_size_25(self):
        """Test default test sample size is 25."""
        config = ControlConfig()
        assert config.test_sample_size == 25


class TestMaterialityConfig:
    """Tests for MaterialityConfig sub-config."""

    def test_default_overall_5_pct(self):
        """Test default overall materiality is 5%."""
        config = MaterialityConfig()
        assert_decimal_equal(config.overall_materiality_pct, Decimal("5.0"))

    def test_default_performance_65_pct(self):
        """Test default performance materiality is 65%."""
        config = MaterialityConfig()
        assert_decimal_equal(config.performance_materiality_pct, Decimal("65.0"))

    def test_default_trivial_5_pct(self):
        """Test default clearly trivial threshold is 5%."""
        config = MaterialityConfig()
        assert_decimal_equal(config.clearly_trivial_pct, Decimal("5.0"))

    def test_invalid_revision_frequency_raises(self):
        """Test invalid revision_frequency raises ValueError."""
        with pytest.raises(ValueError, match="revision_frequency"):
            MaterialityConfig(revision_frequency="WEEKLY")

    def test_scope_weights_default_sum_to_1(self):
        """Test default scope weights sum to 1.0."""
        config = MaterialityConfig()
        total = sum(config.scope_weights.values())
        assert_decimal_equal(total, Decimal("1.0"))


class TestSamplingConfig:
    """Tests for SamplingConfig sub-config."""

    def test_default_method_mus(self):
        """Test default sampling method is MUS."""
        config = SamplingConfig()
        assert config.primary_method == SamplingMethod.MUS

    def test_default_confidence_95(self):
        """Test default confidence level is 95%."""
        config = SamplingConfig()
        assert_decimal_equal(config.confidence_level_pct, Decimal("95.0"))

    def test_default_minimum_sample_size_25(self):
        """Test default minimum sample size is 25."""
        config = SamplingConfig()
        assert config.minimum_sample_size == 25


class TestRegulatoryConfig:
    """Tests for RegulatoryConfig sub-config."""

    def test_default_eu_csrd_jurisdiction(self):
        """Test default jurisdiction is EU CSRD."""
        config = RegulatoryConfig()
        assert Jurisdiction.EU_CSRD in config.jurisdictions

    def test_default_company_size_large(self):
        """Test default company size is LARGE."""
        config = RegulatoryConfig()
        assert config.company_size == CompanySize.LARGE


class TestCostTimelineConfig:
    """Tests for CostTimelineConfig sub-config."""

    def test_default_assurance_level_limited(self):
        """Test default assurance level is LIMITED."""
        config = CostTimelineConfig()
        assert config.assurance_level == AssuranceLevel.LIMITED

    def test_default_currency_eur(self):
        """Test default currency is EUR."""
        config = CostTimelineConfig()
        assert config.currency == "EUR"

    def test_default_target_12_weeks(self):
        """Test default target completion is 12 weeks."""
        config = CostTimelineConfig()
        assert config.target_completion_weeks == 12


class TestReportingConfig:
    """Tests for ReportingConfig sub-config."""

    def test_default_formats_html_json(self):
        """Test default report formats include HTML and JSON."""
        config = ReportingConfig()
        assert ReportFormat.HTML in config.formats
        assert ReportFormat.JSON in config.formats

    def test_default_10_sections(self):
        """Test default report includes 10 sections."""
        config = ReportingConfig()
        assert len(config.sections) == 10

    def test_executive_summary_included(self):
        """Test default sections include executive_summary."""
        config = ReportingConfig()
        assert "executive_summary" in config.sections


class TestAlertConfig:
    """Tests for AlertConfig sub-config."""

    def test_default_6_alert_types(self):
        """Test default includes 6 alert types."""
        config = AlertConfig()
        assert len(config.alert_types) == 6

    def test_readiness_gap_alert_included(self):
        """Test READINESS_GAP alert is included by default."""
        config = AlertConfig()
        assert "READINESS_GAP" in config.alert_types

    def test_provenance_mismatch_alert_included(self):
        """Test PROVENANCE_MISMATCH alert is included by default."""
        config = AlertConfig()
        assert "PROVENANCE_MISMATCH" in config.alert_types


class TestSecurityConfig:
    """Tests for SecurityConfig sub-config."""

    def test_rbac_enabled_by_default(self):
        """Test RBAC is enabled by default."""
        config = SecurityConfig()
        assert config.rbac_enabled is True

    def test_audit_trail_enabled_by_default(self):
        """Test audit trail is enabled by default."""
        config = SecurityConfig()
        assert config.audit_trail_enabled is True

    def test_11_roles_defined(self):
        """Test 11 RBAC roles are defined for assurance management."""
        config = SecurityConfig()
        assert len(config.roles) == 11

    def test_verifier_access_controls_enabled(self):
        """Test verifier access controls are enabled by default."""
        config = SecurityConfig()
        assert config.verifier_access_controls is True


class TestPerformanceConfig:
    """Tests for PerformanceConfig sub-config."""

    def test_default_max_calc_time_300(self):
        """Test default max calculation time is 300 seconds."""
        config = PerformanceConfig()
        assert config.max_calculation_time_seconds == 300

    def test_default_batch_size_500(self):
        """Test default batch size is 500."""
        config = PerformanceConfig()
        assert config.batch_size == 500


# ---------------------------------------------------------------------------
# AssurancePackConfig Tests
# ---------------------------------------------------------------------------


class TestAssurancePackConfig:
    """Tests for AssurancePackConfig main config model."""

    def test_default_creation(self):
        """Test AssurancePackConfig can be created with defaults."""
        config = AssurancePackConfig()
        assert config is not None
        assert config.assurance_standard == AssuranceStandard.ISAE_3410
        assert config.assurance_level == AssuranceLevel.LIMITED

    def test_base_year_after_reporting_year_raises(self):
        """Test base_year after reporting_year raises ValueError."""
        with pytest.raises(ValueError, match="base_year"):
            AssurancePackConfig(base_year=2030, reporting_year=2025)

    def test_base_year_equals_reporting_year_allowed(self):
        """Test base_year equal to reporting_year is allowed."""
        config = AssurancePackConfig(base_year=2025, reporting_year=2025)
        assert config.base_year == 2025

    def test_all_14_sub_configs_populated(self):
        """Test all 14 sub-config sections are populated."""
        config = AssurancePackConfig()
        assert config.evidence is not None
        assert config.readiness is not None
        assert config.provenance is not None
        assert config.controls is not None
        assert config.verifier is not None
        assert config.materiality is not None
        assert config.sampling is not None
        assert config.regulatory is not None
        assert config.cost_timeline is not None
        assert config.reporting is not None
        assert config.alerts is not None
        assert config.performance is not None
        assert config.security is not None
        assert config.engagement is not None

    def test_default_country_de(self):
        """Test default country is DE."""
        config = AssurancePackConfig()
        assert config.country == "DE"

    def test_default_reporting_year_2026(self):
        """Test default reporting year is 2026."""
        config = AssurancePackConfig()
        assert config.reporting_year == 2026


# ---------------------------------------------------------------------------
# PackConfig Tests
# ---------------------------------------------------------------------------


class TestPackConfig:
    """Tests for PackConfig wrapper model."""

    def test_default_creation(self):
        """Test PackConfig can be created with defaults."""
        config = PackConfig()
        assert config.pack_id == "PACK-048-assurance-prep"
        assert config.config_version == "1.0.0"

    def test_get_config_hash_is_sha256(self):
        """Test get_config_hash returns a 64-char SHA-256 hex string."""
        config = PackConfig()
        h = config.get_config_hash()
        assert len(h) == 64
        int(h, 16)  # Should not raise (valid hex)

    def test_get_config_hash_deterministic(self):
        """Test identical configs produce identical hashes."""
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()

    def test_to_dict_returns_dict(self):
        """Test to_dict returns a plain dictionary."""
        config = PackConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "pack" in d

    def test_from_preset_unknown_raises(self):
        """Test from_preset with unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            PackConfig.from_preset("nonexistent_preset")

    def test_from_yaml_missing_file_raises(self):
        """Test from_yaml with missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PackConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_merge_overrides_apply(self):
        """Test merge applies overrides to base config."""
        base = PackConfig()
        merged = PackConfig.merge(base, {"company_name": "Merged Corp"})
        assert merged.pack.company_name == "Merged Corp"

    def test_merge_preserves_base_values(self):
        """Test merge preserves non-overridden base values."""
        base = PackConfig(pack=AssurancePackConfig(company_name="Original"))
        merged = PackConfig.merge(base, {"country": "US"})
        assert merged.pack.company_name == "Original"
        assert merged.pack.country == "US"

    def test_validate_completeness_returns_list(self):
        """Test validate_completeness returns list of warnings."""
        config = PackConfig()
        warnings = config.validate_completeness()
        assert isinstance(warnings, list)

    def test_validate_completeness_warns_missing_company_name(self):
        """Test validate_completeness warns about missing company name."""
        config = PackConfig()
        warnings = config.validate_completeness()
        found = any("company_name" in w for w in warnings)
        assert found is True

    def test_deep_merge_nested_dict(self):
        """Test _deep_merge correctly merges nested dicts."""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}}
        result = PackConfig._deep_merge(base, override)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["d"] == 3


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_get_default_config(self):
        """Test get_default_config returns an AssurancePackConfig."""
        config = get_default_config()
        assert isinstance(config, AssurancePackConfig)

    def test_get_default_config_with_standard(self):
        """Test get_default_config accepts an assurance standard parameter."""
        config = get_default_config(standard=AssuranceStandard.SSAE_18)
        assert config.assurance_standard == AssuranceStandard.SSAE_18

    def test_list_available_presets(self):
        """Test list_available_presets returns all 8 presets."""
        presets = list_available_presets()
        assert len(presets) == 8
        assert "corporate_general" in presets

    def test_get_standard_controls(self):
        """Test get_standard_controls returns all 25 controls."""
        controls = get_standard_controls()
        assert len(controls) == 25
        assert "DC-01" in controls
        assert "IT-05" in controls

    def test_get_jurisdiction_requirements_eu_csrd(self):
        """Test get_jurisdiction_requirements returns EU CSRD data."""
        req = get_jurisdiction_requirements("EU_CSRD")
        assert req is not None
        assert "ISAE_3410" in req["standard"]

    def test_get_jurisdiction_requirements_unknown(self):
        """Test get_jurisdiction_requirements returns None for unknown."""
        req = get_jurisdiction_requirements("NONEXISTENT")
        assert req is None

    def test_get_isae3410_categories(self):
        """Test get_isae3410_categories returns all 10 categories."""
        categories = get_isae3410_categories()
        assert len(categories) == 10

    def test_get_cost_estimate_large_limited(self):
        """Test get_cost_estimate returns correct value for LARGE limited."""
        cost = get_cost_estimate("LARGE", "limited")
        assert cost is not None
        assert cost == Decimal("60000")

    def test_get_cost_estimate_large_reasonable(self):
        """Test get_cost_estimate returns correct value for LARGE reasonable."""
        cost = get_cost_estimate("LARGE", "reasonable")
        assert cost is not None
        assert cost == Decimal("150000")

    def test_get_cost_estimate_unknown_size(self):
        """Test get_cost_estimate returns None for unknown size."""
        cost = get_cost_estimate("MEGA_ULTRA", "limited")
        assert cost is None

    def test_get_materiality_defaults(self):
        """Test get_materiality_defaults returns 3 threshold defaults."""
        defaults = get_materiality_defaults()
        assert len(defaults) == 3
        assert "overall_pct" in defaults


# ---------------------------------------------------------------------------
# validate_config Tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for validate_config domain-specific validation."""

    def test_valid_default_config_returns_warnings(self):
        """Test default config produces expected warnings (e.g., no company_name)."""
        config = AssurancePackConfig()
        warnings = validate_config(config)
        assert isinstance(warnings, list)
        assert any("company_name" in w for w in warnings)

    def test_complete_config_fewer_warnings(self):
        """Test fully populated config produces fewer warnings."""
        config = AssurancePackConfig(
            company_name="Test Corp",
            total_emissions_tco2e=Decimal("23000"),
            engagement=EngagementConfig(verifier_firm="Big Four LLP"),
        )
        warnings = validate_config(config)
        assert not any("company_name" in w for w in warnings)
        assert not any("total_emissions_tco2e" in w for w in warnings)

    def test_scope_3_with_limited_warning(self):
        """Test warns when Scope 3 is in scope with limited assurance."""
        config = AssurancePackConfig(
            scopes_in_scope=["SCOPE_1", "SCOPE_2", "SCOPE_3"],
            assurance_level=AssuranceLevel.LIMITED,
        )
        warnings = validate_config(config)
        scope_3_warnings = [w for w in warnings if "Scope 3" in w]
        assert len(scope_3_warnings) >= 1

    def test_multi_jurisdiction_cost_inconsistency_warning(self):
        """Test warns when multiple jurisdictions but cost flag is off."""
        config = AssurancePackConfig(
            regulatory=RegulatoryConfig(
                jurisdictions=[Jurisdiction.EU_CSRD, Jurisdiction.US_SEC],
            ),
            cost_timeline=CostTimelineConfig(multi_jurisdiction=False),
        )
        warnings = validate_config(config)
        multi_jur_warnings = [w for w in warnings if "jurisdictions" in w.lower() or "multi" in w.lower()]
        assert len(multi_jur_warnings) >= 1

    def test_no_verifier_firm_warning(self):
        """Test warns when no verifier_firm is configured."""
        config = AssurancePackConfig()
        warnings = validate_config(config)
        verifier_warnings = [w for w in warnings if "verifier_firm" in w]
        assert len(verifier_warnings) >= 1

    def test_missing_total_emissions_warning(self):
        """Test warns when total_emissions_tco2e is not set."""
        config = AssurancePackConfig(total_emissions_tco2e=None)
        warnings = validate_config(config)
        emissions_warnings = [w for w in warnings if "total_emissions_tco2e" in w]
        assert len(emissions_warnings) >= 1


# ---------------------------------------------------------------------------
# Environment Override Tests
# ---------------------------------------------------------------------------


class TestEnvironmentOverrides:
    """Tests for environment variable override loading."""

    def test_load_env_overrides_parses_prefix(self):
        """Test _load_env_overrides parses ASSURANCE_PACK_ prefixed vars."""
        os.environ["ASSURANCE_PACK_COUNTRY"] = "US"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("country") == "US"
        finally:
            del os.environ["ASSURANCE_PACK_COUNTRY"]

    def test_load_env_overrides_nested_key(self):
        """Test _load_env_overrides parses double-underscore nesting."""
        os.environ["ASSURANCE_PACK_CONTROLS__TEST_SAMPLE_SIZE"] = "50"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("controls", {}).get("test_sample_size") == 50
        finally:
            del os.environ["ASSURANCE_PACK_CONTROLS__TEST_SAMPLE_SIZE"]

    def test_load_env_overrides_bool_true(self):
        """Test _load_env_overrides converts 'true' to True."""
        os.environ["ASSURANCE_PACK_SOME_FLAG"] = "true"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("some_flag") is True
        finally:
            del os.environ["ASSURANCE_PACK_SOME_FLAG"]

    def test_load_env_overrides_bool_false(self):
        """Test _load_env_overrides converts 'false' to False."""
        os.environ["ASSURANCE_PACK_SOME_FLAG"] = "false"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("some_flag") is False
        finally:
            del os.environ["ASSURANCE_PACK_SOME_FLAG"]
