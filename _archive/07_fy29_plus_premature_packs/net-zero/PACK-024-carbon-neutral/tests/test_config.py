# -*- coding: utf-8 -*-
"""
Tests for PACK-024 configuration system.

Covers:
  - Runtime Config Models (25 tests)
  - Config Validation (15 tests)
  - Config Loading (10 tests)
  - Environment Overrides (10 tests)
  - Demo Config (10 tests)

Total: 70 tests
"""
import sys
from pathlib import Path
import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

try:
    from config.runtime_config import (
        CarbonNeutralConfig, PackConfig, validate_config, merge_config,
        get_env_overrides, list_available_presets, SUPPORTED_PRESETS,
        ICVCM_CCP_DIMENSIONS, CREDIT_QUALITY_RATINGS, CREDIT_PROJECT_TYPES,
        SUPPORTED_REGISTRIES, NeutralityType, ConsolidationApproach,
        CreditCategory, ClaimType, AssuranceLevel, BalanceMethod,
        OrganizationConfig, BoundaryConfig, FootprintConfig,
        CarbonManagementPlanConfig, CreditQualityConfig,
        PortfolioOptimizationConfig, NeutralizationBalanceConfig,
        ClaimsSubstantiationConfig, VerificationPackageConfig,
        AnnualCycleConfig, PermanenceRiskConfig,
    )
    CONFIG_AVAILABLE = True
except Exception:
    CONFIG_AVAILABLE = False


# ===========================================================================
# Runtime Config Model Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestRuntimeConfigModels:
    def test_carbon_neutral_config_instantiation(self):
        config = CarbonNeutralConfig()
        assert config is not None

    def test_default_reporting_year(self):
        config = CarbonNeutralConfig()
        assert config.reporting_year == 2025

    def test_default_base_year(self):
        config = CarbonNeutralConfig()
        assert config.base_year == 2024

    def test_organization_config(self):
        org = OrganizationConfig(name="Test Corp")
        assert org.name == "Test Corp"

    def test_organization_default_sector(self):
        org = OrganizationConfig()
        assert org.sector is not None

    def test_boundary_config(self):
        boundary = BoundaryConfig()
        assert boundary.consolidation_approach is not None

    def test_boundary_currency_validation(self):
        boundary = BoundaryConfig(reporting_currency="USD")
        assert boundary.reporting_currency == "USD"

    def test_boundary_invalid_currency(self):
        with pytest.raises(ValueError):
            BoundaryConfig(reporting_currency="INVALID")

    def test_scope3_categories_validation(self):
        boundary = BoundaryConfig(scope3_categories=[1, 2, 3])
        assert boundary.scope3_categories == [1, 2, 3]

    def test_scope3_invalid_category(self):
        with pytest.raises(ValueError):
            BoundaryConfig(scope3_categories=[0, 16])

    def test_footprint_config(self):
        fp = FootprintConfig()
        assert fp.gwp_source == "IPCC_AR6"

    def test_carbon_management_plan_config(self):
        cmp = CarbonManagementPlanConfig()
        assert cmp.reduction_first_strategy is True

    def test_credit_quality_config(self):
        cq = CreditQualityConfig()
        assert cq.min_quality_score >= 0

    def test_portfolio_optimization_config(self):
        po = PortfolioOptimizationConfig()
        assert po.max_nature_based_pct <= 100.0

    def test_neutralization_balance_config(self):
        nb = NeutralizationBalanceConfig()
        assert nb.balance_method is not None

    def test_claims_substantiation_config(self):
        cs = ClaimsSubstantiationConfig()
        assert cs.claim_type is not None

    def test_verification_package_config(self):
        vp = VerificationPackageConfig()
        assert vp.assurance_level is not None

    def test_annual_cycle_config(self):
        ac = AnnualCycleConfig()
        assert 1 <= ac.cycle_start_month <= 12

    def test_permanence_risk_config(self):
        pr = PermanenceRiskConfig()
        assert pr.buffer_contribution_pct >= 0

    def test_base_year_after_reporting_year_rejected(self):
        with pytest.raises(ValueError):
            CarbonNeutralConfig(base_year=2026, reporting_year=2025)

    def test_get_enabled_engines(self):
        config = CarbonNeutralConfig()
        engines = config.get_enabled_engines()
        assert len(engines) == 10

    def test_enabled_engines_include_footprint(self):
        config = CarbonNeutralConfig()
        engines = config.get_enabled_engines()
        assert "footprint_quantification" in engines

    def test_enabled_engines_include_credit_quality(self):
        config = CarbonNeutralConfig()
        engines = config.get_enabled_engines()
        assert "credit_quality" in engines

    def test_enabled_engines_include_permanence_risk(self):
        config = CarbonNeutralConfig()
        engines = config.get_enabled_engines()
        assert "permanence_risk" in engines

    def test_fiscal_year_end_validation(self):
        org = OrganizationConfig(fiscal_year_end="03-31")
        assert org.fiscal_year_end == "03-31"


# ===========================================================================
# Config Validation Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestConfigValidation:
    def test_empty_org_name_warning(self):
        config = CarbonNeutralConfig()
        warnings = validate_config(config)
        assert any("name" in w.lower() for w in warnings)

    def test_high_offset_reliance_warning(self):
        config = CarbonNeutralConfig()
        config.carbon_management_plan.max_offset_reliance_pct = 90.0
        warnings = validate_config(config)
        assert any("offset" in w.lower() for w in warnings)

    def test_low_credit_quality_warning(self):
        config = CarbonNeutralConfig()
        config.credit_quality.min_quality_score = 30
        warnings = validate_config(config)
        assert any("quality" in w.lower() for w in warnings)

    def test_low_removal_pct_warning(self):
        config = CarbonNeutralConfig()
        config.portfolio_optimization.min_removal_pct = 5.0
        warnings = validate_config(config)
        assert any("removal" in w.lower() for w in warnings)

    def test_old_vintage_warning(self):
        config = CarbonNeutralConfig()
        config.portfolio_optimization.vintage_max_age_years = 9
        warnings = validate_config(config)
        assert any("vintage" in w.lower() for w in warnings)

    def test_no_public_disclosure_warning(self):
        config = CarbonNeutralConfig()
        config.claims_substantiation.public_disclosure_required = False
        warnings = validate_config(config)
        assert any("disclosure" in w.lower() for w in warnings)

    def test_no_third_party_verification_warning(self):
        config = CarbonNeutralConfig()
        config.claims_substantiation.third_party_verification = False
        warnings = validate_config(config)
        assert any("verification" in w.lower() for w in warnings)

    def test_forward_credits_warning(self):
        config = CarbonNeutralConfig()
        config.neutralization_balance.allow_forward_credits = True
        warnings = validate_config(config)
        assert any("forward" in w.lower() for w in warnings)

    def test_no_standard_mapping_warning(self):
        config = CarbonNeutralConfig()
        config.reporting.include_iso14068_mapping = False
        config.reporting.include_pas2060_mapping = False
        warnings = validate_config(config)
        assert any("standard" in w.lower() or "mapping" in w.lower() for w in warnings)

    def test_valid_config_has_warnings(self):
        config = CarbonNeutralConfig(organization=OrganizationConfig(name="Valid Corp"))
        warnings = validate_config(config)
        assert isinstance(warnings, list)

    def test_all_defaults_produce_some_warnings(self):
        config = CarbonNeutralConfig()
        warnings = validate_config(config)
        assert len(warnings) >= 1  # At least empty org name

    def test_fully_configured_few_warnings(self):
        config = CarbonNeutralConfig(
            organization=OrganizationConfig(name="Test Corp"),
        )
        warnings = validate_config(config)
        assert isinstance(warnings, list)

    def test_validation_returns_list(self):
        config = CarbonNeutralConfig()
        result = validate_config(config)
        assert isinstance(result, list)

    def test_validation_items_are_strings(self):
        config = CarbonNeutralConfig()
        warnings = validate_config(config)
        for w in warnings:
            assert isinstance(w, str)

    def test_max_offset_at_boundary(self):
        config = CarbonNeutralConfig()
        config.carbon_management_plan.max_offset_reliance_pct = 80.0
        warnings = validate_config(config)
        assert isinstance(warnings, list)


# ===========================================================================
# Config Loading Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestConfigLoading:
    def test_pack_config_instantiation(self):
        config = PackConfig()
        assert config is not None

    def test_pack_config_default_id(self):
        config = PackConfig()
        assert config.pack_id == "PACK-024-carbon-neutral"

    def test_pack_config_version(self):
        config = PackConfig()
        assert config.config_version == "1.0.0"

    def test_config_hash_generation(self):
        config = PackConfig()
        h = config.get_config_hash()
        assert len(h) == 64

    def test_config_hash_deterministic(self):
        c1 = PackConfig()
        c2 = PackConfig()
        assert c1.get_config_hash() == c2.get_config_hash()

    def test_validate_config_via_pack(self):
        config = PackConfig()
        warnings = config.validate_config()
        assert isinstance(warnings, list)

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert len(presets) == 8

    def test_presets_include_corporate(self):
        presets = list_available_presets()
        assert "corporate_neutrality" in presets

    def test_presets_include_sme(self):
        presets = list_available_presets()
        assert "sme_neutrality" in presets

    def test_presets_include_event(self):
        presets = list_available_presets()
        assert "event_neutrality" in presets


# ===========================================================================
# Environment Override Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestEnvironmentOverrides:
    def test_get_env_overrides_empty(self):
        result = get_env_overrides("NONEXISTENT_PREFIX_")
        assert result == {}

    def test_merge_config_simple(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = merge_config(base, override)
        assert result["b"] == 3

    def test_merge_config_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3}}
        result = merge_config(base, override)
        assert result["a"]["y"] == 3
        assert result["a"]["x"] == 1

    def test_merge_config_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_config(base, override)
        assert result["b"] == 2

    def test_merge_config_preserves_base(self):
        base = {"a": 1}
        override = {}
        result = merge_config(base, override)
        assert result["a"] == 1

    def test_merge_config_deep_nested(self):
        base = {"a": {"b": {"c": 1}}}
        override = {"a": {"b": {"c": 2}}}
        result = merge_config(base, override)
        assert result["a"]["b"]["c"] == 2

    def test_merge_config_empty_base(self):
        result = merge_config({}, {"a": 1})
        assert result["a"] == 1

    def test_merge_config_empty_override(self):
        result = merge_config({"a": 1}, {})
        assert result["a"] == 1

    def test_merge_config_both_empty(self):
        result = merge_config({}, {})
        assert result == {}

    def test_merge_config_list_override(self):
        base = {"a": [1, 2]}
        override = {"a": [3, 4]}
        result = merge_config(base, override)
        assert result["a"] == [3, 4]


# ===========================================================================
# Demo Config Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestDemoConfig:
    def test_demo_config_path_exists(self):
        from config.demo import DEMO_CONFIG_PATH
        assert DEMO_CONFIG_PATH is not None

    def test_load_demo_config_function(self):
        from config.demo import load_demo_config
        path = load_demo_config()
        assert path is not None

    def test_demo_config_file_exists(self):
        from config.demo import DEMO_CONFIG_PATH
        assert Path(DEMO_CONFIG_PATH).exists()

    def test_demo_config_is_valid_yaml(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_demo_config_has_organization(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "organization" in data

    def test_demo_config_has_reporting_year(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "reporting_year" in data

    def test_demo_config_has_credit_quality(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "credit_quality" in data

    def test_demo_config_has_neutralization_balance(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "neutralization_balance" in data

    def test_demo_config_has_claims(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "claims_substantiation" in data

    def test_demo_config_has_verification(self):
        import yaml
        from config.demo import DEMO_CONFIG_PATH
        with open(DEMO_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "verification_package" in data


# ===========================================================================
# Constants Tests
# ===========================================================================

@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config not available")
class TestConstants:
    def test_icvcm_dimensions_count(self):
        assert len(ICVCM_CCP_DIMENSIONS) == 12

    def test_icvcm_weights_sum_to_one(self):
        total = sum(d["weight"] for d in ICVCM_CCP_DIMENSIONS.values())
        assert abs(total - 1.0) < 0.001

    def test_credit_quality_ratings_count(self):
        assert len(CREDIT_QUALITY_RATINGS) >= 7

    def test_credit_project_types_count(self):
        assert len(CREDIT_PROJECT_TYPES) >= 10

    def test_supported_registries_count(self):
        assert len(SUPPORTED_REGISTRIES) >= 4

    def test_supported_presets_count(self):
        assert len(SUPPORTED_PRESETS) == 8
