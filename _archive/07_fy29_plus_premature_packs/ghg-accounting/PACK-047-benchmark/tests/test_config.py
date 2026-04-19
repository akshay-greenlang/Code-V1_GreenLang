"""
Unit tests for PACK-047 Configuration (pack_config.py).

Tests all 18 enums, 9 SBTi sector pathways, 3 IPCC carbon budgets,
5 PCAF quality levels, 6 peer size bands, 5 transition risk weights,
8 presets, 15 sub-configs, main BenchmarkPackConfig, PackConfig,
and utility functions.

70+ tests covering:
  - Enum member counts and values
  - Reference data integrity (pathways, budgets, quality thresholds)
  - Sub-config defaults and validation
  - BenchmarkPackConfig creation and field validation
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
    AlertConfig,
    AlertType,
    BenchmarkMetric,
    BenchmarkPackConfig,
    ConsolidationApproach,
    DataQualityConfig,
    DataSourceType,
    DisclosureConfig,
    DisclosureFramework,
    ExternalDataConfig,
    GWP_CONVERSION_FACTORS,
    GWPVersion,
    IPCC_CARBON_BUDGETS,
    ITRConfig,
    ITRMethod,
    NormalisationConfig,
    NormalisationStep,
    PackConfig,
    PathwayConfig,
    PathwayScenario,
    PathwayType,
    PCAFScore,
    PCAF_QUALITY_THRESHOLDS,
    PEER_SIZE_BANDS,
    PeerGroupConfig,
    PeerSizeBand,
    PerformanceConfig,
    PortfolioAssetClass,
    PortfolioConfig,
    QualityDimension,
    ReportFormat,
    ReportingConfig,
    SBTI_SECTOR_PATHWAYS,
    ScopeAlignment,
    SectorClassification,
    SecurityConfig,
    TRANSITION_RISK_WEIGHTS,
    TrajectoryConfig,
    TransitionRiskCategory,
    TransitionRiskConfig,
    get_carbon_budget,
    get_default_config,
    get_pcaf_quality_info,
    get_peer_size_band,
    get_sbti_pathway,
    list_available_presets,
    validate_config,
)

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Enum Counts Tests (18 enums)
# ---------------------------------------------------------------------------


class TestEnumCounts:
    """Tests for enum member counts."""

    def test_sector_classification_14_members(self):
        """Test SectorClassification has 14 members (GICS/NACE/ISIC/SIC/CUSTOM)."""
        assert len(SectorClassification) == 14

    def test_peer_size_band_6_members(self):
        """Test PeerSizeBand has 6 members."""
        assert len(PeerSizeBand) == 6

    def test_scope_alignment_5_members(self):
        """Test ScopeAlignment has 5 members."""
        assert len(ScopeAlignment) == 5

    def test_consolidation_approach_3_members(self):
        """Test ConsolidationApproach has 3 members."""
        assert len(ConsolidationApproach) == 3

    def test_gwp_version_3_members(self):
        """Test GWPVersion has 3 members (AR4, AR5, AR6)."""
        assert len(GWPVersion) == 3

    def test_pathway_type_8_members(self):
        """Test PathwayType has 8 members."""
        assert len(PathwayType) == 8

    def test_pathway_scenario_3_members(self):
        """Test PathwayScenario has 3 temperature scenarios."""
        assert len(PathwayScenario) == 3

    def test_itr_method_3_members(self):
        """Test ITRMethod has 3 calculation methods."""
        assert len(ITRMethod) == 3

    def test_portfolio_asset_class_6_members(self):
        """Test PortfolioAssetClass has 6 PCAF-aligned classes."""
        assert len(PortfolioAssetClass) == 6

    def test_pcaf_score_5_members(self):
        """Test PCAFScore has 5 quality levels (1-5)."""
        assert len(PCAFScore) == 5

    def test_data_source_type_6_members(self):
        """Test DataSourceType has 6 sources."""
        assert len(DataSourceType) == 6

    def test_transition_risk_category_5_members(self):
        """Test TransitionRiskCategory has 5 categories."""
        assert len(TransitionRiskCategory) == 5

    def test_benchmark_metric_6_members(self):
        """Test BenchmarkMetric has 6 metric types."""
        assert len(BenchmarkMetric) == 6

    def test_report_format_6_members(self):
        """Test ReportFormat has 6 output formats."""
        assert len(ReportFormat) == 6

    def test_disclosure_framework_6_members(self):
        """Test DisclosureFramework has 6 frameworks."""
        assert len(DisclosureFramework) == 6

    def test_alert_type_6_members(self):
        """Test AlertType has 6 trigger types."""
        assert len(AlertType) == 6

    def test_quality_dimension_5_members(self):
        """Test QualityDimension has 5 PCAF dimensions."""
        assert len(QualityDimension) == 5

    def test_normalisation_step_8_members(self):
        """Test NormalisationStep has 8 pipeline steps."""
        assert len(NormalisationStep) == 8


# ---------------------------------------------------------------------------
# Reference Data Tests
# ---------------------------------------------------------------------------


class TestIPCCCarbonBudgets:
    """Tests for IPCC_CARBON_BUDGETS reference data."""

    def test_3_temperature_scenarios(self):
        """Test 3 temperature scenarios defined (1.5C, 1.7C, 2.0C)."""
        assert len(IPCC_CARBON_BUDGETS) == 3

    def test_1_5c_budget_400_gt(self):
        """Test 1.5C scenario has 400 GtCO2 remaining budget."""
        b = IPCC_CARBON_BUDGETS["1.5C"]
        assert_decimal_equal(b["remaining_budget_gt_co2"], Decimal("400"))

    def test_2_0c_budget_1150_gt(self):
        """Test 2.0C scenario has 1150 GtCO2 remaining budget."""
        b = IPCC_CARBON_BUDGETS["2.0C"]
        assert_decimal_equal(b["remaining_budget_gt_co2"], Decimal("1150"))

    def test_budgets_increase_with_temperature(self):
        """Test higher temperature scenarios have larger carbon budgets."""
        b_15 = IPCC_CARBON_BUDGETS["1.5C"]["remaining_budget_gt_co2"]
        b_17 = IPCC_CARBON_BUDGETS["1.7C"]["remaining_budget_gt_co2"]
        b_20 = IPCC_CARBON_BUDGETS["2.0C"]["remaining_budget_gt_co2"]
        assert b_15 < b_17 < b_20

    def test_every_budget_has_required_fields(self):
        """Test every carbon budget entry has required fields."""
        required = {"temperature", "remaining_budget_gt_co2", "from_year", "probability_pct", "source"}
        for scenario, data in IPCC_CARBON_BUDGETS.items():
            for field in required:
                assert field in data, f"Scenario {scenario} missing field '{field}'"


class TestSBTiSectorPathways:
    """Tests for SBTI_SECTOR_PATHWAYS reference data."""

    def test_9_sector_pathways(self):
        """Test 9 SBTi sector pathways defined."""
        assert len(SBTI_SECTOR_PATHWAYS) == 9

    def test_power_pathway_exists(self):
        """Test power generation pathway exists."""
        assert "power" in SBTI_SECTOR_PATHWAYS
        pwr = SBTI_SECTOR_PATHWAYS["power"]
        assert pwr["base_year"] == 2020
        assert "2030" in pwr["pathway_1_5c"]

    def test_steel_pathway_exists(self):
        """Test steel (iron and steel) pathway exists."""
        assert "steel" in SBTI_SECTOR_PATHWAYS

    def test_cement_pathway_exists(self):
        """Test cement pathway exists."""
        assert "cement" in SBTI_SECTOR_PATHWAYS

    def test_food_flag_pathway_exists(self):
        """Test food and agriculture (FLAG) pathway exists."""
        assert "food" in SBTI_SECTOR_PATHWAYS

    def test_1_5c_pathway_decreases_over_time(self):
        """Test 1.5C pathway intensity decreases over time for all sectors."""
        for sector, pathway in SBTI_SECTOR_PATHWAYS.items():
            p15c = pathway["pathway_1_5c"]
            years = sorted(p15c.keys())
            for i in range(1, len(years)):
                assert p15c[years[i]] <= p15c[years[i - 1]], (
                    f"Sector {sector}: pathway_1_5c not decreasing between "
                    f"{years[i-1]}={p15c[years[i-1]]} and {years[i]}={p15c[years[i]]}"
                )

    def test_target_intensity_lower_than_base(self):
        """Test target intensity is lower than base intensity for all sectors."""
        for sector, pathway in SBTI_SECTOR_PATHWAYS.items():
            assert pathway["target_intensity"] < pathway["base_intensity"], (
                f"Sector {sector}: target {pathway['target_intensity']} not less "
                f"than base {pathway['base_intensity']}"
            )


class TestPCAFQualityThresholds:
    """Tests for PCAF_QUALITY_THRESHOLDS reference data."""

    def test_5_quality_levels(self):
        """Test 5 PCAF quality levels defined."""
        assert len(PCAF_QUALITY_THRESHOLDS) == 5

    def test_level_1_is_audited(self):
        """Test level 1 is audited/verified quality."""
        assert "Audited" in PCAF_QUALITY_THRESHOLDS[1]["label"]

    def test_level_5_is_estimated(self):
        """Test level 5 is estimated/asset class quality."""
        assert "Estimated" in PCAF_QUALITY_THRESHOLDS[5]["label"]

    def test_uncertainty_increases_with_level(self):
        """Test uncertainty percentage increases from level 1 to 5."""
        for level in range(1, 5):
            curr = PCAF_QUALITY_THRESHOLDS[level]["typical_uncertainty_pct"]
            next_val = PCAF_QUALITY_THRESHOLDS[level + 1]["typical_uncertainty_pct"]
            assert next_val > curr, (
                f"Level {level} uncertainty ({curr}) should be < level {level+1} ({next_val})"
            )


class TestPeerSizeBands:
    """Tests for PEER_SIZE_BANDS reference data."""

    def test_6_size_bands(self):
        """Test 6 peer size bands defined."""
        assert len(PEER_SIZE_BANDS) == 6

    def test_bands_are_contiguous(self):
        """Test revenue bands are contiguous (no gaps)."""
        band_order = ["MICRO", "SMALL", "MEDIUM", "LARGE", "ENTERPRISE", "MEGA"]
        for i in range(len(band_order) - 1):
            current_max = PEER_SIZE_BANDS[band_order[i]]["revenue_max_meur"]
            next_min = PEER_SIZE_BANDS[band_order[i + 1]]["revenue_min_meur"]
            assert current_max == next_min, (
                f"Gap between {band_order[i]} max ({current_max}) and "
                f"{band_order[i+1]} min ({next_min})"
            )


class TestTransitionRiskWeights:
    """Tests for TRANSITION_RISK_WEIGHTS reference data."""

    def test_5_risk_categories(self):
        """Test 5 transition risk weight categories."""
        assert len(TRANSITION_RISK_WEIGHTS) == 5

    def test_weights_sum_to_1(self):
        """Test transition risk weights sum to 1.0."""
        total = sum(TRANSITION_RISK_WEIGHTS.values())
        assert_decimal_equal(total, Decimal("1.0"), tolerance=Decimal("0.001"))


class TestGWPConversionFactors:
    """Tests for GWP_CONVERSION_FACTORS reference data."""

    def test_2_conversion_sets(self):
        """Test 2 GWP conversion sets (AR4_to_AR6, AR5_to_AR6)."""
        assert "AR4_to_AR6" in GWP_CONVERSION_FACTORS
        assert "AR5_to_AR6" in GWP_CONVERSION_FACTORS

    def test_co2_factor_is_1_for_all(self):
        """Test CO2 conversion factor is always 1.0 (no change across ARs)."""
        for conversion_set, gases in GWP_CONVERSION_FACTORS.items():
            assert_decimal_equal(
                gases["CO2"]["factor"], Decimal("1.0"),
                msg=f"CO2 factor in {conversion_set} is not 1.0",
            )


class TestAvailablePresets:
    """Tests for AVAILABLE_PRESETS reference data."""

    def test_8_presets_available(self):
        """Test 8 benchmark presets available."""
        assert len(AVAILABLE_PRESETS) == 8

    def test_corporate_general_preset(self):
        """Test corporate_general preset exists."""
        assert "corporate_general" in AVAILABLE_PRESETS

    def test_power_utilities_preset(self):
        """Test power_utilities preset exists."""
        assert "power_utilities" in AVAILABLE_PRESETS

    def test_heavy_industry_preset(self):
        """Test heavy_industry preset exists."""
        assert "heavy_industry" in AVAILABLE_PRESETS

    def test_real_estate_preset(self):
        """Test real_estate preset exists."""
        assert "real_estate" in AVAILABLE_PRESETS

    def test_financial_services_preset(self):
        """Test financial_services preset exists."""
        assert "financial_services" in AVAILABLE_PRESETS

    def test_all_presets_have_descriptions(self):
        """Test all presets have non-empty descriptions."""
        for name, desc in AVAILABLE_PRESETS.items():
            assert len(desc) > 20, f"Preset '{name}' description is too short: '{desc}'"


# ---------------------------------------------------------------------------
# Sub-Config Model Tests
# ---------------------------------------------------------------------------


class TestPeerGroupConfig:
    """Tests for PeerGroupConfig sub-config."""

    def test_default_config(self):
        """Test default PeerGroupConfig values."""
        config = PeerGroupConfig()
        assert config.sector_classification == SectorClassification.GICS_4DIG
        assert config.size_band == PeerSizeBand.LARGE
        assert config.min_peers == 5
        assert config.max_peers == 50

    def test_region_scope_validation(self):
        """Test region_scope validator accepts valid values."""
        config = PeerGroupConfig(region_scope="EU")
        assert config.region_scope == "EU"

    def test_invalid_region_scope_raises(self):
        """Test invalid region_scope raises ValueError."""
        with pytest.raises(ValueError, match="region_scope"):
            PeerGroupConfig(region_scope="INVALID_REGION")


class TestNormalisationConfig:
    """Tests for NormalisationConfig sub-config."""

    def test_default_scope_alignment(self):
        """Test default scope alignment is S1_S2M (market-based)."""
        config = NormalisationConfig()
        assert config.scope_alignment == ScopeAlignment.S1_S2M

    def test_default_gwp_is_ar6(self):
        """Test default GWP version is AR6."""
        config = NormalisationConfig()
        assert config.target_gwp == GWPVersion.AR6

    def test_period_alignment_validation(self):
        """Test period_alignment validator rejects invalid values."""
        with pytest.raises(ValueError, match="period_alignment"):
            NormalisationConfig(period_alignment="WEEKLY")

    def test_biogenic_treatment_validation(self):
        """Test biogenic_treatment validator rejects invalid values."""
        with pytest.raises(ValueError, match="biogenic_treatment"):
            NormalisationConfig(biogenic_treatment="UNKNOWN")

    def test_data_gap_strategy_validation(self):
        """Test data_gap_strategy validator rejects invalid values."""
        with pytest.raises(ValueError, match="data_gap_strategy"):
            NormalisationConfig(data_gap_strategy="DROP_ALL")

    def test_default_normalisation_steps(self):
        """Test default normalisation steps include expected pipeline."""
        config = NormalisationConfig()
        assert NormalisationStep.SCOPE_ALIGN in config.normalisation_steps
        assert NormalisationStep.GWP in config.normalisation_steps
        assert NormalisationStep.CURRENCY in config.normalisation_steps


class TestPathwayConfig:
    """Tests for PathwayConfig sub-config."""

    def test_default_pathways(self):
        """Test default pathways include IEA NZE and SBTi SDA."""
        config = PathwayConfig()
        assert PathwayType.IEA_NZE in config.pathways
        assert PathwayType.SBTI_SDA in config.pathways

    def test_default_scenario_1_5c(self):
        """Test default scenario is 1.5C."""
        config = PathwayConfig()
        assert config.primary_scenario == PathwayScenario.ONE_POINT_FIVE_C


class TestITRConfig:
    """Tests for ITRConfig sub-config."""

    def test_default_methods(self):
        """Test default ITR methods include budget-based and sector-relative."""
        config = ITRConfig()
        assert ITRMethod.BUDGET_BASED in config.methods
        assert ITRMethod.SECTOR_RELATIVE in config.methods

    def test_default_primary_method(self):
        """Test default primary method is budget-based."""
        config = ITRConfig()
        assert config.primary_method == ITRMethod.BUDGET_BASED


class TestPortfolioConfig:
    """Tests for PortfolioConfig sub-config."""

    def test_default_asset_classes(self):
        """Test default portfolio includes listed equity and corporate bonds."""
        config = PortfolioConfig()
        assert PortfolioAssetClass.LISTED_EQUITY in config.asset_classes
        assert PortfolioAssetClass.CORPORATE_BONDS in config.asset_classes

    def test_attribution_method_validation(self):
        """Test attribution_method validator rejects invalid values."""
        with pytest.raises(ValueError, match="attribution_method"):
            PortfolioConfig(attribution_method="INVALID_METHOD")


class TestDataQualityConfig:
    """Tests for DataQualityConfig sub-config."""

    def test_default_dimensions(self):
        """Test default quality includes all 5 PCAF dimensions."""
        config = DataQualityConfig()
        assert len(config.dimensions) == 5

    def test_dimension_weights_sum_to_1(self):
        """Test default dimension weights sum to 1.0."""
        config = DataQualityConfig()
        total = sum(config.dimension_weights.values())
        assert_decimal_equal(total, Decimal("1.0"), tolerance=Decimal("0.01"))

    def test_invalid_weights_sum_raises(self):
        """Test dimension weights not summing to 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataQualityConfig(
                dimension_weights={
                    "TEMPORAL": Decimal("0.50"),
                    "GEOGRAPHIC": Decimal("0.50"),
                    "TECHNOLOGICAL": Decimal("0.50"),
                    "COMPLETENESS": Decimal("0.50"),
                    "RELIABILITY": Decimal("0.50"),
                }
            )


class TestTransitionRiskConfig:
    """Tests for TransitionRiskConfig sub-config."""

    def test_default_categories(self):
        """Test default includes all 5 risk categories."""
        config = TransitionRiskConfig()
        assert len(config.categories) == 5

    def test_category_weights_sum_to_1(self):
        """Test default category weights sum to 1.0."""
        config = TransitionRiskConfig()
        total = sum(config.category_weights.values())
        assert_decimal_equal(total, Decimal("1.0"), tolerance=Decimal("0.01"))

    def test_carbon_price_scenario_validation(self):
        """Test carbon_price_scenario validator rejects invalid values."""
        with pytest.raises(ValueError, match="carbon_price_scenario"):
            TransitionRiskConfig(carbon_price_scenario="INVALID")


class TestTrajectoryConfig:
    """Tests for TrajectoryConfig sub-config."""

    def test_default_historical_years(self):
        """Test default historical window is 5 years."""
        config = TrajectoryConfig()
        assert config.historical_years == 5

    def test_regression_model_validation(self):
        """Test regression_model validator rejects invalid values."""
        with pytest.raises(ValueError, match="regression_model"):
            TrajectoryConfig(regression_model="NEURAL_NET")


class TestReportingConfig:
    """Tests for ReportingConfig sub-config."""

    def test_default_formats(self):
        """Test default report formats include HTML and JSON."""
        config = ReportingConfig()
        assert ReportFormat.HTML in config.formats
        assert ReportFormat.JSON in config.formats

    def test_default_sections_include_executive_summary(self):
        """Test default sections include executive summary."""
        config = ReportingConfig()
        assert "executive_summary" in config.sections


class TestDisclosureConfig:
    """Tests for DisclosureConfig sub-config."""

    def test_default_frameworks(self):
        """Test default disclosure frameworks include ESRS, CDP, TCFD."""
        config = DisclosureConfig()
        assert DisclosureFramework.ESRS in config.frameworks
        assert DisclosureFramework.CDP in config.frameworks
        assert DisclosureFramework.TCFD in config.frameworks


class TestAlertConfig:
    """Tests for AlertConfig sub-config."""

    def test_default_alert_types(self):
        """Test default alert types include threshold and pathway deviation."""
        config = AlertConfig()
        assert AlertType.THRESHOLD in config.alert_types
        assert AlertType.PATHWAY_DEVIATION in config.alert_types


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

    def test_7_roles_defined(self):
        """Test 7 RBAC roles are defined."""
        config = SecurityConfig()
        assert len(config.roles) == 7


# ---------------------------------------------------------------------------
# BenchmarkPackConfig Tests
# ---------------------------------------------------------------------------


class TestBenchmarkPackConfig:
    """Tests for BenchmarkPackConfig main config model."""

    def test_default_creation(self):
        """Test BenchmarkPackConfig can be created with defaults."""
        config = BenchmarkPackConfig()
        assert config is not None
        assert config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL

    def test_base_year_before_reporting_year(self):
        """Test base_year must be before reporting_year."""
        with pytest.raises(ValueError, match="base_year"):
            BenchmarkPackConfig(base_year=2030, reporting_year=2025)

    def test_base_year_equals_reporting_year_allowed(self):
        """Test base_year equal to reporting_year is allowed."""
        config = BenchmarkPackConfig(base_year=2025, reporting_year=2025)
        assert config.base_year == 2025

    def test_all_sub_configs_populated(self):
        """Test all 14 sub-config sections are populated."""
        config = BenchmarkPackConfig()
        assert config.peer_group is not None
        assert config.normalisation is not None
        assert config.external_data is not None
        assert config.pathway is not None
        assert config.itr is not None
        assert config.trajectory is not None
        assert config.portfolio is not None
        assert config.data_quality is not None
        assert config.transition_risk is not None
        assert config.reporting is not None
        assert config.disclosure is not None
        assert config.alerts is not None
        assert config.performance is not None
        assert config.security is not None


# ---------------------------------------------------------------------------
# PackConfig Tests
# ---------------------------------------------------------------------------


class TestPackConfig:
    """Tests for PackConfig wrapper model."""

    def test_default_creation(self):
        """Test PackConfig can be created with defaults."""
        config = PackConfig()
        assert config.pack_id == "PACK-047-benchmark"
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
        base = PackConfig(pack=BenchmarkPackConfig(company_name="Original"))
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
        """Test get_default_config returns a BenchmarkPackConfig."""
        config = get_default_config()
        assert isinstance(config, BenchmarkPackConfig)

    def test_list_available_presets(self):
        """Test list_available_presets returns all 8 presets."""
        presets = list_available_presets()
        assert len(presets) == 8
        assert "corporate_general" in presets

    def test_get_sbti_pathway_power(self):
        """Test get_sbti_pathway returns power sector data."""
        pathway = get_sbti_pathway("power")
        assert pathway is not None
        assert "pathway_1_5c" in pathway

    def test_get_sbti_pathway_unknown(self):
        """Test get_sbti_pathway returns None for unknown sector."""
        pathway = get_sbti_pathway("nonexistent_sector")
        assert pathway is None

    def test_get_carbon_budget_1_5c(self):
        """Test get_carbon_budget returns 1.5C scenario data."""
        budget = get_carbon_budget("1.5C")
        assert budget is not None
        assert_decimal_equal(budget["remaining_budget_gt_co2"], Decimal("400"))

    def test_get_carbon_budget_unknown(self):
        """Test get_carbon_budget returns None for unknown scenario."""
        budget = get_carbon_budget("10.0C")
        assert budget is None

    def test_get_pcaf_quality_info_score_1(self):
        """Test get_pcaf_quality_info returns level 1 data."""
        info = get_pcaf_quality_info(1)
        assert info is not None
        assert "Audited" in info["label"]

    def test_get_pcaf_quality_info_invalid_score(self):
        """Test get_pcaf_quality_info returns None for invalid score."""
        info = get_pcaf_quality_info(0)
        assert info is None
        info6 = get_pcaf_quality_info(6)
        assert info6 is None

    def test_get_peer_size_band_micro(self):
        """Test get_peer_size_band classifies micro revenue."""
        band = get_peer_size_band(Decimal("1"))
        assert band == "MICRO"

    def test_get_peer_size_band_enterprise(self):
        """Test get_peer_size_band classifies enterprise revenue."""
        band = get_peer_size_band(Decimal("10000"))
        assert band == "ENTERPRISE"

    def test_get_peer_size_band_mega(self):
        """Test get_peer_size_band classifies mega revenue."""
        band = get_peer_size_band(Decimal("100000"))
        assert band == "MEGA"


# ---------------------------------------------------------------------------
# validate_config Tests
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for validate_config domain-specific validation."""

    def test_valid_default_config_returns_warnings(self):
        """Test default config produces expected warnings (e.g., no company_name)."""
        config = BenchmarkPackConfig()
        warnings = validate_config(config)
        assert isinstance(warnings, list)
        # Default has no company name, should warn
        assert any("company_name" in w for w in warnings)

    def test_complete_config_fewer_warnings(self):
        """Test fully populated config produces fewer warnings."""
        config = BenchmarkPackConfig(
            company_name="Test Corp",
            sector_code="2010",
            base_year=2020,
            reporting_year=2025,
        )
        warnings = validate_config(config)
        assert not any("company_name" in w for w in warnings)
        assert not any("sector_code" in w for w in warnings)

    def test_scope_3_inconsistency_warning(self):
        """Test warns when normalisation includes Scope 3 but ITR excludes it."""
        config = BenchmarkPackConfig(
            normalisation=NormalisationConfig(scope_alignment=ScopeAlignment.S1_S2_S3),
            itr=ITRConfig(include_scope_3=False),
        )
        warnings = validate_config(config)
        scope_3_warnings = [w for w in warnings if "Scope 3" in w]
        assert len(scope_3_warnings) >= 1

    def test_gresb_non_real_estate_warning(self):
        """Test warns when GRESB is enabled for non-real-estate sector."""
        config = BenchmarkPackConfig(
            sector_code="2010",  # Non real-estate (not 60xx)
            external_data=ExternalDataConfig(
                sources=[DataSourceType.CDP, DataSourceType.GRESB],
            ),
        )
        warnings = validate_config(config)
        gresb_warnings = [w for w in warnings if "GRESB" in w]
        assert len(gresb_warnings) >= 1

    def test_sovereign_asset_class_consistency_warning(self):
        """Test warns when include_sovereign is True but SOVEREIGN_DEBT not in asset_classes."""
        config = BenchmarkPackConfig(
            portfolio=PortfolioConfig(
                include_sovereign=True,
                asset_classes=[PortfolioAssetClass.LISTED_EQUITY],
            ),
        )
        warnings = validate_config(config)
        sovereign_warnings = [w for w in warnings if "sovereign" in w.lower() or "SOVEREIGN" in w]
        assert len(sovereign_warnings) >= 1


# ---------------------------------------------------------------------------
# Environment Override Tests
# ---------------------------------------------------------------------------


class TestEnvironmentOverrides:
    """Tests for environment variable override loading."""

    def test_load_env_overrides_parses_prefix(self):
        """Test _load_env_overrides parses BENCHMARK_PACK_ prefixed vars."""
        os.environ["BENCHMARK_PACK_COUNTRY"] = "US"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("country") == "US"
        finally:
            del os.environ["BENCHMARK_PACK_COUNTRY"]

    def test_load_env_overrides_nested_key(self):
        """Test _load_env_overrides parses double-underscore nesting."""
        os.environ["BENCHMARK_PACK_PEER_GROUP__MIN_PEERS"] = "10"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("peer_group", {}).get("min_peers") == 10
        finally:
            del os.environ["BENCHMARK_PACK_PEER_GROUP__MIN_PEERS"]

    def test_load_env_overrides_bool_true(self):
        """Test _load_env_overrides converts 'true' to True."""
        os.environ["BENCHMARK_PACK_SOME_FLAG"] = "true"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("some_flag") is True
        finally:
            del os.environ["BENCHMARK_PACK_SOME_FLAG"]

    def test_load_env_overrides_bool_false(self):
        """Test _load_env_overrides converts 'false' to False."""
        os.environ["BENCHMARK_PACK_SOME_FLAG"] = "false"
        try:
            overrides = PackConfig._load_env_overrides()
            assert overrides.get("some_flag") is False
        finally:
            del os.environ["BENCHMARK_PACK_SOME_FLAG"]
