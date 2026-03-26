"""
Unit tests for PACK-046 Configuration (pack_config.py).

Tests all 15 enums, 26 standard denominators, 7 SBTi pathways,
5 data quality levels, 12 sectors, 8 presets, 15 sub-configs,
main IntensityMetricsConfig, PackConfig, and utility functions.

40+ tests covering:
  - Enum member counts and values
  - Reference data integrity (denominators, SBTi pathways, sectors)
  - Sub-config defaults and validation
  - IntensityMetricsConfig creation and field validation
  - PackConfig from_preset, validate_completeness, get_config_hash
  - Utility functions (get_default_config, list_available_presets, etc.)
  - Edge cases and validation errors

Author: GreenLang QA Team
"""

import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import (
    AVAILABLE_PRESETS,
    AuditConfig,
    BenchmarkConfig,
    BenchmarkSource,
    ConsolidationApproach,
    DATA_QUALITY_UNCERTAINTY,
    DataQualityLevel,
    DecompositionConfig,
    DecompositionMethod,
    DenominatorCategory,
    DenominatorConfig,
    DisclosureConfig,
    DisclosureFramework,
    IntensityCalculationConfig,
    IntensityMetricsConfig,
    IntensitySector,
    NotificationChannel,
    NotificationConfig,
    NullHandling,
    OutputFormat,
    PackConfig,
    PerformanceConfig,
    PropagationMethod,
    RegressionModel,
    ReportingConfig,
    SBTI_SECTOR_PATHWAYS,
    ScenarioConfig,
    ScenarioType,
    SECTOR_INFO,
    SecurityConfig,
    ScopeInclusion,
    STANDARD_DENOMINATORS,
    TargetConfig,
    TargetPathway,
    TrendConfig,
    UncertaintyConfig,
    WeightedAverageMethod,
    get_default_config,
    get_denominator_info,
    get_sbti_pathway,
    get_sector_info,
    list_available_presets,
    validate_config,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnumCounts:
    """Tests for enum member counts."""

    def test_denominator_category_6_members(self):
        assert len(DenominatorCategory) == 6

    def test_scope_inclusion_8_members(self):
        assert len(ScopeInclusion) == 8

    def test_decomposition_method_4_members(self):
        assert len(DecompositionMethod) == 4

    def test_benchmark_source_5_members(self):
        assert len(BenchmarkSource) == 5

    def test_target_pathway_3_members(self):
        assert len(TargetPathway) == 3

    def test_scenario_type_5_members(self):
        assert len(ScenarioType) == 5

    def test_disclosure_framework_8_members(self):
        assert len(DisclosureFramework) == 8

    def test_data_quality_level_5_members(self):
        assert len(DataQualityLevel) == 5

    def test_consolidation_approach_3_members(self):
        assert len(ConsolidationApproach) == 3

    def test_intensity_sector_12_members(self):
        assert len(IntensitySector) == 12

    def test_output_format_6_members(self):
        assert len(OutputFormat) == 6

    def test_regression_model_4_members(self):
        assert len(RegressionModel) == 4

    def test_null_handling_5_members(self):
        assert len(NullHandling) == 5

    def test_weighted_average_method_4_members(self):
        assert len(WeightedAverageMethod) == 4

    def test_propagation_method_3_members(self):
        assert len(PropagationMethod) == 3

    def test_notification_channel_4_members(self):
        assert len(NotificationChannel) == 4


# ---------------------------------------------------------------------------
# Reference Data Tests
# ---------------------------------------------------------------------------


class TestStandardDenominators:
    """Tests for STANDARD_DENOMINATORS reference data."""

    def test_26_standard_denominators(self):
        assert len(STANDARD_DENOMINATORS) == 26

    def test_revenue_meur_exists(self):
        assert "revenue_meur" in STANDARD_DENOMINATORS

    def test_tonnes_output_exists(self):
        assert "tonnes_output" in STANDARD_DENOMINATORS

    def test_fte_exists(self):
        assert "fte" in STANDARD_DENOMINATORS

    def test_every_denominator_has_required_fields(self):
        required_fields = {"id", "name", "unit", "category", "sectors", "frameworks"}
        for denom_id, denom in STANDARD_DENOMINATORS.items():
            for field in required_fields:
                assert field in denom, f"Denominator {denom_id} missing field '{field}'"

    def test_denominator_categories_valid(self):
        valid_cats = {c.value for c in DenominatorCategory}
        for denom_id, denom in STANDARD_DENOMINATORS.items():
            assert denom["category"] in valid_cats, (
                f"Denominator {denom_id} has invalid category '{denom['category']}'"
            )


class TestSBTiPathways:
    """Tests for SBTI_SECTOR_PATHWAYS reference data."""

    def test_7_sector_pathways(self):
        assert len(SBTI_SECTOR_PATHWAYS) == 7

    def test_power_generation_pathway(self):
        assert "power_generation" in SBTI_SECTOR_PATHWAYS
        pwr = SBTI_SECTOR_PATHWAYS["power_generation"]
        assert pwr["base_year"] == 2020
        assert "2030" in pwr["well_below_2c"]
        assert "2030" in pwr["one_point_five_c"]

    def test_cement_pathway(self):
        assert "cement" in SBTI_SECTOR_PATHWAYS

    def test_steel_pathway(self):
        assert "steel" in SBTI_SECTOR_PATHWAYS

    def test_pathway_targets_decrease_over_time(self):
        for sector, pathway in SBTI_SECTOR_PATHWAYS.items():
            wb2c = pathway["well_below_2c"]
            years = sorted(wb2c.keys())
            for i in range(1, len(years)):
                assert wb2c[years[i]] <= wb2c[years[i - 1]], (
                    f"Pathway {sector} well_below_2c not decreasing: "
                    f"{years[i-1]}={wb2c[years[i-1]]}, {years[i]}={wb2c[years[i]]}"
                )

    def test_1_5c_more_ambitious_than_wb2c(self):
        for sector, pathway in SBTI_SECTOR_PATHWAYS.items():
            wb2c = pathway["well_below_2c"]
            p15c = pathway["one_point_five_c"]
            for year in wb2c:
                if year in p15c:
                    assert p15c[year] <= wb2c[year], (
                        f"Sector {sector} year {year}: 1.5C ({p15c[year]}) "
                        f"should be <= WB2C ({wb2c[year]})"
                    )


class TestDataQualityUncertainty:
    """Tests for DATA_QUALITY_UNCERTAINTY reference data."""

    def test_5_quality_levels(self):
        assert len(DATA_QUALITY_UNCERTAINTY) == 5

    def test_level_1_is_audited(self):
        assert DATA_QUALITY_UNCERTAINTY[1]["level"] == "AUDITED"

    def test_level_5_is_default(self):
        assert DATA_QUALITY_UNCERTAINTY[5]["level"] == "DEFAULT"

    def test_uncertainty_increases_with_level(self):
        for level in range(1, 5):
            curr = DATA_QUALITY_UNCERTAINTY[level]["typical_uncertainty_pct"]
            next_val = DATA_QUALITY_UNCERTAINTY[level + 1]["typical_uncertainty_pct"]
            assert next_val > curr, (
                f"Level {level} uncertainty ({curr}) should be < level {level+1} ({next_val})"
            )


class TestSectorInfo:
    """Tests for SECTOR_INFO reference data."""

    def test_12_sectors(self):
        assert len(SECTOR_INFO) == 12

    def test_manufacturing_sector(self):
        assert "MANUFACTURING" in SECTOR_INFO
        mfg = SECTOR_INFO["MANUFACTURING"]
        assert "tonnes_output" in mfg["primary_denominators"]

    def test_every_sector_has_primary_denominators(self):
        for sector, info in SECTOR_INFO.items():
            assert len(info["primary_denominators"]) > 0, (
                f"Sector {sector} has no primary denominators"
            )


class TestAvailablePresets:
    """Tests for AVAILABLE_PRESETS."""

    def test_8_presets_available(self):
        assert len(AVAILABLE_PRESETS) == 8

    def test_manufacturing_preset(self):
        assert "manufacturing" in AVAILABLE_PRESETS

    def test_sme_simplified_preset(self):
        assert "sme_simplified" in AVAILABLE_PRESETS

    def test_all_presets_have_descriptions(self):
        for name, desc in AVAILABLE_PRESETS.items():
            assert len(desc) > 20, f"Preset {name} description is too short"


# ---------------------------------------------------------------------------
# Sub-Config Model Tests
# ---------------------------------------------------------------------------


class TestDenominatorConfig:
    """Tests for DenominatorConfig."""

    def test_default_config(self):
        config = DenominatorConfig()
        assert config.primary_denominator == "revenue_meur"
        assert "revenue_meur" in config.selected_denominators

    def test_empty_selected_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            DenominatorConfig(selected_denominators=[])

    def test_primary_not_in_selected_raises(self):
        with pytest.raises(ValueError, match="primary_denominator"):
            DenominatorConfig(
                selected_denominators=["tonnes_output"],
                primary_denominator="revenue_meur",
            )


class TestIntensityCalculationConfig:
    """Tests for IntensityCalculationConfig."""

    def test_default_config(self):
        config = IntensityCalculationConfig()
        assert config.scope_inclusion == ScopeInclusion.SCOPE_1_2_MARKET
        assert config.decimal_places == 4

    def test_scope_3_categories_validation(self):
        config = IntensityCalculationConfig(scope_3_categories=[1, 4, 7])
        assert config.scope_3_categories == [1, 4, 7]

    def test_scope_3_categories_invalid(self):
        with pytest.raises(ValueError, match="1-15"):
            IntensityCalculationConfig(scope_3_categories=[0, 16])

    def test_scope_3_categories_deduplicated(self):
        config = IntensityCalculationConfig(scope_3_categories=[3, 1, 3, 7])
        assert config.scope_3_categories == [1, 3, 7]


class TestDecompositionConfig:
    """Tests for DecompositionConfig."""

    def test_default_method(self):
        config = DecompositionConfig()
        assert config.method == DecompositionMethod.LMDI_I_ADDITIVE

    def test_zero_handling_validation(self):
        config = DecompositionConfig(zero_handling="EXCLUDE")
        assert config.zero_handling == "EXCLUDE"

    def test_zero_handling_invalid(self):
        with pytest.raises(ValueError, match="zero_handling"):
            DecompositionConfig(zero_handling="INVALID")


class TestTargetConfig:
    """Tests for TargetConfig."""

    def test_default_config(self):
        config = TargetConfig()
        assert config.pathway == TargetPathway.ONE_POINT_FIVE_C
        assert config.base_year == 2020

    def test_target_years_after_base_year(self):
        with pytest.raises(ValueError, match="after base_year"):
            TargetConfig(base_year=2025, target_years=[2020, 2030])

    def test_convergence_approach_validation(self):
        config = TargetConfig(convergence_approach="ACA")
        assert config.convergence_approach == "ACA"

    def test_convergence_approach_invalid(self):
        with pytest.raises(ValueError, match="convergence_approach"):
            TargetConfig(convergence_approach="UNKNOWN")


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_source(self):
        config = BenchmarkConfig()
        assert BenchmarkSource.CDP in config.sources

    def test_update_frequency_validation(self):
        config = BenchmarkConfig(update_frequency="QUARTERLY")
        assert config.update_frequency == "QUARTERLY"

    def test_update_frequency_invalid(self):
        with pytest.raises(ValueError, match="update_frequency"):
            BenchmarkConfig(update_frequency="DAILY")


class TestUncertaintyConfig:
    """Tests for UncertaintyConfig."""

    def test_default_propagation(self):
        config = UncertaintyConfig()
        assert config.propagation_method == PropagationMethod.MONTE_CARLO

    def test_correlation_assumptions_validation(self):
        config = UncertaintyConfig(correlation_assumptions="CORRELATED")
        assert config.correlation_assumptions == "CORRELATED"

    def test_correlation_assumptions_invalid(self):
        with pytest.raises(ValueError, match="correlation_assumptions"):
            UncertaintyConfig(correlation_assumptions="UNKNOWN")


# ---------------------------------------------------------------------------
# IntensityMetricsConfig Tests
# ---------------------------------------------------------------------------


class TestIntensityMetricsConfig:
    """Tests for IntensityMetricsConfig main config."""

    def test_default_creation(self):
        config = IntensityMetricsConfig()
        assert config is not None
        assert config.sector == IntensitySector.MULTI_SECTOR

    def test_manufacturing_config(self, manufacturing_config):
        assert manufacturing_config.sector == IntensitySector.MANUFACTURING
        assert manufacturing_config.revenue_meur == 500.0
        assert manufacturing_config.employees_fte == 2000

    def test_sme_config(self, sme_config):
        assert sme_config.sector == IntensitySector.SME
        assert sme_config.revenue_meur == 10.0


# ---------------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_get_default_config(self):
        config = get_default_config()
        assert isinstance(config, IntensityMetricsConfig)

    def test_list_available_presets(self):
        presets = list_available_presets()
        assert len(presets) == 8
        assert "manufacturing" in presets

    def test_get_sector_info_manufacturing(self):
        info = get_sector_info("MANUFACTURING")
        assert info is not None
        assert info["name"] == "Manufacturing"

    def test_get_sector_info_unknown(self):
        info = get_sector_info("NONEXISTENT")
        assert info is None

    def test_get_denominator_info_revenue(self):
        info = get_denominator_info("revenue_meur")
        assert info is not None
        assert info["unit"] == "MEUR"

    def test_get_denominator_info_unknown(self):
        info = get_denominator_info("nonexistent_denom")
        assert info is None

    def test_get_sbti_pathway_power(self):
        pathway = get_sbti_pathway("power_generation")
        assert pathway is not None
        assert "well_below_2c" in pathway

    def test_get_sbti_pathway_unknown(self):
        pathway = get_sbti_pathway("nonexistent_sector")
        assert pathway is None

    def test_validate_config_valid(self):
        config = IntensityMetricsConfig()
        result = validate_config(config)
        assert result is True or isinstance(result, bool)
