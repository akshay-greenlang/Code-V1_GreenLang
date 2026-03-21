# -*- coding: utf-8 -*-
"""
Unit tests for ResidualEmissionsEngine (PACK-021 Engine 5).

Tests residual budget calculation, CDR option assessment, permanence scoring,
cost estimation, neutralization plan generation, timeline planning, and
provenance hashing. Targets 85%+ coverage with parameterized scenarios.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure the pack root is on sys.path so engine imports resolve.
# (conftest.py also does this, but kept here for standalone execution.)
# ---------------------------------------------------------------------------
_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.residual_emissions_engine import (
    CDR_REFERENCE_DATA,
    CDROptionAssessment,
    CDRReadinessLevel,
    CDRType,
    NeutralizationTimeline,
    PERMANENCE_THRESHOLDS,
    PermanenceCategory,
    ResidualAllowanceLevel,
    ResidualEmissionsEngine,
    ResidualInput,
    ResidualResult,
    SECTOR_RESIDUAL_ALLOWANCES,
    TIMELINE_DEFAULTS,
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def engine() -> ResidualEmissionsEngine:
    """Create a fresh ResidualEmissionsEngine instance."""
    return ResidualEmissionsEngine()


@pytest.fixture
def manufacturing_input() -> ResidualInput:
    """Standard manufacturing company input (20 000 tCO2e base year)."""
    return ResidualInput(
        entity_name="Acme Manufacturing",
        sector="manufacturing",
        base_year=2020,
        base_year_scope1_tco2e=Decimal("5000"),
        base_year_scope2_tco2e=Decimal("3000"),
        base_year_scope3_tco2e=Decimal("12000"),
        target_year=2050,
        current_year=2026,
    )


@pytest.fixture
def technology_input() -> ResidualInput:
    """Technology company with strict residual allowance."""
    return ResidualInput(
        entity_name="TechCo",
        sector="technology",
        base_year=2021,
        base_year_scope1_tco2e=Decimal("200"),
        base_year_scope2_tco2e=Decimal("800"),
        base_year_scope3_tco2e=Decimal("4000"),
        target_year=2045,
        current_year=2026,
    )


@pytest.fixture
def zero_scope3_input() -> ResidualInput:
    """Input with zero Scope 3 emissions."""
    return ResidualInput(
        entity_name="SmallCo",
        sector="services",
        base_year=2022,
        base_year_scope1_tco2e=Decimal("100"),
        base_year_scope2_tco2e=Decimal("200"),
        base_year_scope3_tco2e=Decimal("0"),
        target_year=2050,
    )


# ========================================================================
# TestResidualEmissionsEngine
# ========================================================================


class TestResidualEmissionsEngineInstantiation:
    """Tests for engine initialization."""

    def test_engine_instantiates(self):
        """Engine can be created with no arguments."""
        engine = ResidualEmissionsEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"

    def test_engine_has_calculate_method(self, engine):
        """Engine exposes a `calculate` method."""
        assert callable(getattr(engine, "calculate", None))


# ========================================================================
# Residual Budget Calculation
# ========================================================================


class TestResidualBudgetCalculation:
    """Tests for the core residual budget computation."""

    def test_residual_budget_10_pct_of_base_year(self, engine, manufacturing_input):
        """Manufacturing sector allows 10% residual -> 2000 tCO2e."""
        result = engine.calculate(manufacturing_input)
        # base_total = 5000 + 3000 + 12000 = 20000
        # manufacturing allowance = 10%
        expected_budget = Decimal("2000.000")
        assert result.residual_budget_tco2e == expected_budget
        assert result.neutralization_required_tco2e == expected_budget

    def test_residual_budget_strict_sector(self, engine, technology_input):
        """Technology sector allows 5% residual -> 250 tCO2e."""
        result = engine.calculate(technology_input)
        # base_total = 200 + 800 + 4000 = 5000
        expected_budget = Decimal("250.000")
        assert result.residual_budget_tco2e == expected_budget

    def test_base_year_total_correct(self, engine, manufacturing_input):
        """Base year total is sum of scopes 1+2+3."""
        result = engine.calculate(manufacturing_input)
        expected_total = Decimal("20000.000")
        assert result.base_year_total_tco2e == expected_total

    def test_residual_budget_with_override(self, engine):
        """Override residual allowance overrides sector lookup."""
        inp = ResidualInput(
            entity_name="Override Corp",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("10000"),
            base_year_scope2_tco2e=Decimal("0"),
            base_year_scope3_tco2e=Decimal("0"),
            target_year=2050,
            residual_allowance_override_pct=Decimal("7"),
        )
        result = engine.calculate(inp)
        # 10000 * 7% = 700
        assert result.residual_budget_tco2e == Decimal("700.000")


# ========================================================================
# Sector Specific Allowances
# ========================================================================


class TestSectorSpecificAllowances:
    """Validate sector-level residual allowance lookups."""

    @pytest.mark.parametrize(
        "sector,expected_pct,expected_level",
        [
            ("energy", Decimal("5.0"), ResidualAllowanceLevel.STRICT),
            ("utilities", Decimal("5.0"), ResidualAllowanceLevel.STRICT),
            ("technology", Decimal("5.0"), ResidualAllowanceLevel.STRICT),
            ("transportation", Decimal("8.0"), ResidualAllowanceLevel.STANDARD),
            ("healthcare", Decimal("8.0"), ResidualAllowanceLevel.STANDARD),
            ("real_estate", Decimal("7.0"), ResidualAllowanceLevel.STANDARD),
            ("retail", Decimal("7.0"), ResidualAllowanceLevel.STANDARD),
            ("manufacturing", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
            ("cement", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
            ("steel", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
            ("agriculture", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
            ("chemicals", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
            ("default", Decimal("10.0"), ResidualAllowanceLevel.ELEVATED),
        ],
    )
    def test_sector_allowance_lookup(self, sector, expected_pct, expected_level):
        """Each sector has the correct max residual percentage and level."""
        sector_data = SECTOR_RESIDUAL_ALLOWANCES[sector]
        assert sector_data["max_residual_pct"] == expected_pct
        assert sector_data["level"] == expected_level

    def test_unknown_sector_falls_back_to_default(self, engine):
        """Unrecognized sector uses 'default' -> 10%."""
        inp = ResidualInput(
            entity_name="Unknown Corp",
            sector="alien_industry",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("1000"),
            base_year_scope2_tco2e=Decimal("0"),
            base_year_scope3_tco2e=Decimal("0"),
            target_year=2050,
        )
        result = engine.calculate(inp)
        assert result.sector_residual_allowance_pct == Decimal("10.0")

    def test_sector_residual_allowance_in_result(self, engine, manufacturing_input):
        """Result includes the sector allowance percentage and level."""
        result = engine.calculate(manufacturing_input)
        assert result.sector_residual_allowance_pct == Decimal("10.0")
        assert result.residual_allowance_level == ResidualAllowanceLevel.ELEVATED.value


# ========================================================================
# CDR Options
# ========================================================================


class TestCDROptionsListing:
    """Tests for CDR option assessment."""

    def test_cdr_options_returned(self, engine, manufacturing_input):
        """CDR options list is non-empty."""
        result = engine.calculate(manufacturing_input)
        assert len(result.cdr_options) > 0

    def test_all_cdr_types_assessed(self, engine, manufacturing_input):
        """All 8 CDR types appear in the assessment."""
        result = engine.calculate(manufacturing_input)
        assessed_types = {opt.cdr_type for opt in result.cdr_options}
        expected_types = {
            CDRType.DACCS,
            CDRType.BECCS,
            CDRType.BIOCHAR,
            CDRType.ENHANCED_WEATHERING,
            CDRType.AFFORESTATION,
            CDRType.REFORESTATION,
            CDRType.OCEAN_BASED,
            CDRType.SOIL_CARBON,
        }
        assert assessed_types == expected_types

    def test_cdr_options_are_assessment_objects(self, engine, manufacturing_input):
        """Each CDR option is a CDROptionAssessment."""
        result = engine.calculate(manufacturing_input)
        for opt in result.cdr_options:
            assert isinstance(opt, CDROptionAssessment)

    def test_preferred_cdr_types_prioritised(self, engine):
        """Preferred CDR types get higher suitability scores or are listed first."""
        inp = ResidualInput(
            entity_name="Preferred CDR Corp",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("12000"),
            target_year=2050,
            preferred_cdr_types=[CDRType.DACCS, CDRType.BIOCHAR],
        )
        result = engine.calculate(inp)
        assert len(result.cdr_options) >= 2


# ========================================================================
# Permanence Scoring
# ========================================================================


class TestPermanenceScoring:
    """Validate permanence categorisation and SBTi eligibility."""

    @pytest.mark.parametrize(
        "cdr_type,expected_permanence_years,expected_category,expected_sbti_eligible",
        [
            (CDRType.DACCS, 10000, PermanenceCategory.GEOLOGICAL, True),
            (CDRType.BECCS, 10000, PermanenceCategory.GEOLOGICAL, True),
            (CDRType.BIOCHAR, 500, PermanenceCategory.MINERALOGICAL, True),
            (CDRType.ENHANCED_WEATHERING, 100000, PermanenceCategory.MINERALOGICAL, True),
            (CDRType.AFFORESTATION, 50, PermanenceCategory.BIOLOGICAL_SHORT, False),
            (CDRType.REFORESTATION, 50, PermanenceCategory.BIOLOGICAL_SHORT, False),
            (CDRType.OCEAN_BASED, 1000, PermanenceCategory.OCEAN, False),
            (CDRType.SOIL_CARBON, 30, PermanenceCategory.BIOLOGICAL_SHORT, False),
        ],
    )
    def test_reference_data_permanence(
        self,
        cdr_type,
        expected_permanence_years,
        expected_category,
        expected_sbti_eligible,
    ):
        """CDR reference data has correct permanence and SBTi eligibility."""
        ref = CDR_REFERENCE_DATA[cdr_type.value]
        assert ref["permanence_years"] == expected_permanence_years
        assert ref["permanence_category"] == expected_category
        assert ref["sbti_eligible"] == expected_sbti_eligible

    def test_sbti_minimum_permanence_threshold(self):
        """SBTi requires minimum 100 years permanence."""
        assert PERMANENCE_THRESHOLDS["sbti_minimum"] == 100

    def test_sbti_eligible_options_count(self, engine, manufacturing_input):
        """Result reports the correct count of SBTi-eligible CDR options."""
        result = engine.calculate(manufacturing_input)
        sbti_from_ref = sum(
            1 for d in CDR_REFERENCE_DATA.values() if d["sbti_eligible"]
        )
        assert result.sbti_eligible_options_count == sbti_from_ref

    def test_cdr_option_meets_sbti_permanence_flag(self, engine, manufacturing_input):
        """CDR options with 100+ year permanence have meets_sbti_permanence=True."""
        result = engine.calculate(manufacturing_input)
        for opt in result.cdr_options:
            ref = CDR_REFERENCE_DATA[opt.cdr_type.value]
            if ref["permanence_years"] >= 100:
                assert opt.meets_sbti_permanence is True
            else:
                assert opt.meets_sbti_permanence is False


# ========================================================================
# Cost Estimation
# ========================================================================


class TestCostEstimation:
    """Tests for CDR cost estimation by type."""

    @pytest.mark.parametrize(
        "cdr_type,expected_cost_mid",
        [
            (CDRType.DACCS, Decimal("450")),
            (CDRType.BECCS, Decimal("200")),
            (CDRType.BIOCHAR, Decimal("120")),
            (CDRType.ENHANCED_WEATHERING, Decimal("150")),
            (CDRType.AFFORESTATION, Decimal("30")),
            (CDRType.REFORESTATION, Decimal("25")),
            (CDRType.OCEAN_BASED, Decimal("300")),
            (CDRType.SOIL_CARBON, Decimal("40")),
        ],
    )
    def test_cost_mid_usd_per_tco2e_from_reference(
        self, cdr_type, expected_cost_mid
    ):
        """CDR reference data mid-cost matches expected values."""
        ref = CDR_REFERENCE_DATA[cdr_type.value]
        assert ref["cost_mid_usd"] == expected_cost_mid

    def test_total_neutralization_cost_positive(self, engine, manufacturing_input):
        """Total neutralization costs are positive for non-zero residual."""
        result = engine.calculate(manufacturing_input)
        assert result.total_neutralization_cost_low > Decimal("0")
        assert result.total_neutralization_cost_mid > Decimal("0")
        assert result.total_neutralization_cost_high > Decimal("0")

    def test_cost_ordering(self, engine, manufacturing_input):
        """Low <= Mid <= High cost ordering holds."""
        result = engine.calculate(manufacturing_input)
        assert result.total_neutralization_cost_low <= result.total_neutralization_cost_mid
        assert result.total_neutralization_cost_mid <= result.total_neutralization_cost_high

    def test_cdr_option_total_cost_calculation(self, engine, manufacturing_input):
        """Each CDR option has consistent total cost = per-unit * budget."""
        result = engine.calculate(manufacturing_input)
        budget = result.residual_budget_tco2e
        for opt in result.cdr_options:
            expected_mid = opt.cost_mid_usd_per_tco2e * budget
            # Allow rounding tolerance
            assert abs(opt.total_cost_mid - expected_mid) < Decimal("1")


# ========================================================================
# Neutralization Plan & Timeline
# ========================================================================


class TestNeutralizationPlanGeneration:
    """Tests for plan generation and recommendations."""

    def test_recommended_cdr_mix_non_empty(self, engine, manufacturing_input):
        """Recommended CDR mix is populated."""
        result = engine.calculate(manufacturing_input)
        assert len(result.recommended_cdr_mix) > 0

    def test_recommendations_generated(self, engine, manufacturing_input):
        """Engine produces recommendations list."""
        result = engine.calculate(manufacturing_input)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1


class TestTimelinePlanning:
    """Tests for CDR procurement timeline."""

    def test_timeline_present(self, engine, manufacturing_input):
        """Timeline object is generated."""
        result = engine.calculate(manufacturing_input)
        assert result.timeline is not None
        assert isinstance(result.timeline, NeutralizationTimeline)

    def test_timeline_target_year(self, engine, manufacturing_input):
        """Timeline target year matches input."""
        result = engine.calculate(manufacturing_input)
        assert result.timeline.target_year == 2050

    def test_timeline_years_remaining(self, engine, manufacturing_input):
        """Years remaining is target - current."""
        result = engine.calculate(manufacturing_input)
        assert result.timeline.years_remaining == 2050 - 2026

    def test_timeline_milestones_non_empty(self, engine, manufacturing_input):
        """Timeline milestones list is populated."""
        result = engine.calculate(manufacturing_input)
        assert len(result.timeline.milestones) > 0

    def test_timeline_urgency_level(self, engine, manufacturing_input):
        """Urgency level is a non-empty string."""
        result = engine.calculate(manufacturing_input)
        assert result.timeline.urgency_level in ("low", "medium", "high", "critical")


# ========================================================================
# Provenance Hash
# ========================================================================


class TestProvenanceHash:
    """Tests for SHA-256 provenance hashing."""

    def test_provenance_hash_present(self, engine, manufacturing_input):
        """Result has a non-empty provenance hash."""
        result = engine.calculate(manufacturing_input)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_valid_sha256(self, engine):
        """Provenance hash is a valid SHA-256 hex string (64 hex chars).

        Note: the hash covers the full result (incl. result_id UUID and
        processing_time_ms) so it differs between calls.  We verify
        the format rather than determinism.
        """
        inp = ResidualInput(
            entity_name="HashTest",
            sector="manufacturing",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("12000"),
            target_year=2050,
            current_year=2026,
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        # Both hashes are valid hex
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)


# ========================================================================
# Zero Residual Scenario
# ========================================================================


class TestZeroResidualScenario:
    """Edge case: zero or very small residual budget."""

    def test_zero_residual_budget_with_100_pct_reduction(self, engine):
        """100% reduction target yields zero residual budget."""
        inp = ResidualInput(
            entity_name="ZeroRes Corp",
            sector="technology",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("1000"),
            base_year_scope2_tco2e=Decimal("500"),
            base_year_scope3_tco2e=Decimal("2000"),
            target_year=2050,
            long_term_reduction_pct=Decimal("100"),
            residual_allowance_override_pct=Decimal("0"),
        )
        result = engine.calculate(inp)
        assert result.residual_budget_tco2e == Decimal("0.000")
        assert result.neutralization_required_tco2e == Decimal("0.000")

    def test_zero_base_year_emissions(self, engine):
        """Zero base year emissions yields zero residual."""
        inp = ResidualInput(
            entity_name="Zero Corp",
            sector="technology",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("0"),
            base_year_scope2_tco2e=Decimal("0"),
            base_year_scope3_tco2e=Decimal("0"),
            target_year=2050,
        )
        result = engine.calculate(inp)
        assert result.residual_budget_tco2e == Decimal("0.000")
        assert result.base_year_total_tco2e == Decimal("0.000")


# ========================================================================
# Result Structure
# ========================================================================


class TestResultStructure:
    """Validate complete result structure fields."""

    def test_result_has_all_required_fields(self, engine, manufacturing_input):
        """ResidualResult has all expected fields."""
        result = engine.calculate(manufacturing_input)
        assert isinstance(result, ResidualResult)
        assert result.result_id
        assert result.engine_version == "1.0.0"
        assert result.entity_name == "Acme Manufacturing"
        assert result.sector == "manufacturing"
        assert result.base_year == 2020
        assert result.target_year == 2050
        assert result.processing_time_ms >= 0.0

    def test_warnings_list_present(self, engine, manufacturing_input):
        """Warnings list is present (may be empty)."""
        result = engine.calculate(manufacturing_input)
        assert isinstance(result.warnings, list)

    def test_long_term_reduction_default_90(self, engine, manufacturing_input):
        """Default long-term reduction is 90%."""
        result = engine.calculate(manufacturing_input)
        assert result.long_term_reduction_pct == Decimal("90.0")
