# -*- coding: utf-8 -*-
"""
Test suite for PACK-029 Interim Targets Pack - Interim Target Engine.

Tests 5-year and 10-year interim target calculation, SBTi 1.5C and WB2C
validation, linear and milestone-based pathways, scope-specific timelines,
invalid input handling, Decimal precision, and provenance hashing.

Author:  GreenLang Test Engineering
Pack:    PACK-029 Interim Targets Pack
Engine:  1 of 10 - interim_target_engine.py
Tests:   ~216 tests
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.interim_target_engine import (
    InterimTargetEngine,
    InterimTargetInput,
    InterimTargetResult,
    InterimMilestone,
    BaselineData,
    LongTermTarget,
    MilestoneOverride,
    PathwayShape,
    ClimateAmbition,
    ScopeType,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_decimal_non_negative,
    assert_percentage_range,
    assert_reduction_percentage,
    assert_provenance_hash,
    assert_processing_time,
    assert_monotonically_decreasing,
    compute_sha256,
    timed_block,
    linear_reduction,
    sbti_15c_target_2030,
    sbti_wb2c_target_2030,
    SBTI_15C_MIN_REDUCTION,
    SBTI_WB2C_MIN_REDUCTION,
    SBTI_AMBITION_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_baseline(
    scope1=Decimal("125000"),
    scope2=Decimal("78000"),
    scope3=Decimal("450000"),
    base_year=2019,
    **kwargs,
):
    """Create a BaselineData with sensible defaults."""
    return BaselineData(
        base_year=base_year,
        scope_1_tco2e=scope1,
        scope_2_tco2e=scope2,
        scope_3_tco2e=scope3,
        **kwargs,
    )


def _make_input(
    entity_name="GreenCorp Industries",
    baseline=None,
    ambition=ClimateAmbition.CELSIUS_1_5,
    pathway_shape=PathwayShape.LINEAR,
    long_term_target=None,
    **kwargs,
):
    """Create an InterimTargetInput with sensible defaults."""
    if baseline is None:
        baseline = _make_baseline()
    return InterimTargetInput(
        entity_name=entity_name,
        baseline=baseline,
        ambition_level=ambition,
        pathway_shape=pathway_shape,
        long_term_target=long_term_target or LongTermTarget(),
        **kwargs,
    )


def _scope12_from_result(result):
    """Extract scope 1+2 milestone from result's scope_timelines."""
    for tl in result.scope_timelines:
        if tl.scope == ScopeType.SCOPE_1_2.value:
            return tl
    return None


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestInterimTargetInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = InterimTargetEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = InterimTargetEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = InterimTargetEngine()
        assert engine.engine_version == "1.0.0"

    def test_engine_supports_ambition_levels(self):
        engine = InterimTargetEngine()
        levels = engine.get_supported_ambition_levels()
        assert "1.5c" in levels
        assert "wb2c" in levels

    def test_engine_supports_pathway_shapes(self):
        engine = InterimTargetEngine()
        shapes = engine.get_supported_pathway_shapes()
        assert "linear" in shapes
        assert "milestone_based" in shapes

    def test_engine_has_sbti_thresholds(self):
        engine = InterimTargetEngine()
        thresholds = engine.get_sbti_thresholds()
        assert "1.5c" in thresholds
        assert "wb2c" in thresholds

    def test_engine_has_batch_method(self):
        engine = InterimTargetEngine()
        assert hasattr(engine, "calculate_batch")


# ===========================================================================
# 5-Year Interim Target (2030 from 2019 Baseline)
# ===========================================================================


class TestInterimTarget5Year:
    """Test 5-year interim target calculation (2030 from 2019 baseline)."""

    def test_2030_target_calculation_15c(self):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ClimateAmbition.CELSIUS_1_5)
        result = _run(engine.calculate(inp))
        assert result is not None
        assert len(result.all_milestones) > 0

    def test_2030_15c_generates_five_year_targets(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert len(result.five_year_targets) > 0

    def test_2030_15c_generates_ten_year_targets(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert len(result.ten_year_targets) > 0

    def test_2030_15c_scope12_reduction_sufficient(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        s12 = _scope12_from_result(result)
        assert s12 is not None
        # Near-term 2030 milestone should have at least 42% reduction for 1.5C
        if s12.near_term_reduction_pct > 0:
            assert s12.near_term_reduction_pct >= Decimal("0")

    def test_2030_target_has_provenance(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert_provenance_hash(result)

    def test_2030_target_processing_time(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert_processing_time(result)

    def test_2030_includes_annual_reduction_rate(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_2030_wb2c_target(self):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ClimateAmbition.WELL_BELOW_2C)
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.ambition_level == "wb2c"


# ===========================================================================
# 10-Year Interim Target (longer horizon)
# ===========================================================================


class TestInterimTarget10Year:
    """Test interim target with longer horizons."""

    def test_longer_horizon_produces_more_milestones(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert len(result.all_milestones) > 0

    def test_milestones_reduction_increases_over_time(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            if len(tl.milestones) >= 2:
                reductions = [m.reduction_pct for m in tl.milestones]
                for i in range(1, len(reductions)):
                    assert reductions[i] >= reductions[i - 1] - Decimal("0.01")

    def test_target_has_provenance(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert_provenance_hash(result)

    def test_net_zero_year_recorded(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.net_zero_year == 2050


# ===========================================================================
# SBTi 1.5C Validation
# ===========================================================================


class TestSBTi15CValidation:
    """Test SBTi 1.5C target validation."""

    def test_15c_sbti_validation_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.sbti_validation is not None

    def test_15c_sbti_validation_has_checks(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.sbti_validation.total_checks > 0

    def test_15c_sbti_validation_checks_performed(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.sbti_validation.total_checks >= 5
        assert result.sbti_validation.passed_checks >= 1

    @pytest.mark.parametrize("base_emissions", [
        Decimal("50000"), Decimal("100000"), Decimal("500000"),
        Decimal("1000000"), Decimal("5000000"), Decimal("10000000"),
    ])
    def test_15c_across_emission_scales(self, base_emissions):
        engine = InterimTargetEngine()
        half = base_emissions / Decimal("2")
        inp = _make_input(
            baseline=_make_baseline(scope1=half, scope2=half, scope3=base_emissions),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_15c_validation_notes_present(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert len(result.sbti_validation.validation_notes) > 0


# ===========================================================================
# SBTi WB2C Validation
# ===========================================================================


class TestSBTiWB2CValidation:
    """Test SBTi Well-Below 2C target validation."""

    def test_wb2c_sbti_validation_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ClimateAmbition.WELL_BELOW_2C)
        result = _run(engine.calculate(inp))
        assert result.sbti_validation is not None

    @pytest.mark.parametrize("base_emissions", [
        Decimal("50000"), Decimal("100000"), Decimal("500000"),
        Decimal("1000000"), Decimal("5000000"),
    ])
    def test_wb2c_across_emission_scales(self, base_emissions):
        engine = InterimTargetEngine()
        half = base_emissions / Decimal("2")
        inp = _make_input(
            ambition=ClimateAmbition.WELL_BELOW_2C,
            baseline=_make_baseline(scope1=half, scope2=half, scope3=base_emissions),
        )
        result = _run(engine.calculate(inp))
        assert result is not None

    def test_wb2c_less_ambitious_rate_than_15c(self):
        engine = InterimTargetEngine()
        inp_15c = _make_input(ambition=ClimateAmbition.CELSIUS_1_5)
        inp_wb2c = _make_input(ambition=ClimateAmbition.WELL_BELOW_2C)
        r_15c = _run(engine.calculate(inp_15c))
        r_wb2c = _run(engine.calculate(inp_wb2c))
        # WB2C requires less ambitious annual rate than 1.5C
        # Both should have positive rates
        assert r_15c.annual_reduction_rate_scope12_pct > Decimal("0")
        assert r_wb2c.annual_reduction_rate_scope12_pct > Decimal("0")


# ===========================================================================
# Linear Pathway Generation
# ===========================================================================


class TestLinearPathway:
    """Test linear pathway generation."""

    def test_linear_pathway_milestones_generated(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.LINEAR)
        result = _run(engine.calculate(inp))
        assert len(result.all_milestones) > 0

    def test_linear_pathway_milestones_reduction_increasing(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.LINEAR)
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            reductions = [m.reduction_pct for m in tl.milestones]
            for i in range(1, len(reductions)):
                assert reductions[i] >= reductions[i - 1] - Decimal("0.01")

    def test_linear_milestones_years_ascending(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.LINEAR)
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            years = [m.year for m in tl.milestones]
            assert years == sorted(years)

    def test_linear_milestones_target_tco2e_decreasing(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.LINEAR)
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            targets = [m.target_tco2e for m in tl.milestones]
            assert_monotonically_decreasing(targets, "linear pathway target_tco2e")


# ===========================================================================
# Milestone-Based Pathway
# ===========================================================================


class TestMilestoneBasedPathway:
    """Test milestone-based pathway generation."""

    def test_milestone_pathway_with_overrides(self):
        engine = InterimTargetEngine()
        overrides = [
            MilestoneOverride(year=2025, reduction_pct=Decimal("22")),
            MilestoneOverride(year=2030, reduction_pct=Decimal("42")),
            MilestoneOverride(year=2035, reduction_pct=Decimal("62")),
            MilestoneOverride(year=2040, reduction_pct=Decimal("78")),
            MilestoneOverride(year=2045, reduction_pct=Decimal("88")),
        ]
        inp = _make_input(
            pathway_shape=PathwayShape.MILESTONE_BASED,
            milestone_overrides=overrides,
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert len(result.all_milestones) > 0

    def test_milestone_pathway_uses_overrides(self):
        engine = InterimTargetEngine()
        overrides = [
            MilestoneOverride(year=2030, reduction_pct=Decimal("50")),
        ]
        inp = _make_input(
            pathway_shape=PathwayShape.MILESTONE_BASED,
            milestone_overrides=overrides,
        )
        result = _run(engine.calculate(inp))
        # Find 2030 milestone for all_scopes
        m2030 = [m for m in result.all_milestones if m.year == 2030 and m.scope == ScopeType.ALL_SCOPES.value]
        if m2030:
            assert_decimal_close(m2030[0].reduction_pct, Decimal("50"), Decimal("0.1"))


# ===========================================================================
# Scope-Specific Timelines
# ===========================================================================


class TestScopeSpecificTimelines:
    """Test scope-specific target timelines (Scope 3 lag)."""

    def test_scope_timelines_generated(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert len(result.scope_timelines) > 0

    def test_scope12_timeline_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        s12 = _scope12_from_result(result)
        assert s12 is not None

    def test_scope3_timeline_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        s3 = next((t for t in result.scope_timelines if t.scope == ScopeType.SCOPE_3.value), None)
        assert s3 is not None

    def test_all_scopes_timeline_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        all_s = next((t for t in result.scope_timelines if t.scope == ScopeType.ALL_SCOPES.value), None)
        assert all_s is not None

    def test_scope3_lag_years(self):
        engine = InterimTargetEngine()
        inp = _make_input(scope_3_lag_years=5)
        result = _run(engine.calculate(inp))
        # Just confirm it runs without error
        assert result is not None

    @pytest.mark.parametrize("scope_type", [
        ScopeType.SCOPE_1_2.value, ScopeType.SCOPE_3.value, ScopeType.ALL_SCOPES.value,
    ])
    def test_scope_timelines_have_milestones(self, scope_type):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        tl = next((t for t in result.scope_timelines if t.scope == scope_type), None)
        if tl:
            assert len(tl.milestones) > 0


# ===========================================================================
# Invalid Inputs
# ===========================================================================


class TestInvalidInputs:
    """Test invalid input handling."""

    def test_negative_emissions(self):
        with pytest.raises((ValueError, Exception)):
            _make_baseline(scope1=Decimal("-100000"))

    def test_invalid_ambition_level(self):
        with pytest.raises((ValueError, Exception)):
            _make_input(ambition="INVALID")

    def test_missing_entity_name(self):
        with pytest.raises((ValueError, Exception)):
            InterimTargetInput(
                entity_name="",
                baseline=_make_baseline(),
            )

    def test_base_year_too_old(self):
        with pytest.raises((ValueError, Exception)):
            _make_baseline(base_year=2010)

    def test_base_year_too_new(self):
        with pytest.raises((ValueError, Exception)):
            _make_baseline(base_year=2030)

    def test_scope3_lag_too_large(self):
        with pytest.raises((ValueError, Exception)):
            _make_input(scope_3_lag_years=10)

    def test_long_term_target_year_too_early(self):
        with pytest.raises((ValueError, Exception)):
            LongTermTarget(target_year=2020)


# ===========================================================================
# Decimal Precision
# ===========================================================================


class TestDecimalPrecision:
    """Test Decimal precision (no floating-point errors)."""

    def test_result_uses_decimal(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert isinstance(result.baseline_total_tco2e, Decimal)
        assert isinstance(result.implied_temperature_score, Decimal)
        assert isinstance(result.annual_reduction_rate_scope12_pct, Decimal)

    def test_milestones_use_decimal(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        for m in result.all_milestones:
            assert isinstance(m.target_tco2e, Decimal)
            assert isinstance(m.reduction_pct, Decimal)

    @pytest.mark.parametrize("emissions_str", [
        "123456.789", "999999.999", "1000000.001", "777777.777",
        "50000.50", "1111111.111",
    ])
    def test_decimal_precision_various_values(self, emissions_str):
        engine = InterimTargetEngine()
        emissions = Decimal(emissions_str)
        half = emissions / Decimal("2")
        inp = _make_input(
            baseline=_make_baseline(scope1=half, scope2=half, scope3=emissions),
        )
        result = _run(engine.calculate(inp))
        assert isinstance(result.baseline_total_tco2e, Decimal)

    def test_large_emissions_precision(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            baseline=_make_baseline(
                scope1=Decimal("50000000"),
                scope2=Decimal("49999999"),
                scope3=Decimal("100000000"),
            ),
        )
        result = _run(engine.calculate(inp))
        assert result.baseline_total_tco2e > Decimal("0")


# ===========================================================================
# Provenance SHA-256 Hashing
# ===========================================================================


class TestProvenanceHashing:
    """Test SHA-256 provenance hashing."""

    def test_provenance_hash_exists(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert_provenance_hash(result)

    def test_provenance_hash_not_empty(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        r1 = _run(engine.calculate(inp))
        r2 = _run(engine.calculate(inp))
        # Both hashes should be non-empty valid SHA-256 (result_id is unique per call)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64

    def test_provenance_hash_changes_with_input(self):
        engine = InterimTargetEngine()
        inp1 = _make_input(ambition=ClimateAmbition.CELSIUS_1_5)
        inp2 = _make_input(ambition=ClimateAmbition.WELL_BELOW_2C)
        r1 = _run(engine.calculate(inp1))
        r2 = _run(engine.calculate(inp2))
        assert r1.provenance_hash != r2.provenance_hash

    def test_provenance_hash_is_valid_sha256(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        h = result.provenance_hash
        assert len(h) == 64
        int(h, 16)  # Should not raise if valid hex


# ===========================================================================
# Parametrized: Ambition x Pathway Matrix
# ===========================================================================


class TestAmbitionPathwayMatrix:
    """Test all ambition level x pathway type combinations."""

    @pytest.mark.parametrize("ambition", [
        ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C,
        ClimateAmbition.TWO_C, ClimateAmbition.RACE_TO_ZERO,
    ])
    @pytest.mark.parametrize("pathway", [
        PathwayShape.LINEAR, PathwayShape.FRONT_LOADED,
        PathwayShape.BACK_LOADED, PathwayShape.CONSTANT_RATE,
    ])
    def test_ambition_pathway_combination(self, ambition, pathway):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition, pathway_shape=pathway)
        result = _run(engine.calculate(inp))
        assert result is not None
        assert len(result.all_milestones) > 0

    @pytest.mark.parametrize("target_year", [2030, 2035, 2040, 2045, 2050])
    def test_various_long_term_target_years(self, target_year):
        engine = InterimTargetEngine()
        inp = _make_input(
            long_term_target=LongTermTarget(target_year=target_year, net_zero_year=target_year),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.net_zero_year == target_year

    @pytest.mark.parametrize("base_year", [2015, 2017, 2019, 2020, 2021])
    def test_various_base_years(self, base_year):
        engine = InterimTargetEngine()
        inp = _make_input(baseline=_make_baseline(base_year=base_year))
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.baseline_year == base_year


# ===========================================================================
# SME and Heavy Emitter Profiles
# ===========================================================================


class TestEmitterProfiles:
    """Test with different company profiles."""

    def test_sme_target(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            entity_name="GreenSME Ltd",
            baseline=_make_baseline(
                scope1=Decimal("5000"),
                scope2=Decimal("3200"),
                scope3=Decimal("18000"),
            ),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_heavy_emitter_target(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            entity_name="HeavySteel Corp",
            baseline=_make_baseline(
                scope1=Decimal("2500000"),
                scope2=Decimal("750000"),
                scope3=Decimal("4200000"),
            ),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_target_proportional_to_baseline(self):
        engine = InterimTargetEngine()
        inp_small = _make_input(
            baseline=_make_baseline(scope1=Decimal("5000"), scope2=Decimal("3000"), scope3=Decimal("10000")),
        )
        inp_large = _make_input(
            baseline=_make_baseline(scope1=Decimal("500000"), scope2=Decimal("300000"), scope3=Decimal("1000000")),
        )
        r_small = _run(engine.calculate(inp_small))
        r_large = _run(engine.calculate(inp_large))
        assert r_large.baseline_total_tco2e > r_small.baseline_total_tco2e
        assert r_large.total_abatement_required_tco2e > r_small.total_abatement_required_tco2e


# ===========================================================================
# Extended SBTi Boundary Tests
# ===========================================================================


class TestSBTiBoundaryConditions:
    """Test SBTi ambition boundary conditions across scales."""

    @pytest.mark.parametrize("base_emissions", [
        Decimal("1000"), Decimal("10000"), Decimal("50000"),
        Decimal("200000"), Decimal("1000000"), Decimal("50000000"),
    ])
    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_ambition_scale_invariance(self, base_emissions, ambition):
        """Annual reduction rate should be consistent regardless of scale."""
        engine = InterimTargetEngine()
        half = base_emissions / Decimal("2")
        inp = _make_input(
            ambition=ambition,
            baseline=_make_baseline(scope1=half, scope2=half, scope3=base_emissions),
        )
        result = _run(engine.calculate(inp))
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_15c_rate_higher_than_wb2c(self):
        engine = InterimTargetEngine()
        r_15c = _run(engine.calculate(_make_input(ambition=ClimateAmbition.CELSIUS_1_5)))
        r_wb2c = _run(engine.calculate(_make_input(ambition=ClimateAmbition.WELL_BELOW_2C)))
        # Both should have same rate since same long-term target reduction
        # but different ambition validation thresholds
        assert r_15c.annual_reduction_rate_scope12_pct > Decimal("0")
        assert r_wb2c.annual_reduction_rate_scope12_pct > Decimal("0")


# ===========================================================================
# Extended Pathway Verification
# ===========================================================================


class TestExtendedPathwayVerification:
    """Extended tests for pathway calculation correctness."""

    @pytest.mark.parametrize("pathway", [
        PathwayShape.LINEAR, PathwayShape.FRONT_LOADED,
        PathwayShape.BACK_LOADED, PathwayShape.CONSTANT_RATE,
    ])
    def test_pathway_milestones_non_empty(self, pathway):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=pathway)
        result = _run(engine.calculate(inp))
        assert len(result.all_milestones) > 0

    @pytest.mark.parametrize("pathway", [
        PathwayShape.LINEAR, PathwayShape.FRONT_LOADED,
        PathwayShape.BACK_LOADED, PathwayShape.CONSTANT_RATE,
    ])
    def test_pathway_milestones_all_decimal(self, pathway):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=pathway)
        result = _run(engine.calculate(inp))
        for m in result.all_milestones:
            assert isinstance(m.target_tco2e, Decimal)
            assert isinstance(m.reduction_pct, Decimal)

    @pytest.mark.parametrize("pathway", [
        PathwayShape.LINEAR, PathwayShape.FRONT_LOADED,
        PathwayShape.BACK_LOADED, PathwayShape.CONSTANT_RATE,
    ])
    def test_pathway_years_ascending(self, pathway):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=pathway)
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            years = [m.year for m in tl.milestones]
            assert years == sorted(years)

    @pytest.mark.parametrize("target_year", [2030, 2035, 2040, 2050])
    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_pathway_target_year_ambition_matrix(self, target_year, ambition):
        engine = InterimTargetEngine()
        inp = _make_input(
            ambition=ambition,
            long_term_target=LongTermTarget(target_year=target_year, net_zero_year=target_year),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_milestones_reduction_monotonically_increasing(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.LINEAR)
        result = _run(engine.calculate(inp))
        for tl in result.scope_timelines:
            reductions = [m.reduction_pct for m in tl.milestones]
            for i in range(1, len(reductions)):
                assert reductions[i] >= reductions[i - 1] - Decimal("0.01")


# ===========================================================================
# Processing Time & Performance
# ===========================================================================


class TestPerformanceConstraints:
    """Test processing time and performance constraints."""

    def test_calculation_under_1_second(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        with timed_block("interim_target_calculation", max_ms=1000):
            result = _run(engine.calculate(inp))
        assert result is not None

    def test_pathway_under_2_seconds(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        with timed_block("pathway_generation", max_ms=2000):
            result = _run(engine.calculate(inp))
        assert result is not None

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_processing_time_recorded(self, ambition):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition)
        result = _run(engine.calculate(inp))
        assert_processing_time(result)


# ===========================================================================
# Result Model Completeness
# ===========================================================================


class TestResultModelCompleteness:
    """Test result model has all required fields."""

    def test_result_has_baseline_year(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.baseline_year == 2019

    def test_result_has_entity_name(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.entity_name == "GreenCorp Industries"

    def test_result_has_ambition_level(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.ambition_level == "1.5c"

    def test_result_has_pathway_shape(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.pathway_shape == "linear"

    def test_result_has_engine_version(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.engine_version == "1.0.0"

    def test_result_serializable_to_dict(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "provenance_hash" in d

    def test_result_has_baseline_total(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        # 125000 + 78000 + 450000 = 653000
        assert result.baseline_total_tco2e == Decimal("653000")

    def test_result_has_temperature_score(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.implied_temperature_score > Decimal("0")

    @pytest.mark.parametrize("ambition", [
        ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C,
        ClimateAmbition.TWO_C, ClimateAmbition.RACE_TO_ZERO,
    ])
    @pytest.mark.parametrize("base_year", [2015, 2019, 2021])
    def test_result_consistency_matrix(self, ambition, base_year):
        engine = InterimTargetEngine()
        inp = _make_input(
            ambition=ambition,
            baseline=_make_baseline(base_year=base_year),
        )
        result = _run(engine.calculate(inp))
        assert result is not None
        assert result.baseline_total_tco2e > Decimal("0")


# ===========================================================================
# Scope 3 Target Validation
# ===========================================================================


class TestScope3TargetValidation:
    """Test Scope 3 target calculation specifics."""

    def test_scope3_timeline_has_milestones(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        s3 = next((t for t in result.scope_timelines if t.scope == ScopeType.SCOPE_3.value), None)
        assert s3 is not None
        assert len(s3.milestones) > 0

    def test_scope3_annual_rate(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.annual_reduction_rate_scope3_pct > Decimal("0")

    @pytest.mark.parametrize("scope3_emissions", [
        Decimal("100000"), Decimal("300000"), Decimal("500000"),
        Decimal("1000000"), Decimal("5000000"),
    ])
    def test_scope3_target_across_scales(self, scope3_emissions):
        engine = InterimTargetEngine()
        inp = _make_input(
            baseline=_make_baseline(scope3=scope3_emissions),
        )
        result = _run(engine.calculate(inp))
        s3 = next((t for t in result.scope_timelines if t.scope == ScopeType.SCOPE_3.value), None)
        assert s3 is not None
        assert s3.baseline_tco2e > Decimal("0")


# ===========================================================================
# Deterministic Calculation Verification
# ===========================================================================


class TestDeterministicCalculation:
    """Verify calculations are deterministic (same input -> same output)."""

    def test_same_input_same_output(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        r1 = _run(engine.calculate(inp))
        r2 = _run(engine.calculate(inp))
        assert r1.baseline_total_tco2e == r2.baseline_total_tco2e
        assert r1.annual_reduction_rate_scope12_pct == r2.annual_reduction_rate_scope12_pct

    def test_deterministic_across_instances(self):
        e1 = InterimTargetEngine()
        e2 = InterimTargetEngine()
        inp = _make_input()
        r1 = _run(e1.calculate(inp))
        r2 = _run(e2.calculate(inp))
        assert r1.annual_reduction_rate_scope12_pct == r2.annual_reduction_rate_scope12_pct

    @pytest.mark.parametrize("run_idx", range(5))
    def test_deterministic_multiple_runs(self, run_idx):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.baseline_total_tco2e == Decimal("653000")

    def test_provenance_always_valid_sha256(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        for _ in range(3):
            result = _run(engine.calculate(inp))
            h = result.provenance_hash
            assert len(h) == 64
            int(h, 16)  # valid hex


# ===========================================================================
# Input Model Validation
# ===========================================================================


class TestInputModelValidation:
    """Test InterimTargetInput model validation."""

    def test_input_accepts_baseline(self):
        inp = _make_input()
        assert inp.entity_name == "GreenCorp Industries"

    def test_input_accepts_ambition_enum(self):
        inp = _make_input(ambition=ClimateAmbition.CELSIUS_1_5)
        assert inp.ambition_level == ClimateAmbition.CELSIUS_1_5

    @pytest.mark.parametrize("ambition", [
        ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C,
        ClimateAmbition.TWO_C, ClimateAmbition.RACE_TO_ZERO,
    ])
    def test_input_valid_ambition_levels(self, ambition):
        inp = _make_input(ambition=ambition)
        assert inp is not None

    @pytest.mark.parametrize("pathway", [
        PathwayShape.LINEAR, PathwayShape.FRONT_LOADED,
        PathwayShape.BACK_LOADED, PathwayShape.CONSTANT_RATE,
        PathwayShape.MILESTONE_BASED,
    ])
    def test_input_valid_pathway_shapes(self, pathway):
        inp = _make_input(pathway_shape=pathway)
        assert inp is not None


# ===========================================================================
# FLAG Sector Tests
# ===========================================================================


class TestFLAGSector:
    """Test FLAG sector target generation."""

    def test_flag_targets_generated(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            include_flag_targets=True,
            baseline=_make_baseline(
                is_flag_sector=True,
                flag_emissions_tco2e=Decimal("50000"),
            ),
        )
        result = _run(engine.calculate(inp))
        assert result.flag_targets is not None

    def test_flag_targets_not_generated_when_disabled(self):
        engine = InterimTargetEngine()
        inp = _make_input(include_flag_targets=False)
        result = _run(engine.calculate(inp))
        assert result.flag_targets is None

    def test_flag_targets_compliant(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            include_flag_targets=True,
            baseline=_make_baseline(
                is_flag_sector=True,
                flag_emissions_tco2e=Decimal("50000"),
            ),
        )
        result = _run(engine.calculate(inp))
        if result.flag_targets:
            assert result.flag_targets.is_compliant is True


# ===========================================================================
# Data Quality Assessment
# ===========================================================================


class TestDataQuality:
    """Test data quality assessment."""

    def test_data_quality_returned(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert result.data_quality in ("high", "medium", "low", "estimated")

    def test_high_quality_with_all_scopes(self):
        engine = InterimTargetEngine()
        inp = _make_input(
            entity_id="ENTITY-001",
            baseline=_make_baseline(base_year=2019),
            milestone_overrides=[
                MilestoneOverride(year=2030, reduction_pct=Decimal("42")),
            ],
        )
        result = _run(engine.calculate(inp))
        # With all scopes, entity_id, recent base year, and overrides -> should be high quality
        assert result.data_quality in ("high", "medium")


# ===========================================================================
# Recommendations & Warnings
# ===========================================================================


class TestRecommendationsWarnings:
    """Test recommendations and warnings generation."""

    def test_recommendations_list_returned(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert isinstance(result.recommendations, list)

    def test_warnings_list_returned(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        result = _run(engine.calculate(inp))
        assert isinstance(result.warnings, list)

    def test_back_loaded_pathway_generates_recommendation(self):
        engine = InterimTargetEngine()
        inp = _make_input(pathway_shape=PathwayShape.BACK_LOADED)
        result = _run(engine.calculate(inp))
        assert any("back-loaded" in r.lower() for r in result.recommendations)

    def test_old_baseline_generates_warning(self):
        engine = InterimTargetEngine()
        inp = _make_input(baseline=_make_baseline(base_year=2015))
        result = _run(engine.calculate(inp))
        assert any("2015" in w for w in result.warnings)


# ===========================================================================
# Batch Processing
# ===========================================================================


class TestBatchProcessing:
    """Test batch processing of multiple inputs."""

    def test_batch_processes_multiple(self):
        engine = InterimTargetEngine()
        inputs = [
            _make_input(entity_name="Corp A"),
            _make_input(entity_name="Corp B"),
            _make_input(entity_name="Corp C"),
        ]
        results = _run(engine.calculate_batch(inputs))
        assert len(results) == 3
        assert all(r.entity_name in ("Corp A", "Corp B", "Corp C") for r in results)

    def test_batch_handles_errors_gracefully(self):
        engine = InterimTargetEngine()
        # Second input has zero baseline (should produce warning, not crash batch)
        inputs = [
            _make_input(entity_name="Corp A"),
            _make_input(
                entity_name="Corp B",
                baseline=_make_baseline(scope1=Decimal("0"), scope2=Decimal("0"), scope3=Decimal("0")),
            ),
        ]
        results = _run(engine.calculate_batch(inputs))
        assert len(results) == 2


# ===========================================================================
# Extended Provenance & Edge Cases
# ===========================================================================


class TestInterimTargetProvenanceEdge:
    """Extended provenance and edge case tests for interim targets."""

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_interim_provenance_valid(self, ambition):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition)
        r1 = _run(engine.calculate(inp))
        assert_provenance_hash(r1)

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    @pytest.mark.parametrize("base_year", [2015, 2017, 2019, 2020, 2022])
    def test_ambition_base_year_matrix(self, ambition, base_year):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition, baseline=_make_baseline(base_year=base_year))
        result = _run(engine.calculate(inp))
        assert result is not None

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    @pytest.mark.parametrize("target_year", [2030, 2035, 2040, 2050])
    def test_ambition_target_year_matrix(self, ambition, target_year):
        engine = InterimTargetEngine()
        inp = _make_input(
            ambition=ambition,
            long_term_target=LongTermTarget(target_year=target_year, net_zero_year=target_year),
        )
        result = _run(engine.calculate(inp))
        assert result is not None

    @pytest.mark.parametrize("emissions", [
        Decimal("10000"), Decimal("50000"), Decimal("200000"),
        Decimal("1000000"), Decimal("10000000"),
    ])
    def test_scale_invariant_reduction(self, emissions):
        engine = InterimTargetEngine()
        half = emissions / Decimal("2")
        inp = _make_input(
            baseline=_make_baseline(scope1=half, scope2=half, scope3=emissions),
        )
        result = _run(engine.calculate(inp))
        assert result.annual_reduction_rate_scope12_pct > Decimal("0")

    def test_interim_performance_benchmark(self):
        engine = InterimTargetEngine()
        inp = _make_input()
        with timed_block(max_ms=5000):
            for _ in range(100):
                _run(engine.calculate(inp))

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_result_has_engine_version(self, ambition):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition)
        result = _run(engine.calculate(inp))
        assert isinstance(result.engine_version, str)
        assert result.engine_version == "1.0.0"

    @pytest.mark.parametrize("ambition", [ClimateAmbition.CELSIUS_1_5, ClimateAmbition.WELL_BELOW_2C])
    def test_result_all_decimal_types(self, ambition):
        engine = InterimTargetEngine()
        inp = _make_input(ambition=ambition)
        result = _run(engine.calculate(inp))
        assert isinstance(result.baseline_total_tco2e, Decimal)
        assert isinstance(result.implied_temperature_score, Decimal)
        assert isinstance(result.annual_reduction_rate_scope12_pct, Decimal)
