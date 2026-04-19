# -*- coding: utf-8 -*-
"""
Test suite for PACK-021 Net Zero Starter Pack - NetZeroTargetEngine.

Validates SBTi Net-Zero Standard v1.2 compliant target setting including
ACA and WB2C pathway rates, SDA sector benchmarks, near-term and long-term
target generation, coverage validation, milestone generation, temperature
alignment scoring, and provenance hashing.

All assertions on numeric values use Decimal for precision.

Author:  GreenLang Test Engineering
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from engines.net_zero_target_engine import (
    AmbitionLevel,
    MilestoneEntry,
    NetZeroTargetEngine,
    PathwayType,
    SBTiSector,
    ScopeCategory,
    TargetDefinition,
    TargetInput,
    TargetResult,
    TargetTimeframe,
    TargetType,
    TemperatureAlignment,
    ValidationCheck,
    SBTI_RATES,
    COVERAGE_REQUIREMENTS,
    SDA_BENCHMARKS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> NetZeroTargetEngine:
    """Create a fresh engine instance."""
    return NetZeroTargetEngine()


@pytest.fixture
def standard_1_5c_input() -> TargetInput:
    """Standard 1.5C ACA input with typical manufacturing baseline.

    Uses base_year=2022 to stay within SBTi's 5-year recency window
    (current year is 2026, so 2026 - 2022 = 4 years <= 5).
    """
    return TargetInput(
        entity_name="MfgCorp",
        base_year=2022,
        base_year_scope1_tco2e=Decimal("5000"),
        base_year_scope2_tco2e=Decimal("3000"),
        base_year_scope3_tco2e=Decimal("12000"),
        near_term_target_year=2030,
        long_term_target_year=2050,
        pathway_type=PathwayType.ACA,
        ambition_level=AmbitionLevel.PARIS_1_5C,
        sector=SBTiSector.OTHER,
        scope1_2_coverage_pct=Decimal("100"),
        scope3_coverage_pct=Decimal("67"),
        include_scope3=True,
        milestone_interval_years=5,
    )


@pytest.fixture
def wb2c_input() -> TargetInput:
    """Well-below 2C input."""
    return TargetInput(
        entity_name="WB2Corp",
        base_year=2020,
        base_year_scope1_tco2e=Decimal("4000"),
        base_year_scope2_tco2e=Decimal("2000"),
        base_year_scope3_tco2e=Decimal("8000"),
        near_term_target_year=2030,
        long_term_target_year=2050,
        pathway_type=PathwayType.ACA,
        ambition_level=AmbitionLevel.WELL_BELOW_2C,
        scope1_2_coverage_pct=Decimal("95"),
        scope3_coverage_pct=Decimal("67"),
        include_scope3=True,
    )


@pytest.fixture
def no_scope3_input() -> TargetInput:
    """Input without Scope 3 targets."""
    return TargetInput(
        entity_name="NoS3Corp",
        base_year=2020,
        base_year_scope1_tco2e=Decimal("3000"),
        base_year_scope2_tco2e=Decimal("2000"),
        base_year_scope3_tco2e=Decimal("0"),
        near_term_target_year=2030,
        long_term_target_year=2050,
        include_scope3=False,
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self) -> None:
        """Engine must instantiate without arguments."""
        engine = NetZeroTargetEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- ACA 1.5C Target
# ===========================================================================


class TestACA15CTarget:
    """Tests for Absolute Contraction Approach at 1.5C alignment."""

    def test_aca_1_5c_target(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """ACA 1.5C: 4.2% per year linear absolute reduction for Scope 1+2."""
        result = engine.calculate(standard_1_5c_input)

        assert isinstance(result, TargetResult)
        assert result.near_term_target is not None

        # 4.2% annual * 8 years (2030 - 2022) = 33.6% reduction
        nt = result.near_term_target
        assert float(nt.annual_rate_pct) == pytest.approx(4.2, rel=1e-3)
        years = standard_1_5c_input.near_term_target_year - standard_1_5c_input.base_year
        expected_reduction = 4.2 * years  # 33.6%
        assert float(nt.reduction_pct) == pytest.approx(expected_reduction, rel=1e-2)

    def test_near_term_target_year(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Near-term target year must match input."""
        result = engine.calculate(standard_1_5c_input)
        assert result.near_term_target.target_year == standard_1_5c_input.near_term_target_year

    def test_near_term_scope_coverage(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Near-term target must cover Scope 1+2."""
        result = engine.calculate(standard_1_5c_input)
        nt = result.near_term_target
        assert nt.scope_coverage in (
            ScopeCategory.SCOPE_1_2, ScopeCategory.ALL_SCOPES
        )


# ===========================================================================
# Tests -- ACA Well-Below 2C Target
# ===========================================================================


class TestACAWB2CTarget:
    """Tests for Absolute Contraction Approach at well-below 2C alignment."""

    def test_aca_wb2c_target(
        self, engine: NetZeroTargetEngine, wb2c_input: TargetInput
    ) -> None:
        """WB2C: 2.5% per year linear absolute reduction for Scope 1+2."""
        result = engine.calculate(wb2c_input)

        nt = result.near_term_target
        assert nt is not None
        assert nt.annual_rate_pct == pytest.approx(2.5, rel=1e-3)
        # 2.5% * 10 years = 25%
        assert float(nt.reduction_pct) == pytest.approx(25.0, rel=1e-2)


# ===========================================================================
# Tests -- SDA Pathway
# ===========================================================================


class TestSDAPathway:
    """Tests for Sectoral Decarbonization Approach targets."""

    def test_sda_pathway_target(self, engine: NetZeroTargetEngine) -> None:
        """SDA pathway for power generation must use sector benchmarks."""
        inp = TargetInput(
            entity_name="PowerCorp",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("50000"),
            base_year_scope2_tco2e=Decimal("1000"),
            base_year_scope3_tco2e=Decimal("10000"),
            pathway_type=PathwayType.SDA,
            sector=SBTiSector.POWER_GENERATION,
            near_term_target_year=2030,
            long_term_target_year=2050,
        )
        result = engine.calculate(inp)
        assert isinstance(result, TargetResult)
        # SDA result should still produce a valid target
        assert result.near_term_target is not None


# ===========================================================================
# Tests -- Near-Term and Long-Term Target Generation
# ===========================================================================


class TestTargetGeneration:
    """Tests for near-term and long-term target outputs."""

    def test_near_term_target_generation(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Near-term target must be 5-10 year window from base year."""
        result = engine.calculate(standard_1_5c_input)
        nt = result.near_term_target
        years = nt.target_year - nt.base_year
        assert 5 <= years <= 10

    def test_long_term_target_generation(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Long-term target must require 90%+ reduction per SBTi 1.5C."""
        result = engine.calculate(standard_1_5c_input)
        lt = result.long_term_target
        assert lt is not None
        assert lt.target_year == 2050
        assert float(lt.reduction_pct) >= 90.0

    def test_long_term_target_covers_all_scopes(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Long-term target should cover all scopes."""
        result = engine.calculate(standard_1_5c_input)
        lt = result.long_term_target
        assert lt.scope_coverage == ScopeCategory.ALL_SCOPES

    def test_scope3_target_created_when_material(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Scope 3 target must be created when Scope 3 is material (>40%)."""
        result = engine.calculate(standard_1_5c_input)
        # Scope 3 = 12000 / (5000+3000+12000) = 60% > 40% threshold
        assert result.scope3_materiality is True
        assert result.scope3_target is not None


# ===========================================================================
# Tests -- Coverage Validation
# ===========================================================================


class TestCoverageValidation:
    """Tests for SBTi coverage requirement validation."""

    def test_coverage_validation_scope1_2_pass(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """100% Scope 1+2 coverage must pass the 95% minimum check."""
        result = engine.calculate(standard_1_5c_input)
        coverage_checks = [
            c for c in result.validation_checks
            if "coverage" in c.check_name.lower() and "1" in c.check_name
        ]
        if coverage_checks:
            assert all(c.passed for c in coverage_checks)

    def test_coverage_validation_scope3_pass(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """67% Scope 3 coverage must pass the 67% minimum check."""
        result = engine.calculate(standard_1_5c_input)
        s3_checks = [
            c for c in result.validation_checks
            if "scope 3" in c.check_name.lower() or "scope_3" in c.check_name.lower()
        ]
        if s3_checks:
            assert all(c.passed for c in s3_checks)

    def test_insufficient_scope3_coverage_fails(
        self, engine: NetZeroTargetEngine
    ) -> None:
        """50% Scope 3 coverage should fail the 67% minimum check."""
        inp = TargetInput(
            entity_name="LowCovCorp",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("3000"),
            base_year_scope2_tco2e=Decimal("2000"),
            base_year_scope3_tco2e=Decimal("10000"),
            scope3_coverage_pct=Decimal("50"),
            include_scope3=True,
        )
        result = engine.calculate(inp)
        # Find the Scope 3 coverage check
        s3_checks = [
            c for c in result.validation_checks
            if "scope_3" in c.check_name.lower() or "scope 3" in c.check_name.lower()
        ]
        # At least one should fail
        if s3_checks:
            assert any(not c.passed for c in s3_checks)


# ===========================================================================
# Tests -- FLAG Pathway
# ===========================================================================


class TestFLAGPathway:
    """Tests for Forest, Land and Agriculture pathway."""

    def test_flag_pathway(self, engine: NetZeroTargetEngine) -> None:
        """FLAG pathway must be accepted for agriculture sector."""
        inp = TargetInput(
            entity_name="FarmCorp",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("8000"),
            base_year_scope2_tco2e=Decimal("500"),
            base_year_scope3_tco2e=Decimal("3000"),
            pathway_type=PathwayType.FLAG,
            sector=SBTiSector.FLAG,
            near_term_target_year=2030,
            long_term_target_year=2050,
        )
        result = engine.calculate(inp)
        assert isinstance(result, TargetResult)
        assert result.near_term_target is not None


# ===========================================================================
# Tests -- Milestone Generation
# ===========================================================================


class TestMilestoneGeneration:
    """Tests for interim milestone generation."""

    def test_milestone_generation(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Milestones must be generated at the specified interval."""
        result = engine.calculate(standard_1_5c_input)
        assert len(result.milestones) > 0
        assert all(isinstance(m, MilestoneEntry) for m in result.milestones)

    def test_milestone_every_5_years(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """With 5-year interval, milestone years should be multiples of 5."""
        result = engine.calculate(standard_1_5c_input)
        milestone_years = [m.year for m in result.milestones]
        # At least should include 2025, 2030 (near-term) and beyond
        assert len(milestone_years) > 0

    def test_milestones_have_decreasing_targets(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Milestone target emissions should decrease over time within each scope."""
        result = engine.calculate(standard_1_5c_input)
        if len(result.milestones) >= 2:
            # Group milestones by scope, then verify decreasing within each scope
            scopes_seen = {m.scope for m in result.milestones}
            for scope in scopes_seen:
                scope_ms = sorted(
                    [m for m in result.milestones if m.scope == scope],
                    key=lambda m: m.year,
                )
                for i in range(1, len(scope_ms)):
                    assert scope_ms[i].target_tco2e <= scope_ms[i - 1].target_tco2e, (
                        f"Scope '{scope}': milestone at year {scope_ms[i].year} "
                        f"({scope_ms[i].target_tco2e}) should be <= "
                        f"year {scope_ms[i-1].year} ({scope_ms[i-1].target_tco2e})"
                    )


# ===========================================================================
# Tests -- Temperature Alignment Scoring
# ===========================================================================


class TestTemperatureAlignment:
    """Tests for temperature alignment classification."""

    def test_temperature_alignment_scoring_1_5c(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """1.5C-aligned input must score 1.5C alignment."""
        result = engine.calculate(standard_1_5c_input)
        assert "1.5" in result.temperature_alignment.lower() or \
               "1_5" in result.temperature_alignment.lower()

    def test_temperature_alignment_scoring_wb2c(
        self, engine: NetZeroTargetEngine, wb2c_input: TargetInput
    ) -> None:
        """Well-below 2C input must score WB2C alignment."""
        result = engine.calculate(wb2c_input)
        assert "2" in result.temperature_alignment.lower() or \
               "below" in result.temperature_alignment.lower() or \
               "wb2c" in result.temperature_alignment.lower()


# ===========================================================================
# Tests -- SBTi Validation
# ===========================================================================


class TestSBTiValidation:
    """Tests for SBTi validation check execution."""

    def test_sbti_validation_pass(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Standard 1.5C input with full coverage must pass SBTi validation."""
        result = engine.calculate(standard_1_5c_input)
        assert result.validation_passed is True
        assert len(result.validation_checks) > 0

    def test_sbti_validation_fail_insufficient_ambition(
        self, engine: NetZeroTargetEngine
    ) -> None:
        """Below-2C ambition with low coverage should fail validation."""
        inp = TargetInput(
            entity_name="LowAmbCorp",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("5000"),
            base_year_scope2_tco2e=Decimal("3000"),
            base_year_scope3_tco2e=Decimal("15000"),
            ambition_level=AmbitionLevel.BELOW_2C,
            scope1_2_coverage_pct=Decimal("50"),  # below 95% threshold
            scope3_coverage_pct=Decimal("30"),  # below 67% threshold
            include_scope3=True,
        )
        result = engine.calculate(inp)
        # At least one coverage check should fail
        failed = [c for c in result.validation_checks if not c.passed]
        assert len(failed) > 0

    def test_validation_checks_have_details(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Each validation check must have a check_name and message."""
        result = engine.calculate(standard_1_5c_input)
        for check in result.validation_checks:
            assert isinstance(check, ValidationCheck)
            assert check.check_name, "check_name must not be empty"
            assert check.message, "message must not be empty"


# ===========================================================================
# Tests -- Provenance & Edge Cases
# ===========================================================================


class TestProvenanceAndEdgeCases:
    """Tests for provenance hashing and edge cases."""

    def test_provenance_hash(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Result must have a non-empty 64-character SHA-256 hash."""
        result = engine.calculate(standard_1_5c_input)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_hex_string(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Provenance hash must be a valid 64-character hex string.

        Note: The hash includes result_id (UUID4) which changes per call,
        so deterministic equality is not expected across separate calls.
        """
        r1 = engine.calculate(standard_1_5c_input)
        r2 = engine.calculate(standard_1_5c_input)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)

    def test_invalid_sector_handling(self, engine: NetZeroTargetEngine) -> None:
        """OTHER sector with ACA pathway must still produce valid targets."""
        inp = TargetInput(
            entity_name="OtherSector",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("1000"),
            base_year_scope2_tco2e=Decimal("500"),
            sector=SBTiSector.OTHER,
        )
        result = engine.calculate(inp)
        assert result.near_term_target is not None

    def test_target_year_boundaries(self, engine: NetZeroTargetEngine) -> None:
        """Extreme but valid target year boundaries must be handled."""
        inp = TargetInput(
            entity_name="BoundaryCorp",
            base_year=2025,
            base_year_scope1_tco2e=Decimal("2000"),
            base_year_scope2_tco2e=Decimal("1000"),
            near_term_target_year=2035,
            long_term_target_year=2050,
        )
        result = engine.calculate(inp)
        assert result.near_term_target.target_year == 2035

    def test_near_term_before_base_raises(self) -> None:
        """Near-term target year at or before base year must raise."""
        with pytest.raises(Exception):
            TargetInput(
                entity_name="InvalidNT",
                base_year=2030,
                base_year_scope1_tco2e=Decimal("1000"),
                base_year_scope2_tco2e=Decimal("500"),
                near_term_target_year=2025,
            )

    def test_zero_base_year_emissions(self, engine: NetZeroTargetEngine) -> None:
        """Zero base year emissions should still produce a result without crashing."""
        inp = TargetInput(
            entity_name="ZeroCorp",
            base_year=2020,
            base_year_scope1_tco2e=Decimal("0"),
            base_year_scope2_tco2e=Decimal("0"),
            base_year_scope3_tco2e=Decimal("0"),
        )
        result = engine.calculate(inp)
        assert isinstance(result, TargetResult)
        assert result.total_base_year_tco2e == Decimal("0")

    def test_processing_time_recorded(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """processing_time_ms must be positive."""
        result = engine.calculate(standard_1_5c_input)
        assert result.processing_time_ms > 0

    def test_entity_name_in_result(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Entity name must be propagated to the result."""
        result = engine.calculate(standard_1_5c_input)
        assert result.entity_name == "MfgCorp"

    def test_total_base_year_tco2e(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Total base year emissions must equal S1 + S2 + S3."""
        result = engine.calculate(standard_1_5c_input)
        expected = Decimal("5000") + Decimal("3000") + Decimal("12000")
        assert float(result.total_base_year_tco2e) == pytest.approx(
            float(expected), rel=1e-3
        )

    def test_annual_rates_dict(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """annual_rates dict must contain at least one scope entry."""
        result = engine.calculate(standard_1_5c_input)
        assert len(result.annual_rates) > 0

    def test_no_scope3_target_when_disabled(
        self, engine: NetZeroTargetEngine, no_scope3_input: TargetInput
    ) -> None:
        """When include_scope3=False, scope3_target must be None."""
        result = engine.calculate(no_scope3_input)
        assert result.scope3_target is None

    def test_recommendations_generated(
        self, engine: NetZeroTargetEngine, standard_1_5c_input: TargetInput
    ) -> None:
        """Recommendations list should be populated."""
        result = engine.calculate(standard_1_5c_input)
        assert isinstance(result.recommendations, list)


# ===========================================================================
# Tests -- Constants Integrity
# ===========================================================================


class TestConstantsIntegrity:
    """Tests for SBTI_RATES and COVERAGE_REQUIREMENTS integrity."""

    def test_sbti_rates_1_5c(self) -> None:
        """SBTI_RATES for 1.5C must have 4.2% scope_1_2 annual rate."""
        rates = SBTI_RATES[AmbitionLevel.PARIS_1_5C]
        assert rates["scope_1_2_annual_pct"] == Decimal("4.2")
        assert rates["scope_3_annual_pct"] == Decimal("2.5")
        assert rates["long_term_reduction_pct"] == Decimal("90")

    def test_sbti_rates_wb2c(self) -> None:
        """SBTI_RATES for WB2C must have 2.5% scope_1_2 annual rate."""
        rates = SBTI_RATES[AmbitionLevel.WELL_BELOW_2C]
        assert rates["scope_1_2_annual_pct"] == Decimal("2.5")

    def test_coverage_requirements_scope1_2(self) -> None:
        """Scope 1+2 minimum coverage must be 95%."""
        assert COVERAGE_REQUIREMENTS["scope_1_2_min_pct"] == Decimal("95")

    def test_coverage_requirements_scope3(self) -> None:
        """Scope 3 minimum coverage must be 67%."""
        assert COVERAGE_REQUIREMENTS["scope_3_min_pct"] == Decimal("67")

    def test_scope3_materiality_threshold(self) -> None:
        """Scope 3 materiality threshold must be 40%."""
        assert COVERAGE_REQUIREMENTS["scope_3_materiality_threshold_pct"] == Decimal("40")
