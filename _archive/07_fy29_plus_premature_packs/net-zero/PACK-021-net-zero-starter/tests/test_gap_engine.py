# -*- coding: utf-8 -*-
"""
Test suite for PACK-021 Net Zero Starter Pack - NetZeroGapEngine.

Validates gap-to-net-zero trajectory analysis including on-track, at-risk,
and off-track scenarios, per-scope gap breakdown, year-by-year gap, budget
remaining calculation, required acceleration, linear projection, and
provenance hashing.

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

from engines.net_zero_gap_engine import (
    BudgetAnalysis,
    GapInput,
    GapResult,
    GapSeverity,
    HistoricalDataPoint,
    NetZeroGapEngine,
    ProjectionMethod,
    RiskAssessment,
    ScopeEmissions,
    ScopeGap,
    TrajectoryStatus,
    YearlyGap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> NetZeroGapEngine:
    """Create a fresh engine instance."""
    return NetZeroGapEngine()


@pytest.fixture
def on_track_input() -> GapInput:
    """On-track scenario: current emissions below target pathway.

    Base year 2020: 10000 tCO2e, 4.2% annual rate.
    By 2025 (5 yrs): expected = 10000 * (1 - 0.042*5) = 10000 * 0.79 = 7900.
    Current = 7500 (below 7900) -> on-track.
    """
    return GapInput(
        entity_name="OnTrackCorp",
        current_year=2025,
        base_year=2020,
        base_year_total_tco2e=Decimal("10000"),
        current_total_tco2e=Decimal("7500"),
        target_total_tco2e=Decimal("1000"),
        target_year=2050,
        annual_rate_pct=Decimal("4.2"),
        scope_data=[
            ScopeEmissions(
                scope_label="scope_1",
                current_tco2e=Decimal("3000"),
                base_year_tco2e=Decimal("4000"),
                target_tco2e=Decimal("400"),
                target_year=2050,
                annual_rate_pct=Decimal("4.2"),
            ),
            ScopeEmissions(
                scope_label="scope_2",
                current_tco2e=Decimal("2000"),
                base_year_tco2e=Decimal("3000"),
                target_tco2e=Decimal("300"),
                target_year=2050,
                annual_rate_pct=Decimal("4.2"),
            ),
            ScopeEmissions(
                scope_label="scope_3",
                current_tco2e=Decimal("2500"),
                base_year_tco2e=Decimal("3000"),
                target_tco2e=Decimal("300"),
                target_year=2050,
                annual_rate_pct=Decimal("2.5"),
            ),
        ],
        historical_data=[
            HistoricalDataPoint(year=2020, total_tco2e=Decimal("10000")),
            HistoricalDataPoint(year=2021, total_tco2e=Decimal("9600")),
            HistoricalDataPoint(year=2022, total_tco2e=Decimal("9100")),
            HistoricalDataPoint(year=2023, total_tco2e=Decimal("8500")),
            HistoricalDataPoint(year=2024, total_tco2e=Decimal("8000")),
            HistoricalDataPoint(year=2025, total_tco2e=Decimal("7500")),
        ],
        projection_method=ProjectionMethod.LINEAR,
    )


@pytest.fixture
def at_risk_input() -> GapInput:
    """At-risk scenario: slightly above pathway (0-20% gap).

    Base year 2020: 10000 tCO2e, 4.2% annual rate.
    By 2025: expected = 7900. Current = 8600 -> gap = 700 / 7900 = ~8.9% -> at-risk.
    """
    return GapInput(
        entity_name="AtRiskCorp",
        current_year=2025,
        base_year=2020,
        base_year_total_tco2e=Decimal("10000"),
        current_total_tco2e=Decimal("8600"),
        target_total_tco2e=Decimal("1000"),
        target_year=2050,
        annual_rate_pct=Decimal("4.2"),
        projection_method=ProjectionMethod.LINEAR,
    )


@pytest.fixture
def off_track_input() -> GapInput:
    """Off-track scenario: significantly above pathway (>20% gap).

    Base year 2020: 10000 tCO2e, 4.2% annual rate.
    By 2025: expected = 7900. Current = 10500 -> gap = 2600 / 7900 = ~32.9% -> off-track.
    """
    return GapInput(
        entity_name="OffTrackCorp",
        current_year=2025,
        base_year=2020,
        base_year_total_tco2e=Decimal("10000"),
        current_total_tco2e=Decimal("10500"),
        target_total_tco2e=Decimal("1000"),
        target_year=2050,
        annual_rate_pct=Decimal("4.2"),
        projection_method=ProjectionMethod.LINEAR,
    )


@pytest.fixture
def perfect_alignment_input() -> GapInput:
    """Perfect alignment: current exactly on pathway."""
    return GapInput(
        entity_name="PerfectCorp",
        current_year=2025,
        base_year=2020,
        base_year_total_tco2e=Decimal("10000"),
        current_total_tco2e=Decimal("7900"),  # exactly on pathway
        target_total_tco2e=Decimal("1000"),
        target_year=2050,
        annual_rate_pct=Decimal("4.2"),
        projection_method=ProjectionMethod.LINEAR,
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self) -> None:
        """Engine must instantiate without arguments."""
        engine = NetZeroGapEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- On-Track Scenario
# ===========================================================================


class TestOnTrackScenario:
    """Tests for on-track trajectory assessment."""

    def test_on_track_scenario(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Current below pathway must yield on-track status."""
        result = engine.calculate(on_track_input)

        assert isinstance(result, GapResult)
        # Gap should be negative (below pathway is good)
        assert result.current_gap_tco2e < Decimal("0")
        assert result.trajectory_status in (
            TrajectoryStatus.ON_TRACK.value,
            TrajectoryStatus.ACHIEVED.value,
            "on_track", "achieved",
        )

    def test_on_track_gap_percentage_negative(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """On-track gap percentage must be negative or zero."""
        result = engine.calculate(on_track_input)
        assert result.current_gap_pct <= Decimal("0")


# ===========================================================================
# Tests -- At-Risk Scenario
# ===========================================================================


class TestAtRiskScenario:
    """Tests for at-risk trajectory assessment."""

    def test_at_risk_scenario(
        self, engine: NetZeroGapEngine, at_risk_input: GapInput
    ) -> None:
        """Emissions slightly above pathway (0-20%) must yield at-risk status."""
        result = engine.calculate(at_risk_input)

        assert result.current_gap_tco2e > Decimal("0")
        assert result.trajectory_status in (
            TrajectoryStatus.AT_RISK.value,
            "at_risk",
        )

    def test_at_risk_gap_under_20_pct(
        self, engine: NetZeroGapEngine, at_risk_input: GapInput
    ) -> None:
        """At-risk gap percentage must be 0-20%."""
        result = engine.calculate(at_risk_input)
        assert Decimal("0") < result.current_gap_pct <= Decimal("20")


# ===========================================================================
# Tests -- Off-Track Scenario
# ===========================================================================


class TestOffTrackScenario:
    """Tests for off-track trajectory assessment."""

    def test_off_track_scenario(
        self, engine: NetZeroGapEngine, off_track_input: GapInput
    ) -> None:
        """Emissions >20% above pathway must yield off-track status."""
        result = engine.calculate(off_track_input)

        assert result.current_gap_tco2e > Decimal("0")
        assert result.trajectory_status in (
            TrajectoryStatus.OFF_TRACK.value,
            "off_track",
        )

    def test_off_track_gap_above_20_pct(
        self, engine: NetZeroGapEngine, off_track_input: GapInput
    ) -> None:
        """Off-track gap percentage must be above 20%."""
        result = engine.calculate(off_track_input)
        assert result.current_gap_pct > Decimal("20")


# ===========================================================================
# Tests -- Per-Scope Gap
# ===========================================================================


class TestScopeGap:
    """Tests for per-scope gap analysis."""

    def test_gap_by_scope(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """gap_by_scope must contain entries for each provided scope."""
        result = engine.calculate(on_track_input)
        assert len(result.gap_by_scope) == 3  # scope_1, scope_2, scope_3

    def test_scope_gap_labels(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Scope gap labels must match input scope labels."""
        result = engine.calculate(on_track_input)
        labels = {sg.scope_label for sg in result.gap_by_scope}
        assert "scope_1" in labels
        assert "scope_2" in labels
        assert "scope_3" in labels

    def test_scope_gap_has_acceleration(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Each scope gap must have a required_acceleration factor."""
        result = engine.calculate(on_track_input)
        for sg in result.gap_by_scope:
            assert isinstance(sg, ScopeGap)
            assert sg.required_acceleration is not None


# ===========================================================================
# Tests -- Year-by-Year Gap
# ===========================================================================


class TestYearlyGap:
    """Tests for year-by-year gap analysis."""

    def test_gap_by_year(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """gap_by_year must span from base year to target year."""
        result = engine.calculate(on_track_input)
        years = [g.year for g in result.gap_by_year]
        assert min(years) == on_track_input.base_year
        assert max(years) == on_track_input.target_year

    def test_gap_by_year_expected_decreasing(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Expected emissions on the pathway should decrease year over year."""
        result = engine.calculate(on_track_input)
        sorted_gaps = sorted(result.gap_by_year, key=lambda g: g.year)
        for i in range(1, len(sorted_gaps)):
            assert sorted_gaps[i].expected_tco2e <= sorted_gaps[i-1].expected_tco2e

    def test_gap_by_year_has_status(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Each yearly gap entry must have a status field."""
        result = engine.calculate(on_track_input)
        for yg in result.gap_by_year:
            assert isinstance(yg, YearlyGap)
            assert yg.status  # non-empty


# ===========================================================================
# Tests -- Budget Analysis
# ===========================================================================


class TestBudgetAnalysis:
    """Tests for cumulative emissions budget analysis."""

    def test_budget_remaining_calculation(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Budget analysis must have pathway_budget and years_remaining."""
        result = engine.calculate(on_track_input)
        budget = result.budget_analysis
        assert isinstance(budget, BudgetAnalysis)
        assert budget.pathway_budget_tco2e > Decimal("0")
        assert budget.years_remaining > 0

    def test_budget_gap_sign(
        self, engine: NetZeroGapEngine, off_track_input: GapInput
    ) -> None:
        """Off-track scenario must have positive budget gap (overshoot)."""
        result = engine.calculate(off_track_input)
        budget = result.budget_analysis
        assert budget.budget_gap_tco2e > Decimal("0")

    def test_budget_remaining_positive(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """On-track scenario should still have remaining budget."""
        result = engine.calculate(on_track_input)
        budget = result.budget_analysis
        assert budget.budget_remaining_tco2e > Decimal("0")


# ===========================================================================
# Tests -- Required Acceleration
# ===========================================================================


class TestRequiredAcceleration:
    """Tests for required acceleration rate calculation."""

    def test_required_acceleration_off_track(
        self, engine: NetZeroGapEngine, off_track_input: GapInput
    ) -> None:
        """Off-track scenario must have acceleration factor >= 1.

        When emissions have increased (negative actual reduction), the engine
        defaults acceleration_factor to 1.0 because a meaningful ratio
        cannot be computed from a negative base rate.
        """
        result = engine.calculate(off_track_input)
        risk = result.risk_assessment
        assert isinstance(risk, RiskAssessment)
        assert risk.acceleration_factor >= Decimal("1.0")

    def test_required_acceleration_on_track(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """On-track scenario must have acceleration factor <= 1."""
        result = engine.calculate(on_track_input)
        risk = result.risk_assessment
        assert risk.acceleration_factor <= Decimal("1.0")


# ===========================================================================
# Tests -- Linear Projection
# ===========================================================================


class TestLinearProjection:
    """Tests for linear emission projection."""

    def test_linear_projection(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Linear projection must extend actual emissions into future years."""
        result = engine.calculate(on_track_input)
        # Future years (2026-2050) should have projected actual emissions
        future_gaps = [
            g for g in result.gap_by_year if g.year > on_track_input.current_year
        ]
        assert len(future_gaps) > 0

    def test_projection_method_in_result(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Projection method must be recorded in result."""
        result = engine.calculate(on_track_input)
        assert result.projection_method == "linear"


# ===========================================================================
# Tests -- Trajectory Status Determination
# ===========================================================================


class TestTrajectoryStatus:
    """Tests for trajectory status determination logic."""

    def test_trajectory_status_determination(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Trajectory status must be a recognized status value."""
        result = engine.calculate(on_track_input)
        valid_statuses = {
            "on_track", "at_risk", "off_track", "achieved", "expired"
        }
        assert result.trajectory_status in valid_statuses

    @pytest.mark.parametrize("current_emissions,expected_status_group", [
        (Decimal("7000"), {"on_track", "achieved"}),
        (Decimal("8600"), {"at_risk"}),
        (Decimal("10500"), {"off_track"}),
    ])
    def test_status_by_emission_level(
        self,
        engine: NetZeroGapEngine,
        current_emissions: Decimal,
        expected_status_group: set,
    ) -> None:
        """Different emission levels must yield the correct RAG status."""
        inp = GapInput(
            entity_name="StatusTest",
            current_year=2025,
            base_year=2020,
            base_year_total_tco2e=Decimal("10000"),
            current_total_tco2e=current_emissions,
            target_total_tco2e=Decimal("1000"),
            target_year=2050,
            annual_rate_pct=Decimal("4.2"),
        )
        result = engine.calculate(inp)
        assert result.trajectory_status in expected_status_group


# ===========================================================================
# Tests -- Provenance & Edge Cases
# ===========================================================================


class TestProvenanceAndEdgeCases:
    """Tests for provenance hashing and edge cases."""

    def test_provenance_hash(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Result must have a non-empty 64-character SHA-256 hash."""
        result = engine.calculate(on_track_input)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_hex_string(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Provenance hash must be a valid 64-character hex string.

        Note: The hash includes result_id (UUID4) which changes per call,
        so deterministic equality is not expected across separate calls.
        """
        r1 = engine.calculate(on_track_input)
        r2 = engine.calculate(on_track_input)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)

    def test_zero_gap_perfect_alignment(
        self, engine: NetZeroGapEngine, perfect_alignment_input: GapInput
    ) -> None:
        """When current exactly matches expected pathway, gap should be ~0."""
        result = engine.calculate(perfect_alignment_input)
        assert abs(float(result.current_gap_tco2e)) < 1.0

    def test_processing_time_recorded(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """processing_time_ms must be positive."""
        result = engine.calculate(on_track_input)
        assert result.processing_time_ms > 0

    def test_entity_name_propagated(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Entity name must be propagated to result."""
        result = engine.calculate(on_track_input)
        assert result.entity_name == "OnTrackCorp"

    def test_result_years_match_input(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Result base_year, current_year, target_year must match input."""
        result = engine.calculate(on_track_input)
        assert result.base_year == 2020
        assert result.current_year == 2025
        assert result.target_year == 2050

    def test_no_scope_data_still_works(self, engine: NetZeroGapEngine) -> None:
        """Gap analysis without per-scope data must still produce a result."""
        inp = GapInput(
            entity_name="NoScopeCorp",
            current_year=2025,
            base_year=2020,
            base_year_total_tco2e=Decimal("10000"),
            current_total_tco2e=Decimal("8000"),
            target_total_tco2e=Decimal("1000"),
            target_year=2050,
            annual_rate_pct=Decimal("4.2"),
        )
        result = engine.calculate(inp)
        assert isinstance(result, GapResult)
        assert len(result.gap_by_scope) == 0  # no scope data provided

    def test_risk_assessment_populated(
        self, engine: NetZeroGapEngine, on_track_input: GapInput
    ) -> None:
        """Risk assessment must be populated with relevant fields."""
        result = engine.calculate(on_track_input)
        risk = result.risk_assessment
        assert risk.overall_status
        assert risk.severity
        assert risk.required_annual_reduction_tco2e >= Decimal("0")


# ===========================================================================
# Tests -- Enum Values
# ===========================================================================


class TestEnumValues:
    """Tests for gap engine enum value integrity."""

    def test_trajectory_status_values(self) -> None:
        """TrajectoryStatus must have all expected members."""
        assert TrajectoryStatus.ON_TRACK.value == "on_track"
        assert TrajectoryStatus.AT_RISK.value == "at_risk"
        assert TrajectoryStatus.OFF_TRACK.value == "off_track"
        assert TrajectoryStatus.ACHIEVED.value == "achieved"
        assert TrajectoryStatus.EXPIRED.value == "expired"

    def test_gap_severity_values(self) -> None:
        """GapSeverity must have all expected members."""
        assert GapSeverity.NONE.value == "none"
        assert GapSeverity.MINOR.value == "minor"
        assert GapSeverity.MODERATE.value == "moderate"
        assert GapSeverity.SIGNIFICANT.value == "significant"
        assert GapSeverity.CRITICAL.value == "critical"

    def test_projection_method_values(self) -> None:
        """ProjectionMethod must have all expected members."""
        assert ProjectionMethod.LINEAR.value == "linear"
        assert ProjectionMethod.FLAT.value == "flat"
        assert ProjectionMethod.COMPOUND.value == "compound"
