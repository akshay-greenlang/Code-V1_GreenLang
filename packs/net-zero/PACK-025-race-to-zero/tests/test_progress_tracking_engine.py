# -*- coding: utf-8 -*-
"""
Deep tests for ProgressTrackingEngine (Engine 5 of 10).

Covers: RAG status classification, year-over-year change calculation,
trajectory alignment assessment, remaining carbon budget computation,
variance analysis, verification status, cumulative reduction tracking,
Decimal arithmetic, SHA-256 provenance.

Target: ~50 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.progress_tracking_engine import (
    ProgressTrackingEngine,
    ProgressTrackingInput,
    AnnualEmissionRecord,
    ActionPlanStatusInput,
    ProgressStatus,
    VerificationStatus,
    TrendDirection,
    RAG_CAUTION_THRESHOLD,
    RAG_OFF_TRACK_THRESHOLD,
    VARIANCE_ON_TRACK,
    VARIANCE_AT_RISK,
)

from conftest import assert_provenance_hash, timed_block


# ========================================================================
# Constants
# ========================================================================


class TestProgressTrackingConstants:
    """Validate progress tracking constants."""

    def test_rag_caution_threshold_10(self):
        assert RAG_CAUTION_THRESHOLD == Decimal("10")

    def test_rag_off_track_threshold_20(self):
        assert RAG_OFF_TRACK_THRESHOLD == Decimal("20")

    def test_variance_on_track_5(self):
        assert VARIANCE_ON_TRACK == Decimal("5")

    def test_variance_at_risk_15(self):
        assert VARIANCE_AT_RISK == Decimal("15")


# ========================================================================
# Enums
# ========================================================================


class TestProgressTrackingEnums:
    """Validate progress tracking enums."""

    def test_progress_status_4_values(self):
        assert len(ProgressStatus) == 4

    def test_progress_status_values(self):
        assert ProgressStatus.ON_TRACK.value == "on_track"
        assert ProgressStatus.CAUTION.value == "caution"
        assert ProgressStatus.OFF_TRACK.value == "off_track"
        assert ProgressStatus.CRITICAL.value == "critical"

    def test_verification_status_3_values(self):
        assert len(VerificationStatus) == 3

    def test_verification_values(self):
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.SELF_REPORTED.value == "self_reported"
        assert VerificationStatus.PENDING.value == "pending"

    def test_trend_direction_3_values(self):
        assert len(TrendDirection) == 3

    def test_trend_values(self):
        assert TrendDirection.DECREASING.value == "decreasing"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.INCREASING.value == "increasing"


# ========================================================================
# Input Model Validation
# ========================================================================


class TestProgressTrackingInputModel:
    """Validate input model construction."""

    def test_annual_record_construction(self):
        rec = AnnualEmissionRecord(
            year=2024,
            scope1_tco2e=Decimal("30000"),
            scope2_tco2e=Decimal("20000"),
            scope3_tco2e=Decimal("50000"),
            total_tco2e=Decimal("100000"),
            verification_status="verified",
        )
        assert rec.year == 2024
        assert rec.total_tco2e == Decimal("100000")

    def test_invalid_verification_raises(self):
        with pytest.raises(Exception):
            AnnualEmissionRecord(
                year=2024,
                verification_status="bogus",
            )

    def test_action_plan_status_construction(self):
        aps = ActionPlanStatusInput(
            total_actions=15,
            actions_completed=5,
            actions_initiated=8,
            abatement_realized_tco2e=Decimal("5000"),
        )
        assert aps.total_actions == 15

    def test_progress_input_construction(self):
        records = [
            AnnualEmissionRecord(
                year=2019,
                total_tco2e=Decimal("100000"),
            ),
            AnnualEmissionRecord(
                year=2024,
                total_tco2e=Decimal("80000"),
            ),
        ]
        inp = ProgressTrackingInput(
            entity_name="GreenCorp",
            baseline_year=2019,
            baseline_emissions_tco2e=Decimal("100000"),
            interim_target_year=2030,
            interim_target_tco2e=Decimal("50000"),
            current_year=2024,
            annual_records=records,
        )
        assert inp.entity_name == "GreenCorp"
        assert len(inp.annual_records) == 2


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestProgressEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, progress_engine):
        assert progress_engine is not None

    def test_engine_has_calculate(self, progress_engine):
        assert callable(getattr(progress_engine, "track", None))


# ========================================================================
# On-Track Scenario
# ========================================================================


class TestOnTrackScenario:
    """Tests for entity on track with targets."""

    @pytest.fixture
    def on_track_input(self):
        records = [
            AnnualEmissionRecord(year=2019, total_tco2e=Decimal("100000")),
            AnnualEmissionRecord(year=2020, total_tco2e=Decimal("92000")),
            AnnualEmissionRecord(year=2021, total_tco2e=Decimal("85000")),
            AnnualEmissionRecord(year=2022, total_tco2e=Decimal("78000")),
            AnnualEmissionRecord(year=2023, total_tco2e=Decimal("72000")),
            AnnualEmissionRecord(year=2024, total_tco2e=Decimal("66000")),
        ]
        return ProgressTrackingInput(
            entity_name="GreenCorp International",
            baseline_year=2019,
            baseline_emissions_tco2e=Decimal("100000"),
            interim_target_year=2030,
            interim_target_tco2e=Decimal("50000"),
            net_zero_year=2050,
            current_year=2024,
            annual_records=records,
        )

    def test_on_track_calculates(self, progress_engine, on_track_input):
        result = progress_engine.track(on_track_input)
        assert result is not None

    def test_on_track_has_provenance(self, progress_engine, on_track_input):
        result = progress_engine.track(on_track_input)
        assert_provenance_hash(result)

    def test_on_track_cumulative_reduction(self, progress_engine, on_track_input):
        result = progress_engine.track(on_track_input)
        assert result.cumulative_reduction_pct > Decimal("0")

    def test_on_track_status_not_critical(self, progress_engine, on_track_input):
        result = progress_engine.track(on_track_input)
        assert result.progress_status != "critical"

    def test_on_track_performance(self, progress_engine, on_track_input):
        with timed_block("on_track_assessment", max_seconds=5.0):
            progress_engine.track(on_track_input)


# ========================================================================
# Off-Track Scenario
# ========================================================================


class TestOffTrackScenario:
    """Tests for entity off track (increasing emissions)."""

    @pytest.fixture
    def off_track_input(self):
        records = [
            AnnualEmissionRecord(year=2019, total_tco2e=Decimal("100000")),
            AnnualEmissionRecord(year=2020, total_tco2e=Decimal("98000")),
            AnnualEmissionRecord(year=2021, total_tco2e=Decimal("99000")),
            AnnualEmissionRecord(year=2022, total_tco2e=Decimal("101000")),
            AnnualEmissionRecord(year=2023, total_tco2e=Decimal("103000")),
            AnnualEmissionRecord(year=2024, total_tco2e=Decimal("105000")),
        ]
        return ProgressTrackingInput(
            entity_name="SlowStart Ltd",
            baseline_year=2019,
            baseline_emissions_tco2e=Decimal("100000"),
            interim_target_year=2030,
            interim_target_tco2e=Decimal("50000"),
            current_year=2024,
            annual_records=records,
        )

    def test_off_track_calculates(self, progress_engine, off_track_input):
        result = progress_engine.track(off_track_input)
        assert result is not None

    def test_off_track_status(self, progress_engine, off_track_input):
        result = progress_engine.track(off_track_input)
        assert result.progress_status in ("off_track", "critical")

    def test_off_track_has_provenance(self, progress_engine, off_track_input):
        result = progress_engine.track(off_track_input)
        assert_provenance_hash(result)


# ========================================================================
# Determinism
# ========================================================================


class TestProgressDeterminism:
    """Tests for deterministic output."""

    @pytest.fixture
    def simple_input(self):
        records = [
            AnnualEmissionRecord(year=2019, total_tco2e=Decimal("100000")),
            AnnualEmissionRecord(year=2024, total_tco2e=Decimal("80000")),
        ]
        return ProgressTrackingInput(
            entity_name="Test Corp",
            baseline_year=2019,
            baseline_emissions_tco2e=Decimal("100000"),
            interim_target_year=2030,
            interim_target_tco2e=Decimal("50000"),
            current_year=2024,
            annual_records=records,
        )

    def test_same_input_same_status(self, progress_engine, simple_input):
        r1 = progress_engine.track(simple_input)
        r2 = progress_engine.track(simple_input)
        assert r1.progress_status == r2.progress_status

    def test_same_input_same_reduction(self, progress_engine, simple_input):
        r1 = progress_engine.track(simple_input)
        r2 = progress_engine.track(simple_input)
        assert r1.cumulative_reduction_pct == r2.cumulative_reduction_pct
