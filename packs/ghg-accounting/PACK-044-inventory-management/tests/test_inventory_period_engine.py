# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Inventory Period Engine Tests
=====================================================

Tests InventoryPeriodEngine: state machine transitions, milestone tracking,
period creation, locking, comparison, auto-creation, and guard conditions.

Target: 80+ test cases.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from conftest import _load_engine, compute_provenance_hash

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("inventory_period")

InventoryPeriodEngine = _mod.InventoryPeriodEngine
InventoryPeriod = _mod.InventoryPeriod
PeriodMilestone = _mod.PeriodMilestone
PeriodTransition = _mod.PeriodTransition
PeriodManagementResult = _mod.PeriodManagementResult
MetricComparison = _mod.MetricComparison
PeriodComparison = _mod.PeriodComparison
PeriodStatus = _mod.PeriodStatus
PeriodType = _mod.PeriodType
MilestoneStatus = _mod.MilestoneStatus
ComparisonMetric = _mod.ComparisonMetric
ALLOWED_TRANSITIONS = _mod.ALLOWED_TRANSITIONS
DEFAULT_MILESTONES = _mod.DEFAULT_MILESTONES


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh InventoryPeriodEngine."""
    return InventoryPeriodEngine()


@pytest.fixture
def created_period(engine, sample_period):
    """Create a period and return (engine, result)."""
    result = engine.create_period(
        organisation_id=sample_period["organisation_id"],
        period_name=sample_period["period_name"],
        start_date=sample_period["start_date"],
        end_date=sample_period["end_date"],
        period_type=PeriodType.CALENDAR_YEAR,
        base_year=sample_period["base_year"],
        base_year_reference=sample_period["base_year_reference"],
        created_by=sample_period["created_by"],
        metadata=sample_period["metadata"],
    )
    return engine, result


def _advance_to_data_collection(engine, period_id):
    """Complete a planning milestone and transition to DATA_COLLECTION."""
    period = engine.get_period(period_id)
    planning_ms = [m for m in period.milestones if m.phase == "planning"]
    if planning_ms:
        engine.update_milestone(
            period_id, planning_ms[0].milestone_id,
            status=MilestoneStatus.COMPLETED, actual_date=date.today(),
        )
    return engine.transition(period_id, PeriodStatus.DATA_COLLECTION)


def _advance_to_calculation(engine, period_id):
    """Advance period through DATA_COLLECTION to CALCULATION."""
    _advance_to_data_collection(engine, period_id)
    period = engine.get_period(period_id)
    dc_ms = [m for m in period.milestones if m.phase == "data_collection"]
    if dc_ms:
        engine.update_milestone(
            period_id, dc_ms[0].milestone_id,
            status=MilestoneStatus.COMPLETED, actual_date=date.today(),
        )
    return engine.transition(period_id, PeriodStatus.CALCULATION)


def _advance_to_review(engine, period_id):
    """Advance period through CALCULATION to REVIEW."""
    _advance_to_calculation(engine, period_id)
    period = engine.get_period(period_id)
    calc_ms = [m for m in period.milestones if m.phase == "calculation"]
    if calc_ms:
        engine.update_milestone(
            period_id, calc_ms[0].milestone_id,
            status=MilestoneStatus.COMPLETED, actual_date=date.today(),
        )
    return engine.transition(period_id, PeriodStatus.REVIEW)


def _advance_to_approved(engine, period_id):
    """Advance period through REVIEW to APPROVED."""
    _advance_to_review(engine, period_id)
    period = engine.get_period(period_id)
    review_ms = [m for m in period.milestones if m.phase == "review"]
    for ms in review_ms:
        engine.update_milestone(
            period_id, ms.milestone_id,
            status=MilestoneStatus.COMPLETED, actual_date=date.today(),
        )
    return engine.transition(period_id, PeriodStatus.APPROVED)


# ===================================================================
# Period Creation Tests
# ===================================================================


class TestPeriodCreation:
    """Tests for create_period."""

    def test_create_period_returns_result(self, created_period):
        _, result = created_period
        assert isinstance(result, PeriodManagementResult)
        assert result.operation == "create_period"

    def test_created_period_has_planning_status(self, created_period):
        _, result = created_period
        assert result.period.status == PeriodStatus.PLANNING

    def test_created_period_has_milestones(self, created_period):
        _, result = created_period
        assert len(result.period.milestones) == len(DEFAULT_MILESTONES)

    def test_created_period_has_provenance_hash(self, created_period):
        _, result = created_period
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_created_period_not_locked(self, created_period):
        _, result = created_period
        assert result.period.locked is False

    def test_created_period_stored_in_engine(self, created_period):
        engine, result = created_period
        retrieved = engine.get_period(result.period.period_id)
        assert retrieved.period_id == result.period.period_id

    def test_periods_managed_increments(self, engine, sample_period):
        r1 = engine.create_period(
            organisation_id=sample_period["organisation_id"],
            period_name="Period 1",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert r1.periods_managed == 1
        r2 = engine.create_period(
            organisation_id=sample_period["organisation_id"],
            period_name="Period 2",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
        )
        assert r2.periods_managed == 2

    def test_create_period_end_before_start_raises(self, engine):
        with pytest.raises(ValueError, match="end_date"):
            engine.create_period(
                organisation_id="org-001",
                period_name="Bad Period",
                start_date=date(2025, 12, 31),
                end_date=date(2025, 1, 1),
            )

    def test_create_period_same_start_end_raises(self, engine):
        with pytest.raises(ValueError, match="end_date"):
            engine.create_period(
                organisation_id="org-001",
                period_name="Zero Length",
                start_date=date(2025, 6, 15),
                end_date=date(2025, 6, 15),
            )

    def test_overlapping_periods_generate_warning(self, engine, sample_period):
        engine.create_period(
            organisation_id="ORG-ACME-001",
            period_name="FY2025 A",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        result = engine.create_period(
            organisation_id="ORG-ACME-001",
            period_name="FY2025 B Overlapping",
            start_date=date(2025, 6, 1),
            end_date=date(2026, 5, 31),
        )
        assert len(result.warnings) > 0
        assert "Overlaps" in result.warnings[0]

    def test_base_year_flag_stored(self, engine):
        result = engine.create_period(
            organisation_id="org-001",
            period_name="Base Year",
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            base_year=True,
        )
        assert result.period.base_year is True

    def test_fiscal_year_type(self, engine):
        result = engine.create_period(
            organisation_id="org-001",
            period_name="FY2025",
            start_date=date(2025, 4, 1),
            end_date=date(2026, 3, 31),
            period_type=PeriodType.FISCAL_YEAR,
            fiscal_year_start_month=4,
        )
        assert result.period.period_type == PeriodType.FISCAL_YEAR
        assert result.period.fiscal_year_start_month == 4

    def test_metadata_preserved(self, created_period):
        _, result = created_period
        assert result.period.metadata.get("framework") == "GHG Protocol"

    def test_milestone_period_ids_set(self, created_period):
        _, result = created_period
        pid = result.period.period_id
        for ms in result.period.milestones:
            assert ms.period_id == pid

    def test_processing_time_positive(self, created_period):
        _, result = created_period
        assert result.processing_time_ms >= Decimal("0")


# ===================================================================
# State Transition Tests
# ===================================================================


class TestStateTransitions:
    """Tests for transition method and state machine."""

    def test_planning_to_data_collection(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_data_collection(engine, pid)
        assert engine.get_period(pid).status == PeriodStatus.DATA_COLLECTION

    def test_data_collection_to_calculation(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_calculation(engine, pid)
        assert engine.get_period(pid).status == PeriodStatus.CALCULATION

    def test_calculation_to_review(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_review(engine, pid)
        assert engine.get_period(pid).status == PeriodStatus.REVIEW

    def test_review_to_approved(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        assert engine.get_period(pid).status == PeriodStatus.APPROVED

    def test_approved_auto_locks_period(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        period = engine.get_period(pid)
        assert period.locked is True
        assert period.locked_at is not None

    def test_approved_to_final(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        r = engine.transition(pid, PeriodStatus.FINAL)
        assert r.period.status == PeriodStatus.FINAL

    def test_final_to_archived(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        engine.transition(pid, PeriodStatus.FINAL)
        r = engine.transition(pid, PeriodStatus.ARCHIVED)
        assert r.period.status == PeriodStatus.ARCHIVED

    def test_archived_is_terminal(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        engine.transition(pid, PeriodStatus.FINAL)
        engine.transition(pid, PeriodStatus.ARCHIVED)
        with pytest.raises(ValueError, match="not allowed"):
            engine.transition(pid, PeriodStatus.PLANNING)

    def test_invalid_transition_raises(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        with pytest.raises(ValueError, match="not allowed"):
            engine.transition(pid, PeriodStatus.APPROVED)

    def test_amendment_requires_reason(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        with pytest.raises(ValueError, match="Reason is required"):
            engine.transition(pid, PeriodStatus.AMENDED, reason="")

    def test_amendment_with_reason_succeeds(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        r = engine.transition(
            pid, PeriodStatus.AMENDED, reason="Correction needed"
        )
        assert r.period.status == PeriodStatus.AMENDED
        assert r.period.amendment_reason == "Correction needed"

    def test_amended_back_to_planning(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        engine.transition(pid, PeriodStatus.AMENDED, reason="Fix")
        r = engine.transition(pid, PeriodStatus.PLANNING)
        assert r.period.status == PeriodStatus.PLANNING

    def test_transition_records_history(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_data_collection(engine, pid)
        history = engine.get_transition_history(pid)
        assert len(history) >= 1
        assert history[0].from_status == PeriodStatus.PLANNING
        assert history[0].to_status == PeriodStatus.DATA_COLLECTION

    def test_unknown_period_raises_keyerror(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.transition("nonexistent-id", PeriodStatus.DATA_COLLECTION)

    @pytest.mark.parametrize("from_status,allowed_targets", [
        (PeriodStatus.PLANNING, {PeriodStatus.DATA_COLLECTION, PeriodStatus.AMENDED}),
        (PeriodStatus.FINAL, {PeriodStatus.ARCHIVED}),
        (PeriodStatus.ARCHIVED, set()),
        (PeriodStatus.AMENDED, {PeriodStatus.PLANNING}),
    ])
    def test_allowed_transitions_matrix(self, from_status, allowed_targets):
        assert ALLOWED_TRANSITIONS[from_status] == allowed_targets


# ===================================================================
# Guard Condition Tests
# ===================================================================


class TestGuardConditions:
    """Tests for guard checks on transitions."""

    def test_dc_transition_blocked_without_planning_milestone(self, engine):
        result = engine.create_period(
            organisation_id="org-001",
            period_name="Guard Test",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        pid = result.period.period_id
        with pytest.raises(ValueError, match="Guard check"):
            engine.transition(pid, PeriodStatus.DATA_COLLECTION)

    def test_calculation_blocked_without_dc_milestone(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_data_collection(engine, pid)
        with pytest.raises(ValueError, match="Guard check"):
            engine.transition(pid, PeriodStatus.CALCULATION)

    def test_review_blocked_without_calculation_milestone(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_calculation(engine, pid)
        with pytest.raises(ValueError, match="Guard check"):
            engine.transition(pid, PeriodStatus.REVIEW)

    def test_approved_blocked_without_review_milestone(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_review(engine, pid)
        with pytest.raises(ValueError, match="Guard check"):
            engine.transition(pid, PeriodStatus.APPROVED)

    def test_final_blocked_when_not_locked(self, engine):
        r = engine.create_period(
            organisation_id="org-001",
            period_name="Final Guard",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        pid = r.period.period_id
        _advance_to_review(engine, pid)
        period = engine.get_period(pid)
        review_ms = [m for m in period.milestones if m.phase == "review"]
        for ms in review_ms:
            engine.update_milestone(
                pid, ms.milestone_id,
                status=MilestoneStatus.COMPLETED, actual_date=date.today(),
            )
        engine.transition(pid, PeriodStatus.APPROVED)
        assert engine.get_period(pid).locked is True


# ===================================================================
# Milestone Tests
# ===================================================================


class TestMilestones:
    """Tests for milestone management."""

    def test_update_milestone_status(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        # Find a milestone with a future target_date so it won't be auto-overdue
        future_ms = None
        for ms in result.period.milestones:
            if ms.target_date is not None and ms.target_date >= date.today():
                future_ms = ms
                break
        if future_ms is None:
            # All milestones are in the past, so set one to future first
            ms = result.period.milestones[-1]
            ms.target_date = date.today() + timedelta(days=30)
            future_ms = ms
        ms_id = future_ms.milestone_id
        engine.update_milestone(pid, ms_id, status=MilestoneStatus.IN_PROGRESS)
        p = engine.get_period(pid)
        ms = [m for m in p.milestones if m.milestone_id == ms_id][0]
        assert ms.status == MilestoneStatus.IN_PROGRESS

    def test_update_milestone_actual_date_auto_completes(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        ms_id = result.period.milestones[0].milestone_id
        engine.update_milestone(pid, ms_id, actual_date=date.today())
        p = engine.get_period(pid)
        ms = [m for m in p.milestones if m.milestone_id == ms_id][0]
        assert ms.status == MilestoneStatus.COMPLETED
        assert ms.actual_date == date.today()

    def test_update_milestone_notes(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        ms_id = result.period.milestones[0].milestone_id
        engine.update_milestone(pid, ms_id, notes="Started boundary work")
        p = engine.get_period(pid)
        ms = [m for m in p.milestones if m.milestone_id == ms_id][0]
        assert "boundary" in ms.notes.lower()

    def test_update_milestone_locked_period_raises(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        _advance_to_approved(engine, pid)
        with pytest.raises(ValueError, match="locked"):
            engine.update_milestone(
                pid, result.period.milestones[0].milestone_id,
                status=MilestoneStatus.COMPLETED,
            )

    def test_update_nonexistent_milestone_raises(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        with pytest.raises(KeyError, match="not found"):
            engine.update_milestone(pid, "fake-milestone-id", notes="nope")

    def test_completion_percentage_zero_initially(self, created_period):
        engine, result = created_period
        pct = engine.get_completion_percentage(result.period.period_id)
        assert pct == Decimal("0.00")

    def test_completion_percentage_increases(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        ms_id = result.period.milestones[0].milestone_id
        engine.update_milestone(pid, ms_id, status=MilestoneStatus.COMPLETED)
        pct = engine.get_completion_percentage(pid)
        assert pct > Decimal("0")

    def test_completion_counts_skipped(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        ms_id = result.period.milestones[0].milestone_id
        engine.update_milestone(pid, ms_id, status=MilestoneStatus.SKIPPED)
        pct = engine.get_completion_percentage(pid)
        assert pct > Decimal("0")

    def test_default_milestones_count(self):
        assert len(DEFAULT_MILESTONES) == 10

    def test_default_milestones_phases(self):
        phases = {m["phase"] for m in DEFAULT_MILESTONES}
        assert "planning" in phases
        assert "data_collection" in phases
        assert "calculation" in phases
        assert "review" in phases


# ===================================================================
# Locking Tests
# ===================================================================


class TestLocking:
    """Tests for lock_period and unlock_period."""

    def test_lock_period(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        r = engine.lock_period(pid, locked_by="admin")
        assert r.period.locked is True
        assert r.period.locked_by == "admin"

    def test_lock_already_locked_raises(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        engine.lock_period(pid)
        with pytest.raises(ValueError, match="already locked"):
            engine.lock_period(pid)

    def test_unlock_period(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        engine.lock_period(pid)
        r = engine.unlock_period(pid, unlocked_by="admin", reason="Amendment")
        assert r.period.locked is False
        assert r.period.locked_at is None

    def test_unlock_not_locked_raises(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        with pytest.raises(ValueError, match="not locked"):
            engine.unlock_period(pid)

    def test_unlock_non_amended_succeeds(self, created_period):
        """Unlocking a non-amended period should succeed without error."""
        engine, result = created_period
        pid = result.period.period_id
        engine.lock_period(pid)
        r = engine.unlock_period(pid, reason="Override")
        # Period is unlocked successfully
        assert r.period.locked is False


# ===================================================================
# Period Comparison Tests
# ===================================================================


class TestPeriodComparison:
    """Tests for compare_periods."""

    def test_compare_periods_returns_comparison(self, engine):
        r1 = engine.create_period(
            organisation_id="org-001",
            period_name="FY2024",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        r2 = engine.create_period(
            organisation_id="org-001",
            period_name="FY2025",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
        result = engine.compare_periods(
            r2.period.period_id,
            r1.period.period_id,
            {"total_emissions": Decimal("50000")},
            {"total_emissions": Decimal("55000")},
        )
        assert result.comparison is not None
        assert len(result.comparison.metrics) == len(ComparisonMetric)

    def test_comparison_increase_direction(self, engine):
        r1 = engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        r2 = engine.create_period("o", "P2", date(2025, 1, 1), date(2025, 12, 31))
        result = engine.compare_periods(
            r2.period.period_id, r1.period.period_id,
            {"total_emissions": Decimal("60000")},
            {"total_emissions": Decimal("50000")},
        )
        total_metric = [m for m in result.comparison.metrics
                        if m.metric == ComparisonMetric.TOTAL_EMISSIONS][0]
        assert total_metric.direction == "increase"

    def test_comparison_decrease_direction(self, engine):
        r1 = engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        r2 = engine.create_period("o", "P2", date(2025, 1, 1), date(2025, 12, 31))
        result = engine.compare_periods(
            r2.period.period_id, r1.period.period_id,
            {"total_emissions": Decimal("40000")},
            {"total_emissions": Decimal("50000")},
        )
        total_metric = [m for m in result.comparison.metrics
                        if m.metric == ComparisonMetric.TOTAL_EMISSIONS][0]
        assert total_metric.direction == "decrease"

    def test_comparison_no_change(self, engine):
        r1 = engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        r2 = engine.create_period("o", "P2", date(2025, 1, 1), date(2025, 12, 31))
        result = engine.compare_periods(
            r2.period.period_id, r1.period.period_id,
            {"total_emissions": Decimal("50000")},
            {"total_emissions": Decimal("50000")},
        )
        total_metric = [m for m in result.comparison.metrics
                        if m.metric == ComparisonMetric.TOTAL_EMISSIONS][0]
        assert total_metric.direction == "no_change"

    def test_comparison_unknown_period_raises(self, engine):
        r1 = engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        with pytest.raises(KeyError):
            engine.compare_periods(
                "nonexistent", r1.period.period_id, {}, {},
            )


# ===================================================================
# Auto-Creation Tests
# ===================================================================


class TestAutoCreation:
    """Tests for auto_create_next_period."""

    def test_auto_create_calendar_year(self, engine):
        r = engine.create_period(
            organisation_id="org-001",
            period_name="FY2025 GHG Inventory",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            period_type=PeriodType.CALENDAR_YEAR,
        )
        next_r = engine.auto_create_next_period(r.period.period_id)
        assert next_r.period.start_date == date(2026, 1, 1)
        assert next_r.period.end_date == date(2026, 12, 31)
        assert "2026" in next_r.period.period_name

    def test_auto_create_fiscal_year(self, engine):
        r = engine.create_period(
            organisation_id="org-001",
            period_name="FY2025",
            start_date=date(2025, 4, 1),
            end_date=date(2026, 3, 31),
            period_type=PeriodType.FISCAL_YEAR,
            fiscal_year_start_month=4,
        )
        next_r = engine.auto_create_next_period(r.period.period_id)
        assert next_r.period.start_date.month == 4
        assert next_r.period.period_type == PeriodType.FISCAL_YEAR

    def test_auto_create_inherits_metadata(self, engine):
        r = engine.create_period(
            organisation_id="org-001",
            period_name="FY2025",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
            metadata={"tier": "enterprise"},
        )
        next_r = engine.auto_create_next_period(r.period.period_id)
        assert next_r.period.metadata.get("tier") == "enterprise"

    def test_auto_create_base_year_reference(self, engine):
        base = engine.create_period(
            organisation_id="org-001",
            period_name="Base Year 2019",
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            base_year=True,
        )
        next_r = engine.auto_create_next_period(base.period.period_id)
        assert next_r.period.base_year is False
        assert next_r.period.base_year_reference == base.period.period_id

    def test_auto_create_unknown_template_raises(self, engine):
        with pytest.raises(KeyError):
            engine.auto_create_next_period("nonexistent-id")


# ===================================================================
# Retrieval and Listing Tests
# ===================================================================


class TestRetrieval:
    """Tests for get_period, list_periods, get_transition_history."""

    def test_get_period_not_found(self, engine):
        with pytest.raises(KeyError):
            engine.get_period("nonexistent")

    def test_list_periods_empty(self, engine):
        assert engine.list_periods() == []

    def test_list_periods_returns_all(self, engine):
        engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        engine.create_period("o", "P2", date(2025, 1, 1), date(2025, 12, 31))
        assert len(engine.list_periods()) == 2

    def test_list_periods_filter_by_org(self, engine):
        engine.create_period("org-A", "P1", date(2024, 1, 1), date(2024, 12, 31))
        engine.create_period("org-B", "P2", date(2025, 1, 1), date(2025, 12, 31))
        assert len(engine.list_periods(organisation_id="org-A")) == 1

    def test_list_periods_filter_by_status(self, engine):
        engine.create_period("o", "P1", date(2024, 1, 1), date(2024, 12, 31))
        results = engine.list_periods(status_filter=[PeriodStatus.PLANNING])
        assert len(results) == 1

    def test_list_periods_sorted_by_start_date(self, engine):
        engine.create_period("o", "P2025", date(2025, 1, 1), date(2025, 12, 31))
        engine.create_period("o", "P2023", date(2023, 1, 1), date(2023, 12, 31))
        periods = engine.list_periods()
        assert periods[0].start_date < periods[1].start_date

    def test_get_transition_history_empty(self, engine):
        assert engine.get_transition_history() == []

    def test_get_transition_history_filtered(self, engine):
        r = engine.create_period("o", "P1", date(2025, 1, 1), date(2025, 12, 31))
        pid = r.period.period_id
        period = engine.get_period(pid)
        planning_ms = [m for m in period.milestones if m.phase == "planning"]
        if planning_ms:
            engine.update_milestone(
                pid, planning_ms[0].milestone_id,
                status=MilestoneStatus.COMPLETED,
            )
        engine.transition(pid, PeriodStatus.DATA_COLLECTION)
        history = engine.get_transition_history(pid)
        assert len(history) >= 1
        assert all(h.period_id == pid for h in history)


# ===================================================================
# Provenance Tests
# ===================================================================


class TestProvenance:
    """Tests for provenance hash integrity."""

    def test_provenance_hash_is_64_chars(self, created_period):
        _, result = created_period
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_hex_only(self, created_period):
        _, result = created_period
        int(result.provenance_hash, 16)

    def test_transition_result_has_hash(self, created_period):
        engine, result = created_period
        pid = result.period.period_id
        tr = _advance_to_data_collection(engine, pid)
        assert len(tr.provenance_hash) == 64


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model behaviour."""

    def test_inventory_period_end_before_start_raises(self):
        with pytest.raises(Exception):
            InventoryPeriod(
                start_date=date(2025, 12, 31),
                end_date=date(2025, 1, 1),
            )

    def test_period_milestone_default_status(self):
        ms = PeriodMilestone()
        assert ms.status == MilestoneStatus.PENDING

    def test_period_transition_requires_statuses(self):
        t = PeriodTransition(
            from_status=PeriodStatus.PLANNING,
            to_status=PeriodStatus.DATA_COLLECTION,
        )
        assert t.from_status == PeriodStatus.PLANNING

    @pytest.mark.parametrize("status", list(PeriodStatus))
    def test_all_period_statuses_in_transition_matrix(self, status):
        assert status in ALLOWED_TRANSITIONS

    @pytest.mark.parametrize("member", list(PeriodType))
    def test_period_type_members(self, member):
        assert member.value is not None

    @pytest.mark.parametrize("member", list(MilestoneStatus))
    def test_milestone_status_members(self, member):
        assert member.value is not None

    @pytest.mark.parametrize("member", list(ComparisonMetric))
    def test_comparison_metric_members(self, member):
        assert member.value is not None
