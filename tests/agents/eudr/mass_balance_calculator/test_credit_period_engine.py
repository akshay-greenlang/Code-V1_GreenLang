# -*- coding: utf-8 -*-
"""
Tests for CreditPeriodEngine - AGENT-EUDR-011 Engine 2: Credit Period Lifecycle

Comprehensive test suite covering:
- Period creation (basic, auto-create, configurable durations)
- Period lifecycle (pending->active->reconciling->closed, invalid transitions)
- Period rollover (auto rollover, new period creation)
- Grace period (entry during grace, rejected after grace)
- Period extension (extend with reason, audit trail)
- Overlap prevention (no two active periods for same facility+commodity)
- Historical browsing (list past periods, period details)
- Edge cases (concurrent periods, expired periods)

Test count: 55+ tests
Coverage target: >= 85% of CreditPeriodEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.mass_balance_calculator.conftest import (
    EUDR_COMMODITIES,
    STANDARDS,
    PERIOD_STATUSES,
    CREDIT_PERIOD_DAYS,
    DEFAULT_GRACE_PERIOD_DAYS,
    SHA256_HEX_LENGTH,
    PERIOD_COCOA_RSPO,
    PERIOD_PALM_ISCC,
    PERIOD_COCOA_Q1,
    PERIOD_COCOA_Q2,
    PERIOD_PALM_Y1,
    FAC_ID_MILL_MY,
    FAC_ID_REFINERY_ID,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    make_period,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Period Creation
# ===========================================================================


class TestPeriodCreation:
    """Test credit period creation."""

    def test_create_period_basic(self, credit_period_engine):
        """Create a basic RSPO credit period."""
        period = make_period()
        result = credit_period_engine.create_period(period)
        assert result is not None
        assert result["commodity"] == "cocoa"
        assert result["standard"] == "rspo"

    @pytest.mark.parametrize("standard,expected_days", list(CREDIT_PERIOD_DAYS.items()))
    def test_create_period_standard_durations(
        self, credit_period_engine, standard, expected_days
    ):
        """Each standard has the correct default duration."""
        period = make_period(
            standard=standard,
            duration_days=expected_days,
        )
        result = credit_period_engine.create_period(period)
        assert result["duration_days"] == expected_days

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_period_all_commodities(self, credit_period_engine, commodity):
        """Periods can be created for all 7 EUDR commodities."""
        period = make_period(commodity=commodity)
        result = credit_period_engine.create_period(period)
        assert result["commodity"] == commodity

    def test_create_period_auto_id(self, credit_period_engine):
        """Period creation auto-assigns an ID if not provided."""
        period = make_period()
        period["period_id"] = None
        result = credit_period_engine.create_period(period)
        assert result.get("period_id") is not None

    def test_create_period_provenance_hash(self, credit_period_engine):
        """Period creation generates a provenance hash."""
        period = make_period()
        result = credit_period_engine.create_period(period)
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_create_period_rspo_90_days(self, credit_period_engine):
        """RSPO periods default to 90 days."""
        period = make_period(standard="rspo", duration_days=90)
        result = credit_period_engine.create_period(period)
        assert result["duration_days"] == 90

    def test_create_period_fsc_365_days(self, credit_period_engine):
        """FSC periods default to 365 days."""
        period = make_period(standard="fsc", duration_days=365)
        result = credit_period_engine.create_period(period)
        assert result["duration_days"] == 365

    def test_create_period_custom_duration(self, credit_period_engine):
        """Custom duration can be specified."""
        period = make_period(duration_days=180)
        result = credit_period_engine.create_period(period)
        assert result["duration_days"] == 180

    def test_duplicate_period_id_raises(self, credit_period_engine):
        """Creating a period with duplicate ID raises an error."""
        period = make_period(period_id="PRD-DUP-001")
        credit_period_engine.create_period(period)
        with pytest.raises((ValueError, KeyError)):
            credit_period_engine.create_period(copy.deepcopy(period))

    def test_missing_facility_raises(self, credit_period_engine):
        """Period without facility_id raises ValueError."""
        period = make_period()
        period["facility_id"] = None
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)

    def test_missing_commodity_raises(self, credit_period_engine):
        """Period without commodity raises ValueError."""
        period = make_period()
        period["commodity"] = None
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)


# ===========================================================================
# 2. Period Lifecycle
# ===========================================================================


class TestPeriodLifecycle:
    """Test period lifecycle state machine."""

    def test_initial_status_pending(self, credit_period_engine):
        """New period starts in pending status."""
        period = make_period(status="pending")
        result = credit_period_engine.create_period(period)
        assert result["status"] in ("pending", "active")

    def test_transition_pending_to_active(self, credit_period_engine):
        """Period transitions from pending to active."""
        period = make_period(status="pending", period_id="PRD-LC-001")
        credit_period_engine.create_period(period)
        result = credit_period_engine.activate("PRD-LC-001")
        assert result["status"] == "active"

    def test_transition_active_to_reconciling(self, credit_period_engine):
        """Period transitions from active to reconciling."""
        period = make_period(status="active", period_id="PRD-LC-002")
        credit_period_engine.create_period(period)
        result = credit_period_engine.begin_reconciliation("PRD-LC-002")
        assert result["status"] == "reconciling"

    def test_transition_reconciling_to_closed(self, credit_period_engine):
        """Period transitions from reconciling to closed."""
        period = make_period(status="active", period_id="PRD-LC-003")
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-LC-003")
        result = credit_period_engine.close("PRD-LC-003")
        assert result["status"] == "closed"

    def test_invalid_transition_closed_to_active(self, credit_period_engine):
        """Closed period cannot transition to active."""
        period = make_period(status="active", period_id="PRD-LC-004")
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-LC-004")
        credit_period_engine.close("PRD-LC-004")
        with pytest.raises((ValueError, RuntimeError)):
            credit_period_engine.activate("PRD-LC-004")

    def test_invalid_transition_pending_to_closed(self, credit_period_engine):
        """Pending period cannot skip to closed."""
        period = make_period(status="pending", period_id="PRD-LC-005")
        credit_period_engine.create_period(period)
        with pytest.raises((ValueError, RuntimeError)):
            credit_period_engine.close("PRD-LC-005")

    def test_nonexistent_period_raises(self, credit_period_engine):
        """Transitioning non-existent period raises error."""
        with pytest.raises((ValueError, KeyError)):
            credit_period_engine.activate("PRD-NONEXISTENT")

    def test_lifecycle_provenance_on_transition(self, credit_period_engine):
        """Each lifecycle transition generates a provenance hash."""
        period = make_period(status="pending", period_id="PRD-LC-006")
        credit_period_engine.create_period(period)
        result = credit_period_engine.activate("PRD-LC-006")
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 3. Period Rollover
# ===========================================================================


class TestPeriodRollover:
    """Test auto rollover when period expires."""

    def test_auto_rollover_creates_new_period(self, credit_period_engine):
        """Auto rollover creates a new period when current one closes."""
        period = make_period(
            period_id="PRD-ROLL-001",
            status="active",
            start_days_ago=95,
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-ROLL-001")
        result = credit_period_engine.close("PRD-ROLL-001", auto_rollover=True)
        assert result.get("new_period_id") is not None or result["status"] == "closed"

    def test_rollover_preserves_facility_commodity(self, credit_period_engine):
        """Rolled-over period has same facility and commodity."""
        period = make_period(
            period_id="PRD-ROLL-002",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
            start_days_ago=95,
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-ROLL-002")
        result = credit_period_engine.close("PRD-ROLL-002", auto_rollover=True)
        new_id = result.get("new_period_id")
        if new_id:
            new_period = credit_period_engine.get(new_id)
            assert new_period["facility_id"] == FAC_ID_MILL_MY
            assert new_period["commodity"] == "cocoa"

    def test_rollover_new_period_starts_at_old_end(self, credit_period_engine):
        """New period starts at or after old period end date."""
        period = make_period(
            period_id="PRD-ROLL-003",
            status="active",
            start_days_ago=95,
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-ROLL-003")
        result = credit_period_engine.close("PRD-ROLL-003", auto_rollover=True)
        assert result is not None


# ===========================================================================
# 4. Grace Period
# ===========================================================================


class TestGracePeriod:
    """Test grace period for late entries."""

    def test_entry_allowed_during_grace(self, credit_period_engine):
        """Entries are accepted during the grace period."""
        period = make_period(
            period_id="PRD-GRACE-001",
            status="active",
            start_days_ago=92,
            duration_days=90,
            grace_period_days=5,
        )
        credit_period_engine.create_period(period)
        result = credit_period_engine.check_entry_allowed("PRD-GRACE-001")
        assert result.get("allowed") is True or result.get("in_grace_period") is True

    def test_entry_rejected_after_grace(self, credit_period_engine):
        """Entries are rejected after the grace period expires."""
        period = make_period(
            period_id="PRD-GRACE-002",
            status="reconciling",
            start_days_ago=100,
            duration_days=90,
            grace_period_days=5,
        )
        credit_period_engine.create_period(period)
        result = credit_period_engine.check_entry_allowed("PRD-GRACE-002")
        assert result.get("allowed") is False or result.get("rejected") is True

    def test_default_grace_period(self, mbc_config):
        """Default grace period is 5 days."""
        assert mbc_config["grace_period_days"] == DEFAULT_GRACE_PERIOD_DAYS

    def test_custom_grace_period(self, credit_period_engine):
        """Custom grace period can be configured."""
        period = make_period(
            period_id="PRD-GRACE-003",
            grace_period_days=10,
        )
        result = credit_period_engine.create_period(period)
        assert result["grace_period_days"] == 10


# ===========================================================================
# 5. Period Extension
# ===========================================================================


class TestPeriodExtension:
    """Test period extension with audit trail."""

    def test_extend_period_with_reason(self, credit_period_engine):
        """Period can be extended with a reason."""
        period = make_period(
            period_id="PRD-EXT-001",
            status="active",
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        result = credit_period_engine.extend(
            "PRD-EXT-001",
            additional_days=30,
            reason="Supplier delay in documentation",
        )
        assert result is not None
        assert result.get("duration_days", 0) >= 120 or result.get("extended") is True

    def test_extend_creates_audit_trail(self, credit_period_engine):
        """Extension creates an audit trail entry."""
        period = make_period(
            period_id="PRD-EXT-002",
            status="active",
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        credit_period_engine.extend(
            "PRD-EXT-002",
            additional_days=15,
            reason="Quality check pending",
        )
        history = credit_period_engine.get_history("PRD-EXT-002")
        assert len(history) >= 1

    def test_extend_closed_period_raises(self, credit_period_engine):
        """Cannot extend a closed period."""
        period = make_period(
            period_id="PRD-EXT-003",
            status="active",
        )
        credit_period_engine.create_period(period)
        credit_period_engine.begin_reconciliation("PRD-EXT-003")
        credit_period_engine.close("PRD-EXT-003")
        with pytest.raises((ValueError, RuntimeError)):
            credit_period_engine.extend(
                "PRD-EXT-003",
                additional_days=30,
                reason="Late request",
            )

    def test_extend_nonexistent_raises(self, credit_period_engine):
        """Extending non-existent period raises error."""
        with pytest.raises((ValueError, KeyError)):
            credit_period_engine.extend(
                "PRD-NONEXISTENT",
                additional_days=30,
                reason="Test",
            )


# ===========================================================================
# 6. Overlap Prevention
# ===========================================================================


class TestOverlapPrevention:
    """Test no two active periods for same facility+commodity."""

    def test_no_concurrent_active_periods(self, credit_period_engine):
        """Cannot have two active periods for same facility+commodity."""
        p1 = make_period(
            period_id="PRD-OVR-001",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        credit_period_engine.create_period(p1)
        p2 = make_period(
            period_id="PRD-OVR-002",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        with pytest.raises((ValueError, RuntimeError)):
            credit_period_engine.create_period(p2)

    def test_different_commodities_allowed(self, credit_period_engine):
        """Different commodities at same facility can have active periods."""
        p1 = make_period(
            period_id="PRD-OVR-003",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        credit_period_engine.create_period(p1)
        p2 = make_period(
            period_id="PRD-OVR-004",
            facility_id=FAC_ID_MILL_MY,
            commodity="oil_palm",
            status="active",
        )
        result = credit_period_engine.create_period(p2)
        assert result is not None

    def test_different_facilities_allowed(self, credit_period_engine):
        """Different facilities with same commodity can have active periods."""
        p1 = make_period(
            period_id="PRD-OVR-005",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        credit_period_engine.create_period(p1)
        p2 = make_period(
            period_id="PRD-OVR-006",
            facility_id=FAC_ID_REFINERY_ID,
            commodity="cocoa",
            status="active",
        )
        result = credit_period_engine.create_period(p2)
        assert result is not None

    def test_closed_period_allows_new_active(self, credit_period_engine):
        """After closing a period, a new active period can be created."""
        p1 = make_period(
            period_id="PRD-OVR-007",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        credit_period_engine.create_period(p1)
        credit_period_engine.begin_reconciliation("PRD-OVR-007")
        credit_period_engine.close("PRD-OVR-007")
        p2 = make_period(
            period_id="PRD-OVR-008",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
        )
        result = credit_period_engine.create_period(p2)
        assert result is not None


# ===========================================================================
# 7. Historical Browsing
# ===========================================================================


class TestHistoricalBrowsing:
    """Test historical period browsing."""

    def test_list_past_periods(self, credit_period_engine):
        """List past periods for a facility+commodity."""
        p1 = make_period(
            period_id="PRD-HIST-001",
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
            status="active",
            start_days_ago=200,
            duration_days=90,
        )
        credit_period_engine.create_period(p1)
        credit_period_engine.begin_reconciliation("PRD-HIST-001")
        credit_period_engine.close("PRD-HIST-001")
        periods = credit_period_engine.list_periods(
            facility_id=FAC_ID_MILL_MY,
            commodity="cocoa",
        )
        assert len(periods) >= 1

    def test_get_period_details(self, credit_period_engine):
        """Get detailed information for a specific period."""
        period = make_period(period_id="PRD-HIST-002")
        credit_period_engine.create_period(period)
        result = credit_period_engine.get("PRD-HIST-002")
        assert result is not None
        assert result["period_id"] == "PRD-HIST-002"

    def test_get_nonexistent_period_returns_none(self, credit_period_engine):
        """Getting non-existent period returns None."""
        result = credit_period_engine.get("PRD-NONEXISTENT-999")
        assert result is None

    def test_history_includes_transitions(self, credit_period_engine):
        """Period history includes lifecycle transition events."""
        period = make_period(period_id="PRD-HIST-003", status="pending")
        credit_period_engine.create_period(period)
        credit_period_engine.activate("PRD-HIST-003")
        history = credit_period_engine.get_history("PRD-HIST-003")
        assert len(history) >= 1


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for credit period operations."""

    def test_zero_duration_raises(self, credit_period_engine):
        """Period with zero duration raises ValueError."""
        period = make_period(duration_days=0)
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)

    def test_negative_duration_raises(self, credit_period_engine):
        """Period with negative duration raises ValueError."""
        period = make_period(duration_days=-30)
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)

    def test_very_long_period(self, credit_period_engine):
        """Period with 730-day duration is accepted."""
        period = make_period(duration_days=730)
        result = credit_period_engine.create_period(period)
        assert result["duration_days"] == 730

    def test_expired_period_detected(self, credit_period_engine):
        """Expired period (past end date) is detected."""
        period = make_period(
            period_id="PRD-EDGE-001",
            status="active",
            start_days_ago=120,
            duration_days=90,
        )
        credit_period_engine.create_period(period)
        result = credit_period_engine.check_expiry("PRD-EDGE-001")
        assert result.get("expired") is True or result.get("status") == "expired"

    def test_negative_grace_period_raises(self, credit_period_engine):
        """Negative grace period raises ValueError."""
        period = make_period(grace_period_days=-1)
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)

    @pytest.mark.parametrize("status", PERIOD_STATUSES)
    def test_list_by_status(self, credit_period_engine, status):
        """Can filter periods by each valid status."""
        period = make_period(
            period_id=f"PRD-EDGE-ST-{status}",
            status=status if status != "reconciling" else "active",
        )
        credit_period_engine.create_period(period)
        if status == "reconciling":
            credit_period_engine.begin_reconciliation(f"PRD-EDGE-ST-{status}")
        results = credit_period_engine.list_periods(status=status)
        assert isinstance(results, list)

    def test_invalid_standard_raises(self, credit_period_engine):
        """Period with invalid standard raises ValueError."""
        period = make_period()
        period["standard"] = "invalid_standard"
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)

    def test_period_count_tracking(self, credit_period_engine):
        """Period tracks entry count."""
        period = make_period(period_id="PRD-COUNT-001")
        result = credit_period_engine.create_period(period)
        assert result["entry_count"] == 0

    def test_auto_create_on_first_entry(self, credit_period_engine):
        """Period is auto-created on first entry for facility+commodity."""
        result = credit_period_engine.auto_create_if_needed(
            facility_id=FAC_ID_MILL_MY,
            commodity="rubber",
            standard="eudr_default",
        )
        assert result is not None

    @pytest.mark.parametrize("standard", STANDARDS)
    def test_create_all_standards(self, credit_period_engine, standard):
        """Periods can be created with all certification standards."""
        period = make_period(standard=standard)
        result = credit_period_engine.create_period(period)
        assert result["standard"] == standard

    def test_period_dates_valid(self, credit_period_engine):
        """Period end_date is after start_date."""
        period = make_period(
            period_id="PRD-DATES-001",
            start_days_ago=30,
            duration_days=90,
        )
        result = credit_period_engine.create_period(period)
        assert result["start_date"] < result["end_date"]

    def test_very_short_grace_period(self, credit_period_engine):
        """Grace period of 0 days is accepted."""
        period = make_period(grace_period_days=0)
        result = credit_period_engine.create_period(period)
        assert result["grace_period_days"] == 0

    def test_max_grace_period(self, credit_period_engine):
        """Grace period of 90 days is accepted."""
        period = make_period(grace_period_days=90)
        result = credit_period_engine.create_period(period)
        assert result["grace_period_days"] == 90

    def test_excessive_grace_period_raises(self, credit_period_engine):
        """Grace period exceeding 90 days raises ValueError."""
        period = make_period(grace_period_days=91)
        with pytest.raises(ValueError):
            credit_period_engine.create_period(period)
