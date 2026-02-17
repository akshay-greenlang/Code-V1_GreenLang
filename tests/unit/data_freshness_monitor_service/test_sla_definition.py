# -*- coding: utf-8 -*-
"""
Unit tests for SLADefinitionEngine - AGENT-DATA-016 Engine 2.

Tests all 16 public methods of SLADefinitionEngine with 85%+ coverage.
Validates SLA lifecycle (create, read, update, delete), evaluation logic,
boundary conditions, template management, business hours computation,
statistics, provenance tracking, and error handling.

Target: 70+ tests across 11 test classes.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from greenlang.data_freshness_monitor.sla_definition import (
    SLADefinitionEngine,
    SLAStatus,
    BreachSeverity,
    EscalationPolicy,
)


# ---------------------------------------------------------------------------
# Helper to extract string value from enum or plain string
# ---------------------------------------------------------------------------

def _val(v) -> str:
    """Extract lowercase string value from enum or plain string."""
    if hasattr(v, "value"):
        return str(v.value).lower()
    return str(v).lower()


def _to_dict(obj) -> dict:
    """Serialize model to dict (handles both Pydantic and fallback)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return vars(obj)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> SLADefinitionEngine:
    """Create a fresh SLADefinitionEngine instance for each test."""
    return SLADefinitionEngine()


@pytest.fixture
def populated_engine(engine: SLADefinitionEngine):
    """Engine pre-populated with 3 SLAs on different datasets."""
    engine.create_sla(
        dataset_id="ds-001",
        warning_hours=12.0,
        critical_hours=48.0,
        breach_severity="low",
        escalation_policy="notify",
    )
    engine.create_sla(
        dataset_id="ds-002",
        warning_hours=24.0,
        critical_hours=72.0,
        breach_severity="medium",
        escalation_policy="escalate",
    )
    engine.create_sla(
        dataset_id="ds-003",
        warning_hours=6.0,
        critical_hours=24.0,
        breach_severity="high",
        escalation_policy="page",
        business_hours_only=True,
    )
    return engine


# ===========================================================================
# Test Class: create_sla
# ===========================================================================


class TestCreateSLA:
    """Tests for SLADefinitionEngine.create_sla."""

    def test_create_basic(self, engine: SLADefinitionEngine):
        """Create an SLA with minimal required fields and defaults."""
        sla = engine.create_sla(dataset_id="ds-001")
        assert sla.id is not None
        assert len(sla.id) == 32  # UUID4 hex
        assert sla.dataset_id == "ds-001"
        assert sla.warning_hours == 24.0
        assert sla.critical_hours == 72.0
        assert _val(sla.breach_severity) == "medium"
        assert _val(sla.escalation_policy) == "notify"
        assert sla.business_hours_only is False

    def test_create_all_fields(self, engine: SLADefinitionEngine):
        """Create an SLA with all optional fields explicitly set."""
        sla = engine.create_sla(
            dataset_id="ds-full",
            warning_hours=8.0,
            critical_hours=36.0,
            breach_severity="critical",
            escalation_policy="page",
            business_hours_only=True,
        )
        assert sla.dataset_id == "ds-full"
        assert sla.warning_hours == 8.0
        assert sla.critical_hours == 36.0
        assert _val(sla.breach_severity) == "critical"
        assert _val(sla.escalation_policy) == "page"
        assert sla.business_hours_only is True

    def test_create_sets_timestamps(self, engine: SLADefinitionEngine):
        """Created SLA should have created_at and updated_at timestamps."""
        sla = engine.create_sla(dataset_id="ds-ts")
        assert sla.created_at is not None
        assert sla.updated_at is not None
        assert sla.created_at.tzinfo is not None  # UTC-aware

    def test_create_sets_provenance_hash(self, engine: SLADefinitionEngine):
        """Created SLA should have a 64-char SHA-256 provenance hash."""
        sla = engine.create_sla(dataset_id="ds-prov")
        assert sla.provenance_hash is not None
        assert len(sla.provenance_hash) == 64

    def test_create_empty_dataset_id_raises(self, engine: SLADefinitionEngine):
        """Empty dataset_id should raise ValueError."""
        with pytest.raises(ValueError, match="dataset_id must not be empty"):
            engine.create_sla(dataset_id="")

    def test_create_whitespace_dataset_id_raises(self, engine: SLADefinitionEngine):
        """Whitespace-only dataset_id should raise ValueError."""
        with pytest.raises(ValueError, match="dataset_id must not be empty"):
            engine.create_sla(dataset_id="   ")

    def test_create_strips_dataset_id(self, engine: SLADefinitionEngine):
        """Dataset ID with leading/trailing whitespace should be stripped."""
        sla = engine.create_sla(dataset_id="  ds-strip  ")
        assert sla.dataset_id == "ds-strip"

    def test_create_warning_hours_zero_raises(self, engine: SLADefinitionEngine):
        """Warning hours of zero should raise ValueError."""
        with pytest.raises(ValueError, match="warning_hours must be > 0"):
            engine.create_sla(dataset_id="ds-001", warning_hours=0.0)

    def test_create_warning_hours_negative_raises(self, engine: SLADefinitionEngine):
        """Negative warning hours should raise ValueError."""
        with pytest.raises(ValueError, match="warning_hours must be > 0"):
            engine.create_sla(dataset_id="ds-001", warning_hours=-5.0)

    def test_create_critical_equal_warning_raises(self, engine: SLADefinitionEngine):
        """Critical hours equal to warning hours should raise ValueError."""
        with pytest.raises(ValueError, match="critical_hours.*must be >.*warning_hours"):
            engine.create_sla(
                dataset_id="ds-001", warning_hours=24.0, critical_hours=24.0,
            )

    def test_create_critical_less_than_warning_raises(self, engine: SLADefinitionEngine):
        """Critical hours less than warning hours should raise ValueError."""
        with pytest.raises(ValueError, match="critical_hours.*must be >.*warning_hours"):
            engine.create_sla(
                dataset_id="ds-001", warning_hours=48.0, critical_hours=24.0,
            )

    def test_create_duplicate_dataset_raises(self, engine: SLADefinitionEngine):
        """Creating a second SLA for the same dataset should raise ValueError."""
        engine.create_sla(dataset_id="ds-dup")
        with pytest.raises(ValueError, match="already has an SLA"):
            engine.create_sla(dataset_id="ds-dup")

    def test_create_invalid_breach_severity_raises(self, engine: SLADefinitionEngine):
        """Invalid breach severity string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid breach severity"):
            engine.create_sla(dataset_id="ds-001", breach_severity="super_bad")

    def test_create_invalid_escalation_policy_raises(self, engine: SLADefinitionEngine):
        """Invalid escalation policy string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid escalation policy"):
            engine.create_sla(dataset_id="ds-001", escalation_policy="yell")

    @pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
    def test_create_all_valid_severities(
        self, engine: SLADefinitionEngine, severity: str,
    ):
        """All four valid breach severities should be accepted."""
        sla = engine.create_sla(
            dataset_id=f"ds-sev-{severity}", breach_severity=severity,
        )
        assert _val(sla.breach_severity) == severity

    @pytest.mark.parametrize("policy", ["none", "notify", "escalate", "page"])
    def test_create_all_valid_policies(
        self, engine: SLADefinitionEngine, policy: str,
    ):
        """All four valid escalation policies should be accepted."""
        sla = engine.create_sla(
            dataset_id=f"ds-pol-{policy}", escalation_policy=policy,
        )
        assert _val(sla.escalation_policy) == policy

    def test_create_case_insensitive_severity(self, engine: SLADefinitionEngine):
        """Breach severity should be case-insensitive."""
        sla = engine.create_sla(dataset_id="ds-case", breach_severity="HIGH")
        assert _val(sla.breach_severity) == "high"

    def test_create_case_insensitive_policy(self, engine: SLADefinitionEngine):
        """Escalation policy should be case-insensitive."""
        sla = engine.create_sla(dataset_id="ds-case", escalation_policy="PAGE")
        assert _val(sla.escalation_policy) == "page"

    def test_create_serializable(self, engine: SLADefinitionEngine):
        """SLA to_dict should return a serializable dictionary."""
        sla = engine.create_sla(dataset_id="ds-ser")
        d = _to_dict(sla)
        assert d["dataset_id"] == "ds-ser"
        assert "id" in d
        assert "warning_hours" in d
        assert "critical_hours" in d

    def test_create_fractional_hours(self, engine: SLADefinitionEngine):
        """Fractional warning and critical hours should be accepted."""
        sla = engine.create_sla(
            dataset_id="ds-frac", warning_hours=0.5, critical_hours=1.5,
        )
        assert sla.warning_hours == 0.5
        assert sla.critical_hours == 1.5


# ===========================================================================
# Test Class: get_sla
# ===========================================================================


class TestGetSLA:
    """Tests for SLADefinitionEngine.get_sla."""

    def test_get_existing(self, engine: SLADefinitionEngine):
        """Retrieving an existing SLA by ID should return it."""
        created = engine.create_sla(dataset_id="ds-001")
        retrieved = engine.get_sla(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.dataset_id == "ds-001"

    def test_get_nonexistent_returns_none(self, engine: SLADefinitionEngine):
        """Retrieving a non-existent SLA should return None."""
        result = engine.get_sla("nonexistent-id-abc")
        assert result is None

    def test_get_after_delete_returns_none(self, engine: SLADefinitionEngine):
        """After deleting an SLA, get_sla should return None."""
        sla = engine.create_sla(dataset_id="ds-del")
        engine.delete_sla(sla.id)
        assert engine.get_sla(sla.id) is None


# ===========================================================================
# Test Class: get_sla_for_dataset
# ===========================================================================


class TestGetSLAForDataset:
    """Tests for SLADefinitionEngine.get_sla_for_dataset."""

    def test_get_sla_for_existing_dataset(self, engine: SLADefinitionEngine):
        """Should return the SLA associated with a dataset."""
        created = engine.create_sla(dataset_id="ds-lookup")
        result = engine.get_sla_for_dataset("ds-lookup")
        assert result is not None
        assert result.id == created.id

    def test_get_sla_for_unknown_dataset_returns_none(
        self, engine: SLADefinitionEngine,
    ):
        """Should return None for a dataset with no SLA."""
        result = engine.get_sla_for_dataset("ds-unknown")
        assert result is None

    def test_get_sla_for_dataset_after_delete(self, engine: SLADefinitionEngine):
        """After SLA is deleted, dataset lookup should return None."""
        sla = engine.create_sla(dataset_id="ds-delcheck")
        engine.delete_sla(sla.id)
        assert engine.get_sla_for_dataset("ds-delcheck") is None


# ===========================================================================
# Test Class: list_slas
# ===========================================================================


class TestListSLAs:
    """Tests for SLADefinitionEngine.list_slas."""

    def test_list_empty(self, engine: SLADefinitionEngine):
        """Listing SLAs on an empty engine should return empty list."""
        result = engine.list_slas()
        assert result == []

    def test_list_returns_all(self, populated_engine: SLADefinitionEngine):
        """Listing without filter should return all SLAs."""
        result = populated_engine.list_slas()
        assert len(result) == 3

    def test_list_filter_by_dataset(self, populated_engine: SLADefinitionEngine):
        """Filtering by dataset_id should return only matching SLAs."""
        result = populated_engine.list_slas(dataset_id="ds-002")
        assert len(result) == 1
        assert result[0].dataset_id == "ds-002"

    def test_list_filter_unknown_dataset(self, populated_engine: SLADefinitionEngine):
        """Filtering by non-existent dataset should return empty list."""
        result = populated_engine.list_slas(dataset_id="ds-999")
        assert result == []

    def test_list_limit(self, populated_engine: SLADefinitionEngine):
        """Limit parameter should cap the number of results."""
        result = populated_engine.list_slas(limit=2)
        assert len(result) == 2

    def test_list_offset(self, populated_engine: SLADefinitionEngine):
        """Offset parameter should skip leading results."""
        all_slas = populated_engine.list_slas()
        offset_slas = populated_engine.list_slas(offset=1)
        assert len(offset_slas) == len(all_slas) - 1

    def test_list_limit_and_offset(self, populated_engine: SLADefinitionEngine):
        """Combined limit and offset should paginate correctly."""
        result = populated_engine.list_slas(limit=1, offset=1)
        assert len(result) == 1

    def test_list_offset_beyond_total(self, populated_engine: SLADefinitionEngine):
        """Offset past the total count should return empty list."""
        result = populated_engine.list_slas(offset=100)
        assert result == []

    def test_list_sorted_newest_first(self, engine: SLADefinitionEngine):
        """Results should be sorted by created_at descending (newest first)."""
        engine.create_sla(dataset_id="ds-a", warning_hours=1.0, critical_hours=2.0)
        engine.create_sla(dataset_id="ds-b", warning_hours=1.0, critical_hours=2.0)
        engine.create_sla(dataset_id="ds-c", warning_hours=1.0, critical_hours=2.0)
        result = engine.list_slas()
        # Newest first: created_at should be non-increasing
        for i in range(len(result) - 1):
            assert result[i].created_at >= result[i + 1].created_at


# ===========================================================================
# Test Class: update_sla
# ===========================================================================


class TestUpdateSLA:
    """Tests for SLADefinitionEngine.update_sla."""

    def test_update_warning_hours(self, engine: SLADefinitionEngine):
        """Updating warning_hours should change the value."""
        sla = engine.create_sla(dataset_id="ds-upd", warning_hours=24.0, critical_hours=72.0)
        updated = engine.update_sla(sla.id, warning_hours=12.0)
        assert updated.warning_hours == 12.0

    def test_update_critical_hours(self, engine: SLADefinitionEngine):
        """Updating critical_hours should change the value."""
        sla = engine.create_sla(dataset_id="ds-upd", warning_hours=24.0, critical_hours=72.0)
        updated = engine.update_sla(sla.id, critical_hours=96.0)
        assert updated.critical_hours == 96.0

    def test_update_breach_severity(self, engine: SLADefinitionEngine):
        """Updating breach_severity should change the value."""
        sla = engine.create_sla(dataset_id="ds-upd")
        updated = engine.update_sla(sla.id, breach_severity="critical")
        assert _val(updated.breach_severity) == "critical"

    def test_update_escalation_policy(self, engine: SLADefinitionEngine):
        """Updating escalation_policy should change the value."""
        sla = engine.create_sla(dataset_id="ds-upd")
        updated = engine.update_sla(sla.id, escalation_policy="page")
        assert _val(updated.escalation_policy) == "page"

    def test_update_business_hours_only(self, engine: SLADefinitionEngine):
        """Updating business_hours_only should change the flag."""
        sla = engine.create_sla(dataset_id="ds-upd")
        assert sla.business_hours_only is False
        updated = engine.update_sla(sla.id, business_hours_only=True)
        assert updated.business_hours_only is True

    def test_update_multiple_fields(self, engine: SLADefinitionEngine):
        """Updating multiple fields at once should work."""
        sla = engine.create_sla(
            dataset_id="ds-multi", warning_hours=24.0, critical_hours=72.0,
        )
        updated = engine.update_sla(
            sla.id,
            warning_hours=6.0,
            critical_hours=48.0,
            breach_severity="high",
        )
        assert updated.warning_hours == 6.0
        assert updated.critical_hours == 48.0
        assert _val(updated.breach_severity) == "high"

    def test_update_refreshes_updated_at(self, engine: SLADefinitionEngine):
        """Updating an SLA should refresh the updated_at timestamp."""
        sla = engine.create_sla(dataset_id="ds-ts")
        original_updated = sla.updated_at
        updated = engine.update_sla(sla.id, warning_hours=12.0)
        assert updated.updated_at >= original_updated

    def test_update_refreshes_provenance_hash(self, engine: SLADefinitionEngine):
        """Updating an SLA should compute a new provenance hash."""
        sla = engine.create_sla(dataset_id="ds-prov")
        original_hash = sla.provenance_hash
        updated = engine.update_sla(sla.id, warning_hours=12.0)
        assert updated.provenance_hash != original_hash
        assert len(updated.provenance_hash) == 64

    def test_update_nonexistent_raises_keyerror(self, engine: SLADefinitionEngine):
        """Updating a non-existent SLA should raise KeyError."""
        with pytest.raises(KeyError, match="SLA not found"):
            engine.update_sla("nonexistent-id", warning_hours=12.0)

    def test_update_invalid_field_raises(self, engine: SLADefinitionEngine):
        """Updating a non-updatable field should raise ValueError."""
        sla = engine.create_sla(dataset_id="ds-upd")
        with pytest.raises(ValueError, match="Cannot update fields"):
            engine.update_sla(sla.id, dataset_id="ds-new")

    def test_update_invalid_severity_raises(self, engine: SLADefinitionEngine):
        """Updating to an invalid breach severity should raise ValueError."""
        sla = engine.create_sla(dataset_id="ds-upd")
        with pytest.raises(ValueError, match="Invalid breach severity"):
            engine.update_sla(sla.id, breach_severity="extreme")

    def test_update_invalid_policy_raises(self, engine: SLADefinitionEngine):
        """Updating to an invalid escalation policy should raise ValueError."""
        sla = engine.create_sla(dataset_id="ds-upd")
        with pytest.raises(ValueError, match="Invalid escalation policy"):
            engine.update_sla(sla.id, escalation_policy="shout")

    def test_update_warning_hours_zero_raises(self, engine: SLADefinitionEngine):
        """Updating warning_hours to zero should raise ValueError."""
        sla = engine.create_sla(dataset_id="ds-upd")
        with pytest.raises(ValueError, match="warning_hours must be > 0"):
            engine.update_sla(sla.id, warning_hours=0.0)

    def test_update_critical_hours_zero_raises(self, engine: SLADefinitionEngine):
        """Updating critical_hours to zero should raise ValueError."""
        sla = engine.create_sla(dataset_id="ds-upd")
        with pytest.raises(ValueError, match="critical_hours must be > 0"):
            engine.update_sla(sla.id, critical_hours=0.0)

    def test_update_cross_validation_critical_le_warning(
        self, engine: SLADefinitionEngine,
    ):
        """After update, critical_hours must remain > warning_hours."""
        sla = engine.create_sla(
            dataset_id="ds-cross", warning_hours=24.0, critical_hours=72.0,
        )
        with pytest.raises(ValueError, match="critical_hours.*must be >.*warning_hours"):
            engine.update_sla(sla.id, critical_hours=20.0)

    def test_update_cross_validation_warning_exceeds_critical(
        self, engine: SLADefinitionEngine,
    ):
        """Raising warning_hours above current critical should fail cross-validation."""
        sla = engine.create_sla(
            dataset_id="ds-cross2", warning_hours=24.0, critical_hours=72.0,
        )
        with pytest.raises(ValueError, match="critical_hours.*must be >.*warning_hours"):
            engine.update_sla(sla.id, warning_hours=80.0)


# ===========================================================================
# Test Class: delete_sla
# ===========================================================================


class TestDeleteSLA:
    """Tests for SLADefinitionEngine.delete_sla."""

    def test_delete_existing(self, engine: SLADefinitionEngine):
        """Deleting an existing SLA should return True."""
        sla = engine.create_sla(dataset_id="ds-del")
        assert engine.delete_sla(sla.id) is True

    def test_delete_removes_from_storage(self, engine: SLADefinitionEngine):
        """Deleted SLA should no longer be retrievable."""
        sla = engine.create_sla(dataset_id="ds-del")
        engine.delete_sla(sla.id)
        assert engine.get_sla(sla.id) is None

    def test_delete_removes_dataset_mapping(self, engine: SLADefinitionEngine):
        """After deletion, dataset lookup should return None."""
        sla = engine.create_sla(dataset_id="ds-del")
        engine.delete_sla(sla.id)
        assert engine.get_sla_for_dataset("ds-del") is None

    def test_delete_nonexistent_returns_false(self, engine: SLADefinitionEngine):
        """Deleting a non-existent SLA should return False."""
        assert engine.delete_sla("nonexistent-id") is False

    def test_delete_allows_re_creation(self, engine: SLADefinitionEngine):
        """After deletion, a new SLA can be created for the same dataset."""
        sla = engine.create_sla(dataset_id="ds-recreate")
        engine.delete_sla(sla.id)
        sla2 = engine.create_sla(dataset_id="ds-recreate")
        assert sla2.id != sla.id
        assert sla2.dataset_id == "ds-recreate"

    def test_delete_decrements_count(self, engine: SLADefinitionEngine):
        """Deleting an SLA should decrement the total SLA count."""
        sla = engine.create_sla(dataset_id="ds-cnt")
        assert engine.get_statistics()["total_slas"] == 1
        engine.delete_sla(sla.id)
        assert engine.get_statistics()["total_slas"] == 0


# ===========================================================================
# Test Class: evaluate_sla
# ===========================================================================


class TestEvaluateSLA:
    """Tests for SLADefinitionEngine.evaluate_sla and evaluate_dataset_sla.

    Evaluation logic (warning=24h, critical=72h):
      - age <= 24  -> COMPLIANT
      - 24 < age <= 72  -> WARNING
      - 72 < age <= 144 -> BREACHED
      - age > 144       -> CRITICAL
    """

    @pytest.fixture
    def sla(self, engine: SLADefinitionEngine):
        """Create a standard SLA for evaluation tests (24h/72h)."""
        return engine.create_sla(
            dataset_id="ds-eval",
            warning_hours=24.0,
            critical_hours=72.0,
        )

    def test_evaluate_compliant_zero(self, engine: SLADefinitionEngine, sla):
        """Age of zero should be COMPLIANT."""
        status = engine.evaluate_sla(sla.id, 0.0)
        assert status == SLAStatus.COMPLIANT

    def test_evaluate_compliant_below_warning(self, engine: SLADefinitionEngine, sla):
        """Age below warning threshold should be COMPLIANT."""
        status = engine.evaluate_sla(sla.id, 12.0)
        assert status == SLAStatus.COMPLIANT

    def test_evaluate_compliant_at_warning_boundary(
        self, engine: SLADefinitionEngine, sla,
    ):
        """Age exactly equal to warning_hours should be COMPLIANT."""
        status = engine.evaluate_sla(sla.id, 24.0)
        assert status == SLAStatus.COMPLIANT

    def test_evaluate_warning_just_above(self, engine: SLADefinitionEngine, sla):
        """Age slightly above warning_hours should be WARNING."""
        status = engine.evaluate_sla(sla.id, 24.001)
        assert status == SLAStatus.WARNING

    def test_evaluate_warning_midrange(self, engine: SLADefinitionEngine, sla):
        """Age between warning and critical should be WARNING."""
        status = engine.evaluate_sla(sla.id, 48.0)
        assert status == SLAStatus.WARNING

    def test_evaluate_warning_at_critical_boundary(
        self, engine: SLADefinitionEngine, sla,
    ):
        """Age exactly at critical_hours should be WARNING (<=)."""
        status = engine.evaluate_sla(sla.id, 72.0)
        assert status == SLAStatus.WARNING

    def test_evaluate_breached_just_above_critical(
        self, engine: SLADefinitionEngine, sla,
    ):
        """Age slightly above critical_hours should be BREACHED."""
        status = engine.evaluate_sla(sla.id, 72.001)
        assert status == SLAStatus.BREACHED

    def test_evaluate_breached_midrange(self, engine: SLADefinitionEngine, sla):
        """Age between critical and 2x critical should be BREACHED."""
        status = engine.evaluate_sla(sla.id, 100.0)
        assert status == SLAStatus.BREACHED

    def test_evaluate_breached_at_2x_critical_boundary(
        self, engine: SLADefinitionEngine, sla,
    ):
        """Age exactly at 2x critical_hours should be BREACHED (<=)."""
        status = engine.evaluate_sla(sla.id, 144.0)
        assert status == SLAStatus.BREACHED

    def test_evaluate_critical_just_above_2x(self, engine: SLADefinitionEngine, sla):
        """Age slightly above 2x critical_hours should be CRITICAL."""
        status = engine.evaluate_sla(sla.id, 144.001)
        assert status == SLAStatus.CRITICAL

    def test_evaluate_critical_very_large(self, engine: SLADefinitionEngine, sla):
        """Very large age should be CRITICAL."""
        status = engine.evaluate_sla(sla.id, 10000.0)
        assert status == SLAStatus.CRITICAL

    def test_evaluate_negative_age_raises(self, engine: SLADefinitionEngine, sla):
        """Negative age_hours should raise ValueError."""
        with pytest.raises(ValueError, match="age_hours must be >= 0"):
            engine.evaluate_sla(sla.id, -1.0)

    def test_evaluate_nonexistent_sla_raises(self, engine: SLADefinitionEngine):
        """Evaluating a non-existent SLA should raise KeyError."""
        with pytest.raises(KeyError, match="SLA not found"):
            engine.evaluate_sla("nonexistent", 10.0)

    @pytest.mark.parametrize("age,expected", [
        (0.0, "compliant"),
        (24.0, "compliant"),
        (24.001, "warning"),
        (72.0, "warning"),
        (72.001, "breached"),
        (144.0, "breached"),
        (144.001, "critical"),
    ])
    def test_evaluate_parametrized_boundaries(
        self, engine: SLADefinitionEngine, sla, age: float, expected: str,
    ):
        """Parametrized test across all evaluation boundaries."""
        status = engine.evaluate_sla(sla.id, age)
        assert _val(status) == expected

    def test_evaluate_dataset_sla(self, engine: SLADefinitionEngine, sla):
        """evaluate_dataset_sla should delegate correctly to evaluate_sla."""
        status = engine.evaluate_dataset_sla("ds-eval", 48.0)
        assert status == SLAStatus.WARNING

    def test_evaluate_dataset_sla_nonexistent_raises(
        self, engine: SLADefinitionEngine,
    ):
        """evaluate_dataset_sla with unknown dataset should raise KeyError."""
        with pytest.raises(KeyError, match="No SLA defined for dataset"):
            engine.evaluate_dataset_sla("ds-unknown", 10.0)

    def test_evaluate_dataset_sla_negative_age_raises(
        self, engine: SLADefinitionEngine, sla,
    ):
        """evaluate_dataset_sla with negative age should raise ValueError."""
        with pytest.raises(ValueError, match="age_hours must be >= 0"):
            engine.evaluate_dataset_sla("ds-eval", -5.0)


# ===========================================================================
# Test Class: Templates (create_template, apply_template, list_templates)
# ===========================================================================


class TestTemplates:
    """Tests for SLA template management."""

    def test_create_template_basic(self, engine: SLADefinitionEngine):
        """Create a template with defaults and verify returned object."""
        tmpl = engine.create_template(name="Standard SLA")
        assert tmpl.id is not None
        assert len(tmpl.id) == 32
        assert tmpl.name == "Standard SLA"
        assert tmpl.warning_hours == 24.0
        assert tmpl.critical_hours == 72.0
        assert _val(tmpl.breach_severity) == "medium"
        assert _val(tmpl.escalation_policy) == "notify"
        assert tmpl.created_at is not None

    def test_create_template_custom_values(self, engine: SLADefinitionEngine):
        """Create a template with fully custom parameters."""
        tmpl = engine.create_template(
            name="Strict SLA",
            warning_hours=4.0,
            critical_hours=12.0,
            breach_severity="critical",
            escalation_policy="page",
        )
        assert tmpl.warning_hours == 4.0
        assert tmpl.critical_hours == 12.0
        assert _val(tmpl.breach_severity) == "critical"
        assert _val(tmpl.escalation_policy) == "page"

    def test_create_template_empty_name_raises(self, engine: SLADefinitionEngine):
        """Empty template name should raise ValueError."""
        with pytest.raises(ValueError, match="Template name must not be empty"):
            engine.create_template(name="")

    def test_create_template_whitespace_name_raises(self, engine: SLADefinitionEngine):
        """Whitespace-only template name should raise ValueError."""
        with pytest.raises(ValueError, match="Template name must not be empty"):
            engine.create_template(name="   ")

    def test_create_template_duplicate_name_raises(self, engine: SLADefinitionEngine):
        """Duplicate template name should raise ValueError."""
        engine.create_template(name="Daily Check")
        with pytest.raises(ValueError, match="already exists"):
            engine.create_template(name="Daily Check")

    def test_create_template_invalid_thresholds_raises(
        self, engine: SLADefinitionEngine,
    ):
        """Template with critical <= warning should raise ValueError."""
        with pytest.raises(ValueError, match="critical_hours.*must be >.*warning_hours"):
            engine.create_template(
                name="Bad", warning_hours=48.0, critical_hours=24.0,
            )

    def test_create_template_zero_warning_raises(self, engine: SLADefinitionEngine):
        """Template with warning_hours=0 should raise ValueError."""
        with pytest.raises(ValueError, match="warning_hours must be > 0"):
            engine.create_template(name="Bad", warning_hours=0.0)

    def test_create_template_serializable(self, engine: SLADefinitionEngine):
        """Template to_dict should return a serializable dictionary."""
        tmpl = engine.create_template(name="Serializable")
        d = _to_dict(tmpl)
        assert d["name"] == "Serializable"
        assert "id" in d

    def test_apply_template_creates_slas(self, engine: SLADefinitionEngine):
        """Applying a template should create SLAs for each dataset."""
        tmpl = engine.create_template(
            name="Batch", warning_hours=6.0, critical_hours=24.0,
        )
        slas = engine.apply_template(tmpl.id, ["ds-a", "ds-b", "ds-c"])
        assert len(slas) == 3
        dataset_ids = {s.dataset_id for s in slas}
        assert dataset_ids == {"ds-a", "ds-b", "ds-c"}
        for s in slas:
            assert s.warning_hours == 6.0
            assert s.critical_hours == 24.0

    def test_apply_template_skips_existing(self, engine: SLADefinitionEngine):
        """Datasets with existing SLAs should be skipped during apply."""
        engine.create_sla(dataset_id="ds-existing")
        tmpl = engine.create_template(name="SkipTest")
        slas = engine.apply_template(tmpl.id, ["ds-existing", "ds-new"])
        assert len(slas) == 1
        assert slas[0].dataset_id == "ds-new"

    def test_apply_template_nonexistent_raises(self, engine: SLADefinitionEngine):
        """Applying a non-existent template should raise KeyError."""
        with pytest.raises(KeyError, match="Template not found"):
            engine.apply_template("nonexistent", ["ds-a"])

    def test_apply_template_empty_dataset_ids_raises(
        self, engine: SLADefinitionEngine,
    ):
        """Applying a template with empty dataset list should raise ValueError."""
        tmpl = engine.create_template(name="Empty")
        with pytest.raises(ValueError, match="dataset_ids must not be empty"):
            engine.apply_template(tmpl.id, [])

    def test_list_templates_empty(self, engine: SLADefinitionEngine):
        """Listing templates on empty engine should return empty list."""
        assert engine.list_templates() == []

    def test_list_templates_sorted_by_name(self, engine: SLADefinitionEngine):
        """Templates should be listed sorted alphabetically by name."""
        engine.create_template(name="Zebra")
        engine.create_template(name="Alpha")
        engine.create_template(name="Middle")
        templates = engine.list_templates()
        names = [t.name for t in templates]
        assert names == ["Alpha", "Middle", "Zebra"]

    def test_list_templates_count(self, engine: SLADefinitionEngine):
        """List should return all registered templates."""
        engine.create_template(name="T1")
        engine.create_template(name="T2")
        assert len(engine.list_templates()) == 2


# ===========================================================================
# Test Class: Breach Severity
# ===========================================================================


class TestBreachSeverity:
    """Tests for SLADefinitionEngine.get_breach_severity_for_age.

    Severity logic (warning=24h, critical=72h):
      - age <= 36 (24*1.5) -> LOW
      - 36 < age <= 72     -> MEDIUM
      - 72 < age <= 108 (72*1.5) -> HIGH
      - age > 108          -> CRITICAL
    """

    @pytest.fixture
    def sla(self, engine: SLADefinitionEngine):
        """Create a standard SLA for severity tests (24h/72h)."""
        return engine.create_sla(
            dataset_id="ds-sev",
            warning_hours=24.0,
            critical_hours=72.0,
        )

    def test_severity_low(self, engine: SLADefinitionEngine, sla):
        """Age at or below 1.5x warning should be LOW."""
        severity = engine.get_breach_severity_for_age(sla.id, 30.0)
        assert severity == BreachSeverity.LOW

    def test_severity_low_at_boundary(self, engine: SLADefinitionEngine, sla):
        """Age exactly at 1.5x warning should be LOW."""
        severity = engine.get_breach_severity_for_age(sla.id, 36.0)
        assert severity == BreachSeverity.LOW

    def test_severity_medium(self, engine: SLADefinitionEngine, sla):
        """Age between 1.5x warning and critical should be MEDIUM."""
        severity = engine.get_breach_severity_for_age(sla.id, 50.0)
        assert severity == BreachSeverity.MEDIUM

    def test_severity_medium_at_critical_boundary(
        self, engine: SLADefinitionEngine, sla,
    ):
        """Age exactly at critical should be MEDIUM."""
        severity = engine.get_breach_severity_for_age(sla.id, 72.0)
        assert severity == BreachSeverity.MEDIUM

    def test_severity_high(self, engine: SLADefinitionEngine, sla):
        """Age between critical and 1.5x critical should be HIGH."""
        severity = engine.get_breach_severity_for_age(sla.id, 90.0)
        assert severity == BreachSeverity.HIGH

    def test_severity_high_at_boundary(self, engine: SLADefinitionEngine, sla):
        """Age exactly at 1.5x critical should be HIGH."""
        severity = engine.get_breach_severity_for_age(sla.id, 108.0)
        assert severity == BreachSeverity.HIGH

    def test_severity_critical(self, engine: SLADefinitionEngine, sla):
        """Age above 1.5x critical should be CRITICAL."""
        severity = engine.get_breach_severity_for_age(sla.id, 200.0)
        assert severity == BreachSeverity.CRITICAL

    def test_severity_negative_age_raises(self, engine: SLADefinitionEngine, sla):
        """Negative age_hours should raise ValueError."""
        with pytest.raises(ValueError, match="age_hours must be >= 0"):
            engine.get_breach_severity_for_age(sla.id, -1.0)

    def test_severity_nonexistent_sla_raises(self, engine: SLADefinitionEngine):
        """Non-existent SLA should raise KeyError."""
        with pytest.raises(KeyError, match="SLA not found"):
            engine.get_breach_severity_for_age("nonexistent", 10.0)


# ===========================================================================
# Test Class: Business Hours
# ===========================================================================


class TestBusinessHours:
    """Tests for SLADefinitionEngine.is_business_hours and get_effective_sla_hours."""

    def test_business_hours_monday_10am(self, engine: SLADefinitionEngine):
        """Monday 10:00 UTC should be business hours."""
        dt = datetime(2026, 2, 16, 10, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is True

    def test_business_hours_friday_17pm(self, engine: SLADefinitionEngine):
        """Friday 17:00 UTC (last business hour) should be business hours."""
        dt = datetime(2026, 2, 20, 17, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is True

    def test_non_business_hours_saturday(self, engine: SLADefinitionEngine):
        """Saturday should not be business hours."""
        dt = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is False

    def test_non_business_hours_sunday(self, engine: SLADefinitionEngine):
        """Sunday should not be business hours."""
        dt = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is False

    def test_non_business_hours_before_9am(self, engine: SLADefinitionEngine):
        """Monday 08:59 UTC should not be business hours."""
        dt = datetime(2026, 2, 16, 8, 59, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is False

    def test_business_hours_at_9am(self, engine: SLADefinitionEngine):
        """Monday 09:00 UTC should be business hours (inclusive start)."""
        dt = datetime(2026, 2, 16, 9, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is True

    def test_non_business_hours_at_18(self, engine: SLADefinitionEngine):
        """Monday 18:00 UTC should not be business hours (exclusive end)."""
        dt = datetime(2026, 2, 16, 18, 0, tzinfo=timezone.utc)
        assert engine.is_business_hours(dt) is False

    def test_effective_hours_no_business_hours(self, engine: SLADefinitionEngine):
        """Non-business-hours SLA should return critical_hours unchanged."""
        sla = engine.create_sla(
            dataset_id="ds-eff",
            warning_hours=1.0,
            critical_hours=72.0,
            business_hours_only=False,
        )
        effective = engine.get_effective_sla_hours(sla.id)
        assert effective == 72.0

    def test_effective_hours_business_hours_scaling(
        self, engine: SLADefinitionEngine,
    ):
        """Business-hours SLA should scale critical_hours by (24/9)*(7/5)."""
        sla = engine.create_sla(
            dataset_id="ds-biz",
            warning_hours=1.0,
            critical_hours=9.0,
            business_hours_only=True,
        )
        effective = engine.get_effective_sla_hours(sla.id)
        # 9.0 * (24/9) * (7/5) = 9.0 * 2.6667 * 1.4 = 33.6
        expected = 9.0 * (24.0 / 9.0) * (7.0 / 5.0)
        assert abs(effective - expected) < 0.01
        assert abs(effective - 33.6) < 0.01

    def test_effective_hours_nonexistent_raises(self, engine: SLADefinitionEngine):
        """Non-existent SLA ID should raise KeyError."""
        with pytest.raises(KeyError, match="SLA not found"):
            engine.get_effective_sla_hours("nonexistent")


# ===========================================================================
# Test Class: Statistics
# ===========================================================================


class TestStatistics:
    """Tests for SLADefinitionEngine.get_statistics."""

    def test_stats_empty_engine(self, engine: SLADefinitionEngine):
        """Empty engine should report zero counts."""
        stats = engine.get_statistics()
        assert stats["total_slas"] == 0
        assert stats["total_templates"] == 0
        assert stats["total_datasets_covered"] == 0
        assert stats["business_hours_slas"] == 0
        assert stats["avg_warning_hours"] == 0.0
        assert stats["avg_critical_hours"] == 0.0
        assert stats["min_warning_hours"] == 0.0
        assert stats["max_critical_hours"] == 0.0
        assert stats["escalation_policies"] == {}
        assert stats["breach_severities"] == {}

    def test_stats_total_slas(self, populated_engine: SLADefinitionEngine):
        """Total SLAs should match the number created."""
        stats = populated_engine.get_statistics()
        assert stats["total_slas"] == 3

    def test_stats_datasets_covered(self, populated_engine: SLADefinitionEngine):
        """Datasets covered should match datasets with SLAs."""
        stats = populated_engine.get_statistics()
        assert stats["total_datasets_covered"] == 3

    def test_stats_business_hours_count(self, populated_engine: SLADefinitionEngine):
        """Business hours SLA count should reflect configuration."""
        stats = populated_engine.get_statistics()
        assert stats["business_hours_slas"] == 1  # ds-003 has biz hours

    def test_stats_avg_warning_hours(self, populated_engine: SLADefinitionEngine):
        """Average warning hours should be computed correctly."""
        stats = populated_engine.get_statistics()
        expected = (12.0 + 24.0 + 6.0) / 3.0
        assert abs(stats["avg_warning_hours"] - expected) < 0.01

    def test_stats_avg_critical_hours(self, populated_engine: SLADefinitionEngine):
        """Average critical hours should be computed correctly."""
        stats = populated_engine.get_statistics()
        expected = (48.0 + 72.0 + 24.0) / 3.0
        assert abs(stats["avg_critical_hours"] - expected) < 0.01

    def test_stats_min_warning(self, populated_engine: SLADefinitionEngine):
        """Min warning hours should be the smallest configured threshold."""
        stats = populated_engine.get_statistics()
        assert stats["min_warning_hours"] == 6.0

    def test_stats_max_critical(self, populated_engine: SLADefinitionEngine):
        """Max critical hours should be the largest configured threshold."""
        stats = populated_engine.get_statistics()
        assert stats["max_critical_hours"] == 72.0

    def test_stats_escalation_distribution(
        self, populated_engine: SLADefinitionEngine,
    ):
        """Escalation policy distribution should be accurate."""
        stats = populated_engine.get_statistics()
        policies = stats["escalation_policies"]
        assert policies.get("notify", 0) == 1
        assert policies.get("escalate", 0) == 1
        assert policies.get("page", 0) == 1

    def test_stats_severity_distribution(
        self, populated_engine: SLADefinitionEngine,
    ):
        """Breach severity distribution should be accurate."""
        stats = populated_engine.get_statistics()
        severities = stats["breach_severities"]
        assert severities.get("low", 0) == 1
        assert severities.get("medium", 0) == 1
        assert severities.get("high", 0) == 1

    def test_stats_provenance_entries(self, populated_engine: SLADefinitionEngine):
        """Provenance entries should be positive after SLA creation."""
        stats = populated_engine.get_statistics()
        assert stats["provenance_entries"] >= 3

    def test_stats_with_templates(self, engine: SLADefinitionEngine):
        """Template count should reflect created templates."""
        engine.create_template(name="T1")
        engine.create_template(name="T2")
        stats = engine.get_statistics()
        assert stats["total_templates"] == 2


# ===========================================================================
# Test Class: Reset
# ===========================================================================


class TestReset:
    """Tests for SLADefinitionEngine.reset."""

    def test_reset_clears_slas(self, populated_engine: SLADefinitionEngine):
        """Reset should remove all SLAs."""
        populated_engine.reset()
        assert populated_engine.get_statistics()["total_slas"] == 0

    def test_reset_clears_templates(self, engine: SLADefinitionEngine):
        """Reset should remove all templates."""
        engine.create_template(name="T1")
        engine.reset()
        assert engine.list_templates() == []

    def test_reset_clears_dataset_mappings(
        self, populated_engine: SLADefinitionEngine,
    ):
        """Reset should clear dataset-to-SLA mappings."""
        populated_engine.reset()
        assert populated_engine.get_sla_for_dataset("ds-001") is None

    def test_reset_resets_provenance(self, populated_engine: SLADefinitionEngine):
        """Reset should reinitialize provenance tracker."""
        populated_engine.reset()
        stats = populated_engine.get_statistics()
        assert stats["provenance_entries"] == 0

    def test_reset_allows_recreation(self, populated_engine: SLADefinitionEngine):
        """After reset, previously used dataset IDs can be reused."""
        populated_engine.reset()
        sla = populated_engine.create_sla(dataset_id="ds-001")
        assert sla.dataset_id == "ds-001"


# ===========================================================================
# Test Class: Provenance
# ===========================================================================


class TestProvenance:
    """Tests for provenance tracking across SLA operations."""

    def test_create_generates_provenance(self, engine: SLADefinitionEngine):
        """SLA creation should add a provenance chain entry."""
        engine.create_sla(dataset_id="ds-p1")
        stats = engine.get_statistics()
        assert stats["provenance_entries"] >= 1

    def test_update_adds_provenance_entry(self, engine: SLADefinitionEngine):
        """SLA update should add an additional provenance chain entry."""
        sla = engine.create_sla(dataset_id="ds-p2")
        entries_before = engine.get_statistics()["provenance_entries"]
        engine.update_sla(sla.id, warning_hours=12.0)
        entries_after = engine.get_statistics()["provenance_entries"]
        assert entries_after > entries_before

    def test_delete_adds_provenance_entry(self, engine: SLADefinitionEngine):
        """SLA deletion should add a provenance chain entry."""
        sla = engine.create_sla(dataset_id="ds-p3")
        entries_before = engine.get_statistics()["provenance_entries"]
        engine.delete_sla(sla.id)
        entries_after = engine.get_statistics()["provenance_entries"]
        assert entries_after > entries_before

    def test_template_creation_adds_provenance(self, engine: SLADefinitionEngine):
        """Template creation should add a provenance chain entry."""
        entries_before = engine.get_statistics()["provenance_entries"]
        engine.create_template(name="Prov Template")
        entries_after = engine.get_statistics()["provenance_entries"]
        assert entries_after > entries_before

    def test_apply_template_adds_provenance(self, engine: SLADefinitionEngine):
        """Applying a template should add provenance entries."""
        tmpl = engine.create_template(name="Apply Prov")
        entries_before = engine.get_statistics()["provenance_entries"]
        engine.apply_template(tmpl.id, ["ds-x", "ds-y"])
        entries_after = engine.get_statistics()["provenance_entries"]
        # Should add entries for each created SLA + 1 for the batch operation
        assert entries_after > entries_before

    def test_provenance_hashes_are_unique(self, engine: SLADefinitionEngine):
        """Different SLAs should have different provenance hashes."""
        sla1 = engine.create_sla(dataset_id="ds-u1")
        sla2 = engine.create_sla(dataset_id="ds-u2")
        assert sla1.provenance_hash != sla2.provenance_hash

    def test_provenance_hash_changes_on_update(self, engine: SLADefinitionEngine):
        """Updating an SLA should change its provenance hash."""
        sla = engine.create_sla(dataset_id="ds-hc")
        original_hash = sla.provenance_hash
        updated = engine.update_sla(sla.id, warning_hours=12.0)
        assert updated.provenance_hash != original_hash

    def test_provenance_hash_is_64_char_hex(self, engine: SLADefinitionEngine):
        """Provenance hash should be a valid 64-character hex string (SHA-256)."""
        sla = engine.create_sla(dataset_id="ds-hex")
        assert len(sla.provenance_hash) == 64
        int(sla.provenance_hash, 16)  # Should not raise


# ===========================================================================
# Test Class: Edge Cases and Integration
# ===========================================================================


class TestEdgeCases:
    """Edge case and cross-method integration tests."""

    def test_very_small_thresholds(self, engine: SLADefinitionEngine):
        """Very small but valid thresholds should be accepted."""
        sla = engine.create_sla(
            dataset_id="ds-small",
            warning_hours=0.001,
            critical_hours=0.002,
        )
        assert sla.warning_hours == 0.001
        assert sla.critical_hours == 0.002

    def test_very_large_thresholds(self, engine: SLADefinitionEngine):
        """Very large thresholds should be accepted."""
        sla = engine.create_sla(
            dataset_id="ds-big",
            warning_hours=8760.0,  # 1 year
            critical_hours=17520.0,  # 2 years
        )
        assert sla.warning_hours == 8760.0
        assert sla.critical_hours == 17520.0

    def test_many_slas(self, engine: SLADefinitionEngine):
        """Creating many SLAs should work without errors."""
        for i in range(50):
            engine.create_sla(
                dataset_id=f"ds-batch-{i}",
                warning_hours=float(i + 1),
                critical_hours=float(i + 100),
            )
        stats = engine.get_statistics()
        assert stats["total_slas"] == 50
        assert stats["total_datasets_covered"] == 50

    def test_apply_template_all_existing_datasets(
        self, engine: SLADefinitionEngine,
    ):
        """Applying template when all datasets already have SLAs returns empty."""
        engine.create_sla(dataset_id="ds-e1")
        engine.create_sla(dataset_id="ds-e2")
        tmpl = engine.create_template(name="NoOp")
        slas = engine.apply_template(tmpl.id, ["ds-e1", "ds-e2"])
        assert len(slas) == 0

    def test_evaluate_after_update(self, engine: SLADefinitionEngine):
        """Evaluation should use updated thresholds after an SLA update."""
        sla = engine.create_sla(
            dataset_id="ds-eu",
            warning_hours=24.0,
            critical_hours=72.0,
        )
        # Before update: 36h -> WARNING (24 < 36 <= 72)
        assert engine.evaluate_sla(sla.id, 36.0) == SLAStatus.WARNING
        # Update warning to 48 hours
        engine.update_sla(sla.id, warning_hours=48.0)
        # After update: 36h -> COMPLIANT (36 <= 48)
        assert engine.evaluate_sla(sla.id, 36.0) == SLAStatus.COMPLIANT

    def test_sla_to_dict_roundtrip(self, engine: SLADefinitionEngine):
        """SLA serialization should include all essential fields."""
        sla = engine.create_sla(
            dataset_id="ds-rt",
            warning_hours=10.0,
            critical_hours=40.0,
            breach_severity="high",
            escalation_policy="escalate",
            business_hours_only=True,
        )
        d = _to_dict(sla)
        assert d["dataset_id"] == "ds-rt"
        assert d["warning_hours"] == 10.0
        assert d["critical_hours"] == 40.0
        assert d["business_hours_only"] is True
        assert d["provenance_hash"] is not None
