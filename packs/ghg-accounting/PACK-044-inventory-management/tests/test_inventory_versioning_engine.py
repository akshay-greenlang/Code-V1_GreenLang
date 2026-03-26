# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Inventory Versioning Engine Tests
=========================================================

Tests InventoryVersioningEngine: version creation, lifecycle transitions,
field-level diffs, rollback, optimistic locking, and version chains.

Target: 60+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("inventory_versioning")

InventoryVersioningEngine = _mod.InventoryVersioningEngine
InventoryVersion = _mod.InventoryVersion
VersionDiff = _mod.VersionDiff
FieldChange = _mod.FieldChange
VersionComparison = _mod.VersionComparison
VersioningResult = _mod.VersioningResult
VersionStatus = _mod.VersionStatus
ChangeType = _mod.ChangeType


# ===================================================================
# Fixtures
# ===================================================================

SAMPLE_DATA_V1 = {
    "scope1_total": 10000,
    "scope2_location": 5000,
    "scope2_market": 3000,
    "total_scope12": 15000,
    "data_quality_score": 82,
    "completeness_pct": 95,
}

SAMPLE_DATA_V2 = {
    "scope1_total": 9500,
    "scope2_location": 5200,
    "scope2_market": 3100,
    "total_scope12": 14700,
    "data_quality_score": 85,
    "completeness_pct": 97,
}


@pytest.fixture
def engine():
    """Create a fresh InventoryVersioningEngine."""
    return InventoryVersioningEngine()


@pytest.fixture
def v1(engine):
    """Create the first version and return (engine, result)."""
    result = engine.create_version(
        inventory_id="inv-2025-001",
        reporting_year=2025,
        data=dict(SAMPLE_DATA_V1),
        created_by="user-analyst-001",
        created_by_name="Data Analyst",
        notes="Initial draft",
    )
    return engine, result


# ===================================================================
# Version Creation Tests
# ===================================================================


class TestVersionCreation:
    """Tests for create_version."""

    def test_create_returns_result(self, v1):
        _, result = v1
        assert isinstance(result, VersioningResult)
        assert result.action == "create"

    def test_version_number_is_one(self, v1):
        _, result = v1
        assert result.version.version_number == 1

    def test_version_status_draft(self, v1):
        _, result = v1
        assert result.version.status == VersionStatus.DRAFT

    def test_version_data_stored(self, v1):
        _, result = v1
        assert result.version.data["scope1_total"] == 10000

    def test_version_provenance_hash(self, v1):
        _, result = v1
        assert len(result.version.provenance_hash) == 64

    def test_version_label_default(self, v1):
        _, result = v1
        assert "v1" in result.version.label.lower()

    def test_version_created_by(self, v1):
        _, result = v1
        assert result.version.created_by == "user-analyst-001"

    def test_result_provenance_hash(self, v1):
        _, result = v1
        assert len(result.provenance_hash) == 64


# ===================================================================
# Next Version Tests
# ===================================================================


class TestNextVersion:
    """Tests for create_next_version (incremental versions)."""

    def test_create_next_version(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        assert r2.version.version_number == 2

    def test_next_version_links_to_previous(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        assert r2.version.previous_version_id == r1.version.version_id

    def test_next_version_inherits_inventory_id(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        assert r2.version.inventory_id == r1.version.inventory_id

    def test_next_version_data_updated(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        assert r2.version.data["scope1_total"] == 9500

    def test_multiple_versions(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        data_v3 = dict(SAMPLE_DATA_V2)
        data_v3["scope1_total"] = 9000
        r3 = engine.create_next_version(r2.version, data_v3)
        assert r3.version.version_number == 3


# ===================================================================
# Version Lifecycle Tests
# ===================================================================


class TestVersionLifecycle:
    """Tests for version status transitions."""

    def test_draft_to_under_review(self, v1):
        engine, r1 = v1
        r = engine.transition_status(
            r1.version, VersionStatus.UNDER_REVIEW,
            "user-001", "User",
        )
        assert r.version.status == VersionStatus.UNDER_REVIEW

    def test_under_review_to_final(self, v1):
        engine, r1 = v1
        engine.transition_status(
            r1.version, VersionStatus.UNDER_REVIEW,
            "u1", "User 1",
        )
        r = engine.transition_status(
            r1.version, VersionStatus.FINAL,
            "u2", "User 2",
        )
        assert r.version.status == VersionStatus.FINAL

    def test_final_sets_finalised_timestamp(self, v1):
        engine, r1 = v1
        engine.transition_status(
            r1.version, VersionStatus.UNDER_REVIEW,
            "u1", "User 1",
        )
        r = engine.transition_status(
            r1.version, VersionStatus.FINAL,
            "u2", "User 2",
        )
        assert r.version.finalised_at is not None
        assert r.version.finalised_by == "u2"

    def test_final_to_amended(self, v1):
        engine, r1 = v1
        v = r1.version
        engine.transition_status(v, VersionStatus.UNDER_REVIEW, "u1", "U1")
        engine.transition_status(v, VersionStatus.FINAL, "u2", "U2")
        r = engine.transition_status(v, VersionStatus.AMENDED, "u3", "U3")
        assert r.version.status == VersionStatus.AMENDED

    def test_under_review_back_to_draft(self, v1):
        engine, r1 = v1
        v = r1.version
        engine.transition_status(v, VersionStatus.UNDER_REVIEW, "u1", "U1")
        r = engine.transition_status(v, VersionStatus.DRAFT, "u1", "U1")
        assert r.version.status == VersionStatus.DRAFT

    def test_invalid_transition_raises(self, v1):
        engine, r1 = v1
        with pytest.raises((ValueError, Exception)):
            engine.transition_status(
                r1.version, VersionStatus.SUPERSEDED, "u1", "U1"
            )

    def test_immutable_after_final(self, v1):
        engine, r1 = v1
        v = r1.version
        engine.transition_status(v, VersionStatus.UNDER_REVIEW, "u1", "U1")
        engine.transition_status(v, VersionStatus.FINAL, "u2", "U2")
        with pytest.raises((ValueError, Exception)):
            engine.update_data(
                v, {"scope1_total": 999},
                "u3", "U3",
                expected_lock_version=v.lock_version,
            )


# ===================================================================
# Diff Computation Tests
# ===================================================================


class TestDiffComputation:
    """Tests for compute_diff."""

    def test_diff_identifies_changes(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        diff_result = engine.compute_diff(r1.version, r2.version)
        assert diff_result.diff is not None
        assert diff_result.diff.fields_modified > 0

    def test_diff_no_changes(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, dict(SAMPLE_DATA_V1))
        diff_result = engine.compute_diff(r1.version, r2.version)
        assert diff_result.diff.fields_modified == 0

    def test_diff_field_changes_list(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        diff_result = engine.compute_diff(r1.version, r2.version)
        assert len(diff_result.diff.changes) > 0

    def test_diff_change_types(self, v1):
        engine, r1 = v1
        data_with_new_field = dict(SAMPLE_DATA_V2)
        data_with_new_field["new_field"] = 42
        r2 = engine.create_next_version(r1.version, data_with_new_field)
        diff_result = engine.compute_diff(r1.version, r2.version)
        change_types = set()
        for c in diff_result.diff.changes:
            ct = c.change_type
            change_types.add(ct.value if hasattr(ct, "value") else str(ct))
        assert "added" in change_types or "modified" in change_types

    def test_diff_summary_populated(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        diff_result = engine.compute_diff(r1.version, r2.version)
        assert diff_result.diff.summary != ""

    def test_diff_provenance_hash(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        diff_result = engine.compute_diff(r1.version, r2.version)
        assert len(diff_result.diff.provenance_hash) == 64


# ===================================================================
# Rollback Tests
# ===================================================================


class TestRollback:
    """Tests for version rollback."""

    def test_rollback_to_previous(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        rollback_result = engine.rollback(r2.version, r1.version)
        assert rollback_result.version is not None
        assert rollback_result.version.data["scope1_total"] == SAMPLE_DATA_V1["scope1_total"]

    def test_rollback_creates_new_version(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        rollback_result = engine.rollback(r2.version, r1.version)
        assert rollback_result.version.version_number >= 3

    def test_rollback_source_stored(self, v1):
        engine, r1 = v1
        r2 = engine.create_next_version(r1.version, SAMPLE_DATA_V2)
        rollback_result = engine.rollback(r2.version, r1.version)
        assert rollback_result.rollback_source is not None


# ===================================================================
# Optimistic Locking Tests
# ===================================================================


class TestOptimisticLocking:
    """Tests for concurrency control via lock_version."""

    def test_lock_version_starts_at_one(self, v1):
        _, result = v1
        assert result.version.lock_version == 1

    def test_update_increments_lock_version(self, v1):
        engine, r1 = v1
        v = r1.version
        r = engine.update_data(
            v, {"scope1_total": 9800},
            "user-001", "User",
            expected_lock_version=1,
        )
        assert r.version.lock_version == 2

    def test_stale_lock_version_raises(self, v1):
        engine, r1 = v1
        v = r1.version
        engine.update_data(v, {"scope1_total": 9800}, "u1", "U1", expected_lock_version=1)
        with pytest.raises((ValueError, Exception)):
            engine.update_data(v, {"scope1_total": 9700}, "u2", "U2", expected_lock_version=1)


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults and enum values."""

    @pytest.mark.parametrize("status", list(VersionStatus))
    def test_version_statuses(self, status):
        assert status.value is not None

    def test_inventory_version_defaults(self):
        v = InventoryVersion(
            inventory_id="inv-001",
            reporting_year=2025,
            data={},
        )
        assert v.status == VersionStatus.DRAFT
        assert v.version_number == 1
        assert v.lock_version == 1

    def test_field_change_defaults(self):
        fc = FieldChange(field_path="test_field")
        assert fc.change_type == ChangeType.UNCHANGED

    def test_version_diff_defaults(self):
        vd = VersionDiff()
        assert vd.fields_modified == 0

    def test_version_comparison_defaults(self):
        vc = VersionComparison()
        assert vc.is_material_change is False

    def test_versioning_result_defaults(self):
        vr = VersioningResult()
        assert vr.action == ""
        assert vr.processing_time_ms == Decimal("0")
