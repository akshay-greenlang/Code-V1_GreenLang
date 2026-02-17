# -*- coding: utf-8 -*-
"""
Unit tests for DatasetRegistryEngine - AGENT-DATA-016 Engine 1.

Tests all public methods of DatasetRegistryEngine with 85%+ coverage.
Validates business logic, error handling, edge cases, provenance tracking,
group management, bulk operations, and aggregate statistics.

Target: 70+ tests.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from greenlang.data_freshness_monitor.dataset_registry import (
    DatasetRegistryEngine,
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
def engine() -> DatasetRegistryEngine:
    """Create a fresh DatasetRegistryEngine instance for each test."""
    return DatasetRegistryEngine()


@pytest.fixture
def populated_engine(engine: DatasetRegistryEngine):
    """Engine pre-populated with 3 datasets from different sources."""
    engine.register_dataset(
        name="ERP Spend Data",
        source_name="SAP",
        source_type="erp",
        owner="finance-team",
        refresh_cadence="daily",
        priority="high",
        tags=["finance", "erp"],
    )
    engine.register_dataset(
        name="Cloud Storage Logs",
        source_name="S3",
        source_type="file",
        owner="infra-team",
        refresh_cadence="hourly",
        priority="medium",
    )
    engine.register_dataset(
        name="Warehouse Sales",
        source_name="Snowflake",
        source_type="warehouse",
        owner="analytics-team",
        refresh_cadence="weekly",
        priority="low",
    )
    return engine


# ===========================================================================
# Test Class: register_dataset
# ===========================================================================


class TestRegisterDataset:
    """Tests for DatasetRegistryEngine.register_dataset."""

    def test_register_basic(self, engine: DatasetRegistryEngine):
        """Register a dataset with minimal required fields."""
        ds = engine.register_dataset(name="Test", source_name="SAP")
        assert ds.id is not None
        assert len(ds.id) >= 12
        assert ds.name == "Test"
        assert ds.source_name == "SAP"

    def test_register_all_fields(self, engine: DatasetRegistryEngine):
        """Register a dataset with all optional fields populated."""
        ds = engine.register_dataset(
            name="Full Dataset",
            source_name="Oracle",
            source_type="database",
            owner="dba-team",
            refresh_cadence="hourly",
            priority="critical",
            tags=["tag1", "tag2"],
            metadata={"region": "us-east-1"},
        )
        assert ds.name == "Full Dataset"
        assert ds.source_name == "Oracle"
        assert ds.source_type == "database"
        assert ds.owner == "dba-team"
        assert _val(ds.refresh_cadence) == "hourly"
        assert _val(ds.priority) == "critical"
        assert list(ds.tags) == ["tag1", "tag2"]
        assert dict(ds.metadata) == {"region": "us-east-1"}

    def test_register_status_is_active(self, engine: DatasetRegistryEngine):
        """Newly registered dataset status should be active."""
        ds = engine.register_dataset(name="A", source_name="S3")
        assert _val(ds.status) == "active"

    def test_register_provenance_hash_set(self, engine: DatasetRegistryEngine):
        """Provenance hash should be a 64-char SHA-256 hex string."""
        ds = engine.register_dataset(name="B", source_name="SAP")
        assert ds.provenance_hash is not None
        assert len(ds.provenance_hash) == 64

    def test_register_increments_dataset_count(self, engine: DatasetRegistryEngine):
        """Registering a dataset should increment the dataset count."""
        assert engine.get_dataset_count() == 0
        engine.register_dataset(name="X", source_name="S3")
        assert engine.get_dataset_count() == 1
        engine.register_dataset(name="Y", source_name="S3")
        assert engine.get_dataset_count() == 2

    def test_register_increments_operation_count(self, engine: DatasetRegistryEngine):
        """Each registration should increment the operation counter."""
        assert engine.operation_count == 0
        engine.register_dataset(name="X", source_name="S3")
        assert engine.operation_count == 1

    @pytest.mark.parametrize("cadence", [
        "realtime", "hourly", "daily", "weekly",
        "monthly", "quarterly", "annual",
    ])
    def test_register_all_valid_cadences(
        self, engine: DatasetRegistryEngine, cadence: str,
    ):
        """All seven valid cadences should be accepted."""
        ds = engine.register_dataset(
            name=f"DS-{cadence}", source_name="SAP", refresh_cadence=cadence,
        )
        assert _val(ds.refresh_cadence) == cadence

    @pytest.mark.parametrize("priority", ["critical", "high", "medium", "low"])
    def test_register_all_valid_priorities(
        self, engine: DatasetRegistryEngine, priority: str,
    ):
        """All four valid priorities should be accepted."""
        ds = engine.register_dataset(
            name=f"DS-{priority}", source_name="SAP", priority=priority,
        )
        assert _val(ds.priority) == priority

    def test_register_case_insensitive_cadence(self, engine: DatasetRegistryEngine):
        """Cadence should be case-insensitive and stored as lowercase."""
        ds = engine.register_dataset(
            name="A", source_name="SAP", refresh_cadence="DAILY",
        )
        assert _val(ds.refresh_cadence) == "daily"

    def test_register_case_insensitive_priority(self, engine: DatasetRegistryEngine):
        """Priority should be case-insensitive and stored as lowercase."""
        ds = engine.register_dataset(
            name="A", source_name="SAP", priority="HIGH",
        )
        assert _val(ds.priority) == "high"

    def test_register_strips_whitespace_name(self, engine: DatasetRegistryEngine):
        """Name with leading/trailing whitespace should be stripped."""
        ds = engine.register_dataset(name="  Trimmed  ", source_name="SAP")
        assert ds.name == "Trimmed"

    def test_register_empty_name_raises(self, engine: DatasetRegistryEngine):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="name must not be empty"):
            engine.register_dataset(name="", source_name="SAP")

    def test_register_whitespace_name_raises(self, engine: DatasetRegistryEngine):
        """Whitespace-only name should raise ValueError."""
        with pytest.raises(ValueError, match="name must not be empty"):
            engine.register_dataset(name="   ", source_name="SAP")

    def test_register_empty_source_raises(self, engine: DatasetRegistryEngine):
        """Empty source_name should raise ValueError."""
        with pytest.raises(ValueError, match="Source name must not be empty"):
            engine.register_dataset(name="A", source_name="")

    def test_register_invalid_cadence_raises(self, engine: DatasetRegistryEngine):
        """Invalid cadence should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid refresh_cadence"):
            engine.register_dataset(
                name="A", source_name="SAP", refresh_cadence="biweekly",
            )

    def test_register_invalid_priority_raises(self, engine: DatasetRegistryEngine):
        """Invalid priority should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            engine.register_dataset(
                name="A", source_name="SAP", priority="urgent",
            )

    def test_register_default_cadence_is_daily(self, engine: DatasetRegistryEngine):
        """Default cadence should be daily when not specified."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        assert _val(ds.refresh_cadence) == "daily"

    def test_register_default_priority_is_medium(self, engine: DatasetRegistryEngine):
        """Default priority should be medium when not specified."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        assert _val(ds.priority) == "medium"

    def test_register_provenance_chain_grows(self, engine: DatasetRegistryEngine):
        """Each registration should add an entry to the provenance chain."""
        assert engine.provenance_chain_length == 0
        engine.register_dataset(name="A", source_name="SAP")
        assert engine.provenance_chain_length == 1
        engine.register_dataset(name="B", source_name="S3")
        assert engine.provenance_chain_length == 2

    def test_register_unique_ids(self, engine: DatasetRegistryEngine):
        """Each registered dataset should get a unique ID."""
        ds1 = engine.register_dataset(name="A", source_name="SAP")
        ds2 = engine.register_dataset(name="B", source_name="SAP")
        assert ds1.id != ds2.id

    def test_register_unique_provenance_hashes(self, engine: DatasetRegistryEngine):
        """Each registration should get a unique provenance hash."""
        ds1 = engine.register_dataset(name="A", source_name="SAP")
        ds2 = engine.register_dataset(name="B", source_name="SAP")
        assert ds1.provenance_hash != ds2.provenance_hash


# ===========================================================================
# Test Class: get_dataset
# ===========================================================================


class TestGetDataset:
    """Tests for DatasetRegistryEngine.get_dataset."""

    def test_get_existing_dataset(self, engine: DatasetRegistryEngine):
        """Getting an existing dataset should return its definition."""
        ds = engine.register_dataset(name="Get Me", source_name="S3")
        retrieved = engine.get_dataset(ds.id)
        assert retrieved is not None
        assert retrieved.name == "Get Me"
        assert retrieved.id == ds.id

    def test_get_nonexistent_returns_none(self, engine: DatasetRegistryEngine):
        """Getting a nonexistent dataset ID should return None."""
        assert engine.get_dataset("nonexistent-id") is None

    def test_get_after_removal_returns_none(self, engine: DatasetRegistryEngine):
        """Getting a removed dataset should return None."""
        ds = engine.register_dataset(name="Temp", source_name="SAP")
        engine.remove_dataset(ds.id)
        assert engine.get_dataset(ds.id) is None

    def test_get_preserves_all_fields(self, engine: DatasetRegistryEngine):
        """Getting a dataset should preserve all registered fields."""
        ds = engine.register_dataset(
            name="Full", source_name="SAP", source_type="erp",
            owner="team", tags=["a"],
        )
        retrieved = engine.get_dataset(ds.id)
        assert retrieved.name == "Full"
        assert retrieved.source_name == "SAP"
        assert retrieved.owner == "team"


# ===========================================================================
# Test Class: list_datasets
# ===========================================================================


class TestListDatasets:
    """Tests for DatasetRegistryEngine.list_datasets."""

    def test_list_empty(self, engine: DatasetRegistryEngine):
        """Listing with no registered datasets returns empty list."""
        assert engine.list_datasets() == []

    def test_list_all(self, populated_engine: DatasetRegistryEngine):
        """Listing all datasets should return all 3."""
        result = populated_engine.list_datasets()
        assert len(result) == 3

    def test_list_sorted_by_name(self, populated_engine: DatasetRegistryEngine):
        """Datasets should be sorted by name ascending."""
        result = populated_engine.list_datasets()
        names = [d.name for d in result]
        assert names == sorted(names, key=str.lower)

    def test_list_filter_by_priority(self, populated_engine: DatasetRegistryEngine):
        """Filtering by priority should return matching datasets only."""
        high = populated_engine.list_datasets(priority="high")
        assert len(high) == 1
        assert high[0].name == "ERP Spend Data"

    def test_list_filter_by_status(self, populated_engine: DatasetRegistryEngine):
        """Filtering by status should return matching datasets."""
        active = populated_engine.list_datasets(status="active")
        assert len(active) == 3

    def test_list_filter_by_source_name(self, populated_engine: DatasetRegistryEngine):
        """Filtering by source_name should substring match."""
        sap_datasets = populated_engine.list_datasets(source_name="SAP")
        assert len(sap_datasets) == 1

    def test_list_filter_by_cadence(self, populated_engine: DatasetRegistryEngine):
        """Filtering by cadence should return matching datasets."""
        daily = populated_engine.list_datasets(cadence="daily")
        assert len(daily) == 1

    def test_list_with_limit(self, populated_engine: DatasetRegistryEngine):
        """Limit should cap the number of returned datasets."""
        result = populated_engine.list_datasets(limit=2)
        assert len(result) == 2

    def test_list_with_offset(self, populated_engine: DatasetRegistryEngine):
        """Offset should skip the first N results."""
        all_ds = populated_engine.list_datasets()
        offset_ds = populated_engine.list_datasets(offset=1)
        assert len(offset_ds) == len(all_ds) - 1

    def test_list_combined_filters(self, populated_engine: DatasetRegistryEngine):
        """Multiple filters should combine with AND logic."""
        result = populated_engine.list_datasets(
            status="active", priority="low",
        )
        assert len(result) == 1
        assert result[0].name == "Warehouse Sales"

    def test_list_no_match_returns_empty(self, populated_engine: DatasetRegistryEngine):
        """Filters matching nothing should return empty list."""
        result = populated_engine.list_datasets(priority="critical")
        assert result == []

    def test_list_limit_and_offset_combined(self, populated_engine: DatasetRegistryEngine):
        """Limit and offset should work together."""
        result = populated_engine.list_datasets(offset=1, limit=1)
        assert len(result) == 1


# ===========================================================================
# Test Class: update_dataset
# ===========================================================================


class TestUpdateDataset:
    """Tests for DatasetRegistryEngine.update_dataset."""

    def test_update_name(self, engine: DatasetRegistryEngine):
        """Updating name should change the name field."""
        ds = engine.register_dataset(name="Old", source_name="SAP")
        updated = engine.update_dataset(ds.id, name="New")
        assert updated.name == "New"

    def test_update_priority(self, engine: DatasetRegistryEngine):
        """Updating priority should change the priority field."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(ds.id, priority="critical")
        assert _val(updated.priority) == "critical"

    def test_update_bumps_version(self, engine: DatasetRegistryEngine):
        """Each update should increment the version counter."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        initial_version = ds.version
        updated = engine.update_dataset(ds.id, priority="high")
        assert updated.version == initial_version + 1

    def test_update_changes_provenance_hash(self, engine: DatasetRegistryEngine):
        """Updating a dataset should produce a new provenance hash."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        old_hash = ds.provenance_hash
        updated = engine.update_dataset(ds.id, priority="high")
        assert updated.provenance_hash != old_hash

    def test_update_nonexistent_raises_key_error(self, engine: DatasetRegistryEngine):
        """Updating a nonexistent dataset should raise KeyError."""
        with pytest.raises(KeyError, match="Dataset not found"):
            engine.update_dataset("nonexistent", name="X")

    def test_update_invalid_field_raises(self, engine: DatasetRegistryEngine):
        """Updating unsupported fields should raise ValueError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(ValueError, match="Cannot update fields"):
            engine.update_dataset(ds.id, id="new-id")

    def test_update_empty_name_raises(self, engine: DatasetRegistryEngine):
        """Updating name to empty should raise ValueError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(ValueError, match="name must not be empty"):
            engine.update_dataset(ds.id, name="")

    def test_update_invalid_cadence_raises(self, engine: DatasetRegistryEngine):
        """Updating cadence to invalid value should raise ValueError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(ValueError, match="Invalid refresh_cadence"):
            engine.update_dataset(ds.id, refresh_cadence="biweekly")

    def test_update_invalid_priority_raises(self, engine: DatasetRegistryEngine):
        """Updating priority to invalid value should raise ValueError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(ValueError, match="Invalid priority"):
            engine.update_dataset(ds.id, priority="super-high")

    def test_update_invalid_status_raises(self, engine: DatasetRegistryEngine):
        """Updating status to invalid value should raise ValueError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(ValueError, match="Invalid status"):
            engine.update_dataset(ds.id, status="unknown")

    def test_update_status_to_stale(self, engine: DatasetRegistryEngine):
        """Status can be updated to stale."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(ds.id, status="stale")
        assert _val(updated.status) == "stale"

    def test_update_tags(self, engine: DatasetRegistryEngine):
        """Tags can be updated."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(ds.id, tags=["new-tag"])
        assert list(updated.tags) == ["new-tag"]

    def test_update_metadata(self, engine: DatasetRegistryEngine):
        """Metadata can be updated."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(ds.id, metadata={"key": "val"})
        assert dict(updated.metadata) == {"key": "val"}

    def test_update_source_name(self, engine: DatasetRegistryEngine):
        """Source name can be updated."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(ds.id, source_name="Oracle")
        assert updated.source_name == "Oracle"

    def test_update_multiple_fields(self, engine: DatasetRegistryEngine):
        """Updating multiple fields at once should work."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        updated = engine.update_dataset(
            ds.id, name="B", priority="high", status="inactive",
        )
        assert updated.name == "B"
        assert _val(updated.priority) == "high"
        assert _val(updated.status) == "inactive"


# ===========================================================================
# Test Class: remove_dataset
# ===========================================================================


class TestRemoveDataset:
    """Tests for DatasetRegistryEngine.remove_dataset."""

    def test_remove_existing(self, engine: DatasetRegistryEngine):
        """Removing an existing dataset should return True."""
        ds = engine.register_dataset(name="ToRemove", source_name="SAP")
        assert engine.remove_dataset(ds.id) is True
        assert engine.get_dataset_count() == 0

    def test_remove_nonexistent_returns_false(self, engine: DatasetRegistryEngine):
        """Removing a nonexistent dataset should return False."""
        assert engine.remove_dataset("nonexistent") is False

    def test_remove_clears_refresh_history(self, engine: DatasetRegistryEngine):
        """Removing a dataset should also remove its refresh history."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        engine.record_refresh(ds.id)
        engine.remove_dataset(ds.id)
        assert engine.get_refresh_history(ds.id) == []

    def test_remove_detaches_from_groups(self, engine: DatasetRegistryEngine):
        """Removing a dataset should detach it from all groups."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1", dataset_ids=[ds.id])
        engine.remove_dataset(ds.id)
        refreshed_grp = engine.get_group(grp.id)
        assert ds.id not in refreshed_grp.dataset_ids

    def test_remove_twice_returns_false_second_time(
        self, engine: DatasetRegistryEngine,
    ):
        """Removing an already removed dataset should return False."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        assert engine.remove_dataset(ds.id) is True
        assert engine.remove_dataset(ds.id) is False


# ===========================================================================
# Test Class: record_refresh / get_refresh_history / get_last_refresh
# ===========================================================================


class TestRefreshEvents:
    """Tests for refresh event recording and retrieval."""

    def test_record_refresh_basic(self, engine: DatasetRegistryEngine):
        """Recording a refresh should return a RefreshEvent."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        event = engine.record_refresh(ds.id)
        assert event.dataset_id == ds.id
        assert event.id is not None
        assert len(event.provenance_hash) == 64

    def test_record_refresh_with_size_and_count(self, engine: DatasetRegistryEngine):
        """Recording with size and count should store those values."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        event = engine.record_refresh(
            ds.id, data_size_bytes=1024, record_count=500,
        )
        assert event.data_size_bytes == 1024
        assert event.record_count == 500

    def test_record_refresh_with_custom_timestamp(self, engine: DatasetRegistryEngine):
        """Recording with a custom refreshed_at timestamp should use it."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        custom_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        event = engine.record_refresh(ds.id, refreshed_at=custom_time)
        assert event.refreshed_at == custom_time

    def test_record_refresh_with_source_info(self, engine: DatasetRegistryEngine):
        """Source info dict should be recorded."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        event = engine.record_refresh(
            ds.id, source_info={"pipeline_run": "abc123"},
        )
        assert event.source_info == {"pipeline_run": "abc123"}

    def test_record_refresh_unknown_dataset_raises(self, engine: DatasetRegistryEngine):
        """Recording a refresh for unknown dataset should raise KeyError."""
        with pytest.raises(KeyError, match="Dataset not found"):
            engine.record_refresh("nonexistent")

    def test_get_refresh_history_empty(self, engine: DatasetRegistryEngine):
        """Getting history for dataset with no refreshes returns empty."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        assert engine.get_refresh_history(ds.id) == []

    def test_get_refresh_history_ordered_desc(self, engine: DatasetRegistryEngine):
        """History should be sorted by refreshed_at descending."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        engine.record_refresh(ds.id, refreshed_at=t1)
        engine.record_refresh(ds.id, refreshed_at=t3)
        engine.record_refresh(ds.id, refreshed_at=t2)
        history = engine.get_refresh_history(ds.id)
        assert len(history) == 3
        assert history[0].refreshed_at == t3
        assert history[1].refreshed_at == t2
        assert history[2].refreshed_at == t1

    def test_get_refresh_history_with_limit(self, engine: DatasetRegistryEngine):
        """Limit should cap the number of returned events."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        for i in range(5):
            engine.record_refresh(ds.id)
        history = engine.get_refresh_history(ds.id, limit=2)
        assert len(history) == 2

    def test_get_refresh_history_nonexistent_returns_empty(
        self, engine: DatasetRegistryEngine,
    ):
        """Getting history for unknown dataset returns empty list."""
        assert engine.get_refresh_history("no-such-id") == []

    def test_get_last_refresh_none_when_no_events(
        self, engine: DatasetRegistryEngine,
    ):
        """get_last_refresh returns None when no refresh events exist."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        assert engine.get_last_refresh(ds.id) is None

    def test_get_last_refresh_returns_latest(self, engine: DatasetRegistryEngine):
        """get_last_refresh returns the most recent refresh timestamp."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        t1 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        engine.record_refresh(ds.id, refreshed_at=t1)
        engine.record_refresh(ds.id, refreshed_at=t2)
        assert engine.get_last_refresh(ds.id) == t2

    def test_record_refresh_increments_operation_count(
        self, engine: DatasetRegistryEngine,
    ):
        """Each refresh recording should increment operation count."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        ops_before = engine.operation_count
        engine.record_refresh(ds.id)
        assert engine.operation_count == ops_before + 1


# ===========================================================================
# Test Class: Groups
# ===========================================================================


class TestGroups:
    """Tests for group management methods."""

    def test_create_group_basic(self, engine: DatasetRegistryEngine):
        """Creating a group should return a DatasetGroup."""
        grp = engine.create_group(name="Finance")
        assert grp.name == "Finance"
        assert grp.id is not None
        assert list(grp.dataset_ids) == []

    def test_create_group_with_datasets(self, engine: DatasetRegistryEngine):
        """Creating a group with existing dataset IDs should include them."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1", dataset_ids=[ds.id])
        assert ds.id in grp.dataset_ids

    def test_create_group_filters_nonexistent_ids(
        self, engine: DatasetRegistryEngine,
    ):
        """Nonexistent dataset IDs should be silently filtered out."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(
            name="G1", dataset_ids=[ds.id, "nonexistent"],
        )
        assert len(grp.dataset_ids) == 1
        assert ds.id in grp.dataset_ids

    def test_create_group_empty_name_raises(self, engine: DatasetRegistryEngine):
        """Empty group name should raise ValueError."""
        with pytest.raises(ValueError, match="Group name must not be empty"):
            engine.create_group(name="")

    def test_create_group_invalid_priority_raises(
        self, engine: DatasetRegistryEngine,
    ):
        """Invalid priority should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid priority"):
            engine.create_group(name="G1", priority="super-high")

    def test_create_group_with_sla_id(self, engine: DatasetRegistryEngine):
        """SLA ID should be stored on the group."""
        grp = engine.create_group(name="G1", sla_id="sla-abc")
        assert grp.sla_id == "sla-abc"

    def test_get_group_existing(self, engine: DatasetRegistryEngine):
        """Getting an existing group should return it."""
        grp = engine.create_group(name="G1")
        retrieved = engine.get_group(grp.id)
        assert retrieved is not None
        assert retrieved.name == "G1"

    def test_get_group_nonexistent(self, engine: DatasetRegistryEngine):
        """Getting a nonexistent group should return None."""
        assert engine.get_group("no-such-group") is None

    def test_list_groups_empty(self, engine: DatasetRegistryEngine):
        """Listing groups when none exist returns empty list."""
        assert engine.list_groups() == []

    def test_list_groups_sorted_by_name(self, engine: DatasetRegistryEngine):
        """Groups should be sorted by name ascending."""
        engine.create_group(name="Zulu")
        engine.create_group(name="Alpha")
        groups = engine.list_groups()
        assert groups[0].name == "Alpha"
        assert groups[1].name == "Zulu"

    def test_add_to_group(self, engine: DatasetRegistryEngine):
        """Adding a dataset to a group should include it in dataset_ids."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1")
        updated = engine.add_to_group(grp.id, ds.id)
        assert ds.id in updated.dataset_ids

    def test_add_to_group_idempotent(self, engine: DatasetRegistryEngine):
        """Adding the same dataset twice should be a no-op."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1", dataset_ids=[ds.id])
        updated = engine.add_to_group(grp.id, ds.id)
        assert list(updated.dataset_ids).count(ds.id) == 1

    def test_add_to_group_nonexistent_group_raises(
        self, engine: DatasetRegistryEngine,
    ):
        """Adding to a nonexistent group should raise KeyError."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        with pytest.raises(KeyError, match="Group not found"):
            engine.add_to_group("no-group", ds.id)

    def test_add_to_group_nonexistent_dataset_raises(
        self, engine: DatasetRegistryEngine,
    ):
        """Adding a nonexistent dataset should raise KeyError."""
        grp = engine.create_group(name="G1")
        with pytest.raises(KeyError, match="Dataset not found"):
            engine.add_to_group(grp.id, "no-dataset")

    def test_remove_from_group(self, engine: DatasetRegistryEngine):
        """Removing a dataset from a group should exclude it."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1", dataset_ids=[ds.id])
        updated = engine.remove_from_group(grp.id, ds.id)
        assert ds.id not in updated.dataset_ids

    def test_remove_from_group_not_in_group_is_noop(
        self, engine: DatasetRegistryEngine,
    ):
        """Removing a dataset not in the group should be a no-op."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        grp = engine.create_group(name="G1")
        updated = engine.remove_from_group(grp.id, ds.id)
        assert ds.id not in updated.dataset_ids

    def test_remove_from_group_nonexistent_group_raises(
        self, engine: DatasetRegistryEngine,
    ):
        """Removing from a nonexistent group should raise KeyError."""
        with pytest.raises(KeyError, match="Group not found"):
            engine.remove_from_group("no-group", "ds-id")

    def test_create_group_with_description(self, engine: DatasetRegistryEngine):
        """Description should be stored on the group."""
        grp = engine.create_group(name="G1", description="Test group")
        assert grp.description == "Test group"


# ===========================================================================
# Test Class: bulk_register
# ===========================================================================


class TestBulkRegister:
    """Tests for DatasetRegistryEngine.bulk_register."""

    def test_bulk_register_multiple(self, engine: DatasetRegistryEngine):
        """Bulk registering multiple datasets should return all."""
        data = [
            {"name": "A", "source_name": "SAP", "source_type": "erp"},
            {"name": "B", "source_name": "S3", "source_type": "file"},
            {"name": "C", "source_name": "Oracle", "source_type": "database"},
        ]
        results = engine.bulk_register(data)
        assert len(results) == 3
        assert engine.get_dataset_count() == 3

    def test_bulk_register_skips_invalid(self, engine: DatasetRegistryEngine):
        """Invalid entries should be skipped; valid ones should succeed."""
        data = [
            {"name": "Valid", "source_name": "SAP"},
            {"name": "", "source_name": "SAP"},  # Invalid: empty name
        ]
        results = engine.bulk_register(data)
        assert len(results) == 1
        assert results[0].name == "Valid"

    def test_bulk_register_empty_list_raises(self, engine: DatasetRegistryEngine):
        """Bulk registering an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.bulk_register([])

    def test_bulk_register_all_invalid(self, engine: DatasetRegistryEngine):
        """If all entries are invalid, should return empty list."""
        data = [
            {"name": "", "source_name": "SAP"},
            {"name": "X", "source_name": ""},
        ]
        results = engine.bulk_register(data)
        assert len(results) == 0

    def test_bulk_register_with_optional_fields(
        self, engine: DatasetRegistryEngine,
    ):
        """Bulk register with optional fields should populate them."""
        data = [
            {
                "name": "Full",
                "source_name": "SAP",
                "refresh_cadence": "hourly",
                "priority": "critical",
                "tags": ["tag1"],
                "metadata": {"key": "val"},
            },
        ]
        results = engine.bulk_register(data)
        assert len(results) == 1
        assert _val(results[0].refresh_cadence) == "hourly"
        assert _val(results[0].priority) == "critical"

    def test_bulk_register_single_entry(self, engine: DatasetRegistryEngine):
        """Bulk registering a single entry should work."""
        data = [{"name": "Solo", "source_name": "S3"}]
        results = engine.bulk_register(data)
        assert len(results) == 1
        assert results[0].name == "Solo"


# ===========================================================================
# Test Class: get_datasets_by_source
# ===========================================================================


class TestGetDatasetsBySource:
    """Tests for DatasetRegistryEngine.get_datasets_by_source."""

    def test_get_by_source_matches(self, populated_engine: DatasetRegistryEngine):
        """Should return datasets matching the given source."""
        result = populated_engine.get_datasets_by_source("SAP")
        assert len(result) == 1
        assert result[0].source_name == "SAP"

    def test_get_by_source_case_insensitive(
        self, populated_engine: DatasetRegistryEngine,
    ):
        """Source matching should be case-insensitive."""
        result = populated_engine.get_datasets_by_source("sap")
        assert len(result) == 1

    def test_get_by_source_no_match(self, populated_engine: DatasetRegistryEngine):
        """Should return empty list when source does not match."""
        result = populated_engine.get_datasets_by_source("NonExistent")
        assert result == []

    def test_get_by_source_multiple_matches(self, engine: DatasetRegistryEngine):
        """Multiple datasets from same source should all be returned."""
        engine.register_dataset(name="A", source_name="SAP")
        engine.register_dataset(name="B", source_name="SAP")
        result = engine.get_datasets_by_source("SAP")
        assert len(result) == 2


# ===========================================================================
# Test Class: Statistics & Reset
# ===========================================================================


class TestStatisticsAndReset:
    """Tests for get_statistics and reset methods."""

    def test_statistics_empty(self, engine: DatasetRegistryEngine):
        """Statistics on empty engine should have zero counts."""
        stats = engine.get_statistics()
        assert stats["total_datasets"] == 0
        assert stats["total_refresh_events"] == 0
        assert stats["total_groups"] == 0
        assert stats["total_operations"] == 0

    def test_statistics_after_register(self, engine: DatasetRegistryEngine):
        """Statistics should reflect registered datasets."""
        engine.register_dataset(name="A", source_name="SAP", priority="high")
        stats = engine.get_statistics()
        assert stats["total_datasets"] == 1
        assert stats["by_priority"]["high"] == 1
        assert stats["by_status"]["active"] == 1
        assert stats["by_cadence"]["daily"] == 1

    def test_statistics_refresh_events_count(self, engine: DatasetRegistryEngine):
        """Statistics should count refresh events."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        engine.record_refresh(ds.id)
        engine.record_refresh(ds.id)
        stats = engine.get_statistics()
        assert stats["total_refresh_events"] == 2

    def test_statistics_groups_count(self, engine: DatasetRegistryEngine):
        """Statistics should count groups."""
        engine.create_group(name="G1")
        stats = engine.get_statistics()
        assert stats["total_groups"] == 1

    def test_statistics_provenance_chain_length(
        self, engine: DatasetRegistryEngine,
    ):
        """Statistics should include provenance chain length."""
        engine.register_dataset(name="A", source_name="SAP")
        stats = engine.get_statistics()
        assert stats["provenance_chain_length"] >= 1

    def test_reset_clears_everything(self, populated_engine: DatasetRegistryEngine):
        """Reset should clear all datasets, groups, history, and counters."""
        populated_engine.create_group(name="G1")
        ds_ids = populated_engine.get_all_dataset_ids()
        for ds_id in ds_ids:
            populated_engine.record_refresh(ds_id)
        populated_engine.reset()
        assert populated_engine.get_dataset_count() == 0
        assert populated_engine.group_count == 0
        assert populated_engine.operation_count == 0
        assert populated_engine.provenance_chain_length == 0

    def test_statistics_multiple_priorities(self, engine: DatasetRegistryEngine):
        """Statistics should correctly count multiple priorities."""
        engine.register_dataset(name="A", source_name="SAP", priority="high")
        engine.register_dataset(name="B", source_name="S3", priority="high")
        engine.register_dataset(name="C", source_name="X", priority="low")
        stats = engine.get_statistics()
        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["low"] == 1


# ===========================================================================
# Test Class: Properties and Helpers
# ===========================================================================


class TestPropertiesAndHelpers:
    """Tests for engine properties and helper methods."""

    def test_dataset_count_property(self, engine: DatasetRegistryEngine):
        """dataset_count property should track registered datasets."""
        assert engine.dataset_count == 0
        engine.register_dataset(name="A", source_name="SAP")
        assert engine.dataset_count == 1

    def test_group_count_property(self, engine: DatasetRegistryEngine):
        """group_count property should track groups."""
        assert engine.group_count == 0
        engine.create_group(name="G1")
        assert engine.group_count == 1

    def test_get_all_dataset_ids(self, populated_engine: DatasetRegistryEngine):
        """get_all_dataset_ids should return sorted list of all IDs."""
        ids = populated_engine.get_all_dataset_ids()
        assert len(ids) == 3
        assert ids == sorted(ids)

    def test_get_provenance_chain(self, engine: DatasetRegistryEngine):
        """get_provenance_chain should return list of provenance entries."""
        engine.register_dataset(name="A", source_name="SAP")
        chain = engine.get_provenance_chain()
        assert len(chain) == 1
        assert chain[0]["operation"] == "register_dataset"

    def test_model_dump_on_dataset(self, engine: DatasetRegistryEngine):
        """Dataset should be serializable via model_dump or to_dict."""
        ds = engine.register_dataset(
            name="A", source_name="SAP", tags=["t1"],
        )
        d = _to_dict(ds)
        assert d["name"] == "A"
        assert d["source_name"] == "SAP"
        assert "id" in d
        assert "provenance_hash" in d

    def test_provenance_chain_length_property(self, engine: DatasetRegistryEngine):
        """provenance_chain_length should track entries."""
        assert engine.provenance_chain_length == 0
        engine.register_dataset(name="A", source_name="SAP")
        assert engine.provenance_chain_length >= 1

    def test_operation_count_increments_with_updates(
        self, engine: DatasetRegistryEngine,
    ):
        """Operation count should increment for updates."""
        ds = engine.register_dataset(name="A", source_name="SAP")
        ops_after_register = engine.operation_count
        engine.update_dataset(ds.id, name="B")
        assert engine.operation_count == ops_after_register + 1
