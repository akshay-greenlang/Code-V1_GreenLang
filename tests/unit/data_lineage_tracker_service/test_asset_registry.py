# -*- coding: utf-8 -*-
"""
Unit Tests for AssetRegistryEngine - AGENT-DATA-018 Data Lineage Tracker
========================================================================

Tests all public methods of AssetRegistryEngine (Engine 1 of 7) with 100+
tests covering registration, retrieval, update, deletion, search, bulk
operations, statistics, export/import, and thread safety.

Test Classes:
    TestRegisterAsset          - 12 tests: registration flows and validation
    TestGetAsset               - 4 tests:  retrieval by ID and qualified name
    TestUpdateAsset            - 11 tests: field updates and validation
    TestDeleteAsset            - 4 tests:  soft and hard deletion
    TestSearchAssets            - 9 tests:  multi-criteria search and pagination
    TestBulkOperations         - 4 tests:  bulk register, export, import
    TestStatistics             - 3 tests:  aggregate statistics
    TestMisc                   - 4 tests:  list methods, clear, thread safety
    TestAssetRegistryEdgeCases - 60+ tests: validation edge cases, boundaries

Total: 111+ tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List

import pytest

from greenlang.data_lineage_tracker.asset_registry import (
    AssetRegistryEngine,
    VALID_ASSET_TYPES,
    VALID_CLASSIFICATIONS,
    VALID_STATUSES,
    MAX_QUALIFIED_NAME_LENGTH,
    MAX_DISPLAY_NAME_LENGTH,
    MAX_TAG_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_BULK_IMPORT,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# TestRegisterAsset
# ============================================================================


class TestRegisterAsset:
    """Tests for AssetRegistryEngine.register_asset()."""

    def test_register_basic(self, asset_registry, sample_asset_params):
        """Register a basic asset and verify all returned fields."""
        result = asset_registry.register_asset(**sample_asset_params)

        assert result is not None
        assert "asset_id" in result
        assert result["qualified_name"] == "emissions.scope3.spend_data"
        assert result["asset_type"] == "dataset"
        assert result["display_name"] == "Scope 3 Spend Data"
        assert result["owner"] == "data-team"
        assert result["classification"] == "confidential"
        assert result["status"] == "active"
        assert result["version"] == 1
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64  # SHA-256
        assert result["created_at"] != ""
        assert result["updated_at"] != ""

    @pytest.mark.parametrize("asset_type", sorted(VALID_ASSET_TYPES))
    def test_register_all_types(self, asset_registry, asset_type):
        """Verify registration succeeds for every valid asset type."""
        result = asset_registry.register_asset(
            qualified_name=f"test.{asset_type}.asset",
            asset_type=asset_type,
        )
        assert result["asset_type"] == asset_type
        assert result["status"] == "active"

    @pytest.mark.parametrize("classification", sorted(VALID_CLASSIFICATIONS))
    def test_register_all_classifications(self, asset_registry, classification):
        """Verify registration succeeds for every valid classification."""
        result = asset_registry.register_asset(
            qualified_name=f"test.{classification}.asset",
            asset_type="dataset",
            classification=classification,
        )
        assert result["classification"] == classification

    def test_register_with_tags(self, asset_registry):
        """Verify tags are normalized, deduplicated, and sorted."""
        result = asset_registry.register_asset(
            qualified_name="test.tags",
            asset_type="dataset",
            tags=["Spend", "  scope3 ", "SPEND", "emissions"],
        )
        # Should be lowercased, stripped, deduplicated, and sorted
        assert result["tags"] == ["emissions", "scope3", "spend"]

    def test_register_with_metadata(self, asset_registry):
        """Verify metadata dict is stored correctly."""
        meta = {"source": "SAP", "version": 3, "nested": {"key": "value"}}
        result = asset_registry.register_asset(
            qualified_name="test.meta",
            asset_type="dataset",
            metadata=meta,
        )
        assert result["metadata"]["source"] == "SAP"
        assert result["metadata"]["version"] == 3
        assert result["metadata"]["nested"]["key"] == "value"

    def test_register_duplicate_name_fails(self, asset_registry):
        """Registering two active assets with the same qualified_name must fail."""
        asset_registry.register_asset(
            qualified_name="duplicate.name",
            asset_type="dataset",
        )
        with pytest.raises(ValueError, match="already exists"):
            asset_registry.register_asset(
                qualified_name="duplicate.name",
                asset_type="pipeline",
            )

    def test_register_invalid_type(self, asset_registry):
        """Invalid asset_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid asset_type"):
            asset_registry.register_asset(
                qualified_name="bad.type",
                asset_type="nonexistent_type",
            )

    def test_register_invalid_classification(self, asset_registry):
        """Invalid classification raises ValueError."""
        with pytest.raises(ValueError, match="Invalid classification"):
            asset_registry.register_asset(
                qualified_name="bad.classification",
                asset_type="dataset",
                classification="top_secret",
            )

    def test_register_generates_uuid(self, asset_registry):
        """Every registration produces a unique UUID asset_id."""
        r1 = asset_registry.register_asset(
            qualified_name="uuid.test.1", asset_type="dataset"
        )
        r2 = asset_registry.register_asset(
            qualified_name="uuid.test.2", asset_type="dataset"
        )
        assert r1["asset_id"] != r2["asset_id"]
        # Validate it is a valid UUID
        uuid.UUID(r1["asset_id"])
        uuid.UUID(r2["asset_id"])

    def test_register_records_provenance(self, asset_registry, provenance_tracker):
        """Registration must create a provenance chain entry."""
        initial_count = provenance_tracker.entry_count
        asset_registry.register_asset(
            qualified_name="prov.test", asset_type="dataset"
        )
        assert provenance_tracker.entry_count == initial_count + 1

    def test_register_records_metric(self, asset_registry):
        """Registration should not raise even when metrics are no-op."""
        # This test verifies that the metric recording path does not throw.
        result = asset_registry.register_asset(
            qualified_name="metric.test",
            asset_type="dataset",
            classification="public",
        )
        assert result["asset_type"] == "dataset"

    def test_register_default_display_name(self, asset_registry):
        """When display_name is omitted, it defaults to qualified_name."""
        result = asset_registry.register_asset(
            qualified_name="default.display",
            asset_type="dataset",
        )
        assert result["display_name"] == "default.display"


# ============================================================================
# TestGetAsset
# ============================================================================


class TestGetAsset:
    """Tests for AssetRegistryEngine.get_asset() and get_asset_by_name()."""

    def test_get_by_id(self, asset_registry):
        """Retrieve an asset by its UUID."""
        registered = asset_registry.register_asset(
            qualified_name="get.by.id", asset_type="dataset"
        )
        retrieved = asset_registry.get_asset(registered["asset_id"])
        assert retrieved is not None
        assert retrieved["qualified_name"] == "get.by.id"
        assert retrieved["asset_id"] == registered["asset_id"]

    def test_get_nonexistent(self, asset_registry):
        """Getting a non-existent asset_id returns None."""
        result = asset_registry.get_asset("nonexistent-uuid")
        assert result is None

    def test_get_by_name(self, asset_registry):
        """Retrieve an asset by its qualified_name."""
        asset_registry.register_asset(
            qualified_name="get.by.name", asset_type="pipeline"
        )
        result = asset_registry.get_asset_by_name("get.by.name")
        assert result is not None
        assert result["qualified_name"] == "get.by.name"
        assert result["asset_type"] == "pipeline"

    def test_get_by_name_nonexistent(self, asset_registry):
        """Getting a non-existent qualified_name returns None."""
        result = asset_registry.get_asset_by_name("does.not.exist")
        assert result is None


# ============================================================================
# TestUpdateAsset
# ============================================================================


class TestUpdateAsset:
    """Tests for AssetRegistryEngine.update_asset()."""

    def test_update_display_name(self, asset_registry):
        """Update display_name and verify version increments."""
        registered = asset_registry.register_asset(
            qualified_name="update.display", asset_type="dataset"
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], display_name="New Display Name"
        )
        assert updated is not None
        assert updated["display_name"] == "New Display Name"
        assert updated["version"] == 2

    def test_update_owner(self, asset_registry):
        """Update owner and verify owner index reconciliation."""
        registered = asset_registry.register_asset(
            qualified_name="update.owner",
            asset_type="dataset",
            owner="old-team",
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], owner="new-team"
        )
        assert updated["owner"] == "new-team"

        # Verify owner index was reconciled
        owners = asset_registry.list_owners()
        assert "new-team" in owners
        assert owners.get("old-team", 0) == 0

    def test_update_tags(self, asset_registry):
        """Update tags and verify tag index reconciliation."""
        registered = asset_registry.register_asset(
            qualified_name="update.tags",
            asset_type="dataset",
            tags=["old-tag", "common-tag"],
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], tags=["new-tag", "common-tag"]
        )
        assert "common-tag" in updated["tags"]
        assert "new-tag" in updated["tags"]
        assert "old-tag" not in updated["tags"]

    def test_update_classification(self, asset_registry):
        """Update classification to a valid value."""
        registered = asset_registry.register_asset(
            qualified_name="update.classification",
            asset_type="dataset",
            classification="internal",
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], classification="restricted"
        )
        assert updated["classification"] == "restricted"

    def test_update_status(self, asset_registry):
        """Update status to deprecated."""
        registered = asset_registry.register_asset(
            qualified_name="update.status", asset_type="dataset"
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], status="deprecated"
        )
        assert updated["status"] == "deprecated"

    def test_update_description(self, asset_registry):
        """Update description field."""
        registered = asset_registry.register_asset(
            qualified_name="update.desc", asset_type="dataset"
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], description="Updated description."
        )
        assert updated["description"] == "Updated description."

    def test_update_metadata(self, asset_registry):
        """Update metadata dictionary."""
        registered = asset_registry.register_asset(
            qualified_name="update.meta", asset_type="dataset"
        )
        new_meta = {"refresh": "hourly", "priority": "high"}
        updated = asset_registry.update_asset(
            registered["asset_id"], metadata=new_meta
        )
        assert updated["metadata"]["refresh"] == "hourly"
        assert updated["metadata"]["priority"] == "high"

    def test_update_nonexistent(self, asset_registry):
        """Updating a non-existent asset returns None."""
        result = asset_registry.update_asset(
            "nonexistent-uuid", display_name="Nope"
        )
        assert result is None

    def test_update_invalid_status(self, asset_registry):
        """Updating with an invalid status raises ValueError."""
        registered = asset_registry.register_asset(
            qualified_name="invalid.status", asset_type="dataset"
        )
        with pytest.raises(ValueError, match="Invalid status"):
            asset_registry.update_asset(
                registered["asset_id"], status="nonexistent_status"
            )

    def test_update_records_provenance(self, asset_registry, provenance_tracker):
        """Update must record a provenance chain entry."""
        registered = asset_registry.register_asset(
            qualified_name="update.prov", asset_type="dataset"
        )
        count_after_register = provenance_tracker.entry_count
        asset_registry.update_asset(
            registered["asset_id"], display_name="Provenance Test"
        )
        assert provenance_tracker.entry_count == count_after_register + 1

    def test_update_invalid_field_rejected(self, asset_registry):
        """Passing a non-updatable field raises ValueError."""
        registered = asset_registry.register_asset(
            qualified_name="invalid.field", asset_type="dataset"
        )
        with pytest.raises(ValueError, match="Cannot update fields"):
            asset_registry.update_asset(
                registered["asset_id"], qualified_name="new.name"
            )


# ============================================================================
# TestDeleteAsset
# ============================================================================


class TestDeleteAsset:
    """Tests for AssetRegistryEngine.delete_asset()."""

    def test_soft_delete(self, asset_registry):
        """Soft delete sets status to archived and asset is still retrievable."""
        registered = asset_registry.register_asset(
            qualified_name="soft.delete", asset_type="dataset"
        )
        result = asset_registry.delete_asset(registered["asset_id"])
        assert result is True

        retrieved = asset_registry.get_asset(registered["asset_id"])
        assert retrieved is not None
        assert retrieved["status"] == "archived"

    def test_hard_delete(self, asset_registry):
        """Hard delete permanently removes the asset."""
        registered = asset_registry.register_asset(
            qualified_name="hard.delete", asset_type="dataset"
        )
        result = asset_registry.delete_asset(registered["asset_id"], hard=True)
        assert result is True

        retrieved = asset_registry.get_asset(registered["asset_id"])
        assert retrieved is None

    def test_delete_nonexistent(self, asset_registry):
        """Deleting a non-existent asset returns False."""
        result = asset_registry.delete_asset("nonexistent-uuid")
        assert result is False

    def test_soft_delete_sets_archived(self, asset_registry):
        """Verify soft delete specifically sets status to 'archived'."""
        registered = asset_registry.register_asset(
            qualified_name="archive.check", asset_type="pipeline"
        )
        asset_registry.delete_asset(registered["asset_id"])
        retrieved = asset_registry.get_asset(registered["asset_id"])
        assert retrieved["status"] == "archived"
        # Version should be incremented
        assert retrieved["version"] == 2


# ============================================================================
# TestSearchAssets
# ============================================================================


class TestSearchAssets:
    """Tests for AssetRegistryEngine.search_assets()."""

    def _register_test_assets(self, registry):
        """Helper to register a set of test assets for search tests."""
        registry.register_asset(
            qualified_name="search.dataset.alpha",
            asset_type="dataset",
            owner="team-a",
            classification="internal",
            tags=["scope3", "spend"],
        )
        registry.register_asset(
            qualified_name="search.dataset.beta",
            asset_type="dataset",
            owner="team-b",
            classification="confidential",
            tags=["scope3", "energy"],
        )
        registry.register_asset(
            qualified_name="search.pipeline.gamma",
            asset_type="pipeline",
            owner="team-a",
            classification="internal",
            tags=["etl"],
        )
        registry.register_asset(
            qualified_name="search.report.delta",
            asset_type="report",
            owner="team-b",
            classification="public",
            tags=["csrd"],
        )

    def test_search_by_type(self, asset_registry):
        """Search filtering by asset_type returns correct results."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(asset_type="dataset")
        assert len(results) == 2
        assert all(r["asset_type"] == "dataset" for r in results)

    def test_search_by_owner(self, asset_registry):
        """Search filtering by owner."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(owner="team-a")
        assert len(results) == 2
        assert all(r["owner"] == "team-a" for r in results)

    def test_search_by_classification(self, asset_registry):
        """Search filtering by classification."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(classification="internal")
        assert len(results) == 2
        assert all(r["classification"] == "internal" for r in results)

    def test_search_by_status(self, asset_registry):
        """Search filtering by status."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(status="active")
        assert len(results) == 4

    def test_search_by_tags(self, asset_registry):
        """Search filtering by tags (AND logic)."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(tags=["scope3"])
        assert len(results) == 2

        results = asset_registry.search_assets(tags=["scope3", "spend"])
        assert len(results) == 1
        assert results[0]["qualified_name"] == "search.dataset.alpha"

    def test_search_by_name_pattern(self, asset_registry):
        """Search filtering by regex name_pattern."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(name_pattern="dataset")
        assert len(results) == 2

        results = asset_registry.search_assets(name_pattern="alpha$")
        assert len(results) == 1

    def test_search_pagination(self, asset_registry):
        """Search with limit and offset for pagination."""
        self._register_test_assets(asset_registry)
        page1 = asset_registry.search_assets(limit=2, offset=0)
        page2 = asset_registry.search_assets(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap
        page1_ids = {r["asset_id"] for r in page1}
        page2_ids = {r["asset_id"] for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_search_empty_results(self, asset_registry):
        """Search returns empty list when no assets match."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(asset_type="nonexistent")
        assert results == []

    def test_search_combined_filters(self, asset_registry):
        """Search with multiple combined filters (AND logic)."""
        self._register_test_assets(asset_registry)
        results = asset_registry.search_assets(
            asset_type="dataset",
            owner="team-a",
            classification="internal",
        )
        assert len(results) == 1
        assert results[0]["qualified_name"] == "search.dataset.alpha"


# ============================================================================
# TestBulkOperations
# ============================================================================


class TestBulkOperations:
    """Tests for bulk_register, export_assets, import_assets."""

    def test_bulk_register(self, asset_registry):
        """Bulk register multiple assets successfully."""
        assets = [
            {"qualified_name": f"bulk.asset.{i}", "asset_type": "dataset"}
            for i in range(5)
        ]
        result = asset_registry.bulk_register(assets)
        assert result["registered"] == 5
        assert result["failed"] == 0
        assert len(result["asset_ids"]) == 5
        assert result["provenance_hash"] != ""

    def test_bulk_register_with_failures(self, asset_registry):
        """Bulk register with some invalid entries skips failures."""
        assets = [
            {"qualified_name": "valid.asset", "asset_type": "dataset"},
            {"qualified_name": "", "asset_type": "dataset"},  # Invalid: empty name
            {"qualified_name": "valid.asset.2", "asset_type": "bad_type"},  # Invalid type
        ]
        result = asset_registry.bulk_register(assets)
        assert result["registered"] == 1
        assert result["failed"] == 2
        assert len(result["errors"]) == 2

    def test_export_assets(self, asset_registry):
        """Export all assets as a list of dicts."""
        asset_registry.register_asset(
            qualified_name="export.a", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="export.b", asset_type="pipeline"
        )
        exported = asset_registry.export_assets()
        assert len(exported) == 2
        # Sorted by qualified_name
        assert exported[0]["qualified_name"] == "export.a"
        assert exported[1]["qualified_name"] == "export.b"

    def test_import_assets(self, asset_registry):
        """Import previously exported assets into a new registry."""
        asset_registry.register_asset(
            qualified_name="import.source", asset_type="dataset"
        )
        exported = asset_registry.export_assets()

        # Create a new registry and import
        new_provenance = ProvenanceTracker()
        new_registry = AssetRegistryEngine(provenance=new_provenance)
        result = new_registry.import_assets(exported)

        assert result["imported"] == 1
        assert result["failed"] == 0
        assert len(result["asset_ids"]) == 1

        # Verify the imported asset is retrievable
        imported_asset = new_registry.get_asset_by_name("import.source")
        assert imported_asset is not None
        assert imported_asset["asset_type"] == "dataset"


# ============================================================================
# TestStatistics
# ============================================================================


class TestStatistics:
    """Tests for AssetRegistryEngine.get_statistics()."""

    def test_get_statistics(self, asset_registry):
        """Verify statistics reflect registered assets."""
        asset_registry.register_asset(
            qualified_name="stats.a",
            asset_type="dataset",
            owner="team-a",
            classification="internal",
        )
        asset_registry.register_asset(
            qualified_name="stats.b",
            asset_type="pipeline",
            owner="team-b",
            classification="confidential",
        )
        stats = asset_registry.get_statistics()

        assert stats["total_assets"] == 2
        assert stats["by_type"]["dataset"] == 1
        assert stats["by_type"]["pipeline"] == 1
        assert stats["by_classification"]["internal"] == 1
        assert stats["by_classification"]["confidential"] == 1
        assert stats["by_status"]["active"] == 2
        assert stats["by_owner"]["team-a"] == 1
        assert stats["by_owner"]["team-b"] == 1
        assert stats["provenance_entries"] >= 2

    def test_statistics_by_type(self, asset_registry):
        """Verify list_asset_types returns correct counts per type."""
        asset_registry.register_asset(
            qualified_name="types.d1", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="types.d2", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="types.p1", asset_type="pipeline"
        )
        types = asset_registry.list_asset_types()
        assert types["dataset"] == 2
        assert types["pipeline"] == 1

    def test_statistics_empty(self, asset_registry):
        """Statistics on empty registry show all zeros."""
        stats = asset_registry.get_statistics()
        assert stats["total_assets"] == 0
        assert stats["by_type"] == {}
        assert stats["total_tags"] == 0


# ============================================================================
# TestMisc
# ============================================================================


class TestMisc:
    """Miscellaneous tests for list methods, clear, and thread safety."""

    def test_list_asset_types(self, asset_registry):
        """list_asset_types returns counts for each registered type."""
        asset_registry.register_asset(
            qualified_name="list.d1", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="list.p1", asset_type="pipeline"
        )
        types = asset_registry.list_asset_types()
        assert "dataset" in types
        assert "pipeline" in types

    def test_list_owners(self, asset_registry):
        """list_owners returns counts for each registered owner."""
        asset_registry.register_asset(
            qualified_name="owner.a", asset_type="dataset", owner="team-x"
        )
        asset_registry.register_asset(
            qualified_name="owner.b", asset_type="dataset", owner="team-y"
        )
        owners = asset_registry.list_owners()
        assert owners["team-x"] == 1
        assert owners["team-y"] == 1

    def test_clear(self, asset_registry):
        """clear() removes all assets and resets provenance."""
        asset_registry.register_asset(
            qualified_name="clear.test", asset_type="dataset"
        )
        assert asset_registry.asset_count == 1
        asset_registry.clear()
        assert asset_registry.asset_count == 0
        assert asset_registry.get_statistics()["total_assets"] == 0

    def test_thread_safety(self, provenance_tracker):
        """Concurrent registration from multiple threads must not corrupt state."""
        engine = AssetRegistryEngine(provenance=provenance_tracker)
        num_threads = 10
        assets_per_thread = 5
        errors: List[str] = []

        def register_batch(thread_id: int):
            try:
                for i in range(assets_per_thread):
                    engine.register_asset(
                        qualified_name=f"thread.{thread_id}.asset.{i}",
                        asset_type="dataset",
                        owner=f"thread-{thread_id}",
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=register_batch, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert engine.asset_count == num_threads * assets_per_thread


# ============================================================================
# TestAssetRegistryEdgeCases
# ============================================================================


class TestAssetRegistryEdgeCases:
    """Edge case and boundary condition tests for AssetRegistryEngine."""

    # -- Qualified name validation --

    def test_empty_qualified_name_raises(self, asset_registry):
        """Empty string qualified_name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            asset_registry.register_asset(
                qualified_name="", asset_type="dataset"
            )

    def test_whitespace_only_qualified_name_raises(self, asset_registry):
        """Whitespace-only qualified_name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            asset_registry.register_asset(
                qualified_name="   ", asset_type="dataset"
            )

    def test_max_length_qualified_name(self, asset_registry):
        """Qualified name at exactly MAX_QUALIFIED_NAME_LENGTH succeeds."""
        long_name = "a" * MAX_QUALIFIED_NAME_LENGTH
        result = asset_registry.register_asset(
            qualified_name=long_name,
            asset_type="dataset",
            display_name="short_display",  # Avoid display_name defaulting to long qualified_name
        )
        assert result["qualified_name"] == long_name

    def test_exceeded_qualified_name_length_raises(self, asset_registry):
        """Qualified name exceeding MAX_QUALIFIED_NAME_LENGTH raises ValueError."""
        too_long = "a" * (MAX_QUALIFIED_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            asset_registry.register_asset(
                qualified_name=too_long, asset_type="dataset"
            )

    # -- Display name validation --

    def test_max_display_name_length(self, asset_registry):
        """Display name at exactly MAX_DISPLAY_NAME_LENGTH succeeds."""
        long_display = "b" * MAX_DISPLAY_NAME_LENGTH
        result = asset_registry.register_asset(
            qualified_name="display.max",
            asset_type="dataset",
            display_name=long_display,
        )
        assert result["display_name"] == long_display

    def test_exceeded_display_name_length_raises(self, asset_registry):
        """Display name exceeding MAX_DISPLAY_NAME_LENGTH raises ValueError."""
        too_long = "b" * (MAX_DISPLAY_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="display_name exceeds"):
            asset_registry.register_asset(
                qualified_name="display.toolong",
                asset_type="dataset",
                display_name=too_long,
            )

    # -- Description validation --

    def test_max_description_length(self, asset_registry):
        """Description at exactly MAX_DESCRIPTION_LENGTH succeeds."""
        long_desc = "c" * MAX_DESCRIPTION_LENGTH
        result = asset_registry.register_asset(
            qualified_name="desc.max",
            asset_type="dataset",
            description=long_desc,
        )
        assert len(result["description"]) == MAX_DESCRIPTION_LENGTH

    def test_exceeded_description_length_raises(self, asset_registry):
        """Description exceeding MAX_DESCRIPTION_LENGTH raises ValueError."""
        too_long = "c" * (MAX_DESCRIPTION_LENGTH + 1)
        with pytest.raises(ValueError, match="Description exceeds"):
            asset_registry.register_asset(
                qualified_name="desc.toolong",
                asset_type="dataset",
                description=too_long,
            )

    # -- Tag validation --

    def test_tag_max_length(self, asset_registry):
        """Tag at exactly MAX_TAG_LENGTH succeeds."""
        long_tag = "d" * MAX_TAG_LENGTH
        result = asset_registry.register_asset(
            qualified_name="tag.max",
            asset_type="dataset",
            tags=[long_tag],
        )
        assert long_tag in result["tags"]

    def test_tag_exceeded_length_raises(self, asset_registry):
        """Tag exceeding MAX_TAG_LENGTH raises ValueError."""
        too_long = "d" * (MAX_TAG_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            asset_registry.register_asset(
                qualified_name="tag.toolong",
                asset_type="dataset",
                tags=[too_long],
            )

    def test_empty_tags_list(self, asset_registry):
        """Empty tags list results in no tags."""
        result = asset_registry.register_asset(
            qualified_name="tag.empty",
            asset_type="dataset",
            tags=[],
        )
        assert result["tags"] == []

    def test_none_tags(self, asset_registry):
        """None tags defaults to empty list."""
        result = asset_registry.register_asset(
            qualified_name="tag.none",
            asset_type="dataset",
            tags=None,
        )
        assert result["tags"] == []

    # -- Owner defaults --

    def test_default_owner(self, asset_registry):
        """Default owner is 'system' when not specified."""
        result = asset_registry.register_asset(
            qualified_name="owner.default", asset_type="dataset"
        )
        assert result["owner"] == "system"

    def test_empty_owner_defaults_to_system(self, asset_registry):
        """Empty string owner defaults to 'system'."""
        result = asset_registry.register_asset(
            qualified_name="owner.empty",
            asset_type="dataset",
            owner="",
        )
        assert result["owner"] == "system"

    # -- Classification defaults --

    def test_default_classification(self, asset_registry):
        """Default classification is 'internal'."""
        result = asset_registry.register_asset(
            qualified_name="class.default", asset_type="dataset"
        )
        assert result["classification"] == "internal"

    # -- Asset type case insensitivity --

    def test_asset_type_case_insensitive(self, asset_registry):
        """Asset type is normalized to lowercase."""
        result = asset_registry.register_asset(
            qualified_name="case.type",
            asset_type="DATASET",
        )
        assert result["asset_type"] == "dataset"

    def test_classification_case_insensitive(self, asset_registry):
        """Classification is normalized to lowercase."""
        result = asset_registry.register_asset(
            qualified_name="case.class",
            asset_type="dataset",
            classification="CONFIDENTIAL",
        )
        assert result["classification"] == "confidential"

    # -- Whitespace stripping --

    def test_qualified_name_stripped(self, asset_registry):
        """Leading/trailing whitespace in qualified_name is stripped."""
        result = asset_registry.register_asset(
            qualified_name="  strip.name  ",
            asset_type="dataset",
        )
        assert result["qualified_name"] == "strip.name"

    def test_asset_type_stripped(self, asset_registry):
        """Leading/trailing whitespace in asset_type is stripped."""
        result = asset_registry.register_asset(
            qualified_name="strip.type",
            asset_type="  dataset  ",
        )
        assert result["asset_type"] == "dataset"

    # -- Schema ref --

    def test_schema_ref_stored(self, asset_registry):
        """schema_ref is stored correctly when provided."""
        result = asset_registry.register_asset(
            qualified_name="schema.ref",
            asset_type="dataset",
            schema_ref="schema-uuid-123",
        )
        assert result["schema_ref"] == "schema-uuid-123"

    def test_schema_ref_none_by_default(self, asset_registry):
        """schema_ref defaults to None when not provided."""
        result = asset_registry.register_asset(
            qualified_name="schema.default",
            asset_type="dataset",
        )
        assert result["schema_ref"] is None

    # -- Metadata isolation --

    def test_metadata_deep_copy(self, asset_registry):
        """Modifying the original metadata dict does not affect stored asset."""
        meta = {"key": "original"}
        result = asset_registry.register_asset(
            qualified_name="meta.deepcopy",
            asset_type="dataset",
            metadata=meta,
        )
        meta["key"] = "mutated"
        retrieved = asset_registry.get_asset(result["asset_id"])
        assert retrieved["metadata"]["key"] == "original"

    def test_retrieved_asset_is_deep_copy(self, asset_registry):
        """Modifying a retrieved asset dict does not affect the registry."""
        result = asset_registry.register_asset(
            qualified_name="copy.test",
            asset_type="dataset",
            metadata={"immutable": True},
        )
        retrieved = asset_registry.get_asset(result["asset_id"])
        retrieved["metadata"]["immutable"] = False

        retrieved2 = asset_registry.get_asset(result["asset_id"])
        assert retrieved2["metadata"]["immutable"] is True

    # -- Re-registration after archive --

    def test_reregister_after_archive(self, asset_registry):
        """Registering a qualified_name after archiving the original succeeds."""
        r1 = asset_registry.register_asset(
            qualified_name="reregister.test", asset_type="dataset"
        )
        asset_registry.delete_asset(r1["asset_id"])  # Soft delete -> archived

        r2 = asset_registry.register_asset(
            qualified_name="reregister.test", asset_type="pipeline"
        )
        assert r2["asset_id"] != r1["asset_id"]
        assert r2["asset_type"] == "pipeline"
        assert r2["status"] == "active"

    # -- Hard delete removes from all indexes --

    def test_hard_delete_removes_from_name_index(self, asset_registry):
        """Hard delete removes asset from the name index."""
        registered = asset_registry.register_asset(
            qualified_name="harddelete.name", asset_type="dataset"
        )
        asset_registry.delete_asset(registered["asset_id"], hard=True)
        assert asset_registry.get_asset_by_name("harddelete.name") is None

    def test_hard_delete_removes_from_type_index(self, asset_registry):
        """Hard delete removes asset from the type index."""
        registered = asset_registry.register_asset(
            qualified_name="harddelete.type", asset_type="metric"
        )
        asset_registry.delete_asset(registered["asset_id"], hard=True)
        results = asset_registry.search_assets(asset_type="metric")
        assert len(results) == 0

    def test_hard_delete_removes_from_owner_index(self, asset_registry):
        """Hard delete removes asset from the owner index."""
        registered = asset_registry.register_asset(
            qualified_name="harddelete.owner",
            asset_type="dataset",
            owner="delete-team",
        )
        asset_registry.delete_asset(registered["asset_id"], hard=True)
        owners = asset_registry.list_owners()
        assert "delete-team" not in owners

    def test_hard_delete_removes_from_tag_index(self, asset_registry):
        """Hard delete removes asset from the tag index."""
        registered = asset_registry.register_asset(
            qualified_name="harddelete.tag",
            asset_type="dataset",
            tags=["removeme"],
        )
        asset_registry.delete_asset(registered["asset_id"], hard=True)
        results = asset_registry.search_assets(tags=["removeme"])
        assert len(results) == 0

    # -- Bulk operation edge cases --

    def test_bulk_register_empty_raises(self, asset_registry):
        """Bulk register with empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            asset_registry.bulk_register([])

    def test_bulk_register_exceeds_limit_raises(self, asset_registry):
        """Bulk register exceeding MAX_BULK_IMPORT raises ValueError."""
        assets = [
            {"qualified_name": f"bulk.{i}", "asset_type": "dataset"}
            for i in range(MAX_BULK_IMPORT + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            asset_registry.bulk_register(assets)

    def test_import_empty_raises(self, asset_registry):
        """Import with empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            asset_registry.import_assets([])

    def test_import_exceeds_limit_raises(self, asset_registry):
        """Import exceeding MAX_BULK_IMPORT raises ValueError."""
        assets = [
            {"qualified_name": f"import.{i}", "asset_type": "dataset"}
            for i in range(MAX_BULK_IMPORT + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            asset_registry.import_assets(assets)

    # -- Export filtering --

    def test_export_by_type(self, asset_registry):
        """Export filtered by asset_type returns only matching assets."""
        asset_registry.register_asset(
            qualified_name="export.d", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="export.p", asset_type="pipeline"
        )
        exported = asset_registry.export_assets(asset_type="dataset")
        assert len(exported) == 1
        assert exported[0]["asset_type"] == "dataset"

    def test_export_empty_registry(self, asset_registry):
        """Export from empty registry returns empty list."""
        exported = asset_registry.export_assets()
        assert exported == []

    # -- Property tests --

    def test_asset_count_property(self, asset_registry):
        """asset_count property returns correct count."""
        assert asset_registry.asset_count == 0
        asset_registry.register_asset(
            qualified_name="count.1", asset_type="dataset"
        )
        assert asset_registry.asset_count == 1
        asset_registry.register_asset(
            qualified_name="count.2", asset_type="dataset"
        )
        assert asset_registry.asset_count == 2

    def test_provenance_chain_length_property(self, asset_registry):
        """provenance_chain_length property increments on registration."""
        initial = asset_registry.provenance_chain_length
        asset_registry.register_asset(
            qualified_name="prov.chain", asset_type="dataset"
        )
        assert asset_registry.provenance_chain_length == initial + 1

    def test_get_all_asset_ids(self, asset_registry):
        """get_all_asset_ids returns sorted list of all IDs."""
        r1 = asset_registry.register_asset(
            qualified_name="ids.a", asset_type="dataset"
        )
        r2 = asset_registry.register_asset(
            qualified_name="ids.b", asset_type="pipeline"
        )
        ids = asset_registry.get_all_asset_ids()
        assert len(ids) == 2
        assert r1["asset_id"] in ids
        assert r2["asset_id"] in ids
        # Sorted
        assert ids == sorted(ids)

    def test_get_provenance_chain(self, asset_registry):
        """get_provenance_chain returns list of provenance entries."""
        asset_registry.register_asset(
            qualified_name="chain.test", asset_type="dataset"
        )
        chain = asset_registry.get_provenance_chain()
        assert len(chain) >= 1
        assert isinstance(chain[0], dict)
        assert "hash_value" in chain[0]

    # -- Update edge cases --

    def test_update_classification_case_insensitive(self, asset_registry):
        """Updating classification is case-insensitive."""
        registered = asset_registry.register_asset(
            qualified_name="update.case.class", asset_type="dataset"
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], classification="RESTRICTED"
        )
        assert updated["classification"] == "restricted"

    def test_update_invalid_classification_raises(self, asset_registry):
        """Updating with invalid classification raises ValueError."""
        registered = asset_registry.register_asset(
            qualified_name="update.bad.class", asset_type="dataset"
        )
        with pytest.raises(ValueError, match="Invalid classification"):
            asset_registry.update_asset(
                registered["asset_id"], classification="top_secret"
            )

    def test_update_display_name_too_long_raises(self, asset_registry):
        """Updating display_name beyond limit raises ValueError."""
        registered = asset_registry.register_asset(
            qualified_name="update.long.display", asset_type="dataset"
        )
        too_long = "x" * (MAX_DISPLAY_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="display_name exceeds"):
            asset_registry.update_asset(
                registered["asset_id"], display_name=too_long
            )

    def test_update_description_too_long_raises(self, asset_registry):
        """Updating description beyond limit raises ValueError."""
        registered = asset_registry.register_asset(
            qualified_name="update.long.desc", asset_type="dataset"
        )
        too_long = "y" * (MAX_DESCRIPTION_LENGTH + 1)
        with pytest.raises(ValueError, match="description exceeds"):
            asset_registry.update_asset(
                registered["asset_id"], description=too_long
            )

    def test_update_multiple_fields(self, asset_registry):
        """Updating multiple fields simultaneously."""
        registered = asset_registry.register_asset(
            qualified_name="update.multi",
            asset_type="dataset",
            owner="old-owner",
            classification="internal",
        )
        updated = asset_registry.update_asset(
            registered["asset_id"],
            owner="new-owner",
            classification="confidential",
            display_name="Multi Update",
        )
        assert updated["owner"] == "new-owner"
        assert updated["classification"] == "confidential"
        assert updated["display_name"] == "Multi Update"
        assert updated["version"] == 2

    def test_update_no_actual_change_still_bumps_version(self, asset_registry):
        """Even if fields have same values, version is still bumped."""
        registered = asset_registry.register_asset(
            qualified_name="update.same",
            asset_type="dataset",
            owner="same-owner",
        )
        updated = asset_registry.update_asset(
            registered["asset_id"], owner="same-owner"
        )
        # Version still bumps because update_asset always increments
        assert updated["version"] == 2

    # -- Search edge cases --

    def test_search_invalid_regex_ignored(self, asset_registry):
        """Invalid regex in name_pattern is ignored, returns unfiltered results."""
        asset_registry.register_asset(
            qualified_name="regex.test", asset_type="dataset"
        )
        # Invalid regex with unmatched bracket
        results = asset_registry.search_assets(name_pattern="[invalid")
        # Should return results since pattern is silently ignored
        assert len(results) >= 1

    def test_search_negative_limit_raises(self, asset_registry):
        """Negative limit raises ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 0"):
            asset_registry.search_assets(limit=-1)

    def test_search_negative_offset_raises(self, asset_registry):
        """Negative offset raises ValueError."""
        with pytest.raises(ValueError, match="offset must be >= 0"):
            asset_registry.search_assets(offset=-1)

    def test_search_zero_limit_returns_all(self, asset_registry):
        """Zero limit returns all matching results (no slicing)."""
        for i in range(3):
            asset_registry.register_asset(
                qualified_name=f"zero.limit.{i}", asset_type="dataset"
            )
        results = asset_registry.search_assets(limit=0)
        assert len(results) == 3

    def test_search_large_offset_returns_empty(self, asset_registry):
        """Offset beyond total results returns empty list."""
        asset_registry.register_asset(
            qualified_name="large.offset", asset_type="dataset"
        )
        results = asset_registry.search_assets(offset=100)
        assert results == []

    # -- Multiple registrations and statistics consistency --

    def test_statistics_after_hard_delete(self, asset_registry):
        """Statistics reflect state after hard delete."""
        r1 = asset_registry.register_asset(
            qualified_name="stat.del.1", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="stat.del.2", asset_type="dataset"
        )
        asset_registry.delete_asset(r1["asset_id"], hard=True)

        stats = asset_registry.get_statistics()
        assert stats["total_assets"] == 1

    def test_statistics_after_soft_delete(self, asset_registry):
        """Statistics include soft-deleted (archived) assets."""
        r1 = asset_registry.register_asset(
            qualified_name="stat.soft.1", asset_type="dataset"
        )
        asset_registry.register_asset(
            qualified_name="stat.soft.2", asset_type="dataset"
        )
        asset_registry.delete_asset(r1["asset_id"])  # Soft delete

        stats = asset_registry.get_statistics()
        assert stats["total_assets"] == 2  # Still counts archived
        assert stats["by_status"].get("archived", 0) == 1
        assert stats["by_status"].get("active", 0) == 1

    def test_clear_resets_provenance(self, asset_registry, provenance_tracker):
        """clear() resets the provenance tracker."""
        asset_registry.register_asset(
            qualified_name="clear.prov", asset_type="dataset"
        )
        assert provenance_tracker.entry_count > 0
        asset_registry.clear()
        assert provenance_tracker.entry_count == 0
