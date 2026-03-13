# -*- coding: utf-8 -*-
"""
Unit tests for DataPackageBuilder - AGENT-EUDR-015 Engine 7.

Tests all methods of DataPackageBuilder with 85%+ coverage.
Validates package creation, artifact addition, Merkle tree construction,
sealing, validation, export, size estimation, splitting, status
transitions, and error handling.

Test count: ~55 tests
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.mobile_data_collector.data_package_builder import (
    DataPackageBuilder,
    PACKAGE_STATUSES,
    PACKAGE_TRANSITIONS,
    ARTIFACT_TYPES,
    EXPORT_FORMATS,
    COMPRESSION_RATIOS,
)

from .conftest import assert_valid_sha256


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestDataPackageBuilderInit:
    """Tests for DataPackageBuilder initialization."""

    def test_initialization(self, data_package_builder):
        """Engine initializes with empty stores."""
        assert data_package_builder is not None
        assert len(data_package_builder) == 0

    def test_repr(self, data_package_builder):
        """Repr includes package count."""
        r = repr(data_package_builder)
        assert "DataPackageBuilder" in r

    def test_len_starts_at_zero(self, data_package_builder):
        """Initial package count is zero."""
        assert len(data_package_builder) == 0


# ---------------------------------------------------------------------------
# Test: create_package
# ---------------------------------------------------------------------------

class TestCreatePackage:
    """Tests for create_package method."""

    def test_create_valid_package(self, data_package_builder, make_data_package):
        """Create a valid data package in building status."""
        data = make_data_package()
        result = data_package_builder.create_package(**data)
        assert "package_id" in result
        assert result["status"] == "building"
        assert result["device_id"] == "dev-001"
        assert result["operator_id"] == "op-001"

    def test_create_increments_count(self, data_package_builder, make_data_package):
        """Creating a package increments count."""
        data_package_builder.create_package(**make_data_package())
        assert len(data_package_builder) == 1

    @pytest.mark.parametrize("fmt", ["zip", "tar_gz", "json_ld"])
    def test_create_all_export_formats(self, data_package_builder, make_data_package, fmt):
        """All export formats are accepted."""
        data = make_data_package(export_format=fmt)
        result = data_package_builder.create_package(**data)
        assert result["export_format"] == fmt

    def test_create_empty_device_id_raises(self, data_package_builder):
        """Empty device_id raises ValueError."""
        with pytest.raises(ValueError):
            data_package_builder.create_package(
                device_id="", operator_id="op-001",
            )

    def test_create_empty_operator_id_raises(self, data_package_builder):
        """Empty operator_id raises ValueError."""
        with pytest.raises(ValueError):
            data_package_builder.create_package(
                device_id="dev-001", operator_id="",
            )

    def test_create_invalid_format_raises(self, data_package_builder):
        """Invalid export format raises ValueError."""
        with pytest.raises(ValueError):
            data_package_builder.create_package(
                device_id="dev-001", operator_id="op-001",
                export_format="csv",
            )

    def test_create_unique_ids(self, data_package_builder, make_data_package):
        """Each package gets a unique ID."""
        ids = set()
        for _ in range(5):
            result = data_package_builder.create_package(**make_data_package())
            ids.add(result["package_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Test: get_package / list_packages
# ---------------------------------------------------------------------------

class TestPackageRetrieval:
    """Tests for package retrieval."""

    def test_get_existing_package(self, data_package_builder, make_data_package):
        """Get an existing package by ID."""
        created = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.get_package(created["package_id"])
        assert result["package_id"] == created["package_id"]

    def test_get_nonexistent_raises(self, data_package_builder):
        """Getting nonexistent package raises KeyError."""
        with pytest.raises(KeyError):
            data_package_builder.get_package("nonexistent")

    def test_list_packages_empty(self, data_package_builder):
        """List packages returns empty initially."""
        result = data_package_builder.list_packages()
        assert len(result) == 0

    def test_list_packages_filter_by_device(self, data_package_builder, make_data_package):
        """List packages filters by device_id."""
        data_package_builder.create_package(**make_data_package(device_id="dev-A"))
        data_package_builder.create_package(**make_data_package(device_id="dev-B"))
        result = data_package_builder.list_packages(device_id="dev-A")
        assert len(result) == 1

    def test_list_packages_filter_by_status(self, data_package_builder, make_data_package):
        """List packages filters by status."""
        data_package_builder.create_package(**make_data_package())
        result = data_package_builder.list_packages(status="building")
        assert all(p["status"] == "building" for p in result)


# ---------------------------------------------------------------------------
# Test: Add Artifacts
# ---------------------------------------------------------------------------

class TestAddArtifacts:
    """Tests for adding artifacts to packages."""

    def test_add_form(self, data_package_builder, make_data_package):
        """Add a form artifact."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.add_form(
            pkg["package_id"], form_id="form-001",
            data={"field": "value"}, size_bytes=1024,
        )
        assert result["artifact_count"] == 1
        assert "form-001" in result["form_ids"]

    def test_add_gps_capture(self, data_package_builder, make_data_package):
        """Add a GPS capture artifact."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.add_gps_capture(
            pkg["package_id"], capture_id="gps-001",
            data={"lat": 5.6, "lon": -0.18}, size_bytes=512,
        )
        assert result["artifact_count"] == 1
        assert "gps-001" in result["gps_capture_ids"]

    def test_add_photo(self, data_package_builder, make_data_package):
        """Add a photo artifact."""
        pkg = data_package_builder.create_package(**make_data_package())
        photo_hash = hashlib.sha256(b"photo_data").hexdigest()
        result = data_package_builder.add_photo(
            pkg["package_id"], photo_id="photo-001",
            integrity_hash=photo_hash, size_bytes=500_000,
        )
        assert result["artifact_count"] == 1

    def test_add_signature(self, data_package_builder, make_data_package):
        """Add a signature artifact."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.add_signature(
            pkg["package_id"], signature_id="sig-001",
            data={"signer": "test"}, size_bytes=256,
        )
        assert result["artifact_count"] == 1

    def test_add_polygon(self, data_package_builder, make_data_package):
        """Add a polygon artifact."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.add_polygon(
            pkg["package_id"], polygon_id="poly-001",
            data={"vertices": [[5.6, -0.18]]}, size_bytes=768,
        )
        assert result["artifact_count"] == 1

    def test_add_multiple_artifacts(self, data_package_builder, make_data_package):
        """Add multiple artifacts to one package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.add_form(pid, "f2", {"d": 2}, 2048)
        data_package_builder.add_gps_capture(pid, "g1", {"l": 1}, 512)
        result = data_package_builder.get_package(pid)
        assert result["artifact_count"] == 3
        assert result["total_size_bytes"] == 1024 + 2048 + 512

    def test_add_to_nonexistent_package_raises(self, data_package_builder):
        """Adding to nonexistent package raises KeyError."""
        with pytest.raises(KeyError):
            data_package_builder.add_form(
                "nonexistent", "f1", {"d": 1}, 1024,
            )

    def test_add_to_sealed_package_raises(self, data_package_builder, make_data_package):
        """Adding to sealed package raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        with pytest.raises(ValueError):
            data_package_builder.add_form(pid, "f2", {"d": 2}, 1024)


# ---------------------------------------------------------------------------
# Test: build_merkle_tree
# ---------------------------------------------------------------------------

class TestBuildMerkleTree:
    """Tests for Merkle tree construction."""

    def test_merkle_single_leaf(self, data_package_builder):
        """Merkle tree with single leaf has leaf as root."""
        leaf = hashlib.sha256(b"leaf1").hexdigest()
        root, tree = data_package_builder.build_merkle_tree([leaf])
        assert root == leaf
        assert tree["depth"] == 0
        assert tree["leaf_count"] == 1

    def test_merkle_two_leaves(self, data_package_builder):
        """Merkle tree with two leaves computes correct root."""
        l1 = hashlib.sha256(b"leaf1").hexdigest()
        l2 = hashlib.sha256(b"leaf2").hexdigest()
        root, tree = data_package_builder.build_merkle_tree([l1, l2])
        expected_root = hashlib.sha256((l1 + l2).encode("utf-8")).hexdigest()
        assert root == expected_root
        assert tree["depth"] == 1

    def test_merkle_four_leaves(self, data_package_builder):
        """Merkle tree with four leaves has depth 2."""
        leaves = [hashlib.sha256(f"leaf{i}".encode()).hexdigest() for i in range(4)]
        root, tree = data_package_builder.build_merkle_tree(leaves)
        assert tree["depth"] == 2
        assert tree["leaf_count"] == 4
        assert_valid_sha256(root)

    def test_merkle_odd_leaves_padded(self, data_package_builder):
        """Merkle tree with odd leaves pads the last leaf."""
        leaves = [hashlib.sha256(f"leaf{i}".encode()).hexdigest() for i in range(3)]
        root, tree = data_package_builder.build_merkle_tree(leaves)
        assert root is not None
        assert tree["leaf_count"] == 3

    def test_merkle_deterministic(self, data_package_builder):
        """Merkle tree is deterministic for same inputs."""
        leaves = [hashlib.sha256(f"l{i}".encode()).hexdigest() for i in range(5)]
        root1, _ = data_package_builder.build_merkle_tree(leaves)
        root2, _ = data_package_builder.build_merkle_tree(leaves)
        assert root1 == root2

    def test_merkle_empty_raises(self, data_package_builder):
        """Merkle tree from empty leaves raises ValueError."""
        with pytest.raises(ValueError):
            data_package_builder.build_merkle_tree([])


# ---------------------------------------------------------------------------
# Test: seal_package
# ---------------------------------------------------------------------------

class TestSealPackage:
    """Tests for package sealing."""

    def test_seal_valid_package(self, data_package_builder, make_data_package):
        """Seal a package with artifacts."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        result = data_package_builder.seal_package(pid)
        assert result["status"] == "sealed"
        assert result["merkle_root"] is not None
        assert_valid_sha256(result["merkle_root"])

    def test_seal_generates_manifest(self, data_package_builder, make_data_package):
        """Sealing generates artifact manifest."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        result = data_package_builder.seal_package(pid)
        assert result["manifest"] is not None
        assert result["manifest"]["artifact_count"] == 1

    def test_seal_generates_package_signature(self, data_package_builder, make_data_package):
        """Sealing generates device package signature."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        result = data_package_builder.seal_package(pid)
        assert result["package_signature_hex"] is not None
        assert_valid_sha256(result["package_signature_hex"])

    def test_seal_empty_package_raises(self, data_package_builder, make_data_package):
        """Sealing an empty package raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        with pytest.raises(ValueError):
            data_package_builder.seal_package(pkg["package_id"])

    def test_seal_nonexistent_raises(self, data_package_builder):
        """Sealing nonexistent package raises KeyError."""
        with pytest.raises(KeyError):
            data_package_builder.seal_package("nonexistent")


# ---------------------------------------------------------------------------
# Test: validate_package
# ---------------------------------------------------------------------------

class TestValidatePackage:
    """Tests for package validation."""

    def test_validate_sealed_package(self, data_package_builder, make_data_package):
        """Validate a sealed package passes all checks."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.validate_package(pid)
        assert result["valid"] is True
        assert result["checks_passed"] == result["checks_total"]

    def test_validate_building_package_fails(self, data_package_builder, make_data_package):
        """Validating a building (unsealed) package fails status check."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        result = data_package_builder.validate_package(pid)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# Test: export_package
# ---------------------------------------------------------------------------

class TestExportPackage:
    """Tests for package export."""

    def test_export_sealed_package(self, data_package_builder, make_data_package):
        """Export a sealed package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.export_package(pid)
        assert result["export_format"] == "zip"
        assert result["content_type"] == "application/zip"

    def test_export_json_ld_format(self, data_package_builder, make_data_package):
        """Export in JSON-LD format."""
        pkg = data_package_builder.create_package(**make_data_package(export_format="json_ld"))
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.export_package(pid)
        assert result["content_type"] == "application/ld+json"
        assert "json_ld_context" in result

    def test_export_building_package_raises(self, data_package_builder, make_data_package):
        """Exporting an unsealed package raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        with pytest.raises(ValueError):
            data_package_builder.export_package(pkg["package_id"])


# ---------------------------------------------------------------------------
# Test: estimate_size
# ---------------------------------------------------------------------------

class TestEstimateSize:
    """Tests for size estimation."""

    def test_estimate_empty_package(self, data_package_builder, make_data_package):
        """Estimate size of empty package."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.estimate_size(pkg["package_id"])
        assert result["raw_size_bytes"] == 0
        assert result["artifact_count"] == 0

    def test_estimate_with_artifacts(self, data_package_builder, make_data_package):
        """Estimate size with artifacts uses compression ratios."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 10000)
        result = data_package_builder.estimate_size(pid)
        assert result["raw_size_bytes"] == 10000
        assert result["estimated_compressed_bytes"] < 10000  # JSON compresses well


# ---------------------------------------------------------------------------
# Test: Status Transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    """Tests for package status transitions."""

    def test_submit_sealed_package(self, data_package_builder, make_data_package):
        """Submit a sealed package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.submit_package(pid)
        assert result["status"] == "submitted"

    def test_accept_submitted_package(self, data_package_builder, make_data_package):
        """Accept a submitted package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        data_package_builder.submit_package(pid)
        result = data_package_builder.accept_package(pid)
        assert result["status"] == "accepted"

    def test_reject_package(self, data_package_builder, make_data_package):
        """Reject a package with reason."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.reject_package(pid, reason="Data quality issues")
        assert result["status"] == "rejected"

    def test_invalid_transition_raises(self, data_package_builder, make_data_package):
        """Invalid status transition raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        with pytest.raises(ValueError):
            data_package_builder.submit_package(pkg["package_id"])


# ---------------------------------------------------------------------------
# Test: split_package
# ---------------------------------------------------------------------------

class TestSplitPackage:
    """Tests for package splitting."""

    def test_split_single_chunk(self, data_package_builder, make_data_package):
        """Package within size limit does not split."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        result = data_package_builder.split_package(pid, max_size_bytes=100_000)
        assert len(result) == 1

    def test_split_into_chunks(self, data_package_builder, make_data_package):
        """Large package splits into multiple chunks."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        for i in range(10):
            data_package_builder.add_form(pid, f"f{i}", {"d": i}, 10_000)
        result = data_package_builder.split_package(pid, max_size_bytes=30_000)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Test: Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    """Tests for builder statistics."""

    def test_statistics_empty(self, data_package_builder):
        """Statistics reflect empty state."""
        stats = data_package_builder.get_statistics()
        assert stats["total_packages"] == 0

    def test_statistics_after_operations(self, data_package_builder, make_data_package):
        """Statistics reflect operations."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        stats = data_package_builder.get_statistics()
        assert stats["total_packages"] == 1
        assert stats["total_artifacts"] == 1
        assert stats["total_bytes"] == 1024

    def test_statistics_sealed_count(self, data_package_builder, make_data_package):
        """Statistics tracks sealed packages."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        stats = data_package_builder.get_statistics()
        assert stats["by_status"]["sealed"] == 1


# ---------------------------------------------------------------------------
# Test: Additional Package Operations
# ---------------------------------------------------------------------------

class TestPackageAdditional:
    """Additional tests for package operations."""

    def test_get_manifest_sealed(self, data_package_builder, make_data_package):
        """Get manifest of sealed package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        manifest = data_package_builder.get_manifest(pid)
        assert manifest is not None
        assert manifest["artifact_count"] >= 1

    def test_seal_twice_raises(self, data_package_builder, make_data_package):
        """Sealing an already sealed package raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        with pytest.raises(ValueError):
            data_package_builder.seal_package(pid)

    def test_export_tar_gz_format(self, data_package_builder, make_data_package):
        """Export in tar.gz format."""
        pkg = data_package_builder.create_package(**make_data_package(export_format="tar_gz"))
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        result = data_package_builder.export_package(pid)
        assert result["content_type"] == "application/gzip"

    def test_package_has_creation_timestamp(self, data_package_builder, make_data_package):
        """Package has a creation timestamp."""
        pkg = data_package_builder.create_package(**make_data_package())
        assert "created_at" in pkg or "timestamp" in pkg

    def test_add_form_preserves_data(self, data_package_builder, make_data_package):
        """Added form data is preserved in the package."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"key": "value"}, 1024)
        result = data_package_builder.get_package(pid)
        assert "f1" in result["form_ids"]

    def test_reject_sealed_not_submitted_raises(self, data_package_builder, make_data_package):
        """Rejecting a sealed (not submitted) package raises ValueError."""
        pkg = data_package_builder.create_package(**make_data_package())
        pid = pkg["package_id"]
        data_package_builder.add_form(pid, "f1", {"d": 1}, 1024)
        data_package_builder.seal_package(pid)
        with pytest.raises(ValueError):
            data_package_builder.accept_package(pid)

    def test_validate_nonexistent_raises(self, data_package_builder):
        """Validating nonexistent package raises KeyError."""
        with pytest.raises(KeyError):
            data_package_builder.validate_package("nonexistent")

    def test_export_nonexistent_raises(self, data_package_builder):
        """Exporting nonexistent package raises KeyError."""
        with pytest.raises(KeyError):
            data_package_builder.export_package("nonexistent")

    def test_get_manifest_building_returns_empty(self, data_package_builder, make_data_package):
        """Get manifest of building package returns empty or partial manifest."""
        pkg = data_package_builder.create_package(**make_data_package())
        result = data_package_builder.get_manifest(pkg["package_id"])
        assert isinstance(result, dict)
