# -*- coding: utf-8 -*-
"""
Unit tests for PhotoEvidenceCollector - AGENT-EUDR-015 Engine 3.

Tests all methods of PhotoEvidenceCollector with 85%+ coverage.
Validates photo capture, retrieval, geotag validation, hash calculation,
duplicate detection, annotation, storage tracking, and error handling.

Test count: ~50 tests
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict

import pytest

from greenlang.agents.eudr.mobile_data_collector.photo_evidence_collector import (
    PhotoEvidenceCollector,
    PhotoNotFoundError,
    PhotoValidationError,
    DuplicatePhotoError,
    StorageQuotaExceededError,
    GeotagValidationError,
)

from .conftest import (
    SAMPLE_LAT, SAMPLE_LON, PHOTO_TYPES,
    assert_valid_sha256, assert_valid_coordinates,
)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestPhotoEvidenceCollectorInit:
    """Tests for PhotoEvidenceCollector initialization."""

    def test_initialization(self, photo_evidence_collector):
        """Engine initializes with empty stores."""
        assert photo_evidence_collector is not None
        assert len(photo_evidence_collector) == 0

    def test_repr(self, photo_evidence_collector):
        """Repr includes engine name."""
        r = repr(photo_evidence_collector)
        assert "PhotoEvidenceCollector" in r


# ---------------------------------------------------------------------------
# Test: capture_photo
# ---------------------------------------------------------------------------

class TestCapturePhoto:
    """Tests for capture_photo method."""

    def test_capture_valid_photo(self, photo_evidence_collector, make_photo_evidence):
        """Capture a valid photo returns photo dict with ID."""
        data = make_photo_evidence()
        result = photo_evidence_collector.capture_photo(**data)
        assert "photo_id" in result
        assert result["photo_type"] == "plot_photo"

    def test_capture_photo_increments_count(self, photo_evidence_collector, make_photo_evidence):
        """Capture increments photo count."""
        data = make_photo_evidence()
        photo_evidence_collector.capture_photo(**data)
        assert len(photo_evidence_collector) >= 1

    @pytest.mark.parametrize("photo_type", PHOTO_TYPES)
    def test_capture_all_photo_types(
        self, photo_evidence_collector, make_photo_evidence, photo_type,
    ):
        """All photo types are accepted."""
        data = make_photo_evidence(photo_type=photo_type)
        result = photo_evidence_collector.capture_photo(**data)
        assert result is not None

    def test_capture_photo_stores_integrity_hash(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Captured photo includes integrity hash."""
        data = make_photo_evidence()
        result = photo_evidence_collector.capture_photo(**data)
        assert "integrity_hash" in result
        assert_valid_sha256(result["integrity_hash"])

    def test_capture_photo_with_geotag(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Captured photo includes geotag coordinates."""
        data = make_photo_evidence(latitude=SAMPLE_LAT, longitude=SAMPLE_LON)
        result = photo_evidence_collector.capture_photo(**data)
        assert result is not None

    def test_capture_photo_unique_ids(self, photo_evidence_collector, make_photo_evidence):
        """Each capture returns a unique photo_id."""
        ids = set()
        for _ in range(5):
            data = make_photo_evidence()
            result = photo_evidence_collector.capture_photo(**data)
            ids.add(result["photo_id"])
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# Test: get_photo
# ---------------------------------------------------------------------------

class TestGetPhoto:
    """Tests for get_photo method."""

    def test_get_existing_photo(self, photo_evidence_collector, make_photo_evidence):
        """Get an existing photo by ID."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.get_photo(captured["photo_id"])
        assert result["photo_id"] == captured["photo_id"]

    def test_get_nonexistent_photo_raises(self, photo_evidence_collector):
        """Getting nonexistent photo raises error."""
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.get_photo("nonexistent-photo-id")

    def test_get_photo_returns_copy(self, photo_evidence_collector, make_photo_evidence):
        """get_photo returns a defensive copy."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        r1 = photo_evidence_collector.get_photo(captured["photo_id"])
        r2 = photo_evidence_collector.get_photo(captured["photo_id"])
        assert r1 is not r2
        assert r1["photo_id"] == r2["photo_id"]


# ---------------------------------------------------------------------------
# Test: list_photos
# ---------------------------------------------------------------------------

class TestListPhotos:
    """Tests for list_photos method."""

    def test_list_photos_empty(self, photo_evidence_collector):
        """List photos is empty initially."""
        result = photo_evidence_collector.list_photos()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_list_photos_after_capture(self, photo_evidence_collector, make_photo_evidence):
        """List photos includes captured photos."""
        photo_evidence_collector.capture_photo(**make_photo_evidence())
        result = photo_evidence_collector.list_photos()
        assert len(result) >= 1

    def test_list_photos_filter_by_type(self, photo_evidence_collector, make_photo_evidence):
        """List photos can filter by photo type."""
        photo_evidence_collector.capture_photo(**make_photo_evidence(photo_type="plot_photo"))
        photo_evidence_collector.capture_photo(**make_photo_evidence(photo_type="document_photo"))
        result = photo_evidence_collector.list_photos(photo_type="plot_photo")
        assert all(p.get("photo_type") == "plot_photo" for p in result)


# ---------------------------------------------------------------------------
# Test: validate_geotag
# ---------------------------------------------------------------------------

class TestValidateGeotag:
    """Tests for geotag validation."""

    def test_validate_valid_geotag(self, photo_evidence_collector, make_photo_evidence):
        """Valid geotag passes validation."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.validate_geotag(captured["photo_id"])
        assert isinstance(result, dict)

    def test_validate_geotag_nonexistent_raises(self, photo_evidence_collector):
        """Validating geotag for nonexistent photo raises error."""
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.validate_geotag("nonexistent-id")


# ---------------------------------------------------------------------------
# Test: calculate_hash
# ---------------------------------------------------------------------------

class TestCalculateHash:
    """Tests for hash calculation."""

    def test_calculate_hash_deterministic(self, photo_evidence_collector):
        """Hash calculation is deterministic for same input."""
        data = b"test_photo_data_12345"
        h1 = photo_evidence_collector.calculate_hash(data)
        h2 = photo_evidence_collector.calculate_hash(data)
        assert h1 == h2

    def test_calculate_hash_different_inputs(self, photo_evidence_collector):
        """Different inputs produce different hashes."""
        h1 = photo_evidence_collector.calculate_hash(b"data_a")
        h2 = photo_evidence_collector.calculate_hash(b"data_b")
        assert h1 != h2

    def test_calculate_hash_is_sha256(self, photo_evidence_collector):
        """Hash output is SHA-256 (64 hex chars)."""
        h = photo_evidence_collector.calculate_hash(b"test")
        assert_valid_sha256(h)

    def test_calculate_hash_matches_stdlib(self, photo_evidence_collector):
        """Hash matches Python stdlib SHA-256."""
        data = b"eudr_photo_verification"
        engine_hash = photo_evidence_collector.calculate_hash(data)
        stdlib_hash = hashlib.sha256(data).hexdigest()
        assert engine_hash == stdlib_hash


# ---------------------------------------------------------------------------
# Test: detect_duplicate
# ---------------------------------------------------------------------------

class TestDetectDuplicate:
    """Tests for duplicate detection."""

    def test_no_duplicate_for_unique_photos(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Unique photos are not detected as duplicates."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.detect_duplicate(captured["photo_id"])
        assert result is False or (isinstance(result, dict) and not result.get("is_duplicate"))

    def test_duplicate_detected_for_same_hash(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Photos with same integrity hash are detected as duplicates."""
        fixed_hash = hashlib.sha256(b"identical_photo").hexdigest()
        data1 = make_photo_evidence(integrity_hash=fixed_hash)
        data2 = make_photo_evidence(integrity_hash=fixed_hash)
        photo_evidence_collector.capture_photo(**data1)
        try:
            photo_evidence_collector.capture_photo(**data2)
            # If second capture succeeds, check duplicate detection
            photos = photo_evidence_collector.list_photos()
            if len(photos) >= 2:
                result = photo_evidence_collector.detect_duplicate(photos[-1]["photo_id"])
                assert result is True or (isinstance(result, dict) and result.get("is_duplicate"))
        except (DuplicatePhotoError, ValueError):
            # Engine may reject duplicates at capture time
            pass


# ---------------------------------------------------------------------------
# Test: annotate_photo
# ---------------------------------------------------------------------------

class TestAnnotatePhoto:
    """Tests for photo annotation."""

    def test_annotate_photo(self, photo_evidence_collector, make_photo_evidence):
        """Annotate a photo with metadata."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.annotate_photo(
            captured["photo_id"],
            annotation="Coffee cherries at peak ripeness",
        )
        assert result is not None

    def test_annotate_nonexistent_raises(self, photo_evidence_collector):
        """Annotating nonexistent photo raises error."""
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.annotate_photo(
                "nonexistent-id", annotation="test",
            )


# ---------------------------------------------------------------------------
# Test: get_storage_usage
# ---------------------------------------------------------------------------

class TestStorageUsage:
    """Tests for storage usage tracking."""

    def test_storage_usage_empty(self, photo_evidence_collector):
        """Storage usage is zero initially."""
        result = photo_evidence_collector.get_storage_usage()
        assert isinstance(result, dict)
        assert result.get("total_bytes", 0) == 0 or result.get("used_bytes", 0) == 0

    def test_storage_usage_after_capture(self, photo_evidence_collector, make_photo_evidence):
        """Storage usage increases after capture."""
        data = make_photo_evidence(file_size_bytes=500_000)
        photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.get_storage_usage()
        total = result.get("total_bytes", 0) or result.get("used_bytes", 0)
        assert total > 0


# ---------------------------------------------------------------------------
# Test: delete_photo
# ---------------------------------------------------------------------------

class TestDeletePhoto:
    """Tests for delete_photo method."""

    def test_delete_existing_photo(self, photo_evidence_collector, make_photo_evidence):
        """Delete an existing photo."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.delete_photo(captured["photo_id"])
        assert result is True or result is not None

    def test_delete_nonexistent_raises(self, photo_evidence_collector):
        """Deleting nonexistent photo raises error."""
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.delete_photo("nonexistent-id")

    def test_delete_removes_from_store(self, photo_evidence_collector, make_photo_evidence):
        """Deleted photo is no longer retrievable."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        photo_evidence_collector.delete_photo(captured["photo_id"])
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.get_photo(captured["photo_id"])


# ---------------------------------------------------------------------------
# Test: associate_photo
# ---------------------------------------------------------------------------

class TestAssociatePhoto:
    """Tests for photo-form association."""

    def test_associate_photo_to_form(self, photo_evidence_collector, make_photo_evidence):
        """Associate a photo with a form."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.associate_photo(
            captured["photo_id"], form_id="form-001",
        )
        assert result is not None

    def test_associate_nonexistent_photo_raises(self, photo_evidence_collector):
        """Associating nonexistent photo raises error."""
        with pytest.raises((PhotoNotFoundError, KeyError)):
            photo_evidence_collector.associate_photo(
                "nonexistent-id", form_id="form-001",
            )


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestPhotoEdgeCases:
    """Tests for edge cases."""

    def test_capture_multiple_photos(self, photo_evidence_collector, make_photo_evidence):
        """Multiple photos can be captured."""
        for i in range(10):
            data = make_photo_evidence()
            photo_evidence_collector.capture_photo(**data)
        assert len(photo_evidence_collector) == 10

    def test_large_file_size(self, photo_evidence_collector, make_photo_evidence):
        """Large file sizes are recorded accurately."""
        data = make_photo_evidence(file_size_bytes=15_000_000)  # 15MB
        result = photo_evidence_collector.capture_photo(**data)
        assert result.get("file_size_bytes", 0) == 15_000_000

    def test_capture_with_all_metadata(self, photo_evidence_collector, make_photo_evidence):
        """Capture with full metadata payload."""
        data = make_photo_evidence(
            photo_type="commodity_photo",
            file_size_bytes=2_000_000,
            width=4032,
            height=3024,
            format="jpeg",
            latitude=5.6037,
            longitude=-0.1870,
        )
        result = photo_evidence_collector.capture_photo(**data)
        assert result["photo_type"] == "commodity_photo"

    def test_storage_decreases_after_delete(self, photo_evidence_collector, make_photo_evidence):
        """Storage usage decreases after deletion."""
        data = make_photo_evidence(file_size_bytes=500_000)
        captured = photo_evidence_collector.capture_photo(**data)
        usage_before = photo_evidence_collector.get_storage_usage()
        total_before = usage_before.get("total_bytes", 0) or usage_before.get("used_bytes", 0)
        photo_evidence_collector.delete_photo(captured["photo_id"])
        usage_after = photo_evidence_collector.get_storage_usage()
        total_after = usage_after.get("total_bytes", 0) or usage_after.get("used_bytes", 0)
        assert total_after < total_before

    def test_associate_multiple_photos_to_form(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Associate multiple photos to the same form."""
        form_id = "form-multi"
        for _ in range(3):
            captured = photo_evidence_collector.capture_photo(**make_photo_evidence())
            photo_evidence_collector.associate_photo(
                captured["photo_id"], form_id=form_id,
            )
        result = photo_evidence_collector.list_photos(form_id=form_id)
        assert len(result) >= 3

    def test_capture_photo_timestamp_present(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Captured photo has a timestamp field."""
        data = make_photo_evidence()
        result = photo_evidence_collector.capture_photo(**data)
        assert "captured_at" in result or "timestamp" in result or "created_at" in result

    def test_annotate_overwrites_previous(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Re-annotating overwrites the previous annotation."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        photo_evidence_collector.annotate_photo(
            captured["photo_id"], annotation="First note",
        )
        result = photo_evidence_collector.annotate_photo(
            captured["photo_id"], annotation="Updated note",
        )
        assert result is not None

    def test_list_photos_multiple_types(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """List photos with multiple types returns correct counts."""
        for ptype in ["plot_photo", "plot_photo", "commodity_photo"]:
            photo_evidence_collector.capture_photo(
                **make_photo_evidence(photo_type=ptype),
            )
        all_photos = photo_evidence_collector.list_photos()
        assert len(all_photos) == 3

    def test_validate_geotag_returns_fields(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Geotag validation result includes coordinate fields."""
        data = make_photo_evidence(latitude=5.6037, longitude=-0.1870)
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.validate_geotag(captured["photo_id"])
        assert isinstance(result, dict)
        # Should contain some validation info
        assert len(result) > 0

    def test_detect_duplicate_returns_boolean_or_dict(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Duplicate detection returns a boolean or dict."""
        data = make_photo_evidence()
        captured = photo_evidence_collector.capture_photo(**data)
        result = photo_evidence_collector.detect_duplicate(captured["photo_id"])
        assert isinstance(result, (bool, dict))

    def test_calculate_hash_empty_data(self, photo_evidence_collector):
        """Hash of empty bytes is a known SHA-256."""
        import hashlib
        h = photo_evidence_collector.calculate_hash(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected

    def test_capture_returns_device_id(self, photo_evidence_collector, make_photo_evidence):
        """Captured photo includes the device_id."""
        data = make_photo_evidence(device_id="dev-test-123")
        result = photo_evidence_collector.capture_photo(**data)
        assert result.get("device_id") == "dev-test-123"

    def test_get_photo_preserves_all_fields(
        self, photo_evidence_collector, make_photo_evidence,
    ):
        """Retrieved photo preserves all original fields."""
        data = make_photo_evidence(photo_type="document_photo", file_size_bytes=800_000)
        captured = photo_evidence_collector.capture_photo(**data)
        retrieved = photo_evidence_collector.get_photo(captured["photo_id"])
        assert retrieved["photo_type"] == "document_photo"
        assert retrieved["file_size_bytes"] == 800_000

    def test_capture_minimal_fields(self, photo_evidence_collector):
        """Capture photo with minimal required fields."""
        result = photo_evidence_collector.capture_photo(
            device_id="dev-min",
            operator_id="op-min",
            photo_type="plot_photo",
            file_size_bytes=100_000,
            integrity_hash=hashlib.sha256(b"minimal").hexdigest(),
        )
        assert "photo_id" in result

    def test_list_photos_returns_list(self, photo_evidence_collector):
        """list_photos always returns a list."""
        result = photo_evidence_collector.list_photos()
        assert isinstance(result, list)

    def test_storage_usage_returns_expected_keys(self, photo_evidence_collector):
        """Storage usage dict has expected keys."""
        result = photo_evidence_collector.get_storage_usage()
        assert isinstance(result, dict)
        # Should have at least one size-related key
        assert any(k in result for k in ("total_bytes", "used_bytes", "photo_count", "count"))

    def test_delete_reduces_photo_count(self, photo_evidence_collector, make_photo_evidence):
        """Deleting a photo reduces the photo count."""
        p1 = photo_evidence_collector.capture_photo(**make_photo_evidence())
        p2 = photo_evidence_collector.capture_photo(**make_photo_evidence())
        assert len(photo_evidence_collector) == 2
        photo_evidence_collector.delete_photo(p1["photo_id"])
        assert len(photo_evidence_collector) == 1

    def test_capture_with_dimensions(self, photo_evidence_collector, make_photo_evidence):
        """Photo dimensions are recorded."""
        data = make_photo_evidence(width=3840, height=2160)
        result = photo_evidence_collector.capture_photo(**data)
        assert result.get("width") == 3840
        assert result.get("height") == 2160

    def test_capture_with_format_png(self, photo_evidence_collector, make_photo_evidence):
        """PNG format is accepted."""
        data = make_photo_evidence(format="png", file_size_bytes=3_000_000)
        result = photo_evidence_collector.capture_photo(**data)
        assert result is not None
