# -*- coding: utf-8 -*-
"""
Photo Evidence Collector - AGENT-EUDR-015 Mobile Data Collector (Engine 3)

Production-grade geotagged photo evidence collection engine with SHA-256
integrity hashing for EUDR compliance covering photo metadata recording
(dimensions, format, size, capture timestamp), EXIF-like metadata
extraction (lat, lon, altitude, bearing, device model), SHA-256 content
hash at capture time for content-addressable storage, photo type
classification (field_photo, product_photo, document_scan, aerial_view,
infrastructure, evidence), JPEG quality tiers (high/medium/low), batch
photo sequencing, geotag validation with configurable proximity threshold,
annotation support, photo-to-form/GPS/signature association, duplicate
detection via hash comparison, and per-device storage quota tracking.

Zero-Hallucination Guarantees:
    - All hash computations use deterministic SHA-256
    - Distance calculations for geotag validation use Haversine formula
    - Storage quota tracking is simple arithmetic
    - No LLM calls in any photo processing path
    - SHA-256 provenance recorded for every mutation

PRD: PRD-AGENT-EUDR-015 Feature F3 (Photo Evidence Collection)
Agent ID: GL-EUDR-MDC-015
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 14

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mobile_data_collector.config import get_config
from greenlang.agents.eudr.mobile_data_collector.metrics import (
    observe_photo_upload_duration,
    record_api_error,
    record_photo_captured,
    set_storage_used_bytes,
)
from greenlang.agents.eudr.mobile_data_collector.models import (
    PhotoEvidence,
    PhotoResponse,
    PhotoType,
)
from greenlang.agents.eudr.mobile_data_collector.provenance import (
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mean Earth radius in meters for Haversine distance calculation.
_EARTH_RADIUS_M: float = 6371008.8

#: Default geotag proximity threshold in meters.
_DEFAULT_GEOTAG_THRESHOLD_M: float = 500.0

#: Mapping of photo type string aliases to PhotoType enum values.
_PHOTO_TYPE_MAP: Dict[str, PhotoType] = {
    "field_photo": PhotoType.PLOT_PHOTO,
    "product_photo": PhotoType.COMMODITY_PHOTO,
    "document_scan": PhotoType.DOCUMENT_PHOTO,
    "aerial_view": PhotoType.FACILITY_PHOTO,
    "infrastructure": PhotoType.TRANSPORT_PHOTO,
    "evidence": PhotoType.IDENTITY_PHOTO,
    # Direct enum values
    "plot_photo": PhotoType.PLOT_PHOTO,
    "commodity_photo": PhotoType.COMMODITY_PHOTO,
    "document_photo": PhotoType.DOCUMENT_PHOTO,
    "facility_photo": PhotoType.FACILITY_PHOTO,
    "transport_photo": PhotoType.TRANSPORT_PHOTO,
    "identity_photo": PhotoType.IDENTITY_PHOTO,
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PhotoEvidenceError(Exception):
    """Base exception for photo evidence collector operations."""


class PhotoNotFoundError(PhotoEvidenceError):
    """Raised when a photo record cannot be found."""


class PhotoValidationError(PhotoEvidenceError):
    """Raised when photo metadata fails validation."""


class DuplicatePhotoError(PhotoEvidenceError):
    """Raised when a duplicate photo is detected via hash comparison."""


class StorageQuotaExceededError(PhotoEvidenceError):
    """Raised when device storage quota is exceeded."""


class GeotagValidationError(PhotoEvidenceError):
    """Raised when photo geotag fails proximity validation."""


# ---------------------------------------------------------------------------
# PhotoEvidenceCollector
# ---------------------------------------------------------------------------


class PhotoEvidenceCollector:
    """Geotagged photo evidence collection engine with integrity hashing.

    Manages the complete lifecycle of photo evidence from capture through
    validation, storage, duplicate detection, and form association for
    EUDR Article 9 compliance.

    Thread Safety:
        All public methods are protected by a reentrant lock for
        concurrent access from multiple API handlers.

    Attributes:
        _config: Agent configuration instance.
        _photos: In-memory photo store keyed by photo_id.
        _hash_index: Reverse index from integrity_hash to photo_id
            for O(1) duplicate detection.
        _device_storage: Per-device storage usage in bytes.
        _provenance: Provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> collector = PhotoEvidenceCollector()
        >>> result = collector.capture_photo(
        ...     device_id="dev-001",
        ...     operator_id="op-001",
        ...     form_id="form-001",
        ...     photo_type="plot_photo",
        ...     file_name="IMG_001.jpg",
        ...     file_size_bytes=2048000,
        ...     width_px=3840,
        ...     height_px=2160,
        ...     content_bytes=b"...",
        ... )
        >>> assert result.integrity_hash is not None
    """

    __slots__ = (
        "_config",
        "_photos",
        "_hash_index",
        "_device_storage",
        "_provenance",
        "_lock",
    )

    def __init__(self) -> None:
        """Initialize the PhotoEvidenceCollector with empty stores."""
        self._config = get_config()
        self._photos: Dict[str, PhotoEvidence] = {}
        self._hash_index: Dict[str, str] = {}
        self._device_storage: Dict[str, int] = {}
        self._provenance = get_provenance_tracker()
        self._lock = threading.RLock()
        logger.info(
            "PhotoEvidenceCollector initialized: max_size=%dMB, "
            "quality=%d/%d/%d, max_per_form=%d, "
            "min_res=%dx%d, formats=%s",
            self._config.max_photo_size_mb,
            self._config.compression_quality_high,
            self._config.compression_quality_medium,
            self._config.compression_quality_low,
            self._config.max_photos_per_form,
            self._config.min_photo_width,
            self._config.min_photo_height,
            self._config.supported_photo_formats,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_photo(
        self,
        device_id: str,
        operator_id: str,
        form_id: str,
        photo_type: str,
        file_name: str,
        file_size_bytes: int,
        width_px: int,
        height_px: int,
        content_bytes: Optional[bytes] = None,
        integrity_hash: Optional[str] = None,
        file_format: str = "jpeg",
        capture_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude_m: Optional[float] = None,
        bearing: Optional[float] = None,
        device_model: Optional[str] = None,
        exif_timestamp: Optional[datetime] = None,
        device_timestamp: Optional[datetime] = None,
        compression_quality: Optional[int] = None,
        annotation: Optional[str] = None,
        sequence_number: Optional[int] = None,
        batch_group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhotoResponse:
        """Capture and record a geotagged photo with integrity hash.

        Validates photo metadata, computes SHA-256 hash of content (or
        uses provided hash), checks for duplicates, stores the record,
        and updates per-device storage tracking.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent identifier.
            form_id: Associated form submission identifier.
            photo_type: Photo category classification.
            file_name: Original file name on device.
            file_size_bytes: Photo file size in bytes.
            width_px: Image width in pixels.
            height_px: Image height in pixels.
            content_bytes: Raw image bytes for hash computation.
            integrity_hash: Pre-computed SHA-256 hash (if content_bytes
                not provided).
            file_format: Image format (jpeg, png, heic).
            capture_id: Associated GPS capture identifier.
            latitude: GPS latitude where photo was taken.
            longitude: GPS longitude where photo was taken.
            altitude_m: Altitude above sea level in meters.
            bearing: Camera bearing in degrees.
            device_model: Device hardware model name.
            exif_timestamp: EXIF timestamp from photo metadata.
            device_timestamp: Device system time at capture.
            compression_quality: JPEG compression quality applied.
            annotation: Optional text annotation.
            sequence_number: Sequence number within batch capture.
            batch_group_id: Batch capture group identifier.
            metadata: Additional photo metadata.

        Returns:
            PhotoResponse with photo_id, integrity_hash, and provenance.

        Raises:
            PhotoValidationError: If photo metadata fails validation.
            DuplicatePhotoError: If an identical photo already exists.
            StorageQuotaExceededError: If device quota is exceeded.
            PhotoEvidenceError: If capture processing fails.
        """
        start_time = time.monotonic()
        try:
            # Resolve photo type
            resolved_type = self._resolve_photo_type(photo_type)

            # Validate metadata
            self._validate_photo_metadata(
                file_size_bytes=file_size_bytes,
                width_px=width_px,
                height_px=height_px,
                file_format=file_format,
            )

            # Compute or use provided hash
            computed_hash = self._compute_or_use_hash(
                content_bytes, integrity_hash,
            )

            # Check for duplicates
            self._check_duplicate(computed_hash)

            # Check storage quota
            self._check_storage_quota(device_id, file_size_bytes)

            # Check form photo count limit
            self._check_form_photo_limit(form_id)

            # Build EXIF-like metadata
            exif_meta = self._build_exif_metadata(
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude_m,
                bearing=bearing,
                device_model=device_model,
            )

            # Merge into metadata
            final_metadata = dict(metadata or {})
            if exif_meta:
                final_metadata["exif"] = exif_meta

            # Build photo record
            now = datetime.now(timezone.utc).replace(microsecond=0)
            photo = PhotoEvidence(
                form_id=form_id,
                capture_id=capture_id,
                device_id=device_id,
                operator_id=operator_id,
                photo_type=resolved_type,
                file_name=file_name,
                file_size_bytes=file_size_bytes,
                file_format=file_format.lower(),
                width_px=width_px,
                height_px=height_px,
                integrity_hash=computed_hash,
                hash_algorithm="sha256",
                latitude=latitude,
                longitude=longitude,
                exif_timestamp=exif_timestamp,
                device_timestamp=device_timestamp or now,
                compression_quality=compression_quality,
                annotation=annotation,
                sequence_number=sequence_number,
                batch_group_id=batch_group_id,
                metadata=final_metadata,
                created_at=now,
            )

            # Store photo and update indices
            with self._lock:
                self._photos[photo.photo_id] = photo
                self._hash_index[computed_hash] = photo.photo_id
                self._device_storage[device_id] = (
                    self._device_storage.get(device_id, 0)
                    + file_size_bytes
                )

            # Update metrics
            total_storage = sum(self._device_storage.values())
            set_storage_used_bytes(total_storage)

            # Record provenance
            provenance_entry = self._provenance.record(
                entity_type="photo_evidence",
                action="capture",
                entity_id=photo.photo_id,
                data={
                    "file_name": file_name,
                    "file_size": file_size_bytes,
                    "integrity_hash": computed_hash,
                    "photo_type": resolved_type.value,
                    "dimensions": f"{width_px}x{height_px}",
                },
                metadata={
                    "device_id": device_id,
                    "form_id": form_id,
                },
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            observe_photo_upload_duration(elapsed_ms / 1000)
            record_photo_captured(resolved_type.value)

            logger.info(
                "Photo captured: id=%s type=%s file=%s "
                "size=%d hash_prefix=%s elapsed=%.1fms",
                photo.photo_id, resolved_type.value, file_name,
                file_size_bytes, computed_hash[:16], elapsed_ms,
            )

            return PhotoResponse(
                photo_id=photo.photo_id,
                integrity_hash=computed_hash,
                provenance_hash=provenance_entry.hash_value,
                processing_time_ms=elapsed_ms,
                message="Photo captured successfully",
                photo=photo,
            )

        except (
            PhotoValidationError,
            DuplicatePhotoError,
            StorageQuotaExceededError,
        ):
            record_api_error("capture")
            raise
        except Exception as e:
            record_api_error("capture")
            logger.error(
                "Photo capture failed: %s", str(e), exc_info=True,
            )
            raise PhotoEvidenceError(
                f"Photo capture failed: {str(e)}"
            ) from e

    def get_photo(self, photo_id: str) -> PhotoEvidence:
        """Retrieve a photo record by its identifier.

        Args:
            photo_id: Photo evidence identifier.

        Returns:
            PhotoEvidence instance.

        Raises:
            PhotoNotFoundError: If the photo_id does not exist.
        """
        with self._lock:
            photo = self._photos.get(photo_id)
        if photo is None:
            raise PhotoNotFoundError(
                f"Photo not found: photo_id={photo_id}"
            )
        return photo

    def list_photos(
        self,
        form_id: Optional[str] = None,
        device_id: Optional[str] = None,
        photo_type: Optional[str] = None,
        batch_group_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[PhotoEvidence]:
        """List photo records with optional filters.

        Args:
            form_id: Filter by associated form identifier.
            device_id: Filter by device identifier.
            photo_type: Filter by photo type.
            batch_group_id: Filter by batch group identifier.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching PhotoEvidence instances.
        """
        with self._lock:
            photos = list(self._photos.values())

        if form_id:
            photos = [p for p in photos if p.form_id == form_id]
        if device_id:
            photos = [p for p in photos if p.device_id == device_id]
        if photo_type:
            resolved = self._resolve_photo_type(photo_type)
            photos = [
                p for p in photos if p.photo_type == resolved
            ]
        if batch_group_id:
            photos = [
                p for p in photos
                if p.batch_group_id == batch_group_id
            ]

        # Sort by sequence_number then created_at
        photos.sort(
            key=lambda p: (
                p.sequence_number or 0,
                p.created_at,
            )
        )

        return photos[offset: offset + limit]

    def validate_geotag(
        self,
        photo_id: str,
        reference_lat: float,
        reference_lon: float,
        threshold_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate photo geotag against a reference GPS position.

        Checks that the photo's GPS coordinates are within the
        configured proximity threshold of the reference point.

        Args:
            photo_id: Photo evidence identifier.
            reference_lat: Reference latitude for proximity check.
            reference_lon: Reference longitude for proximity check.
            threshold_m: Maximum distance in meters. Defaults to
                config or 500m.

        Returns:
            Dictionary with is_valid, distance_m, threshold_m, and
            photo coordinates.

        Raises:
            PhotoNotFoundError: If the photo does not exist.
            GeotagValidationError: If photo has no GPS coordinates.
        """
        photo = self.get_photo(photo_id)
        if photo.latitude is None or photo.longitude is None:
            raise GeotagValidationError(
                f"Photo {photo_id} has no GPS coordinates"
            )

        if threshold_m is None:
            threshold_m = _DEFAULT_GEOTAG_THRESHOLD_M

        distance = self._haversine_distance(
            photo.latitude, photo.longitude,
            reference_lat, reference_lon,
        )

        is_valid = distance <= threshold_m

        if not is_valid:
            logger.warning(
                "Geotag validation failed: photo=%s distance=%.1fm "
                "threshold=%.1fm",
                photo_id, distance, threshold_m,
            )

        return {
            "is_valid": is_valid,
            "distance_m": round(distance, 3),
            "threshold_m": threshold_m,
            "photo_latitude": photo.latitude,
            "photo_longitude": photo.longitude,
            "reference_latitude": reference_lat,
            "reference_longitude": reference_lon,
        }

    def calculate_hash(
        self,
        content_bytes: bytes,
    ) -> str:
        """Compute SHA-256 hash of raw image bytes.

        Args:
            content_bytes: Raw image bytes.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        return hashlib.sha256(content_bytes).hexdigest()

    def detect_duplicate(
        self,
        integrity_hash: str,
    ) -> Dict[str, Any]:
        """Check if a photo with the given hash already exists.

        Args:
            integrity_hash: SHA-256 hash of photo content.

        Returns:
            Dictionary with is_duplicate and existing_photo_id.
        """
        with self._lock:
            existing_id = self._hash_index.get(integrity_hash)

        return {
            "is_duplicate": existing_id is not None,
            "integrity_hash": integrity_hash,
            "existing_photo_id": existing_id,
        }

    def annotate_photo(
        self,
        photo_id: str,
        annotation: str,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    ) -> PhotoEvidence:
        """Add text annotation and optional bounding boxes to a photo.

        Args:
            photo_id: Photo evidence identifier.
            annotation: Text annotation to add.
            bounding_boxes: Optional list of region bounding boxes,
                each with x, y, width, height, and label keys.

        Returns:
            Updated PhotoEvidence instance.

        Raises:
            PhotoNotFoundError: If the photo does not exist.
        """
        with self._lock:
            photo = self._photos.get(photo_id)
            if photo is None:
                raise PhotoNotFoundError(
                    f"Photo not found: photo_id={photo_id}"
                )

            photo.annotation = annotation
            if bounding_boxes:
                photo.metadata["bounding_boxes"] = bounding_boxes
            photo.metadata["annotated_at"] = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )

        self._provenance.record(
            entity_type="photo_evidence",
            action="update",
            entity_id=photo_id,
            data={
                "annotation": annotation,
                "has_bounding_boxes": bounding_boxes is not None,
            },
        )

        logger.info(
            "Photo annotated: id=%s annotation_len=%d boxes=%d",
            photo_id, len(annotation),
            len(bounding_boxes) if bounding_boxes else 0,
        )

        return photo

    def get_storage_usage(
        self,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get storage usage statistics for a device or all devices.

        Args:
            device_id: Optional device identifier. If None, returns
                aggregate across all devices.

        Returns:
            Dictionary with total_bytes, photo_count, and per-device
            breakdown.
        """
        with self._lock:
            if device_id:
                device_bytes = self._device_storage.get(device_id, 0)
                device_photos = sum(
                    1 for p in self._photos.values()
                    if p.device_id == device_id
                )
                return {
                    "device_id": device_id,
                    "total_bytes": device_bytes,
                    "total_mb": round(device_bytes / (1024 * 1024), 2),
                    "photo_count": device_photos,
                }

            total_bytes = sum(self._device_storage.values())
            total_photos = len(self._photos)
            per_device = {
                did: {
                    "total_bytes": usage,
                    "total_mb": round(usage / (1024 * 1024), 2),
                }
                for did, usage in self._device_storage.items()
            }

            return {
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 2),
                "photo_count": total_photos,
                "device_count": len(self._device_storage),
                "per_device": per_device,
            }

    def delete_photo(self, photo_id: str) -> Dict[str, Any]:
        """Delete a photo record and update storage tracking.

        Args:
            photo_id: Photo evidence identifier.

        Returns:
            Dictionary with deleted photo_id and freed bytes.

        Raises:
            PhotoNotFoundError: If the photo does not exist.
        """
        with self._lock:
            photo = self._photos.get(photo_id)
            if photo is None:
                raise PhotoNotFoundError(
                    f"Photo not found: photo_id={photo_id}"
                )

            freed_bytes = photo.file_size_bytes
            del self._photos[photo_id]
            self._hash_index.pop(photo.integrity_hash, None)

            if photo.device_id in self._device_storage:
                self._device_storage[photo.device_id] = max(
                    0,
                    self._device_storage[photo.device_id] - freed_bytes,
                )

        self._provenance.record(
            entity_type="photo_evidence",
            action="update",
            entity_id=photo_id,
            data={"action": "delete", "freed_bytes": freed_bytes},
        )

        logger.info(
            "Photo deleted: id=%s freed=%d bytes", photo_id, freed_bytes,
        )

        return {
            "photo_id": photo_id,
            "deleted": True,
            "freed_bytes": freed_bytes,
        }

    def associate_photo(
        self,
        photo_id: str,
        form_id: Optional[str] = None,
        capture_id: Optional[str] = None,
        signature_id: Optional[str] = None,
    ) -> PhotoEvidence:
        """Associate a photo with forms, GPS captures, or signatures.

        Args:
            photo_id: Photo evidence identifier.
            form_id: Form submission identifier to associate.
            capture_id: GPS capture identifier to associate.
            signature_id: Digital signature identifier to associate.

        Returns:
            Updated PhotoEvidence instance.

        Raises:
            PhotoNotFoundError: If the photo does not exist.
        """
        with self._lock:
            photo = self._photos.get(photo_id)
            if photo is None:
                raise PhotoNotFoundError(
                    f"Photo not found: photo_id={photo_id}"
                )

            if form_id:
                photo.form_id = form_id
            if capture_id:
                photo.capture_id = capture_id
            if signature_id:
                if "signature_ids" not in photo.metadata:
                    photo.metadata["signature_ids"] = []
                if signature_id not in photo.metadata["signature_ids"]:
                    photo.metadata["signature_ids"].append(signature_id)

        associations = {
            k: v for k, v in {
                "form_id": form_id,
                "capture_id": capture_id,
                "signature_id": signature_id,
            }.items() if v is not None
        }

        self._provenance.record(
            entity_type="photo_evidence",
            action="update",
            entity_id=photo_id,
            data={"associations": associations},
        )

        logger.info(
            "Photo associated: id=%s associations=%s",
            photo_id, associations,
        )

        return photo

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def photo_count(self) -> int:
        """Return the total number of stored photos."""
        with self._lock:
            return len(self._photos)

    @property
    def total_storage_bytes(self) -> int:
        """Return total storage used across all devices."""
        with self._lock:
            return sum(self._device_storage.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_photo_type(self, photo_type: str) -> PhotoType:
        """Resolve a photo type string to a PhotoType enum value.

        Args:
            photo_type: Photo type string or alias.

        Returns:
            Resolved PhotoType enum value.

        Raises:
            PhotoValidationError: If photo_type is not recognized.
        """
        normalized = photo_type.lower().strip()
        resolved = _PHOTO_TYPE_MAP.get(normalized)
        if resolved is None:
            try:
                resolved = PhotoType(normalized)
            except ValueError:
                valid = sorted(set(_PHOTO_TYPE_MAP.keys()))
                raise PhotoValidationError(
                    f"Invalid photo_type '{photo_type}'; "
                    f"must be one of {valid}"
                )
        return resolved

    def _validate_photo_metadata(
        self,
        file_size_bytes: int,
        width_px: int,
        height_px: int,
        file_format: str,
    ) -> None:
        """Validate photo metadata against configuration limits.

        Args:
            file_size_bytes: Photo file size in bytes.
            width_px: Image width in pixels.
            height_px: Image height in pixels.
            file_format: Image format string.

        Raises:
            PhotoValidationError: If any metadata value is invalid.
        """
        errors: List[str] = []

        max_bytes = self._config.max_photo_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            errors.append(
                f"File size {file_size_bytes} bytes exceeds max "
                f"{self._config.max_photo_size_mb}MB"
            )

        min_bytes = self._config.min_photo_file_size_bytes
        if file_size_bytes < min_bytes:
            errors.append(
                f"File size {file_size_bytes} bytes is below min "
                f"{min_bytes} bytes (possible corruption)"
            )

        if width_px < self._config.min_photo_width:
            errors.append(
                f"Width {width_px}px is below min "
                f"{self._config.min_photo_width}px"
            )

        if height_px < self._config.min_photo_height:
            errors.append(
                f"Height {height_px}px is below min "
                f"{self._config.min_photo_height}px"
            )

        normalized_format = file_format.lower().strip()
        if normalized_format not in self._config.supported_photo_formats:
            errors.append(
                f"Unsupported format '{file_format}'; "
                f"must be one of {self._config.supported_photo_formats}"
            )

        if errors:
            raise PhotoValidationError(
                "Photo validation failed: " + "; ".join(errors)
            )

    def _compute_or_use_hash(
        self,
        content_bytes: Optional[bytes],
        integrity_hash: Optional[str],
    ) -> str:
        """Compute SHA-256 hash from content bytes or use provided hash.

        Args:
            content_bytes: Raw image bytes (preferred).
            integrity_hash: Pre-computed hash (fallback).

        Returns:
            Hex-encoded SHA-256 hash string.

        Raises:
            PhotoValidationError: If neither bytes nor hash provided.
        """
        if content_bytes is not None:
            return hashlib.sha256(content_bytes).hexdigest()
        if integrity_hash is not None:
            return integrity_hash
        raise PhotoValidationError(
            "Either content_bytes or integrity_hash must be provided"
        )

    def _check_duplicate(self, integrity_hash: str) -> None:
        """Check if a photo with this hash already exists.

        Args:
            integrity_hash: SHA-256 hash to check.

        Raises:
            DuplicatePhotoError: If a duplicate exists.
        """
        with self._lock:
            existing_id = self._hash_index.get(integrity_hash)
        if existing_id is not None:
            raise DuplicatePhotoError(
                f"Duplicate photo detected: hash={integrity_hash[:16]}... "
                f"matches existing photo_id={existing_id}"
            )

    def _check_storage_quota(
        self,
        device_id: str,
        file_size_bytes: int,
    ) -> None:
        """Check that adding this photo will not exceed storage quota.

        Uses the max_photo_size_mb * max_photos_per_form * 10 as an
        approximate per-device storage cap.

        Args:
            device_id: Device identifier.
            file_size_bytes: Size of the new photo in bytes.

        Raises:
            StorageQuotaExceededError: If quota would be exceeded.
        """
        quota_bytes = (
            self._config.max_photo_size_mb
            * self._config.max_photos_per_form
            * 10
            * 1024 * 1024
        )

        with self._lock:
            current_usage = self._device_storage.get(device_id, 0)

        if current_usage + file_size_bytes > quota_bytes:
            raise StorageQuotaExceededError(
                f"Device {device_id} storage quota exceeded: "
                f"current={current_usage}, new={file_size_bytes}, "
                f"quota={quota_bytes}"
            )

    def _check_form_photo_limit(self, form_id: str) -> None:
        """Check that the form has not exceeded its photo limit.

        Args:
            form_id: Form submission identifier.

        Raises:
            PhotoValidationError: If limit exceeded.
        """
        with self._lock:
            count = sum(
                1 for p in self._photos.values()
                if p.form_id == form_id
            )
        if count >= self._config.max_photos_per_form:
            raise PhotoValidationError(
                f"Form {form_id} has reached the max "
                f"{self._config.max_photos_per_form} photos"
            )

    def _build_exif_metadata(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        altitude_m: Optional[float],
        bearing: Optional[float],
        device_model: Optional[str],
    ) -> Dict[str, Any]:
        """Build EXIF-like metadata dictionary from capture parameters.

        Args:
            latitude: GPS latitude.
            longitude: GPS longitude.
            altitude_m: Altitude in meters.
            bearing: Camera bearing in degrees.
            device_model: Device model name.

        Returns:
            Dictionary of EXIF-like metadata fields.
        """
        exif: Dict[str, Any] = {}
        if latitude is not None:
            exif["gps_latitude"] = latitude
        if longitude is not None:
            exif["gps_longitude"] = longitude
        if altitude_m is not None:
            exif["gps_altitude_m"] = altitude_m
        if bearing is not None:
            exif["gps_bearing"] = bearing
        if device_model is not None:
            exif["device_model"] = device_model
        return exif

    @staticmethod
    def _haversine_distance(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points using Haversine formula.

        Args:
            lat1: Latitude of first point.
            lon1: Longitude of first point.
            lat2: Latitude of second point.
            lon2: Longitude of second point.

        Returns:
            Distance in meters.
        """
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r)
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(_EARTH_RADIUS_M * c, 3)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"PhotoEvidenceCollector(photos={self.photo_count}, "
            f"storage={self.total_storage_bytes}B)"
        )

    def __len__(self) -> int:
        """Return the total number of stored photos."""
        return self.photo_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PhotoEvidenceCollector",
    "PhotoEvidenceError",
    "PhotoNotFoundError",
    "PhotoValidationError",
    "DuplicatePhotoError",
    "StorageQuotaExceededError",
    "GeotagValidationError",
]
