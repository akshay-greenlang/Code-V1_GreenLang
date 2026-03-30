# -*- coding: utf-8 -*-
"""
Custody Event Tracker Engine - AGENT-EUDR-009: Chain of Custody (Feature 1)

Records, validates, and manages every change-of-custody event within EUDR
supply chains. Each custody transfer is immutably recorded with temporal,
actor, and location continuity validation. Supports 10 event types (transfer,
receipt, storage_in, storage_out, processing_in, processing_out, export,
import, inspection, sampling), amendment tracking with original preservation,
and bulk import from EDI/XML/CSV external formats.

Zero-Hallucination Guarantees:
    - All validation is deterministic (datetime comparison, string equality,
      numeric thresholds).
    - Temporal ordering uses strict datetime comparison (no ML/LLM).
    - Actor continuity is exact string match (receiver == next sender).
    - Location continuity is exact facility_id equality.
    - Gap detection is pure arithmetic (hours between events).
    - SHA-256 provenance hashes on all event records and results.
    - No ML/LLM used for any validation or tracking logic.

Performance Targets:
    - Single event recording: <5ms
    - Temporal order validation: <2ms
    - Actor/location continuity check: <2ms
    - Gap detection (100 events): <10ms
    - Bulk import (10,000 events): <5 seconds

Regulatory References:
    - EUDR Article 4: Due diligence obligations requiring traceability.
    - EUDR Article 9: Information requirements including custody chain.
    - EUDR Article 10: Risk assessment based on custody gaps.
    - ISO 22095: Chain of custody -- General terminology and models.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Feature 1: Custody Event Tracking)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ValidationSeverity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum gap threshold in hours before a custody gap is flagged.
DEFAULT_GAP_THRESHOLD_HOURS: float = 72.0

#: Maximum number of events in a single bulk import batch.
MAX_BULK_IMPORT_SIZE: int = 50_000

#: Supported source formats for bulk import.
SUPPORTED_IMPORT_FORMATS: Tuple[str, ...] = ("edi", "xml", "csv", "json")

#: Maximum number of amendments allowed per event.
MAX_AMENDMENTS_PER_EVENT: int = 50

#: Supported amendment fields (fields that can be amended on an event).
AMENDABLE_FIELDS: Tuple[str, ...] = (
    "quantity_kg",
    "notes",
    "documents",
    "quality_grade",
    "transport_mode",
    "temperature_celsius",
    "humidity_pct",
    "handler_name",
    "handler_id",
    "transport_vehicle_id",
    "seal_numbers",
)

#: Fields that must NEVER be amended (immutable core).
IMMUTABLE_FIELDS: Tuple[str, ...] = (
    "event_id",
    "batch_id",
    "event_type",
    "timestamp",
    "created_at",
    "provenance_hash",
)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Custody event types per EUDR supply chain flow."""

    TRANSFER = "transfer"
    RECEIPT = "receipt"
    STORAGE_IN = "storage_in"
    STORAGE_OUT = "storage_out"
    PROCESSING_IN = "processing_in"
    PROCESSING_OUT = "processing_out"
    EXPORT = "export"
    IMPORT = "import"
    INSPECTION = "inspection"
    SAMPLING = "sampling"

class EventStatus(str, Enum):
    """Status of a custody event."""

    ACTIVE = "active"
    AMENDED = "amended"
    SUPERSEDED = "superseded"
    VOIDED = "voided"

class ImportSourceFormat(str, Enum):
    """Supported external import formats."""

    EDI = "edi"
    XML = "xml"
    CSV = "csv"
    JSON = "json"

# ---------------------------------------------------------------------------
# Event type ordering rules
# ---------------------------------------------------------------------------

#: Valid predecessor event types for each event type.
#: Empty tuple means the event can be the first event in a chain.
VALID_PREDECESSORS: Dict[str, Tuple[str, ...]] = {
    EventType.TRANSFER: (
        EventType.STORAGE_OUT,
        EventType.PROCESSING_OUT,
        EventType.RECEIPT,
        EventType.TRANSFER,
    ),
    EventType.RECEIPT: (
        EventType.TRANSFER,
        EventType.IMPORT,
    ),
    EventType.STORAGE_IN: (
        EventType.RECEIPT,
        EventType.PROCESSING_OUT,
        EventType.IMPORT,
    ),
    EventType.STORAGE_OUT: (
        EventType.STORAGE_IN,
        EventType.INSPECTION,
        EventType.SAMPLING,
    ),
    EventType.PROCESSING_IN: (
        EventType.STORAGE_OUT,
        EventType.RECEIPT,
    ),
    EventType.PROCESSING_OUT: (
        EventType.PROCESSING_IN,
    ),
    EventType.EXPORT: (
        EventType.STORAGE_OUT,
        EventType.PROCESSING_OUT,
        EventType.INSPECTION,
    ),
    EventType.IMPORT: (
        EventType.EXPORT,
        EventType.TRANSFER,
    ),
    EventType.INSPECTION: (
        EventType.STORAGE_IN,
        EventType.RECEIPT,
        EventType.PROCESSING_OUT,
        EventType.IMPORT,
    ),
    EventType.SAMPLING: (
        EventType.STORAGE_IN,
        EventType.RECEIPT,
        EventType.INSPECTION,
        EventType.IMPORT,
    ),
}

#: Event types that represent physical movement (need actor/location continuity).
MOVEMENT_EVENT_TYPES: Tuple[str, ...] = (
    EventType.TRANSFER,
    EventType.RECEIPT,
    EventType.EXPORT,
    EventType.IMPORT,
)

#: Event types that occur within a single facility.
FACILITY_EVENT_TYPES: Tuple[str, ...] = (
    EventType.STORAGE_IN,
    EventType.STORAGE_OUT,
    EventType.PROCESSING_IN,
    EventType.PROCESSING_OUT,
    EventType.INSPECTION,
    EventType.SAMPLING,
)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CustodyEvent:
    """A single custody event in the chain of custody.

    Attributes:
        event_id: Unique identifier for this event.
        batch_id: Identifier of the commodity batch.
        event_type: Type of custody event.
        timestamp: When the event occurred (UTC).
        sender_id: Actor who sent / released the goods.
        sender_name: Human-readable name of the sender.
        receiver_id: Actor who received / accepted the goods.
        receiver_name: Human-readable name of the receiver.
        facility_id: Facility where the event took place.
        facility_name: Human-readable name of the facility.
        facility_country: ISO 3166-1 alpha-2 country code.
        commodity: Commodity type (e.g., 'cocoa', 'palm_oil', 'soy').
        quantity_kg: Quantity in kilograms.
        quality_grade: Quality grade or classification.
        transport_mode: Transport mode (truck, rail, vessel, air, pipeline).
        transport_vehicle_id: Vehicle/vessel identifier.
        seal_numbers: Tamper-evident seal numbers.
        temperature_celsius: Temperature during transit/storage.
        humidity_pct: Humidity percentage during transit/storage.
        handler_name: Name of the person handling the goods.
        handler_id: Identifier of the handler.
        documents: List of associated document references.
        notes: Free-text notes.
        status: Event status (active, amended, superseded, voided).
        amendment_ids: IDs of amendments linked to this event.
        preceding_event_id: ID of the previous event in the chain.
        created_at: When this record was created (UTC).
        updated_at: When this record was last updated (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    event_id: str = ""
    batch_id: str = ""
    event_type: str = ""
    timestamp: Optional[datetime] = None
    sender_id: str = ""
    sender_name: str = ""
    receiver_id: str = ""
    receiver_name: str = ""
    facility_id: str = ""
    facility_name: str = ""
    facility_country: str = ""
    commodity: str = ""
    quantity_kg: float = 0.0
    quality_grade: str = ""
    transport_mode: str = ""
    transport_vehicle_id: str = ""
    seal_numbers: List[str] = field(default_factory=list)
    temperature_celsius: Optional[float] = None
    humidity_pct: Optional[float] = None
    handler_name: str = ""
    handler_id: str = ""
    documents: List[str] = field(default_factory=list)
    notes: str = ""
    status: str = EventStatus.ACTIVE
    amendment_ids: List[str] = field(default_factory=list)
    preceding_event_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to a dictionary for hashing and serialization."""
        return {
            "event_id": self.event_id,
            "batch_id": self.batch_id,
            "event_type": self.event_type,
            "timestamp": str(self.timestamp) if self.timestamp else "",
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "receiver_id": self.receiver_id,
            "receiver_name": self.receiver_name,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "facility_country": self.facility_country,
            "commodity": self.commodity,
            "quantity_kg": self.quantity_kg,
            "quality_grade": self.quality_grade,
            "transport_mode": self.transport_mode,
            "transport_vehicle_id": self.transport_vehicle_id,
            "seal_numbers": list(self.seal_numbers),
            "temperature_celsius": self.temperature_celsius,
            "humidity_pct": self.humidity_pct,
            "handler_name": self.handler_name,
            "handler_id": self.handler_id,
            "documents": list(self.documents),
            "notes": self.notes,
            "status": self.status,
            "amendment_ids": list(self.amendment_ids),
            "preceding_event_id": self.preceding_event_id,
            "created_at": str(self.created_at) if self.created_at else "",
            "updated_at": str(self.updated_at) if self.updated_at else "",
        }

@dataclass
class EventValidation:
    """Result of validating a custody event.

    Attributes:
        validation_id: Unique identifier for this validation.
        event_id: ID of the event being validated.
        batch_id: Batch ID associated with the event.
        is_valid: Whether the event passes all validations.
        errors: List of error-level issues.
        warnings: List of warning-level issues.
        info_messages: List of informational messages.
        temporal_valid: Whether temporal ordering is valid.
        actor_valid: Whether actor continuity is valid.
        location_valid: Whether location continuity is valid.
        type_sequence_valid: Whether event type sequence is valid.
        quantity_valid: Whether quantity is within acceptable range.
        validated_at: When validation was performed (UTC).
        processing_time_ms: Time taken for validation in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """

    validation_id: str = ""
    event_id: str = ""
    batch_id: str = ""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info_messages: List[str] = field(default_factory=list)
    temporal_valid: bool = True
    actor_valid: bool = True
    location_valid: bool = True
    type_sequence_valid: bool = True
    quantity_valid: bool = True
    validated_at: Optional[datetime] = None
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "validation_id": self.validation_id,
            "event_id": self.event_id,
            "batch_id": self.batch_id,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "info_messages": list(self.info_messages),
            "temporal_valid": self.temporal_valid,
            "actor_valid": self.actor_valid,
            "location_valid": self.location_valid,
            "type_sequence_valid": self.type_sequence_valid,
            "quantity_valid": self.quantity_valid,
            "validated_at": str(self.validated_at) if self.validated_at else "",
            "processing_time_ms": self.processing_time_ms,
        }

@dataclass
class CustodyGap:
    """A temporal gap detected in the custody chain.

    Attributes:
        gap_id: Unique identifier for this gap.
        batch_id: Batch ID where the gap was detected.
        preceding_event_id: ID of the event before the gap.
        following_event_id: ID of the event after the gap.
        preceding_event_type: Type of the preceding event.
        following_event_type: Type of the following event.
        preceding_timestamp: Timestamp of the preceding event.
        following_timestamp: Timestamp of the following event.
        gap_hours: Duration of the gap in hours.
        threshold_hours: Threshold that was exceeded.
        severity: Severity of the gap (warning or error).
        location_before: Facility ID before the gap.
        location_after: Facility ID after the gap.
        detected_at: When the gap was detected (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    gap_id: str = ""
    batch_id: str = ""
    preceding_event_id: str = ""
    following_event_id: str = ""
    preceding_event_type: str = ""
    following_event_type: str = ""
    preceding_timestamp: Optional[datetime] = None
    following_timestamp: Optional[datetime] = None
    gap_hours: float = 0.0
    threshold_hours: float = DEFAULT_GAP_THRESHOLD_HOURS
    severity: str = ValidationSeverity.WARNING
    location_before: str = ""
    location_after: str = ""
    detected_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert gap record to dictionary."""
        return {
            "gap_id": self.gap_id,
            "batch_id": self.batch_id,
            "preceding_event_id": self.preceding_event_id,
            "following_event_id": self.following_event_id,
            "preceding_event_type": self.preceding_event_type,
            "following_event_type": self.following_event_type,
            "preceding_timestamp": str(self.preceding_timestamp) if self.preceding_timestamp else "",
            "following_timestamp": str(self.following_timestamp) if self.following_timestamp else "",
            "gap_hours": self.gap_hours,
            "threshold_hours": self.threshold_hours,
            "severity": self.severity,
            "location_before": self.location_before,
            "location_after": self.location_after,
            "detected_at": str(self.detected_at) if self.detected_at else "",
        }

@dataclass
class EventAmendment:
    """An amendment to an existing custody event.

    The original event is preserved; amendments are recorded as a linked list
    with the reason for each change.

    Attributes:
        amendment_id: Unique identifier for this amendment.
        event_id: ID of the event being amended.
        batch_id: Batch ID associated with the event.
        amendment_number: Sequential amendment number for this event.
        field_name: Name of the field being amended.
        old_value: Previous value (serialized as string).
        new_value: New value (serialized as string).
        reason: Reason for the amendment.
        amended_by: ID of the user who made the amendment.
        amended_by_name: Name of the user who made the amendment.
        created_at: When this amendment was created (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    amendment_id: str = ""
    event_id: str = ""
    batch_id: str = ""
    amendment_number: int = 0
    field_name: str = ""
    old_value: str = ""
    new_value: str = ""
    reason: str = ""
    amended_by: str = ""
    amended_by_name: str = ""
    created_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert amendment to dictionary."""
        return {
            "amendment_id": self.amendment_id,
            "event_id": self.event_id,
            "batch_id": self.batch_id,
            "amendment_number": self.amendment_number,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "amended_by": self.amended_by,
            "amended_by_name": self.amended_by_name,
            "created_at": str(self.created_at) if self.created_at else "",
        }

@dataclass
class EventChain:
    """An ordered sequence of custody events for a batch.

    Attributes:
        chain_id: Unique identifier for this chain snapshot.
        batch_id: Batch ID this chain belongs to.
        events: Ordered list of custody events.
        total_events: Total number of events in the chain.
        first_event_timestamp: Timestamp of the earliest event.
        last_event_timestamp: Timestamp of the most recent event.
        chain_duration_hours: Total duration of the chain in hours.
        unique_actors: Number of unique actors (senders + receivers).
        unique_facilities: Number of unique facilities.
        gaps_detected: Number of custody gaps detected.
        is_complete: Whether the chain has no gaps or issues.
        created_at: When this chain snapshot was created (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    chain_id: str = ""
    batch_id: str = ""
    events: List[CustodyEvent] = field(default_factory=list)
    total_events: int = 0
    first_event_timestamp: Optional[datetime] = None
    last_event_timestamp: Optional[datetime] = None
    chain_duration_hours: float = 0.0
    unique_actors: int = 0
    unique_facilities: int = 0
    gaps_detected: int = 0
    is_complete: bool = True
    created_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "batch_id": self.batch_id,
            "total_events": self.total_events,
            "first_event_timestamp": str(self.first_event_timestamp) if self.first_event_timestamp else "",
            "last_event_timestamp": str(self.last_event_timestamp) if self.last_event_timestamp else "",
            "chain_duration_hours": self.chain_duration_hours,
            "unique_actors": self.unique_actors,
            "unique_facilities": self.unique_facilities,
            "gaps_detected": self.gaps_detected,
            "is_complete": self.is_complete,
            "created_at": str(self.created_at) if self.created_at else "",
        }

@dataclass
class BulkImportResult:
    """Result of a bulk import operation.

    Attributes:
        import_id: Unique identifier for this import.
        source_format: Format of the source data.
        total_records: Total number of records in the source.
        imported_count: Number of successfully imported records.
        skipped_count: Number of skipped records (duplicates, etc.).
        failed_count: Number of failed records.
        validation_errors: List of validation error messages per record.
        imported_event_ids: IDs of successfully imported events.
        processing_time_ms: Total processing time in milliseconds.
        started_at: When the import started (UTC).
        completed_at: When the import completed (UTC).
        provenance_hash: SHA-256 provenance hash.
    """

    import_id: str = ""
    source_format: str = ""
    total_records: int = 0
    imported_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    imported_event_ids: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert import result to dictionary."""
        return {
            "import_id": self.import_id,
            "source_format": self.source_format,
            "total_records": self.total_records,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "validation_errors": list(self.validation_errors),
            "imported_event_ids": list(self.imported_event_ids),
            "processing_time_ms": self.processing_time_ms,
            "started_at": str(self.started_at) if self.started_at else "",
            "completed_at": str(self.completed_at) if self.completed_at else "",
        }

# ---------------------------------------------------------------------------
# CustodyEventTracker
# ---------------------------------------------------------------------------

class CustodyEventTracker:
    """Production-grade custody event tracking engine for EUDR compliance.

    Records every change of custody for commodity batches with comprehensive
    validation of temporal ordering, actor continuity, and location continuity.
    Supports amendment tracking (original event preserved), bulk import from
    EDI/XML/CSV, and custody gap detection.

    All operations are deterministic with zero LLM/ML involvement. Every result
    object carries a SHA-256 provenance hash for complete audit trail.

    Example::

        tracker = CustodyEventTracker()
        result = tracker.record_event({
            "batch_id": "BATCH-001",
            "event_type": "transfer",
            "timestamp": "2025-06-15T10:00:00Z",
            "sender_id": "COOP-A",
            "receiver_id": "TRADER-B",
            "facility_id": "FAC-001",
            "commodity": "cocoa",
            "quantity_kg": 5000.0,
        })
        assert result.provenance_hash != ""
        chain = tracker.get_event_chain("BATCH-001")
        assert chain.total_events == 1

    Attributes:
        events: In-memory store of all custody events keyed by event_id.
        batch_events: Mapping from batch_id to ordered list of event_ids.
        amendments: In-memory store of all amendments keyed by amendment_id.
        gap_threshold_hours: Default gap detection threshold in hours.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the CustodyEventTracker.

        Args:
            config: Optional configuration object. Supports attributes:
                - gap_threshold_hours (float): Default gap threshold.
                - max_bulk_import_size (int): Max bulk import batch size.
        """
        self.gap_threshold_hours: float = DEFAULT_GAP_THRESHOLD_HOURS
        self.max_bulk_import_size: int = MAX_BULK_IMPORT_SIZE

        if config is not None:
            self.gap_threshold_hours = float(
                getattr(config, "gap_threshold_hours", DEFAULT_GAP_THRESHOLD_HOURS)
            )
            self.max_bulk_import_size = int(
                getattr(config, "max_bulk_import_size", MAX_BULK_IMPORT_SIZE)
            )

        # In-memory event store: event_id -> CustodyEvent
        self._events: Dict[str, CustodyEvent] = {}

        # Batch index: batch_id -> [event_id, ...] in chronological order
        self._batch_events: Dict[str, List[str]] = {}

        # Amendment store: amendment_id -> EventAmendment
        self._amendments: Dict[str, EventAmendment] = {}

        # Event amendment index: event_id -> [amendment_id, ...]
        self._event_amendments: Dict[str, List[str]] = {}

        logger.info(
            "CustodyEventTracker initialized: gap_threshold=%.1fh, "
            "max_bulk_size=%d",
            self.gap_threshold_hours,
            self.max_bulk_import_size,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Return total number of recorded events."""
        return len(self._events)

    @property
    def batch_count(self) -> int:
        """Return total number of distinct batches."""
        return len(self._batch_events)

    # ------------------------------------------------------------------
    # Public API: record_event
    # ------------------------------------------------------------------

    def record_event(self, event_data: Dict[str, Any]) -> CustodyEvent:
        """Record a new custody event with full validation.

        Creates an immutable custody event record. Before recording, validates
        temporal ordering, actor continuity, location continuity, and event
        type sequencing against the existing chain for the batch.

        Args:
            event_data: Dictionary containing event attributes. Required keys:
                - batch_id (str): Batch identifier.
                - event_type (str): One of the 10 EventType values.
                - timestamp (str|datetime): When the event occurred.
                - facility_id (str): Facility where the event occurred.
                - commodity (str): Commodity type.
                - quantity_kg (float): Quantity in kilograms.
                Optional keys: sender_id, receiver_id, sender_name,
                receiver_name, facility_name, facility_country,
                quality_grade, transport_mode, transport_vehicle_id,
                seal_numbers, temperature_celsius, humidity_pct,
                handler_name, handler_id, documents, notes.

        Returns:
            The recorded CustodyEvent with provenance hash.

        Raises:
            ValueError: If required fields are missing or invalid.
            ValueError: If validation fails (temporal, actor, location).
        """
        start_time = time.monotonic()

        # Validate required fields
        self._validate_required_fields(event_data)

        # Parse and build event
        event = self._build_event_from_data(event_data)

        # Run chain validations if batch has prior events
        validation = self._validate_new_event(event)
        if not validation.is_valid:
            error_msg = "; ".join(validation.errors)
            raise ValueError(
                f"Event validation failed for batch '{event.batch_id}': {error_msg}"
            )

        # Link to preceding event
        if event.batch_id in self._batch_events and self._batch_events[event.batch_id]:
            event.preceding_event_id = self._batch_events[event.batch_id][-1]

        # Compute provenance hash
        event.provenance_hash = _compute_hash(event.to_dict())

        # Store the event
        self._events[event.event_id] = event

        # Update batch index
        if event.batch_id not in self._batch_events:
            self._batch_events[event.batch_id] = []
        self._batch_events[event.batch_id].append(event.event_id)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Recorded custody event %s for batch %s [type=%s, qty=%.2fkg] in %.2fms",
            event.event_id,
            event.batch_id,
            event.event_type,
            event.quantity_kg,
            elapsed_ms,
        )

        return event

    # ------------------------------------------------------------------
    # Public API: validate_temporal_order
    # ------------------------------------------------------------------

    def validate_temporal_order(
        self, batch_id: str, new_event: Dict[str, Any]
    ) -> EventValidation:
        """Validate that a new event is chronologically ordered within its batch.

        Events for a batch must have strictly non-decreasing timestamps.
        A new event's timestamp must be >= the last event's timestamp.

        Args:
            batch_id: Batch identifier.
            new_event: Dictionary with at least 'timestamp' key.

        Returns:
            EventValidation with temporal_valid flag and any errors.

        Raises:
            ValueError: If timestamp is missing or unparseable.
        """
        start_time = time.monotonic()
        validation = EventValidation(
            validation_id=_generate_id(),
            batch_id=batch_id,
            validated_at=utcnow(),
        )

        new_ts = self._parse_timestamp(new_event.get("timestamp"))
        if new_ts is None:
            validation.is_valid = False
            validation.temporal_valid = False
            validation.errors.append("Missing or invalid timestamp in new event.")
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # If batch has no events, temporal order is trivially valid
        if batch_id not in self._batch_events or not self._batch_events[batch_id]:
            validation.temporal_valid = True
            validation.info_messages.append(
                "First event for batch; temporal order check not applicable."
            )
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # Get last event timestamp
        last_event_id = self._batch_events[batch_id][-1]
        last_event = self._events[last_event_id]
        last_ts = last_event.timestamp

        if last_ts is not None and new_ts < last_ts:
            validation.is_valid = False
            validation.temporal_valid = False
            validation.errors.append(
                f"Temporal order violation: new event timestamp "
                f"({new_ts.isoformat()}) is before last event timestamp "
                f"({last_ts.isoformat()}) for batch '{batch_id}'."
            )
        else:
            validation.temporal_valid = True
            validation.info_messages.append("Temporal order valid.")

        validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
        validation.provenance_hash = _compute_hash(validation.to_dict())
        return validation

    # ------------------------------------------------------------------
    # Public API: validate_actor_continuity
    # ------------------------------------------------------------------

    def validate_actor_continuity(
        self, batch_id: str, new_event: Dict[str, Any]
    ) -> EventValidation:
        """Validate actor continuity: receiver of previous = sender of new.

        For movement events, the sender of the new event must match the
        receiver of the preceding event, ensuring no custody break.

        Args:
            batch_id: Batch identifier.
            new_event: Dictionary with 'sender_id' key.

        Returns:
            EventValidation with actor_valid flag and any errors.
        """
        start_time = time.monotonic()
        validation = EventValidation(
            validation_id=_generate_id(),
            batch_id=batch_id,
            validated_at=utcnow(),
        )

        new_sender = str(new_event.get("sender_id", "")).strip()
        new_event_type = str(new_event.get("event_type", "")).strip()

        # Only validate actor continuity for movement events
        if new_event_type not in MOVEMENT_EVENT_TYPES:
            validation.actor_valid = True
            validation.info_messages.append(
                f"Actor continuity not required for event type '{new_event_type}'."
            )
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # If batch has no events, actor continuity is trivially valid
        if batch_id not in self._batch_events or not self._batch_events[batch_id]:
            validation.actor_valid = True
            validation.info_messages.append(
                "First event for batch; actor continuity check not applicable."
            )
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # Get previous event's receiver
        last_event_id = self._batch_events[batch_id][-1]
        last_event = self._events[last_event_id]
        last_receiver = last_event.receiver_id.strip()

        if not last_receiver:
            # If previous event has no receiver (e.g., inspection), skip check
            validation.actor_valid = True
            validation.warnings.append(
                "Previous event has no receiver_id; actor continuity "
                "check skipped."
            )
        elif not new_sender:
            validation.is_valid = False
            validation.actor_valid = False
            validation.errors.append(
                f"Missing sender_id on new event for batch '{batch_id}'."
            )
        elif new_sender != last_receiver:
            validation.is_valid = False
            validation.actor_valid = False
            validation.errors.append(
                f"Actor continuity break: previous receiver '{last_receiver}' "
                f"does not match new sender '{new_sender}' for batch '{batch_id}'."
            )
        else:
            validation.actor_valid = True
            validation.info_messages.append("Actor continuity valid.")

        validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
        validation.provenance_hash = _compute_hash(validation.to_dict())
        return validation

    # ------------------------------------------------------------------
    # Public API: validate_location_continuity
    # ------------------------------------------------------------------

    def validate_location_continuity(
        self, batch_id: str, new_event: Dict[str, Any]
    ) -> EventValidation:
        """Validate location continuity for facility-internal events.

        For facility events (storage, processing, inspection, sampling), the
        facility_id must match the facility of the preceding event. Goods
        cannot teleport between facilities without a transfer/export event.

        Args:
            batch_id: Batch identifier.
            new_event: Dictionary with 'facility_id' and 'event_type' keys.

        Returns:
            EventValidation with location_valid flag and any errors.
        """
        start_time = time.monotonic()
        validation = EventValidation(
            validation_id=_generate_id(),
            batch_id=batch_id,
            validated_at=utcnow(),
        )

        new_facility = str(new_event.get("facility_id", "")).strip()
        new_event_type = str(new_event.get("event_type", "")).strip()

        # Movement events naturally change location
        if new_event_type in MOVEMENT_EVENT_TYPES:
            validation.location_valid = True
            validation.info_messages.append(
                f"Location continuity not required for movement event "
                f"type '{new_event_type}'."
            )
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # If batch has no events, location continuity is trivially valid
        if batch_id not in self._batch_events or not self._batch_events[batch_id]:
            validation.location_valid = True
            validation.info_messages.append(
                "First event for batch; location continuity check not applicable."
            )
            validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
            validation.provenance_hash = _compute_hash(validation.to_dict())
            return validation

        # Find the most recent event with a facility_id
        last_facility = self._get_last_facility(batch_id)

        if not last_facility:
            validation.location_valid = True
            validation.warnings.append(
                "No previous facility found for batch; location check skipped."
            )
        elif not new_facility:
            validation.is_valid = False
            validation.location_valid = False
            validation.errors.append(
                f"Missing facility_id on facility event for batch '{batch_id}'."
            )
        elif new_facility != last_facility:
            validation.is_valid = False
            validation.location_valid = False
            validation.errors.append(
                f"Location continuity break: goods at facility "
                f"'{last_facility}' cannot appear at '{new_facility}' "
                f"without a transfer/export event for batch '{batch_id}'."
            )
        else:
            validation.location_valid = True
            validation.info_messages.append("Location continuity valid.")

        validation.processing_time_ms = (time.monotonic() - start_time) * 1000.0
        validation.provenance_hash = _compute_hash(validation.to_dict())
        return validation

    # ------------------------------------------------------------------
    # Public API: detect_custody_gaps
    # ------------------------------------------------------------------

    def detect_custody_gaps(
        self, batch_id: str, threshold_hours: Optional[float] = None
    ) -> List[CustodyGap]:
        """Detect temporal gaps in the custody chain exceeding a threshold.

        Scans all consecutive event pairs for a batch and flags any pair
        whose temporal distance exceeds the threshold. A gap indicates
        potential untracked custody and is a risk indicator per EUDR
        Article 10.

        Args:
            batch_id: Batch identifier.
            threshold_hours: Maximum allowed gap in hours. If None, uses
                engine default (72 hours).

        Returns:
            List of CustodyGap records, sorted by gap_hours descending.

        Raises:
            ValueError: If batch_id is not found.
        """
        start_time = time.monotonic()
        threshold = threshold_hours if threshold_hours is not None else self.gap_threshold_hours

        if batch_id not in self._batch_events:
            raise ValueError(f"Batch '{batch_id}' not found.")

        event_ids = self._batch_events[batch_id]
        if len(event_ids) < 2:
            logger.info(
                "Batch '%s' has %d event(s); no gaps to detect.",
                batch_id,
                len(event_ids),
            )
            return []

        gaps: List[CustodyGap] = []
        now = utcnow()

        for i in range(len(event_ids) - 1):
            preceding = self._events[event_ids[i]]
            following = self._events[event_ids[i + 1]]

            if preceding.timestamp is None or following.timestamp is None:
                continue

            delta = following.timestamp - preceding.timestamp
            gap_hours = delta.total_seconds() / 3600.0

            if gap_hours > threshold:
                severity = self._classify_gap_severity(gap_hours, threshold)
                gap = CustodyGap(
                    gap_id=_generate_id(),
                    batch_id=batch_id,
                    preceding_event_id=preceding.event_id,
                    following_event_id=following.event_id,
                    preceding_event_type=preceding.event_type,
                    following_event_type=following.event_type,
                    preceding_timestamp=preceding.timestamp,
                    following_timestamp=following.timestamp,
                    gap_hours=round(gap_hours, 2),
                    threshold_hours=threshold,
                    severity=severity,
                    location_before=preceding.facility_id,
                    location_after=following.facility_id,
                    detected_at=now,
                )
                gap.provenance_hash = _compute_hash(gap.to_dict())
                gaps.append(gap)

        # Sort by gap_hours descending (largest gaps first)
        gaps.sort(key=lambda g: g.gap_hours, reverse=True)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Detected %d custody gap(s) for batch '%s' (threshold=%.1fh) "
            "in %.2fms",
            len(gaps),
            batch_id,
            threshold,
            elapsed_ms,
        )

        return gaps

    # ------------------------------------------------------------------
    # Public API: get_event_chain
    # ------------------------------------------------------------------

    def get_event_chain(self, batch_id: str) -> EventChain:
        """Return the ordered event sequence for a batch.

        Constructs a complete chain snapshot with summary statistics
        including duration, unique actors/facilities, and gap count.

        Args:
            batch_id: Batch identifier.

        Returns:
            EventChain with events in chronological order.

        Raises:
            ValueError: If batch_id is not found.
        """
        start_time = time.monotonic()

        if batch_id not in self._batch_events:
            raise ValueError(f"Batch '{batch_id}' not found.")

        event_ids = self._batch_events[batch_id]
        events = [self._events[eid] for eid in event_ids]

        chain = EventChain(
            chain_id=_generate_id(),
            batch_id=batch_id,
            events=events,
            total_events=len(events),
            created_at=utcnow(),
        )

        if events:
            chain.first_event_timestamp = events[0].timestamp
            chain.last_event_timestamp = events[-1].timestamp

            if chain.first_event_timestamp and chain.last_event_timestamp:
                delta = chain.last_event_timestamp - chain.first_event_timestamp
                chain.chain_duration_hours = round(
                    delta.total_seconds() / 3600.0, 2
                )

            # Count unique actors
            actors: set = set()
            facilities: set = set()
            for evt in events:
                if evt.sender_id:
                    actors.add(evt.sender_id)
                if evt.receiver_id:
                    actors.add(evt.receiver_id)
                if evt.facility_id:
                    facilities.add(evt.facility_id)

            chain.unique_actors = len(actors)
            chain.unique_facilities = len(facilities)

            # Count gaps
            gaps = self.detect_custody_gaps(batch_id)
            chain.gaps_detected = len(gaps)
            chain.is_complete = len(gaps) == 0

        chain.provenance_hash = _compute_hash(chain.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Built event chain for batch '%s': %d events, %.1fh duration, "
            "%d gaps in %.2fms",
            batch_id,
            chain.total_events,
            chain.chain_duration_hours,
            chain.gaps_detected,
            elapsed_ms,
        )

        return chain

    # ------------------------------------------------------------------
    # Public API: amend_event
    # ------------------------------------------------------------------

    def amend_event(
        self,
        event_id: str,
        amendments: Dict[str, Any],
        reason: str,
        amended_by: str = "",
        amended_by_name: str = "",
    ) -> List[EventAmendment]:
        """Create amendments for an existing event (original preserved).

        Each field amendment is recorded as a separate EventAmendment record.
        The original event values are preserved in the amendment's old_value.
        The event's status is updated to AMENDED and the amendment IDs are
        linked.

        Args:
            event_id: ID of the event to amend.
            amendments: Dictionary of field_name -> new_value pairs.
            reason: Reason for the amendments.
            amended_by: ID of the user making the amendment.
            amended_by_name: Name of the user making the amendment.

        Returns:
            List of EventAmendment records created.

        Raises:
            ValueError: If event_id not found.
            ValueError: If attempting to amend an immutable field.
            ValueError: If amendments dict is empty.
            ValueError: If maximum amendments exceeded.
        """
        start_time = time.monotonic()

        if event_id not in self._events:
            raise ValueError(f"Event '{event_id}' not found.")

        if not amendments:
            raise ValueError("Amendments dictionary must not be empty.")

        if not reason or not reason.strip():
            raise ValueError("Amendment reason must be provided.")

        event = self._events[event_id]

        # Check max amendments
        existing_count = len(self._event_amendments.get(event_id, []))
        if existing_count + len(amendments) > MAX_AMENDMENTS_PER_EVENT:
            raise ValueError(
                f"Maximum amendments ({MAX_AMENDMENTS_PER_EVENT}) would be "
                f"exceeded for event '{event_id}'."
            )

        # Validate no immutable fields
        for field_name in amendments:
            if field_name in IMMUTABLE_FIELDS:
                raise ValueError(
                    f"Field '{field_name}' is immutable and cannot be amended."
                )
            if field_name not in AMENDABLE_FIELDS:
                raise ValueError(
                    f"Field '{field_name}' is not an amendable field. "
                    f"Allowed: {AMENDABLE_FIELDS}"
                )

        now = utcnow()
        created_amendments: List[EventAmendment] = []
        amendment_number_start = existing_count + 1

        for idx, (field_name, new_value) in enumerate(amendments.items()):
            old_value = getattr(event, field_name, None)

            amendment = EventAmendment(
                amendment_id=_generate_id(),
                event_id=event_id,
                batch_id=event.batch_id,
                amendment_number=amendment_number_start + idx,
                field_name=field_name,
                old_value=str(old_value) if old_value is not None else "",
                new_value=str(new_value),
                reason=reason.strip(),
                amended_by=amended_by,
                amended_by_name=amended_by_name,
                created_at=now,
            )
            amendment.provenance_hash = _compute_hash(amendment.to_dict())

            # Store amendment
            self._amendments[amendment.amendment_id] = amendment

            # Link to event
            if event_id not in self._event_amendments:
                self._event_amendments[event_id] = []
            self._event_amendments[event_id].append(amendment.amendment_id)
            event.amendment_ids.append(amendment.amendment_id)

            # Apply the change to the event object
            self._apply_amendment_to_event(event, field_name, new_value)

            created_amendments.append(amendment)

        # Update event metadata
        event.status = EventStatus.AMENDED
        event.updated_at = now
        event.provenance_hash = _compute_hash(event.to_dict())

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Created %d amendment(s) for event '%s' (reason: %s) in %.2fms",
            len(created_amendments),
            event_id,
            reason[:50],
            elapsed_ms,
        )

        return created_amendments

    # ------------------------------------------------------------------
    # Public API: bulk_import
    # ------------------------------------------------------------------

    def bulk_import(
        self,
        events: List[Dict[str, Any]],
        source_format: str = "json",
    ) -> BulkImportResult:
        """Import events from external format (EDI/XML/CSV/JSON).

        Processes each event record, applying format-specific parsing,
        validation, and recording. Failed records are captured in the
        result with their error details.

        Args:
            events: List of event dictionaries in the source format.
            source_format: Source format ('edi', 'xml', 'csv', 'json').

        Returns:
            BulkImportResult with import statistics and errors.

        Raises:
            ValueError: If source_format is not supported.
            ValueError: If events exceeds maximum bulk import size.
        """
        start_time = time.monotonic()
        now = utcnow()

        fmt = source_format.lower().strip()
        if fmt not in SUPPORTED_IMPORT_FORMATS:
            raise ValueError(
                f"Unsupported import format '{source_format}'. "
                f"Supported: {SUPPORTED_IMPORT_FORMATS}"
            )

        if len(events) > self.max_bulk_import_size:
            raise ValueError(
                f"Bulk import size ({len(events)}) exceeds maximum "
                f"({self.max_bulk_import_size})."
            )

        result = BulkImportResult(
            import_id=_generate_id(),
            source_format=fmt,
            total_records=len(events),
            started_at=now,
        )

        for idx, raw_event in enumerate(events):
            try:
                # Normalize from source format
                normalized = self._normalize_import_record(raw_event, fmt)

                # Check for duplicate event (same batch, timestamp, sender)
                if self._is_duplicate_event(normalized):
                    result.skipped_count += 1
                    result.validation_errors.append({
                        "record_index": idx,
                        "error": "Duplicate event detected; skipped.",
                        "batch_id": normalized.get("batch_id", ""),
                    })
                    continue

                # Record the event
                recorded = self.record_event(normalized)
                result.imported_event_ids.append(recorded.event_id)
                result.imported_count += 1

            except (ValueError, KeyError, TypeError) as exc:
                result.failed_count += 1
                result.validation_errors.append({
                    "record_index": idx,
                    "error": str(exc),
                    "batch_id": raw_event.get("batch_id", ""),
                })
                logger.warning(
                    "Bulk import record %d failed: %s", idx, str(exc)
                )

        result.completed_at = utcnow()
        result.processing_time_ms = (time.monotonic() - start_time) * 1000.0
        result.provenance_hash = _compute_hash(result.to_dict())

        logger.info(
            "Bulk import completed: %d imported, %d skipped, %d failed "
            "out of %d total in %.2fms",
            result.imported_count,
            result.skipped_count,
            result.failed_count,
            result.total_records,
            result.processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: batch_record
    # ------------------------------------------------------------------

    def batch_record(
        self, events: List[Dict[str, Any]]
    ) -> List[CustodyEvent]:
        """Record multiple events in batch with collective validation.

        Events are validated and recorded in order. If any event fails
        validation, it is skipped and a warning is logged. Successfully
        recorded events are returned.

        Args:
            events: List of event data dictionaries.

        Returns:
            List of successfully recorded CustodyEvent objects.
        """
        start_time = time.monotonic()
        recorded: List[CustodyEvent] = []

        for idx, event_data in enumerate(events):
            try:
                event = self.record_event(event_data)
                recorded.append(event)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Batch record: event %d failed: %s", idx, str(exc)
                )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Batch record completed: %d/%d events recorded in %.2fms",
            len(recorded),
            len(events),
            elapsed_ms,
        )

        return recorded

    # ------------------------------------------------------------------
    # Public API: get_event
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> CustodyEvent:
        """Retrieve a single custody event by ID.

        Args:
            event_id: Event identifier.

        Returns:
            The CustodyEvent.

        Raises:
            ValueError: If event_id not found.
        """
        if event_id not in self._events:
            raise ValueError(f"Event '{event_id}' not found.")
        return self._events[event_id]

    # ------------------------------------------------------------------
    # Public API: get_amendments
    # ------------------------------------------------------------------

    def get_amendments(self, event_id: str) -> List[EventAmendment]:
        """Retrieve all amendments for a given event.

        Args:
            event_id: Event identifier.

        Returns:
            List of EventAmendment records, ordered by amendment_number.
        """
        if event_id not in self._events:
            raise ValueError(f"Event '{event_id}' not found.")

        amendment_ids = self._event_amendments.get(event_id, [])
        amendments = [self._amendments[aid] for aid in amendment_ids]
        amendments.sort(key=lambda a: a.amendment_number)
        return amendments

    # ------------------------------------------------------------------
    # Public API: get_batch_event_ids
    # ------------------------------------------------------------------

    def get_batch_event_ids(self, batch_id: str) -> List[str]:
        """Retrieve ordered event IDs for a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            List of event IDs in chronological order.

        Raises:
            ValueError: If batch_id not found.
        """
        if batch_id not in self._batch_events:
            raise ValueError(f"Batch '{batch_id}' not found.")
        return list(self._batch_events[batch_id])

    # ------------------------------------------------------------------
    # Internal: validation helpers
    # ------------------------------------------------------------------

    def _validate_required_fields(self, event_data: Dict[str, Any]) -> None:
        """Validate that all required fields are present and non-empty.

        Args:
            event_data: Event data dictionary.

        Raises:
            ValueError: If a required field is missing or empty.
        """
        required = ("batch_id", "event_type", "timestamp", "facility_id",
                     "commodity", "quantity_kg")
        for field_name in required:
            val = event_data.get(field_name)
            if val is None:
                raise ValueError(f"Required field '{field_name}' is missing.")
            if isinstance(val, str) and not val.strip():
                raise ValueError(f"Required field '{field_name}' is empty.")

        # Validate event type
        event_type = str(event_data["event_type"]).strip().lower()
        valid_types = {e.value for e in EventType}
        if event_type not in valid_types:
            raise ValueError(
                f"Invalid event_type '{event_data['event_type']}'. "
                f"Valid types: {sorted(valid_types)}"
            )

        # Validate quantity is positive
        qty = float(event_data["quantity_kg"])
        if qty <= 0:
            raise ValueError(
                f"quantity_kg must be positive, got {qty}."
            )

    def _build_event_from_data(self, event_data: Dict[str, Any]) -> CustodyEvent:
        """Build a CustodyEvent from a data dictionary.

        Args:
            event_data: Event data dictionary.

        Returns:
            Populated CustodyEvent (without provenance hash).
        """
        now = utcnow()
        ts = self._parse_timestamp(event_data.get("timestamp"))

        event = CustodyEvent(
            event_id=_generate_id(),
            batch_id=str(event_data["batch_id"]).strip(),
            event_type=str(event_data["event_type"]).strip().lower(),
            timestamp=ts,
            sender_id=str(event_data.get("sender_id", "")).strip(),
            sender_name=str(event_data.get("sender_name", "")).strip(),
            receiver_id=str(event_data.get("receiver_id", "")).strip(),
            receiver_name=str(event_data.get("receiver_name", "")).strip(),
            facility_id=str(event_data["facility_id"]).strip(),
            facility_name=str(event_data.get("facility_name", "")).strip(),
            facility_country=str(event_data.get("facility_country", "")).strip().upper(),
            commodity=str(event_data["commodity"]).strip().lower(),
            quantity_kg=float(event_data["quantity_kg"]),
            quality_grade=str(event_data.get("quality_grade", "")).strip(),
            transport_mode=str(event_data.get("transport_mode", "")).strip().lower(),
            transport_vehicle_id=str(event_data.get("transport_vehicle_id", "")).strip(),
            seal_numbers=list(event_data.get("seal_numbers", [])),
            temperature_celsius=self._parse_optional_float(
                event_data.get("temperature_celsius")
            ),
            humidity_pct=self._parse_optional_float(
                event_data.get("humidity_pct")
            ),
            handler_name=str(event_data.get("handler_name", "")).strip(),
            handler_id=str(event_data.get("handler_id", "")).strip(),
            documents=list(event_data.get("documents", [])),
            notes=str(event_data.get("notes", "")).strip(),
            status=EventStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

        return event

    def _validate_new_event(self, event: CustodyEvent) -> EventValidation:
        """Run all validations on a new event before recording.

        Combines temporal, actor, location, and type-sequence validations.

        Args:
            event: The event to validate.

        Returns:
            Combined EventValidation result.
        """
        validation = EventValidation(
            validation_id=_generate_id(),
            event_id=event.event_id,
            batch_id=event.batch_id,
            validated_at=utcnow(),
        )

        event_dict = event.to_dict()

        # Temporal order validation
        temporal = self.validate_temporal_order(event.batch_id, event_dict)
        validation.temporal_valid = temporal.temporal_valid
        if not temporal.temporal_valid:
            validation.is_valid = False
            validation.errors.extend(temporal.errors)
        validation.warnings.extend(temporal.warnings)
        validation.info_messages.extend(temporal.info_messages)

        # Actor continuity validation
        actor = self.validate_actor_continuity(event.batch_id, event_dict)
        validation.actor_valid = actor.actor_valid
        if not actor.actor_valid:
            validation.is_valid = False
            validation.errors.extend(actor.errors)
        validation.warnings.extend(actor.warnings)
        validation.info_messages.extend(actor.info_messages)

        # Location continuity validation
        location = self.validate_location_continuity(event.batch_id, event_dict)
        validation.location_valid = location.location_valid
        if not location.location_valid:
            validation.is_valid = False
            validation.errors.extend(location.errors)
        validation.warnings.extend(location.warnings)
        validation.info_messages.extend(location.info_messages)

        # Event type sequence validation
        type_valid = self._validate_event_type_sequence(event)
        validation.type_sequence_valid = type_valid
        if not type_valid:
            validation.is_valid = False
            validation.errors.append(
                f"Invalid event type sequence: '{event.event_type}' cannot "
                f"follow the preceding event for batch '{event.batch_id}'."
            )

        # Quantity validation
        qty_valid = self._validate_quantity(event)
        validation.quantity_valid = qty_valid
        if not qty_valid:
            validation.warnings.append(
                f"Quantity {event.quantity_kg}kg may be unusual for "
                f"commodity '{event.commodity}'."
            )

        validation.provenance_hash = _compute_hash(validation.to_dict())
        return validation

    def _validate_event_type_sequence(self, event: CustodyEvent) -> bool:
        """Validate event type follows valid predecessor rules.

        Args:
            event: The event to validate.

        Returns:
            True if the event type sequence is valid.
        """
        batch_id = event.batch_id
        if batch_id not in self._batch_events or not self._batch_events[batch_id]:
            # First event - any type is valid
            return True

        last_event_id = self._batch_events[batch_id][-1]
        last_event = self._events[last_event_id]
        last_type = last_event.event_type

        valid_predecessors = VALID_PREDECESSORS.get(event.event_type, ())

        # If no predecessor rules defined, allow any sequence
        if not valid_predecessors:
            return True

        return last_type in valid_predecessors

    def _validate_quantity(self, event: CustodyEvent) -> bool:
        """Validate quantity is within reasonable bounds.

        Args:
            event: The event to validate.

        Returns:
            True if quantity is within acceptable range.
        """
        # Basic bounds check: 0.001 kg to 1,000,000 kg (1,000 tonnes)
        return 0.001 <= event.quantity_kg <= 1_000_000.0

    # ------------------------------------------------------------------
    # Internal: gap severity classification
    # ------------------------------------------------------------------

    def _classify_gap_severity(
        self, gap_hours: float, threshold: float
    ) -> str:
        """Classify gap severity based on how much it exceeds threshold.

        Args:
            gap_hours: Duration of the gap in hours.
            threshold: Threshold in hours.

        Returns:
            Severity level string.
        """
        ratio = gap_hours / threshold if threshold > 0 else 1.0
        if ratio >= 5.0:
            return ValidationSeverity.ERROR
        elif ratio >= 2.0:
            return ValidationSeverity.WARNING
        else:
            return ValidationSeverity.WARNING

    # ------------------------------------------------------------------
    # Internal: facility lookup
    # ------------------------------------------------------------------

    def _get_last_facility(self, batch_id: str) -> str:
        """Find the most recent facility_id for a batch.

        Args:
            batch_id: Batch identifier.

        Returns:
            Facility ID string, or empty string if none found.
        """
        if batch_id not in self._batch_events:
            return ""

        # Walk backwards to find last event with a facility_id
        for eid in reversed(self._batch_events[batch_id]):
            evt = self._events[eid]
            if evt.facility_id:
                return evt.facility_id
        return ""

    # ------------------------------------------------------------------
    # Internal: import normalization
    # ------------------------------------------------------------------

    def _normalize_import_record(
        self, raw: Dict[str, Any], fmt: str
    ) -> Dict[str, Any]:
        """Normalize an import record from external format to internal schema.

        Applies format-specific field name mappings and type conversions.

        Args:
            raw: Raw record dictionary.
            fmt: Source format identifier.

        Returns:
            Normalized event data dictionary.
        """
        if fmt == "edi":
            return self._normalize_edi_record(raw)
        elif fmt == "xml":
            return self._normalize_xml_record(raw)
        elif fmt == "csv":
            return self._normalize_csv_record(raw)
        else:
            # JSON format is assumed to match internal schema
            return dict(raw)

    def _normalize_edi_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an EDI-formatted record.

        EDI records use abbreviated field names. This method maps them
        to the internal schema.

        Args:
            raw: Raw EDI record.

        Returns:
            Normalized dictionary.
        """
        edi_mapping = {
            "BID": "batch_id",
            "EVT": "event_type",
            "TS": "timestamp",
            "SND": "sender_id",
            "RCV": "receiver_id",
            "FAC": "facility_id",
            "CMD": "commodity",
            "QTY": "quantity_kg",
            "GRD": "quality_grade",
            "TRM": "transport_mode",
            "VEH": "transport_vehicle_id",
            "SEAL": "seal_numbers",
            "TEMP": "temperature_celsius",
            "HUM": "humidity_pct",
            "HDL": "handler_name",
            "HDLID": "handler_id",
            "DOC": "documents",
            "NOTE": "notes",
            "SNAME": "sender_name",
            "RNAME": "receiver_name",
            "FNAME": "facility_name",
            "FCTY": "facility_country",
        }
        normalized: Dict[str, Any] = {}
        for edi_key, internal_key in edi_mapping.items():
            if edi_key in raw:
                normalized[internal_key] = raw[edi_key]

        # Pass through any non-mapped keys that match internal schema
        for key, value in raw.items():
            if key not in edi_mapping and key not in normalized:
                normalized[key] = value

        return normalized

    def _normalize_xml_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an XML-parsed record.

        XML records use PascalCase field names. This method maps them
        to the internal snake_case schema.

        Args:
            raw: Raw XML-parsed record.

        Returns:
            Normalized dictionary.
        """
        xml_mapping = {
            "BatchId": "batch_id",
            "EventType": "event_type",
            "Timestamp": "timestamp",
            "SenderId": "sender_id",
            "SenderName": "sender_name",
            "ReceiverId": "receiver_id",
            "ReceiverName": "receiver_name",
            "FacilityId": "facility_id",
            "FacilityName": "facility_name",
            "FacilityCountry": "facility_country",
            "Commodity": "commodity",
            "QuantityKg": "quantity_kg",
            "QualityGrade": "quality_grade",
            "TransportMode": "transport_mode",
            "TransportVehicleId": "transport_vehicle_id",
            "SealNumbers": "seal_numbers",
            "TemperatureCelsius": "temperature_celsius",
            "HumidityPct": "humidity_pct",
            "HandlerName": "handler_name",
            "HandlerId": "handler_id",
            "Documents": "documents",
            "Notes": "notes",
        }
        normalized: Dict[str, Any] = {}
        for xml_key, internal_key in xml_mapping.items():
            if xml_key in raw:
                normalized[internal_key] = raw[xml_key]

        # Pass through non-mapped keys
        for key, value in raw.items():
            if key not in xml_mapping and key not in normalized:
                normalized[key] = value

        return normalized

    def _normalize_csv_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a CSV-parsed record.

        CSV records may use UPPER_CASE or mixed naming. This method
        normalizes to internal snake_case schema.

        Args:
            raw: Raw CSV-parsed record.

        Returns:
            Normalized dictionary.
        """
        csv_mapping = {
            "BATCH_ID": "batch_id",
            "EVENT_TYPE": "event_type",
            "TIMESTAMP": "timestamp",
            "SENDER_ID": "sender_id",
            "SENDER_NAME": "sender_name",
            "RECEIVER_ID": "receiver_id",
            "RECEIVER_NAME": "receiver_name",
            "FACILITY_ID": "facility_id",
            "FACILITY_NAME": "facility_name",
            "FACILITY_COUNTRY": "facility_country",
            "COMMODITY": "commodity",
            "QUANTITY_KG": "quantity_kg",
            "QUALITY_GRADE": "quality_grade",
            "TRANSPORT_MODE": "transport_mode",
            "TRANSPORT_VEHICLE_ID": "transport_vehicle_id",
            "SEAL_NUMBERS": "seal_numbers",
            "TEMPERATURE_CELSIUS": "temperature_celsius",
            "HUMIDITY_PCT": "humidity_pct",
            "HANDLER_NAME": "handler_name",
            "HANDLER_ID": "handler_id",
            "DOCUMENTS": "documents",
            "NOTES": "notes",
        }
        normalized: Dict[str, Any] = {}
        for csv_key, internal_key in csv_mapping.items():
            if csv_key in raw:
                val = raw[csv_key]
                # CSV values may need type conversion
                if internal_key == "quantity_kg":
                    val = float(val) if val else 0.0
                elif internal_key == "temperature_celsius":
                    val = float(val) if val else None
                elif internal_key == "humidity_pct":
                    val = float(val) if val else None
                elif internal_key in ("seal_numbers", "documents"):
                    if isinstance(val, str):
                        val = [s.strip() for s in val.split(";") if s.strip()]
                normalized[internal_key] = val

        # Also check lowercase keys as pass-through
        for key, value in raw.items():
            lower_key = key.lower()
            if key not in csv_mapping and lower_key not in normalized:
                normalized[lower_key] = value

        return normalized

    # ------------------------------------------------------------------
    # Internal: duplicate detection
    # ------------------------------------------------------------------

    def _is_duplicate_event(self, event_data: Dict[str, Any]) -> bool:
        """Check if an event is a duplicate of an existing event.

        Duplicate is defined as same batch_id + timestamp + sender_id +
        event_type + facility_id.

        Args:
            event_data: Event data dictionary.

        Returns:
            True if duplicate is found.
        """
        batch_id = str(event_data.get("batch_id", "")).strip()
        if batch_id not in self._batch_events:
            return False

        new_ts = self._parse_timestamp(event_data.get("timestamp"))
        new_sender = str(event_data.get("sender_id", "")).strip()
        new_type = str(event_data.get("event_type", "")).strip().lower()
        new_facility = str(event_data.get("facility_id", "")).strip()

        for eid in self._batch_events[batch_id]:
            existing = self._events[eid]
            if (
                existing.timestamp == new_ts
                and existing.sender_id == new_sender
                and existing.event_type == new_type
                and existing.facility_id == new_facility
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Internal: amendment application
    # ------------------------------------------------------------------

    def _apply_amendment_to_event(
        self, event: CustodyEvent, field_name: str, new_value: Any
    ) -> None:
        """Apply an amendment value to an event field.

        Args:
            event: The event to modify.
            field_name: Field name to update.
            new_value: New value to set.
        """
        if field_name == "quantity_kg":
            setattr(event, field_name, float(new_value))
        elif field_name in ("temperature_celsius", "humidity_pct"):
            setattr(event, field_name, float(new_value) if new_value is not None else None)
        elif field_name in ("seal_numbers", "documents"):
            if isinstance(new_value, list):
                setattr(event, field_name, list(new_value))
            elif isinstance(new_value, str):
                setattr(event, field_name, [s.strip() for s in new_value.split(";") if s.strip()])
            else:
                setattr(event, field_name, [str(new_value)])
        else:
            setattr(event, field_name, str(new_value))

    # ------------------------------------------------------------------
    # Internal: timestamp parsing
    # ------------------------------------------------------------------

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        """Parse a timestamp from various formats.

        Supports datetime objects, ISO format strings, and Unix timestamps.

        Args:
            value: Timestamp value to parse.

        Returns:
            Parsed datetime (UTC), or None if unparseable.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            # Try ISO format (including Python's str(datetime) output)
            for fmt in (
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue

            logger.warning("Unable to parse timestamp: %s", value)
            return None

        return None

    def _parse_optional_float(self, value: Any) -> Optional[float]:
        """Parse an optional float value.

        Args:
            value: Value to parse.

        Returns:
            Float value, or None.
        """
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
