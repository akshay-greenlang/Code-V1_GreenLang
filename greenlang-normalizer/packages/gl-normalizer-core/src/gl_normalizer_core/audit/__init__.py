"""
Audit event system for GL-FOUND-X-003 Unit & Reference Normalizer.

This package provides a complete audit trail system for the GreenLang
Normalizer, supporting governance-grade audit requirements with:
- Immutable audit event records
- Tamper-evident hash chaining
- Complete conversion and resolution traces
- Multiple serialization formats (JSON, Kafka, Parquet)

The audit system satisfies the following requirements from the PRD:
- FR-070: Emit one audit event per normalization request
- FR-071: Ensure audit log is append-only (immutable)
- FR-072: Include raw inputs, parsed AST, dimension signature
- FR-073: Include all conversion steps with factors
- FR-074: Include entity resolution method, confidence, vocabulary version
- FR-075: Include agent version, policy mode, overrides
- FR-076: Generate stable normalization_event_id
- FR-077: Ensure 100% audit coverage
- FR-078: Support tamper-evident hashing
- FR-079: Enable audit replay for identical outputs
- FR-080: Support audit event streaming to Kafka
- NFR-035: Tamper-evident audit hashing with hash chaining

Modules:
    schema: Pydantic models for audit event schemas
    chain: Hash chain generator for tamper-evident linking
    builder: Fluent builder for constructing audit events
    serializer: Serialization to JSON, Kafka, and Parquet formats

Quick Start:
    >>> from gl_normalizer_core.audit import (
    ...     AuditPayloadBuilder,
    ...     ConversionResult,
    ...     ResolutionResult,
    ...     VersionMetadata,
    ...     EventStatus,
    ... )
    >>>
    >>> # Build an audit event
    >>> builder = AuditPayloadBuilder()
    >>> event = (
    ...     builder
    ...     .set_request_context(request_id="req-123", source_record_id="meter-001")
    ...     .set_org_context(org_id="org-acme", policy_mode="STRICT")
    ...     .set_versions(
    ...         vocab_version="2026.01.0",
    ...         policy_version="1.0.0",
    ...         unit_registry_version="2026.01.0",
    ...         validator_version="1.0.0",
    ...         api_revision="v1"
    ...     )
    ...     .add_measurement_audit({
    ...         "field": "energy_consumption",
    ...         "raw_value": 1500,
    ...         "raw_unit": "kWh",
    ...         "expected_dimension": "energy",
    ...         "canonical_value": 5400.0,
    ...         "canonical_unit": "MJ",
    ...         "dimension": "energy",
    ...     })
    ...     .build()
    ... )
    >>>
    >>> # Serialize to various formats
    >>> from gl_normalizer_core.audit import to_json, to_kafka_message
    >>> json_str = to_json(event, pretty=True)
    >>> kafka_msg = to_kafka_message(event)

Hash Chain Verification:
    >>> from gl_normalizer_core.audit import verify_chain_integrity
    >>> is_valid, error = verify_chain_integrity(events)
    >>> if not is_valid:
    ...     print(f"Chain integrity violation: {error}")

Legacy Support:
    The following classes are maintained for backward compatibility:
    - AuditLogger, AuditTrail, AuditEntry, ProvenanceRecord
    - AuditEventType (legacy), AuditSeverity

Example Usage:
    See individual module docstrings for detailed examples.
"""

# =============================================================================
# New Audit Event System Exports (GL-FOUND-X-003)
# =============================================================================

# Schema exports
from .schema import (
    # Enums
    EventStatus,
    ConversionMethod,
    MatchMethod,
    EntityType,
    # Nested models
    ReferenceConditions,
    ConversionStep,
    PrecisionConfig,
    UnitAST,
    ResolutionCandidate,
    # Audit record models
    MeasurementAudit,
    EntityAudit,
    AuditError,
    AuditWarning,
    # Main event model
    NormalizationEvent,
    # Type aliases
    AuditEventType as AuditEventTypeNew,
    ConversionStepType,
    MeasurementAuditType,
    EntityAuditType,
)

# Chain exports
from .chain import (
    # Classes
    HashChainGenerator,
    ChainState,
    ChainIntegrityError,
    # Functions
    get_default_generator,
    generate_event_id,
    compute_payload_hash,
    compute_event_hash,
    verify_chain_integrity,
)

# Builder exports
from .builder import (
    # Classes
    AuditPayloadBuilder,
    ConversionResult,
    ResolutionResult,
    VersionMetadata,
    # Functions
    build_measurement_audit,
    build_entity_audit,
)

# Serializer exports
from .serializer import (
    # Classes
    AuditEventSerializer,
    KafkaMessage,
    ParquetColumn,
    ParquetSchema,
    # Functions
    get_default_serializer,
    to_json,
    to_kafka_message,
    to_parquet_row,
    from_json,
)

# =============================================================================
# Legacy Audit System (maintained for backward compatibility)
# =============================================================================

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    PARSE = "parse"
    CONVERSION = "conversion"
    RESOLUTION = "resolution"
    POLICY_CHECK = "policy_check"
    VOCAB_ACCESS = "vocab_access"
    VOCAB_UPDATE = "vocab_update"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProvenanceRecord(BaseModel):
    """
    A provenance record for data lineage tracking.

    Attributes:
        id: Unique record identifier
        timestamp: When the record was created
        event_type: Type of event
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        provenance_hash: Combined hash for full lineage
        parent_hash: Hash of parent record (for chaining)
        operation: Operation performed
        parameters: Operation parameters
        metadata: Additional metadata
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Record ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    event_type: AuditEventType = Field(..., description="Event type")
    input_hash: str = Field(..., description="SHA-256 hash of input")
    output_hash: str = Field(..., description="SHA-256 hash of output")
    provenance_hash: str = Field(..., description="Combined provenance hash")
    parent_hash: Optional[str] = Field(None, description="Parent record hash")
    operation: str = Field(..., description="Operation performed")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation parameters"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    user_id: Optional[str] = Field(None, description="User who performed operation")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @classmethod
    def create(
        cls,
        event_type: AuditEventType,
        operation: str,
        input_data: Any,
        output_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        parent_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "ProvenanceRecord":
        """
        Create a provenance record with calculated hashes.

        Args:
            event_type: Type of event
            operation: Operation name
            input_data: Input data to hash
            output_data: Output data to hash
            parameters: Operation parameters
            parent_hash: Hash of parent record
            metadata: Additional metadata
            user_id: User identifier
            session_id: Session identifier

        Returns:
            New ProvenanceRecord with calculated hashes
        """
        # Calculate input hash
        input_str = cls._serialize_for_hash(input_data)
        input_hash = hashlib.sha256(input_str.encode()).hexdigest()

        # Calculate output hash
        output_str = cls._serialize_for_hash(output_data)
        output_hash = hashlib.sha256(output_str.encode()).hexdigest()

        # Calculate provenance hash (combines input, output, and parent)
        provenance_str = f"{input_hash}|{output_hash}|{parent_hash or 'root'}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return cls(
            event_type=event_type,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_hash=provenance_hash,
            parent_hash=parent_hash,
            operation=operation,
            parameters=parameters or {},
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
        )

    @staticmethod
    def _serialize_for_hash(data: Any) -> str:
        """Serialize data for hashing."""
        if isinstance(data, BaseModel):
            return data.model_dump_json(exclude_none=True)
        elif isinstance(data, (dict, list)):
            return json.dumps(data, sort_keys=True, default=str)
        else:
            return str(data)


class AuditEntry(BaseModel):
    """
    A single audit log entry.

    Attributes:
        id: Unique entry identifier
        timestamp: When the entry was created
        event_type: Type of event
        severity: Severity level
        message: Log message
        details: Detailed information
        provenance_record: Associated provenance record
        correlation_id: ID for correlating related entries
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Entry ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    event_type: AuditEventType = Field(..., description="Event type")
    severity: AuditSeverity = Field(default=AuditSeverity.INFO, description="Severity")
    message: str = Field(..., description="Log message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Details")
    provenance_record: Optional[ProvenanceRecord] = Field(
        None,
        description="Associated provenance"
    )
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    user_id: Optional[str] = Field(None, description="User ID")
    source_ip: Optional[str] = Field(None, description="Source IP address")


class AuditTrail(BaseModel):
    """
    A complete audit trail for an operation.

    Attributes:
        id: Trail identifier
        entries: List of audit entries
        start_time: When the trail started
        end_time: When the trail ended
        summary: Summary of the trail
    """

    id: str = Field(default_factory=lambda: str(uuid4()), description="Trail ID")
    entries: List[AuditEntry] = Field(default_factory=list, description="Audit entries")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Start time"
    )
    end_time: Optional[datetime] = Field(None, description="End time")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Trail summary")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Correlation ID"
    )

    def add_entry(self, entry: AuditEntry) -> None:
        """Add an entry to the trail."""
        entry.correlation_id = self.correlation_id
        self.entries.append(entry)

    def close(self) -> None:
        """Close the audit trail."""
        self.end_time = datetime.now(timezone.utc)
        self.summary = {
            "entry_count": len(self.entries),
            "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000,
            "event_types": list(set(e.event_type for e in self.entries)),
            "has_errors": any(e.severity == AuditSeverity.ERROR for e in self.entries),
        }


class AuditLogger:
    """
    Logger for audit events.

    This class provides a centralized interface for logging audit events
    and creating provenance records.

    Attributes:
        storage: Audit storage backend
        enabled: Whether auditing is enabled

    Example:
        >>> audit_logger = AuditLogger()
        >>> record = audit_logger.log_conversion(
        ...     source_quantity, target_quantity, conversion_result
        ... )
    """

    def __init__(
        self,
        enabled: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Initialize AuditLogger.

        Args:
            enabled: Whether auditing is enabled
            user_id: Default user ID for entries
            session_id: Session identifier
        """
        self.enabled = enabled
        self.user_id = user_id
        self.session_id = session_id or str(uuid4())
        self._entries: List[AuditEntry] = []
        self._current_trail: Optional[AuditTrail] = None

        logger.info(
            "AuditLogger initialized",
            enabled=enabled,
            session_id=self.session_id,
        )

    def log_parse(
        self,
        input_string: str,
        result: Any,
        success: bool,
    ) -> ProvenanceRecord:
        """
        Log a parse operation.

        Args:
            input_string: Input string that was parsed
            result: Parse result
            success: Whether parsing succeeded

        Returns:
            ProvenanceRecord for the operation
        """
        record = ProvenanceRecord.create(
            event_type=AuditEventType.PARSE,
            operation="parse",
            input_data=input_string,
            output_data=result,
            parameters={"success": success},
            user_id=self.user_id,
            session_id=self.session_id,
        )

        self._log_entry(
            event_type=AuditEventType.PARSE,
            message=f"Parsed input: {input_string[:50]}...",
            details={"success": success, "input_length": len(input_string)},
            provenance_record=record,
        )

        return record

    def log_conversion(
        self,
        source: Any,
        target_unit: str,
        result: Any,
        success: bool,
    ) -> ProvenanceRecord:
        """
        Log a conversion operation.

        Args:
            source: Source quantity
            target_unit: Target unit
            result: Conversion result
            success: Whether conversion succeeded

        Returns:
            ProvenanceRecord for the operation
        """
        record = ProvenanceRecord.create(
            event_type=AuditEventType.CONVERSION,
            operation="convert",
            input_data={"source": source, "target_unit": target_unit},
            output_data=result,
            parameters={"target_unit": target_unit, "success": success},
            user_id=self.user_id,
            session_id=self.session_id,
        )

        self._log_entry(
            event_type=AuditEventType.CONVERSION,
            message=f"Converted to {target_unit}",
            details={"target_unit": target_unit, "success": success},
            provenance_record=record,
        )

        return record

    def log_resolution(
        self,
        query: str,
        vocabulary: str,
        result: Any,
        success: bool,
    ) -> ProvenanceRecord:
        """
        Log a resolution operation.

        Args:
            query: Resolution query
            vocabulary: Vocabulary used
            result: Resolution result
            success: Whether resolution succeeded

        Returns:
            ProvenanceRecord for the operation
        """
        record = ProvenanceRecord.create(
            event_type=AuditEventType.RESOLUTION,
            operation="resolve",
            input_data={"query": query, "vocabulary": vocabulary},
            output_data=result,
            parameters={"vocabulary": vocabulary, "success": success},
            user_id=self.user_id,
            session_id=self.session_id,
        )

        self._log_entry(
            event_type=AuditEventType.RESOLUTION,
            message=f"Resolved '{query}' in {vocabulary}",
            details={"vocabulary": vocabulary, "success": success},
            provenance_record=record,
        )

        return record

    def log_error(
        self,
        operation: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an error event.

        Args:
            operation: Operation that failed
            error: Exception that occurred
            context: Additional context

        Returns:
            AuditEntry for the error
        """
        entry = self._log_entry(
            event_type=AuditEventType.ERROR,
            severity=AuditSeverity.ERROR,
            message=f"Error in {operation}: {str(error)}",
            details={
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            },
        )
        return entry

    def start_trail(self) -> AuditTrail:
        """Start a new audit trail."""
        self._current_trail = AuditTrail(correlation_id=str(uuid4()))
        return self._current_trail

    def end_trail(self) -> Optional[AuditTrail]:
        """End the current audit trail."""
        if self._current_trail:
            self._current_trail.close()
            trail = self._current_trail
            self._current_trail = None
            return trail
        return None

    def get_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Get audit entries with optional filtering.

        Args:
            event_type: Filter by event type
            limit: Maximum entries to return

        Returns:
            List of matching audit entries
        """
        entries = self._entries
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        return entries[-limit:]

    def _log_entry(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        provenance_record: Optional[ProvenanceRecord] = None,
    ) -> AuditEntry:
        """Create and store an audit entry."""
        if not self.enabled:
            return AuditEntry(
                event_type=event_type,
                message=message,
                severity=severity,
            )

        entry = AuditEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            details=details or {},
            provenance_record=provenance_record,
            user_id=self.user_id,
        )

        self._entries.append(entry)

        if self._current_trail:
            self._current_trail.add_entry(entry)

        logger.debug(
            "Audit entry logged",
            event_type=event_type,
            message=message[:100],
        )

        return entry


__all__ = [
    # ==========================================================================
    # New Audit Event System (GL-FOUND-X-003) - Preferred API
    # ==========================================================================
    # Schema - Enums
    "EventStatus",
    "ConversionMethod",
    "MatchMethod",
    "EntityType",
    # Schema - Nested models
    "ReferenceConditions",
    "ConversionStep",
    "PrecisionConfig",
    "UnitAST",
    "ResolutionCandidate",
    # Schema - Audit record models
    "MeasurementAudit",
    "EntityAudit",
    "AuditError",
    "AuditWarning",
    # Schema - Main event model
    "NormalizationEvent",
    # Schema - Type aliases
    "ConversionStepType",
    "MeasurementAuditType",
    "EntityAuditType",
    # Chain - Classes
    "HashChainGenerator",
    "ChainState",
    "ChainIntegrityError",
    # Chain - Functions
    "get_default_generator",
    "generate_event_id",
    "compute_payload_hash",
    "compute_event_hash",
    "verify_chain_integrity",
    # Builder - Classes
    "AuditPayloadBuilder",
    "ConversionResult",
    "ResolutionResult",
    "VersionMetadata",
    # Builder - Functions
    "build_measurement_audit",
    "build_entity_audit",
    # Serializer - Classes
    "AuditEventSerializer",
    "KafkaMessage",
    "ParquetColumn",
    "ParquetSchema",
    # Serializer - Functions
    "get_default_serializer",
    "to_json",
    "to_kafka_message",
    "to_parquet_row",
    "from_json",
    # ==========================================================================
    # Legacy Audit System (maintained for backward compatibility)
    # ==========================================================================
    "AuditLogger",
    "AuditTrail",
    "AuditEntry",
    "ProvenanceRecord",
    "AuditEventType",
    "AuditSeverity",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
