"""
GL-012 STEAMQUAL - Audit Trail

Comprehensive audit trail for all calculations and explanations in the
Steam Quality Controller, ensuring complete traceability per playbook requirement.

This module provides:
1. Input/output SHA-256 hashes for all calculations
2. Model version tracking
3. Configuration version tracking
4. Timestamp tracking for all operations
5. Tamper-evident chain hashing
6. Export capabilities for compliance reporting

All explanations are traceable to data and assumptions per playbook requirement.

Reference:
    - IEC 62443 Security for Industrial Automation
    - NIST SP 800-53 Security and Privacy Controls
    - GreenLang Audit Framework Standards

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AuditEventType(Enum):
    """Types of audit events."""

    # Calculation events
    CALCULATION_STARTED = "calculation_started"
    CALCULATION_COMPLETED = "calculation_completed"
    CALCULATION_FAILED = "calculation_failed"

    # Explanation events
    EXPLANATION_GENERATED = "explanation_generated"
    EXPLANATION_ACCESSED = "explanation_accessed"
    EXPLANATION_EXPORTED = "explanation_exported"

    # Root cause events
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    ROOT_CAUSE_ACCESSED = "root_cause_accessed"

    # Recommendation events
    RECOMMENDATION_GENERATED = "recommendation_generated"
    RECOMMENDATION_ACCESSED = "recommendation_accessed"
    RECOMMENDATION_ACCEPTED = "recommendation_accepted"
    RECOMMENDATION_REJECTED = "recommendation_rejected"

    # Model/config events
    MODEL_LOADED = "model_loaded"
    MODEL_UPDATED = "model_updated"
    CONFIG_LOADED = "config_loaded"
    CONFIG_UPDATED = "config_updated"

    # Data events
    DATA_INGESTED = "data_ingested"
    DATA_VALIDATED = "data_validated"
    DATA_REJECTED = "data_rejected"

    # System events
    AUDIT_STARTED = "audit_started"
    AUDIT_EXPORTED = "audit_exported"
    AUDIT_VERIFIED = "audit_verified"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HashAlgorithm(Enum):
    """Hash algorithms supported."""

    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelVersion:
    """Model version information."""

    model_id: str
    model_name: str
    version: str
    model_hash: str
    trained_timestamp: Optional[datetime] = None
    deployed_timestamp: Optional[datetime] = None

    # Metadata
    feature_count: int = 0
    training_samples: int = 0
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Provenance
    training_data_hash: str = ""
    hyperparameters_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_hash": self.model_hash,
            "trained_timestamp": self.trained_timestamp.isoformat() if self.trained_timestamp else None,
            "deployed_timestamp": self.deployed_timestamp.isoformat() if self.deployed_timestamp else None,
            "feature_count": self.feature_count,
            "training_samples": self.training_samples,
            "validation_metrics": self.validation_metrics,
            "training_data_hash": self.training_data_hash,
            "hyperparameters_hash": self.hyperparameters_hash,
        }


@dataclass
class ConfigVersion:
    """Configuration version information."""

    config_id: str
    config_name: str
    version: str
    config_hash: str
    created_timestamp: datetime
    activated_timestamp: Optional[datetime] = None

    # Content
    parameters: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Change tracking
    previous_version: Optional[str] = None
    change_description: str = ""
    changed_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "version": self.version,
            "config_hash": self.config_hash,
            "created_timestamp": self.created_timestamp.isoformat(),
            "activated_timestamp": self.activated_timestamp.isoformat() if self.activated_timestamp else None,
            "parameters": self.parameters,
            "thresholds": self.thresholds,
            "previous_version": self.previous_version,
            "change_description": self.change_description,
            "changed_by": self.changed_by,
        }


@dataclass
class AuditEntry:
    """Single audit trail entry."""

    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity

    # Context
    agent_id: str
    header_id: str = ""
    session_id: str = ""

    # Operation details
    operation_name: str = ""
    operation_id: str = ""

    # Input/Output hashes
    input_hash: str = ""
    output_hash: str = ""
    computation_hash: str = ""

    # Version tracking
    model_version: str = ""
    config_version: str = ""
    agent_version: str = ""

    # Execution details
    execution_time_ms: float = 0.0
    success: bool = True
    error_message: str = ""

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Chain hash for tamper detection
    previous_entry_hash: str = ""
    entry_hash: str = ""

    def __post_init__(self):
        """Compute entry hash if not provided."""
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this entry for chain integrity."""
        data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "operation_id": self.operation_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "previous_entry_hash": self.previous_entry_hash,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "header_id": self.header_id,
            "session_id": self.session_id,
            "operation_name": self.operation_name,
            "operation_id": self.operation_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "computation_hash": self.computation_hash,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "agent_version": self.agent_version,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "details": self.details,
            "tags": self.tags,
            "previous_entry_hash": self.previous_entry_hash,
            "entry_hash": self.entry_hash,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_provenance_hash(
    data: Any,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> str:
    """
    Compute provenance hash for any data.

    Args:
        data: Data to hash (dict, list, or serializable object)
        algorithm: Hash algorithm to use

    Returns:
        Hexadecimal hash string
    """
    if hasattr(data, 'model_dump'):
        # Pydantic model
        data = data.model_dump()
    elif hasattr(data, '__dict__'):
        # Dataclass or object
        data = data.__dict__

    json_str = json.dumps(data, sort_keys=True, default=str)

    if algorithm == HashAlgorithm.SHA256:
        return hashlib.sha256(json_str.encode()).hexdigest()
    elif algorithm == HashAlgorithm.SHA384:
        return hashlib.sha384(json_str.encode()).hexdigest()
    elif algorithm == HashAlgorithm.SHA512:
        return hashlib.sha512(json_str.encode()).hexdigest()
    else:
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# EXPLANATION AUDIT TRAIL
# =============================================================================

class ExplanationAuditTrail:
    """
    Audit trail for all calculations and explanations.

    Provides comprehensive logging of all operations with:
    - SHA-256 input/output hashes
    - Model and config version tracking
    - Tamper-evident chain hashing
    - Query capabilities for compliance

    Example:
        >>> audit = ExplanationAuditTrail(agent_id="GL-012")
        >>> audit.log_calculation(
        ...     operation="dryness_estimation",
        ...     inputs=sensor_data,
        ...     outputs=quality_estimate
        ... )
        >>> export = audit.export_for_compliance(
        ...     start_time=start,
        ...     end_time=end
        ... )

    Attributes:
        agent_id: Agent identifier
        max_entries: Maximum entries to retain in memory
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-012",
        max_entries: int = 100000,
        on_audit_event: Optional[Callable[[AuditEntry], None]] = None,
    ) -> None:
        """
        Initialize ExplanationAuditTrail.

        Args:
            agent_id: Agent identifier
            max_entries: Maximum entries to retain
            on_audit_event: Optional callback for audit events
        """
        self.agent_id = agent_id
        self.max_entries = max_entries
        self._on_audit_event = on_audit_event

        # State
        self._entries: List[AuditEntry] = []
        self._chain_hash = "0" * 64  # Genesis hash
        self._session_id = str(uuid.uuid4())

        # Version tracking
        self._model_versions: Dict[str, ModelVersion] = {}
        self._config_versions: Dict[str, ConfigVersion] = {}
        self._current_model_version: str = "1.0.0"
        self._current_config_version: str = "1.0.0"

        # Statistics
        self._stats = {
            "total_entries": 0,
            "calculations": 0,
            "explanations": 0,
            "root_cause_analyses": 0,
            "recommendations": 0,
            "errors": 0,
            "average_execution_time_ms": 0.0,
        }

        # Log audit start
        self._log_system_event(AuditEventType.AUDIT_STARTED)

        logger.info(f"ExplanationAuditTrail initialized: {agent_id}")

    def log_calculation(
        self,
        operation: str,
        inputs: Any,
        outputs: Any,
        header_id: str = "",
        execution_time_ms: float = 0.0,
        success: bool = True,
        error_message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log a calculation event.

        Args:
            operation: Name of the calculation operation
            inputs: Input data (will be hashed)
            outputs: Output data (will be hashed)
            header_id: Steam header identifier
            execution_time_ms: Execution time in milliseconds
            success: Whether calculation succeeded
            error_message: Error message if failed
            details: Additional details

        Returns:
            AuditEntry for the logged event
        """
        event_type = (
            AuditEventType.CALCULATION_COMPLETED if success
            else AuditEventType.CALCULATION_FAILED
        )
        severity = AuditSeverity.INFO if success else AuditSeverity.ERROR

        # Compute hashes
        input_hash = compute_provenance_hash(inputs)
        output_hash = compute_provenance_hash(outputs) if success else ""
        computation_hash = compute_provenance_hash({
            "input_hash": input_hash,
            "output_hash": output_hash,
            "operation": operation,
            "agent_id": self.agent_id,
        })

        entry = self._create_entry(
            event_type=event_type,
            severity=severity,
            header_id=header_id,
            operation_name=operation,
            operation_id=str(uuid.uuid4())[:8],
            input_hash=input_hash,
            output_hash=output_hash,
            computation_hash=computation_hash,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            details=details or {},
            tags=["calculation", operation],
        )

        self._update_stats("calculations", execution_time_ms)

        return entry

    def log_explanation(
        self,
        explanation_type: str,
        explanation_id: str,
        inputs: Any,
        outputs: Any,
        header_id: str = "",
        execution_time_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an explanation generation event.

        Args:
            explanation_type: Type of explanation generated
            explanation_id: Unique explanation identifier
            inputs: Input data used
            outputs: Explanation output
            header_id: Steam header identifier
            execution_time_ms: Execution time
            details: Additional details

        Returns:
            AuditEntry for the logged event
        """
        input_hash = compute_provenance_hash(inputs)
        output_hash = compute_provenance_hash(outputs)

        entry = self._create_entry(
            event_type=AuditEventType.EXPLANATION_GENERATED,
            severity=AuditSeverity.INFO,
            header_id=header_id,
            operation_name=f"explain_{explanation_type}",
            operation_id=explanation_id,
            input_hash=input_hash,
            output_hash=output_hash,
            computation_hash=compute_provenance_hash({
                "input_hash": input_hash,
                "output_hash": output_hash,
                "explanation_type": explanation_type,
            }),
            execution_time_ms=execution_time_ms,
            success=True,
            details={
                **(details or {}),
                "explanation_type": explanation_type,
                "explanation_id": explanation_id,
            },
            tags=["explanation", explanation_type],
        )

        self._update_stats("explanations", execution_time_ms)

        return entry

    def log_explanation_access(
        self,
        explanation_id: str,
        accessor: str = "",
        access_method: str = "api",
        header_id: str = "",
    ) -> AuditEntry:
        """Log explanation access event."""
        return self._create_entry(
            event_type=AuditEventType.EXPLANATION_ACCESSED,
            severity=AuditSeverity.INFO,
            header_id=header_id,
            operation_name="access_explanation",
            operation_id=explanation_id,
            details={
                "explanation_id": explanation_id,
                "accessor": accessor,
                "access_method": access_method,
            },
            tags=["access", "explanation"],
        )

    def log_root_cause_analysis(
        self,
        analysis_id: str,
        event_id: str,
        event_type: str,
        inputs: Any,
        outputs: Any,
        header_id: str = "",
        execution_time_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log root cause analysis event.

        Args:
            analysis_id: Analysis identifier
            event_id: Quality event identifier
            event_type: Type of quality event
            inputs: Input event data
            outputs: Analysis results
            header_id: Steam header identifier
            execution_time_ms: Execution time
            details: Additional details

        Returns:
            AuditEntry for the logged event
        """
        input_hash = compute_provenance_hash(inputs)
        output_hash = compute_provenance_hash(outputs)

        entry = self._create_entry(
            event_type=AuditEventType.ROOT_CAUSE_ANALYSIS,
            severity=AuditSeverity.INFO,
            header_id=header_id,
            operation_name=f"root_cause_{event_type}",
            operation_id=analysis_id,
            input_hash=input_hash,
            output_hash=output_hash,
            computation_hash=compute_provenance_hash({
                "input_hash": input_hash,
                "output_hash": output_hash,
                "analysis_id": analysis_id,
            }),
            execution_time_ms=execution_time_ms,
            success=True,
            details={
                **(details or {}),
                "analysis_id": analysis_id,
                "event_id": event_id,
                "event_type": event_type,
            },
            tags=["root_cause", event_type],
        )

        self._update_stats("root_cause_analyses", execution_time_ms)

        return entry

    def log_recommendation(
        self,
        recommendation_id: str,
        action_type: str,
        inputs: Any,
        outputs: Any,
        header_id: str = "",
        execution_time_ms: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log recommendation generation event.

        Args:
            recommendation_id: Recommendation identifier
            action_type: Type of recommended action
            inputs: Input state data
            outputs: Recommendation output
            header_id: Steam header identifier
            execution_time_ms: Execution time
            details: Additional details

        Returns:
            AuditEntry for the logged event
        """
        input_hash = compute_provenance_hash(inputs)
        output_hash = compute_provenance_hash(outputs)

        entry = self._create_entry(
            event_type=AuditEventType.RECOMMENDATION_GENERATED,
            severity=AuditSeverity.INFO,
            header_id=header_id,
            operation_name=f"recommend_{action_type}",
            operation_id=recommendation_id,
            input_hash=input_hash,
            output_hash=output_hash,
            computation_hash=compute_provenance_hash({
                "input_hash": input_hash,
                "output_hash": output_hash,
                "recommendation_id": recommendation_id,
            }),
            execution_time_ms=execution_time_ms,
            success=True,
            details={
                **(details or {}),
                "recommendation_id": recommendation_id,
                "action_type": action_type,
            },
            tags=["recommendation", action_type],
        )

        self._update_stats("recommendations", execution_time_ms)

        return entry

    def log_recommendation_response(
        self,
        recommendation_id: str,
        accepted: bool,
        responder: str = "",
        reason: str = "",
        header_id: str = "",
    ) -> AuditEntry:
        """Log response to a recommendation."""
        event_type = (
            AuditEventType.RECOMMENDATION_ACCEPTED if accepted
            else AuditEventType.RECOMMENDATION_REJECTED
        )

        return self._create_entry(
            event_type=event_type,
            severity=AuditSeverity.INFO,
            header_id=header_id,
            operation_name="recommendation_response",
            operation_id=recommendation_id,
            details={
                "recommendation_id": recommendation_id,
                "accepted": accepted,
                "responder": responder,
                "reason": reason,
            },
            tags=["recommendation_response", "accepted" if accepted else "rejected"],
        )

    def log_model_update(
        self,
        model_version: ModelVersion,
    ) -> AuditEntry:
        """Log model version update."""
        self._model_versions[model_version.model_id] = model_version
        self._current_model_version = model_version.version

        return self._create_entry(
            event_type=AuditEventType.MODEL_UPDATED,
            severity=AuditSeverity.INFO,
            operation_name="update_model",
            operation_id=model_version.model_id,
            details=model_version.to_dict(),
            tags=["model", "update"],
        )

    def log_config_update(
        self,
        config_version: ConfigVersion,
    ) -> AuditEntry:
        """Log configuration version update."""
        self._config_versions[config_version.config_id] = config_version
        self._current_config_version = config_version.version

        return self._create_entry(
            event_type=AuditEventType.CONFIG_UPDATED,
            severity=AuditSeverity.INFO,
            operation_name="update_config",
            operation_id=config_version.config_id,
            details=config_version.to_dict(),
            tags=["config", "update"],
        )

    def log_data_ingestion(
        self,
        data_source: str,
        record_count: int,
        data_hash: str,
        header_id: str = "",
        validated: bool = True,
        rejection_reason: str = "",
    ) -> AuditEntry:
        """Log data ingestion event."""
        if validated:
            event_type = AuditEventType.DATA_VALIDATED
            severity = AuditSeverity.INFO
        else:
            event_type = AuditEventType.DATA_REJECTED
            severity = AuditSeverity.WARNING

        return self._create_entry(
            event_type=event_type,
            severity=severity,
            header_id=header_id,
            operation_name="ingest_data",
            input_hash=data_hash,
            success=validated,
            error_message=rejection_reason if not validated else "",
            details={
                "data_source": data_source,
                "record_count": record_count,
                "validated": validated,
            },
            tags=["data", "ingestion"],
        )

    def _create_entry(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        header_id: str = "",
        operation_name: str = "",
        operation_id: str = "",
        input_hash: str = "",
        output_hash: str = "",
        computation_hash: str = "",
        execution_time_ms: float = 0.0,
        success: bool = True,
        error_message: str = "",
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> AuditEntry:
        """Create and store an audit entry."""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            agent_id=self.agent_id,
            header_id=header_id,
            session_id=self._session_id,
            operation_name=operation_name,
            operation_id=operation_id or str(uuid.uuid4())[:8],
            input_hash=input_hash,
            output_hash=output_hash,
            computation_hash=computation_hash,
            model_version=self._current_model_version,
            config_version=self._current_config_version,
            agent_version=self.VERSION,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error_message,
            details=details or {},
            tags=tags or [],
            previous_entry_hash=self._chain_hash,
        )

        # Update chain hash
        self._chain_hash = entry.entry_hash

        # Store entry
        self._entries.append(entry)

        # Enforce max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]

        # Update stats
        self._stats["total_entries"] += 1
        if not success:
            self._stats["errors"] += 1

        # Callback
        if self._on_audit_event:
            self._on_audit_event(entry)

        logger.debug(f"Audit entry: {event_type.value} - {operation_name}")

        return entry

    def _log_system_event(self, event_type: AuditEventType) -> None:
        """Log system-level event."""
        self._create_entry(
            event_type=event_type,
            severity=AuditSeverity.INFO,
            operation_name="system",
            tags=["system"],
        )

    def _update_stats(self, category: str, execution_time_ms: float) -> None:
        """Update statistics."""
        self._stats[category] = self._stats.get(category, 0) + 1

        # Update average execution time
        total = self._stats["total_entries"]
        old_avg = self._stats["average_execution_time_ms"]
        self._stats["average_execution_time_ms"] = (
            (old_avg * (total - 1) + execution_time_ms) / total
        )

    def get_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        header_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Query audit entries with filters.

        Args:
            event_type: Filter by event type
            header_id: Filter by header ID
            operation_name: Filter by operation name
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            success_only: Only return successful operations
            limit: Maximum entries to return

        Returns:
            List of matching AuditEntry objects
        """
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if header_id:
            entries = [e for e in entries if e.header_id == header_id]
        if operation_name:
            entries = [e for e in entries if e.operation_name == operation_name]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        if success_only:
            entries = [e for e in entries if e.success]

        # Sort by timestamp descending
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        return entries[:limit]

    def get_entry_by_id(self, entry_id: str) -> Optional[AuditEntry]:
        """Get specific entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def get_entry_by_operation(self, operation_id: str) -> List[AuditEntry]:
        """Get all entries for a specific operation."""
        return [e for e in self._entries if e.operation_id == operation_id]

    def verify_chain_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of the audit chain.

        Returns:
            Tuple of (is_valid, list of any error messages)
        """
        errors = []

        if not self._entries:
            return True, []

        # Verify each entry's hash chain
        expected_hash = "0" * 64  # Genesis

        for i, entry in enumerate(self._entries):
            # Check chain link
            if entry.previous_entry_hash != expected_hash:
                errors.append(
                    f"Entry {i} ({entry.entry_id}): Chain hash mismatch. "
                    f"Expected {expected_hash[:16]}..., got {entry.previous_entry_hash[:16]}..."
                )

            # Verify entry hash
            computed_hash = entry._compute_hash()
            if entry.entry_hash != computed_hash:
                errors.append(
                    f"Entry {i} ({entry.entry_id}): Entry hash tampering detected. "
                    f"Expected {computed_hash[:16]}..., got {entry.entry_hash[:16]}..."
                )

            expected_hash = entry.entry_hash

        # Verify final chain hash
        if expected_hash != self._chain_hash:
            errors.append(
                f"Final chain hash mismatch. "
                f"Expected {expected_hash[:16]}..., got {self._chain_hash[:16]}..."
            )

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Audit chain integrity verified successfully")
        else:
            logger.error(f"Audit chain integrity check failed: {len(errors)} errors")

        return is_valid, errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            **self._stats,
            "entries_count": len(self._entries),
            "session_id": self._session_id,
            "chain_hash": self._chain_hash[:16] + "...",
            "current_model_version": self._current_model_version,
            "current_config_version": self._current_config_version,
        }

    def export_for_compliance(
        self,
        start_time: datetime,
        end_time: datetime,
        header_id: Optional[str] = None,
        include_chain_verification: bool = True,
    ) -> Dict[str, Any]:
        """
        Export audit data for regulatory compliance.

        Args:
            start_time: Start of period
            end_time: End of period
            header_id: Optional filter by header
            include_chain_verification: Include chain integrity check

        Returns:
            Compliance export dictionary
        """
        entries = self.get_entries(
            header_id=header_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        export = {
            "agent_id": self.agent_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "header_id": header_id or "all",
            "entry_count": len(entries),
            "entries": [e.to_dict() for e in reversed(entries)],  # Chronological
            "statistics": self.get_statistics(),
            "model_versions": {
                k: v.to_dict() for k, v in self._model_versions.items()
            },
            "config_versions": {
                k: v.to_dict() for k, v in self._config_versions.items()
            },
        }

        if include_chain_verification:
            is_valid, errors = self.verify_chain_integrity()
            export["chain_verified"] = is_valid
            export["chain_hash"] = self._chain_hash
            if errors:
                export["chain_errors"] = errors

        # Log export
        self._create_entry(
            event_type=AuditEventType.AUDIT_EXPORTED,
            severity=AuditSeverity.INFO,
            operation_name="export_compliance",
            details={
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "entry_count": len(entries),
            },
            tags=["export", "compliance"],
        )

        return export

    def clear(self) -> None:
        """Clear all audit entries (for testing only)."""
        self._entries.clear()
        self._chain_hash = "0" * 64
        self._stats = {
            "total_entries": 0,
            "calculations": 0,
            "explanations": 0,
            "root_cause_analyses": 0,
            "recommendations": 0,
            "errors": 0,
            "average_execution_time_ms": 0.0,
        }
        logger.warning("Audit trail cleared")


# =============================================================================
# AUDIT EXPORTER
# =============================================================================

class AuditExporter:
    """
    Export audit data in various formats.

    Supports JSON, CSV, and compliance report formats.
    """

    @staticmethod
    def to_json(
        entries: List[AuditEntry],
        indent: int = 2,
    ) -> str:
        """Export entries to JSON string."""
        return json.dumps(
            [e.to_dict() for e in entries],
            indent=indent,
        )

    @staticmethod
    def to_csv(entries: List[AuditEntry]) -> str:
        """Export entries to CSV string."""
        if not entries:
            return ""

        headers = [
            "entry_id", "timestamp", "event_type", "severity",
            "header_id", "operation_name", "operation_id",
            "input_hash", "output_hash", "execution_time_ms",
            "success", "model_version", "config_version"
        ]

        lines = [",".join(headers)]

        for entry in entries:
            row = [
                entry.entry_id,
                entry.timestamp.isoformat(),
                entry.event_type.value,
                entry.severity.value,
                entry.header_id,
                entry.operation_name,
                entry.operation_id,
                entry.input_hash[:16] + "..." if entry.input_hash else "",
                entry.output_hash[:16] + "..." if entry.output_hash else "",
                str(entry.execution_time_ms),
                str(entry.success),
                entry.model_version,
                entry.config_version,
            ]
            lines.append(",".join(row))

        return "\n".join(lines)

    @staticmethod
    def to_compliance_report(
        audit_trail: ExplanationAuditTrail,
        start_time: datetime,
        end_time: datetime,
        header_id: Optional[str] = None,
    ) -> str:
        """Generate compliance report as text."""
        export = audit_trail.export_for_compliance(
            start_time=start_time,
            end_time=end_time,
            header_id=header_id,
        )

        lines = [
            "STEAM QUALITY AUDIT COMPLIANCE REPORT",
            "=" * 60,
            "",
            f"Agent: {export['agent_id']}",
            f"Export Time: {export['export_timestamp']}",
            f"Period: {export['period_start']} to {export['period_end']}",
            f"Header: {export['header_id']}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Entries: {export['entry_count']}",
            f"Chain Verified: {export.get('chain_verified', 'N/A')}",
            f"Chain Hash: {export.get('chain_hash', 'N/A')[:32]}...",
            "",
            "STATISTICS",
            "-" * 40,
        ]

        for key, value in export['statistics'].items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "MODEL VERSIONS",
            "-" * 40,
        ])
        for model_id, model in export['model_versions'].items():
            lines.append(f"  {model_id}: v{model['version']}")

        lines.extend([
            "",
            "CONFIG VERSIONS",
            "-" * 40,
        ])
        for config_id, config in export['config_versions'].items():
            lines.append(f"  {config_id}: v{config['version']}")

        if export.get('chain_errors'):
            lines.extend([
                "",
                "CHAIN INTEGRITY ERRORS",
                "-" * 40,
            ])
            for error in export['chain_errors']:
                lines.append(f"  ERROR: {error}")

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
        ])

        return "\n".join(lines)
