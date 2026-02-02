# -*- coding: utf-8 -*-
"""
Audit Schemas for GL-010 EmissionsGuardian

This module defines all data models for the audit and lineage tracking subsystem,
supporting complete traceability from raw data to final calculations with
immutable audit trails and regulatory compliance.

Standards Compliance:
    - EPA 40 CFR Part 75 (3+ years data retention)
    - SOX (Sarbanes-Oxley) audit requirements
    - SOC 2 Type II compliance controls
    - EPA electronic reporting requirements

Zero-Hallucination Principle:
    - Complete provenance tracking via SHA-256 hash chains
    - Deterministic transformation recording
    - Immutable audit entries with tamper detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json

from pydantic import BaseModel, Field, field_validator, ConfigDict


class AuditAction(str, Enum):
    """Standard audit action types for emissions data lifecycle."""
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    ARCHIVE = "ARCHIVE"
    CALCULATION = "CALCULATION"
    VALIDATION = "VALIDATION"
    TRANSFORMATION = "TRANSFORMATION"
    AGGREGATION = "AGGREGATION"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    EXCEEDANCE_DETECTED = "EXCEEDANCE_DETECTED"
    DEVIATION_REPORTED = "DEVIATION_REPORTED"
    PERMIT_LIMIT_EVALUATED = "PERMIT_LIMIT_EVALUATED"
    CEMS_DATA_RECEIVED = "CEMS_DATA_RECEIVED"
    CALIBRATION_PERFORMED = "CALIBRATION_PERFORMED"
    DATA_SUBSTITUTION = "DATA_SUBSTITUTION"
    RATA_COMPLETED = "RATA_COMPLETED"
    TRADE_EXECUTED = "TRADE_EXECUTED"
    OFFSET_RETIRED = "OFFSET_RETIRED"
    CERTIFICATE_VALIDATED = "CERTIFICATE_VALIDATED"
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    PERMISSION_CHANGED = "PERMISSION_CHANGED"
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    ERROR_OCCURRED = "ERROR_OCCURRED"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"


class ResourceType(str, Enum):
    """Types of resources tracked in audit logs."""
    EMISSION_RECORD = "emission_record"
    CEMS_DATA = "cems_data"
    COMPLIANCE_RESULT = "compliance_result"
    PERMIT_LIMIT = "permit_limit"
    CALIBRATION = "calibration"
    RATA_TEST = "rata_test"
    FUGITIVE_DETECTION = "fugitive_detection"
    CARBON_TRADE = "carbon_trade"
    OFFSET_CERTIFICATE = "offset_certificate"
    USER_ACCOUNT = "user_account"
    CONFIGURATION = "configuration"
    REPORT = "report"
    ALERT = "alert"
    CALCULATION = "calculation"


class ActorType(str, Enum):
    """Types of actors that can perform auditable actions."""
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    SCHEDULER = "scheduler"
    API = "api"
    INTEGRATION = "integration"


class TransformationOperation(str, Enum):
    """Standard transformation operations in data lineage."""
    EXTRACT = "extract"
    VALIDATE = "validate"
    CLEAN = "clean"
    NORMALIZE = "normalize"
    CONVERT = "convert"
    CALCULATE = "calculate"
    AGGREGATE = "aggregate"
    FILTER = "filter"
    JOIN = "join"
    ENRICH = "enrich"
    SUBSTITUTE = "substitute"
    CORRECT = "correct"


class ChainVerificationStatus(str, Enum):
    """Status of hash chain verification."""
    VALID = "valid"
    INVALID = "invalid"
    BROKEN = "broken"
    INCOMPLETE = "incomplete"


def _serialize_value(value: Any) -> Any:
    """Serialize value for consistent hashing."""
    if value is None:
        return None
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in sorted(value.items())}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, Enum):
        return value.value
    return value


@dataclass
class TransformationStep:
    """Records a single transformation step in data lineage."""
    step_number: int
    operation: TransformationOperation
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    function_name: str
    calculation_trace: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        op_value = self.operation.value if isinstance(self.operation, TransformationOperation) else str(self.operation)
        hash_data = {
            "step_number": self.step_number,
            "operation": op_value,
            "inputs": _serialize_value(self.inputs),
            "outputs": _serialize_value(self.outputs),
            "function_name": self.function_name,
            "calculation_trace": self.calculation_trace,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        op_value = self.operation.value if isinstance(self.operation, TransformationOperation) else str(self.operation)
        return {
            "step_number": self.step_number,
            "operation": op_value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "function_name": self.function_name,
            "calculation_trace": self.calculation_trace,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class DataLineage:
    """Complete lineage record for a piece of data."""
    lineage_id: str
    data_id: str
    source_type: str
    source_id: str
    transformations: List[TransformationStep] = field(default_factory=list)
    final_hash: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_transformation(self, step: TransformationStep) -> None:
        self.transformations.append(step)

    def complete(self, final_data: Any) -> None:
        self.completed_at = datetime.utcnow()
        self.final_hash = self._calculate_final_hash(final_data)

    def _calculate_final_hash(self, data: Any) -> str:
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, sort_keys=True, default=str)
        else:
            json_str = str(data)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def get_transformation_chain_hash(self) -> str:
        chain_data = {
            "lineage_id": self.lineage_id,
            "data_id": self.data_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "transformation_hashes": [t.provenance_hash for t in self.transformations],
            "created_at": self.created_at.isoformat()
        }
        json_str = json.dumps(chain_data, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "data_id": self.data_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "transformations": [t.to_dict() for t in self.transformations],
            "final_hash": self.final_hash,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }

    def to_dot(self) -> str:
        """Generate DOT graph representation of data lineage."""
        lines = ["digraph DataLineage {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")
        source_label = f"Source: {self.source_type}
{self.source_id}"
        lines.append(f'    source [label="{source_label}", fillcolor=lightblue];')
        prev_node = "source"
        for step in self.transformations:
            node_id = f"step_{step.step_number}"
            op_val = step.operation.value if isinstance(step.operation, TransformationOperation) else step.operation
            func_short = step.function_name.split(".")[-1]
            label = f"Step {step.step_number}
{op_val}
{func_short}"
            lines.append(f'    {node_id} [label="{label}", fillcolor=lightyellow];')
            lines.append(f"    {prev_node} -> {node_id};")
            prev_node = node_id
        hash_preview = self.final_hash[:16] if self.final_hash else "pending"
        final_label = f"Final Data
{self.data_id}
hash: {hash_preview}..."
        lines.append(f'    final [label="{final_label}", fillcolor=lightgreen];')
        lines.append(f"    {prev_node} -> final;")
        lines.append("}")
        return "
".join(lines)


@dataclass
class AuditEntry:
    """Immutable audit log entry with hash chain linking."""
    entry_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor: str = ""
    actor_type: ActorType = ActorType.SYSTEM
    action: AuditAction = AuditAction.CREATE
    resource_type: ResourceType = ResourceType.EMISSION_RECORD
    resource_id: str = ""
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    parent_hash: str = ""
    session_id: str = ""
    correlation_id: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            self.provenance_hash = self.calculate_provenance_hash()

    def calculate_provenance_hash(self) -> str:
        actor_type_val = self.actor_type.value if isinstance(self.actor_type, ActorType) else str(self.actor_type)
        action_val = self.action.value if isinstance(self.action, AuditAction) else str(self.action)
        resource_type_val = self.resource_type.value if isinstance(self.resource_type, ResourceType) else str(self.resource_type)
        hash_data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "actor_type": actor_type_val,
            "action": action_val,
            "resource_type": resource_type_val,
            "resource_id": self.resource_id,
            "old_value": _serialize_value(self.old_value),
            "new_value": _serialize_value(self.new_value),
            "details": _serialize_value(self.details),
            "parent_hash": self.parent_hash
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def calculate_chain_hash(self) -> str:
        chain_data = f"{self.parent_hash}{self.provenance_hash}"
        return hashlib.sha256(chain_data.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        actor_type_val = self.actor_type.value if isinstance(self.actor_type, ActorType) else str(self.actor_type)
        action_val = self.action.value if isinstance(self.action, AuditAction) else str(self.action)
        resource_type_val = self.resource_type.value if isinstance(self.resource_type, ResourceType) else str(self.resource_type)
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "actor_type": actor_type_val,
            "action": action_val,
            "resource_type": resource_type_val,
            "resource_id": self.resource_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "details": self.details,
            "provenance_hash": self.provenance_hash,
            "parent_hash": self.parent_hash,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id
        }


class AuditQueryFilter(BaseModel):
    """Filter criteria for audit log queries."""
    model_config = ConfigDict(use_enum_values=True)
    entry_ids: Optional[List[str]] = Field(None, description="Filter by specific entry IDs")
    actors: Optional[List[str]] = Field(None, description="Filter by actor identifiers")
    actor_types: Optional[List[ActorType]] = Field(None, description="Filter by actor types")
    actions: Optional[List[AuditAction]] = Field(None, description="Filter by action types")
    resource_types: Optional[List[ResourceType]] = Field(None, description="Filter by resource types")
    resource_ids: Optional[List[str]] = Field(None, description="Filter by resource identifiers")
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")
    session_ids: Optional[List[str]] = Field(None, description="Filter by session identifiers")
    correlation_ids: Optional[List[str]] = Field(None, description="Filter by correlation identifiers")
    text_search: Optional[str] = Field(None, description="Full-text search in details")

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def parse_datetime(cls, v: Any) -> Optional[datetime]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


@dataclass
class AuditQuery:
    """Query specification for searching audit logs."""
    filters: AuditQueryFilter = field(default_factory=AuditQueryFilter)
    sort_by: str = "timestamp"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 100
    include_chain: bool = False

    def __post_init__(self) -> None:
        if self.sort_order not in ("asc", "desc"):
            raise ValueError("sort_order must be asc or desc")
        if self.page < 1:
            raise ValueError("page must be >= 1")
        if self.page_size < 1 or self.page_size > 10000:
            raise ValueError("page_size must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        filters_dict = self.filters.model_dump() if hasattr(self.filters, "model_dump") else {}
        return {
            "filters": filters_dict,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "page": self.page,
            "page_size": self.page_size,
            "include_chain": self.include_chain
        }


class AuditStatistics(BaseModel):
    """Summary statistics for audit report."""
    total_entries: int = Field(0, description="Total number of audit entries")
    entries_by_action: Dict[str, int] = Field(default_factory=dict)
    entries_by_resource: Dict[str, int] = Field(default_factory=dict)
    entries_by_actor: Dict[str, int] = Field(default_factory=dict)
    entries_by_day: Dict[str, int] = Field(default_factory=dict)
    first_entry_time: Optional[datetime] = Field(None)
    last_entry_time: Optional[datetime] = Field(None)
    chain_integrity_status: ChainVerificationStatus = Field(ChainVerificationStatus.VALID)
    broken_chain_count: int = Field(0)


@dataclass
class AuditReport:
    """Complete audit report with statistics and entries."""
    report_id: str
    report_type: str = "compliance"
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    statistics: AuditStatistics = field(default_factory=AuditStatistics)
    entries: List[AuditEntry] = field(default_factory=list)
    query_used: Optional[AuditQuery] = None
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provenance_hash and self.entries:
            self.provenance_hash = self._calculate_report_hash()

    def _calculate_report_hash(self) -> str:
        stats_dict = self.statistics.model_dump() if hasattr(self.statistics, "model_dump") else {}
        hash_data = {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "entry_hashes": [e.provenance_hash for e in self.entries],
            "statistics": stats_dict
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        stats_dict = self.statistics.model_dump() if hasattr(self.statistics, "model_dump") else {}
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "statistics": stats_dict,
            "entries": [e.to_dict() for e in self.entries],
            "query_used": self.query_used.to_dict() if self.query_used else None,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata
        }


@dataclass
class ChainVerificationResult:
    """Result of hash chain verification."""
    status: ChainVerificationStatus
    total_entries: int = 0
    valid_links: int = 0
    broken_links: List[str] = field(default_factory=list)
    first_broken_at: Optional[str] = None
    verification_time: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status == ChainVerificationStatus.VALID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "total_entries": self.total_entries,
            "valid_links": self.valid_links,
            "broken_links": self.broken_links,
            "first_broken_at": self.first_broken_at,
            "verification_time": self.verification_time.isoformat(),
            "details": self.details
        }


class RetentionPolicy(BaseModel):
    """Data retention policy configuration."""
    model_config = ConfigDict(use_enum_values=True)
    policy_id: str = Field(..., description="Unique policy identifier")
    policy_name: str = Field(..., description="Human-readable policy name")
    resource_type: ResourceType = Field(..., description="Resource type this policy applies to")
    retention_days: int = Field(1095, ge=365, le=3650)
    archive_after_days: int = Field(365, ge=30, le=1825)
    regulatory_citation: str = Field("EPA 40 CFR Part 75")
    secure_deletion: bool = Field(True)
    deletion_requires_approval: bool = Field(True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("system")


class RetentionStatus(BaseModel):
    """Current status of data retention."""
    total_records: int = Field(0)
    active_records: int = Field(0)
    archived_records: int = Field(0)
    pending_deletion: int = Field(0)
    oldest_record_date: Optional[datetime] = Field(None)
    storage_used_gb: float = Field(0.0)
    compliance_status: str = Field("compliant")
    next_archive_date: Optional[datetime] = Field(None)
    next_deletion_date: Optional[datetime] = Field(None)


@dataclass
class ArchiveResult:
    """Result of archive operation."""
    success: bool
    archived_count: int = 0
    failed_count: int = 0
    archive_location: str = ""
    archive_hash: str = ""
    archived_at: datetime = field(default_factory=datetime.utcnow)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "archived_count": self.archived_count,
            "failed_count": self.failed_count,
            "archive_location": self.archive_location,
            "archive_hash": self.archive_hash,
            "archived_at": self.archived_at.isoformat(),
            "errors": self.errors
        }


@dataclass
class RetentionResult:
    """Result of retention policy application."""
    success: bool
    policy_id: str = ""
    records_evaluated: int = 0
    records_archived: int = 0
    records_deleted: int = 0
    records_retained: int = 0
    execution_time_ms: float = 0.0
    executed_at: datetime = field(default_factory=datetime.utcnow)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "policy_id": self.policy_id,
            "records_evaluated": self.records_evaluated,
            "records_archived": self.records_archived,
            "records_deleted": self.records_deleted,
            "records_retained": self.records_retained,
            "execution_time_ms": self.execution_time_ms,
            "executed_at": self.executed_at.isoformat(),
            "errors": self.errors
        }
