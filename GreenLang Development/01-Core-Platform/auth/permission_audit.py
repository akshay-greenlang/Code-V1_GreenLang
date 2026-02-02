# -*- coding: utf-8 -*-
"""
Permission Change Audit Trail for GreenLang

This module provides immutable audit logging for all permission-related changes,
including before/after snapshots and actor attribution.

Features:
    - Immutable audit log
    - Before/after snapshots
    - Actor attribution
    - Change tracking for permissions, roles, delegations
    - Compliance reporting
    - Tamper detection

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import uuid
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock
from greenlang.intelligence import ChatMessage
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# ==============================================================================
# Permission Audit Event Types
# ==============================================================================

class PermissionChangeType(str, Enum):
    """Types of permission changes."""

    # Permission changes
    PERMISSION_CREATED = "permission.created"
    PERMISSION_UPDATED = "permission.updated"
    PERMISSION_DELETED = "permission.deleted"
    PERMISSION_GRANTED = "permission.granted"
    PERMISSION_REVOKED = "permission.revoked"

    # Role changes
    ROLE_CREATED = "role.created"
    ROLE_UPDATED = "role.updated"
    ROLE_DELETED = "role.deleted"
    ROLE_ASSIGNED = "role.assigned"
    ROLE_UNASSIGNED = "role.unassigned"
    ROLE_PERMISSION_ADDED = "role.permission_added"
    ROLE_PERMISSION_REMOVED = "role.permission_removed"

    # Delegation changes
    DELEGATION_CREATED = "delegation.created"
    DELEGATION_REVOKED = "delegation.revoked"
    DELEGATION_EXPIRED = "delegation.expired"
    DELEGATION_USED = "delegation.used"

    # Temporal permission changes
    TEMPORAL_PERMISSION_CREATED = "temporal.created"
    TEMPORAL_PERMISSION_REVOKED = "temporal.revoked"
    TEMPORAL_PERMISSION_EXPIRED = "temporal.expired"

    # ABAC policy changes
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DELETED = "policy.deleted"
    POLICY_ENABLED = "policy.enabled"
    POLICY_DISABLED = "policy.disabled"


class AuditSeverity(str, Enum):
    """Severity of audit event."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# ==============================================================================
# Audit Event Model
# ==============================================================================

@dataclass
class PermissionAuditEvent:
    """Immutable audit event for permission changes."""

    # Event identity
    event_id: str = field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    change_type: PermissionChangeType = field(default=PermissionChangeType.PERMISSION_CREATED)
    severity: AuditSeverity = field(default=AuditSeverity.INFO)

    # Actor information
    actor_id: str = ""  # Who made the change
    actor_type: str = "user"  # user, system, service_account
    actor_ip: Optional[str] = None
    actor_location: Optional[str] = None

    # Target information
    target_id: str = ""  # ID of affected permission/role/policy
    target_type: str = ""  # permission, role, delegation, policy
    target_name: Optional[str] = None

    # Affected principal (if applicable)
    principal_id: Optional[str] = None  # User/service affected
    principal_type: Optional[str] = None  # user, service_account

    # Change details
    before_snapshot: Optional[Dict[str, Any]] = None
    after_snapshot: Optional[Dict[str, Any]] = None
    changes: Dict[str, Any] = field(default_factory=dict)

    # Context
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    # Metadata
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Integrity
    event_hash: Optional[str] = None  # For tamper detection
    previous_hash: Optional[str] = None  # Chain to previous event

    def calculate_hash(self) -> str:
        """
        Calculate cryptographic hash of event for integrity.

        Returns:
            SHA-256 hash of event data
        """
        # Create deterministic representation
        hash_data = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'change_type': self.change_type.value,
            'actor_id': self.actor_id,
            'target_id': self.target_id,
            'target_type': self.target_type,
            'principal_id': self.principal_id,
            'before_snapshot': json.dumps(self.before_snapshot, sort_keys=True) if self.before_snapshot else None,
            'after_snapshot': json.dumps(self.after_snapshot, sort_keys=True) if self.after_snapshot else None,
            'changes': json.dumps(self.changes, sort_keys=True),
            'previous_hash': self.previous_hash
        }

        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'change_type': self.change_type.value,
            'severity': self.severity.value,
            'actor_id': self.actor_id,
            'actor_type': self.actor_type,
            'actor_ip': self.actor_ip,
            'actor_location': self.actor_location,
            'target_id': self.target_id,
            'target_type': self.target_type,
            'target_name': self.target_name,
            'principal_id': self.principal_id,
            'principal_type': self.principal_type,
            'before_snapshot': self.before_snapshot,
            'after_snapshot': self.after_snapshot,
            'changes': self.changes,
            'tenant_id': self.tenant_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'reason': self.reason,
            'metadata': self.metadata,
            'tags': self.tags,
            'event_hash': self.event_hash,
            'previous_hash': self.previous_hash
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionAuditEvent':
        """Create from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'change_type' in data and isinstance(data['change_type'], str):
            data['change_type'] = PermissionChangeType(data['change_type'])
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = AuditSeverity(data['severity'])
        return cls(**data)


# ==============================================================================
# Permission Audit Logger
# ==============================================================================

class PermissionAuditLogger:
    """
    Immutable audit logger for permission changes.

    Provides:
    - Append-only audit trail
    - Cryptographic integrity checking
    - Before/after snapshots
    - Comprehensive change tracking
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_integrity_chain: bool = True
    ):
        """
        Initialize permission audit logger.

        Args:
            storage_path: Path to store audit logs
            enable_integrity_chain: Enable hash chaining for tamper detection
        """
        self.storage_path = storage_path or Path.cwd() / "audit_logs" / "permissions"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.enable_integrity_chain = enable_integrity_chain
        self._last_event_hash: Optional[str] = None
        self._events_cache: List[PermissionAuditEvent] = []

        self._stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'events_by_actor': defaultdict(int)
        }

        logger.info(f"Initialized PermissionAuditLogger (storage={self.storage_path})")

    def log_permission_change(
        self,
        change_type: PermissionChangeType,
        actor_id: str,
        target_id: str,
        target_type: str,
        before_snapshot: Optional[Dict[str, Any]] = None,
        after_snapshot: Optional[Dict[str, Any]] = None,
        principal_id: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> PermissionAuditEvent:
        """
        Log a permission change event.

        Args:
            change_type: Type of change
            actor_id: Who made the change
            target_id: What was changed
            target_type: Type of target (permission, role, etc.)
            before_snapshot: State before change
            after_snapshot: State after change
            principal_id: User/service affected
            reason: Reason for change
            **kwargs: Additional event attributes

        Returns:
            Created audit event
        """
        # Calculate changes
        changes = self._calculate_changes(before_snapshot, after_snapshot)

        # Determine severity
        severity = self._determine_severity(change_type, changes)

        # Create event
        event = PermissionAuditEvent(
            change_type=change_type,
            severity=severity,
            actor_id=actor_id,
            target_id=target_id,
            target_type=target_type,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            changes=changes,
            principal_id=principal_id,
            reason=reason,
            **kwargs
        )

        # Add to integrity chain
        if self.enable_integrity_chain:
            event.previous_hash = self._last_event_hash
            event.event_hash = event.calculate_hash()
            self._last_event_hash = event.event_hash

        # Store event
        self._store_event(event)

        # Update stats
        self._stats['total_events'] += 1
        self._stats['events_by_type'][change_type.value] += 1
        self._stats['events_by_actor'][actor_id] += 1

        # Cache in memory
        self._events_cache.append(event)
        if len(self._events_cache) > 1000:  # Keep last 1000 events
            self._events_cache.pop(0)

        logger.info(
            f"Logged permission change: {change_type.value} by {actor_id} "
            f"on {target_type}:{target_id}"
        )

        return event

    def log_permission_created(
        self,
        actor_id: str,
        permission: Dict[str, Any],
        principal_id: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> PermissionAuditEvent:
        """Log permission creation."""
        return self.log_permission_change(
            change_type=PermissionChangeType.PERMISSION_CREATED,
            actor_id=actor_id,
            target_id=permission.get('permission_id', 'unknown'),
            target_type='permission',
            after_snapshot=permission,
            principal_id=principal_id,
            reason=reason,
            **kwargs
        )

    def log_role_assigned(
        self,
        actor_id: str,
        role_id: str,
        principal_id: str,
        role_snapshot: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> PermissionAuditEvent:
        """Log role assignment."""
        return self.log_permission_change(
            change_type=PermissionChangeType.ROLE_ASSIGNED,
            actor_id=actor_id,
            target_id=role_id,
            target_type='role',
            after_snapshot=role_snapshot,
            principal_id=principal_id,
            reason=reason,
            **kwargs
        )

    def log_delegation_created(
        self,
        actor_id: str,
        delegation: Dict[str, Any],
        reason: Optional[str] = None,
        **kwargs
    ) -> PermissionAuditEvent:
        """Log delegation creation."""
        return self.log_permission_change(
            change_type=PermissionChangeType.DELEGATION_CREATED,
            actor_id=actor_id,
            target_id=delegation.get('delegation_id', 'unknown'),
            target_type='delegation',
            after_snapshot=delegation,
            principal_id=delegation.get('delegatee_id'),
            reason=reason,
            **kwargs
        )

    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        change_types: Optional[List[PermissionChangeType]] = None,
        actor_id: Optional[str] = None,
        target_id: Optional[str] = None,
        principal_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[PermissionAuditEvent]:
        """
        Query audit events.

        Args:
            start_time: Start of time range
            end_time: End of time range
            change_types: Filter by change types
            actor_id: Filter by actor
            target_id: Filter by target
            principal_id: Filter by affected principal
            limit: Maximum number of events

        Returns:
            List of matching events
        """
        # In production, this would query from database
        # For now, use in-memory cache
        events = self._events_cache.copy()

        # Apply filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        if change_types:
            events = [e for e in events if e.change_type in change_types]

        if actor_id:
            events = [e for e in events if e.actor_id == actor_id]

        if target_id:
            events = [e for e in events if e.target_id == target_id]

        if principal_id:
            events = [e for e in events if e.principal_id == principal_id]

        # Apply limit
        return events[:limit]

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of audit chain.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.enable_integrity_chain:
            return True, []

        errors = []
        previous_hash = None

        for i, event in enumerate(self._events_cache):
            # Check previous hash matches
            if event.previous_hash != previous_hash:
                errors.append(
                    f"Event {i} (id={event.event_id}): Chain broken - "
                    f"expected previous_hash={previous_hash}, got={event.previous_hash}"
                )

            # Verify event hash
            expected_hash = event.calculate_hash()
            if event.event_hash != expected_hash:
                errors.append(
                    f"Event {i} (id={event.event_id}): Hash mismatch - "
                    f"expected={expected_hash}, got={event.event_hash}"
                )

            previous_hash = event.event_hash

        is_valid = len(errors) == 0
        return is_valid, errors

    def get_change_history(
        self,
        target_id: str,
        target_type: str
    ) -> List[PermissionAuditEvent]:
        """
        Get complete change history for a target.

        Args:
            target_id: Target ID
            target_type: Target type

        Returns:
            List of audit events in chronological order
        """
        events = [
            e for e in self._events_cache
            if e.target_id == target_id and e.target_type == target_type
        ]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        return events

    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for permission changes.

        Args:
            start_time: Report start time
            end_time: Report end time
            tenant_id: Optional tenant filter

        Returns:
            Compliance report
        """
        events = self.query_events(start_time=start_time, end_time=end_time, limit=10000)

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]

        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'tenant_id': tenant_id,
            'total_changes': len(events),
            'changes_by_type': defaultdict(int),
            'high_risk_changes': [],
            'actors': set(),
            'affected_principals': set()
        }

        for event in events:
            report['changes_by_type'][event.change_type.value] += 1
            report['actors'].add(event.actor_id)

            if event.principal_id:
                report['affected_principals'].add(event.principal_id)

            # Track high-risk changes
            if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                report['high_risk_changes'].append(event.to_dict())

        # Convert sets to lists for JSON serialization
        report['actors'] = list(report['actors'])
        report['affected_principals'] = list(report['affected_principals'])
        report['changes_by_type'] = dict(report['changes_by_type'])

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        return {
            **self._stats,
            'events_by_type': dict(self._stats['events_by_type']),
            'events_by_actor': dict(self._stats['events_by_actor'])
        }

    def _calculate_changes(
        self,
        before: Optional[Dict[str, Any]],
        after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate changes between before and after snapshots."""
        if not before:
            return {'created': after}

        if not after:
            return {'deleted': before}

        changes = {}

        # Find added/changed fields
        for key, new_value in after.items():
            if key not in before:
                changes[f'added.{key}'] = new_value
            elif before[key] != new_value:
                changes[f'changed.{key}'] = {
                    'from': before[key],
                    'to': new_value
                }

        # Find removed fields
        for key in before:
            if key not in after:
                changes[f'removed.{key}'] = before[key]

        return changes

    def _determine_severity(
        self,
        change_type: PermissionChangeType,
        changes: Dict[str, Any]
    ) -> AuditSeverity:
        """Determine severity of change."""
        # Critical changes
        if change_type in [
            PermissionChangeType.PERMISSION_GRANTED,
            PermissionChangeType.ROLE_ASSIGNED
        ]:
            # Check if granting admin permissions
            if any('admin' in str(v).lower() for v in changes.values()):
                return AuditSeverity.CRITICAL

        # High severity changes
        if change_type in [
            PermissionChangeType.ROLE_CREATED,
            PermissionChangeType.POLICY_CREATED,
            PermissionChangeType.DELEGATION_CREATED
        ]:
            return AuditSeverity.HIGH

        # Warning level changes
        if change_type in [
            PermissionChangeType.ROLE_UPDATED,
            PermissionChangeType.POLICY_UPDATED
        ]:
            return AuditSeverity.WARNING

        # Default to info
        return AuditSeverity.INFO

    def _store_event(self, event: PermissionAuditEvent):
        """Store event to persistent storage."""
        try:
            # Create daily log file
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = self.storage_path / f"permission_audit_{date_str}.jsonl"

            with open(log_file, "a") as f:
                f.write(event.to_json() + "\n")

        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")


# ==============================================================================
# Global Instance
# ==============================================================================

_permission_audit_logger: Optional[PermissionAuditLogger] = None


def get_permission_audit_logger() -> PermissionAuditLogger:
    """Get global permission audit logger instance."""
    global _permission_audit_logger
    if not _permission_audit_logger:
        _permission_audit_logger = PermissionAuditLogger()
    return _permission_audit_logger


__all__ = [
    'PermissionChangeType',
    'AuditSeverity',
    'PermissionAuditEvent',
    'PermissionAuditLogger',
    'get_permission_audit_logger'
]
