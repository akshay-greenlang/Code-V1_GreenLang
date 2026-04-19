# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

SAFETY_SYSTEM_ACCESS_PROHIBITED = True
SCADA_READ_ONLY = True
DAHS_READ_ONLY = True
CONTROL_SYSTEM_ISOLATION = True
APPROVAL_TIMEOUT_SECONDS = 3600

class SafetyLevel(str, Enum):
    SIL_1 = "sil_1"
    SIL_2 = "sil_2"
    SIL_3 = "sil_3"
    SIL_4 = "sil_4"

class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"
    CONFIGURE = "configure"
    EMERGENCY = "emergency"

class SystemType(str, Enum):
    SCADA = "scada"
    DAHS = "dahs"
    PLC = "plc"
    DCS = "dcs"
    SIS = "sis"
    CEMS = "cems"
    ERP = "erp"

class ApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"

class InterlockState(str, Enum):
    ACTIVE = "active"
    BYPASSED = "bypassed"
    TRIPPED = "tripped"
    FAULT = "fault"
    UNKNOWN = "unknown"

class SafetyViolationError(Exception):
    pass

class WriteAccessDeniedError(SafetyViolationError):
    pass

class ApprovalRequiredError(SafetyViolationError):
    pass

class InterlockViolationError(SafetyViolationError):
    pass

class EmergencyShutdownError(SafetyViolationError):
    pass

@dataclass
class SafetyAuditEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation_type: OperationType = OperationType.READ
    system_type: SystemType = SystemType.DAHS
    user_id: Optional[str] = None
    action: str = ""
    allowed: bool = False
    reason: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self.calculate_provenance()

    def calculate_provenance(self) -> str:
        content = f"{self.entry_id}|{self.timestamp.isoformat()}|{self.operation_type.value}|{self.allowed}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class WriteApprovalRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_type: SystemType = SystemType.DAHS
    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)
    state: ApprovalState = ApprovalState.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=APPROVAL_TIMEOUT_SECONDS))
    safety_level: SafetyLevel = SafetyLevel.SIL_2
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self.calculate_provenance()

    def calculate_provenance(self) -> str:
        content = f"{self.request_id}|{self.system_type.value}|{self.operation}"
        return hashlib.sha256(content.encode()).hexdigest()

    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    def approve(self, approved_by: str) -> None:
        if self.is_expired():
            self.state = ApprovalState.EXPIRED
            raise SafetyViolationError("Approval request has expired")
        self.state = ApprovalState.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()

    def reject(self, rejected_by: str, reason: str) -> None:
        self.state = ApprovalState.REJECTED
        self.approved_by = rejected_by
        self.rejection_reason = reason


@dataclass
class SafetyInterlock:
    interlock_id: str
    name: str
    system_type: SystemType
    state: InterlockState = InterlockState.ACTIVE
    description: str = ""
    trip_conditions: List[str] = field(default_factory=list)
    last_checked: datetime = field(default_factory=datetime.utcnow)

    def is_safe(self) -> bool:
        return self.state == InterlockState.ACTIVE

    def trip(self, reason: str) -> None:
        self.state = InterlockState.TRIPPED
        logger.critical(f"Interlock {self.name} TRIPPED: {reason}")

    def reset(self) -> None:
        self.state = InterlockState.ACTIVE
        logger.warning(f"Interlock {self.name} RESET")



class SafetyController:
    _instance: Optional['SafetyController'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'SafetyController':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._audit_log: List[SafetyAuditEntry] = []
        self._approval_requests: Dict[str, WriteApprovalRequest] = {}
        self._interlocks: Dict[str, SafetyInterlock] = {}
        self._emergency_shutdown_active = False
        self._audit_lock = threading.Lock()
        self._read_only_systems: Set[SystemType] = {SystemType.SCADA, SystemType.DAHS, SystemType.SIS}
        logger.info('SafetyController initialized with READ-ONLY enforcement')

    def _log_audit(self, entry: SafetyAuditEntry) -> None:
        with self._audit_lock:
            self._audit_log.append(entry)
            log_msg = f'SAFETY AUDIT: {entry.action} - allowed={entry.allowed} reason={entry.reason}'
            if entry.allowed:
                logger.info(log_msg)
            else:
                logger.warning(log_msg)

    def is_read_only_enforced(self, system_type: SystemType) -> bool:
        if SAFETY_SYSTEM_ACCESS_PROHIBITED:
            return True
        if system_type == SystemType.SCADA and SCADA_READ_ONLY:
            return True
        if system_type == SystemType.DAHS and DAHS_READ_ONLY:
            return True
        return system_type in self._read_only_systems

    def check_read_operation(self, system_type: SystemType, action: str, user_id: Optional[str] = None) -> bool:
        entry = SafetyAuditEntry(
            operation_type=OperationType.READ,
            system_type=system_type,
            user_id=user_id,
            action=action,
            allowed=True,
            reason='Read operations are always permitted'
        )
        self._log_audit(entry)
        return True

    def check_write_operation(self, system_type: SystemType, action: str, user_id: Optional[str] = None) -> bool:
        if self._emergency_shutdown_active:
            entry = SafetyAuditEntry(
                operation_type=OperationType.WRITE,
                system_type=system_type,
                user_id=user_id,
                action=action,
                allowed=False,
                reason='Emergency shutdown is active'
            )
            self._log_audit(entry)
            raise EmergencyShutdownError('Emergency shutdown is active - all write operations blocked')

        if self.is_read_only_enforced(system_type):
            entry = SafetyAuditEntry(
                operation_type=OperationType.WRITE,
                system_type=system_type,
                user_id=user_id,
                action=action,
                allowed=False,
                reason=f'{system_type.value} is configured as READ-ONLY'
            )
            self._log_audit(entry)
            raise WriteAccessDeniedError(f'Write access to {system_type.value} is prohibited. System is READ-ONLY.')

        return False

    def request_write_approval(self, system_type: SystemType, operation: str, parameters: Dict[str, Any], requested_by: str) -> WriteApprovalRequest:
        if self.is_read_only_enforced(system_type):
            raise WriteAccessDeniedError(f'Write operations to {system_type.value} are prohibited')

        request = WriteApprovalRequest(
            system_type=system_type,
            operation=operation,
            parameters=parameters,
            requested_by=requested_by
        )
        self._approval_requests[request.request_id] = request
        logger.warning(f'Write approval requested: {request.request_id} for {operation} on {system_type.value}')
        return request

    def approve_write_request(self, request_id: str, approved_by: str) -> WriteApprovalRequest:
        request = self._approval_requests.get(request_id)
        if not request:
            raise ValueError(f'Approval request {request_id} not found')
        request.approve(approved_by)
        entry = SafetyAuditEntry(
            operation_type=OperationType.WRITE,
            system_type=request.system_type,
            user_id=approved_by,
            action=f'APPROVED: {request.operation}',
            allowed=True,
            reason=f'Approved by {approved_by}'
        )
        self._log_audit(entry)
        return request

    def reject_write_request(self, request_id: str, rejected_by: str, reason: str) -> WriteApprovalRequest:
        request = self._approval_requests.get(request_id)
        if not request:
            raise ValueError(f'Approval request {request_id} not found')
        request.reject(rejected_by, reason)
        entry = SafetyAuditEntry(
            operation_type=OperationType.WRITE,
            system_type=request.system_type,
            user_id=rejected_by,
            action=f'REJECTED: {request.operation}',
            allowed=False,
            reason=reason
        )
        self._log_audit(entry)
        return request


    def register_interlock(self, interlock: SafetyInterlock) -> None:
        self._interlocks[interlock.interlock_id] = interlock
        logger.info(f'Registered safety interlock: {interlock.name}')

    def check_interlocks(self, system_type: SystemType) -> bool:
        for interlock in self._interlocks.values():
            if interlock.system_type == system_type:
                if not interlock.is_safe():
                    raise InterlockViolationError(f'Interlock {interlock.name} is not in safe state: {interlock.state.value}')
        return True

    def trigger_emergency_shutdown(self, reason: str, triggered_by: str) -> None:
        self._emergency_shutdown_active = True
        entry = SafetyAuditEntry(
            operation_type=OperationType.EMERGENCY,
            system_type=SystemType.SIS,
            user_id=triggered_by,
            action='EMERGENCY SHUTDOWN TRIGGERED',
            allowed=True,
            reason=reason
        )
        self._log_audit(entry)
        logger.critical(f'EMERGENCY SHUTDOWN TRIGGERED by {triggered_by}: {reason}')
        for interlock in self._interlocks.values():
            interlock.trip(reason)

    def clear_emergency_shutdown(self, cleared_by: str, authorization_code: str) -> None:
        if not authorization_code:
            raise SafetyViolationError('Authorization code required to clear emergency shutdown')
        self._emergency_shutdown_active = False
        entry = SafetyAuditEntry(
            operation_type=OperationType.EMERGENCY,
            system_type=SystemType.SIS,
            user_id=cleared_by,
            action='EMERGENCY SHUTDOWN CLEARED',
            allowed=True,
            reason=f'Cleared by {cleared_by} with authorization'
        )
        self._log_audit(entry)
        logger.warning(f'Emergency shutdown cleared by {cleared_by}')

    def get_audit_log(self, limit: int = 100) -> List[SafetyAuditEntry]:
        with self._audit_lock:
            return list(self._audit_log[-limit:])

    def is_emergency_shutdown_active(self) -> bool:
        return self._emergency_shutdown_active

    def get_interlock_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            iid: {
                'name': i.name,
                'system_type': i.system_type.value,
                'state': i.state.value,
                'is_safe': i.is_safe()
            }
            for iid, i in self._interlocks.items()
        }


_default_controller: Optional[SafetyController] = None


def get_safety_controller() -> SafetyController:
    global _default_controller
    if _default_controller is None:
        _default_controller = SafetyController()
    return _default_controller


def check_read_access(system_type: SystemType, action: str, user_id: Optional[str] = None) -> bool:
    return get_safety_controller().check_read_operation(system_type, action, user_id)


def check_write_access(system_type: SystemType, action: str, user_id: Optional[str] = None) -> bool:
    return get_safety_controller().check_write_operation(system_type, action, user_id)
