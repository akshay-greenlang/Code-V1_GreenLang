#!/usr/bin/env python
# Script to create approvals.py

content = '''# -*- coding: utf-8 -*-
"""FR-043: Signed Approvals/Attestations for GL-FOUND-X-001"""

from __future__ import annotations
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519PrivateKey = None
    InvalidSignature = Exception

try:
    from greenlang.utilities.serialization.canonical import canonical_dumps
except ImportError:
    import json
    def canonical_dumps(obj):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

try:
    from greenlang.utilities.determinism.clock import DeterministicClock
except ImportError:
    class DeterministicClock:
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz or timezone.utc).replace(microsecond=0)

try:
    from greenlang.orchestrator.governance.policy_engine import ApprovalRequirement, ApprovalType
except ImportError:
    class ApprovalType(str, Enum):
        MANAGER = "manager"
        SECURITY = "security"
        DATA_OWNER = "data_owner"
        COMPLIANCE = "compliance"
        COST_CENTER = "cost_center"

    class ApprovalRequirement(BaseModel):
        approval_type: ApprovalType
        approver_id: Optional[str] = None
        approver_role: Optional[str] = None
        reason: str
        deadline_hours: int = 24
        auto_deny_on_timeout: bool = True

try:
    from greenlang.orchestrator.audit.event_store import EventFactory, EventType
except ImportError:
    EventFactory = None
    EventType = None

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalAttestation(BaseModel):
    approver_id: str = Field(..., min_length=1)
    approver_name: Optional[str] = None
    approver_role: Optional[str] = None
    decision: ApprovalDecision
    reason: Optional[str] = Field(None, max_length=2000)
    timestamp: datetime = Field(default_factory=lambda: DeterministicClock.now(timezone.utc))
    signature: str
    public_key: str
    attestation_hash: str = ""

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.replace(microsecond=0)

    def compute_content_hash(self):
        content = {"approver_id": self.approver_id, "decision": self.decision.value,
                   "reason": self.reason, "timestamp": self.timestamp.isoformat()}
        return hashlib.sha256(canonical_dumps(content).encode()).hexdigest()

    def get_signable_content(self):
        content = {"approver_id": self.approver_id, "decision": self.decision.value,
                   "reason": self.reason, "timestamp": self.timestamp.isoformat()}
        return canonical_dumps(content).encode("utf-8")


class ApprovalRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    approval_type: ApprovalType
    reason: str = Field(..., max_length=2000)
    requested_by: Optional[str] = None
    requested_at: datetime = Field(default_factory=lambda: DeterministicClock.now(timezone.utc))
    deadline: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    attestation: Optional[ApprovalAttestation] = None
    requirement: Optional[ApprovalRequirement] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("requested_at", "deadline", mode="before")
    @classmethod
    def ensure_utc_timestamps(cls, v):
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.replace(microsecond=0)

    def is_expired(self):
        return DeterministicClock.now(timezone.utc) > self.deadline

    def compute_provenance_hash(self):
        content = {"request_id": self.request_id, "run_id": self.run_id, "step_id": self.step_id,
                   "approval_type": self.approval_type.value, "reason": self.reason,
                   "requested_at": self.requested_at.isoformat(), "deadline": self.deadline.isoformat(),
                   "status": self.status.value}
        if self.attestation:
            content["attestation_hash"] = self.attestation.attestation_hash
        return hashlib.sha256(canonical_dumps(content).encode()).hexdigest()


@runtime_checkable
class ApprovalStore(Protocol):
    async def save(self, request: ApprovalRequest) -> str: ...
    async def get(self, request_id: str) -> Optional[ApprovalRequest]: ...
    async def get_by_run(self, run_id: str) -> List[ApprovalRequest]: ...
    async def get_by_step(self, run_id: str, step_id: str) -> Optional[ApprovalRequest]: ...
    async def get_pending(self, run_id: Optional[str] = None) -> List[ApprovalRequest]: ...
    async def update(self, request: ApprovalRequest) -> bool: ...
    async def expire_stale(self) -> int: ...


class InMemoryApprovalStore:
    def __init__(self):
        self._requests = {}
        self._by_run = {}
        self._by_step = {}
        logger.info("InMemoryApprovalStore initialized")

    async def save(self, request):
        request.provenance_hash = request.compute_provenance_hash()
        self._requests[request.request_id] = request
        if request.run_id not in self._by_run:
            self._by_run[request.run_id] = []
        if request.request_id not in self._by_run[request.run_id]:
            self._by_run[request.run_id].append(request.request_id)
        self._by_step[f"{request.run_id}:{request.step_id}"] = request.request_id
        return request.request_id

    async def get(self, request_id):
        return self._requests.get(request_id)

    async def get_by_run(self, run_id):
        return [self._requests[rid] for rid in self._by_run.get(run_id, []) if rid in self._requests]

    async def get_by_step(self, run_id, step_id):
        request_id = self._by_step.get(f"{run_id}:{step_id}")
        return self._requests.get(request_id) if request_id else None

    async def get_pending(self, run_id=None):
        requests = await self.get_by_run(run_id) if run_id else list(self._requests.values())
        return [r for r in requests if r.status == ApprovalStatus.PENDING]

    async def update(self, request):
        if request.request_id not in self._requests:
            return False
        request.provenance_hash = request.compute_provenance_hash()
        self._requests[request.request_id] = request
        return True

    async def expire_stale(self):
        count = 0
        now = DeterministicClock.now(timezone.utc)
        for request in self._requests.values():
            if request.status == ApprovalStatus.PENDING and request.deadline < now:
                request.status = ApprovalStatus.EXPIRED
                count += 1
        return count

    async def clear(self):
        self._requests.clear()
        self._by_run.clear()
        self._by_step.clear()


class SignatureUtils:
    @staticmethod
    def generate_keypair():
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        private_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
        public_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo)
        return private_pem, public_pem

    @staticmethod
    def sign(content, private_key_pem):
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        return private_key.sign(content)

    @staticmethod
    def verify(content, signature, public_key_pem):
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography library required")
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            public_key.verify(signature, content)
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def bytes_to_base64(data):
        import base64
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def base64_to_bytes(data):
        import base64
        return base64.b64decode(data.encode("ascii"))


class ApprovalWorkflow:
    def __init__(self, store, event_factory=None, default_deadline_hours=24):
        self._store = store
        self._event_factory = event_factory
        self._default_deadline_hours = default_deadline_hours

    async def request_approval(self, run_id, step_id, requirement, requested_by=None, metadata=None):
        existing = await self._store.get_by_step(run_id, step_id)
        if existing and existing.status == ApprovalStatus.PENDING:
            return existing.request_id
        deadline = DeterministicClock.now(timezone.utc) + timedelta(hours=requirement.deadline_hours or self._default_deadline_hours)
        request_id = f"apr-{uuid4().hex[:12]}"
        request = ApprovalRequest(request_id=request_id, run_id=run_id, step_id=step_id,
            approval_type=requirement.approval_type, reason=requirement.reason,
            requested_by=requested_by, deadline=deadline, requirement=requirement, metadata=metadata or {})
        await self._store.save(request)
        await self._emit_event(run_id, "APPROVAL_REQUESTED", {"request_id": request_id, "step_id": step_id}, step_id)
        return request_id

    async def submit_approval(self, approval_id, approver_id, decision, private_key, public_key, reason=None,
                              approver_name=None, approver_role=None):
        request = await self._store.get(approval_id)
        if not request:
            raise ValueError(f"Approval not found: {approval_id}")
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Already decided: {request.status.value}")
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            await self._store.update(request)
            raise ValueError("Expired")
        timestamp = DeterministicClock.now(timezone.utc)
        attestation = ApprovalAttestation(approver_id=approver_id, approver_name=approver_name,
            approver_role=approver_role, decision=decision, reason=reason, timestamp=timestamp,
            signature="", public_key=SignatureUtils.bytes_to_base64(public_key))
        content = attestation.get_signable_content()
        signature = SignatureUtils.sign(content, private_key)
        attestation.signature = SignatureUtils.bytes_to_base64(signature)
        attestation.attestation_hash = attestation.compute_content_hash()
        request.attestation = attestation
        request.status = ApprovalStatus.APPROVED if decision == ApprovalDecision.APPROVED else ApprovalStatus.REJECTED
        await self._store.update(request)
        await self._emit_event(request.run_id, "APPROVAL_DECISION", {"decision": decision.value}, request.step_id)
        return attestation

    async def check_approval_status(self, approval_id):
        request = await self._store.get(approval_id)
        if not request:
            raise ValueError(f"Not found: {approval_id}")
        if request.status == ApprovalStatus.PENDING and request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            await self._store.update(request)
        return request.status

    async def verify_attestation(self, approval_id):
        request = await self._store.get(approval_id)
        if not request or not request.attestation:
            raise ValueError(f"No attestation: {approval_id}")
        attestation = request.attestation
        if attestation.attestation_hash != attestation.compute_content_hash():
            return False
        content = attestation.get_signable_content()
        signature = SignatureUtils.base64_to_bytes(attestation.signature)
        public_key = SignatureUtils.base64_to_bytes(attestation.public_key)
        try:
            return SignatureUtils.verify(content, signature, public_key)
        except Exception:
            return False

    async def get_pending_approvals(self, run_id=None):
        return await self._store.get_pending(run_id)

    async def get_approval(self, approval_id):
        return await self._store.get(approval_id)

    async def get_step_approval(self, run_id, step_id):
        return await self._store.get_by_step(run_id, step_id)

    async def _emit_event(self, run_id, event_type, payload, step_id=None):
        if self._event_factory and EventType:
            try:
                et = getattr(EventType, event_type, None)
                if et:
                    await self._event_factory.emit(run_id=run_id, event_type=et, payload=payload, step_id=step_id)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")


class ApprovalError(Exception):
    pass

class ApprovalNotFoundError(ApprovalError):
    pass

class ApprovalExpiredError(ApprovalError):
    pass

class ApprovalAlreadyDecidedError(ApprovalError):
    pass

class SignatureVerificationError(ApprovalError):
    pass


__all__ = [
    "ApprovalStatus", "ApprovalDecision", "ApprovalAttestation", "ApprovalRequest",
    "ApprovalStore", "InMemoryApprovalStore", "SignatureUtils", "ApprovalWorkflow",
    "ApprovalError", "ApprovalNotFoundError", "ApprovalExpiredError",
    "ApprovalAlreadyDecidedError", "SignatureVerificationError", "CRYPTO_AVAILABLE",
]
'''

import os
filepath = 'greenlang/orchestrator/governance/approvals.py'
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Created {filepath}')
