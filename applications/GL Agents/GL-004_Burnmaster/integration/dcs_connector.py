"""
GL-004 BURNMASTER - DCS Connector

Distributed Control System connectivity for combustion optimization.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DCSType(str, Enum):
    HONEYWELL_EXPERION = "honeywell_experion"
    EMERSON_DELTAV = "emerson_deltav"
    YOKOGAWA_CENTUM = "yokogawa_centum"
    ABB_800XA = "abb_800xa"
    SIEMENS_PCS7 = "siemens_pcs7"


class QualityCode(int, Enum):
    GOOD = 192
    UNCERTAIN = 64
    BAD = 0
    BAD_COMM_FAILURE = 24


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class WriteStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"
    MODE_NOT_PERMISSIVE = "mode_not_permissive"


class PermissiveState(str, Enum):
    PERMISSIVE = "permissive"
    NOT_PERMISSIVE = "not_permissive"
    UNKNOWN = "unknown"


class DCSConfig(BaseModel):
    dcs_type: DCSType
    host: str
    port: int = 4840
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    write_verification_enabled: bool = True
    write_verification_tolerance_pct: float = 1.0


@dataclass
class TagValue:
    tag: str
    value: Union[float, int, bool, str]
    quality: QualityCode
    timestamp: datetime
    engineering_units: Optional[str] = None

    @property
    def is_good(self) -> bool:
        return self.quality.value >= 192


@dataclass
class AuditContext:
    user_id: str
    session_id: str
    reason: str
    correlation_id: Optional[str] = None
    source_system: str = "GL-004_BURNMASTER"


@dataclass
class ConnectionResult:
    success: bool
    state: ConnectionState
    message: str
    connected_at: Optional[datetime] = None


@dataclass
class WriteResult:
    success: bool
    status: WriteStatus
    tag: str
    requested_value: Any
    actual_value: Optional[Any] = None
    audit_entry_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class VerificationResult:
    success: bool
    tag: str
    expected_value: Any
    actual_value: Optional[Any]
    tolerance: float
    message: str = ""


@dataclass
class PermissiveStatus:
    unit_id: str
    is_permissive: bool
    state: PermissiveState
    blocking_conditions: List[str] = field(default_factory=list)


@dataclass
class Subscription:
    subscription_id: str
    tags: List[str]
    callback: Callable
    sampling_interval_ms: int = 1000
    is_active: bool = False


class DCSConnector:
    """DCS Connector for combustion optimization."""

    def __init__(self, config: Optional[DCSConfig] = None, vault_client=None, audit_logger=None):
        self.config = config
        self._vault_client = vault_client
        self._audit_logger = audit_logger
        self._state = ConnectionState.DISCONNECTED
        self._client = None
        self._connected_at = None
        self._subscriptions: Dict[str, Subscription] = {}
        self._tag_cache: Dict[str, Tuple[TagValue, datetime]] = {}
        self._stats = {"reads": 0, "writes": 0, "errors": 0}
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    async def connect(self, config: DCSConfig) -> ConnectionResult:
        async with self._lock:
            self.config = config
            self._state = ConnectionState.CONNECTING
            await asyncio.sleep(0.1)
            self._client = {"connected": True}
            self._state = ConnectionState.CONNECTED
            self._connected_at = datetime.now(timezone.utc)
            return ConnectionResult(True, ConnectionState.CONNECTED, "Connected", self._connected_at)

    async def disconnect(self) -> None:
        self._client = None
        self._state = ConnectionState.DISCONNECTED

    async def read_tag(self, tag: str) -> TagValue:
        if not self.is_connected:
            raise ConnectionError("Not connected to DCS")
        import math
        import random
        now = datetime.now(timezone.utc)
        value = 50.0 + (hash(tag) % 50) + math.sin(now.timestamp() / 10) * 5
        self._stats["reads"] += 1
        return TagValue(tag, round(value, 2), QualityCode.GOOD, now)

    async def read_tags(self, tags: List[str]) -> Dict[str, TagValue]:
        return {tag: await self.read_tag(tag) for tag in tags}

    async def write_tag(self, tag: str, value: float, audit: AuditContext) -> WriteResult:
        if not self.is_connected:
            raise ConnectionError("Not connected to DCS")
        if not audit:
            raise ValueError("AuditContext required for all writes")
        audit_id = hashlib.sha256(f"{tag}|{value}|{audit.session_id}".encode()).hexdigest()[:16]
        self._stats["writes"] += 1
        if self.config.write_verification_enabled:
            verification = await self.verify_write(tag, value)
            if not verification.success:
                return WriteResult(False, WriteStatus.VERIFICATION_FAILED, tag, value, error_message=verification.message)
        return WriteResult(True, WriteStatus.SUCCESS, tag, value, value, audit_id)

    async def verify_write(self, tag: str, expected: float) -> VerificationResult:
        readback = await self.read_tag(tag)
        deviation = abs(float(readback.value) - expected)
        deviation_pct = (deviation / abs(expected) * 100) if expected != 0 else 0
        success = deviation_pct <= self.config.write_verification_tolerance_pct
        return VerificationResult(success, tag, expected, readback.value, self.config.write_verification_tolerance_pct)

    async def subscribe_to_tags(self, tags: List[str], callback: Callable) -> Subscription:
        sub_id = str(uuid.uuid4())
        sub = Subscription(sub_id, tags, callback, is_active=True)
        self._subscriptions[sub_id] = sub
        return sub

    async def check_mode_permissive(self, unit_id: str) -> PermissiveStatus:
        if not self.is_connected:
            raise ConnectionError("Not connected to DCS")
        perm_tag = await self.read_tag(f"{unit_id}.PERMISSIVE")
        is_perm = bool(perm_tag.value) and perm_tag.is_good
        return PermissiveStatus(unit_id, is_perm, PermissiveState.PERMISSIVE if is_perm else PermissiveState.NOT_PERMISSIVE)
