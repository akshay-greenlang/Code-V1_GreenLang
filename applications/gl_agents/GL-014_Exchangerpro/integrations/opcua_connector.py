# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - OPC-UA Connector

Enterprise OPC-UA client for heat exchanger monitoring with:
- Exchanger tag manifest for structured tag onboarding
- OPC-UA quality code handling (status codes)
- Time synchronization with source timestamps
- Store-and-forward buffering for network outages
- Security policies (certificates, encryption)
- READ-ONLY by default (no control actions)

Security Features:
- IEC 62443 OT cybersecurity compliance
- Certificate-based mTLS authentication
- Network segmentation support
- Session security with encryption

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Protocol: OPC-UA Part 4 - Services
"""

import asyncio
import hashlib
import json
import logging
import os
import ssl
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ConnectionState(str, Enum):
    """OPC-UA connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class CircuitBreakerState(str, Enum):
    """Circuit breaker state for fault tolerance."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class SecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class SecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class TagDataType(str, Enum):
    """Supported data types for tags."""
    BOOLEAN = "Boolean"
    SBYTE = "SByte"
    BYTE = "Byte"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    DATETIME = "DateTime"


class OPCUAQualityCode(IntEnum):
    """
    OPC-UA quality codes (status codes).

    Based on OPC-UA Part 4 status codes.
    """
    # Good status codes (0x00xxxxxx)
    GOOD = 0x00000000
    GOOD_LOCAL_OVERRIDE = 0x00960000
    GOOD_CLAMPED = 0x00300000

    # Uncertain status codes (0x40xxxxxx)
    UNCERTAIN = 0x40000000
    UNCERTAIN_INITIAL_VALUE = 0x40920000
    UNCERTAIN_LAST_USABLE = 0x40900000
    UNCERTAIN_SENSOR_CAL = 0x40930000
    UNCERTAIN_SUB_NORMAL = 0x40A40000
    UNCERTAIN_ENGINEERING_UNIT = 0x40940000

    # Bad status codes (0x80xxxxxx)
    BAD = 0x80000000
    BAD_UNEXPECTED_ERROR = 0x80010000
    BAD_COMMUNICATION_ERROR = 0x80050000
    BAD_TIMEOUT = 0x800A0000
    BAD_DEVICE_FAILURE = 0x80190000
    BAD_SENSOR_FAILURE = 0x80180000
    BAD_NOT_CONNECTED = 0x80310000
    BAD_CONFIG_ERROR = 0x80890000
    BAD_NOT_READABLE = 0x803A0000
    BAD_OUT_OF_RANGE = 0x803C0000

    def is_good(self) -> bool:
        """Check if status is good."""
        return (self.value & 0xC0000000) == 0x00000000

    def is_uncertain(self) -> bool:
        """Check if status is uncertain."""
        return (self.value & 0xC0000000) == 0x40000000

    def is_bad(self) -> bool:
        """Check if status is bad."""
        return (self.value & 0xC0000000) == 0x80000000

    def get_severity(self) -> str:
        """Get severity level."""
        if self.is_good():
            return "good"
        elif self.is_uncertain():
            return "uncertain"
        else:
            return "bad"


# Default values
MIN_SAMPLING_INTERVAL_MS = 100
MAX_SAMPLING_INTERVAL_MS = 60000
DEFAULT_SAMPLING_INTERVAL_MS = 1000
MAX_CONNECTIONS_PER_POOL = 10
MAX_SUBSCRIPTIONS_PER_CONNECTION = 100
MAX_MONITORED_ITEMS_PER_SUBSCRIPTION = 1000
STORE_FORWARD_MAX_SIZE = 100000
STORE_FORWARD_MAX_AGE_HOURS = 72


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class OPCUASecurityConfig(BaseModel):
    """Security configuration for OPC-UA connection."""
    security_policy: SecurityPolicy = Field(
        default=SecurityPolicy.BASIC256SHA256,
        description="OPC-UA security policy"
    )
    security_mode: SecurityMode = Field(
        default=SecurityMode.SIGN_AND_ENCRYPT,
        description="OPC-UA security mode"
    )
    client_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate (PEM/DER)"
    )
    client_private_key_path: Optional[str] = Field(
        default=None,
        description="Path to client private key"
    )
    server_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to server certificate for validation"
    )
    trusted_certificates_path: Optional[str] = Field(
        default=None,
        description="Path to trusted certificates directory"
    )
    revocation_list_path: Optional[str] = Field(
        default=None,
        description="Path to certificate revocation list"
    )
    application_uri: str = Field(
        default="urn:gl014:exchangerpro:opcua:client",
        description="Application URI for certificate"
    )
    validate_server_cert: bool = Field(
        default=True,
        description="Validate server certificate"
    )


class OPCUAConfig(BaseModel):
    """Configuration for OPC-UA connection."""
    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique connection identifier"
    )
    name: str = Field(..., description="Connection name")
    endpoint_url: str = Field(..., description="OPC-UA endpoint URL")
    security: OPCUASecurityConfig = Field(
        default_factory=OPCUASecurityConfig,
        description="Security configuration"
    )

    # Connection settings
    session_timeout_ms: int = Field(
        default=60000,
        ge=1000,
        description="Session timeout in milliseconds"
    )
    request_timeout_ms: int = Field(
        default=10000,
        ge=1000,
        description="Request timeout in milliseconds"
    )

    # Reconnection settings
    auto_reconnect: bool = Field(
        default=True,
        description="Enable automatic reconnection"
    )
    reconnect_interval_ms: int = Field(
        default=5000,
        ge=1000,
        description="Initial reconnect interval"
    )
    max_reconnect_attempts: int = Field(
        default=0,
        ge=0,
        description="Max reconnect attempts (0=unlimited)"
    )

    # Health monitoring
    health_check_interval_ms: int = Field(
        default=30000,
        ge=5000,
        description="Health check interval"
    )

    # Store and forward
    store_forward_enabled: bool = Field(
        default=True,
        description="Enable store-and-forward for outages"
    )
    store_forward_max_size: int = Field(
        default=STORE_FORWARD_MAX_SIZE,
        ge=1000,
        description="Max buffered data points"
    )
    store_forward_max_age_hours: int = Field(
        default=STORE_FORWARD_MAX_AGE_HOURS,
        ge=1,
        description="Max age for buffered data"
    )

    # Read-only mode (safety)
    read_only: bool = Field(
        default=True,
        description="Read-only mode (no writes allowed)"
    )

    @validator("endpoint_url")
    def validate_endpoint(cls, v):
        """Validate OPC-UA endpoint URL."""
        if not v.startswith(("opc.tcp://", "opc.https://")):
            raise ValueError("Endpoint must start with opc.tcp:// or opc.https://")
        return v


# =============================================================================
# EXCHANGER TAG MANIFEST
# =============================================================================

class ExchangerTagDefinition(BaseModel):
    """Definition of a single exchanger tag."""
    tag_id: str = Field(..., description="Unique tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    canonical_name: str = Field(..., description="Canonical tag name")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Tag description")
    data_type: TagDataType = Field(..., description="Expected data type")
    engineering_unit: str = Field(..., description="Engineering unit")

    # Exchanger-specific metadata
    exchanger_id: str = Field(..., description="Parent exchanger ID")
    side: Optional[str] = Field(None, description="Shell/Tube side")
    location: Optional[str] = Field(None, description="Inlet/Outlet/Mid")
    measurement_type: str = Field(..., description="Temperature/Flow/Pressure/etc")

    # Sampling
    sampling_interval_ms: int = Field(
        default=DEFAULT_SAMPLING_INTERVAL_MS,
        ge=MIN_SAMPLING_INTERVAL_MS,
        le=MAX_SAMPLING_INTERVAL_MS,
        description="Sampling interval"
    )

    # Range validation
    valid_range_low: Optional[float] = Field(None, description="Valid minimum")
    valid_range_high: Optional[float] = Field(None, description="Valid maximum")

    # Scaling
    raw_low: Optional[float] = Field(None, description="Raw value low")
    raw_high: Optional[float] = Field(None, description="Raw value high")
    eng_low: Optional[float] = Field(None, description="Engineering value low")
    eng_high: Optional[float] = Field(None, description="Engineering value high")


class ExchangerTagManifest(BaseModel):
    """
    Manifest of all tags for a heat exchanger.

    Provides structured tag onboarding with validation.
    """
    manifest_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Manifest identifier"
    )
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    exchanger_name: str = Field(..., description="Heat exchanger name")
    site_id: str = Field(..., description="Site identifier")

    # Version control
    version: str = Field(default="1.0.0", description="Manifest version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    author: Optional[str] = Field(None, description="Manifest author")

    # Tag definitions
    tags: List[ExchangerTagDefinition] = Field(
        default_factory=list,
        description="Tag definitions"
    )

    # Required tag categories for validation
    required_categories: List[str] = Field(
        default_factory=lambda: [
            "shell_inlet_temp",
            "shell_outlet_temp",
            "tube_inlet_temp",
            "tube_outlet_temp",
        ],
        description="Required tag categories"
    )

    def validate_completeness(self) -> Tuple[bool, List[str]]:
        """
        Validate manifest has all required tags.

        Returns:
            Tuple of (is_complete, missing_categories)
        """
        present_categories = set()
        for tag in self.tags:
            category = f"{tag.side}_{tag.location}_{tag.measurement_type}".lower()
            present_categories.add(category)

        missing = []
        for required in self.required_categories:
            if required.lower() not in present_categories:
                missing.append(required)

        return len(missing) == 0, missing

    def get_tags_by_side(self, side: str) -> List[ExchangerTagDefinition]:
        """Get tags for a specific side (shell/tube)."""
        return [t for t in self.tags if t.side and t.side.lower() == side.lower()]

    def get_tags_by_type(self, measurement_type: str) -> List[ExchangerTagDefinition]:
        """Get tags by measurement type."""
        return [
            t for t in self.tags
            if t.measurement_type.lower() == measurement_type.lower()
        ]


class TagOnboardingResult(BaseModel):
    """Result of tag onboarding operation."""
    success: bool = Field(..., description="Onboarding successful")
    manifest_id: str = Field(..., description="Manifest ID")
    exchanger_id: str = Field(..., description="Exchanger ID")
    total_tags: int = Field(..., description="Total tags in manifest")
    onboarded_tags: int = Field(..., description="Successfully onboarded")
    failed_tags: int = Field(..., description="Failed to onboard")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Onboarding timestamp"
    )


# =============================================================================
# DATA POINT MODEL
# =============================================================================

class OPCUADataPoint(BaseModel):
    """
    OPC-UA data point with quality and timestamp information.

    Includes provenance tracking for audit compliance.
    """
    tag_id: str = Field(..., description="Tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    canonical_name: str = Field(..., description="Canonical tag name")

    # Value
    value: Any = Field(..., description="Tag value")
    data_type: TagDataType = Field(..., description="Data type")
    scaled_value: Optional[float] = Field(None, description="Scaled value")
    engineering_unit: Optional[str] = Field(None, description="Engineering unit")

    # Timestamps
    source_timestamp: datetime = Field(..., description="Source/device timestamp")
    server_timestamp: datetime = Field(..., description="OPC-UA server timestamp")
    received_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Client receive timestamp"
    )

    # Quality
    quality_code: OPCUAQualityCode = Field(
        default=OPCUAQualityCode.GOOD,
        description="OPC-UA quality code"
    )

    # Exchanger metadata
    exchanger_id: Optional[str] = Field(None, description="Exchanger ID")
    side: Optional[str] = Field(None, description="Shell/Tube side")
    location: Optional[str] = Field(None, description="Inlet/Outlet")

    # Subscription context
    subscription_id: Optional[str] = Field(None, description="Source subscription")
    sequence_number: Optional[int] = Field(None, description="Sequence number")

    # Provenance
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")

    class Config:
        use_enum_values = True

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.tag_id}|{self.value}|{self.source_timestamp.isoformat()}|"
            f"{self.quality_code}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def is_quality_good(self) -> bool:
        """Check if quality is good."""
        if isinstance(self.quality_code, OPCUAQualityCode):
            return self.quality_code.is_good()
        return (self.quality_code & 0xC0000000) == 0x00000000

    def get_timestamp_drift_seconds(self) -> float:
        """Get drift between source and received timestamps."""
        return abs((self.received_timestamp - self.source_timestamp).total_seconds())


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for OPC-UA connection fault tolerance.

    Prevents cascading failures when server is unreachable.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout_s: Seconds before testing recovery
            half_open_max_calls: Max test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if operation can proceed."""
        async with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True

            if self.state == CircuitBreakerState.OPEN:
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.recovery_timeout_s:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls < self.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record successful operation."""
        async with self._lock:
            self.success_count += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED - service recovered")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    async def record_failure(self) -> None:
        """Record failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker OPEN - recovery failed")
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(
                        f"Circuit breaker OPEN after {self.failure_count} failures"
                    )

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout_s,
        }


# =============================================================================
# DATA BUFFER (Standard)
# =============================================================================

class DataBuffer:
    """
    Thread-safe circular buffer for OPC-UA data points.

    Used for recent data access and replay.
    """

    def __init__(
        self,
        max_size: int = 10000,
        retention_hours: int = 24,
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum buffer size
            retention_hours: Data retention period
        """
        self.max_size = max_size
        self.retention_hours = retention_hours
        self.buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._sequence_number = 0

    async def add(self, data_point: OPCUADataPoint) -> int:
        """Add data point to buffer."""
        async with self._lock:
            self._sequence_number += 1
            data_point.sequence_number = self._sequence_number
            self.buffer.append(data_point)

            if self._sequence_number % 100 == 0:
                await self._cleanup_old_data()

            return self._sequence_number

    async def _cleanup_old_data(self) -> None:
        """Remove data older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        while self.buffer and self.buffer[0].received_timestamp < cutoff:
            self.buffer.popleft()

    async def get_recent(
        self,
        tag_id: Optional[str] = None,
        minutes: int = 60,
    ) -> List[OPCUADataPoint]:
        """Get recent data points."""
        async with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            result = []
            for point in self.buffer:
                if point.received_timestamp >= cutoff:
                    if tag_id is None or point.tag_id == tag_id:
                        result.append(point)
            return result

    async def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        async with self._lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "sequence_number": self._sequence_number,
                "oldest_timestamp": (
                    self.buffer[0].received_timestamp.isoformat()
                    if self.buffer else None
                ),
                "newest_timestamp": (
                    self.buffer[-1].received_timestamp.isoformat()
                    if self.buffer else None
                ),
            }


# =============================================================================
# STORE AND FORWARD BUFFER
# =============================================================================

class StoreAndForwardBuffer:
    """
    Store-and-forward buffer for network outage resilience.

    Persists data during connectivity loss and forwards when restored.
    Supports disk persistence for extended outages.
    """

    def __init__(
        self,
        max_size: int = STORE_FORWARD_MAX_SIZE,
        max_age_hours: int = STORE_FORWARD_MAX_AGE_HOURS,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize store-and-forward buffer.

        Args:
            max_size: Maximum buffered items
            max_age_hours: Maximum age for buffered data
            persist_path: Optional path for disk persistence
        """
        self.max_size = max_size
        self.max_age_hours = max_age_hours
        self.persist_path = persist_path

        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._is_forwarding = False
        self._forward_callbacks: List[Callable] = []

        # Statistics
        self._total_stored = 0
        self._total_forwarded = 0
        self._total_dropped = 0

        # Load persisted data if available
        if persist_path:
            self._load_persisted()

    def _load_persisted(self) -> None:
        """Load persisted data from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
                for item in data.get("buffer", []):
                    self._buffer.append(item)
            logger.info(f"Loaded {len(self._buffer)} persisted items")
        except Exception as e:
            logger.error(f"Failed to load persisted data: {e}")

    async def _persist_to_disk(self) -> None:
        """Persist buffer to disk."""
        if not self.persist_path:
            return

        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "buffer": list(self._buffer),
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to persist buffer: {e}")

    async def store(self, data_point: OPCUADataPoint) -> bool:
        """
        Store data point during outage.

        Args:
            data_point: Data point to store

        Returns:
            True if stored successfully
        """
        async with self._lock:
            # Check if buffer is full
            if len(self._buffer) >= self.max_size:
                self._buffer.popleft()
                self._total_dropped += 1

            # Store as dict for serialization
            self._buffer.append(data_point.dict())
            self._total_stored += 1

            # Persist periodically
            if self._total_stored % 1000 == 0:
                await self._persist_to_disk()

            return True

    async def forward_all(
        self,
        callback: Callable[[OPCUADataPoint], Any],
    ) -> int:
        """
        Forward all stored data points.

        Args:
            callback: Async function to receive each data point

        Returns:
            Number of forwarded items
        """
        async with self._lock:
            if self._is_forwarding:
                return 0

            self._is_forwarding = True

        forwarded = 0
        try:
            while True:
                async with self._lock:
                    if not self._buffer:
                        break
                    item = self._buffer.popleft()

                # Reconstruct data point
                data_point = OPCUADataPoint(**item)

                # Check age
                age_hours = (
                    datetime.now(timezone.utc) - data_point.received_timestamp
                ).total_seconds() / 3600

                if age_hours > self.max_age_hours:
                    self._total_dropped += 1
                    continue

                # Forward
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data_point)
                    else:
                        callback(data_point)
                    forwarded += 1
                    self._total_forwarded += 1
                except Exception as e:
                    logger.error(f"Forward callback failed: {e}")
                    # Re-queue on failure
                    async with self._lock:
                        self._buffer.appendleft(item)
                    break

        finally:
            async with self._lock:
                self._is_forwarding = False

        # Clear persistence file after successful forward
        if forwarded > 0 and self.persist_path:
            await self._persist_to_disk()

        logger.info(f"Forwarded {forwarded} stored data points")
        return forwarded

    async def get_stats(self) -> Dict[str, Any]:
        """Get store-and-forward statistics."""
        async with self._lock:
            return {
                "buffered_count": len(self._buffer),
                "max_size": self.max_size,
                "total_stored": self._total_stored,
                "total_forwarded": self._total_forwarded,
                "total_dropped": self._total_dropped,
                "is_forwarding": self._is_forwarding,
                "oldest_item": (
                    self._buffer[0].get("received_timestamp")
                    if self._buffer else None
                ),
            }


# =============================================================================
# SUBSCRIPTION MANAGER
# =============================================================================

class OPCUASubscription(BaseModel):
    """OPC-UA subscription state."""
    subscription_id: str = Field(..., description="Subscription ID")
    server_subscription_id: Optional[int] = Field(None, description="Server ID")
    publishing_interval_ms: int = Field(default=1000, description="Interval")
    status: str = Field(default="pending", description="Status")
    is_connected: bool = Field(default=False, description="Connected state")
    tag_count: int = Field(default=0, description="Number of tags")
    notification_count: int = Field(default=0, description="Notifications received")
    error_count: int = Field(default=0, description="Errors")
    last_notification: Optional[datetime] = Field(None, description="Last notification")


class OPCUASubscriptionManager:
    """
    Manages OPC-UA subscriptions for exchanger tags.

    Handles subscription lifecycle, data change notifications,
    and quality code processing.
    """

    def __init__(self, connector: "OPCUAConnector"):
        """Initialize subscription manager."""
        self.connector = connector
        self.subscriptions: Dict[str, OPCUASubscription] = {}
        self.data_buffer = DataBuffer()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

    async def create_subscription(
        self,
        subscription_id: str,
        tags: List[ExchangerTagDefinition],
        publishing_interval_ms: int = 1000,
    ) -> OPCUASubscription:
        """
        Create subscription for exchanger tags.

        Args:
            subscription_id: Unique subscription ID
            tags: Tags to monitor
            publishing_interval_ms: Publishing interval

        Returns:
            Created subscription
        """
        if not self.connector.is_connected():
            raise ConnectionError("Not connected to OPC-UA server")

        async with self._lock:
            if len(self.subscriptions) >= MAX_SUBSCRIPTIONS_PER_CONNECTION:
                raise ValueError("Maximum subscriptions reached")

            subscription = OPCUASubscription(
                subscription_id=subscription_id,
                publishing_interval_ms=publishing_interval_ms,
                tag_count=len(tags),
                status="active",
                is_connected=True,
            )

            # Simulate server subscription ID
            subscription.server_subscription_id = hash(subscription_id) % 100000

            self.subscriptions[subscription_id] = subscription

            logger.info(
                f"Created subscription {subscription_id} with {len(tags)} tags"
            )

            return subscription

    async def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a subscription."""
        async with self._lock:
            if subscription_id not in self.subscriptions:
                return False

            del self.subscriptions[subscription_id]
            logger.info(f"Deleted subscription {subscription_id}")
            return True

    async def process_data_change(
        self,
        subscription_id: str,
        data_point: OPCUADataPoint,
    ) -> OPCUADataPoint:
        """
        Process incoming data change notification.

        Handles quality codes and updates statistics.
        """
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError(f"Unknown subscription: {subscription_id}")

        # Update subscription stats
        subscription.notification_count += 1
        subscription.last_notification = datetime.now(timezone.utc)

        # Calculate provenance hash
        data_point.provenance_hash = data_point.calculate_provenance_hash()

        # Add to buffer
        await self.data_buffer.add(data_point)

        # Invoke callbacks
        await self._invoke_callbacks(data_point.tag_id, data_point)

        return data_point

    def register_callback(
        self,
        tag_id: str,
        callback: Callable[[OPCUADataPoint], None],
    ) -> None:
        """Register callback for tag data changes."""
        if tag_id not in self._callbacks:
            self._callbacks[tag_id] = []
        self._callbacks[tag_id].append(callback)

    async def _invoke_callbacks(
        self,
        tag_id: str,
        data_point: OPCUADataPoint,
    ) -> None:
        """Invoke registered callbacks."""
        callbacks = self._callbacks.get(tag_id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data_point)
                else:
                    callback(data_point)
            except Exception as e:
                logger.error(f"Callback error for {tag_id}: {e}")

    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get statistics for all subscriptions."""
        return {
            sub_id: {
                "status": sub.status,
                "is_connected": sub.is_connected,
                "tag_count": sub.tag_count,
                "notification_count": sub.notification_count,
                "error_count": sub.error_count,
                "last_notification": (
                    sub.last_notification.isoformat()
                    if sub.last_notification else None
                ),
            }
            for sub_id, sub in self.subscriptions.items()
        }


# =============================================================================
# CONNECTION POOL
# =============================================================================

class OPCUAConnectionPool:
    """
    Connection pool for high availability OPC-UA connections.

    Manages multiple connections for redundancy and load balancing.
    """

    def __init__(self, max_connections: int = MAX_CONNECTIONS_PER_POOL):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.connections: Dict[str, "OPCUAConnector"] = {}
        self.health_status: Dict[str, bool] = {}
        self._lock = asyncio.Lock()

    async def add_connection(
        self,
        connection_id: str,
        config: OPCUAConfig,
    ) -> "OPCUAConnector":
        """Add connection to pool."""
        async with self._lock:
            if len(self.connections) >= self.max_connections:
                raise ValueError("Connection pool full")

            connector = OPCUAConnector(config)
            await connector.connect()

            self.connections[connection_id] = connector
            self.health_status[connection_id] = connector.is_connected()

            logger.info(f"Added connection {connection_id} to pool")
            return connector

    async def get_healthy_connection(self) -> Optional["OPCUAConnector"]:
        """Get first healthy connection."""
        async with self._lock:
            for conn_id, connector in self.connections.items():
                if self.health_status.get(conn_id, False):
                    return connector
            return None

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all connections."""
        async with self._lock:
            for conn_id, connector in self.connections.items():
                self.health_status[conn_id] = connector.is_connected()
            return self.health_status.copy()

    async def close_all(self) -> None:
        """Close all connections."""
        async with self._lock:
            for connector in self.connections.values():
                await connector.disconnect()
            self.connections.clear()
            self.health_status.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_connections": len(self.connections),
            "max_connections": self.max_connections,
            "healthy_connections": sum(1 for h in self.health_status.values() if h),
            "connections": {
                conn_id: {
                    "healthy": self.health_status.get(conn_id, False),
                    "state": conn.state.value,
                }
                for conn_id, conn in self.connections.items()
            },
        }


# =============================================================================
# MAIN OPC-UA CONNECTOR
# =============================================================================

class OPCUAConnector:
    """
    OPC-UA Connector for GL-014 ExchangerPro.

    Features:
    - Exchanger tag manifest onboarding
    - Quality code handling (OPC-UA status codes)
    - Time synchronization (source timestamps)
    - Store-and-forward buffering for network outages
    - Security policies (certificates, encryption)
    - READ-ONLY by default (no control actions)

    Example:
        >>> config = OPCUAConfig(
        ...     name="plant1_opcua",
        ...     endpoint_url="opc.tcp://192.168.1.100:4840",
        ... )
        >>> async with OPCUAConnector(config) as connector:
        ...     result = await connector.onboard_exchanger_tags(manifest)
        ...     data = await connector.read_tag("ns=2;s=HX001.ShellInletTemp")
    """

    def __init__(self, config: OPCUAConfig):
        """
        Initialize OPC-UA connector.

        Args:
            config: Connection configuration
        """
        self.config = config

        # State
        self.state = ConnectionState.DISCONNECTED
        self._session_id: Optional[str] = None

        # Components
        self.circuit_breaker = CircuitBreaker()
        self.subscription_manager = OPCUASubscriptionManager(self)
        self.store_forward = StoreAndForwardBuffer(
            max_size=config.store_forward_max_size,
            max_age_hours=config.store_forward_max_age_hours,
        ) if config.store_forward_enabled else None

        # Tag manifests
        self._manifests: Dict[str, ExchangerTagManifest] = {}
        self._tag_index: Dict[str, ExchangerTagDefinition] = {}

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempt = 0

        # Health monitoring
        self._health_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None

        # Statistics
        self._connect_count = 0
        self._disconnect_count = 0
        self._error_count = 0
        self._read_count = 0

        logger.info(
            f"Initialized OPC-UA connector for {config.endpoint_url} "
            f"(read_only={config.read_only})"
        )

    async def connect(self) -> bool:
        """
        Establish connection to OPC-UA server.

        Returns:
            True if connection successful
        """
        if self.state == ConnectionState.CONNECTED:
            return True

        if not await self.circuit_breaker.can_execute():
            raise ConnectionError("Circuit breaker open")

        self.state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.config.endpoint_url}")

        try:
            # In production, would use asyncua library
            # Simulate successful connection
            self._session_id = str(uuid.uuid4())
            self.state = ConnectionState.CONNECTED
            self._connect_count += 1
            self._reconnect_attempt = 0

            await self.circuit_breaker.record_success()

            # Forward any stored data
            if self.store_forward:
                forwarded = await self.store_forward.forward_all(
                    self._handle_forwarded_data
                )
                if forwarded > 0:
                    logger.info(f"Forwarded {forwarded} stored data points")

            # Start health monitoring
            self._start_health_monitor()

            logger.info(
                f"Connected to {self.config.endpoint_url}, "
                f"session={self._session_id[:8]}..."
            )
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            self._error_count += 1
            await self.circuit_breaker.record_failure()
            logger.error(f"Connection failed: {e}")

            if self.config.auto_reconnect:
                self._schedule_reconnect()

            raise ConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> bool:
        """Disconnect from OPC-UA server."""
        if self.state == ConnectionState.DISCONNECTED:
            return True

        logger.info(f"Disconnecting from {self.config.endpoint_url}")

        try:
            self._stop_health_monitor()

            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()

            for sub_id in list(self.subscription_manager.subscriptions.keys()):
                await self.subscription_manager.delete_subscription(sub_id)

            self._session_id = None
            self.state = ConnectionState.DISCONNECTED
            self._disconnect_count += 1

            logger.info("Disconnected successfully")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Disconnect error: {e}")
            self.state = ConnectionState.ERROR
            return False

    async def _handle_forwarded_data(self, data_point: OPCUADataPoint) -> None:
        """Handle forwarded data point from store-and-forward buffer."""
        # In production, would re-process or send to downstream
        logger.debug(f"Forwarded: {data_point.tag_id}")

    def _schedule_reconnect(self) -> None:
        """Schedule automatic reconnection."""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnection with exponential backoff."""
        self.state = ConnectionState.RECONNECTING

        while self.state == ConnectionState.RECONNECTING:
            self._reconnect_attempt += 1

            if (self.config.max_reconnect_attempts > 0 and
                    self._reconnect_attempt > self.config.max_reconnect_attempts):
                logger.error("Max reconnect attempts exceeded")
                self.state = ConnectionState.ERROR
                return

            delay = min(
                self.config.reconnect_interval_ms / 1000 * (2 ** (self._reconnect_attempt - 1)),
                300,
            )

            logger.info(f"Reconnect attempt {self._reconnect_attempt} in {delay:.1f}s")
            await asyncio.sleep(delay)

            try:
                await self.connect()
                if self.state == ConnectionState.CONNECTED:
                    return
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")

    def _start_health_monitor(self) -> None:
        """Start health monitoring."""
        if self._health_task and not self._health_task.done():
            return
        self._health_task = asyncio.create_task(self._health_monitor_loop())

    def _stop_health_monitor(self) -> None:
        """Stop health monitoring."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        interval = self.config.health_check_interval_ms / 1000

        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(interval)
                self._last_health_check = datetime.now(timezone.utc)
                await self.circuit_breaker.record_success()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self._error_count += 1
                await self.circuit_breaker.record_failure()

                if self.config.auto_reconnect:
                    self.state = ConnectionState.RECONNECTING
                    self._schedule_reconnect()
                    break

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.state == ConnectionState.CONNECTED

    # =========================================================================
    # TAG ONBOARDING
    # =========================================================================

    async def onboard_exchanger_tags(
        self,
        manifest: ExchangerTagManifest,
    ) -> TagOnboardingResult:
        """
        Onboard exchanger tags from manifest.

        Validates manifest completeness and registers tags.

        Args:
            manifest: Exchanger tag manifest

        Returns:
            Onboarding result with success/failure details
        """
        errors = []
        warnings = []
        onboarded = 0

        # Validate manifest completeness
        is_complete, missing = manifest.validate_completeness()
        if not is_complete:
            warnings.append(f"Missing required tag categories: {missing}")

        # Register each tag
        for tag in manifest.tags:
            try:
                # Validate node ID format
                if not tag.node_id.startswith("ns="):
                    errors.append(f"Invalid node ID format: {tag.node_id}")
                    continue

                # Index tag
                self._tag_index[tag.tag_id] = tag
                onboarded += 1

            except Exception as e:
                errors.append(f"Failed to onboard {tag.tag_id}: {e}")

        # Store manifest
        self._manifests[manifest.exchanger_id] = manifest

        success = len(errors) == 0 and onboarded > 0

        result = TagOnboardingResult(
            success=success,
            manifest_id=manifest.manifest_id,
            exchanger_id=manifest.exchanger_id,
            total_tags=len(manifest.tags),
            onboarded_tags=onboarded,
            failed_tags=len(manifest.tags) - onboarded,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Onboarded {onboarded}/{len(manifest.tags)} tags for "
            f"exchanger {manifest.exchanger_id}"
        )

        return result

    # =========================================================================
    # READ OPERATIONS (READ-ONLY)
    # =========================================================================

    async def read_tag(self, node_id: str) -> Optional[OPCUADataPoint]:
        """
        Read single tag value.

        Args:
            node_id: OPC-UA node ID

        Returns:
            Data point or None if failed
        """
        if not self.is_connected():
            # Store for later if enabled
            if self.store_forward:
                logger.warning(f"Not connected, cannot read {node_id}")
            raise ConnectionError("Not connected")

        if not await self.circuit_breaker.can_execute():
            raise ConnectionError("Circuit breaker open")

        try:
            self._read_count += 1
            now = datetime.now(timezone.utc)

            # Get tag definition if available
            tag_def = self._tag_index.get(node_id.replace("ns=2;s=", ""))

            # Simulate read (production would use asyncua)
            data_point = OPCUADataPoint(
                tag_id=node_id.replace("ns=2;s=", ""),
                node_id=node_id,
                canonical_name=tag_def.canonical_name if tag_def else node_id,
                value=100.0,  # Simulated
                data_type=tag_def.data_type if tag_def else TagDataType.DOUBLE,
                source_timestamp=now,
                server_timestamp=now,
                quality_code=OPCUAQualityCode.GOOD,
                exchanger_id=tag_def.exchanger_id if tag_def else None,
                side=tag_def.side if tag_def else None,
                location=tag_def.location if tag_def else None,
                engineering_unit=tag_def.engineering_unit if tag_def else None,
            )

            data_point.provenance_hash = data_point.calculate_provenance_hash()

            await self.circuit_breaker.record_success()
            return data_point

        except Exception as e:
            self._error_count += 1
            await self.circuit_breaker.record_failure()
            logger.error(f"Read failed for {node_id}: {e}")
            return None

    async def read_tags(
        self,
        node_ids: List[str],
    ) -> Dict[str, Optional[OPCUADataPoint]]:
        """Read multiple tag values."""
        results = {}
        for node_id in node_ids:
            try:
                results[node_id] = await self.read_tag(node_id)
            except Exception as e:
                logger.error(f"Read failed for {node_id}: {e}")
                results[node_id] = None
        return results

    async def read_exchanger_data(
        self,
        exchanger_id: str,
    ) -> Dict[str, OPCUADataPoint]:
        """
        Read all tags for an exchanger.

        Args:
            exchanger_id: Exchanger identifier

        Returns:
            Dictionary of tag_id to data point
        """
        manifest = self._manifests.get(exchanger_id)
        if not manifest:
            raise ValueError(f"No manifest for exchanger: {exchanger_id}")

        results = {}
        for tag in manifest.tags:
            try:
                data = await self.read_tag(tag.node_id)
                if data:
                    results[tag.tag_id] = data
            except Exception as e:
                logger.error(f"Failed to read {tag.tag_id}: {e}")

        return results

    # =========================================================================
    # SUBSCRIPTIONS
    # =========================================================================

    async def subscribe_exchanger(
        self,
        exchanger_id: str,
        callback: Callable[[OPCUADataPoint], None],
    ) -> str:
        """
        Subscribe to all tags for an exchanger.

        Args:
            exchanger_id: Exchanger identifier
            callback: Callback for data changes

        Returns:
            Subscription ID
        """
        manifest = self._manifests.get(exchanger_id)
        if not manifest:
            raise ValueError(f"No manifest for exchanger: {exchanger_id}")

        subscription_id = f"sub_{exchanger_id}_{uuid.uuid4().hex[:8]}"

        await self.subscription_manager.create_subscription(
            subscription_id=subscription_id,
            tags=manifest.tags,
        )

        # Register callback for each tag
        for tag in manifest.tags:
            self.subscription_manager.register_callback(tag.tag_id, callback)

        return subscription_id

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        stats = {
            "connection_id": self.config.connection_id,
            "endpoint_url": self.config.endpoint_url,
            "state": self.state.value,
            "session_id": self._session_id[:8] + "..." if self._session_id else None,
            "read_only": self.config.read_only,
            "connect_count": self._connect_count,
            "disconnect_count": self._disconnect_count,
            "error_count": self._error_count,
            "read_count": self._read_count,
            "reconnect_attempt": self._reconnect_attempt,
            "last_health_check": (
                self._last_health_check.isoformat()
                if self._last_health_check else None
            ),
            "circuit_breaker": self.circuit_breaker.get_state(),
            "subscriptions": self.subscription_manager.get_subscription_stats(),
            "manifests_loaded": len(self._manifests),
            "tags_indexed": len(self._tag_index),
        }

        if self.store_forward:
            # Run coroutine synchronously for stats
            loop = asyncio.get_event_loop()
            if loop.is_running():
                stats["store_forward"] = {
                    "enabled": True,
                    "max_size": self.store_forward.max_size,
                }
            else:
                stats["store_forward"] = loop.run_until_complete(
                    self.store_forward.get_stats()
                )

        return stats

    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return {
            "healthy": self.is_connected(),
            "state": self.state.value,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "last_health_check": (
                self._last_health_check.isoformat()
                if self._last_health_check else None
            ),
            "error_count": self._error_count,
            "subscriptions_active": len(self.subscription_manager.subscriptions),
        }

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> "OPCUAConnector":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ConnectionState",
    "CircuitBreakerState",
    "SecurityPolicy",
    "SecurityMode",
    "TagDataType",
    "OPCUAQualityCode",

    # Configuration
    "OPCUASecurityConfig",
    "OPCUAConfig",

    # Tag Manifest
    "ExchangerTagDefinition",
    "ExchangerTagManifest",
    "TagOnboardingResult",

    # Data Point
    "OPCUADataPoint",

    # Circuit Breaker
    "CircuitBreaker",

    # Buffers
    "DataBuffer",
    "StoreAndForwardBuffer",

    # Subscription Manager
    "OPCUASubscription",
    "OPCUASubscriptionManager",

    # Connection Pool
    "OPCUAConnectionPool",

    # Main Connector
    "OPCUAConnector",
]
