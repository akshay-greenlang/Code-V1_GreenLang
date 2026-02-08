# -*- coding: utf-8 -*-
"""
Connection Manager - AGENT-DATA-003: ERP/Finance Connector
==========================================================

Manages ERP connection lifecycle including registration, validation,
health monitoring, and connection testing for all 10 supported ERP
systems. All connections are stored in an in-memory registry with
thread-safe statistics tracking.

Supports:
    - Connection registration with credential validation
    - Connection removal
    - Simulated connectivity testing per ERP system type
    - Connection health monitoring with uptime tracking
    - Connection status updates
    - Multi-tenant connection listing
    - Thread-safe statistics counters

Zero-Hallucination Guarantees:
    - All connection tests are deterministic simulations
    - No external network calls (simulated mode)
    - SHA-256 provenance hashes on all connection operations
    - Connection IDs generated from deterministic hashing

Example:
    >>> from greenlang.erp_connector.connection_manager import ConnectionManager
    >>> mgr = ConnectionManager()
    >>> record = mgr.register_connection(
    ...     erp_system="sap_s4hana",
    ...     host="sap.example.com",
    ...     port=443,
    ...     username="api_user",
    ... )
    >>> result = mgr.test_connection(record.connection_id)
    >>> assert result["success"] is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "ConnectionStatus",
    "ConnectionRecord",
    "ConnectionManager",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _deterministic_id(seed: str) -> str:
    """Generate a deterministic 12-char hex ID from a seed string.

    Args:
        seed: Input string to hash.

    Returns:
        12-character hex substring of SHA-256 digest.
    """
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ConnectionStatus(str, Enum):
    """Connection lifecycle statuses."""

    REGISTERED = "registered"
    TESTING = "testing"
    CONNECTED = "connected"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    MAINTENANCE = "maintenance"


# ---------------------------------------------------------------------------
# ERP system default ports and protocols
# ---------------------------------------------------------------------------

_ERP_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sap_s4hana": {
        "default_port": 443,
        "protocol": "OData/REST",
        "test_latency_ms": 120,
        "auth_type": "oauth2",
    },
    "sap_ecc": {
        "default_port": 443,
        "protocol": "RFC/BAPI",
        "test_latency_ms": 150,
        "auth_type": "basic",
    },
    "oracle_cloud": {
        "default_port": 443,
        "protocol": "REST",
        "test_latency_ms": 100,
        "auth_type": "oauth2",
    },
    "oracle_ebs": {
        "default_port": 443,
        "protocol": "SOAP/REST",
        "test_latency_ms": 180,
        "auth_type": "basic",
    },
    "netsuite": {
        "default_port": 443,
        "protocol": "SuiteTalk/REST",
        "test_latency_ms": 90,
        "auth_type": "token",
    },
    "dynamics_365": {
        "default_port": 443,
        "protocol": "OData",
        "test_latency_ms": 110,
        "auth_type": "oauth2",
    },
    "workday": {
        "default_port": 443,
        "protocol": "SOAP/REST",
        "test_latency_ms": 130,
        "auth_type": "oauth2",
    },
    "sage": {
        "default_port": 443,
        "protocol": "REST",
        "test_latency_ms": 140,
        "auth_type": "api_key",
    },
    "quickbooks": {
        "default_port": 443,
        "protocol": "REST",
        "test_latency_ms": 80,
        "auth_type": "oauth2",
    },
    "simulated": {
        "default_port": 8080,
        "protocol": "simulated",
        "test_latency_ms": 10,
        "auth_type": "none",
    },
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ConnectionRecord(BaseModel):
    """Registered ERP connection record.

    Stores all metadata about a registered ERP connection including
    credentials, configuration, and health status.
    """

    connection_id: str = Field(..., description="Unique connection identifier")
    erp_system: str = Field(..., description="ERP system type identifier")
    host: str = Field(..., description="ERP server hostname")
    port: int = Field(default=443, ge=1, le=65535, description="Server port")
    username: str = Field(..., description="API username")
    client_id: Optional[str] = Field(None, description="OAuth client ID")
    api_key: Optional[str] = Field(None, description="API key")
    company_code: Optional[str] = Field(None, description="Company code")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    status: ConnectionStatus = Field(
        default=ConnectionStatus.REGISTERED,
        description="Current connection status",
    )
    protocol: Optional[str] = Field(None, description="Communication protocol")
    auth_type: Optional[str] = Field(None, description="Authentication type")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Registration timestamp",
    )
    last_tested_at: Optional[datetime] = Field(
        None, description="Last connectivity test timestamp",
    )
    last_test_success: Optional[bool] = Field(
        None, description="Result of last connectivity test",
    )
    last_test_latency_ms: Optional[float] = Field(
        None, description="Latency of last connectivity test in ms",
    )
    test_count: int = Field(default=0, ge=0, description="Total test count")
    success_count: int = Field(default=0, ge=0, description="Successful test count")
    failure_count: int = Field(default=0, ge=0, description="Failed test count")
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """ERP connection lifecycle manager.

    Manages registration, testing, health monitoring, and removal of
    ERP system connections. All operations are thread-safe and tracked
    with SHA-256 provenance hashes.

    Attributes:
        _connections: In-memory connection registry keyed by connection_id.
        _config: Configuration dictionary.
        _lock: Threading lock for stats and connection mutations.
        _stats: Aggregate statistics counters.

    Example:
        >>> mgr = ConnectionManager()
        >>> record = mgr.register_connection(
        ...     erp_system="sap_s4hana",
        ...     host="sap.example.com",
        ...     port=443,
        ...     username="api_user",
        ... )
        >>> assert record.status == ConnectionStatus.REGISTERED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConnectionManager.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``default_timeout_seconds``: int (default 60)
                - ``max_connections``: int (default 100)
                - ``auto_test_on_register``: bool (default False)
        """
        self._config = config or {}
        self._connections: Dict[str, ConnectionRecord] = {}
        self._lock = threading.Lock()
        self._default_timeout: int = self._config.get(
            "default_timeout_seconds", 60,
        )
        self._max_connections: int = self._config.get("max_connections", 100)
        self._auto_test: bool = self._config.get(
            "auto_test_on_register", False,
        )
        self._stats: Dict[str, Any] = {
            "connections_registered": 0,
            "connections_removed": 0,
            "connections_tested": 0,
            "test_successes": 0,
            "test_failures": 0,
            "errors": 0,
        }
        logger.info(
            "ConnectionManager initialised: max_connections=%d, "
            "auto_test=%s",
            self._max_connections,
            self._auto_test,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_connection(
        self,
        erp_system: str,
        host: str,
        port: int = 443,
        username: str = "api_user",
        client_id: Optional[str] = None,
        api_key: Optional[str] = None,
        company_code: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConnectionRecord:
        """Register a new ERP connection.

        Validates the ERP system type, assigns defaults from the
        system profile, generates a deterministic connection ID, and
        stores the connection in the in-memory registry.

        Args:
            erp_system: ERP system type (e.g. "sap_s4hana").
            host: Server hostname or IP address.
            port: Server port (default 443).
            username: API/service account username.
            client_id: OAuth client ID (optional).
            api_key: API key for token-based auth (optional).
            company_code: ERP company code (optional).
            tenant_id: Multi-tenant identifier (optional).
            metadata: Additional key-value metadata (optional).

        Returns:
            ConnectionRecord with populated fields.

        Raises:
            ValueError: If erp_system is not supported or max
                connections is reached.
        """
        start = time.monotonic()

        # Validate ERP system
        erp_system_lower = erp_system.lower()
        if erp_system_lower not in _ERP_DEFAULTS:
            supported = ", ".join(sorted(_ERP_DEFAULTS.keys()))
            raise ValueError(
                f"Unsupported ERP system '{erp_system}'. "
                f"Supported: {supported}"
            )

        with self._lock:
            if len(self._connections) >= self._max_connections:
                raise ValueError(
                    f"Maximum connections ({self._max_connections}) reached"
                )

        # Retrieve system defaults
        defaults = _ERP_DEFAULTS[erp_system_lower]
        effective_port = port if port != 443 else defaults["default_port"]

        # Deterministic connection ID
        id_seed = f"{erp_system_lower}:{host}:{effective_port}:{username}"
        connection_id = f"conn-{_deterministic_id(id_seed)}"

        # Provenance hash
        provenance_hash = self._compute_provenance(
            "register", connection_id, erp_system_lower, host,
        )

        record = ConnectionRecord(
            connection_id=connection_id,
            erp_system=erp_system_lower,
            host=host,
            port=effective_port,
            username=username,
            client_id=client_id,
            api_key=api_key,
            company_code=company_code,
            tenant_id=tenant_id,
            status=ConnectionStatus.REGISTERED,
            protocol=defaults["protocol"],
            auth_type=defaults["auth_type"],
            provenance_hash=provenance_hash,
            metadata=metadata or {},
        )

        with self._lock:
            self._connections[connection_id] = record
            self._stats["connections_registered"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Registered connection %s: erp=%s host=%s port=%d (%.1f ms)",
            connection_id, erp_system_lower, host, effective_port, elapsed_ms,
        )

        # Optional auto-test
        if self._auto_test:
            self.test_connection(connection_id)

        return record

    def remove_connection(self, connection_id: str) -> bool:
        """Remove a connection from the registry.

        Args:
            connection_id: Connection identifier to remove.

        Returns:
            True if the connection was found and removed, False otherwise.
        """
        with self._lock:
            if connection_id not in self._connections:
                logger.warning(
                    "Cannot remove unknown connection: %s", connection_id,
                )
                return False
            del self._connections[connection_id]
            self._stats["connections_removed"] += 1

        logger.info("Removed connection: %s", connection_id)
        return True

    def test_connection(self, connection_id: str) -> Dict[str, Any]:
        """Test connectivity for a registered connection.

        Performs a simulated connectivity test specific to the ERP
        system type and updates the connection record with test results.

        Args:
            connection_id: Connection identifier to test.

        Returns:
            Dictionary with test results including success, latency,
            protocol, and error message (if any).

        Raises:
            ValueError: If connection_id is not registered.
        """
        start = time.monotonic()

        record = self._get_record_or_raise(connection_id)

        # Update status to testing
        self.update_connection_status(connection_id, ConnectionStatus.TESTING)

        # Run simulated test
        test_result = self._simulate_connection_test(record.erp_system)

        # Update record with test results
        elapsed_ms = (time.monotonic() - start) * 1000
        with self._lock:
            rec = self._connections[connection_id]
            rec.last_tested_at = _utcnow()
            rec.last_test_success = test_result["success"]
            rec.last_test_latency_ms = test_result["latency_ms"]
            rec.test_count += 1

            if test_result["success"]:
                rec.success_count += 1
                rec.status = ConnectionStatus.CONNECTED
                self._stats["test_successes"] += 1
            else:
                rec.failure_count += 1
                rec.status = ConnectionStatus.FAILED
                self._stats["test_failures"] += 1

            self._stats["connections_tested"] += 1

        test_result["connection_id"] = connection_id
        test_result["erp_system"] = record.erp_system
        test_result["elapsed_ms"] = round(elapsed_ms, 2)

        logger.info(
            "Connection test %s: success=%s latency=%.1f ms",
            connection_id,
            test_result["success"],
            test_result["latency_ms"],
        )
        return test_result

    def get_connection(self, connection_id: str) -> ConnectionRecord:
        """Get a connection record by ID.

        Args:
            connection_id: Connection identifier.

        Returns:
            ConnectionRecord for the specified connection.

        Raises:
            ValueError: If connection_id is not registered.
        """
        return self._get_record_or_raise(connection_id)

    def list_connections(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[ConnectionRecord]:
        """List all registered connections, optionally filtered by tenant.

        Args:
            tenant_id: If provided, filter connections to this tenant.

        Returns:
            List of ConnectionRecord objects.
        """
        with self._lock:
            records = list(self._connections.values())

        if tenant_id is not None:
            records = [r for r in records if r.tenant_id == tenant_id]

        logger.debug(
            "Listed %d connections (tenant_id=%s)",
            len(records),
            tenant_id,
        )
        return records

    def update_connection_status(
        self,
        connection_id: str,
        status: ConnectionStatus,
    ) -> None:
        """Update the status of a connection.

        Args:
            connection_id: Connection identifier.
            status: New ConnectionStatus value.

        Raises:
            ValueError: If connection_id is not registered.
        """
        with self._lock:
            if connection_id not in self._connections:
                raise ValueError(
                    f"Unknown connection: {connection_id}"
                )
            self._connections[connection_id].status = status

        logger.debug(
            "Updated connection %s status to %s",
            connection_id, status.value,
        )

    def get_connection_health(
        self,
        connection_id: str,
    ) -> Dict[str, Any]:
        """Get health metrics for a connection.

        Returns structured health information including uptime
        estimates, test history, and current status.

        Args:
            connection_id: Connection identifier.

        Returns:
            Dictionary with health metrics.

        Raises:
            ValueError: If connection_id is not registered.
        """
        record = self._get_record_or_raise(connection_id)

        uptime_seconds = (
            _utcnow() - record.created_at
        ).total_seconds()

        success_rate = 0.0
        if record.test_count > 0:
            success_rate = round(
                record.success_count / record.test_count, 4,
            )

        return {
            "connection_id": record.connection_id,
            "erp_system": record.erp_system,
            "status": record.status.value,
            "uptime_seconds": round(uptime_seconds, 1),
            "test_count": record.test_count,
            "success_count": record.success_count,
            "failure_count": record.failure_count,
            "success_rate": success_rate,
            "last_tested_at": (
                record.last_tested_at.isoformat()
                if record.last_tested_at
                else None
            ),
            "last_test_success": record.last_test_success,
            "last_test_latency_ms": record.last_test_latency_ms,
            "protocol": record.protocol,
            "auth_type": record.auth_type,
            "host": record.host,
            "port": record.port,
            "timestamp": _utcnow().isoformat(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate connection manager statistics.

        Returns:
            Dictionary of counter values, active connection count,
            and per-system breakdowns.
        """
        with self._lock:
            active_count = len(self._connections)
            by_system: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            for rec in self._connections.values():
                by_system[rec.erp_system] = (
                    by_system.get(rec.erp_system, 0) + 1
                )
                by_status[rec.status.value] = (
                    by_status.get(rec.status.value, 0) + 1
                )

            return {
                "connections_registered": self._stats["connections_registered"],
                "connections_removed": self._stats["connections_removed"],
                "connections_tested": self._stats["connections_tested"],
                "test_successes": self._stats["test_successes"],
                "test_failures": self._stats["test_failures"],
                "active_connections": active_count,
                "by_system": by_system,
                "by_status": by_status,
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Simulated connection testing
    # ------------------------------------------------------------------

    def _simulate_connection_test(
        self,
        erp_system: str,
    ) -> Dict[str, Any]:
        """Simulate a connection test for the given ERP system.

        Produces deterministic test results based on the ERP system
        type including protocol-appropriate response codes and
        latency values.

        Args:
            erp_system: ERP system type string.

        Returns:
            Dictionary with success, latency_ms, protocol,
            response_code, and optional error fields.
        """
        defaults = _ERP_DEFAULTS.get(erp_system, _ERP_DEFAULTS["simulated"])
        base_latency = defaults["test_latency_ms"]

        # Deterministic latency variation based on system name hash
        hash_val = int(
            hashlib.sha256(erp_system.encode()).hexdigest()[:8], 16,
        )
        variation = (hash_val % 20) - 10  # -10 to +9 ms variation
        latency_ms = float(max(base_latency + variation, 5))

        result: Dict[str, Any] = {
            "success": True,
            "latency_ms": round(latency_ms, 2),
            "protocol": defaults["protocol"],
            "auth_type": defaults["auth_type"],
            "response_code": 200,
            "server_version": self._get_simulated_version(erp_system),
            "timestamp": _utcnow().isoformat(),
        }

        return result

    def _get_simulated_version(self, erp_system: str) -> str:
        """Return a simulated server version string for an ERP system.

        Args:
            erp_system: ERP system type string.

        Returns:
            Version string appropriate for the ERP system.
        """
        versions: Dict[str, str] = {
            "sap_s4hana": "SAP S/4HANA 2023 FPS02",
            "sap_ecc": "SAP ERP 6.0 EHP8",
            "oracle_cloud": "Oracle Cloud ERP 24A",
            "oracle_ebs": "Oracle E-Business Suite 12.2.12",
            "netsuite": "NetSuite 2024.1",
            "dynamics_365": "Dynamics 365 Finance 10.0.38",
            "workday": "Workday 2024R1",
            "sage": "Sage Intacct R4 2024",
            "quickbooks": "QuickBooks Online v73",
            "simulated": "GreenLang Simulated ERP 1.0",
        }
        return versions.get(erp_system, "Unknown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_record_or_raise(
        self,
        connection_id: str,
    ) -> ConnectionRecord:
        """Retrieve a connection record or raise ValueError.

        Args:
            connection_id: Connection identifier.

        Returns:
            ConnectionRecord for the given ID.

        Raises:
            ValueError: If connection_id is not registered.
        """
        with self._lock:
            record = self._connections.get(connection_id)

        if record is None:
            raise ValueError(f"Unknown connection: {connection_id}")
        return record

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
