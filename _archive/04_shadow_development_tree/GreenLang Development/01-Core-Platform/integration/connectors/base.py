# -*- coding: utf-8 -*-
"""
Enhanced Connector Base Classes
================================

Extends the SDK Connector with async support, capabilities, and snapshot management.

Following patterns from:
- greenlang/intelligence/providers/base.py (LLMProvider)
- greenlang/sdk/base.py (original Connector)
- greenlang/intelligence/determinism.py (caching)
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import asyncio
import logging

logger = logging.getLogger(__name__)

# Type variables for generic connector
TQuery = TypeVar('TQuery', bound=BaseModel)
TPayload = TypeVar('TPayload', bound=BaseModel)
TConfig = TypeVar('TConfig', bound=BaseModel)


class ConnectorCapabilities(BaseModel):
    """
    Connector capabilities declaration

    Similar to LLMCapabilities pattern in intelligence/providers/base.py
    """
    supports_streaming: bool = Field(default=False, description="Supports streaming data")
    supports_pagination: bool = Field(default=False, description="Supports paginated queries")
    supports_push: bool = Field(default=False, description="Supports data push/write")
    requires_auth: bool = Field(default=True, description="Requires authentication")
    rate_limit_per_hour: Optional[int] = Field(default=None, description="Rate limit (requests/hour)")
    max_batch_size: int = Field(default=100, description="Maximum batch size")
    supports_time_series: bool = Field(default=False, description="Supports time-series data")
    min_resolution: Optional[str] = Field(default=None, description="Minimum time resolution (e.g., 'hour', 'day')")

    class Config:
        frozen = True


class ConnectorProvenance(BaseModel):
    """
    Connector provenance metadata for data lineage and auditing

    Required for compliance and reproducibility
    """
    connector_id: str = Field(..., description="Unique connector identifier")
    connector_version: str = Field(..., description="Connector version (semver)")
    mode: str = Field(..., description="Execution mode (record/replay/golden)")
    query_hash: str = Field(..., description="SHA-256 hash of canonical query")
    schema_hash: str = Field(..., description="SHA-256 hash of payload schema")
    seed: Optional[str] = Field(default=None, description="Deterministic seed (hex)")
    snapshot_id: Optional[str] = Field(default=None, description="SHA-256 hash of snapshot")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Connector(ABC, Generic[TQuery, TPayload, TConfig]):
    """
    Enhanced Base Connector with async support and capabilities

    Key enhancements over SDK version:
    - Generic type parameters for type safety
    - Async-first interface (fetch instead of read)
    - Capabilities property (following LLMProvider pattern)
    - Snapshot/restore for deterministic replay
    - Provenance tracking for auditing
    - Context-aware execution (ConnectorContext)

    Backward compatibility:
    - Maintains context manager support (__enter__/__exit__)
    - Provides sync wrappers for legacy code
    - Can be used as drop-in replacement for SDK Connector

    Example:
        class MyConnector(Connector[MyQuery, MyPayload, MyConfig]):
            connector_id = "my-connector"
            connector_version = "1.0.0"

            @property
            def capabilities(self) -> ConnectorCapabilities:
                return ConnectorCapabilities(
                    supports_time_series=True,
                    min_resolution="hour"
                )

            async def fetch(self, query: MyQuery, ctx: ConnectorContext):
                # Implementation
                payload = MyPayload(...)
                prov = ConnectorProvenance(...)
                return payload, prov
    """

    # Class attributes (must be set by subclass)
    connector_id: str
    connector_version: str = "0.1.0"

    def __init__(self, config: Optional[TConfig] = None):
        """Initialize connector with configuration"""
        self.config = config
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.{self.connector_id}")

    @property
    @abstractmethod
    def capabilities(self) -> ConnectorCapabilities:
        """
        Declare connector capabilities

        Must be implemented by subclass to describe what the connector can do.
        Used for validation and capability negotiation.

        Returns:
            ConnectorCapabilities with supported features
        """

    @abstractmethod
    async def fetch(
        self,
        query: TQuery,
        ctx: 'ConnectorContext'
    ) -> Tuple[TPayload, ConnectorProvenance]:
        """
        Fetch data from source (async)

        This is the primary data retrieval method. Must be implemented by all connectors.

        Args:
            query: Typed query specification (Pydantic model)
            ctx: Connector context (mode, cache settings, security)

        Returns:
            Tuple of (payload, provenance)
            - payload: Typed data payload (Pydantic model)
            - provenance: Metadata for auditing and lineage

        Raises:
            ConnectorReplayRequired: If in replay mode without snapshot
            ConnectorNetworkError: If network request fails
            ConnectorAuthError: If authentication fails
        """

    def snapshot(self, payload: TPayload, prov: ConnectorProvenance) -> bytes:
        """
        Create canonical snapshot for replay

        Uses existing snapshot infrastructure from greenlang/connectors/snapshot.py

        Args:
            payload: Data payload to snapshot
            prov: Provenance metadata

        Returns:
            Canonical bytes (deterministic, byte-exact)
        """
        from greenlang.connectors.snapshot import write_canonical_snapshot
        return write_canonical_snapshot(
            connector_id=self.connector_id,
            connector_version=self.connector_version,
            payload=payload,
            provenance=prov
        )

    def restore(self, raw: bytes) -> Tuple[TPayload, ConnectorProvenance]:
        """
        Restore payload and provenance from snapshot

        Args:
            raw: Snapshot bytes

        Returns:
            Tuple of (payload, provenance)

        Raises:
            ConnectorError: If snapshot is invalid or corrupted
        """
        from greenlang.connectors.snapshot import read_canonical_snapshot
        return read_canonical_snapshot(raw, payload_type=type(self).__orig_bases__[0].__args__[1])

    # Sync wrapper for backward compatibility
    def fetch_sync(
        self,
        query: TQuery,
        ctx: 'ConnectorContext'
    ) -> Tuple[TPayload, ConnectorProvenance]:
        """
        Synchronous wrapper for fetch() - for legacy code

        Args:
            query: Query specification
            ctx: Connector context

        Returns:
            Tuple of (payload, provenance)
        """
        return asyncio.run(self.fetch(query, ctx))

    # Context manager support (SDK compatibility)
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def connect(self) -> bool:
        """
        Establish connection (optional, for stateful connectors)

        Returns:
            True if connected successfully
        """
        self.connected = True
        return True

    def disconnect(self) -> bool:
        """
        Close connection (optional, for stateful connectors)

        Returns:
            True if disconnected successfully
        """
        self.connected = False
        return True

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connected

    # Legacy SDK method compatibility (deprecated, use fetch)
    def read(self, query: Any) -> Any:
        """
        Legacy read method - deprecated, use fetch() instead

        Provided for SDK compatibility only.
        """
        from greenlang.sdk.base import Result
        self.logger.warning(
            f"Connector.read() is deprecated, use fetch() instead. "
            f"Called from {self.connector_id}"
        )
        try:
            # Import here to avoid circular dependency
            from greenlang.connectors.context import ConnectorContext, CacheMode
            ctx = ConnectorContext(
                mode=CacheMode.RECORD,
                connector_id=self.connector_id
            )
            payload, prov = self.fetch_sync(query, ctx)
            return Result(success=True, data=payload, metadata=prov.dict())
        except Exception as e:
            return Result(success=False, error=str(e))

    def write(self, data: Any) -> Any:
        """
        Legacy write method - not implemented

        For push-capable connectors, implement a separate push() method.
        """
        from greenlang.sdk.base import Result
        return Result(
            success=False,
            error="write() not supported, use connector-specific push() if available"
        )


class StreamConnector(Connector[TQuery, TPayload, TConfig]):
    """
    Base class for streaming connectors

    For connectors that provide real-time or incremental data streams.
    Not needed for W1 (DATA-301), but provides placeholder for future.
    """

    @abstractmethod
    async def stream(
        self,
        query: TQuery,
        ctx: 'ConnectorContext'
    ) -> Any:  # AsyncIterator[TPayload] in Python 3.9+
        """
        Stream data incrementally

        Args:
            query: Query specification
            ctx: Connector context

        Yields:
            Individual payload items
        """
        raise NotImplementedError("Streaming not implemented in W1")
