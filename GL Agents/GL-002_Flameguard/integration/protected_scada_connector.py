"""
GL-002 FLAMEGUARD - Protected SCADA Connector

This module provides a fault-tolerant wrapper around the SCADA connector
with circuit breaker protection, graceful degradation, and value caching.

Features:
    - Circuit breaker protection for all external calls
    - Last-known-good value caching when breaker opens
    - Fallback strategies for degraded operation
    - Automatic recovery with health monitoring
    - Comprehensive metrics and audit trails

Standards Compliance:
    - IEC 61511 (Functional Safety - fail-safe behavior)
    - ISA-95 (Enterprise-Control Integration)
    - OPC Foundation UA specifications

Example:
    >>> config = SCADAConnectionConfig(protocol=SCADAProtocol.MODBUS_TCP, ...)
    >>> connector = ProtectedSCADAConnector(config)
    >>> await connector.connect()
    >>> values = await connector.read_tags_safe(["drum_pressure", "o2_percent"])
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import hashlib
import logging
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitHalfOpenError,
    CircuitState,
)
from .scada_connector import (
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,
    TagMapping,
    TagValue,
    TagQuality,
    ProtocolHandler,
    ModbusTCPHandler,
    OPCUAHandler,
)

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategies when circuit breaker opens."""
    LAST_KNOWN_VALUE = "last_known_value"  # Return cached value
    SAFE_DEFAULT = "safe_default"          # Return safe default value
    RAISE_ERROR = "raise_error"            # Raise exception
    DEGRADED_MODE = "degraded_mode"        # Enter degraded operation mode


class DegradedModeLevel(Enum):
    """Levels of degraded operation."""
    NORMAL = "normal"           # Full operation
    LIMITED = "limited"         # Reduced polling frequency
    READ_ONLY = "read_only"     # No writes allowed
    CACHED_ONLY = "cached_only" # Only cached values
    OFFLINE = "offline"         # No SCADA access


@dataclass
class CachedValue:
    """Cached tag value with metadata."""
    tag: str
    value: Any
    quality: TagQuality
    timestamp: datetime
    cached_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cache_hits: int = 0
    source: str = "scada"  # "scada", "fallback", "interpolated"

    def is_stale(self, max_age_s: float = 300.0) -> bool:
        """Check if cached value is stale."""
        age = (datetime.now(timezone.utc) - self.cached_at).total_seconds()
        return age > max_age_s

    def to_tag_value(self, mark_as_cached: bool = True) -> TagValue:
        """Convert to TagValue, optionally marking quality as cached."""
        quality = TagQuality.LAST_KNOWN if mark_as_cached else self.quality
        return TagValue(
            tag=self.tag,
            value=self.value,
            quality=quality,
            timestamp=self.timestamp,
        )


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    strategy: FallbackStrategy = FallbackStrategy.LAST_KNOWN_VALUE
    cache_max_age_s: float = 300.0          # Max age for cached values
    safe_defaults: Dict[str, Any] = field(default_factory=dict)
    stale_value_quality: TagQuality = TagQuality.LAST_KNOWN
    degraded_poll_multiplier: float = 5.0   # Slow down polling in degraded mode
    max_cache_size: int = 10000


@dataclass
class ProtectedConnectorMetrics:
    """Metrics for protected SCADA connector."""
    total_reads: int = 0
    successful_reads: int = 0
    failed_reads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_activations: int = 0
    total_writes: int = 0
    successful_writes: int = 0
    failed_writes: int = 0
    connection_attempts: int = 0
    connection_failures: int = 0
    degraded_mode_entries: int = 0
    recovery_successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )
        read_success_rate = (
            self.successful_reads / self.total_reads
            if self.total_reads > 0 else 0.0
        )
        return {
            "total_reads": self.total_reads,
            "successful_reads": self.successful_reads,
            "failed_reads": self.failed_reads,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "read_success_rate": read_success_rate,
            "fallback_activations": self.fallback_activations,
            "total_writes": self.total_writes,
            "successful_writes": self.successful_writes,
            "failed_writes": self.failed_writes,
            "connection_attempts": self.connection_attempts,
            "connection_failures": self.connection_failures,
            "degraded_mode_entries": self.degraded_mode_entries,
            "recovery_successes": self.recovery_successes,
        }


class ProtectedSCADAConnector:
    """
    Fault-tolerant SCADA connector with circuit breaker protection.

    This class wraps the standard SCADAConnector with:
    - Circuit breaker protection for connection, read, and write operations
    - Automatic caching of last-known-good values
    - Configurable fallback strategies
    - Graceful degradation modes
    - Comprehensive metrics and monitoring

    Attributes:
        config: SCADA connection configuration
        fallback_config: Fallback behavior configuration
        degraded_mode: Current degraded mode level
        metrics: Collected metrics

    Example:
        >>> connector = ProtectedSCADAConnector(scada_config)
        >>> await connector.connect()
        >>>
        >>> # Safe read with fallback
        >>> values = await connector.read_tags_safe(["pressure", "temperature"])
        >>>
        >>> # Check breaker status
        >>> status = connector.get_circuit_status()
    """

    def __init__(
        self,
        config: SCADAConnectionConfig,
        fallback_config: Optional[FallbackConfig] = None,
        breaker_config: Optional[CircuitBreakerConfig] = None,
        on_degraded_mode_change: Optional[Callable[[DegradedModeLevel], None]] = None,
        on_circuit_state_change: Optional[Callable[[str, CircuitState], None]] = None,
    ) -> None:
        """
        Initialize ProtectedSCADAConnector.

        Args:
            config: SCADA connection configuration
            fallback_config: Fallback behavior configuration
            breaker_config: Circuit breaker configuration
            on_degraded_mode_change: Callback for degraded mode changes
            on_circuit_state_change: Callback for circuit state changes
        """
        self.config = config
        self.fallback_config = fallback_config or FallbackConfig()
        self.breaker_config = breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout_s=30.0,
            half_open_max_calls=3,
            success_threshold=2,
        )

        # Callbacks
        self._on_degraded_mode_change = on_degraded_mode_change
        self._on_circuit_state_change = on_circuit_state_change

        # Initialize underlying connector
        self._connector = SCADAConnector(
            config,
            on_data_callback=self._on_data_received,
            on_quality_change=self._on_quality_changed,
        )

        # Circuit breakers for different operations
        self._registry = CircuitBreakerRegistry()
        self._init_circuit_breakers()

        # Value cache
        self._value_cache: Dict[str, CachedValue] = {}
        self._cache_lock = asyncio.Lock()

        # State
        self.degraded_mode = DegradedModeLevel.NORMAL
        self.metrics = ProtectedConnectorMetrics()

        # Health check state
        self._last_successful_read: Optional[datetime] = None
        self._last_successful_write: Optional[datetime] = None

        logger.info(
            f"ProtectedSCADAConnector initialized: {config.protocol.value}"
        )

    def _init_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different operations."""
        prefix = f"scada_{self.config.protocol.value}"

        # Connection circuit breaker
        self._connection_breaker = self._registry.get_or_create(
            f"{prefix}_connection",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_s=60.0,
                half_open_max_calls=1,
            ),
            on_state_change=self._handle_breaker_state_change,
        )

        # Read circuit breaker
        self._read_breaker = self._registry.get_or_create(
            f"{prefix}_read",
            config=self.breaker_config,
            on_state_change=self._handle_breaker_state_change,
        )

        # Write circuit breaker
        self._write_breaker = self._registry.get_or_create(
            f"{prefix}_write",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_s=45.0,
                half_open_max_calls=2,
            ),
            on_state_change=self._handle_breaker_state_change,
        )

    def _handle_breaker_state_change(
        self,
        name: str,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        """Handle circuit breaker state changes."""
        logger.info(
            f"Circuit breaker '{name}' state change: "
            f"{old_state.value} -> {new_state.value}"
        )

        if self._on_circuit_state_change:
            try:
                self._on_circuit_state_change(name, new_state)
            except Exception as e:
                logger.error(f"Circuit state change callback failed: {e}")

        # Update degraded mode based on circuit states
        self._update_degraded_mode()

    def _update_degraded_mode(self) -> None:
        """Update degraded mode based on circuit breaker states."""
        old_mode = self.degraded_mode

        # Determine new mode based on breaker states
        if self._connection_breaker.is_open:
            new_mode = DegradedModeLevel.OFFLINE
        elif self._read_breaker.is_open and self._write_breaker.is_open:
            new_mode = DegradedModeLevel.CACHED_ONLY
        elif self._write_breaker.is_open:
            new_mode = DegradedModeLevel.READ_ONLY
        elif self._read_breaker.is_open:
            new_mode = DegradedModeLevel.LIMITED
        elif any(b.is_half_open for b in [
            self._connection_breaker,
            self._read_breaker,
            self._write_breaker,
        ]):
            new_mode = DegradedModeLevel.LIMITED
        else:
            new_mode = DegradedModeLevel.NORMAL

        if new_mode != old_mode:
            self.degraded_mode = new_mode
            self.metrics.degraded_mode_entries += 1

            logger.warning(
                f"Degraded mode changed: {old_mode.value} -> {new_mode.value}"
            )

            if self._on_degraded_mode_change:
                try:
                    self._on_degraded_mode_change(new_mode)
                except Exception as e:
                    logger.error(f"Degraded mode change callback failed: {e}")

    def _on_data_received(self, values: Dict[str, TagValue]) -> None:
        """Handle data received from underlying connector."""
        # Update cache with new values
        asyncio.create_task(self._update_cache(values))

    def _on_quality_changed(self, tag: str, quality: TagQuality) -> None:
        """Handle quality change for a tag."""
        logger.debug(f"Quality changed for {tag}: {quality.value}")

    async def _update_cache(self, values: Dict[str, TagValue]) -> None:
        """Update value cache with new values."""
        async with self._cache_lock:
            for tag, value in values.items():
                if value.quality == TagQuality.GOOD:
                    self._value_cache[tag] = CachedValue(
                        tag=tag,
                        value=value.value,
                        quality=value.quality,
                        timestamp=value.timestamp,
                        source="scada",
                    )

            # Enforce cache size limit
            if len(self._value_cache) > self.fallback_config.max_cache_size:
                # Remove oldest entries
                sorted_entries = sorted(
                    self._value_cache.items(),
                    key=lambda x: x[1].cached_at,
                )
                entries_to_remove = len(self._value_cache) - self.fallback_config.max_cache_size
                for tag, _ in sorted_entries[:entries_to_remove]:
                    del self._value_cache[tag]

    async def _get_cached_value(self, tag: str) -> Optional[CachedValue]:
        """Get cached value for a tag."""
        async with self._cache_lock:
            cached = self._value_cache.get(tag)
            if cached:
                cached.cache_hits += 1
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            return cached

    def _get_fallback_value(
        self,
        tag: str,
        cached: Optional[CachedValue],
    ) -> Optional[TagValue]:
        """Get fallback value based on configured strategy."""
        strategy = self.fallback_config.strategy
        self.metrics.fallback_activations += 1

        if strategy == FallbackStrategy.RAISE_ERROR:
            return None

        if strategy == FallbackStrategy.LAST_KNOWN_VALUE and cached:
            if not cached.is_stale(self.fallback_config.cache_max_age_s):
                return cached.to_tag_value(mark_as_cached=True)

        if strategy in [
            FallbackStrategy.SAFE_DEFAULT,
            FallbackStrategy.LAST_KNOWN_VALUE,
        ]:
            if tag in self.fallback_config.safe_defaults:
                return TagValue(
                    tag=tag,
                    value=self.fallback_config.safe_defaults[tag],
                    quality=TagQuality.UNCERTAIN,
                    timestamp=datetime.now(timezone.utc),
                )

        if strategy == FallbackStrategy.DEGRADED_MODE and cached:
            return cached.to_tag_value(mark_as_cached=True)

        return None

    def add_tag(self, mapping: TagMapping) -> None:
        """Add tag mapping to connector."""
        self._connector.add_tag(mapping)

        # Add safe default if not specified
        if mapping.internal_name not in self.fallback_config.safe_defaults:
            # Use midpoint of limits as safe default
            if mapping.low_limit is not None and mapping.high_limit is not None:
                self.fallback_config.safe_defaults[mapping.internal_name] = (
                    (mapping.low_limit + mapping.high_limit) / 2
                )

    async def connect(self) -> bool:
        """
        Connect to SCADA system with circuit breaker protection.

        Returns:
            True if connection successful
        """
        self.metrics.connection_attempts += 1

        try:
            result = await self._connection_breaker.call(
                self._connector.connect
            )

            if result:
                self.metrics.recovery_successes += 1
                logger.info("SCADA connection established")
            else:
                self.metrics.connection_failures += 1

            return result

        except CircuitOpenError as e:
            logger.warning(f"Connection circuit open: {e}")
            self.metrics.connection_failures += 1
            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.metrics.connection_failures += 1
            return False

    async def disconnect(self) -> None:
        """Disconnect from SCADA system."""
        await self._connector.disconnect()

    async def start_polling(self) -> None:
        """Start polling with circuit breaker protection."""
        if self.degraded_mode == DegradedModeLevel.OFFLINE:
            logger.warning("Cannot start polling in OFFLINE mode")
            return

        # Adjust polling interval based on degraded mode
        if self.degraded_mode == DegradedModeLevel.LIMITED:
            # Slow down polling in limited mode
            original_interval = self.config.poll_interval_ms
            self.config.poll_interval_ms = int(
                original_interval * self.fallback_config.degraded_poll_multiplier
            )

        await self._connector.start_polling()

    async def stop_polling(self) -> None:
        """Stop polling."""
        await self._connector.stop_polling()

    async def read_tag_safe(
        self,
        internal_name: str,
        use_cache: bool = True,
    ) -> Optional[TagValue]:
        """
        Read single tag with circuit breaker and fallback.

        Args:
            internal_name: Internal tag name
            use_cache: Whether to use cached value on failure

        Returns:
            TagValue or None if read failed and no fallback available
        """
        self.metrics.total_reads += 1

        # Check if in cached-only mode
        if self.degraded_mode == DegradedModeLevel.CACHED_ONLY:
            cached = await self._get_cached_value(internal_name)
            if cached:
                return cached.to_tag_value(mark_as_cached=True)
            return self._get_fallback_value(internal_name, None)

        try:
            result = await self._read_breaker.call(
                self._connector.read_tag,
                internal_name,
            )

            if result and result.quality == TagQuality.GOOD:
                self.metrics.successful_reads += 1
                self._last_successful_read = datetime.now(timezone.utc)

                # Update cache
                await self._update_cache({internal_name: result})

                return result

            # Bad quality - try cache/fallback
            self.metrics.failed_reads += 1
            if use_cache:
                cached = await self._get_cached_value(internal_name)
                return self._get_fallback_value(internal_name, cached) or result

            return result

        except CircuitOpenError:
            self.metrics.failed_reads += 1
            if use_cache:
                cached = await self._get_cached_value(internal_name)
                return self._get_fallback_value(internal_name, cached)
            return None

        except Exception as e:
            logger.error(f"Read failed for {internal_name}: {e}")
            self.metrics.failed_reads += 1
            if use_cache:
                cached = await self._get_cached_value(internal_name)
                return self._get_fallback_value(internal_name, cached)
            return None

    async def read_tags_safe(
        self,
        internal_names: List[str],
        use_cache: bool = True,
    ) -> Dict[str, TagValue]:
        """
        Read multiple tags with circuit breaker and fallback.

        Args:
            internal_names: List of internal tag names
            use_cache: Whether to use cached values on failure

        Returns:
            Dictionary of tag names to TagValues
        """
        self.metrics.total_reads += 1

        # Check if in cached-only mode
        if self.degraded_mode == DegradedModeLevel.CACHED_ONLY:
            results = {}
            for name in internal_names:
                cached = await self._get_cached_value(name)
                if cached:
                    results[name] = cached.to_tag_value(mark_as_cached=True)
                else:
                    fallback = self._get_fallback_value(name, None)
                    if fallback:
                        results[name] = fallback
            return results

        try:
            result = await self._read_breaker.call(
                self._connector.read_tags,
                internal_names,
            )

            # Update cache with good values
            good_values = {
                k: v for k, v in result.items()
                if v.quality == TagQuality.GOOD
            }
            if good_values:
                await self._update_cache(good_values)
                self.metrics.successful_reads += 1
                self._last_successful_read = datetime.now(timezone.utc)

            # Fill in missing values from cache/fallback
            if use_cache:
                for name in internal_names:
                    if name not in result or result[name].quality != TagQuality.GOOD:
                        cached = await self._get_cached_value(name)
                        fallback = self._get_fallback_value(name, cached)
                        if fallback:
                            result[name] = fallback

            return result

        except CircuitOpenError:
            self.metrics.failed_reads += 1
            if use_cache:
                results = {}
                for name in internal_names:
                    cached = await self._get_cached_value(name)
                    fallback = self._get_fallback_value(name, cached)
                    if fallback:
                        results[name] = fallback
                return results
            return {}

        except Exception as e:
            logger.error(f"Batch read failed: {e}")
            self.metrics.failed_reads += 1
            if use_cache:
                results = {}
                for name in internal_names:
                    cached = await self._get_cached_value(name)
                    fallback = self._get_fallback_value(name, cached)
                    if fallback:
                        results[name] = fallback
                return results
            return {}

    async def write_tag_safe(
        self,
        internal_name: str,
        value: float,
        verify: bool = True,
    ) -> bool:
        """
        Write tag value with circuit breaker protection.

        Args:
            internal_name: Internal tag name
            value: Value to write
            verify: Whether to verify write by reading back

        Returns:
            True if write successful
        """
        self.metrics.total_writes += 1

        # Check if writes allowed
        if self.degraded_mode in [
            DegradedModeLevel.READ_ONLY,
            DegradedModeLevel.CACHED_ONLY,
            DegradedModeLevel.OFFLINE,
        ]:
            logger.warning(
                f"Write rejected in {self.degraded_mode.value} mode: {internal_name}"
            )
            self.metrics.failed_writes += 1
            return False

        try:
            result = await self._write_breaker.call(
                self._connector.write_tag,
                internal_name,
                value,
                verify,
            )

            if result:
                self.metrics.successful_writes += 1
                self._last_successful_write = datetime.now(timezone.utc)
            else:
                self.metrics.failed_writes += 1

            return result

        except CircuitOpenError:
            logger.warning(f"Write circuit open for {internal_name}")
            self.metrics.failed_writes += 1
            return False

        except Exception as e:
            logger.error(f"Write failed for {internal_name}: {e}")
            self.metrics.failed_writes += 1
            return False

    def is_connected(self) -> bool:
        """Check if connected to SCADA."""
        return self._connector.is_connected()

    def is_healthy(self) -> bool:
        """Check if connector is healthy."""
        return (
            self.degraded_mode == DegradedModeLevel.NORMAL and
            self._connection_breaker.is_closed and
            self._read_breaker.is_closed and
            self._write_breaker.is_closed
        )

    def get_circuit_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            "connection": self._connection_breaker.get_status(),
            "read": self._read_breaker.get_status(),
            "write": self._write_breaker.get_status(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive connector status."""
        return {
            "connected": self.is_connected(),
            "healthy": self.is_healthy(),
            "degraded_mode": self.degraded_mode.value,
            "protocol": self.config.protocol.value,
            "host": self.config.host,
            "port": self.config.port,
            "circuit_breakers": self.get_circuit_status(),
            "metrics": self.metrics.to_dict(),
            "cache_size": len(self._value_cache),
            "last_successful_read": (
                self._last_successful_read.isoformat()
                if self._last_successful_read else None
            ),
            "last_successful_write": (
                self._last_successful_write.isoformat()
                if self._last_successful_write else None
            ),
        }

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status."""
        now = datetime.now(timezone.utc)
        stale_count = sum(
            1 for v in self._value_cache.values()
            if v.is_stale(self.fallback_config.cache_max_age_s)
        )
        return {
            "size": len(self._value_cache),
            "max_size": self.fallback_config.max_cache_size,
            "stale_entries": stale_count,
            "entries": {
                tag: {
                    "value": cv.value,
                    "quality": cv.quality.value,
                    "age_s": (now - cv.cached_at).total_seconds(),
                    "hits": cv.cache_hits,
                    "source": cv.source,
                }
                for tag, cv in self._value_cache.items()
            },
        }

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        self._connection_breaker.reset()
        self._read_breaker.reset()
        self._write_breaker.reset()
        self._update_degraded_mode()
        logger.info("All circuit breakers reset")

    def clear_cache(self) -> None:
        """Clear the value cache."""
        self._value_cache.clear()
        logger.info("Value cache cleared")

    def get_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        state_str = (
            f"{self.degraded_mode.value}|"
            f"{self._connection_breaker.state.value}|"
            f"{self._read_breaker.state.value}|"
            f"{self._write_breaker.state.value}|"
            f"{self.metrics.total_reads}|{self.metrics.total_writes}"
        )
        return hashlib.sha256(state_str.encode()).hexdigest()


class ProtectedModbusClient:
    """
    Protected Modbus TCP client with circuit breaker.

    Provides a simplified interface for Modbus operations with
    automatic circuit breaker protection.
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        timeout_ms: int = 5000,
        breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize protected Modbus client."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host=host,
            port=port,
            unit_id=unit_id,
            timeout_ms=timeout_ms,
        )
        self._connector = ProtectedSCADAConnector(config, breaker_config=breaker_config)

    async def connect(self) -> bool:
        """Connect to Modbus device."""
        return await self._connector.connect()

    async def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        await self._connector.disconnect()

    async def read_registers(
        self,
        tag_mappings: List[TagMapping],
    ) -> Dict[str, TagValue]:
        """Read Modbus registers."""
        for mapping in tag_mappings:
            self._connector.add_tag(mapping)

        names = [m.internal_name for m in tag_mappings]
        return await self._connector.read_tags_safe(names)

    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return self._connector.get_status()


class ProtectedOPCUAClient:
    """
    Protected OPC-UA client with circuit breaker.

    Provides a simplified interface for OPC-UA operations with
    automatic circuit breaker protection.
    """

    def __init__(
        self,
        endpoint_url: str,
        security_policy: str = "None",
        security_mode: str = "None",
        timeout_ms: int = 10000,
        breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize protected OPC-UA client."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="",  # Not used for OPC-UA
            port=0,
            endpoint_url=endpoint_url,
            security_policy=security_policy,
            security_mode=security_mode,
            timeout_ms=timeout_ms,
        )
        self._connector = ProtectedSCADAConnector(config, breaker_config=breaker_config)

    async def connect(self) -> bool:
        """Connect to OPC-UA server."""
        return await self._connector.connect()

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        await self._connector.disconnect()

    async def read_nodes(
        self,
        tag_mappings: List[TagMapping],
    ) -> Dict[str, TagValue]:
        """Read OPC-UA nodes."""
        for mapping in tag_mappings:
            self._connector.add_tag(mapping)

        names = [m.internal_name for m in tag_mappings]
        return await self._connector.read_tags_safe(names)

    async def write_node(
        self,
        internal_name: str,
        value: Any,
    ) -> bool:
        """Write to OPC-UA node."""
        return await self._connector.write_tag_safe(internal_name, value)

    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return self._connector.get_status()
