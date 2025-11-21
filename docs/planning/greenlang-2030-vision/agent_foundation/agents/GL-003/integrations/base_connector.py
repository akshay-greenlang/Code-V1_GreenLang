# -*- coding: utf-8 -*-
"""
Base Connector Abstract Class for GL-003 SteamSystemAnalyzer

Provides common infrastructure for all integration connectors including:
- Connection lifecycle management
- Retry logic with exponential backoff
- Circuit breaker pattern
- Health checking
- Metrics integration
- Thread-safe operations
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import time
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"


class CircuitState(Enum):
    """Circuit breaker state."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit broken, fail fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ConnectionConfig:
    """Base configuration for connectors."""
    host: str
    port: int
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: int = 300
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 60
    health_check_interval_seconds: int = 30
    connection_pool_size: int = 5
    enable_metrics: bool = True


@dataclass
class HealthStatus:
    """Health check status."""
    is_healthy: bool
    state: ConnectionState
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when error threshold is reached.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        async with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self.last_failure_time:
                    elapsed = (DeterministicClock.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.timeout_seconds:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker is OPEN. Retry after {self.timeout_seconds - elapsed:.0f}s"
                        )

            # In HALF_OPEN, limit number of calls
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN call limit reached")
                self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Success - reset or close circuit
            async with self._lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    logger.info("Circuit breaker closed after successful recovery")
                self.failure_count = 0

            return result

        except Exception as e:
            # Failure - increment count and potentially open circuit
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = DeterministicClock.utcnow()

                if self.state == CircuitState.HALF_OPEN:
                    # Failed during recovery, reopen circuit
                    self.state = CircuitState.OPEN
                    logger.warning("Circuit breaker reopened after recovery failure")
                elif self.failure_count >= self.failure_threshold:
                    # Threshold exceeded, open circuit
                    self.state = CircuitState.OPEN
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise

    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        logger.info("Circuit breaker manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class BaseConnector(ABC):
    """
    Abstract base class for all integration connectors.

    Provides common functionality:
    - Connection management
    - Retry logic
    - Circuit breaker
    - Health checks
    - Metrics collection
    """

    def __init__(self, config: ConnectionConfig):
        """
        Initialize base connector.

        Args:
            config: Connection configuration
        """
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.connection = None
        self.circuit_breaker = None

        if config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_failure_threshold,
                timeout_seconds=config.circuit_timeout_seconds
            )

        self.health_status = HealthStatus(
            is_healthy=False,
            state=ConnectionState.DISCONNECTED,
            last_check=DeterministicClock.utcnow()
        )

        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_retries': 0,
            'avg_response_time_ms': 0.0,
            'last_call_time': None
        }

        self._health_check_task = None
        self._reconnect_task = None
        self._response_times = deque(maxlen=100)

    @abstractmethod
    async def _connect_impl(self) -> bool:
        """
        Implement actual connection logic.

        Must be implemented by subclasses.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    async def _disconnect_impl(self):
        """
        Implement actual disconnection logic.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """
        Implement health check logic.

        Must be implemented by subclasses.

        Returns:
            True if healthy
        """
        pass

    async def connect(self) -> bool:
        """
        Connect to remote system with retry logic.

        Returns:
            True if connected successfully
        """
        if self.state == ConnectionState.CONNECTED:
            logger.info("Already connected")
            return True

        self.state = ConnectionState.CONNECTING

        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{self.config.max_retries}")

                # Call implementation
                success = await asyncio.wait_for(
                    self._connect_impl(),
                    timeout=self.config.timeout_seconds
                )

                if success:
                    self.state = ConnectionState.CONNECTED
                    self.health_status.is_healthy = True
                    self.health_status.state = ConnectionState.CONNECTED

                    # Start health check loop
                    if self.config.health_check_interval_seconds > 0:
                        self._health_check_task = asyncio.create_task(
                            self._health_check_loop()
                        )

                    logger.info("Successfully connected")
                    return True

            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")

            # Calculate retry delay with exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = min(
                    self.config.retry_delay_seconds * (self.config.retry_backoff_multiplier ** attempt),
                    self.config.max_retry_delay_seconds
                )
                logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
                self.metrics['total_retries'] += 1

        # All attempts failed
        self.state = ConnectionState.FAILED
        self.health_status.is_healthy = False
        self.health_status.state = ConnectionState.FAILED

        logger.error("Failed to connect after all retry attempts")
        return False

    async def disconnect(self):
        """Disconnect from remote system."""
        if self.state == ConnectionState.DISCONNECTED:
            return

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

        # Cancel reconnect task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        # Call implementation
        try:
            await self._disconnect_impl()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        self.state = ConnectionState.DISCONNECTED
        self.health_status.is_healthy = False
        self.health_status.state = ConnectionState.DISCONNECTED
        self.connection = None

        logger.info("Disconnected")

    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def health_check(self) -> HealthStatus:
        """
        Perform health check.

        Returns:
            Current health status
        """
        start_time = time.time()

        try:
            is_healthy = await asyncio.wait_for(
                self._health_check_impl(),
                timeout=self.config.timeout_seconds
            )

            response_time = (time.time() - start_time) * 1000  # ms

            self.health_status = HealthStatus(
                is_healthy=is_healthy,
                state=self.state,
                last_check=DeterministicClock.utcnow(),
                consecutive_failures=0 if is_healthy else self.health_status.consecutive_failures + 1,
                last_error=None if is_healthy else "Health check failed",
                response_time_ms=response_time
            )

            if not is_healthy and self.state == ConnectionState.CONNECTED:
                logger.warning("Health check failed, initiating reconnection")
                self._reconnect_task = asyncio.create_task(self._reconnect())

        except Exception as e:
            response_time = (time.time() - start_time) * 1000

            self.health_status = HealthStatus(
                is_healthy=False,
                state=self.state,
                last_check=DeterministicClock.utcnow(),
                consecutive_failures=self.health_status.consecutive_failures + 1,
                last_error=str(e),
                response_time_ms=response_time
            )

            logger.error(f"Health check error: {e}")

        return self.health_status

    async def _reconnect(self):
        """Attempt to reconnect after connection loss."""
        if self.state == ConnectionState.RECONNECTING:
            return

        self.state = ConnectionState.RECONNECTING
        logger.info("Attempting reconnection...")

        # Disconnect first
        try:
            await self._disconnect_impl()
        except Exception:
            pass

        # Reconnect
        success = await self.connect()

        if not success:
            self.state = ConnectionState.FAILED
            logger.error("Reconnection failed")

    async def execute_with_retry(self, func, *args, **kwargs):
        """
        Execute function with retry logic and circuit breaker.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        start_time = time.time()
        self.metrics['total_calls'] += 1

        # Check circuit breaker
        if self.circuit_breaker:
            try:
                result = await self.circuit_breaker.call(func, *args, **kwargs)

                # Track success metrics
                response_time = (time.time() - start_time) * 1000
                self._response_times.append(response_time)
                self.metrics['successful_calls'] += 1
                self.metrics['avg_response_time_ms'] = sum(self._response_times) / len(self._response_times)
                self.metrics['last_call_time'] = DeterministicClock.utcnow()

                return result

            except CircuitBreakerOpenError:
                self.metrics['failed_calls'] += 1
                raise

        # Execute with retry
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )

                # Track success metrics
                response_time = (time.time() - start_time) * 1000
                self._response_times.append(response_time)
                self.metrics['successful_calls'] += 1
                self.metrics['avg_response_time_ms'] = sum(self._response_times) / len(self._response_times)
                self.metrics['last_call_time'] = DeterministicClock.utcnow()

                return result

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    delay = min(
                        self.config.retry_delay_seconds * (self.config.retry_backoff_multiplier ** attempt),
                        self.config.max_retry_delay_seconds
                    )
                    await asyncio.sleep(delay)
                    self.metrics['total_retries'] += 1

        # All retries failed
        self.metrics['failed_calls'] += 1
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connector metrics.

        Returns:
            Metrics dictionary
        """
        return {
            **self.metrics,
            'state': self.state.value,
            'health': {
                'is_healthy': self.health_status.is_healthy,
                'consecutive_failures': self.health_status.consecutive_failures,
                'last_check': self.health_status.last_check.isoformat(),
                'response_time_ms': self.health_status.response_time_ms
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.state.value if self.circuit_breaker else None,
                'failure_count': self.circuit_breaker.failure_count if self.circuit_breaker else 0
            } if self.circuit_breaker else None
        }
