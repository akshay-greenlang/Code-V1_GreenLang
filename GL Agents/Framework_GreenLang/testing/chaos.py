"""
GreenLang Framework - Chaos Testing Module

Chaos engineering utilities for resilience testing of GreenLang agents.

Provides:
- Network failure simulation
- Timeout injection
- Resource exhaustion tests
- Latency injection
- Fault injection
- Recovery testing

Target: Verify agents gracefully handle failures and maintain data integrity.

Author: GreenLang QA Team
Version: 1.0.0
"""

import asyncio
import contextlib
import functools
import gc
import os
import random
import signal
import socket
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest.mock import MagicMock, patch

import numpy as np


# =============================================================================
# Type Variables and Enums
# =============================================================================

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ChaosType(Enum):
    """Types of chaos that can be injected."""
    NETWORK_FAILURE = "network_failure"
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_STRESS = "cpu_stress"
    DISK_FULL = "disk_full"
    PROCESS_CRASH = "process_crash"
    EXCEPTION_INJECTION = "exception_injection"
    DATA_CORRUPTION = "data_corruption"
    CLOCK_SKEW = "clock_skew"


class SeverityLevel(Enum):
    """Severity level of chaos injection."""
    LOW = "low"          # Occasional failures, quick recovery
    MEDIUM = "medium"    # Regular failures, moderate recovery
    HIGH = "high"        # Frequent failures, slow recovery
    EXTREME = "extreme"  # Persistent failures


@dataclass
class ChaosConfig:
    """Configuration for chaos injection."""
    chaos_type: ChaosType
    severity: SeverityLevel = SeverityLevel.MEDIUM
    duration_seconds: float = 10.0
    probability: float = 0.5  # Probability of failure occurring
    recovery_time_seconds: float = 1.0
    target_functions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosEvent:
    """Record of a chaos event that occurred."""
    timestamp: datetime
    chaos_type: ChaosType
    severity: SeverityLevel
    target: str
    details: str
    recovered: bool = False
    recovery_time_seconds: Optional[float] = None


# =============================================================================
# Base Classes
# =============================================================================

class ChaosInjector(ABC):
    """Abstract base class for chaos injectors."""

    def __init__(self, config: ChaosConfig):
        """Initialize chaos injector with configuration."""
        self.config = config
        self.events: List[ChaosEvent] = []
        self.is_active = False
        self._start_time: Optional[datetime] = None

    @abstractmethod
    def inject(self) -> None:
        """Inject chaos."""
        pass

    @abstractmethod
    def recover(self) -> None:
        """Recover from chaos."""
        pass

    def record_event(
        self,
        target: str,
        details: str,
        recovered: bool = False,
        recovery_time: Optional[float] = None,
    ) -> ChaosEvent:
        """Record a chaos event."""
        event = ChaosEvent(
            timestamp=datetime.now(timezone.utc),
            chaos_type=self.config.chaos_type,
            severity=self.config.severity,
            target=target,
            details=details,
            recovered=recovered,
            recovery_time_seconds=recovery_time,
        )
        self.events.append(event)
        return event

    def should_trigger(self) -> bool:
        """Determine if chaos should trigger based on probability."""
        return random.random() < self.config.probability

    @contextmanager
    def chaos_context(self) -> Generator[None, None, None]:
        """Context manager for chaos injection."""
        self._start_time = datetime.now(timezone.utc)
        self.is_active = True
        try:
            self.inject()
            yield
        finally:
            recovery_start = time.perf_counter()
            self.recover()
            recovery_time = time.perf_counter() - recovery_start
            self.is_active = False

            # Record recovery
            if self.events:
                last_event = self.events[-1]
                last_event.recovered = True
                last_event.recovery_time_seconds = recovery_time


# =============================================================================
# Network Failure Simulator
# =============================================================================

class NetworkFailureSimulator(ChaosInjector):
    """
    Simulate network failures for resilience testing.

    Supports:
    - Complete network outages
    - Intermittent connectivity
    - DNS failures
    - Connection timeouts
    - Connection refused errors
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        failure_type: str = "connection_refused",
    ):
        """
        Initialize network failure simulator.

        Args:
            config: Chaos configuration
            failure_type: Type of network failure to simulate
                - "connection_refused": Socket connection refused
                - "connection_timeout": Socket timeout
                - "dns_failure": DNS resolution failure
                - "intermittent": Random failures
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.NETWORK_FAILURE,
            severity=SeverityLevel.MEDIUM,
        )
        super().__init__(config)
        self.failure_type = failure_type
        self._original_socket_connect = None
        self._original_getaddrinfo = None
        self._patchers: List[Any] = []

    def inject(self) -> None:
        """Inject network failure."""
        self.record_event(
            target="socket",
            details=f"Injecting {self.failure_type} network failure",
        )

        if self.failure_type == "connection_refused":
            self._inject_connection_refused()
        elif self.failure_type == "connection_timeout":
            self._inject_connection_timeout()
        elif self.failure_type == "dns_failure":
            self._inject_dns_failure()
        elif self.failure_type == "intermittent":
            self._inject_intermittent()

    def _inject_connection_refused(self) -> None:
        """Simulate connection refused errors."""
        original_connect = socket.socket.connect

        def failing_connect(self_socket, address):
            if random.random() < self.config.probability:
                raise ConnectionRefusedError(f"Connection refused to {address}")
            return original_connect(self_socket, address)

        self._original_socket_connect = original_connect
        socket.socket.connect = failing_connect

    def _inject_connection_timeout(self) -> None:
        """Simulate connection timeouts."""
        original_connect = socket.socket.connect

        def timeout_connect(self_socket, address):
            if random.random() < self.config.probability:
                # Simulate timeout by sleeping then raising
                time.sleep(self.config.metadata.get("timeout_seconds", 5))
                raise socket.timeout(f"Connection to {address} timed out")
            return original_connect(self_socket, address)

        self._original_socket_connect = original_connect
        socket.socket.connect = timeout_connect

    def _inject_dns_failure(self) -> None:
        """Simulate DNS resolution failures."""
        original_getaddrinfo = socket.getaddrinfo

        def failing_getaddrinfo(host, port, *args, **kwargs):
            if random.random() < self.config.probability:
                raise socket.gaierror(8, f"nodename nor servname provided for {host}")
            return original_getaddrinfo(host, port, *args, **kwargs)

        self._original_getaddrinfo = original_getaddrinfo
        socket.getaddrinfo = failing_getaddrinfo

    def _inject_intermittent(self) -> None:
        """Simulate intermittent network failures."""
        original_connect = socket.socket.connect

        def intermittent_connect(self_socket, address):
            if random.random() < self.config.probability:
                error_type = random.choice([
                    ConnectionRefusedError,
                    ConnectionResetError,
                    socket.timeout,
                    BrokenPipeError,
                ])
                raise error_type(f"Intermittent network error to {address}")
            return original_connect(self_socket, address)

        self._original_socket_connect = original_connect
        socket.socket.connect = intermittent_connect

    def recover(self) -> None:
        """Recover from network failure injection."""
        if self._original_socket_connect:
            socket.socket.connect = self._original_socket_connect
            self._original_socket_connect = None

        if self._original_getaddrinfo:
            socket.getaddrinfo = self._original_getaddrinfo
            self._original_getaddrinfo = None

        for patcher in self._patchers:
            patcher.stop()
        self._patchers.clear()

    @contextmanager
    def network_down(
        self,
        duration_seconds: float = 5.0,
    ) -> Generator[None, None, None]:
        """Context manager for simulating network down."""
        self.config.probability = 1.0  # 100% failure rate
        self.config.duration_seconds = duration_seconds

        with self.chaos_context():
            yield


class NetworkLatencyInjector(ChaosInjector):
    """
    Inject network latency for testing timeout handling.

    Supports:
    - Fixed latency
    - Variable latency (normal distribution)
    - Latency spikes
    - Jitter simulation
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        base_latency_ms: float = 100.0,
        jitter_ms: float = 50.0,
        spike_probability: float = 0.1,
        spike_multiplier: float = 10.0,
    ):
        """
        Initialize latency injector.

        Args:
            config: Chaos configuration
            base_latency_ms: Base latency to add (milliseconds)
            jitter_ms: Standard deviation of jitter (milliseconds)
            spike_probability: Probability of latency spike
            spike_multiplier: Multiplier for spike latency
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.NETWORK_LATENCY,
            severity=SeverityLevel.MEDIUM,
        )
        super().__init__(config)
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.spike_probability = spike_probability
        self.spike_multiplier = spike_multiplier
        self._original_recv = None
        self._original_send = None

    def _calculate_latency(self) -> float:
        """Calculate latency to inject (in seconds)."""
        latency_ms = self.base_latency_ms + np.random.normal(0, self.jitter_ms)

        # Add spike if triggered
        if random.random() < self.spike_probability:
            latency_ms *= self.spike_multiplier

        return max(0, latency_ms) / 1000.0  # Convert to seconds

    def inject(self) -> None:
        """Inject network latency."""
        self.record_event(
            target="socket",
            details=f"Injecting latency: {self.base_latency_ms}ms +/- {self.jitter_ms}ms",
        )

        original_recv = socket.socket.recv

        def delayed_recv(self_socket, bufsize, flags=0):
            time.sleep(self._calculate_latency())
            return original_recv(self_socket, bufsize, flags)

        self._original_recv = original_recv
        socket.socket.recv = delayed_recv

    def recover(self) -> None:
        """Remove latency injection."""
        if self._original_recv:
            socket.socket.recv = self._original_recv
            self._original_recv = None

    def add_latency(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to add latency to function calls."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(self._calculate_latency())
            return func(*args, **kwargs)
        return wrapper


# =============================================================================
# Timeout Injector
# =============================================================================

class TimeoutInjector(ChaosInjector):
    """
    Inject timeouts into function calls.

    Useful for testing:
    - Timeout handling in async operations
    - Retry logic
    - Circuit breaker patterns
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        timeout_seconds: float = 5.0,
        timeout_probability: float = 0.3,
    ):
        """
        Initialize timeout injector.

        Args:
            config: Chaos configuration
            timeout_seconds: Duration before timeout
            timeout_probability: Probability of triggering timeout
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.TIMEOUT,
            severity=SeverityLevel.MEDIUM,
        )
        super().__init__(config)
        self.timeout_seconds = timeout_seconds
        self.timeout_probability = timeout_probability
        self._wrapped_functions: Dict[str, Callable] = {}

    def inject(self) -> None:
        """Enable timeout injection."""
        self.record_event(
            target="functions",
            details=f"Timeout injection enabled: {self.timeout_seconds}s",
        )

    def recover(self) -> None:
        """Disable timeout injection."""
        self._wrapped_functions.clear()

    def wrap_with_timeout(
        self,
        func: Callable[..., T],
        custom_timeout: Optional[float] = None,
    ) -> Callable[..., T]:
        """
        Wrap a function with timeout injection.

        Args:
            func: Function to wrap
            custom_timeout: Override default timeout

        Returns:
            Wrapped function that may timeout
        """
        timeout = custom_timeout or self.timeout_seconds

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < self.timeout_probability:
                # Simulate slow operation that will timeout
                time.sleep(timeout + 1)
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")
            return func(*args, **kwargs)

        self._wrapped_functions[func.__name__] = wrapper
        return wrapper

    def wrap_async_with_timeout(
        self,
        func: Callable[..., T],
        custom_timeout: Optional[float] = None,
    ) -> Callable[..., T]:
        """Wrap async function with timeout injection."""
        timeout = custom_timeout or self.timeout_seconds

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if random.random() < self.timeout_probability:
                await asyncio.sleep(timeout + 1)
                raise asyncio.TimeoutError(f"Async function {func.__name__} timed out")
            return await func(*args, **kwargs)

        return wrapper

    @contextmanager
    def timeout_context(
        self,
        timeout: Optional[float] = None,
        probability: Optional[float] = None,
    ) -> Generator[None, None, None]:
        """Context manager for timeout zone."""
        timeout = timeout or self.timeout_seconds
        probability = probability or self.timeout_probability

        if random.random() < probability:
            self.record_event(
                target="context",
                details=f"Timeout will trigger in {timeout}s",
            )
            time.sleep(timeout)
            raise TimeoutError(f"Operation timed out after {timeout}s")

        yield


# =============================================================================
# Resource Exhaustion Simulator
# =============================================================================

class ResourceExhaustionSimulator(ChaosInjector):
    """
    Simulate resource exhaustion conditions.

    Tests agent behavior under:
    - Memory pressure
    - CPU saturation
    - Disk full conditions
    - File descriptor exhaustion
    - Thread pool exhaustion
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        resource_type: str = "memory",
    ):
        """
        Initialize resource exhaustion simulator.

        Args:
            config: Chaos configuration
            resource_type: Type of resource to exhaust
                - "memory": Consume memory
                - "cpu": Consume CPU
                - "file_descriptors": Open many files
                - "threads": Create many threads
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.RESOURCE_EXHAUSTION,
            severity=SeverityLevel.MEDIUM,
        )
        super().__init__(config)
        self.resource_type = resource_type
        self._allocated_memory: List[bytes] = []
        self._cpu_threads: List[threading.Thread] = []
        self._open_files: List[Any] = []
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

    def inject(self) -> None:
        """Inject resource exhaustion."""
        self.record_event(
            target=self.resource_type,
            details=f"Exhausting {self.resource_type} resources",
        )

        if self.resource_type == "memory":
            self._exhaust_memory()
        elif self.resource_type == "cpu":
            self._exhaust_cpu()
        elif self.resource_type == "file_descriptors":
            self._exhaust_file_descriptors()
        elif self.resource_type == "threads":
            self._exhaust_threads()

    def _exhaust_memory(self) -> None:
        """Allocate memory to create pressure."""
        target_mb = self.config.metadata.get("target_mb", 256)
        chunk_size = 1024 * 1024  # 1 MB chunks

        for _ in range(target_mb):
            try:
                self._allocated_memory.append(b'x' * chunk_size)
            except MemoryError:
                break

    def _exhaust_cpu(self) -> None:
        """Create CPU load."""
        num_threads = self.config.metadata.get("num_threads", 4)

        def cpu_burn():
            while not self._stop_event.is_set():
                # CPU-intensive operation
                _ = sum(i * i for i in range(10000))

        for _ in range(num_threads):
            t = threading.Thread(target=cpu_burn, daemon=True)
            t.start()
            self._cpu_threads.append(t)

    def _exhaust_file_descriptors(self) -> None:
        """Open many file descriptors."""
        target_files = self.config.metadata.get("target_files", 100)

        for i in range(target_files):
            try:
                # Create temp files
                import tempfile
                f = tempfile.NamedTemporaryFile(delete=False)
                self._open_files.append(f)
            except OSError:
                break

    def _exhaust_threads(self) -> None:
        """Create many threads."""
        target_threads = self.config.metadata.get("target_threads", 100)

        def idle_thread():
            while not self._stop_event.is_set():
                time.sleep(0.1)

        for _ in range(target_threads):
            try:
                t = threading.Thread(target=idle_thread, daemon=True)
                t.start()
                self._threads.append(t)
            except RuntimeError:
                break

    def recover(self) -> None:
        """Release exhausted resources."""
        # Stop CPU threads
        self._stop_event.set()
        for t in self._cpu_threads:
            t.join(timeout=1)
        self._cpu_threads.clear()

        # Release memory
        self._allocated_memory.clear()
        gc.collect()

        # Close files
        for f in self._open_files:
            try:
                f.close()
                os.unlink(f.name)
            except Exception:
                pass
        self._open_files.clear()

        # Stop threads
        for t in self._threads:
            t.join(timeout=0.1)
        self._threads.clear()

        self._stop_event.clear()

    @contextmanager
    def memory_pressure(
        self,
        mb: int = 256,
    ) -> Generator[None, None, None]:
        """Context manager for memory pressure."""
        self.config.metadata["target_mb"] = mb
        self.resource_type = "memory"

        with self.chaos_context():
            yield

    @contextmanager
    def cpu_stress(
        self,
        threads: int = 4,
        duration_seconds: float = 5.0,
    ) -> Generator[None, None, None]:
        """Context manager for CPU stress."""
        self.config.metadata["num_threads"] = threads
        self.config.duration_seconds = duration_seconds
        self.resource_type = "cpu"

        with self.chaos_context():
            time.sleep(duration_seconds)


# =============================================================================
# Exception Injection
# =============================================================================

class ExceptionInjector(ChaosInjector):
    """
    Inject exceptions into function calls.

    Useful for testing:
    - Error handling paths
    - Exception propagation
    - Recovery procedures
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        exception_types: List[Type[Exception]] = None,
        probability: float = 0.3,
    ):
        """
        Initialize exception injector.

        Args:
            config: Chaos configuration
            exception_types: List of exception types to inject
            probability: Probability of exception occurring
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.EXCEPTION_INJECTION,
            severity=SeverityLevel.MEDIUM,
        )
        super().__init__(config)
        self.exception_types = exception_types or [
            RuntimeError,
            ValueError,
            IOError,
            ConnectionError,
        ]
        self.probability = probability

    def inject(self) -> None:
        """Enable exception injection."""
        self.record_event(
            target="functions",
            details=f"Exception injection enabled: {[e.__name__ for e in self.exception_types]}",
        )

    def recover(self) -> None:
        """Disable exception injection."""
        pass

    def maybe_raise(
        self,
        message: str = "Chaos-injected exception",
    ) -> None:
        """Maybe raise an exception based on probability."""
        if random.random() < self.probability:
            exception_type = random.choice(self.exception_types)
            raise exception_type(message)

    def wrap_with_exceptions(
        self,
        func: Callable[..., T],
    ) -> Callable[..., T]:
        """Wrap function with exception injection."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.maybe_raise(f"Exception in {func.__name__}")
            return func(*args, **kwargs)
        return wrapper

    @contextmanager
    def exception_zone(
        self,
        exception_type: Type[Exception] = RuntimeError,
        probability: float = None,
    ) -> Generator[None, None, None]:
        """Context manager that may raise exception."""
        probability = probability or self.probability

        if random.random() < probability:
            self.record_event(
                target="context",
                details=f"Raising {exception_type.__name__}",
            )
            raise exception_type("Chaos-injected exception in context")

        yield


# =============================================================================
# Data Corruption Simulator
# =============================================================================

class DataCorruptionSimulator(ChaosInjector):
    """
    Simulate data corruption for data integrity testing.

    Useful for testing:
    - Data validation
    - Checksum verification
    - Provenance hash validation
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        corruption_type: str = "bit_flip",
        probability: float = 0.1,
    ):
        """
        Initialize data corruption simulator.

        Args:
            config: Chaos configuration
            corruption_type: Type of corruption
                - "bit_flip": Flip random bits
                - "truncation": Truncate data
                - "substitution": Replace values
                - "duplication": Duplicate data
            probability: Probability of corruption
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.DATA_CORRUPTION,
            severity=SeverityLevel.LOW,
        )
        super().__init__(config)
        self.corruption_type = corruption_type
        self.probability = probability

    def inject(self) -> None:
        """Enable data corruption."""
        self.record_event(
            target="data",
            details=f"Data corruption enabled: {self.corruption_type}",
        )

    def recover(self) -> None:
        """Disable data corruption."""
        pass

    def corrupt_string(self, data: str) -> str:
        """Corrupt a string value."""
        if random.random() >= self.probability:
            return data

        if self.corruption_type == "bit_flip":
            # Flip a random character
            if len(data) == 0:
                return data
            idx = random.randint(0, len(data) - 1)
            chars = list(data)
            chars[idx] = chr(ord(chars[idx]) ^ random.randint(1, 255))
            return ''.join(chars)

        elif self.corruption_type == "truncation":
            # Truncate at random point
            if len(data) == 0:
                return data
            return data[:random.randint(0, len(data) - 1)]

        elif self.corruption_type == "substitution":
            # Replace with random string
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=len(data)))

        elif self.corruption_type == "duplication":
            # Duplicate a section
            if len(data) < 2:
                return data
            idx = random.randint(0, len(data) - 1)
            length = random.randint(1, min(5, len(data) - idx))
            return data[:idx] + data[idx:idx+length] * 2 + data[idx+length:]

        return data

    def corrupt_number(self, data: float) -> float:
        """Corrupt a numeric value."""
        if random.random() >= self.probability:
            return data

        if self.corruption_type == "bit_flip":
            # Add small noise
            return data * (1 + random.gauss(0, 0.1))

        elif self.corruption_type == "substitution":
            # Replace with random value
            return random.uniform(-1e10, 1e10)

        return data

    def corrupt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Corrupt dictionary values."""
        if random.random() >= self.probability:
            return data

        corrupted = dict(data)
        if not corrupted:
            return corrupted

        # Corrupt a random key's value
        key = random.choice(list(corrupted.keys()))
        value = corrupted[key]

        if isinstance(value, str):
            corrupted[key] = self.corrupt_string(value)
        elif isinstance(value, (int, float)):
            corrupted[key] = self.corrupt_number(float(value))
        elif isinstance(value, dict):
            corrupted[key] = self.corrupt_dict(value)

        return corrupted


# =============================================================================
# Clock Skew Simulator
# =============================================================================

class ClockSkewSimulator(ChaosInjector):
    """
    Simulate clock skew for testing time-dependent operations.

    Useful for testing:
    - Timestamp validation
    - Token expiration
    - Cache invalidation
    - Scheduling logic
    """

    def __init__(
        self,
        config: ChaosConfig = None,
        skew_seconds: float = 300.0,  # 5 minutes
        direction: str = "forward",
    ):
        """
        Initialize clock skew simulator.

        Args:
            config: Chaos configuration
            skew_seconds: Amount of time skew
            direction: "forward" or "backward"
        """
        config = config or ChaosConfig(
            chaos_type=ChaosType.CLOCK_SKEW,
            severity=SeverityLevel.LOW,
        )
        super().__init__(config)
        self.skew_seconds = skew_seconds
        self.direction = direction
        self._original_datetime_now = None
        self._original_time_time = None

    def _get_skewed_time(self) -> float:
        """Get current time with skew applied."""
        import time as time_module
        original_time = self._original_time_time()
        skew = self.skew_seconds if self.direction == "forward" else -self.skew_seconds
        return original_time + skew

    def inject(self) -> None:
        """Inject clock skew."""
        self.record_event(
            target="time",
            details=f"Clock skew: {self.skew_seconds}s {self.direction}",
        )

        # Note: In practice, mocking time requires careful patching
        # This is a simplified demonstration
        import time as time_module
        self._original_time_time = time_module.time
        time_module.time = self._get_skewed_time

    def recover(self) -> None:
        """Remove clock skew."""
        if self._original_time_time:
            import time as time_module
            time_module.time = self._original_time_time
            self._original_time_time = None

    def get_skewed_datetime(
        self,
        dt: datetime = None,
    ) -> datetime:
        """Get a datetime with skew applied."""
        dt = dt or datetime.now(timezone.utc)
        skew = timedelta(seconds=self.skew_seconds)

        if self.direction == "forward":
            return dt + skew
        else:
            return dt - skew


# =============================================================================
# Chaos Controller
# =============================================================================

class ChaosController:
    """
    Central controller for chaos testing.

    Provides:
    - Coordinated chaos injection
    - Event logging
    - Recovery management
    - Chaos scheduling
    """

    def __init__(
        self,
        seed: int = 42,
        enabled: bool = True,
    ):
        """
        Initialize chaos controller.

        Args:
            seed: Random seed for reproducibility
            enabled: Whether chaos is enabled
        """
        self.seed = seed
        self.enabled = enabled
        self.injectors: Dict[str, ChaosInjector] = {}
        self.events: List[ChaosEvent] = []
        self._active_chaos: Set[str] = set()

        random.seed(seed)
        np.random.seed(seed)

    def register(
        self,
        name: str,
        injector: ChaosInjector,
    ) -> None:
        """Register a chaos injector."""
        self.injectors[name] = injector

    def create_network_failure(
        self,
        name: str = "network",
        failure_type: str = "connection_refused",
        probability: float = 0.5,
    ) -> NetworkFailureSimulator:
        """Create and register a network failure simulator."""
        config = ChaosConfig(
            chaos_type=ChaosType.NETWORK_FAILURE,
            probability=probability,
        )
        injector = NetworkFailureSimulator(config, failure_type)
        self.register(name, injector)
        return injector

    def create_timeout_injector(
        self,
        name: str = "timeout",
        timeout_seconds: float = 5.0,
        probability: float = 0.3,
    ) -> TimeoutInjector:
        """Create and register a timeout injector."""
        injector = TimeoutInjector(
            timeout_seconds=timeout_seconds,
            timeout_probability=probability,
        )
        self.register(name, injector)
        return injector

    def create_resource_exhaustion(
        self,
        name: str = "resources",
        resource_type: str = "memory",
    ) -> ResourceExhaustionSimulator:
        """Create and register a resource exhaustion simulator."""
        injector = ResourceExhaustionSimulator(resource_type=resource_type)
        self.register(name, injector)
        return injector

    def create_exception_injector(
        self,
        name: str = "exceptions",
        exception_types: List[Type[Exception]] = None,
        probability: float = 0.3,
    ) -> ExceptionInjector:
        """Create and register an exception injector."""
        injector = ExceptionInjector(
            exception_types=exception_types,
            probability=probability,
        )
        self.register(name, injector)
        return injector

    def start(self, name: str) -> None:
        """Start a registered chaos injector."""
        if not self.enabled:
            return

        if name not in self.injectors:
            raise KeyError(f"Unknown injector: {name}")

        injector = self.injectors[name]
        injector.inject()
        self._active_chaos.add(name)

    def stop(self, name: str) -> None:
        """Stop a registered chaos injector."""
        if name not in self.injectors:
            raise KeyError(f"Unknown injector: {name}")

        injector = self.injectors[name]
        injector.recover()
        self._active_chaos.discard(name)

        # Collect events
        self.events.extend(injector.events)

    def stop_all(self) -> None:
        """Stop all active chaos injectors."""
        for name in list(self._active_chaos):
            self.stop(name)

    @contextmanager
    def chaos_session(
        self,
        *names: str,
    ) -> Generator[None, None, None]:
        """Context manager for chaos session with multiple injectors."""
        if not self.enabled:
            yield
            return

        for name in names:
            self.start(name)

        try:
            yield
        finally:
            for name in names:
                self.stop(name)

    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of all chaos events."""
        if not self.events:
            return {"total_events": 0}

        by_type = {}
        for event in self.events:
            type_name = event.chaos_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(event)

        return {
            "total_events": len(self.events),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "recovered": sum(1 for e in self.events if e.recovered),
            "avg_recovery_time": np.mean([
                e.recovery_time_seconds
                for e in self.events
                if e.recovery_time_seconds is not None
            ]) if any(e.recovery_time_seconds for e in self.events) else 0,
        }


# =============================================================================
# Pytest Fixtures for Chaos Testing
# =============================================================================

def pytest_chaos_fixtures():
    """
    Return pytest fixture functions for chaos testing.

    Usage in conftest.py:
        from greenlang_framework.testing.chaos import pytest_chaos_fixtures
        fixtures = pytest_chaos_fixtures()
        chaos_controller = fixtures['chaos_controller']
    """
    import pytest

    @pytest.fixture
    def chaos_controller():
        """Provide chaos controller for tests."""
        controller = ChaosController(seed=42, enabled=True)
        yield controller
        controller.stop_all()

    @pytest.fixture
    def network_chaos(chaos_controller):
        """Provide network failure simulator."""
        return chaos_controller.create_network_failure()

    @pytest.fixture
    def timeout_chaos(chaos_controller):
        """Provide timeout injector."""
        return chaos_controller.create_timeout_injector()

    @pytest.fixture
    def resource_chaos(chaos_controller):
        """Provide resource exhaustion simulator."""
        return chaos_controller.create_resource_exhaustion()

    @pytest.fixture
    def exception_chaos(chaos_controller):
        """Provide exception injector."""
        return chaos_controller.create_exception_injector()

    return {
        'chaos_controller': chaos_controller,
        'network_chaos': network_chaos,
        'timeout_chaos': timeout_chaos,
        'resource_chaos': resource_chaos,
        'exception_chaos': exception_chaos,
    }


# =============================================================================
# Export all public classes and functions
# =============================================================================

__all__ = [
    # Enums
    "ChaosType",
    "SeverityLevel",
    # Configuration
    "ChaosConfig",
    "ChaosEvent",
    # Base classes
    "ChaosInjector",
    # Network chaos
    "NetworkFailureSimulator",
    "NetworkLatencyInjector",
    # Timeout
    "TimeoutInjector",
    # Resource
    "ResourceExhaustionSimulator",
    # Exception
    "ExceptionInjector",
    # Data
    "DataCorruptionSimulator",
    # Clock
    "ClockSkewSimulator",
    # Controller
    "ChaosController",
    # Fixtures
    "pytest_chaos_fixtures",
]
