"""
GL-001 ThermalCommand - Fault Injectors

This module provides fault injection capabilities for chaos engineering tests.
All injectors are designed to be safe for CI/CD execution and use simulation
rather than actual infrastructure manipulation.

Fault Categories:
- Network faults (latency, partition, packet loss)
- Resource faults (CPU, memory, disk)
- Service faults (unavailability, slow response)
- State faults (corruption, inconsistency)

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


# =============================================================================
# Base Classes
# =============================================================================

class FaultInjector(ABC):
    """Abstract base class for fault injectors."""

    def __init__(self):
        self._active = False
        self._injection_time: Optional[datetime] = None
        self._params: Dict[str, Any] = {}

    @abstractmethod
    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject the fault. Returns True if successful."""
        pass

    @abstractmethod
    async def rollback(self) -> bool:
        """Rollback the fault. Returns True if successful."""
        pass

    def is_active(self) -> bool:
        """Check if fault is currently active."""
        return self._active


class NoOpFaultInjector(FaultInjector):
    """No-operation fault injector for unknown fault types."""

    async def inject(self, params: Dict[str, Any]) -> bool:
        logger.debug("NoOpFaultInjector: inject called (no-op)")
        self._active = True
        return True

    async def rollback(self) -> bool:
        logger.debug("NoOpFaultInjector: rollback called (no-op)")
        self._active = False
        return True


# =============================================================================
# Network Fault Injectors
# =============================================================================

class NetworkLatencyInjector(FaultInjector):
    """
    Inject network latency into simulated network calls.

    This injector wraps async functions to add configurable delays,
    simulating network latency without affecting actual network calls.

    Example:
        >>> injector = NetworkLatencyInjector()
        >>> await injector.inject({"delay_ms": 200, "jitter_ms": 50})
        >>> # All wrapped async calls now have 200ms +/- 50ms latency
        >>> await injector.rollback()
    """

    def __init__(self):
        super().__init__()
        self._original_functions: Dict[str, Callable] = {}
        self._delay_ms: float = 0
        self._jitter_ms: float = 0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject network latency."""
        try:
            self._delay_ms = params.get("delay_ms", 100)
            self._jitter_ms = params.get("jitter_ms", 10)
            self._params = params

            logger.info(
                f"NetworkLatencyInjector: Injecting {self._delay_ms}ms latency "
                f"(jitter: {self._jitter_ms}ms)"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"NetworkLatencyInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove network latency injection."""
        try:
            logger.info("NetworkLatencyInjector: Rolling back latency injection")
            self._active = False
            self._delay_ms = 0
            self._jitter_ms = 0
            return True

        except Exception as e:
            logger.error(f"NetworkLatencyInjector: Rollback failed: {e}")
            return False

    def get_delay(self) -> float:
        """Get current delay with jitter in seconds."""
        if not self._active:
            return 0

        jitter = random.uniform(-self._jitter_ms, self._jitter_ms)
        return (self._delay_ms + jitter) / 1000.0

    async def simulate_latency(self):
        """Apply simulated latency delay."""
        if self._active:
            await asyncio.sleep(self.get_delay())


class NetworkPartitionInjector(FaultInjector):
    """
    Simulate network partition (split-brain) scenarios.

    This injector simulates network partitions by blocking communication
    between specified node groups in a simulated distributed system.

    Example:
        >>> injector = NetworkPartitionInjector()
        >>> await injector.inject({
        ...     "partition_groups": [["node1", "node2"], ["node3", "node4"]],
        ...     "bidirectional": True
        ... })
    """

    def __init__(self):
        super().__init__()
        self._blocked_routes: Set[tuple] = set()
        self._partition_groups: List[List[str]] = []

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject network partition."""
        try:
            self._partition_groups = params.get("partition_groups", [])
            bidirectional = params.get("bidirectional", True)

            # Block routes between partition groups
            for i, group1 in enumerate(self._partition_groups):
                for group2 in self._partition_groups[i + 1:]:
                    for node1 in group1:
                        for node2 in group2:
                            self._blocked_routes.add((node1, node2))
                            if bidirectional:
                                self._blocked_routes.add((node2, node1))

            logger.info(
                f"NetworkPartitionInjector: Created partition with "
                f"{len(self._blocked_routes)} blocked routes"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params
            return True

        except Exception as e:
            logger.error(f"NetworkPartitionInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove network partition."""
        try:
            logger.info("NetworkPartitionInjector: Healing network partition")
            self._blocked_routes.clear()
            self._partition_groups.clear()
            self._active = False
            return True

        except Exception as e:
            logger.error(f"NetworkPartitionInjector: Rollback failed: {e}")
            return False

    def is_route_blocked(self, source: str, destination: str) -> bool:
        """Check if route is blocked by partition."""
        return (source, destination) in self._blocked_routes


class PacketLossInjector(FaultInjector):
    """
    Simulate network packet loss.

    This injector simulates packet loss by randomly failing
    a percentage of network operations.

    Example:
        >>> injector = PacketLossInjector()
        >>> await injector.inject({"loss_percentage": 20})  # 20% packet loss
    """

    def __init__(self):
        super().__init__()
        self._loss_percentage: float = 0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject packet loss."""
        try:
            self._loss_percentage = params.get("loss_percentage", 10)
            self._params = params

            logger.info(f"PacketLossInjector: Injecting {self._loss_percentage}% packet loss")

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"PacketLossInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove packet loss injection."""
        try:
            logger.info("PacketLossInjector: Rolling back packet loss injection")
            self._active = False
            self._loss_percentage = 0
            return True

        except Exception as e:
            logger.error(f"PacketLossInjector: Rollback failed: {e}")
            return False

    def should_drop_packet(self) -> bool:
        """Determine if current packet should be dropped."""
        if not self._active:
            return False
        return random.random() * 100 < self._loss_percentage


class NetworkFaultInjector(FaultInjector):
    """
    Composite network fault injector combining latency, partition, and packet loss.

    Example:
        >>> injector = NetworkFaultInjector()
        >>> await injector.inject({
        ...     "latency_ms": 100,
        ...     "packet_loss_percent": 5,
        ...     "partition_nodes": ["node3"]
        ... })
    """

    def __init__(self):
        super().__init__()
        self._latency_injector = NetworkLatencyInjector()
        self._partition_injector = NetworkPartitionInjector()
        self._packet_loss_injector = PacketLossInjector()

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject composite network faults."""
        try:
            success = True

            if "latency_ms" in params or "delay_ms" in params:
                success &= await self._latency_injector.inject({
                    "delay_ms": params.get("latency_ms") or params.get("delay_ms", 0),
                    "jitter_ms": params.get("jitter_ms", 10),
                })

            if "partition_groups" in params:
                success &= await self._partition_injector.inject({
                    "partition_groups": params["partition_groups"],
                    "bidirectional": params.get("bidirectional", True),
                })

            if "packet_loss_percent" in params or "loss_percentage" in params:
                success &= await self._packet_loss_injector.inject({
                    "loss_percentage": params.get("packet_loss_percent") or params.get("loss_percentage", 0),
                })

            self._active = success
            self._injection_time = datetime.now(timezone.utc)
            self._params = params
            return success

        except Exception as e:
            logger.error(f"NetworkFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Rollback all network faults."""
        try:
            success = True
            success &= await self._latency_injector.rollback()
            success &= await self._partition_injector.rollback()
            success &= await self._packet_loss_injector.rollback()
            self._active = False
            return success

        except Exception as e:
            logger.error(f"NetworkFaultInjector: Rollback failed: {e}")
            return False


# =============================================================================
# Resource Fault Injectors
# =============================================================================

class CPUStressInjector(FaultInjector):
    """
    Simulate CPU stress/exhaustion.

    This injector simulates CPU pressure by performing CPU-intensive
    operations in background threads (configurable intensity).

    Note: For CI safety, this uses simulated stress rather than actual
    CPU exhaustion.

    Example:
        >>> injector = CPUStressInjector()
        >>> await injector.inject({"target_cpu_percent": 80, "duration_seconds": 30})
    """

    def __init__(self):
        super().__init__()
        self._stress_threads: List[threading.Thread] = []
        self._stop_flag = threading.Event()
        self._target_cpu_percent: float = 0
        self._simulated_load: float = 0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject CPU stress."""
        try:
            self._target_cpu_percent = params.get("target_cpu_percent", 50)
            num_cores = params.get("num_cores", 1)

            logger.info(
                f"CPUStressInjector: Simulating {self._target_cpu_percent}% CPU load "
                f"on {num_cores} cores"
            )

            # In CI-safe mode, we just track the simulated load
            # Real CPU stress would spawn busy-loop threads
            self._simulated_load = self._target_cpu_percent
            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"CPUStressInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove CPU stress."""
        try:
            logger.info("CPUStressInjector: Rolling back CPU stress")

            self._stop_flag.set()

            for thread in self._stress_threads:
                thread.join(timeout=2.0)

            self._stress_threads.clear()
            self._stop_flag.clear()
            self._simulated_load = 0
            self._active = False

            return True

        except Exception as e:
            logger.error(f"CPUStressInjector: Rollback failed: {e}")
            return False

    def get_simulated_load(self) -> float:
        """Get current simulated CPU load percentage."""
        return self._simulated_load if self._active else 0


class MemoryPressureInjector(FaultInjector):
    """
    Simulate memory pressure.

    This injector simulates memory pressure by tracking allocated memory
    without actually consuming large amounts of RAM (CI-safe).

    Example:
        >>> injector = MemoryPressureInjector()
        >>> await injector.inject({"target_memory_mb": 512})
    """

    def __init__(self):
        super().__init__()
        self._allocated_blocks: List[bytes] = []
        self._target_memory_mb: float = 0
        self._simulated_memory_mb: float = 0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject memory pressure."""
        try:
            self._target_memory_mb = params.get("target_memory_mb", 256)
            allocate_real = params.get("allocate_real", False)

            logger.info(f"MemoryPressureInjector: Simulating {self._target_memory_mb}MB memory pressure")

            if allocate_real:
                # Allocate small chunks to simulate pressure
                # Limited to prevent actual memory issues
                chunk_size = min(10, self._target_memory_mb) * 1024 * 1024  # Max 10MB
                self._allocated_blocks.append(bytes(chunk_size))

            self._simulated_memory_mb = self._target_memory_mb
            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"MemoryPressureInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Release memory pressure."""
        try:
            logger.info("MemoryPressureInjector: Rolling back memory pressure")

            self._allocated_blocks.clear()
            self._simulated_memory_mb = 0
            self._active = False

            return True

        except Exception as e:
            logger.error(f"MemoryPressureInjector: Rollback failed: {e}")
            return False

    def get_simulated_memory_usage(self) -> float:
        """Get current simulated memory usage in MB."""
        return self._simulated_memory_mb if self._active else 0


class DiskIOFaultInjector(FaultInjector):
    """
    Simulate disk I/O failures.

    This injector simulates disk I/O issues by introducing delays
    and failures into file operations (simulated, not actual disk ops).

    Example:
        >>> injector = DiskIOFaultInjector()
        >>> await injector.inject({
        ...     "read_latency_ms": 500,
        ...     "write_failure_percent": 10,
        ...     "disk_full": False
        ... })
    """

    def __init__(self):
        super().__init__()
        self._read_latency_ms: float = 0
        self._write_latency_ms: float = 0
        self._read_failure_percent: float = 0
        self._write_failure_percent: float = 0
        self._disk_full: bool = False

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject disk I/O faults."""
        try:
            self._read_latency_ms = params.get("read_latency_ms", 0)
            self._write_latency_ms = params.get("write_latency_ms", 0)
            self._read_failure_percent = params.get("read_failure_percent", 0)
            self._write_failure_percent = params.get("write_failure_percent", 0)
            self._disk_full = params.get("disk_full", False)

            logger.info(
                f"DiskIOFaultInjector: Simulating disk I/O faults "
                f"(read_latency={self._read_latency_ms}ms, "
                f"write_failure={self._write_failure_percent}%)"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"DiskIOFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove disk I/O faults."""
        try:
            logger.info("DiskIOFaultInjector: Rolling back disk I/O faults")

            self._read_latency_ms = 0
            self._write_latency_ms = 0
            self._read_failure_percent = 0
            self._write_failure_percent = 0
            self._disk_full = False
            self._active = False

            return True

        except Exception as e:
            logger.error(f"DiskIOFaultInjector: Rollback failed: {e}")
            return False

    def should_fail_read(self) -> bool:
        """Determine if read operation should fail."""
        if not self._active:
            return False
        return random.random() * 100 < self._read_failure_percent

    def should_fail_write(self) -> bool:
        """Determine if write operation should fail."""
        if not self._active:
            return False
        if self._disk_full:
            return True
        return random.random() * 100 < self._write_failure_percent


class ResourceFaultInjector(FaultInjector):
    """
    Composite resource fault injector combining CPU, memory, and disk faults.

    Example:
        >>> injector = ResourceFaultInjector()
        >>> await injector.inject({
        ...     "cpu_percent": 70,
        ...     "memory_mb": 256,
        ...     "disk_latency_ms": 100
        ... })
    """

    def __init__(self):
        super().__init__()
        self._cpu_injector = CPUStressInjector()
        self._memory_injector = MemoryPressureInjector()
        self._disk_injector = DiskIOFaultInjector()

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject composite resource faults."""
        try:
            success = True

            if "cpu_percent" in params or "target_cpu_percent" in params:
                success &= await self._cpu_injector.inject({
                    "target_cpu_percent": params.get("cpu_percent") or params.get("target_cpu_percent", 0),
                })

            if "memory_mb" in params or "target_memory_mb" in params:
                success &= await self._memory_injector.inject({
                    "target_memory_mb": params.get("memory_mb") or params.get("target_memory_mb", 0),
                })

            if any(k in params for k in ["disk_latency_ms", "read_latency_ms", "write_failure_percent"]):
                success &= await self._disk_injector.inject({
                    "read_latency_ms": params.get("disk_latency_ms") or params.get("read_latency_ms", 0),
                    "write_latency_ms": params.get("write_latency_ms", 0),
                    "write_failure_percent": params.get("write_failure_percent", 0),
                })

            self._active = success
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return success

        except Exception as e:
            logger.error(f"ResourceFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Rollback all resource faults."""
        try:
            success = True
            success &= await self._cpu_injector.rollback()
            success &= await self._memory_injector.rollback()
            success &= await self._disk_injector.rollback()
            self._active = False
            return success

        except Exception as e:
            logger.error(f"ResourceFaultInjector: Rollback failed: {e}")
            return False


# =============================================================================
# Service Fault Injectors
# =============================================================================

class ServiceUnavailabilityInjector(FaultInjector):
    """
    Simulate external service unavailability.

    This injector simulates service outages by returning errors
    for specified services.

    Example:
        >>> injector = ServiceUnavailabilityInjector()
        >>> await injector.inject({
        ...     "services": ["database", "cache", "api"],
        ...     "failure_mode": "timeout"  # or "connection_refused", "error_500"
        ... })
    """

    def __init__(self):
        super().__init__()
        self._unavailable_services: Set[str] = set()
        self._failure_mode: str = "timeout"

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject service unavailability."""
        try:
            services = params.get("services", [])
            self._failure_mode = params.get("failure_mode", "timeout")

            self._unavailable_services = set(services)

            logger.info(
                f"ServiceUnavailabilityInjector: Services unavailable: "
                f"{services} (mode={self._failure_mode})"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"ServiceUnavailabilityInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Restore service availability."""
        try:
            logger.info("ServiceUnavailabilityInjector: Restoring service availability")

            self._unavailable_services.clear()
            self._active = False

            return True

        except Exception as e:
            logger.error(f"ServiceUnavailabilityInjector: Rollback failed: {e}")
            return False

    def is_service_available(self, service_name: str) -> bool:
        """Check if service is available."""
        return service_name not in self._unavailable_services

    def get_failure_mode(self) -> str:
        """Get current failure mode."""
        return self._failure_mode if self._active else "none"


class SlowServiceInjector(FaultInjector):
    """
    Simulate slow service responses.

    Example:
        >>> injector = SlowServiceInjector()
        >>> await injector.inject({
        ...     "services": {"database": 500, "api": 200},  # service: latency_ms
        ... })
    """

    def __init__(self):
        super().__init__()
        self._service_latencies: Dict[str, float] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject slow service responses."""
        try:
            self._service_latencies = params.get("services", {})

            logger.info(
                f"SlowServiceInjector: Injecting latencies: {self._service_latencies}"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"SlowServiceInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove slow service injection."""
        try:
            logger.info("SlowServiceInjector: Rolling back service latencies")

            self._service_latencies.clear()
            self._active = False

            return True

        except Exception as e:
            logger.error(f"SlowServiceInjector: Rollback failed: {e}")
            return False

    def get_latency(self, service_name: str) -> float:
        """Get latency for service in seconds."""
        if not self._active:
            return 0
        return self._service_latencies.get(service_name, 0) / 1000.0


class ServiceFaultInjector(FaultInjector):
    """
    Composite service fault injector.

    Example:
        >>> injector = ServiceFaultInjector()
        >>> await injector.inject({
        ...     "unavailable_services": ["cache"],
        ...     "slow_services": {"database": 500}
        ... })
    """

    def __init__(self):
        super().__init__()
        self._unavailability_injector = ServiceUnavailabilityInjector()
        self._slow_injector = SlowServiceInjector()

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject composite service faults."""
        try:
            success = True

            if "unavailable_services" in params or "services" in params:
                services = params.get("unavailable_services") or params.get("services", [])
                if isinstance(services, list):
                    success &= await self._unavailability_injector.inject({
                        "services": services,
                        "failure_mode": params.get("failure_mode", "timeout"),
                    })

            if "slow_services" in params:
                success &= await self._slow_injector.inject({
                    "services": params["slow_services"],
                })

            self._active = success
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return success

        except Exception as e:
            logger.error(f"ServiceFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Rollback all service faults."""
        try:
            success = True
            success &= await self._unavailability_injector.rollback()
            success &= await self._slow_injector.rollback()
            self._active = False
            return success

        except Exception as e:
            logger.error(f"ServiceFaultInjector: Rollback failed: {e}")
            return False


# =============================================================================
# State Fault Injectors
# =============================================================================

class StateFaultInjector(FaultInjector):
    """
    Simulate state corruption and inconsistency.

    This injector simulates various state-related faults such as
    cache poisoning, stale data, and inconsistent state.

    Example:
        >>> injector = StateFaultInjector()
        >>> await injector.inject({
        ...     "corruption_type": "stale_cache",
        ...     "affected_keys": ["user_session", "rate_limit"],
        ...     "corruption_percent": 30
        ... })
    """

    def __init__(self):
        super().__init__()
        self._corruption_type: str = ""
        self._affected_keys: Set[str] = set()
        self._corruption_percent: float = 0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject state corruption."""
        try:
            self._corruption_type = params.get("corruption_type", "random")
            self._affected_keys = set(params.get("affected_keys", []))
            self._corruption_percent = params.get("corruption_percent", 50)

            logger.info(
                f"StateFaultInjector: Injecting {self._corruption_type} corruption "
                f"({self._corruption_percent}% of keys)"
            )

            self._active = True
            self._injection_time = datetime.now(timezone.utc)
            self._params = params

            return True

        except Exception as e:
            logger.error(f"StateFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove state corruption."""
        try:
            logger.info("StateFaultInjector: Rolling back state corruption")

            self._corruption_type = ""
            self._affected_keys.clear()
            self._corruption_percent = 0
            self._active = False

            return True

        except Exception as e:
            logger.error(f"StateFaultInjector: Rollback failed: {e}")
            return False

    def is_key_corrupted(self, key: str) -> bool:
        """Check if a key should return corrupted data."""
        if not self._active:
            return False

        if self._affected_keys and key not in self._affected_keys:
            return False

        return random.random() * 100 < self._corruption_percent

    def get_corruption_type(self) -> str:
        """Get active corruption type."""
        return self._corruption_type if self._active else "none"


# =============================================================================
# Factory Function
# =============================================================================

def get_fault_injector(fault_type: str) -> FaultInjector:
    """
    Factory function to get appropriate fault injector.

    Args:
        fault_type: Type of fault injector needed

    Returns:
        Appropriate FaultInjector instance
    """
    injector_map = {
        "network_latency": NetworkLatencyInjector,
        "network_partition": NetworkPartitionInjector,
        "packet_loss": PacketLossInjector,
        "network": NetworkFaultInjector,
        "cpu_stress": CPUStressInjector,
        "memory_pressure": MemoryPressureInjector,
        "disk_io": DiskIOFaultInjector,
        "resource": ResourceFaultInjector,
        "service_unavailable": ServiceUnavailabilityInjector,
        "slow_service": SlowServiceInjector,
        "service": ServiceFaultInjector,
        "state_corruption": StateFaultInjector,
        "state": StateFaultInjector,
    }

    injector_class = injector_map.get(fault_type, NoOpFaultInjector)
    return injector_class()
