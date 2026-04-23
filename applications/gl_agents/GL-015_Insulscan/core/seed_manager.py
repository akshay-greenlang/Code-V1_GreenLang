# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Seed Manager for Deterministic RNG

Implements centralized seed management for reproducible random number generation
across the INSULSCAN agent. Essential for audit compliance and debugging
of ML-based anomaly detection and pattern recognition components.

Design Principles:
    - Reproducibility: Exact reproduction of any processing run
    - Auditability: Complete seed logging for regulatory compliance
    - Isolation: Independent seed contexts for parallel processing
    - Zero-Hallucination: Deterministic behavior for all calculations

Usage:
    This module manages seeds for:
    - Anomaly detection ML models (hot spot pattern recognition)
    - Degradation prediction models
    - Monte Carlo simulations for uncertainty quantification
    - Data sampling operations
    - Test data generation

    Note: Thermal calculations and heat loss formulas NEVER use random numbers.
    Random numbers are only for ML prediction support and pattern analysis.

Example:
    >>> manager = SeedManager()
    >>> manager.set_global_seed(42)
    >>>
    >>> # Get reproducibility context for audit
    >>> context = manager.get_reproducibility_context()
    >>>
    >>> # Use domain-specific seed
    >>> with manager.seed_context(SeedDomain.ANOMALY_DETECTION):
    ...     result = anomaly_model.predict(data)
    >>>
    >>> # Reset all seeds
    >>> manager.reset_seeds()

Author: GreenLang GL-015 INSULSCAN
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic operations
T = TypeVar("T")

# Check for numpy availability (optional dependency)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False
    logger.info("NumPy not available - SeedManager will manage Python random only")


# =============================================================================
# ENUMS
# =============================================================================


class SeedDomain(str, Enum):
    """
    Domains for seed isolation and management.

    Each domain gets a deterministically derived seed from the global seed,
    ensuring reproducibility while allowing independent control.
    """
    GLOBAL = "global"
    THERMAL_MODEL = "thermal_model"
    ANOMALY_DETECTION = "anomaly_detection"
    HOT_SPOT_DETECTION = "hot_spot_detection"
    DEGRADATION_PREDICTION = "degradation_prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    MONTE_CARLO = "monte_carlo"
    DATA_SAMPLING = "data_sampling"
    CROSS_VALIDATION = "cross_validation"
    TEST_GENERATION = "test_generation"
    SIMULATION = "simulation"
    UNCERTAINTY = "uncertainty"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SeedRecord:
    """
    Record of a seed operation for audit trail.

    Captures all information needed to reproduce a random state
    at any point in processing.

    Attributes:
        record_id: Unique identifier for this record
        timestamp: When the seed was set/used
        domain: Which subsystem used this seed
        seed_value: The actual seed value
        operation: What operation triggered this
        caller: Module/function that requested the seed
        provenance_hash: SHA-256 hash for audit trail
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    domain: SeedDomain = SeedDomain.GLOBAL
    seed_value: int = 0
    operation: str = ""
    caller: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this record."""
        content = (
            f"{self.record_id}|{self.timestamp.isoformat()}|"
            f"{self.domain.value}|{self.seed_value}|"
            f"{self.operation}|{self.caller}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "domain": self.domain.value,
            "seed_value": self.seed_value,
            "operation": self.operation,
            "caller": self.caller,
            "context": self.context,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRecord":
        """Create record from dictionary."""
        return cls(
            record_id=data["record_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            domain=SeedDomain(data["domain"]),
            seed_value=data["seed_value"],
            operation=data["operation"],
            caller=data["caller"],
            context=data.get("context", {}),
            provenance_hash=data.get("provenance_hash", ""),
        )


@dataclass
class ReproducibilityContext:
    """
    Complete context needed to reproduce a processing run.

    Contains all information necessary to exactly reproduce
    the random state at any point in processing.

    Attributes:
        context_id: Unique identifier for this context
        created_at: When this context was created
        global_seed: The master seed used
        domain_seeds: Derived seeds for each domain
        python_version: Python version used
        numpy_version: NumPy version (if available)
        platform: Operating system platform
        hostname: Machine hostname
        process_id: Process ID
        seed_sequence: Sequence of seed operations
        provenance_hash: SHA-256 hash for audit trail
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    global_seed: int = 42
    domain_seeds: Dict[str, int] = field(default_factory=dict)
    python_version: str = ""
    numpy_version: Optional[str] = None
    platform: str = ""
    hostname: str = ""
    process_id: int = 0
    seed_sequence: List[SeedRecord] = field(default_factory=list)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Initialize system information and provenance hash."""
        import platform as plat
        import socket

        self.python_version = plat.python_version()
        self.platform = plat.platform()
        try:
            self.hostname = socket.gethostname()
        except Exception:
            self.hostname = "unknown"
        self.process_id = os.getpid()

        if NUMPY_AVAILABLE:
            self.numpy_version = np.__version__

        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this context."""
        content = (
            f"{self.context_id}|{self.created_at.isoformat()}|"
            f"{self.global_seed}|{self.python_version}|"
            f"{self.process_id}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "global_seed": self.global_seed,
            "domain_seeds": self.domain_seeds,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
            "hostname": self.hostname,
            "process_id": self.process_id,
            "seed_count": len(self.seed_sequence),
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# MAIN SEED MANAGER CLASS
# =============================================================================


class SeedManager:
    """
    Centralized seed management for deterministic RNG across INSULSCAN.

    Manages seeds for Python's random module and NumPy (if available).
    Provides audit logging, domain isolation, and reproducibility contexts.

    Thread-safe singleton implementation ensures consistent seed management
    across all components of the agent.

    Example:
        >>> manager = SeedManager()
        >>> manager.set_global_seed(42)
        >>>
        >>> # Get reproducibility context for audit
        >>> context = manager.get_reproducibility_context()
        >>>
        >>> # Use domain-specific seed
        >>> with manager.seed_context(SeedDomain.ANOMALY_DETECTION):
        ...     result = model.predict(data)
        >>>
        >>> # Reset all seeds
        >>> manager.reset_seeds()

    Attributes:
        global_seed: The master seed used to derive domain seeds
        enable_audit_logging: Whether to log all seed operations
    """

    _instance: Optional["SeedManager"] = None
    _lock = threading.Lock()

    # Default seed for reproducibility
    DEFAULT_SEED = 42

    def __new__(cls) -> "SeedManager":
        """Singleton implementation with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the seed manager."""
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._global_seed = self.DEFAULT_SEED
        self._domain_seeds: Dict[SeedDomain, int] = {}
        self._seed_history: List[SeedRecord] = []
        self._current_context: Optional[ReproducibilityContext] = None
        self._state_lock = threading.RLock()
        self._enable_audit_logging = True
        self._max_history_size = 10000

        # Initialize all RNG systems
        self._initialize_seeds()

        logger.info(
            f"SeedManager initialized: global_seed={self._global_seed}, "
            f"numpy_available={NUMPY_AVAILABLE}"
        )

    def _initialize_seeds(self) -> None:
        """Initialize all RNG systems with the global seed."""
        self._set_python_seed(self._global_seed)
        self._set_numpy_seed(self._global_seed)
        self._domain_seeds.clear()
        self._record_seed_operation(
            SeedDomain.GLOBAL,
            self._global_seed,
            "initialize",
            "SeedManager.__init__"
        )

    def _set_python_seed(self, seed: int) -> None:
        """Set Python random module seed."""
        random.seed(seed)

    def _set_numpy_seed(self, seed: int) -> None:
        """Set NumPy random seed if available."""
        if NUMPY_AVAILABLE:
            np.random.seed(seed)

    def _record_seed_operation(
        self,
        domain: SeedDomain,
        seed: int,
        operation: str,
        caller: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SeedRecord:
        """Record a seed operation for audit trail."""
        record = SeedRecord(
            domain=domain,
            seed_value=seed,
            operation=operation,
            caller=caller,
            context=context or {}
        )

        with self._state_lock:
            self._seed_history.append(record)
            # Trim history if needed
            if len(self._seed_history) > self._max_history_size:
                self._seed_history = self._seed_history[-self._max_history_size // 2:]

        if self._enable_audit_logging:
            logger.debug(
                f"Seed operation: {operation} domain={domain.value} "
                f"seed={seed} caller={caller}"
            )

        return record

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def global_seed(self) -> int:
        """Get the current global seed."""
        return self._global_seed

    @property
    def enable_audit_logging(self) -> bool:
        """Check if audit logging is enabled."""
        return self._enable_audit_logging

    @enable_audit_logging.setter
    def enable_audit_logging(self, value: bool) -> None:
        """Enable or disable audit logging."""
        self._enable_audit_logging = value

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def set_global_seed(self, seed: int, caller: str = "unknown") -> None:
        """
        Set the global seed for all RNG systems.

        This resets all domain seeds and reinitializes all RNG systems
        with the new global seed.

        Args:
            seed: The seed value to use (must be non-negative)
            caller: Identifier for who set the seed (for audit)

        Example:
            >>> manager.set_global_seed(42, caller="main")
        """
        if seed < 0:
            raise ValueError(f"Seed must be non-negative, got {seed}")

        with self._state_lock:
            self._global_seed = seed
            self._set_python_seed(seed)
            self._set_numpy_seed(seed)
            self._domain_seeds.clear()
            self._record_seed_operation(
                SeedDomain.GLOBAL,
                seed,
                "set_global_seed",
                caller
            )

        logger.info(f"Global seed set to {seed} by {caller}")

    def get_domain_seed(self, domain: SeedDomain, caller: str = "unknown") -> int:
        """
        Get a deterministic seed for a specific domain.

        Domain seeds are derived from the global seed using SHA-256 hashing
        to ensure reproducibility while allowing domain isolation.

        Args:
            domain: The domain requesting a seed
            caller: Identifier for audit

        Returns:
            A deterministic seed for this domain

        Example:
            >>> seed = manager.get_domain_seed(SeedDomain.ANOMALY_DETECTION)
        """
        with self._state_lock:
            if domain not in self._domain_seeds:
                domain_seed = self._derive_domain_seed(domain)
                self._domain_seeds[domain] = domain_seed
                self._record_seed_operation(
                    domain,
                    domain_seed,
                    "derive_domain_seed",
                    caller
                )
            return self._domain_seeds[domain]

    def _derive_domain_seed(self, domain: SeedDomain) -> int:
        """
        Derive a domain-specific seed from the global seed.

        Uses SHA-256 hashing to create a deterministic but unique
        seed for each domain.
        """
        content = f"{self._global_seed}|{domain.value}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        # Use first 4 bytes, ensuring it fits in Python's random.seed() range
        return int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)

    def reset_seeds(self, caller: str = "unknown") -> None:
        """
        Reset all seeds to the global seed.

        Clears domain seeds and reinitializes all RNG systems.

        Args:
            caller: Identifier for audit

        Example:
            >>> manager.reset_seeds(caller="test_setup")
        """
        with self._state_lock:
            self._set_python_seed(self._global_seed)
            self._set_numpy_seed(self._global_seed)
            self._domain_seeds.clear()
            self._record_seed_operation(
                SeedDomain.GLOBAL,
                self._global_seed,
                "reset_seeds",
                caller
            )

        logger.info(f"All seeds reset to global_seed={self._global_seed} by {caller}")

    def get_reproducibility_context(self) -> ReproducibilityContext:
        """
        Get complete reproducibility context for audit trail.

        Returns a snapshot of all information needed to reproduce
        the current random state.

        Returns:
            ReproducibilityContext with all seed information

        Example:
            >>> context = manager.get_reproducibility_context()
            >>> print(context.to_dict())
        """
        with self._state_lock:
            context = ReproducibilityContext(
                global_seed=self._global_seed,
                domain_seeds={d.value: s for d, s in self._domain_seeds.items()},
                seed_sequence=list(self._seed_history[-100:]),
            )
            self._current_context = context
            return context

    @contextmanager
    def seed_context(
        self,
        domain: SeedDomain,
        caller: str = "unknown"
    ) -> Generator[int, None, None]:
        """
        Context manager for domain-specific seeding.

        Sets the RNG to the domain seed on entry and restores
        the previous state on exit.

        Args:
            domain: The domain to use
            caller: Identifier for audit

        Yields:
            The domain seed being used

        Example:
            >>> with manager.seed_context(SeedDomain.ANOMALY_DETECTION) as seed:
            ...     # All random operations here use the domain seed
            ...     result = model.predict(data)
        """
        with self._state_lock:
            # Save current state
            python_state = random.getstate()
            numpy_state = np.random.get_state() if NUMPY_AVAILABLE else None

            # Set domain seed
            domain_seed = self.get_domain_seed(domain, caller)
            self._set_python_seed(domain_seed)
            self._set_numpy_seed(domain_seed)
            self._record_seed_operation(
                domain,
                domain_seed,
                "enter_context",
                caller
            )

        try:
            yield domain_seed
        finally:
            with self._state_lock:
                # Restore previous state
                random.setstate(python_state)
                if NUMPY_AVAILABLE and numpy_state is not None:
                    np.random.set_state(numpy_state)
                self._record_seed_operation(
                    domain,
                    domain_seed,
                    "exit_context",
                    caller
                )

    @contextmanager
    def temporary_seed(
        self,
        seed: int,
        caller: str = "unknown"
    ) -> Generator[int, None, None]:
        """
        Context manager for temporary seed override.

        Temporarily sets a specific seed and restores the previous
        state on exit. Useful for testing or one-off operations.

        Args:
            seed: The temporary seed to use
            caller: Identifier for audit

        Yields:
            The temporary seed being used

        Example:
            >>> with manager.temporary_seed(12345) as seed:
            ...     # All random operations here use seed 12345
            ...     result = generate_test_data()
        """
        with self._state_lock:
            # Save current state
            python_state = random.getstate()
            numpy_state = np.random.get_state() if NUMPY_AVAILABLE else None

            # Set temporary seed
            self._set_python_seed(seed)
            self._set_numpy_seed(seed)
            self._record_seed_operation(
                SeedDomain.GLOBAL,
                seed,
                "enter_temporary",
                caller
            )

        try:
            yield seed
        finally:
            with self._state_lock:
                # Restore previous state
                random.setstate(python_state)
                if NUMPY_AVAILABLE and numpy_state is not None:
                    np.random.set_state(numpy_state)
                self._record_seed_operation(
                    SeedDomain.GLOBAL,
                    seed,
                    "exit_temporary",
                    caller
                )

    def seed_from_run_id(self, run_id: str, caller: str = "unknown") -> int:
        """
        Generate a deterministic seed from a run ID.

        Useful for ensuring reproducibility based on unique run identifiers.

        Args:
            run_id: Unique identifier for the processing run
            caller: Identifier for audit

        Returns:
            A deterministic seed derived from the run ID

        Example:
            >>> seed = manager.seed_from_run_id("analysis-2024-001")
        """
        content = f"{self._global_seed}|run|{run_id}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)

        self._record_seed_operation(
            SeedDomain.GLOBAL,
            seed,
            "seed_from_run_id",
            caller,
            {"run_id": run_id}
        )

        return seed

    def seed_from_asset_id(
        self,
        asset_id: str,
        caller: str = "unknown"
    ) -> int:
        """
        Generate a deterministic seed from an asset ID.

        Useful for ensuring consistent analysis for specific assets.

        Args:
            asset_id: Insulation asset identifier
            caller: Identifier for audit

        Returns:
            A deterministic seed for this asset

        Example:
            >>> seed = manager.seed_from_asset_id("INS-001")
        """
        content = f"{self._global_seed}|asset|{asset_id}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)

        self._record_seed_operation(
            SeedDomain.ANOMALY_DETECTION,
            seed,
            "seed_from_asset_id",
            caller,
            {"asset_id": asset_id}
        )

        return seed

    # =========================================================================
    # AUDIT AND STATUS METHODS
    # =========================================================================

    def get_seed_history(
        self,
        limit: int = 100,
        domain: Optional[SeedDomain] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent seed operation history.

        Args:
            limit: Maximum number of records to return
            domain: Optional filter by domain

        Returns:
            List of seed operation records as dictionaries
        """
        with self._state_lock:
            records = self._seed_history[-limit:]
            if domain is not None:
                records = [r for r in records if r.domain == domain]
            return [r.to_dict() for r in records]

    def get_status(self) -> Dict[str, Any]:
        """
        Get current seed manager status.

        Returns:
            Dictionary with current status information
        """
        with self._state_lock:
            return {
                "global_seed": self._global_seed,
                "domain_seeds": {d.value: s for d, s in self._domain_seeds.items()},
                "numpy_available": NUMPY_AVAILABLE,
                "audit_logging_enabled": self._enable_audit_logging,
                "history_size": len(self._seed_history),
                "max_history_size": self._max_history_size,
            }

    def clear_history(self) -> None:
        """Clear seed history (for memory management)."""
        with self._state_lock:
            self._seed_history.clear()
        logger.info("Seed history cleared")


# =============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# =============================================================================


_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """
    Get the global SeedManager singleton.

    Returns:
        The global SeedManager instance

    Example:
        >>> manager = get_seed_manager()
        >>> manager.set_global_seed(42)
    """
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = SeedManager()
    return _seed_manager


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def set_global_seed(seed: int, caller: str = "unknown") -> None:
    """
    Set the global seed for all RNG systems.

    Convenience function that uses the global SeedManager.

    Args:
        seed: The seed value to use
        caller: Identifier for audit
    """
    get_seed_manager().set_global_seed(seed, caller)


def reset_seeds(caller: str = "unknown") -> None:
    """
    Reset all seeds to the global seed.

    Convenience function that uses the global SeedManager.

    Args:
        caller: Identifier for audit
    """
    get_seed_manager().reset_seeds(caller)


def get_reproducibility_context() -> ReproducibilityContext:
    """
    Get complete reproducibility context for audit trail.

    Convenience function that uses the global SeedManager.

    Returns:
        ReproducibilityContext with all seed information
    """
    return get_seed_manager().get_reproducibility_context()


def seed_context(
    domain: SeedDomain,
    caller: str = "unknown"
) -> contextmanager:
    """
    Context manager for domain-specific seeding.

    Convenience function that uses the global SeedManager.

    Args:
        domain: The domain to use
        caller: Identifier for audit

    Returns:
        Context manager yielding the domain seed
    """
    return get_seed_manager().seed_context(domain, caller)


# =============================================================================
# DECORATOR
# =============================================================================


def deterministic(
    domain: SeedDomain = SeedDomain.GLOBAL
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to make a function deterministic using domain seeding.

    The decorated function will execute with deterministic random state
    based on the specified domain.

    Args:
        domain: The seed domain to use

    Returns:
        Decorator function

    Example:
        >>> @deterministic(SeedDomain.ANOMALY_DETECTION)
        ... def detect_anomalies(data: np.ndarray) -> List[bool]:
        ...     return ml_model.predict(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            manager = get_seed_manager()
            with manager.seed_context(domain, func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Core class
    "SeedManager",
    # Data classes
    "SeedRecord",
    "ReproducibilityContext",
    # Enums
    "SeedDomain",
    # Singleton accessor
    "get_seed_manager",
    # Convenience functions
    "set_global_seed",
    "reset_seeds",
    "get_reproducibility_context",
    "seed_context",
    # Decorators
    "deterministic",
    # Constants
    "NUMPY_AVAILABLE",
]
