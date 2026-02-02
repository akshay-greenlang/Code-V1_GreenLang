# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Seed Manager for Deterministic RNG

Implements centralized seed management for reproducible random number generation
across the EmissionsGuardian agent. Essential for audit compliance and debugging
of ML-based fugitive detection components.

Design Principles:
    - Reproducibility: Exact reproduction of any processing run
    - Auditability: Complete seed logging for regulatory compliance
    - Isolation: Independent seed contexts for parallel processing
    - Zero-Hallucination: Deterministic behavior for compliance calculations

Usage:
    This module manages seeds for:
    - Fugitive detection ML models (Isolation Forest, etc.)
    - Monte Carlo simulations for risk analysis
    - Data sampling operations
    - Test data generation

    Note: Compliance calculations NEVER use random numbers.
    Random numbers are only for ML classification/detection support.

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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


class SeedDomain(str, Enum):
    """Domains for seed isolation and management."""
    GLOBAL = "global"
    FUGITIVE_DETECTION = "fugitive_detection"
    ANOMALY_MODEL = "anomaly_model"
    MONTE_CARLO = "monte_carlo"
    DATA_SAMPLING = "data_sampling"
    TEST_GENERATION = "test_generation"
    SIMULATION = "simulation"


@dataclass
class SeedRecord:
    """
    Record of a seed operation for audit trail.
    
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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    domain: SeedDomain = SeedDomain.GLOBAL
    seed_value: int = 0
    operation: str = ""
    caller: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        content = (
            f"{self.record_id}|{self.timestamp.isoformat()}|"
            f"{self.domain.value}|{self.seed_value}|"
            f"{self.operation}|{self.caller}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class ReproducibilityContext:
    """
    Complete context needed to reproduce a processing run.
    
    Contains all information necessary to exactly reproduce
    the random state at any point in processing.
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
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
        import platform as plat
        import socket
        self.python_version = plat.python_version()
        self.platform = plat.platform()
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        if NUMPY_AVAILABLE:
            self.numpy_version = np.__version__
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        content = (
            f"{self.context_id}|{self.created_at.isoformat()}|"
            f"{self.global_seed}|{self.python_version}|"
            f"{self.process_id}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
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


class SeedManager:
    """
    Centralized seed management for deterministic RNG across EmissionsGuardian.
    
    Manages seeds for Python's random module and NumPy (if available).
    Provides audit logging, domain isolation, and reproducibility contexts.
    
    Thread-safe singleton implementation.
    
    Example:
        >>> manager = SeedManager()
        >>> manager.set_global_seed(42)
        >>> 
        >>> # Get reproducibility context for audit
        >>> context = manager.get_reproducibility_context()
        >>> 
        >>> # Use domain-specific seed
        >>> with manager.seed_context(SeedDomain.FUGITIVE_DETECTION):
        ...     result = anomaly_detector.detect(data)
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
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._global_seed = self.DEFAULT_SEED
        self._domain_seeds: Dict[SeedDomain, int] = {}
        self._seed_history: List[SeedRecord] = []
        self._current_context: Optional[ReproducibilityContext] = None
        self._state_lock = threading.RLock()
        self._enable_audit_logging = True
        self._initialize_seeds()
        logger.info(f"SeedManager initialized with global_seed={self._global_seed}")

    def _initialize_seeds(self) -> None:
        """Initialize all RNG systems with the global seed."""
        self._set_python_seed(self._global_seed)
        self._set_numpy_seed(self._global_seed)
        self._domain_seeds.clear()
        self._record_seed_operation(SeedDomain.GLOBAL, self._global_seed, "initialize", "SeedManager.__init__")

    def _set_python_seed(self, seed: int) -> None:
        """Set Python random module seed."""
        random.seed(seed)

    def _set_numpy_seed(self, seed: int) -> None:
        """Set NumPy random seed if available."""
        if NUMPY_AVAILABLE:
            np.random.seed(seed)

    def _record_seed_operation(self, domain: SeedDomain, seed: int, operation: str, caller: str,
                               context: Optional[Dict[str, Any]] = None) -> SeedRecord:
        """Record a seed operation for audit trail."""
        record = SeedRecord(domain=domain, seed_value=seed, operation=operation, caller=caller, context=context or {})
        with self._state_lock:
            self._seed_history.append(record)
            if len(self._seed_history) > 10000:
                self._seed_history = self._seed_history[-5000:]
        if self._enable_audit_logging:
            logger.debug(f"Seed operation: {operation} domain={domain.value} seed={seed} caller={caller}")
        return record

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
        self._enable_audit_logging = value

    def set_global_seed(self, seed: int, caller: str = "unknown") -> None:
        """
        Set the global seed for all RNG systems.
        
        Args:
            seed: The seed value to use
            caller: Identifier for who set the seed (for audit)
        """
        with self._state_lock:
            self._global_seed = seed
            self._set_python_seed(seed)
            self._set_numpy_seed(seed)
            self._domain_seeds.clear()
            self._record_seed_operation(SeedDomain.GLOBAL, seed, "set_global_seed", caller)
            logger.info(f"Global seed set to {seed} by {caller}")

    def get_domain_seed(self, domain: SeedDomain, caller: str = "unknown") -> int:
        """
        Get a deterministic seed for a specific domain.
        
        Domain seeds are derived from the global seed to ensure
        reproducibility while allowing domain isolation.
        
        Args:
            domain: The domain requesting a seed
            caller: Identifier for audit
            
        Returns:
            A deterministic seed for this domain
        """
        with self._state_lock:
            if domain not in self._domain_seeds:
                domain_seed = self._derive_domain_seed(domain)
                self._domain_seeds[domain] = domain_seed
                self._record_seed_operation(domain, domain_seed, "derive_domain_seed", caller)
            return self._domain_seeds[domain]

    def _derive_domain_seed(self, domain: SeedDomain) -> int:
        """Derive a domain-specific seed from the global seed."""
        content = f"{self._global_seed}|{domain.value}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)


    def reset_seeds(self, caller: str = "unknown") -> None:
        """
        Reset all seeds to the global seed.
        
        Clears domain seeds and reinitializes all RNG systems.
        
        Args:
            caller: Identifier for audit
        """
        with self._state_lock:
            self._set_python_seed(self._global_seed)
            self._set_numpy_seed(self._global_seed)
            self._domain_seeds.clear()
            self._record_seed_operation(SeedDomain.GLOBAL, self._global_seed, "reset_seeds", caller)
            logger.info(f"All seeds reset to global_seed={self._global_seed} by {caller}")

    def get_reproducibility_context(self) -> ReproducibilityContext:
        """
        Get complete reproducibility context for audit trail.
        
        Returns a snapshot of all information needed to reproduce
        the current random state.
        
        Returns:
            ReproducibilityContext with all seed information
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
    def seed_context(self, domain: SeedDomain, caller: str = "unknown") -> Generator[int, None, None]:
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
            >>> with seed_manager.seed_context(SeedDomain.FUGITIVE_DETECTION) as seed:
            ...     # All random operations here use the domain seed
            ...     result = model.predict(data)
        """
        with self._state_lock:
            python_state = random.getstate()
            numpy_state = np.random.get_state() if NUMPY_AVAILABLE else None
            domain_seed = self.get_domain_seed(domain, caller)
            self._set_python_seed(domain_seed)
            self._set_numpy_seed(domain_seed)
            self._record_seed_operation(domain, domain_seed, "enter_context", caller)

        try:
            yield domain_seed
        finally:
            with self._state_lock:
                random.setstate(python_state)
                if NUMPY_AVAILABLE and numpy_state is not None:
                    np.random.set_state(numpy_state)
                self._record_seed_operation(domain, domain_seed, "exit_context", caller)

    @contextmanager
    def temporary_seed(self, seed: int, caller: str = "unknown") -> Generator[int, None, None]:
        """
        Context manager for temporary seed override.
        
        Temporarily sets a specific seed and restores the previous
        state on exit.
        
        Args:
            seed: The temporary seed to use
            caller: Identifier for audit
            
        Yields:
            The temporary seed being used
        """
        with self._state_lock:
            python_state = random.getstate()
            numpy_state = np.random.get_state() if NUMPY_AVAILABLE else None
            self._set_python_seed(seed)
            self._set_numpy_seed(seed)
            self._record_seed_operation(SeedDomain.GLOBAL, seed, "enter_temporary", caller)

        try:
            yield seed
        finally:
            with self._state_lock:
                random.setstate(python_state)
                if NUMPY_AVAILABLE and numpy_state is not None:
                    np.random.set_state(numpy_state)
                self._record_seed_operation(SeedDomain.GLOBAL, seed, "exit_temporary", caller)

    def seed_from_run_id(self, run_id: str, caller: str = "unknown") -> int:
        """
        Generate a deterministic seed from a run ID.
        
        Useful for ensuring reproducibility based on unique run identifiers.
        
        Args:
            run_id: Unique identifier for the processing run
            caller: Identifier for audit
            
        Returns:
            A deterministic seed derived from the run ID
        """
        content = f"{self._global_seed}|run|{run_id}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big") % (2**31 - 1)
        self._record_seed_operation(SeedDomain.GLOBAL, seed, "seed_from_run_id", caller, {"run_id": run_id})
        return seed

    def get_seed_history(self, limit: int = 100, domain: Optional[SeedDomain] = None) -> List[Dict[str, Any]]:
        """
        Get recent seed operation history.
        
        Args:
            limit: Maximum number of records to return
            domain: Optional filter by domain
            
        Returns:
            List of seed operation records
        """
        with self._state_lock:
            records = self._seed_history[-limit:]
            if domain is not None:
                records = [r for r in records if r.domain == domain]
            return [r.to_dict() for r in records]

    def get_status(self) -> Dict[str, Any]:
        """Get current seed manager status."""
        with self._state_lock:
            return {
                "global_seed": self._global_seed,
                "domain_seeds": {d.value: s for d, s in self._domain_seeds.items()},
                "numpy_available": NUMPY_AVAILABLE,
                "audit_logging_enabled": self._enable_audit_logging,
                "history_size": len(self._seed_history),
            }

    def clear_history(self) -> None:
        """Clear seed history (for memory management)."""
        with self._state_lock:
            self._seed_history.clear()
            logger.info("Seed history cleared")


# Module-level singleton accessor
_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """
    Get the global SeedManager singleton.
    
    Returns:
        The global SeedManager instance
    """
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = SeedManager()
    return _seed_manager


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


def seed_context(domain: SeedDomain, caller: str = "unknown") -> contextmanager:
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


def deterministic(domain: SeedDomain = SeedDomain.GLOBAL) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to make a function deterministic using domain seeding.
    
    The decorated function will execute with deterministic random state
    based on the specified domain.
    
    Args:
        domain: The seed domain to use
        
    Returns:
        Decorator function
        
    Example:
        >>> @deterministic(SeedDomain.FUGITIVE_DETECTION)
        >>> def detect_anomalies(data: np.ndarray) -> np.ndarray:
        ...     return isolation_forest.fit_predict(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        from functools import wraps
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            manager = get_seed_manager()
            with manager.seed_context(domain, func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


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
