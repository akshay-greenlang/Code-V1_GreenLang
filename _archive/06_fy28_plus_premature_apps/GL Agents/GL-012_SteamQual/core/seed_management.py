# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL - Seed Management Module

This module provides comprehensive seed management for reproducible steam
quality calculations. Ensures deterministic behavior for all numerical
operations used in quality estimation and control.

Reproducibility Guarantees:
    - Same seed produces identical quality calculations
    - All numpy/scipy random state is controlled
    - Provenance hashes include seed information
    - Test reproducibility is guaranteed across runs

Standards Compliance:
    - EU AI Act (Reproducibility requirements)
    - ISO 22514-7 (Statistical methods - Capability indices)
    - GreenLang Zero-Hallucination principle

Example:
    >>> from core.seed_management import SeedManager, set_global_seed
    >>> manager = SeedManager(global_seed=42)
    >>> state = manager.apply_all_seeds()
    >>> # All subsequent calculations are deterministic
    >>> print(f"Provenance: {state.provenance_hash}")

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
import hashlib
import json
import logging
import os
import random
import contextlib

logger = logging.getLogger(__name__)


# =============================================================================
# SEED CONFIGURATION
# =============================================================================

@dataclass
class SeedConfig:
    """
    Configuration for deterministic seed management.

    Attributes:
        global_seed: Master seed for all operations (default: 42)
        numpy_seed: Separate seed for NumPy random generators
        estimation_seed: Seed for quality estimation algorithms
        simulation_seed: Seed for Monte Carlo uncertainty analysis
        deterministic_mode: Enable fully deterministic behavior
        log_seed_usage: Log seed application events
    """

    global_seed: int = 42
    numpy_seed: Optional[int] = None
    estimation_seed: Optional[int] = None
    simulation_seed: Optional[int] = None
    control_seed: Optional[int] = None
    deterministic_mode: bool = True
    log_seed_usage: bool = True

    def __post_init__(self) -> None:
        """Validate seed configuration."""
        if not 0 <= self.global_seed <= 2**32 - 1:
            raise ValueError(
                f"global_seed must be in range 0-{2**32-1}, got {self.global_seed}"
            )

        # Validate optional seeds if provided
        for name, seed in [
            ("numpy_seed", self.numpy_seed),
            ("estimation_seed", self.estimation_seed),
            ("simulation_seed", self.simulation_seed),
            ("control_seed", self.control_seed),
        ]:
            if seed is not None and not 0 <= seed <= 2**32 - 1:
                raise ValueError(
                    f"{name} must be in range 0-{2**32-1}, got {seed}"
                )

    def get_effective_seed(self, component: str) -> int:
        """
        Get effective seed for a specific component.

        Args:
            component: Component name (numpy, estimation, simulation, control)

        Returns:
            Seed value to use (component-specific or global)
        """
        seed_map = {
            "numpy": self.numpy_seed,
            "estimation": self.estimation_seed,
            "simulation": self.simulation_seed,
            "control": self.control_seed,
        }
        return seed_map.get(component) or self.global_seed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "global_seed": self.global_seed,
            "numpy_seed": self.numpy_seed,
            "estimation_seed": self.estimation_seed,
            "simulation_seed": self.simulation_seed,
            "control_seed": self.control_seed,
            "deterministic_mode": self.deterministic_mode,
        }


@dataclass
class SeedState:
    """
    Record of applied seed state for provenance tracking.

    Immutable snapshot of the seed configuration at time of application,
    enabling full reproducibility verification.

    Attributes:
        timestamp: When seeds were applied (UTC)
        applied_seeds: Dictionary mapping component to seed value
        deterministic_mode: Whether deterministic mode was enabled
        provenance_hash: SHA-256 hash of seed configuration
        numpy_available: Whether NumPy was available
        scipy_available: Whether SciPy was available
    """

    timestamp: datetime
    applied_seeds: Dict[str, int]
    deterministic_mode: bool
    provenance_hash: str
    numpy_available: bool = False
    scipy_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "applied_seeds": self.applied_seeds,
            "deterministic_mode": self.deterministic_mode,
            "provenance_hash": self.provenance_hash,
            "numpy_available": self.numpy_available,
            "scipy_available": self.scipy_available,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeedState':
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def verify(self) -> bool:
        """
        Verify provenance hash matches current state.

        Returns:
            True if hash matches, False otherwise
        """
        computed_hash = _compute_hash({
            "applied_seeds": self.applied_seeds,
            "deterministic_mode": self.deterministic_mode,
            "numpy_available": self.numpy_available,
            "scipy_available": self.scipy_available,
        })
        return computed_hash == self.provenance_hash


# =============================================================================
# SEED MANAGER
# =============================================================================

class SeedManager:
    """
    Comprehensive seed manager for reproducible steam quality calculations.

    Provides centralized control over all random number generators used
    in the STEAMQUAL system, ensuring deterministic behavior for quality
    estimation, uncertainty analysis, and control algorithms.

    Features:
        - Global seed propagation to Python, NumPy, SciPy
        - Component-specific seed overrides
        - Provenance hash generation for audit trails
        - Context manager for scoped determinism
        - Seed history for reproducibility verification

    Example:
        >>> manager = SeedManager(global_seed=42)
        >>> state = manager.apply_all_seeds()
        >>> print(f"Provenance: {state.provenance_hash}")

        # Use as context manager for scoped operations
        >>> with manager.deterministic_context():
        ...     result = estimate_quality(data)
    """

    VERSION = "1.0.0"
    AGENT_ID = "GL-012"

    def __init__(
        self,
        global_seed: int = 42,
        numpy_seed: Optional[int] = None,
        estimation_seed: Optional[int] = None,
        simulation_seed: Optional[int] = None,
        control_seed: Optional[int] = None,
        deterministic_mode: bool = True,
        auto_apply: bool = False,
    ) -> None:
        """
        Initialize the seed manager.

        Args:
            global_seed: Master seed for all operations (0 to 2^32-1)
            numpy_seed: Optional override for NumPy random state
            estimation_seed: Optional override for estimation algorithms
            simulation_seed: Optional override for Monte Carlo simulations
            control_seed: Optional override for control algorithms
            deterministic_mode: Enable fully deterministic mode
            auto_apply: Automatically apply seeds on initialization
        """
        self.config = SeedConfig(
            global_seed=global_seed,
            numpy_seed=numpy_seed,
            estimation_seed=estimation_seed,
            simulation_seed=simulation_seed,
            control_seed=control_seed,
            deterministic_mode=deterministic_mode,
        )

        self._applied_state: Optional[SeedState] = None
        self._state_history: List[SeedState] = []
        self._numpy_available = False
        self._scipy_available = False

        # Check library availability
        self._check_libraries()

        if auto_apply:
            self.apply_all_seeds()

        logger.info(
            f"SeedManager initialized: global_seed={global_seed}, "
            f"deterministic={deterministic_mode}, "
            f"numpy={self._numpy_available}, scipy={self._scipy_available}"
        )

    def _check_libraries(self) -> None:
        """Check availability of numerical libraries."""
        try:
            import numpy as np
            self._numpy_available = True
            logger.debug("NumPy available for seeding")
        except ImportError:
            logger.debug("NumPy not available")

        try:
            import scipy
            self._scipy_available = True
            logger.debug("SciPy available")
        except ImportError:
            logger.debug("SciPy not available")

    @property
    def is_applied(self) -> bool:
        """Check if seeds have been applied."""
        return self._applied_state is not None

    @property
    def applied_state(self) -> Optional[SeedState]:
        """Get the current applied seed state."""
        return self._applied_state

    @property
    def state_history(self) -> List[SeedState]:
        """Get history of applied seed states."""
        return self._state_history.copy()

    def get_effective_seed(self, component: str) -> int:
        """
        Get effective seed for a specific component.

        Args:
            component: Component name (numpy, estimation, simulation, control)

        Returns:
            Seed value to use for that component
        """
        return self.config.get_effective_seed(component)

    def apply_all_seeds(self) -> SeedState:
        """
        Apply all seeds to their respective libraries.

        Sets random state for:
        - Python random module
        - NumPy random (if available)
        - PYTHONHASHSEED environment variable (deterministic mode)

        Returns:
            SeedState record of applied seeds

        Example:
            >>> manager = SeedManager(global_seed=42)
            >>> state = manager.apply_all_seeds()
            >>> print(f"Seeds: {state.applied_seeds}")
        """
        timestamp = datetime.now(timezone.utc)
        applied_seeds: Dict[str, int] = {}

        # 1. Python random module
        random.seed(self.config.global_seed)
        applied_seeds["python_random"] = self.config.global_seed

        # 2. NumPy random state
        if self._numpy_available:
            import numpy as np
            np_seed = self.get_effective_seed("numpy")
            np.random.seed(np_seed)
            applied_seeds["numpy_legacy"] = np_seed

            # Also create new-style generator (NumPy >= 1.17)
            try:
                self._numpy_rng = np.random.default_rng(np_seed)
                applied_seeds["numpy_generator"] = np_seed
            except AttributeError:
                self._numpy_rng = None

        # 3. Component-specific seeds
        applied_seeds["estimation"] = self.get_effective_seed("estimation")
        applied_seeds["simulation"] = self.get_effective_seed("simulation")
        applied_seeds["control"] = self.get_effective_seed("control")

        # 4. Set PYTHONHASHSEED for deterministic dict/set ordering
        if self.config.deterministic_mode:
            os.environ["PYTHONHASHSEED"] = str(self.config.global_seed)
            applied_seeds["pythonhashseed"] = self.config.global_seed

        # 5. Compute provenance hash
        provenance_hash = _compute_hash({
            "applied_seeds": applied_seeds,
            "deterministic_mode": self.config.deterministic_mode,
            "numpy_available": self._numpy_available,
            "scipy_available": self._scipy_available,
        })

        # Create state record
        self._applied_state = SeedState(
            timestamp=timestamp,
            applied_seeds=applied_seeds,
            deterministic_mode=self.config.deterministic_mode,
            provenance_hash=provenance_hash,
            numpy_available=self._numpy_available,
            scipy_available=self._scipy_available,
        )

        # Add to history (bounded)
        self._state_history.append(self._applied_state)
        if len(self._state_history) > 100:
            self._state_history = self._state_history[-100:]

        if self.config.log_seed_usage:
            logger.info(f"Seeds applied: {applied_seeds}")
            logger.info(f"Provenance hash: {provenance_hash[:16]}...")

        return self._applied_state

    def reset_seeds(self) -> SeedState:
        """
        Reset all seeds to their initial state.

        Useful for running multiple reproducible experiments.

        Returns:
            New SeedState after reset
        """
        logger.info("Resetting all seeds")
        return self.apply_all_seeds()

    def get_numpy_rng(self, seed_offset: int = 0) -> Any:
        """
        Get a NumPy random Generator with controlled seed.

        Args:
            seed_offset: Offset to add to base seed for parallel operations

        Returns:
            NumPy random Generator instance

        Raises:
            RuntimeError: If NumPy not available

        Example:
            >>> rng = manager.get_numpy_rng(offset=1)
            >>> samples = rng.normal(0, 1, size=100)
        """
        if not self._numpy_available:
            raise RuntimeError("NumPy not available")

        import numpy as np
        seed = self.get_effective_seed("numpy") + seed_offset
        return np.random.default_rng(seed)

    def get_estimation_rng(self) -> Any:
        """
        Get random generator for quality estimation algorithms.

        Returns:
            NumPy Generator or Python Random instance
        """
        if self._numpy_available:
            return self.get_numpy_rng(seed_offset=1000)
        else:
            rng = random.Random(self.get_effective_seed("estimation"))
            return rng

    def get_simulation_rng(self) -> Any:
        """
        Get random generator for Monte Carlo simulations.

        Returns:
            NumPy Generator or Python Random instance
        """
        if self._numpy_available:
            return self.get_numpy_rng(seed_offset=2000)
        else:
            rng = random.Random(self.get_effective_seed("simulation"))
            return rng

    def get_control_rng(self) -> Any:
        """
        Get random generator for control algorithms.

        Returns:
            NumPy Generator or Python Random instance
        """
        if self._numpy_available:
            return self.get_numpy_rng(seed_offset=3000)
        else:
            rng = random.Random(self.get_effective_seed("control"))
            return rng

    @contextlib.contextmanager
    def deterministic_context(self, seed_offset: int = 0):
        """
        Context manager for deterministic operations.

        Applies seeds on entry and restores on exit (if changed).

        Args:
            seed_offset: Optional seed offset for this context

        Yields:
            SeedState for the context

        Example:
            >>> with manager.deterministic_context() as state:
            ...     result = compute_quality(data)
            ...     print(f"Computed with seed: {state.provenance_hash[:8]}")
        """
        # Store current state
        previous_state = self._applied_state

        # Apply seeds with optional offset
        if seed_offset != 0:
            original_global = self.config.global_seed
            self.config.global_seed += seed_offset

        state = self.apply_all_seeds()

        try:
            yield state
        finally:
            # Restore original seed if modified
            if seed_offset != 0:
                self.config.global_seed = original_global
                self.apply_all_seeds()

    def get_provenance_hash(self) -> str:
        """
        Get provenance hash for current seed state.

        Returns:
            SHA-256 hash (16 chars) of seed configuration
        """
        if self._applied_state is None:
            return _compute_hash({
                "global_seed": self.config.global_seed,
                "status": "not_applied",
            })
        return self._applied_state.provenance_hash

    def get_provenance_info(self) -> Dict[str, Any]:
        """
        Get complete provenance information for audit trails.

        Returns:
            Dictionary with all seed and state information
        """
        info = {
            "agent_id": self.AGENT_ID,
            "manager_version": self.VERSION,
            "config": self.config.to_dict(),
            "numpy_available": self._numpy_available,
            "scipy_available": self._scipy_available,
            "is_applied": self.is_applied,
            "history_length": len(self._state_history),
        }

        if self._applied_state:
            info["applied_state"] = self._applied_state.to_dict()

        return info

    def verify_reproducibility(
        self,
        expected_hash: str,
    ) -> bool:
        """
        Verify current state matches expected provenance hash.

        Args:
            expected_hash: Expected provenance hash

        Returns:
            True if current state matches expected hash
        """
        if self._applied_state is None:
            logger.warning("Cannot verify: seeds not applied")
            return False

        matches = self._applied_state.provenance_hash == expected_hash
        if not matches:
            logger.warning(
                f"Provenance mismatch: expected {expected_hash[:16]}..., "
                f"got {self._applied_state.provenance_hash[:16]}..."
            )
        return matches

    def create_child_manager(
        self,
        component: str,
        seed_offset: int = 0,
    ) -> 'SeedManager':
        """
        Create a child seed manager for a specific component.

        Useful for creating independent random streams for parallel
        processing while maintaining reproducibility.

        Args:
            component: Component name for seed lookup
            seed_offset: Additional offset for independence

        Returns:
            New SeedManager instance
        """
        base_seed = self.get_effective_seed(component) + seed_offset
        return SeedManager(
            global_seed=base_seed,
            deterministic_mode=self.config.deterministic_mode,
            auto_apply=True,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SeedManager(global_seed={self.config.global_seed}, "
            f"deterministic={self.config.deterministic_mode}, "
            f"applied={self.is_applied})"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash for provenance tracking.

    Args:
        data: Data to hash

    Returns:
        SHA-256 hash as hex string (16 chars)
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def set_global_seed(seed: int = 42, deterministic: bool = True) -> SeedState:
    """
    Convenience function to set global seed for all libraries.

    Args:
        seed: The seed value to use
        deterministic: Enable deterministic mode

    Returns:
        SeedState record of applied seeds

    Example:
        >>> from core.seed_management import set_global_seed
        >>> state = set_global_seed(42)
        >>> print(f"Seeds applied: {state.provenance_hash[:8]}...")
    """
    manager = SeedManager(global_seed=seed, deterministic_mode=deterministic)
    return manager.apply_all_seeds()


def get_reproducibility_hash(config: Dict[str, Any]) -> str:
    """
    Get a reproducibility hash for a configuration.

    Args:
        config: Dictionary of configuration values

    Returns:
        SHA-256 hash (16 chars) for provenance

    Example:
        >>> hash = get_reproducibility_hash({"seed": 42, "method": "iapws"})
    """
    return _compute_hash(config)


def verify_determinism(
    func: Callable[[], Any],
    iterations: int = 10,
    seed: int = 42,
) -> bool:
    """
    Verify a function produces deterministic output.

    Runs the function multiple times with the same seed and
    verifies all outputs match.

    Args:
        func: Function to test (should be pure with seeded randomness)
        iterations: Number of times to run
        seed: Seed to use

    Returns:
        True if all outputs match

    Example:
        >>> def compute():
        ...     return estimate_quality(data)
        >>> assert verify_determinism(compute, iterations=5, seed=42)
    """
    manager = SeedManager(global_seed=seed, deterministic_mode=True)
    results = []

    for i in range(iterations):
        manager.apply_all_seeds()
        result = func()
        # Hash the result for comparison
        if hasattr(result, 'model_dump'):
            result_hash = _compute_hash(result.model_dump())
        elif hasattr(result, '__dict__'):
            result_hash = _compute_hash(result.__dict__)
        else:
            result_hash = _compute_hash({"value": str(result)})
        results.append(result_hash)

    # Check all hashes match
    all_match = len(set(results)) == 1

    if not all_match:
        logger.error(
            f"Determinism check failed: {len(set(results))} unique results "
            f"from {iterations} runs"
        )

    return all_match


# =============================================================================
# MODULE-LEVEL DEFAULT MANAGER
# =============================================================================

_default_manager: Optional[SeedManager] = None


def get_default_manager() -> SeedManager:
    """
    Get the default seed manager instance.

    Creates one with default settings if not already initialized.

    Returns:
        The default SeedManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = SeedManager(global_seed=42, auto_apply=True)
    return _default_manager


def initialize_default_seeds(seed: int = 42) -> SeedState:
    """
    Initialize the default seed manager with a specific seed.

    Args:
        seed: The global seed to use

    Returns:
        SeedState after applying seeds
    """
    global _default_manager
    _default_manager = SeedManager(global_seed=seed, auto_apply=True)
    if _default_manager._applied_state is None:
        raise RuntimeError("Failed to apply seeds")
    return _default_manager._applied_state


def reset_default_seeds() -> SeedState:
    """
    Reset the default seed manager to initial state.

    Returns:
        SeedState after reset
    """
    manager = get_default_manager()
    return manager.reset_seeds()


# =============================================================================
# DECORATORS FOR DETERMINISTIC FUNCTIONS
# =============================================================================

def deterministic(seed: Optional[int] = None):
    """
    Decorator to ensure a function runs with deterministic seeds.

    Args:
        seed: Optional seed override (uses default if None)

    Returns:
        Decorated function

    Example:
        >>> @deterministic(seed=42)
        ... def estimate_quality(data):
        ...     return compute_estimate(data)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_default_manager()
            if seed is not None:
                temp_manager = SeedManager(global_seed=seed, auto_apply=True)
                result = func(*args, **kwargs)
                # Restore default manager state
                manager.apply_all_seeds()
                return result
            else:
                manager.apply_all_seeds()
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_provenance(func: Callable) -> Callable:
    """
    Decorator to add provenance tracking to a function.

    Logs seed state before and after function execution.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function

    Example:
        >>> @with_provenance
        ... def calculate_quality(data):
        ...     return quality_result
    """
    def wrapper(*args, **kwargs):
        manager = get_default_manager()
        before_hash = manager.get_provenance_hash()

        result = func(*args, **kwargs)

        after_hash = manager.get_provenance_hash()

        logger.debug(
            f"Provenance: {func.__name__} executed with "
            f"seed state {before_hash[:8]}... -> {after_hash[:8]}..."
        )

        return result
    return wrapper
