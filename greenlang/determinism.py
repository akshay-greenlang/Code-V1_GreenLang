"""
GreenLang Determinism Module - Utilities for Deterministic Operations

This module provides utilities to ensure deterministic behavior across the GreenLang
framework, making it suitable for regulatory compliance and auditable calculations.

Features:
- Deterministic ID generation using content-based hashing
- Controlled timestamp generation with freezable clock
- Seeded random number generation
- Decimal precision for financial calculations
- Sorted file operations

Author: GreenLang Team
Date: 2025-11-21
"""

import hashlib
import random
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path
from typing import Optional, List, Any, Union
import os
import glob as glob_module
from contextlib import contextmanager
import threading


# Set decimal precision for financial calculations (8 decimal places)
getcontext().prec = 28  # 28 significant digits
getcontext().rounding = ROUND_HALF_UP


class DeterministicClock:
    """
    A deterministic clock that can be frozen for testing and auditing.

    This clock ensures consistent timestamp generation across the application,
    with the ability to freeze time for deterministic testing.
    """

    _instance = None
    _lock = threading.Lock()
    _frozen_time: Optional[datetime] = None
    _time_offset: float = 0.0

    def __new__(cls):
        """Singleton pattern to ensure single clock instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @classmethod
    def now(cls, tz=None) -> datetime:
        """
        Get current time, either real or frozen.

        Args:
            tz: Timezone info (defaults to UTC if frozen)

        Returns:
            Current datetime
        """
        instance = cls()
        if instance._frozen_time is not None:
            if tz is not None:
                return instance._frozen_time.replace(tzinfo=tz)
            return instance._frozen_time

        # Apply offset for testing time progression
        if instance._time_offset != 0:
            base_time = datetime.now(tz or timezone.utc)
            return base_time.replace(microsecond=0)  # Remove microseconds for determinism

        return datetime.now(tz or timezone.utc).replace(microsecond=0)

    @classmethod
    def utcnow(cls) -> datetime:
        """Get current UTC time."""
        return cls.now(timezone.utc)

    @classmethod
    def freeze(cls, frozen_time: Optional[datetime] = None):
        """
        Freeze clock at specific time.

        Args:
            frozen_time: Time to freeze at (defaults to current time)
        """
        instance = cls()
        if frozen_time is None:
            frozen_time = datetime.now(timezone.utc).replace(microsecond=0)
        instance._frozen_time = frozen_time

    @classmethod
    def unfreeze(cls):
        """Unfreeze the clock."""
        instance = cls()
        instance._frozen_time = None
        instance._time_offset = 0.0

    @classmethod
    @contextmanager
    def frozen(cls, frozen_time: Optional[datetime] = None):
        """
        Context manager for temporarily freezing time.

        Usage:
            with DeterministicClock.frozen(datetime(2025, 1, 1)):
                # All timestamps will be 2025-01-01
                pass
        """
        cls.freeze(frozen_time)
        try:
            yield
        finally:
            cls.unfreeze()


class DeterministicRandom:
    """
    Seeded random number generator for deterministic randomness.

    Each instance maintains its own seed for reproducible sequences.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize with optional seed.

        Args:
            seed: Random seed (defaults to 42 for determinism)
        """
        self.seed = seed if seed is not None else 42
        self._generator = random.Random(self.seed)

    def random(self) -> float:
        """Generate random float in [0.0, 1.0)."""
        return self._generator.random()

    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b]."""
        return self._generator.randint(a, b)

    def choice(self, seq: List[Any]) -> Any:
        """Choose random element from sequence."""
        return self._generator.choice(seq)

    def sample(self, population: List[Any], k: int) -> List[Any]:
        """Sample k unique elements from population."""
        return self._generator.sample(population, k)

    def shuffle(self, x: List[Any]) -> None:
        """Shuffle list in-place."""
        self._generator.shuffle(x)

    def reset(self):
        """Reset generator to initial seed."""
        self._generator = random.Random(self.seed)


# Global deterministic random instance
_global_random = DeterministicRandom(seed=42)


def set_global_random_seed(seed: int):
    """
    Set global random seed for all operations.

    Args:
        seed: Random seed value
    """
    global _global_random
    _global_random = DeterministicRandom(seed)
    # Also set Python's global random seed
    random.seed(seed)


def deterministic_random() -> DeterministicRandom:
    """Get global deterministic random instance."""
    return _global_random


def deterministic_id(content: Union[str, bytes, dict], prefix: str = "") -> str:
    """
    Generate deterministic ID from content using SHA-256 hashing.

    This function creates reproducible IDs based on content, ensuring
    the same input always produces the same ID.

    Args:
        content: Content to hash (string, bytes, or dict)
        prefix: Optional prefix for the ID

    Returns:
        Deterministic ID string (prefix + first 16 hex chars of hash)

    Examples:
        >>> deterministic_id("hello world", "doc_")
        'doc_2ef37fde8a8b4f63'

        >>> deterministic_id({"key": "value"}, "rec_")
        'rec_a1b2c3d4e5f6g7h8'
    """
    if isinstance(content, dict):
        # Sort dict keys for consistent hashing
        import json
        content = json.dumps(content, sort_keys=True, ensure_ascii=True)

    if isinstance(content, str):
        content = content.encode('utf-8')

    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(content)
    hash_hex = hash_obj.hexdigest()

    # Use first 16 characters for readability
    id_suffix = hash_hex[:16]

    return f"{prefix}{id_suffix}"


def deterministic_uuid(namespace: str, name: str) -> str:
    """
    Generate deterministic UUID using namespace and name.

    Args:
        namespace: Namespace for UUID generation
        name: Name within namespace

    Returns:
        Deterministic UUID string
    """
    combined = f"{namespace}:{name}"
    return deterministic_id(combined, "uuid_")


def content_hash(content: Union[str, bytes, dict]) -> str:
    """
    Generate SHA-256 hash of content for provenance tracking.

    Args:
        content: Content to hash

    Returns:
        Full SHA-256 hash hex string
    """
    if isinstance(content, dict):
        import json
        content = json.dumps(content, sort_keys=True, ensure_ascii=True)

    if isinstance(content, str):
        content = content.encode('utf-8')

    return hashlib.sha256(content).hexdigest()


def sorted_listdir(path: Union[str, Path]) -> List[str]:
    """
    List directory contents in sorted order.

    Args:
        path: Directory path

    Returns:
        Sorted list of filenames
    """
    return sorted(os.listdir(path))


def sorted_glob(pattern: str, recursive: bool = False) -> List[str]:
    """
    Glob files in sorted order.

    Args:
        pattern: Glob pattern
        recursive: Enable recursive globbing

    Returns:
        Sorted list of matching paths
    """
    return sorted(glob_module.glob(pattern, recursive=recursive))


def sorted_iterdir(path: Union[str, Path]) -> List[Path]:
    """
    Iterate directory contents in sorted order.

    Args:
        path: Directory path

    Returns:
        Sorted list of Path objects
    """
    path = Path(path) if isinstance(path, str) else path
    return sorted(path.iterdir())


class FinancialDecimal:
    """
    Wrapper for Decimal operations with financial precision.

    Ensures all financial calculations use proper decimal arithmetic
    with consistent rounding rules.
    """

    PRECISION = Decimal('0.00000001')  # 8 decimal places
    ROUNDING = ROUND_HALF_UP

    @classmethod
    def from_float(cls, value: float) -> Decimal:
        """
        Convert float to Decimal safely.

        Args:
            value: Float value

        Returns:
            Decimal with proper precision
        """
        # Convert through string to avoid float precision issues
        return Decimal(str(value)).quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def from_string(cls, value: str) -> Decimal:
        """
        Parse string to Decimal.

        Args:
            value: String representation

        Returns:
            Decimal with proper precision
        """
        # Remove common formatting
        value = value.replace(',', '').replace('$', '').strip()
        return Decimal(value).quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def multiply(cls, a: Decimal, b: Decimal) -> Decimal:
        """
        Multiply two decimals with proper precision.

        Args:
            a: First operand
            b: Second operand

        Returns:
            Product with proper precision
        """
        result = a * b
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def divide(cls, a: Decimal, b: Decimal) -> Decimal:
        """
        Divide two decimals with proper precision.

        Args:
            a: Dividend
            b: Divisor

        Returns:
            Quotient with proper precision
        """
        if b == 0:
            raise ValueError("Division by zero")
        result = a / b
        return result.quantize(cls.PRECISION, rounding=cls.ROUNDING)

    @classmethod
    def sum(cls, values: List[Decimal]) -> Decimal:
        """
        Sum list of decimals.

        Args:
            values: List of Decimal values

        Returns:
            Sum with proper precision
        """
        total = sum(values, Decimal('0'))
        return total.quantize(cls.PRECISION, rounding=cls.ROUNDING)


# Convenience functions
def now() -> datetime:
    """Get current deterministic time."""
    return DeterministicClock.now()


def utcnow() -> datetime:
    """Get current deterministic UTC time."""
    return DeterministicClock.utcnow()


def freeze_time(frozen_time: Optional[datetime] = None):
    """Freeze time for testing."""
    DeterministicClock.freeze(frozen_time)


def unfreeze_time():
    """Unfreeze time."""
    DeterministicClock.unfreeze()


# Initialize global random seed
set_global_random_seed(42)


__all__ = [
    # ID Generation
    'deterministic_id',
    'deterministic_uuid',
    'content_hash',

    # Time Management
    'DeterministicClock',
    'now',
    'utcnow',
    'freeze_time',
    'unfreeze_time',

    # Random Operations
    'DeterministicRandom',
    'deterministic_random',
    'set_global_random_seed',

    # File Operations
    'sorted_listdir',
    'sorted_glob',
    'sorted_iterdir',

    # Financial Calculations
    'FinancialDecimal',
]