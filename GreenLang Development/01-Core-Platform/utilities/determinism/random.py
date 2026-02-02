"""
GreenLang Deterministic Random - Seeded Random Number Generation

This module provides seeded random number generation for reproducible results.

Features:
- Seeded RNG with reproducible sequences
- Global random instance for convenience
- Thread-safe seed management

Author: GreenLang Team
Date: 2025-11-21
"""

import random
from typing import Optional, List, Any


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


# Initialize global random seed on module import
set_global_random_seed(42)


__all__ = [
    'DeterministicRandom',
    'deterministic_random',
    'set_global_random_seed',
]
