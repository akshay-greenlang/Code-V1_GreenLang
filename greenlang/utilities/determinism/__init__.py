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

Organization:
- clock: Time management and freezable clock
- uuid: ID and UUID generation
- random: Seeded random number generation
- decimal: Financial decimal precision
- files: Sorted file operations

Author: GreenLang Team
Date: 2025-11-21
"""

# Import all components from separated modules
from greenlang.utilities.determinism.clock import (
    DeterministicClock,
    now,
    utcnow,
    freeze_time,
    unfreeze_time,
)

from greenlang.utilities.determinism.uuid import (
    deterministic_id,
    deterministic_uuid,
    content_hash,
)

from greenlang.utilities.determinism.random import (
    DeterministicRandom,
    deterministic_random,
    set_global_random_seed,
)

from greenlang.utilities.determinism.decimal import (
    FinancialDecimal,
    safe_decimal,
    safe_decimal_multiply,
    safe_decimal_divide,
    safe_decimal_add,
    safe_decimal_sum,
    round_for_reporting,
)

from greenlang.utilities.determinism.files import (
    sorted_listdir,
    sorted_glob,
    sorted_iterdir,
)


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

    # Financial/Decimal Calculations
    'FinancialDecimal',

    # Safe Decimal Helper Functions
    'safe_decimal',
    'safe_decimal_multiply',
    'safe_decimal_divide',
    'safe_decimal_add',
    'safe_decimal_sum',
    'round_for_reporting',
]
