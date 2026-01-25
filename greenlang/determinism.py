"""
GreenLang Determinism Module - Utilities for Deterministic Operations

This module provides utilities to ensure deterministic behavior across the GreenLang
framework, making it suitable for regulatory compliance and auditable calculations.

DEPRECATED: This module is now organized into submodules for better separation of concerns.
Please import from the specific submodules:
- greenlang.determinism.clock - Time management
- greenlang.determinism.uuid - ID generation
- greenlang.determinism.random - Random number generation
- greenlang.determinism.decimal - Financial decimal operations
- greenlang.determinism.files - File operations

This file provides backward-compatible re-exports.

Author: GreenLang Team
Date: 2025-11-21
"""

import warnings

# Show deprecation warning when importing from this module
warnings.warn(
    "Importing from greenlang.utilities.determinism is deprecated. "
    "Please import from specific submodules: "
    "greenlang.determinism.clock, greenlang.determinism.uuid, "
    "greenlang.determinism.random, greenlang.determinism.decimal, "
    "greenlang.determinism.files",
    DeprecationWarning,
    stacklevel=2
)

# Backward-compatible re-exports from separated modules
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
