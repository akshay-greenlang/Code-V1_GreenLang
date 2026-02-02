# -*- coding: utf-8 -*-
"""
GreenLang Pipeline Package

This module provides pipeline utilities including checkpointing, recovery,
and idempotency guarantees for GreenLang pipelines.
"""

from .checkpointing import (
    CheckpointManager,
    CheckpointStrategy,
    CheckpointStatus,
    POSTGRES_AVAILABLE,
    REDIS_AVAILABLE,
)
from .idempotency import (
    IdempotencyManager,
    IdempotencyStatus,
    IdempotentPipelineBase,
)

__all__ = [
    # Checkpointing
    "CheckpointManager",
    "CheckpointStrategy",
    "CheckpointStatus",
    # Availability flags
    "POSTGRES_AVAILABLE",
    "REDIS_AVAILABLE",
    # Idempotency
    "IdempotencyManager",
    "IdempotencyStatus",
    "IdempotentPipelineBase",
]
