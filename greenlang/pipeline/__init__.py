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
    # Idempotency
    "IdempotencyManager",
    "IdempotencyStatus",
    "IdempotentPipelineBase",
]
