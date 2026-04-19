"""
GreenLang Execution Layer

Consolidated module containing core execution, pipeline, runtime, infrastructure, and resilience.

This module provides:
- Core execution logic (from core/)
- Pipeline orchestration (from pipeline/)
- Runtime environment (from runtime/)
- Infrastructure components (from infrastructure/)
- Resilience patterns (from resilience/)

Sub-modules:
- execution.core: Core execution logic
- execution.pipeline: Checkpointing and idempotency
- execution.runtime: Executor, golden tests, guard
- execution.infrastructure: Agent templates, caching, scheduling
- execution.resilience: Circuit breaker, fallback patterns

Author: GreenLang Team
Version: 2.0.0
"""

__version__ = "2.0.0"

__all__ = []
