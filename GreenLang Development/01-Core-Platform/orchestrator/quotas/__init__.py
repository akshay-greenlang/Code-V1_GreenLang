# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Namespace Concurrency Quotas (FR-024)
=============================================================

Multi-tenant quota management for resource isolation and fair scheduling.

This module provides:
- QuotaConfig: Configuration for namespace quotas
- QuotaUsage: Real-time usage tracking per namespace
- QuotaManager: Thread-safe quota enforcement and priority scheduling
- QuotaEvents: Event types for slot acquisition/release/timeout

Example:
    >>> from greenlang.orchestrator.quotas import QuotaManager, QuotaConfig
    >>>
    >>> # Initialize quota manager
    >>> manager = QuotaManager()
    >>>
    >>> # Set namespace quota
    >>> manager.set_quota("production", QuotaConfig(
    ...     max_concurrent_runs=50,
    ...     max_concurrent_steps=200,
    ...     max_queued_runs=100,
    ...     priority_weight=2.0
    ... ))
    >>>
    >>> # Check admission
    >>> if manager.can_submit_run("production"):
    ...     run_id = submit_run(...)
    ...     manager.acquire_run_slot("production", run_id)

Author: GreenLang Framework Team
Date: January 2026
GL-FOUND-X-001: FR-024 Namespace Concurrency Quotas
Status: Production Ready
"""

from greenlang.orchestrator.quotas.manager import (
    QuotaConfig,
    QuotaUsage,
    QuotaManager,
    QuotaEvent,
    QuotaEventType,
    QueuedRun,
    QuotaMetrics,
)

__all__ = [
    "QuotaConfig",
    "QuotaUsage",
    "QuotaManager",
    "QuotaEvent",
    "QuotaEventType",
    "QueuedRun",
    "QuotaMetrics",
]

__version__ = "1.0.0"
