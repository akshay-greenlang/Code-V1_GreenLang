# -*- coding: utf-8 -*-
"""
Feature Flags Analytics - INFRA-008

Prometheus metrics, evaluation tracking, and periodic metrics collection
for the GreenLang feature flag system.

Provides:
    - Prometheus metric definitions (Counters, Histograms, Gauges)
    - Helper functions for recording evaluation events
    - Background MetricsCollector for periodic gauge updates

Example:
    >>> from greenlang.infrastructure.feature_flags.analytics import (
    ...     record_evaluation,
    ...     record_cache_event,
    ...     MetricsCollector,
    ... )
    >>> record_evaluation("my-flag", enabled=True, environment="prod")
    >>> collector = MetricsCollector(service)
    >>> await collector.start()
"""

from greenlang.infrastructure.feature_flags.analytics.metrics import (
    ff_cache_hit_total,
    ff_cache_miss_total,
    ff_evaluation_duration_seconds,
    ff_evaluation_total,
    ff_flag_state,
    ff_kill_switch_active,
    ff_stale_flags_total,
    ff_storage_errors_total,
    record_cache_event,
    record_evaluation,
    update_flag_state,
)
from greenlang.infrastructure.feature_flags.analytics.collector import (
    MetricsCollector,
)

__all__ = [
    "MetricsCollector",
    "ff_cache_hit_total",
    "ff_cache_miss_total",
    "ff_evaluation_duration_seconds",
    "ff_evaluation_total",
    "ff_flag_state",
    "ff_kill_switch_active",
    "ff_stale_flags_total",
    "ff_storage_errors_total",
    "record_cache_event",
    "record_evaluation",
    "update_flag_state",
]
