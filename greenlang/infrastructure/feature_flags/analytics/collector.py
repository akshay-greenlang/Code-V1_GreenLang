# -*- coding: utf-8 -*-
"""
Feature Flags Metrics Collector - INFRA-008

Background asyncio task that periodically collects and updates Prometheus
gauge metrics for the feature flag system. Runs every 60 seconds by default
and reports:
    - Flag state counts (active, killed, archived, etc.)
    - Stale flag counts
    - Kill switch states

The collector is designed to be started once during service initialization
and stopped during graceful shutdown.

Example:
    >>> from greenlang.infrastructure.feature_flags.analytics.collector import MetricsCollector
    >>> collector = MetricsCollector(service)
    >>> await collector.start()
    >>> # ... application runs ...
    >>> await collector.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from greenlang.infrastructure.feature_flags.analytics.metrics import (
    record_kill_switch,
    update_flag_state,
    update_stale_count,
)
from greenlang.infrastructure.feature_flags.models import FlagStatus

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Background metrics collector for feature flag gauge updates.

    Periodically queries the feature flag service for current state
    and updates Prometheus gauge metrics. This ensures that dashboards
    and alerts have up-to-date flag state information even when flags
    are not being actively evaluated.

    Attributes:
        _service: The FeatureFlagService to query.
        _interval_seconds: Collection interval in seconds.
        _task: The running background task, or None if stopped.
        _running: Whether the collector is currently active.
        _environment: Deployment environment for stale count labels.
    """

    def __init__(
        self,
        service: "FeatureFlagService",  # noqa: F821 - forward reference
        interval_seconds: int = 60,
        environment: str = "dev",
    ) -> None:
        """Initialize the metrics collector.

        Args:
            service: The FeatureFlagService instance to query.
            interval_seconds: How often to collect metrics (default 60s).
            environment: Environment label for stale flag metrics.
        """
        # Import here to avoid circular imports at module level
        from greenlang.infrastructure.feature_flags.service import FeatureFlagService

        self._service: FeatureFlagService = service
        self._interval_seconds = max(10, interval_seconds)  # Minimum 10s
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._environment = environment
        logger.info(
            "MetricsCollector initialized (interval=%ds, env=%s)",
            self._interval_seconds, self._environment,
        )

    async def start(self) -> None:
        """Start the background metrics collection task.

        Idempotent: calling start() when already running is a no-op.
        """
        if self._running:
            logger.debug("MetricsCollector already running")
            return

        self._running = True
        self._task = asyncio.create_task(
            self._collection_loop(),
            name="ff-metrics-collector",
        )
        logger.info("MetricsCollector started")

    async def stop(self) -> None:
        """Stop the background metrics collection task.

        Waits for the current collection cycle to complete before stopping.
        """
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("MetricsCollector stopped")

    @property
    def is_running(self) -> bool:
        """Return whether the collector is currently running."""
        return self._running

    async def _collection_loop(self) -> None:
        """Main collection loop. Runs until stopped.

        Each iteration collects flag states, stale counts, and kill
        switch states. Errors in individual collection steps do not
        stop the loop.
        """
        logger.debug("MetricsCollector loop started")
        while self._running:
            try:
                await self._collect_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "MetricsCollector collection error: %s", exc, exc_info=True
                )

            try:
                await asyncio.sleep(self._interval_seconds)
            except asyncio.CancelledError:
                break

    async def _collect_once(self) -> None:
        """Perform a single metrics collection cycle.

        Collects:
            1. Flag state gauges for each flag
            2. Kill switch state for each killed flag
            3. Stale flag count
        """
        try:
            await self._collect_flag_states()
        except Exception as exc:
            logger.warning("Failed to collect flag states: %s", exc)

        try:
            await self._collect_stale_flags()
        except Exception as exc:
            logger.warning("Failed to collect stale flags: %s", exc)

        try:
            await self._collect_kill_switch_states()
        except Exception as exc:
            logger.warning("Failed to collect kill switch states: %s", exc)

    async def _collect_flag_states(self) -> None:
        """Update flag state gauges for all flags.

        Queries all flags from the service and updates the
        ff_flag_state gauge for each flag.
        """
        flags = await self._service.list_flags(offset=0, limit=10000)
        for flag in flags:
            update_flag_state(flag.key, flag.status.value)

        logger.debug("Updated flag state gauges for %d flags", len(flags))

    async def _collect_stale_flags(self) -> None:
        """Count and report stale flags.

        A flag is considered stale if it has not been updated within
        the configured stale detection threshold and is not archived
        or permanent.
        """
        flags = await self._service.list_flags(offset=0, limit=10000)
        stale_threshold_days = 30  # Default; overridden by config if available

        try:
            config = self._service._config
            stale_threshold_days = config.stale_detection_days
        except AttributeError:
            pass

        now = datetime.now(timezone.utc)
        stale_count = 0
        for flag in flags:
            if flag.status in (FlagStatus.ARCHIVED, FlagStatus.PERMANENT):
                continue
            age_days = (now - flag.updated_at).days
            if age_days >= stale_threshold_days:
                stale_count += 1

        update_stale_count(self._environment, stale_count)
        logger.debug(
            "Stale flag count: %d (threshold=%d days)",
            stale_count, stale_threshold_days,
        )

    async def _collect_kill_switch_states(self) -> None:
        """Update kill switch gauge for all flags.

        Sets the kill switch gauge to 1 for killed flags and 0 for others.
        """
        flags = await self._service.list_flags(offset=0, limit=10000)
        for flag in flags:
            is_killed = flag.status == FlagStatus.KILLED
            record_kill_switch(flag.key, is_killed)

    async def collect_now(self) -> None:
        """Trigger an immediate metrics collection cycle.

        Useful for testing or forcing a refresh after a flag change.
        """
        await self._collect_once()
        logger.info("Manual metrics collection completed")


__all__ = ["MetricsCollector"]
