# -*- coding: utf-8 -*-
"""
Base Notification Channel - OBS-004: Unified Alerting Service

Abstract base class for all notification channels. Every channel must
implement ``send()`` and ``health_check()``.  Optional lifecycle hooks
(``acknowledge``, ``resolve``) are provided with default no-op fallbacks
for channels that support bidirectional integration (PagerDuty, Opsgenie).

Example:
    >>> class MyChannel(BaseNotificationChannel):
    ...     async def send(self, alert, rendered_message):
    ...         ...
    ...     async def health_check(self):
    ...         return True

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from greenlang.infrastructure.alerting_service.models import (
    Alert,
    NotificationChannel,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BaseNotificationChannel
# ---------------------------------------------------------------------------


class BaseNotificationChannel(ABC):
    """Abstract base for notification delivery channels.

    Subclasses must implement ``send()`` and ``health_check()``.
    ``acknowledge()`` and ``resolve()`` are optional lifecycle hooks that
    default to a SKIPPED result.

    Attributes:
        name: Machine-readable channel name (e.g. ``slack``).
        enabled: Whether the channel is active.
    """

    name: str = "base"
    enabled: bool = True

    @abstractmethod
    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Deliver a notification for the given alert.

        Args:
            alert: The alert to notify about.
            rendered_message: Pre-rendered message body.

        Returns:
            NotificationResult with delivery outcome.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify channel connectivity.

        Returns:
            True if the channel is reachable and healthy.
        """

    async def acknowledge(self, alert_id: str) -> NotificationResult:
        """Acknowledge an alert in the upstream provider.

        Override in channels that support bidirectional ack (PagerDuty,
        Opsgenie).

        Args:
            alert_id: The alert or dedup key to acknowledge.

        Returns:
            NotificationResult (default: SKIPPED).
        """
        return self._make_result(
            status=NotificationStatus.SKIPPED,
            recipient="n/a",
            error_message="acknowledge not supported by this channel",
        )

    async def resolve(self, alert_id: str) -> NotificationResult:
        """Resolve an alert in the upstream provider.

        Override in channels that support bidirectional resolve.

        Args:
            alert_id: The alert or dedup key to resolve.

        Returns:
            NotificationResult (default: SKIPPED).
        """
        return self._make_result(
            status=NotificationStatus.SKIPPED,
            recipient="n/a",
            error_message="resolve not supported by this channel",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_result(
        self,
        status: NotificationStatus,
        recipient: str = "",
        duration_ms: float = 0.0,
        response_code: int = 0,
        error_message: str = "",
    ) -> NotificationResult:
        """Build a NotificationResult for this channel.

        Args:
            status: Delivery outcome.
            recipient: Target address.
            duration_ms: Delivery latency.
            response_code: HTTP or provider status code.
            error_message: Error detail.

        Returns:
            Populated NotificationResult.
        """
        channel_enum = NotificationChannel.WEBHOOK
        try:
            channel_enum = NotificationChannel(self.name)
        except ValueError:
            pass

        return NotificationResult(
            channel=channel_enum,
            status=status,
            recipient=recipient,
            duration_ms=duration_ms,
            response_code=response_code,
            error_message=error_message,
            sent_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _timed() -> float:
        """Return a high-resolution timestamp for latency tracking.

        Returns:
            Current time in seconds (perf_counter).
        """
        return time.perf_counter()
