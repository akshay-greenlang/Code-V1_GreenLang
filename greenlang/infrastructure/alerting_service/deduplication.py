# -*- coding: utf-8 -*-
"""
Alert Deduplication - OBS-004: Unified Alerting Service

Prevents duplicate notifications for the same incident by tracking alert
fingerprints within a configurable time window. Correlated alerts that
share label subsets are also detected.

Example:
    >>> from greenlang.infrastructure.alerting_service.deduplication import (
    ...     AlertDeduplicator,
    ... )
    >>> dedup = AlertDeduplicator(window_minutes=60)
    >>> alert, is_new = dedup.process(alert)
    >>> if not is_new:
    ...     print("Duplicate suppressed")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from greenlang.infrastructure.alerting_service.metrics import record_dedup
from greenlang.infrastructure.alerting_service.models import Alert

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlertDeduplicator
# ---------------------------------------------------------------------------


class AlertDeduplicator:
    """Fingerprint-based alert deduplication within a sliding time window.

    Alerts that share the same fingerprint (``source + name + labels``)
    within the window are treated as duplicates. The first occurrence is
    passed through; subsequent duplicates increment ``notification_count``
    on the original alert.

    Attributes:
        window_minutes: Deduplication window size.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        """Initialize the deduplicator.

        Args:
            window_minutes: Duration in minutes for the dedup window.
        """
        self.window_minutes = window_minutes
        self._active_fingerprints: Dict[str, Alert] = {}
        self._fingerprint_timestamps: Dict[str, datetime] = {}
        self._dedup_count: int = 0
        self._lock = threading.RLock()
        logger.info(
            "AlertDeduplicator initialized: window=%dm", window_minutes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def generate_fingerprint(
        source: str,
        name: str,
        labels: Dict[str, str],
    ) -> str:
        """Generate a stable dedup fingerprint identical to Alert.generate_fingerprint.

        Args:
            source: Alert source.
            name: Alert rule name.
            labels: Label key-value pairs.

        Returns:
            Hex MD5 digest.
        """
        sorted_labels = "&".join(
            f"{k}={v}" for k, v in sorted(labels.items())
        )
        raw = f"{source}|{name}|{sorted_labels}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def is_duplicate(self, alert: Alert) -> bool:
        """Check if an alert is a duplicate within the current window.

        Args:
            alert: The alert to check.

        Returns:
            True if a matching fingerprint is already active.
        """
        fp = alert.fingerprint or self.generate_fingerprint(
            alert.source, alert.name, alert.labels,
        )
        with self._lock:
            if fp not in self._fingerprint_timestamps:
                return False
            first_seen = self._fingerprint_timestamps[fp]
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=self.window_minutes,
            )
            return first_seen > cutoff

    def process(self, alert: Alert) -> Tuple[Alert, bool]:
        """Process an alert through deduplication.

        If the alert is new, it is registered and returned as-is.
        If it is a duplicate, the existing alert's ``notification_count``
        is incremented and ``(existing_alert, False)`` is returned.

        Args:
            alert: The incoming alert.

        Returns:
            Tuple of ``(alert, is_new)``.
        """
        fp = alert.fingerprint or self.generate_fingerprint(
            alert.source, alert.name, alert.labels,
        )
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.window_minutes)

        with self._lock:
            # Check for existing active fingerprint
            if fp in self._fingerprint_timestamps:
                first_seen = self._fingerprint_timestamps[fp]
                if first_seen > cutoff:
                    existing = self._active_fingerprints[fp]
                    existing.notification_count += 1
                    self._dedup_count += 1
                    record_dedup()
                    logger.debug(
                        "Dedup suppressed: fp=%s, count=%d",
                        fp[:12], existing.notification_count,
                    )
                    return existing, False

            # New alert — register fingerprint
            self._active_fingerprints[fp] = alert
            self._fingerprint_timestamps[fp] = now
            logger.debug("Dedup registered: fp=%s", fp[:12])
            return alert, True

    def correlate(self, alert: Alert) -> Optional[str]:
        """Find a related alert by shared labels.

        Looks for active alerts that share at least one label key-value
        pair with the given alert (excluding trivial labels).

        Args:
            alert: The alert to correlate.

        Returns:
            The ``alert_id`` of a related alert, or None.
        """
        trivial_keys = {"job", "instance", "__name__"}
        significant_labels = {
            k: v for k, v in alert.labels.items() if k not in trivial_keys
        }
        if not significant_labels:
            return None

        with self._lock:
            for fp, existing in self._active_fingerprints.items():
                if existing.alert_id == alert.alert_id:
                    continue
                for k, v in significant_labels.items():
                    if existing.labels.get(k) == v:
                        logger.debug(
                            "Correlated alert %s with %s via label %s=%s",
                            alert.alert_id[:8], existing.alert_id[:8], k, v,
                        )
                        return existing.alert_id
        return None

    def cleanup(self) -> int:
        """Remove expired fingerprints outside the dedup window.

        Returns:
            Number of fingerprints removed.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=self.window_minutes)
        removed = 0

        with self._lock:
            expired = [
                fp
                for fp, ts in self._fingerprint_timestamps.items()
                if ts <= cutoff
            ]
            for fp in expired:
                del self._fingerprint_timestamps[fp]
                self._active_fingerprints.pop(fp, None)
                removed += 1

        if removed:
            logger.info("Dedup cleanup: removed %d expired fingerprints", removed)
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Return deduplication statistics.

        Returns:
            Dictionary with ``dedup_count``, ``active_fingerprints``,
            ``window_minutes``.
        """
        with self._lock:
            return {
                "dedup_count": self._dedup_count,
                "active_fingerprints": len(self._active_fingerprints),
                "window_minutes": self.window_minutes,
            }
