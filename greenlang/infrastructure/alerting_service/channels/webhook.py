# -*- coding: utf-8 -*-
"""
Generic Webhook Notification Channel - OBS-004: Unified Alerting Service

Delivers alert notifications to an arbitrary HTTPS endpoint. The payload
is JSON-encoded and signed with HMAC-SHA256 so the receiver can verify
authenticity.

Example:
    >>> channel = WebhookChannel(
    ...     url="https://hooks.example.com/alerts",
    ...     secret="s3cret",
    ... )
    >>> result = await channel.send(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional httpx import
# ---------------------------------------------------------------------------

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# WebhookChannel
# ---------------------------------------------------------------------------


class WebhookChannel(BaseNotificationChannel):
    """Generic HMAC-signed webhook notification channel.

    Posts a JSON payload to the configured URL and adds an
    ``X-GL-Signature`` header containing the HMAC-SHA256 digest of the
    request body.

    Attributes:
        url: Target webhook URL.
        secret: HMAC signing secret.
        extra_headers: Additional headers to include.
    """

    name = "webhook"

    def __init__(
        self,
        url: str = "",
        secret: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.url = url
        self.secret = secret
        self.extra_headers = headers or {}
        self.enabled = bool(url) and HTTPX_AVAILABLE

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """POST the alert payload to the webhook URL.

        Args:
            alert: Alert to deliver.
            rendered_message: Pre-rendered description text.

        Returns:
            NotificationResult.
        """
        if not HTTPX_AVAILABLE:
            return self._make_result(
                NotificationStatus.FAILED,
                error_message="httpx not installed",
            )
        if not self.url:
            return self._make_result(
                NotificationStatus.SKIPPED,
                error_message="No webhook URL configured",
            )

        payload = alert.to_dict()
        payload["rendered_message"] = rendered_message

        body_bytes = json.dumps(payload, default=str).encode("utf-8")
        timestamp = str(int(time.time()))

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "X-GL-Timestamp": timestamp,
        }
        headers.update(self.extra_headers)

        if self.secret:
            signature = self._sign_payload(body_bytes, self.secret)
            headers["X-GL-Signature"] = signature

        start = self._timed()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.url,
                    content=body_bytes,
                    headers=headers,
                )

            duration_ms = (self._timed() - start) * 1000

            if 200 <= resp.status_code < 300:
                logger.info(
                    "Webhook notification sent: url=%s, alert=%s",
                    self.url, alert.alert_id[:8],
                )
                return self._make_result(
                    NotificationStatus.SENT,
                    recipient=self.url,
                    duration_ms=duration_ms,
                    response_code=resp.status_code,
                )

            logger.warning(
                "Webhook send failed: status=%d, body=%s",
                resp.status_code, resp.text[:200],
            )
            return self._make_result(
                NotificationStatus.FAILED,
                recipient=self.url,
                duration_ms=duration_ms,
                response_code=resp.status_code,
                error_message=resp.text[:200],
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Webhook send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                recipient=self.url,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        """Check webhook URL reachability.

        Returns:
            True if the endpoint responds.
        """
        if not HTTPX_AVAILABLE or not self.url:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.head(self.url)
            return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # HMAC signing
    # ------------------------------------------------------------------

    @staticmethod
    def _sign_payload(payload_bytes: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature over the payload.

        Args:
            payload_bytes: Raw request body bytes.
            secret: Signing secret.

        Returns:
            Hex-encoded HMAC digest.
        """
        return hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
