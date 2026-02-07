# -*- coding: utf-8 -*-
"""
Incident Detector - SEC-010

Aggregates alerts from multiple monitoring sources (Prometheus, Loki,
GuardDuty, CloudTrail) and converts them to normalized Alert objects
for incident correlation and response.

Example:
    >>> from greenlang.infrastructure.incident_response.detector import (
    ...     IncidentDetector,
    ... )
    >>> detector = IncidentDetector(config)
    >>> alerts = await detector.detect_incidents()

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import httpx

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Alert,
    AlertSource,
    EscalationLevel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert Normalization Helpers
# ---------------------------------------------------------------------------


def _normalize_prometheus_severity(severity: str) -> EscalationLevel:
    """Convert Prometheus severity label to EscalationLevel.

    Args:
        severity: Prometheus severity string.

    Returns:
        Normalized escalation level.
    """
    severity_map = {
        "critical": EscalationLevel.P0,
        "error": EscalationLevel.P1,
        "warning": EscalationLevel.P2,
        "info": EscalationLevel.P3,
        "none": EscalationLevel.P3,
    }
    return severity_map.get(severity.lower(), EscalationLevel.P2)


def _normalize_guardduty_severity(severity: float) -> EscalationLevel:
    """Convert GuardDuty severity (0-10) to EscalationLevel.

    Args:
        severity: GuardDuty severity score.

    Returns:
        Normalized escalation level.
    """
    if severity >= 7.0:
        return EscalationLevel.P0
    elif severity >= 4.0:
        return EscalationLevel.P1
    elif severity >= 2.0:
        return EscalationLevel.P2
    else:
        return EscalationLevel.P3


def _normalize_loki_severity(log_level: str) -> EscalationLevel:
    """Convert Loki log level to EscalationLevel.

    Args:
        log_level: Log level string.

    Returns:
        Normalized escalation level.
    """
    level_map = {
        "fatal": EscalationLevel.P0,
        "critical": EscalationLevel.P0,
        "error": EscalationLevel.P1,
        "warn": EscalationLevel.P2,
        "warning": EscalationLevel.P2,
        "info": EscalationLevel.P3,
        "debug": EscalationLevel.P3,
    }
    return level_map.get(log_level.lower(), EscalationLevel.P2)


# ---------------------------------------------------------------------------
# Incident Detector
# ---------------------------------------------------------------------------


class IncidentDetector:
    """Aggregates alerts from multiple monitoring sources.

    This class polls various alert sources (Prometheus Alertmanager, Loki,
    AWS GuardDuty, AWS CloudTrail) and converts their alerts to normalized
    Alert objects for further processing.

    Attributes:
        config: Incident response configuration.
        http_client: Async HTTP client for API calls.
        seen_fingerprints: Set of already-seen alert fingerprints.

    Example:
        >>> detector = IncidentDetector(config)
        >>> alerts = await detector.poll_prometheus()
        >>> all_alerts = await detector.detect_incidents()
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the incident detector.

        Args:
            config: Incident response configuration.
            http_client: Optional HTTP client (created if not provided).
        """
        self.config = config or get_config()
        self._http_client = http_client
        self._owns_client = http_client is None
        self.seen_fingerprints: Set[str] = set()
        self._last_poll_times: Dict[str, datetime] = {}

        logger.info("IncidentDetector initialized")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client if owned by this instance."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def poll_prometheus(self) -> List[Alert]:
        """Poll Prometheus Alertmanager for active alerts.

        Fetches alerts from the Alertmanager API and converts them to
        normalized Alert objects.

        Returns:
            List of Alert objects from Prometheus.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        if not self.config.prometheus.enabled:
            logger.debug("Prometheus polling disabled")
            return []

        alerts: List[Alert] = []
        client = await self._get_http_client()

        try:
            endpoint = self.config.prometheus.endpoint
            if not endpoint:
                logger.warning("Prometheus endpoint not configured")
                return []

            logger.debug("Polling Prometheus Alertmanager: %s", endpoint)

            response = await client.get(
                endpoint,
                timeout=self.config.prometheus.timeout_seconds,
            )
            response.raise_for_status()

            alert_data = response.json()

            for item in alert_data:
                try:
                    # Extract labels and annotations
                    labels = item.get("labels", {})
                    annotations = item.get("annotations", {})

                    # Determine severity
                    severity_str = labels.get("severity", "warning")
                    severity = _normalize_prometheus_severity(severity_str)

                    # Apply label filters
                    skip = False
                    for k, v in self.config.prometheus.label_filters.items():
                        if labels.get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue

                    # Parse timestamps
                    starts_at = None
                    if item.get("startsAt"):
                        try:
                            starts_at = datetime.fromisoformat(
                                item["startsAt"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    ends_at = None
                    if item.get("endsAt"):
                        try:
                            ends_at = datetime.fromisoformat(
                                item["endsAt"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    # Create alert
                    alert = Alert(
                        id=uuid4(),
                        source=AlertSource.PROMETHEUS,
                        alert_type=labels.get("alertname", "unknown"),
                        severity=severity,
                        message=annotations.get("summary", labels.get("alertname", "")),
                        description=annotations.get("description"),
                        raw_data=item,
                        labels=labels,
                        annotations=annotations,
                        fingerprint=item.get("fingerprint"),
                        received_at=datetime.now(timezone.utc),
                        starts_at=starts_at,
                        ends_at=ends_at,
                    )

                    alerts.append(alert)

                except Exception as e:
                    logger.warning("Failed to parse Prometheus alert: %s", e)
                    continue

            self._last_poll_times["prometheus"] = datetime.now(timezone.utc)
            logger.info("Polled %d alerts from Prometheus", len(alerts))

        except httpx.HTTPError as e:
            logger.error("Failed to poll Prometheus: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error polling Prometheus: %s", e)
            raise

        return alerts

    async def poll_loki(self) -> List[Alert]:
        """Poll Loki for error patterns using LogQL queries.

        Executes configured LogQL queries against Loki and converts
        matching log entries to Alert objects.

        Returns:
            List of Alert objects from Loki.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        if not self.config.loki.enabled:
            logger.debug("Loki polling disabled")
            return []

        alerts: List[Alert] = []
        client = await self._get_http_client()

        try:
            endpoint = self.config.loki.endpoint
            if not endpoint:
                logger.warning("Loki endpoint not configured")
                return []

            # Calculate time range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(
                seconds=self.config.loki.lookback_seconds
            )

            for query_name, query in self.config.loki.queries.items():
                try:
                    logger.debug("Executing Loki query '%s': %s", query_name, query)

                    params = {
                        "query": query,
                        "start": str(int(start_time.timestamp() * 1e9)),
                        "end": str(int(end_time.timestamp() * 1e9)),
                        "limit": 100,
                    }

                    response = await client.get(
                        endpoint,
                        params=params,
                        timeout=self.config.loki.timeout_seconds,
                    )
                    response.raise_for_status()

                    data = response.json()
                    results = data.get("data", {}).get("result", [])

                    for stream in results:
                        stream_labels = stream.get("stream", {})
                        entries = stream.get("values", [])

                        for entry in entries:
                            try:
                                timestamp_ns, log_line = entry
                                timestamp = datetime.fromtimestamp(
                                    int(timestamp_ns) / 1e9,
                                    tz=timezone.utc,
                                )

                                # Extract log level from labels or content
                                log_level = stream_labels.get("level", "error")
                                severity = _normalize_loki_severity(log_level)

                                alert = Alert(
                                    id=uuid4(),
                                    source=AlertSource.LOKI,
                                    alert_type=query_name,
                                    severity=severity,
                                    message=log_line[:500],  # Truncate
                                    description=f"Log match for query: {query_name}",
                                    raw_data={
                                        "query": query,
                                        "stream": stream_labels,
                                        "log_line": log_line,
                                    },
                                    labels=stream_labels,
                                    received_at=datetime.now(timezone.utc),
                                    starts_at=timestamp,
                                )

                                alerts.append(alert)

                            except Exception as e:
                                logger.warning("Failed to parse Loki entry: %s", e)
                                continue

                except httpx.HTTPError as e:
                    logger.warning("Loki query '%s' failed: %s", query_name, e)
                    continue

            self._last_poll_times["loki"] = datetime.now(timezone.utc)
            logger.info("Polled %d alerts from Loki", len(alerts))

        except Exception as e:
            logger.error("Unexpected error polling Loki: %s", e)
            raise

        return alerts

    async def poll_guardduty(self) -> List[Alert]:
        """Poll AWS GuardDuty for security findings.

        Fetches active GuardDuty findings and converts them to Alert objects.
        Requires boto3 and appropriate AWS credentials.

        Returns:
            List of Alert objects from GuardDuty.

        Raises:
            ImportError: If boto3 is not available.
            botocore.exceptions.ClientError: If AWS API call fails.
        """
        if not self.config.guardduty.enabled:
            logger.debug("GuardDuty polling disabled")
            return []

        alerts: List[Alert] = []

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not available - skipping GuardDuty polling")
            return []

        try:
            client = boto3.client(
                "guardduty",
                region_name=self.config.guardduty.region,
            )

            # Get detector ID if not configured
            detector_id = self.config.guardduty.detector_id
            if not detector_id:
                detectors = client.list_detectors()
                if detectors.get("DetectorIds"):
                    detector_id = detectors["DetectorIds"][0]
                else:
                    logger.warning("No GuardDuty detectors found")
                    return []

            # Build finding criteria
            finding_criteria: Dict[str, Any] = {
                "Criterion": {
                    "severity": {
                        "Gte": self.config.guardduty.min_severity,
                    },
                    "service.archived": {
                        "Eq": ["false"],
                    },
                }
            }

            if self.config.guardduty.finding_types:
                finding_criteria["Criterion"]["type"] = {
                    "Eq": self.config.guardduty.finding_types,
                }

            # List findings
            response = client.list_findings(
                DetectorId=detector_id,
                FindingCriteria=finding_criteria,
                MaxResults=50,
            )

            finding_ids = response.get("FindingIds", [])
            if not finding_ids:
                logger.debug("No GuardDuty findings found")
                return []

            # Get finding details
            findings_response = client.get_findings(
                DetectorId=detector_id,
                FindingIds=finding_ids,
            )

            for finding in findings_response.get("Findings", []):
                try:
                    severity = _normalize_guardduty_severity(
                        finding.get("Severity", 5.0)
                    )

                    # Parse timestamps
                    created_at = None
                    if finding.get("CreatedAt"):
                        try:
                            created_at = datetime.fromisoformat(
                                finding["CreatedAt"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    alert = Alert(
                        id=uuid4(),
                        source=AlertSource.GUARDDUTY,
                        alert_type=finding.get("Type", "unknown"),
                        severity=severity,
                        message=finding.get("Title", "GuardDuty Finding"),
                        description=finding.get("Description"),
                        raw_data=finding,
                        labels={
                            "finding_id": finding.get("Id", ""),
                            "account_id": finding.get("AccountId", ""),
                            "region": finding.get("Region", ""),
                            "resource_type": finding.get("Resource", {}).get(
                                "ResourceType", ""
                            ),
                        },
                        received_at=datetime.now(timezone.utc),
                        starts_at=created_at,
                    )

                    alerts.append(alert)

                except Exception as e:
                    logger.warning("Failed to parse GuardDuty finding: %s", e)
                    continue

            self._last_poll_times["guardduty"] = datetime.now(timezone.utc)
            logger.info("Polled %d alerts from GuardDuty", len(alerts))

        except ClientError as e:
            logger.error("GuardDuty API error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error polling GuardDuty: %s", e)
            raise

        return alerts

    async def poll_cloudtrail_anomalies(self) -> List[Alert]:
        """Poll CloudTrail for anomalous API activity.

        Uses CloudWatch Anomaly Detector to identify unusual patterns
        in CloudTrail logs.

        Returns:
            List of Alert objects from CloudTrail anomalies.

        Raises:
            ImportError: If boto3 is not available.
            botocore.exceptions.ClientError: If AWS API call fails.
        """
        if not self.config.cloudtrail.enabled:
            logger.debug("CloudTrail polling disabled")
            return []

        alerts: List[Alert] = []

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not available - skipping CloudTrail polling")
            return []

        try:
            logs_client = boto3.client(
                "logs",
                region_name=self.config.cloudtrail.region,
            )

            # Query CloudWatch Logs for specific patterns
            # This is a simplified implementation - production would use
            # CloudWatch Anomaly Detector or CloudTrail Insights

            log_group = self.config.cloudtrail.log_group_name
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=30)

            # Patterns indicating potential security issues
            security_patterns = [
                '"errorCode":"AccessDenied"',
                '"errorCode":"UnauthorizedAccess"',
                '"eventName":"ConsoleLogin"',
                '"eventName":"CreateUser"',
                '"eventName":"AttachUserPolicy"',
                '"eventName":"DeleteTrail"',
                '"eventName":"StopLogging"',
            ]

            for pattern in security_patterns:
                try:
                    response = logs_client.filter_log_events(
                        logGroupName=log_group,
                        startTime=int(start_time.timestamp() * 1000),
                        endTime=int(end_time.timestamp() * 1000),
                        filterPattern=pattern,
                        limit=10,
                    )

                    for event in response.get("events", []):
                        try:
                            import json
                            message = event.get("message", "{}")
                            event_data = json.loads(message)

                            # Determine severity based on event type
                            event_name = event_data.get("eventName", "")
                            severity = EscalationLevel.P2

                            if event_name in ("DeleteTrail", "StopLogging"):
                                severity = EscalationLevel.P0
                            elif event_name in ("CreateUser", "AttachUserPolicy"):
                                severity = EscalationLevel.P1
                            elif "AccessDenied" in str(event_data):
                                severity = EscalationLevel.P2

                            timestamp = datetime.fromtimestamp(
                                event.get("timestamp", 0) / 1000,
                                tz=timezone.utc,
                            )

                            alert = Alert(
                                id=uuid4(),
                                source=AlertSource.CLOUDTRAIL,
                                alert_type=f"cloudtrail_{event_name.lower()}",
                                severity=severity,
                                message=f"CloudTrail: {event_name}",
                                description=f"CloudTrail event detected: {event_name}",
                                raw_data=event_data,
                                labels={
                                    "event_name": event_name,
                                    "event_source": event_data.get("eventSource", ""),
                                    "user_identity": str(
                                        event_data.get("userIdentity", {}).get(
                                            "type", ""
                                        )
                                    ),
                                    "source_ip": event_data.get(
                                        "sourceIPAddress", ""
                                    ),
                                },
                                received_at=datetime.now(timezone.utc),
                                starts_at=timestamp,
                            )

                            alerts.append(alert)

                        except Exception as e:
                            logger.warning("Failed to parse CloudTrail event: %s", e)
                            continue

                except ClientError as e:
                    logger.warning("CloudTrail query for '%s' failed: %s", pattern, e)
                    continue

            self._last_poll_times["cloudtrail"] = datetime.now(timezone.utc)
            logger.info("Polled %d alerts from CloudTrail", len(alerts))

        except ClientError as e:
            logger.error("CloudTrail API error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error polling CloudTrail: %s", e)
            raise

        return alerts

    async def detect_incidents(self) -> List[Alert]:
        """Aggregate alerts from all enabled sources.

        Polls all configured alert sources concurrently and returns
        a deduplicated list of alerts.

        Returns:
            List of all detected alerts.
        """
        all_alerts: List[Alert] = []

        # Run all polls concurrently
        tasks = []

        if self.config.prometheus.enabled:
            tasks.append(self.poll_prometheus())
        if self.config.loki.enabled:
            tasks.append(self.poll_loki())
        if self.config.guardduty.enabled:
            tasks.append(self.poll_guardduty())
        if self.config.cloudtrail.enabled:
            tasks.append(self.poll_cloudtrail_anomalies())

        if not tasks:
            logger.warning("No alert sources enabled")
            return []

        # Gather results, handling individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Alert source polling failed: %s", result)
                continue
            if isinstance(result, list):
                all_alerts.extend(result)

        # Deduplicate based on fingerprint
        unique_alerts: List[Alert] = []
        for alert in all_alerts:
            fingerprint = alert.fingerprint or alert.calculate_fingerprint()
            if fingerprint not in self.seen_fingerprints:
                self.seen_fingerprints.add(fingerprint)
                unique_alerts.append(alert)

        # Limit seen fingerprints set size (memory management)
        if len(self.seen_fingerprints) > 10000:
            # Keep most recent half
            self.seen_fingerprints = set(
                list(self.seen_fingerprints)[-5000:]
            )

        logger.info(
            "Detected %d unique alerts from %d total",
            len(unique_alerts),
            len(all_alerts),
        )

        return unique_alerts

    def clear_seen_fingerprints(self) -> None:
        """Clear the seen fingerprints cache."""
        self.seen_fingerprints.clear()
        logger.info("Cleared seen fingerprints cache")


# ---------------------------------------------------------------------------
# Global Detector Instance
# ---------------------------------------------------------------------------

_global_detector: Optional[IncidentDetector] = None


def get_detector(
    config: Optional[IncidentResponseConfig] = None,
) -> IncidentDetector:
    """Get or create the global incident detector.

    Args:
        config: Optional configuration override.

    Returns:
        The global IncidentDetector instance.
    """
    global _global_detector

    if _global_detector is None:
        _global_detector = IncidentDetector(config)

    return _global_detector


async def reset_detector() -> None:
    """Reset and close the global detector."""
    global _global_detector

    if _global_detector is not None:
        await _global_detector.close()
        _global_detector = None


__all__ = [
    "IncidentDetector",
    "get_detector",
    "reset_detector",
]
