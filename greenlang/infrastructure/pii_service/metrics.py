# -*- coding: utf-8 -*-
"""
PII Service Prometheus Metrics - SEC-011

Comprehensive metrics for monitoring the PII Detection/Redaction service including:
- Detection metrics (by type, source, confidence level)
- Enforcement metrics (actions taken, blocked requests)
- Token vault metrics (tokens by tenant, tokenization operations)
- Streaming metrics (processed/blocked messages, errors)
- Remediation metrics (actions, pending items)
- Quarantine metrics (items by type)
- Allowlist metrics (matches, entries)

Metrics follow GreenLang naming conventions with 'gl_' prefix.

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Detection Metrics
# =============================================================================

pii_detections_total = Counter(
    "gl_pii_detections_total",
    "Total PII detections across all scanners",
    ["pii_type", "source", "confidence_level"],
)

pii_detection_latency_seconds = Histogram(
    "gl_pii_detection_latency_seconds",
    "PII detection latency in seconds",
    ["scanner_type"],  # regex, ml, hybrid
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


# =============================================================================
# Enforcement Metrics
# =============================================================================

pii_enforcement_actions_total = Counter(
    "gl_pii_enforcement_actions_total",
    "Total enforcement actions taken when PII is detected",
    ["action", "pii_type", "context"],
)

pii_blocked_requests_total = Counter(
    "gl_pii_blocked_requests_total",
    "Total requests blocked due to PII detection",
    ["pii_type", "endpoint"],
)


# =============================================================================
# Token Vault Metrics
# =============================================================================

pii_tokens_total = Gauge(
    "gl_pii_tokens_total",
    "Total active tokens in the vault",
    ["tenant_id", "pii_type"],
)

pii_tokenization_total = Counter(
    "gl_pii_tokenization_total",
    "Total tokenization operations",
    ["pii_type", "status"],  # success, failed
)

pii_detokenization_total = Counter(
    "gl_pii_detokenization_total",
    "Total detokenization operations",
    ["pii_type", "status"],  # success, failed, denied, expired
)


# =============================================================================
# Streaming Metrics
# =============================================================================

pii_stream_processed_total = Counter(
    "gl_pii_stream_processed_total",
    "Total stream messages processed for PII scanning",
    ["topic", "action"],
)

pii_stream_blocked_total = Counter(
    "gl_pii_stream_blocked_total",
    "Total stream messages blocked due to PII",
    ["topic", "pii_type"],
)

pii_stream_errors_total = Counter(
    "gl_pii_stream_errors_total",
    "Total errors during stream PII processing",
    ["topic"],
)


# =============================================================================
# Remediation Metrics
# =============================================================================

pii_remediation_total = Counter(
    "gl_pii_remediation_total",
    "Total remediation actions executed",
    ["action", "pii_type", "source"],
)

pii_remediation_pending = Gauge(
    "gl_pii_remediation_pending",
    "Current number of pending remediation items",
    ["pii_type"],
)


# =============================================================================
# Quarantine Metrics
# =============================================================================

pii_quarantine_items = Gauge(
    "gl_pii_quarantine_items",
    "Current number of items in quarantine",
    ["pii_type"],
)


# =============================================================================
# Allowlist Metrics
# =============================================================================

pii_allowlist_matches_total = Counter(
    "gl_pii_allowlist_matches_total",
    "Total allowlist matches (false positives avoided)",
    ["pii_type", "pattern"],
)

pii_allowlist_entries = Gauge(
    "gl_pii_allowlist_entries",
    "Total active allowlist entries",
    ["pii_type", "tenant_id"],
)


# =============================================================================
# Helper Functions
# =============================================================================


def record_detection(
    pii_type: str,
    source: str,
    confidence: float,
) -> None:
    """Record a PII detection with confidence-based level.

    Args:
        pii_type: The type of PII detected (e.g., 'ssn', 'email').
        source: The source of the detection (e.g., 'api', 'storage', 'streaming').
        confidence: The confidence score (0.0-1.0).
    """
    if confidence >= 0.9:
        level = "high"
    elif confidence >= 0.7:
        level = "medium"
    else:
        level = "low"

    pii_detections_total.labels(
        pii_type=pii_type,
        source=source,
        confidence_level=level,
    ).inc()


def record_enforcement_action(
    action: str,
    pii_type: str,
    context: str,
) -> None:
    """Record an enforcement action taken.

    Args:
        action: The action taken (allow, redact, block, quarantine, transform).
        pii_type: The type of PII that triggered the action.
        context: The enforcement context (api_request, api_response, storage, etc.).
    """
    pii_enforcement_actions_total.labels(
        action=action,
        pii_type=pii_type,
        context=context,
    ).inc()


def record_blocked_request(
    pii_type: str,
    endpoint: str,
) -> None:
    """Record a blocked request due to PII detection.

    Args:
        pii_type: The type of PII that caused the block.
        endpoint: The API endpoint where the request was blocked.
    """
    pii_blocked_requests_total.labels(
        pii_type=pii_type,
        endpoint=endpoint,
    ).inc()


def record_tokenization(
    pii_type: str,
    success: bool,
) -> None:
    """Record a tokenization operation.

    Args:
        pii_type: The type of PII being tokenized.
        success: Whether the tokenization succeeded.
    """
    status = "success" if success else "failed"
    pii_tokenization_total.labels(
        pii_type=pii_type,
        status=status,
    ).inc()


def record_detokenization(
    pii_type: str,
    status: str,
) -> None:
    """Record a detokenization operation.

    Args:
        pii_type: The type of PII being detokenized.
        status: The result status (success, failed, denied, expired).
    """
    pii_detokenization_total.labels(
        pii_type=pii_type,
        status=status,
    ).inc()


def record_stream_processed(
    topic: str,
    action: str,
) -> None:
    """Record a processed stream message.

    Args:
        topic: The Kafka/Kinesis topic.
        action: The action taken (allowed, redacted, blocked).
    """
    pii_stream_processed_total.labels(
        topic=topic,
        action=action,
    ).inc()


def record_stream_blocked(
    topic: str,
    pii_type: str,
) -> None:
    """Record a blocked stream message.

    Args:
        topic: The Kafka/Kinesis topic.
        pii_type: The type of PII that caused the block.
    """
    pii_stream_blocked_total.labels(
        topic=topic,
        pii_type=pii_type,
    ).inc()


def record_stream_error(
    topic: str,
) -> None:
    """Record a stream processing error.

    Args:
        topic: The Kafka/Kinesis topic where the error occurred.
    """
    pii_stream_errors_total.labels(
        topic=topic,
    ).inc()


def record_remediation(
    action: str,
    pii_type: str,
    source: str,
) -> None:
    """Record a remediation action.

    Args:
        action: The remediation action (delete, anonymize, archive, notify_only).
        pii_type: The type of PII remediated.
        source: The data source (postgresql, s3, redis, etc.).
    """
    pii_remediation_total.labels(
        action=action,
        pii_type=pii_type,
        source=source,
    ).inc()


def record_allowlist_match(
    pii_type: str,
    pattern: str,
) -> None:
    """Record an allowlist match (false positive avoided).

    Args:
        pii_type: The type of PII that matched the allowlist.
        pattern: The pattern that matched (truncated if too long).
    """
    # Truncate pattern to avoid high cardinality
    truncated_pattern = pattern[:50] if len(pattern) > 50 else pattern
    pii_allowlist_matches_total.labels(
        pii_type=pii_type,
        pattern=truncated_pattern,
    ).inc()


def set_token_count(
    tenant_id: str,
    pii_type: str,
    count: int,
) -> None:
    """Set the current token count for a tenant/type combination.

    Args:
        tenant_id: The tenant identifier.
        pii_type: The type of PII.
        count: The current count of tokens.
    """
    pii_tokens_total.labels(
        tenant_id=tenant_id,
        pii_type=pii_type,
    ).set(count)


def set_pending_remediation(
    pii_type: str,
    count: int,
) -> None:
    """Set the count of pending remediation items.

    Args:
        pii_type: The type of PII.
        count: The current count of pending items.
    """
    pii_remediation_pending.labels(
        pii_type=pii_type,
    ).set(count)


def set_quarantine_count(
    pii_type: str,
    count: int,
) -> None:
    """Set the count of quarantined items.

    Args:
        pii_type: The type of PII.
        count: The current count of quarantined items.
    """
    pii_quarantine_items.labels(
        pii_type=pii_type,
    ).set(count)


def set_allowlist_entry_count(
    pii_type: str,
    tenant_id: str,
    count: int,
) -> None:
    """Set the count of allowlist entries.

    Args:
        pii_type: The type of PII.
        tenant_id: The tenant identifier (or 'global' for global entries).
        count: The current count of allowlist entries.
    """
    pii_allowlist_entries.labels(
        pii_type=pii_type,
        tenant_id=tenant_id,
    ).set(count)


# =============================================================================
# Metrics Helper Class
# =============================================================================


class PIIMetrics:
    """Metrics helper class providing convenient access to all PII metrics.

    This class provides a unified interface for accessing PII service metrics,
    making it easy to inject and use metrics throughout the service.

    Example:
        >>> metrics = get_pii_metrics()
        >>> metrics.detections_total.labels(pii_type='ssn', source='api', confidence_level='high').inc()

    Attributes:
        detections_total: Counter for total PII detections.
        detection_latency: Histogram for detection latency.
        enforcement_actions: Counter for enforcement actions.
        blocked_requests: Counter for blocked requests.
        tokens_total: Gauge for active tokens.
        tokenization_total: Counter for tokenization operations.
        detokenization_total: Counter for detokenization operations.
        stream_processed: Counter for processed stream messages.
        stream_blocked: Counter for blocked stream messages.
        stream_errors: Counter for stream errors.
        remediation_total: Counter for remediation actions.
        remediation_pending: Gauge for pending remediations.
        quarantine_items: Gauge for quarantined items.
        allowlist_matches: Counter for allowlist matches.
        allowlist_entries: Gauge for allowlist entries.
    """

    # Detection metrics
    detections_total = pii_detections_total
    detection_latency = pii_detection_latency_seconds

    # Enforcement metrics
    enforcement_actions = pii_enforcement_actions_total
    blocked_requests = pii_blocked_requests_total

    # Token vault metrics
    tokens_total = pii_tokens_total
    tokenization_total = pii_tokenization_total
    detokenization_total = pii_detokenization_total

    # Streaming metrics
    stream_processed = pii_stream_processed_total
    stream_blocked = pii_stream_blocked_total
    stream_errors = pii_stream_errors_total

    # Remediation metrics
    remediation_total = pii_remediation_total
    remediation_pending = pii_remediation_pending

    # Quarantine metrics
    quarantine_items = pii_quarantine_items

    # Allowlist metrics
    allowlist_matches = pii_allowlist_matches_total
    allowlist_entries = pii_allowlist_entries


# Singleton instance
_pii_metrics: Optional[PIIMetrics] = None


def get_pii_metrics() -> PIIMetrics:
    """Get or create the PII metrics helper instance.

    Returns:
        PIIMetrics singleton instance.
    """
    global _pii_metrics

    if _pii_metrics is None:
        _pii_metrics = PIIMetrics()

    return _pii_metrics


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Raw metrics
    "pii_detections_total",
    "pii_detection_latency_seconds",
    "pii_enforcement_actions_total",
    "pii_blocked_requests_total",
    "pii_tokens_total",
    "pii_tokenization_total",
    "pii_detokenization_total",
    "pii_stream_processed_total",
    "pii_stream_blocked_total",
    "pii_stream_errors_total",
    "pii_remediation_total",
    "pii_remediation_pending",
    "pii_quarantine_items",
    "pii_allowlist_matches_total",
    "pii_allowlist_entries",
    # Helper functions
    "record_detection",
    "record_enforcement_action",
    "record_blocked_request",
    "record_tokenization",
    "record_detokenization",
    "record_stream_processed",
    "record_stream_blocked",
    "record_stream_error",
    "record_remediation",
    "record_allowlist_match",
    "set_token_count",
    "set_pending_remediation",
    "set_quarantine_count",
    "set_allowlist_entry_count",
    # Classes
    "PIIMetrics",
    "get_pii_metrics",
]
