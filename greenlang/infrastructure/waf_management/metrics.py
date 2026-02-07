# -*- coding: utf-8 -*-
"""
WAF Management Prometheus Metrics - SEC-010

Defines all Prometheus metrics for the GreenLang WAF and DDoS protection
system and provides helper functions for recording events.

Metrics are organized by subsystem:
    - WAF: Request counts, blocks, rule latency
    - DDoS: Attack detection, mitigation, duration
    - Traffic: Requests per second, endpoint metrics
    - Shield: Protection status, attack statistics

All metrics use the ``gl_secops_`` prefix to align with SEC-010
security operations naming convention.

Example:
    >>> from greenlang.infrastructure.waf_management.metrics import (
    ...     record_waf_request, record_ddos_attack, record_traffic_rps,
    ... )
    >>> record_waf_request(rule="RateLimitPerIP", action="block")
    >>> record_ddos_attack(attack_type="volumetric")
    >>> record_traffic_rps(endpoint="/api", rps=150.0)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]
    Summary = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:

    # -- WAF Request Metrics ------------------------------------------------

    gl_secops_waf_requests_total = Counter(
        "gl_secops_waf_requests_total",
        "Total number of requests evaluated by WAF rules",
        labelnames=["rule", "action"],
    )
    """Counter: Total WAF evaluations. Labels: rule (rule name), action (allow/block/count)."""

    gl_secops_waf_blocked_total = Counter(
        "gl_secops_waf_blocked_total",
        "Total number of requests blocked by WAF",
        labelnames=["rule", "reason"],
    )
    """Counter: Blocked requests. Labels: rule (rule name), reason (sql_injection/xss/rate_limit/etc)."""

    gl_secops_waf_rule_latency_seconds = Histogram(
        "gl_secops_waf_rule_latency_seconds",
        "Time taken to evaluate WAF rules in seconds",
        labelnames=["rule"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    )
    """Histogram: Rule evaluation latency. Labels: rule (rule name)."""

    gl_secops_waf_rules_total = Gauge(
        "gl_secops_waf_rules_total",
        "Total number of WAF rules by status",
        labelnames=["status"],
    )
    """Gauge: Total rules. Labels: status (active/disabled/draft)."""

    gl_secops_waf_false_positives_total = Counter(
        "gl_secops_waf_false_positives_total",
        "Total number of reported false positives",
        labelnames=["rule"],
    )
    """Counter: False positives. Labels: rule (rule name)."""

    # -- DDoS Attack Metrics ------------------------------------------------

    gl_secops_ddos_attacks_total = Counter(
        "gl_secops_ddos_attacks_total",
        "Total number of DDoS attacks detected",
        labelnames=["type", "severity"],
    )
    """Counter: Attacks detected. Labels: type (volumetric/slowloris/etc), severity."""

    gl_secops_ddos_mitigated_total = Counter(
        "gl_secops_ddos_mitigated_total",
        "Total number of DDoS attacks successfully mitigated",
        labelnames=["type"],
    )
    """Counter: Attacks mitigated. Labels: type (attack type)."""

    gl_secops_ddos_attack_duration_seconds = Histogram(
        "gl_secops_ddos_attack_duration_seconds",
        "Duration of DDoS attacks in seconds",
        labelnames=["type"],
        buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800),
    )
    """Histogram: Attack duration. Labels: type (attack type)."""

    gl_secops_ddos_mitigation_time_seconds = Histogram(
        "gl_secops_ddos_mitigation_time_seconds",
        "Time to mitigate DDoS attacks in seconds",
        labelnames=["type"],
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    )
    """Histogram: Mitigation time. Labels: type (attack type)."""

    gl_secops_ddos_active_attacks = Gauge(
        "gl_secops_ddos_active_attacks",
        "Number of currently active DDoS attacks",
        labelnames=["type"],
    )
    """Gauge: Active attacks. Labels: type (attack type)."""

    # -- Traffic Metrics ----------------------------------------------------

    gl_secops_traffic_rps = Gauge(
        "gl_secops_traffic_rps",
        "Current requests per second",
        labelnames=["endpoint"],
    )
    """Gauge: RPS by endpoint. Labels: endpoint (API path)."""

    gl_secops_traffic_blocked_rps = Gauge(
        "gl_secops_traffic_blocked_rps",
        "Current blocked requests per second",
        labelnames=["endpoint"],
    )
    """Gauge: Blocked RPS. Labels: endpoint (API path)."""

    gl_secops_traffic_bytes_per_second = Gauge(
        "gl_secops_traffic_bytes_per_second",
        "Current traffic in bytes per second",
        labelnames=["direction"],
    )
    """Gauge: Bytes per second. Labels: direction (inbound/outbound)."""

    gl_secops_traffic_latency_seconds = Summary(
        "gl_secops_traffic_latency_seconds",
        "Request latency distribution",
        labelnames=["endpoint"],
    )
    """Summary: Latency distribution. Labels: endpoint (API path)."""

    gl_secops_traffic_unique_ips = Gauge(
        "gl_secops_traffic_unique_ips",
        "Number of unique source IP addresses",
        labelnames=["window"],
    )
    """Gauge: Unique IPs. Labels: window (1m/5m/1h)."""

    gl_secops_traffic_by_country = Gauge(
        "gl_secops_traffic_by_country",
        "Request count by source country",
        labelnames=["country"],
    )
    """Gauge: Requests by country. Labels: country (ISO code)."""

    # -- Anomaly Detection Metrics ------------------------------------------

    gl_secops_anomaly_detections_total = Counter(
        "gl_secops_anomaly_detections_total",
        "Total number of traffic anomalies detected",
        labelnames=["type", "severity"],
    )
    """Counter: Anomalies detected. Labels: type, severity."""

    gl_secops_baseline_rps = Gauge(
        "gl_secops_baseline_rps",
        "Current baseline requests per second",
        labelnames=["percentile"],
    )
    """Gauge: Baseline RPS. Labels: percentile (p50/p95/p99)."""

    gl_secops_baseline_age_seconds = Gauge(
        "gl_secops_baseline_age_seconds",
        "Age of the current traffic baseline in seconds",
    )
    """Gauge: Baseline age."""

    # -- Shield Metrics -----------------------------------------------------

    gl_secops_shield_protections_total = Gauge(
        "gl_secops_shield_protections_total",
        "Total number of Shield protections",
        labelnames=["resource_type"],
    )
    """Gauge: Shield protections. Labels: resource_type (ALB/CloudFront/etc)."""

    gl_secops_shield_subscription_active = Gauge(
        "gl_secops_shield_subscription_active",
        "Whether Shield Advanced subscription is active (1=active, 0=inactive)",
    )
    """Gauge: Shield subscription status."""

    gl_secops_shield_attacks_mitigated_total = Counter(
        "gl_secops_shield_attacks_mitigated_total",
        "Total attacks mitigated by Shield",
        labelnames=["resource_arn"],
    )
    """Counter: Shield mitigated attacks. Labels: resource_arn."""

    # -- Rate Limiting Metrics ----------------------------------------------

    gl_secops_rate_limit_exceeded_total = Counter(
        "gl_secops_rate_limit_exceeded_total",
        "Total number of rate limit violations",
        labelnames=["rule", "source_ip"],
    )
    """Counter: Rate limit violations. Labels: rule, source_ip."""

    gl_secops_rate_limit_current = Gauge(
        "gl_secops_rate_limit_current",
        "Current request count in rate limit window",
        labelnames=["rule"],
    )
    """Gauge: Current rate limit count. Labels: rule."""

    # -- Geo Blocking Metrics -----------------------------------------------

    gl_secops_geo_blocked_total = Counter(
        "gl_secops_geo_blocked_total",
        "Total requests blocked by geo-blocking",
        labelnames=["country"],
    )
    """Counter: Geo-blocked requests. Labels: country (ISO code)."""

else:
    # Provide no-op stubs when prometheus_client is not installed
    logger.info(
        "prometheus_client not available; WAF metrics will be no-ops"
    )

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is not installed."""

        def labels(self, *args, **kwargs):  # noqa: ANN
            """Return self for chaining."""
            return self

        def inc(self, amount=1):  # noqa: ANN
            """No-op increment."""

        def dec(self, amount=1):  # noqa: ANN
            """No-op decrement."""

        def set(self, value):  # noqa: ANN
            """No-op set."""

        def observe(self, amount):  # noqa: ANN
            """No-op observe."""

    gl_secops_waf_requests_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_waf_blocked_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_waf_rule_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_waf_rules_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_waf_false_positives_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_ddos_attacks_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_ddos_mitigated_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_ddos_attack_duration_seconds = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_ddos_mitigation_time_seconds = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_ddos_active_attacks = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_rps = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_blocked_rps = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_bytes_per_second = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_latency_seconds = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_unique_ips = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_traffic_by_country = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_anomaly_detections_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_baseline_rps = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_baseline_age_seconds = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_shield_protections_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_shield_subscription_active = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_shield_attacks_mitigated_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_rate_limit_exceeded_total = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_rate_limit_current = _NoOpMetric()  # type: ignore[assignment]
    gl_secops_geo_blocked_total = _NoOpMetric()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_waf_request(
    rule: str,
    action: str,
    latency_seconds: float = 0.0,
    reason: Optional[str] = None,
) -> None:
    """Record a WAF rule evaluation.

    Args:
        rule: Name of the WAF rule.
        action: Action taken (allow, block, count, captcha).
        latency_seconds: Evaluation time in seconds.
        reason: Optional reason for the action (for blocks).
    """
    try:
        gl_secops_waf_requests_total.labels(rule=rule, action=action).inc()

        if latency_seconds > 0:
            gl_secops_waf_rule_latency_seconds.labels(rule=rule).observe(latency_seconds)

        if action == "block" and reason:
            gl_secops_waf_blocked_total.labels(rule=rule, reason=reason).inc()

    except Exception as exc:
        logger.debug("Failed to record WAF request metric: %s", exc)


def record_waf_block(
    rule: str,
    reason: str,
) -> None:
    """Record a blocked request.

    Args:
        rule: Name of the WAF rule that blocked.
        reason: Reason for blocking (sql_injection, xss, rate_limit, etc).
    """
    try:
        gl_secops_waf_blocked_total.labels(rule=rule, reason=reason).inc()
    except Exception as exc:
        logger.debug("Failed to record WAF block metric: %s", exc)


def record_ddos_attack(
    attack_type: str,
    severity: str = "medium",
    mitigated: bool = False,
    duration_seconds: Optional[float] = None,
    mitigation_time_seconds: Optional[float] = None,
) -> None:
    """Record a DDoS attack detection or mitigation.

    Args:
        attack_type: Type of attack (volumetric, slowloris, etc).
        severity: Attack severity (low, medium, high, critical).
        mitigated: Whether the attack was mitigated.
        duration_seconds: Attack duration in seconds (if ended).
        mitigation_time_seconds: Time to mitigate in seconds.
    """
    try:
        gl_secops_ddos_attacks_total.labels(type=attack_type, severity=severity).inc()

        if mitigated:
            gl_secops_ddos_mitigated_total.labels(type=attack_type).inc()

        if duration_seconds is not None:
            gl_secops_ddos_attack_duration_seconds.labels(type=attack_type).observe(
                duration_seconds
            )

        if mitigation_time_seconds is not None:
            gl_secops_ddos_mitigation_time_seconds.labels(type=attack_type).observe(
                mitigation_time_seconds
            )

    except Exception as exc:
        logger.debug("Failed to record DDoS attack metric: %s", exc)


def update_active_attacks(
    attack_type: str,
    count: int,
) -> None:
    """Update the count of active attacks.

    Args:
        attack_type: Type of attack.
        count: Current number of active attacks.
    """
    try:
        gl_secops_ddos_active_attacks.labels(type=attack_type).set(count)
    except Exception as exc:
        logger.debug("Failed to update active attacks metric: %s", exc)


def record_traffic_rps(
    endpoint: str,
    rps: float,
    blocked_rps: float = 0.0,
) -> None:
    """Record traffic requests per second.

    Args:
        endpoint: API endpoint or path.
        rps: Requests per second.
        blocked_rps: Blocked requests per second.
    """
    try:
        gl_secops_traffic_rps.labels(endpoint=endpoint).set(rps)
        if blocked_rps > 0:
            gl_secops_traffic_blocked_rps.labels(endpoint=endpoint).set(blocked_rps)
    except Exception as exc:
        logger.debug("Failed to record traffic RPS metric: %s", exc)


def record_traffic_latency(
    endpoint: str,
    latency_seconds: float,
) -> None:
    """Record request latency.

    Args:
        endpoint: API endpoint or path.
        latency_seconds: Request latency in seconds.
    """
    try:
        gl_secops_traffic_latency_seconds.labels(endpoint=endpoint).observe(
            latency_seconds
        )
    except Exception as exc:
        logger.debug("Failed to record traffic latency metric: %s", exc)


def update_unique_ips(
    window: str,
    count: int,
) -> None:
    """Update unique IP count.

    Args:
        window: Time window (1m, 5m, 1h).
        count: Number of unique IPs.
    """
    try:
        gl_secops_traffic_unique_ips.labels(window=window).set(count)
    except Exception as exc:
        logger.debug("Failed to update unique IPs metric: %s", exc)


def update_traffic_by_country(
    country_counts: dict,
) -> None:
    """Update traffic counts by country.

    Args:
        country_counts: Dictionary mapping country codes to request counts.
    """
    try:
        for country, count in country_counts.items():
            gl_secops_traffic_by_country.labels(country=country).set(count)
    except Exception as exc:
        logger.debug("Failed to update traffic by country metric: %s", exc)


def record_anomaly_detection(
    anomaly_type: str,
    severity: str,
) -> None:
    """Record an anomaly detection event.

    Args:
        anomaly_type: Type of anomaly detected.
        severity: Severity level.
    """
    try:
        gl_secops_anomaly_detections_total.labels(
            type=anomaly_type, severity=severity
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record anomaly detection metric: %s", exc)


def update_baseline_metrics(
    rps_p50: float,
    rps_p95: float,
    rps_p99: float,
    age_seconds: float,
) -> None:
    """Update baseline metrics.

    Args:
        rps_p50: 50th percentile RPS.
        rps_p95: 95th percentile RPS.
        rps_p99: 99th percentile RPS.
        age_seconds: Age of the baseline in seconds.
    """
    try:
        gl_secops_baseline_rps.labels(percentile="p50").set(rps_p50)
        gl_secops_baseline_rps.labels(percentile="p95").set(rps_p95)
        gl_secops_baseline_rps.labels(percentile="p99").set(rps_p99)
        gl_secops_baseline_age_seconds.set(age_seconds)
    except Exception as exc:
        logger.debug("Failed to update baseline metrics: %s", exc)


def update_waf_rules_count(
    active: int,
    disabled: int,
    draft: int,
) -> None:
    """Update WAF rules count gauge.

    Args:
        active: Number of active rules.
        disabled: Number of disabled rules.
        draft: Number of draft rules.
    """
    try:
        gl_secops_waf_rules_total.labels(status="active").set(active)
        gl_secops_waf_rules_total.labels(status="disabled").set(disabled)
        gl_secops_waf_rules_total.labels(status="draft").set(draft)
    except Exception as exc:
        logger.debug("Failed to update WAF rules count metric: %s", exc)


def update_shield_status(
    subscription_active: bool,
    protection_counts: dict,
) -> None:
    """Update Shield status metrics.

    Args:
        subscription_active: Whether Shield subscription is active.
        protection_counts: Dictionary mapping resource types to counts.
    """
    try:
        gl_secops_shield_subscription_active.set(1 if subscription_active else 0)
        for resource_type, count in protection_counts.items():
            gl_secops_shield_protections_total.labels(
                resource_type=resource_type
            ).set(count)
    except Exception as exc:
        logger.debug("Failed to update Shield status metric: %s", exc)


def record_rate_limit_exceeded(
    rule: str,
    source_ip: str,
) -> None:
    """Record a rate limit violation.

    Args:
        rule: Rate limit rule name.
        source_ip: Source IP that exceeded the limit.
    """
    try:
        # Anonymize IP for privacy (use /24 subnet for IPv4)
        if "." in source_ip:
            parts = source_ip.split(".")
            anonymized_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        else:
            anonymized_ip = "ipv6"

        gl_secops_rate_limit_exceeded_total.labels(
            rule=rule, source_ip=anonymized_ip
        ).inc()
    except Exception as exc:
        logger.debug("Failed to record rate limit exceeded metric: %s", exc)


def record_geo_block(
    country: str,
) -> None:
    """Record a geo-blocking event.

    Args:
        country: ISO country code that was blocked.
    """
    try:
        gl_secops_geo_blocked_total.labels(country=country).inc()
    except Exception as exc:
        logger.debug("Failed to record geo block metric: %s", exc)


__all__ = [
    # Metrics
    "gl_secops_waf_requests_total",
    "gl_secops_waf_blocked_total",
    "gl_secops_waf_rule_latency_seconds",
    "gl_secops_waf_rules_total",
    "gl_secops_waf_false_positives_total",
    "gl_secops_ddos_attacks_total",
    "gl_secops_ddos_mitigated_total",
    "gl_secops_ddos_attack_duration_seconds",
    "gl_secops_ddos_mitigation_time_seconds",
    "gl_secops_ddos_active_attacks",
    "gl_secops_traffic_rps",
    "gl_secops_traffic_blocked_rps",
    "gl_secops_traffic_bytes_per_second",
    "gl_secops_traffic_latency_seconds",
    "gl_secops_traffic_unique_ips",
    "gl_secops_traffic_by_country",
    "gl_secops_anomaly_detections_total",
    "gl_secops_baseline_rps",
    "gl_secops_baseline_age_seconds",
    "gl_secops_shield_protections_total",
    "gl_secops_shield_subscription_active",
    "gl_secops_shield_attacks_mitigated_total",
    "gl_secops_rate_limit_exceeded_total",
    "gl_secops_rate_limit_current",
    "gl_secops_geo_blocked_total",
    # Helper functions
    "record_waf_request",
    "record_waf_block",
    "record_ddos_attack",
    "update_active_attacks",
    "record_traffic_rps",
    "record_traffic_latency",
    "update_unique_ips",
    "update_traffic_by_country",
    "record_anomaly_detection",
    "update_baseline_metrics",
    "update_waf_rules_count",
    "update_shield_status",
    "record_rate_limit_exceeded",
    "record_geo_block",
]
