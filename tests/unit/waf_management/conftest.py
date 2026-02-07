"""
Test fixtures for waf_management module.

Provides mock traffic data, WAF rules, detection results, and configuration
for comprehensive unit testing of WAF management components.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from greenlang.infrastructure.waf_management.models import (
    WAFRule,
    RuleType,
    RuleAction,
    RulePriority,
    TrafficSample,
    DetectionResult,
    DetectionType,
    Severity,
    TrafficBaseline,
    AnomalyReport,
    MitigationAction,
)
from greenlang.infrastructure.waf_management.config import (
    WAFConfig,
    RuleBuilderConfig,
    AnomalyDetectorConfig,
)


# -----------------------------------------------------------------------------
# WAF Rule Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_rate_limit_rule() -> WAFRule:
    """Create a sample rate limiting rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="API Rate Limit",
        description="Limit API requests to 100/minute per IP",
        rule_type=RuleType.RATE_LIMIT,
        action=RuleAction.BLOCK,
        priority=RulePriority.MEDIUM,
        conditions={
            "path_pattern": "/api/.*",
            "method": ["GET", "POST", "PUT", "DELETE"],
        },
        parameters={
            "limit": 100,
            "window_seconds": 60,
            "key": "ip_address",
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={
            "category": "rate_limiting",
            "compliance": ["soc2"],
        },
    )


@pytest.fixture
def sample_geo_block_rule() -> WAFRule:
    """Create a sample geo-blocking rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="Geo Block High-Risk Countries",
        description="Block traffic from high-risk countries",
        rule_type=RuleType.GEO_BLOCK,
        action=RuleAction.BLOCK,
        priority=RulePriority.HIGH,
        conditions={
            "country_codes": ["CN", "RU", "KP", "IR"],
        },
        parameters={
            "allow_list": ["approved-partner-cn.example.com"],
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={},
    )


@pytest.fixture
def sample_ip_reputation_rule() -> WAFRule:
    """Create a sample IP reputation rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="Block Malicious IPs",
        description="Block IPs from threat intelligence feeds",
        rule_type=RuleType.IP_REPUTATION,
        action=RuleAction.BLOCK,
        priority=RulePriority.HIGH,
        conditions={
            "reputation_threshold": 30,
            "feeds": ["spamhaus", "abuseipdb", "alientvault"],
        },
        parameters={
            "cache_ttl_hours": 24,
            "whitelist": ["10.0.0.0/8", "192.168.0.0/16"],
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={},
    )


@pytest.fixture
def sample_sqli_rule() -> WAFRule:
    """Create a sample SQL injection rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="SQL Injection Detection",
        description="Detect and block SQL injection attempts",
        rule_type=RuleType.SQL_INJECTION,
        action=RuleAction.BLOCK,
        priority=RulePriority.CRITICAL,
        conditions={
            "patterns": [
                r"(?i)(\bunion\b.*\bselect\b)",
                r"(?i)(\bor\b.*\b=\b)",
                r"(?i)(--|\#|/\*)",
                r"(?i)(\bdrop\b.*\btable\b)",
            ],
            "targets": ["query_params", "body", "headers"],
        },
        parameters={
            "sensitivity": "high",
            "false_positive_threshold": 0.01,
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={
            "owasp_category": "A03:2021",
        },
    )


@pytest.fixture
def sample_xss_rule() -> WAFRule:
    """Create a sample XSS rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="XSS Detection",
        description="Detect and block cross-site scripting attempts",
        rule_type=RuleType.XSS,
        action=RuleAction.BLOCK,
        priority=RulePriority.CRITICAL,
        conditions={
            "patterns": [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
            ],
            "targets": ["query_params", "body", "headers"],
        },
        parameters={
            "sensitivity": "high",
            "encode_output": True,
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={
            "owasp_category": "A03:2021",
        },
    )


@pytest.fixture
def sample_custom_regex_rule() -> WAFRule:
    """Create a sample custom regex rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="Block Sensitive File Access",
        description="Block access to sensitive file paths",
        rule_type=RuleType.CUSTOM_REGEX,
        action=RuleAction.BLOCK,
        priority=RulePriority.HIGH,
        conditions={
            "pattern": r"\.(env|git|htaccess|config|bak|sql)$",
            "targets": ["path"],
        },
        parameters={
            "case_insensitive": True,
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={},
    )


@pytest.fixture
def sample_bot_control_rule() -> WAFRule:
    """Create a sample bot control rule."""
    return WAFRule(
        rule_id=str(uuid4()),
        name="Bot Control",
        description="Detect and manage bot traffic",
        rule_type=RuleType.BOT_CONTROL,
        action=RuleAction.CHALLENGE,
        priority=RulePriority.MEDIUM,
        conditions={
            "bot_categories": ["scraper", "crawler", "automation"],
            "exempt_bots": ["googlebot", "bingbot"],
        },
        parameters={
            "challenge_type": "captcha",
            "good_bot_action": "allow",
        },
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="security-team",
        metadata={},
    )


@pytest.fixture
def multiple_rules(
    sample_rate_limit_rule,
    sample_geo_block_rule,
    sample_sqli_rule,
    sample_xss_rule,
    sample_bot_control_rule,
) -> List[WAFRule]:
    """Create a list of multiple WAF rules."""
    return [
        sample_rate_limit_rule,
        sample_geo_block_rule,
        sample_sqli_rule,
        sample_xss_rule,
        sample_bot_control_rule,
    ]


# -----------------------------------------------------------------------------
# Traffic Sample Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def normal_traffic_sample() -> TrafficSample:
    """Create a normal traffic sample."""
    return TrafficSample(
        sample_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        source_ip="192.168.1.100",
        destination_ip="10.0.0.50",
        source_port=54321,
        destination_port=443,
        protocol="HTTPS",
        method="GET",
        path="/api/v1/users/profile",
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        },
        query_params={"include": "settings"},
        body=None,
        response_code=200,
        response_time_ms=45,
        bytes_sent=1024,
        bytes_received=2048,
        country_code="US",
        asn="AS15169",
        metadata={},
    )


@pytest.fixture
def sqli_attack_sample() -> TrafficSample:
    """Create a SQL injection attack traffic sample."""
    return TrafficSample(
        sample_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        source_ip="45.33.32.156",
        destination_ip="10.0.0.50",
        source_port=43210,
        destination_port=443,
        protocol="HTTPS",
        method="POST",
        path="/api/v1/users/login",
        headers={
            "User-Agent": "curl/7.68.0",
            "Content-Type": "application/json",
        },
        query_params={},
        body='{"username": "admin\' OR 1=1--", "password": "test"}',
        response_code=403,
        response_time_ms=5,
        bytes_sent=256,
        bytes_received=128,
        country_code="RO",
        asn="AS44901",
        metadata={"blocked_by": "sqli_rule"},
    )


@pytest.fixture
def xss_attack_sample() -> TrafficSample:
    """Create an XSS attack traffic sample."""
    return TrafficSample(
        sample_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        source_ip="185.220.101.1",
        destination_ip="10.0.0.50",
        source_port=12345,
        destination_port=443,
        protocol="HTTPS",
        method="POST",
        path="/api/v1/comments",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/json",
        },
        query_params={},
        body='{"comment": "<script>alert(document.cookie)</script>"}',
        response_code=403,
        response_time_ms=3,
        bytes_sent=512,
        bytes_received=128,
        country_code="DE",
        asn="AS24940",
        metadata={"blocked_by": "xss_rule"},
    )


@pytest.fixture
def rate_limit_sample() -> TrafficSample:
    """Create a rate limit violation traffic sample."""
    return TrafficSample(
        sample_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        source_ip="203.0.113.50",
        destination_ip="10.0.0.50",
        source_port=55555,
        destination_port=443,
        protocol="HTTPS",
        method="GET",
        path="/api/v1/data",
        headers={
            "User-Agent": "python-requests/2.28.0",
        },
        query_params={},
        body=None,
        response_code=429,
        response_time_ms=2,
        bytes_sent=256,
        bytes_received=64,
        country_code="AU",
        asn="AS7545",
        metadata={"rate_limit_exceeded": True, "request_count": 150},
    )


@pytest.fixture
def bot_traffic_sample() -> TrafficSample:
    """Create a bot traffic sample."""
    return TrafficSample(
        sample_id=str(uuid4()),
        timestamp=datetime.utcnow(),
        source_ip="198.51.100.25",
        destination_ip="10.0.0.50",
        source_port=33333,
        destination_port=443,
        protocol="HTTPS",
        method="GET",
        path="/",
        headers={
            "User-Agent": "Scrapy/2.7.0 (+https://scrapy.org)",
        },
        query_params={},
        body=None,
        response_code=200,
        response_time_ms=100,
        bytes_sent=256,
        bytes_received=50000,
        country_code="NL",
        asn="AS60781",
        metadata={"bot_detected": True, "bot_type": "scraper"},
    )


@pytest.fixture
def multiple_traffic_samples(
    normal_traffic_sample,
    sqli_attack_sample,
    xss_attack_sample,
    rate_limit_sample,
    bot_traffic_sample,
) -> List[TrafficSample]:
    """Create multiple traffic samples for testing."""
    return [
        normal_traffic_sample,
        sqli_attack_sample,
        xss_attack_sample,
        rate_limit_sample,
        bot_traffic_sample,
    ]


# -----------------------------------------------------------------------------
# Detection Result Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sqli_detection_result(sqli_attack_sample) -> DetectionResult:
    """Create a SQL injection detection result."""
    return DetectionResult(
        detection_id=str(uuid4()),
        detection_type=DetectionType.SQL_INJECTION,
        severity=Severity.CRITICAL,
        confidence=0.95,
        traffic_sample=sqli_attack_sample,
        matched_patterns=[r"(?i)(\bor\b.*\b=\b)", r"(?i)(--)"],
        rule_ids=["sqli_rule_1"],
        description="SQL injection attempt detected in login request",
        recommendations=[
            "Block source IP temporarily",
            "Review authentication logs",
            "Enable additional monitoring",
        ],
        detected_at=datetime.utcnow(),
        metadata={
            "attack_category": "injection",
            "owasp_id": "A03:2021",
        },
    )


@pytest.fixture
def ddos_detection_result() -> DetectionResult:
    """Create a DDoS detection result."""
    return DetectionResult(
        detection_id=str(uuid4()),
        detection_type=DetectionType.DDOS,
        severity=Severity.HIGH,
        confidence=0.85,
        traffic_sample=None,  # Aggregate detection
        matched_patterns=[],
        rule_ids=[],
        description="DDoS attack detected: 10x normal traffic volume from multiple sources",
        recommendations=[
            "Enable rate limiting",
            "Activate DDoS mitigation",
            "Contact upstream provider",
        ],
        detected_at=datetime.utcnow(),
        metadata={
            "traffic_multiplier": 10.5,
            "unique_sources": 5000,
            "attack_type": "volumetric",
        },
    )


@pytest.fixture
def anomaly_detection_result(normal_traffic_sample) -> DetectionResult:
    """Create an anomaly detection result."""
    return DetectionResult(
        detection_id=str(uuid4()),
        detection_type=DetectionType.ANOMALY,
        severity=Severity.MEDIUM,
        confidence=0.7,
        traffic_sample=normal_traffic_sample,
        matched_patterns=[],
        rule_ids=[],
        description="Unusual traffic pattern: 3x normal request rate from single IP",
        recommendations=[
            "Monitor IP for 24 hours",
            "Apply rate limiting if pattern continues",
        ],
        detected_at=datetime.utcnow(),
        metadata={
            "baseline_deviation": 3.0,
            "metric": "requests_per_minute",
        },
    )


# -----------------------------------------------------------------------------
# Traffic Baseline Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_traffic_baseline() -> TrafficBaseline:
    """Create a sample traffic baseline."""
    return TrafficBaseline(
        baseline_id=str(uuid4()),
        name="Production API Baseline",
        endpoint_pattern="/api/.*",
        time_window_hours=24,
        metrics={
            "requests_per_minute": {
                "mean": 1000,
                "std_dev": 200,
                "min": 500,
                "max": 2000,
                "p50": 950,
                "p95": 1500,
                "p99": 1800,
            },
            "unique_ips_per_hour": {
                "mean": 5000,
                "std_dev": 1000,
                "min": 3000,
                "max": 8000,
            },
            "error_rate": {
                "mean": 0.02,
                "std_dev": 0.005,
                "min": 0.01,
                "max": 0.05,
            },
            "response_time_ms": {
                "mean": 50,
                "std_dev": 20,
                "min": 10,
                "max": 200,
                "p50": 45,
                "p95": 100,
                "p99": 150,
            },
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        sample_count=1000000,
        metadata={
            "environment": "production",
            "region": "us-east-1",
        },
    )


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def rule_builder_config() -> RuleBuilderConfig:
    """Create rule builder configuration."""
    return RuleBuilderConfig(
        default_action=RuleAction.BLOCK,
        default_priority=RulePriority.MEDIUM,
        enable_logging=True,
        rate_limit_defaults={
            "window_seconds": 60,
            "limit": 100,
        },
        sqli_sensitivity="high",
        xss_sensitivity="high",
        custom_patterns_path="/etc/greenlang/waf/patterns",
    )


@pytest.fixture
def anomaly_detector_config() -> AnomalyDetectorConfig:
    """Create anomaly detector configuration."""
    return AnomalyDetectorConfig(
        baseline_window_hours=24,
        detection_threshold_std_devs=3.0,
        min_samples_for_baseline=1000,
        update_interval_minutes=15,
        metrics_to_monitor=[
            "requests_per_minute",
            "unique_ips_per_hour",
            "error_rate",
            "response_time_ms",
            "bytes_per_request",
        ],
        auto_mitigation_enabled=True,
        auto_mitigation_threshold=Severity.HIGH,
    )


@pytest.fixture
def waf_config(
    rule_builder_config,
    anomaly_detector_config,
) -> WAFConfig:
    """Create full WAF configuration."""
    return WAFConfig(
        rule_builder=rule_builder_config,
        anomaly_detector=anomaly_detector_config,
        enabled=True,
        mode="blocking",  # or "detection"
        log_all_requests=False,
        log_blocked_requests=True,
        notification_channels=["slack", "pagerduty"],
        retention_days=90,
    )


# -----------------------------------------------------------------------------
# Mock Service Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_aws_waf_client():
    """Create mock AWS WAF client."""
    client = MagicMock()
    client.create_rule.return_value = {"RuleId": str(uuid4())}
    client.update_rule.return_value = {}
    client.delete_rule.return_value = {}
    client.list_rules.return_value = {"Rules": []}
    client.get_sampled_requests.return_value = {"SampledRequests": []}
    return client


@pytest.fixture
def mock_cloudflare_client():
    """Create mock Cloudflare client."""
    client = MagicMock()
    client.create_filter.return_value = {"id": str(uuid4())}
    client.create_firewall_rule.return_value = {"id": str(uuid4())}
    client.list_firewall_rules.return_value = {"result": []}
    return client


@pytest.fixture
def mock_metrics_client():
    """Create mock metrics client for traffic data."""
    client = AsyncMock()
    client.query_range.return_value = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"path": "/api/v1/users"},
                    "values": [
                        [1609459200, "1000"],
                        [1609459260, "1050"],
                        [1609459320, "980"],
                    ],
                }
            ]
        },
    }
    return client


@pytest.fixture
def mock_threat_intel_client():
    """Create mock threat intelligence client."""
    client = AsyncMock()
    client.check_ip_reputation.return_value = {
        "ip": "45.33.32.156",
        "reputation_score": 15,  # Low score = bad reputation
        "categories": ["malware", "botnet"],
        "last_seen": datetime.utcnow().isoformat(),
    }
    return client


# -----------------------------------------------------------------------------
# API Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def waf_admin_headers() -> Dict[str, str]:
    """Create headers for WAF admin role."""
    return {
        "Authorization": "Bearer test-waf-admin-token",
        "X-User-Id": "waf-admin",
        "X-User-Roles": "waf-admin,security-analyst,viewer",
    }


@pytest.fixture
def waf_viewer_headers() -> Dict[str, str]:
    """Create headers for WAF viewer role."""
    return {
        "Authorization": "Bearer test-waf-viewer-token",
        "X-User-Id": "waf-viewer",
        "X-User-Roles": "waf-viewer,viewer",
    }
