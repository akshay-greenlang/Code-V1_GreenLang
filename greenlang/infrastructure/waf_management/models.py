# -*- coding: utf-8 -*-
"""
WAF Management Data Models - SEC-010

Pydantic v2 models for the GreenLang WAF and DDoS protection system.
Provides strongly-typed data structures for WAF rules, attack detection,
mitigation results, and traffic metrics.

All datetime fields use UTC. All models enforce strict validation via
Pydantic v2 field validators and model configuration.

Models:
    - RuleType: Enumeration of WAF rule types
    - RuleAction: Actions that can be taken by WAF rules
    - WAFRule: WAF rule definition with conditions and metrics
    - RuleCondition: Targeting condition for rule matching
    - AttackType: Types of DDoS/application attacks
    - Attack: Detected attack with source information
    - MitigationResult: Outcome of attack mitigation
    - TrafficMetrics: Real-time traffic statistics

Example:
    >>> from greenlang.infrastructure.waf_management.models import (
    ...     WAFRule, RuleType, RuleAction, RuleCondition
    ... )
    >>> rule = WAFRule(
    ...     name="Block Known Bad IPs",
    ...     rule_type=RuleType.IP_REPUTATION,
    ...     priority=10,
    ...     action=RuleAction.BLOCK,
    ...     conditions=[RuleCondition(field="ip", operator="in_list", values=["1.2.3.4"])],
    ... )
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RULE_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{1,127}$")
"""Valid rule name: alphanumeric start, 2-128 chars, underscores/hyphens allowed."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RuleType(str, Enum):
    """Types of WAF rules supported by the system.

    Each type determines the evaluation logic and required configuration.
    """

    RATE_LIMIT = "rate_limit"
    """Rate-based rule that blocks IPs exceeding request threshold."""

    GEO_BLOCK = "geo_block"
    """Geographic blocking based on country codes."""

    IP_REPUTATION = "ip_reputation"
    """Block or allow based on IP reputation lists."""

    SQL_INJECTION = "sql_injection"
    """Detect and block SQL injection attacks."""

    XSS = "xss"
    """Detect and block cross-site scripting attacks."""

    CUSTOM_REGEX = "custom_regex"
    """Custom regex pattern matching."""

    BOT_CONTROL = "bot_control"
    """Bot detection and management rules."""

    PATH_TRAVERSAL = "path_traversal"
    """Detect and block path traversal attacks."""

    COMMAND_INJECTION = "command_injection"
    """Detect and block command injection attacks."""

    SIZE_CONSTRAINT = "size_constraint"
    """Limit request body or header sizes."""


class RuleAction(str, Enum):
    """Actions that can be taken when a WAF rule matches.

    Actions are executed in order of severity when multiple rules match.
    """

    ALLOW = "allow"
    """Allow the request to proceed."""

    BLOCK = "block"
    """Block the request with 403 Forbidden."""

    COUNT = "count"
    """Count the request for monitoring without blocking."""

    CAPTCHA = "captcha"
    """Challenge the client with a CAPTCHA."""

    CHALLENGE = "challenge"
    """Challenge the client with a JavaScript challenge."""

    RATE_LIMIT = "rate_limit"
    """Apply rate limiting to the client."""

    LOG = "log"
    """Log the request for analysis."""


class RuleConditionOperator(str, Enum):
    """Operators for rule condition matching."""

    EQUALS = "equals"
    """Exact string match."""

    CONTAINS = "contains"
    """Substring match."""

    STARTS_WITH = "starts_with"
    """Prefix match."""

    ENDS_WITH = "ends_with"
    """Suffix match."""

    REGEX = "regex"
    """Regular expression match."""

    IN_LIST = "in_list"
    """Value is in a list."""

    NOT_IN_LIST = "not_in_list"
    """Value is not in a list."""

    GREATER_THAN = "greater_than"
    """Numeric greater than."""

    LESS_THAN = "less_than"
    """Numeric less than."""

    IP_IN_CIDR = "ip_in_cidr"
    """IP address is in CIDR range."""

    GEO_IN = "geo_in"
    """Request originates from country."""


class AttackType(str, Enum):
    """Types of attacks detected by the anomaly detection system."""

    VOLUMETRIC = "volumetric"
    """High-volume request flood (DDoS)."""

    SLOWLORIS = "slowloris"
    """Slow connection attacks that exhaust server resources."""

    APPLICATION_LAYER = "application_layer"
    """Layer 7 attacks targeting application logic."""

    BOT_FLOOD = "bot_flood"
    """Automated bot traffic flood."""

    CREDENTIAL_STUFFING = "credential_stuffing"
    """Automated credential testing attacks."""

    SCRAPING = "scraping"
    """Aggressive content scraping."""

    API_ABUSE = "api_abuse"
    """API endpoint abuse patterns."""

    SYN_FLOOD = "syn_flood"
    """TCP SYN flood attack."""

    UDP_FLOOD = "udp_flood"
    """UDP packet flood."""

    AMPLIFICATION = "amplification"
    """Reflection/amplification attacks."""


class AttackSeverity(str, Enum):
    """Severity levels for detected attacks."""

    LOW = "low"
    """Minor impact, can be handled with standard rules."""

    MEDIUM = "medium"
    """Moderate impact, requires monitoring."""

    HIGH = "high"
    """Significant impact, requires immediate action."""

    CRITICAL = "critical"
    """Severe impact, emergency response required."""


class MitigationStatus(str, Enum):
    """Status of attack mitigation efforts."""

    PENDING = "pending"
    """Mitigation not yet started."""

    IN_PROGRESS = "in_progress"
    """Mitigation actions being applied."""

    MITIGATED = "mitigated"
    """Attack successfully mitigated."""

    FAILED = "failed"
    """Mitigation failed, escalation required."""

    ESCALATED = "escalated"
    """Escalated to AWS Shield Response Team."""


class RuleStatus(str, Enum):
    """Lifecycle status of a WAF rule."""

    DRAFT = "draft"
    """Rule is defined but not deployed."""

    TESTING = "testing"
    """Rule is being tested."""

    ACTIVE = "active"
    """Rule is deployed and active."""

    DISABLED = "disabled"
    """Rule is disabled but not deleted."""

    ARCHIVED = "archived"
    """Rule is archived and no longer evaluating."""


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class RuleCondition(BaseModel):
    """Targeting condition for WAF rule matching.

    Conditions define what aspects of a request to inspect and how
    to match them. Multiple conditions can be combined with AND logic.

    Attributes:
        field: Request field to inspect (ip, uri, headers, body, etc.).
        operator: Comparison operator for matching.
        values: List of values to match against.
        negated: If True, inverts the match result.
        transform: Optional text transformation before matching.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    field: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Request field to inspect (ip, uri, headers, body, query_string, method).",
    )
    operator: RuleConditionOperator = Field(
        default=RuleConditionOperator.EQUALS,
        description="Comparison operator for matching.",
    )
    values: List[str] = Field(
        default_factory=list,
        description="Values to match against.",
    )
    negated: bool = Field(
        default=False,
        description="If True, inverts the match result (NOT operator).",
    )
    transform: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Text transformation: lowercase, uppercase, url_decode, html_decode, compress_whitespace.",
    )

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate field name is a known request attribute."""
        allowed_fields = {
            "ip",
            "source_ip",
            "uri",
            "uri_path",
            "query_string",
            "body",
            "method",
            "headers",
            "user_agent",
            "referer",
            "cookie",
            "host",
            "country",
            "region",
            "content_type",
            "content_length",
        }
        v_lower = v.strip().lower()
        if v_lower not in allowed_fields:
            raise ValueError(
                f"Invalid field '{v}'. Allowed fields: {sorted(allowed_fields)}"
            )
        return v_lower

    @field_validator("transform")
    @classmethod
    def validate_transform(cls, v: Optional[str]) -> Optional[str]:
        """Validate transformation function."""
        if v is None:
            return v
        allowed_transforms = {
            "lowercase",
            "uppercase",
            "url_decode",
            "html_decode",
            "compress_whitespace",
            "base64_decode",
            "hex_decode",
            "none",
        }
        v_lower = v.strip().lower()
        if v_lower not in allowed_transforms:
            raise ValueError(
                f"Invalid transform '{v}'. Allowed: {sorted(allowed_transforms)}"
            )
        return v_lower


class WAFRuleMetrics(BaseModel):
    """Metrics associated with a WAF rule.

    Tracks rule performance and effectiveness over time.

    Attributes:
        requests_evaluated: Total requests evaluated against this rule.
        requests_matched: Requests that matched this rule.
        requests_blocked: Requests blocked by this rule.
        false_positives_reported: Reported false positives.
        average_latency_ms: Average evaluation latency in milliseconds.
        last_matched_at: Last time a request matched this rule.
    """

    model_config = ConfigDict(extra="forbid")

    requests_evaluated: int = Field(
        default=0,
        ge=0,
        description="Total requests evaluated against this rule.",
    )
    requests_matched: int = Field(
        default=0,
        ge=0,
        description="Requests that matched this rule.",
    )
    requests_blocked: int = Field(
        default=0,
        ge=0,
        description="Requests blocked by this rule.",
    )
    false_positives_reported: int = Field(
        default=0,
        ge=0,
        description="Reported false positives.",
    )
    average_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average evaluation latency in milliseconds.",
    )
    last_matched_at: Optional[datetime] = Field(
        default=None,
        description="Last time a request matched this rule.",
    )


class WAFRule(BaseModel):
    """WAF rule definition for request filtering.

    Represents a complete WAF rule with its conditions, action,
    and associated metadata. Rules are evaluated in priority order.

    Attributes:
        id: Unique identifier for the rule (UUID).
        name: Human-readable rule name.
        description: Detailed description of rule purpose.
        rule_type: Type of WAF rule (rate_limit, geo_block, etc.).
        priority: Evaluation priority (lower = evaluated first).
        action: Action to take when rule matches.
        conditions: List of conditions that must match (AND logic).
        enabled: Whether rule is actively evaluating.
        status: Lifecycle status of the rule.
        rate_limit_threshold: For rate_limit rules, requests per window.
        rate_limit_window_seconds: For rate_limit rules, window duration.
        blocked_countries: For geo_block rules, country codes to block.
        ip_set_arn: For IP reputation rules, ARN of IP set.
        regex_pattern: For custom_regex rules, the pattern.
        managed_rule_group: For managed rules, the AWS managed rule group name.
        aws_rule_id: AWS WAF rule ID after deployment.
        created_at: When the rule was created.
        updated_at: When the rule was last modified.
        deployed_at: When the rule was last deployed to AWS WAF.
        created_by: User who created the rule.
        metrics: Rule performance metrics.
        metadata: Arbitrary key-value metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "RateLimitPerIP",
                    "rule_type": "rate_limit",
                    "priority": 1,
                    "action": "block",
                    "rate_limit_threshold": 2000,
                    "rate_limit_window_seconds": 300,
                }
            ]
        },
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique rule identifier (UUID).",
    )
    name: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Human-readable rule name.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Detailed description of the rule's purpose.",
    )
    rule_type: RuleType = Field(
        ...,
        description="Type of WAF rule.",
    )
    priority: int = Field(
        default=100,
        ge=0,
        le=10000,
        description="Evaluation priority (lower = higher priority).",
    )
    action: RuleAction = Field(
        default=RuleAction.BLOCK,
        description="Action to take when rule matches.",
    )
    conditions: List[RuleCondition] = Field(
        default_factory=list,
        description="Conditions that must match (AND logic).",
    )
    enabled: bool = Field(
        default=True,
        description="Whether rule is actively evaluating.",
    )
    status: RuleStatus = Field(
        default=RuleStatus.DRAFT,
        description="Lifecycle status of the rule.",
    )

    # Rate limiting specific
    rate_limit_threshold: int = Field(
        default=2000,
        ge=100,
        le=100000000,
        description="For rate_limit rules: requests per window before blocking.",
    )
    rate_limit_window_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="For rate_limit rules: evaluation window in seconds.",
    )

    # Geo-blocking specific
    blocked_countries: List[str] = Field(
        default_factory=list,
        description="For geo_block rules: ISO 3166-1 alpha-2 country codes to block.",
    )

    # IP reputation specific
    ip_set_arn: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="For ip_reputation rules: ARN of AWS WAF IP set.",
    )

    # Custom regex specific
    regex_pattern: Optional[str] = Field(
        default=None,
        max_length=512,
        description="For custom_regex rules: regex pattern to match.",
    )

    # Managed rule groups
    managed_rule_group: Optional[str] = Field(
        default=None,
        max_length=256,
        description="AWS managed rule group name (e.g., AWSManagedRulesCommonRuleSet).",
    )

    # AWS deployment state
    aws_rule_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="AWS WAF rule ID after deployment.",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )
    deployed_at: Optional[datetime] = Field(
        default=None,
        description="Last deployment timestamp (UTC).",
    )

    # Audit
    created_by: str = Field(
        default="",
        max_length=256,
        description="User or service that created this rule.",
    )

    # Metrics and metadata
    metrics: WAFRuleMetrics = Field(
        default_factory=WAFRuleMetrics,
        description="Rule performance metrics.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )

    # -- Field Validators --------------------------------------------------

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Ensure rule name matches the required pattern."""
        if not _RULE_NAME_PATTERN.match(v):
            raise ValueError(
                f"Rule name '{v}' is invalid. Must start with a letter, "
                f"contain only [a-zA-Z0-9_-], and be 2-128 characters."
            )
        return v

    @field_validator("blocked_countries")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Validate and normalize country codes to uppercase."""
        normalized: List[str] = []
        for code in v:
            code_upper = code.strip().upper()
            if len(code_upper) != 2 or not code_upper.isalpha():
                raise ValueError(
                    f"Invalid country code '{code}'. Must be ISO 3166-1 alpha-2 (2 letters)."
                )
            normalized.append(code_upper)
        return list(set(normalized))  # Deduplicate

    @field_validator("regex_pattern")
    @classmethod
    def validate_regex_pattern(cls, v: Optional[str]) -> Optional[str]:
        """Validate regex pattern is syntactically correct."""
        if v is None:
            return v
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        return v

    @field_validator("created_at", "updated_at", "deployed_at")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    # -- Model Validators ---------------------------------------------------

    @model_validator(mode="after")
    def validate_rule_type_requirements(self) -> "WAFRule":
        """Validate rule-type-specific requirements."""
        if self.rule_type == RuleType.GEO_BLOCK and not self.blocked_countries:
            raise ValueError("geo_block rules require at least one blocked_countries entry.")

        if self.rule_type == RuleType.CUSTOM_REGEX and not self.regex_pattern:
            raise ValueError("custom_regex rules require a regex_pattern.")

        return self


class Attack(BaseModel):
    """Detected attack with source and timing information.

    Represents an attack detected by the anomaly detection system,
    including source information, magnitude, and timeline.

    Attributes:
        id: Unique attack identifier.
        attack_type: Type of attack detected.
        severity: Attack severity level.
        source_ips: IP addresses involved in the attack.
        target_endpoints: Endpoints being targeted.
        requests_per_second: Peak request rate during attack.
        total_requests: Total malicious requests.
        bytes_per_second: Traffic volume in bytes/sec.
        started_at: When the attack was first detected.
        detected_at: When the system detected the attack.
        mitigated_at: When the attack was successfully mitigated.
        ended_at: When the attack stopped.
        status: Current mitigation status.
        detection_source: How the attack was detected.
        attack_signature: Pattern signature if identified.
        geographic_distribution: Source country distribution.
        metadata: Additional attack metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attack identifier.",
    )
    attack_type: AttackType = Field(
        ...,
        description="Type of attack detected.",
    )
    severity: AttackSeverity = Field(
        default=AttackSeverity.MEDIUM,
        description="Attack severity level.",
    )
    source_ips: List[str] = Field(
        default_factory=list,
        description="IP addresses involved in the attack.",
    )
    target_endpoints: List[str] = Field(
        default_factory=list,
        description="API endpoints being targeted.",
    )
    requests_per_second: int = Field(
        default=0,
        ge=0,
        description="Peak request rate during attack.",
    )
    total_requests: int = Field(
        default=0,
        ge=0,
        description="Total malicious requests observed.",
    )
    bytes_per_second: int = Field(
        default=0,
        ge=0,
        description="Traffic volume in bytes per second.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the attack first started.",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the system detected the attack.",
    )
    mitigated_at: Optional[datetime] = Field(
        default=None,
        description="When the attack was successfully mitigated.",
    )
    ended_at: Optional[datetime] = Field(
        default=None,
        description="When the attack stopped.",
    )
    status: MitigationStatus = Field(
        default=MitigationStatus.PENDING,
        description="Current mitigation status.",
    )
    detection_source: str = Field(
        default="anomaly_detector",
        max_length=128,
        description="How the attack was detected (anomaly_detector, shield, waf, manual).",
    )
    attack_signature: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Attack pattern signature if identified.",
    )
    geographic_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Request count by source country.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attack metadata.",
    )

    @field_validator("source_ips")
    @classmethod
    def validate_source_ips(cls, v: List[str]) -> List[str]:
        """Basic validation of IP addresses."""
        validated: List[str] = []
        for ip in v:
            ip_stripped = ip.strip()
            # Basic length check for IPv4/IPv6
            if len(ip_stripped) < 3 or len(ip_stripped) > 45:
                continue  # Skip invalid IPs
            validated.append(ip_stripped)
        return validated

    @field_validator("started_at", "detected_at", "mitigated_at", "ended_at")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate attack duration in seconds if ended."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    @property
    def detection_delay_seconds(self) -> float:
        """Calculate time from attack start to detection."""
        return (self.detected_at - self.started_at).total_seconds()


class MitigationAction(BaseModel):
    """Individual mitigation action taken during attack response.

    Attributes:
        action_type: Type of mitigation action.
        target: Target of the action (IP, endpoint, etc.).
        timestamp: When the action was taken.
        success: Whether the action succeeded.
        details: Additional action details.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(
        ...,
        max_length=64,
        description="Type of action: rate_limit, geo_block, ip_block, scale, shield_engage.",
    )
    target: str = Field(
        default="",
        max_length=512,
        description="Target of the action (IP, endpoint, resource ARN).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the action was taken.",
    )
    success: bool = Field(
        default=True,
        description="Whether the action succeeded.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional action details.",
    )


class MitigationResult(BaseModel):
    """Outcome of attack mitigation efforts.

    Tracks all mitigation actions taken and their effectiveness.

    Attributes:
        id: Unique mitigation result identifier.
        attack_id: ID of the attack being mitigated.
        actions_taken: List of mitigation actions performed.
        effectiveness_score: Overall mitigation effectiveness (0.0-1.0).
        traffic_reduction_percent: Percentage of malicious traffic blocked.
        false_positive_rate: Estimated false positive rate.
        duration_seconds: Total mitigation duration.
        started_at: When mitigation started.
        completed_at: When mitigation completed.
        status: Final mitigation status.
        shield_engaged: Whether AWS Shield DRT was engaged.
        rules_created: WAF rules created during mitigation.
        recommendations: Post-incident recommendations.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mitigation result identifier.",
    )
    attack_id: str = Field(
        ...,
        description="ID of the attack being mitigated.",
    )
    actions_taken: List[MitigationAction] = Field(
        default_factory=list,
        description="Mitigation actions performed.",
    )
    effectiveness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall mitigation effectiveness (0.0-1.0).",
    )
    traffic_reduction_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of malicious traffic blocked.",
    )
    false_positive_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated false positive rate.",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total mitigation duration in seconds.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When mitigation started.",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When mitigation completed.",
    )
    status: MitigationStatus = Field(
        default=MitigationStatus.IN_PROGRESS,
        description="Final mitigation status.",
    )
    shield_engaged: bool = Field(
        default=False,
        description="Whether AWS Shield DRT was engaged.",
    )
    rules_created: List[str] = Field(
        default_factory=list,
        description="IDs of WAF rules created during mitigation.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Post-incident recommendations.",
    )


class TrafficMetrics(BaseModel):
    """Real-time traffic statistics for anomaly detection.

    Captures traffic patterns at a point in time for baseline
    comparison and attack detection.

    Attributes:
        timestamp: When these metrics were captured.
        requests_per_second: Total requests per second.
        blocked_per_second: Blocked requests per second.
        allowed_per_second: Allowed requests per second.
        latency_p50_ms: 50th percentile latency in milliseconds.
        latency_p95_ms: 95th percentile latency.
        latency_p99_ms: 99th percentile latency.
        error_rate: Percentage of 5xx responses.
        unique_ips: Count of unique source IPs.
        bytes_per_second: Total bytes per second.
        endpoint_breakdown: Requests per endpoint.
        country_breakdown: Requests per country.
        user_agent_breakdown: Requests per user agent category.
        status_code_breakdown: Requests per HTTP status code.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When these metrics were captured.",
    )
    requests_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Total requests per second.",
    )
    blocked_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Blocked requests per second.",
    )
    allowed_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Allowed requests per second.",
    )
    latency_p50_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="50th percentile latency in milliseconds.",
    )
    latency_p95_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="95th percentile latency in milliseconds.",
    )
    latency_p99_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="99th percentile latency in milliseconds.",
    )
    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of 5xx responses.",
    )
    unique_ips: int = Field(
        default=0,
        ge=0,
        description="Count of unique source IP addresses.",
    )
    bytes_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Total bytes per second.",
    )
    endpoint_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Request count per endpoint.",
    )
    country_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Request count per source country.",
    )
    user_agent_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Request count per user agent category (browser, bot, mobile, etc.).",
    )
    status_code_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Request count per HTTP status code.",
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc_datetime(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class ShieldProtection(BaseModel):
    """AWS Shield Advanced protection configuration.

    Attributes:
        id: Protection ID from AWS Shield.
        resource_arn: ARN of the protected resource.
        protection_name: Human-readable protection name.
        health_check_arn: Optional Route 53 health check ARN.
        auto_remediate: Whether auto-remediation is enabled.
        proactive_engagement: Whether DRT proactive engagement is enabled.
        created_at: When protection was enabled.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Protection ID from AWS Shield.",
    )
    resource_arn: str = Field(
        ...,
        max_length=2048,
        description="ARN of the protected resource.",
    )
    protection_name: str = Field(
        default="",
        max_length=256,
        description="Human-readable protection name.",
    )
    health_check_arn: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Optional Route 53 health check ARN.",
    )
    auto_remediate: bool = Field(
        default=True,
        description="Whether auto-remediation is enabled.",
    )
    proactive_engagement: bool = Field(
        default=False,
        description="Whether DRT proactive engagement is enabled.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When protection was enabled.",
    )


class ProtectionGroup(BaseModel):
    """AWS Shield protection group configuration.

    Attributes:
        id: Protection group ID.
        aggregation: Aggregation type (SUM, MEAN, MAX).
        pattern: Resource pattern (ALL, ARBITRARY, BY_RESOURCE_TYPE).
        resource_type: Optional resource type filter.
        members: List of resource ARNs in the group.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Protection group ID.",
    )
    aggregation: str = Field(
        default="SUM",
        description="Aggregation type: SUM, MEAN, MAX.",
    )
    pattern: str = Field(
        default="ARBITRARY",
        description="Resource pattern: ALL, ARBITRARY, BY_RESOURCE_TYPE.",
    )
    resource_type: Optional[str] = Field(
        default=None,
        description="Resource type filter.",
    )
    members: List[str] = Field(
        default_factory=list,
        description="List of resource ARNs in the group.",
    )


__all__ = [
    # Enums
    "RuleType",
    "RuleAction",
    "RuleConditionOperator",
    "AttackType",
    "AttackSeverity",
    "MitigationStatus",
    "RuleStatus",
    # Models
    "RuleCondition",
    "WAFRuleMetrics",
    "WAFRule",
    "Attack",
    "MitigationAction",
    "MitigationResult",
    "TrafficMetrics",
    "ShieldProtection",
    "ProtectionGroup",
]
