# -*- coding: utf-8 -*-
"""
Incident Response Configuration - SEC-010

Configuration dataclasses for the incident response automation module.
Provides settings for alert sources, escalation thresholds, notification
channels, and playbook execution.

Follows the GreenLang pattern of dataclass-based configuration with
sensible defaults and environment variable overrides.

Example:
    >>> from greenlang.infrastructure.incident_response.config import (
    ...     IncidentResponseConfig,
    ... )
    >>> config = IncidentResponseConfig.from_environment()
    >>> print(config.escalation_thresholds["P0"])

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation Thresholds (minutes)
# ---------------------------------------------------------------------------

DEFAULT_ESCALATION_THRESHOLDS: Dict[str, int] = {
    "P0": 15,      # Critical - 15 minutes
    "P1": 60,      # High - 1 hour
    "P2": 240,     # Medium - 4 hours
    "P3": 1440,    # Low - 24 hours
}

# Response time SLAs (minutes)
DEFAULT_RESPONSE_SLAS: Dict[str, int] = {
    "P0": 5,       # Critical - 5 min acknowledge
    "P1": 15,      # High - 15 min acknowledge
    "P2": 60,      # Medium - 1 hour acknowledge
    "P3": 240,     # Low - 4 hours acknowledge
}

# Resolution time SLAs (minutes)
DEFAULT_RESOLUTION_SLAS: Dict[str, int] = {
    "P0": 60,      # Critical - 1 hour resolution
    "P1": 240,     # High - 4 hours resolution
    "P2": 1440,    # Medium - 24 hours resolution
    "P3": 10080,   # Low - 7 days resolution
}


# ---------------------------------------------------------------------------
# Alert Source Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlertSourceConfig:
    """Configuration for an alert source.

    Attributes:
        name: Alert source identifier.
        enabled: Whether this source is enabled.
        endpoint: API endpoint URL.
        api_key_env: Environment variable name for API key.
        poll_interval_seconds: How often to poll (for pull-based sources).
        timeout_seconds: Request timeout.
        verify_ssl: Whether to verify SSL certificates.
        extra_headers: Additional HTTP headers.
    """

    name: str
    enabled: bool = True
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    poll_interval_seconds: int = 30
    timeout_seconds: int = 30
    verify_ssl: bool = True
    extra_headers: Dict[str, str] = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable.

        Returns:
            API key or None if not set.
        """
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class PrometheusAlertConfig(AlertSourceConfig):
    """Prometheus Alertmanager configuration.

    Attributes:
        alertmanager_url: Alertmanager API URL.
        silence_duration_seconds: Default silence duration.
        label_filters: Labels to filter alerts.
    """

    alertmanager_url: str = "http://alertmanager:9093"
    silence_duration_seconds: int = 3600
    label_filters: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        self.name = "prometheus"
        if not self.endpoint:
            self.endpoint = f"{self.alertmanager_url}/api/v2/alerts"


@dataclass
class LokiAlertConfig(AlertSourceConfig):
    """Loki log aggregation configuration.

    Attributes:
        loki_url: Loki API URL.
        queries: LogQL queries for incident detection.
        lookback_seconds: Time window for queries.
    """

    loki_url: str = "http://loki:3100"
    queries: Dict[str, str] = field(default_factory=dict)
    lookback_seconds: int = 300

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        self.name = "loki"
        if not self.endpoint:
            self.endpoint = f"{self.loki_url}/loki/api/v1/query_range"
        if not self.queries:
            self.queries = {
                "security_errors": '{level="error"} |~ "security|unauthorized|forbidden"',
                "auth_failures": '{job="auth-service"} |~ "authentication failed"',
                "rate_limit": '{job="api-gateway"} |~ "rate limit exceeded"',
            }


@dataclass
class GuardDutyConfig(AlertSourceConfig):
    """AWS GuardDuty configuration.

    Attributes:
        region: AWS region.
        detector_id: GuardDuty detector ID (auto-detected if not provided).
        min_severity: Minimum severity to fetch (0.0-10.0).
        finding_types: Specific finding types to monitor.
    """

    region: str = "us-east-1"
    detector_id: Optional[str] = None
    min_severity: float = 4.0
    finding_types: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        self.name = "guardduty"
        self.api_key_env = "AWS_ACCESS_KEY_ID"


@dataclass
class CloudTrailConfig(AlertSourceConfig):
    """AWS CloudTrail anomaly detection configuration.

    Attributes:
        region: AWS region.
        trail_name: CloudTrail trail name.
        log_group_name: CloudWatch log group for CloudTrail.
        anomaly_detector_arn: CloudWatch Anomaly Detector ARN.
    """

    region: str = "us-east-1"
    trail_name: str = "greenlang-cloudtrail"
    log_group_name: str = "/aws/cloudtrail/greenlang"
    anomaly_detector_arn: Optional[str] = None

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        self.name = "cloudtrail"
        self.api_key_env = "AWS_ACCESS_KEY_ID"


# ---------------------------------------------------------------------------
# Notification Configuration
# ---------------------------------------------------------------------------


@dataclass
class PagerDutyConfig:
    """PagerDuty integration configuration.

    Attributes:
        enabled: Whether PagerDuty is enabled.
        routing_key: Events API v2 routing key.
        api_key: REST API key for lookups.
        base_url: PagerDuty API base URL.
        default_severity: Default PagerDuty severity.
        dedup_key_prefix: Prefix for deduplication keys.
    """

    enabled: bool = True
    routing_key: Optional[str] = None
    api_key: Optional[str] = None
    base_url: str = "https://events.pagerduty.com/v2/enqueue"
    api_base_url: str = "https://api.pagerduty.com"
    default_severity: str = "error"
    dedup_key_prefix: str = "gl-incident"

    def get_routing_key(self) -> Optional[str]:
        """Get routing key from config or environment."""
        return self.routing_key or os.environ.get("PAGERDUTY_ROUTING_KEY")

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        return self.api_key or os.environ.get("PAGERDUTY_API_KEY")


@dataclass
class SlackConfig:
    """Slack integration configuration.

    Attributes:
        enabled: Whether Slack is enabled.
        webhook_url: Slack Incoming Webhook URL.
        bot_token: Slack Bot Token for advanced features.
        default_channel: Default channel for notifications.
        channels_by_severity: Channel mapping by severity.
        mention_users: User IDs to mention for critical incidents.
    """

    enabled: bool = True
    webhook_url: Optional[str] = None
    bot_token: Optional[str] = None
    default_channel: str = "#security-incidents"
    channels_by_severity: Dict[str, str] = field(default_factory=dict)
    mention_users: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.channels_by_severity:
            self.channels_by_severity = {
                "P0": "#security-critical",
                "P1": "#security-high",
                "P2": "#security-incidents",
                "P3": "#security-incidents",
            }
        if not self.mention_users:
            self.mention_users = {
                "P0": [],  # Add user IDs here
                "P1": [],
            }

    def get_webhook_url(self) -> Optional[str]:
        """Get webhook URL from config or environment."""
        return self.webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def get_bot_token(self) -> Optional[str]:
        """Get bot token from config or environment."""
        return self.bot_token or os.environ.get("SLACK_BOT_TOKEN")


@dataclass
class EmailConfig:
    """Email notification configuration.

    Attributes:
        enabled: Whether email is enabled.
        smtp_host: SMTP server host.
        smtp_port: SMTP server port.
        smtp_user: SMTP username.
        smtp_password_env: Environment variable for password.
        use_tls: Whether to use TLS.
        from_address: From email address.
        reply_to: Reply-to address.
        recipients_by_severity: Recipients by severity level.
    """

    enabled: bool = True
    smtp_host: str = "email-smtp.us-east-1.amazonaws.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password_env: str = "SMTP_PASSWORD"
    use_tls: bool = True
    from_address: str = "security@greenlang.io"
    reply_to: str = "security@greenlang.io"
    recipients_by_severity: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.recipients_by_severity:
            self.recipients_by_severity = {
                "P0": ["security-oncall@greenlang.io", "ciso@greenlang.io"],
                "P1": ["security-oncall@greenlang.io"],
                "P2": ["security-team@greenlang.io"],
                "P3": ["security-team@greenlang.io"],
            }

    def get_smtp_password(self) -> Optional[str]:
        """Get SMTP password from environment."""
        return os.environ.get(self.smtp_password_env)


@dataclass
class SMSConfig:
    """SMS notification configuration (AWS SNS).

    Attributes:
        enabled: Whether SMS is enabled.
        region: AWS region.
        topic_arn: SNS topic ARN for SMS.
        phone_numbers_by_severity: Phone numbers by severity.
    """

    enabled: bool = True
    region: str = "us-east-1"
    topic_arn: Optional[str] = None
    phone_numbers_by_severity: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.phone_numbers_by_severity:
            self.phone_numbers_by_severity = {
                "P0": [],  # Add phone numbers for critical incidents
            }


# ---------------------------------------------------------------------------
# Playbook Configuration
# ---------------------------------------------------------------------------


@dataclass
class PlaybookConfig:
    """Playbook execution configuration.

    Attributes:
        enabled: Whether automated playbook execution is enabled.
        dry_run: Run playbooks in dry-run mode (no actual changes).
        max_concurrent_executions: Maximum concurrent playbook runs.
        step_timeout_seconds: Default timeout per playbook step.
        total_timeout_seconds: Maximum total playbook execution time.
        retry_on_failure: Whether to retry failed steps.
        max_retries: Maximum retry attempts.
        rollback_on_failure: Whether to rollback on step failure.
        require_approval_for: Incident types requiring manual approval.
    """

    enabled: bool = True
    dry_run: bool = False
    max_concurrent_executions: int = 5
    step_timeout_seconds: int = 300
    total_timeout_seconds: int = 1800
    retry_on_failure: bool = True
    max_retries: int = 3
    rollback_on_failure: bool = True
    require_approval_for: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set defaults after initialization."""
        if not self.require_approval_for:
            self.require_approval_for = [
                "data_breach",
                "credential_compromise",
            ]


# ---------------------------------------------------------------------------
# Tracking Configuration
# ---------------------------------------------------------------------------


@dataclass
class JiraConfig:
    """Jira integration configuration.

    Attributes:
        enabled: Whether Jira is enabled.
        base_url: Jira instance URL.
        project_key: Default project key.
        issue_type: Default issue type.
        api_token_env: Environment variable for API token.
        user_email: Jira user email.
        custom_fields: Custom field mappings.
    """

    enabled: bool = True
    base_url: Optional[str] = None
    project_key: str = "SEC"
    issue_type: str = "Incident"
    api_token_env: str = "JIRA_API_TOKEN"
    user_email: Optional[str] = None
    custom_fields: Dict[str, str] = field(default_factory=dict)

    def get_api_token(self) -> Optional[str]:
        """Get API token from environment."""
        return os.environ.get(self.api_token_env)


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------


@dataclass
class IncidentResponseConfig:
    """Main configuration for incident response automation.

    Attributes:
        environment: Deployment environment (dev, staging, prod).
        enable_auto_detection: Enable automatic incident detection.
        enable_auto_escalation: Enable automatic escalation.
        enable_auto_remediation: Enable automated playbook execution.
        correlation_window_seconds: Time window for alert correlation.
        deduplication_window_seconds: Time window for deduplication.
        incident_number_prefix: Prefix for incident numbers.
        escalation_thresholds: Minutes before escalation by priority.
        response_slas: Response time SLAs by priority.
        resolution_slas: Resolution time SLAs by priority.
        prometheus: Prometheus Alertmanager config.
        loki: Loki config.
        guardduty: AWS GuardDuty config.
        cloudtrail: AWS CloudTrail config.
        pagerduty: PagerDuty config.
        slack: Slack config.
        email: Email config.
        sms: SMS config.
        playbook: Playbook execution config.
        jira: Jira config.
        database_url: PostgreSQL connection URL.
        redis_url: Redis connection URL.
    """

    environment: str = "production"
    enable_auto_detection: bool = True
    enable_auto_escalation: bool = True
    enable_auto_remediation: bool = True
    correlation_window_seconds: int = 300  # 5 minutes
    deduplication_window_seconds: int = 3600  # 1 hour
    incident_number_prefix: str = "INC"

    # SLA Configuration
    escalation_thresholds: Dict[str, int] = field(
        default_factory=lambda: DEFAULT_ESCALATION_THRESHOLDS.copy()
    )
    response_slas: Dict[str, int] = field(
        default_factory=lambda: DEFAULT_RESPONSE_SLAS.copy()
    )
    resolution_slas: Dict[str, int] = field(
        default_factory=lambda: DEFAULT_RESOLUTION_SLAS.copy()
    )

    # Alert Sources
    prometheus: PrometheusAlertConfig = field(default_factory=PrometheusAlertConfig)
    loki: LokiAlertConfig = field(default_factory=LokiAlertConfig)
    guardduty: GuardDutyConfig = field(default_factory=GuardDutyConfig)
    cloudtrail: CloudTrailConfig = field(default_factory=CloudTrailConfig)

    # Notification Channels
    pagerduty: PagerDutyConfig = field(default_factory=PagerDutyConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    sms: SMSConfig = field(default_factory=SMSConfig)

    # Playbook Execution
    playbook: PlaybookConfig = field(default_factory=PlaybookConfig)

    # Tracking
    jira: JiraConfig = field(default_factory=JiraConfig)

    # Database
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    @classmethod
    def from_environment(cls) -> IncidentResponseConfig:
        """Create configuration from environment variables.

        Environment variables:
            GL_ENVIRONMENT: Deployment environment
            GL_INCIDENT_AUTO_DETECTION: Enable auto detection (true/false)
            GL_INCIDENT_AUTO_ESCALATION: Enable auto escalation (true/false)
            GL_INCIDENT_AUTO_REMEDIATION: Enable auto remediation (true/false)
            GL_INCIDENT_CORRELATION_WINDOW: Correlation window in seconds
            GL_INCIDENT_DEDUP_WINDOW: Deduplication window in seconds
            GL_INCIDENT_NUMBER_PREFIX: Incident number prefix
            GL_DATABASE_URL: PostgreSQL connection URL
            GL_REDIS_URL: Redis connection URL

        Alert Source URLs:
            GL_ALERTMANAGER_URL: Prometheus Alertmanager URL
            GL_LOKI_URL: Loki URL
            AWS_REGION: AWS region for GuardDuty/CloudTrail

        Notification Channels:
            PAGERDUTY_ROUTING_KEY: PagerDuty Events API routing key
            PAGERDUTY_API_KEY: PagerDuty REST API key
            SLACK_WEBHOOK_URL: Slack webhook URL
            SLACK_BOT_TOKEN: Slack bot token
            SMTP_PASSWORD: SMTP password for email

        Returns:
            Configuration populated from environment.
        """
        def _bool_env(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        def _int_env(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, str(default)))
            except ValueError:
                return default

        # Create alert source configs
        prometheus = PrometheusAlertConfig(
            alertmanager_url=os.environ.get(
                "GL_ALERTMANAGER_URL", "http://alertmanager:9093"
            ),
            enabled=_bool_env("GL_PROMETHEUS_ENABLED", True),
        )

        loki = LokiAlertConfig(
            loki_url=os.environ.get("GL_LOKI_URL", "http://loki:3100"),
            enabled=_bool_env("GL_LOKI_ENABLED", True),
        )

        guardduty = GuardDutyConfig(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            enabled=_bool_env("GL_GUARDDUTY_ENABLED", True),
        )

        cloudtrail = CloudTrailConfig(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            enabled=_bool_env("GL_CLOUDTRAIL_ENABLED", True),
        )

        # Create notification configs
        pagerduty = PagerDutyConfig(
            enabled=_bool_env("GL_PAGERDUTY_ENABLED", True),
        )

        slack = SlackConfig(
            enabled=_bool_env("GL_SLACK_ENABLED", True),
            default_channel=os.environ.get(
                "GL_SLACK_DEFAULT_CHANNEL", "#security-incidents"
            ),
        )

        email = EmailConfig(
            enabled=_bool_env("GL_EMAIL_ENABLED", True),
            smtp_host=os.environ.get(
                "GL_SMTP_HOST", "email-smtp.us-east-1.amazonaws.com"
            ),
            smtp_port=_int_env("GL_SMTP_PORT", 587),
        )

        sms = SMSConfig(
            enabled=_bool_env("GL_SMS_ENABLED", True),
            region=os.environ.get("AWS_REGION", "us-east-1"),
        )

        # Create playbook config
        playbook = PlaybookConfig(
            enabled=_bool_env("GL_PLAYBOOK_ENABLED", True),
            dry_run=_bool_env("GL_PLAYBOOK_DRY_RUN", False),
            max_concurrent_executions=_int_env("GL_PLAYBOOK_MAX_CONCURRENT", 5),
        )

        # Create Jira config
        jira = JiraConfig(
            enabled=_bool_env("GL_JIRA_ENABLED", True),
            base_url=os.environ.get("GL_JIRA_URL"),
            project_key=os.environ.get("GL_JIRA_PROJECT", "SEC"),
        )

        return cls(
            environment=os.environ.get("GL_ENVIRONMENT", "production"),
            enable_auto_detection=_bool_env("GL_INCIDENT_AUTO_DETECTION", True),
            enable_auto_escalation=_bool_env("GL_INCIDENT_AUTO_ESCALATION", True),
            enable_auto_remediation=_bool_env("GL_INCIDENT_AUTO_REMEDIATION", True),
            correlation_window_seconds=_int_env("GL_INCIDENT_CORRELATION_WINDOW", 300),
            deduplication_window_seconds=_int_env("GL_INCIDENT_DEDUP_WINDOW", 3600),
            incident_number_prefix=os.environ.get("GL_INCIDENT_NUMBER_PREFIX", "INC"),
            prometheus=prometheus,
            loki=loki,
            guardduty=guardduty,
            cloudtrail=cloudtrail,
            pagerduty=pagerduty,
            slack=slack,
            email=email,
            sms=sms,
            playbook=playbook,
            jira=jira,
            database_url=os.environ.get("GL_DATABASE_URL"),
            redis_url=os.environ.get("GL_REDIS_URL"),
        )

    def get_enabled_alert_sources(self) -> List[AlertSourceConfig]:
        """Get list of enabled alert sources.

        Returns:
            List of enabled alert source configurations.
        """
        sources = []
        if self.prometheus.enabled:
            sources.append(self.prometheus)
        if self.loki.enabled:
            sources.append(self.loki)
        if self.guardduty.enabled:
            sources.append(self.guardduty)
        if self.cloudtrail.enabled:
            sources.append(self.cloudtrail)
        return sources

    def get_escalation_threshold_minutes(self, severity: str) -> int:
        """Get escalation threshold for a severity level.

        Args:
            severity: Severity level (P0, P1, P2, P3).

        Returns:
            Threshold in minutes.
        """
        return self.escalation_thresholds.get(severity, 60)

    def get_response_sla_minutes(self, severity: str) -> int:
        """Get response SLA for a severity level.

        Args:
            severity: Severity level (P0, P1, P2, P3).

        Returns:
            SLA in minutes.
        """
        return self.response_slas.get(severity, 60)

    def get_resolution_sla_minutes(self, severity: str) -> int:
        """Get resolution SLA for a severity level.

        Args:
            severity: Severity level (P0, P1, P2, P3).

        Returns:
            SLA in minutes.
        """
        return self.resolution_slas.get(severity, 1440)


# ---------------------------------------------------------------------------
# Global Configuration Instance
# ---------------------------------------------------------------------------

_global_config: Optional[IncidentResponseConfig] = None


def get_config() -> IncidentResponseConfig:
    """Get or create the global incident response configuration.

    Returns:
        The global IncidentResponseConfig instance.
    """
    global _global_config

    if _global_config is None:
        _global_config = IncidentResponseConfig.from_environment()
        logger.info(
            "Incident response config initialized (environment=%s)",
            _global_config.environment,
        )

    return _global_config


def configure(config: IncidentResponseConfig) -> None:
    """Set the global incident response configuration.

    Args:
        config: Configuration to use globally.
    """
    global _global_config
    _global_config = config
    logger.info("Incident response config updated")


__all__ = [
    # Constants
    "DEFAULT_ESCALATION_THRESHOLDS",
    "DEFAULT_RESPONSE_SLAS",
    "DEFAULT_RESOLUTION_SLAS",
    # Alert Source Configs
    "AlertSourceConfig",
    "PrometheusAlertConfig",
    "LokiAlertConfig",
    "GuardDutyConfig",
    "CloudTrailConfig",
    # Notification Configs
    "PagerDutyConfig",
    "SlackConfig",
    "EmailConfig",
    "SMSConfig",
    # Other Configs
    "PlaybookConfig",
    "JiraConfig",
    # Main Config
    "IncidentResponseConfig",
    # Functions
    "get_config",
    "configure",
]
