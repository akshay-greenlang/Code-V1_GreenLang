"""
Workday RaaS Connector Configuration Management
GL-VCCI Scope 3 Platform

Configuration management for Workday connector including environment variables,
OAuth settings, RaaS report endpoints, and service settings.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator


class WorkdayEnvironment(str, Enum):
    """Workday environment types."""
    SANDBOX = "sandbox"
    IMPLEMENTATION = "implementation"
    PRODUCTION = "production"


class RaaSReport(BaseModel):
    """
    Configuration for a single RaaS report.

    Attributes:
        name: Report name in Workday
        owner: Report owner (typically a user or integration system)
        category: Report category (HCM, Finance, etc.)
        description: Human-readable description
        enabled: Whether report extraction is enabled
        format: Response format (json or xml)
        batch_size: Default batch size for pagination
    """
    name: str
    owner: str
    category: str
    description: str = ""
    enabled: bool = True
    format: str = Field(default="json", pattern="^(json|xml)$")
    batch_size: int = Field(default=1000, ge=1, le=10000)


class RetryConfig(BaseModel):
    """
    Configuration for retry logic with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        backoff_multiplier: Multiplier for exponential backoff
        retry_on_status_codes: HTTP status codes to retry on
    """
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    max_delay: float = Field(default=8.0, ge=1.0, le=300.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_on_status_codes: List[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504]
    )

    @validator('max_delay')
    def max_delay_must_be_greater_than_base(cls, v, values):
        """Validate max_delay is greater than base_delay."""
        if 'base_delay' in values and v < values['base_delay']:
            raise ValueError('max_delay must be greater than base_delay')
        return v


class RateLimitConfig(BaseModel):
    """
    Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute
        enabled: Whether rate limiting is enabled
        burst_size: Maximum burst size for token bucket algorithm
    """
    requests_per_minute: int = Field(default=10, ge=1, le=1000)
    enabled: bool = True
    burst_size: int = Field(default=5, ge=1, le=100)


class OAuth2Config(BaseModel):
    """
    Configuration for OAuth 2.0 authentication.

    Attributes:
        client_id: OAuth client ID (Integration System User)
        client_secret: OAuth client secret
        token_url: OAuth token endpoint URL
        refresh_token: Refresh token (for OAuth refresh flow)
        grant_type: OAuth grant type
        token_cache_ttl: Token cache TTL in seconds
    """
    client_id: str
    client_secret: str
    token_url: str
    refresh_token: Optional[str] = None
    grant_type: str = "client_credentials"
    token_cache_ttl: int = Field(default=3300, ge=60, le=86400)  # 55 minutes

    class Config:
        # Don't expose secrets in string representation
        json_encoders = {
            str: lambda v: '***' if 'secret' in str(v).lower() else v
        }


class TimeoutConfig(BaseModel):
    """
    Configuration for request timeouts.

    Attributes:
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        total_timeout: Total timeout in seconds
    """
    connect_timeout: float = Field(default=10.0, ge=1.0, le=60.0)
    read_timeout: float = Field(default=60.0, ge=1.0, le=600.0)
    total_timeout: float = Field(default=120.0, ge=1.0, le=1200.0)


class WorkdayConnectorConfig(BaseModel):
    """
    Main configuration class for Workday RaaS connector.

    This class consolidates all configuration settings and provides
    methods to load from environment variables.
    """
    # Environment
    environment: WorkdayEnvironment = WorkdayEnvironment.SANDBOX

    # Workday tenant URL (e.g., https://impl.workday.com/tenant_name)
    tenant_url: str

    # Tenant name
    tenant_name: str

    # OAuth configuration
    oauth: OAuth2Config

    # Rate limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Timeout configuration
    timeout: TimeoutConfig = Field(default_factory=TimeoutConfig)

    # RaaS reports
    reports: Dict[str, RaaSReport] = Field(default_factory=dict)

    # Default batch size
    default_batch_size: int = Field(default=1000, ge=1, le=10000)

    # Connection pool settings
    pool_connections: int = Field(default=10, ge=1, le=100)
    pool_maxsize: int = Field(default=20, ge=1, le=200)

    # Enable detailed logging
    debug_mode: bool = False

    class Config:
        use_enum_values = True

    def __init__(self, **data):
        """Initialize and set up default reports."""
        super().__init__(**data)
        if not self.reports:
            self._init_default_reports()

    def _init_default_reports(self):
        """Initialize default RaaS report configurations."""
        self.reports = {
            "expense_reports": RaaSReport(
                name="Expense_Report_for_Carbon",
                owner="Integration_System_User",
                category="HCM",
                description="Expense reports for Category 6: Business Travel emissions",
                enabled=True,
                format="json",
                batch_size=self.default_batch_size
            ),
            "commute_surveys": RaaSReport(
                name="Commute_Survey_Results",
                owner="Integration_System_User",
                category="HCM",
                description="Employee commute surveys for Category 7: Employee Commuting emissions",
                enabled=True,
                format="json",
                batch_size=self.default_batch_size
            ),
        }

    @classmethod
    def from_env(cls, environment: Optional[str] = None) -> "WorkdayConnectorConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            WORKDAY_ENVIRONMENT: Environment (sandbox, implementation, production)
            WORKDAY_TENANT_URL: Workday tenant URL
            WORKDAY_TENANT_NAME: Workday tenant name
            WORKDAY_CLIENT_ID: OAuth client ID
            WORKDAY_CLIENT_SECRET: OAuth client secret
            WORKDAY_TOKEN_URL: OAuth token URL
            WORKDAY_REFRESH_TOKEN: OAuth refresh token (optional)
            WORKDAY_RATE_LIMIT_RPM: Rate limit (requests per minute)
            WORKDAY_BATCH_SIZE: Default batch size for pagination
            WORKDAY_DEBUG_MODE: Enable debug logging

        Args:
            environment: Override environment from env var

        Returns:
            WorkdayConnectorConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        # Determine environment
        env = environment or os.getenv("WORKDAY_ENVIRONMENT", "sandbox")
        workday_environment = WorkdayEnvironment(env)

        # Load tenant URL and name (required)
        tenant_url = os.getenv("WORKDAY_TENANT_URL")
        tenant_name = os.getenv("WORKDAY_TENANT_NAME")

        if not tenant_url or not tenant_name:
            raise ValueError("WORKDAY_TENANT_URL and WORKDAY_TENANT_NAME environment variables are required")

        # Load OAuth config (required)
        client_id = os.getenv("WORKDAY_CLIENT_ID")
        client_secret = os.getenv("WORKDAY_CLIENT_SECRET")
        token_url = os.getenv("WORKDAY_TOKEN_URL")

        if not all([client_id, client_secret, token_url]):
            raise ValueError(
                "WORKDAY_CLIENT_ID, WORKDAY_CLIENT_SECRET, and WORKDAY_TOKEN_URL "
                "environment variables are required"
            )

        oauth_config = OAuth2Config(
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
            refresh_token=os.getenv("WORKDAY_REFRESH_TOKEN")
        )

        # Rate limit configuration
        rate_limit_config = RateLimitConfig(
            requests_per_minute=int(os.getenv("WORKDAY_RATE_LIMIT_RPM", "10")),
            enabled=os.getenv("WORKDAY_RATE_LIMIT_ENABLED", "true").lower() == "true"
        )

        # Retry configuration
        retry_config = RetryConfig(
            max_retries=int(os.getenv("WORKDAY_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("WORKDAY_RETRY_BASE_DELAY", "1.0")),
            max_delay=float(os.getenv("WORKDAY_RETRY_MAX_DELAY", "8.0"))
        )

        # Timeout configuration
        timeout_config = TimeoutConfig(
            connect_timeout=float(os.getenv("WORKDAY_CONNECT_TIMEOUT", "10.0")),
            read_timeout=float(os.getenv("WORKDAY_READ_TIMEOUT", "60.0")),
            total_timeout=float(os.getenv("WORKDAY_TOTAL_TIMEOUT", "120.0"))
        )

        # Default batch size
        default_batch_size = int(os.getenv("WORKDAY_BATCH_SIZE", "1000"))

        # Debug mode
        debug_mode = os.getenv("WORKDAY_DEBUG_MODE", "false").lower() == "true"

        return cls(
            environment=workday_environment,
            tenant_url=tenant_url,
            tenant_name=tenant_name,
            oauth=oauth_config,
            rate_limit=rate_limit_config,
            retry=retry_config,
            timeout=timeout_config,
            default_batch_size=default_batch_size,
            debug_mode=debug_mode
        )

    def get_report_config(self, report_name: str) -> Optional[RaaSReport]:
        """
        Get configuration for a specific report.

        Args:
            report_name: Name of the report

        Returns:
            RaaSReport if found, None otherwise
        """
        return self.reports.get(report_name)

    def is_report_enabled(self, report_name: str) -> bool:
        """
        Check if a report is enabled.

        Args:
            report_name: Name of the report

        Returns:
            True if report is enabled, False otherwise
        """
        report = self.get_report_config(report_name)
        return report.enabled if report else False

    def get_raas_url(self, report_name: str) -> Optional[str]:
        """
        Get full RaaS URL for a report.

        Args:
            report_name: Name of the report

        Returns:
            Full RaaS URL if report exists, None otherwise

        Example:
            https://impl.workday.com/tenant_name/ccx/service/tenant_name/RaaS/owner/report_name
        """
        report = self.get_report_config(report_name)
        if not report:
            return None

        # Remove trailing slash from tenant_url
        base = self.tenant_url.rstrip('/')

        # Build RaaS URL: /ccx/service/{tenant}/RaaS/{owner}/{report_name}
        return f"{base}/ccx/service/{self.tenant_name}/RaaS/{report.owner}/{report.name}"

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate tenant URL
        if not self.tenant_url:
            errors.append("tenant_url is required")
        elif not (self.tenant_url.startswith('http://') or self.tenant_url.startswith('https://')):
            errors.append("tenant_url must start with http:// or https://")

        # Validate tenant name
        if not self.tenant_name:
            errors.append("tenant_name is required")

        # Validate OAuth config
        if not self.oauth.client_id:
            errors.append("OAuth client_id is required")
        if not self.oauth.client_secret:
            errors.append("OAuth client_secret is required")
        if not self.oauth.token_url:
            errors.append("OAuth token_url is required")

        # Validate at least one report is enabled
        enabled_reports = [r for r in self.reports.values() if r.enabled]
        if not enabled_reports:
            errors.append("At least one report must be enabled")

        return errors


# Global configuration instance
_config: Optional[WorkdayConnectorConfig] = None


def get_config() -> WorkdayConnectorConfig:
    """
    Get global configuration instance.

    Loads configuration from environment on first call,
    then returns cached instance.

    Returns:
        WorkdayConnectorConfig instance
    """
    global _config

    if _config is None:
        _config = WorkdayConnectorConfig.from_env()

    return _config


def reset_config():
    """
    Reset global configuration instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _config
    _config = None
