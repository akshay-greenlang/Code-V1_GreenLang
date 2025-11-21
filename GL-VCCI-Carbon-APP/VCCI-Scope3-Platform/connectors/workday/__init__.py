# -*- coding: utf-8 -*-
"""
Workday RaaS (Report as a Service) Connector
GL-VCCI Scope 3 Platform

Complete Workday connector for extracting HCM data (expense reports and commute surveys)
for Scope 3 Categories 6 (Business Travel) and 7 (Employee Commuting).

Architecture:
- OAuth 2.0 authentication with token caching
- RaaS API client with pagination and retry logic
- HCM data extractors with delta sync support
- Mappers to VCCI schemas (logistics_v1.0.json for travel, custom for commute)
- Celery jobs for scheduled synchronization
- Redis-based deduplication

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

from .config import (
    WorkdayConnectorConfig,
    WorkdayEnvironment,
    RaaSReport,
    OAuth2Config,
    get_config,
    reset_config
)
from .auth import (
    WorkdayAuthHandler,
    TokenCache,
    get_auth_handler,
    reset_auth_handlers
)
from .client import WorkdayRaaSClient
from .exceptions import (
    WorkdayConnectorError,
    WorkdayConnectionError,
    WorkdayAuthenticationError,
    WorkdayRateLimitError,
    WorkdayDataError,
    WorkdayTimeoutError,
    WorkdayConfigurationError,
)
from .extractors import (
    BaseExtractor,
    HCMExtractor,
    ExpenseReportData,
    CommuteData,
)
from .mappers import (
    ExpenseMapper,
    CommuteMapper,
)
from .jobs import (
    sync_expense_reports,
    sync_commute_surveys,
)

__version__ = "1.0.0"
__author__ = "GL-VCCI Team"

__all__ = [
    # Config
    "WorkdayConnectorConfig",
    "WorkdayEnvironment",
    "RaaSReport",
    "OAuth2Config",
    "get_config",
    "reset_config",
    # Auth
    "WorkdayAuthHandler",
    "TokenCache",
    "get_auth_handler",
    "reset_auth_handlers",
    # Client
    "WorkdayRaaSClient",
    # Exceptions
    "WorkdayConnectorError",
    "WorkdayConnectionError",
    "WorkdayAuthenticationError",
    "WorkdayRateLimitError",
    "WorkdayDataError",
    "WorkdayTimeoutError",
    "WorkdayConfigurationError",
    # Extractors
    "BaseExtractor",
    "HCMExtractor",
    "ExpenseReportData",
    "CommuteData",
    # Mappers
    "ExpenseMapper",
    "CommuteMapper",
    # Jobs
    "sync_expense_reports",
    "sync_commute_surveys",
]


def create_workday_connector(
    tenant_url: str,
    tenant_name: str,
    client_id: str,
    client_secret: str,
    token_url: str,
    environment: str = "sandbox"
) -> WorkdayRaaSClient:
    """
    Factory function to create a configured Workday connector.

    Args:
        tenant_url: Workday tenant URL
        tenant_name: Workday tenant name
        client_id: OAuth client ID
        client_secret: OAuth client secret
        token_url: OAuth token URL
        environment: Environment (sandbox, implementation, production)

    Returns:
        Configured WorkdayRaaSClient instance

    Example:
        >>> client = create_workday_connector(
        ...     tenant_url="https://impl.workday.com/acme",
        ...     tenant_name="acme",
        ...     client_id="ISU_client_123",
        ...     client_secret="secret",
        ...     token_url="https://impl.workday.com/acme/oauth2/token",
        ...     environment="sandbox"
        ... )
        >>> expenses = client.get_report("expense_reports")
    """
    oauth_config = OAuth2Config(
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url
    )

    config = WorkdayConnectorConfig(
        environment=WorkdayEnvironment(environment),
        tenant_url=tenant_url,
        tenant_name=tenant_name,
        oauth=oauth_config
    )

    return WorkdayRaaSClient(config)
