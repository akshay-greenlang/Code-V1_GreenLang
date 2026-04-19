# -*- coding: utf-8 -*-
# Oracle Fusion Cloud Connector
# REST API integration for procurement and supply chain data extraction

"""
Oracle Fusion Cloud Connector
==============================

Core infrastructure for Oracle Fusion Cloud REST API integration.

Modules Supported:
-----------------
- Procurement Cloud: Purchase orders, suppliers, requisitions
- Supply Chain Management: Shipments, transportation orders
- Financials Cloud: Fixed assets

Core Components:
---------------
1. Configuration Management (config.py):
   - Environment-based configuration
   - OAuth 2.0 settings
   - REST endpoint definitions
   - Rate limiting and retry settings

2. Authentication Handler (auth.py):
   - OAuth 2.0 client credentials flow
   - Token caching and refresh
   - Multi-environment support

3. REST Client (client.py):
   - GET/POST/PATCH operations
   - Automatic pagination handling
   - Query builder for Oracle REST API
   - Error handling and retry logic

4. Custom Exceptions (exceptions.py):
   - OracleConnectionError
   - OracleAuthenticationError
   - OracleRateLimitError
   - OracleDataError
   - OracleTimeoutError

Authentication:
--------------
OAuth 2.0 client credentials flow with token caching

Usage:
------
```python
from connectors.oracle import (
    OracleConnectorConfig,
    OracleRESTClient,
    create_query,
    get_config
)

# Load configuration from environment
config = OracleConnectorConfig.from_env()

# Create REST client
client = OracleRESTClient(config)

# Build query
query = create_query().q("LastUpdateDate >= '2024-01-01T00:00:00'").limit(1000)

# Query endpoint with pagination
for batch in client.query_paginated("purchase_orders", query.build()):
    for item in batch:
        print(item)

# Close client
client.close()
```
"""

__version__ = "1.0.0"

# Configuration
from .config import (
    OracleConnectorConfig,
    OracleEnvironment,
    OracleModule,
    RESTEndpoint,
    OAuth2Config,
    RetryConfig,
    RateLimitConfig,
    TimeoutConfig,
    get_config,
    reset_config,
)

# Authentication
from .auth import (
    OracleAuthHandler,
    TokenCache,
    get_auth_handler,
    reset_auth_handlers,
)

# REST Client
from .client import (
    OracleRESTClient,
    RESTQueryBuilder,
    RateLimiter,
    create_query,
)

# Exceptions
from .exceptions import (
    OracleConnectorError,
    OracleConnectionError,
    OracleAuthenticationError,
    OracleRateLimitError,
    OracleDataError,
    OracleTimeoutError,
    OracleConfigurationError,
    get_exception_for_status_code,
)

__all__ = [
    # Configuration
    "OracleConnectorConfig",
    "OracleEnvironment",
    "OracleModule",
    "RESTEndpoint",
    "OAuth2Config",
    "RetryConfig",
    "RateLimitConfig",
    "TimeoutConfig",
    "get_config",
    "reset_config",
    # Authentication
    "OracleAuthHandler",
    "TokenCache",
    "get_auth_handler",
    "reset_auth_handlers",
    # REST Client
    "OracleRESTClient",
    "RESTQueryBuilder",
    "RateLimiter",
    "create_query",
    # Exceptions
    "OracleConnectorError",
    "OracleConnectionError",
    "OracleAuthenticationError",
    "OracleRateLimitError",
    "OracleDataError",
    "OracleTimeoutError",
    "OracleConfigurationError",
    "get_exception_for_status_code",
]
