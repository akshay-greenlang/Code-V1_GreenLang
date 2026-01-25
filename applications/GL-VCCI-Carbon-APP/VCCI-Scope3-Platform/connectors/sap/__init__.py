# -*- coding: utf-8 -*-
"""
SAP S/4HANA Connector for GL-VCCI Scope 3 Platform
GL-VCCI Scope 3 Platform

Production-ready SAP S/4HANA OData connector with OAuth 2.0 authentication,
automatic pagination, retry logic, and comprehensive error handling.

Version: 1.0.0
Phase: 4 (Weeks 19-22)
Date: 2025-11-06

Modules Supported:
-----------------
- MM (Materials Management): Procurement transactions
- SD (Sales & Distribution): Logistics, transportation
- FI (Financial Accounting): Capital expenditures

Data Extracted:
--------------
1. Procurement (Category 1):
   - Purchase orders
   - Goods receipts
   - Vendor master data
   - Material master data

2. Logistics (Category 4):
   - Inbound deliveries
   - Transportation orders
   - Freight data

3. Capital Goods (Category 2):
   - Fixed asset acquisitions
   - Capital expenditures

Authentication:
--------------
OAuth 2.0 client credentials flow with token caching

Usage:
------
```python
from connectors.sap import SAPODataClient, SAPConnectorConfig, create_query

# Load configuration from environment
config = SAPConnectorConfig.from_env()

# Create client
client = SAPODataClient(config)

# Query with filters
query = create_query().filter("PostingDate ge '2024-01-01'").top(100)
results = client.query("purchase_orders", query)

# Query with pagination
for batch in client.query_paginated("vendor_master"):
    process_batch(batch)

# Close client
client.close()
```
"""

from .config import (
    SAPConnectorConfig,
    SAPEnvironment,
    SAPModule,
    ODataEndpoint,
    OAuth2Config,
    RetryConfig,
    RateLimitConfig,
    TimeoutConfig,
    get_config,
    reset_config,
)

from .auth import (
    SAPAuthHandler,
    TokenCache,
    get_auth_handler,
    reset_auth_handlers,
)

from .client import (
    SAPODataClient,
    ODataQueryBuilder,
    RateLimiter,
    create_query,
)

from .exceptions import (
    SAPConnectorError,
    SAPConnectionError,
    SAPAuthenticationError,
    SAPRateLimitError,
    SAPDataError,
    SAPTimeoutError,
    SAPConfigurationError,
    get_exception_for_status_code,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "SAPConnectorConfig",
    "SAPEnvironment",
    "SAPModule",
    "ODataEndpoint",
    "OAuth2Config",
    "RetryConfig",
    "RateLimitConfig",
    "TimeoutConfig",
    "get_config",
    "reset_config",
    # Auth
    "SAPAuthHandler",
    "TokenCache",
    "get_auth_handler",
    "reset_auth_handlers",
    # Client
    "SAPODataClient",
    "ODataQueryBuilder",
    "RateLimiter",
    "create_query",
    # Exceptions
    "SAPConnectorError",
    "SAPConnectionError",
    "SAPAuthenticationError",
    "SAPRateLimitError",
    "SAPDataError",
    "SAPTimeoutError",
    "SAPConfigurationError",
    "get_exception_for_status_code",
]
