# SAP Connector Utilities
# Supporting infrastructure for SAP integration

"""
SAP Utilities
=============

Supporting utilities for SAP S/4HANA integration.

Modules:
--------
- retry_logic: Exponential backoff retry decorator
- rate_limiter: Token bucket rate limiting
- audit_logger: Comprehensive audit logging
- deduplication: Transaction ID deduplication

Usage:
------
```python
from connectors.sap.utils import (
    retry_with_backoff,
    RateLimiter,
    AuditLogger,
    DeduplicationCache
)

# Retry logic
@retry_with_backoff(max_retries=4)
def call_api():
    # Will retry with exponential backoff
    pass

# Rate limiting
limiter = RateLimiter(rate=10, per=60)
if limiter.acquire("/api/purchaseorders"):
    # Make API call
    pass

# Audit logging
logger = AuditLogger()
logger.log_api_call("/api/purchaseorders", "GET", 200, duration=0.5)

# Deduplication
cache = DeduplicationCache()
if not cache.is_duplicate("PO-12345"):
    # Process transaction
    cache.mark_processed("PO-12345")
```
"""

__version__ = "1.0.0"

__all__ = [
    "retry_with_backoff",
    "RateLimiter",
    "AuditLogger",
    "DeduplicationCache",
]
