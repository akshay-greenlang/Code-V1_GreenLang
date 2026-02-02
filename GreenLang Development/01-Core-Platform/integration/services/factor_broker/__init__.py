# -*- coding: utf-8 -*-
"""
Factor Broker Service
GL-VCCI Scope 3 Platform

Runtime emission factor resolution with multi-source cascading,
license compliance, and caching.

Version: 1.0.0
Compliance: ecoinvent license terms, no bulk redistribution
"""

from .broker import FactorBroker
from .sources import (
    DESNZSource,
    EPASource,
    EcoinventSource,
    ProxySource
)
from .models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    DataQualityIndicator
)
from .cache import FactorCache
from .exceptions import (
    FactorBrokerError,
    FactorNotFoundError,
    LicenseViolationError,
    RateLimitExceededError
)

__version__ = "1.0.0"
__all__ = [
    "FactorBroker",
    "DESNZSource",
    "EPASource",
    "EcoinventSource",
    "ProxySource",
    "FactorRequest",
    "FactorResponse",
    "FactorMetadata",
    "DataQualityIndicator",
    "FactorCache",
    "FactorBrokerError",
    "FactorNotFoundError",
    "LicenseViolationError",
    "RateLimitExceededError",
]
