"""
GreenLang Integration Layer

Consolidated module containing API, SDK, connectors, integrations, services, and adapters.

This module provides:
- REST API interfaces (from api/)
- SDK client libraries (from sdk/)
- External system connectors (from connectors/, integrations/)
- Service layer (from services/)
- Adapter patterns (from adapters/)

Sub-modules:
- integration.api: FastAPI service for emission factor queries
- integration.sdk: SDK client and builder patterns
- integration.connectors: Base connector and context handling
- integration.integrations: CEMS, CMMS, and external system integrations
- integration.services: Service layer components
- integration.adapters: Factor broker and other adapters

Author: GreenLang Team
Version: 2.0.0
"""

__version__ = "2.0.0"

# Re-export commonly used components
# Individual modules can be imported as needed

__all__ = []
