# -*- coding: utf-8 -*-
"""
Compliance Automation API - SEC-010 Phase 5

FastAPI router for the GreenLang multi-compliance automation system.
Provides REST API endpoints for compliance assessment, DSAR processing,
consent management, and framework-specific operations.

API Endpoints:
- /api/v1/secops/compliance/status - Overall compliance dashboard
- /api/v1/secops/compliance/iso27001 - ISO 27001 status
- /api/v1/secops/compliance/gdpr - GDPR status
- /api/v1/secops/compliance/pci-dss - PCI-DSS status
- /api/v1/secops/dsar - DSAR management
- /api/v1/secops/consent - Consent management

Public API:
    - compliance_router: FastAPI router for compliance endpoints.

Example:
    >>> from greenlang.infrastructure.compliance_automation.api import (
    ...     compliance_router,
    ... )
    >>> app.include_router(compliance_router, prefix="/api/v1/secops")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from greenlang.infrastructure.compliance_automation.api.compliance_routes import (
    router as compliance_router,
)

__all__ = [
    "compliance_router",
]
