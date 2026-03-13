# -*- coding: utf-8 -*-
"""
AGENT-EUDR-023: Legal Compliance Verifier - API Package

FastAPI router package providing 37+ REST endpoints for EUDR legal compliance
verification operations including legal framework management, document
verification, certification validation, red flag detection, compliance
assessment, audit integration, reporting, and batch processing.

Route Modules:
    - framework_routes: Legal framework CRUD + search (register, list, detail, update, search)
    - document_routes: Document verification (verify, list, detail, validity-check, expiring)
    - certification_routes: Certification validation (validate, list, detail, eudr-equivalence)
    - red_flag_routes: Red flag detection (detect, list, detail, suppress)
    - compliance_routes: Compliance assessment (assess, check-category, list, detail, history)
    - audit_routes: Audit integration (ingest, list, findings, corrective-actions)
    - report_routes: Reporting (generate, list, download, schedule)
    - batch_routes: Batch processing (batch-assess, batch-verify, batch-status)
    - router: Main router registration with /v1/eudr-lcv prefix

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-lcv:* permissions (20 permissions)
    - Rate limiting via middleware decorator (100/30/10/5 req/min tiers)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from greenlang.agents.eudr.legal_compliance_verifier.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
