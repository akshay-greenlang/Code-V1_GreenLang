# -*- coding: utf-8 -*-
"""
Document Authentication REST API - AGENT-EUDR-012

FastAPI router package providing 37 REST endpoints for EUDR document
authentication operations including document classification, digital
signature verification, hash integrity validation, certificate chain
validation, metadata extraction, fraud pattern detection, cross-reference
verification against external registries, and compliance reporting with
evidence packages.

Route Modules:
    - classify_routes: Document classification, batch classify, templates
    - signature_routes: Signature verification, batch verify, history
    - hash_routes: Hash computation, verification, registry, Merkle tree
    - certificate_routes: Certificate chain validation, trusted CAs
    - metadata_routes: Metadata extraction, retrieval, validation
    - fraud_routes: Fraud detection, batch detect, alerts, rules
    - crossref_routes: Cross-reference verification, batch, cache stats
    - report_routes: Authentication reports, evidence packages, dashboard
    - router: Main router with batch job and health endpoints

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-dav:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012 (Document Authentication)
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from greenlang.agents.eudr.document_authentication.api.router import (
    router,
    get_router,
)

__all__ = [
    "router",
    "get_router",
]
