# -*- coding: utf-8 -*-
"""
QR Code Generator REST API - AGENT-EUDR-014

FastAPI router package providing 37 REST endpoints for EUDR QR code
generation operations including QR code creation, data payload
composition, label rendering, batch code generation, verification URL
construction, anti-counterfeiting checks, bulk generation job
orchestration, and QR code lifecycle management.

Route Modules:
    - qr_routes: QR code generation (generate, data-matrix, detail, image)
    - payload_routes: Payload composition (compose, validate, detail, schemas)
    - label_routes: Label rendering (generate, batch, detail, download, templates)
    - batch_code_routes: Batch codes (generate, reserve, lookup, hierarchy)
    - verification_routes: Verification (build-url, signature, status, offline)
    - counterfeit_routes: Anti-counterfeiting (check, revoke, revocation-list, analytics)
    - bulk_routes: Bulk generation (submit, status, download, cancel, manifest)
    - lifecycle_routes: Lifecycle (activate, deactivate, revoke, scan, history)
    - router: Main router with health endpoint

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-qrg:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 (QR Code Generator)
Agent ID: GL-EUDR-QRG-014
Status: Production Ready
"""

from greenlang.agents.eudr.qr_code_generator.api.router import (
    get_router,
    router,
)

__all__ = [
    "router",
    "get_router",
]
