# -*- coding: utf-8 -*-
"""
PII Enforcement Engine - SEC-011

Production-grade PII enforcement system for the GreenLang Climate OS platform.
Provides real-time scanning, policy-based enforcement, and audit trails for
regulatory compliance (GDPR, CCPA, HIPAA, PCI-DSS).

This module exports the core enforcement components:

Classes:
    - PIIEnforcementEngine: Core enforcement logic
    - EnforcementAction: Actions (ALLOW, REDACT, BLOCK, QUARANTINE, TRANSFORM)
    - EnforcementPolicy: Policy for a specific PII type
    - EnforcementContext: Context for enforcement decisions
    - PIIEnforcementMiddleware: FastAPI middleware for HTTP enforcement

Models:
    - PIIDetection: Detected PII instance
    - ActionTaken: Record of enforcement action
    - QuarantineItem: Item stored for review
    - EnforcementResult: Complete enforcement result

Enums:
    - PIIType: Supported PII types (SSN, CREDIT_CARD, EMAIL, etc.)
    - ContextType: Context types (api_request, storage, streaming, etc.)
    - TransformationType: Transformation types (tokenize, hash, mask, encrypt)

Example:
    >>> from greenlang.infrastructure.pii_service.enforcement import (
    ...     PIIEnforcementEngine,
    ...     EnforcementContext,
    ...     EnforcementPolicy,
    ...     EnforcementAction,
    ...     PIIType,
    ... )
    >>>
    >>> # Create engine
    >>> engine = PIIEnforcementEngine()
    >>>
    >>> # Define context
    >>> context = EnforcementContext(
    ...     context_type="api_request",
    ...     path="/api/v1/reports",
    ...     method="POST",
    ...     tenant_id="tenant-acme",
    ... )
    >>>
    >>> # Enforce
    >>> result = await engine.enforce(request_body, context)
    >>> if result.blocked:
    ...     return error_response(result.blocked_types)
    >>> return result.modified_content

Middleware Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.pii_service.enforcement import (
    ...     PIIEnforcementMiddleware,
    ...     PIIEnforcementEngine,
    ... )
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(
    ...     PIIEnforcementMiddleware,
    ...     engine=PIIEnforcementEngine(),
    ...     scan_requests=True,
    ...     exclude_paths=["/health", "/metrics"],
    ... )

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import logging

# Import from policies module
from greenlang.infrastructure.pii_service.enforcement.policies import (
    PIIType,
    EnforcementAction,
    TransformationType,
    ContextType,
    EnforcementPolicy,
    EnforcementContext,
    DEFAULT_POLICIES,
    get_default_policy,
)

# Import from actions module
from greenlang.infrastructure.pii_service.enforcement.actions import (
    PIIDetection,
    ActionTaken,
    QuarantineStatus,
    QuarantineItem,
    EnforcementResult,
)

# Import from engine module
from greenlang.infrastructure.pii_service.enforcement.engine import (
    PIIEnforcementEngine,
    EnforcementConfig,
    SimplePatternScanner,
    PIIScannerProtocol,
    AllowlistManagerProtocol,
    NotifierProtocol,
    QuarantineStorageProtocol,
    TokenVaultProtocol,
    get_enforcement_engine,
    reset_engine,
)

# Import from middleware module
from greenlang.infrastructure.pii_service.enforcement.middleware import (
    MiddlewareConfig,
    PIIEnforcementMiddleware,
    PIIEnforcementASGIMiddleware,
    PIIEnforcementDependency,
    create_pii_error_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Enums
    "PIIType",
    "EnforcementAction",
    "TransformationType",
    "ContextType",
    "QuarantineStatus",
    # Policy Models
    "EnforcementPolicy",
    "EnforcementContext",
    "DEFAULT_POLICIES",
    "get_default_policy",
    # Action Models
    "PIIDetection",
    "ActionTaken",
    "QuarantineItem",
    "EnforcementResult",
    # Engine
    "PIIEnforcementEngine",
    "EnforcementConfig",
    "SimplePatternScanner",
    "get_enforcement_engine",
    "reset_engine",
    # Protocols (for type hints and custom implementations)
    "PIIScannerProtocol",
    "AllowlistManagerProtocol",
    "NotifierProtocol",
    "QuarantineStorageProtocol",
    "TokenVaultProtocol",
    # Middleware
    "MiddlewareConfig",
    "PIIEnforcementMiddleware",
    "PIIEnforcementASGIMiddleware",
    "PIIEnforcementDependency",
    "create_pii_error_response",
]
