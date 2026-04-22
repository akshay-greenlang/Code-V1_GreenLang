# -*- coding: utf-8 -*-
"""Security hardening utilities for the GreenLang Factors API.

Provides input validation/sanitization, API key management,
and security audit helpers.
"""

from greenlang.factors.security.input_validation import (
    sanitize_search_query,
    validate_factor_id,
    validate_edition_id,
    sanitize_geography,
)
from greenlang.factors.security.api_key_manager import (
    generate_api_key,
    hash_api_key,
    validate_api_key_format,
    rotate_api_key,
)
from greenlang.factors.security.audit import (
    SecurityFinding,
    check_headers,
    check_cors_config,
    check_auth_config,
)
from greenlang.factors.security.tenant_vault_transit import (
    TenantKeyAccessError,
    TenantVaultTransit,
    TenantVaultTransitError,
    TransitAuditEntry,
    VaultUnavailableInProductionError,
    default_transit,
    reset_default_transit,
)

__all__ = [
    "sanitize_search_query",
    "validate_factor_id",
    "validate_edition_id",
    "sanitize_geography",
    "generate_api_key",
    "hash_api_key",
    "validate_api_key_format",
    "rotate_api_key",
    "SecurityFinding",
    "check_headers",
    "check_cors_config",
    "check_auth_config",
    "TenantVaultTransit",
    "TenantVaultTransitError",
    "TenantKeyAccessError",
    "VaultUnavailableInProductionError",
    "TransitAuditEntry",
    "default_transit",
    "reset_default_transit",
]
