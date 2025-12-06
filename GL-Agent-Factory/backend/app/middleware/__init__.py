"""
API Middleware

This package contains middleware for:
- JWT Authentication
- Rate Limiting
- Request Validation
- Tenant Context (Multi-tenancy)
- Security Headers (SOC2/ISO27001 compliance)
"""

from app.middleware.auth import JWTAuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_validation import RequestValidationMiddleware
from app.middleware.tenant_context import (
    TenantContextMiddleware,
    TenantContext,
    get_tenant_context,
    set_tenant_context,
    clear_tenant_context,
    require_feature,
    require_quota,
    require_role,
)
from app.middleware.security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    ContentSecurityPolicy,
    CORSPolicy,
    RateLimitConfig,
    RequestSigningConfig,
    create_api_security_config,
    create_strict_security_config,
)

__all__ = [
    "JWTAuthMiddleware",
    "RateLimitMiddleware",
    "RequestValidationMiddleware",
    "TenantContextMiddleware",
    "TenantContext",
    "get_tenant_context",
    "set_tenant_context",
    "clear_tenant_context",
    "require_feature",
    "require_quota",
    "require_role",
    # Security Headers
    "SecurityHeadersMiddleware",
    "SecurityHeadersConfig",
    "ContentSecurityPolicy",
    "CORSPolicy",
    "RateLimitConfig",
    "RequestSigningConfig",
    "create_api_security_config",
    "create_strict_security_config",
]
