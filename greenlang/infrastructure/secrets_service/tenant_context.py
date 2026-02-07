# -*- coding: utf-8 -*-
"""
Tenant Context for Secrets Service - SEC-006

Provides tenant isolation for secrets access. Uses ContextVar to track
the current tenant context and validates cross-tenant access attempts.

Example:
    >>> from greenlang.infrastructure.secrets_service.tenant_context import (
    ...     TenantSecretContext,
    ...     get_current_tenant,
    ...     set_current_tenant,
    ... )
    >>> set_current_tenant("acme-corp")
    >>> ctx = TenantSecretContext()
    >>> path = ctx.build_path("database/main")  # -> tenants/acme-corp/database/main

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context Variable for Current Tenant
# ---------------------------------------------------------------------------

_current_tenant: ContextVar[Optional[str]] = ContextVar(
    "current_tenant", default=None
)


def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID from the context.

    Returns:
        Current tenant ID or None if not set.
    """
    return _current_tenant.get()


def set_current_tenant(tenant_id: Optional[str]) -> None:
    """Set the current tenant ID in the context.

    Args:
        tenant_id: Tenant ID to set (None to clear).
    """
    _current_tenant.set(tenant_id)


def clear_tenant_context() -> None:
    """Clear the current tenant context."""
    _current_tenant.set(None)


# ---------------------------------------------------------------------------
# Tenant Secret Context
# ---------------------------------------------------------------------------


class TenantAccessDeniedError(Exception):
    """Raised when a tenant attempts to access another tenant's secrets."""

    def __init__(
        self,
        requesting_tenant: Optional[str],
        target_tenant: Optional[str],
        path: str,
    ):
        self.requesting_tenant = requesting_tenant
        self.target_tenant = target_tenant
        self.path = path
        super().__init__(
            f"Tenant '{requesting_tenant}' cannot access secrets for "
            f"tenant '{target_tenant}' at path '{path}'"
        )


class InvalidSecretPathError(Exception):
    """Raised when a secret path is invalid or contains forbidden characters."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Invalid secret path '{path}': {reason}")


@dataclass
class TenantSecretContext:
    """Context for tenant-isolated secret access.

    Handles path building, access validation, and tenant isolation
    for the Secrets Service.

    Attributes:
        tenant_path_prefix: Path prefix for tenant-scoped secrets.
        platform_path_prefix: Path prefix for platform-wide secrets.
        platform_access_roles: Roles that can access platform secrets.
        cross_tenant_access_roles: Roles that can access any tenant's secrets.
    """

    tenant_path_prefix: str = "secret/data/tenants"
    platform_path_prefix: str = "secret/data/greenlang"

    # Roles with special access (injected from auth context)
    platform_access_roles: Set[str] = field(
        default_factory=lambda: {"platform_admin", "system"}
    )
    cross_tenant_access_roles: Set[str] = field(
        default_factory=lambda: {"platform_admin", "system", "super_admin"}
    )

    # Path validation pattern
    _valid_path_pattern: str = r"^[a-zA-Z0-9][a-zA-Z0-9/_-]*$"
    _max_path_length: int = 512

    def build_path(
        self,
        base_path: str,
        tenant_id: Optional[str] = None,
        is_platform: bool = False,
    ) -> str:
        """Build a full Vault path for a secret.

        If tenant_id is provided, builds a tenant-scoped path.
        If is_platform is True, builds a platform-scoped path.
        Otherwise, uses the current tenant context.

        Args:
            base_path: Relative path within the tenant/platform namespace.
            tenant_id: Explicit tenant ID (overrides context).
            is_platform: Whether this is a platform-level secret.

        Returns:
            Full Vault path.

        Raises:
            InvalidSecretPathError: If path is invalid.
            ValueError: If no tenant context and not a platform secret.
        """
        # Validate base path
        self._validate_path(base_path)

        # Normalize path (remove leading/trailing slashes)
        base_path = base_path.strip("/")

        if is_platform:
            return f"{self.platform_path_prefix}/{base_path}"

        # Determine tenant
        effective_tenant = tenant_id or get_current_tenant()

        if effective_tenant is None:
            raise ValueError(
                "No tenant context available. Either set_current_tenant() "
                "or pass tenant_id explicitly."
            )

        # Validate tenant ID format
        self._validate_tenant_id(effective_tenant)

        return f"{self.tenant_path_prefix}/{effective_tenant}/{base_path}"

    def build_tenant_path(self, tenant_id: str, base_path: str) -> str:
        """Build a tenant-scoped path explicitly.

        Args:
            tenant_id: Tenant identifier.
            base_path: Relative path within tenant namespace.

        Returns:
            Full tenant-scoped Vault path.
        """
        return self.build_path(base_path, tenant_id=tenant_id, is_platform=False)

    def build_platform_path(self, base_path: str) -> str:
        """Build a platform-scoped path.

        Args:
            base_path: Relative path within platform namespace.

        Returns:
            Full platform-scoped Vault path.
        """
        return self.build_path(base_path, is_platform=True)

    def validate_access(
        self,
        tenant_id: Optional[str],
        path: str,
        user_roles: Optional[Set[str]] = None,
    ) -> bool:
        """Validate that access to a secret path is allowed.

        Checks:
        1. Cross-tenant access requires special roles.
        2. Platform secret access requires platform roles.
        3. Tenant-scoped access is allowed for same tenant.

        Args:
            tenant_id: Tenant ID of the requester.
            path: Full secret path being accessed.
            user_roles: Set of roles assigned to the requester.

        Returns:
            True if access is allowed.

        Raises:
            TenantAccessDeniedError: If cross-tenant access is denied.
        """
        user_roles = user_roles or set()

        # Super admins can access anything
        if user_roles & self.cross_tenant_access_roles:
            logger.debug(
                "Access granted: user has cross-tenant role",
                extra={
                    "event_category": "secrets",
                    "tenant_id": tenant_id,
                    "path": path,
                },
            )
            return True

        # Check if this is a platform secret
        if self.is_platform_secret(path):
            if not (user_roles & self.platform_access_roles):
                logger.warning(
                    "Access denied: platform secret requires platform role",
                    extra={
                        "event_category": "secrets",
                        "tenant_id": tenant_id,
                        "path": path,
                    },
                )
                raise TenantAccessDeniedError(tenant_id, "platform", path)
            return True

        # Extract tenant from path
        path_tenant = self.extract_tenant_from_path(path)

        if path_tenant is None:
            # Path is not tenant-scoped, treat as platform
            if not (user_roles & self.platform_access_roles):
                raise TenantAccessDeniedError(tenant_id, "platform", path)
            return True

        # Check tenant match
        if tenant_id != path_tenant:
            logger.warning(
                "Access denied: cross-tenant access attempt",
                extra={
                    "event_category": "secrets",
                    "requesting_tenant": tenant_id,
                    "target_tenant": path_tenant,
                    "path": path,
                },
            )
            raise TenantAccessDeniedError(tenant_id, path_tenant, path)

        return True

    def is_platform_secret(self, path: str) -> bool:
        """Check if a path refers to a platform-level secret.

        Args:
            path: Full secret path.

        Returns:
            True if this is a platform secret.
        """
        normalized_path = path.strip("/")
        platform_prefix = self.platform_path_prefix.strip("/")
        return normalized_path.startswith(platform_prefix)

    def is_tenant_secret(self, path: str) -> bool:
        """Check if a path refers to a tenant-scoped secret.

        Args:
            path: Full secret path.

        Returns:
            True if this is a tenant-scoped secret.
        """
        normalized_path = path.strip("/")
        tenant_prefix = self.tenant_path_prefix.strip("/")
        return normalized_path.startswith(tenant_prefix)

    def extract_tenant_from_path(self, path: str) -> Optional[str]:
        """Extract the tenant ID from a tenant-scoped path.

        Args:
            path: Full secret path.

        Returns:
            Tenant ID if path is tenant-scoped, None otherwise.
        """
        if not self.is_tenant_secret(path):
            return None

        # Remove prefix and extract tenant segment
        normalized_path = path.strip("/")
        tenant_prefix = self.tenant_path_prefix.strip("/")

        remaining = normalized_path[len(tenant_prefix) :].strip("/")
        if not remaining:
            return None

        # First segment after prefix is tenant ID
        parts = remaining.split("/")
        return parts[0] if parts else None

    def _validate_path(self, path: str) -> None:
        """Validate a secret path for correctness.

        Args:
            path: Path to validate.

        Raises:
            InvalidSecretPathError: If path is invalid.
        """
        if not path:
            raise InvalidSecretPathError(path, "Path cannot be empty")

        if len(path) > self._max_path_length:
            raise InvalidSecretPathError(
                path, f"Path exceeds maximum length of {self._max_path_length}"
            )

        # Check for path traversal
        if ".." in path:
            raise InvalidSecretPathError(path, "Path traversal (..) not allowed")

        # Check pattern (after stripping slashes for validation)
        normalized = path.strip("/")
        if not re.match(self._valid_path_pattern, normalized):
            raise InvalidSecretPathError(
                path,
                "Path must start with alphanumeric and contain only "
                "alphanumeric, underscore, hyphen, or slash",
            )

    def _validate_tenant_id(self, tenant_id: str) -> None:
        """Validate a tenant ID format.

        Args:
            tenant_id: Tenant ID to validate.

        Raises:
            InvalidSecretPathError: If tenant ID is invalid.
        """
        if not tenant_id:
            raise InvalidSecretPathError(tenant_id, "Tenant ID cannot be empty")

        if len(tenant_id) > 128:
            raise InvalidSecretPathError(
                tenant_id, "Tenant ID exceeds maximum length of 128"
            )

        # Tenant IDs should be simple identifiers
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$", tenant_id):
            raise InvalidSecretPathError(
                tenant_id,
                "Tenant ID must start with alphanumeric and contain only "
                "alphanumeric, underscore, or hyphen",
            )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def build_tenant_secret_path(
    base_path: str,
    tenant_id: Optional[str] = None,
    tenant_prefix: str = "secret/data/tenants",
) -> str:
    """Build a tenant-scoped secret path.

    Convenience function that creates a TenantSecretContext and builds the path.

    Args:
        base_path: Relative path within tenant namespace.
        tenant_id: Tenant ID (uses context if not provided).
        tenant_prefix: Override for tenant path prefix.

    Returns:
        Full tenant-scoped Vault path.
    """
    ctx = TenantSecretContext(tenant_path_prefix=tenant_prefix)
    return ctx.build_path(base_path, tenant_id=tenant_id)


def build_platform_secret_path(
    base_path: str,
    platform_prefix: str = "secret/data/greenlang",
) -> str:
    """Build a platform-scoped secret path.

    Args:
        base_path: Relative path within platform namespace.
        platform_prefix: Override for platform path prefix.

    Returns:
        Full platform-scoped Vault path.
    """
    ctx = TenantSecretContext(platform_path_prefix=platform_prefix)
    return ctx.build_path(base_path, is_platform=True)
