# -*- coding: utf-8 -*-
"""
Tests for Route Protector - JWT Authentication Service (SEC-001)

Covers:
    - Permission wildcard matching
    - Public path detection
    - AuthDependency behaviour (public bypass, bearer extraction, 401)
    - PermissionDependency (grant, deny, wildcard)
    - TenantDependency (match, mismatch, super_admin bypass)
    - protect_router() bulk injection
    - Decorator helpers (require_auth, require_permissions, require_roles, require_tenant)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.auth_service.route_protector import (
    PUBLIC_PATHS,
    PERMISSION_MAP,
    AuthContext,
    AuthDependency,
    PermissionDependency,
    TenantDependency,
    _is_public_path,
    _normalise_path,
    _permission_matches,
    _user_has_permission,
    _lookup_permission_for_route,
    _extract_request,
    _extract_client_ip,
    protect_router,
    require_auth,
    require_permissions,
    require_roles,
    require_tenant,
    permission_matches,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_auth_context(
    user_id: str = "usr-1",
    tenant_id: str = "t-acme",
    roles: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
    scopes: Optional[List[str]] = None,
) -> AuthContext:
    """Create an AuthContext for testing."""
    return AuthContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles or [],
        permissions=permissions or [],
        scopes=scopes or [],
        auth_method="jwt",
    )


class _FakeRequest:
    """Minimal stand-in for a FastAPI Request."""

    def __init__(
        self,
        path: str = "/api/v1/agents",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        auth: Any = None,
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
    ) -> None:
        self.method = method

        class _URL:
            pass

        self.url = _URL()
        self.url.path = path  # type: ignore[attr-defined]
        self.headers = headers or {}
        self.path_params = path_params or {}
        self.query_params = query_params or {}
        self.client = MagicMock(host="127.0.0.1")
        self.state = MagicMock()
        self.state.auth = auth


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestNormalisePath:
    """Tests for _normalise_path."""

    def test_strips_trailing_slash(self) -> None:
        assert _normalise_path("/api/v1/agents/") == "/api/v1/agents"

    def test_root_stays_slash(self) -> None:
        assert _normalise_path("/") == "/"

    def test_no_change_when_no_trailing_slash(self) -> None:
        assert _normalise_path("/health") == "/health"


class TestIsPublicPath:
    """Tests for _is_public_path."""

    def test_exact_match(self) -> None:
        assert _is_public_path("/health") is True
        assert _is_public_path("/docs") is True
        assert _is_public_path("/auth/login") is True
        assert _is_public_path("/metrics") is True

    def test_prefix_match(self) -> None:
        assert _is_public_path("/docs/swagger") is True

    def test_non_public(self) -> None:
        assert _is_public_path("/api/v1/agents") is False
        assert _is_public_path("/auth/admin/users") is False

    def test_trailing_slash(self) -> None:
        assert _is_public_path("/health/") is True


class TestPermissionMatches:
    """Tests for _permission_matches wildcard logic."""

    def test_exact_match(self) -> None:
        assert _permission_matches("agents:execute", "agents:execute") is True

    def test_mismatch(self) -> None:
        assert _permission_matches("agents:list", "agents:execute") is False

    def test_star_matches_everything(self) -> None:
        assert _permission_matches("*", "agents:execute") is True
        assert _permission_matches("*", "admin:users:list") is True

    def test_wildcard_single_level(self) -> None:
        assert _permission_matches("agents:*", "agents:execute") is True
        assert _permission_matches("agents:*", "agents:list") is True

    def test_wildcard_does_not_cross_namespace(self) -> None:
        assert _permission_matches("agents:*", "emissions:list") is False

    def test_hierarchical_wildcard(self) -> None:
        assert _permission_matches("admin:*", "admin:users:list") is True
        assert _permission_matches("admin:*", "admin:users:unlock") is True

    def test_public_alias(self) -> None:
        """permission_matches is the public alias."""
        assert permission_matches("agents:*", "agents:read") is True


class TestUserHasPermission:
    """Tests for _user_has_permission."""

    def test_direct_match(self) -> None:
        auth = _make_auth_context(permissions=["agents:execute"])
        assert _user_has_permission(auth, "agents:execute") is True

    def test_wildcard_match(self) -> None:
        auth = _make_auth_context(permissions=["agents:*"])
        assert _user_has_permission(auth, "agents:execute") is True

    def test_no_match(self) -> None:
        auth = _make_auth_context(permissions=["emissions:list"])
        assert _user_has_permission(auth, "agents:execute") is False

    def test_super_admin_bypasses(self) -> None:
        auth = _make_auth_context(roles=["super_admin"], permissions=[])
        assert _user_has_permission(auth, "agents:execute") is True

    def test_empty_permissions(self) -> None:
        auth = _make_auth_context(permissions=[])
        assert _user_has_permission(auth, "agents:execute") is False


class TestLookupPermissionForRoute:
    """Tests for _lookup_permission_for_route."""

    def test_exact_match(self) -> None:
        perm = _lookup_permission_for_route("GET", "/api/v1/agents")
        assert perm == "agents:list"

    def test_case_insensitive_method(self) -> None:
        perm = _lookup_permission_for_route("get", "/api/v1/agents")
        assert perm == "agents:list"

    def test_no_match(self) -> None:
        perm = _lookup_permission_for_route("DELETE", "/api/v1/unknown")
        assert perm is None

    def test_custom_map(self) -> None:
        custom = {"GET:/custom": "custom:read"}
        perm = _lookup_permission_for_route("GET", "/custom", custom)
        assert perm == "custom:read"


# ---------------------------------------------------------------------------
# Unit tests: AuthDependency
# ---------------------------------------------------------------------------


class TestAuthDependency:
    """Tests for AuthDependency."""

    @pytest.mark.asyncio
    async def test_public_path_returns_none(self) -> None:
        dep = AuthDependency(required=True)
        request = _FakeRequest(path="/health")
        result = await dep(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_existing_auth_returned(self) -> None:
        auth = _make_auth_context()
        dep = AuthDependency(required=True)
        request = _FakeRequest(path="/api/v1/agents", auth=auth)
        result = await dep(request)
        assert result is auth

    @pytest.mark.asyncio
    async def test_no_auth_required_false_returns_none(self) -> None:
        dep = AuthDependency(required=False)
        request = _FakeRequest(path="/api/v1/agents", auth=None)
        result = await dep(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_auth_required_true_raises_401(self) -> None:
        dep = AuthDependency(required=True)
        request = _FakeRequest(path="/api/v1/agents", auth=None)
        with pytest.raises(Exception) as exc_info:
            await dep(request)
        # HTTPException should have status 401
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_token_with_token_service(self) -> None:
        mock_claims = MagicMock()
        mock_claims.sub = "usr-1"
        mock_claims.tenant_id = "t-acme"
        mock_claims.jti = "jti-abc"
        mock_claims.roles = ["viewer"]
        mock_claims.permissions = ["agents:list"]
        mock_claims.scopes = []
        mock_claims.email = "usr@example.com"
        mock_claims.name = "Test User"

        mock_service = AsyncMock()
        mock_service.validate_token = AsyncMock(return_value=mock_claims)

        dep = AuthDependency(token_service=mock_service, required=True)
        request = _FakeRequest(
            path="/api/v1/agents",
            auth=None,
            headers={"Authorization": "Bearer fake-token-123"},
        )
        result = await dep(request)
        assert result is not None
        assert result.user_id == "usr-1"
        assert result.tenant_id == "t-acme"
        mock_service.validate_token.assert_awaited_once_with("fake-token-123")

    @pytest.mark.asyncio
    async def test_bearer_token_invalid_format_ignored(self) -> None:
        dep = AuthDependency(required=False)
        request = _FakeRequest(
            path="/api/v1/agents",
            auth=None,
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        result = await dep(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_bearer_token_service_exception_returns_none(self) -> None:
        mock_service = AsyncMock()
        mock_service.validate_token = AsyncMock(side_effect=ValueError("bad token"))

        dep = AuthDependency(token_service=mock_service, required=False)
        request = _FakeRequest(
            path="/api/v1/agents",
            auth=None,
            headers={"Authorization": "Bearer bad-token"},
        )
        result = await dep(request)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: PermissionDependency
# ---------------------------------------------------------------------------


class TestPermissionDependency:
    """Tests for PermissionDependency."""

    @pytest.mark.asyncio
    async def test_grants_when_permission_matches(self) -> None:
        auth = _make_auth_context(permissions=["agents:execute"])
        dep = PermissionDependency("agents:execute")
        request = _FakeRequest(auth=auth)
        # Should not raise
        await dep(request)

    @pytest.mark.asyncio
    async def test_grants_with_wildcard(self) -> None:
        auth = _make_auth_context(permissions=["agents:*"])
        dep = PermissionDependency("agents:execute")
        request = _FakeRequest(auth=auth)
        await dep(request)  # no exception

    @pytest.mark.asyncio
    async def test_denies_when_no_match(self) -> None:
        auth = _make_auth_context(permissions=["emissions:list"])
        dep = PermissionDependency("agents:execute")
        request = _FakeRequest(auth=auth)
        with pytest.raises(Exception) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_raises_401_when_no_auth(self) -> None:
        dep = PermissionDependency("agents:execute")
        request = _FakeRequest(auth=None)
        with pytest.raises(Exception) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Unit tests: TenantDependency
# ---------------------------------------------------------------------------


class TestTenantDependency:
    """Tests for TenantDependency."""

    @pytest.mark.asyncio
    async def test_matching_tenant_returns_id(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")
        dep = TenantDependency()
        request = _FakeRequest(
            auth=auth,
            path_params={"tenant_id": "t-acme"},
        )
        result = await dep(request)
        assert result == "t-acme"

    @pytest.mark.asyncio
    async def test_no_explicit_tenant_uses_auth_tenant(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")
        dep = TenantDependency()
        request = _FakeRequest(auth=auth)
        result = await dep(request)
        assert result == "t-acme"

    @pytest.mark.asyncio
    async def test_mismatched_tenant_raises_403(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")
        dep = TenantDependency()
        request = _FakeRequest(
            auth=auth,
            path_params={"tenant_id": "t-other"},
        )
        with pytest.raises(Exception) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_super_admin_bypasses_tenant_check(self) -> None:
        auth = _make_auth_context(
            tenant_id="t-acme", roles=["super_admin"]
        )
        dep = TenantDependency()
        request = _FakeRequest(
            auth=auth,
            path_params={"tenant_id": "t-other"},
        )
        result = await dep(request)
        assert result == "t-other"

    @pytest.mark.asyncio
    async def test_raises_401_when_no_auth(self) -> None:
        dep = TenantDependency()
        request = _FakeRequest(auth=None)
        with pytest.raises(Exception) as exc_info:
            await dep(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_resolves_from_query_param(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")
        dep = TenantDependency()
        request = _FakeRequest(
            auth=auth,
            query_params={"tenant_id": "t-acme"},
        )
        result = await dep(request)
        assert result == "t-acme"

    @pytest.mark.asyncio
    async def test_resolves_from_header(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")
        dep = TenantDependency(header_name="X-Tenant-ID")
        request = _FakeRequest(
            auth=auth,
            headers={"X-Tenant-ID": "t-acme"},
        )
        result = await dep(request)
        assert result == "t-acme"


# ---------------------------------------------------------------------------
# Unit tests: decorators
# ---------------------------------------------------------------------------


class TestRequireAuthDecorator:
    """Tests for the require_auth decorator."""

    @pytest.mark.asyncio
    async def test_passes_when_authenticated(self) -> None:
        auth = _make_auth_context()

        @require_auth
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_401_when_no_auth(self) -> None:
        @require_auth
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=None)
        with pytest.raises(Exception) as exc_info:
            await handler(request=request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_callable_with_parens(self) -> None:
        """@require_auth() also works."""
        auth = _make_auth_context()

        @require_auth()
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"


class TestRequirePermissionsDecorator:
    """Tests for the require_permissions decorator."""

    @pytest.mark.asyncio
    async def test_passes_with_matching_permission(self) -> None:
        auth = _make_auth_context(permissions=["agents:execute"])

        @require_permissions("agents:execute")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_denies_without_permission(self) -> None:
        auth = _make_auth_context(permissions=["emissions:list"])

        @require_permissions("agents:execute")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        with pytest.raises(Exception) as exc_info:
            await handler(request=request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_wildcard_permission_passes(self) -> None:
        auth = _make_auth_context(permissions=["agents:*"])

        @require_permissions("agents:execute")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_multiple_permissions_any_passes(self) -> None:
        auth = _make_auth_context(permissions=["emissions:list"])

        @require_permissions("agents:execute", "emissions:list")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"


class TestRequireRolesDecorator:
    """Tests for the require_roles decorator."""

    @pytest.mark.asyncio
    async def test_passes_with_matching_role(self) -> None:
        auth = _make_auth_context(roles=["admin"])

        @require_roles("admin")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_denies_without_role(self) -> None:
        auth = _make_auth_context(roles=["viewer"])

        @require_roles("admin")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        with pytest.raises(Exception) as exc_info:
            await handler(request=request)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_multiple_roles_any_passes(self) -> None:
        auth = _make_auth_context(roles=["viewer"])

        @require_roles("admin", "viewer")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_raises_401_when_no_auth(self) -> None:
        @require_roles("admin")
        async def handler(request: Any) -> str:
            return "ok"

        request = _FakeRequest(auth=None)
        with pytest.raises(Exception) as exc_info:
            await handler(request=request)
        assert exc_info.value.status_code == 401


class TestRequireTenantDecorator:
    """Tests for the require_tenant decorator."""

    @pytest.mark.asyncio
    async def test_passes_when_tenant_matches(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")

        @require_tenant
        async def handler(request: Any, tenant_id: str = "t-acme") -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request, tenant_id="t-acme")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_denies_when_tenant_mismatches(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")

        @require_tenant
        async def handler(request: Any, tenant_id: str = "t-other") -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        with pytest.raises(Exception) as exc_info:
            await handler(request=request, tenant_id="t-other")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_super_admin_bypasses(self) -> None:
        auth = _make_auth_context(
            tenant_id="t-acme", roles=["super_admin"]
        )

        @require_tenant
        async def handler(request: Any, tenant_id: str = "t-other") -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request, tenant_id="t-other")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_callable_with_parens(self) -> None:
        auth = _make_auth_context(tenant_id="t-acme")

        @require_tenant()
        async def handler(request: Any, tenant_id: str = "t-acme") -> str:
            return "ok"

        request = _FakeRequest(auth=auth)
        result = await handler(request=request, tenant_id="t-acme")
        assert result == "ok"


# ---------------------------------------------------------------------------
# Unit tests: extract helpers
# ---------------------------------------------------------------------------


class TestExtractRequest:
    """Tests for _extract_request."""

    def test_from_kwargs(self) -> None:
        request = _FakeRequest()
        result = _extract_request((), {"request": request})
        assert result is request

    def test_from_args_with_real_request(self) -> None:
        """_extract_request checks isinstance(arg, Request).

        We use a MagicMock(spec=Request) to satisfy the isinstance check
        since _FakeRequest is not a real Request subclass.
        """
        from fastapi import Request as RealRequest

        mock_request = MagicMock(spec=RealRequest)
        result = _extract_request((mock_request,), {})
        assert result is mock_request

    def test_returns_none_when_missing(self) -> None:
        result = _extract_request(("not-a-request",), {"other": "val"})
        assert result is None


class TestExtractClientIp:
    """Tests for _extract_client_ip."""

    def test_from_forwarded_header(self) -> None:
        request = _FakeRequest(headers={"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})
        assert _extract_client_ip(request) == "1.2.3.4"

    def test_from_client_host(self) -> None:
        request = _FakeRequest()
        assert _extract_client_ip(request) == "127.0.0.1"

    def test_unknown_when_no_client(self) -> None:
        request = _FakeRequest()
        request.client = None
        assert _extract_client_ip(request) == "unknown"


# ---------------------------------------------------------------------------
# Unit tests: protect_router
# ---------------------------------------------------------------------------


class TestProtectRouter:
    """Tests for protect_router bulk injection."""

    def test_injects_dependencies(self) -> None:
        from fastapi import APIRouter
        from fastapi.routing import APIRoute

        test_router = APIRouter()

        @test_router.get("/agents")
        async def list_agents() -> dict:
            return {}

        @test_router.post("/agents/{agent_id}/execute")
        async def execute_agent(agent_id: str) -> dict:
            return {}

        auth_dep = AuthDependency(required=True)
        custom_map = {
            "GET:/agents": "agents:list",
            "POST:/agents/{agent_id}/execute": "agents:execute",
        }

        protect_router(test_router, auth_dep=auth_dep, permission_map=custom_map)

        # Each route should now have dependencies injected
        for route in test_router.routes:
            if isinstance(route, APIRoute):
                assert len(route.dependencies) > 0

    def test_skips_public_paths(self) -> None:
        from fastapi import APIRouter
        from fastapi.routing import APIRoute

        test_router = APIRouter()

        @test_router.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        protect_router(test_router)

        for route in test_router.routes:
            if isinstance(route, APIRoute) and route.path == "/health":
                # Should have no injected dependencies
                assert len(route.dependencies) == 0

    def test_skips_excluded_paths(self) -> None:
        from fastapi import APIRouter
        from fastapi.routing import APIRoute

        test_router = APIRouter()

        @test_router.get("/custom-skip")
        async def custom_skip() -> dict:
            return {}

        protect_router(test_router, exclude_paths={"/custom-skip"})

        for route in test_router.routes:
            if isinstance(route, APIRoute) and route.path == "/custom-skip":
                assert len(route.dependencies) == 0


# ---------------------------------------------------------------------------
# Unit tests: constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_public_paths_contains_health(self) -> None:
        assert "/health" in PUBLIC_PATHS
        assert "/docs" in PUBLIC_PATHS
        assert "/openapi.json" in PUBLIC_PATHS

    def test_permission_map_has_agent_routes(self) -> None:
        assert "GET:/api/v1/agents" in PERMISSION_MAP
        assert PERMISSION_MAP["GET:/api/v1/agents"] == "agents:list"

    def test_permission_map_has_admin_routes(self) -> None:
        assert "GET:/auth/admin/users" in PERMISSION_MAP
        assert PERMISSION_MAP["GET:/auth/admin/users"] == "admin:users:list"
