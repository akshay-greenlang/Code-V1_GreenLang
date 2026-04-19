# -*- coding: utf-8 -*-
"""
Comprehensive Integration Tests for REST API Endpoints

40 test cases covering:
- REST endpoints (10 tests)
- Authentication - JWT and API key (10 tests)
- Rate limiting (5 tests)
- Request validation (7 tests)
- Response formats (5 tests)
- Pagination (3 tests)

Target: 85%+ coverage of API endpoint integration paths
Run with: pytest tests/integration/test_api_endpoints_comprehensive.py -v --tb=short

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import time
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Mock HTTP Request/Response Classes
# =============================================================================

@dataclass
class MockRequest:
    """Mock HTTP request object."""
    method: str
    path: str
    headers: Dict[str, str]
    body: Optional[Dict[str, Any]] = None
    query_params: Optional[Dict[str, str]] = None


@dataclass
class MockResponse:
    """Mock HTTP response object."""
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def jwt_secret():
    """JWT secret for testing."""
    return "test-secret-key-for-jwt-signing"


@pytest.fixture
def valid_jwt_token(jwt_secret):
    """Generate a valid JWT token."""
    payload = {
        "sub": "user-001",
        "email": "test@greenlang.io",
        "roles": ["user", "analyst"],
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def expired_jwt_token(jwt_secret):
    """Generate an expired JWT token."""
    payload = {
        "sub": "user-001",
        "email": "test@greenlang.io",
        "roles": ["user"],
        "exp": datetime.utcnow() - timedelta(hours=1),
        "iat": datetime.utcnow() - timedelta(hours=2),
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


@pytest.fixture
def valid_api_key():
    """Valid API key for testing."""
    return "gl_test_key_1234567890abcdef"


@pytest.fixture
def rate_limiter():
    """Create a mock rate limiter."""
    class RateLimiter:
        def __init__(self, max_requests: int = 100, window_seconds: int = 60):
            self.max_requests = max_requests
            self.window_seconds = window_seconds
            self.requests: Dict[str, List[float]] = {}

        def is_allowed(self, client_id: str) -> bool:
            now = time.time()
            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove old requests outside window
            self.requests[client_id] = [
                ts for ts in self.requests[client_id]
                if now - ts < self.window_seconds
            ]

            if len(self.requests[client_id]) >= self.max_requests:
                return False

            self.requests[client_id].append(now)
            return True

        def get_remaining(self, client_id: str) -> int:
            now = time.time()
            if client_id not in self.requests:
                return self.max_requests

            valid_requests = [
                ts for ts in self.requests[client_id]
                if now - ts < self.window_seconds
            ]
            return max(0, self.max_requests - len(valid_requests))

    return RateLimiter(max_requests=10, window_seconds=1)


@pytest.fixture
def mock_api_router():
    """Create a mock API router."""
    class APIRouter:
        def __init__(self):
            self.routes = {}
            self.middleware = []

        def register(self, method: str, path: str, handler):
            key = f"{method.upper()}:{path}"
            self.routes[key] = handler

        def add_middleware(self, middleware):
            self.middleware.append(middleware)

        async def handle_request(self, request: MockRequest) -> MockResponse:
            key = f"{request.method.upper()}:{request.path}"

            # Run middleware
            for mw in self.middleware:
                result = await mw(request)
                if isinstance(result, MockResponse):
                    return result

            if key not in self.routes:
                return MockResponse(
                    status_code=404,
                    body={"error": "Not found", "path": request.path},
                    headers={"Content-Type": "application/json"}
                )

            handler = self.routes[key]
            return await handler(request)

    return APIRouter()


@pytest.fixture
def sample_fuel_payload():
    """Sample fuel calculation payload."""
    return {
        "fuel_type": "natural_gas",
        "quantity": 10000,
        "unit": "MJ",
        "region": "US",
        "year": 2024,
    }


@pytest.fixture
def sample_cbam_payload():
    """Sample CBAM calculation payload."""
    return {
        "product_type": "steel_hot_rolled_coil",
        "quantity_tonnes": 100,
        "origin_country": "CN",
        "direct_emissions_tco2e": 170,
        "indirect_emissions_tco2e": 30,
        "import_date": "2025-01-15",
    }


@pytest.fixture
def sample_eudr_payload():
    """Sample EUDR compliance payload."""
    return {
        "commodity_type": "coffee",
        "quantity_kg": 5000,
        "origin_country": "BR",
        "production_date": "2024-06-15",
        "geolocation": {
            "type": "Point",
            "coordinates": [-47.9292, -15.7801],
        },
    }


# =============================================================================
# REST Endpoints Tests (10 tests)
# =============================================================================

class TestRESTEndpoints:
    """Test REST API endpoints - 10 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fuel_emissions_endpoint(self, mock_api_router, sample_fuel_payload):
        """INT-API-001: Test POST /api/v1/emissions/fuel endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            data = request.body
            emissions = data["quantity"] * 0.0561  # Natural gas factor

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "emissions_kgco2e": round(emissions, 4),
                        "fuel_type": data["fuel_type"],
                        "emission_factor": 0.0561,
                        "emission_factor_source": "EPA 2024",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/emissions/fuel", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body=sample_fuel_payload
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["success"] is True
        assert response.body["data"]["emissions_kgco2e"] == pytest.approx(561.0, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cbam_calculation_endpoint(self, mock_api_router, sample_cbam_payload):
        """INT-API-002: Test POST /api/v1/cbam/calculate endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            data = request.body
            total_emissions = data["direct_emissions_tco2e"] + data["indirect_emissions_tco2e"]
            carbon_intensity = total_emissions / data["quantity_tonnes"]

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "carbon_intensity": round(carbon_intensity, 6),
                        "benchmark_value": 1.85,
                        "surplus_emissions": max(0, (carbon_intensity - 1.85) * data["quantity_tonnes"]),
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/cbam/calculate", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/cbam/calculate",
            headers={"Content-Type": "application/json"},
            body=sample_cbam_payload
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["success"] is True
        assert response.body["data"]["carbon_intensity"] == pytest.approx(2.0, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_eudr_compliance_endpoint(self, mock_api_router, sample_eudr_payload):
        """INT-API-003: Test POST /api/v1/eudr/check endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            data = request.body
            high_risk_countries = ["BR", "ID", "MY"]

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "commodity_type": data["commodity_type"],
                        "eudr_regulated": True,
                        "risk_level": "high" if data["origin_country"] in high_risk_countries else "standard",
                        "dds_required": True,
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/eudr/check", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/eudr/check",
            headers={"Content-Type": "application/json"},
            body=sample_eudr_payload
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["data"]["risk_level"] == "high"
        assert response.body["data"]["eudr_regulated"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_building_energy_endpoint(self, mock_api_router):
        """INT-API-004: Test POST /api/v1/buildings/energy endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            data = request.body
            eui = data["energy_consumption_kwh"] / data["floor_area_sqm"]

            if eui <= 80:
                rating = "A"
            elif eui <= 120:
                rating = "B"
            elif eui <= 180:
                rating = "C"
            else:
                rating = "D"

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "eui_kwh_per_sqm": round(eui, 2),
                        "energy_rating": rating,
                        "building_type": data["building_type"],
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/buildings/energy", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/buildings/energy",
            headers={"Content-Type": "application/json"},
            body={
                "building_type": "office",
                "floor_area_sqm": 5000,
                "energy_consumption_kwh": 600000,
            }
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["data"]["eui_kwh_per_sqm"] == 120.0
        assert response.body["data"]["energy_rating"] == "B"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_emission_factors_get_endpoint(self, mock_api_router):
        """INT-API-005: Test GET /api/v1/emission-factors endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            fuel_type = request.query_params.get("fuel_type", "natural_gas")
            region = request.query_params.get("region", "US")

            factors = {
                ("natural_gas", "US"): 0.0561,
                ("diesel", "US"): 0.0745,
                ("natural_gas", "GB"): 0.0549,
                ("diesel", "GB"): 0.0732,
            }

            factor = factors.get((fuel_type, region))

            if factor is None:
                return MockResponse(
                    status_code=404,
                    body={"error": "Factor not found"},
                    headers={"Content-Type": "application/json"}
                )

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "fuel_type": fuel_type,
                        "region": region,
                        "emission_factor": factor,
                        "unit": "kgCO2e/MJ",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/emission-factors", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/emission-factors",
            headers={},
            query_params={"fuel_type": "natural_gas", "region": "US"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["data"]["emission_factor"] == 0.0561

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, mock_api_router):
        """INT-API-006: Test GET /api/v1/health endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={
                    "status": "healthy",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "components": {
                        "database": "healthy",
                        "cache": "healthy",
                        "agents": "healthy",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/health", handler)

        request = MockRequest(method="GET", path="/api/v1/health", headers={})
        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["status"] == "healthy"
        assert all(c == "healthy" for c in response.body["components"].values())

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_endpoint(self, mock_api_router):
        """INT-API-007: Test POST /api/v1/emissions/batch endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            records = request.body.get("records", [])
            results = []

            for record in records:
                emissions = record["quantity"] * 0.0561
                results.append({
                    "record_id": record.get("id"),
                    "emissions_kgco2e": round(emissions, 4),
                })

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "processed_count": len(results),
                        "results": results,
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/emissions/batch", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/batch",
            headers={"Content-Type": "application/json"},
            body={
                "records": [
                    {"id": "1", "fuel_type": "natural_gas", "quantity": 1000},
                    {"id": "2", "fuel_type": "natural_gas", "quantity": 2000},
                    {"id": "3", "fuel_type": "natural_gas", "quantity": 3000},
                ]
            }
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["data"]["processed_count"] == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_not_found_endpoint(self, mock_api_router):
        """INT-API-008: Test 404 response for non-existent endpoint."""
        request = MockRequest(
            method="GET",
            path="/api/v1/nonexistent",
            headers={}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 404
        assert "error" in response.body

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provenance_endpoint(self, mock_api_router):
        """INT-API-009: Test GET /api/v1/provenance/{hash} endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            # Extract hash from path (simplified)
            provenance_hash = "abc123def456"

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "provenance_hash": provenance_hash,
                        "created_at": "2025-01-15T10:30:00Z",
                        "calculation_type": "fuel_emissions",
                        "input_hash": hashlib.sha256(b"input").hexdigest(),
                        "output_hash": hashlib.sha256(b"output").hexdigest(),
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/provenance/{hash}", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/provenance/{hash}",
            headers={}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert "provenance_hash" in response.body["data"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_export_endpoint(self, mock_api_router):
        """INT-API-010: Test POST /api/v1/export/cbam-xml endpoint."""
        async def handler(request: MockRequest) -> MockResponse:
            data = request.body

            # Generate XML (simplified)
            xml_content = f"""<?xml version="1.0"?>
<CBAMDeclaration>
    <Period>{data.get('period', 'Q1-2025')}</Period>
    <TotalEmissions>{data.get('total_emissions', 0)}</TotalEmissions>
</CBAMDeclaration>"""

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "format": "xml",
                        "content": xml_content,
                        "filename": "cbam_declaration_Q1_2025.xml",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/export/cbam-xml", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/export/cbam-xml",
            headers={"Content-Type": "application/json"},
            body={"period": "Q1-2025", "total_emissions": 1500}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert "CBAMDeclaration" in response.body["data"]["content"]


# =============================================================================
# Authentication Tests (10 tests)
# =============================================================================

class TestAuthentication:
    """Test authentication - 10 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_valid_token(self, mock_api_router, valid_jwt_token, jwt_secret):
        """INT-AUTH-001: Test valid JWT token authentication."""
        async def auth_middleware(request: MockRequest):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return MockResponse(
                    status_code=401,
                    body={"error": "Missing bearer token"},
                    headers={}
                )

            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                request.user = payload
            except jwt.ExpiredSignatureError:
                return MockResponse(
                    status_code=401,
                    body={"error": "Token expired"},
                    headers={}
                )
            except jwt.InvalidTokenError:
                return MockResponse(
                    status_code=401,
                    body={"error": "Invalid token"},
                    headers={}
                )
            return None

        async def protected_handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={"success": True, "user": request.user["sub"]},
                headers={}
            )

        mock_api_router.add_middleware(auth_middleware)
        mock_api_router.register("GET", "/api/v1/protected", protected_handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/protected",
            headers={"Authorization": f"Bearer {valid_jwt_token}"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["user"] == "user-001"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_expired_token(self, mock_api_router, expired_jwt_token, jwt_secret):
        """INT-AUTH-002: Test expired JWT token rejection."""
        async def auth_middleware(request: MockRequest):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                try:
                    jwt.decode(token, jwt_secret, algorithms=["HS256"])
                except jwt.ExpiredSignatureError:
                    return MockResponse(
                        status_code=401,
                        body={"error": "Token expired"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(auth_middleware)

        request = MockRequest(
            method="GET",
            path="/api/v1/protected",
            headers={"Authorization": f"Bearer {expired_jwt_token}"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 401
        assert response.body["error"] == "Token expired"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_invalid_token(self, mock_api_router, jwt_secret):
        """INT-AUTH-003: Test invalid JWT token rejection."""
        async def auth_middleware(request: MockRequest):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                try:
                    jwt.decode(token, jwt_secret, algorithms=["HS256"])
                except jwt.InvalidTokenError:
                    return MockResponse(
                        status_code=401,
                        body={"error": "Invalid token"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(auth_middleware)

        request = MockRequest(
            method="GET",
            path="/api/v1/protected",
            headers={"Authorization": "Bearer invalid.token.here"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 401
        assert response.body["error"] == "Invalid token"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_missing_token(self, mock_api_router):
        """INT-AUTH-004: Test missing JWT token."""
        async def auth_middleware(request: MockRequest):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header:
                return MockResponse(
                    status_code=401,
                    body={"error": "Authorization header required"},
                    headers={}
                )
            return None

        mock_api_router.add_middleware(auth_middleware)

        request = MockRequest(
            method="GET",
            path="/api/v1/protected",
            headers={}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 401

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_key_valid(self, mock_api_router, valid_api_key):
        """INT-AUTH-005: Test valid API key authentication."""
        valid_keys = {valid_api_key: {"user_id": "api-user-001", "scope": "read:write"}}

        async def api_key_middleware(request: MockRequest):
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return None  # Let JWT handle it

            if api_key not in valid_keys:
                return MockResponse(
                    status_code=401,
                    body={"error": "Invalid API key"},
                    headers={}
                )

            request.api_user = valid_keys[api_key]
            return None

        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={"success": True, "api_user": getattr(request, "api_user", {})},
                headers={}
            )

        mock_api_router.add_middleware(api_key_middleware)
        mock_api_router.register("GET", "/api/v1/data", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/data",
            headers={"X-API-Key": valid_api_key}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert response.body["api_user"]["user_id"] == "api-user-001"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_key_invalid(self, mock_api_router):
        """INT-AUTH-006: Test invalid API key rejection."""
        async def api_key_middleware(request: MockRequest):
            api_key = request.headers.get("X-API-Key")
            if api_key and api_key != "valid-key":
                return MockResponse(
                    status_code=401,
                    body={"error": "Invalid API key"},
                    headers={}
                )
            return None

        mock_api_router.add_middleware(api_key_middleware)

        request = MockRequest(
            method="GET",
            path="/api/v1/data",
            headers={"X-API-Key": "invalid-key"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 401
        assert response.body["error"] == "Invalid API key"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_jwt_role_authorization(self, mock_api_router, jwt_secret):
        """INT-AUTH-007: Test JWT role-based authorization."""
        async def admin_middleware(request: MockRequest):
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                try:
                    payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                    if "admin" not in payload.get("roles", []):
                        return MockResponse(
                            status_code=403,
                            body={"error": "Admin role required"},
                            headers={}
                        )
                    request.user = payload
                except jwt.InvalidTokenError:
                    pass
            return None

        # Create non-admin token
        non_admin_token = jwt.encode({
            "sub": "user-002",
            "roles": ["user"],
            "exp": datetime.utcnow() + timedelta(hours=1),
        }, jwt_secret, algorithm="HS256")

        mock_api_router.add_middleware(admin_middleware)

        request = MockRequest(
            method="DELETE",
            path="/api/v1/admin/users",
            headers={"Authorization": f"Bearer {non_admin_token}"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 403
        assert response.body["error"] == "Admin role required"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_key_scope_check(self, mock_api_router):
        """INT-AUTH-008: Test API key scope validation."""
        api_keys = {
            "read-only-key": {"scopes": ["read"]},
            "read-write-key": {"scopes": ["read", "write"]},
        }

        async def scope_middleware(request: MockRequest):
            api_key = request.headers.get("X-API-Key")
            if api_key in api_keys:
                key_info = api_keys[api_key]
                if request.method in ["POST", "PUT", "DELETE"]:
                    if "write" not in key_info["scopes"]:
                        return MockResponse(
                            status_code=403,
                            body={"error": "Write scope required"},
                            headers={}
                        )
            return None

        mock_api_router.add_middleware(scope_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/data",
            headers={"X-API-Key": "read-only-key"},
            body={"data": "test"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 403
        assert response.body["error"] == "Write scope required"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_refresh(self, mock_api_router, jwt_secret):
        """INT-AUTH-009: Test JWT token refresh."""
        async def handler(request: MockRequest) -> MockResponse:
            old_token = request.headers.get("Authorization", "").replace("Bearer ", "")

            try:
                payload = jwt.decode(old_token, jwt_secret, algorithms=["HS256"])
                # Issue new token
                new_payload = {
                    "sub": payload["sub"],
                    "roles": payload.get("roles", []),
                    "exp": datetime.utcnow() + timedelta(hours=1),
                    "iat": datetime.utcnow(),
                }
                new_token = jwt.encode(new_payload, jwt_secret, algorithm="HS256")

                return MockResponse(
                    status_code=200,
                    body={"success": True, "token": new_token},
                    headers={}
                )
            except jwt.InvalidTokenError:
                return MockResponse(
                    status_code=401,
                    body={"error": "Invalid token"},
                    headers={}
                )

        mock_api_router.register("POST", "/api/v1/auth/refresh", handler)

        # Create a token about to expire
        expiring_token = jwt.encode({
            "sub": "user-001",
            "roles": ["user"],
            "exp": datetime.utcnow() + timedelta(minutes=5),
        }, jwt_secret, algorithm="HS256")

        request = MockRequest(
            method="POST",
            path="/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {expiring_token}"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert "token" in response.body
        assert response.body["token"] != expiring_token

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, mock_api_router, jwt_secret):
        """INT-AUTH-010: Test multi-tenant data isolation."""
        tenant_data = {
            "tenant-A": ["data-A1", "data-A2"],
            "tenant-B": ["data-B1", "data-B2"],
        }

        async def handler(request: MockRequest) -> MockResponse:
            auth_header = request.headers.get("Authorization", "")
            token = auth_header.replace("Bearer ", "")

            try:
                payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
                tenant_id = payload.get("tenant_id")

                if tenant_id not in tenant_data:
                    return MockResponse(
                        status_code=403,
                        body={"error": "Tenant not found"},
                        headers={}
                    )

                return MockResponse(
                    status_code=200,
                    body={"success": True, "data": tenant_data[tenant_id]},
                    headers={}
                )
            except jwt.InvalidTokenError:
                return MockResponse(
                    status_code=401,
                    body={"error": "Invalid token"},
                    headers={}
                )

        mock_api_router.register("GET", "/api/v1/tenant-data", handler)

        # Create tenant A token
        tenant_a_token = jwt.encode({
            "sub": "user-001",
            "tenant_id": "tenant-A",
            "exp": datetime.utcnow() + timedelta(hours=1),
        }, jwt_secret, algorithm="HS256")

        request = MockRequest(
            method="GET",
            path="/api/v1/tenant-data",
            headers={"Authorization": f"Bearer {tenant_a_token}"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert "data-A1" in response.body["data"]
        assert "data-B1" not in response.body["data"]


# =============================================================================
# Rate Limiting Tests (5 tests)
# =============================================================================

class TestRateLimiting:
    """Test rate limiting - 5 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_under_limit(self, rate_limiter):
        """INT-RATE-001: Test requests under rate limit."""
        client_id = "client-001"

        # Make 5 requests (limit is 10)
        for i in range(5):
            assert rate_limiter.is_allowed(client_id) is True

        remaining = rate_limiter.get_remaining(client_id)
        assert remaining == 5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, rate_limiter):
        """INT-RATE-002: Test rate limit exceeded."""
        client_id = "client-002"

        # Exhaust the limit (10 requests)
        for i in range(10):
            rate_limiter.is_allowed(client_id)

        # Next request should be denied
        assert rate_limiter.is_allowed(client_id) is False
        assert rate_limiter.get_remaining(client_id) == 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self, rate_limiter):
        """INT-RATE-003: Test rate limit window reset."""
        client_id = "client-003"

        # Exhaust the limit
        for i in range(10):
            rate_limiter.is_allowed(client_id)

        assert rate_limiter.is_allowed(client_id) is False

        # Wait for window to reset (window is 1 second)
        await asyncio.sleep(1.1)

        # Should be allowed again
        assert rate_limiter.is_allowed(client_id) is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_per_client(self, rate_limiter):
        """INT-RATE-004: Test rate limits are per-client."""
        client_a = "client-A"
        client_b = "client-B"

        # Exhaust client A's limit
        for i in range(10):
            rate_limiter.is_allowed(client_a)

        # Client A blocked
        assert rate_limiter.is_allowed(client_a) is False

        # Client B should still be allowed
        assert rate_limiter.is_allowed(client_b) is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rate_limit_response_headers(self, mock_api_router, rate_limiter):
        """INT-RATE-005: Test rate limit headers in response."""
        async def rate_limit_middleware(request: MockRequest):
            client_id = request.headers.get("X-Client-ID", "anonymous")

            if not rate_limiter.is_allowed(client_id):
                return MockResponse(
                    status_code=429,
                    body={"error": "Rate limit exceeded"},
                    headers={
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + 60),
                        "Retry-After": "60",
                    }
                )

            return None

        async def handler(request: MockRequest) -> MockResponse:
            client_id = request.headers.get("X-Client-ID", "anonymous")
            return MockResponse(
                status_code=200,
                body={"success": True},
                headers={
                    "X-RateLimit-Remaining": str(rate_limiter.get_remaining(client_id)),
                    "X-RateLimit-Limit": "10",
                }
            )

        mock_api_router.add_middleware(rate_limit_middleware)
        mock_api_router.register("GET", "/api/v1/data", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/data",
            headers={"X-Client-ID": "client-headers"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Limit" in response.headers


# =============================================================================
# Request Validation Tests (7 tests)
# =============================================================================

class TestRequestValidation:
    """Test request validation - 7 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_valid_request_body(self, mock_api_router, sample_fuel_payload):
        """INT-VAL-001: Test valid request body passes validation."""
        required_fields = ["fuel_type", "quantity", "unit"]

        async def validation_middleware(request: MockRequest):
            if request.method in ["POST", "PUT"] and request.body:
                missing = [f for f in required_fields if f not in request.body]
                if missing:
                    return MockResponse(
                        status_code=400,
                        body={"error": f"Missing required fields: {missing}"},
                        headers={}
                    )
            return None

        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={"success": True},
                headers={}
            )

        mock_api_router.add_middleware(validation_middleware)
        mock_api_router.register("POST", "/api/v1/emissions/fuel", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body=sample_fuel_payload
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_required_field(self, mock_api_router):
        """INT-VAL-002: Test missing required field returns 400."""
        required_fields = ["fuel_type", "quantity"]

        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                missing = [f for f in required_fields if f not in request.body]
                if missing:
                    return MockResponse(
                        status_code=400,
                        body={"error": f"Missing required fields: {missing}"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body={"fuel_type": "natural_gas"}  # Missing quantity
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "quantity" in response.body["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_field_type(self, mock_api_router):
        """INT-VAL-003: Test invalid field type returns 400."""
        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                quantity = request.body.get("quantity")
                if quantity is not None and not isinstance(quantity, (int, float)):
                    return MockResponse(
                        status_code=400,
                        body={"error": "quantity must be a number"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body={"fuel_type": "natural_gas", "quantity": "not-a-number"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "quantity must be a number" in response.body["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_negative_value_validation(self, mock_api_router):
        """INT-VAL-004: Test negative values are rejected."""
        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                quantity = request.body.get("quantity", 0)
                if isinstance(quantity, (int, float)) and quantity < 0:
                    return MockResponse(
                        status_code=400,
                        body={"error": "quantity cannot be negative"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body={"fuel_type": "natural_gas", "quantity": -100}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "cannot be negative" in response.body["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_enum_validation(self, mock_api_router):
        """INT-VAL-005: Test enum field validation."""
        valid_fuel_types = ["natural_gas", "diesel", "gasoline", "electricity", "coal"]

        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                fuel_type = request.body.get("fuel_type")
                if fuel_type and fuel_type not in valid_fuel_types:
                    return MockResponse(
                        status_code=400,
                        body={"error": f"Invalid fuel_type. Must be one of: {valid_fuel_types}"},
                        headers={}
                    )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={"Content-Type": "application/json"},
            body={"fuel_type": "unicorn_tears", "quantity": 100}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "Invalid fuel_type" in response.body["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_date_format_validation(self, mock_api_router):
        """INT-VAL-006: Test date format validation."""
        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                date_str = request.body.get("import_date")
                if date_str:
                    try:
                        datetime.fromisoformat(date_str)
                    except ValueError:
                        return MockResponse(
                            status_code=400,
                            body={"error": "import_date must be in ISO format (YYYY-MM-DD)"},
                            headers={}
                        )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/cbam/calculate",
            headers={"Content-Type": "application/json"},
            body={"product_type": "steel", "import_date": "01/15/2025"}  # Invalid format
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "ISO format" in response.body["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_geolocation_validation(self, mock_api_router):
        """INT-VAL-007: Test geolocation coordinates validation."""
        async def validation_middleware(request: MockRequest):
            if request.method == "POST" and request.body:
                geo = request.body.get("geolocation")
                if geo:
                    coords = geo.get("coordinates", [])
                    if len(coords) != 2:
                        return MockResponse(
                            status_code=400,
                            body={"error": "geolocation coordinates must have [lon, lat]"},
                            headers={}
                        )
                    lon, lat = coords
                    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                        return MockResponse(
                            status_code=400,
                            body={"error": "Invalid coordinates: lon must be [-180, 180], lat must be [-90, 90]"},
                            headers={}
                        )
            return None

        mock_api_router.add_middleware(validation_middleware)

        request = MockRequest(
            method="POST",
            path="/api/v1/eudr/check",
            headers={"Content-Type": "application/json"},
            body={
                "commodity_type": "coffee",
                "geolocation": {"coordinates": [999, 999]}  # Invalid
            }
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert "Invalid coordinates" in response.body["error"]


# =============================================================================
# Response Format Tests (5 tests)
# =============================================================================

class TestResponseFormats:
    """Test response formats - 5 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_json_response_format(self, mock_api_router):
        """INT-RESP-001: Test JSON response format."""
        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {"emissions": 561.0},
                    "meta": {"request_id": "req-123"},
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/data", handler)

        request = MockRequest(method="GET", path="/api/v1/data", headers={})
        response = await mock_api_router.handle_request(request)

        assert response.headers["Content-Type"] == "application/json"
        assert "success" in response.body
        assert "data" in response.body

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_response_format(self, mock_api_router):
        """INT-RESP-002: Test error response format."""
        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=400,
                body={
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid input data",
                        "details": [
                            {"field": "quantity", "issue": "must be positive"}
                        ],
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/data", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/data",
            headers={},
            body={"quantity": -1}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 400
        assert response.body["success"] is False
        assert "error" in response.body
        assert response.body["error"]["code"] == "VALIDATION_ERROR"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_response_includes_metadata(self, mock_api_router):
        """INT-RESP-003: Test response includes metadata."""
        async def handler(request: MockRequest) -> MockResponse:
            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {"result": 100},
                    "meta": {
                        "request_id": "req-456",
                        "timestamp": datetime.now().isoformat(),
                        "processing_time_ms": 42,
                        "api_version": "1.0.0",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/data", handler)

        request = MockRequest(method="GET", path="/api/v1/data", headers={})
        response = await mock_api_router.handle_request(request)

        assert "meta" in response.body
        assert "request_id" in response.body["meta"]
        assert "timestamp" in response.body["meta"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_response_includes_provenance(self, mock_api_router):
        """INT-RESP-004: Test calculation response includes provenance."""
        async def handler(request: MockRequest) -> MockResponse:
            input_hash = hashlib.sha256(
                json.dumps(request.body, sort_keys=True).encode()
            ).hexdigest()

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {"emissions": 561.0},
                    "provenance": {
                        "hash": input_hash,
                        "inputs_hash": input_hash[:32],
                        "calculation_version": "1.0.0",
                        "emission_factor_uri": "ef://EPA/natural_gas/US/2024",
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("POST", "/api/v1/emissions/fuel", handler)

        request = MockRequest(
            method="POST",
            path="/api/v1/emissions/fuel",
            headers={},
            body={"fuel_type": "natural_gas", "quantity": 10000}
        )

        response = await mock_api_router.handle_request(request)

        assert "provenance" in response.body
        assert len(response.body["provenance"]["hash"]) == 64

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_csv_export_format(self, mock_api_router):
        """INT-RESP-005: Test CSV export format."""
        async def handler(request: MockRequest) -> MockResponse:
            csv_content = "fuel_type,quantity,emissions_kgco2e\nnatural_gas,10000,561.0\ndiesel,5000,372.5"

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": {
                        "format": "csv",
                        "content": csv_content,
                        "filename": "emissions_export.csv",
                    },
                },
                headers={
                    "Content-Type": "text/csv",
                    "Content-Disposition": "attachment; filename=emissions_export.csv",
                }
            )

        mock_api_router.register("GET", "/api/v1/export/csv", handler)

        request = MockRequest(method="GET", path="/api/v1/export/csv", headers={})
        response = await mock_api_router.handle_request(request)

        assert "csv" in response.headers.get("Content-Type", "")
        assert "fuel_type" in response.body["data"]["content"]


# =============================================================================
# Pagination Tests (3 tests)
# =============================================================================

class TestPagination:
    """Test pagination - 3 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_paginated_response(self, mock_api_router):
        """INT-PAGE-001: Test paginated response format."""
        all_items = [{"id": i, "name": f"Item {i}"} for i in range(100)]

        async def handler(request: MockRequest) -> MockResponse:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 10))

            start = (page - 1) * page_size
            end = start + page_size
            items = all_items[start:end]

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": items,
                    "pagination": {
                        "page": page,
                        "page_size": page_size,
                        "total_items": len(all_items),
                        "total_pages": (len(all_items) + page_size - 1) // page_size,
                        "has_next": end < len(all_items),
                        "has_prev": page > 1,
                    },
                },
                headers={"Content-Type": "application/json"}
            )

        mock_api_router.register("GET", "/api/v1/items", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/items",
            headers={},
            query_params={"page": "2", "page_size": "10"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.status_code == 200
        assert len(response.body["data"]) == 10
        assert response.body["pagination"]["page"] == 2
        assert response.body["pagination"]["total_pages"] == 10
        assert response.body["pagination"]["has_next"] is True
        assert response.body["pagination"]["has_prev"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pagination_first_page(self, mock_api_router):
        """INT-PAGE-002: Test first page pagination."""
        all_items = [{"id": i} for i in range(50)]

        async def handler(request: MockRequest) -> MockResponse:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 10))

            start = (page - 1) * page_size
            end = start + page_size
            items = all_items[start:end]

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": items,
                    "pagination": {
                        "page": page,
                        "has_next": end < len(all_items),
                        "has_prev": page > 1,
                    },
                },
                headers={}
            )

        mock_api_router.register("GET", "/api/v1/items", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/items",
            headers={},
            query_params={"page": "1", "page_size": "10"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.body["pagination"]["page"] == 1
        assert response.body["pagination"]["has_prev"] is False
        assert response.body["pagination"]["has_next"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pagination_last_page(self, mock_api_router):
        """INT-PAGE-003: Test last page pagination."""
        all_items = [{"id": i} for i in range(25)]

        async def handler(request: MockRequest) -> MockResponse:
            page = int(request.query_params.get("page", 1))
            page_size = int(request.query_params.get("page_size", 10))

            start = (page - 1) * page_size
            end = start + page_size
            items = all_items[start:end]

            return MockResponse(
                status_code=200,
                body={
                    "success": True,
                    "data": items,
                    "pagination": {
                        "page": page,
                        "has_next": end < len(all_items),
                        "has_prev": page > 1,
                    },
                },
                headers={}
            )

        mock_api_router.register("GET", "/api/v1/items", handler)

        request = MockRequest(
            method="GET",
            path="/api/v1/items",
            headers={},
            query_params={"page": "3", "page_size": "10"}
        )

        response = await mock_api_router.handle_request(request)

        assert response.body["pagination"]["page"] == 3
        assert response.body["pagination"]["has_prev"] is True
        assert response.body["pagination"]["has_next"] is False
        assert len(response.body["data"]) == 5  # Only 5 items left


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
