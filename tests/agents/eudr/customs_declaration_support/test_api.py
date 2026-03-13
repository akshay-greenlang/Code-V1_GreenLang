# -*- coding: utf-8 -*-
"""
Unit tests for FastAPI Router - AGENT-EUDR-039

Tests all 28+ API endpoints for customs declaration creation, CN code
mapping, HS code validation, tariff calculation, origin verification,
compliance checking, SAD form generation, customs submission,
MRN operations, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from greenlang.agents.eudr.customs_declaration_support.api import (
    CreateDeclarationRequest,
    MapCNCodesRequest,
    ValidateHSCodeRequest,
    CalculateTariffRequest,
    VerifyOriginRequest,
    RunComplianceCheckRequest,
    GenerateSADFormRequest,
    SubmitDeclarationRequest,
    UpdateStatusRequest,
    CalculateValueRequest,
    ConvertCurrencyRequest,
    ErrorResponse,
    get_router,
    router,
)


# ---------------------------------------------------------------------------
# Helper: create FastAPI test app with mocked service
# ---------------------------------------------------------------------------

def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with the customs declaration router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def app():
    return _build_app()


@pytest.fixture
def mock_service():
    """Create a mock CustomsDeclarationService."""
    svc = AsyncMock()
    svc.create_declaration = AsyncMock(return_value={"declaration_id": "DECL-001", "status": "draft"})
    svc.get_declaration = AsyncMock(return_value=None)
    svc.list_declarations = AsyncMock(return_value=[])
    svc.update_declaration_status = AsyncMock(return_value={"status": "updated"})
    svc.map_cn_codes = AsyncMock(return_value=[{"cn_code": "18010000", "commodity": "cocoa"}])
    svc.lookup_cn_code = AsyncMock(return_value={"cn_code": "18010000", "commodity": "cocoa"})
    svc.validate_hs_code = AsyncMock(return_value={"hs_code": "180100", "eudr_regulated": True})
    svc.validate_hs_codes_batch = AsyncMock(return_value=[])
    svc.calculate_tariff = AsyncMock(return_value={"total_payable": "5250.00"})
    svc.calculate_duty = AsyncMock(return_value={"duty_amount": "4800.00"})
    svc.calculate_customs_value = AsyncMock(return_value={"customs_value": "25000.00"})
    svc.convert_currency = AsyncMock(return_value={"converted_amount": "920.00"})
    svc.verify_origin = AsyncMock(return_value={"result": "verified", "confidence_score": "95.00"})
    svc.verify_origin_batch = AsyncMock(return_value=[])
    svc.run_compliance_check = AsyncMock(return_value={"overall_status": "compliant"})
    svc.check_dds_reference = AsyncMock(return_value={"result": "pass"})
    svc.generate_sad_form = AsyncMock(return_value={"form_id": "SAD-001"})
    svc.submit_to_ncts = AsyncMock(return_value={"status": "accepted", "mrn": "26NL0003960000001A"})
    svc.submit_to_ais = AsyncMock(return_value={"status": "accepted", "mrn": "26BE0003960000003C"})
    svc.submit_declaration = AsyncMock(return_value={"status": "submitted"})
    svc.check_submission_status = AsyncMock(return_value={"status": "processing"})
    svc.get_mrn_status = AsyncMock(return_value={"mrn_status": "accepted"})
    svc.get_tariff_summary = AsyncMock(return_value={"total_value": "25000.00"})
    svc.get_compliance_report = AsyncMock(return_value={"overall_status": "compliant"})
    svc.cancel_declaration = AsyncMock(return_value={"status": "cancelled"})
    svc.health_check = AsyncMock(return_value={"agent_id": "GL-EUDR-CDS-039", "status": "healthy"})
    return svc


@pytest_asyncio.fixture
async def client(app, mock_service):
    """Create an async httpx client patching get_service."""
    with patch(
        "greenlang.agents.eudr.customs_declaration_support.api.get_service",
        return_value=mock_service,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ====================================================================
# Request Schema Tests
# ====================================================================


class TestRequestSchemas:
    """Test request schema construction and defaults."""

    def test_create_declaration_request(self):
        req = CreateDeclarationRequest(
            operator_id="OP-001",
            operator_name="Acme Trading",
            commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert req.operator_id == "OP-001"
        assert req.declaration_type == "import"
        assert req.incoterms == "CIF"

    def test_create_declaration_request_defaults(self):
        req = CreateDeclarationRequest(
            operator_id="OP-001",
            commodities=["cocoa"],
            country_of_origin="CI",
        )
        assert req.operator_eori == ""
        assert req.port_of_entry == ""
        assert req.currency == "EUR"

    def test_map_cn_codes_request(self):
        req = MapCNCodesRequest(commodity="cocoa")
        assert req.commodity == "cocoa"

    def test_validate_hs_code_request(self):
        req = ValidateHSCodeRequest(hs_code="180100")
        assert req.hs_code == "180100"

    def test_calculate_tariff_request(self):
        req = CalculateTariffRequest(
            cn_code="18010000",
            customs_value=25000.00,
            quantity=10000.00,
            origin_country="CI",
        )
        assert req.cn_code == "18010000"
        assert req.currency == "EUR"

    def test_verify_origin_request(self):
        req = VerifyOriginRequest(
            declared_origin="CI",
            supply_chain_origins=["CI", "GH"],
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert req.declared_origin == "CI"

    def test_run_compliance_check_request(self):
        req = RunComplianceCheckRequest(
            dds_reference="GL-DDS-20260313-ABCDEF",
            cn_codes=["18010000"],
            declared_origin="CI",
        )
        assert req.deforestation_free is False
        assert req.risk_level == "standard"

    def test_submit_declaration_request(self):
        req = SubmitDeclarationRequest(system="ncts")
        assert req.system == "ncts"

    def test_update_status_request(self):
        req = UpdateStatusRequest(status="submitted")
        assert req.status == "submitted"

    def test_calculate_value_request(self):
        req = CalculateValueRequest(
            fob_value=10000.00,
            freight_cost=500.00,
            insurance_cost=100.00,
            incoterms="CIF",
        )
        assert req.incoterms == "CIF"

    def test_convert_currency_request(self):
        req = ConvertCurrencyRequest(
            amount=1000.00,
            from_currency="USD",
            to_currency="EUR",
        )
        assert req.from_currency == "USD"

    def test_error_response(self):
        resp = ErrorResponse(detail="Something went wrong")
        assert resp.error_code == "internal_error"


# ====================================================================
# Router Configuration Tests
# ====================================================================


class TestRouterConfig:
    def test_router_prefix(self):
        assert router.prefix == "/api/v1/eudr/customs-declaration"

    def test_router_tags(self):
        assert "EUDR Customs Declaration Support" in router.tags

    def test_get_router_returns_same(self):
        r = get_router()
        assert r is router

    def test_router_has_routes(self):
        paths = [route.path for route in router.routes]
        assert len(paths) > 0

    def test_router_contains_declarations(self):
        paths = [route.path for route in router.routes]
        assert any("declarations" in p or "declaration" in p for p in paths)

    def test_router_contains_health(self):
        paths = [route.path for route in router.routes]
        assert any("health" in p for p in paths)


# ====================================================================
# Declaration CRUD Endpoint Tests
# ====================================================================


class TestCreateDeclarationEndpoint:
    @pytest.mark.asyncio
    async def test_create_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations",
            json={
                "operator_id": "OP-001",
                "operator_name": "Acme Trading",
                "commodities": ["cocoa"],
                "country_of_origin": "CI",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["declaration_id"] == "DECL-001"

    @pytest.mark.asyncio
    async def test_create_value_error(self, client, mock_service):
        mock_service.create_declaration.side_effect = ValueError("Empty operator_id")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations",
            json={
                "operator_id": "",
                "commodities": ["cocoa"],
                "country_of_origin": "CI",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_internal_error(self, client, mock_service):
        mock_service.create_declaration.side_effect = RuntimeError("DB down")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations",
            json={
                "operator_id": "OP-001",
                "commodities": ["cocoa"],
                "country_of_origin": "CI",
            },
        )
        assert resp.status_code == 500


class TestListDeclarationsEndpoint:
    @pytest.mark.asyncio
    async def test_list_empty(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/customs-declaration/declarations")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_with_filters(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/declarations",
            params={"operator_id": "OP-001", "status": "draft"},
        )
        assert resp.status_code == 200


class TestGetDeclarationEndpoint:
    @pytest.mark.asyncio
    async def test_get_not_found(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/customs-declaration/declarations/DECL-NONEXISTENT")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_found(self, client, mock_service):
        mock_service.get_declaration.return_value = {"declaration_id": "DECL-001"}
        resp = await client.get("/api/v1/eudr/customs-declaration/declarations/DECL-001")
        assert resp.status_code == 200
        assert resp.json()["declaration_id"] == "DECL-001"


class TestUpdateStatusEndpoint:
    @pytest.mark.asyncio
    async def test_update_status_success(self, client, mock_service):
        resp = await client.put(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/status",
            json={"status": "submitted"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_status_value_error(self, client, mock_service):
        mock_service.update_declaration_status.side_effect = ValueError("Invalid")
        resp = await client.put(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/status",
            json={"status": "invalid_xyz"},
        )
        assert resp.status_code == 422


class TestCancelDeclarationEndpoint:
    @pytest.mark.asyncio
    async def test_cancel_success(self, client, mock_service):
        resp = await client.delete("/api/v1/eudr/customs-declaration/declarations/DECL-001")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, client, mock_service):
        mock_service.cancel_declaration.side_effect = ValueError("Not found")
        resp = await client.delete("/api/v1/eudr/customs-declaration/declarations/DECL-001")
        assert resp.status_code == 404


# ====================================================================
# CN Code Mapping Endpoint Tests
# ====================================================================


class TestCNCodeEndpoints:
    @pytest.mark.asyncio
    async def test_map_cn_codes(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/cn-codes/map",
            json={"commodity": "cocoa"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lookup_cn_code(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/cn-codes/18010000",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_lookup_cn_code_not_found(self, client, mock_service):
        mock_service.lookup_cn_code.return_value = None
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/cn-codes/99999999",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_map_cn_codes_invalid_commodity(self, client, mock_service):
        mock_service.map_cn_codes.side_effect = ValueError("Unknown commodity")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/cn-codes/map",
            json={"commodity": "unknown_xyz"},
        )
        assert resp.status_code == 422


# ====================================================================
# HS Code Validation Endpoint Tests
# ====================================================================


class TestHSCodeEndpoints:
    @pytest.mark.asyncio
    async def test_validate_hs_code(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/hs-codes/validate",
            json={"hs_code": "180100"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_hs_code_invalid(self, client, mock_service):
        mock_service.validate_hs_code.side_effect = ValueError("Invalid format")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/hs-codes/validate",
            json={"hs_code": "XYZ"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_validate_hs_codes_batch(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/hs-codes/validate/batch",
            json=["180100", "090111", "440111"],
        )
        assert resp.status_code == 200


# ====================================================================
# Tariff Calculation Endpoint Tests
# ====================================================================


class TestTariffEndpoints:
    @pytest.mark.asyncio
    async def test_calculate_tariff(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/tariff",
            json={
                "cn_code": "18010000",
                "customs_value": 25000.00,
                "quantity": 10000.00,
                "origin_country": "CI",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_calculate_customs_value(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/customs-value",
            json={
                "fob_value": 10000.00,
                "freight_cost": 500.00,
                "insurance_cost": 100.00,
                "incoterms": "CIF",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_tariff_summary(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/tariff/summary",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_convert_currency(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/currency/convert",
            json={
                "amount": 1000.00,
                "from_currency": "USD",
                "to_currency": "EUR",
            },
        )
        assert resp.status_code == 200


# ====================================================================
# Origin Verification Endpoint Tests
# ====================================================================


class TestOriginEndpoints:
    @pytest.mark.asyncio
    async def test_verify_origin(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/origin/verify",
            json={
                "declared_origin": "CI",
                "supply_chain_origins": ["CI", "GH"],
                "dds_reference": "GL-DDS-20260313-ABCDEF",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_origin_invalid(self, client, mock_service):
        mock_service.verify_origin.side_effect = ValueError("Invalid origin")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/origin/verify",
            json={
                "declared_origin": "",
                "supply_chain_origins": [],
                "dds_reference": "",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_verify_origin_batch(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/origin/verify/batch",
            json=[
                {"declared_origin": "CI", "supply_chain_origins": ["CI"], "dds_reference": "REF-1"},
            ],
        )
        assert resp.status_code == 200


# ====================================================================
# Compliance Check Endpoint Tests
# ====================================================================


class TestComplianceEndpoints:
    @pytest.mark.asyncio
    async def test_run_compliance_check(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/compliance",
            json={
                "dds_reference": "GL-DDS-20260313-ABCDEF",
                "cn_codes": ["18010000"],
                "declared_origin": "CI",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_compliance_report(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/compliance",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_compliance_not_found(self, client, mock_service):
        mock_service.get_compliance_report.return_value = None
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/compliance",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_check_dds_reference(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/compliance/dds-check",
            json={"dds_reference": "GL-DDS-20260313-ABCDEF"},
        )
        assert resp.status_code == 200


# ====================================================================
# SAD Form Endpoint Tests
# ====================================================================


class TestSADFormEndpoints:
    @pytest.mark.asyncio
    async def test_generate_sad_form(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/sad-form",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_sad_form_not_found(self, client, mock_service):
        mock_service.generate_sad_form.side_effect = ValueError("Not found")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-NONEXISTENT/sad-form",
        )
        assert resp.status_code == 404


# ====================================================================
# Customs Submission Endpoint Tests
# ====================================================================


class TestSubmissionEndpoints:
    @pytest.mark.asyncio
    async def test_submit_to_ncts(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/submit",
            json={"system": "ncts"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_to_ais(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/submit",
            json={"system": "ais"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_invalid_system(self, client, mock_service):
        mock_service.submit_declaration.side_effect = ValueError("Invalid system")
        resp = await client.post(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/submit",
            json={"system": "invalid"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_check_submission_status(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/declarations/DECL-001/submit/status",
        )
        assert resp.status_code == 200


# ====================================================================
# MRN Endpoint Tests
# ====================================================================


class TestMRNEndpoints:
    @pytest.mark.asyncio
    async def test_get_mrn_status(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/customs-declaration/mrn/26NL0003960000001A/status",
        )
        assert resp.status_code == 200


# ====================================================================
# Health Endpoint Tests
# ====================================================================


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/customs-declaration/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "GL-EUDR-CDS-039"
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_error_fallback(self, client, mock_service):
        mock_service.health_check.side_effect = RuntimeError("Engine crash")
        resp = await client.get("/api/v1/eudr/customs-declaration/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
