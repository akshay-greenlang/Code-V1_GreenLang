# -*- coding: utf-8 -*-
"""
Unit tests for FastAPI Router - AGENT-EUDR-037

Tests all 30+ API endpoints for DDS creation, assembly, geolocation
formatting, risk integration, supply chain compilation, compliance
validation, document packaging, signing, amendments, version control,
submission, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from greenlang.agents.eudr.due_diligence_statement_creator.api import (
    AddDocumentRequest,
    ApplySignatureRequest,
    AssembleDDSRequest,
    CompileSupplyChainRequest,
    CreateAmendmentRequest,
    CreateDDSRequest,
    ErrorResponse,
    FormatGeolocationRequest,
    IntegrateRiskRequest,
    SubmitDDSRequest,
    UpdateStatusRequest,
    get_router,
    router,
)


# ---------------------------------------------------------------------------
# Helper: create FastAPI test app with mocked service
# ---------------------------------------------------------------------------

def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with the DDS router attached."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def app():
    return _build_app()


@pytest.fixture
def mock_service():
    """Create a mock DDSCreatorService."""
    svc = AsyncMock()
    svc.create_statement = AsyncMock(return_value={"statement_id": "DDS-001", "status": "draft"})
    svc.assemble_statement = AsyncMock(return_value={"statement_id": "DDS-002", "status": "draft"})
    svc.get_statement = AsyncMock(return_value=None)
    svc.list_statements = AsyncMock(return_value=[])
    svc.get_statement_summary = AsyncMock(return_value=None)
    svc.update_statement_status = AsyncMock(return_value={"status": "updated"})
    svc.withdraw_statement = AsyncMock(return_value={"status": "withdrawn"})
    svc.format_geolocation = AsyncMock(return_value={"plot_id": "PLT-001"})
    svc.format_geolocations_batch = AsyncMock(return_value=[])
    svc.export_geojson = AsyncMock(return_value={"type": "FeatureCollection", "features": []})
    svc.integrate_risk = AsyncMock(return_value={"risk_id": "RISK-001"})
    svc.integrate_risk_batch = AsyncMock(return_value=[])
    svc.get_overall_risk = AsyncMock(return_value={"overall_risk_level": "standard"})
    svc.compile_supply_chain = AsyncMock(return_value={"supply_chain_id": "SC-001"})
    svc.validate_supply_chain = AsyncMock(return_value={"complete": True})
    svc.get_supply_chain_countries = AsyncMock(return_value={"CI": 5})
    svc.validate_statement = AsyncMock(return_value={"overall_result": "pass"})
    svc.get_compliance_report = AsyncMock(return_value={"overall_result": "pass"})
    svc.add_document = AsyncMock(return_value={"document_id": "DOC-001"})
    svc.create_submission_package = AsyncMock(return_value={"package_id": "PKG-001"})
    svc.validate_package = AsyncMock(return_value={"valid": True})
    svc.get_package_manifest = AsyncMock(return_value={"total_documents": 0})
    svc.apply_signature = AsyncMock(return_value={"signature_id": "SIG-001"})
    svc.validate_signature = AsyncMock(return_value={"valid": True})
    svc.create_amendment = AsyncMock(return_value={"amendment_id": "AMD-001"})
    svc.get_versions = AsyncMock(return_value=[])
    svc.get_latest_version = AsyncMock(return_value=None)
    svc.get_amendments = AsyncMock(return_value=[])
    svc.submit_statement = AsyncMock(return_value={"status": "submitted"})
    svc.health_check = AsyncMock(return_value={"agent_id": "GL-EUDR-DDSC-037", "status": "healthy"})
    return svc


@pytest_asyncio.fixture
async def client(app, mock_service):
    """Create an async httpx client patching get_service."""
    with patch(
        "greenlang.agents.eudr.due_diligence_statement_creator.api.get_service",
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

    def test_create_dds_request_required_fields(self):
        req = CreateDDSRequest(
            operator_id="OP-001",
            operator_name="Acme Corp",
            commodities=["cocoa"],
        )
        assert req.operator_id == "OP-001"
        assert req.statement_type == "placing"
        assert req.language == "en"

    def test_create_dds_request_defaults(self):
        req = CreateDDSRequest(
            operator_id="OP-001",
            operator_name="Acme Corp",
            commodities=["cocoa"],
        )
        assert req.operator_address == ""
        assert req.operator_eori_number == ""

    def test_assemble_dds_request_defaults(self):
        req = AssembleDDSRequest(
            operator_id="OP-001",
            operator_name="Acme Corp",
            commodities=["cocoa"],
        )
        assert req.total_quantity == 0.0
        assert req.quantity_unit == "metric_tonnes"
        assert req.deforestation_free is False

    def test_update_status_request(self):
        req = UpdateStatusRequest(status="submitted")
        assert req.status == "submitted"

    def test_format_geolocation_request(self):
        req = FormatGeolocationRequest(
            plot_id="PLT-001",
            latitude=5.123,
            longitude=-3.456,
        )
        assert req.area_hectares == 0.0
        assert req.collection_method == "gps_field_survey"

    def test_integrate_risk_request(self):
        req = IntegrateRiskRequest(
            risk_id="RISK-001",
            source_agent="EUDR-016",
            risk_category="country",
        )
        assert req.risk_level == "standard"
        assert req.risk_score == 0.0

    def test_compile_supply_chain_request(self):
        req = CompileSupplyChainRequest(
            supply_chain_id="SC-001",
            commodity="cocoa",
        )
        assert req.chain_of_custody_model == "segregation"
        assert req.traceability_score == 0.0

    def test_add_document_request(self):
        req = AddDocumentRequest(
            document_type="certificate_of_origin",
            filename="cert.pdf",
        )
        assert req.size_bytes == 0
        assert req.mime_type == "application/pdf"
        assert req.language == "en"

    def test_apply_signature_request(self):
        req = ApplySignatureRequest(signer_name="John Smith")
        assert req.signature_type == "qualified_electronic"
        assert req.signer_role == ""

    def test_create_amendment_request(self):
        req = CreateAmendmentRequest(
            reason="correction_of_error",
            description="Fix operator name",
            previous_version=1,
        )
        assert req.changed_by == ""
        assert req.approved_by == ""

    def test_submit_dds_request(self):
        req = SubmitDDSRequest()
        assert req.additional_documents is None

    def test_error_response(self):
        resp = ErrorResponse(detail="Something went wrong")
        assert resp.error_code == "internal_error"


# ====================================================================
# Router Tests
# ====================================================================


class TestRouterConfig:
    """Test router configuration."""

    def test_router_prefix(self):
        assert router.prefix == "/api/v1/eudr/dds-creator"

    def test_router_tags(self):
        assert "EUDR Due Diligence Statement Creator" in router.tags

    def test_get_router_returns_same(self):
        r = get_router()
        assert r is router

    def test_router_has_routes(self):
        paths = [route.path for route in router.routes]
        assert len(paths) > 0

    def test_router_contains_create_dds(self):
        paths = [route.path for route in router.routes]
        assert any("create-dds" in p for p in paths)

    def test_router_contains_health(self):
        paths = [route.path for route in router.routes]
        assert any("health" in p for p in paths)


# ====================================================================
# DDS Core Endpoint Tests
# ====================================================================


class TestCreateDDSEndpoint:
    @pytest.mark.asyncio
    async def test_create_dds_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/create-dds",
            json={
                "operator_id": "OP-001",
                "operator_name": "Acme Corp",
                "commodities": ["cocoa"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["statement_id"] == "DDS-001"

    @pytest.mark.asyncio
    async def test_create_dds_value_error(self, client, mock_service):
        mock_service.create_statement.side_effect = ValueError("Empty operator_id")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/create-dds",
            json={
                "operator_id": "",
                "operator_name": "Acme Corp",
                "commodities": ["cocoa"],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_dds_internal_error(self, client, mock_service):
        mock_service.create_statement.side_effect = RuntimeError("DB down")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/create-dds",
            json={
                "operator_id": "OP-001",
                "operator_name": "Acme Corp",
                "commodities": ["cocoa"],
            },
        )
        assert resp.status_code == 500


class TestAssembleDDSEndpoint:
    @pytest.mark.asyncio
    async def test_assemble_dds_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/assemble-dds",
            json={
                "operator_id": "OP-001",
                "operator_name": "Acme Corp",
                "commodities": ["cocoa"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["statement_id"] == "DDS-002"

    @pytest.mark.asyncio
    async def test_assemble_dds_value_error(self, client, mock_service):
        mock_service.assemble_statement.side_effect = ValueError("Invalid")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/assemble-dds",
            json={
                "operator_id": "",
                "operator_name": "Acme Corp",
                "commodities": ["cocoa"],
            },
        )
        assert resp.status_code == 422


class TestListDDSEndpoint:
    @pytest.mark.asyncio
    async def test_list_dds_empty(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/dds-creator/dds")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_dds_with_filters(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds",
            params={"operator_id": "OP-001", "status": "draft"},
        )
        assert resp.status_code == 200
        mock_service.list_statements.assert_called_once()


class TestGetDDSEndpoint:
    @pytest.mark.asyncio
    async def test_get_dds_not_found(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/dds-creator/dds/DDS-NONEXISTENT")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_dds_found(self, client, mock_service):
        mock_service.get_statement.return_value = {"statement_id": "DDS-001"}
        resp = await client.get("/api/v1/eudr/dds-creator/dds/DDS-001")
        assert resp.status_code == 200
        assert resp.json()["statement_id"] == "DDS-001"


class TestGetDDSSummaryEndpoint:
    @pytest.mark.asyncio
    async def test_summary_not_found(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/dds-creator/dds/DDS-NONEXISTENT/summary")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_summary_found(self, client, mock_service):
        mock_service.get_statement_summary.return_value = {"statement_id": "DDS-001", "status": "draft"}
        resp = await client.get("/api/v1/eudr/dds-creator/dds/DDS-001/summary")
        assert resp.status_code == 200


class TestUpdateStatusEndpoint:
    @pytest.mark.asyncio
    async def test_update_status_success(self, client, mock_service):
        resp = await client.put(
            "/api/v1/eudr/dds-creator/dds/DDS-001/status",
            json={"status": "submitted"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_update_status_value_error(self, client, mock_service):
        mock_service.update_statement_status.side_effect = ValueError("Invalid")
        resp = await client.put(
            "/api/v1/eudr/dds-creator/dds/DDS-001/status",
            json={"status": "invalid_xyz"},
        )
        assert resp.status_code == 422


class TestWithdrawDDSEndpoint:
    @pytest.mark.asyncio
    async def test_withdraw_success(self, client, mock_service):
        resp = await client.delete("/api/v1/eudr/dds-creator/dds/DDS-001")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_withdraw_not_found(self, client, mock_service):
        mock_service.withdraw_statement.side_effect = ValueError("Not found")
        resp = await client.delete("/api/v1/eudr/dds-creator/dds/DDS-001")
        assert resp.status_code == 404


# ====================================================================
# Geolocation Endpoint Tests
# ====================================================================


class TestGeolocationEndpoints:
    @pytest.mark.asyncio
    async def test_format_geolocation_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/geolocations",
            json={
                "plot_id": "PLT-001",
                "latitude": 5.123,
                "longitude": -3.456,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_format_geolocation_invalid(self, client, mock_service):
        mock_service.format_geolocation.side_effect = ValueError("Out of bounds")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/geolocations",
            json={
                "plot_id": "PLT-001",
                "latitude": 999.0,
                "longitude": -3.456,
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_format_geolocations(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/geolocations/batch",
            json=[
                {"plot_id": "PLT-001", "latitude": 5.123, "longitude": -3.456},
                {"plot_id": "PLT-002", "latitude": 5.200, "longitude": -3.500},
            ],
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_export_geojson_success(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/geolocations/geojson",
        )
        assert resp.status_code == 200
        assert resp.json()["type"] == "FeatureCollection"

    @pytest.mark.asyncio
    async def test_export_geojson_not_found(self, client, mock_service):
        mock_service.export_geojson.side_effect = ValueError("Not found")
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/geolocations/geojson",
        )
        assert resp.status_code == 404


# ====================================================================
# Risk Data Endpoint Tests
# ====================================================================


class TestRiskEndpoints:
    @pytest.mark.asyncio
    async def test_integrate_risk_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/risk-references",
            json={
                "risk_id": "RISK-001",
                "source_agent": "EUDR-016",
                "risk_category": "country",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_integrate_risk_invalid(self, client, mock_service):
        mock_service.integrate_risk.side_effect = ValueError("Invalid")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/risk-references",
            json={
                "risk_id": "RISK-001",
                "source_agent": "EUDR-016",
                "risk_category": "country",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_integrate_risk(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/risk-references/batch",
            json=[
                {"risk_id": "R-1", "source_agent": "EUDR-016", "risk_category": "country"},
            ],
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_overall_risk(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/risk-references/overall",
        )
        assert resp.status_code == 200


# ====================================================================
# Supply Chain Endpoint Tests
# ====================================================================


class TestSupplyChainEndpoints:
    @pytest.mark.asyncio
    async def test_compile_supply_chain(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/supply-chain",
            json={
                "supply_chain_id": "SC-001",
                "commodity": "cocoa",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_supply_chain(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/supply-chain/completeness",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_countries_summary(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/supply-chain/countries",
        )
        assert resp.status_code == 200


# ====================================================================
# Compliance Endpoint Tests
# ====================================================================


class TestComplianceEndpoints:
    @pytest.mark.asyncio
    async def test_validate_dds(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/validate",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_dds_not_found(self, client, mock_service):
        mock_service.validate_statement.side_effect = ValueError("Not found")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-NONEXISTENT/validate",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_compliance_report(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/compliance",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_compliance_report_not_found(self, client, mock_service):
        mock_service.get_compliance_report.return_value = None
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/compliance",
        )
        assert resp.status_code == 404


# ====================================================================
# Document Packaging Endpoint Tests
# ====================================================================


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_add_document(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/documents",
            json={
                "document_type": "certificate_of_origin",
                "filename": "cert.pdf",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_package(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/package",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_package(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/package/validate",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_manifest(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/package/manifest",
        )
        assert resp.status_code == 200


# ====================================================================
# Signature Endpoint Tests
# ====================================================================


class TestSignatureEndpoints:
    @pytest.mark.asyncio
    async def test_apply_signature(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/sign",
            json={"signer_name": "John Smith"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_signature(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/signature/validate",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_apply_signature_invalid(self, client, mock_service):
        mock_service.apply_signature.side_effect = ValueError("Invalid signer")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/sign",
            json={"signer_name": ""},
        )
        assert resp.status_code == 422


# ====================================================================
# Amendment / Version Endpoint Tests
# ====================================================================


class TestAmendmentEndpoints:
    @pytest.mark.asyncio
    async def test_create_amendment(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/amend",
            json={
                "reason": "correction_of_error",
                "description": "Fix operator name",
                "previous_version": 1,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_create_amendment_invalid(self, client, mock_service):
        mock_service.create_amendment.side_effect = ValueError("Invalid")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/amend",
            json={
                "reason": "correction_of_error",
                "description": "Fix",
                "previous_version": 1,
            },
        )
        assert resp.status_code == 422


class TestVersionEndpoints:
    @pytest.mark.asyncio
    async def test_get_versions(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/versions",
        )
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_get_latest_version_not_found(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/versions/latest",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_latest_version_found(self, client, mock_service):
        mock_service.get_latest_version.return_value = {"version_number": 2}
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/versions/latest",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_amendments(self, client, mock_service):
        resp = await client.get(
            "/api/v1/eudr/dds-creator/dds/DDS-001/amendments",
        )
        assert resp.status_code == 200
        assert resp.json() == []


# ====================================================================
# Submission Endpoint Tests
# ====================================================================


class TestSubmitEndpoint:
    @pytest.mark.asyncio
    async def test_submit_dds_success(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/submit",
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_dds_with_body(self, client, mock_service):
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/submit",
            json={"additional_documents": [{"filename": "extra.pdf"}]},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_dds_validation_fail(self, client, mock_service):
        mock_service.submit_statement.side_effect = ValueError("Validation failed")
        resp = await client.post(
            "/api/v1/eudr/dds-creator/dds/DDS-001/submit",
        )
        assert resp.status_code == 422


# ====================================================================
# Health Endpoint Tests
# ====================================================================


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client, mock_service):
        resp = await client.get("/api/v1/eudr/dds-creator/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "GL-EUDR-DDSC-037"
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_error_fallback(self, client, mock_service):
        mock_service.health_check.side_effect = RuntimeError("Engine crash")
        resp = await client.get("/api/v1/eudr/dds-creator/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
