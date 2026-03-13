# -*- coding: utf-8 -*-
"""
Unit tests for FastAPI Router - AGENT-EUDR-040

Tests all 30+ API endpoints for authority communication management,
information request handling, inspection coordination, non-compliance
processing, appeal management, document exchange, notification routing,
multi-language template rendering, authority listing, and health checks.

70+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from greenlang.agents.eudr.authority_communication_manager.api import (
    CreateCommunicationRequest,
    RespondRequest,
    InformationRequestBody,
    ScheduleInspectionRequest,
    UpdateInspectionStatusRequest,
    RecordFindingsRequest,
    RecordViolationRequest,
    FileAppealRequest,
    AppealDecisionRequest,
    UploadDocumentRequest,
    SendNotificationRequest,
    RenderTemplateRequest,
    ErrorResponse,
    get_router,
    router,
)


# ---------------------------------------------------------------------------
# Helper: create FastAPI test app with mocked service
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    """Build a minimal FastAPI app with the authority communication router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def app():
    return _build_app()


@pytest.fixture
def mock_service():
    """Create a mock AuthorityCommunicationManagerService."""
    svc = AsyncMock()
    # Communication management
    svc.create_communication = AsyncMock(return_value={
        "communication_id": "COMM-001",
        "status": "pending",
        "priority": "normal",
    })
    svc.get_communication = AsyncMock(return_value=None)
    svc.respond_to_communication = AsyncMock(return_value={
        "communication_id": "COMM-001",
        "status": "responded",
    })
    svc.list_pending_communications = AsyncMock(return_value=[])
    svc.list_overdue_communications = AsyncMock(return_value=[])

    # Information request
    svc.handle_information_request = AsyncMock(return_value={
        "request_id": "REQ-001",
        "request_type": "dds_verification",
    })

    # Inspection
    svc.schedule_inspection = AsyncMock(return_value={
        "inspection_id": "INSP-001",
        "inspection_type": "on_the_spot",
    })

    # Non-compliance
    svc.record_violation = AsyncMock(return_value={
        "non_compliance_id": "NC-001",
        "severity": "major",
        "penalty_amount": "5000.00",
    })

    # Appeal
    svc.file_appeal = AsyncMock(return_value={
        "appeal_id": "APP-001",
        "decision": "pending",
    })

    # Engine accessor
    appeal_engine = AsyncMock()
    appeal_engine.record_decision = AsyncMock(return_value=MagicMock(
        model_dump=MagicMock(return_value={"appeal_id": "APP-001", "decision": "upheld"}),
    ))
    appeal_engine.grant_extension = AsyncMock(return_value=MagicMock(
        model_dump=MagicMock(return_value={"appeal_id": "APP-001", "extensions_granted": 1}),
    ))
    appeal_engine.withdraw_appeal = AsyncMock(return_value=MagicMock(
        model_dump=MagicMock(return_value={"appeal_id": "APP-001", "decision": "withdrawn"}),
    ))

    doc_engine = AsyncMock()
    doc_engine.get_document_metadata = AsyncMock(return_value=None)
    doc_engine.verify_integrity = AsyncMock(return_value={"valid": True})

    def _get_engine(name):
        if name == "appeal_processor":
            return appeal_engine
        if name == "document_exchange":
            return doc_engine
        return None

    svc.get_engine = MagicMock(side_effect=_get_engine)

    # Document
    svc.upload_document = AsyncMock(return_value={
        "document_id": "DOC-001",
        "document_type": "certificate",
    })

    # Notification
    svc.send_notification = AsyncMock(return_value={
        "notification_id": "NOTIF-001",
        "channel": "email",
    })

    # Template
    svc.render_template = AsyncMock(return_value={
        "subject": "Rendered Subject",
        "body": "Rendered Body",
    })
    svc.list_templates = AsyncMock(return_value=[])

    # Authorities
    svc.get_authorities = AsyncMock(return_value=[
        {"member_state": "DE", "authority_name": "BfN"},
    ])

    # Health
    svc.health_check = AsyncMock(return_value={
        "agent_id": "GL-EUDR-ACM-040",
        "status": "healthy",
        "version": "1.0.0",
    })

    return svc


@pytest_asyncio.fixture
async def client(app, mock_service):
    """Create an async httpx client patching get_service."""
    with patch(
        "greenlang.agents.eudr.authority_communication_manager.api.get_service",
        return_value=mock_service,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


PREFIX = "/api/v1/eudr/authority-communication-manager"


# ====================================================================
# Request Schema Tests
# ====================================================================


class TestRequestSchemas:
    """Test request schema construction and defaults."""

    def test_create_communication_request(self):
        req = CreateCommunicationRequest(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            member_state="DE",
            communication_type="information_request",
            subject="DDS Verification",
        )
        assert req.operator_id == "OP-001"
        assert req.priority == "normal"
        assert req.language == "en"
        assert req.body == ""
        assert req.dds_reference == ""
        assert req.document_ids == []

    def test_create_communication_required_fields(self):
        req = CreateCommunicationRequest(
            operator_id="OP-001",
            authority_id="AUTH-FR-001",
            member_state="FR",
            communication_type="general_correspondence",
            subject="Test subject",
            body="Test body",
            priority="urgent",
            language="fr",
        )
        assert req.member_state == "FR"
        assert req.language == "fr"

    def test_respond_request(self):
        req = RespondRequest(
            responder_id="OP-001",
            body="Here is our response.",
        )
        assert req.responder_id == "OP-001"
        assert req.document_ids == []

    def test_information_request_body(self):
        req = InformationRequestBody(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            request_type="dds_verification",
            items_requested=["DDS Statement", "Geolocation data"],
        )
        assert req.request_type == "dds_verification"
        assert req.priority == "normal"
        assert req.language == "en"
        assert req.dds_reference == ""
        assert req.commodity == ""

    def test_schedule_inspection_request(self):
        req = ScheduleInspectionRequest(
            operator_id="OP-001",
            authority_id="AUTH-NL-001",
            inspection_type="on_the_spot",
            scheduled_date="2026-04-01T09:00:00Z",
        )
        assert req.inspection_type == "on_the_spot"
        assert req.location == ""
        assert req.scope == ""
        assert req.inspector_name == ""

    def test_update_inspection_status_request(self):
        req = UpdateInspectionStatusRequest(new_status="in_progress")
        assert req.new_status == "in_progress"
        assert req.notes == ""

    def test_record_findings_request(self):
        req = RecordFindingsRequest(
            findings=["Finding 1", "Finding 2"],
        )
        assert len(req.findings) == 2
        assert req.corrective_actions == []

    def test_record_violation_request(self):
        req = RecordViolationRequest(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="major",
            description="Missing DDS for cocoa import batch B-2025-001",
        )
        assert req.violation_type == "missing_dds"
        assert req.corrective_deadline_days == 30
        assert req.penalty_override is None
        assert req.evidence_references == []
        assert req.corrective_actions_required == []
        assert req.commodity == ""
        assert req.dds_reference == ""

    def test_record_violation_with_penalty_override(self):
        req = RecordViolationRequest(
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            violation_type="missing_dds",
            severity="critical",
            description="Critical: Missing DDS for cocoa import batch",
            penalty_override="9999.99",
        )
        assert req.penalty_override == "9999.99"

    def test_file_appeal_request(self):
        req = FileAppealRequest(
            non_compliance_id="NC-001",
            operator_id="OP-001",
            authority_id="AUTH-DE-001",
            grounds="Geolocation data was submitted but not properly processed.",
        )
        assert req.non_compliance_id == "NC-001"
        assert req.supporting_evidence == []

    def test_appeal_decision_request(self):
        req = AppealDecisionRequest(decision="upheld")
        assert req.decision == "upheld"
        assert req.reason == ""

    def test_upload_document_request(self):
        req = UploadDocumentRequest(
            communication_id="COMM-001",
            document_type="dds_statement",
            title="DDS Statement 2026",
            uploaded_by="OP-001",
        )
        assert req.document_type == "dds_statement"
        assert req.language == "en"
        assert req.mime_type == "application/pdf"
        assert req.encrypt is None
        assert req.description == ""

    def test_send_notification_request(self):
        req = SendNotificationRequest(
            communication_id="COMM-001",
            channel="email",
            recipient_type="operator",
            recipient_id="OP-001",
        )
        assert req.channel == "email"
        assert req.recipient_address == ""
        assert req.subject == ""
        assert req.body == ""
        assert req.language == "en"

    def test_render_template_request(self):
        req = RenderTemplateRequest(template_name="information_request_response")
        assert req.template_name == "information_request_response"
        assert req.language == "en"
        assert req.variables == {}

    def test_error_response(self):
        resp = ErrorResponse(detail="Something went wrong")
        assert resp.error_code == "internal_error"
        assert resp.timestamp is None


# ====================================================================
# Router Configuration Tests
# ====================================================================


class TestRouterConfig:
    def test_router_prefix(self):
        assert router.prefix == "/api/v1/eudr/authority-communication-manager"

    def test_router_tags(self):
        assert "EUDR Authority Communication Manager" in router.tags

    def test_get_router_returns_same(self):
        r = get_router()
        assert r is router

    def test_router_has_routes(self):
        paths = [route.path for route in router.routes]
        assert len(paths) > 0

    def test_router_contains_communication(self):
        paths = [route.path for route in router.routes]
        assert any("communication" in p for p in paths)

    def test_router_contains_health(self):
        paths = [route.path for route in router.routes]
        assert any("health" in p for p in paths)

    def test_router_contains_appeal(self):
        paths = [route.path for route in router.routes]
        assert any("appeal" in p for p in paths)

    def test_router_contains_inspection(self):
        paths = [route.path for route in router.routes]
        assert any("inspection" in p for p in paths)

    def test_router_contains_document(self):
        paths = [route.path for route in router.routes]
        assert any("document" in p for p in paths)

    def test_router_contains_templates(self):
        paths = [route.path for route in router.routes]
        assert any("templates" in p for p in paths)


# ====================================================================
# Communication Endpoint Tests
# ====================================================================


class TestCreateCommunicationEndpoint:
    @pytest.mark.asyncio
    async def test_create_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/communication",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "member_state": "DE",
                "communication_type": "information_request",
                "subject": "DDS Verification Request",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["communication_id"] == "COMM-001"

    @pytest.mark.asyncio
    async def test_create_value_error(self, client, mock_service):
        mock_service.create_communication.side_effect = ValueError("Invalid type")
        resp = await client.post(
            f"{PREFIX}/communication",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "member_state": "DE",
                "communication_type": "invalid_type",
                "subject": "Test",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_internal_error(self, client, mock_service):
        mock_service.create_communication.side_effect = RuntimeError("DB down")
        resp = await client.post(
            f"{PREFIX}/communication",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "member_state": "DE",
                "communication_type": "information_request",
                "subject": "Test",
            },
        )
        assert resp.status_code == 500


class TestGetCommunicationEndpoint:
    @pytest.mark.asyncio
    async def test_get_not_found(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/communication/COMM-NONEXISTENT")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_found(self, client, mock_service):
        mock_service.get_communication.return_value = {
            "communication_id": "COMM-001",
            "status": "pending",
        }
        resp = await client.get(f"{PREFIX}/communication/COMM-001")
        assert resp.status_code == 200
        assert resp.json()["communication_id"] == "COMM-001"

    @pytest.mark.asyncio
    async def test_get_internal_error(self, client, mock_service):
        mock_service.get_communication.side_effect = RuntimeError("Error")
        resp = await client.get(f"{PREFIX}/communication/COMM-001")
        assert resp.status_code == 500


class TestRespondEndpoint:
    @pytest.mark.asyncio
    async def test_respond_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/communication/COMM-001/respond",
            json={
                "responder_id": "OP-001",
                "body": "Here is our response with supporting documents.",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_respond_value_error(self, client, mock_service):
        mock_service.respond_to_communication.side_effect = ValueError("Not found")
        resp = await client.post(
            f"{PREFIX}/communication/COMM-001/respond",
            json={
                "responder_id": "OP-001",
                "body": "Response text.",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_respond_internal_error(self, client, mock_service):
        mock_service.respond_to_communication.side_effect = RuntimeError("Error")
        resp = await client.post(
            f"{PREFIX}/communication/COMM-001/respond",
            json={
                "responder_id": "OP-001",
                "body": "Response text.",
            },
        )
        assert resp.status_code == 500


class TestListCommunicationsEndpoint:
    @pytest.mark.asyncio
    async def test_list_pending_empty(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/communications/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_pending_with_filter(self, client, mock_service):
        resp = await client.get(
            f"{PREFIX}/communications/pending",
            params={"operator_id": "OP-001"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_overdue_empty(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/communications/overdue")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_pending_internal_error(self, client, mock_service):
        mock_service.list_pending_communications.side_effect = RuntimeError("Error")
        resp = await client.get(f"{PREFIX}/communications/pending")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_list_overdue_internal_error(self, client, mock_service):
        mock_service.list_overdue_communications.side_effect = RuntimeError("Error")
        resp = await client.get(f"{PREFIX}/communications/overdue")
        assert resp.status_code == 500


# ====================================================================
# Information Request Endpoint Tests
# ====================================================================


class TestInformationRequestEndpoint:
    @pytest.mark.asyncio
    async def test_handle_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/information-request",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "request_type": "dds_verification",
                "items_requested": ["DDS Statement", "Geolocation data"],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_value_error(self, client, mock_service):
        mock_service.handle_information_request.side_effect = ValueError("Invalid type")
        resp = await client.post(
            f"{PREFIX}/information-request",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "request_type": "invalid",
                "items_requested": ["Item"],
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_handle_runtime_error(self, client, mock_service):
        mock_service.handle_information_request.side_effect = RuntimeError("Unavailable")
        resp = await client.post(
            f"{PREFIX}/information-request",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "request_type": "dds_verification",
                "items_requested": ["DDS Statement"],
            },
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_handle_internal_error(self, client, mock_service):
        mock_service.handle_information_request.side_effect = Exception("Unexpected")
        resp = await client.post(
            f"{PREFIX}/information-request",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "request_type": "dds_verification",
                "items_requested": ["DDS Statement"],
            },
        )
        assert resp.status_code == 500


# ====================================================================
# Inspection Endpoint Tests
# ====================================================================


class TestInspectionEndpoints:
    @pytest.mark.asyncio
    async def test_schedule_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/inspection",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "inspection_type": "on_the_spot",
                "scheduled_date": "2026-04-01T09:00:00Z",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_schedule_value_error(self, client, mock_service):
        mock_service.schedule_inspection.side_effect = ValueError("Invalid type")
        resp = await client.post(
            f"{PREFIX}/inspection",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "inspection_type": "invalid",
                "scheduled_date": "2026-04-01T09:00:00Z",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_schedule_runtime_error(self, client, mock_service):
        mock_service.schedule_inspection.side_effect = RuntimeError("Unavailable")
        resp = await client.post(
            f"{PREFIX}/inspection",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "inspection_type": "on_the_spot",
                "scheduled_date": "2026-04-01T09:00:00Z",
            },
        )
        assert resp.status_code == 503


# ====================================================================
# Non-Compliance Endpoint Tests
# ====================================================================


class TestNonComplianceEndpoints:
    @pytest.mark.asyncio
    async def test_record_violation_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/non-compliance",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "violation_type": "missing_dds",
                "severity": "major",
                "description": "Missing DDS for cocoa import batch B-2025-001",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_record_violation_with_penalty_override(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/non-compliance",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "violation_type": "false_information",
                "severity": "critical",
                "description": "Deliberate falsification of origin data in DDS.",
                "penalty_override": "50000.00",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_record_violation_value_error(self, client, mock_service):
        mock_service.record_violation.side_effect = ValueError("Invalid violation type")
        resp = await client.post(
            f"{PREFIX}/non-compliance",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "violation_type": "invalid",
                "severity": "minor",
                "description": "Test violation with invalid type to check error.",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_record_violation_runtime_error(self, client, mock_service):
        mock_service.record_violation.side_effect = RuntimeError("Engine unavailable")
        resp = await client.post(
            f"{PREFIX}/non-compliance",
            json={
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "violation_type": "missing_dds",
                "severity": "minor",
                "description": "Test violation while engine is unavailable for processing.",
            },
        )
        assert resp.status_code == 503


# ====================================================================
# Appeal Endpoint Tests
# ====================================================================


class TestAppealEndpoints:
    @pytest.mark.asyncio
    async def test_file_appeal_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/appeal",
            json={
                "non_compliance_id": "NC-001",
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "grounds": "Geolocation data was submitted but not properly processed by the portal.",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_file_appeal_value_error(self, client, mock_service):
        mock_service.file_appeal.side_effect = ValueError("Grounds too short")
        resp = await client.post(
            f"{PREFIX}/appeal",
            json={
                "non_compliance_id": "NC-001",
                "operator_id": "OP-001",
                "authority_id": "AUTH-DE-001",
                "grounds": "Geolocation data was submitted but not properly processed.",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_record_decision_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/appeal/APP-001/decision",
            json={"decision": "upheld", "reason": "Evidence substantiated."},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_record_decision_value_error(self, client, mock_service):
        appeal_engine = mock_service.get_engine("appeal_processor")
        appeal_engine.record_decision.side_effect = ValueError("Invalid decision")
        resp = await client.post(
            f"{PREFIX}/appeal/APP-001/decision",
            json={"decision": "invalid_decision", "reason": "Test."},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_grant_extension_success(self, client, mock_service):
        resp = await client.post(f"{PREFIX}/appeal/APP-001/extension")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_grant_extension_with_days(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/appeal/APP-001/extension",
            params={"additional_days": 15},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_grant_extension_value_error(self, client, mock_service):
        appeal_engine = mock_service.get_engine("appeal_processor")
        appeal_engine.grant_extension.side_effect = ValueError("Max extensions exceeded")
        resp = await client.post(f"{PREFIX}/appeal/APP-001/extension")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_withdraw_appeal_success(self, client, mock_service):
        resp = await client.post(f"{PREFIX}/appeal/APP-001/withdraw")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_withdraw_appeal_value_error(self, client, mock_service):
        appeal_engine = mock_service.get_engine("appeal_processor")
        appeal_engine.withdraw_appeal.side_effect = ValueError("Not found")
        resp = await client.post(f"{PREFIX}/appeal/APP-NONEXISTENT/withdraw")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_appeal_engine_unavailable(self, client, mock_service):
        mock_service.get_engine = MagicMock(return_value=None)
        resp = await client.post(
            f"{PREFIX}/appeal/APP-001/decision",
            json={"decision": "upheld"},
        )
        assert resp.status_code == 503


# ====================================================================
# Document Endpoint Tests
# ====================================================================


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_upload_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/document/upload",
            json={
                "communication_id": "COMM-001",
                "document_type": "dds_statement",
                "title": "DDS Statement 2026",
                "uploaded_by": "OP-001",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_value_error(self, client, mock_service):
        mock_service.upload_document.side_effect = ValueError("Invalid type")
        resp = await client.post(
            f"{PREFIX}/document/upload",
            json={
                "communication_id": "COMM-001",
                "document_type": "invalid",
                "title": "Test",
                "uploaded_by": "OP-001",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/document/DOC-NONEXISTENT/metadata")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_metadata_found(self, client, mock_service):
        doc_engine = mock_service.get_engine("document_exchange")
        doc_engine.get_document_metadata.return_value = MagicMock(
            model_dump=MagicMock(return_value={"document_id": "DOC-001", "title": "Test"}),
        )
        resp = await client.get(f"{PREFIX}/document/DOC-001/metadata")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_integrity(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/document/DOC-001/verify")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_verify_integrity_not_found(self, client, mock_service):
        doc_engine = mock_service.get_engine("document_exchange")
        doc_engine.verify_integrity.side_effect = ValueError("Not found")
        resp = await client.get(f"{PREFIX}/document/DOC-NONEXISTENT/verify")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_document_engine_unavailable(self, client, mock_service):
        mock_service.get_engine = MagicMock(return_value=None)
        resp = await client.get(f"{PREFIX}/document/DOC-001/metadata")
        assert resp.status_code == 503


# ====================================================================
# Notification Endpoint Tests
# ====================================================================


class TestNotificationEndpoints:
    @pytest.mark.asyncio
    async def test_send_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/notification/send",
            json={
                "communication_id": "COMM-001",
                "channel": "email",
                "recipient_type": "operator",
                "recipient_id": "OP-001",
                "recipient_address": "compliance@acme.com",
                "subject": "Information Request",
                "body": "Please provide documents.",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_send_value_error(self, client, mock_service):
        mock_service.send_notification.side_effect = ValueError("Invalid channel")
        resp = await client.post(
            f"{PREFIX}/notification/send",
            json={
                "communication_id": "COMM-001",
                "channel": "invalid_channel",
                "recipient_type": "operator",
                "recipient_id": "OP-001",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_send_runtime_error(self, client, mock_service):
        mock_service.send_notification.side_effect = RuntimeError("Engine unavailable")
        resp = await client.post(
            f"{PREFIX}/notification/send",
            json={
                "communication_id": "COMM-001",
                "channel": "email",
                "recipient_type": "operator",
                "recipient_id": "OP-001",
            },
        )
        assert resp.status_code == 503


# ====================================================================
# Template Endpoint Tests
# ====================================================================


class TestTemplateEndpoints:
    @pytest.mark.asyncio
    async def test_list_templates_empty(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/templates")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_templates_with_language(self, client, mock_service):
        resp = await client.get(
            f"{PREFIX}/templates",
            params={"language": "en"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_templates_with_type(self, client, mock_service):
        resp = await client.get(
            f"{PREFIX}/templates",
            params={"communication_type": "information_request"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_templates_by_language(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/templates/en")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_templates_by_language_de(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/templates/de")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_render_template_success(self, client, mock_service):
        resp = await client.post(
            f"{PREFIX}/templates/render",
            json={
                "template_name": "information_request_response",
                "language": "en",
                "variables": {
                    "operator_name": "Acme Corp",
                    "reference_number": "REF-001",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "subject" in data
        assert "body" in data

    @pytest.mark.asyncio
    async def test_render_template_not_found(self, client, mock_service):
        mock_service.render_template.side_effect = ValueError("Template not found")
        resp = await client.post(
            f"{PREFIX}/templates/render",
            json={
                "template_name": "nonexistent",
                "variables": {},
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_render_template_runtime_error(self, client, mock_service):
        mock_service.render_template.side_effect = RuntimeError("Engine unavailable")
        resp = await client.post(
            f"{PREFIX}/templates/render",
            json={
                "template_name": "test",
                "variables": {},
            },
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_list_templates_internal_error(self, client, mock_service):
        mock_service.list_templates.side_effect = RuntimeError("Error")
        resp = await client.get(f"{PREFIX}/templates")
        assert resp.status_code == 500


# ====================================================================
# Authority Endpoint Tests
# ====================================================================


class TestAuthorityEndpoints:
    @pytest.mark.asyncio
    async def test_list_authorities(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/authorities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_list_authorities_by_state(self, client, mock_service):
        resp = await client.get(
            f"{PREFIX}/authorities",
            params={"member_state": "DE"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_list_authorities_internal_error(self, client, mock_service):
        mock_service.get_authorities.side_effect = RuntimeError("Error")
        resp = await client.get(f"{PREFIX}/authorities")
        assert resp.status_code == 500


# ====================================================================
# Health Endpoint Tests
# ====================================================================


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client, mock_service):
        resp = await client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "GL-EUDR-ACM-040"
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_error_fallback(self, client, mock_service):
        mock_service.health_check.side_effect = RuntimeError("Engine crash")
        resp = await client.get(f"{PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["agent_id"] == "GL-EUDR-ACM-040"
