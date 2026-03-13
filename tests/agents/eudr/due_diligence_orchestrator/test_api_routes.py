# -*- coding: utf-8 -*-
"""
Unit tests for API Routes - AGENT-EUDR-026 Due Diligence Orchestrator

Tests all 9 route modules (workflow, execution, status, quality gate,
checkpoint, template, package, monitoring, dependencies) by directly
invoking route handler functions with mocked service dependencies.

Test count: 101+ tests across 10 test classes.
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from fastapi import HTTPException

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    VERSION,
    ALL_EUDR_AGENTS,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    PHASE_3_AGENTS,
    AGENT_NAMES,
    SUPPORTED_COMMODITIES,
    AgentExecutionRecord,
    AgentExecutionStatus,
    AgentNode,
    CircuitBreakerRecord,
    CircuitBreakerState,
    CreateWorkflowRequest,
    DDSField,
    DDSSection,
    DeadLetterEntry,
    DueDiligencePackage,
    DueDiligencePhase,
    EUDRCommodity,
    ErrorClassification,
    GeneratePackageRequest,
    PackageGenerationResponse,
    QualityGateCheck,
    QualityGateEvaluation,
    QualityGateId,
    QualityGateResponse,
    QualityGateResultEnum,
    WorkflowCheckpoint,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowProgressResponse,
    WorkflowState,
    WorkflowStatus,
    WorkflowStatusResponse,
    WorkflowType,
    _utcnow,
)
from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    PaginationParams,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_auth_user(
    user_id: str = "user-001",
    tenant_id: str = "tenant-001",
    operator_id: str = "op-001",
    roles: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
) -> AuthUser:
    """Create an AuthUser for testing."""
    return AuthUser(
        user_id=user_id,
        email=f"{user_id}@test.com",
        tenant_id=tenant_id,
        operator_id=operator_id,
        roles=roles if roles is not None else ["operator"],
        permissions=permissions if permissions is not None else ["eudr-ddo:*"],
    )


def _make_request_mock() -> MagicMock:
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.url.path = "/api/v1/ddo/workflows"
    request.state = MagicMock()
    return request


def _make_workflow_state(
    workflow_id: str = "wf-001",
    status: WorkflowStatus = WorkflowStatus.CREATED,
    commodity: EUDRCommodity = EUDRCommodity.COCOA,
    workflow_type: WorkflowType = WorkflowType.STANDARD,
    progress_pct: Decimal = Decimal("0"),
    eta_seconds: Optional[int] = None,
    agent_executions: Optional[Dict[str, AgentExecutionRecord]] = None,
    quality_gates: Optional[Dict[str, QualityGateEvaluation]] = None,
    checkpoints: Optional[List[WorkflowCheckpoint]] = None,
) -> WorkflowState:
    """Create a WorkflowState for testing."""
    return WorkflowState(
        workflow_id=workflow_id,
        definition_id="def-001",
        status=status,
        workflow_type=workflow_type,
        commodity=commodity,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        operator_id="op-001",
        operator_name="Test Operator GmbH",
        country_codes=["GH", "CI"],
        progress_pct=progress_pct,
        eta_seconds=eta_seconds,
        agent_executions=agent_executions or {},
        quality_gates=quality_gates or {},
        checkpoints=checkpoints or [],
    )


def _make_status_response(
    workflow_id: str = "wf-001",
    status: WorkflowStatus = WorkflowStatus.CREATED,
) -> WorkflowStatusResponse:
    """Create a WorkflowStatusResponse for testing."""
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=status,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        progress_pct=Decimal("0"),
        agents_total=25,
        agents_completed=0,
        agents_running=0,
        agents_failed=0,
    )


def _make_progress_response(
    workflow_id: str = "wf-001",
) -> WorkflowProgressResponse:
    """Create a WorkflowProgressResponse for testing."""
    agent_statuses = {
        f"EUDR-{i:03d}": "completed" for i in range(1, 8)
    }
    agent_statuses.update({
        f"EUDR-{i:03d}": "running" for i in range(8, 11)
    })
    agent_statuses.update({
        f"EUDR-{i:03d}": "pending" for i in range(11, 26)
    })
    return WorkflowProgressResponse(
        workflow_id=workflow_id,
        status=WorkflowStatus.RUNNING,
        current_phase=DueDiligencePhase.INFORMATION_GATHERING,
        progress_pct=Decimal("30"),
        agent_statuses=agent_statuses,
    )


def _make_workflow_definition(
    definition_id: str = "def-001",
    commodity: EUDRCommodity = EUDRCommodity.COCOA,
) -> WorkflowDefinition:
    """Create a WorkflowDefinition for testing."""
    nodes = [
        AgentNode(
            agent_id="EUDR-001",
            name="Supply Chain Mapping Master",
            phase=DueDiligencePhase.INFORMATION_GATHERING,
        ),
        AgentNode(
            agent_id="EUDR-002",
            name="Geolocation Verification",
            phase=DueDiligencePhase.INFORMATION_GATHERING,
        ),
    ]
    edges = [WorkflowEdge(source="EUDR-001", target="EUDR-002")]
    return WorkflowDefinition(
        definition_id=definition_id,
        name="Test Workflow",
        workflow_type=WorkflowType.STANDARD,
        commodity=commodity,
        nodes=nodes,
        edges=edges,
        quality_gates=["QG-1", "QG-2", "QG-3"],
    )


def _make_quality_gate_evaluation(
    gate_id: QualityGateId = QualityGateId.QG1,
    result: QualityGateResultEnum = QualityGateResultEnum.PASSED,
    checks: Optional[List[QualityGateCheck]] = None,
) -> QualityGateEvaluation:
    """Create a QualityGateEvaluation for testing."""
    default_checks = checks or [
        QualityGateCheck(
            name="Coverage Check",
            weight=Decimal("0.50"),
            measured_value=Decimal("95"),
            threshold=Decimal("90"),
            passed=True,
            source_agents=["EUDR-001"],
        ),
        QualityGateCheck(
            name="Completeness Check",
            weight=Decimal("0.50"),
            measured_value=Decimal("92"),
            threshold=Decimal("90"),
            passed=True,
            source_agents=["EUDR-002"],
        ),
    ]
    return QualityGateEvaluation(
        workflow_id="wf-001",
        gate_id=gate_id,
        phase_from=DueDiligencePhase.INFORMATION_GATHERING,
        phase_to=DueDiligencePhase.RISK_ASSESSMENT,
        result=result,
        weighted_score=Decimal("0.93"),
        threshold=Decimal("0.90"),
        checks=default_checks,
    )


def _make_checkpoint(
    checkpoint_id: str = "cp-001",
    workflow_id: str = "wf-001",
    sequence_number: int = 1,
    phase: DueDiligencePhase = DueDiligencePhase.INFORMATION_GATHERING,
) -> WorkflowCheckpoint:
    """Create a WorkflowCheckpoint for testing."""
    return WorkflowCheckpoint(
        checkpoint_id=checkpoint_id,
        workflow_id=workflow_id,
        sequence_number=sequence_number,
        phase=phase,
        agent_id="EUDR-001",
        agent_statuses={"EUDR-001": "completed"},
        cumulative_provenance_hash=hashlib.sha256(b"test").hexdigest(),
    )


def _make_dds_package(
    package_id: str = "pkg-001",
    workflow_id: str = "wf-005",
) -> DueDiligencePackage:
    """Create a DueDiligencePackage for testing."""
    return DueDiligencePackage(
        package_id=package_id,
        workflow_id=workflow_id,
        commodity=EUDRCommodity.COCOA,
        workflow_type=WorkflowType.STANDARD,
        operator_id="op-001",
        operator_name="Test Operator GmbH",
        total_agents_executed=25,
        total_duration_ms=Decimal("240000"),
        language="en",
        integrity_hash=hashlib.sha256(b"package").hexdigest(),
        download_urls={"json": "https://s3.example.com/pkg-001.json"},
        sections=[
            DDSSection(
                section_number=1,
                title="Operator Information",
                completeness_pct=Decimal("100"),
                fields=[
                    DDSField(
                        article_ref="12(2)(a)",
                        field_name="operator_name",
                        value="Test Operator GmbH",
                    ),
                ],
            ),
            DDSSection(
                section_number=2,
                title="Product Description",
                completeness_pct=Decimal("100"),
                fields=[
                    DDSField(
                        article_ref="12(2)(b)",
                        field_name="product_description",
                        value="Cocoa beans",
                    ),
                ],
            ),
        ],
    )


def _make_dead_letter_entry(
    dlq_id: str = "dlq-001",
    agent_id: str = "EUDR-003",
    resolved: bool = False,
) -> MagicMock:
    """Create a mock dead letter entry for testing."""
    entry = MagicMock()
    entry.dlq_id = dlq_id
    entry.workflow_id = "wf-001"
    entry.agent_id = agent_id
    entry.error_type = "ConnectionError"
    entry.error_message = "Satellite provider unavailable"
    entry.attempt_count = 3
    entry.resolved = resolved
    entry.created_at = datetime.now(timezone.utc)
    return entry


# ---------------------------------------------------------------------------
# Test: Workflow CRUD Routes
# ---------------------------------------------------------------------------


class TestWorkflowRoutes:
    """Test workflow CRUD routes (6 endpoints, ~15 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for all workflow route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_create_workflow_success(self):
        """POST /workflows -- 201 on valid creation request."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            create_workflow,
        )

        body = CreateWorkflowRequest(
            workflow_type=WorkflowType.STANDARD,
            commodity=EUDRCommodity.COCOA,
            operator_id="op-001",
            operator_name="Test Operator",
            request_id="req-001",
        )
        expected = _make_status_response()
        self.mock_service.create_workflow.return_value = expected

        result = await create_workflow(
            request=self.request, body=body, user=self.user, _rate=self.user,
        )

        assert result.workflow_id == expected.workflow_id
        assert result.status == WorkflowStatus.CREATED
        self.mock_service.create_workflow.assert_called_once_with(body)

    @pytest.mark.asyncio
    async def test_create_workflow_invalid_raises_400(self):
        """POST /workflows -- 400 on ValueError from service."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            create_workflow,
        )

        body = CreateWorkflowRequest(
            workflow_type=WorkflowType.STANDARD,
            commodity=EUDRCommodity.COCOA,
            operator_id="op-001",
        )
        self.mock_service.create_workflow.side_effect = ValueError("Invalid params")

        with pytest.raises(HTTPException) as exc_info:
            await create_workflow(
                request=self.request, body=body, user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    async def test_create_workflow_all_commodities(self, commodity):
        """POST /workflows -- accepts all 7 EUDR commodities."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            create_workflow,
        )

        body = CreateWorkflowRequest(
            workflow_type=WorkflowType.STANDARD,
            commodity=commodity,
            operator_id="op-001",
        )
        self.mock_service.create_workflow.return_value = _make_status_response()

        result = await create_workflow(
            request=self.request, body=body, user=self.user, _rate=self.user,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_list_workflows_success(self):
        """GET /workflows -- 200 with paginated list."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            list_workflows,
        )

        state = _make_workflow_state()
        self.mock_service._state_manager.list_workflows.return_value = [state]
        self.mock_service._state_manager.count_workflows.return_value = 1
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_workflows(
            request=self.request,
            user=self.user,
            pagination=pagination,
            commodity=None,
            workflow_type=None,
            status_filter=None,
            _rate=self.user,
        )

        assert "workflows" in result
        assert "meta" in result
        assert result["meta"]["total"] == 1
        assert len(result["workflows"]) == 1

    @pytest.mark.asyncio
    async def test_list_workflows_with_commodity_filter(self):
        """GET /workflows?commodity=cocoa -- filters by commodity."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            list_workflows,
        )

        self.mock_service._state_manager.list_workflows.return_value = []
        self.mock_service._state_manager.count_workflows.return_value = 0
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_workflows(
            request=self.request,
            user=self.user,
            pagination=pagination,
            commodity="cocoa",
            workflow_type=None,
            status_filter=None,
            _rate=self.user,
        )

        assert result["meta"]["total"] == 0
        self.mock_service._state_manager.list_workflows.assert_called_once_with(
            tenant_id="tenant-001",
            commodity="cocoa",
            workflow_type=None,
            status_filter=None,
            limit=50,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_get_workflow_success(self):
        """GET /workflows/{id} -- 200 returns workflow details."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            get_workflow,
        )

        expected = _make_status_response()
        self.mock_service.get_workflow_status.return_value = expected

        result = await get_workflow(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result.workflow_id == "wf-001"

    @pytest.mark.asyncio
    async def test_get_workflow_not_found_404(self):
        """GET /workflows/{id} -- 404 when workflow does not exist."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            get_workflow,
        )

        self.mock_service.get_workflow_status.side_effect = ValueError("Not found")

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_workflow_success(self):
        """DELETE /workflows/{id} -- 200 archives a non-running workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            delete_workflow,
        )

        state = _make_workflow_state(status=WorkflowStatus.COMPLETED)
        self.mock_service._state_manager.get_state.return_value = state
        self.mock_service._state_manager.archive_workflow.return_value = None

        result = await delete_workflow(
            request=self.request, workflow_id="wf-001", reason="Test cleanup",
            user=self.user, _rate=self.user,
        )

        assert result["status"] == "archived"
        assert result["workflow_id"] == "wf-001"
        assert result["archived_by"] == "user-001"

    @pytest.mark.asyncio
    async def test_delete_running_workflow_raises_400(self):
        """DELETE /workflows/{id} -- 400 when workflow is running."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            delete_workflow,
        )

        state = _make_workflow_state(status=WorkflowStatus.RUNNING)
        self.mock_service._state_manager.get_state.return_value = state

        with pytest.raises(HTTPException) as exc_info:
            await delete_workflow(
                request=self.request, workflow_id="wf-001", reason=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_workflow_not_found_404(self):
        """DELETE /workflows/{id} -- 404 when workflow does not exist."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            delete_workflow,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await delete_workflow(
                request=self.request, workflow_id="wf-nonexistent", reason=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_validate_workflow_success(self):
        """POST /workflows/{id}/validate -- 200 with valid DAG."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            validate_workflow,
        )

        state = _make_workflow_state()
        self.mock_service._state_manager.get_state.return_value = state

        definition = _make_workflow_definition()
        self.mock_service._workflow_engine.get_definition.return_value = definition
        self.mock_service._workflow_engine.validate_definition.return_value = (True, [])
        self.mock_service._workflow_engine.get_execution_layers.return_value = [
            ["EUDR-001"], ["EUDR-002"],
        ]

        result = await validate_workflow(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["dag_metadata"]["total_nodes"] == 2

    @pytest.mark.asyncio
    async def test_validate_workflow_not_found_404(self):
        """POST /workflows/{id}/validate -- 404 when workflow is missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            validate_workflow,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await validate_workflow(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_clone_workflow_success(self):
        """POST /workflows/{id}/clone -- 201 clones existing workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            clone_workflow,
        )

        state = _make_workflow_state()
        self.mock_service._state_manager.get_state.return_value = state
        expected = _make_status_response(workflow_id="wf-clone-001")
        self.mock_service.create_workflow.return_value = expected

        result = await clone_workflow(
            request=self.request, workflow_id="wf-001", name="Clone Test",
            commodity=None, user=self.user, _rate=self.user,
        )

        assert result.workflow_id == "wf-clone-001"

    @pytest.mark.asyncio
    async def test_clone_workflow_not_found_404(self):
        """POST /workflows/{id}/clone -- 404 when source workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            clone_workflow,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await clone_workflow(
                request=self.request, workflow_id="wf-nonexistent",
                name=None, commodity=None, user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test: Execution Control Routes
# ---------------------------------------------------------------------------


class TestExecutionRoutes:
    """Test execution control routes (5 endpoints, ~12 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for execution route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_start_workflow_success(self):
        """POST /workflows/{id}/start -- 200 starts a CREATED workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            start_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.CREATED)
        self.mock_service.get_workflow_status.return_value = current
        started = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.start_workflow.return_value = started

        result = await start_workflow(
            request=self.request, workflow_id="wf-001", body=None,
            user=self.user, _rate=self.user,
        )

        assert result.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_workflow_not_found_404(self):
        """POST /workflows/{id}/start -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            start_workflow,
        )

        self.mock_service.get_workflow_status.side_effect = ValueError("Not found")

        with pytest.raises(HTTPException) as exc_info:
            await start_workflow(
                request=self.request, workflow_id="wf-nonexistent", body=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_start_running_workflow_raises_409(self):
        """POST /workflows/{id}/start -- 409 when workflow already running."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            start_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.get_workflow_status.return_value = current

        with pytest.raises(HTTPException) as exc_info:
            await start_workflow(
                request=self.request, workflow_id="wf-001", body=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_pause_workflow_success(self):
        """POST /workflows/{id}/pause -- 200 pauses a RUNNING workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            pause_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.get_workflow_status.return_value = current
        paused = _make_status_response(status=WorkflowStatus.PAUSED)
        self.mock_service.pause_workflow.return_value = paused

        result = await pause_workflow(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result.status == WorkflowStatus.PAUSED

    @pytest.mark.asyncio
    async def test_pause_non_running_raises_409(self):
        """POST /workflows/{id}/pause -- 409 when workflow is not running."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            pause_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.PAUSED)
        self.mock_service.get_workflow_status.return_value = current

        with pytest.raises(HTTPException) as exc_info:
            await pause_workflow(
                request=self.request, workflow_id="wf-001",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_resume_workflow_success(self):
        """POST /workflows/{id}/resume -- 200 resumes a PAUSED workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            resume_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.PAUSED)
        self.mock_service.get_workflow_status.return_value = current
        resumed = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.resume_workflow.return_value = resumed

        result = await resume_workflow(
            request=self.request, workflow_id="wf-001", body=None,
            user=self.user, _rate=self.user,
        )

        assert result.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_status", [
        WorkflowStatus.CREATED, WorkflowStatus.RUNNING,
        WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED,
    ])
    async def test_resume_invalid_status_raises_409(self, invalid_status):
        """POST /workflows/{id}/resume -- 409 for non-resumable states."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            resume_workflow,
        )

        current = _make_status_response(status=invalid_status)
        self.mock_service.get_workflow_status.return_value = current

        with pytest.raises(HTTPException) as exc_info:
            await resume_workflow(
                request=self.request, workflow_id="wf-001", body=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_workflow_success(self):
        """POST /workflows/{id}/cancel -- 200 cancels a running workflow."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            cancel_workflow,
        )

        current = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.get_workflow_status.return_value = current
        cancelled = _make_status_response(status=WorkflowStatus.CANCELLED)
        self.mock_service.cancel_workflow.return_value = cancelled

        result = await cancel_workflow(
            request=self.request, workflow_id="wf-001", body=None,
            user=self.user, _rate=self.user,
        )

        assert result.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    @pytest.mark.parametrize("terminal_status", [
        WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED,
        WorkflowStatus.TERMINATED,
    ])
    async def test_cancel_terminal_raises_409(self, terminal_status):
        """POST /workflows/{id}/cancel -- 409 for terminal states."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            cancel_workflow,
        )

        current = _make_status_response(status=terminal_status)
        self.mock_service.get_workflow_status.return_value = current

        with pytest.raises(HTTPException) as exc_info:
            await cancel_workflow(
                request=self.request, workflow_id="wf-001", body=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_rollback_workflow_success(self):
        """POST /workflows/{id}/rollback -- 200 on valid rollback."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            rollback_workflow,
            RollbackRequest,
        )

        state = _make_workflow_state(status=WorkflowStatus.PAUSED)
        self.mock_service._state_manager.get_state.return_value = state
        self.mock_service._state_manager.rollback_to_checkpoint.return_value = {
            "phase": "information_gathering",
            "sequence_number": 5,
        }

        body = RollbackRequest(
            checkpoint_id="cp-001",
            reason="Reverting to known good state",
        )

        result = await rollback_workflow(
            request=self.request, workflow_id="wf-001", body=body,
            user=self.user, _rate=self.user,
        )

        assert result["status"] == "paused"
        assert result["rolled_back_to_checkpoint"] == "cp-001"

    @pytest.mark.asyncio
    async def test_rollback_workflow_not_found_404(self):
        """POST /workflows/{id}/rollback -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.execution_routes import (
            rollback_workflow,
            RollbackRequest,
        )

        self.mock_service._state_manager.get_state.return_value = None
        body = RollbackRequest(checkpoint_id="cp-001")

        with pytest.raises(HTTPException) as exc_info:
            await rollback_workflow(
                request=self.request, workflow_id="wf-nonexistent", body=body,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test: Status Monitoring Routes
# ---------------------------------------------------------------------------


class TestStatusRoutes:
    """Test status monitoring routes (4 endpoints, ~10 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for status route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """GET /workflows/{id}/status -- 200 returns workflow status."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_status,
        )

        expected = _make_status_response(status=WorkflowStatus.RUNNING)
        self.mock_service.get_workflow_status.return_value = expected

        result = await get_workflow_status(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_status_not_found_404(self):
        """GET /workflows/{id}/status -- 404 when missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_status,
        )

        self.mock_service.get_workflow_status.side_effect = ValueError("Not found")

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow_status(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_progress_success(self):
        """GET /workflows/{id}/progress -- 200 returns per-agent progress."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_progress,
        )

        expected = _make_progress_response()
        self.mock_service.get_workflow_progress.return_value = expected

        result = await get_workflow_progress(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result.workflow_id == "wf-001"
        assert result.status == WorkflowStatus.RUNNING
        assert result.progress_pct == Decimal("30")
        completed_count = sum(1 for s in result.agent_statuses.values() if s == "completed")
        assert completed_count == 7

    @pytest.mark.asyncio
    async def test_get_progress_not_found_404(self):
        """GET /workflows/{id}/progress -- 404 when missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_progress,
        )

        self.mock_service.get_workflow_progress.side_effect = ValueError("Not found")

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow_progress(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_phase_status_success(self):
        """GET /workflows/{id}/phase-status -- 200 returns per-phase data."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_phase_status,
        )

        state = _make_workflow_state(status=WorkflowStatus.RUNNING)
        # Add some completed agents in Phase 1
        for agent_id in PHASE_1_AGENTS[:5]:
            state.agent_executions[agent_id] = AgentExecutionRecord(
                workflow_id="wf-001",
                agent_id=agent_id,
                status=AgentExecutionStatus.COMPLETED,
            )
        self.mock_service._state_manager.get_state.return_value = state

        result = await get_phase_status(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert "phases" in result
        assert "information_gathering" in result["phases"]
        assert "risk_assessment" in result["phases"]
        assert "risk_mitigation" in result["phases"]
        ig = result["phases"]["information_gathering"]
        assert ig["agents_completed"] == 5

    @pytest.mark.asyncio
    async def test_get_phase_status_not_found_404(self):
        """GET /workflows/{id}/phase-status -- 404 when missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_phase_status,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_phase_status(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_eta_success(self):
        """GET /workflows/{id}/eta -- 200 returns ETA data."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_eta,
        )

        state = _make_workflow_state(
            status=WorkflowStatus.RUNNING,
            progress_pct=Decimal("40"),
            eta_seconds=120,
        )
        state.agent_executions["EUDR-001"] = AgentExecutionRecord(
            workflow_id="wf-001", agent_id="EUDR-001",
            status=AgentExecutionStatus.COMPLETED,
        )
        self.mock_service._state_manager.get_state.return_value = state

        result = await get_workflow_eta(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result["eta_seconds"] == 120
        assert result["agents_completed"] == 1
        assert result["estimated_completion"] is not None

    @pytest.mark.asyncio
    async def test_get_eta_zero_when_complete(self):
        """GET /workflows/{id}/eta -- eta_seconds=0 for completed workflows."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_eta,
        )

        state = _make_workflow_state(
            status=WorkflowStatus.COMPLETED,
            progress_pct=Decimal("100"),
            eta_seconds=0,
        )
        self.mock_service._state_manager.get_state.return_value = state

        result = await get_workflow_eta(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert result["eta_seconds"] == 0
        assert result["estimated_completion"] is None

    @pytest.mark.asyncio
    async def test_get_eta_not_found_404(self):
        """GET /workflows/{id}/eta -- 404 when missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.status_routes import (
            get_workflow_eta,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_workflow_eta(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test: Quality Gate Routes
# ---------------------------------------------------------------------------


class TestQualityGateRoutes:
    """Test quality gate routes (3 endpoints, ~8 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for quality gate route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_get_quality_gates_success(self):
        """GET /workflows/{id}/gates -- 200 returns all gate results."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            get_quality_gates,
        )

        qg1 = _make_quality_gate_evaluation(QualityGateId.QG1, QualityGateResultEnum.PASSED)
        state = _make_workflow_state(quality_gates={"QG-1": qg1})
        self.mock_service._state_manager.get_state.return_value = state

        result = await get_quality_gates(
            request=self.request, workflow_id="wf-001",
            user=self.user, _rate=self.user,
        )

        assert "gates" in result
        assert result["gates_total"] == 3
        assert result["gates_evaluated"] == 1
        assert result["gates"]["QG-1"]["result"] == "passed"

    @pytest.mark.asyncio
    async def test_get_quality_gates_not_found_404(self):
        """GET /workflows/{id}/gates -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            get_quality_gates,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_quality_gates(
                request=self.request, workflow_id="wf-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_override_quality_gate_success(self):
        """POST /workflows/{id}/gates/{gate_id}/override -- 200 on valid override."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            override_quality_gate,
            GateOverrideRequest,
        )

        qg1 = _make_quality_gate_evaluation(QualityGateId.QG1, QualityGateResultEnum.FAILED)
        state = _make_workflow_state(quality_gates={"QG-1": qg1})
        self.mock_service._state_manager.get_state.return_value = state

        overridden_eval = QualityGateEvaluation(
            workflow_id="wf-001",
            gate_id=QualityGateId.QG1,
            phase_from=DueDiligencePhase.INFORMATION_GATHERING,
            phase_to=DueDiligencePhase.RISK_ASSESSMENT,
            result=QualityGateResultEnum.OVERRIDDEN,
            weighted_score=Decimal("0.93"),
            threshold=Decimal("0.90"),
            override_justification="Approved by compliance officer",
            override_by="user-001",
        )
        expected_response = QualityGateResponse(
            evaluation=overridden_eval,
        )
        self.mock_service.override_quality_gate.return_value = expected_response

        body = GateOverrideRequest(
            justification="Approved by compliance officer - sufficient alternative evidence available",
        )

        result = await override_quality_gate(
            request=self.request, workflow_id="wf-001", gate_id="QG-1",
            body=body, user=self.user, _rate=self.user,
        )

        assert result.evaluation.result == QualityGateResultEnum.OVERRIDDEN

    @pytest.mark.asyncio
    async def test_override_invalid_gate_id_raises_400(self):
        """POST /workflows/{id}/gates/{gate_id}/override -- 400 on invalid gate ID."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            override_quality_gate,
            GateOverrideRequest,
        )

        body = GateOverrideRequest(
            justification="This is a valid justification for overriding the gate",
        )

        with pytest.raises(HTTPException) as exc_info:
            await override_quality_gate(
                request=self.request, workflow_id="wf-001", gate_id="QG-INVALID",
                body=body, user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_override_already_passed_raises_409(self):
        """POST /workflows/{id}/gates/{gate_id}/override -- 409 when gate already passed."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            override_quality_gate,
            GateOverrideRequest,
        )

        qg1 = _make_quality_gate_evaluation(QualityGateId.QG1, QualityGateResultEnum.PASSED)
        state = _make_workflow_state(quality_gates={"QG-1": qg1})
        self.mock_service._state_manager.get_state.return_value = state

        body = GateOverrideRequest(
            justification="This override is not needed because the gate already passed",
        )

        with pytest.raises(HTTPException) as exc_info:
            await override_quality_gate(
                request=self.request, workflow_id="wf-001", gate_id="QG-1",
                body=body, user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_get_gate_details_success(self):
        """GET /workflows/{id}/gates/{gate_id}/details -- 200 with check details."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            get_gate_details,
        )

        qg1 = _make_quality_gate_evaluation(QualityGateId.QG1, QualityGateResultEnum.PASSED)
        state = _make_workflow_state(quality_gates={"QG-1": qg1})
        self.mock_service._state_manager.get_state.return_value = state

        result = await get_gate_details(
            request=self.request, workflow_id="wf-001", gate_id="QG-1",
            user=self.user, _rate=self.user,
        )

        assert result["gate_id"] == "QG-1"
        assert result["result"] == "passed"
        assert result["checks_total"] == 2
        assert result["checks_passed"] == 2

    @pytest.mark.asyncio
    async def test_get_gate_details_invalid_gate_400(self):
        """GET /workflows/{id}/gates/{gate_id}/details -- 400 on invalid gate ID."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            get_gate_details,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_gate_details(
                request=self.request, workflow_id="wf-001", gate_id="INVALID",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_gate_details_not_evaluated_404(self):
        """GET /workflows/{id}/gates/{gate_id}/details -- 404 when gate not evaluated."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.quality_gate_routes import (
            get_gate_details,
        )

        state = _make_workflow_state(quality_gates={})
        self.mock_service._state_manager.get_state.return_value = state

        with pytest.raises(HTTPException) as exc_info:
            await get_gate_details(
                request=self.request, workflow_id="wf-001", gate_id="QG-1",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test: Checkpoint Routes
# ---------------------------------------------------------------------------


class TestCheckpointRoutes:
    """Test checkpoint management routes (3 endpoints, ~8 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for checkpoint route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_list_checkpoints_success(self):
        """GET /workflows/{id}/checkpoints -- 200 returns checkpoint list."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            list_checkpoints,
        )

        cp1 = _make_checkpoint("cp-001", sequence_number=1)
        cp2 = _make_checkpoint("cp-002", sequence_number=2)
        state = _make_workflow_state(checkpoints=[cp1, cp2])
        self.mock_service._state_manager.get_state.return_value = state
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_checkpoints(
            request=self.request, workflow_id="wf-001",
            pagination=pagination, phase=None,
            user=self.user, _rate=self.user,
        )

        assert len(result["checkpoints"]) == 2
        assert result["meta"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_phase_filter(self):
        """GET /workflows/{id}/checkpoints?phase=... -- filters by phase."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            list_checkpoints,
        )

        cp1 = _make_checkpoint("cp-001", phase=DueDiligencePhase.INFORMATION_GATHERING)
        cp2 = _make_checkpoint("cp-002", phase=DueDiligencePhase.RISK_ASSESSMENT)
        state = _make_workflow_state(checkpoints=[cp1, cp2])
        self.mock_service._state_manager.get_state.return_value = state
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_checkpoints(
            request=self.request, workflow_id="wf-001",
            pagination=pagination, phase="risk_assessment",
            user=self.user, _rate=self.user,
        )

        assert len(result["checkpoints"]) == 1
        assert result["checkpoints"][0]["checkpoint_id"] == "cp-002"

    @pytest.mark.asyncio
    async def test_list_checkpoints_not_found_404(self):
        """GET /workflows/{id}/checkpoints -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            list_checkpoints,
        )

        self.mock_service._state_manager.get_state.return_value = None
        pagination = PaginationParams(limit=50, offset=0)

        with pytest.raises(HTTPException) as exc_info:
            await list_checkpoints(
                request=self.request, workflow_id="wf-nonexistent",
                pagination=pagination, phase=None,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_checkpoint_success(self):
        """POST /workflows/{id}/checkpoints -- 201 creates manual checkpoint."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            create_checkpoint,
        )

        state = _make_workflow_state(status=WorkflowStatus.RUNNING)
        self.mock_service._state_manager.get_state.return_value = state
        mock_cp = _make_checkpoint("cp-new")
        self.mock_service._state_manager.create_checkpoint.return_value = mock_cp

        result = await create_checkpoint(
            request=self.request, workflow_id="wf-001",
            reason="Before config change", user=self.user, _rate=self.user,
        )

        assert result["type"] == "manual"
        assert result["workflow_id"] == "wf-001"
        assert result["created_by"] == "user-001"

    @pytest.mark.asyncio
    async def test_create_checkpoint_not_found_404(self):
        """POST /workflows/{id}/checkpoints -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            create_checkpoint,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await create_checkpoint(
                request=self.request, workflow_id="wf-nonexistent",
                reason=None, user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_checkpoint_success(self):
        """GET /checkpoints/{id} -- 200 returns checkpoint details."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            get_checkpoint,
        )

        cp = _make_checkpoint("cp-001")
        self.mock_service._state_manager.get_checkpoint.return_value = cp

        result = await get_checkpoint(
            request=self.request, checkpoint_id="cp-001",
            user=self.user, _rate=self.user,
        )

        assert result["checkpoint_id"] == "cp-001"
        assert result["workflow_id"] == "wf-001"
        assert "cumulative_provenance_hash" in result

    @pytest.mark.asyncio
    async def test_get_checkpoint_not_found_404(self):
        """GET /checkpoints/{id} -- 404 when checkpoint missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            get_checkpoint,
        )

        self.mock_service._state_manager.get_checkpoint.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_checkpoint(
                request=self.request, checkpoint_id="cp-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_checkpoints_pagination(self):
        """GET /workflows/{id}/checkpoints -- respects offset and limit."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            list_checkpoints,
        )

        checkpoints = [
            _make_checkpoint(f"cp-{i:03d}", sequence_number=i)
            for i in range(10)
        ]
        state = _make_workflow_state(checkpoints=checkpoints)
        self.mock_service._state_manager.get_state.return_value = state
        pagination = PaginationParams(limit=3, offset=2)

        result = await list_checkpoints(
            request=self.request, workflow_id="wf-001",
            pagination=pagination, phase=None,
            user=self.user, _rate=self.user,
        )

        assert len(result["checkpoints"]) == 3
        assert result["meta"]["total"] == 10
        assert result["meta"]["has_more"] is True


# ---------------------------------------------------------------------------
# Test: Template Routes
# ---------------------------------------------------------------------------


class TestTemplateRoutes:
    """Test template management routes (4 endpoints, ~10 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service and engine for template route tests."""
        self.mock_service = MagicMock()
        self.mock_engine = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher_service = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher_engine = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes.get_workflow_engine",
            return_value=self.mock_engine,
        )
        self.patcher_service.start()
        self.patcher_engine.start()
        yield
        self.patcher_service.stop()
        self.patcher_engine.stop()

    @pytest.mark.asyncio
    async def test_list_templates_success(self):
        """GET /templates -- 200 returns built-in and custom templates."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            list_templates,
        )

        mock_template = MagicMock()
        mock_template.nodes = [MagicMock() for _ in range(25)]
        mock_template.quality_gates = ["QG-1", "QG-2", "QG-3"]
        self.mock_engine.get_commodity_template.return_value = mock_template
        self.mock_engine.list_custom_templates.return_value = []

        result = await list_templates(
            request=self.request, workflow_type=None,
            user=self.user, _rate=self.user,
        )

        assert "templates" in result
        assert result["total"] >= 0

    @pytest.mark.asyncio
    async def test_list_templates_with_type_filter(self):
        """GET /templates?workflow_type=standard -- filters by type."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            list_templates,
        )

        mock_template = MagicMock()
        mock_template.nodes = [MagicMock()]
        mock_template.quality_gates = ["QG-1"]
        self.mock_engine.get_commodity_template.return_value = mock_template
        self.mock_engine.list_custom_templates.return_value = []

        result = await list_templates(
            request=self.request, workflow_type="standard",
            user=self.user, _rate=self.user,
        )

        assert "templates" in result

    @pytest.mark.asyncio
    async def test_get_commodity_template_success(self):
        """GET /templates/commodity/{commodity} -- 200 returns template."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_commodity_template,
        )

        mock_template = MagicMock()
        mock_template.nodes = [MagicMock() for _ in range(25)]
        mock_template.edges = [MagicMock() for _ in range(30)]
        mock_template.quality_gates = ["QG-1", "QG-2", "QG-3"]
        mock_template.model_dump.return_value = {"name": "test"}
        self.mock_engine.get_commodity_template.return_value = mock_template

        result = await get_commodity_template(
            request=self.request, commodity="cocoa",
            workflow_type="standard", user=self.user, _rate=self.user,
        )

        assert result["commodity"] == "cocoa"
        assert result["agent_count"] == 25

    @pytest.mark.asyncio
    async def test_get_commodity_template_invalid_commodity_400(self):
        """GET /templates/commodity/{commodity} -- 400 for invalid commodity."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_commodity_template,
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_commodity_template(
                request=self.request, commodity="bananas",
                workflow_type="standard", user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_commodity_template_not_found_404(self):
        """GET /templates/commodity/{commodity} -- 404 when template missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_commodity_template,
        )

        self.mock_engine.get_commodity_template.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_commodity_template(
                request=self.request, commodity="cocoa",
                workflow_type="standard", user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_create_template_success(self):
        """POST /templates -- 201 creates custom template."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            create_template,
        )

        body = _make_workflow_definition()
        self.mock_engine.validate_definition.return_value = (True, [])
        self.mock_engine.store_custom_template.return_value = None

        result = await create_template(
            request=self.request, body=body,
            user=self.user, _rate=self.user,
        )

        assert result["valid"] is True
        assert result["created_by"] == "user-001"
        assert result["agent_count"] == 2

    @pytest.mark.asyncio
    async def test_create_template_invalid_dag_400(self):
        """POST /templates -- 400 when DAG validation fails."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            create_template,
        )

        body = _make_workflow_definition()
        self.mock_engine.validate_definition.return_value = (
            False, ["Circular dependency detected"],
        )

        with pytest.raises(HTTPException) as exc_info:
            await create_template(
                request=self.request, body=body,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_template_by_id_builtin(self):
        """GET /templates/{id} -- 200 returns built-in template by commodity_type ID."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_template,
        )

        mock_template = MagicMock()
        mock_template.nodes = [MagicMock()]
        mock_template.edges = [MagicMock()]
        mock_template.quality_gates = ["QG-1"]
        mock_template.model_dump.return_value = {"name": "test"}
        self.mock_engine.get_commodity_template.return_value = mock_template

        result = await get_template(
            request=self.request, template_id="cocoa_standard",
            user=self.user, _rate=self.user,
        )

        assert result["is_builtin"] is True
        assert result["commodity"] == "cocoa"

    @pytest.mark.asyncio
    async def test_get_template_by_id_not_found_404(self):
        """GET /templates/{id} -- 404 when template not found."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_template,
        )

        self.mock_engine.get_commodity_template.return_value = None
        self.mock_engine.get_custom_template.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_template(
                request=self.request, template_id="nonexistent_template",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    async def test_get_commodity_template_all_commodities(self, commodity):
        """GET /templates/commodity/{commodity} -- accepts all 7 commodities."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.template_routes import (
            get_commodity_template,
        )

        mock_template = MagicMock()
        mock_template.nodes = [MagicMock()]
        mock_template.edges = []
        mock_template.quality_gates = []
        mock_template.model_dump.return_value = {}
        self.mock_engine.get_commodity_template.return_value = mock_template

        result = await get_commodity_template(
            request=self.request, commodity=commodity,
            workflow_type="standard", user=self.user, _rate=self.user,
        )
        assert result["commodity"] == commodity


# ---------------------------------------------------------------------------
# Test: Package Routes
# ---------------------------------------------------------------------------


class TestPackageRoutes:
    """Test DDS package routes (4 endpoints, ~10 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for package route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_generate_package_success(self):
        """POST /workflows/{id}/package -- 201 generates DDS package."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            generate_package,
        )

        state = _make_workflow_state(status=WorkflowStatus.COMPLETED)
        self.mock_service._state_manager.get_state.return_value = state

        pkg = DueDiligencePackage(
            package_id="pkg-001",
            workflow_id="wf-001",
        )
        expected = PackageGenerationResponse(
            package=pkg,
        )
        self.mock_service.generate_package.return_value = expected

        result = await generate_package(
            request=self.request, workflow_id="wf-001",
            formats="json,pdf", language="en",
            include_executive_summary=True,
            include_evidence_annexes=True,
            user=self.user, _rate=self.user,
        )

        assert result.package.package_id == "pkg-001"
        assert result.package.workflow_id == "wf-001"

    @pytest.mark.asyncio
    async def test_generate_package_invalid_language_400(self):
        """POST /workflows/{id}/package -- 400 for invalid language."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            generate_package,
        )

        with pytest.raises(HTTPException) as exc_info:
            await generate_package(
                request=self.request, workflow_id="wf-001",
                formats="json", language="xx",
                include_executive_summary=True,
                include_evidence_annexes=True,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_generate_package_wrong_status_409(self):
        """POST /workflows/{id}/package -- 409 when workflow not ready."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            generate_package,
        )

        state = _make_workflow_state(status=WorkflowStatus.CREATED)
        self.mock_service._state_manager.get_state.return_value = state

        with pytest.raises(HTTPException) as exc_info:
            await generate_package(
                request=self.request, workflow_id="wf-001",
                formats="json", language="en",
                include_executive_summary=True,
                include_evidence_annexes=True,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_generate_package_not_found_404(self):
        """POST /workflows/{id}/package -- 404 when workflow missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            generate_package,
        )

        self.mock_service._state_manager.get_state.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await generate_package(
                request=self.request, workflow_id="wf-nonexistent",
                formats="json", language="en",
                include_executive_summary=True,
                include_evidence_annexes=True,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_generate_package_invalid_format_400(self):
        """POST /workflows/{id}/package -- 400 for invalid format."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            generate_package,
        )

        state = _make_workflow_state(status=WorkflowStatus.COMPLETED)
        self.mock_service._state_manager.get_state.return_value = state

        with pytest.raises(HTTPException) as exc_info:
            await generate_package(
                request=self.request, workflow_id="wf-001",
                formats="docx,xlsx", language="en",
                include_executive_summary=True,
                include_evidence_annexes=True,
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_package_success(self):
        """GET /packages/{id} -- 200 returns package metadata."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            get_package,
        )

        pkg = _make_dds_package()
        self.mock_service._package_generator.get_package.return_value = pkg

        result = await get_package(
            request=self.request, package_id="pkg-001",
            user=self.user, _rate=self.user,
        )

        assert result["package_id"] == "pkg-001"
        assert len(result["sections"]) == 2

    @pytest.mark.asyncio
    async def test_get_package_not_found_404(self):
        """GET /packages/{id} -- 404 when package missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            get_package,
        )

        self.mock_service._package_generator.get_package.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_package(
                request=self.request, package_id="pkg-nonexistent",
                user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_download_package_with_url(self):
        """GET /packages/{id}/download -- 200 returns download URL."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            download_package,
        )

        pkg = _make_dds_package()
        self.mock_service._package_generator.get_package.return_value = pkg

        result = await download_package(
            request=self.request, package_id="pkg-001",
            format="json", user=self.user, _rate=self.user,
        )

        assert result["download_url"] == "https://s3.example.com/pkg-001.json"
        assert result["expires_in_seconds"] == 3600

    @pytest.mark.asyncio
    async def test_download_package_invalid_format_400(self):
        """GET /packages/{id}/download -- 400 for invalid format."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            download_package,
        )

        with pytest.raises(HTTPException) as exc_info:
            await download_package(
                request=self.request, package_id="pkg-001",
                format="docx", user=self.user, _rate=self.user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_package_success(self):
        """POST /packages/validate -- 200 returns validation result."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.package_routes import (
            validate_package,
        )

        pkg = _make_dds_package()
        self.mock_service._package_generator.get_package.return_value = pkg
        self.mock_service._package_generator.validate_dds_schema.return_value = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        result = await validate_package(
            request=self.request, package_id="pkg-001",
            user=self.user, _rate=self.user,
        )

        assert result["valid"] is True
        assert result["error_count"] == 0
        assert result["field_coverage"]["total_fields"] == 2
        assert result["field_coverage"]["populated_fields"] == 2


# ---------------------------------------------------------------------------
# Test: Monitoring Routes
# ---------------------------------------------------------------------------


class TestMonitoringRoutes:
    """Test monitoring and operations routes (6 endpoints, ~12 tests)."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked service for monitoring route tests."""
        self.mock_service = MagicMock()
        self.user = _make_auth_user()
        self.request = _make_request_mock()
        self.patcher = patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes.get_ddo_service",
            return_value=self.mock_service,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """GET /health -- returns healthy status."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            health_check,
        )

        self.mock_service.health_check = AsyncMock(return_value={
            "database": "healthy",
            "redis": "healthy",
            "s3": "healthy",
        })

        result = await health_check(_rate=None)

        assert result["status"] == "healthy"
        assert result["agent"] == "GL-EUDR-DDO-026"
        assert result["version"] == VERSION

    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """GET /health -- returns degraded status on exception."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            health_check,
        )

        self.mock_service.health_check = AsyncMock(
            side_effect=Exception("Database connection failed"),
        )

        result = await health_check(_rate=None)

        assert result["status"] == "degraded"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_metrics_summary_success(self):
        """GET /metrics -- returns metrics data."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            metrics_summary,
        )

        self.mock_service.get_metrics = AsyncMock(return_value={
            "workflows_total": 42,
            "workflows_running": 5,
        })

        result = await metrics_summary(user=self.user, _rate=None)

        assert result["agent"] == "GL-EUDR-DDO-026"
        assert result["metrics"]["workflows_total"] == 42

    @pytest.mark.asyncio
    async def test_metrics_summary_error_raises_500(self):
        """GET /metrics -- 500 on metrics retrieval failure."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            metrics_summary,
        )

        self.mock_service.get_metrics = AsyncMock(
            side_effect=Exception("Prometheus unreachable"),
        )

        with pytest.raises(HTTPException) as exc_info:
            await metrics_summary(user=self.user, _rate=None)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_version_info(self):
        """GET /version -- returns version and feature info."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            version_info,
        )

        result = await version_info(_rate=None)

        assert result["agent_id"] == "GL-EUDR-DDO-026"
        assert result["version"] == VERSION
        assert "workflow_definition_engine" in result["features"]
        assert len(result["supported_commodities"]) == 7
        assert result["upstream_agents"] == 25
        assert result["quality_gates"] == 3

    @pytest.mark.asyncio
    async def test_get_circuit_breakers_success(self):
        """GET /circuit-breakers -- returns all circuit breaker states."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            get_circuit_breakers,
        )

        cb_record = CircuitBreakerRecord(
            agent_id="EUDR-003",
            state=CircuitBreakerState.CLOSED,
            failure_count=0,
        )
        self.mock_service._error_manager.get_all_circuit_breaker_states.return_value = {
            "EUDR-003": cb_record,
        }

        result = await get_circuit_breakers(user=self.user, _rate=None)

        assert result["total_agents"] == 25
        assert "EUDR-003" in result["circuit_breakers"]

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_success(self):
        """POST /circuit-breakers/{id}/reset -- resets circuit breaker."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            reset_circuit_breaker,
        )

        self.mock_service._error_manager.reset_circuit_breaker.return_value = None

        result = await reset_circuit_breaker(
            agent_id="EUDR-003", user=self.user, _rate=None,
        )

        assert result["agent_id"] == "EUDR-003"
        assert result["new_state"] == "closed"
        assert result["reset_by"] == "user-001"

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_not_found_404(self):
        """POST /circuit-breakers/{id}/reset -- 404 for unknown agent."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            reset_circuit_breaker,
        )

        self.mock_service._error_manager.reset_circuit_breaker.side_effect = (
            ValueError("Agent not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await reset_circuit_breaker(
                agent_id="EUDR-INVALID", user=self.user, _rate=None,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_dead_letter_queue_success(self):
        """GET /dead-letter-queue -- returns DLQ entries."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            list_dead_letter_queue,
        )

        entries = [
            _make_dead_letter_entry("dlq-001", "EUDR-003"),
            _make_dead_letter_entry("dlq-002", "EUDR-020"),
        ]
        self.mock_service._error_manager.get_dead_letter_entries.return_value = entries
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_dead_letter_queue(
            user=self.user, resolved=None, agent_id=None,
            pagination=pagination, _rate=None,
        )

        assert result["total"] == 2
        assert len(result["entries"]) == 2

    @pytest.mark.asyncio
    async def test_dead_letter_queue_filter_by_agent(self):
        """GET /dead-letter-queue?agent_id=... -- filters by agent."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            list_dead_letter_queue,
        )

        entries = [
            _make_dead_letter_entry("dlq-001", "EUDR-003"),
            _make_dead_letter_entry("dlq-002", "EUDR-020"),
        ]
        self.mock_service._error_manager.get_dead_letter_entries.return_value = entries
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_dead_letter_queue(
            user=self.user, resolved=None, agent_id="EUDR-003",
            pagination=pagination, _rate=None,
        )

        assert result["total"] == 1
        assert result["entries"][0]["agent_id"] == "EUDR-003"

    @pytest.mark.asyncio
    async def test_dead_letter_queue_filter_by_resolved(self):
        """GET /dead-letter-queue?resolved=false -- filters by resolution status."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.monitoring_routes import (
            list_dead_letter_queue,
        )

        entries = [
            _make_dead_letter_entry("dlq-001", "EUDR-003", resolved=False),
            _make_dead_letter_entry("dlq-002", "EUDR-020", resolved=True),
        ]
        self.mock_service._error_manager.get_dead_letter_entries.return_value = entries
        pagination = PaginationParams(limit=50, offset=0)

        result = await list_dead_letter_queue(
            user=self.user, resolved=False, agent_id=None,
            pagination=pagination, _rate=None,
        )

        assert result["total"] == 1
        assert result["entries"][0]["resolved"] is False


# ---------------------------------------------------------------------------
# Test: Auth and Permission Checks
# ---------------------------------------------------------------------------


class TestAuthAndPermissions:
    """Test authentication and authorization patterns (~10 tests)."""

    @pytest.mark.asyncio
    async def test_require_permission_grants_admin(self):
        """Admin role bypasses permission checks."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        admin_user = _make_auth_user(roles=["admin"], permissions=[])
        check = require_permission("eudr-ddo:workflows:read")

        result = await check(user=admin_user)
        assert result.user_id == admin_user.user_id

    @pytest.mark.asyncio
    async def test_require_permission_grants_exact_match(self):
        """Exact permission match grants access."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(
            roles=["operator"],
            permissions=["eudr-ddo:workflows:read"],
        )
        check = require_permission("eudr-ddo:workflows:read")

        result = await check(user=user)
        assert result.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_require_permission_grants_wildcard(self):
        """Wildcard permission grants access."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(
            roles=["operator"],
            permissions=["eudr-ddo:*"],
        )
        check = require_permission("eudr-ddo:workflows:read")

        result = await check(user=user)
        assert result.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_require_permission_denies_missing_permission(self):
        """Missing permission raises 403."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(
            roles=["viewer"],
            permissions=["eudr-ddo:workflows:read"],
        )
        check = require_permission("eudr-ddo:workflows:create")

        with pytest.raises(HTTPException) as exc_info:
            await check(user=user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_permission_platform_admin_bypasses(self):
        """platform_admin role bypasses all permission checks."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(roles=["platform_admin"], permissions=[])
        check = require_permission("eudr-ddo:gates:override")

        result = await check(user=user)
        assert result.user_id == user.user_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("permission", [
        "eudr-ddo:workflows:create",
        "eudr-ddo:workflows:read",
        "eudr-ddo:workflows:manage",
        "eudr-ddo:workflows:delete",
        "eudr-ddo:gates:read",
        "eudr-ddo:gates:override",
        "eudr-ddo:checkpoints:read",
        "eudr-ddo:checkpoints:rollback",
        "eudr-ddo:templates:read",
        "eudr-ddo:templates:manage",
        "eudr-ddo:packages:generate",
        "eudr-ddo:packages:read",
        "eudr-ddo:packages:download",
        "eudr-ddo:circuit-breakers:read",
        "eudr-ddo:circuit-breakers:manage",
        "eudr-ddo:dlq:read",
    ])
    async def test_each_permission_is_enforced(self, permission):
        """Each documented permission denies access when missing."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(roles=["viewer"], permissions=[])
        check = require_permission(permission)

        with pytest.raises(HTTPException) as exc_info:
            await check(user=user)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_partial_wildcard_permission(self):
        """Partial wildcard (eudr-ddo:workflows:*) grants sub-permissions."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            require_permission,
        )

        user = _make_auth_user(
            roles=["operator"],
            permissions=["eudr-ddo:workflows:*"],
        )
        check = require_permission("eudr-ddo:workflows:read")

        result = await check(user=user)
        assert result.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self):
        """Missing auth credentials raises 401."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            get_current_user,
        )

        request = _make_request_mock()
        request.state.auth = None
        delattr(request.state, "auth")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request=request, token=None, api_key=None)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Test: Pagination
# ---------------------------------------------------------------------------


class TestPagination:
    """Test pagination parameter handling (~6 tests)."""

    def test_default_pagination_params(self):
        """Default pagination has limit=50, offset=0."""
        # get_pagination() uses FastAPI Query() defaults that only resolve
        # inside the DI framework. We test the same logic by calling it
        # with the documented defaults directly.
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            get_pagination,
        )

        params = get_pagination(limit=50, offset=0)
        assert params.limit == 50
        assert params.offset == 0

    def test_custom_pagination_params(self):
        """Custom pagination respects provided values."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
            get_pagination,
        )

        params = get_pagination(limit=10, offset=20)
        assert params.limit == 10
        assert params.offset == 20

    def test_pagination_max_limit(self):
        """Pagination limit respects maximum of 1000."""
        params = PaginationParams(limit=1000, offset=0)
        assert params.limit == 1000

    @pytest.mark.asyncio
    async def test_list_workflows_pagination_has_more_true(self):
        """Pagination meta.has_more is true when more results exist."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            list_workflows,
        )

        mock_service = MagicMock()
        state = _make_workflow_state()
        mock_service._state_manager.list_workflows.return_value = [state]
        mock_service._state_manager.count_workflows.return_value = 100

        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes.get_ddo_service",
            return_value=mock_service,
        ):
            pagination = PaginationParams(limit=10, offset=0)
            result = await list_workflows(
                request=_make_request_mock(),
                user=_make_auth_user(),
                pagination=pagination,
                commodity=None,
                workflow_type=None,
                status_filter=None,
                _rate=_make_auth_user(),
            )

        assert result["meta"]["has_more"] is True
        assert result["meta"]["total"] == 100

    @pytest.mark.asyncio
    async def test_list_workflows_pagination_has_more_false(self):
        """Pagination meta.has_more is false at end of results."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes import (
            list_workflows,
        )

        mock_service = MagicMock()
        mock_service._state_manager.list_workflows.return_value = []
        mock_service._state_manager.count_workflows.return_value = 5

        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.workflow_routes.get_ddo_service",
            return_value=mock_service,
        ):
            pagination = PaginationParams(limit=50, offset=0)
            result = await list_workflows(
                request=_make_request_mock(),
                user=_make_auth_user(),
                pagination=pagination,
                commodity=None,
                workflow_type=None,
                status_filter=None,
                _rate=_make_auth_user(),
            )

        assert result["meta"]["has_more"] is False

    @pytest.mark.asyncio
    async def test_checkpoint_pagination_offset(self):
        """Checkpoint list respects offset parameter."""
        from greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes import (
            list_checkpoints,
        )

        mock_service = MagicMock()
        checkpoints = [
            _make_checkpoint(f"cp-{i:03d}", sequence_number=i)
            for i in range(5)
        ]
        state = _make_workflow_state(checkpoints=checkpoints)
        mock_service._state_manager.get_state.return_value = state

        with patch(
            "greenlang.agents.eudr.due_diligence_orchestrator.api.checkpoint_routes.get_ddo_service",
            return_value=mock_service,
        ):
            pagination = PaginationParams(limit=2, offset=3)
            result = await list_checkpoints(
                request=_make_request_mock(),
                workflow_id="wf-001",
                pagination=pagination,
                phase=None,
                user=_make_auth_user(),
                _rate=_make_auth_user(),
            )

        assert len(result["checkpoints"]) == 2
        assert result["meta"]["offset"] == 3
        assert result["meta"]["total"] == 5
