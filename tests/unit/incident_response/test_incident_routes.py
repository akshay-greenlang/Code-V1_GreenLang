"""
Unit tests for incident_routes API.

Tests all incident response API endpoints including incident CRUD,
playbook execution, timeline events, and post-mortem management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.incident_response.api.incident_routes import router
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentType,
    IncidentStatus,
    EscalationLevel,
    Alert,
    AlertSource,
    PlaybookExecution,
    PlaybookStatus,
    TimelineEvent,
    PostMortem,
)


def _build_test_app() -> FastAPI:
    """Build FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/incidents")
    return app


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    app = _build_test_app()
    return TestClient(app)


@pytest.fixture
def mock_incident_service():
    """Create mock incident service."""
    service = AsyncMock()
    return service


class TestIncidentListEndpoint:
    """Test GET /api/v1/incidents endpoint."""

    def test_list_incidents_returns_200(
        self, test_client, incident_responder_headers
    ):
        """Test listing incidents returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200

    def test_list_incidents_returns_incidents(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test listing incidents returns incident data."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_incidents.return_value = [sample_incident]
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["incidents"]) == 1

    def test_list_incidents_filter_by_status(
        self, test_client, incident_responder_headers
    ):
        """Test filtering incidents by status."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents?status=open",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            mock_service.list_incidents.assert_called_once()
            call_kwargs = mock_service.list_incidents.call_args[1]
            assert call_kwargs.get("status") == "open"

    def test_list_incidents_filter_by_escalation_level(
        self, test_client, incident_responder_headers
    ):
        """Test filtering incidents by escalation level."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents?escalation_level=P0",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200

    def test_list_incidents_pagination(
        self, test_client, incident_responder_headers
    ):
        """Test pagination parameters."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents?page=2&page_size=25",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200

    def test_list_incidents_unauthorized(self, test_client):
        """Test listing incidents without auth returns 401."""
        response = test_client.get("/api/v1/incidents")

        assert response.status_code in [401, 403]


class TestIncidentGetEndpoint:
    """Test GET /api/v1/incidents/{incident_id} endpoint."""

    def test_get_incident_returns_200(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test getting incident returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/incidents/{sample_incident.incident_id}",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200

    def test_get_incident_returns_incident_data(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test getting incident returns full incident data."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/incidents/{sample_incident.incident_id}",
                headers=incident_responder_headers,
            )

            data = response.json()
            assert data["incident_id"] == sample_incident.incident_id
            assert data["title"] == sample_incident.title

    def test_get_incident_not_found(
        self, test_client, incident_responder_headers
    ):
        """Test getting nonexistent incident returns 404."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = None
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents/nonexistent-id",
                headers=incident_responder_headers,
            )

            assert response.status_code == 404


class TestIncidentCreateEndpoint:
    """Test POST /api/v1/incidents endpoint."""

    def test_create_incident_returns_201(
        self, test_client, security_admin_headers
    ):
        """Test creating incident returns 201."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_incident.return_value = Incident(
                incident_id=str(uuid4()),
                title="New incident",
                description="Description",
                incident_type=IncidentType.INFRASTRUCTURE,
                status=IncidentStatus.OPEN,
                escalation_level=EscalationLevel.P2,
                alerts=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=[],
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/incidents",
                headers=security_admin_headers,
                json={
                    "title": "New incident",
                    "description": "Description",
                    "incident_type": "infrastructure",
                    "escalation_level": "P2",
                },
            )

            assert response.status_code == 201

    def test_create_incident_returns_incident_id(
        self, test_client, security_admin_headers
    ):
        """Test creating incident returns incident ID."""
        incident_id = str(uuid4())

        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_incident.return_value = Incident(
                incident_id=incident_id,
                title="New incident",
                description="Description",
                incident_type=IncidentType.INFRASTRUCTURE,
                status=IncidentStatus.OPEN,
                escalation_level=EscalationLevel.P2,
                alerts=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                tags=[],
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/incidents",
                headers=security_admin_headers,
                json={
                    "title": "New incident",
                    "description": "Description",
                    "incident_type": "infrastructure",
                },
            )

            data = response.json()
            assert data["incident_id"] == incident_id

    def test_create_incident_validation_error(
        self, test_client, security_admin_headers
    ):
        """Test creating incident with invalid data returns 422."""
        response = test_client.post(
            "/api/v1/incidents",
            headers=security_admin_headers,
            json={
                "title": "",  # Empty title
                "description": "",
            },
        )

        assert response.status_code == 422


class TestIncidentUpdateEndpoint:
    """Test PUT /api/v1/incidents/{incident_id} endpoint."""

    def test_update_incident_returns_200(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test updating incident returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.update_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/incidents/{sample_incident.incident_id}",
                headers=incident_responder_headers,
                json={
                    "title": "Updated title",
                    "description": "Updated description",
                },
            )

            assert response.status_code == 200

    def test_update_incident_status(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test updating incident status."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            updated_incident = sample_incident
            updated_incident.status = IncidentStatus.INVESTIGATING
            mock_service.update_incident.return_value = updated_incident
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/incidents/{sample_incident.incident_id}",
                headers=incident_responder_headers,
                json={
                    "status": "investigating",
                },
            )

            assert response.status_code == 200

    def test_update_incident_not_found(
        self, test_client, incident_responder_headers
    ):
        """Test updating nonexistent incident returns 404."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = None
            mock_get_service.return_value = mock_service

            response = test_client.put(
                "/api/v1/incidents/nonexistent-id",
                headers=incident_responder_headers,
                json={"title": "Updated"},
            )

            assert response.status_code == 404


class TestIncidentAssignEndpoint:
    """Test POST /api/v1/incidents/{incident_id}/assign endpoint."""

    def test_assign_incident_returns_200(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test assigning incident returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.assign_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/assign",
                headers=incident_responder_headers,
                json={"assignee": "platform-team"},
            )

            assert response.status_code == 200

    def test_assign_incident_updates_assignee(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test assigning incident updates assignee."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            sample_incident.assigned_to = "platform-team"
            mock_service.assign_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/assign",
                headers=incident_responder_headers,
                json={"assignee": "platform-team"},
            )

            data = response.json()
            assert data["assigned_to"] == "platform-team"


class TestIncidentEscalateEndpoint:
    """Test POST /api/v1/incidents/{incident_id}/escalate endpoint."""

    def test_escalate_incident_returns_200(
        self, test_client, security_admin_headers, sample_incident
    ):
        """Test escalating incident returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.escalate_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/escalate",
                headers=security_admin_headers,
                json={"escalation_level": "P1", "reason": "Customer impact"},
            )

            assert response.status_code == 200

    def test_escalate_incident_updates_level(
        self, test_client, security_admin_headers, sample_incident
    ):
        """Test escalating incident updates escalation level."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            sample_incident.escalation_level = EscalationLevel.P1
            mock_service.escalate_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/escalate",
                headers=security_admin_headers,
                json={"escalation_level": "P1"},
            )

            data = response.json()
            assert data["escalation_level"] == "P1"


class TestIncidentResolveEndpoint:
    """Test POST /api/v1/incidents/{incident_id}/resolve endpoint."""

    def test_resolve_incident_returns_200(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test resolving incident returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.resolve_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/resolve",
                headers=incident_responder_headers,
                json={"resolution": "Issue fixed by restarting pods"},
            )

            assert response.status_code == 200

    def test_resolve_incident_updates_status(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test resolving incident updates status to resolved."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            sample_incident.status = IncidentStatus.RESOLVED
            mock_service.resolve_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/resolve",
                headers=incident_responder_headers,
                json={"resolution": "Fixed"},
            )

            data = response.json()
            assert data["status"] == "resolved"


class TestPlaybookExecutionEndpoints:
    """Test playbook execution API endpoints."""

    def test_list_available_playbooks(
        self, test_client, incident_responder_headers
    ):
        """Test listing available playbooks."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_playbook_executor"
        ) as mock_get_executor:
            mock_executor = MagicMock()
            mock_executor.get_available_playbooks.return_value = [
                {"id": "pod_restart", "name": "Pod Restart"},
                {"id": "scale_up", "name": "Scale Up"},
            ]
            mock_get_executor.return_value = mock_executor

            response = test_client.get(
                "/api/v1/incidents/playbooks",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["playbooks"]) >= 2

    def test_execute_playbook_returns_200(
        self, test_client, security_admin_headers, sample_incident
    ):
        """Test executing playbook returns 200."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service, patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_playbook_executor"
        ) as mock_get_executor:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            mock_executor = AsyncMock()
            mock_executor.execute.return_value = PlaybookExecution(
                execution_id=str(uuid4()),
                playbook_id="pod_restart",
                incident_id=sample_incident.incident_id,
                status=PlaybookStatus.RUNNING,
                steps=[],
                current_step=0,
                started_at=datetime.utcnow(),
                completed_at=None,
                executed_by="test-user",
                results={},
            )
            mock_get_executor.return_value = mock_executor

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/playbooks/pod_restart/execute",
                headers=security_admin_headers,
            )

            assert response.status_code == 200

    def test_get_playbook_execution_status(
        self, test_client, incident_responder_headers, sample_playbook_execution
    ):
        """Test getting playbook execution status."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_playbook_executor"
        ) as mock_get_executor:
            mock_executor = AsyncMock()
            mock_executor.get_execution.return_value = sample_playbook_execution
            mock_get_executor.return_value = mock_executor

            response = test_client.get(
                f"/api/v1/incidents/playbooks/executions/{sample_playbook_execution.execution_id}",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["execution_id"] == sample_playbook_execution.execution_id


class TestTimelineEndpoints:
    """Test incident timeline API endpoints."""

    def test_get_incident_timeline(
        self, test_client, incident_responder_headers, incident_with_timeline
    ):
        """Test getting incident timeline."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = incident_with_timeline
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/incidents/{incident_with_timeline.incident_id}/timeline",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["events"]) == len(incident_with_timeline.timeline)

    def test_add_timeline_event(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test adding timeline event."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.add_timeline_event.return_value = TimelineEvent(
                event_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                event_type="note",
                description="Investigation update",
                actor="test-user",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/timeline",
                headers=incident_responder_headers,
                json={
                    "event_type": "note",
                    "description": "Investigation update",
                },
            )

            assert response.status_code == 201


class TestPostMortemEndpoints:
    """Test post-mortem API endpoints."""

    def test_create_post_mortem(
        self, test_client, security_admin_headers, sample_incident
    ):
        """Test creating post-mortem."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.create_post_mortem.return_value = PostMortem(
                post_mortem_id=str(uuid4()),
                incident_id=sample_incident.incident_id,
                title=f"Post-Mortem: {sample_incident.title}",
                summary="Summary",
                timeline=[],
                root_cause="Root cause",
                contributing_factors=[],
                impact={},
                action_items=[],
                lessons_learned=[],
                created_at=datetime.utcnow(),
                created_by="test-user",
                status="draft",
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/post-mortem",
                headers=security_admin_headers,
                json={
                    "summary": "Summary",
                    "root_cause": "Root cause",
                },
            )

            assert response.status_code == 201

    def test_get_post_mortem(
        self, test_client, incident_responder_headers, sample_post_mortem
    ):
        """Test getting post-mortem."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_post_mortem.return_value = sample_post_mortem
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/incidents/{sample_post_mortem.incident_id}/post-mortem",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["post_mortem_id"] == sample_post_mortem.post_mortem_id

    def test_update_post_mortem(
        self, test_client, security_admin_headers, sample_post_mortem
    ):
        """Test updating post-mortem."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_post_mortem.return_value = sample_post_mortem
            mock_service.update_post_mortem.return_value = sample_post_mortem
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/incidents/{sample_post_mortem.incident_id}/post-mortem",
                headers=security_admin_headers,
                json={
                    "summary": "Updated summary",
                    "lessons_learned": ["New lesson"],
                },
            )

            assert response.status_code == 200


class TestAlertsEndpoints:
    """Test alert-related API endpoints."""

    def test_list_alerts_for_incident(
        self, test_client, incident_responder_headers, sample_incident
    ):
        """Test listing alerts for incident."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/incidents/{sample_incident.incident_id}/alerts",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "alerts" in data

    def test_add_alert_to_incident(
        self, test_client, security_admin_headers, sample_incident
    ):
        """Test adding alert to incident."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_incident.return_value = sample_incident
            mock_service.add_alert_to_incident.return_value = sample_incident
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/incidents/{sample_incident.incident_id}/alerts",
                headers=security_admin_headers,
                json={
                    "alert_id": str(uuid4()),
                    "title": "New alert",
                    "severity": "high",
                    "source": "prometheus",
                },
            )

            assert response.status_code == 200


class TestMetricsEndpoints:
    """Test incident metrics API endpoints."""

    def test_get_incident_metrics(
        self, test_client, incident_responder_headers
    ):
        """Test getting incident metrics."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_metrics.return_value = {
                "total_incidents": 100,
                "open_incidents": 5,
                "mttr_minutes": 45,
                "incidents_by_type": {"security": 20, "infrastructure": 80},
            }
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents/metrics",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "total_incidents" in data
            assert "mttr_minutes" in data

    def test_get_incident_metrics_with_date_range(
        self, test_client, incident_responder_headers
    ):
        """Test getting incident metrics with date range."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_metrics.return_value = {}
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents/metrics?start_date=2025-01-01&end_date=2025-01-31",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200


class TestSearchEndpoint:
    """Test incident search API endpoint."""

    def test_search_incidents(
        self, test_client, incident_responder_headers
    ):
        """Test searching incidents."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents/search?q=database",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200

    def test_search_incidents_with_filters(
        self, test_client, incident_responder_headers
    ):
        """Test searching incidents with filters."""
        with patch(
            "greenlang.infrastructure.incident_response.api.incident_routes.get_incident_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_incidents.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/incidents/search?q=database&incident_type=infrastructure&status=open",
                headers=incident_responder_headers,
            )

            assert response.status_code == 200
