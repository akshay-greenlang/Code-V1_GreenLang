"""
Unit tests for threat_routes API.

Tests all threat modeling API endpoints including threat model CRUD,
threat analysis, risk scoring, and mitigation management.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.threat_modeling.api.threat_routes import router
from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataFlow,
    TrustBoundary,
    Threat,
    ThreatCategory,
    ThreatModel,
    RiskLevel,
    Mitigation,
    MitigationStatus,
)


def _build_test_app() -> FastAPI:
    """Build FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/threat-models")
    return app


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    app = _build_test_app()
    return TestClient(app)


@pytest.fixture
def mock_threat_service():
    """Create mock threat modeling service."""
    service = AsyncMock()
    return service


class TestThreatModelListEndpoint:
    """Test GET /api/v1/threat-models endpoint."""

    def test_list_threat_models_returns_200(
        self, test_client, security_analyst_headers
    ):
        """Test listing threat models returns 200."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_threat_models.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/threat-models",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200

    def test_list_threat_models_returns_models(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test listing threat models returns model data."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_threat_models.return_value = [sample_threat_model]
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/threat-models",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["threat_models"]) == 1

    def test_list_threat_models_filter_by_status(
        self, test_client, security_analyst_headers
    ):
        """Test filtering threat models by status."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.list_threat_models.return_value = []
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/threat-models?status=active",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200

    def test_list_threat_models_unauthorized(self, test_client):
        """Test listing threat models without auth returns 401."""
        response = test_client.get("/api/v1/threat-models")

        assert response.status_code in [401, 403]


class TestThreatModelGetEndpoint:
    """Test GET /api/v1/threat-models/{model_id} endpoint."""

    def test_get_threat_model_returns_200(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test getting threat model returns 200."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200

    def test_get_threat_model_returns_full_data(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test getting threat model returns full model data."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}",
                headers=security_analyst_headers,
            )

            data = response.json()
            assert data["model_id"] == sample_threat_model.model_id
            assert "components" in data
            assert "data_flows" in data
            assert "threats" in data

    def test_get_threat_model_not_found(
        self, test_client, security_analyst_headers
    ):
        """Test getting nonexistent threat model returns 404."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = None
            mock_get_service.return_value = mock_service

            response = test_client.get(
                "/api/v1/threat-models/nonexistent-id",
                headers=security_analyst_headers,
            )

            assert response.status_code == 404


class TestThreatModelCreateEndpoint:
    """Test POST /api/v1/threat-models endpoint."""

    def test_create_threat_model_returns_201(
        self, test_client, threat_model_admin_headers
    ):
        """Test creating threat model returns 201."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_threat_model.return_value = ThreatModel(
                model_id=str(uuid4()),
                name="New Model",
                version="1.0.0",
                description="Test model",
                scope="Test scope",
                components=[],
                data_flows=[],
                trust_boundaries=[],
                threats=[],
                overall_risk_score=0.0,
                overall_risk_level=RiskLevel.LOW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="test-user",
                status="draft",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/threat-models",
                headers=threat_model_admin_headers,
                json={
                    "name": "New Model",
                    "description": "Test model",
                    "scope": "Test scope",
                },
            )

            assert response.status_code == 201

    def test_create_threat_model_with_components(
        self, test_client, threat_model_admin_headers
    ):
        """Test creating threat model with components."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.create_threat_model.return_value = ThreatModel(
                model_id=str(uuid4()),
                name="Model with Components",
                version="1.0.0",
                description="Test",
                scope="Test",
                components=[],
                data_flows=[],
                trust_boundaries=[],
                threats=[],
                overall_risk_score=0.0,
                overall_risk_level=RiskLevel.LOW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="test-user",
                status="draft",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                "/api/v1/threat-models",
                headers=threat_model_admin_headers,
                json={
                    "name": "Model with Components",
                    "description": "Test",
                    "scope": "Test",
                    "components": [
                        {
                            "name": "api-service",
                            "component_type": "service",
                            "description": "API service",
                            "data_classification": "confidential",
                        }
                    ],
                },
            )

            assert response.status_code == 201

    def test_create_threat_model_validation_error(
        self, test_client, threat_model_admin_headers
    ):
        """Test creating threat model with invalid data returns 422."""
        response = test_client.post(
            "/api/v1/threat-models",
            headers=threat_model_admin_headers,
            json={
                "name": "",  # Empty name
            },
        )

        assert response.status_code == 422


class TestAnalyzeEndpoints:
    """Test threat analysis API endpoints."""

    def test_analyze_component_returns_threats(
        self, test_client, security_analyst_headers
    ):
        """Test analyzing component returns threats."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_stride_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.analyze_component.return_value = [
                Threat(
                    threat_id=str(uuid4()),
                    title="Test Threat",
                    description="Test",
                    category=ThreatCategory.SPOOFING,
                    affected_component_ids=[],
                    affected_data_flow_ids=[],
                    attack_vector="Test",
                    prerequisites=[],
                    potential_impact="Test",
                    likelihood=0.5,
                    impact=0.5,
                    risk_score=0.25,
                    risk_level=RiskLevel.MEDIUM,
                    mitigations=[],
                    status="identified",
                    identified_at=datetime.utcnow(),
                    identified_by="system",
                )
            ]
            mock_get_engine.return_value = mock_engine

            response = test_client.post(
                "/api/v1/threat-models/analyze/component",
                headers=security_analyst_headers,
                json={
                    "name": "test-service",
                    "component_type": "service",
                    "description": "Test service",
                    "data_classification": "confidential",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "threats" in data

    def test_analyze_data_flow_returns_threats(
        self, test_client, security_analyst_headers
    ):
        """Test analyzing data flow returns threats."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_stride_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.analyze_data_flow.return_value = []
            mock_get_engine.return_value = mock_engine

            response = test_client.post(
                "/api/v1/threat-models/analyze/data-flow",
                headers=security_analyst_headers,
                json={
                    "name": "API to DB",
                    "source_component_id": str(uuid4()),
                    "destination_component_id": str(uuid4()),
                    "data_types": ["user_data"],
                    "protocol": "TCP",
                    "encrypted": True,
                },
            )

            assert response.status_code == 200

    def test_analyze_trust_boundary_returns_threats(
        self, test_client, security_analyst_headers
    ):
        """Test analyzing trust boundary returns threats."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_stride_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.analyze_trust_boundary.return_value = []
            mock_get_engine.return_value = mock_engine

            response = test_client.post(
                "/api/v1/threat-models/analyze/trust-boundary",
                headers=security_analyst_headers,
                json={
                    "name": "Network Boundary",
                    "boundary_type": "network",
                    "description": "DMZ boundary",
                    "controls": ["firewall", "waf"],
                },
            )

            assert response.status_code == 200


class TestRiskScoringEndpoints:
    """Test risk scoring API endpoints."""

    def test_calculate_risk_score(
        self, test_client, security_analyst_headers, spoofing_threat
    ):
        """Test calculating risk score for threat."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_risk_scorer"
        ) as mock_get_scorer:
            mock_scorer = MagicMock()
            spoofing_threat.risk_score = 0.65
            spoofing_threat.risk_level = RiskLevel.HIGH
            mock_scorer.calculate_risk_score.return_value = spoofing_threat
            mock_get_scorer.return_value = mock_scorer

            response = test_client.post(
                "/api/v1/threat-models/threats/score",
                headers=security_analyst_headers,
                json={
                    "threat_id": spoofing_threat.threat_id,
                    "title": spoofing_threat.title,
                    "category": "spoofing",
                    "attack_vector": spoofing_threat.attack_vector,
                    "prerequisites": spoofing_threat.prerequisites,
                    "potential_impact": spoofing_threat.potential_impact,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "risk_score" in data
            assert "risk_level" in data

    def test_calculate_cvss_score(
        self, test_client, security_analyst_headers
    ):
        """Test calculating CVSS score."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_risk_scorer"
        ) as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.calculate_cvss.return_value = 8.5
            mock_get_scorer.return_value = mock_scorer

            response = test_client.post(
                "/api/v1/threat-models/threats/cvss",
                headers=security_analyst_headers,
                json={
                    "attack_vector": "NETWORK",
                    "attack_complexity": "LOW",
                    "privileges_required": "NONE",
                    "user_interaction": "NONE",
                    "scope": "CHANGED",
                    "confidentiality_impact": "HIGH",
                    "integrity_impact": "HIGH",
                    "availability_impact": "HIGH",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "cvss_score" in data
            assert 0 <= data["cvss_score"] <= 10

    def test_prioritize_threats(
        self, test_client, security_analyst_headers
    ):
        """Test prioritizing threats."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_risk_scorer"
        ) as mock_get_scorer:
            mock_scorer = MagicMock()
            mock_scorer.prioritize_threats.return_value = []
            mock_get_scorer.return_value = mock_scorer

            response = test_client.post(
                "/api/v1/threat-models/threats/prioritize",
                headers=security_analyst_headers,
                json={
                    "threat_ids": [str(uuid4()), str(uuid4())],
                },
            )

            assert response.status_code == 200


class TestThreatEndpoints:
    """Test individual threat management endpoints."""

    def test_list_threats_for_model(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test listing threats for a threat model."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/threats",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "threats" in data

    def test_get_threat_details(
        self, test_client, security_analyst_headers, spoofing_threat
    ):
        """Test getting threat details."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat.return_value = spoofing_threat
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/threats/{spoofing_threat.threat_id}",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["threat_id"] == spoofing_threat.threat_id

    def test_update_threat_status(
        self, test_client, threat_model_admin_headers, spoofing_threat
    ):
        """Test updating threat status."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat.return_value = spoofing_threat
            mock_service.update_threat.return_value = spoofing_threat
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/threat-models/threats/{spoofing_threat.threat_id}",
                headers=threat_model_admin_headers,
                json={
                    "status": "mitigated",
                },
            )

            assert response.status_code == 200


class TestMitigationEndpoints:
    """Test mitigation management endpoints."""

    def test_list_mitigations_for_threat(
        self, test_client, security_analyst_headers, spoofing_threat
    ):
        """Test listing mitigations for a threat."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat.return_value = spoofing_threat
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/threats/{spoofing_threat.threat_id}/mitigations",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200

    def test_add_mitigation_to_threat(
        self, test_client, threat_model_admin_headers, spoofing_threat
    ):
        """Test adding mitigation to a threat."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat.return_value = spoofing_threat
            mock_service.add_mitigation.return_value = Mitigation(
                mitigation_id=str(uuid4()),
                title="New Mitigation",
                description="Implement MFA",
                threat_ids=[spoofing_threat.threat_id],
                control_type="preventive",
                implementation_effort="medium",
                effectiveness=0.8,
                status=MitigationStatus.PLANNED,
                owner="security-team",
                due_date=datetime.utcnow(),
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/threat-models/threats/{spoofing_threat.threat_id}/mitigations",
                headers=threat_model_admin_headers,
                json={
                    "title": "New Mitigation",
                    "description": "Implement MFA",
                    "control_type": "preventive",
                    "effectiveness": 0.8,
                },
            )

            assert response.status_code == 201

    def test_update_mitigation_status(
        self, test_client, threat_model_admin_headers, sample_mitigation
    ):
        """Test updating mitigation status."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_mitigation.return_value = sample_mitigation
            mock_service.update_mitigation.return_value = sample_mitigation
            mock_get_service.return_value = mock_service

            response = test_client.put(
                f"/api/v1/threat-models/mitigations/{sample_mitigation.mitigation_id}",
                headers=threat_model_admin_headers,
                json={
                    "status": "implemented",
                },
            )

            assert response.status_code == 200


class TestComponentEndpoints:
    """Test component management endpoints."""

    def test_add_component_to_model(
        self, test_client, threat_model_admin_headers, sample_threat_model
    ):
        """Test adding component to threat model."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_service.add_component.return_value = Component(
                component_id=str(uuid4()),
                name="new-service",
                component_type=ComponentType.SERVICE,
                description="New service",
                technology_stack=["Python"],
                data_classification="confidential",
                authentication_required=True,
                authorization_level="role_based",
                exposed_ports=[8000],
                protocols=["HTTPS"],
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/components",
                headers=threat_model_admin_headers,
                json={
                    "name": "new-service",
                    "component_type": "service",
                    "description": "New service",
                    "data_classification": "confidential",
                },
            )

            assert response.status_code == 201

    def test_remove_component_from_model(
        self, test_client, threat_model_admin_headers, sample_threat_model
    ):
        """Test removing component from threat model."""
        component_id = sample_threat_model.components[0].component_id if sample_threat_model.components else str(uuid4())

        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_service.remove_component.return_value = True
            mock_get_service.return_value = mock_service

            response = test_client.delete(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/components/{component_id}",
                headers=threat_model_admin_headers,
            )

            assert response.status_code in [200, 204]


class TestDataFlowEndpoints:
    """Test data flow management endpoints."""

    def test_add_data_flow_to_model(
        self, test_client, threat_model_admin_headers, sample_threat_model
    ):
        """Test adding data flow to threat model."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_service.add_data_flow.return_value = DataFlow(
                flow_id=str(uuid4()),
                name="New Flow",
                source_component_id=str(uuid4()),
                destination_component_id=str(uuid4()),
                data_types=["user_data"],
                protocol="HTTPS",
                encrypted=True,
                authentication="jwt",
                data_classification="confidential",
                metadata={},
            )
            mock_get_service.return_value = mock_service

            response = test_client.post(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/data-flows",
                headers=threat_model_admin_headers,
                json={
                    "name": "New Flow",
                    "source_component_id": str(uuid4()),
                    "destination_component_id": str(uuid4()),
                    "data_types": ["user_data"],
                    "protocol": "HTTPS",
                    "encrypted": True,
                },
            )

            assert response.status_code == 201


class TestExportEndpoints:
    """Test threat model export endpoints."""

    def test_export_threat_model_json(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test exporting threat model as JSON."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/export?format=json",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200
            assert response.headers.get("content-type") == "application/json"

    def test_export_threat_model_pdf(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test exporting threat model as PDF."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_service.export_to_pdf.return_value = b"PDF content"
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/export?format=pdf",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200


class TestMetricsEndpoint:
    """Test threat model metrics endpoint."""

    def test_get_threat_model_metrics(
        self, test_client, security_analyst_headers, sample_threat_model
    ):
        """Test getting threat model metrics."""
        with patch(
            "greenlang.infrastructure.threat_modeling.api.threat_routes.get_threat_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_threat_model.return_value = sample_threat_model
            mock_service.get_metrics.return_value = {
                "total_threats": 6,
                "threats_by_category": {
                    "spoofing": 1,
                    "tampering": 1,
                    "repudiation": 1,
                    "information_disclosure": 1,
                    "denial_of_service": 1,
                    "elevation_of_privilege": 1,
                },
                "threats_by_risk_level": {
                    "critical": 1,
                    "high": 3,
                    "medium": 1,
                    "low": 1,
                },
                "mitigations_implemented": 2,
                "mitigations_planned": 4,
            }
            mock_get_service.return_value = mock_service

            response = test_client.get(
                f"/api/v1/threat-models/{sample_threat_model.model_id}/metrics",
                headers=security_analyst_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert "total_threats" in data
            assert "threats_by_category" in data
