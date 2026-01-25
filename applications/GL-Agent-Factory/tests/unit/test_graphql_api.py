"""
Unit Tests for GreenLang Process Heat GraphQL API

Tests the GraphQL schema, types, resolvers, and middleware.

Run with:
    pytest tests/unit/test_graphql_api.py -v
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import json


# =============================================================================
# Test GraphQL Types
# =============================================================================


class TestAgentTypes:
    """Test GraphQL agent type definitions."""

    def test_agent_status_enum(self):
        """Test AgentStatusEnum values."""
        from app.graphql.types.agent import AgentStatusEnum

        assert AgentStatusEnum.AVAILABLE.value == "available"
        assert AgentStatusEnum.BUSY.value == "busy"
        assert AgentStatusEnum.DEGRADED.value == "degraded"
        assert AgentStatusEnum.OFFLINE.value == "offline"

    def test_agent_category_enum(self):
        """Test AgentCategoryEnum values."""
        from app.graphql.types.agent import AgentCategoryEnum

        assert AgentCategoryEnum.STEAM_SYSTEMS.value == "Steam Systems"
        assert AgentCategoryEnum.EMISSIONS.value == "Emissions"
        assert AgentCategoryEnum.HEAT_RECOVERY.value == "Heat Recovery"

    def test_process_heat_agent_type_creation(self):
        """Test ProcessHeatAgentType instantiation."""
        from app.graphql.types.agent import ProcessHeatAgentType, AgentStatusEnum

        agent = ProcessHeatAgentType(
            id="test-id",
            agent_id="GL-022",
            name="SUPERHEAT-CTRL",
            category="Steam Systems",
            type="Controller",
            complexity="Medium",
            priority="P2",
            status=AgentStatusEnum.AVAILABLE,
            health_score=100.0,
            last_run=None,
            description="Test agent",
            market_size="$5B",
            standards=["ISO 50001"],
            tags=["steam", "controller"],
            module_path="gl_022_superheater_control",
            class_name="SuperheaterControlAgent",
            version="1.0.0",
            deterministic=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        assert agent.agent_id == "GL-022"
        assert agent.name == "SUPERHEAT-CTRL"
        assert agent.category == "Steam Systems"
        assert agent.health_score == 100.0

    def test_health_status_type(self):
        """Test HealthStatusType instantiation."""
        from app.graphql.types.agent import HealthStatusType, HealthStatusLevel

        health = HealthStatusType(
            level=HealthStatusLevel.HEALTHY,
            score=95.5,
            last_check=datetime.now(timezone.utc),
            response_time_ms=150.0,
            error_rate=0.01,
            availability=99.9,
            message="Operating normally",
            issues=[],
        )

        assert health.level == HealthStatusLevel.HEALTHY
        assert health.score == 95.5
        assert health.availability == 99.9


class TestCalculationTypes:
    """Test GraphQL calculation type definitions."""

    def test_calculation_status_enum(self):
        """Test CalculationStatusEnum values."""
        from app.graphql.types.calculation import CalculationStatusEnum

        assert CalculationStatusEnum.PENDING.value == "pending"
        assert CalculationStatusEnum.RUNNING.value == "running"
        assert CalculationStatusEnum.COMPLETED.value == "completed"
        assert CalculationStatusEnum.FAILED.value == "failed"

    def test_data_quality_tier_enum(self):
        """Test DataQualityTier values."""
        from app.graphql.types.calculation import DataQualityTier

        assert DataQualityTier.TIER_1.value == "tier_1"
        assert DataQualityTier.TIER_2.value == "tier_2"

    def test_calculation_result_type(self):
        """Test CalculationResultType instantiation."""
        from app.graphql.types.calculation import (
            CalculationResultType,
            CalculationStatusEnum,
        )

        result = CalculationResultType(
            id="calc-123",
            execution_id="exec-456",
            agent_id="GL-022",
            status=CalculationStatusEnum.COMPLETED,
            progress_percent=100,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_ms=250.0,
            confidence_score=0.95,
            methodology="ghg_protocol",
            inputs={},
            metadata={},
            tenant_id="default",
        )

        assert result.id == "calc-123"
        assert result.status == CalculationStatusEnum.COMPLETED
        assert result.confidence_score == 0.95


class TestEventTypes:
    """Test GraphQL event type definitions."""

    def test_event_type_enum(self):
        """Test EventTypeEnum values."""
        from app.graphql.types.events import EventTypeEnum

        assert EventTypeEnum.AGENT_CREATED.value == "agent.created"
        assert EventTypeEnum.EXECUTION_STARTED.value == "execution.started"
        assert EventTypeEnum.CALCULATION_COMPLETED.value == "calculation.completed"

    def test_agent_event_type(self):
        """Test AgentEventType instantiation."""
        from app.graphql.types.events import (
            AgentEventType,
            EventTypeEnum,
            EventSeverityEnum,
            EventSourceEnum,
        )

        event = AgentEventType(
            event_id="evt-123",
            event_type=EventTypeEnum.AGENT_STATUS_CHANGED,
            timestamp=datetime.now(timezone.utc),
            source=EventSourceEnum.AGENT,
            agent_id="GL-022",
            severity=EventSeverityEnum.INFO,
            message="Agent status changed",
            data={},
            tenant_id="default",
        )

        assert event.event_id == "evt-123"
        assert event.event_type == EventTypeEnum.AGENT_STATUS_CHANGED
        assert event.agent_id == "GL-022"


# =============================================================================
# Test Middleware
# =============================================================================


class TestAuthMiddleware:
    """Test authentication middleware."""

    def test_permission_enum(self):
        """Test Permission enum values."""
        from app.graphql.middleware.auth import Permission

        assert Permission.AGENT_READ.value == "agent:read"
        assert Permission.AGENT_EXECUTE.value == "agent:execute"
        assert Permission.ADMIN_USERS.value == "admin:users"

    def test_role_enum(self):
        """Test Role enum values."""
        from app.graphql.middleware.auth import Role

        assert Role.VIEWER.value == "viewer"
        assert Role.OPERATOR.value == "operator"
        assert Role.ADMIN.value == "admin"

    def test_auth_context_creation(self):
        """Test AuthContext creation."""
        from app.graphql.middleware.auth import AuthContext, Role, Permission

        context = AuthContext(
            user_id="user-123",
            email="test@example.com",
            tenant_id="tenant-456",
            roles=[Role.OPERATOR],
            permissions={Permission.AGENT_READ, Permission.AGENT_EXECUTE},
            authenticated=True,
            auth_method="jwt",
        )

        assert context.user_id == "user-123"
        assert context.authenticated is True
        assert context.has_permission(Permission.AGENT_READ) is True
        assert context.has_permission(Permission.ADMIN_USERS) is False
        assert context.has_role(Role.OPERATOR) is True
        assert context.has_role(Role.ADMIN) is False

    def test_jwt_token_creation_and_verification(self):
        """Test JWT token creation and verification."""
        try:
            import jwt
        except ImportError:
            pytest.skip("PyJWT not installed")

        from app.graphql.middleware.auth import create_jwt_token, verify_jwt_token

        token = create_jwt_token(
            user_id="user-123",
            email="test@example.com",
            tenant_id="tenant-456",
            roles=["operator"],
            name="Test User",
        )

        assert token is not None
        assert isinstance(token, str)

        # Verify the token
        payload = verify_jwt_token(token)

        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"
        assert payload["tenant_id"] == "tenant-456"
        assert "operator" in payload["roles"]


class TestLoggingMiddleware:
    """Test logging middleware."""

    def test_graphql_metrics_creation(self):
        """Test GraphQLMetrics creation."""
        from app.graphql.middleware.logging import GraphQLMetrics

        metrics = GraphQLMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.success_rate == 1.0

    def test_graphql_metrics_recording(self):
        """Test metrics recording."""
        from app.graphql.middleware.logging import GraphQLMetrics

        metrics = GraphQLMetrics()

        metrics.record_request(
            operation_name="getAgent",
            operation_type="query",
            latency=0.5,
            success=True,
        )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.queries == 1
        assert metrics.average_latency == 0.5

        metrics.record_request(
            operation_name="runAgent",
            operation_type="mutation",
            latency=1.0,
            success=False,
            error_type="ValidationError",
        )

        assert metrics.total_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.mutations == 1
        assert "ValidationError" in metrics.errors_by_type


# =============================================================================
# Test Resolvers
# =============================================================================


class TestAgentResolvers:
    """Test agent query and mutation resolvers."""

    @pytest.mark.asyncio
    async def test_get_agent_resolver(self):
        """Test get_agent resolver with mock registry."""
        from app.graphql.resolvers.agents import get_agent_info_from_registry

        # Mock the registry
        with patch(
            "app.graphql.resolvers.agents.get_agent_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_info = MagicMock()
            mock_info.agent_id = "GL-022"
            mock_info.agent_name = "SUPERHEAT-CTRL"
            mock_info.module_path = "gl_022_superheater_control"
            mock_info.class_name = "SuperheaterControlAgent"
            mock_info.category = "Steam Systems"
            mock_info.agent_type = "Controller"
            mock_info.complexity = "Medium"
            mock_info.priority = "P2"
            mock_info.market_size = "$5B"
            mock_info.description = "Test agent"
            mock_info.standards = []
            mock_info.status = "Implemented"

            mock_registry.get_info.return_value = mock_info
            mock_get_registry.return_value = mock_registry

            result = get_agent_info_from_registry("GL-022")

            assert result is not None
            assert result["agent_id"] == "GL-022"
            assert result["agent_name"] == "SUPERHEAT-CTRL"
            assert result["category"] == "Steam Systems"

    @pytest.mark.asyncio
    async def test_list_agents_resolver(self):
        """Test list_agents resolver with mock registry."""
        from app.graphql.resolvers.agents import list_agents_from_registry

        with patch(
            "app.graphql.resolvers.agents.get_agent_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()

            mock_agent1 = MagicMock()
            mock_agent1.agent_id = "GL-022"
            mock_agent1.agent_name = "SUPERHEAT-CTRL"
            mock_agent1.module_path = "gl_022"
            mock_agent1.class_name = "SuperheaterControlAgent"
            mock_agent1.category = "Steam Systems"
            mock_agent1.agent_type = "Controller"
            mock_agent1.complexity = "Medium"
            mock_agent1.priority = "P2"
            mock_agent1.market_size = "$5B"
            mock_agent1.description = "Test"
            mock_agent1.standards = []
            mock_agent1.status = "Implemented"

            mock_agent2 = MagicMock()
            mock_agent2.agent_id = "GL-023"
            mock_agent2.agent_name = "LOADBALANCER"
            mock_agent2.module_path = "gl_023"
            mock_agent2.class_name = "HeatLoadBalancerAgent"
            mock_agent2.category = "Optimization"
            mock_agent2.agent_type = "Optimizer"
            mock_agent2.complexity = "High"
            mock_agent2.priority = "P1"
            mock_agent2.market_size = "$9B"
            mock_agent2.description = "Test"
            mock_agent2.standards = []
            mock_agent2.status = "Implemented"

            mock_registry.list_agents.return_value = [mock_agent1, mock_agent2]
            mock_get_registry.return_value = mock_registry

            results = list_agents_from_registry()

            assert len(results) == 2
            assert results[0]["agent_id"] == "GL-022"
            assert results[1]["agent_id"] == "GL-023"

            # Test with filter
            mock_registry.list_agents.return_value = [mock_agent1]
            filtered = list_agents_from_registry(category="Steam Systems")

            assert len(filtered) == 1
            assert filtered[0]["category"] == "Steam Systems"


# =============================================================================
# Test Schema
# =============================================================================


class TestSchema:
    """Test GraphQL schema configuration."""

    def test_schema_creation(self):
        """Test schema can be created."""
        from app.graphql.schema import get_schema

        schema = get_schema()

        assert schema is not None

    def test_schema_has_query(self):
        """Test schema has Query type."""
        from app.graphql.schema import Query

        # Check Query class exists and has expected methods
        assert hasattr(Query, "agent")
        assert hasattr(Query, "agents")
        assert hasattr(Query, "agent_health")
        assert hasattr(Query, "calculation")
        assert hasattr(Query, "registry_stats")
        assert hasattr(Query, "search_agents")

    def test_schema_has_mutation(self):
        """Test schema has Mutation type."""
        from app.graphql.schema import Mutation

        # Check Mutation class exists and has expected methods
        assert hasattr(Mutation, "run_agent")
        assert hasattr(Mutation, "configure_agent")

    def test_schema_has_subscription(self):
        """Test schema has Subscription type."""
        from app.graphql.schema import Subscription

        # Check Subscription class exists and has expected methods
        assert hasattr(Subscription, "agent_events")
        assert hasattr(Subscription, "calculation_progress")
        assert hasattr(Subscription, "system_events")

    def test_export_schema_sdl(self):
        """Test schema SDL export."""
        from app.graphql.schema import export_schema_sdl

        sdl = export_schema_sdl()

        assert sdl is not None
        assert isinstance(sdl, str)
        assert "Query" in sdl
        assert "Mutation" in sdl
        assert "Subscription" in sdl


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphQLIntegration:
    """Integration tests for GraphQL API."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test GraphQL health check function."""
        from app.graphql.schema import graphql_health_check

        health = await graphql_health_check()

        assert health["status"] == "healthy"
        assert health["schema_loaded"] is True
        assert "timestamp" in health
        assert "metrics" in health

    def test_create_graphql_app(self):
        """Test GraphQL router creation."""
        from app.graphql import create_graphql_app

        router = create_graphql_app(graphiql=True)

        assert router is not None

    def test_context_creation(self):
        """Test GreenLangContext creation."""
        from app.graphql.schema import GreenLangContext
        from app.graphql.middleware.auth import AuthContext

        mock_request = MagicMock()
        auth = AuthContext(
            user_id="user-123",
            tenant_id="tenant-456",
            authenticated=True,
        )

        context = GreenLangContext(request=mock_request, auth=auth)

        assert context.user_id == "user-123"
        assert context.tenant_id == "tenant-456"
        assert context.authenticated is True


# =============================================================================
# Example Queries (for documentation)
# =============================================================================


EXAMPLE_QUERIES = {
    "get_agent": """
        query GetAgent {
            agent(id: "GL-022") {
                id
                agentId
                name
                category
                type
                status
                healthScore
                description
            }
        }
    """,
    "list_agents": """
        query ListAgents {
            agents(category: "Steam Systems", first: 10) {
                edges {
                    node {
                        agentId
                        name
                        status
                    }
                    cursor
                }
                pageInfo {
                    hasNextPage
                    totalCount
                }
            }
        }
    """,
    "search_agents": """
        query SearchAgents {
            searchAgents(query: "steam optimizer", limit: 5) {
                agentId
                name
                category
                description
            }
        }
    """,
    "run_agent": """
        mutation RunAgent {
            runAgent(
                id: "GL-022",
                input: {
                    steamPressure: 150,
                    feedwaterTemp: 220,
                    fuelType: "natural_gas"
                }
            ) {
                success
                calculation {
                    id
                    status
                    progressPercent
                    result {
                        value
                        unit
                    }
                }
                error
            }
        }
    """,
    "agent_events": """
        subscription AgentEvents {
            agentEvents(agentId: "GL-022") {
                eventId
                eventType
                timestamp
                severity
                message
                data
            }
        }
    """,
    "calculation_progress": """
        subscription CalculationProgress {
            calculationProgress(calculationId: "calc-abc123") {
                calculationId
                status
                progress {
                    percent
                    currentStep
                    estimatedRemainingSeconds
                }
                intermediateValue
                intermediateUnit
            }
        }
    """,
}


class TestExampleQueries:
    """Test that example queries are valid GraphQL."""

    def test_example_queries_parse(self):
        """Test that example queries are syntactically valid."""
        # This is a basic validation - full validation requires graphql-core
        for name, query in EXAMPLE_QUERIES.items():
            assert "query" in query or "mutation" in query or "subscription" in query, (
                f"Query '{name}' should contain query, mutation, or subscription"
            )
            assert "{" in query and "}" in query, (
                f"Query '{name}' should contain braces"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
