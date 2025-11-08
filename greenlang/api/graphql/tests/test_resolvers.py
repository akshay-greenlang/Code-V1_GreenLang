"""
Unit Tests for GraphQL Resolvers
Tests all query and mutation resolvers
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import strawberry

from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.core.workflow import Workflow, WorkflowStep
from greenlang.api.graphql.context import GraphQLContext
from greenlang.api.graphql.resolvers import Query, Mutation, _convert_agent
from greenlang.api.graphql.types import (
    CreateAgentInput,
    UpdateAgentInput,
    CreateWorkflowInput,
    WorkflowStepInput,
    ExecuteWorkflowInput,
    PaginationInput,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def orchestrator():
    """Create test orchestrator"""
    return Orchestrator()


@pytest.fixture
def auth_manager():
    """Create test auth manager"""
    return AuthManager()


@pytest.fixture
def rbac_manager():
    """Create test RBAC manager"""
    return RBACManager()


@pytest.fixture
def test_user(auth_manager, rbac_manager):
    """Create test user with permissions"""
    user_id = auth_manager.create_user(
        tenant_id="test-tenant",
        username="testuser",
        email="test@example.com",
        password="testpass123",
    )

    # Assign super admin role for testing
    rbac_manager.assign_role(user_id, "super_admin")

    return user_id


@pytest.fixture
def context(orchestrator, auth_manager, rbac_manager, test_user):
    """Create test GraphQL context"""
    return GraphQLContext(
        user_id=test_user,
        tenant_id="test-tenant",
        orchestrator=orchestrator,
        auth_manager=auth_manager,
        rbac_manager=rbac_manager,
    )


@pytest.fixture
def mock_info(context):
    """Create mock Strawberry info object"""
    info = Mock()
    info.context = context
    return info


@pytest.fixture
def test_agent(orchestrator):
    """Create and register test agent"""
    class TestAgent(BaseAgent):
        def execute(self, input_data):
            return AgentResult(
                success=True,
                data={"result": "test output"},
            )

    config = AgentConfig(
        name="Test Agent",
        description="Agent for testing",
        version="1.0.0",
    )

    agent = TestAgent(config)
    agent_id = "test-agent-1"
    orchestrator.register_agent(agent_id, agent)

    return agent_id, agent


# ==============================================================================
# Query Tests
# ==============================================================================

class TestQueryResolvers:
    """Test Query resolvers"""

    @pytest.mark.asyncio
    async def test_agent_query(self, mock_info, test_agent):
        """Test agent query resolver"""
        agent_id, _ = test_agent
        query = Query()

        agent = await query.agent(mock_info, strawberry.ID(agent_id))

        assert agent is not None
        assert agent.name == "Test Agent"
        assert agent.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_agents_query(self, mock_info, test_agent):
        """Test agents list query"""
        query = Query()

        result = await query.agents(
            mock_info,
            pagination=PaginationInput(page=1, page_size=10),
        )

        assert result is not None
        assert result.total_count >= 1
        assert len(result.nodes) >= 1

    @pytest.mark.asyncio
    async def test_agent_not_found(self, mock_info):
        """Test agent query with non-existent ID"""
        query = Query()

        agent = await query.agent(mock_info, strawberry.ID("non-existent"))

        assert agent is None

    @pytest.mark.asyncio
    async def test_workflows_query(self, mock_info, orchestrator):
        """Test workflows query"""
        # Create test workflow
        workflow = Workflow(
            name="Test Workflow",
            description="Workflow for testing",
            version="1.0.0",
            steps=[],
        )
        orchestrator.register_workflow("test-workflow-1", workflow)

        query = Query()
        result = await query.workflows(
            mock_info,
            pagination=PaginationInput(page=1, page_size=10),
        )

        assert result is not None
        assert result.total_count >= 1

    def test_current_user(self, mock_info):
        """Test current user query"""
        query = Query()

        user = query.current_user(mock_info)

        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_system_health(self, mock_info):
        """Test system health query"""
        query = Query()

        health = query.system_health(mock_info)

        assert health is not None
        assert health.status == "healthy"
        assert health.agent_count >= 0

    def test_check_permission_allowed(self, mock_info):
        """Test permission check (allowed)"""
        query = Query()

        result = query.check_permission(
            mock_info,
            resource="agent",
            action="read",
        )

        assert result is True

    def test_check_permission_denied(self, mock_info, rbac_manager):
        """Test permission check (denied)"""
        # Remove all roles
        rbac_manager.user_roles[mock_info.context.user_id] = set()

        query = Query()

        result = query.check_permission(
            mock_info,
            resource="agent",
            action="admin",
        )

        assert result is False


# ==============================================================================
# Mutation Tests
# ==============================================================================

class TestMutationResolvers:
    """Test Mutation resolvers"""

    @pytest.mark.asyncio
    async def test_create_agent(self, mock_info):
        """Test create agent mutation"""
        mutation = Mutation()

        input_data = CreateAgentInput(
            name="New Agent",
            description="Created by test",
            version="1.0.0",
            enabled=True,
        )

        agent = await mutation.create_agent(mock_info, input_data)

        assert agent is not None
        assert agent.name == "New Agent"
        assert agent.enabled is True

    @pytest.mark.asyncio
    async def test_update_agent(self, mock_info, test_agent):
        """Test update agent mutation"""
        agent_id, _ = test_agent
        mutation = Mutation()

        input_data = UpdateAgentInput(
            description="Updated description",
            enabled=False,
        )

        agent = await mutation.update_agent(
            mock_info,
            strawberry.ID(agent_id),
            input_data,
        )

        assert agent is not None
        assert agent.description == "Updated description"
        assert agent.enabled is False

    @pytest.mark.asyncio
    async def test_delete_agent(self, mock_info, test_agent):
        """Test delete agent mutation"""
        agent_id, _ = test_agent
        mutation = Mutation()

        result = await mutation.delete_agent(
            mock_info,
            strawberry.ID(agent_id),
        )

        assert result is True

        # Verify deletion
        assert agent_id not in mock_info.context.orchestrator.agents

    @pytest.mark.asyncio
    async def test_create_workflow(self, mock_info, test_agent):
        """Test create workflow mutation"""
        agent_id, _ = test_agent
        mutation = Mutation()

        input_data = CreateWorkflowInput(
            name="Test Workflow",
            description="Created by test",
            version="1.0.0",
            steps=[
                WorkflowStepInput(
                    name="Step 1",
                    agent_id=strawberry.ID(agent_id),
                    description="Test step",
                ),
            ],
        )

        workflow = await mutation.create_workflow(mock_info, input_data)

        assert workflow is not None
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 1

    @pytest.mark.asyncio
    async def test_execute_workflow(self, mock_info, orchestrator, test_agent):
        """Test execute workflow mutation"""
        agent_id, _ = test_agent

        # Create workflow
        workflow = Workflow(
            name="Test Workflow",
            description="For testing",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    name="Step 1",
                    agent_id=agent_id,
                ),
            ],
        )
        workflow_id = "test-workflow-exec"
        orchestrator.register_workflow(workflow_id, workflow)

        mutation = Mutation()

        input_data = ExecuteWorkflowInput(
            workflow_id=strawberry.ID(workflow_id),
            input_data={"test": "data"},
        )

        result = await mutation.execute_workflow(mock_info, input_data)

        assert result is not None
        assert result.execution is not None
        # Success depends on workflow execution

    @pytest.mark.asyncio
    async def test_create_role(self, mock_info):
        """Test create role mutation"""
        from greenlang.api.graphql.types import CreateRoleInput, PermissionInput

        mutation = Mutation()

        input_data = CreateRoleInput(
            name="test_role",
            description="Test role",
            permissions=[
                PermissionInput(
                    resource="agent",
                    action="read",
                ),
            ],
        )

        role = mutation.create_role(mock_info, input_data)

        assert role is not None
        assert role.name == "test_role"
        assert len(role.permissions) >= 1

    @pytest.mark.asyncio
    async def test_batch_create_agents(self, mock_info):
        """Test batch create agents"""
        mutation = Mutation()

        inputs = [
            CreateAgentInput(
                name=f"Batch Agent {i}",
                description="Created in batch",
                version="1.0.0",
            )
            for i in range(3)
        ]

        agents = await mutation.batch_create_agents(mock_info, inputs)

        assert len(agents) == 3
        assert all(agent.name.startswith("Batch Agent") for agent in agents)


# ==============================================================================
# Permission Tests
# ==============================================================================

class TestPermissions:
    """Test permission checks in resolvers"""

    @pytest.mark.asyncio
    async def test_query_without_permission(self, mock_info, rbac_manager):
        """Test query fails without permission"""
        # Remove all permissions
        rbac_manager.user_roles[mock_info.context.user_id] = set()

        query = Query()

        with pytest.raises(PermissionError):
            await query.agents(mock_info)

    @pytest.mark.asyncio
    async def test_mutation_without_permission(self, mock_info, rbac_manager):
        """Test mutation fails without permission"""
        # Remove all permissions
        rbac_manager.user_roles[mock_info.context.user_id] = set()

        mutation = Mutation()
        input_data = CreateAgentInput(
            name="Unauthorized",
            description="Should fail",
        )

        with pytest.raises(PermissionError):
            await mutation.create_agent(mock_info, input_data)


# ==============================================================================
# DataLoader Tests
# ==============================================================================

class TestDataLoaders:
    """Test DataLoader functionality"""

    @pytest.mark.asyncio
    async def test_agent_loader_batching(self, context, test_agent):
        """Test agent loader batches requests"""
        agent_id, _ = test_agent

        # Load same agent multiple times
        results = await asyncio.gather(
            context.agent_loader.load(agent_id),
            context.agent_loader.load(agent_id),
            context.agent_loader.load(agent_id),
        )

        # All results should be the same
        assert all(r is not None for r in results)
        assert len(set(r.id for r in results)) == 1

    @pytest.mark.asyncio
    async def test_agent_loader_caching(self, context, test_agent):
        """Test agent loader caches results"""
        agent_id, _ = test_agent

        # First load
        agent1 = await context.agent_loader.load(agent_id)

        # Second load should use cache
        agent2 = await context.agent_loader.load(agent_id)

        assert agent1 is agent2  # Same object from cache


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Test error handling in resolvers"""

    @pytest.mark.asyncio
    async def test_invalid_agent_id(self, mock_info):
        """Test handling of invalid agent ID"""
        query = Query()

        agent = await query.agent(mock_info, strawberry.ID("invalid"))

        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_agent(self, mock_info):
        """Test deleting non-existent agent"""
        mutation = Mutation()

        result = await mutation.delete_agent(
            mock_info,
            strawberry.ID("nonexistent"),
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_nonexistent_agent(self, mock_info):
        """Test updating non-existent agent"""
        mutation = Mutation()

        input_data = UpdateAgentInput(description="Update")

        with pytest.raises(ValueError):
            await mutation.update_agent(
                mock_info,
                strawberry.ID("nonexistent"),
                input_data,
            )
