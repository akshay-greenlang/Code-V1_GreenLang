# -*- coding: utf-8 -*-
"""
Integration Tests for GraphQL API
End-to-end tests for the complete GraphQL API
"""

import pytest
from fastapi.testclient import TestClient
import json

from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager
from greenlang.api.graphql.server import create_graphql_app


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def test_services():
    """Create test services"""
    orchestrator = Orchestrator()
    auth_manager = AuthManager()
    rbac_manager = RBACManager()

    # Create test user
    user_id = auth_manager.create_user(
        tenant_id="test-tenant",
        username="testuser",
        email="test@example.com",
        password="testpass123",
    )

    # Assign admin role
    rbac_manager.assign_role(user_id, "super_admin")

    # Create auth token
    token = auth_manager.create_token(
        tenant_id="test-tenant",
        user_id=user_id,
        name="Test Token",
    )

    return {
        "orchestrator": orchestrator,
        "auth_manager": auth_manager,
        "rbac_manager": rbac_manager,
        "user_id": user_id,
        "token": token.token_value,
    }


@pytest.fixture
def test_client(test_services):
    """Create test client"""
    app = create_graphql_app(
        orchestrator=test_services["orchestrator"],
        auth_manager=test_services["auth_manager"],
        rbac_manager=test_services["rbac_manager"],
        enable_playground=True,
        enable_introspection=True,
        debug=True,
    )

    client = TestClient(app)
    client.headers = {"Authorization": f"Bearer {test_services['token']}"}

    return client


# ==============================================================================
# Health Check Tests
# ==============================================================================

def test_health_endpoint(test_client):
    """Test health check endpoint"""
    response = test_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metrics_endpoint(test_client):
    """Test metrics endpoint"""
    response = test_client.get("/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "agents_total" in data
    assert "workflows_total" in data


def test_playground_endpoint(test_client):
    """Test playground endpoint"""
    response = test_client.get("/playground")

    assert response.status_code == 200
    assert "GraphQL Playground" in response.text


# ==============================================================================
# Query Integration Tests
# ==============================================================================

def test_system_health_query(test_client):
    """Test system health query"""
    query = """
    query {
      systemHealth {
        status
        version
        agentCount
        workflowCount
        checks {
          name
          status
        }
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"]["systemHealth"]["status"] == "healthy"


def test_current_user_query(test_client):
    """Test current user query"""
    query = """
    query {
      currentUser {
        username
        email
        active
        roles {
          name
        }
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["currentUser"]["username"] == "testuser"


def test_agents_query(test_client):
    """Test agents list query"""
    query = """
    query {
      agents(pagination: { page: 1, pageSize: 10 }) {
        nodes {
          id
          name
          version
        }
        pageInfo {
          totalCount
          hasNextPage
        }
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "agents" in data["data"]


def test_workflows_query(test_client):
    """Test workflows list query"""
    query = """
    query {
      workflows(pagination: { page: 1, pageSize: 10 }) {
        nodes {
          id
          name
          version
        }
        pageInfo {
          totalCount
        }
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert "data" in data


def test_query_with_variables(test_client):
    """Test query with variables"""
    query = """
    query CheckPerm($resource: String!, $action: String!) {
      checkPermission(resource: $resource, action: $action)
    }
    """

    variables = {
        "resource": "agent",
        "action": "read",
    }

    response = test_client.post(
        "/graphql",
        json={"query": query, "variables": variables},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["checkPermission"] is True


# ==============================================================================
# Mutation Integration Tests
# ==============================================================================

def test_create_agent_mutation(test_client):
    """Test create agent mutation"""
    mutation = """
    mutation CreateAgent($input: CreateAgentInput!) {
      createAgent(input: $input) {
        id
        name
        description
        version
        enabled
      }
    }
    """

    variables = {
        "input": {
            "name": "Integration Test Agent",
            "description": "Created in integration test",
            "version": "1.0.0",
            "enabled": True,
        }
    }

    response = test_client.post(
        "/graphql",
        json={"query": mutation, "variables": variables},
    )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    agent = data["data"]["createAgent"]
    assert agent["name"] == "Integration Test Agent"


def test_create_and_update_agent(test_client):
    """Test create and update agent"""
    # Create agent
    create_mutation = """
    mutation CreateAgent($input: CreateAgentInput!) {
      createAgent(input: $input) {
        id
        name
        enabled
      }
    }
    """

    create_vars = {
        "input": {
            "name": "Agent to Update",
            "description": "Will be updated",
        }
    }

    create_response = test_client.post(
        "/graphql",
        json={"query": create_mutation, "variables": create_vars},
    )

    agent_id = create_response.json()["data"]["createAgent"]["id"]

    # Update agent
    update_mutation = """
    mutation UpdateAgent($id: ID!, $input: UpdateAgentInput!) {
      updateAgent(id: $id, input: $input) {
        id
        name
        enabled
      }
    }
    """

    update_vars = {
        "id": agent_id,
        "input": {
            "enabled": False,
        }
    }

    update_response = test_client.post(
        "/graphql",
        json={"query": update_mutation, "variables": update_vars},
    )

    assert update_response.status_code == 200
    updated_agent = update_response.json()["data"]["updateAgent"]
    assert updated_agent["enabled"] is False


def test_create_workflow_mutation(test_client, test_services):
    """Test create workflow mutation"""
    # First create an agent to use in workflow
    from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult

    class TestAgent(BaseAgent):
        def execute(self, input_data):
            return AgentResult(success=True, data={})

    agent = TestAgent(AgentConfig(name="Test", description="Test"))
    agent_id = "test-agent-workflow"
    test_services["orchestrator"].register_agent(agent_id, agent)

    mutation = """
    mutation CreateWorkflow($input: CreateWorkflowInput!) {
      createWorkflow(input: $input) {
        id
        name
        steps {
          name
          agentId
        }
      }
    }
    """

    variables = {
        "input": {
            "name": "Integration Test Workflow",
            "description": "Created in integration test",
            "steps": [
                {
                    "name": "Step 1",
                    "agentId": agent_id,
                }
            ],
        }
    }

    response = test_client.post(
        "/graphql",
        json={"query": mutation, "variables": variables},
    )

    assert response.status_code == 200
    data = response.json()
    workflow = data["data"]["createWorkflow"]
    assert workflow["name"] == "Integration Test Workflow"
    assert len(workflow["steps"]) == 1


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_unauthorized_request(test_services):
    """Test request without auth token"""
    app = create_graphql_app(
        orchestrator=test_services["orchestrator"],
        auth_manager=test_services["auth_manager"],
        rbac_manager=test_services["rbac_manager"],
    )

    client = TestClient(app)

    query = """
    query {
      currentUser {
        username
      }
    }
    """

    response = client.post("/graphql", json={"query": query})

    # Should fail without token
    assert response.status_code == 403 or "errors" in response.json()


def test_invalid_query(test_client):
    """Test invalid GraphQL query"""
    query = """
    query {
      invalidField {
        doesNotExist
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    # Should return error
    data = response.json()
    assert "errors" in data


def test_malformed_query(test_client):
    """Test malformed GraphQL syntax"""
    query = "this is not valid GraphQL"

    response = test_client.post("/graphql", json={"query": query})

    data = response.json()
    assert "errors" in data


# ==============================================================================
# Pagination Tests
# ==============================================================================

def test_pagination(test_client, test_services):
    """Test pagination in list queries"""
    # Create multiple agents
    from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult

    class TestAgent(BaseAgent):
        def execute(self, input_data):
            return AgentResult(success=True, data={})

    for i in range(5):
        agent = TestAgent(AgentConfig(name=f"Agent {i}", description=f"Agent {i}"))
        test_services["orchestrator"].register_agent(f"agent-{i}", agent)

    query = """
    query($page: Int, $pageSize: Int) {
      agents(pagination: { page: $page, pageSize: $pageSize }) {
        nodes {
          id
          name
        }
        pageInfo {
          totalCount
          currentPage
          hasNextPage
        }
      }
    }
    """

    # Page 1
    response = test_client.post(
        "/graphql",
        json={"query": query, "variables": {"page": 1, "pageSize": 2}},
    )

    data = response.json()["data"]["agents"]
    assert len(data["nodes"]) <= 2
    assert data["pageInfo"]["currentPage"] == 1


# ==============================================================================
# Complex Query Tests
# ==============================================================================

def test_nested_query(test_client):
    """Test nested query with multiple levels"""
    query = """
    query {
      currentUser {
        username
        roles {
          name
          permissions {
            resource
            action
          }
        }
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    user = data["data"]["currentUser"]
    assert "roles" in user
    if user["roles"]:
        assert "permissions" in user["roles"][0]


def test_multiple_queries(test_client):
    """Test multiple queries in single request"""
    query = """
    query {
      health: systemHealth {
        status
      }
      user: currentUser {
        username
      }
    }
    """

    response = test_client.post("/graphql", json={"query": query})

    assert response.status_code == 200
    data = response.json()["data"]
    assert "health" in data
    assert "user" in data
