"""
GL-001 ThermalCommand Orchestrator - REST/GraphQL API Module

This module provides REST and GraphQL API endpoints for the ThermalCommand
Orchestrator, enabling external access to orchestration capabilities.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(default=None, description="Request ID")


class WorkflowRequest(BaseModel):
    """Request to execute a workflow."""

    workflow_type: str = Field(..., description="Workflow type identifier")
    name: str = Field(..., description="Workflow name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow parameters"
    )
    priority: str = Field(default="normal", description="Priority level")
    timeout_s: float = Field(default=300.0, description="Timeout in seconds")


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent."""

    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Agent type")
    name: str = Field(..., description="Agent name")
    capabilities: List[str] = Field(
        default_factory=list,
        description="Agent capabilities"
    )
    endpoint: Optional[str] = Field(default=None, description="Agent endpoint")


class SafetyPermitRequest(BaseModel):
    """Request for a safety permit."""

    permit_type: str = Field(..., description="Permit type")
    equipment_id: str = Field(..., description="Equipment identifier")
    requested_by: str = Field(..., description="Requester name")
    duration_hours: float = Field(default=8.0, description="Duration in hours")


# =============================================================================
# REST API CONTROLLER
# =============================================================================

class RESTAPIController:
    """
    REST API controller for ThermalCommand Orchestrator.

    Provides RESTful endpoints for orchestrator operations.
    In production, this would be integrated with FastAPI/Flask.
    """

    def __init__(self, orchestrator: Any) -> None:
        """
        Initialize the REST API controller.

        Args:
            orchestrator: ThermalCommandOrchestrator instance
        """
        self._orchestrator = orchestrator
        self._routes: Dict[str, Dict[str, callable]] = {}
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes."""
        self._routes = {
            "/api/v1/status": {
                "GET": self.get_status,
            },
            "/api/v1/agents": {
                "GET": self.list_agents,
                "POST": self.register_agent,
            },
            "/api/v1/agents/{agent_id}": {
                "GET": self.get_agent,
                "DELETE": self.deregister_agent,
            },
            "/api/v1/workflows": {
                "GET": self.list_workflows,
                "POST": self.execute_workflow,
            },
            "/api/v1/workflows/{workflow_id}": {
                "GET": self.get_workflow,
                "DELETE": self.cancel_workflow,
            },
            "/api/v1/safety/status": {
                "GET": self.get_safety_status,
            },
            "/api/v1/safety/esd": {
                "POST": self.trigger_esd,
                "DELETE": self.reset_esd,
            },
            "/api/v1/safety/permits": {
                "GET": self.list_permits,
                "POST": self.request_permit,
            },
            "/api/v1/metrics": {
                "GET": self.get_metrics,
            },
        }

    async def get_status(self) -> APIResponse:
        """GET /api/v1/status - Get system status."""
        try:
            status = self._orchestrator.get_system_status()
            return APIResponse(success=True, data=status.dict())
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return APIResponse(success=False, error=str(e))

    async def list_agents(self) -> APIResponse:
        """GET /api/v1/agents - List registered agents."""
        try:
            agents = []
            for agent_id in self._orchestrator._registered_agents:
                status = self._orchestrator.get_agent_status(agent_id)
                if status:
                    agents.append(status.dict())
            return APIResponse(success=True, data={"agents": agents})
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return APIResponse(success=False, error=str(e))

    async def register_agent(
        self,
        request: AgentRegistrationRequest,
    ) -> APIResponse:
        """POST /api/v1/agents - Register a new agent."""
        try:
            from greenlang.agents.process_heat.shared.coordination import (
                AgentRegistration,
                AgentRole,
            )

            registration = AgentRegistration(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                name=request.name,
                role=AgentRole.WORKER,
                capabilities=set(request.capabilities),
                endpoint=request.endpoint,
            )

            success = self._orchestrator.register_agent(registration)

            if success:
                return APIResponse(
                    success=True,
                    data={"agent_id": request.agent_id, "registered": True}
                )
            else:
                return APIResponse(
                    success=False,
                    error="Agent already registered"
                )

        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return APIResponse(success=False, error=str(e))

    async def get_agent(self, agent_id: str) -> APIResponse:
        """GET /api/v1/agents/{agent_id} - Get agent status."""
        try:
            status = self._orchestrator.get_agent_status(agent_id)
            if status:
                return APIResponse(success=True, data=status.dict())
            else:
                return APIResponse(success=False, error="Agent not found")
        except Exception as e:
            logger.error(f"Error getting agent: {e}")
            return APIResponse(success=False, error=str(e))

    async def deregister_agent(self, agent_id: str) -> APIResponse:
        """DELETE /api/v1/agents/{agent_id} - Deregister an agent."""
        try:
            success = self._orchestrator.deregister_agent(agent_id)
            return APIResponse(success=success, data={"deregistered": success})
        except Exception as e:
            logger.error(f"Error deregistering agent: {e}")
            return APIResponse(success=False, error=str(e))

    async def list_workflows(self) -> APIResponse:
        """GET /api/v1/workflows - List active workflows."""
        try:
            workflows = self._orchestrator._workflow_coordinator.get_active_workflows()
            return APIResponse(success=True, data={"workflows": workflows})
        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            return APIResponse(success=False, error=str(e))

    async def execute_workflow(self, request: WorkflowRequest) -> APIResponse:
        """POST /api/v1/workflows - Execute a workflow."""
        try:
            from greenlang.agents.process_heat.gl_001_thermal_command.schemas import (
                WorkflowSpec,
                WorkflowType,
                Priority,
            )

            # Map priority
            priority_map = {
                "low": Priority.LOW,
                "normal": Priority.NORMAL,
                "high": Priority.HIGH,
                "critical": Priority.CRITICAL,
            }

            spec = WorkflowSpec(
                workflow_type=WorkflowType.OPTIMIZATION,
                name=request.name,
                priority=priority_map.get(request.priority.lower(), Priority.NORMAL),
                timeout_s=request.timeout_s,
                parameters=request.parameters,
            )

            result = await self._orchestrator.execute_workflow(spec)
            return APIResponse(success=True, data=result.dict())

        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return APIResponse(success=False, error=str(e))

    async def get_workflow(self, workflow_id: str) -> APIResponse:
        """GET /api/v1/workflows/{workflow_id} - Get workflow status."""
        try:
            status = self._orchestrator._workflow_coordinator.get_workflow_status(
                workflow_id
            )
            if status:
                return APIResponse(
                    success=True,
                    data={"workflow_id": workflow_id, "status": status.value if hasattr(status, 'value') else status}
                )
            else:
                return APIResponse(success=False, error="Workflow not found")
        except Exception as e:
            logger.error(f"Error getting workflow: {e}")
            return APIResponse(success=False, error=str(e))

    async def cancel_workflow(self, workflow_id: str) -> APIResponse:
        """DELETE /api/v1/workflows/{workflow_id} - Cancel a workflow."""
        try:
            success = await self._orchestrator.cancel_workflow(workflow_id)
            return APIResponse(success=success, data={"cancelled": success})
        except Exception as e:
            logger.error(f"Error cancelling workflow: {e}")
            return APIResponse(success=False, error=str(e))

    async def get_safety_status(self) -> APIResponse:
        """GET /api/v1/safety/status - Get safety system status."""
        try:
            return APIResponse(
                success=True,
                data={
                    "state": self._orchestrator._safety_coordinator.safety_state,
                    "esd_triggered": self._orchestrator._safety_coordinator.is_esd_triggered,
                    "alarms": self._orchestrator._event_handlers["safety"].get_alarm_summary(),
                }
            )
        except Exception as e:
            logger.error(f"Error getting safety status: {e}")
            return APIResponse(success=False, error=str(e))

    async def trigger_esd(self, reason: str) -> APIResponse:
        """POST /api/v1/safety/esd - Trigger emergency shutdown."""
        try:
            await self._orchestrator.trigger_emergency_shutdown(reason)
            return APIResponse(success=True, data={"esd_triggered": True})
        except Exception as e:
            logger.error(f"Error triggering ESD: {e}")
            return APIResponse(success=False, error=str(e))

    async def reset_esd(self, authorized_by: str) -> APIResponse:
        """DELETE /api/v1/safety/esd - Reset emergency shutdown."""
        try:
            success = await self._orchestrator.reset_emergency_shutdown(authorized_by)
            return APIResponse(success=success, data={"esd_reset": success})
        except Exception as e:
            logger.error(f"Error resetting ESD: {e}")
            return APIResponse(success=False, error=str(e))

    async def list_permits(self) -> APIResponse:
        """GET /api/v1/safety/permits - List safety permits."""
        try:
            permits = list(self._orchestrator._safety_coordinator._permits.values())
            return APIResponse(
                success=True,
                data={"permits": [p.dict() for p in permits]}
            )
        except Exception as e:
            logger.error(f"Error listing permits: {e}")
            return APIResponse(success=False, error=str(e))

    async def request_permit(self, request: SafetyPermitRequest) -> APIResponse:
        """POST /api/v1/safety/permits - Request a safety permit."""
        try:
            permit_id = self._orchestrator._safety_coordinator.request_permit(
                permit_type=request.permit_type,
                equipment_id=request.equipment_id,
                requested_by=request.requested_by,
                duration_hours=request.duration_hours,
            )

            if permit_id:
                return APIResponse(success=True, data={"permit_id": permit_id})
            else:
                return APIResponse(success=False, error="Permit denied")

        except Exception as e:
            logger.error(f"Error requesting permit: {e}")
            return APIResponse(success=False, error=str(e))

    async def get_metrics(self) -> APIResponse:
        """GET /api/v1/metrics - Get orchestrator metrics."""
        try:
            metrics = self._orchestrator.get_metrics()
            return APIResponse(success=True, data=metrics)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return APIResponse(success=False, error=str(e))


# =============================================================================
# GRAPHQL SCHEMA
# =============================================================================

GRAPHQL_SCHEMA = '''
type Query {
    status: SystemStatus!
    agents: [Agent!]!
    agent(id: ID!): Agent
    workflows: [Workflow!]!
    workflow(id: ID!): Workflow
    safetyStatus: SafetyStatus!
    metrics: Metrics!
}

type Mutation {
    registerAgent(input: AgentRegistrationInput!): AgentRegistrationResult!
    deregisterAgent(id: ID!): Boolean!
    executeWorkflow(input: WorkflowInput!): WorkflowResult!
    cancelWorkflow(id: ID!): Boolean!
    triggerESD(reason: String!): Boolean!
    resetESD(authorizedBy: String!): Boolean!
    requestPermit(input: PermitInput!): PermitResult!
}

type SystemStatus {
    orchestratorId: ID!
    orchestratorName: String!
    status: String!
    uptimeSeconds: Float!
    registeredAgents: Int!
    healthyAgents: Int!
    activeWorkflows: Int!
    safetyLevel: String!
    safetyStatus: String!
}

type Agent {
    id: ID!
    type: String!
    name: String!
    health: String!
    version: String!
    lastHeartbeat: String!
    activeTasks: Int!
    capabilities: [String!]!
}

type Workflow {
    id: ID!
    name: String!
    status: String!
    startTime: String!
    endTime: String
    durationMs: Float!
    tasksCompleted: Int!
    tasksFailed: Int!
    tasksTotal: Int!
}

type SafetyStatus {
    state: String!
    esdTriggered: Boolean!
    activeAlarms: Int!
    alarms: [Alarm!]!
}

type Alarm {
    id: ID!
    type: String!
    severity: String!
    description: String!
    acknowledged: Boolean!
}

type Metrics {
    workflowsExecuted: Int!
    workflowsFailed: Int!
    tasksExecuted: Int!
    safetyEvents: Int!
    uptimeSeconds: Float!
}

type WorkflowResult {
    workflowId: ID!
    status: String!
    error: String
}

type AgentRegistrationResult {
    agentId: ID!
    registered: Boolean!
    error: String
}

type PermitResult {
    permitId: ID
    approved: Boolean!
    error: String
}

input AgentRegistrationInput {
    agentId: ID!
    agentType: String!
    name: String!
    capabilities: [String!]!
    endpoint: String
}

input WorkflowInput {
    workflowType: String!
    name: String!
    parameters: JSON
    priority: String
    timeoutS: Float
}

input PermitInput {
    permitType: String!
    equipmentId: String!
    requestedBy: String!
    durationHours: Float
}

scalar JSON
'''


class GraphQLController:
    """
    GraphQL controller for ThermalCommand Orchestrator.

    Provides GraphQL API for orchestrator operations.
    In production, this would be integrated with Strawberry/Ariadne.
    """

    def __init__(self, orchestrator: Any) -> None:
        """Initialize GraphQL controller."""
        self._orchestrator = orchestrator
        self._schema = GRAPHQL_SCHEMA

    @property
    def schema(self) -> str:
        """Get GraphQL schema."""
        return self._schema

    async def execute(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """
        Execute a GraphQL query.

        In production, this would use a proper GraphQL library.
        This is a simplified implementation for demonstration.
        """
        # Placeholder - in production use Strawberry/Ariadne
        return {
            "data": None,
            "errors": [{"message": "GraphQL execution not implemented"}]
        }
