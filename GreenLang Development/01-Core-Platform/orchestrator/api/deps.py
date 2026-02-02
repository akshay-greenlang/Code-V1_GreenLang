# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Control Plane API - Dependency Injection
================================================================

This module provides FastAPI dependency injection for the Control Plane API.

Dependencies include:
- Orchestrator instance (GLIPOrchestrator)
- Agent Registry (VersionedAgentRegistry)
- Event Store (for audit trails)
- API Key authentication
- Request tracing
- Rate limiting context

All dependencies are designed for async/await usage and proper resource cleanup.

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Control Plane API Dependencies
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class APIConfig:
    """
    API Configuration settings.

    Loaded from environment variables or config files.
    """

    def __init__(
        self,
        api_version: str = "1.0.0",
        api_title: str = "GreenLang Orchestrator Control Plane",
        api_description: str = "REST API for GreenLang Pipeline Orchestration",
        debug: bool = False,
        cors_origins: Optional[list] = None,
        rate_limit_requests: int = 100,
        rate_limit_window_seconds: int = 60,
        api_key_header: str = "X-API-Key",
        require_auth: bool = True,
        valid_api_keys: Optional[set] = None,
        database_url: Optional[str] = None,
        k8s_namespace: str = "greenlang",
    ):
        self.api_version = api_version
        self.api_title = api_title
        self.api_description = api_description
        self.debug = debug
        self.cors_origins = cors_origins or ["https://*.greenlang.io", "http://localhost:*"]
        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.api_key_header = api_key_header
        self.require_auth = require_auth
        self.valid_api_keys = valid_api_keys or set()
        self.database_url = database_url
        self.k8s_namespace = k8s_namespace


@lru_cache()
def get_config() -> APIConfig:
    """
    Get API configuration (cached singleton).

    In production, this would load from environment variables or a config service.

    Returns:
        APIConfig instance
    """
    import os

    return APIConfig(
        api_version=os.getenv("API_VERSION", "1.0.0"),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        require_auth=os.getenv("REQUIRE_AUTH", "true").lower() == "true",
        database_url=os.getenv("DATABASE_URL"),
        k8s_namespace=os.getenv("K8S_NAMESPACE", "greenlang"),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
    )


# =============================================================================
# ORCHESTRATOR DEPENDENCY
# =============================================================================


# Global orchestrator instance (initialized at startup)
_orchestrator_instance = None
_orchestrator_lock = None


async def get_orchestrator():
    """
    Get the GLIPOrchestrator instance.

    The orchestrator is a singleton that manages pipeline execution,
    scheduling, and coordination with K8s.

    Returns:
        GLIPOrchestrator instance

    Raises:
        HTTPException: If orchestrator is not initialized
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        # Lazy initialization - try to import and create
        try:
            from greenlang.orchestrator.glip_orchestrator import GLIPOrchestrator

            _orchestrator_instance = GLIPOrchestrator()
            logger.info("GLIPOrchestrator initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"GLIPOrchestrator not available: {e}")
            # Return a mock for development/testing
            _orchestrator_instance = MockOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize GLIPOrchestrator: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Orchestrator service unavailable",
            )

    return _orchestrator_instance


def set_orchestrator(orchestrator) -> None:
    """
    Set the orchestrator instance (for testing or custom initialization).

    Args:
        orchestrator: GLIPOrchestrator instance or compatible mock
    """
    global _orchestrator_instance
    _orchestrator_instance = orchestrator


# =============================================================================
# AGENT REGISTRY DEPENDENCY
# =============================================================================


_agent_registry_instance = None


async def get_agent_registry():
    """
    Get the VersionedAgentRegistry instance.

    The registry maintains metadata about all available agents,
    their capabilities, versions, and execution modes.

    Returns:
        VersionedAgentRegistry instance

    Raises:
        HTTPException: If registry is not available
    """
    global _agent_registry_instance

    if _agent_registry_instance is None:
        try:
            from greenlang.agents.foundation.agent_registry import (
                VersionedAgentRegistry,
            )

            _agent_registry_instance = VersionedAgentRegistry()
            logger.info("VersionedAgentRegistry initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"VersionedAgentRegistry not available: {e}")
            _agent_registry_instance = MockAgentRegistry()
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent registry service unavailable",
            )

    return _agent_registry_instance


def set_agent_registry(registry) -> None:
    """
    Set the agent registry instance (for testing or custom initialization).

    Args:
        registry: VersionedAgentRegistry instance or compatible mock
    """
    global _agent_registry_instance
    _agent_registry_instance = registry


# =============================================================================
# EVENT STORE DEPENDENCY
# =============================================================================


_event_store_instance = None


async def get_event_store():
    """
    Get the EventStore instance for audit trails.

    The event store provides hash-chained audit logging for
    tamper-evident compliance.

    Returns:
        EventStore instance (InMemoryEventStore or PostgresEventStore)

    Raises:
        HTTPException: If event store is not available
    """
    global _event_store_instance

    if _event_store_instance is None:
        try:
            from greenlang.orchestrator.audit.event_store import InMemoryEventStore

            _event_store_instance = InMemoryEventStore()
            logger.info("InMemoryEventStore initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"EventStore not available: {e}")
            _event_store_instance = MockEventStore()
        except Exception as e:
            logger.error(f"Failed to initialize event store: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Event store service unavailable",
            )

    return _event_store_instance


def set_event_store(store) -> None:
    """
    Set the event store instance (for testing or custom initialization).

    Args:
        store: EventStore instance or compatible mock
    """
    global _event_store_instance
    _event_store_instance = store


# =============================================================================
# POLICY ENGINE DEPENDENCY
# =============================================================================


_policy_engine_instance = None


async def get_policy_engine():
    """
    Get the PolicyEngine instance for governance checks.

    Returns:
        PolicyEngine instance

    Raises:
        HTTPException: If policy engine is not available
    """
    global _policy_engine_instance

    if _policy_engine_instance is None:
        try:
            from greenlang.orchestrator.governance.policy_engine import PolicyEngine

            _policy_engine_instance = PolicyEngine()
            logger.info("PolicyEngine initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"PolicyEngine not available: {e}")
            _policy_engine_instance = MockPolicyEngine()
        except Exception as e:
            logger.error(f"Failed to initialize policy engine: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy engine service unavailable",
            )

    return _policy_engine_instance


def set_policy_engine(engine) -> None:
    """
    Set the policy engine instance (for testing or custom initialization).

    Args:
        engine: PolicyEngine instance or compatible mock
    """
    global _policy_engine_instance
    _policy_engine_instance = engine


# =============================================================================
# AUTHENTICATION DEPENDENCY
# =============================================================================


class AuthContext:
    """
    Authentication context for API requests.

    Contains information about the authenticated API key and associated metadata.
    """

    def __init__(
        self,
        api_key: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scopes: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.scopes = scopes or []
        self.metadata = metadata or {}
        self.authenticated_at = datetime.now(timezone.utc)

    def has_scope(self, scope: str) -> bool:
        """Check if this auth context has a specific scope."""
        return scope in self.scopes or "*" in self.scopes

    def can_access_tenant(self, tenant_id: str) -> bool:
        """Check if this auth context can access a specific tenant."""
        return self.tenant_id is None or self.tenant_id == tenant_id


async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    config: APIConfig = Depends(get_config),
) -> AuthContext:
    """
    Validate API key and return authentication context.

    The API key can be provided via:
    1. X-API-Key header (preferred)
    2. api_key query parameter (for testing/debugging only)

    Args:
        request: FastAPI request
        x_api_key: API key from header
        config: API configuration

    Returns:
        AuthContext with authentication details

    Raises:
        HTTPException: If authentication fails
    """
    # Skip auth if not required (development mode)
    if not config.require_auth:
        logger.debug("Authentication disabled - using default context")
        return AuthContext(
            api_key="development",
            tenant_id=None,
            user_id="dev-user",
            scopes=["*"],
        )

    # Check for API key
    api_key = x_api_key or request.query_params.get("api_key")

    if not api_key:
        logger.warning(f"Missing API key from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key
    # In production, this would look up the key in a database or cache
    if config.valid_api_keys and api_key not in config.valid_api_keys:
        logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Create auth context
    # In production, this would look up tenant/user info from the key
    auth_context = AuthContext(
        api_key=api_key[:8] + "...",  # Masked for logging
        tenant_id=request.headers.get("X-Tenant-ID"),
        user_id=request.headers.get("X-User-ID"),
        scopes=["pipelines:read", "pipelines:write", "runs:read", "runs:write"],
    )

    logger.debug(f"Authenticated request: tenant={auth_context.tenant_id}")
    return auth_context


# =============================================================================
# REQUEST TRACING DEPENDENCY
# =============================================================================


class RequestTrace:
    """
    Request tracing context for observability.

    Provides correlation IDs and timing for distributed tracing.
    """

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str] = None,
        start_time: Optional[float] = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time = start_time or time.time()
        self.attributes: Dict[str, Any] = {}

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a trace attribute."""
        self.attributes[key] = value


async def get_request_trace(
    request: Request,
    x_trace_id: Optional[str] = Header(None, alias="X-Trace-ID"),
    x_span_id: Optional[str] = Header(None, alias="X-Span-ID"),
) -> RequestTrace:
    """
    Get or create request tracing context.

    Supports propagation of trace context from upstream services.

    Args:
        request: FastAPI request
        x_trace_id: Trace ID from header
        x_span_id: Span ID from header

    Returns:
        RequestTrace context
    """
    trace_id = x_trace_id or f"trace-{uuid4().hex[:16]}"
    span_id = f"span-{uuid4().hex[:8]}"
    parent_span_id = x_span_id

    trace = RequestTrace(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
    )

    # Set initial attributes
    trace.set_attribute("http.method", request.method)
    trace.set_attribute("http.url", str(request.url))
    trace.set_attribute("http.client_ip", request.client.host if request.client else "unknown")

    return trace


# =============================================================================
# MOCK IMPLEMENTATIONS (for development/testing)
# =============================================================================


class MockOrchestrator:
    """Mock orchestrator for development and testing."""

    def __init__(self):
        self._pipelines: Dict[str, Any] = {}
        self._runs: Dict[str, Any] = {}
        logger.info("MockOrchestrator initialized")

    async def register_pipeline(self, definition: Any) -> str:
        """Register a mock pipeline."""
        pipeline_id = f"pipe-{uuid4().hex[:8]}"
        self._pipelines[pipeline_id] = {
            "id": pipeline_id,
            "definition": definition,
            "created_at": datetime.now(timezone.utc),
        }
        return pipeline_id

    async def get_pipeline(self, pipeline_id: str) -> Optional[Dict]:
        """Get a mock pipeline."""
        return self._pipelines.get(pipeline_id)

    async def list_pipelines(self, **kwargs) -> list:
        """List mock pipelines."""
        return list(self._pipelines.values())

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a mock pipeline."""
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
            return True
        return False

    async def submit_run(self, pipeline_id: str, parameters: Dict, **kwargs) -> str:
        """Submit a mock run."""
        run_id = f"run-{uuid4().hex[:8]}"
        self._runs[run_id] = {
            "id": run_id,
            "pipeline_id": pipeline_id,
            "parameters": parameters,
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            **kwargs,
        }
        return run_id

    async def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a mock run."""
        return self._runs.get(run_id)

    async def list_runs(self, **kwargs) -> list:
        """List mock runs."""
        return list(self._runs.values())

    async def cancel_run(self, run_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a mock run."""
        if run_id in self._runs:
            self._runs[run_id]["status"] = "canceled"
            self._runs[run_id]["cancel_reason"] = reason
            return True
        return False


class MockAgentRegistry:
    """Mock agent registry for development and testing."""

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        logger.info("MockAgentRegistry initialized")

    def get_agent(self, agent_id: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get a mock agent."""
        return self._agents.get(agent_id)

    def query_agents(self, query: Any) -> Any:
        """Query mock agents."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            agents: list
            total_count: int

        return MockResult(agents=list(self._agents.values()), total_count=len(self._agents))

    def get_statistics(self) -> Dict[str, Any]:
        """Get mock registry statistics."""
        return {
            "total_agents": len(self._agents),
            "total_versions": len(self._agents),
        }


class MockEventStore:
    """Mock event store for development and testing."""

    def __init__(self):
        self._events: Dict[str, list] = {}
        logger.info("MockEventStore initialized")

    async def get_events(self, run_id: str) -> list:
        """Get events for a run."""
        return self._events.get(run_id, [])

    async def verify_chain(self, run_id: str) -> bool:
        """Verify hash chain."""
        return True

    async def export_audit_package(self, run_id: str) -> Dict:
        """Export audit package."""
        events = self._events.get(run_id, [])
        return {
            "run_id": run_id,
            "events": events,
            "chain_valid": True,
            "exported_at": datetime.now(timezone.utc),
        }


class MockPolicyEngine:
    """Mock policy engine for development and testing."""

    def __init__(self):
        logger.info("MockPolicyEngine initialized")

    async def evaluate(self, context: Any) -> Any:
        """Evaluate mock policy."""
        from dataclasses import dataclass

        @dataclass
        class MockResult:
            allowed: bool = True
            violations: list = None

            def __post_init__(self):
                self.violations = self.violations or []

        return MockResult()


# =============================================================================
# STARTUP AND SHUTDOWN HOOKS
# =============================================================================


async def startup_dependencies() -> None:
    """
    Initialize all dependencies on application startup.

    Called during FastAPI lifespan startup.
    """
    logger.info("Initializing API dependencies...")

    # Pre-initialize to catch errors early
    try:
        await get_orchestrator()
        await get_agent_registry()
        await get_event_store()
        await get_policy_engine()
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise

    logger.info("API dependencies initialized successfully")


async def shutdown_dependencies() -> None:
    """
    Clean up dependencies on application shutdown.

    Called during FastAPI lifespan shutdown.
    """
    logger.info("Shutting down API dependencies...")

    global _orchestrator_instance, _agent_registry_instance, _event_store_instance, _policy_engine_instance

    # Clean up orchestrator
    if _orchestrator_instance and hasattr(_orchestrator_instance, "shutdown"):
        try:
            await _orchestrator_instance.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down orchestrator: {e}")

    # Clean up event store
    if _event_store_instance and hasattr(_event_store_instance, "close"):
        try:
            await _event_store_instance.close()
        except Exception as e:
            logger.warning(f"Error closing event store: {e}")

    # Reset instances
    _orchestrator_instance = None
    _agent_registry_instance = None
    _event_store_instance = None
    _policy_engine_instance = None

    logger.info("API dependencies shut down")




# =============================================================================
# APPROVAL WORKFLOW DEPENDENCY (FR-043)
# =============================================================================


_approval_workflow_instance = None


async def get_approval_workflow():
    """
    Get the ApprovalWorkflow instance for signed approvals.

    Returns:
        ApprovalWorkflow instance

    Raises:
        HTTPException: If approval workflow is not available
    """
    global _approval_workflow_instance

    if _approval_workflow_instance is None:
        try:
            from greenlang.orchestrator.governance.approvals import (
                ApprovalWorkflow,
                InMemoryApprovalStore,
            )
            from greenlang.orchestrator.audit.event_store import EventFactory

            store = InMemoryApprovalStore()
            event_store = await get_event_store()
            event_factory = EventFactory(event_store) if event_store else None

            _approval_workflow_instance = ApprovalWorkflow(
                store=store,
                event_factory=event_factory,
                default_deadline_hours=24,
            )
            logger.info("ApprovalWorkflow initialized via dependency injection")
        except ImportError as e:
            logger.warning(f"ApprovalWorkflow not available: {e}")
            _approval_workflow_instance = MockApprovalWorkflow()
        except Exception as e:
            logger.error(f"Failed to initialize approval workflow: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Approval workflow service unavailable",
            )

    return _approval_workflow_instance


def set_approval_workflow(workflow) -> None:
    """
    Set the approval workflow instance (for testing).

    Args:
        workflow: ApprovalWorkflow instance or compatible mock
    """
    global _approval_workflow_instance
    _approval_workflow_instance = workflow


class MockApprovalWorkflow:
    """Mock approval workflow for development and testing."""

    def __init__(self):
        self._requests = {}
        logger.info("MockApprovalWorkflow initialized")

    async def request_approval(self, run_id, step_id, requirement, **kwargs):
        from uuid import uuid4
        request_id = f"apr-{uuid4().hex[:12]}"
        self._requests[request_id] = {
            "request_id": request_id,
            "run_id": run_id,
            "step_id": step_id,
            "status": "pending",
        }
        return request_id

    async def submit_approval(self, approval_id, approver_id, decision, **kwargs):
        from dataclasses import dataclass
        from datetime import datetime, timezone

        @dataclass
        class MockAttestation:
            approver_id: str = approver_id
            approver_name: str = None
            approver_role: str = None
            decision: type = None
            reason: str = None
            timestamp: datetime = None
            signature: str = "mock-signature"
            attestation_hash: str = "mock-hash"

        return MockAttestation(timestamp=datetime.now(timezone.utc))

    async def check_approval_status(self, approval_id):
        return "pending"

    async def verify_attestation(self, approval_id):
        return True

    async def get_pending_approvals(self, run_id=None):
        return []

    async def get_approval(self, approval_id):
        return self._requests.get(approval_id)

    async def get_step_approval(self, run_id, step_id):
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "APIConfig",
    "get_config",
    # Orchestrator
    "get_orchestrator",
    "set_orchestrator",
    # Agent Registry
    "get_agent_registry",
    "set_agent_registry",
    # Event Store
    "get_event_store",
    "set_event_store",
    # Policy Engine
    "get_policy_engine",
    "set_policy_engine",
    # Authentication
    "AuthContext",
    "get_api_key",
    # Tracing
    "RequestTrace",
    "get_request_trace",
    # Mocks
    "MockOrchestrator",
    "MockAgentRegistry",
    "MockEventStore",
    "MockPolicyEngine",
    # Lifecycle
    "startup_dependencies",
    "shutdown_dependencies",
    # Approval Workflow
    "get_approval_workflow",
    "set_approval_workflow",
    "MockApprovalWorkflow",
]
