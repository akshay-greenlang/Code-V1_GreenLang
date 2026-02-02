# -*- coding: utf-8 -*-
"""
GraphQL Context
Provides request context with authentication, authorization, and DataLoaders
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager, AuthToken
from greenlang.api.graphql.dataloaders import (
    AgentLoader,
    WorkflowLoader,
    ExecutionLoader,
    UserLoader,
    DataLoaderFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphQLContext:
    """
    GraphQL execution context

    Provides:
    - Authentication state
    - Authorization services
    - DataLoaders for efficient data fetching
    - Core services (orchestrator, auth, RBAC)
    """

    # Authentication
    user_id: str
    tenant_id: str
    token: Optional[AuthToken] = None

    # Core services
    orchestrator: Orchestrator = None
    auth_manager: AuthManager = None
    rbac_manager: RBACManager = None

    # DataLoaders (created per-request)
    agent_loader: AgentLoader = None
    workflow_loader: WorkflowLoader = None
    execution_loader: ExecutionLoader = None
    user_loader: UserLoader = None

    # Request metadata
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize DataLoaders after context creation"""
        if self.metadata is None:
            self.metadata = {}

        # Create DataLoaders if not provided
        if self.orchestrator and self.auth_manager and self.rbac_manager:
            factory = DataLoaderFactory(
                orchestrator=self.orchestrator,
                auth_manager=self.auth_manager,
                rbac_manager=self.rbac_manager,
            )

            if not self.agent_loader:
                self.agent_loader = factory.create_agent_loader()
            if not self.workflow_loader:
                self.workflow_loader = factory.create_workflow_loader()
            if not self.execution_loader:
                self.execution_loader = factory.create_execution_loader()
            if not self.user_loader:
                self.user_loader = factory.create_user_loader()

    @classmethod
    async def from_request(
        cls,
        request: Any,
        orchestrator: Orchestrator,
        auth_manager: AuthManager,
        rbac_manager: RBACManager,
    ) -> GraphQLContext:
        """
        Create context from HTTP request

        Args:
            request: HTTP request object
            orchestrator: Orchestrator instance
            auth_manager: AuthManager instance
            rbac_manager: RBACManager instance

        Returns:
            GraphQLContext

        Raises:
            PermissionError: If authentication fails
        """
        # Extract authorization header
        auth_header = request.headers.get("Authorization", "")

        # Parse token
        token_value = None
        if auth_header.startswith("Bearer "):
            token_value = auth_header[7:]
        elif auth_header.startswith("Token "):
            token_value = auth_header[6:]

        if not token_value:
            raise PermissionError("No authentication token provided")

        # Validate token
        token = auth_manager.validate_token(token_value)
        if not token:
            raise PermissionError("Invalid or expired token")

        # Extract request metadata
        request_id = request.headers.get("X-Request-ID")
        ip_address = request.headers.get("X-Real-IP") or request.client.host
        user_agent = request.headers.get("User-Agent")

        # Create context
        context = cls(
            user_id=token.user_id,
            tenant_id=token.tenant_id,
            token=token,
            orchestrator=orchestrator,
            auth_manager=auth_manager,
            rbac_manager=rbac_manager,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        logger.info(
            f"Created GraphQL context for user {token.user_id} "
            f"(request_id: {request_id})"
        )

        return context

    @classmethod
    def create_system_context(
        cls,
        orchestrator: Orchestrator,
        auth_manager: AuthManager,
        rbac_manager: RBACManager,
    ) -> GraphQLContext:
        """
        Create system context (for internal use)

        Args:
            orchestrator: Orchestrator instance
            auth_manager: AuthManager instance
            rbac_manager: RBACManager instance

        Returns:
            GraphQLContext with system privileges
        """
        # Create system user if not exists
        if "system" not in auth_manager.users:
            system_user_id = auth_manager.create_user(
                tenant_id="system",
                username="system",
                email="system@greenlang.internal",
                password="system_internal_use_only",
            )
        else:
            system_user_id = [
                uid for uid, u in auth_manager.users.items() if u["username"] == "system"
            ][0]

        # Assign super_admin role
        rbac_manager.assign_role(system_user_id, "super_admin")

        return cls(
            user_id=system_user_id,
            tenant_id="system",
            orchestrator=orchestrator,
            auth_manager=auth_manager,
            rbac_manager=rbac_manager,
            metadata={"is_system": True},
        )

    def has_permission(
        self,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if current user has permission

        Args:
            resource: Resource to access
            action: Action to perform
            context: Additional context for permission check

        Returns:
            True if user has permission
        """
        return self.rbac_manager.check_permission(
            self.user_id,
            resource,
            action,
            context,
        )

    def require_permission(
        self,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Require permission or raise error

        Args:
            resource: Resource to access
            action: Action to perform
            context: Additional context for permission check

        Raises:
            PermissionError: If user lacks permission
        """
        if not self.has_permission(resource, action, context):
            raise PermissionError(
                f"User {self.user_id} lacks permission: {resource}:{action}"
            )

    def clear_caches(self):
        """Clear all DataLoader caches"""
        if self.agent_loader:
            self.agent_loader.clear()
        if self.workflow_loader:
            self.workflow_loader.clear()
        if self.execution_loader:
            self.execution_loader.clear()
        if self.user_loader:
            self.user_loader.clear()

    def is_system(self) -> bool:
        """Check if this is a system context"""
        return self.metadata.get("is_system", False)

    def __repr__(self) -> str:
        return (
            f"GraphQLContext("
            f"user_id={self.user_id}, "
            f"tenant_id={self.tenant_id}, "
            f"request_id={self.request_id})"
        )
