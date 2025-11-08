"""
DataLoader Implementation for N+1 Query Prevention
Provides batching and caching for efficient data loading
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime

import strawberry

from greenlang.api.graphql.types import (
    Agent,
    Workflow,
    Execution,
    User,
    AgentStats,
    WorkflowStep,
    Role,
    Permission,
)
from greenlang.core.orchestrator import Orchestrator
from greenlang.auth.rbac import RBACManager
from greenlang.auth.auth import AuthManager

logger = logging.getLogger(__name__)


# ==============================================================================
# DataLoader Base Class
# ==============================================================================

class DataLoader:
    """Base DataLoader with batching and caching"""

    def __init__(self, batch_load_fn, max_batch_size: int = 100):
        """
        Initialize DataLoader

        Args:
            batch_load_fn: Async function that takes list of keys and returns list of values
            max_batch_size: Maximum number of keys to batch together
        """
        self.batch_load_fn = batch_load_fn
        self.max_batch_size = max_batch_size
        self.cache: Dict[str, Any] = {}
        self.queue: List[tuple] = []
        self.dispatched = False

    async def load(self, key: str) -> Optional[Any]:
        """
        Load single item by key

        Args:
            key: Item key/ID

        Returns:
            Loaded item or None
        """
        # Check cache first
        if key in self.cache:
            logger.debug(f"Cache hit for key: {key}")
            return self.cache[key]

        # Add to batch queue
        future = asyncio.Future()
        self.queue.append((key, future))

        # Dispatch batch if not already dispatched
        if not self.dispatched:
            self.dispatched = True
            asyncio.create_task(self._dispatch_queue())

        # Wait for result
        return await future

    async def load_many(self, keys: List[str]) -> List[Optional[Any]]:
        """
        Load multiple items by keys

        Args:
            keys: List of item keys/IDs

        Returns:
            List of loaded items (None for not found)
        """
        # Load all in parallel
        tasks = [self.load(key) for key in keys]
        return await asyncio.gather(*tasks)

    async def _dispatch_queue(self):
        """Dispatch queued load requests"""
        # Small delay to allow batching
        await asyncio.sleep(0.001)

        # Process queue
        queue = self.queue
        self.queue = []
        self.dispatched = False

        if not queue:
            return

        # Extract keys and futures
        keys = [item[0] for item in queue]
        futures = [item[1] for item in queue]

        try:
            # Batch load
            values = await self.batch_load_fn(keys)

            # Validate result length
            if len(values) != len(keys):
                raise ValueError(
                    f"Batch load returned {len(values)} items for {len(keys)} keys"
                )

            # Cache and resolve futures
            for key, value, future in zip(keys, values, futures):
                self.cache[key] = value
                future.set_result(value)

        except Exception as e:
            logger.error(f"Batch load error: {e}")
            # Reject all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    def clear(self):
        """Clear cache"""
        self.cache.clear()

    def prime(self, key: str, value: Any):
        """Prime cache with value"""
        self.cache[key] = value


# ==============================================================================
# Agent DataLoader
# ==============================================================================

class AgentLoader(DataLoader):
    """DataLoader for agents with batching and caching"""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        super().__init__(self._batch_load_agents)

    async def _batch_load_agents(self, agent_ids: List[str]) -> List[Optional[Agent]]:
        """
        Batch load agents

        Args:
            agent_ids: List of agent IDs to load

        Returns:
            List of Agent objects (None for not found)
        """
        logger.debug(f"Batch loading {len(agent_ids)} agents")

        agents = []
        for agent_id in agent_ids:
            agent_obj = self.orchestrator.agents.get(agent_id)

            if agent_obj:
                # Convert to GraphQL type
                from greenlang.api.graphql.resolvers import _convert_agent

                agent = _convert_agent(agent_id, agent_obj)
                agents.append(agent)
            else:
                agents.append(None)

        return agents


# ==============================================================================
# Workflow DataLoader
# ==============================================================================

class WorkflowLoader(DataLoader):
    """DataLoader for workflows with batching and caching"""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        super().__init__(self._batch_load_workflows)

    async def _batch_load_workflows(
        self, workflow_ids: List[str]
    ) -> List[Optional[Workflow]]:
        """
        Batch load workflows

        Args:
            workflow_ids: List of workflow IDs to load

        Returns:
            List of Workflow objects (None for not found)
        """
        logger.debug(f"Batch loading {len(workflow_ids)} workflows")

        workflows = []
        for workflow_id in workflow_ids:
            workflow_obj = self.orchestrator.workflows.get(workflow_id)

            if workflow_obj:
                # Convert to GraphQL type
                from greenlang.api.graphql.resolvers import _convert_workflow

                workflow = _convert_workflow(workflow_id, workflow_obj)
                workflows.append(workflow)
            else:
                workflows.append(None)

        return workflows


# ==============================================================================
# Execution DataLoader
# ==============================================================================

class ExecutionLoader(DataLoader):
    """DataLoader for executions with batching and caching"""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        super().__init__(self._batch_load_executions)

    async def _batch_load_executions(
        self, execution_ids: List[str]
    ) -> List[Optional[Execution]]:
        """
        Batch load executions

        Args:
            execution_ids: List of execution IDs to load

        Returns:
            List of Execution objects (None for not found)
        """
        logger.debug(f"Batch loading {len(execution_ids)} executions")

        # Get all execution history
        history = self.orchestrator.get_execution_history()

        # Create lookup dict
        execution_map = {rec["execution_id"]: rec for rec in history}

        executions = []
        for execution_id in execution_ids:
            record = execution_map.get(execution_id)

            if record:
                # Convert to GraphQL type
                from greenlang.api.graphql.resolvers import _convert_execution_record

                execution = _convert_execution_record(record)
                executions.append(execution)
            else:
                executions.append(None)

        return executions


# ==============================================================================
# User DataLoader
# ==============================================================================

class UserLoader(DataLoader):
    """DataLoader for users with batching and caching"""

    def __init__(
        self,
        auth_manager: AuthManager,
        rbac_manager: RBACManager,
    ):
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager
        super().__init__(self._batch_load_users)

    async def _batch_load_users(self, user_ids: List[str]) -> List[Optional[User]]:
        """
        Batch load users

        Args:
            user_ids: List of user IDs to load

        Returns:
            List of User objects (None for not found)
        """
        logger.debug(f"Batch loading {len(user_ids)} users")

        users = []
        for user_id in user_ids:
            user_data = self.auth_manager.users.get(user_id)

            if user_data:
                # Get user roles
                role_names = self.rbac_manager.get_user_roles(user_id)
                roles = []
                for name in role_names:
                    role_obj = self.rbac_manager.get_role(name)
                    if role_obj:
                        from greenlang.api.graphql.resolvers import _convert_role

                        roles.append(_convert_role(role_obj))

                # Get user permissions
                perms = self.rbac_manager.get_user_permissions(user_id)
                permissions = []
                for perm in perms:
                    from greenlang.api.graphql.resolvers import _convert_permission

                    permissions.append(_convert_permission(perm))

                # Create User object
                user = User(
                    id=strawberry.ID(user_id),
                    tenant_id=strawberry.ID(user_data["tenant_id"]),
                    username=user_data["username"],
                    email=user_data["email"],
                    active=user_data.get("active", True),
                    roles=roles,
                    permissions=permissions,
                    created_at=user_data.get("created_at", datetime.utcnow()),
                    last_login=None,
                    metadata=user_data.get("metadata", {}),
                )
                users.append(user)
            else:
                users.append(None)

        return users


# ==============================================================================
# DataLoader Factory
# ==============================================================================

@dataclass
class DataLoaderFactory:
    """Factory for creating and managing DataLoaders"""

    orchestrator: Orchestrator
    auth_manager: AuthManager
    rbac_manager: RBACManager

    def create_agent_loader(self) -> AgentLoader:
        """Create AgentLoader"""
        return AgentLoader(self.orchestrator)

    def create_workflow_loader(self) -> WorkflowLoader:
        """Create WorkflowLoader"""
        return WorkflowLoader(self.orchestrator)

    def create_execution_loader(self) -> ExecutionLoader:
        """Create ExecutionLoader"""
        return ExecutionLoader(self.orchestrator)

    def create_user_loader(self) -> UserLoader:
        """Create UserLoader"""
        return UserLoader(self.auth_manager, self.rbac_manager)

    def create_all_loaders(self) -> Dict[str, DataLoader]:
        """Create all loaders"""
        return {
            "agent": self.create_agent_loader(),
            "workflow": self.create_workflow_loader(),
            "execution": self.create_execution_loader(),
            "user": self.create_user_loader(),
        }
