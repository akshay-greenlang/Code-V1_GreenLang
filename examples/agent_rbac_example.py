# -*- coding: utf-8 -*-
"""
Agent RBAC Example
==================

Example demonstrating how to integrate agent-level RBAC into agent implementations.

This example shows:
1. How to check permissions before agent execution
2. How to handle permission denials
3. How to use RBAC in agent orchestrators
4. How to audit agent access
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from core.greenlang.policy.enforcer import PolicyEnforcer, PolicyResult
from core.greenlang.policy.agent_rbac import AgentPermission

logger = logging.getLogger(__name__)


class SecuredAgentExecutor:
    """
    Example agent executor with RBAC enforcement.

    This class shows how to integrate RBAC checks into your agent execution logic.
    """

    def __init__(self, agent_id: str, policy_enforcer: Optional[PolicyEnforcer] = None):
        """
        Initialize secured agent executor.

        Args:
            agent_id: Agent identifier (e.g., "GL-001")
            policy_enforcer: PolicyEnforcer instance (creates new if None)
        """
        self.agent_id = agent_id
        self.enforcer = policy_enforcer or PolicyEnforcer()

    def execute(
        self,
        user: str,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute agent with RBAC permission check.

        Args:
            user: User email or identifier
            input_data: Agent input data
            context: Execution context

        Returns:
            Agent execution result

        Raises:
            PermissionError: If user lacks execute permission
        """
        # Check execute permission
        result = self.enforcer.check_agent_execute(self.agent_id, user, context)

        if not result.allowed:
            logger.error(
                f"User {user} denied execution of agent {self.agent_id}: {result.reason}"
            )
            raise PermissionError(
                f"Access denied: {result.reason}. "
                f"Violated policies: {', '.join(result.violated_policies)}"
            )

        logger.info(f"User {user} executing agent {self.agent_id}")

        # Execute agent logic
        try:
            output = self._execute_agent_logic(input_data)
            logger.info(f"Agent {self.agent_id} execution completed for user {user}")
            return output

        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {e}", exc_info=True)
            raise

    def _execute_agent_logic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal agent execution logic (example).

        In a real implementation, this would call your actual agent logic.
        """
        # Placeholder for actual agent logic
        return {
            "status": "completed",
            "result": f"Processed input: {input_data}",
            "agent_id": self.agent_id
        }

    def read_data(
        self,
        user: str,
        data_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Read agent data with RBAC permission check.

        Args:
            user: User email or identifier
            data_id: Data identifier
            context: Access context

        Returns:
            Agent data

        Raises:
            PermissionError: If user lacks read permission
        """
        # Check read data permission
        result = self.enforcer.check_agent_data_access(
            self.agent_id, user, "read", context
        )

        if not result.allowed:
            logger.error(
                f"User {user} denied read access to agent {self.agent_id} data: {result.reason}"
            )
            raise PermissionError(f"Access denied: {result.reason}")

        logger.info(f"User {user} reading data from agent {self.agent_id}")

        # Read data (example)
        return {
            "data_id": data_id,
            "data": "Example data",
            "agent_id": self.agent_id
        }

    def write_data(
        self,
        user: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Write agent data with RBAC permission check.

        Args:
            user: User email or identifier
            data: Data to write
            context: Access context

        Returns:
            Write result

        Raises:
            PermissionError: If user lacks write permission
        """
        # Check write data permission
        result = self.enforcer.check_agent_data_access(
            self.agent_id, user, "write", context
        )

        if not result.allowed:
            logger.error(
                f"User {user} denied write access to agent {self.agent_id} data: {result.reason}"
            )
            raise PermissionError(f"Access denied: {result.reason}")

        logger.info(f"User {user} writing data to agent {self.agent_id}")

        # Write data (example)
        return {
            "status": "written",
            "data_id": "new-data-123",
            "agent_id": self.agent_id
        }

    def get_config(
        self,
        user: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get agent configuration with RBAC permission check.

        Args:
            user: User email or identifier
            context: Access context

        Returns:
            Agent configuration

        Raises:
            PermissionError: If user lacks read config permission
        """
        # Check read config permission
        result = self.enforcer.check_agent_config_access(
            self.agent_id, user, "read", context
        )

        if not result.allowed:
            logger.error(
                f"User {user} denied read config access to agent {self.agent_id}: {result.reason}"
            )
            raise PermissionError(f"Access denied: {result.reason}")

        logger.info(f"User {user} reading config from agent {self.agent_id}")

        # Return config (example)
        return {
            "agent_id": self.agent_id,
            "version": "1.0.0",
            "settings": {"timeout": 30}
        }

    def update_config(
        self,
        user: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update agent configuration with RBAC permission check.

        Args:
            user: User email or identifier
            config: New configuration
            context: Access context

        Returns:
            Update result

        Raises:
            PermissionError: If user lacks write config permission
        """
        # Check write config permission
        result = self.enforcer.check_agent_config_access(
            self.agent_id, user, "write", context
        )

        if not result.allowed:
            logger.error(
                f"User {user} denied write config access to agent {self.agent_id}: {result.reason}"
            )
            raise PermissionError(f"Access denied: {result.reason}")

        logger.info(f"User {user} updating config for agent {self.agent_id}")

        # Update config (example)
        return {
            "status": "updated",
            "agent_id": self.agent_id
        }


def example_basic_usage():
    """Example: Basic agent execution with RBAC."""
    print("=== Basic Agent Execution with RBAC ===\n")

    # Create enforcer
    enforcer = PolicyEnforcer()

    # Grant permissions
    print("Granting agent_operator role to user@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

    # Create secured agent executor
    agent = SecuredAgentExecutor("GL-001", enforcer)

    # Execute agent (should succeed)
    try:
        print("Executing agent as user@example.com...")
        result = agent.execute("user@example.com", {"input": "test"})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")

    # Try to execute as different user without permission (should fail)
    try:
        print("Executing agent as unauthorized@example.com...")
        result = agent.execute("unauthorized@example.com", {"input": "test"})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Expected error: {e}\n")


def example_data_access():
    """Example: Data access with RBAC."""
    print("=== Agent Data Access with RBAC ===\n")

    enforcer = PolicyEnforcer()

    # Grant operator role (has read but not write)
    print("Granting agent_operator role to operator@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "operator@example.com", "agent_operator")

    # Grant engineer role (has both read and write)
    print("Granting agent_engineer role to engineer@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "engineer@example.com", "agent_engineer")

    agent = SecuredAgentExecutor("GL-001", enforcer)

    # Operator can read
    try:
        print("Operator reading data...")
        data = agent.read_data("operator@example.com", "data-123")
        print(f"Success: {data}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")

    # Operator cannot write
    try:
        print("Operator writing data...")
        result = agent.write_data("operator@example.com", {"new": "data"})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Expected error: {e}\n")

    # Engineer can write
    try:
        print("Engineer writing data...")
        result = agent.write_data("engineer@example.com", {"new": "data"})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")


def example_config_management():
    """Example: Configuration management with RBAC."""
    print("=== Agent Configuration Management with RBAC ===\n")

    enforcer = PolicyEnforcer()

    # Grant viewer role (can read config but not write)
    print("Granting agent_viewer role to viewer@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "viewer@example.com", "agent_viewer")

    # Grant engineer role (can read and write config)
    print("Granting agent_engineer role to engineer@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "engineer@example.com", "agent_engineer")

    agent = SecuredAgentExecutor("GL-001", enforcer)

    # Viewer can read config
    try:
        print("Viewer reading config...")
        config = agent.get_config("viewer@example.com")
        print(f"Success: {config}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")

    # Viewer cannot update config
    try:
        print("Viewer updating config...")
        result = agent.update_config("viewer@example.com", {"timeout": 60})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Expected error: {e}\n")

    # Engineer can update config
    try:
        print("Engineer updating config...")
        result = agent.update_config("engineer@example.com", {"timeout": 60})
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")


def example_audit_trail():
    """Example: Auditing user access."""
    print("=== Agent Access Audit Trail ===\n")

    enforcer = PolicyEnforcer()

    # Grant permissions for multiple agents
    print("Setting up permissions for user@example.com...")
    enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")
    enforcer.grant_agent_role("GL-002", "user@example.com", "agent_engineer")
    enforcer.grant_agent_role("GL-003", "user@example.com", "agent_viewer")

    # Audit user access
    print("Auditing access for user@example.com...\n")
    audit = enforcer.audit_user_agent_access("user@example.com")

    for agent_id, roles in audit.items():
        print(f"Agent {agent_id}:")
        print(f"  Roles: {', '.join(roles)}")

        # Get ACL to show permissions
        acl = enforcer.rbac_manager.get_acl(agent_id)
        if acl:
            permissions = acl.list_user_permissions("user@example.com")
            perm_names = [p.value for p in permissions]
            print(f"  Permissions: {', '.join(perm_names)}")
        print()

    # List available roles
    print("\nAvailable Roles:")
    roles = enforcer.list_available_roles()
    for role_name, description in roles.items():
        print(f"  {role_name}: {description}")


def example_critical_agent_protection():
    """Example: Critical agent protection with approval."""
    print("=== Critical Agent Protection ===\n")

    enforcer = PolicyEnforcer()

    # Grant operator role for GL-001 (critical agent)
    print("Granting agent_operator role to operator@example.com for GL-001...")
    enforcer.grant_agent_role("GL-001", "operator@example.com", "agent_operator")

    agent = SecuredAgentExecutor("GL-001", enforcer)

    # Try to execute without approval (may be denied by OPA policy)
    try:
        print("Executing critical agent GL-001 without approval...")
        context = {"has_approval": False}
        result = agent.execute("operator@example.com", {"input": "test"}, context)
        print(f"Result: {result}\n")
    except PermissionError as e:
        print(f"Note: Critical agents may require approval: {e}\n")

    # Execute with approval
    try:
        print("Executing critical agent GL-001 with approval...")
        context = {"has_approval": True}
        result = agent.execute("operator@example.com", {"input": "test"}, context)
        print(f"Success: {result}\n")
    except PermissionError as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GreenLang Agent RBAC Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_basic_usage()
    example_data_access()
    example_config_management()
    example_audit_trail()
    example_critical_agent_protection()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")
