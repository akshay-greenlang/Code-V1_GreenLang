# -*- coding: utf-8 -*-
"""
Agent-Level RBAC (Role-Based Access Control)
=============================================

Implements fine-grained access control for individual agents.
Provides role-based permissions for agent execution, configuration, and data access.

Example:
    >>> from greenlang.policy.agent_rbac import AgentAccessControl, AgentPermission
    >>> acl = AgentAccessControl(agent_id="GL-001", user_roles={"user@example.com": ["agent_operator"]})
    >>> can_execute = acl.check_permission("user@example.com", AgentPermission.EXECUTE)
    >>> assert can_execute == True
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentPermission(Enum):
    """
    Agent-level permissions.

    These permissions control what operations a user can perform on a specific agent.
    """

    EXECUTE = "execute"  # Execute agent
    READ_CONFIG = "read_config"  # Read agent configuration
    WRITE_CONFIG = "write_config"  # Modify agent configuration
    READ_DATA = "read_data"  # Read agent input/output data
    WRITE_DATA = "write_data"  # Write agent data
    MANAGE_LIFECYCLE = "manage_lifecycle"  # Start/stop agent
    VIEW_METRICS = "view_metrics"  # View agent metrics
    EXPORT_PROVENANCE = "export_provenance"  # Export audit trail
    ADMIN = "admin"  # Full admin access

    def __str__(self) -> str:
        """String representation of permission."""
        return self.value

    @classmethod
    def from_string(cls, permission: str) -> "AgentPermission":
        """Create permission from string."""
        try:
            return cls(permission)
        except ValueError:
            raise ValueError(f"Invalid permission: {permission}. Valid permissions: {[p.value for p in cls]}")


@dataclass
class AgentRole:
    """
    Role with associated permissions.

    A role is a named collection of permissions that can be assigned to users.

    Attributes:
        role_name: Unique identifier for the role
        permissions: Set of permissions granted by this role
        description: Human-readable description of what this role allows
    """

    role_name: str
    permissions: Set[AgentPermission]
    description: str

    def has_permission(self, permission: AgentPermission) -> bool:
        """Check if role has specific permission."""
        return permission in self.permissions or AgentPermission.ADMIN in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert role to dictionary for serialization."""
        return {
            "role_name": self.role_name,
            "permissions": [p.value for p in self.permissions],
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRole":
        """Create role from dictionary."""
        return cls(
            role_name=data["role_name"],
            permissions={AgentPermission(p) for p in data["permissions"]},
            description=data["description"]
        )


# Predefined roles with standard permission sets
PREDEFINED_ROLES = {
    "agent_viewer": AgentRole(
        role_name="agent_viewer",
        permissions={AgentPermission.READ_CONFIG, AgentPermission.VIEW_METRICS},
        description="View agent configuration and metrics (read-only access)"
    ),
    "agent_operator": AgentRole(
        role_name="agent_operator",
        permissions={
            AgentPermission.EXECUTE,
            AgentPermission.READ_CONFIG,
            AgentPermission.READ_DATA,
            AgentPermission.VIEW_METRICS,
            AgentPermission.EXPORT_PROVENANCE
        },
        description="Execute agents and view results (standard operational access)"
    ),
    "agent_engineer": AgentRole(
        role_name="agent_engineer",
        permissions={
            AgentPermission.EXECUTE,
            AgentPermission.READ_CONFIG,
            AgentPermission.WRITE_CONFIG,
            AgentPermission.READ_DATA,
            AgentPermission.WRITE_DATA,
            AgentPermission.MANAGE_LIFECYCLE,
            AgentPermission.VIEW_METRICS,
            AgentPermission.EXPORT_PROVENANCE
        },
        description="Full agent management except admin operations (engineering access)"
    ),
    "agent_admin": AgentRole(
        role_name="agent_admin",
        permissions=set(AgentPermission),  # All permissions
        description="Full administrative access to all agent operations"
    )
}


@dataclass
class AgentAccessControl:
    """
    Access control for a specific agent.

    Manages user-to-role mappings for a single agent, enabling fine-grained
    permission control at the agent level.

    Attributes:
        agent_id: Agent identifier (e.g., "GL-001", "GL-002")
        user_roles: Mapping of user email to list of role names
        custom_roles: Custom roles defined specifically for this agent

    Example:
        >>> acl = AgentAccessControl(
        ...     agent_id="GL-001",
        ...     user_roles={"user@example.com": ["agent_operator"]}
        ... )
        >>> acl.check_permission("user@example.com", AgentPermission.EXECUTE)
        True
    """

    agent_id: str
    user_roles: Dict[str, List[str]] = field(default_factory=dict)
    custom_roles: Dict[str, AgentRole] = field(default_factory=dict)

    def check_permission(self, user: str, permission: AgentPermission) -> bool:
        """
        Check if user has permission.

        Args:
            user: User email or identifier
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        user_role_names = self.user_roles.get(user, [])

        if not user_role_names:
            logger.debug(f"User {user} has no roles for agent {self.agent_id}")
            return False

        # Check each role the user has
        for role_name in user_role_names:
            # Check predefined roles
            role = PREDEFINED_ROLES.get(role_name)
            if role and role.has_permission(permission):
                logger.debug(f"User {user} has permission {permission.value} via role {role_name}")
                return True

            # Check custom roles
            role = self.custom_roles.get(role_name)
            if role and role.has_permission(permission):
                logger.debug(f"User {user} has permission {permission.value} via custom role {role_name}")
                return True

        logger.debug(f"User {user} lacks permission {permission.value} for agent {self.agent_id}")
        return False

    def grant_role(self, user: str, role_name: str) -> None:
        """
        Grant role to user.

        Args:
            user: User email or identifier
            role_name: Name of role to grant

        Raises:
            ValueError: If role doesn't exist
        """
        # Validate role exists
        if role_name not in PREDEFINED_ROLES and role_name not in self.custom_roles:
            raise ValueError(f"Role {role_name} does not exist")

        if user not in self.user_roles:
            self.user_roles[user] = []

        if role_name not in self.user_roles[user]:
            self.user_roles[user].append(role_name)
            logger.info(f"Granted role {role_name} to user {user} for agent {self.agent_id}")

    def revoke_role(self, user: str, role_name: str) -> None:
        """
        Revoke role from user.

        Args:
            user: User email or identifier
            role_name: Name of role to revoke
        """
        if user in self.user_roles and role_name in self.user_roles[user]:
            self.user_roles[user].remove(role_name)
            logger.info(f"Revoked role {role_name} from user {user} for agent {self.agent_id}")

            # Clean up empty user entries
            if not self.user_roles[user]:
                del self.user_roles[user]

    def list_user_roles(self, user: str) -> List[str]:
        """
        List all roles for a user.

        Args:
            user: User email or identifier

        Returns:
            List of role names
        """
        return self.user_roles.get(user, [])

    def list_user_permissions(self, user: str) -> Set[AgentPermission]:
        """
        List all permissions for a user (aggregated across all roles).

        Args:
            user: User email or identifier

        Returns:
            Set of permissions
        """
        permissions = set()

        for role_name in self.user_roles.get(user, []):
            # Check predefined roles
            role = PREDEFINED_ROLES.get(role_name)
            if role:
                permissions.update(role.permissions)

            # Check custom roles
            role = self.custom_roles.get(role_name)
            if role:
                permissions.update(role.permissions)

        return permissions

    def list_all_users(self) -> List[str]:
        """
        List all users with roles for this agent.

        Returns:
            List of user identifiers
        """
        return list(self.user_roles.keys())

    def add_custom_role(self, role: AgentRole) -> None:
        """
        Add custom role for this agent.

        Args:
            role: Custom role to add
        """
        if role.role_name in PREDEFINED_ROLES:
            raise ValueError(f"Cannot override predefined role: {role.role_name}")

        self.custom_roles[role.role_name] = role
        logger.info(f"Added custom role {role.role_name} for agent {self.agent_id}")

    def remove_custom_role(self, role_name: str) -> None:
        """
        Remove custom role.

        Args:
            role_name: Name of custom role to remove

        Raises:
            ValueError: If role doesn't exist or is predefined
        """
        if role_name in PREDEFINED_ROLES:
            raise ValueError(f"Cannot remove predefined role: {role_name}")

        if role_name not in self.custom_roles:
            raise ValueError(f"Custom role {role_name} does not exist")

        # Remove role from all users
        for user in list(self.user_roles.keys()):
            if role_name in self.user_roles[user]:
                self.revoke_role(user, role_name)

        del self.custom_roles[role_name]
        logger.info(f"Removed custom role {role_name} for agent {self.agent_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert ACL to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "user_roles": self.user_roles,
            "custom_roles": {name: role.to_dict() for name, role in self.custom_roles.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentAccessControl":
        """Create ACL from dictionary."""
        custom_roles = {
            name: AgentRole.from_dict(role_data)
            for name, role_data in data.get("custom_roles", {}).items()
        }

        return cls(
            agent_id=data["agent_id"],
            user_roles=data.get("user_roles", {}),
            custom_roles=custom_roles
        )

    def calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of ACL for audit trail.

        Returns:
            Hex digest of SHA-256 hash
        """
        acl_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(acl_json.encode()).hexdigest()


class AgentRBACManager:
    """
    Manager for agent-level RBAC policies.

    Handles persistence, loading, and management of agent access controls.

    Attributes:
        storage_path: Directory where ACL files are stored
        access_controls: In-memory cache of ACLs by agent_id
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize RBAC manager.

        Args:
            storage_path: Directory for storing ACL files (default: ~/.greenlang/rbac)
        """
        self.storage_path = storage_path or Path.home() / ".greenlang" / "rbac"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.access_controls: Dict[str, AgentAccessControl] = {}
        self._load_all_acls()

    def _load_all_acls(self) -> None:
        """Load all ACL files from storage."""
        for acl_file in self.storage_path.glob("*.json"):
            try:
                with open(acl_file) as f:
                    data = json.load(f)

                acl = AgentAccessControl.from_dict(data)
                self.access_controls[acl.agent_id] = acl
                logger.info(f"Loaded ACL for agent {acl.agent_id}")
            except Exception as e:
                logger.error(f"Failed to load ACL from {acl_file}: {e}")

    def get_acl(self, agent_id: str) -> Optional[AgentAccessControl]:
        """
        Get ACL for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentAccessControl or None if not found
        """
        return self.access_controls.get(agent_id)

    def create_acl(self, agent_id: str) -> AgentAccessControl:
        """
        Create new ACL for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            New AgentAccessControl instance

        Raises:
            ValueError: If ACL already exists
        """
        if agent_id in self.access_controls:
            raise ValueError(f"ACL already exists for agent {agent_id}")

        acl = AgentAccessControl(agent_id=agent_id)
        self.access_controls[agent_id] = acl
        self._save_acl(acl)
        logger.info(f"Created ACL for agent {agent_id}")

        return acl

    def delete_acl(self, agent_id: str) -> None:
        """
        Delete ACL for agent.

        Args:
            agent_id: Agent identifier
        """
        if agent_id not in self.access_controls:
            raise ValueError(f"ACL not found for agent {agent_id}")

        del self.access_controls[agent_id]

        # Delete file
        acl_file = self.storage_path / f"{agent_id}.json"
        if acl_file.exists():
            acl_file.unlink()

        logger.info(f"Deleted ACL for agent {agent_id}")

    def _save_acl(self, acl: AgentAccessControl) -> None:
        """Save ACL to storage."""
        acl_file = self.storage_path / f"{acl.agent_id}.json"

        with open(acl_file, "w") as f:
            json.dump(acl.to_dict(), f, indent=2)

        logger.debug(f"Saved ACL for agent {acl.agent_id}")

    def grant_role(self, agent_id: str, user: str, role_name: str) -> None:
        """
        Grant role to user for agent.

        Args:
            agent_id: Agent identifier
            user: User email or identifier
            role_name: Role to grant
        """
        # Get or create ACL
        acl = self.access_controls.get(agent_id)
        if not acl:
            acl = self.create_acl(agent_id)

        acl.grant_role(user, role_name)
        self._save_acl(acl)

    def revoke_role(self, agent_id: str, user: str, role_name: str) -> None:
        """
        Revoke role from user for agent.

        Args:
            agent_id: Agent identifier
            user: User email or identifier
            role_name: Role to revoke
        """
        acl = self.access_controls.get(agent_id)
        if not acl:
            raise ValueError(f"ACL not found for agent {agent_id}")

        acl.revoke_role(user, role_name)
        self._save_acl(acl)

    def check_permission(self, agent_id: str, user: str, permission: AgentPermission) -> bool:
        """
        Check if user has permission for agent.

        Args:
            agent_id: Agent identifier
            user: User email or identifier
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        acl = self.access_controls.get(agent_id)
        if not acl:
            # No ACL defined - default deny
            logger.debug(f"No ACL defined for agent {agent_id}, denying access")
            return False

        return acl.check_permission(user, permission)

    def list_user_agents(self, user: str) -> List[str]:
        """
        List all agents user has access to.

        Args:
            user: User email or identifier

        Returns:
            List of agent identifiers
        """
        agent_ids = []

        for agent_id, acl in self.access_controls.items():
            if user in acl.user_roles:
                agent_ids.append(agent_id)

        return agent_ids

    def audit_user_access(self, user: str) -> Dict[str, List[str]]:
        """
        Audit all access for user across all agents.

        Args:
            user: User email or identifier

        Returns:
            Dictionary mapping agent_id to list of roles
        """
        audit = {}

        for agent_id, acl in self.access_controls.items():
            roles = acl.list_user_roles(user)
            if roles:
                audit[agent_id] = roles

        return audit

    def export_audit_log(self, output_path: Path) -> None:
        """
        Export complete audit log of all ACLs.

        Args:
            output_path: Path to write audit log
        """
        audit_data = {
            "timestamp": Path(__file__).stat().st_mtime,
            "acls": {agent_id: acl.to_dict() for agent_id, acl in self.access_controls.items()},
            "hashes": {agent_id: acl.calculate_hash() for agent_id, acl in self.access_controls.items()}
        }

        with open(output_path, "w") as f:
            json.dump(audit_data, f, indent=2)

        logger.info(f"Exported audit log to {output_path}")
