# -*- coding: utf-8 -*-
"""
Unit Tests for Agent RBAC
==========================

Comprehensive tests for agent-level Role-Based Access Control.

Test Coverage:
    - AgentPermission enum
    - AgentRole model
    - AgentAccessControl operations
    - AgentRBACManager functionality
    - Permission checking logic
    - Role management
    - Audit trail
"""

import pytest
import tempfile
import json
from pathlib import Path

# Import RBAC components
from core.greenlang.policy.agent_rbac import (
    AgentPermission,
    AgentRole,
    AgentAccessControl,
    AgentRBACManager,
    PREDEFINED_ROLES
)


class TestAgentPermission:
    """Test AgentPermission enum."""

    def test_permission_values(self):
        """Test all permission values are correctly defined."""
        assert AgentPermission.EXECUTE.value == "execute"
        assert AgentPermission.READ_CONFIG.value == "read_config"
        assert AgentPermission.WRITE_CONFIG.value == "write_config"
        assert AgentPermission.READ_DATA.value == "read_data"
        assert AgentPermission.WRITE_DATA.value == "write_data"
        assert AgentPermission.MANAGE_LIFECYCLE.value == "manage_lifecycle"
        assert AgentPermission.VIEW_METRICS.value == "view_metrics"
        assert AgentPermission.EXPORT_PROVENANCE.value == "export_provenance"
        assert AgentPermission.ADMIN.value == "admin"

    def test_permission_from_string_valid(self):
        """Test creating permission from valid string."""
        perm = AgentPermission.from_string("execute")
        assert perm == AgentPermission.EXECUTE

        perm = AgentPermission.from_string("read_data")
        assert perm == AgentPermission.READ_DATA

    def test_permission_from_string_invalid(self):
        """Test creating permission from invalid string raises error."""
        with pytest.raises(ValueError, match="Invalid permission"):
            AgentPermission.from_string("invalid_permission")

    def test_permission_string_representation(self):
        """Test string representation of permission."""
        assert str(AgentPermission.EXECUTE) == "execute"
        assert str(AgentPermission.ADMIN) == "admin"


class TestAgentRole:
    """Test AgentRole model."""

    def test_role_creation(self):
        """Test creating a custom role."""
        role = AgentRole(
            role_name="custom_role",
            permissions={AgentPermission.EXECUTE, AgentPermission.READ_DATA},
            description="Custom role for testing"
        )

        assert role.role_name == "custom_role"
        assert len(role.permissions) == 2
        assert AgentPermission.EXECUTE in role.permissions
        assert AgentPermission.READ_DATA in role.permissions

    def test_role_has_permission(self):
        """Test checking if role has permission."""
        role = AgentRole(
            role_name="test_role",
            permissions={AgentPermission.EXECUTE, AgentPermission.READ_DATA},
            description="Test role"
        )

        assert role.has_permission(AgentPermission.EXECUTE)
        assert role.has_permission(AgentPermission.READ_DATA)
        assert not role.has_permission(AgentPermission.WRITE_CONFIG)

    def test_admin_role_has_all_permissions(self):
        """Test admin role has all permissions."""
        role = AgentRole(
            role_name="admin",
            permissions={AgentPermission.ADMIN},
            description="Admin role"
        )

        # Admin permission grants all permissions
        for perm in AgentPermission:
            assert role.has_permission(perm)

    def test_role_to_dict(self):
        """Test serializing role to dictionary."""
        role = AgentRole(
            role_name="test_role",
            permissions={AgentPermission.EXECUTE, AgentPermission.READ_DATA},
            description="Test role"
        )

        role_dict = role.to_dict()

        assert role_dict["role_name"] == "test_role"
        assert set(role_dict["permissions"]) == {"execute", "read_data"}
        assert role_dict["description"] == "Test role"

    def test_role_from_dict(self):
        """Test creating role from dictionary."""
        role_dict = {
            "role_name": "test_role",
            "permissions": ["execute", "read_data"],
            "description": "Test role"
        }

        role = AgentRole.from_dict(role_dict)

        assert role.role_name == "test_role"
        assert len(role.permissions) == 2
        assert AgentPermission.EXECUTE in role.permissions
        assert AgentPermission.READ_DATA in role.permissions


class TestPredefinedRoles:
    """Test predefined roles."""

    def test_all_predefined_roles_exist(self):
        """Test all expected predefined roles exist."""
        expected_roles = ["agent_viewer", "agent_operator", "agent_engineer", "agent_admin"]

        for role_name in expected_roles:
            assert role_name in PREDEFINED_ROLES

    def test_agent_viewer_permissions(self):
        """Test agent_viewer role has correct permissions."""
        role = PREDEFINED_ROLES["agent_viewer"]

        assert AgentPermission.READ_CONFIG in role.permissions
        assert AgentPermission.VIEW_METRICS in role.permissions
        assert AgentPermission.EXECUTE not in role.permissions

    def test_agent_operator_permissions(self):
        """Test agent_operator role has correct permissions."""
        role = PREDEFINED_ROLES["agent_operator"]

        assert AgentPermission.EXECUTE in role.permissions
        assert AgentPermission.READ_CONFIG in role.permissions
        assert AgentPermission.READ_DATA in role.permissions
        assert AgentPermission.WRITE_CONFIG not in role.permissions

    def test_agent_engineer_permissions(self):
        """Test agent_engineer role has correct permissions."""
        role = PREDEFINED_ROLES["agent_engineer"]

        assert AgentPermission.EXECUTE in role.permissions
        assert AgentPermission.WRITE_CONFIG in role.permissions
        assert AgentPermission.MANAGE_LIFECYCLE in role.permissions
        assert AgentPermission.ADMIN not in role.permissions

    def test_agent_admin_permissions(self):
        """Test agent_admin role has all permissions."""
        role = PREDEFINED_ROLES["agent_admin"]

        # Admin should have all permissions
        for perm in AgentPermission:
            assert perm in role.permissions


class TestAgentAccessControl:
    """Test AgentAccessControl."""

    def test_acl_creation(self):
        """Test creating ACL."""
        acl = AgentAccessControl(agent_id="GL-001")

        assert acl.agent_id == "GL-001"
        assert len(acl.user_roles) == 0
        assert len(acl.custom_roles) == 0

    def test_grant_role(self):
        """Test granting role to user."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_operator")

        assert "user@example.com" in acl.user_roles
        assert "agent_operator" in acl.user_roles["user@example.com"]

    def test_grant_multiple_roles(self):
        """Test granting multiple roles to user."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_viewer")
        acl.grant_role("user@example.com", "agent_operator")

        assert len(acl.user_roles["user@example.com"]) == 2
        assert "agent_viewer" in acl.user_roles["user@example.com"]
        assert "agent_operator" in acl.user_roles["user@example.com"]

    def test_grant_duplicate_role(self):
        """Test granting same role twice doesn't create duplicates."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_operator")
        acl.grant_role("user@example.com", "agent_operator")

        assert len(acl.user_roles["user@example.com"]) == 1

    def test_grant_invalid_role(self):
        """Test granting invalid role raises error."""
        acl = AgentAccessControl(agent_id="GL-001")

        with pytest.raises(ValueError, match="does not exist"):
            acl.grant_role("user@example.com", "invalid_role")

    def test_revoke_role(self):
        """Test revoking role from user."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_operator")
        acl.revoke_role("user@example.com", "agent_operator")

        assert "user@example.com" not in acl.user_roles

    def test_check_permission_granted(self):
        """Test checking permission when user has it."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_operator")

        assert acl.check_permission("user@example.com", AgentPermission.EXECUTE)
        assert acl.check_permission("user@example.com", AgentPermission.READ_DATA)

    def test_check_permission_denied(self):
        """Test checking permission when user doesn't have it."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_viewer")

        assert not acl.check_permission("user@example.com", AgentPermission.EXECUTE)
        assert not acl.check_permission("user@example.com", AgentPermission.WRITE_CONFIG)

    def test_check_permission_no_roles(self):
        """Test checking permission when user has no roles."""
        acl = AgentAccessControl(agent_id="GL-001")

        assert not acl.check_permission("user@example.com", AgentPermission.EXECUTE)

    def test_check_permission_admin_has_all(self):
        """Test admin role has all permissions."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("admin@example.com", "agent_admin")

        # Admin should have all permissions
        for perm in AgentPermission:
            assert acl.check_permission("admin@example.com", perm)

    def test_list_user_roles(self):
        """Test listing user roles."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_viewer")
        acl.grant_role("user@example.com", "agent_operator")

        roles = acl.list_user_roles("user@example.com")

        assert len(roles) == 2
        assert "agent_viewer" in roles
        assert "agent_operator" in roles

    def test_list_user_permissions(self):
        """Test listing aggregated user permissions."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user@example.com", "agent_viewer")
        acl.grant_role("user@example.com", "agent_operator")

        permissions = acl.list_user_permissions("user@example.com")

        # Should have permissions from both roles
        assert AgentPermission.READ_CONFIG in permissions
        assert AgentPermission.VIEW_METRICS in permissions
        assert AgentPermission.EXECUTE in permissions
        assert AgentPermission.READ_DATA in permissions

    def test_list_all_users(self):
        """Test listing all users with roles."""
        acl = AgentAccessControl(agent_id="GL-001")

        acl.grant_role("user1@example.com", "agent_operator")
        acl.grant_role("user2@example.com", "agent_viewer")

        users = acl.list_all_users()

        assert len(users) == 2
        assert "user1@example.com" in users
        assert "user2@example.com" in users

    def test_custom_role(self):
        """Test adding and using custom role."""
        acl = AgentAccessControl(agent_id="GL-001")

        custom_role = AgentRole(
            role_name="custom_executor",
            permissions={AgentPermission.EXECUTE},
            description="Custom executor role"
        )

        acl.add_custom_role(custom_role)
        acl.grant_role("user@example.com", "custom_executor")

        assert acl.check_permission("user@example.com", AgentPermission.EXECUTE)
        assert not acl.check_permission("user@example.com", AgentPermission.READ_DATA)

    def test_custom_role_override_predefined(self):
        """Test cannot override predefined role."""
        acl = AgentAccessControl(agent_id="GL-001")

        custom_role = AgentRole(
            role_name="agent_operator",  # Same name as predefined
            permissions={AgentPermission.EXECUTE},
            description="Override attempt"
        )

        with pytest.raises(ValueError, match="Cannot override predefined role"):
            acl.add_custom_role(custom_role)

    def test_remove_custom_role(self):
        """Test removing custom role."""
        acl = AgentAccessControl(agent_id="GL-001")

        custom_role = AgentRole(
            role_name="custom_role",
            permissions={AgentPermission.EXECUTE},
            description="Custom role"
        )

        acl.add_custom_role(custom_role)
        acl.grant_role("user@example.com", "custom_role")

        # Remove custom role should revoke from all users
        acl.remove_custom_role("custom_role")

        assert "custom_role" not in acl.custom_roles
        assert not acl.check_permission("user@example.com", AgentPermission.EXECUTE)

    def test_acl_to_dict(self):
        """Test serializing ACL to dictionary."""
        acl = AgentAccessControl(agent_id="GL-001")
        acl.grant_role("user@example.com", "agent_operator")

        acl_dict = acl.to_dict()

        assert acl_dict["agent_id"] == "GL-001"
        assert "user@example.com" in acl_dict["user_roles"]
        assert "agent_operator" in acl_dict["user_roles"]["user@example.com"]

    def test_acl_from_dict(self):
        """Test creating ACL from dictionary."""
        acl_dict = {
            "agent_id": "GL-001",
            "user_roles": {
                "user@example.com": ["agent_operator"]
            },
            "custom_roles": {}
        }

        acl = AgentAccessControl.from_dict(acl_dict)

        assert acl.agent_id == "GL-001"
        assert "user@example.com" in acl.user_roles
        assert "agent_operator" in acl.user_roles["user@example.com"]

    def test_calculate_hash(self):
        """Test calculating provenance hash."""
        acl = AgentAccessControl(agent_id="GL-001")
        acl.grant_role("user@example.com", "agent_operator")

        hash1 = acl.calculate_hash()

        # Hash should be consistent
        hash2 = acl.calculate_hash()
        assert hash1 == hash2

        # Hash should change when ACL changes
        acl.grant_role("user2@example.com", "agent_viewer")
        hash3 = acl.calculate_hash()
        assert hash1 != hash3


class TestAgentRBACManager:
    """Test AgentRBACManager."""

    def test_manager_creation(self):
        """Test creating RBAC manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            assert manager.storage_path == Path(tmpdir)
            assert len(manager.access_controls) == 0

    def test_create_acl(self):
        """Test creating ACL through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            acl = manager.create_acl("GL-001")

            assert acl.agent_id == "GL-001"
            assert "GL-001" in manager.access_controls

    def test_create_duplicate_acl(self):
        """Test creating duplicate ACL raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.create_acl("GL-001")

            with pytest.raises(ValueError, match="already exists"):
                manager.create_acl("GL-001")

    def test_get_acl(self):
        """Test getting ACL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.create_acl("GL-001")
            acl = manager.get_acl("GL-001")

            assert acl is not None
            assert acl.agent_id == "GL-001"

    def test_get_nonexistent_acl(self):
        """Test getting nonexistent ACL returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            acl = manager.get_acl("GL-999")

            assert acl is None

    def test_delete_acl(self):
        """Test deleting ACL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.create_acl("GL-001")
            manager.delete_acl("GL-001")

            assert "GL-001" not in manager.access_controls
            assert manager.get_acl("GL-001") is None

    def test_grant_role(self):
        """Test granting role through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")

            acl = manager.get_acl("GL-001")
            assert "user@example.com" in acl.user_roles
            assert "agent_operator" in acl.user_roles["user@example.com"]

    def test_grant_role_creates_acl(self):
        """Test granting role creates ACL if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            # Grant role without creating ACL first
            manager.grant_role("GL-001", "user@example.com", "agent_operator")

            # ACL should be created automatically
            acl = manager.get_acl("GL-001")
            assert acl is not None

    def test_revoke_role(self):
        """Test revoking role through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")
            manager.revoke_role("GL-001", "user@example.com", "agent_operator")

            acl = manager.get_acl("GL-001")
            assert "user@example.com" not in acl.user_roles

    def test_check_permission(self):
        """Test checking permission through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")

            assert manager.check_permission("GL-001", "user@example.com", AgentPermission.EXECUTE)
            assert not manager.check_permission("GL-001", "user@example.com", AgentPermission.WRITE_CONFIG)

    def test_check_permission_no_acl(self):
        """Test checking permission when no ACL exists returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            assert not manager.check_permission("GL-001", "user@example.com", AgentPermission.EXECUTE)

    def test_list_user_agents(self):
        """Test listing all agents user has access to."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")
            manager.grant_role("GL-002", "user@example.com", "agent_viewer")

            agents = manager.list_user_agents("user@example.com")

            assert len(agents) == 2
            assert "GL-001" in agents
            assert "GL-002" in agents

    def test_audit_user_access(self):
        """Test auditing user access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")
            manager.grant_role("GL-002", "user@example.com", "agent_viewer")

            audit = manager.audit_user_access("user@example.com")

            assert "GL-001" in audit
            assert "agent_operator" in audit["GL-001"]
            assert "GL-002" in audit
            assert "agent_viewer" in audit["GL-002"]

    def test_persistence(self):
        """Test ACLs are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manager and add ACL
            manager1 = AgentRBACManager(storage_path=Path(tmpdir))
            manager1.grant_role("GL-001", "user@example.com", "agent_operator")

            # Create new manager instance (should load from disk)
            manager2 = AgentRBACManager(storage_path=Path(tmpdir))

            # Check ACL was loaded
            acl = manager2.get_acl("GL-001")
            assert acl is not None
            assert "user@example.com" in acl.user_roles

    def test_export_audit_log(self):
        """Test exporting audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AgentRBACManager(storage_path=Path(tmpdir))

            manager.grant_role("GL-001", "user@example.com", "agent_operator")
            manager.grant_role("GL-002", "admin@example.com", "agent_admin")

            # Export audit log
            output_path = Path(tmpdir) / "audit.json"
            manager.export_audit_log(output_path)

            # Verify file exists
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                audit_data = json.load(f)

            assert "acls" in audit_data
            assert "GL-001" in audit_data["acls"]
            assert "GL-002" in audit_data["acls"]
            assert "hashes" in audit_data
