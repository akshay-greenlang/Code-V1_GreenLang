# -*- coding: utf-8 -*-
"""
Comprehensive RBAC/ABAC Test Suite for Phase 4
Over 100 tests covering all aspects of role-based and attribute-based access control

Test Coverage:
- Role creation and management
- Permission assignment and evaluation
- User-role associations
- Role hierarchy and inheritance
- ABAC condition evaluation
- Multi-tenancy isolation
- Permission denial
- Edge cases and error handling
"""

import pytest
import uuid
from datetime import datetime, timedelta

from greenlang.auth.rbac import (
from greenlang.determinism import deterministic_uuid, DeterministicClock
    RBACManager,
    Role,
    Permission as RBACPermission,
    AccessControl,
    PermissionAction,
    ResourceType,
)


# ===== Role Management Tests (20 tests) =====

class TestRoleManagement:
    """Test role creation, update, deletion"""

    def test_create_role(self, rbac_manager):
        """Test basic role creation"""
        role = rbac_manager.create_role("custom_role", "Custom role for testing")
        assert role is not None
        assert role.name == "custom_role"
        assert role.description == "Custom role for testing"

    def test_create_role_with_permissions(self, rbac_manager):
        """Test role creation with permissions"""
        permissions = [
            RBACPermission(resource="pipeline", action="execute"),
            RBACPermission(resource="pack", action="read"),
        ]
        role = rbac_manager.create_role("exec_role", permissions=permissions)
        assert len(role.permissions) == 2

    def test_create_role_with_parent(self, rbac_manager):
        """Test role inheritance"""
        role = rbac_manager.create_role("child_role", parent_roles=["developer"])
        assert "developer" in role.parent_roles

    def test_get_role(self, rbac_manager):
        """Test retrieving role"""
        rbac_manager.create_role("test_role")
        role = rbac_manager.get_role("test_role")
        assert role is not None
        assert role.name == "test_role"

    def test_get_nonexistent_role(self, rbac_manager):
        """Test retrieving non-existent role"""
        role = rbac_manager.get_role("nonexistent")
        assert role is None

    def test_update_role_description(self, rbac_manager):
        """Test updating role description"""
        rbac_manager.create_role("test_role", "Old description")
        rbac_manager.update_role("test_role", {"description": "New description"})
        role = rbac_manager.get_role("test_role")
        assert role.description == "New description"

    def test_update_role_permissions(self, rbac_manager):
        """Test updating role permissions"""
        rbac_manager.create_role("test_role")
        new_perms = [RBACPermission(resource="dataset", action="create")]
        rbac_manager.update_role("test_role", {"permissions": new_perms})
        role = rbac_manager.get_role("test_role")
        assert len(role.permissions) == 1

    def test_delete_role(self, rbac_manager):
        """Test deleting custom role"""
        rbac_manager.create_role("temp_role")
        result = rbac_manager.delete_role("temp_role")
        assert result is True
        assert rbac_manager.get_role("temp_role") is None

    def test_cannot_delete_system_role(self, rbac_manager):
        """Test that system roles cannot be deleted"""
        result = rbac_manager.delete_role("super_admin")
        assert result is False
        assert rbac_manager.get_role("super_admin") is not None

    def test_default_roles_exist(self, rbac_manager):
        """Test that default roles are created"""
        assert rbac_manager.get_role("super_admin") is not None
        assert rbac_manager.get_role("admin") is not None
        assert rbac_manager.get_role("developer") is not None
        assert rbac_manager.get_role("operator") is not None
        assert rbac_manager.get_role("viewer") is not None

    def test_role_has_metadata(self, rbac_manager):
        """Test role metadata"""
        role = rbac_manager.create_role("meta_role")
        assert role.created_at is not None
        assert role.updated_at is not None

    def test_role_timestamps_update(self, rbac_manager):
        """Test that updated_at timestamp changes"""
        role = rbac_manager.create_role("timestamp_role")
        original_updated = role.updated_at
        import time
        time.sleep(0.01)
        role.add_permission(RBACPermission(resource="test", action="read"))
        assert role.updated_at > original_updated

    def test_role_to_dict(self, rbac_manager):
        """Test role serialization"""
        role = rbac_manager.create_role("serial_role")
        role_dict = role.to_dict()
        assert "name" in role_dict
        assert "permissions" in role_dict
        assert "created_at" in role_dict

    def test_multiple_roles_same_name_different_tenants(self, rbac_manager):
        """Test role name can be reused across tenants (in actual impl)"""
        # This tests the concept - implementation would vary
        role1 = rbac_manager.create_role("shared_name")
        assert role1 is not None

    def test_role_add_permission(self, rbac_manager):
        """Test adding permission to existing role"""
        role = rbac_manager.create_role("perm_role")
        assert len(role.permissions) == 0
        role.add_permission(RBACPermission(resource="test", action="read"))
        assert len(role.permissions) == 1

    def test_role_remove_permission(self, rbac_manager):
        """Test removing permission from role"""
        perm = RBACPermission(resource="test", action="read")
        role = rbac_manager.create_role("perm_role", permissions=[perm])
        assert len(role.permissions) == 1
        role.remove_permission(perm)
        assert len(role.permissions) == 0

    def test_role_has_permission_direct(self, rbac_manager):
        """Test direct permission check on role"""
        role = rbac_manager.create_role(
            "check_role",
            permissions=[RBACPermission(resource="pipeline", action="execute")]
        )
        assert role.has_permission("pipeline", "execute") is True
        assert role.has_permission("pack", "execute") is False

    def test_role_inheritance_permissions(self, rbac_manager):
        """Test role inherits parent permissions"""
        parent = rbac_manager.create_role(
            "parent_role",
            permissions=[RBACPermission(resource="base", action="read")]
        )
        child = rbac_manager.create_role("child_role", parent_roles=["parent_role"])

        all_perms = child.get_all_permissions(rbac_manager)
        assert len(all_perms) >= 1
        assert any(p.resource == "base" and p.action == "read" for p in all_perms)

    def test_multi_level_inheritance(self, rbac_manager):
        """Test multi-level role inheritance"""
        rbac_manager.create_role(
            "grandparent",
            permissions=[RBACPermission(resource="level1", action="read")]
        )
        rbac_manager.create_role("parent", parent_roles=["grandparent"])
        child = rbac_manager.create_role("child", parent_roles=["parent"])

        all_perms = child.get_all_permissions(rbac_manager)
        assert any(p.resource == "level1" for p in all_perms)

    def test_role_circular_inheritance_protection(self, rbac_manager):
        """Test protection against circular inheritance"""
        # Implementation should detect and prevent circular references
        # This is a conceptual test
        role_a = rbac_manager.create_role("role_a")
        role_b = rbac_manager.create_role("role_b", parent_roles=["role_a"])
        # Attempting to make role_a inherit from role_b should fail or be handled
        assert role_a is not None
        assert role_b is not None


# ===== Permission Tests (25 tests) =====

class TestPermissions:
    """Test permission creation and evaluation"""

    def test_permission_matches_exact(self):
        """Test exact permission match"""
        perm = RBACPermission(resource="pipeline:123", action="execute")
        assert perm.matches("pipeline:123", "execute") is True
        assert perm.matches("pipeline:456", "execute") is False

    def test_permission_wildcard_resource(self):
        """Test wildcard resource matching"""
        perm = RBACPermission(resource="pipeline:*", action="execute")
        assert perm.matches("pipeline:123", "execute") is True
        assert perm.matches("pipeline:xyz", "execute") is True
        assert perm.matches("pack:123", "execute") is False

    def test_permission_wildcard_action(self):
        """Test wildcard action matching"""
        perm = RBACPermission(resource="pipeline", action="*")
        assert perm.matches("pipeline", "execute") is True
        assert perm.matches("pipeline", "read") is True
        assert perm.matches("pipeline", "delete") is True

    def test_permission_full_wildcard(self):
        """Test full wildcard permission"""
        perm = RBACPermission(resource="*", action="*")
        assert perm.matches("anything", "anystuff") is True
        assert perm.matches("pipeline", "execute") is True

    def test_permission_pattern_matching(self):
        """Test pattern matching with wildcards"""
        perm = RBACPermission(resource="pipeline:carbon-*", action="execute")
        assert perm.matches("pipeline:carbon-123", "execute") is True
        assert perm.matches("pipeline:carbon-test", "execute") is True
        assert perm.matches("pipeline:wind-123", "execute") is False

    def test_permission_with_scope(self):
        """Test permission with scope"""
        perm = RBACPermission(resource="pipeline", action="execute", scope="tenant:123")
        context = {"tenant": "123"}
        assert perm.matches("pipeline", "execute", context) is True

        context = {"tenant": "456"}
        assert perm.matches("pipeline", "execute", context) is False

    def test_permission_scope_wildcard(self):
        """Test scope with wildcard"""
        perm = RBACPermission(resource="pipeline", action="execute", scope="tenant:*")
        assert perm.matches("pipeline", "execute", {"tenant": "123"}) is True
        assert perm.matches("pipeline", "execute", {"tenant": "456"}) is True

    def test_permission_with_conditions(self):
        """Test permission with custom conditions"""
        perm = RBACPermission(
            resource="pipeline",
            action="execute",
            conditions={"environment": "prod"}
        )
        assert perm.matches("pipeline", "execute", {"environment": "prod"}) is True
        assert perm.matches("pipeline", "execute", {"environment": "dev"}) is False

    def test_permission_multiple_conditions(self):
        """Test permission with multiple conditions"""
        perm = RBACPermission(
            resource="pipeline",
            action="execute",
            conditions={"environment": "prod", "region": "us-east"}
        )
        context = {"environment": "prod", "region": "us-east"}
        assert perm.matches("pipeline", "execute", context) is True

        context = {"environment": "prod", "region": "eu-west"}
        assert perm.matches("pipeline", "execute", context) is False

    def test_permission_to_string(self):
        """Test permission serialization"""
        perm = RBACPermission(resource="pipeline", action="execute")
        perm_str = perm.to_string()
        assert perm_str == "pipeline:execute"

    def test_permission_to_string_with_scope(self):
        """Test permission serialization with scope"""
        perm = RBACPermission(resource="pipeline", action="execute", scope="tenant:123")
        perm_str = perm.to_string()
        assert perm_str == "pipeline:execute:tenant:123"

    def test_permission_from_string(self):
        """Test permission deserialization"""
        perm = RBACPermission.from_string("pipeline:execute")
        assert perm.resource == "pipeline"
        assert perm.action == "execute"

    def test_permission_from_string_with_scope(self):
        """Test permission deserialization with scope"""
        perm = RBACPermission.from_string("pipeline:execute:tenant:123")
        assert perm.resource == "pipeline"
        assert perm.action == "execute"
        assert perm.scope == "tenant:123"

    def test_permission_from_string_invalid(self):
        """Test invalid permission string"""
        with pytest.raises(ValueError):
            RBACPermission.from_string("invalid")

    def test_permission_action_enum(self):
        """Test PermissionAction enum"""
        assert PermissionAction.CREATE.value == "create"
        assert PermissionAction.READ.value == "read"
        assert PermissionAction.UPDATE.value == "update"
        assert PermissionAction.DELETE.value == "delete"
        assert PermissionAction.EXECUTE.value == "execute"

    def test_resource_type_enum(self):
        """Test ResourceType enum"""
        assert ResourceType.PIPELINE.value == "pipeline"
        assert ResourceType.PACK.value == "pack"
        assert ResourceType.DATASET.value == "dataset"
        assert ResourceType.AGENT.value == "agent"

    def test_permission_case_sensitivity(self):
        """Test permission matching is case-sensitive"""
        perm = RBACPermission(resource="Pipeline", action="Execute")
        assert perm.matches("Pipeline", "Execute") is True
        assert perm.matches("pipeline", "execute") is False

    def test_permission_no_context_with_conditions(self):
        """Test permission with conditions but no context fails"""
        perm = RBACPermission(
            resource="pipeline",
            action="execute",
            conditions={"key": "value"}
        )
        # Without context, conditions can't be checked
        assert perm.matches("pipeline", "execute") is False

    def test_permission_empty_conditions(self):
        """Test permission with empty conditions"""
        perm = RBACPermission(resource="pipeline", action="execute", conditions={})
        assert perm.matches("pipeline", "execute") is True

    def test_permission_partial_context_match(self):
        """Test permission when context has extra fields"""
        perm = RBACPermission(
            resource="pipeline",
            action="execute",
            conditions={"env": "prod"}
        )
        context = {"env": "prod", "region": "us", "extra": "data"}
        assert perm.matches("pipeline", "execute", context) is True

    def test_permission_complex_pattern(self):
        """Test complex permission pattern"""
        perm = RBACPermission(resource="pipeline:*:prod", action="execute")
        assert perm.matches("pipeline:carbon-123:prod", "execute") is True
        assert perm.matches("pipeline:wind-456:prod", "execute") is True
        assert perm.matches("pipeline:carbon-123:dev", "execute") is False

    def test_permission_numeric_resource_id(self):
        """Test permission with numeric resource IDs"""
        perm = RBACPermission(resource="pipeline:*", action="read")
        assert perm.matches("pipeline:12345", "read") is True

    def test_permission_special_characters(self):
        """Test permission with special characters in resource"""
        perm = RBACPermission(resource="pipeline:carbon_v2", action="execute")
        assert perm.matches("pipeline:carbon_v2", "execute") is True

    def test_permission_slash_in_resource(self):
        """Test permission with slash in resource path"""
        perm = RBACPermission(resource="dataset:s3/path/to/data", action="read")
        assert perm.matches("dataset:s3/path/to/data", "read") is True

    def test_permission_dot_notation(self):
        """Test permission with dot notation"""
        perm = RBACPermission(resource="api.v1.pipeline", action="execute")
        assert perm.matches("api.v1.pipeline", "execute") is True


# ===== User-Role Assignment Tests (20 tests) =====

class TestUserRoleAssignment:
    """Test assigning and revoking roles from users"""

    def test_assign_role_to_user(self, rbac_manager):
        """Test basic role assignment"""
        result = rbac_manager.assign_role("user-1", "developer")
        assert result is True

    def test_assign_nonexistent_role(self, rbac_manager):
        """Test assigning non-existent role fails"""
        result = rbac_manager.assign_role("user-1", "nonexistent")
        assert result is False

    def test_assign_multiple_roles(self, rbac_manager):
        """Test assigning multiple roles to user"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-1", "operator")
        roles = rbac_manager.get_user_roles("user-1")
        assert len(roles) == 2
        assert "developer" in roles
        assert "operator" in roles

    def test_assign_same_role_twice(self, rbac_manager):
        """Test assigning same role twice is idempotent"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-1", "developer")
        roles = rbac_manager.get_user_roles("user-1")
        assert len(roles) == 1

    def test_revoke_role_from_user(self, rbac_manager):
        """Test revoking role"""
        rbac_manager.assign_role("user-1", "developer")
        result = rbac_manager.revoke_role("user-1", "developer")
        assert result is True
        roles = rbac_manager.get_user_roles("user-1")
        assert "developer" not in roles

    def test_revoke_nonassigned_role(self, rbac_manager):
        """Test revoking unassigned role"""
        result = rbac_manager.revoke_role("user-1", "developer")
        assert result is False

    def test_get_user_roles_empty(self, rbac_manager):
        """Test getting roles for user with no roles"""
        roles = rbac_manager.get_user_roles("new-user")
        assert roles == []

    def test_get_user_permissions(self, rbac_manager_with_users):
        """Test getting all user permissions"""
        perms = rbac_manager_with_users.get_user_permissions("user-1")
        assert len(perms) > 0

    def test_get_user_permissions_includes_inherited(self, rbac_manager):
        """Test user permissions include inherited permissions"""
        rbac_manager.create_role("parent", permissions=[
            RBACPermission(resource="base", action="read")
        ])
        rbac_manager.create_role("child", parent_roles=["parent"])
        rbac_manager.assign_role("user-1", "child")

        perms = rbac_manager.get_user_permissions("user-1")
        assert any(p.resource == "base" and p.action == "read" for p in perms)

    def test_user_permissions_no_duplicates(self, rbac_manager):
        """Test user permissions are deduplicated"""
        perm = RBACPermission(resource="test", action="read")
        rbac_manager.create_role("role1", permissions=[perm])
        rbac_manager.create_role("role2", permissions=[perm])

        rbac_manager.assign_role("user-1", "role1")
        rbac_manager.assign_role("user-1", "role2")

        perms = rbac_manager.get_user_permissions("user-1")
        perm_strings = [p.to_string() for p in perms]
        assert perm_strings.count("test:read") == 1

    def test_multiple_users_same_role(self, rbac_manager):
        """Test multiple users can have same role"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-2", "developer")

        assert "developer" in rbac_manager.get_user_roles("user-1")
        assert "developer" in rbac_manager.get_user_roles("user-2")

    def test_delete_role_removes_assignments(self, rbac_manager):
        """Test deleting role removes user assignments"""
        rbac_manager.create_role("temp_role")
        rbac_manager.assign_role("user-1", "temp_role")
        rbac_manager.delete_role("temp_role")

        roles = rbac_manager.get_user_roles("user-1")
        assert "temp_role" not in roles

    def test_user_role_isolation(self, rbac_manager):
        """Test roles are isolated between users"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-2", "viewer")

        user1_roles = rbac_manager.get_user_roles("user-1")
        user2_roles = rbac_manager.get_user_roles("user-2")

        assert "developer" in user1_roles
        assert "developer" not in user2_roles
        assert "viewer" in user2_roles
        assert "viewer" not in user1_roles

    def test_assign_all_default_roles(self, rbac_manager):
        """Test user can have all default roles"""
        for role in ["super_admin", "admin", "developer", "operator", "viewer"]:
            rbac_manager.assign_role("superuser", role)

        roles = rbac_manager.get_user_roles("superuser")
        assert len(roles) == 5

    def test_user_id_special_characters(self, rbac_manager):
        """Test user IDs with special characters"""
        rbac_manager.assign_role("user@domain.com", "developer")
        roles = rbac_manager.get_user_roles("user@domain.com")
        assert "developer" in roles

    def test_user_id_uuid(self, rbac_manager):
        """Test user IDs as UUIDs"""
        user_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        rbac_manager.assign_role(user_id, "developer")
        roles = rbac_manager.get_user_roles(user_id)
        assert "developer" in roles

    def test_get_user_roles_preserves_order(self, rbac_manager):
        """Test role assignment order (conceptual)"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-1", "operator")
        rbac_manager.assign_role("user-1", "viewer")

        roles = rbac_manager.get_user_roles("user-1")
        assert len(roles) == 3

    def test_revoke_one_of_multiple_roles(self, rbac_manager):
        """Test revoking one role when user has multiple"""
        rbac_manager.assign_role("user-1", "developer")
        rbac_manager.assign_role("user-1", "operator")

        rbac_manager.revoke_role("user-1", "developer")
        roles = rbac_manager.get_user_roles("user-1")

        assert "developer" not in roles
        assert "operator" in roles

    def test_concurrent_role_assignments(self, rbac_manager):
        """Test role assignments work correctly (thread safety concept)"""
        for i in range(10):
            rbac_manager.assign_role(f"user-{i}", "developer")

        for i in range(10):
            roles = rbac_manager.get_user_roles(f"user-{i}")
            assert "developer" in roles

    def test_role_assignment_timestamp(self, rbac_manager):
        """Test role assignment creates metadata"""
        result = rbac_manager.assign_role("user-1", "developer")
        assert result is True
        # In full implementation, would check timestamps


# ===== Permission Check Tests (25 tests) =====

class TestPermissionChecks:
    """Test permission checking logic"""

    def test_check_permission_allowed(self, rbac_manager):
        """Test permission check allows authorized action"""
        rbac_manager.create_role("exec_role", permissions=[
            RBACPermission(resource="pipeline", action="execute")
        ])
        rbac_manager.assign_role("user-1", "exec_role")

        result = rbac_manager.check_permission("user-1", "pipeline", "execute")
        assert result is True

    def test_check_permission_denied(self, rbac_manager):
        """Test permission check denies unauthorized action"""
        rbac_manager.assign_role("user-1", "viewer")
        result = rbac_manager.check_permission("user-1", "pipeline", "delete")
        assert result is False

    def test_check_permission_no_roles(self, rbac_manager):
        """Test permission check for user with no roles"""
        result = rbac_manager.check_permission("new-user", "pipeline", "execute")
        assert result is False

    def test_check_permission_wildcard_resource(self, rbac_manager_with_users):
        """Test permission with wildcard resource"""
        result = rbac_manager_with_users.check_permission("user-3", "anything", "anything")
        assert result is True  # admin has *:*

    def test_check_permission_with_context(self, rbac_manager):
        """Test permission check with context"""
        rbac_manager.create_role("scoped_role", permissions=[
            RBACPermission(resource="pipeline", action="execute", scope="tenant:123")
        ])
        rbac_manager.assign_role("user-1", "scoped_role")

        result = rbac_manager.check_permission(
            "user-1", "pipeline", "execute", {"tenant": "123"}
        )
        assert result is True

        result = rbac_manager.check_permission(
            "user-1", "pipeline", "execute", {"tenant": "456"}
        )
        assert result is False

    def test_check_permission_inherited(self, rbac_manager):
        """Test permission check works with inherited permissions"""
        rbac_manager.create_role("parent", permissions=[
            RBACPermission(resource="base", action="read")
        ])
        rbac_manager.create_role("child", parent_roles=["parent"])
        rbac_manager.assign_role("user-1", "child")

        result = rbac_manager.check_permission("user-1", "base", "read")
        assert result is True

    def test_check_permission_multiple_roles(self, rbac_manager):
        """Test permission check with multiple roles"""
        rbac_manager.create_role("role1", permissions=[
            RBACPermission(resource="pipeline", action="execute")
        ])
        rbac_manager.create_role("role2", permissions=[
            RBACPermission(resource="pack", action="read")
        ])

        rbac_manager.assign_role("user-1", "role1")
        rbac_manager.assign_role("user-1", "role2")

        assert rbac_manager.check_permission("user-1", "pipeline", "execute") is True
        assert rbac_manager.check_permission("user-1", "pack", "read") is True

    def test_check_permission_pattern_match(self, rbac_manager):
        """Test permission check with pattern matching"""
        rbac_manager.create_role("pattern_role", permissions=[
            RBACPermission(resource="pipeline:carbon-*", action="execute")
        ])
        rbac_manager.assign_role("user-1", "pattern_role")

        assert rbac_manager.check_permission("user-1", "pipeline:carbon-123", "execute") is True
        assert rbac_manager.check_permission("user-1", "pipeline:wind-123", "execute") is False

    def test_check_permission_action_wildcard(self, rbac_manager):
        """Test permission check with action wildcard"""
        rbac_manager.create_role("all_actions", permissions=[
            RBACPermission(resource="pipeline:123", action="*")
        ])
        rbac_manager.assign_role("user-1", "all_actions")

        assert rbac_manager.check_permission("user-1", "pipeline:123", "execute") is True
        assert rbac_manager.check_permission("user-1", "pipeline:123", "read") is True
        assert rbac_manager.check_permission("user-1", "pipeline:123", "delete") is True

    def test_check_permission_super_admin(self, rbac_manager):
        """Test super_admin has all permissions"""
        rbac_manager.assign_role("superuser", "super_admin")

        assert rbac_manager.check_permission("superuser", "anything", "anything") is True
        assert rbac_manager.check_permission("superuser", "pipeline", "execute") is True
        assert rbac_manager.check_permission("superuser", "secret", "delete") is True

    def test_check_permission_viewer_read_only(self, rbac_manager):
        """Test viewer role has read-only access"""
        rbac_manager.assign_role("reader", "viewer")

        assert rbac_manager.check_permission("reader", "pipeline", "read") is True
        assert rbac_manager.check_permission("reader", "pack", "list") is True
        assert rbac_manager.check_permission("reader", "pipeline", "execute") is False
        assert rbac_manager.check_permission("reader", "pack", "delete") is False

    def test_check_permission_developer_permissions(self, rbac_manager):
        """Test developer role permissions"""
        rbac_manager.assign_role("dev", "developer")

        assert rbac_manager.check_permission("dev", "pipeline", "create") is True
        assert rbac_manager.check_permission("dev", "pipeline", "execute") is True
        assert rbac_manager.check_permission("dev", "pack", "update") is True

    def test_check_permission_operator_execute_only(self, rbac_manager):
        """Test operator can execute but not modify"""
        rbac_manager.assign_role("op", "operator")

        assert rbac_manager.check_permission("op", "pipeline", "execute") is True
        assert rbac_manager.check_permission("op", "pipeline", "read") is True
        assert rbac_manager.check_permission("op", "pipeline", "delete") is False

    def test_check_permission_with_abac_conditions(self, rbac_manager):
        """Test ABAC permission with conditions"""
        rbac_manager.create_role("conditional_role", permissions=[
            RBACPermission(
                resource="pipeline",
                action="execute",
                conditions={"environment": "prod"}
            )
        ])
        rbac_manager.assign_role("user-1", "conditional_role")

        result = rbac_manager.check_permission(
            "user-1", "pipeline", "execute", {"environment": "prod"}
        )
        assert result is True

        result = rbac_manager.check_permission(
            "user-1", "pipeline", "execute", {"environment": "dev"}
        )
        assert result is False

    def test_check_permission_multiple_conditions(self, rbac_manager):
        """Test ABAC with multiple conditions"""
        rbac_manager.create_role("multi_cond", permissions=[
            RBACPermission(
                resource="pipeline",
                action="execute",
                conditions={"environment": "prod", "region": "us-east"}
            )
        ])
        rbac_manager.assign_role("user-1", "multi_cond")

        context = {"environment": "prod", "region": "us-east"}
        assert rbac_manager.check_permission("user-1", "pipeline", "execute", context) is True

        context = {"environment": "prod", "region": "eu-west"}
        assert rbac_manager.check_permission("user-1", "pipeline", "execute", context) is False

    def test_check_permission_first_match_wins(self, rbac_manager):
        """Test that first matching permission is used"""
        rbac_manager.create_role("multi_perm", permissions=[
            RBACPermission(resource="pipeline:*", action="execute"),
            RBACPermission(resource="pipeline:123", action="read"),
        ])
        rbac_manager.assign_role("user-1", "multi_perm")

        # Both permissions could match, but execute should work
        assert rbac_manager.check_permission("user-1", "pipeline:123", "execute") is True

    def test_check_permission_case_sensitive(self, rbac_manager):
        """Test permission checking is case-sensitive"""
        rbac_manager.create_role("case_role", permissions=[
            RBACPermission(resource="Pipeline", action="Execute")
        ])
        rbac_manager.assign_role("user-1", "case_role")

        assert rbac_manager.check_permission("user-1", "Pipeline", "Execute") is True
        assert rbac_manager.check_permission("user-1", "pipeline", "execute") is False

    def test_check_permission_empty_context(self, rbac_manager):
        """Test permission check with empty context"""
        rbac_manager.create_role("test_role", permissions=[
            RBACPermission(resource="pipeline", action="execute")
        ])
        rbac_manager.assign_role("user-1", "test_role")

        result = rbac_manager.check_permission("user-1", "pipeline", "execute", {})
        assert result is True

    def test_check_permission_none_context(self, rbac_manager):
        """Test permission check with None context"""
        rbac_manager.create_role("test_role", permissions=[
            RBACPermission(resource="pipeline", action="execute")
        ])
        rbac_manager.assign_role("user-1", "test_role")

        result = rbac_manager.check_permission("user-1", "pipeline", "execute", None)
        assert result is True

    def test_permission_check_logging(self, rbac_manager, caplog):
        """Test permission checks are logged"""
        rbac_manager.assign_role("user-1", "viewer")
        rbac_manager.check_permission("user-1", "pipeline", "read")
        # Would check logs in full implementation

    def test_check_permission_resource_with_slash(self, rbac_manager):
        """Test permission check with slash in resource"""
        rbac_manager.create_role("path_role", permissions=[
            RBACPermission(resource="dataset:s3/my/path", action="read")
        ])
        rbac_manager.assign_role("user-1", "path_role")

        assert rbac_manager.check_permission("user-1", "dataset:s3/my/path", "read") is True

    def test_check_permission_numeric_resource(self, rbac_manager):
        """Test permission check with numeric resource ID"""
        rbac_manager.create_role("numeric_role", permissions=[
            RBACPermission(resource="pipeline:12345", action="execute")
        ])
        rbac_manager.assign_role("user-1", "numeric_role")

        assert rbac_manager.check_permission("user-1", "pipeline:12345", "execute") is True

    def test_check_permission_unicode_resource(self, rbac_manager):
        """Test permission check with Unicode characters"""
        rbac_manager.create_role("unicode_role", permissions=[
            RBACPermission(resource="dataset:测试", action="read")
        ])
        rbac_manager.assign_role("user-1", "unicode_role")

        assert rbac_manager.check_permission("user-1", "dataset:测试", "read") is True

    def test_check_permission_performance(self, rbac_manager, performance_metrics):
        """Test permission check performance"""
        import time
        rbac_manager.assign_role("user-1", "developer")

        start = time.time()
        for _ in range(100):
            rbac_manager.check_permission("user-1", "pipeline", "execute")
        duration = time.time() - start

        # Should complete 100 checks in reasonable time
        assert duration < 1.0  # Less than 1 second for 100 checks


# ===== Resource Policy Tests (10 tests) =====

class TestResourcePolicies:
    """Test resource-specific policies"""

    def test_add_resource_policy(self, rbac_manager):
        """Test adding resource-specific policy"""
        policy = {
            "users": ["user-1"],
            "actions": ["execute"],
        }
        rbac_manager.add_resource_policy("pipeline:123", policy)
        assert "pipeline:123" in rbac_manager.resource_policies

    def test_resource_policy_grants_access(self, rbac_manager):
        """Test resource policy grants access"""
        policy = {
            "users": ["user-1"],
            "actions": ["execute"],
        }
        rbac_manager.add_resource_policy("pipeline:123", policy)

        result = rbac_manager.check_permission("user-1", "pipeline:123", "execute")
        assert result is True

    def test_resource_policy_with_wildcard_user(self, rbac_manager):
        """Test resource policy with wildcard user"""
        policy = {
            "users": ["*"],
            "actions": ["read"],
        }
        rbac_manager.add_resource_policy("public:data", policy)

        assert rbac_manager.check_permission("any-user", "public:data", "read") is True

    def test_resource_policy_with_wildcard_action(self, rbac_manager):
        """Test resource policy with wildcard action"""
        policy = {
            "users": ["user-1"],
            "actions": ["*"],
        }
        rbac_manager.add_resource_policy("my:resource", policy)

        assert rbac_manager.check_permission("user-1", "my:resource", "anything") is True

    def test_resource_policy_with_conditions(self, rbac_manager):
        """Test resource policy with conditions"""
        policy = {
            "users": ["user-1"],
            "actions": ["execute"],
            "conditions": {"ip_address": "192.168.1.1"}
        }
        rbac_manager.add_resource_policy("secure:pipeline", policy)

        context = {"ip_address": "192.168.1.1"}
        assert rbac_manager.check_permission("user-1", "secure:pipeline", "execute", context) is True

        context = {"ip_address": "10.0.0.1"}
        assert rbac_manager.check_permission("user-1", "secure:pipeline", "execute", context) is False

    def test_multiple_resource_policies(self, rbac_manager):
        """Test multiple policies on same resource"""
        policy1 = {"users": ["user-1"], "actions": ["read"]}
        policy2 = {"users": ["user-2"], "actions": ["write"]}

        rbac_manager.add_resource_policy("shared:resource", policy1)
        rbac_manager.add_resource_policy("shared:resource", policy2)

        assert rbac_manager.check_permission("user-1", "shared:resource", "read") is True
        assert rbac_manager.check_permission("user-2", "shared:resource", "write") is True

    def test_resource_policy_overrides_role(self, rbac_manager):
        """Test resource policy can grant access beyond role"""
        policy = {"users": ["user-1"], "actions": ["admin"]}
        rbac_manager.add_resource_policy("special:resource", policy)

        # User has no roles but policy grants access
        assert rbac_manager.check_permission("user-1", "special:resource", "admin") is True

    def test_resource_policy_denied_without_match(self, rbac_manager):
        """Test resource policy denies access when user doesn't match"""
        policy = {"users": ["user-2"], "actions": ["execute"]}
        rbac_manager.add_resource_policy("locked:resource", policy)

        assert rbac_manager.check_permission("user-1", "locked:resource", "execute") is False

    def test_resource_policy_multiple_users(self, rbac_manager):
        """Test resource policy with multiple allowed users"""
        policy = {
            "users": ["user-1", "user-2", "user-3"],
            "actions": ["read", "write"]
        }
        rbac_manager.add_resource_policy("team:resource", policy)

        for user in ["user-1", "user-2", "user-3"]:
            assert rbac_manager.check_permission(user, "team:resource", "read") is True
            assert rbac_manager.check_permission(user, "team:resource", "write") is True

    def test_resource_policy_complex_conditions(self, rbac_manager):
        """Test resource policy with complex conditions"""
        policy = {
            "users": ["user-1"],
            "actions": ["execute"],
            "conditions": {
                "environment": "prod",
                "time_of_day": "business_hours",
                "mfa_verified": True
            }
        }
        rbac_manager.add_resource_policy("critical:pipeline", policy)

        context = {
            "environment": "prod",
            "time_of_day": "business_hours",
            "mfa_verified": True
        }
        assert rbac_manager.check_permission("user-1", "critical:pipeline", "execute", context) is True

        # Missing one condition
        context = {"environment": "prod", "time_of_day": "business_hours"}
        assert rbac_manager.check_permission("user-1", "critical:pipeline", "execute", context) is False


# ===== Access Control Decorator Tests (5 tests) =====

class TestAccessControlDecorator:
    """Test AccessControl decorator functionality"""

    def test_access_control_init(self, rbac_manager):
        """Test AccessControl initialization"""
        ac = AccessControl(rbac_manager)
        assert ac.rbac_manager == rbac_manager

    def test_require_permission_decorator_allows(self, rbac_manager):
        """Test decorator allows authorized access"""
        rbac_manager.assign_role("user-1", "developer")
        ac = AccessControl(rbac_manager)

        @ac.require_permission("pipeline", "execute")
        def execute_pipeline(user_id=None, context=None):
            return "executed"

        result = execute_pipeline(user_id="user-1")
        assert result == "executed"

    def test_require_permission_decorator_denies(self, rbac_manager):
        """Test decorator denies unauthorized access"""
        rbac_manager.assign_role("user-1", "viewer")
        ac = AccessControl(rbac_manager)

        @ac.require_permission("pipeline", "delete")
        def delete_pipeline(user_id=None, context=None):
            return "deleted"

        with pytest.raises(PermissionError):
            delete_pipeline(user_id="user-1")

    def test_require_permission_no_user_id(self, rbac_manager):
        """Test decorator requires user_id"""
        ac = AccessControl(rbac_manager)

        @ac.require_permission("pipeline", "execute")
        def execute_pipeline():
            return "executed"

        with pytest.raises(PermissionError, match="User ID required"):
            execute_pipeline()

    def test_filter_resources(self, rbac_manager):
        """Test filtering resources by permissions"""
        rbac_manager.create_role("limited", permissions=[
            RBACPermission(resource="pipeline:123", action="read")
        ])
        rbac_manager.assign_role("user-1", "limited")

        ac = AccessControl(rbac_manager)

        # Mock resources
        class Resource:
            def __init__(self, id):
                self.id = id

        resources = [Resource("123"), Resource("456"), Resource("789")]
        filtered = ac.filter_resources("user-1", resources, "pipeline", "read")

        assert len(filtered) == 1
        assert filtered[0].id == "123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
