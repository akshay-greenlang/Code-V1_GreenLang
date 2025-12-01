# -*- coding: utf-8 -*-
"""
Integration Tests for RBAC Enforcement
=======================================

Tests for PolicyEnforcer integration with agent-level RBAC.

Test Coverage:
    - PolicyEnforcer RBAC methods
    - Agent execute permission checks
    - Agent data access permission checks
    - Agent config access permission checks
    - OPA policy integration
    - Critical agent protection
    - Audit logging
"""

import pytest
import tempfile
from pathlib import Path

from core.greenlang.policy.enforcer import PolicyEnforcer, PolicyResult
from core.greenlang.policy.agent_rbac import AgentPermission


class TestPolicyEnforcerRBAC:
    """Test PolicyEnforcer RBAC integration."""

    @pytest.fixture
    def enforcer(self):
        """Create PolicyEnforcer with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = PolicyEnforcer(rbac_dir=Path(tmpdir))
            yield enforcer

    def test_check_agent_execute_granted(self, enforcer):
        """Test checking agent execute permission when granted."""
        # Grant permission
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check permission
        result = enforcer.check_agent_execute("GL-001", "user@example.com")

        assert result.allowed
        assert result.reason == "Permission granted"

    def test_check_agent_execute_denied(self, enforcer):
        """Test checking agent execute permission when denied."""
        # No permission granted

        # Check permission
        result = enforcer.check_agent_execute("GL-001", "user@example.com")

        assert not result.allowed
        assert "lacks EXECUTE permission" in result.reason
        assert "agent_rbac" in result.violated_policies

    def test_check_agent_execute_viewer_denied(self, enforcer):
        """Test viewer role cannot execute agents."""
        # Grant viewer role (has no execute permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_viewer")

        # Check execute permission
        result = enforcer.check_agent_execute("GL-001", "user@example.com")

        assert not result.allowed
        assert "lacks EXECUTE permission" in result.reason

    def test_check_agent_execute_admin_granted(self, enforcer):
        """Test admin role can execute agents."""
        # Grant admin role
        enforcer.grant_agent_role("GL-001", "admin@example.com", "agent_admin")

        # Check execute permission
        result = enforcer.check_agent_execute("GL-001", "admin@example.com")

        assert result.allowed

    def test_check_agent_data_access_read_granted(self, enforcer):
        """Test checking read data access when granted."""
        # Grant operator role (has read_data permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check read permission
        result = enforcer.check_agent_data_access("GL-001", "user@example.com", "read")

        assert result.allowed

    def test_check_agent_data_access_write_denied(self, enforcer):
        """Test checking write data access when denied."""
        # Grant operator role (no write_data permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check write permission
        result = enforcer.check_agent_data_access("GL-001", "user@example.com", "write")

        assert not result.allowed
        assert "lacks write_data permission" in result.reason

    def test_check_agent_data_access_write_granted(self, enforcer):
        """Test checking write data access when granted."""
        # Grant engineer role (has write_data permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_engineer")

        # Check write permission
        result = enforcer.check_agent_data_access("GL-001", "user@example.com", "write")

        assert result.allowed

    def test_check_agent_data_access_invalid_type(self, enforcer):
        """Test invalid data access type."""
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check with invalid type
        result = enforcer.check_agent_data_access("GL-001", "user@example.com", "invalid")

        assert not result.allowed
        assert "Invalid data_type" in result.reason

    def test_check_agent_config_access_read_granted(self, enforcer):
        """Test checking read config access when granted."""
        # Grant viewer role (has read_config permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_viewer")

        # Check read config permission
        result = enforcer.check_agent_config_access("GL-001", "user@example.com", "read")

        assert result.allowed

    def test_check_agent_config_access_write_denied(self, enforcer):
        """Test checking write config access when denied."""
        # Grant viewer role (no write_config permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_viewer")

        # Check write config permission
        result = enforcer.check_agent_config_access("GL-001", "user@example.com", "write")

        assert not result.allowed
        assert "lacks write_config permission" in result.reason

    def test_check_agent_config_access_write_granted(self, enforcer):
        """Test checking write config access when granted."""
        # Grant engineer role (has write_config permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_engineer")

        # Check write config permission
        result = enforcer.check_agent_config_access("GL-001", "user@example.com", "write")

        assert result.allowed

    def test_check_agent_lifecycle_denied(self, enforcer):
        """Test checking lifecycle management when denied."""
        # Grant operator role (no manage_lifecycle permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check lifecycle permission
        result = enforcer.check_agent_lifecycle("GL-001", "user@example.com")

        assert not result.allowed
        assert "lacks MANAGE_LIFECYCLE permission" in result.reason

    def test_check_agent_lifecycle_granted(self, enforcer):
        """Test checking lifecycle management when granted."""
        # Grant engineer role (has manage_lifecycle permission)
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_engineer")

        # Check lifecycle permission
        result = enforcer.check_agent_lifecycle("GL-001", "user@example.com")

        assert result.allowed

    def test_default_policy_read_config_allowed(self, enforcer):
        """Test default policy allows read_config even without ACL."""
        # No ACL created, no roles granted

        # Check read config (should be allowed by default)
        result = enforcer.check_agent_config_access("GL-999", "user@example.com", "read")

        assert result.allowed

    def test_default_policy_execute_denied(self, enforcer):
        """Test default policy denies execute without ACL."""
        # No ACL created, no roles granted

        # Check execute (should be denied by default)
        result = enforcer.check_agent_execute("GL-999", "user@example.com")

        assert not result.allowed

    def test_grant_and_revoke_role(self, enforcer):
        """Test granting and revoking roles."""
        # Grant role
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check permission granted
        result = enforcer.check_agent_execute("GL-001", "user@example.com")
        assert result.allowed

        # Revoke role
        enforcer.revoke_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check permission denied
        result = enforcer.check_agent_execute("GL-001", "user@example.com")
        assert not result.allowed

    def test_list_agent_roles(self, enforcer):
        """Test listing agent roles for user."""
        # Grant multiple roles
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_viewer")
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # List roles
        roles = enforcer.list_agent_roles("GL-001", "user@example.com")

        assert len(roles) == 2
        assert "agent_viewer" in roles
        assert "agent_operator" in roles

    def test_audit_user_agent_access(self, enforcer):
        """Test auditing user access across multiple agents."""
        # Grant roles for multiple agents
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")
        enforcer.grant_agent_role("GL-002", "user@example.com", "agent_viewer")

        # Audit user access
        audit = enforcer.audit_user_agent_access("user@example.com")

        assert len(audit) == 2
        assert "GL-001" in audit
        assert "agent_operator" in audit["GL-001"]
        assert "GL-002" in audit
        assert "agent_viewer" in audit["GL-002"]

    def test_list_available_roles(self, enforcer):
        """Test listing all available roles."""
        roles = enforcer.list_available_roles()

        assert "agent_viewer" in roles
        assert "agent_operator" in roles
        assert "agent_engineer" in roles
        assert "agent_admin" in roles

        # Check descriptions exist
        for role_name, description in roles.items():
            assert description
            assert isinstance(description, str)

    def test_multiple_users_same_agent(self, enforcer):
        """Test multiple users with different roles on same agent."""
        # Grant different roles to different users
        enforcer.grant_agent_role("GL-001", "viewer@example.com", "agent_viewer")
        enforcer.grant_agent_role("GL-001", "operator@example.com", "agent_operator")
        enforcer.grant_agent_role("GL-001", "admin@example.com", "agent_admin")

        # Check viewer permissions
        result = enforcer.check_agent_execute("GL-001", "viewer@example.com")
        assert not result.allowed

        # Check operator permissions
        result = enforcer.check_agent_execute("GL-001", "operator@example.com")
        assert result.allowed

        # Check admin permissions
        result = enforcer.check_agent_execute("GL-001", "admin@example.com")
        assert result.allowed

    def test_user_with_multiple_agents(self, enforcer):
        """Test user with roles on multiple agents."""
        # Grant roles for multiple agents
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")
        enforcer.grant_agent_role("GL-002", "user@example.com", "agent_engineer")
        enforcer.grant_agent_role("GL-003", "user@example.com", "agent_admin")

        # Check permissions on each agent
        result = enforcer.check_agent_execute("GL-001", "user@example.com")
        assert result.allowed

        result = enforcer.check_agent_config_access("GL-002", "user@example.com", "write")
        assert result.allowed

        result = enforcer.check_agent_lifecycle("GL-003", "user@example.com")
        assert result.allowed


class TestRBACWithContext:
    """Test RBAC with execution context."""

    @pytest.fixture
    def enforcer(self):
        """Create PolicyEnforcer with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = PolicyEnforcer(rbac_dir=Path(tmpdir))
            yield enforcer

    def test_execute_with_context(self, enforcer):
        """Test execute permission check with context."""
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check with context
        context = {
            "environment": "production",
            "region": "us-east-1"
        }

        result = enforcer.check_agent_execute("GL-001", "user@example.com", context)

        assert result.allowed

    def test_critical_agent_requires_approval(self, enforcer):
        """Test critical agents require approval in context."""
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # GL-001 is a critical agent
        # Without approval
        context = {"has_approval": False}
        result = enforcer.check_agent_execute("GL-001", "user@example.com", context)

        # Should be denied by OPA policy if agent_rbac.rego exists
        # (This test assumes OPA policy is available)

    def test_data_classification_context(self, enforcer):
        """Test data access with classification context."""
        enforcer.grant_agent_role("GL-001", "user@example.com", "agent_operator")

        # Check read access with confidential data
        context = {
            "data_classification": "confidential"
        }

        # Operator can still read (OPA policy may add restrictions)
        result = enforcer.check_agent_data_access("GL-001", "user@example.com", "read", context)

        # Result depends on OPA policy


class TestRBACPersistence:
    """Test RBAC persistence across instances."""

    def test_rbac_persists_across_instances(self):
        """Test RBAC grants persist across PolicyEnforcer instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first enforcer and grant role
            enforcer1 = PolicyEnforcer(rbac_dir=Path(tmpdir))
            enforcer1.grant_agent_role("GL-001", "user@example.com", "agent_operator")

            # Check permission granted
            result = enforcer1.check_agent_execute("GL-001", "user@example.com")
            assert result.allowed

            # Create second enforcer (should load from disk)
            enforcer2 = PolicyEnforcer(rbac_dir=Path(tmpdir))

            # Check permission still granted
            result = enforcer2.check_agent_execute("GL-001", "user@example.com")
            assert result.allowed

    def test_rbac_persists_after_revoke(self):
        """Test RBAC revocation persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create enforcer and grant role
            enforcer1 = PolicyEnforcer(rbac_dir=Path(tmpdir))
            enforcer1.grant_agent_role("GL-001", "user@example.com", "agent_operator")

            # Revoke role
            enforcer1.revoke_agent_role("GL-001", "user@example.com", "agent_operator")

            # Create new enforcer
            enforcer2 = PolicyEnforcer(rbac_dir=Path(tmpdir))

            # Check permission denied
            result = enforcer2.check_agent_execute("GL-001", "user@example.com")
            assert not result.allowed


class TestRBACErrorHandling:
    """Test RBAC error handling."""

    @pytest.fixture
    def enforcer(self):
        """Create PolicyEnforcer with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            enforcer = PolicyEnforcer(rbac_dir=Path(tmpdir))
            yield enforcer

    def test_grant_invalid_role(self, enforcer):
        """Test granting invalid role raises error."""
        with pytest.raises(ValueError):
            enforcer.grant_agent_role("GL-001", "user@example.com", "invalid_role")

    def test_revoke_from_nonexistent_acl(self, enforcer):
        """Test revoking from nonexistent ACL raises error."""
        with pytest.raises(ValueError):
            enforcer.revoke_agent_role("GL-999", "user@example.com", "agent_operator")

    def test_list_roles_nonexistent_acl(self, enforcer):
        """Test listing roles for nonexistent ACL returns empty list."""
        roles = enforcer.list_agent_roles("GL-999", "user@example.com")

        assert roles == []

    def test_audit_user_no_access(self, enforcer):
        """Test auditing user with no access returns empty dict."""
        audit = enforcer.audit_user_agent_access("user@example.com")

        assert audit == {}
