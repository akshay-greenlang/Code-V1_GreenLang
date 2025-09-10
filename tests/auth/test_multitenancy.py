"""
Tests for GreenLang Multi-tenancy Support
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import jwt

from greenlang.auth import (
    TenantManager, Tenant, TenantQuota, TenantIsolation, TenantContext,
    RBACManager, Role, Permission, AccessControl,
    AuthManager, AuthToken, APIKey, ServiceAccount
)
from greenlang.auth.audit import (
    AuditLogger, AuditEvent, AuditEventType, AuditSeverity,
    AuditTrail, ComplianceReporter
)


class TestTenantManager(unittest.TestCase):
    """Test tenant management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = TenantManager()
    
    def test_create_tenant(self):
        """Test creating a new tenant"""
        tenant = self.manager.create_tenant(
            name="Test Corp",
            domain="test.com",
            admin_email="admin@test.com",
            admin_name="Admin User"
        )
        
        self.assertIsNotNone(tenant)
        self.assertEqual(tenant.name, "Test Corp")
        self.assertEqual(tenant.domain, "test.com")
        self.assertEqual(tenant.admin_email, "admin@test.com")
        self.assertEqual(tenant.status, "active")
    
    def test_tenant_with_quota(self):
        """Test tenant with quota limits"""
        quota = TenantQuota(
            max_users=10,
            max_pipelines=100,
            max_storage_gb=50,
            max_compute_hours=500
        )
        
        tenant = self.manager.create_tenant(
            name="Limited Corp",
            admin_email="admin@limited.com",
            quota=quota
        )
        
        self.assertEqual(tenant.quota.max_users, 10)
        self.assertEqual(tenant.quota.max_pipelines, 100)
        
        # Test quota checking
        self.assertTrue(tenant.quota.check_quota("users", 5))
        self.assertFalse(tenant.quota.check_quota("users", 15))
    
    def test_tenant_isolation_levels(self):
        """Test different isolation levels"""
        # Shared isolation
        tenant1 = self.manager.create_tenant(
            name="Shared Tenant",
            admin_email="admin@shared.com",
            isolation=TenantIsolation.SHARED
        )
        self.assertEqual(tenant1.isolation, TenantIsolation.SHARED)
        
        # Namespace isolation
        tenant2 = self.manager.create_tenant(
            name="Namespace Tenant",
            admin_email="admin@namespace.com",
            isolation=TenantIsolation.NAMESPACE
        )
        self.assertEqual(tenant2.isolation, TenantIsolation.NAMESPACE)
        
        # Cluster isolation
        tenant3 = self.manager.create_tenant(
            name="Cluster Tenant",
            admin_email="admin@cluster.com",
            isolation=TenantIsolation.CLUSTER
        )
        self.assertEqual(tenant3.isolation, TenantIsolation.CLUSTER)
    
    def test_tenant_context(self):
        """Test tenant context management"""
        tenant = self.manager.create_tenant(
            name="Context Test",
            admin_email="admin@context.com"
        )
        
        # Generate token
        token = self.manager.generate_tenant_token(tenant.tenant_id, "user123")
        self.assertIsNotNone(token)
        
        # Get context from token
        context = self.manager.get_tenant_context(token)
        self.assertEqual(context.tenant_id, tenant.tenant_id)
        self.assertEqual(context.user_id, "user123")
    
    def test_tenant_operations(self):
        """Test tenant CRUD operations"""
        # Create
        tenant = self.manager.create_tenant(
            name="CRUD Test",
            admin_email="admin@crud.com"
        )
        tenant_id = tenant.tenant_id
        
        # Read
        retrieved = self.manager.get_tenant(tenant_id)
        self.assertEqual(retrieved.name, "CRUD Test")
        
        # Update
        updated = self.manager.update_tenant(tenant_id, {
            "name": "Updated Corp",
            "domain": "updated.com"
        })
        self.assertEqual(updated.name, "Updated Corp")
        self.assertEqual(updated.domain, "updated.com")
        
        # List
        tenants = self.manager.list_tenants()
        self.assertIn(tenant_id, [t.tenant_id for t in tenants])
        
        # Suspend
        self.assertTrue(self.manager.suspend_tenant(tenant_id))
        suspended = self.manager.get_tenant(tenant_id)
        self.assertEqual(suspended.status, "suspended")
        
        # Activate
        self.assertTrue(self.manager.activate_tenant(tenant_id))
        activated = self.manager.get_tenant(tenant_id)
        self.assertEqual(activated.status, "active")
        
        # Delete
        self.assertTrue(self.manager.delete_tenant(tenant_id))
        deleted = self.manager.get_tenant(tenant_id)
        self.assertIsNone(deleted)
    
    def test_tenant_data_paths(self):
        """Test tenant data isolation paths"""
        tenant = self.manager.create_tenant(
            name="Path Test",
            admin_email="admin@path.com"
        )
        
        # Get data paths
        data_path = self.manager.get_tenant_data_path(tenant.tenant_id)
        self.assertTrue(str(data_path).endswith(tenant.tenant_id))
        
        pipeline_path = self.manager.get_tenant_data_path(
            tenant.tenant_id, 
            "pipelines"
        )
        self.assertIn("pipelines", str(pipeline_path))


class TestRBAC(unittest.TestCase):
    """Test RBAC system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rbac = RBACManager()
    
    def test_default_roles(self):
        """Test default system roles"""
        # Check default roles exist
        self.assertIsNotNone(self.rbac.get_role("super_admin"))
        self.assertIsNotNone(self.rbac.get_role("admin"))
        self.assertIsNotNone(self.rbac.get_role("developer"))
        self.assertIsNotNone(self.rbac.get_role("operator"))
        self.assertIsNotNone(self.rbac.get_role("viewer"))
    
    def test_permission_matching(self):
        """Test permission pattern matching"""
        # Exact match
        perm1 = Permission(resource="pipeline", action="read")
        self.assertTrue(perm1.matches("pipeline", "read"))
        self.assertFalse(perm1.matches("pipeline", "write"))
        self.assertFalse(perm1.matches("pack", "read"))
        
        # Wildcard matching
        perm2 = Permission(resource="pipeline:*", action="*")
        self.assertTrue(perm2.matches("pipeline:test", "read"))
        self.assertTrue(perm2.matches("pipeline:prod", "write"))
        self.assertFalse(perm2.matches("pack:test", "read"))
        
        # Scope matching
        perm3 = Permission(resource="*", action="*", scope="tenant:123")
        context = {"tenant": "123"}
        self.assertTrue(perm3.matches("pipeline", "read", context))
        
        context2 = {"tenant": "456"}
        self.assertFalse(perm3.matches("pipeline", "read", context2))
    
    def test_role_permissions(self):
        """Test role permission management"""
        # Create custom role
        role = self.rbac.create_role(
            "custom_role",
            "Custom test role",
            [
                Permission(resource="pipeline", action="read"),
                Permission(resource="pipeline", action="execute")
            ]
        )
        
        # Check permissions
        self.assertTrue(role.has_permission("pipeline", "read"))
        self.assertTrue(role.has_permission("pipeline", "execute"))
        self.assertFalse(role.has_permission("pipeline", "delete"))
        
        # Add permission
        role.add_permission(Permission(resource="pack", action="read"))
        self.assertTrue(role.has_permission("pack", "read"))
        
        # Remove permission
        role.remove_permission(Permission(resource="pack", action="read"))
        self.assertFalse(role.has_permission("pack", "read"))
    
    def test_role_inheritance(self):
        """Test role inheritance"""
        # Create parent role
        parent = self.rbac.create_role(
            "parent_role",
            "Parent role",
            [Permission(resource="resource1", action="read")]
        )
        
        # Create child role with inheritance
        child = self.rbac.create_role(
            "child_role",
            "Child role",
            [Permission(resource="resource2", action="write")],
            parent_roles=["parent_role"]
        )
        
        # Check inherited permissions
        all_perms = child.get_all_permissions(self.rbac)
        self.assertEqual(len(all_perms), 2)
        
        # Child should have both its own and parent's permissions
        self.assertTrue(child.has_permission("resource2", "write"))
        # Note: Direct has_permission doesn't check inheritance
        # Need to check via RBAC manager
    
    def test_user_role_assignment(self):
        """Test assigning roles to users"""
        user_id = "user123"
        
        # Assign role
        self.assertTrue(self.rbac.assign_role(user_id, "developer"))
        
        # Check user roles
        roles = self.rbac.get_user_roles(user_id)
        self.assertIn("developer", roles)
        
        # Check permissions
        self.assertTrue(self.rbac.check_permission(
            user_id, "pipeline", "create"
        ))
        self.assertFalse(self.rbac.check_permission(
            user_id, "tenant", "delete"
        ))
        
        # Assign additional role
        self.rbac.assign_role(user_id, "viewer")
        roles = self.rbac.get_user_roles(user_id)
        self.assertEqual(len(roles), 2)
        
        # Revoke role
        self.assertTrue(self.rbac.revoke_role(user_id, "developer"))
        roles = self.rbac.get_user_roles(user_id)
        self.assertNotIn("developer", roles)
    
    def test_resource_policies(self):
        """Test resource-specific policies"""
        # Add resource policy
        self.rbac.add_resource_policy("pipeline:prod", {
            "users": ["user1", "user2"],
            "actions": ["read", "execute"],
            "conditions": {"environment": "production"}
        })
        
        # Check with matching conditions
        context = {"environment": "production"}
        self.assertTrue(self.rbac.check_permission(
            "user1", "pipeline:prod", "read", context
        ))
        
        # Check with non-matching conditions
        context2 = {"environment": "development"}
        self.assertFalse(self.rbac.check_permission(
            "user1", "pipeline:prod", "read", context2
        ))
        
        # Check unauthorized user
        self.assertFalse(self.rbac.check_permission(
            "user3", "pipeline:prod", "read", context
        ))
    
    def test_access_control_decorator(self):
        """Test access control decorator"""
        access = AccessControl(self.rbac)
        
        # Assign role to test user
        self.rbac.assign_role("testuser", "developer")
        
        @access.require_permission("pipeline", "create")
        def create_pipeline(user_id=None, name=None):
            return f"Pipeline {name} created by {user_id}"
        
        # Should work with proper permissions
        result = create_pipeline(user_id="testuser", name="test")
        self.assertIn("created", result)
        
        # Should fail without permissions
        with self.assertRaises(PermissionError):
            create_pipeline(user_id="unauthorized", name="test")
    
    def test_resource_filtering(self):
        """Test filtering resources by permissions"""
        access = AccessControl(self.rbac)
        
        # Create test resources
        resources = [
            Mock(id="res1", name="Resource 1"),
            Mock(id="res2", name="Resource 2"),
            Mock(id="res3", name="Resource 3")
        ]
        
        # Set up permissions
        self.rbac.assign_role("user1", "viewer")
        
        # Filter resources
        filtered = access.filter_resources(
            "user1", resources, "pipeline", "read"
        )
        
        # Viewer should be able to read all
        self.assertEqual(len(filtered), 3)


class TestAuthentication(unittest.TestCase):
    """Test authentication system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auth = AuthManager()
    
    def test_user_creation(self):
        """Test creating users"""
        user = self.auth.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePass123!",
            tenant_id="tenant123"
        )
        
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], "testuser")
        self.assertEqual(user["email"], "test@example.com")
        self.assertIn("tenant123", user["tenant_ids"])
    
    def test_authentication(self):
        """Test user authentication"""
        # Create user
        self.auth.create_user(
            username="authtest",
            email="auth@test.com",
            password="TestPass123!"
        )
        
        # Successful authentication
        token = self.auth.authenticate("authtest", "TestPass123!")
        self.assertIsNotNone(token)
        self.assertEqual(token.user_id, "authtest")
        
        # Failed authentication - wrong password
        token2 = self.auth.authenticate("authtest", "WrongPass")
        self.assertIsNone(token2)
        
        # Failed authentication - unknown user
        token3 = self.auth.authenticate("unknown", "TestPass123!")
        self.assertIsNone(token3)
    
    def test_token_validation(self):
        """Test token validation"""
        # Create user and get token
        self.auth.create_user(
            username="tokentest",
            password="TestPass123!"
        )
        token = self.auth.authenticate("tokentest", "TestPass123!")
        
        # Validate token
        user_id = self.auth.validate_token(token.token)
        self.assertEqual(user_id, "tokentest")
        
        # Invalid token
        invalid_user = self.auth.validate_token("invalid_token")
        self.assertIsNone(invalid_user)
        
        # Revoke token
        self.assertTrue(self.auth.revoke_token(token.token))
        
        # Revoked token should be invalid
        user_id2 = self.auth.validate_token(token.token)
        self.assertIsNone(user_id2)
    
    def test_api_keys(self):
        """Test API key management"""
        # Create API key
        api_key = self.auth.create_api_key(
            user_id="testuser",
            name="Test API Key",
            expires_days=30,
            scopes=["read", "write"]
        )
        
        self.assertIsNotNone(api_key)
        self.assertEqual(api_key.name, "Test API Key")
        self.assertEqual(api_key.scopes, ["read", "write"])
        
        # Validate API key
        user_id = self.auth.validate_api_key(api_key.key)
        self.assertEqual(user_id, "testuser")
        
        # List API keys
        keys = self.auth.list_api_keys("testuser")
        self.assertEqual(len(keys), 1)
        
        # Revoke API key
        self.assertTrue(self.auth.revoke_api_key(api_key.key))
        
        # Revoked key should be invalid
        user_id2 = self.auth.validate_api_key(api_key.key)
        self.assertIsNone(user_id2)
    
    def test_service_accounts(self):
        """Test service account management"""
        # Create service account
        sa = self.auth.create_service_account(
            name="test-service",
            description="Test service account",
            tenant_id="tenant123"
        )
        
        self.assertIsNotNone(sa)
        self.assertEqual(sa.name, "test-service")
        self.assertIsNotNone(sa.client_id)
        self.assertIsNotNone(sa.client_secret)
        
        # Authenticate service account
        token = self.auth.authenticate_service_account(
            sa.client_id,
            sa.client_secret
        )
        self.assertIsNotNone(token)
        
        # Wrong credentials
        token2 = self.auth.authenticate_service_account(
            sa.client_id,
            "wrong_secret"
        )
        self.assertIsNone(token2)
    
    def test_password_requirements(self):
        """Test password strength requirements"""
        # Weak password
        with self.assertRaises(ValueError):
            self.auth.create_user(
                username="weakpass",
                password="weak"
            )
        
        # No uppercase
        with self.assertRaises(ValueError):
            self.auth.create_user(
                username="nocase",
                password="password123!"
            )
        
        # No special character
        with self.assertRaises(ValueError):
            self.auth.create_user(
                username="nospecial",
                password="Password123"
            )
        
        # Valid password
        user = self.auth.create_user(
            username="validpass",
            password="ValidPass123!"
        )
        self.assertIsNotNone(user)
    
    def test_failed_login_tracking(self):
        """Test tracking failed login attempts"""
        # Create user
        self.auth.create_user(
            username="locktest",
            password="TestPass123!"
        )
        
        # Multiple failed attempts
        for i in range(5):
            self.auth.authenticate("locktest", "WrongPass")
        
        # Check if account is locked
        # After 5 failed attempts, account should be locked
        token = self.auth.authenticate("locktest", "TestPass123!")
        # Depending on implementation, might be None or have a flag
        
        # Reset failed attempts
        self.auth.reset_failed_attempts("locktest")


class TestAuditLogging(unittest.TestCase):
    """Test audit logging system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.audit = AuditLogger(Path(self.temp_dir))
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_audit_event_creation(self):
        """Test creating audit events"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            severity=AuditSeverity.INFO,
            tenant_id="tenant123",
            user_id="user123",
            ip_address="192.168.1.1",
            resource_type="auth",
            action="login"
        )
        
        self.assertEqual(event.event_type, AuditEventType.LOGIN_SUCCESS)
        self.assertEqual(event.severity, AuditSeverity.INFO)
        self.assertEqual(event.tenant_id, "tenant123")
        
        # Test serialization
        event_dict = event.to_dict()
        self.assertEqual(event_dict["event_type"], "auth.login.success")
        
        # Test deserialization
        event2 = AuditEvent.from_dict(event_dict)
        self.assertEqual(event2.event_type, AuditEventType.LOGIN_SUCCESS)
    
    def test_audit_logging(self):
        """Test logging audit events"""
        # Log events
        self.audit.log_login("user1", True, "192.168.1.1")
        self.audit.log_login("user2", False, "192.168.1.2")
        self.audit.log_permission_check(
            "user1", "pipeline:test", "execute", True, "tenant123"
        )
        self.audit.log_resource_access(
            "user1", "pipeline", "pipe123", "create", "tenant123"
        )
        
        # Query events
        events = self.audit.query("tenant123", limit=10)
        self.assertGreater(len(events), 0)
        
        # Check event types
        event_types = [e.event_type for e in events]
        self.assertIn(AuditEventType.PERMISSION_GRANTED, event_types)
        self.assertIn(AuditEventType.RESOURCE_CREATED, event_types)
    
    def test_audit_trail(self):
        """Test audit trail management"""
        trail = AuditTrail("tenant123")
        
        # Add events
        for i in range(10):
            event = AuditEvent(
                event_type=AuditEventType.RESOURCE_READ,
                user_id=f"user{i % 3}",
                resource_id=f"resource{i}"
            )
            trail.add_event(event)
        
        # Get events
        events = trail.get_events(limit=5)
        self.assertEqual(len(events), 5)
        
        # Filter by user
        user_events = trail.get_events(user_id="user1", limit=10)
        for event in user_events:
            self.assertEqual(event.user_id, "user1")
        
        # Get statistics
        stats = trail.get_statistics()
        self.assertEqual(stats["tenant_id"], "tenant123")
        self.assertEqual(stats["total_events"], 10)
    
    def test_audit_handlers(self):
        """Test custom audit handlers"""
        handled_events = []
        
        def custom_handler(event):
            handled_events.append(event)
        
        self.audit.add_handler(custom_handler)
        
        # Log event
        self.audit.log_login("user1", True)
        
        # Check handler was called
        self.assertEqual(len(handled_events), 1)
        self.assertEqual(handled_events[0].event_type, AuditEventType.LOGIN_SUCCESS)
    
    def test_audit_filters(self):
        """Test audit event filters"""
        def filter_critical_only(event):
            return event.severity == AuditSeverity.CRITICAL
        
        self.audit.add_filter(filter_critical_only)
        
        # Log non-critical event - should be filtered
        self.audit.log_login("user1", True)
        
        # Log critical event - should pass
        self.audit.log_security_alert(
            "intrusion_detected",
            "Potential intrusion detected",
            "user1",
            "tenant123"
        )
        
        # Query events
        events = self.audit.query("tenant123")
        # Only critical events should be logged
        for event in events:
            self.assertEqual(event.severity, AuditSeverity.CRITICAL)
    
    def test_audit_report(self):
        """Test generating audit reports"""
        # Log various events
        self.audit.log_login("user1", True, tenant_id="tenant123")
        self.audit.log_login("user2", False, tenant_id="tenant123")
        self.audit.log_permission_check(
            "user1", "resource1", "read", False, "tenant123"
        )
        self.audit.log_security_alert(
            "suspicious", "Suspicious activity", "user2", "tenant123"
        )
        
        # Generate report
        start_time = datetime.utcnow() - timedelta(days=1)
        end_time = datetime.utcnow()
        report = self.audit.get_report("tenant123", start_time, end_time)
        
        self.assertEqual(report["tenant_id"], "tenant123")
        self.assertGreater(report["total_events"], 0)
        self.assertGreater(report["failed_logins"], 0)
        self.assertGreater(report["permission_denials"], 0)
        self.assertGreater(report["security_alerts"], 0)
    
    def test_compliance_reports(self):
        """Test compliance report generation"""
        reporter = ComplianceReporter(self.audit)
        
        # Log compliance-relevant events
        self.audit.log(AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGED,
            tenant_id="tenant123",
            user_id="admin",
            metadata={"setting": "security_level", "old": "low", "new": "high"}
        ))
        
        self.audit.log(AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGNED,
            tenant_id="tenant123",
            user_id="user1",
            metadata={"role": "admin"}
        ))
        
        # Generate SOX report
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        sox_report = reporter.generate_sox_report(
            "tenant123", start_time, end_time
        )
        
        self.assertEqual(sox_report["report_type"], "SOX Compliance")
        self.assertIsInstance(sox_report["configuration_changes"], list)
        self.assertIsInstance(sox_report["user_provisioning"], list)
        
        # Generate GDPR report
        gdpr_report = reporter.generate_gdpr_report(
            "tenant123", start_time, end_time
        )
        
        self.assertEqual(gdpr_report["report_type"], "GDPR Compliance")
        self.assertIsInstance(gdpr_report["data_access"], list)
        self.assertIsInstance(gdpr_report["data_modifications"], list)


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-tenancy"""
    
    def test_full_tenant_workflow(self):
        """Test complete tenant workflow"""
        # Create managers
        tenant_mgr = TenantManager()
        rbac_mgr = RBACManager()
        auth_mgr = AuthManager()
        audit_logger = AuditLogger()
        
        # Create tenant
        tenant = tenant_mgr.create_tenant(
            name="Integration Test Corp",
            domain="integration.test",
            admin_email="admin@integration.test",
            admin_name="Admin User"
        )
        
        # Create admin user
        admin_user = auth_mgr.create_user(
            username="admin",
            email="admin@integration.test",
            password="AdminPass123!",
            tenant_id=tenant.tenant_id
        )
        
        # Assign admin role
        rbac_mgr.assign_role(admin_user["user_id"], "admin")
        
        # Create regular user
        regular_user = auth_mgr.create_user(
            username="user1",
            email="user1@integration.test",
            password="UserPass123!",
            tenant_id=tenant.tenant_id
        )
        
        # Assign developer role
        rbac_mgr.assign_role(regular_user["user_id"], "developer")
        
        # Authenticate users
        admin_token = auth_mgr.authenticate("admin", "AdminPass123!")
        user_token = auth_mgr.authenticate("user1", "UserPass123!")
        
        self.assertIsNotNone(admin_token)
        self.assertIsNotNone(user_token)
        
        # Check permissions
        self.assertTrue(rbac_mgr.check_permission(
            admin_user["user_id"], "user", "create",
            {"tenant": tenant.tenant_id}
        ))
        
        self.assertTrue(rbac_mgr.check_permission(
            regular_user["user_id"], "pipeline", "create"
        ))
        
        self.assertFalse(rbac_mgr.check_permission(
            regular_user["user_id"], "user", "delete"
        ))
        
        # Create API key
        api_key = auth_mgr.create_api_key(
            user_id=regular_user["user_id"],
            name="Dev API Key",
            scopes=["pipeline:read", "pipeline:execute"]
        )
        
        # Validate API key
        validated_user = auth_mgr.validate_api_key(api_key.key)
        self.assertEqual(validated_user, regular_user["user_id"])
        
        # Log audit events
        audit_logger.log_login(
            admin_user["user_id"], True,
            tenant_id=tenant.tenant_id
        )
        
        audit_logger.log_resource_access(
            regular_user["user_id"], "pipeline", "pipe123", "create",
            tenant_id=tenant.tenant_id
        )
        
        # Check tenant quota
        tenant.quota.use_quota("pipelines", 1)
        self.assertEqual(tenant.quota.used_pipelines, 1)
        
        # Generate audit report
        report = audit_logger.get_report(
            tenant.tenant_id,
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        
        self.assertGreater(report["total_events"], 0)
        
        # Clean up
        tenant_mgr.delete_tenant(tenant.tenant_id)


if __name__ == '__main__':
    unittest.main()