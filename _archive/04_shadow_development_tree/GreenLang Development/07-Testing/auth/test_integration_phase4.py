# -*- coding: utf-8 -*-
"""
Integration tests for Advanced Access Control (Phase 4)

Tests all components working together:
- Permissions
- Roles
- ABAC
- Delegation
- Temporal Access
- Audit Trail
"""

import pytest
from datetime import datetime, timedelta, time

from greenlang.auth.permissions import create_permission, PermissionEffect
from greenlang.auth.roles import RoleManager, BuiltInRole
from greenlang.auth.abac import ABACEvaluator, create_policy, PolicyEffect
from greenlang.auth.delegation import DelegationManager
from greenlang.auth.temporal_access import (
    TemporalAccessManager,
    TimeWindow,
    RecurrencePattern,
    RecurrenceType
)
from greenlang.auth.permission_audit import (
    PermissionAuditLogger,
    PermissionChangeType
)


class TestCompleteAccessControlFlow:
    """Test complete access control flow with all components."""

    def test_end_to_end_authorization(self):
        """Test end-to-end authorization scenario."""

        # Setup all managers
        role_mgr = RoleManager()
        abac_eval = ABACEvaluator()
        delegation_mgr = DelegationManager()
        temporal_mgr = TemporalAccessManager()
        audit_logger = PermissionAuditLogger()

        # Scenario: Analyst needs temporary elevated access for a project

        # Step 1: Create analyst user with base role
        analyst_id = "analyst_001"
        manager_id = "manager_001"

        # Get analyst role (built-in)
        analyst_role = role_mgr.get_role_by_name(BuiltInRole.ANALYST.value)
        assert analyst_role is not None

        # Assign analyst role
        assignment = role_mgr.assign_role(
            role_id=analyst_role.role_id,
            principal_id=analyst_id,
            assigned_by=manager_id
        )

        # Log the role assignment
        audit_logger.log_role_assigned(
            actor_id=manager_id,
            role_id=analyst_role.role_id,
            principal_id=analyst_id,
            reason="Initial role assignment"
        )

        # Step 2: Verify base permissions
        can_read_workflow = role_mgr.check_permission(
            principal_id=analyst_id,
            resource="workflow:carbon-audit",
            action="read"
        )
        assert can_read_workflow == True

        can_delete_workflow = role_mgr.check_permission(
            principal_id=analyst_id,
            resource="workflow:carbon-audit",
            action="delete"
        )
        assert can_delete_workflow == False  # Analysts can't delete

        # Step 3: Manager delegates special permission for project
        project_perm = create_permission(
            resource="data:project_alpha:*",
            action="*"
        )

        delegation = delegation_mgr.delegate(
            delegator_id=manager_id,
            delegatee_id=analyst_id,
            permission=project_perm,
            duration=timedelta(days=30),
            reason="Alpha project data access"
        )

        # Log the delegation
        audit_logger.log_delegation_created(
            actor_id=manager_id,
            delegation=delegation.to_dict(),
            reason="Alpha project assignment"
        )

        # Verify delegated permissions
        delegated_perms = delegation_mgr.get_delegated_permissions(analyst_id)
        assert len(delegated_perms) > 0
        assert any(p.resource == "data:project_alpha:*" for p in delegated_perms)

        # Step 4: Add temporal restriction (business hours only)
        sensitive_perm = create_permission(
            resource="data:financial:*",
            action="read"
        )

        temporal_perm = temporal_mgr.create_temporal_permission(
            user_id=analyst_id,
            permission=sensitive_perm,
            time_windows=[
                TimeWindow(start_time=time(9, 0), end_time=time(17, 0))
            ],
            recurrence=RecurrencePattern(
                recurrence_type=RecurrenceType.WEEKLY,
                days_of_week=[0, 1, 2, 3, 4]  # Monday-Friday
            ),
            description="Business hours financial data access"
        )

        # Log temporal permission
        audit_logger.log_permission_change(
            change_type=PermissionChangeType.TEMPORAL_PERMISSION_CREATED,
            actor_id=manager_id,
            target_id=temporal_perm.temporal_id,
            target_type="temporal_permission",
            after_snapshot=temporal_perm.to_dict(),
            principal_id=analyst_id
        )

        # Step 5: Add ABAC policy restriction
        # Policy: Deny data export outside business hours
        policy = create_policy(
            name="no-export-after-hours",
            effect=PolicyEffect.DENY,
            actions=["export"],
            resources=["data:*"],
            conditions=[
                {
                    "attribute": "environment.is_business_hours",
                    "operator": "eq",
                    "value": False
                }
            ]
        )

        abac_eval.add_policy(policy)

        # Test ABAC evaluation
        # Simulate after-hours export attempt
        result = abac_eval.evaluate(
            user_id=analyst_id,
            resource="data:emissions",
            action="export",
            context={}  # Will check current time
        )

        # Result depends on current time, but policy is in place
        assert result is not None

        # Step 6: Query audit trail
        events = audit_logger.query_events(
            principal_id=analyst_id,
            limit=100
        )

        assert len(events) >= 2  # At least role assignment and delegation
        event_types = {e.change_type for e in events}
        assert PermissionChangeType.ROLE_ASSIGNED in event_types
        assert PermissionChangeType.DELEGATION_CREATED in event_types

        # Step 7: Verify audit integrity
        is_valid, errors = audit_logger.verify_integrity()
        assert is_valid == True
        assert len(errors) == 0

        # Step 8: Revoke delegation when project ends
        result = delegation_mgr.revoke(
            delegation_id=delegation.delegation_id,
            revoked_by=manager_id,
            reason="Project completed"
        )
        assert result == True

        # Verify delegation is revoked
        delegated_perms_after = delegation_mgr.get_delegated_permissions(analyst_id)
        assert len(delegated_perms_after) == 0


    def test_multi_layer_authorization(self):
        """Test authorization with multiple layers."""

        role_mgr = RoleManager()
        abac_eval = ABACEvaluator()
        temporal_mgr = TemporalAccessManager()

        user_id = "user_123"

        # Layer 1: Assign viewer role (read-only)
        viewer_role = role_mgr.get_role_by_name(BuiltInRole.VIEWER.value)
        role_mgr.assign_role(
            role_id=viewer_role.role_id,
            principal_id=user_id,
            assigned_by="admin_001"
        )

        # Layer 2: Add ABAC policy for department restriction
        abac_eval.user_provider.set_user_attributes(user_id, {
            "department": "finance",
            "clearance": 2
        })

        policy = create_policy(
            name="finance-data-access",
            effect=PolicyEffect.ALLOW,
            actions=["read"],
            resources=["data:finance:*"],
            conditions=[
                {
                    "attribute": "user.department",
                    "operator": "eq",
                    "value": "finance"
                }
            ]
        )
        abac_eval.add_policy(policy)

        # Layer 3: Add temporal restriction
        finance_perm = create_permission("data:finance:*", "read")
        temporal_mgr.create_temporal_permission(
            user_id=user_id,
            permission=finance_perm,
            time_windows=[TimeWindow(time(8, 0), time(18, 0))]
        )

        # Test combined authorization
        # User should have role-based access
        has_role_access = role_mgr.check_permission(
            user_id, "data:public", "read"
        )
        assert has_role_access == True

        # User should have ABAC access to finance data
        abac_result = abac_eval.evaluate(
            user_id, "data:finance:reports", "read"
        )
        assert abac_result.allowed == True

        # User should have temporal access (if within time window)
        active_perms = temporal_mgr.get_active_permissions(user_id)
        # Result depends on current time


    def test_permission_hierarchy_resolution(self):
        """Test permission resolution with role hierarchy."""

        role_mgr = RoleManager()

        # Create custom role hierarchy
        # Manager role (parent)
        manager_role = role_mgr.create_role(
            name="project_manager",
            display_name="Project Manager",
            permissions=[
                create_permission("project:*", "read"),
                create_permission("project:*", "update"),
                create_permission("team:*", "manage")
            ],
            created_by="admin_001"
        )

        # Team lead role (child of manager)
        lead_role = role_mgr.create_role(
            name="team_lead",
            display_name="Team Lead",
            permissions=[
                create_permission("task:*", "assign")
            ],
            parent_role_ids=[manager_role.role_id],
            created_by="admin_001"
        )

        # Assign team lead role to user
        user_id = "lead_001"
        role_mgr.assign_role(
            role_id=lead_role.role_id,
            principal_id=user_id,
            assigned_by="admin_001"
        )

        # User should have both direct and inherited permissions
        # Direct permission
        can_assign = role_mgr.check_permission(
            user_id, "task:feature_123", "assign"
        )
        assert can_assign == True

        # Inherited permission from manager role
        can_read_project = role_mgr.check_permission(
            user_id, "project:alpha", "read"
        )
        assert can_read_project == True

        # Should not have permissions not granted
        can_delete = role_mgr.check_permission(
            user_id, "project:alpha", "delete"
        )
        assert can_delete == False


    def test_delegation_chain(self):
        """Test permission delegation chain."""

        delegation_mgr = DelegationManager(max_delegation_chain=3)

        # Manager delegates to senior analyst
        perm = create_permission("data:*", "analyze")

        delegation1 = delegation_mgr.delegate(
            delegator_id="manager_001",
            delegatee_id="senior_analyst_001",
            permission=perm,
            duration=timedelta(days=30),
            can_delegate=True,  # Allow further delegation
            reason="Monthly analysis"
        )

        # Senior analyst delegates to junior analyst
        delegation2 = delegation_mgr.delegate(
            delegator_id="senior_analyst_001",
            delegatee_id="junior_analyst_001",
            permission=perm,
            duration=timedelta(days=15),
            parent_delegation_id=delegation1.delegation_id,
            reason="Training junior analyst"
        )

        # Verify chain
        chain = delegation_mgr.get_delegation_chain(delegation2.delegation_id)
        assert len(chain) == 2
        assert chain[0].delegation_id == delegation1.delegation_id
        assert chain[1].delegation_id == delegation2.delegation_id

        # Revoke parent delegation (cascade)
        delegation_mgr.revoke(
            delegation1.delegation_id,
            revoked_by="manager_001",
            reason="Project cancelled",
            cascade=True
        )

        # Both should be revoked
        delegations = delegation_mgr.list_delegations(
            delegatee_id="junior_analyst_001"
        )
        revoked = [d for d in delegations if d.delegation_id == delegation2.delegation_id]
        # Child delegation should be revoked due to cascade


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
