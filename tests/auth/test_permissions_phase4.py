"""
Unit tests for Fine-Grained Permission Model (Phase 4)

Tests permissions.py module functionality.
"""

import pytest
from datetime import datetime

from greenlang.auth.permissions import (
    Permission,
    PermissionAction,
    ResourceType,
    PermissionEffect,
    PermissionCondition,
    PermissionEvaluator,
    PermissionStore,
    create_permission,
    parse_permission_string
)


class TestPermission:
    """Test Permission model."""

    def test_create_permission(self):
        """Test creating a basic permission."""
        perm = create_permission(
            resource="agent:*",
            action="read"
        )

        assert perm.resource == "agent:*"
        assert perm.action == "read"
        assert perm.effect == PermissionEffect.ALLOW
        assert perm.permission_id is not None

    def test_permission_with_conditions(self):
        """Test permission with conditions."""
        perm = create_permission(
            resource="data:*",
            action="export",
            conditions=[
                {
                    "attribute": "user.department",
                    "operator": "eq",
                    "value": "finance"
                }
            ]
        )

        assert len(perm.conditions) == 1
        assert perm.conditions[0].attribute == "user.department"

    def test_permission_matches_resource(self):
        """Test permission resource matching."""
        perm = create_permission(resource="agent:*", action="read")

        # Should match
        assert perm.matches_request("agent:carbon-calculator", "read")
        assert perm.matches_request("agent:fuel-analyzer", "read")

        # Should not match
        assert not perm.matches_request("workflow:audit", "read")
        assert not perm.matches_request("agent:carbon-calculator", "write")

    def test_permission_with_deny_effect(self):
        """Test deny permission."""
        perm = create_permission(
            resource="data:confidential:*",
            action="export",
            effect=PermissionEffect.DENY
        )

        assert perm.effect == PermissionEffect.DENY
        assert perm.matches_request("data:confidential:salaries", "export")

    def test_permission_condition_evaluation(self):
        """Test permission condition evaluation."""
        condition = PermissionCondition(
            attribute="user.department",
            operator="eq",
            value="finance"
        )

        # Should match
        context = {"user": {"department": "finance"}}
        assert condition.evaluate(context) == True

        # Should not match
        context = {"user": {"department": "engineering"}}
        assert condition.evaluate(context) == False


class TestPermissionEvaluator:
    """Test permission evaluation engine."""

    def test_evaluate_single_allow(self):
        """Test evaluating single allow permission."""
        evaluator = PermissionEvaluator()
        permissions = [
            create_permission("agent:*", "read")
        ]

        result = evaluator.evaluate(
            permissions,
            resource="agent:carbon-calculator",
            action="read"
        )

        assert result.allowed == True
        assert len(result.matched_permissions) == 1

    def test_evaluate_explicit_deny_wins(self):
        """Test that explicit deny wins over allow."""
        evaluator = PermissionEvaluator()
        permissions = [
            create_permission("data:*", "read"),  # Allow
            create_permission(
                "data:confidential:*",
                "read",
                effect=PermissionEffect.DENY
            )  # Deny
        ]

        result = evaluator.evaluate(
            permissions,
            resource="data:confidential:salaries",
            action="read"
        )

        assert result.allowed == False
        assert result.denied_by is not None
        assert "denied" in result.reason.lower()

    def test_evaluate_default_deny(self):
        """Test default deny when no permissions match."""
        evaluator = PermissionEvaluator()
        permissions = [
            create_permission("agent:*", "read")
        ]

        result = evaluator.evaluate(
            permissions,
            resource="workflow:audit",  # Different resource
            action="read"
        )

        assert result.allowed == False
        assert "no matching" in result.reason.lower()

    def test_evaluate_with_conditions(self):
        """Test evaluation with conditions."""
        evaluator = PermissionEvaluator()
        permissions = [
            create_permission(
                "data:*",
                "export",
                conditions=[
                    {
                        "attribute": "user.clearance",
                        "operator": "gte",
                        "value": 3
                    }
                ]
            )
        ]

        # Should allow with sufficient clearance
        context = {"user": {"clearance": 4}}
        result = evaluator.evaluate(
            permissions,
            "data:emissions",
            "export",
            context
        )
        assert result.allowed == True

        # Should deny with insufficient clearance
        context = {"user": {"clearance": 2}}
        result = evaluator.evaluate(
            permissions,
            "data:emissions",
            "export",
            context
        )
        assert result.allowed == False

    def test_evaluator_caching(self):
        """Test that evaluator uses cache."""
        evaluator = PermissionEvaluator(cache_ttl_seconds=60)
        permissions = [create_permission("agent:*", "read")]

        # First evaluation (cache miss)
        result1 = evaluator.evaluate(
            permissions,
            "agent:carbon",
            "read",
            use_cache=True
        )

        # Second evaluation (cache hit)
        result2 = evaluator.evaluate(
            permissions,
            "agent:carbon",
            "read",
            use_cache=True
        )

        stats = evaluator.get_stats()
        assert stats['cache_hits'] > 0


class TestPermissionStore:
    """Test permission storage."""

    def test_create_and_get_permission(self):
        """Test creating and retrieving permission."""
        store = PermissionStore(storage_backend="memory")

        perm = create_permission("agent:*", "read")
        store.create(perm)

        retrieved = store.get(perm.permission_id)
        assert retrieved is not None
        assert retrieved.resource == "agent:*"

    def test_list_permissions(self):
        """Test listing permissions with filters."""
        store = PermissionStore(storage_backend="memory")

        perm1 = create_permission("agent:*", "read")
        perm2 = create_permission("workflow:*", "execute")
        perm3 = create_permission("agent:*", "write")

        store.create(perm1)
        store.create(perm2)
        store.create(perm3)

        # Filter by resource
        agent_perms = store.list(resource_pattern="agent:*")
        assert len(agent_perms) == 2

        # Filter by action
        read_perms = store.list(action_pattern="read")
        assert len(read_perms) == 1

    def test_update_permission(self):
        """Test updating permission."""
        store = PermissionStore(storage_backend="memory")

        perm = create_permission("agent:*", "read")
        store.create(perm)

        # Update
        perm.action = "write"
        store.update(perm)

        retrieved = store.get(perm.permission_id)
        assert retrieved.action == "write"

    def test_delete_permission(self):
        """Test deleting permission."""
        store = PermissionStore(storage_backend="memory")

        perm = create_permission("agent:*", "read")
        store.create(perm)

        result = store.delete(perm.permission_id)
        assert result == True

        retrieved = store.get(perm.permission_id)
        assert retrieved is None


class TestPermissionHelpers:
    """Test helper functions."""

    def test_parse_permission_string(self):
        """Test parsing permission from string."""
        # Simple format
        perm = parse_permission_string("agent:*:read")
        assert perm.resource == "agent:*"
        assert perm.action == "read"
        assert perm.effect == PermissionEffect.ALLOW

        # With effect
        perm = parse_permission_string("deny:data:*:export")
        assert perm.effect == PermissionEffect.DENY

        # With scope
        perm = parse_permission_string("workflow:*:execute@tenant:123")
        assert perm.scope == "tenant:123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
