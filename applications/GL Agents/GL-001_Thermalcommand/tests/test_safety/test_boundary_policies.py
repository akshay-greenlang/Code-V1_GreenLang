"""
Tests for Boundary Policies

Tests policy definitions, whitelist/blacklist management, and
policy manager functionality.
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.boundary_policies import (
    ThermalPolicyManager,
    get_policy_manager,
    reset_policy_manager,
    TEMPERATURE_POLICIES,
    PRESSURE_POLICIES,
    FLOW_POLICIES,
    LEVEL_POLICIES,
    VALVE_POLICIES,
    ALLOWED_WRITE_TAGS,
    BLACKLISTED_TAGS,
    WHITELIST_POLICY,
)
from safety.safety_schemas import (
    BoundaryPolicy,
    PolicyType,
    ViolationSeverity,
    SafetyLevel,
    RateLimitSpec,
)


class TestDefaultPolicies:
    """Test default policy definitions."""

    def test_temperature_policies_defined(self):
        """Test temperature policies are defined."""
        assert len(TEMPERATURE_POLICIES) > 0
        # Should have absolute and rate limit policies
        types = {p.policy_type for p in TEMPERATURE_POLICIES}
        assert PolicyType.ABSOLUTE_LIMIT in types
        assert PolicyType.RATE_LIMIT in types

    def test_temperature_limits_reasonable(self):
        """Test temperature limits are reasonable."""
        for policy in TEMPERATURE_POLICIES:
            if policy.policy_type == PolicyType.ABSOLUTE_LIMIT:
                if policy.min_value is not None:
                    assert policy.min_value >= -273.15  # Above absolute zero
                if policy.max_value is not None:
                    assert policy.max_value <= 1000.0  # Reasonable industrial max

    def test_pressure_policies_defined(self):
        """Test pressure policies are defined."""
        assert len(PRESSURE_POLICIES) > 0
        for policy in PRESSURE_POLICIES:
            if policy.policy_type == PolicyType.ABSOLUTE_LIMIT:
                if policy.min_value is not None:
                    assert policy.min_value >= 0.0  # No negative pressure

    def test_all_policies_have_ids(self):
        """Test all policies have unique IDs."""
        all_policies = (
            TEMPERATURE_POLICIES +
            PRESSURE_POLICIES +
            FLOW_POLICIES +
            LEVEL_POLICIES +
            VALVE_POLICIES
        )
        ids = [p.policy_id for p in all_policies]
        assert len(ids) == len(set(ids))  # All unique


class TestAllowedWriteTags:
    """Test allowed write tags whitelist."""

    def test_whitelist_contains_controller_tags(self):
        """Test whitelist contains controller tags."""
        controller_patterns = ["TIC-", "PIC-", "FIC-", "LIC-"]
        for pattern in controller_patterns:
            matches = [t for t in ALLOWED_WRITE_TAGS if t.startswith(pattern)]
            assert len(matches) > 0, f"No tags matching {pattern}"

    def test_whitelist_contains_valve_tags(self):
        """Test whitelist contains valve tags."""
        valve_tags = [t for t in ALLOWED_WRITE_TAGS if t.startswith("XV-")]
        assert len(valve_tags) > 0

    def test_whitelist_excludes_sis_tags(self):
        """Test whitelist does not contain SIS tags."""
        for tag in ALLOWED_WRITE_TAGS:
            assert not tag.startswith("SIS-"), f"SIS tag in whitelist: {tag}"
            assert not tag.startswith("ESD-"), f"ESD tag in whitelist: {tag}"
            assert not tag.startswith("TRIP-"), f"TRIP tag in whitelist: {tag}"


class TestBlacklistedTags:
    """Test blacklisted tags."""

    def test_blacklist_contains_sis_patterns(self):
        """Test blacklist contains SIS patterns."""
        sis_patterns = ["SIS-*", "ESD-*", "PSV-*", "TRIP-*"]
        for pattern in sis_patterns:
            assert pattern in BLACKLISTED_TAGS, f"Missing pattern: {pattern}"

    def test_blacklist_contains_safety_valve_patterns(self):
        """Test blacklist contains safety valve patterns."""
        safety_patterns = ["XV-ESD-*", "XV-TRIP-*"]
        for pattern in safety_patterns:
            assert pattern in BLACKLISTED_TAGS


class TestThermalPolicyManager:
    """Test ThermalPolicyManager."""

    @pytest.fixture
    def manager(self):
        """Create fresh policy manager."""
        reset_policy_manager()
        return ThermalPolicyManager()

    def test_initialization(self, manager):
        """Test manager initializes with default policies."""
        stats = manager.get_statistics()
        assert stats["total"] > 0
        assert stats["enabled"] > 0

    def test_get_policy_by_id(self, manager):
        """Test getting policy by ID."""
        policy = manager.get_policy("TEMP_ABS_001")
        assert policy is not None
        assert policy.policy_id == "TEMP_ABS_001"

    def test_get_nonexistent_policy(self, manager):
        """Test getting nonexistent policy returns None."""
        policy = manager.get_policy("NONEXISTENT")
        assert policy is None

    def test_get_policies_for_tag(self, manager):
        """Test getting policies for a tag."""
        policies = manager.get_policies_for_tag("TIC-101")
        assert len(policies) > 0
        # Should have both absolute and rate limits
        types = {p.policy_type for p in policies}
        assert PolicyType.ABSOLUTE_LIMIT in types

    def test_get_policies_for_tag_pattern_matching(self, manager):
        """Test policy pattern matching works."""
        # TI-* pattern should match TI-101
        policies = manager.get_policies_for_tag("TI-101")
        matching = [p for p in policies if "TI-*" in p.tag_pattern or p.tag_pattern == "TI-*"]
        assert len(matching) > 0

    def test_is_tag_allowed(self, manager):
        """Test checking if tag is allowed."""
        assert manager.is_tag_allowed("TIC-101") == True
        assert manager.is_tag_allowed("UNKNOWN-999") == False

    def test_is_tag_blacklisted(self, manager):
        """Test checking if tag is blacklisted."""
        assert manager.is_tag_blacklisted("SIS-101") == True
        assert manager.is_tag_blacklisted("ESD-001") == True
        assert manager.is_tag_blacklisted("TIC-101") == False

    def test_get_limits_for_tag(self, manager):
        """Test getting min/max limits for tag."""
        limits = manager.get_limits_for_tag("TIC-101")
        assert "min" in limits
        assert "max" in limits
        # Temperature controller should have limits
        assert limits["min"] is not None or limits["max"] is not None

    def test_get_rate_limits_for_tag(self, manager):
        """Test getting rate limits for tag."""
        rate_limits = manager.get_rate_limits_for_tag("TIC-101")
        # Temperature controllers should have rate limits
        assert rate_limits is not None
        assert rate_limits.max_change_per_second is not None

    def test_register_policy(self, manager):
        """Test registering a new policy."""
        new_policy = BoundaryPolicy(
            policy_id="CUSTOM_001",
            policy_type=PolicyType.ABSOLUTE_LIMIT,
            tag_pattern="CUSTOM-*",
            max_value=100.0,
        )
        initial_count = len(manager.get_all_policies())
        manager.register_policy(new_policy)
        assert len(manager.get_all_policies()) == initial_count + 1

    def test_register_duplicate_policy_fails(self, manager):
        """Test registering duplicate policy fails."""
        policy = BoundaryPolicy(
            policy_id="TEMP_ABS_001",  # Already exists
            policy_type=PolicyType.ABSOLUTE_LIMIT,
            tag_pattern="TI-*",
            max_value=100.0,
        )
        with pytest.raises(ValueError):
            manager.register_policy(policy)

    def test_enable_disable_policy(self, manager):
        """Test enabling and disabling policies."""
        # Disable a policy
        assert manager.disable_policy("TEMP_RATE_001") == True
        policy = manager.get_policy("TEMP_RATE_001")
        assert policy.enabled == False

        # Re-enable
        assert manager.enable_policy("TEMP_RATE_001") == True
        policy = manager.get_policy("TEMP_RATE_001")
        assert policy.enabled == True

    def test_cannot_disable_whitelist(self, manager):
        """Test cannot disable whitelist policy."""
        result = manager.disable_policy("WHITELIST_001")
        assert result == False  # Should not disable

    def test_get_policies_by_type(self, manager):
        """Test getting policies by type."""
        rate_policies = manager.get_policies_by_type(PolicyType.RATE_LIMIT)
        assert len(rate_policies) > 0
        for policy in rate_policies:
            assert policy.policy_type == PolicyType.RATE_LIMIT

    def test_update_whitelist(self, manager):
        """Test updating whitelist."""
        new_tags = {"CUSTOM-001", "CUSTOM-002"}
        manager.update_whitelist(new_tags)
        assert manager.is_tag_allowed("CUSTOM-001") == True
        assert manager.is_tag_allowed("TIC-101") == False  # Old tags removed

    def test_update_whitelist_rejects_blacklisted(self, manager):
        """Test cannot add blacklisted tags to whitelist."""
        with pytest.raises(ValueError):
            manager.update_whitelist({"SIS-101"})  # Blacklisted

    def test_statistics(self, manager):
        """Test getting statistics."""
        stats = manager.get_statistics()
        assert "total" in stats
        assert "enabled" in stats
        assert "disabled" in stats
        assert stats["enabled"] + stats["disabled"] == stats["total"]


class TestPolicyManagerSingleton:
    """Test policy manager singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_policy_manager()

    def test_get_policy_manager_returns_same_instance(self):
        """Test singleton returns same instance."""
        manager1 = get_policy_manager()
        manager2 = get_policy_manager()
        assert manager1 is manager2

    def test_reset_creates_new_instance(self):
        """Test reset creates new instance."""
        manager1 = get_policy_manager()
        reset_policy_manager()
        manager2 = get_policy_manager()
        assert manager1 is not manager2


class TestConditionalPolicies:
    """Test conditional policy definitions."""

    @pytest.fixture
    def manager(self):
        """Create fresh policy manager."""
        reset_policy_manager()
        return ThermalPolicyManager()

    def test_conditional_policies_exist(self, manager):
        """Test conditional policies are defined."""
        conditional = manager.get_policies_by_type(PolicyType.CONDITIONAL)
        # May or may not have conditional policies enabled
        # Just verify the query works
        assert isinstance(conditional, list)

    def test_interlock_policies_exist(self, manager):
        """Test interlock policies are defined."""
        interlock = manager.get_policies_by_type(PolicyType.INTERLOCK)
        assert isinstance(interlock, list)


class TestPolicyValidation:
    """Test policy validation logic."""

    def test_policy_severity_levels(self):
        """Test all policies have valid severity."""
        all_policies = (
            TEMPERATURE_POLICIES +
            PRESSURE_POLICIES +
            FLOW_POLICIES +
            LEVEL_POLICIES +
            VALVE_POLICIES
        )
        for policy in all_policies:
            assert policy.severity in ViolationSeverity

    def test_policy_safety_levels(self):
        """Test safety-critical policies have safety levels."""
        all_policies = (
            TEMPERATURE_POLICIES +
            PRESSURE_POLICIES
        )
        # Most policies should have safety levels
        policies_with_levels = [p for p in all_policies if p.safety_level is not None]
        assert len(policies_with_levels) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
