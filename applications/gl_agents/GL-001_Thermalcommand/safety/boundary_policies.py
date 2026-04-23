"""
Boundary Policies for GL-001 ThermalCommand Safety System

This module defines operating envelope limits and boundary policies for
thermal management operations. All policies follow zero-hallucination
principles with deterministic enforcement.

Policy Categories:
- Operating envelope limits (temperature, pressure, flow)
- Allowed tags whitelist
- Rate limits per tag type
- Condition-based rules
- Time-based restrictions

Example:
    >>> from boundary_policies import ThermalPolicyManager
    >>> manager = ThermalPolicyManager()
    >>> policies = manager.get_policies_for_tag("TI-101")
    >>> for policy in policies:
    ...     print(policy.policy_id)
"""

from datetime import time
from fnmatch import fnmatch
from typing import Dict, List, Optional, Set
import logging

from .safety_schemas import (
    BoundaryPolicy,
    PolicyType,
    ViolationSeverity,
    SafetyLevel,
    RateLimitSpec,
    TimeRestriction,
    Condition,
    ConditionOperator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OPERATING ENVELOPE LIMITS
# =============================================================================

# Temperature limits (degrees Celsius)
TEMPERATURE_POLICIES: List[BoundaryPolicy] = [
    BoundaryPolicy(
        policy_id="TEMP_ABS_001",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="TI-*",
        min_value=-40.0,
        max_value=200.0,
        engineering_units="degC",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute temperature limits for all temperature indicators",
    ),
    BoundaryPolicy(
        policy_id="TEMP_ABS_002",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="TIC-*",
        min_value=-40.0,
        max_value=200.0,
        engineering_units="degC",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute temperature limits for temperature controllers",
    ),
    BoundaryPolicy(
        policy_id="TEMP_ABS_003",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="TI-CRIT-*",
        min_value=0.0,
        max_value=150.0,
        engineering_units="degC",
        severity=ViolationSeverity.EMERGENCY,
        safety_level=SafetyLevel.SIL_3,
        description="Critical temperature limits for safety-critical sensors",
    ),
    BoundaryPolicy(
        policy_id="TEMP_RATE_001",
        policy_type=PolicyType.RATE_LIMIT,
        tag_pattern="TIC-*",
        rate_limit=RateLimitSpec(
            max_change_per_second=2.0,
            max_change_per_minute=30.0,
            max_writes_per_minute=60,
            cooldown_seconds=1.0,
        ),
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Rate limits for temperature setpoint changes",
    ),
]

# Pressure limits (kPa)
PRESSURE_POLICIES: List[BoundaryPolicy] = [
    BoundaryPolicy(
        policy_id="PRESS_ABS_001",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="PI-*",
        min_value=0.0,
        max_value=1500.0,
        engineering_units="kPa",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute pressure limits for all pressure indicators",
    ),
    BoundaryPolicy(
        policy_id="PRESS_ABS_002",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="PIC-*",
        min_value=0.0,
        max_value=1500.0,
        engineering_units="kPa",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute pressure limits for pressure controllers",
    ),
    BoundaryPolicy(
        policy_id="PRESS_RATE_001",
        policy_type=PolicyType.RATE_LIMIT,
        tag_pattern="PIC-*",
        rate_limit=RateLimitSpec(
            max_change_per_second=10.0,
            max_change_per_minute=100.0,
            max_writes_per_minute=30,
            cooldown_seconds=2.0,
        ),
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Rate limits for pressure setpoint changes",
    ),
]

# Flow limits (m3/h)
FLOW_POLICIES: List[BoundaryPolicy] = [
    BoundaryPolicy(
        policy_id="FLOW_ABS_001",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="FI-*",
        min_value=0.0,
        max_value=500.0,
        engineering_units="m3/h",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute flow limits for all flow indicators",
    ),
    BoundaryPolicy(
        policy_id="FLOW_ABS_002",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="FIC-*",
        min_value=0.0,
        max_value=500.0,
        engineering_units="m3/h",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute flow limits for flow controllers",
    ),
    BoundaryPolicy(
        policy_id="FLOW_RATE_001",
        policy_type=PolicyType.RATE_LIMIT,
        tag_pattern="FIC-*",
        rate_limit=RateLimitSpec(
            max_change_per_second=5.0,
            max_change_per_minute=50.0,
            max_writes_per_minute=30,
            cooldown_seconds=2.0,
        ),
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Rate limits for flow setpoint changes",
    ),
]

# Level limits (%)
LEVEL_POLICIES: List[BoundaryPolicy] = [
    BoundaryPolicy(
        policy_id="LEVEL_ABS_001",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="LI-*",
        min_value=0.0,
        max_value=100.0,
        engineering_units="%",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute level limits for all level indicators",
    ),
    BoundaryPolicy(
        policy_id="LEVEL_ABS_002",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="LIC-*",
        min_value=5.0,
        max_value=95.0,
        engineering_units="%",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute level limits for level controllers (with margin)",
    ),
    BoundaryPolicy(
        policy_id="LEVEL_RATE_001",
        policy_type=PolicyType.RATE_LIMIT,
        tag_pattern="LIC-*",
        rate_limit=RateLimitSpec(
            max_change_per_second=1.0,
            max_change_per_minute=10.0,
            max_writes_per_minute=20,
            cooldown_seconds=3.0,
        ),
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Rate limits for level setpoint changes",
    ),
]

# Valve position limits (%)
VALVE_POLICIES: List[BoundaryPolicy] = [
    BoundaryPolicy(
        policy_id="VALVE_ABS_001",
        policy_type=PolicyType.ABSOLUTE_LIMIT,
        tag_pattern="XV-*",
        min_value=0.0,
        max_value=100.0,
        engineering_units="%",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Absolute valve position limits for all valves",
    ),
    BoundaryPolicy(
        policy_id="VALVE_RATE_001",
        policy_type=PolicyType.RATE_LIMIT,
        tag_pattern="XV-*",
        rate_limit=RateLimitSpec(
            max_change_per_second=10.0,
            max_change_per_minute=100.0,
            max_writes_per_minute=30,
            cooldown_seconds=1.0,
        ),
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Rate limits for valve position changes",
    ),
]


# =============================================================================
# ALLOWED TAGS WHITELIST
# =============================================================================

# Whitelist of tags GL-001 is allowed to write to
ALLOWED_WRITE_TAGS: Set[str] = {
    # Temperature controllers
    "TIC-101", "TIC-102", "TIC-103", "TIC-104", "TIC-105",
    "TIC-201", "TIC-202", "TIC-203", "TIC-204", "TIC-205",
    "TIC-301", "TIC-302", "TIC-303", "TIC-304", "TIC-305",
    # Pressure controllers
    "PIC-101", "PIC-102", "PIC-103",
    "PIC-201", "PIC-202", "PIC-203",
    # Flow controllers
    "FIC-101", "FIC-102", "FIC-103", "FIC-104",
    "FIC-201", "FIC-202", "FIC-203", "FIC-204",
    # Level controllers
    "LIC-101", "LIC-102",
    "LIC-201", "LIC-202",
    # Valves (non-safety)
    "XV-101", "XV-102", "XV-103", "XV-104",
    "XV-201", "XV-202", "XV-203", "XV-204",
}

WHITELIST_POLICY = BoundaryPolicy(
    policy_id="WHITELIST_001",
    policy_type=PolicyType.WHITELIST,
    tag_pattern="*",
    allowed_tags=ALLOWED_WRITE_TAGS,
    severity=ViolationSeverity.CRITICAL,
    safety_level=SafetyLevel.SIL_2,
    description="Whitelist of tags GL-001 is authorized to write to",
)

# Tags that are NEVER writable by GL-001 (safety-critical)
BLACKLISTED_TAGS: Set[str] = {
    # Safety Instrumented System tags - NEVER WRITE
    "SIS-*", "ESD-*", "PSV-*", "TSV-*",
    # Emergency shutdown
    "XV-ESD-*", "XV-TRIP-*",
    # Safety relief valves
    "PSV-101", "PSV-102", "PSV-201", "PSV-202",
    # Trip initiators
    "TRIP-*", "SHUTDOWN-*",
}


# =============================================================================
# CONDITION-BASED RULES
# =============================================================================

CONDITIONAL_POLICIES: List[BoundaryPolicy] = [
    # Cannot increase temperature if pressure is high
    BoundaryPolicy(
        policy_id="COND_TEMP_PRESS_001",
        policy_type=PolicyType.CONDITIONAL,
        tag_pattern="TIC-*",
        max_value=100.0,  # Lower limit when pressure high
        conditions=[
            Condition(
                tag_id="PI-101",
                operator=ConditionOperator.GREATER_THAN,
                value=1000.0,
            )
        ],
        condition_logic="AND",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Limit temperature setpoint when system pressure is high",
    ),
    # Cannot increase flow if level is high
    BoundaryPolicy(
        policy_id="COND_FLOW_LEVEL_001",
        policy_type=PolicyType.CONDITIONAL,
        tag_pattern="FIC-*",
        max_value=200.0,  # Lower limit when level high
        conditions=[
            Condition(
                tag_id="LI-101",
                operator=ConditionOperator.GREATER_THAN,
                value=85.0,
            )
        ],
        condition_logic="AND",
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="Limit flow setpoint when tank level is high",
    ),
    # Interlock-based restriction
    BoundaryPolicy(
        policy_id="COND_INTERLOCK_001",
        policy_type=PolicyType.INTERLOCK,
        tag_pattern="*",
        conditions=[
            Condition(
                tag_id="INTERLOCK-MAIN",
                operator=ConditionOperator.EQUALS,
                value=True,  # Must be permissive
            )
        ],
        severity=ViolationSeverity.EMERGENCY,
        safety_level=SafetyLevel.SIL_3,
        description="All writes blocked when main interlock is not permissive",
    ),
]


# =============================================================================
# TIME-BASED RESTRICTIONS
# =============================================================================

TIME_BASED_POLICIES: List[BoundaryPolicy] = [
    # No large setpoint changes during night shift (reduced staffing)
    BoundaryPolicy(
        policy_id="TIME_NIGHT_001",
        policy_type=PolicyType.TIME_BASED,
        tag_pattern="*IC-*",
        rate_limit=RateLimitSpec(
            max_change_per_minute=5.0,  # Reduced rate at night
            max_writes_per_minute=10,
        ),
        time_restrictions=[
            TimeRestriction(
                start_time=time(22, 0),  # 10 PM
                end_time=time(6, 0),     # 6 AM
                days_of_week=[0, 1, 2, 3, 4, 5, 6],
                timezone="UTC",
            )
        ],
        severity=ViolationSeverity.WARNING,
        safety_level=SafetyLevel.SIL_1,
        description="Reduced rate limits during night shift",
    ),
    # No optimization changes during maintenance windows
    BoundaryPolicy(
        policy_id="TIME_MAINT_001",
        policy_type=PolicyType.TIME_BASED,
        tag_pattern="*",
        max_value=0.0,  # Effectively blocks all writes during maintenance
        time_restrictions=[
            TimeRestriction(
                start_time=time(2, 0),   # 2 AM
                end_time=time(4, 0),     # 4 AM
                days_of_week=[6],        # Sunday only
                timezone="UTC",
            )
        ],
        severity=ViolationSeverity.CRITICAL,
        safety_level=SafetyLevel.SIL_2,
        description="No GL-001 writes during maintenance window",
        enabled=False,  # Enable when maintenance scheduled
    ),
]


# =============================================================================
# POLICY MANAGER
# =============================================================================

class ThermalPolicyManager:
    """
    Manager for thermal boundary policies.

    Provides access to all defined policies and methods to query
    applicable policies for specific tags.

    Attributes:
        policies: All registered policies
        whitelist_policy: Tag whitelist policy
        blacklisted_patterns: Patterns for blacklisted tags

    Example:
        >>> manager = ThermalPolicyManager()
        >>> policies = manager.get_policies_for_tag("TIC-101")
        >>> print(f"Found {len(policies)} applicable policies")
    """

    def __init__(self) -> None:
        """Initialize the policy manager with default policies."""
        self._policies: Dict[str, BoundaryPolicy] = {}
        self._whitelist_policy = WHITELIST_POLICY
        self._blacklisted_patterns = BLACKLISTED_TAGS

        # Register all default policies
        self._register_default_policies()

        logger.info(
            f"ThermalPolicyManager initialized with {len(self._policies)} policies"
        )

    def _register_default_policies(self) -> None:
        """Register all default policies."""
        all_policies = (
            TEMPERATURE_POLICIES +
            PRESSURE_POLICIES +
            FLOW_POLICIES +
            LEVEL_POLICIES +
            VALVE_POLICIES +
            CONDITIONAL_POLICIES +
            TIME_BASED_POLICIES +
            [WHITELIST_POLICY]
        )

        for policy in all_policies:
            self.register_policy(policy)

    def register_policy(self, policy: BoundaryPolicy) -> None:
        """
        Register a boundary policy.

        Args:
            policy: Policy to register

        Raises:
            ValueError: If policy with same ID already exists
        """
        if policy.policy_id in self._policies:
            raise ValueError(f"Policy {policy.policy_id} already registered")

        self._policies[policy.policy_id] = policy
        logger.debug(f"Registered policy: {policy.policy_id}")

    def get_policy(self, policy_id: str) -> Optional[BoundaryPolicy]:
        """
        Get a policy by ID.

        Args:
            policy_id: Policy identifier

        Returns:
            Policy if found, None otherwise
        """
        return self._policies.get(policy_id)

    def get_all_policies(self) -> List[BoundaryPolicy]:
        """
        Get all registered policies.

        Returns:
            List of all policies
        """
        return list(self._policies.values())

    def get_enabled_policies(self) -> List[BoundaryPolicy]:
        """
        Get all enabled policies.

        Returns:
            List of enabled policies
        """
        return [p for p in self._policies.values() if p.enabled]

    def get_policies_for_tag(self, tag_id: str) -> List[BoundaryPolicy]:
        """
        Get all applicable policies for a tag.

        Args:
            tag_id: Tag identifier

        Returns:
            List of policies that apply to this tag
        """
        applicable = []

        for policy in self._policies.values():
            if not policy.enabled:
                continue

            # Check if tag matches policy pattern
            if fnmatch(tag_id, policy.tag_pattern):
                applicable.append(policy)

            # Also check whitelist if it applies to this tag
            if policy.policy_type == PolicyType.WHITELIST:
                if policy.allowed_tags and tag_id in policy.allowed_tags:
                    applicable.append(policy)

        return applicable

    def get_policies_by_type(self, policy_type: PolicyType) -> List[BoundaryPolicy]:
        """
        Get policies of a specific type.

        Args:
            policy_type: Type of policy to filter by

        Returns:
            List of policies of specified type
        """
        return [
            p for p in self._policies.values()
            if p.policy_type == policy_type and p.enabled
        ]

    def is_tag_allowed(self, tag_id: str) -> bool:
        """
        Check if a tag is in the allowed whitelist.

        Args:
            tag_id: Tag identifier

        Returns:
            True if tag is allowed for GL-001 writes
        """
        # Check blacklist first
        for pattern in self._blacklisted_patterns:
            if fnmatch(tag_id, pattern):
                logger.warning(f"Tag {tag_id} matches blacklist pattern {pattern}")
                return False

        # Check whitelist
        if self._whitelist_policy.allowed_tags:
            return tag_id in self._whitelist_policy.allowed_tags

        return False

    def is_tag_blacklisted(self, tag_id: str) -> bool:
        """
        Check if a tag is blacklisted (safety-critical).

        Args:
            tag_id: Tag identifier

        Returns:
            True if tag is blacklisted
        """
        for pattern in self._blacklisted_patterns:
            if fnmatch(tag_id, pattern):
                return True
        return False

    def get_limits_for_tag(
        self,
        tag_id: str
    ) -> Dict[str, Optional[float]]:
        """
        Get min/max limits for a tag.

        Args:
            tag_id: Tag identifier

        Returns:
            Dict with 'min' and 'max' keys
        """
        limits: Dict[str, Optional[float]] = {"min": None, "max": None}

        for policy in self.get_policies_for_tag(tag_id):
            if policy.policy_type == PolicyType.ABSOLUTE_LIMIT:
                # Use most restrictive limits
                if policy.min_value is not None:
                    if limits["min"] is None:
                        limits["min"] = policy.min_value
                    else:
                        limits["min"] = max(limits["min"], policy.min_value)

                if policy.max_value is not None:
                    if limits["max"] is None:
                        limits["max"] = policy.max_value
                    else:
                        limits["max"] = min(limits["max"], policy.max_value)

        return limits

    def get_rate_limits_for_tag(
        self,
        tag_id: str
    ) -> Optional[RateLimitSpec]:
        """
        Get rate limits for a tag.

        Args:
            tag_id: Tag identifier

        Returns:
            Most restrictive rate limit specification
        """
        most_restrictive: Optional[RateLimitSpec] = None

        for policy in self.get_policies_for_tag(tag_id):
            if policy.policy_type == PolicyType.RATE_LIMIT and policy.rate_limit:
                if most_restrictive is None:
                    most_restrictive = policy.rate_limit
                else:
                    # Merge with most restrictive values
                    most_restrictive = RateLimitSpec(
                        max_change_per_second=min(
                            most_restrictive.max_change_per_second or float('inf'),
                            policy.rate_limit.max_change_per_second or float('inf')
                        ) if most_restrictive.max_change_per_second or policy.rate_limit.max_change_per_second else None,
                        max_change_per_minute=min(
                            most_restrictive.max_change_per_minute or float('inf'),
                            policy.rate_limit.max_change_per_minute or float('inf')
                        ) if most_restrictive.max_change_per_minute or policy.rate_limit.max_change_per_minute else None,
                        max_writes_per_minute=min(
                            most_restrictive.max_writes_per_minute or 999999,
                            policy.rate_limit.max_writes_per_minute or 999999
                        ) if most_restrictive.max_writes_per_minute or policy.rate_limit.max_writes_per_minute else None,
                        cooldown_seconds=max(
                            most_restrictive.cooldown_seconds or 0.0,
                            policy.rate_limit.cooldown_seconds or 0.0
                        ) if most_restrictive.cooldown_seconds or policy.rate_limit.cooldown_seconds else None,
                    )

        return most_restrictive

    def update_whitelist(self, allowed_tags: Set[str]) -> None:
        """
        Update the allowed tags whitelist.

        Args:
            allowed_tags: New set of allowed tags

        Note:
            This should only be called during configuration updates
            with proper authorization.
        """
        # Validate no blacklisted tags in whitelist
        for tag in allowed_tags:
            if self.is_tag_blacklisted(tag):
                raise ValueError(f"Cannot whitelist blacklisted tag: {tag}")

        self._whitelist_policy = BoundaryPolicy(
            policy_id="WHITELIST_001",
            policy_type=PolicyType.WHITELIST,
            tag_pattern="*",
            allowed_tags=allowed_tags,
            severity=ViolationSeverity.CRITICAL,
            safety_level=SafetyLevel.SIL_2,
            description="Whitelist of tags GL-001 is authorized to write to",
        )

        # Update in registry
        self._policies["WHITELIST_001"] = self._whitelist_policy
        logger.info(f"Updated whitelist with {len(allowed_tags)} tags")

    def enable_policy(self, policy_id: str) -> bool:
        """
        Enable a policy.

        Args:
            policy_id: Policy identifier

        Returns:
            True if policy was enabled
        """
        policy = self._policies.get(policy_id)
        if policy:
            # Create new policy with enabled=True (since policies should be immutable)
            self._policies[policy_id] = BoundaryPolicy(
                **{**policy.dict(), "enabled": True}
            )
            logger.info(f"Enabled policy: {policy_id}")
            return True
        return False

    def disable_policy(self, policy_id: str) -> bool:
        """
        Disable a policy.

        Args:
            policy_id: Policy identifier

        Returns:
            True if policy was disabled

        Note:
            Some policies cannot be disabled (safety-critical)
        """
        policy = self._policies.get(policy_id)
        if not policy:
            return False

        # Cannot disable whitelist or SIL-3+ policies
        if policy.policy_id == "WHITELIST_001":
            logger.warning("Cannot disable whitelist policy")
            return False

        if policy.safety_level == SafetyLevel.SIL_3:
            logger.warning(f"Cannot disable SIL-3 policy: {policy_id}")
            return False

        self._policies[policy_id] = BoundaryPolicy(
            **{**policy.dict(), "enabled": False}
        )
        logger.info(f"Disabled policy: {policy_id}")
        return True

    def get_statistics(self) -> Dict[str, int]:
        """
        Get policy statistics.

        Returns:
            Dict with policy counts by type and status
        """
        stats = {
            "total": len(self._policies),
            "enabled": len([p for p in self._policies.values() if p.enabled]),
            "disabled": len([p for p in self._policies.values() if not p.enabled]),
        }

        # Count by type
        for policy_type in PolicyType:
            stats[f"type_{policy_type.value}"] = len([
                p for p in self._policies.values()
                if p.policy_type == policy_type
            ])

        # Count by severity
        for severity in ViolationSeverity:
            stats[f"severity_{severity.value}"] = len([
                p for p in self._policies.values()
                if p.severity == severity
            ])

        return stats


# Module-level singleton for convenience
_default_manager: Optional[ThermalPolicyManager] = None


def get_policy_manager() -> ThermalPolicyManager:
    """
    Get the default policy manager singleton.

    Returns:
        ThermalPolicyManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ThermalPolicyManager()
    return _default_manager


def reset_policy_manager() -> None:
    """Reset the default policy manager singleton."""
    global _default_manager
    _default_manager = None
