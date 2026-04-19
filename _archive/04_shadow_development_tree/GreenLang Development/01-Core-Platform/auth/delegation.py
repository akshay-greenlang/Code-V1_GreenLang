# -*- coding: utf-8 -*-
"""
Permission Delegation Mechanism for GreenLang

This module provides temporary permission grants and delegation chains,
allowing users to delegate specific permissions to others for limited time periods.

Delegation Example:
    Manager delegates "workflow:carbon-audit:execute" to Analyst for 7 days
    - Analyst can now execute the workflow
    - Delegation expires after 7 days
    - Delegation can be revoked anytime
    - Delegation chain tracked for audit

Features:
    - Temporary permission grants
    - Delegation chains and tracking
    - Automatic expiration
    - Revocation mechanisms
    - Delegation limits and constraints

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.auth.permissions import Permission, PermissionEffect
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Delegation Models
# ==============================================================================

class DelegationStatus(str, Enum):
    """Status of a delegation."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


class DelegationType(str, Enum):
    """Type of delegation."""
    DIRECT = "direct"  # Direct delegation from one user to another
    ROLE_BASED = "role_based"  # Delegation based on role
    TEMPORARY = "temporary"  # Time-limited delegation
    CONDITIONAL = "conditional"  # Delegation with conditions


class DelegationConstraint(BaseModel):
    """Constraint on a delegation."""

    constraint_type: str = Field(..., description="Type of constraint")
    constraint_value: Any = Field(..., description="Value for the constraint")

    def is_satisfied(self, context: Dict[str, Any]) -> bool:
        """
        Check if constraint is satisfied.

        Args:
            context: Evaluation context

        Returns:
            True if constraint is satisfied
        """
        if self.constraint_type == "max_uses":
            current_uses = context.get("uses", 0)
            return current_uses < self.constraint_value

        elif self.constraint_type == "ip_whitelist":
            ip_address = context.get("ip_address")
            if not ip_address:
                return False
            return ip_address in self.constraint_value

        elif self.constraint_type == "time_window":
            # constraint_value should be {"start_hour": 9, "end_hour": 17}
            current_hour = DeterministicClock.now().hour
            start = self.constraint_value.get("start_hour", 0)
            end = self.constraint_value.get("end_hour", 24)
            return start <= current_hour < end

        elif self.constraint_type == "require_approval":
            return context.get("approved", False)

        return True


class PermissionDelegation(BaseModel):
    """
    Permission delegation from one principal to another.

    Represents a temporary or conditional grant of specific permissions.
    """

    delegation_id: str = Field(
        default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        description="Unique delegation identifier"
    )

    # Who and what
    delegator_id: str = Field(..., description="User delegating the permission")
    delegatee_id: str = Field(..., description="User receiving the delegation")
    permission: Permission = Field(..., description="Permission being delegated")

    # Delegation metadata
    delegation_type: DelegationType = Field(
        default=DelegationType.DIRECT,
        description="Type of delegation"
    )
    status: DelegationStatus = Field(
        default=DelegationStatus.ACTIVE,
        description="Current status"
    )

    # Time constraints
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When delegation was created"
    )
    effective_from: Optional[datetime] = Field(
        None,
        description="When delegation becomes effective"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="When delegation expires"
    )

    # Delegation constraints
    constraints: List[DelegationConstraint] = Field(
        default_factory=list,
        description="Additional constraints on delegation"
    )

    # Usage tracking
    usage_count: int = Field(default=0, description="Number of times used")
    max_uses: Optional[int] = Field(None, description="Maximum allowed uses")

    # Chain tracking
    parent_delegation_id: Optional[str] = Field(
        None,
        description="Parent delegation if this is part of a chain"
    )
    can_delegate: bool = Field(
        default=False,
        description="Whether delegatee can further delegate"
    )

    # Metadata
    reason: Optional[str] = Field(None, description="Reason for delegation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    revoked_at: Optional[datetime] = Field(None, description="When delegation was revoked")
    revoked_by: Optional[str] = Field(None, description="Who revoked the delegation")
    revocation_reason: Optional[str] = Field(None, description="Why it was revoked")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            DelegationStatus: lambda v: v.value,
            DelegationType: lambda v: v.value
        }

    def is_active(self) -> bool:
        """Check if delegation is currently active."""
        if self.status != DelegationStatus.ACTIVE:
            return False

        now = DeterministicClock.utcnow()

        # Check effective_from
        if self.effective_from and now < self.effective_from:
            return False

        # Check expiration
        if self.expires_at and now > self.expires_at:
            return False

        # Check max uses
        if self.max_uses is not None and self.usage_count >= self.max_uses:
            return False

        return True

    def is_expired(self) -> bool:
        """Check if delegation has expired."""
        if self.expires_at:
            return DeterministicClock.utcnow() > self.expires_at
        return False

    def check_constraints(self, context: Dict[str, Any]) -> bool:
        """
        Check if all constraints are satisfied.

        Args:
            context: Evaluation context

        Returns:
            True if all constraints satisfied
        """
        # Add usage count to context
        context['uses'] = self.usage_count

        for constraint in self.constraints:
            if not constraint.is_satisfied(context):
                return False

        return True

    def increment_usage(self):
        """Increment usage count."""
        self.usage_count += 1

    def revoke(self, revoked_by: str, reason: Optional[str] = None):
        """
        Revoke this delegation.

        Args:
            revoked_by: User revoking the delegation
            reason: Reason for revocation
        """
        self.status = DelegationStatus.REVOKED
        self.revoked_at = DeterministicClock.utcnow()
        self.revoked_by = revoked_by
        self.revocation_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'delegation_id': self.delegation_id,
            'delegator_id': self.delegator_id,
            'delegatee_id': self.delegatee_id,
            'permission': self.permission.to_dict(),
            'delegation_type': self.delegation_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'effective_from': self.effective_from.isoformat() if self.effective_from else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'constraints': [c.dict() for c in self.constraints],
            'usage_count': self.usage_count,
            'max_uses': self.max_uses,
            'parent_delegation_id': self.parent_delegation_id,
            'can_delegate': self.can_delegate,
            'reason': self.reason,
            'metadata': self.metadata,
            'revoked_at': self.revoked_at.isoformat() if self.revoked_at else None,
            'revoked_by': self.revoked_by,
            'revocation_reason': self.revocation_reason
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PermissionDelegation':
        """Create from dictionary."""
        if 'permission' in data and isinstance(data['permission'], dict):
            data['permission'] = Permission.from_dict(data['permission'])
        if 'delegation_type' in data and isinstance(data['delegation_type'], str):
            data['delegation_type'] = DelegationType(data['delegation_type'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = DelegationStatus(data['status'])
        if 'constraints' in data and data['constraints']:
            data['constraints'] = [
                DelegationConstraint(**c) if isinstance(c, dict) else c
                for c in data['constraints']
            ]

        # Parse datetime fields
        for field in ['created_at', 'effective_from', 'expires_at', 'revoked_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)


# ==============================================================================
# Delegation Manager
# ==============================================================================

class DelegationManager:
    """
    Manages permission delegations.

    Provides:
    - Creating and revoking delegations
    - Tracking delegation chains
    - Automatic expiration
    - Usage limits enforcement
    """

    def __init__(self, max_delegation_chain: int = 3):
        """
        Initialize delegation manager.

        Args:
            max_delegation_chain: Maximum allowed delegation chain length
        """
        self.max_delegation_chain = max_delegation_chain
        self._delegations: Dict[str, PermissionDelegation] = {}
        self._user_delegations: Dict[str, Set[str]] = defaultdict(set)  # delegatee -> delegation_ids
        self._delegator_grants: Dict[str, Set[str]] = defaultdict(set)  # delegator -> delegation_ids

        self._stats = {
            'total_delegations': 0,
            'active_delegations': 0,
            'expired_delegations': 0,
            'revoked_delegations': 0
        }

        logger.info(f"Initialized DelegationManager (max_chain={max_delegation_chain})")

    def delegate(
        self,
        delegator_id: str,
        delegatee_id: str,
        permission: Permission,
        duration: Optional[timedelta] = None,
        effective_from: Optional[datetime] = None,
        max_uses: Optional[int] = None,
        constraints: Optional[List[DelegationConstraint]] = None,
        can_delegate: bool = False,
        reason: Optional[str] = None,
        parent_delegation_id: Optional[str] = None
    ) -> PermissionDelegation:
        """
        Create a new permission delegation.

        Args:
            delegator_id: User delegating the permission
            delegatee_id: User receiving the delegation
            permission: Permission to delegate
            duration: How long the delegation lasts
            effective_from: When delegation becomes effective
            max_uses: Maximum number of uses
            constraints: Additional constraints
            can_delegate: Whether delegatee can further delegate
            reason: Reason for delegation
            parent_delegation_id: Parent delegation if this is a chain

        Returns:
            Created delegation

        Raises:
            ValueError: If delegation chain is too long or delegator lacks permission
        """
        # Validate delegation chain length
        if parent_delegation_id:
            chain_length = self._get_chain_length(parent_delegation_id)
            if chain_length >= self.max_delegation_chain:
                raise ValueError(
                    f"Delegation chain too long (max={self.max_delegation_chain})"
                )

            # Verify parent delegation allows further delegation
            parent = self._delegations.get(parent_delegation_id)
            if not parent or not parent.can_delegate:
                raise ValueError("Parent delegation does not allow further delegation")

        # Calculate expiration
        expires_at = None
        if duration:
            start = effective_from or DeterministicClock.utcnow()
            expires_at = start + duration

        # Create delegation
        delegation = PermissionDelegation(
            delegator_id=delegator_id,
            delegatee_id=delegatee_id,
            permission=permission,
            effective_from=effective_from,
            expires_at=expires_at,
            max_uses=max_uses,
            constraints=constraints or [],
            can_delegate=can_delegate,
            reason=reason,
            parent_delegation_id=parent_delegation_id
        )

        # Store delegation
        self._delegations[delegation.delegation_id] = delegation
        self._user_delegations[delegatee_id].add(delegation.delegation_id)
        self._delegator_grants[delegator_id].add(delegation.delegation_id)

        # Update stats
        self._stats['total_delegations'] += 1
        self._stats['active_delegations'] += 1

        logger.info(
            f"Created delegation {delegation.delegation_id}: "
            f"{delegator_id} -> {delegatee_id} ({permission.to_string()})"
        )

        return delegation

    def revoke(
        self,
        delegation_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
        cascade: bool = False
    ) -> bool:
        """
        Revoke a delegation.

        Args:
            delegation_id: Delegation to revoke
            revoked_by: User revoking the delegation
            reason: Reason for revocation
            cascade: If True, also revoke child delegations

        Returns:
            True if revoked successfully
        """
        delegation = self._delegations.get(delegation_id)
        if not delegation:
            return False

        # Revoke the delegation
        delegation.revoke(revoked_by, reason)

        # Update stats
        self._stats['active_delegations'] -= 1
        self._stats['revoked_delegations'] += 1

        logger.info(f"Revoked delegation {delegation_id}")

        # Cascade to children if requested
        if cascade:
            children = self._find_child_delegations(delegation_id)
            for child in children:
                self.revoke(child.delegation_id, revoked_by, "Parent delegation revoked", cascade=True)

        return True

    def get_delegation(self, delegation_id: str) -> Optional[PermissionDelegation]:
        """Get delegation by ID."""
        return self._delegations.get(delegation_id)

    def get_user_delegations(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[PermissionDelegation]:
        """
        Get all delegations for a user (where user is delegatee).

        Args:
            user_id: User ID
            active_only: Only return active delegations

        Returns:
            List of delegations
        """
        delegation_ids = self._user_delegations.get(user_id, set())
        delegations = [
            self._delegations[did] for did in delegation_ids
            if did in self._delegations
        ]

        if active_only:
            delegations = [d for d in delegations if d.is_active()]

        return delegations

    def get_delegated_permissions(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Permission]:
        """
        Get all permissions delegated to a user.

        Args:
            user_id: User ID
            context: Evaluation context for constraints

        Returns:
            List of permissions
        """
        context = context or {}
        permissions = []

        delegations = self.get_user_delegations(user_id, active_only=True)

        for delegation in delegations:
            # Check constraints
            if delegation.check_constraints(context):
                permissions.append(delegation.permission)

        return permissions

    def record_usage(self, delegation_id: str) -> bool:
        """
        Record usage of a delegation.

        Args:
            delegation_id: Delegation ID

        Returns:
            True if recorded successfully
        """
        delegation = self._delegations.get(delegation_id)
        if not delegation:
            return False

        delegation.increment_usage()
        return True

    def cleanup_expired(self) -> int:
        """
        Clean up expired delegations.

        Returns:
            Number of delegations cleaned up
        """
        count = 0

        for delegation in list(self._delegations.values()):
            if delegation.status == DelegationStatus.ACTIVE and delegation.is_expired():
                delegation.status = DelegationStatus.EXPIRED
                self._stats['active_delegations'] -= 1
                self._stats['expired_delegations'] += 1
                count += 1
                logger.info(f"Expired delegation {delegation.delegation_id}")

        return count

    def get_delegation_chain(self, delegation_id: str) -> List[PermissionDelegation]:
        """
        Get the full delegation chain for a delegation.

        Args:
            delegation_id: Delegation ID

        Returns:
            List of delegations in the chain (from root to leaf)
        """
        chain = []
        current_id = delegation_id

        while current_id:
            delegation = self._delegations.get(current_id)
            if not delegation:
                break

            chain.insert(0, delegation)
            current_id = delegation.parent_delegation_id

        return chain

    def _get_chain_length(self, delegation_id: str) -> int:
        """Get the length of a delegation chain."""
        return len(self.get_delegation_chain(delegation_id))

    def _find_child_delegations(self, parent_id: str) -> List[PermissionDelegation]:
        """Find all child delegations of a parent."""
        children = []

        for delegation in self._delegations.values():
            if delegation.parent_delegation_id == parent_id:
                children.append(delegation)

        return children

    def list_delegations(
        self,
        delegator_id: Optional[str] = None,
        delegatee_id: Optional[str] = None,
        status: Optional[DelegationStatus] = None
    ) -> List[PermissionDelegation]:
        """
        List delegations matching criteria.

        Args:
            delegator_id: Filter by delegator
            delegatee_id: Filter by delegatee
            status: Filter by status

        Returns:
            List of matching delegations
        """
        delegations = list(self._delegations.values())

        if delegator_id:
            delegations = [d for d in delegations if d.delegator_id == delegator_id]

        if delegatee_id:
            delegations = [d for d in delegations if d.delegatee_id == delegatee_id]

        if status:
            delegations = [d for d in delegations if d.status == status]

        return delegations

    def get_stats(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        return self._stats.copy()


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_temporary_delegation(
    manager: DelegationManager,
    delegator_id: str,
    delegatee_id: str,
    permission: Permission,
    hours: int = 24,
    reason: Optional[str] = None
) -> PermissionDelegation:
    """
    Convenience function to create a temporary delegation.

    Args:
        manager: DelegationManager instance
        delegator_id: User delegating
        delegatee_id: User receiving delegation
        permission: Permission to delegate
        hours: Duration in hours
        reason: Reason for delegation

    Returns:
        Created delegation
    """
    return manager.delegate(
        delegator_id=delegator_id,
        delegatee_id=delegatee_id,
        permission=permission,
        duration=timedelta(hours=hours),
        reason=reason or f"Temporary delegation for {hours} hours"
    )


def create_limited_use_delegation(
    manager: DelegationManager,
    delegator_id: str,
    delegatee_id: str,
    permission: Permission,
    max_uses: int = 10,
    duration: Optional[timedelta] = None,
    reason: Optional[str] = None
) -> PermissionDelegation:
    """
    Convenience function to create a limited-use delegation.

    Args:
        manager: DelegationManager instance
        delegator_id: User delegating
        delegatee_id: User receiving delegation
        permission: Permission to delegate
        max_uses: Maximum number of uses
        duration: Optional time limit
        reason: Reason for delegation

    Returns:
        Created delegation
    """
    return manager.delegate(
        delegator_id=delegator_id,
        delegatee_id=delegatee_id,
        permission=permission,
        max_uses=max_uses,
        duration=duration,
        reason=reason or f"Limited to {max_uses} uses"
    )


__all__ = [
    'DelegationStatus',
    'DelegationType',
    'DelegationConstraint',
    'PermissionDelegation',
    'DelegationManager',
    'create_temporary_delegation',
    'create_limited_use_delegation'
]
