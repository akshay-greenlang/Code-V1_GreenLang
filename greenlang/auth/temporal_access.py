# -*- coding: utf-8 -*-
"""
Time-Based Access Controls for GreenLang

This module provides temporal access control with scheduled permissions,
recurring access windows, and automatic expiration.

Temporal Access Examples:
    - Grant access every weekday from 9 AM to 5 PM
    - Allow weekend access for maintenance
    - Temporary access for 30 days
    - Access during specific date ranges

Features:
    - Scheduled permissions (start/end datetime)
    - Recurring access windows (daily, weekly, custom)
    - Automatic expiration and cleanup
    - Timezone support
    - Business hours enforcement

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, date
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.auth.permissions import Permission
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Temporal Access Models
# ==============================================================================

class RecurrenceType(str, Enum):
    """Type of recurrence pattern."""
    ONCE = "once"  # One-time access
    DAILY = "daily"  # Every day
    WEEKLY = "weekly"  # Specific days of week
    MONTHLY = "monthly"  # Specific days of month
    CUSTOM = "custom"  # Custom cron-like pattern


class DayOfWeek(int, Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class TimeWindow(BaseModel):
    """Defines a time window when access is allowed."""

    start_time: time = Field(..., description="Start time (HH:MM:SS)")
    end_time: time = Field(..., description="End time (HH:MM:SS)")

    def is_active_at(self, check_time: time) -> bool:
        """
        Check if a given time falls within this window.

        Args:
            check_time: Time to check

        Returns:
            True if within window
        """
        if self.start_time <= self.end_time:
            # Normal case: 09:00 - 17:00
            return self.start_time <= check_time <= self.end_time
        else:
            # Crosses midnight: 22:00 - 02:00
            return check_time >= self.start_time or check_time <= self.end_time

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TimeWindow':
        """Create from dictionary."""
        return cls(
            start_time=time.fromisoformat(data['start_time']),
            end_time=time.fromisoformat(data['end_time'])
        )


class RecurrencePattern(BaseModel):
    """Defines when access recurs."""

    recurrence_type: RecurrenceType = Field(..., description="Type of recurrence")

    # For weekly recurrence
    days_of_week: List[int] = Field(
        default_factory=list,
        description="Days of week (0=Monday, 6=Sunday)"
    )

    # For monthly recurrence
    days_of_month: List[int] = Field(
        default_factory=list,
        description="Days of month (1-31)"
    )

    # For custom patterns
    cron_expression: Optional[str] = Field(
        None,
        description="Cron expression for custom patterns"
    )

    def is_active_on(self, check_date: date) -> bool:
        """
        Check if access is active on a given date.

        Args:
            check_date: Date to check

        Returns:
            True if active on this date
        """
        if self.recurrence_type == RecurrenceType.ONCE:
            return True

        elif self.recurrence_type == RecurrenceType.DAILY:
            return True

        elif self.recurrence_type == RecurrenceType.WEEKLY:
            if not self.days_of_week:
                return True
            return check_date.weekday() in self.days_of_week

        elif self.recurrence_type == RecurrenceType.MONTHLY:
            if not self.days_of_month:
                return True
            return check_date.day in self.days_of_month

        elif self.recurrence_type == RecurrenceType.CUSTOM:
            # Placeholder for cron evaluation
            # In production, use croniter or similar library
            logger.warning("Custom cron patterns not yet implemented")
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'recurrence_type': self.recurrence_type.value,
            'days_of_week': self.days_of_week,
            'days_of_month': self.days_of_month,
            'cron_expression': self.cron_expression
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecurrencePattern':
        """Create from dictionary."""
        if 'recurrence_type' in data and isinstance(data['recurrence_type'], str):
            data['recurrence_type'] = RecurrenceType(data['recurrence_type'])
        return cls(**data)


class TemporalPermission(BaseModel):
    """
    Permission with temporal constraints.

    Defines when a permission is active based on dates, times, and recurrence.
    """

    temporal_id: str = Field(
        default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        description="Unique temporal permission identifier"
    )

    permission: Permission = Field(..., description="The underlying permission")

    # Date range
    valid_from: Optional[datetime] = Field(
        None,
        description="When permission becomes valid"
    )
    valid_until: Optional[datetime] = Field(
        None,
        description="When permission expires"
    )

    # Time windows
    time_windows: List[TimeWindow] = Field(
        default_factory=list,
        description="Time windows when permission is active"
    )

    # Recurrence
    recurrence: Optional[RecurrencePattern] = Field(
        None,
        description="Recurrence pattern for the permission"
    )

    # Metadata
    timezone: str = Field(
        default="UTC",
        description="Timezone for temporal evaluation"
    )
    description: str = Field(
        default="",
        description="Description of temporal permission"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            time: lambda v: v.isoformat(),
            RecurrenceType: lambda v: v.value
        }

    def is_active(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if permission is currently active.

        Args:
            check_time: Time to check (defaults to now)

        Returns:
            True if permission is active at the given time
        """
        check_time = check_time or DeterministicClock.utcnow()

        # Check date range
        if self.valid_from and check_time < self.valid_from:
            return False

        if self.valid_until and check_time > self.valid_until:
            return False

        # Check recurrence pattern
        if self.recurrence:
            if not self.recurrence.is_active_on(check_time.date()):
                return False

        # Check time windows
        if self.time_windows:
            current_time = check_time.time()
            is_in_window = any(
                window.is_active_at(current_time)
                for window in self.time_windows
            )
            if not is_in_window:
                return False

        return True

    def is_expired(self) -> bool:
        """Check if permission has permanently expired."""
        if self.valid_until:
            return DeterministicClock.utcnow() > self.valid_until
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'temporal_id': self.temporal_id,
            'permission': self.permission.to_dict(),
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'time_windows': [w.to_dict() for w in self.time_windows],
            'recurrence': self.recurrence.to_dict() if self.recurrence else None,
            'timezone': self.timezone,
            'description': self.description,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalPermission':
        """Create from dictionary."""
        if 'permission' in data and isinstance(data['permission'], dict):
            data['permission'] = Permission.from_dict(data['permission'])

        if 'time_windows' in data and data['time_windows']:
            data['time_windows'] = [
                TimeWindow.from_dict(w) if isinstance(w, dict) else w
                for w in data['time_windows']
            ]

        if 'recurrence' in data and data['recurrence']:
            if isinstance(data['recurrence'], dict):
                data['recurrence'] = RecurrencePattern.from_dict(data['recurrence'])

        # Parse datetime fields
        for field in ['valid_from', 'valid_until', 'created_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)


# ==============================================================================
# Temporal Access Manager
# ==============================================================================

class TemporalAccessManager:
    """
    Manages time-based access controls.

    Provides:
    - Creating scheduled permissions
    - Evaluating temporal permissions
    - Automatic expiration cleanup
    - Recurring access patterns
    """

    def __init__(self, cleanup_interval_minutes: int = 60):
        """
        Initialize temporal access manager.

        Args:
            cleanup_interval_minutes: How often to clean up expired permissions
        """
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self._temporal_permissions: Dict[str, TemporalPermission] = {}
        self._user_permissions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> temporal_ids
        self._last_cleanup: datetime = DeterministicClock.utcnow()

        self._stats = {
            'total_permissions': 0,
            'active_permissions': 0,
            'expired_permissions': 0,
            'cleanups_performed': 0
        }

        logger.info(f"Initialized TemporalAccessManager (cleanup_interval={cleanup_interval_minutes}m)")

    def create_temporal_permission(
        self,
        user_id: str,
        permission: Permission,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        time_windows: Optional[List[TimeWindow]] = None,
        recurrence: Optional[RecurrencePattern] = None,
        timezone: str = "UTC",
        description: str = "",
        created_by: Optional[str] = None
    ) -> TemporalPermission:
        """
        Create a temporal permission.

        Args:
            user_id: User to grant permission to
            permission: The permission to grant
            valid_from: When permission becomes valid
            valid_until: When permission expires
            time_windows: Time windows when permission is active
            recurrence: Recurrence pattern
            timezone: Timezone for evaluation
            description: Description
            created_by: Who created this permission

        Returns:
            Created temporal permission
        """
        temporal_perm = TemporalPermission(
            permission=permission,
            valid_from=valid_from,
            valid_until=valid_until,
            time_windows=time_windows or [],
            recurrence=recurrence,
            timezone=timezone,
            description=description,
            created_by=created_by
        )

        self._temporal_permissions[temporal_perm.temporal_id] = temporal_perm
        self._user_permissions[user_id].add(temporal_perm.temporal_id)

        self._stats['total_permissions'] += 1
        self._stats['active_permissions'] += 1

        logger.info(f"Created temporal permission {temporal_perm.temporal_id} for user {user_id}")
        return temporal_perm

    def get_temporal_permission(self, temporal_id: str) -> Optional[TemporalPermission]:
        """Get temporal permission by ID."""
        return self._temporal_permissions.get(temporal_id)

    def revoke_temporal_permission(self, temporal_id: str) -> bool:
        """
        Revoke a temporal permission.

        Args:
            temporal_id: ID of permission to revoke

        Returns:
            True if revoked successfully
        """
        if temporal_id not in self._temporal_permissions:
            return False

        # Remove from indices
        for user_perms in self._user_permissions.values():
            user_perms.discard(temporal_id)

        # Remove permission
        del self._temporal_permissions[temporal_id]

        self._stats['active_permissions'] -= 1

        logger.info(f"Revoked temporal permission {temporal_id}")
        return True

    def get_user_temporal_permissions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[TemporalPermission]:
        """
        Get all temporal permissions for a user.

        Args:
            user_id: User ID
            active_only: Only return currently active permissions

        Returns:
            List of temporal permissions
        """
        temporal_ids = self._user_permissions.get(user_id, set())
        permissions = [
            self._temporal_permissions[tid]
            for tid in temporal_ids
            if tid in self._temporal_permissions
        ]

        if active_only:
            permissions = [p for p in permissions if p.is_active()]

        return permissions

    def get_active_permissions(
        self,
        user_id: str,
        check_time: Optional[datetime] = None
    ) -> List[Permission]:
        """
        Get all currently active permissions for a user.

        Args:
            user_id: User ID
            check_time: Time to check (defaults to now)

        Returns:
            List of active permissions
        """
        temporal_perms = self.get_user_temporal_permissions(user_id, active_only=False)
        check_time = check_time or DeterministicClock.utcnow()

        active = []
        for temp_perm in temporal_perms:
            if temp_perm.is_active(check_time):
                active.append(temp_perm.permission)

        return active

    def cleanup_expired(self) -> int:
        """
        Clean up expired temporal permissions.

        Returns:
            Number of permissions cleaned up
        """
        count = 0

        for temporal_id, temp_perm in list(self._temporal_permissions.items()):
            if temp_perm.is_expired():
                # Remove from user indices
                for user_perms in self._user_permissions.values():
                    user_perms.discard(temporal_id)

                # Remove permission
                del self._temporal_permissions[temporal_id]

                self._stats['active_permissions'] -= 1
                self._stats['expired_permissions'] += 1
                count += 1

        if count > 0:
            self._stats['cleanups_performed'] += 1
            logger.info(f"Cleaned up {count} expired temporal permissions")

        self._last_cleanup = DeterministicClock.utcnow()
        return count

    def auto_cleanup_if_needed(self):
        """Perform cleanup if interval has passed."""
        elapsed = (DeterministicClock.utcnow() - self._last_cleanup).total_seconds() / 60

        if elapsed >= self.cleanup_interval_minutes:
            self.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        """Get temporal access statistics."""
        return self._stats.copy()

    def list_temporal_permissions(
        self,
        user_id: Optional[str] = None,
        active_only: bool = False
    ) -> List[TemporalPermission]:
        """
        List temporal permissions matching criteria.

        Args:
            user_id: Filter by user ID
            active_only: Only include currently active permissions

        Returns:
            List of matching temporal permissions
        """
        if user_id:
            return self.get_user_temporal_permissions(user_id, active_only)

        permissions = list(self._temporal_permissions.values())

        if active_only:
            permissions = [p for p in permissions if p.is_active()]

        return permissions


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_business_hours_permission(
    manager: TemporalAccessManager,
    user_id: str,
    permission: Permission,
    start_hour: int = 9,
    end_hour: int = 17,
    weekdays_only: bool = True,
    valid_until: Optional[datetime] = None
) -> TemporalPermission:
    """
    Create permission that's only active during business hours.

    Args:
        manager: TemporalAccessManager instance
        user_id: User to grant permission to
        permission: Permission to grant
        start_hour: Business hours start (default 9 AM)
        end_hour: Business hours end (default 5 PM)
        weekdays_only: Only allow on weekdays (default True)
        valid_until: Optional expiration date

    Returns:
        Created temporal permission
    """
    time_windows = [
        TimeWindow(
            start_time=time(hour=start_hour, minute=0),
            end_time=time(hour=end_hour, minute=0)
        )
    ]

    recurrence = None
    if weekdays_only:
        recurrence = RecurrencePattern(
            recurrence_type=RecurrenceType.WEEKLY,
            days_of_week=[0, 1, 2, 3, 4]  # Monday-Friday
        )

    return manager.create_temporal_permission(
        user_id=user_id,
        permission=permission,
        time_windows=time_windows,
        recurrence=recurrence,
        valid_until=valid_until,
        description=f"Business hours access ({start_hour}:00-{end_hour}:00)"
    )

def create_weekend_permission(
    manager: TemporalAccessManager,
    user_id: str,
    permission: Permission,
    valid_until: Optional[datetime] = None
) -> TemporalPermission:
    """
    Create permission that's only active on weekends.

    Args:
        manager: TemporalAccessManager instance
        user_id: User to grant permission to
        permission: Permission to grant
        valid_until: Optional expiration date

    Returns:
        Created temporal permission
    """
    recurrence = RecurrencePattern(
        recurrence_type=RecurrenceType.WEEKLY,
        days_of_week=[5, 6]  # Saturday, Sunday
    )

    return manager.create_temporal_permission(
        user_id=user_id,
        permission=permission,
        recurrence=recurrence,
        valid_until=valid_until,
        description="Weekend access only"
    )


def create_temporary_permission(
    manager: TemporalAccessManager,
    user_id: str,
    permission: Permission,
    days: int = 30
) -> TemporalPermission:
    """
    Create temporary permission that expires after specified days.

    Args:
        manager: TemporalAccessManager instance
        user_id: User to grant permission to
        permission: Permission to grant
        days: Number of days until expiration

    Returns:
        Created temporal permission
    """
    now = DeterministicClock.utcnow()
    expiry = now + timedelta(days=days)

    return manager.create_temporal_permission(
        user_id=user_id,
        permission=permission,
        valid_from=now,
        valid_until=expiry,
        description=f"Temporary access for {days} days"
    )


__all__ = [
    'RecurrenceType',
    'DayOfWeek',
    'TimeWindow',
    'RecurrencePattern',
    'TemporalPermission',
    'TemporalAccessManager',
    'create_business_hours_permission',
    'create_weekend_permission',
    'create_temporary_permission'
]
