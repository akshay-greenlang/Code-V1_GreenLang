# -*- coding: utf-8 -*-
"""
Retention Policy Service - Centralized Audit Logging Service (SEC-005)

Manages retention policies for audit data based on event type, data
classification, and compliance requirements. Determines when audit records
should transition between storage tiers or be marked for archival.

Storage Tiers:
    - hot: 30 days in primary PostgreSQL tables
    - warm: 90 days in compressed PostgreSQL partitions
    - cold: 365 days in S3 Parquet files
    - archive: 7 years in S3 Glacier Deep Archive

Compliance Retention Overrides:
    - financial: 7 years (SOX, FINRA)
    - security: 3 years (SOC 2)
    - user_activity: 1 year (GDPR)
    - compliance_report: 7 years (regulatory)
    - system_config: 3 years (ISO 27001)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class RetentionTier(str, Enum):
    """Storage tier for audit data."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVE = "archive"


class DataClassification(str, Enum):
    """Data classification for compliance-based retention."""

    FINANCIAL = "financial"
    SECURITY = "security"
    USER_ACTIVITY = "user_activity"
    COMPLIANCE_REPORT = "compliance_report"
    SYSTEM_CONFIG = "system_config"
    OPERATIONAL = "operational"
    DEBUG = "debug"


# Retention periods in days for each tier
RETENTION_TIERS: Dict[str, int] = {
    RetentionTier.HOT.value: 30,
    RetentionTier.WARM.value: 90,
    RetentionTier.COLD.value: 365,
    RetentionTier.ARCHIVE.value: 2555,  # 7 years
}

# Compliance-mandated retention periods in days
COMPLIANCE_RETENTION: Dict[str, int] = {
    DataClassification.FINANCIAL.value: 2555,  # 7 years (SOX, FINRA)
    DataClassification.SECURITY.value: 1095,   # 3 years (SOC 2)
    DataClassification.USER_ACTIVITY.value: 365,  # 1 year (GDPR)
    DataClassification.COMPLIANCE_REPORT.value: 2555,  # 7 years
    DataClassification.SYSTEM_CONFIG.value: 1095,  # 3 years (ISO 27001)
    DataClassification.OPERATIONAL.value: 90,   # 90 days
    DataClassification.DEBUG.value: 30,         # 30 days
}

# Event type to data classification mapping
EVENT_TYPE_CLASSIFICATION: Dict[str, DataClassification] = {
    # Authentication events -> security
    "auth.login_success": DataClassification.SECURITY,
    "auth.login_failure": DataClassification.SECURITY,
    "auth.logout": DataClassification.SECURITY,
    "auth.mfa_verified": DataClassification.SECURITY,
    "auth.password_changed": DataClassification.SECURITY,
    "auth.token_revoked": DataClassification.SECURITY,
    "auth.account_locked": DataClassification.SECURITY,
    "auth.account_unlocked": DataClassification.SECURITY,

    # Authorization events -> security
    "rbac.permission_granted": DataClassification.SECURITY,
    "rbac.permission_denied": DataClassification.SECURITY,
    "rbac.role_assigned": DataClassification.SECURITY,
    "rbac.role_revoked": DataClassification.SECURITY,
    "rbac.role_created": DataClassification.SYSTEM_CONFIG,
    "rbac.role_updated": DataClassification.SYSTEM_CONFIG,
    "rbac.role_deleted": DataClassification.SYSTEM_CONFIG,

    # Data access events -> varies by data type
    "data.read": DataClassification.USER_ACTIVITY,
    "data.write": DataClassification.USER_ACTIVITY,
    "data.delete": DataClassification.USER_ACTIVITY,
    "data.export": DataClassification.COMPLIANCE_REPORT,

    # Financial events -> financial
    "billing.charge": DataClassification.FINANCIAL,
    "billing.refund": DataClassification.FINANCIAL,
    "billing.subscription_changed": DataClassification.FINANCIAL,
    "emissions.calculation": DataClassification.COMPLIANCE_REPORT,
    "emissions.report_generated": DataClassification.COMPLIANCE_REPORT,

    # Configuration events -> system_config
    "config.updated": DataClassification.SYSTEM_CONFIG,
    "tenant.created": DataClassification.SYSTEM_CONFIG,
    "tenant.updated": DataClassification.SYSTEM_CONFIG,
    "tenant.deleted": DataClassification.SYSTEM_CONFIG,
    "user.created": DataClassification.SECURITY,
    "user.updated": DataClassification.USER_ACTIVITY,
    "user.deleted": DataClassification.SECURITY,

    # Encryption events -> security
    "encryption.key_rotated": DataClassification.SECURITY,
    "encryption.decrypt": DataClassification.SECURITY,

    # API events -> operational
    "api.request": DataClassification.OPERATIONAL,

    # System events -> debug/operational
    "system.startup": DataClassification.OPERATIONAL,
    "system.shutdown": DataClassification.OPERATIONAL,
    "system.error": DataClassification.OPERATIONAL,
    "job.started": DataClassification.OPERATIONAL,
    "job.completed": DataClassification.OPERATIONAL,
    "job.failed": DataClassification.OPERATIONAL,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RetentionPolicy:
    """Retention policy for an audit event type.

    Attributes:
        event_type: The audit event type this policy applies to.
        data_class: Data classification for compliance.
        retention_days: Total retention period in days.
        hot_days: Days in hot storage (PostgreSQL).
        warm_days: Days in warm storage (compressed).
        cold_days: Days in cold storage (S3 Parquet).
        archive_days: Days in archive (Glacier).
        compliance_hold: Whether the data is under legal hold.
    """

    event_type: str
    data_class: DataClassification
    retention_days: int
    hot_days: int = 30
    warm_days: int = 60
    cold_days: int = 275
    archive_days: int = 2190
    compliance_hold: bool = False


@dataclass
class RetentionMarker:
    """Marker for events to be transitioned between tiers.

    Attributes:
        event_ids: List of event IDs to transition.
        source_tier: Current storage tier.
        target_tier: Destination storage tier.
        transition_date: When the transition should occur.
        reason: Reason for the transition.
    """

    event_ids: List[str]
    source_tier: RetentionTier
    target_tier: RetentionTier
    transition_date: datetime
    reason: str


# ---------------------------------------------------------------------------
# RetentionPolicyService
# ---------------------------------------------------------------------------


class RetentionPolicyService:
    """Manages retention policies and tier transitions for audit data.

    Determines retention periods based on:
    1. Event type classification
    2. Compliance requirements
    3. Tenant-specific overrides
    4. Legal holds

    Thread-safe: All methods are stateless or use async database operations.

    Example:
        >>> service = RetentionPolicyService(db_pool)
        >>> retention_days = service.get_retention_days("auth.login_success", "security")
        >>> print(f"Retain for {retention_days} days")
        Retain for 1095 days
        >>>
        >>> # Mark old events for archival
        >>> await service.mark_for_archival(event_ids, "tier_transition")
    """

    def __init__(
        self,
        db_pool: Any = None,
        custom_policies: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize retention policy service.

        Args:
            db_pool: Async database connection pool.
            custom_policies: Optional custom retention policies
                mapping event_type to retention_days.
        """
        self._db_pool = db_pool
        self._custom_policies = custom_policies or {}

        logger.info(
            "RetentionPolicyService initialized: %d custom policies",
            len(self._custom_policies),
        )

    def get_retention_days(
        self,
        event_type: str,
        data_class: Optional[str] = None,
    ) -> int:
        """Get the retention period in days for an event type.

        Resolution order:
        1. Custom policies (tenant overrides)
        2. Data classification compliance requirements
        3. Event type default classification
        4. Default operational retention

        Args:
            event_type: The audit event type.
            data_class: Optional explicit data classification.

        Returns:
            Retention period in days.

        Example:
            >>> service = RetentionPolicyService()
            >>> service.get_retention_days("auth.login_success")
            1095
            >>> service.get_retention_days("api.request")
            90
        """
        # Check custom policies first
        if event_type in self._custom_policies:
            return self._custom_policies[event_type]

        # Determine data classification
        if data_class is not None:
            classification = data_class
        else:
            # Look up classification from event type
            classification = EVENT_TYPE_CLASSIFICATION.get(
                event_type,
                DataClassification.OPERATIONAL,
            )
            if isinstance(classification, DataClassification):
                classification = classification.value

        # Get compliance retention for classification
        retention = COMPLIANCE_RETENTION.get(
            classification,
            COMPLIANCE_RETENTION[DataClassification.OPERATIONAL.value],
        )

        return retention

    def get_policy(self, event_type: str) -> RetentionPolicy:
        """Get the complete retention policy for an event type.

        Args:
            event_type: The audit event type.

        Returns:
            RetentionPolicy with tier breakdown.
        """
        # Determine classification
        data_class = EVENT_TYPE_CLASSIFICATION.get(
            event_type,
            DataClassification.OPERATIONAL,
        )

        # Get total retention
        total_days = self.get_retention_days(event_type)

        # Calculate tier breakdown
        hot_days = min(RETENTION_TIERS[RetentionTier.HOT.value], total_days)
        remaining = total_days - hot_days

        warm_days = min(
            RETENTION_TIERS[RetentionTier.WARM.value] - hot_days,
            remaining,
        )
        remaining -= warm_days

        cold_days = min(
            RETENTION_TIERS[RetentionTier.COLD.value] - hot_days - warm_days,
            remaining,
        )
        remaining -= cold_days

        archive_days = remaining

        return RetentionPolicy(
            event_type=event_type,
            data_class=data_class,
            retention_days=total_days,
            hot_days=hot_days,
            warm_days=warm_days,
            cold_days=cold_days,
            archive_days=archive_days,
        )

    def get_tier_for_age(self, age_days: int, event_type: str) -> RetentionTier:
        """Determine the storage tier based on event age.

        Args:
            age_days: Age of the event in days.
            event_type: The audit event type.

        Returns:
            Appropriate storage tier for the event age.
        """
        policy = self.get_policy(event_type)

        if age_days <= policy.hot_days:
            return RetentionTier.HOT
        elif age_days <= policy.hot_days + policy.warm_days:
            return RetentionTier.WARM
        elif age_days <= policy.hot_days + policy.warm_days + policy.cold_days:
            return RetentionTier.COLD
        else:
            return RetentionTier.ARCHIVE

    def should_expire(
        self,
        event_timestamp: datetime,
        event_type: str,
        reference_time: Optional[datetime] = None,
    ) -> bool:
        """Check if an event has exceeded its retention period.

        Args:
            event_timestamp: When the event was created.
            event_type: The audit event type.
            reference_time: Reference time for comparison (default: now).

        Returns:
            True if the event should be expired/deleted.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        retention_days = self.get_retention_days(event_type)
        expiry_date = event_timestamp + timedelta(days=retention_days)

        return reference_time > expiry_date

    def get_transition_candidates(
        self,
        event_type: str,
        current_tier: RetentionTier,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime]:
        """Get the date range for events that should transition to next tier.

        Args:
            event_type: The audit event type.
            current_tier: Current storage tier.
            reference_time: Reference time (default: now).

        Returns:
            Tuple of (start_date, end_date) for events to transition.
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        policy = self.get_policy(event_type)

        # Calculate tier boundaries
        if current_tier == RetentionTier.HOT:
            min_age = policy.hot_days
            max_age = policy.hot_days + policy.warm_days
        elif current_tier == RetentionTier.WARM:
            min_age = policy.hot_days + policy.warm_days
            max_age = policy.hot_days + policy.warm_days + policy.cold_days
        elif current_tier == RetentionTier.COLD:
            min_age = (
                policy.hot_days + policy.warm_days + policy.cold_days
            )
            max_age = policy.retention_days
        else:
            # Archive tier - no transition, return empty range
            return reference_time, reference_time

        start_date = reference_time - timedelta(days=max_age)
        end_date = reference_time - timedelta(days=min_age)

        return start_date, end_date

    async def mark_for_archival(
        self,
        event_ids: List[str],
        reason: str = "tier_transition",
    ) -> int:
        """Mark audit events for archival in the database.

        Updates the audit_log table to set archival metadata on the
        specified events. The archival service will pick these up
        during its next run.

        Args:
            event_ids: List of audit event IDs to mark.
            reason: Reason for archival (for audit trail).

        Returns:
            Number of events marked.

        Raises:
            RuntimeError: If database pool is not configured.
        """
        if not event_ids:
            return 0

        if self._db_pool is None:
            logger.warning(
                "Cannot mark events for archival: no database pool configured"
            )
            raise RuntimeError("Database pool not configured")

        now = datetime.now(timezone.utc)

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Update in batches to avoid parameter limits
                batch_size = 1000
                total_updated = 0

                for i in range(0, len(event_ids), batch_size):
                    batch = event_ids[i : i + batch_size]
                    placeholders = ", ".join(["%s"] * len(batch))

                    await cur.execute(
                        f"""
                        UPDATE audit.audit_log
                        SET
                            archival_status = 'pending',
                            archival_marked_at = %s,
                            archival_reason = %s
                        WHERE event_id IN ({placeholders})
                          AND archival_status IS NULL
                        """,
                        [now, reason, *batch],
                    )

                    total_updated += cur.rowcount

                await conn.commit()

        logger.info(
            "Marked %d events for archival: reason=%s",
            total_updated,
            reason,
        )
        return total_updated

    async def get_pending_archival_count(self) -> int:
        """Get the count of events pending archival.

        Returns:
            Number of events marked for archival but not yet processed.
        """
        if self._db_pool is None:
            return 0

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM audit.audit_log
                    WHERE archival_status = 'pending'
                    """
                )
                row = await cur.fetchone()
                return row[0] if row else 0

    async def apply_legal_hold(
        self,
        event_ids: List[str],
        hold_reason: str,
        hold_until: Optional[datetime] = None,
    ) -> int:
        """Apply legal hold to prevent archival/deletion.

        Args:
            event_ids: List of event IDs to hold.
            hold_reason: Legal reason for the hold.
            hold_until: Optional end date for the hold.

        Returns:
            Number of events placed under hold.
        """
        if not event_ids or self._db_pool is None:
            return 0

        now = datetime.now(timezone.utc)

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(event_ids))

                await cur.execute(
                    f"""
                    UPDATE audit.audit_log
                    SET
                        legal_hold = true,
                        legal_hold_reason = %s,
                        legal_hold_applied_at = %s,
                        legal_hold_until = %s
                    WHERE event_id IN ({placeholders})
                    """,
                    [hold_reason, now, hold_until, *event_ids],
                )

                await conn.commit()

                logger.warning(
                    "Legal hold applied to %d events: reason=%s",
                    cur.rowcount,
                    hold_reason,
                )
                return cur.rowcount

    async def release_legal_hold(
        self,
        event_ids: List[str],
        release_reason: str,
    ) -> int:
        """Release legal hold from events.

        Args:
            event_ids: List of event IDs to release.
            release_reason: Reason for releasing the hold.

        Returns:
            Number of events released from hold.
        """
        if not event_ids or self._db_pool is None:
            return 0

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * len(event_ids))

                await cur.execute(
                    f"""
                    UPDATE audit.audit_log
                    SET
                        legal_hold = false,
                        legal_hold_released_at = %s,
                        legal_hold_release_reason = %s
                    WHERE event_id IN ({placeholders})
                      AND legal_hold = true
                    """,
                    [datetime.now(timezone.utc), release_reason, *event_ids],
                )

                await conn.commit()

                logger.info(
                    "Legal hold released from %d events: reason=%s",
                    cur.rowcount,
                    release_reason,
                )
                return cur.rowcount


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Classes
    "RetentionPolicyService",
    "RetentionPolicy",
    "RetentionMarker",
    # Enums
    "RetentionTier",
    "DataClassification",
    # Constants
    "RETENTION_TIERS",
    "COMPLIANCE_RETENTION",
    "EVENT_TYPE_CLASSIFICATION",
]
