# -*- coding: utf-8 -*-
"""
GDPR Data Retention Enforcer - SEC-010 Phase 5

Enforces data retention policies in compliance with GDPR Article 5(1)(e)
(storage limitation principle). Automatically identifies data that has
exceeded its retention period and schedules deletion or anonymization.

Key Features:
- Policy-based retention management
- Automated deletion scheduling
- Anonymization support
- Retention exception handling
- Compliance reporting

Classes:
    - RetentionEnforcer: Main retention enforcement engine.
    - DeletionJob: Scheduled deletion job.
    - AnonymizationJob: Scheduled anonymization job.
    - RetentionReport: Report of retention enforcement actions.

Example:
    >>> enforcer = RetentionEnforcer()
    >>> await enforcer.apply_retention()
    >>> report = await enforcer.generate_report()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.compliance_automation.models import (
    DataCategory,
    RetentionPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job Models
# ---------------------------------------------------------------------------


class DeletionJob(BaseModel):
    """Scheduled data deletion job.

    Attributes:
        id: Unique job identifier.
        source_system: System where data will be deleted.
        source_location: Specific location (table, bucket, etc.).
        criteria: Criteria for selecting records to delete.
        scheduled_at: When the job is scheduled to run.
        executed_at: When the job was executed.
        records_deleted: Number of records deleted.
        status: Job status (pending, running, completed, failed).
        error: Error message if failed.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_system: str
    source_location: str
    criteria: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    executed_at: Optional[datetime] = None
    records_deleted: int = 0
    status: str = "pending"
    error: Optional[str] = None


class AnonymizationJob(BaseModel):
    """Scheduled data anonymization job.

    Attributes:
        id: Unique job identifier.
        source_system: System where data will be anonymized.
        source_location: Specific location.
        fields_to_anonymize: Fields to anonymize.
        anonymization_method: Method to use (hash, mask, generalize).
        scheduled_at: When the job is scheduled.
        executed_at: When the job was executed.
        records_anonymized: Number of records anonymized.
        status: Job status.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_system: str
    source_location: str
    fields_to_anonymize: List[str] = Field(default_factory=list)
    anonymization_method: str = "hash"
    scheduled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    executed_at: Optional[datetime] = None
    records_anonymized: int = 0
    status: str = "pending"


class RetentionReport(BaseModel):
    """Report of retention enforcement actions.

    Attributes:
        id: Unique report identifier.
        generated_at: When the report was generated.
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        policies_evaluated: Number of policies evaluated.
        deletion_jobs: List of deletion jobs executed.
        anonymization_jobs: List of anonymization jobs executed.
        total_records_deleted: Total records deleted.
        total_records_anonymized: Total records anonymized.
        exceptions_logged: Number of exceptions/errors.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    policies_evaluated: int = 0
    deletion_jobs: List[DeletionJob] = Field(default_factory=list)
    anonymization_jobs: List[AnonymizationJob] = Field(default_factory=list)
    total_records_deleted: int = 0
    total_records_anonymized: int = 0
    exceptions_logged: int = 0


# ---------------------------------------------------------------------------
# Retention Enforcer
# ---------------------------------------------------------------------------


class RetentionEnforcer:
    """Enforces data retention policies across GreenLang systems.

    Implements GDPR Article 5(1)(e) storage limitation by automatically
    identifying and deleting or anonymizing data that has exceeded its
    defined retention period.

    Attributes:
        policies: List of active retention policies.
        deletion_jobs: Queue of pending deletion jobs.
        anonymization_jobs: Queue of pending anonymization jobs.

    Example:
        >>> enforcer = RetentionEnforcer()
        >>> enforcer.add_policy(RetentionPolicy(
        ...     name="User PII Retention",
        ...     data_category=DataCategory.PII,
        ...     retention_days=365,
        ...     action="delete",
        ... ))
        >>> await enforcer.apply_retention()
    """

    # Default retention policies by data category
    DEFAULT_RETENTION_DAYS = {
        DataCategory.PII: 365,
        DataCategory.SENSITIVE_PII: 365,
        DataCategory.FINANCIAL: 2555,  # 7 years
        DataCategory.OPERATIONAL: 90,
        DataCategory.AUDIT: 2555,  # 7 years
        DataCategory.SECURITY: 365,
        DataCategory.CONSENT: 2555,  # 7 years (proof of consent)
        DataCategory.BACKUP: 30,
        DataCategory.EMISSIONS: 2555,  # 7 years (regulatory)
        DataCategory.SUPPLY_CHAIN: 365,
    }

    # Data sources with retention support
    DATA_SOURCES = [
        {
            "system": "postgresql",
            "location": "security.user_sessions",
            "category": DataCategory.OPERATIONAL,
            "date_column": "created_at",
        },
        {
            "system": "postgresql",
            "location": "security.audit_logs",
            "category": DataCategory.AUDIT,
            "date_column": "created_at",
        },
        {
            "system": "s3",
            "location": "greenlang-exports",
            "category": DataCategory.OPERATIONAL,
            "date_field": "LastModified",
        },
        {
            "system": "loki",
            "location": "application",
            "category": DataCategory.OPERATIONAL,
            "retention_managed_by": "loki",
        },
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the retention enforcer.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self.policies: Dict[str, RetentionPolicy] = {}
        self.deletion_jobs: List[DeletionJob] = []
        self.anonymization_jobs: List[AnonymizationJob] = []
        self._last_enforcement: Optional[datetime] = None

        # Initialize default policies
        self._initialize_default_policies()

        logger.info("Initialized RetentionEnforcer with %d policies", len(self.policies))

    def add_policy(self, policy: RetentionPolicy) -> None:
        """Add a retention policy.

        Args:
            policy: The retention policy to add.
        """
        self.policies[policy.id] = policy
        logger.info(
            "Added retention policy: %s (%s days, action=%s)",
            policy.name,
            policy.retention_days,
            policy.action,
        )

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a retention policy.

        Args:
            policy_id: The policy ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False

    def get_policy(self, policy_id: str) -> Optional[RetentionPolicy]:
        """Get a retention policy by ID.

        Args:
            policy_id: The policy ID.

        Returns:
            The RetentionPolicy or None.
        """
        return self.policies.get(policy_id)

    async def apply_retention(self) -> RetentionReport:
        """Apply all retention policies.

        Scans data sources, identifies data exceeding retention periods,
        and schedules deletion or anonymization jobs.

        Returns:
            RetentionReport summarizing actions taken.
        """
        logger.info("Applying retention policies")

        report = RetentionReport(
            period_start=self._last_enforcement,
            period_end=datetime.now(timezone.utc),
        )

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            try:
                await self._apply_policy(policy, report)
                report.policies_evaluated += 1
            except Exception as e:
                logger.error("Error applying policy %s: %s", policy.name, str(e))
                report.exceptions_logged += 1

        # Execute pending jobs
        await self._execute_deletion_jobs(report)
        await self._execute_anonymization_jobs(report)

        # Update last enforcement timestamp
        self._last_enforcement = datetime.now(timezone.utc)

        logger.info(
            "Retention enforcement complete: %d records deleted, %d anonymized",
            report.total_records_deleted,
            report.total_records_anonymized,
        )

        return report

    async def schedule_deletion(
        self,
        source_system: str,
        source_location: str,
        criteria: Dict[str, Any],
        scheduled_at: Optional[datetime] = None,
    ) -> DeletionJob:
        """Schedule a deletion job.

        Args:
            source_system: System where data will be deleted.
            source_location: Specific location.
            criteria: Criteria for selecting records.
            scheduled_at: When to execute (default: now).

        Returns:
            The scheduled DeletionJob.
        """
        job = DeletionJob(
            source_system=source_system,
            source_location=source_location,
            criteria=criteria,
            scheduled_at=scheduled_at or datetime.now(timezone.utc),
        )

        self.deletion_jobs.append(job)

        logger.info(
            "Scheduled deletion job %s: %s.%s",
            job.id[:8],
            source_system,
            source_location,
        )

        return job

    async def schedule_anonymization(
        self,
        source_system: str,
        source_location: str,
        fields: List[str],
        method: str = "hash",
        scheduled_at: Optional[datetime] = None,
    ) -> AnonymizationJob:
        """Schedule an anonymization job.

        Args:
            source_system: System where data will be anonymized.
            source_location: Specific location.
            fields: Fields to anonymize.
            method: Anonymization method (hash, mask, generalize).
            scheduled_at: When to execute.

        Returns:
            The scheduled AnonymizationJob.
        """
        job = AnonymizationJob(
            source_system=source_system,
            source_location=source_location,
            fields_to_anonymize=fields,
            anonymization_method=method,
            scheduled_at=scheduled_at or datetime.now(timezone.utc),
        )

        self.anonymization_jobs.append(job)

        logger.info(
            "Scheduled anonymization job %s: %s.%s (fields: %s)",
            job.id[:8],
            source_system,
            source_location,
            fields,
        )

        return job

    async def get_expired_data_summary(self) -> Dict[str, Any]:
        """Get a summary of data that has exceeded retention.

        Returns:
            Summary of expired data by system and category.
        """
        summary: Dict[str, Any] = {
            "total_expired_records": 0,
            "by_system": {},
            "by_category": {},
            "policies_exceeded": [],
        }

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            expired_count = await self._count_expired_records(policy)

            if expired_count > 0:
                summary["total_expired_records"] += expired_count
                summary["policies_exceeded"].append({
                    "policy_id": policy.id,
                    "policy_name": policy.name,
                    "category": policy.data_category.value,
                    "retention_days": policy.retention_days,
                    "expired_records": expired_count,
                })

                category = policy.data_category.value
                summary["by_category"][category] = (
                    summary["by_category"].get(category, 0) + expired_count
                )

        return summary

    async def generate_report(
        self,
        period_days: int = 30,
    ) -> RetentionReport:
        """Generate a retention compliance report.

        Args:
            period_days: Number of days to include in report.

        Returns:
            RetentionReport for the period.
        """
        logger.info("Generating retention report for last %d days", period_days)

        period_start = datetime.now(timezone.utc) - timedelta(days=period_days)

        report = RetentionReport(
            period_start=period_start,
            period_end=datetime.now(timezone.utc),
            policies_evaluated=len([p for p in self.policies.values() if p.enabled]),
        )

        # Filter jobs within period
        report.deletion_jobs = [
            j for j in self.deletion_jobs
            if j.executed_at and j.executed_at >= period_start
        ]
        report.anonymization_jobs = [
            j for j in self.anonymization_jobs
            if j.executed_at and j.executed_at >= period_start
        ]

        # Calculate totals
        report.total_records_deleted = sum(
            j.records_deleted for j in report.deletion_jobs
        )
        report.total_records_anonymized = sum(
            j.records_anonymized for j in report.anonymization_jobs
        )

        return report

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _initialize_default_policies(self) -> None:
        """Initialize default retention policies."""
        for category, days in self.DEFAULT_RETENTION_DAYS.items():
            policy = RetentionPolicy(
                name=f"Default {category.value} Retention",
                data_category=category,
                retention_days=days,
                action="delete" if category == DataCategory.BACKUP else "archive",
                enabled=True,
            )
            self.policies[policy.id] = policy

    async def _apply_policy(
        self,
        policy: RetentionPolicy,
        report: RetentionReport,
    ) -> None:
        """Apply a single retention policy."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy.retention_days)

        # Find applicable data sources
        for source in self.DATA_SOURCES:
            if source.get("category") != policy.data_category:
                continue

            # Check if retention is managed externally
            if source.get("retention_managed_by"):
                logger.debug(
                    "Retention for %s managed by %s",
                    source["location"],
                    source["retention_managed_by"],
                )
                continue

            # Schedule appropriate job based on action
            if policy.action == "delete":
                job = await self.schedule_deletion(
                    source_system=source["system"],
                    source_location=source["location"],
                    criteria={
                        "date_column": source.get("date_column", "created_at"),
                        "cutoff_date": cutoff_date.isoformat(),
                        "category": policy.data_category.value,
                    },
                )
                report.deletion_jobs.append(job)

            elif policy.action == "anonymize":
                job = await self.schedule_anonymization(
                    source_system=source["system"],
                    source_location=source["location"],
                    fields=self._get_pii_fields_for_location(source["location"]),
                    method="hash",
                )
                report.anonymization_jobs.append(job)

            elif policy.action == "archive":
                # Archive to cold storage before deletion
                await self._archive_data(
                    source["system"],
                    source["location"],
                    cutoff_date,
                )
                job = await self.schedule_deletion(
                    source_system=source["system"],
                    source_location=source["location"],
                    criteria={
                        "date_column": source.get("date_column", "created_at"),
                        "cutoff_date": cutoff_date.isoformat(),
                        "archived": True,
                    },
                )
                report.deletion_jobs.append(job)

    async def _execute_deletion_jobs(self, report: RetentionReport) -> None:
        """Execute pending deletion jobs."""
        pending_jobs = [j for j in self.deletion_jobs if j.status == "pending"]

        for job in pending_jobs:
            try:
                job.status = "running"
                deleted = await self._delete_data(
                    job.source_system,
                    job.source_location,
                    job.criteria,
                )
                job.records_deleted = deleted
                job.executed_at = datetime.now(timezone.utc)
                job.status = "completed"
                report.total_records_deleted += deleted

                logger.info(
                    "Deletion job %s completed: %d records deleted from %s.%s",
                    job.id[:8],
                    deleted,
                    job.source_system,
                    job.source_location,
                )

            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                report.exceptions_logged += 1
                logger.error("Deletion job %s failed: %s", job.id[:8], str(e))

    async def _execute_anonymization_jobs(self, report: RetentionReport) -> None:
        """Execute pending anonymization jobs."""
        pending_jobs = [j for j in self.anonymization_jobs if j.status == "pending"]

        for job in pending_jobs:
            try:
                job.status = "running"
                anonymized = await self._anonymize_data(
                    job.source_system,
                    job.source_location,
                    job.fields_to_anonymize,
                    job.anonymization_method,
                )
                job.records_anonymized = anonymized
                job.executed_at = datetime.now(timezone.utc)
                job.status = "completed"
                report.total_records_anonymized += anonymized

                logger.info(
                    "Anonymization job %s completed: %d records anonymized",
                    job.id[:8],
                    anonymized,
                )

            except Exception as e:
                job.status = "failed"
                report.exceptions_logged += 1
                logger.error("Anonymization job %s failed: %s", job.id[:8], str(e))

    async def _delete_data(
        self,
        system: str,
        location: str,
        criteria: Dict[str, Any],
    ) -> int:
        """Execute data deletion.

        In production, this would perform actual deletions.
        """
        logger.debug("Deleting data from %s.%s with criteria: %s", system, location, criteria)
        # Placeholder - return simulated count
        return 100

    async def _anonymize_data(
        self,
        system: str,
        location: str,
        fields: List[str],
        method: str,
    ) -> int:
        """Execute data anonymization.

        In production, this would perform actual anonymization.
        """
        logger.debug(
            "Anonymizing fields %s in %s.%s using %s",
            fields, system, location, method,
        )
        # Placeholder - return simulated count
        return 50

    async def _archive_data(
        self,
        system: str,
        location: str,
        cutoff_date: datetime,
    ) -> None:
        """Archive data before deletion.

        In production, this would copy data to cold storage.
        """
        logger.debug("Archiving data from %s.%s before %s", system, location, cutoff_date)

    async def _count_expired_records(self, policy: RetentionPolicy) -> int:
        """Count records that have exceeded retention.

        In production, this would query actual systems.
        """
        # Placeholder - return simulated count
        return 500

    def _get_pii_fields_for_location(self, location: str) -> List[str]:
        """Get PII fields for a data location."""
        pii_map = {
            "security.users": ["email", "name", "phone"],
            "security.user_sessions": ["ip_address", "user_agent"],
            "security.audit_logs": ["ip_address", "user_agent"],
        }
        return pii_map.get(location, ["email"])


__all__ = [
    "RetentionEnforcer",
    "DeletionJob",
    "AnonymizationJob",
    "RetentionReport",
]
