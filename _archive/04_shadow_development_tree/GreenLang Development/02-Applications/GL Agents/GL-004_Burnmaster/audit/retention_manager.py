"""
AuditRetentionManager - Data retention and lifecycle management for BURNMASTER.

This module implements the AuditRetentionManager for GL-004 BURNMASTER,
handling retention policies, archival, restoration, and purging of audit
records in compliance with regulatory requirements.

Example:
    >>> manager = AuditRetentionManager(config)
    >>> manager.define_retention_policy(policy)
    >>> result = manager.archive_old_records(before_date)
    >>> status = manager.validate_retention_compliance()
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib
import json
import logging
import uuid
import gzip
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class RecordCategory(str, Enum):
    """Categories of records with different retention requirements."""
    SAFETY = "safety"
    EMISSIONS = "emissions"
    OPERATIONAL = "operational"
    OPTIMIZATION = "optimization"
    COMPLIANCE = "compliance"
    MAINTENANCE = "maintenance"
    AUDIT_LOG = "audit_log"
    EVIDENCE = "evidence"


class RetentionAction(str, Enum):
    """Actions to take when retention period expires."""
    ARCHIVE = "archive"
    DELETE = "delete"
    ANONYMIZE = "anonymize"
    RETAIN_INDEFINITELY = "retain_indefinitely"


class ComplianceStatus(str, Enum):
    """Compliance status for retention policies."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"


class ArchiveStatus(str, Enum):
    """Status of archive operations."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    RESTORED = "restored"
    PURGED = "purged"


# =============================================================================
# Policy Models
# =============================================================================

class RetentionRule(BaseModel):
    """Individual retention rule for a record category."""

    rule_id: str = Field(..., description="Unique rule identifier")
    category: RecordCategory = Field(..., description="Record category")
    retention_days: int = Field(..., ge=1, description="Retention period in days")
    action_on_expiry: RetentionAction = Field(..., description="Action when expired")
    regulatory_basis: str = Field(..., description="Regulatory requirement basis")
    priority: int = Field(1, ge=1, le=10, description="Rule priority (1=highest)")
    enabled: bool = Field(True, description="Whether rule is active")


class RetentionPolicy(BaseModel):
    """Complete retention policy definition."""

    policy_id: str = Field(..., description="Unique policy identifier")
    policy_name: str = Field(..., description="Policy name")
    version: str = Field(..., description="Policy version")
    effective_date: datetime = Field(..., description="When policy becomes effective")
    rules: List[RetentionRule] = Field(..., description="Retention rules")

    # Default settings
    default_retention_days: int = Field(2555, ge=365, description="Default retention (7 years)")
    default_action: RetentionAction = Field(RetentionAction.ARCHIVE, description="Default action")

    # Compliance settings
    regulatory_framework: str = Field("EPA", description="Regulatory framework")
    minimum_retention_days: int = Field(2555, ge=365, description="Minimum retention required")

    # Metadata
    created_by: str = Field(..., description="Creator identifier")
    approved_by: Optional[str] = Field(None, description="Approver identifier")
    approval_date: Optional[datetime] = Field(None, description="Approval date")


# =============================================================================
# Result Models
# =============================================================================

class ArchiveRecord(BaseModel):
    """Record of an archived item."""

    archive_id: str = Field(..., description="Unique archive identifier")
    original_id: str = Field(..., description="Original record identifier")
    category: RecordCategory = Field(..., description="Record category")
    archived_at: datetime = Field(..., description="Archive timestamp")
    original_timestamp: datetime = Field(..., description="Original record timestamp")
    archive_location: str = Field(..., description="Archive storage location")
    content_hash: str = Field(..., description="SHA-256 hash of content")
    compressed: bool = Field(True, description="Whether content is compressed")
    size_bytes: int = Field(..., ge=0, description="Size in bytes")
    retention_expires: datetime = Field(..., description="When retention expires")
    status: ArchiveStatus = Field(ArchiveStatus.ARCHIVED, description="Archive status")


class ArchiveResult(BaseModel):
    """Result of an archive operation."""

    operation_id: str = Field(..., description="Operation identifier")
    timestamp: datetime = Field(..., description="Operation timestamp")
    records_processed: int = Field(..., ge=0, description="Records processed")
    records_archived: int = Field(..., ge=0, description="Records successfully archived")
    records_failed: int = Field(..., ge=0, description="Records failed to archive")
    total_size_bytes: int = Field(..., ge=0, description="Total archived size")
    archive_records: List[ArchiveRecord] = Field(
        default_factory=list,
        description="Archive records created"
    )
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    duration_ms: float = Field(..., ge=0, description="Operation duration in ms")


class RestoreResult(BaseModel):
    """Result of a restore operation."""

    operation_id: str = Field(..., description="Operation identifier")
    timestamp: datetime = Field(..., description="Operation timestamp")
    records_requested: int = Field(..., ge=0, description="Records requested")
    records_restored: int = Field(..., ge=0, description="Records successfully restored")
    records_failed: int = Field(..., ge=0, description="Records failed to restore")
    restored_records: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Restored record data"
    )
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    duration_ms: float = Field(..., ge=0, description="Operation duration in ms")


class PurgeResult(BaseModel):
    """Result of a purge operation."""

    operation_id: str = Field(..., description="Operation identifier")
    timestamp: datetime = Field(..., description="Operation timestamp")
    records_eligible: int = Field(..., ge=0, description="Records eligible for purge")
    records_purged: int = Field(..., ge=0, description="Records successfully purged")
    records_retained: int = Field(..., ge=0, description="Records retained (exceptions)")
    space_freed_bytes: int = Field(..., ge=0, description="Storage space freed")
    purge_manifest: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Manifest of purged records"
    )
    retention_holds: List[str] = Field(
        default_factory=list,
        description="Records under retention hold"
    )
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    duration_ms: float = Field(..., ge=0, description="Operation duration in ms")


class RetentionComplianceStatus(BaseModel):
    """Status of retention compliance."""

    check_timestamp: datetime = Field(..., description="Check timestamp")
    overall_status: ComplianceStatus = Field(..., description="Overall compliance status")

    # Policy compliance
    policy_defined: bool = Field(..., description="Whether policy is defined")
    policy_current: bool = Field(..., description="Whether policy is current")
    policy_approved: bool = Field(..., description="Whether policy is approved")

    # Record compliance
    total_records: int = Field(..., ge=0, description="Total records")
    compliant_records: int = Field(..., ge=0, description="Records in compliance")
    non_compliant_records: int = Field(..., ge=0, description="Records out of compliance")
    compliance_percentage: float = Field(..., ge=0, le=100, description="Compliance percentage")

    # Category breakdown
    compliance_by_category: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Compliance by category"
    )

    # Issues and recommendations
    issues: List[str] = Field(default_factory=list, description="Compliance issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


# =============================================================================
# Configuration
# =============================================================================

class RetentionManagerConfig(BaseModel):
    """Configuration for AuditRetentionManager."""

    archive_path: str = Field("./audit_archives", description="Archive storage path")
    enable_compression: bool = Field(True, description="Enable archive compression")
    compression_level: int = Field(6, ge=1, le=9, description="Compression level (1-9)")
    max_archive_size_mb: int = Field(100, ge=1, description="Maximum archive file size")
    retention_check_interval_hours: int = Field(24, ge=1, description="Check interval")
    enable_retention_holds: bool = Field(True, description="Enable retention holds")
    purge_requires_approval: bool = Field(True, description="Require approval for purge")


# =============================================================================
# AuditRetentionManager Implementation
# =============================================================================

class AuditRetentionManager:
    """
    AuditRetentionManager implementation for BURNMASTER.

    This class manages the lifecycle of audit records including retention
    policies, archival, restoration, and purging in compliance with
    regulatory requirements.

    Attributes:
        config: Manager configuration
        _policy: Current retention policy
        _archives: Storage for archive records
        _retention_holds: Records under retention hold

    Example:
        >>> config = RetentionManagerConfig()
        >>> manager = AuditRetentionManager(config)
        >>> manager.define_retention_policy(policy)
        >>> result = manager.archive_old_records(cutoff_date)
    """

    def __init__(self, config: RetentionManagerConfig):
        """
        Initialize AuditRetentionManager.

        Args:
            config: Manager configuration
        """
        self.config = config
        self._policy: Optional[RetentionPolicy] = None
        self._archives: Dict[str, ArchiveRecord] = {}
        self._active_records: Dict[str, Dict[str, Any]] = {}  # Simulated active storage
        self._retention_holds: Dict[str, datetime] = {}  # Record ID -> Hold expiry
        self._purge_log: List[Dict[str, Any]] = []

        logger.info(
            f"AuditRetentionManager initialized with archive_path={config.archive_path}, "
            f"compression={config.enable_compression}"
        )

    def define_retention_policy(self, policy: RetentionPolicy) -> None:
        """
        Define or update the retention policy.

        Args:
            policy: Retention policy to apply

        Raises:
            ValueError: If policy is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Validate policy
            if not policy.rules:
                raise ValueError("Policy must contain at least one rule")

            # Check minimum retention requirements
            for rule in policy.rules:
                if rule.retention_days < policy.minimum_retention_days:
                    if rule.action_on_expiry != RetentionAction.RETAIN_INDEFINITELY:
                        raise ValueError(
                            f"Rule {rule.rule_id} retention ({rule.retention_days} days) "
                            f"is below minimum ({policy.minimum_retention_days} days)"
                        )

            self._policy = policy

            logger.info(
                f"Defined retention policy {policy.policy_id} v{policy.version} "
                f"with {len(policy.rules)} rules, effective {policy.effective_date}"
            )

        except Exception as e:
            logger.error(f"Failed to define retention policy: {str(e)}", exc_info=True)
            raise

    def archive_old_records(
        self,
        before: datetime,
        categories: Optional[List[RecordCategory]] = None,
        dry_run: bool = False
    ) -> ArchiveResult:
        """
        Archive records older than specified date.

        Args:
            before: Archive records older than this date
            categories: Specific categories to archive (None = all)
            dry_run: If True, simulate without actual archival

        Returns:
            Archive operation result

        Raises:
            ValueError: If no policy defined
        """
        import time
        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        try:
            if self._policy is None:
                raise ValueError("No retention policy defined")

            operation_id = str(uuid.uuid4())
            records_processed = 0
            records_archived = 0
            records_failed = 0
            total_size = 0
            archive_records: List[ArchiveRecord] = []
            errors: List[str] = []

            # Find records eligible for archival
            eligible_records = self._find_eligible_records(before, categories)
            records_processed = len(eligible_records)

            for record_id, record_data in eligible_records.items():
                try:
                    # Check for retention holds
                    if self._is_under_hold(record_id):
                        logger.debug(f"Record {record_id} under retention hold, skipping")
                        continue

                    if dry_run:
                        records_archived += 1
                        continue

                    # Archive the record
                    archive_record = self._archive_single_record(
                        record_id,
                        record_data,
                        operation_id
                    )

                    archive_records.append(archive_record)
                    total_size += archive_record.size_bytes
                    records_archived += 1

                except Exception as e:
                    records_failed += 1
                    errors.append(f"Failed to archive {record_id}: {str(e)}")

            duration_ms = (time.perf_counter() - start_perf) * 1000

            result = ArchiveResult(
                operation_id=operation_id,
                timestamp=start_time,
                records_processed=records_processed,
                records_archived=records_archived,
                records_failed=records_failed,
                total_size_bytes=total_size,
                archive_records=archive_records,
                errors=errors,
                duration_ms=duration_ms
            )

            logger.info(
                f"Archive operation {operation_id}: "
                f"{records_archived}/{records_processed} archived, "
                f"{total_size} bytes, {duration_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to archive records: {str(e)}", exc_info=True)
            raise

    def restore_archived_records(
        self,
        period: Optional[Any] = None,  # DateRange from compliance_reporter
        archive_ids: Optional[List[str]] = None,
        categories: Optional[List[RecordCategory]] = None
    ) -> RestoreResult:
        """
        Restore archived records.

        Args:
            period: Date range to restore (optional)
            archive_ids: Specific archive IDs to restore (optional)
            categories: Specific categories to restore (optional)

        Returns:
            Restore operation result
        """
        import time
        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        try:
            operation_id = str(uuid.uuid4())
            records_requested = 0
            records_restored = 0
            records_failed = 0
            restored_records: List[Dict[str, Any]] = []
            errors: List[str] = []

            # Find archives to restore
            archives_to_restore: List[ArchiveRecord] = []

            if archive_ids:
                for aid in archive_ids:
                    if aid in self._archives:
                        archives_to_restore.append(self._archives[aid])
                    else:
                        errors.append(f"Archive {aid} not found")
            else:
                for archive in self._archives.values():
                    if archive.status == ArchiveStatus.ARCHIVED:
                        # Filter by period if specified
                        if period:
                            if (archive.original_timestamp >= period.start_date and
                                archive.original_timestamp <= period.end_date):
                                archives_to_restore.append(archive)
                        # Filter by category if specified
                        elif categories:
                            if archive.category in categories:
                                archives_to_restore.append(archive)
                        else:
                            archives_to_restore.append(archive)

            records_requested = len(archives_to_restore)

            for archive in archives_to_restore:
                try:
                    # Restore the record
                    restored_data = self._restore_single_record(archive)
                    restored_records.append(restored_data)
                    records_restored += 1

                    # Update archive status
                    self._archives[archive.archive_id] = ArchiveRecord(
                        **{**archive.dict(), 'status': ArchiveStatus.RESTORED}
                    )

                except Exception as e:
                    records_failed += 1
                    errors.append(f"Failed to restore {archive.archive_id}: {str(e)}")

            duration_ms = (time.perf_counter() - start_perf) * 1000

            result = RestoreResult(
                operation_id=operation_id,
                timestamp=start_time,
                records_requested=records_requested,
                records_restored=records_restored,
                records_failed=records_failed,
                restored_records=restored_records,
                errors=errors,
                duration_ms=duration_ms
            )

            logger.info(
                f"Restore operation {operation_id}: "
                f"{records_restored}/{records_requested} restored in {duration_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to restore records: {str(e)}", exc_info=True)
            raise

    def purge_expired_records(
        self,
        approval_token: Optional[str] = None,
        dry_run: bool = False
    ) -> PurgeResult:
        """
        Purge records that have exceeded retention period.

        Args:
            approval_token: Required approval token if purge_requires_approval
            dry_run: If True, simulate without actual purge

        Returns:
            Purge operation result

        Raises:
            ValueError: If approval required but not provided
        """
        import time
        start_time = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        try:
            if self.config.purge_requires_approval and not approval_token:
                raise ValueError("Purge approval token required")

            operation_id = str(uuid.uuid4())
            records_eligible = 0
            records_purged = 0
            records_retained = 0
            space_freed = 0
            purge_manifest: List[Dict[str, Any]] = []
            retention_holds: List[str] = []
            errors: List[str] = []

            # Find expired archives
            now = datetime.now(timezone.utc)
            expired_archives = [
                a for a in self._archives.values()
                if a.retention_expires <= now and a.status == ArchiveStatus.ARCHIVED
            ]
            records_eligible = len(expired_archives)

            for archive in expired_archives:
                try:
                    # Check for retention holds
                    if self._is_under_hold(archive.original_id):
                        records_retained += 1
                        retention_holds.append(archive.original_id)
                        continue

                    # Get the retention rule for this category
                    rule = self._get_rule_for_category(archive.category)
                    if rule and rule.action_on_expiry == RetentionAction.RETAIN_INDEFINITELY:
                        records_retained += 1
                        continue

                    if dry_run:
                        records_purged += 1
                        space_freed += archive.size_bytes
                        continue

                    # Purge the record
                    purge_manifest.append({
                        "archive_id": archive.archive_id,
                        "original_id": archive.original_id,
                        "category": archive.category.value,
                        "archived_at": archive.archived_at.isoformat(),
                        "purged_at": now.isoformat(),
                        "content_hash": archive.content_hash
                    })

                    # Update status and free space
                    space_freed += archive.size_bytes
                    self._archives[archive.archive_id] = ArchiveRecord(
                        **{**archive.dict(), 'status': ArchiveStatus.PURGED}
                    )
                    records_purged += 1

                except Exception as e:
                    errors.append(f"Failed to purge {archive.archive_id}: {str(e)}")

            # Log purge operation
            self._purge_log.append({
                "operation_id": operation_id,
                "timestamp": start_time.isoformat(),
                "records_purged": records_purged,
                "approval_token": approval_token,
                "manifest": purge_manifest
            })

            duration_ms = (time.perf_counter() - start_perf) * 1000

            result = PurgeResult(
                operation_id=operation_id,
                timestamp=start_time,
                records_eligible=records_eligible,
                records_purged=records_purged,
                records_retained=records_retained,
                space_freed_bytes=space_freed,
                purge_manifest=purge_manifest,
                retention_holds=retention_holds,
                errors=errors,
                duration_ms=duration_ms
            )

            logger.info(
                f"Purge operation {operation_id}: "
                f"{records_purged}/{records_eligible} purged, "
                f"{space_freed} bytes freed in {duration_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to purge records: {str(e)}", exc_info=True)
            raise

    def validate_retention_compliance(self) -> RetentionComplianceStatus:
        """
        Validate retention compliance status.

        Returns:
            Retention compliance status report
        """
        start_time = datetime.now(timezone.utc)

        try:
            issues: List[str] = []
            recommendations: List[str] = []

            # Check policy status
            policy_defined = self._policy is not None
            policy_current = False
            policy_approved = False

            if self._policy:
                policy_current = self._policy.effective_date <= start_time
                policy_approved = self._policy.approved_by is not None

                if not policy_current:
                    issues.append("Retention policy not yet effective")
                if not policy_approved:
                    issues.append("Retention policy not approved")

            if not policy_defined:
                issues.append("No retention policy defined")
                recommendations.append("Define a retention policy to ensure compliance")

            # Analyze records by category
            total_records = len(self._active_records) + len(self._archives)
            compliant_records = 0
            non_compliant_records = 0
            compliance_by_category: Dict[str, Dict[str, Any]] = {}

            # Check active records
            for record_id, record_data in self._active_records.items():
                category = record_data.get('category', RecordCategory.OPERATIONAL)
                rule = self._get_rule_for_category(category)

                if category.value not in compliance_by_category:
                    compliance_by_category[category.value] = {
                        "total": 0,
                        "compliant": 0,
                        "non_compliant": 0,
                        "retention_days": rule.retention_days if rule else 2555
                    }

                compliance_by_category[category.value]["total"] += 1

                # Check if record should be archived
                record_age = (start_time - record_data.get('timestamp', start_time)).days
                if rule and record_age > rule.retention_days:
                    if record_data.get('status') != 'archived':
                        non_compliant_records += 1
                        compliance_by_category[category.value]["non_compliant"] += 1
                    else:
                        compliant_records += 1
                        compliance_by_category[category.value]["compliant"] += 1
                else:
                    compliant_records += 1
                    compliance_by_category[category.value]["compliant"] += 1

            # Check archives
            for archive in self._archives.values():
                category = archive.category
                if category.value not in compliance_by_category:
                    compliance_by_category[category.value] = {
                        "total": 0,
                        "compliant": 0,
                        "non_compliant": 0,
                        "retention_days": 2555
                    }

                compliance_by_category[category.value]["total"] += 1

                if archive.status == ArchiveStatus.ARCHIVED:
                    compliant_records += 1
                    compliance_by_category[category.value]["compliant"] += 1
                elif archive.status == ArchiveStatus.PURGED:
                    # Check if purge was compliant
                    if archive.retention_expires <= start_time:
                        compliant_records += 1
                        compliance_by_category[category.value]["compliant"] += 1
                    else:
                        non_compliant_records += 1
                        compliance_by_category[category.value]["non_compliant"] += 1
                        issues.append(f"Archive {archive.archive_id} purged before retention expiry")

            # Calculate compliance percentage
            compliance_percentage = (
                (compliant_records / total_records * 100)
                if total_records > 0 else 100.0
            )

            # Determine overall status
            if not policy_defined:
                overall_status = ComplianceStatus.NON_COMPLIANT
            elif non_compliant_records > 0:
                overall_status = ComplianceStatus.NON_COMPLIANT
            elif not policy_approved:
                overall_status = ComplianceStatus.WARNING
            else:
                overall_status = ComplianceStatus.COMPLIANT

            # Generate recommendations
            if non_compliant_records > 0:
                recommendations.append(
                    f"Archive {non_compliant_records} records exceeding retention period"
                )

            status = RetentionComplianceStatus(
                check_timestamp=start_time,
                overall_status=overall_status,
                policy_defined=policy_defined,
                policy_current=policy_current,
                policy_approved=policy_approved,
                total_records=total_records,
                compliant_records=compliant_records,
                non_compliant_records=non_compliant_records,
                compliance_percentage=compliance_percentage,
                compliance_by_category=compliance_by_category,
                issues=issues,
                recommendations=recommendations
            )

            logger.info(
                f"Retention compliance check: status={overall_status.value}, "
                f"{compliant_records}/{total_records} compliant ({compliance_percentage:.1f}%)"
            )

            return status

        except Exception as e:
            logger.error(f"Failed to validate retention compliance: {str(e)}", exc_info=True)
            raise

    def add_retention_hold(
        self,
        record_id: str,
        hold_until: Optional[datetime] = None,
        reason: str = ""
    ) -> None:
        """
        Add a retention hold on a record.

        Args:
            record_id: Record to hold
            hold_until: Hold expiry (None = indefinite)
            reason: Reason for hold
        """
        if not self.config.enable_retention_holds:
            raise ValueError("Retention holds are not enabled")

        self._retention_holds[record_id] = hold_until or datetime.max.replace(tzinfo=timezone.utc)
        logger.info(f"Added retention hold on {record_id} until {hold_until}: {reason}")

    def remove_retention_hold(self, record_id: str) -> None:
        """Remove a retention hold from a record."""
        if record_id in self._retention_holds:
            del self._retention_holds[record_id]
            logger.info(f"Removed retention hold on {record_id}")

    def get_policy(self) -> Optional[RetentionPolicy]:
        """Get the current retention policy."""
        return self._policy

    def get_archive(self, archive_id: str) -> Optional[ArchiveRecord]:
        """Get an archive record by ID."""
        return self._archives.get(archive_id)

    def _find_eligible_records(
        self,
        before: datetime,
        categories: Optional[List[RecordCategory]]
    ) -> Dict[str, Dict[str, Any]]:
        """Find records eligible for archival."""
        eligible = {}
        for record_id, record_data in self._active_records.items():
            record_time = record_data.get('timestamp', datetime.now(timezone.utc))
            record_category = record_data.get('category', RecordCategory.OPERATIONAL)

            if record_time < before:
                if categories is None or record_category in categories:
                    eligible[record_id] = record_data

        return eligible

    def _archive_single_record(
        self,
        record_id: str,
        record_data: Dict[str, Any],
        operation_id: str
    ) -> ArchiveRecord:
        """Archive a single record."""
        archive_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Serialize and optionally compress
        content_json = json.dumps(record_data, sort_keys=True, default=str)
        if self.config.enable_compression:
            content_bytes = gzip.compress(
                content_json.encode('utf-8'),
                compresslevel=self.config.compression_level
            )
        else:
            content_bytes = content_json.encode('utf-8')

        # Calculate hash
        content_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()

        # Get retention period
        category = record_data.get('category', RecordCategory.OPERATIONAL)
        rule = self._get_rule_for_category(category)
        retention_days = rule.retention_days if rule else self._policy.default_retention_days

        archive_record = ArchiveRecord(
            archive_id=archive_id,
            original_id=record_id,
            category=category,
            archived_at=now,
            original_timestamp=record_data.get('timestamp', now),
            archive_location=f"{self.config.archive_path}/{archive_id}.gz",
            content_hash=content_hash,
            compressed=self.config.enable_compression,
            size_bytes=len(content_bytes),
            retention_expires=now + timedelta(days=retention_days),
            status=ArchiveStatus.ARCHIVED
        )

        # Store archive record
        self._archives[archive_id] = archive_record

        # Remove from active records
        if record_id in self._active_records:
            del self._active_records[record_id]

        return archive_record

    def _restore_single_record(self, archive: ArchiveRecord) -> Dict[str, Any]:
        """Restore a single archived record."""
        # In production, would read from archive storage
        # For now, return placeholder data
        return {
            "archive_id": archive.archive_id,
            "original_id": archive.original_id,
            "category": archive.category.value,
            "restored_at": datetime.now(timezone.utc).isoformat(),
            "original_timestamp": archive.original_timestamp.isoformat()
        }

    def _is_under_hold(self, record_id: str) -> bool:
        """Check if a record is under retention hold."""
        if record_id not in self._retention_holds:
            return False

        hold_until = self._retention_holds[record_id]
        return datetime.now(timezone.utc) < hold_until

    def _get_rule_for_category(self, category: RecordCategory) -> Optional[RetentionRule]:
        """Get the retention rule for a category."""
        if self._policy is None:
            return None

        for rule in sorted(self._policy.rules, key=lambda r: r.priority):
            if rule.category == category and rule.enabled:
                return rule

        return None
