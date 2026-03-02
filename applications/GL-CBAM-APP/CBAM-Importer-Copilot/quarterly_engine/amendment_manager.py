# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Amendment Manager Engine v1.1

Manages versioned amendments to submitted CBAM quarterly reports.
Tracks changes with structured diffs, enforces the 60-day amendment window,
and maintains a complete version history for audit trails.

Per EU CBAM Implementing Regulation 2023/1773 Article 9:
  - Importers may correct submitted reports within 60 days after quarter end
  - Each amendment must document the reason and changes made
  - Full version history must be maintained for regulatory inspection

All operations are deterministic. SHA-256 provenance hashing on every version.

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import copy
import hashlib
import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .models import (
    AMENDMENT_WINDOW_DAYS,
    MAX_AMENDMENT_VERSIONS,
    AmendmentReason,
    QuarterlyReport,
    QuarterlyReportPeriod,
    ReportAmendment,
    ReportStatus,
    compute_sha256,
    quantize_decimal,
    validate_status_transition,
)

logger = logging.getLogger(__name__)


class AmendmentManagerEngine:
    """
    Engine for managing versioned amendments to CBAM quarterly reports.

    Provides create, apply, reject, and rollback operations for report
    amendments. Maintains a complete version history with structured diffs
    and SHA-256 provenance hashing for regulatory audit compliance.

    Thread Safety:
        Uses threading.RLock for all mutable state access. Safe for
        concurrent use from multiple API handlers.

    Example:
        >>> manager = AmendmentManagerEngine()
        >>> amendment = manager.create_amendment(
        ...     report_id="CBAM-QR-2025Q4-NL123-...",
        ...     changes={"total_direct_emissions": "2800.000"},
        ...     reason=AmendmentReason.NEW_SUPPLIER_DATA,
        ...     amended_by="compliance@acme.eu",
        ... )
        >>> updated_report = manager.apply_amendment(amendment.amendment_id)
    """

    def __init__(self) -> None:
        """Initialize the amendment manager engine."""
        self._lock = threading.RLock()
        # report_id -> list of QuarterlyReport versions
        self._report_versions: Dict[str, List[QuarterlyReport]] = {}
        # amendment_id -> ReportAmendment
        self._amendments: Dict[str, ReportAmendment] = {}
        # report_id -> list of amendment_ids in order
        self._amendment_chain: Dict[str, List[str]] = {}
        logger.info("AmendmentManagerEngine initialized")

    # ========================================================================
    # REPORT REGISTRATION
    # ========================================================================

    def register_report(self, report: QuarterlyReport) -> None:
        """
        Register a report for amendment tracking.

        Must be called before any amendments can be created for a report.

        Args:
            report: The quarterly report to register.
        """
        with self._lock:
            report_id = report.report_id
            if report_id not in self._report_versions:
                self._report_versions[report_id] = [report]
                self._amendment_chain[report_id] = []
                logger.info(
                    "Report %s registered for amendment tracking (v%d)",
                    report_id, report.version,
                )
            else:
                logger.warning(
                    "Report %s already registered, skipping", report_id
                )

    # ========================================================================
    # AMENDMENT CREATION
    # ========================================================================

    def create_amendment(
        self,
        report_id: str,
        changes: Dict[str, Any],
        reason: AmendmentReason,
        amended_by: str,
        changes_summary: Optional[str] = None,
    ) -> ReportAmendment:
        """
        Create a versioned amendment for a submitted report.

        Validates that:
        1. The report exists and is registered
        2. The amendment window is still open (T+60 days)
        3. The maximum version count has not been exceeded
        4. The changes actually modify the report

        Args:
            report_id: The report being amended.
            changes: Dict of field names to new values.
            reason: Categorized reason for the amendment.
            amended_by: User or system identifier.
            changes_summary: Human-readable description (auto-generated if None).

        Returns:
            ReportAmendment with diff data and hash chain.

        Raises:
            ValueError: If report not found, window closed, or max versions reached.
        """
        with self._lock:
            # Validate report exists
            if report_id not in self._report_versions:
                raise ValueError(
                    f"Report {report_id} not registered. "
                    f"Call register_report() first."
                )

            versions = self._report_versions[report_id]
            current_report = versions[-1]

            # Validate amendment window
            if not self.validate_amendment_window(report_id):
                raise ValueError(
                    f"Amendment window expired for report {report_id}. "
                    f"Deadline was {current_report.period.amendment_deadline}."
                )

            # Validate max versions
            new_version = current_report.version + 1
            if new_version > MAX_AMENDMENT_VERSIONS:
                raise ValueError(
                    f"Maximum amendment versions ({MAX_AMENDMENT_VERSIONS}) "
                    f"reached for report {report_id}. Manual review required."
                )

            # Validate status allows amendment
            if not current_report.status.allows_amendment:
                raise ValueError(
                    f"Report {report_id} status '{current_report.status.value}' "
                    f"does not allow amendments. Must be submitted, accepted, or rejected."
                )

            # Compute previous hash
            previous_hash = current_report.compute_provenance_hash()

            # Compute diff
            diff_data = self._compute_diff(current_report, changes)

            if not diff_data:
                raise ValueError(
                    "No actual changes detected. Amendment must modify the report."
                )

            # Compute what the new hash would be
            # Build a temporary updated report to compute its hash
            temp_report = self._apply_changes_to_report(current_report, changes)
            new_hash = temp_report.compute_provenance_hash()

            if previous_hash == new_hash:
                raise ValueError(
                    "Changes do not alter the provenance hash. "
                    "Ensure changes affect material fields."
                )

            # Auto-generate summary if not provided
            if not changes_summary:
                changes_summary = self._generate_changes_summary(
                    reason, diff_data
                )

            # Create amendment ID
            amendment_id = (
                f"AMEND-{current_report.period.period_label}-"
                f"{uuid.uuid4().hex[:8].upper()}-v{new_version:03d}"
            )

            amendment = ReportAmendment(
                amendment_id=amendment_id,
                report_id=report_id,
                version=new_version,
                reason=reason,
                changes_summary=changes_summary,
                diff_data=diff_data,
                amended_by=amended_by,
                amended_at=datetime.now(timezone.utc),
                previous_hash=previous_hash,
                new_hash=new_hash,
            )

            # Store the amendment (not yet applied)
            self._amendments[amendment_id] = amendment
            self._amendment_chain[report_id].append(amendment_id)

            logger.info(
                "Amendment created: id=%s, report=%s, version=%d, reason=%s",
                amendment_id, report_id, new_version, reason.value,
            )

            return amendment

    # ========================================================================
    # AMENDMENT RETRIEVAL
    # ========================================================================

    def get_amendments(self, report_id: str) -> List[ReportAmendment]:
        """
        List all amendments for a report in chronological order.

        Args:
            report_id: The report identifier.

        Returns:
            List of ReportAmendment objects ordered by version.
        """
        with self._lock:
            amendment_ids = self._amendment_chain.get(report_id, [])
            amendments = [
                self._amendments[aid]
                for aid in amendment_ids
                if aid in self._amendments
            ]
            return sorted(amendments, key=lambda a: a.version)

    def get_amendment(self, amendment_id: str) -> Optional[ReportAmendment]:
        """
        Retrieve a specific amendment by its ID.

        Args:
            amendment_id: The amendment identifier.

        Returns:
            ReportAmendment if found, None otherwise.
        """
        with self._lock:
            return self._amendments.get(amendment_id)

    def get_amendment_diff(self, amendment_id: str) -> Dict[str, Any]:
        """
        Get the detailed diff between versions for a specific amendment.

        Args:
            amendment_id: The amendment identifier.

        Returns:
            Dict with field-level diffs: {field: {old: ..., new: ...}}.

        Raises:
            ValueError: If amendment not found.
        """
        with self._lock:
            amendment = self._amendments.get(amendment_id)
            if not amendment:
                raise ValueError(f"Amendment {amendment_id} not found")
            return amendment.diff_data

    # ========================================================================
    # AMENDMENT WINDOW VALIDATION
    # ========================================================================

    def validate_amendment_window(
        self,
        report_id: str,
        reference_date: Optional[date] = None
    ) -> bool:
        """
        Check if the amendment window is still open for a report.

        The window is T+60 calendar days after the end of the quarter,
        per Implementing Regulation 2023/1773 Article 9.

        Args:
            report_id: The report identifier.
            reference_date: Date to check against (defaults to today).

        Returns:
            True if amendments are still allowed.
        """
        with self._lock:
            if report_id not in self._report_versions:
                return False

            current_report = self._report_versions[report_id][-1]
            if reference_date is None:
                reference_date = date.today()

            return reference_date <= current_report.period.amendment_deadline

    # ========================================================================
    # AMENDMENT APPLICATION
    # ========================================================================

    def apply_amendment(self, amendment_id: str) -> QuarterlyReport:
        """
        Apply a pending amendment to create a new report version.

        Updates the report with the amendment's changes, increments the
        version number, and stores the new version in the version history.

        Args:
            amendment_id: The amendment to apply.

        Returns:
            Updated QuarterlyReport with the new version.

        Raises:
            ValueError: If amendment not found or already applied.
        """
        with self._lock:
            amendment = self._amendments.get(amendment_id)
            if not amendment:
                raise ValueError(f"Amendment {amendment_id} not found")

            report_id = amendment.report_id
            versions = self._report_versions.get(report_id)
            if not versions:
                raise ValueError(f"Report {report_id} not found in version store")

            current_report = versions[-1]

            # Verify amendment hasn't already been applied
            if current_report.version >= amendment.version:
                raise ValueError(
                    f"Amendment v{amendment.version} already applied or superseded "
                    f"(current version: v{current_report.version})"
                )

            # Apply changes
            updated_report = self._apply_changes_to_report(
                current_report, amendment.diff_data, use_new_values=True
            )

            # Update version and status
            updated_report = updated_report.model_copy(update={
                "version": amendment.version,
                "status": ReportStatus.AMENDED,
            })

            # Recompute provenance hash
            new_hash = updated_report.compute_provenance_hash()
            updated_report = updated_report.model_copy(
                update={"provenance_hash": new_hash}
            )

            # Store new version
            versions.append(updated_report)

            logger.info(
                "Amendment %s applied to report %s (v%d -> v%d)",
                amendment_id, report_id,
                current_report.version, amendment.version,
            )

            return updated_report

    def reject_amendment(
        self,
        amendment_id: str,
        rejection_reason: str
    ) -> ReportAmendment:
        """
        Reject a pending amendment with a documented reason.

        Args:
            amendment_id: The amendment to reject.
            rejection_reason: Human-readable reason for rejection.

        Returns:
            Updated ReportAmendment with rejection info in diff_data.

        Raises:
            ValueError: If amendment not found.
        """
        with self._lock:
            amendment = self._amendments.get(amendment_id)
            if not amendment:
                raise ValueError(f"Amendment {amendment_id} not found")

            # Add rejection info to diff_data
            updated_diff = dict(amendment.diff_data)
            updated_diff["_rejection"] = {
                "rejected": True,
                "rejection_reason": rejection_reason,
                "rejected_at": datetime.now(timezone.utc).isoformat(),
            }

            rejected_amendment = amendment.model_copy(
                update={"diff_data": updated_diff}
            )
            self._amendments[amendment_id] = rejected_amendment

            logger.info(
                "Amendment %s rejected: %s", amendment_id, rejection_reason
            )

            return rejected_amendment

    # ========================================================================
    # VERSION HISTORY
    # ========================================================================

    def get_version_history(self, report_id: str) -> List[Dict[str, Any]]:
        """
        Get the full version timeline for a report.

        Returns a chronological list of all versions with their metadata,
        including which amendment caused each version change.

        Args:
            report_id: The report identifier.

        Returns:
            List of version info dicts:
            [
                {
                    "version": 1,
                    "status": "submitted",
                    "created_at": "...",
                    "provenance_hash": "...",
                    "amendment_id": None,
                    "amendment_reason": None,
                },
                {
                    "version": 2,
                    "status": "amended",
                    "created_at": "...",
                    "provenance_hash": "...",
                    "amendment_id": "AMEND-...",
                    "amendment_reason": "new_supplier_data",
                },
            ]

        Raises:
            ValueError: If report not found.
        """
        with self._lock:
            versions = self._report_versions.get(report_id)
            if not versions:
                raise ValueError(f"Report {report_id} not found")

            amendments = self.get_amendments(report_id)
            amendment_by_version: Dict[int, ReportAmendment] = {
                a.version: a for a in amendments
            }

            history: List[Dict[str, Any]] = []
            for report_version in versions:
                amendment = amendment_by_version.get(report_version.version)
                entry = {
                    "version": report_version.version,
                    "status": report_version.status.value,
                    "created_at": report_version.created_at.isoformat(),
                    "provenance_hash": report_version.provenance_hash,
                    "total_embedded_emissions": str(
                        report_version.total_embedded_emissions
                    ),
                    "shipments_count": report_version.shipments_count,
                    "amendment_id": amendment.amendment_id if amendment else None,
                    "amendment_reason": (
                        amendment.reason.value if amendment else None
                    ),
                    "amendment_summary": (
                        amendment.changes_summary if amendment else None
                    ),
                    "amended_by": amendment.amended_by if amendment else None,
                }
                history.append(entry)

            return history

    def rollback_to_version(
        self,
        report_id: str,
        target_version: int
    ) -> QuarterlyReport:
        """
        Rollback a report to a specific previous version.

        Creates a new version (version N+1) that restores the state of
        the target version. Does not delete intermediate versions --
        they are preserved for audit trail.

        Args:
            report_id: The report identifier.
            target_version: The version number to restore.

        Returns:
            New QuarterlyReport at version N+1 with target version's data.

        Raises:
            ValueError: If report or target version not found.
        """
        with self._lock:
            versions = self._report_versions.get(report_id)
            if not versions:
                raise ValueError(f"Report {report_id} not found")

            # Find target version
            target_report = None
            for v in versions:
                if v.version == target_version:
                    target_report = v
                    break

            if target_report is None:
                available = [v.version for v in versions]
                raise ValueError(
                    f"Version {target_version} not found for report {report_id}. "
                    f"Available versions: {available}"
                )

            current_report = versions[-1]
            new_version = current_report.version + 1

            if new_version > MAX_AMENDMENT_VERSIONS:
                raise ValueError(
                    f"Cannot rollback: would exceed max versions "
                    f"({MAX_AMENDMENT_VERSIONS})"
                )

            # Create rollback version
            rollback_report = target_report.model_copy(update={
                "version": new_version,
                "status": ReportStatus.AMENDED,
                "created_at": datetime.now(timezone.utc),
            })

            # Recompute provenance hash
            new_hash = rollback_report.compute_provenance_hash()
            rollback_report = rollback_report.model_copy(
                update={"provenance_hash": new_hash}
            )

            # Store rollback version
            versions.append(rollback_report)

            # Create audit amendment record
            amendment_id = (
                f"AMEND-ROLLBACK-{uuid.uuid4().hex[:8].upper()}-v{new_version:03d}"
            )
            rollback_amendment = ReportAmendment(
                amendment_id=amendment_id,
                report_id=report_id,
                version=new_version,
                reason=AmendmentReason.DATA_CORRECTION,
                changes_summary=(
                    f"Rollback from version {current_report.version} "
                    f"to version {target_version}"
                ),
                diff_data={
                    "_rollback": {
                        "from_version": current_report.version,
                        "to_version": target_version,
                        "new_version": new_version,
                    }
                },
                amended_by="system:rollback",
                previous_hash=current_report.compute_provenance_hash(),
                new_hash=new_hash,
            )

            self._amendments[amendment_id] = rollback_amendment
            self._amendment_chain[report_id].append(amendment_id)

            logger.info(
                "Report %s rolled back: v%d -> v%d (restored from v%d)",
                report_id, current_report.version, new_version, target_version,
            )

            return rollback_report

    def get_current_version(self, report_id: str) -> Optional[QuarterlyReport]:
        """
        Get the latest version of a report.

        Args:
            report_id: The report identifier.

        Returns:
            Latest QuarterlyReport version, or None if not found.
        """
        with self._lock:
            versions = self._report_versions.get(report_id)
            if not versions:
                return None
            return versions[-1]

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _compute_diff(
        self,
        current_report: QuarterlyReport,
        changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute a structured diff between the current report and proposed changes.

        Args:
            current_report: The current report version.
            changes: Dict of field names to new values.

        Returns:
            Dict with field-level diffs: {field: {old: value, new: value}}.
        """
        diff: Dict[str, Any] = {}
        current_dict = current_report.model_dump()

        for field, new_value in changes.items():
            if field.startswith("_"):
                continue  # Skip private/meta fields

            old_value = current_dict.get(field)

            # Convert for comparison
            old_str = str(old_value) if old_value is not None else None
            new_str = str(new_value) if new_value is not None else None

            if old_str != new_str:
                diff[field] = {
                    "old": old_value,
                    "new": new_value,
                }

        return diff

    def _apply_changes_to_report(
        self,
        report: QuarterlyReport,
        changes: Dict[str, Any],
        use_new_values: bool = False
    ) -> QuarterlyReport:
        """
        Apply a set of changes to create an updated report.

        Args:
            report: The base report to modify.
            changes: Dict of changes (either raw field values or diff format).
            use_new_values: If True, changes are in diff format {field: {old, new}}.

        Returns:
            Updated QuarterlyReport (new instance).
        """
        update_dict: Dict[str, Any] = {}

        for field, value in changes.items():
            if field.startswith("_"):
                continue

            if use_new_values and isinstance(value, dict) and "new" in value:
                update_dict[field] = value["new"]
            else:
                update_dict[field] = value

        # Convert string Decimals back to Decimal if needed
        decimal_fields = {
            "total_quantity_mt",
            "total_direct_emissions",
            "total_indirect_emissions",
            "total_embedded_emissions",
        }
        for field in decimal_fields:
            if field in update_dict:
                val = update_dict[field]
                if not isinstance(val, Decimal):
                    update_dict[field] = Decimal(str(val))

        return report.model_copy(update=update_dict)

    def _generate_changes_summary(
        self,
        reason: AmendmentReason,
        diff_data: Dict[str, Any]
    ) -> str:
        """
        Auto-generate a human-readable changes summary.

        Args:
            reason: The amendment reason.
            diff_data: The structured diff.

        Returns:
            Human-readable summary string.
        """
        changed_fields = [
            f for f in diff_data.keys() if not f.startswith("_")
        ]
        field_list = ", ".join(changed_fields[:5])
        if len(changed_fields) > 5:
            field_list += f" (+{len(changed_fields) - 5} more)"

        return (
            f"{reason.description}. "
            f"Modified fields: {field_list}."
        )
