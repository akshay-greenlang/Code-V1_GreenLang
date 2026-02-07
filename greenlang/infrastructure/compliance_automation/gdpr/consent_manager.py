# -*- coding: utf-8 -*-
"""
GDPR Consent Manager - SEC-010 Phase 5

Manages user consent records in compliance with GDPR Articles 6 (lawful
processing) and 7 (conditions for consent). Tracks consent grants,
revocations, and provides audit trails for compliance verification.

Key Features:
- Granular per-purpose consent tracking
- Consent version management
- Revocation handling with propagation
- Audit trail for all consent actions
- Consent status verification

Classes:
    - ConsentManager: Main consent management engine.
    - ConsentAuditEntry: Audit trail entry for consent changes.
    - ConsentStatusResult: Result of consent status check.

Example:
    >>> manager = ConsentManager()
    >>> await manager.record_consent(
    ...     user_id="user-123",
    ...     purpose=ConsentPurpose.MARKETING,
    ...     source="web_form",
    ... )
    >>> status = await manager.get_consent_status("user-123", ConsentPurpose.MARKETING)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.compliance_automation.models import (
    ConsentPurpose,
    ConsentRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConsentAuditEntry(BaseModel):
    """Audit trail entry for consent changes.

    Attributes:
        id: Unique entry identifier.
        consent_record_id: ID of the consent record.
        user_id: User ID.
        action: Action taken (grant, revoke, update).
        purpose: Consent purpose.
        timestamp: When the action occurred.
        performed_by: Who performed the action (user, system).
        ip_address: IP address of the request.
        user_agent: Browser/client user agent.
        details: Additional action details.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    consent_record_id: str
    user_id: str
    action: str
    purpose: ConsentPurpose
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    performed_by: str = "user"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ConsentStatusResult(BaseModel):
    """Result of a consent status check.

    Attributes:
        user_id: The user ID checked.
        purpose: The consent purpose.
        has_consent: Whether active consent exists.
        consent_record: The consent record if exists.
        granted_at: When consent was granted.
        consent_version: Version of consent text.
        can_process: Whether processing is allowed.
    """

    user_id: str
    purpose: ConsentPurpose
    has_consent: bool = False
    consent_record: Optional[ConsentRecord] = None
    granted_at: Optional[datetime] = None
    consent_version: Optional[str] = None
    can_process: bool = False


class ConsentSummary(BaseModel):
    """Summary of consent status for a user.

    Attributes:
        user_id: The user ID.
        total_purposes: Total consent purposes tracked.
        active_consents: Number of active consents.
        revoked_consents: Number of revoked consents.
        consents_by_purpose: Status by purpose.
        last_updated: Last consent activity.
    """

    user_id: str
    total_purposes: int = 0
    active_consents: int = 0
    revoked_consents: int = 0
    consents_by_purpose: Dict[str, bool] = Field(default_factory=dict)
    last_updated: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Consent Manager
# ---------------------------------------------------------------------------


class ConsentManager:
    """Manages user consent records for GDPR compliance.

    Tracks consent grants and revocations for various processing purposes,
    maintains audit trails, and provides consent verification for data
    processing operations.

    Attributes:
        consent_records: Dictionary of user_id -> purpose -> consent record.
        audit_trail: List of all consent audit entries.
        consent_versions: Current version of consent text per purpose.

    Example:
        >>> manager = ConsentManager()
        >>> record = await manager.record_consent(
        ...     user_id="user-123",
        ...     purpose=ConsentPurpose.MARKETING,
        ...     source="registration_form",
        ...     ip_address="192.168.1.1",
        ... )
        >>> print(f"Consent recorded: {record.id}")
    """

    # Current consent text versions by purpose
    CONSENT_VERSIONS = {
        ConsentPurpose.MARKETING: "1.2",
        ConsentPurpose.ANALYTICS: "1.1",
        ConsentPurpose.PERSONALIZATION: "1.0",
        ConsentPurpose.THIRD_PARTY_SHARING: "1.1",
        ConsentPurpose.RESEARCH: "1.0",
        ConsentPurpose.ESSENTIAL: "1.0",
        ConsentPurpose.CARBON_REPORTING: "1.0",
        ConsentPurpose.SUPPLY_CHAIN_VISIBILITY: "1.0",
    }

    # Purposes that require explicit consent (others may have legitimate interest)
    EXPLICIT_CONSENT_REQUIRED = [
        ConsentPurpose.MARKETING,
        ConsentPurpose.THIRD_PARTY_SHARING,
        ConsentPurpose.RESEARCH,
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the consent manager.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        # In-memory storage: user_id -> purpose -> ConsentRecord
        self._records: Dict[str, Dict[ConsentPurpose, ConsentRecord]] = {}
        self._audit_trail: List[ConsentAuditEntry] = []

        logger.info("Initialized ConsentManager")

    async def record_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        source: str = "web_form",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentRecord:
        """Record a new consent grant.

        Creates a consent record for the specified user and purpose.
        If consent already exists, it is updated with new metadata.

        Args:
            user_id: The user granting consent.
            purpose: The purpose for which consent is granted.
            source: How consent was obtained (web_form, api, import).
            ip_address: IP address of the consent request.
            user_agent: Browser/client user agent.
            metadata: Additional metadata.

        Returns:
            The created or updated ConsentRecord.
        """
        logger.info(
            "Recording consent: user=%s, purpose=%s, source=%s",
            user_id,
            purpose.value,
            source,
        )

        # Check for existing consent
        existing = self._get_record(user_id, purpose)
        if existing and existing.is_active:
            logger.info("Consent already exists for user %s, purpose %s", user_id, purpose.value)
            return existing

        # Create new consent record
        record = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            source=source,
            ip_address=ip_address,
            user_agent=user_agent,
            consent_version=self.CONSENT_VERSIONS.get(purpose, "1.0"),
            metadata=metadata or {},
        )

        # Store record
        self._store_record(record)

        # Create audit entry
        audit_entry = ConsentAuditEntry(
            consent_record_id=record.id,
            user_id=user_id,
            action="grant",
            purpose=purpose,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "source": source,
                "consent_version": record.consent_version,
            },
        )
        self._audit_trail.append(audit_entry)

        logger.info(
            "Consent recorded: id=%s, user=%s, purpose=%s",
            record.id[:8],
            user_id,
            purpose.value,
        )

        return record

    async def revoke_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        reason: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> bool:
        """Revoke consent for a specific purpose.

        Marks the consent record as revoked and triggers any necessary
        downstream processing (e.g., stopping marketing emails).

        Args:
            user_id: The user revoking consent.
            purpose: The purpose being revoked.
            reason: Optional reason for revocation.
            ip_address: IP address of the request.
            user_agent: Browser/client user agent.

        Returns:
            True if consent was revoked, False if no active consent found.
        """
        logger.info(
            "Revoking consent: user=%s, purpose=%s",
            user_id,
            purpose.value,
        )

        record = self._get_record(user_id, purpose)
        if not record or not record.is_active:
            logger.warning(
                "No active consent to revoke: user=%s, purpose=%s",
                user_id,
                purpose.value,
            )
            return False

        # Mark as revoked
        record.revoked_at = datetime.now(timezone.utc)

        # Create audit entry
        audit_entry = ConsentAuditEntry(
            consent_record_id=record.id,
            user_id=user_id,
            action="revoke",
            purpose=purpose,
            ip_address=ip_address,
            user_agent=user_agent,
            details={
                "reason": reason,
                "original_granted_at": record.granted_at.isoformat(),
            },
        )
        self._audit_trail.append(audit_entry)

        # Trigger downstream processing
        await self._propagate_revocation(user_id, purpose)

        logger.info(
            "Consent revoked: user=%s, purpose=%s",
            user_id,
            purpose.value,
        )

        return True

    async def get_consent_status(
        self,
        user_id: str,
        purpose: ConsentPurpose,
    ) -> ConsentStatusResult:
        """Check consent status for a user and purpose.

        Determines whether processing is allowed for the specified
        user and purpose based on consent status.

        Args:
            user_id: The user to check.
            purpose: The purpose to check.

        Returns:
            ConsentStatusResult with status details.
        """
        record = self._get_record(user_id, purpose)

        result = ConsentStatusResult(
            user_id=user_id,
            purpose=purpose,
        )

        if record and record.is_active:
            result.has_consent = True
            result.consent_record = record
            result.granted_at = record.granted_at
            result.consent_version = record.consent_version
            result.can_process = True
        elif purpose not in self.EXPLICIT_CONSENT_REQUIRED:
            # Legitimate interest may apply
            result.can_process = True  # Subject to legitimate interest assessment
        else:
            result.can_process = False

        return result

    async def get_user_consent_summary(
        self,
        user_id: str,
    ) -> ConsentSummary:
        """Get a summary of all consents for a user.

        Args:
            user_id: The user ID.

        Returns:
            ConsentSummary with consent status by purpose.
        """
        summary = ConsentSummary(
            user_id=user_id,
            total_purposes=len(ConsentPurpose),
        )

        user_records = self._records.get(user_id, {})
        last_updated: Optional[datetime] = None

        for purpose in ConsentPurpose:
            record = user_records.get(purpose)
            if record:
                is_active = record.is_active
                summary.consents_by_purpose[purpose.value] = is_active

                if is_active:
                    summary.active_consents += 1
                    # Track last activity
                    if last_updated is None or record.granted_at > last_updated:
                        last_updated = record.granted_at
                else:
                    summary.revoked_consents += 1
                    if record.revoked_at:
                        if last_updated is None or record.revoked_at > last_updated:
                            last_updated = record.revoked_at
            else:
                summary.consents_by_purpose[purpose.value] = False

        summary.last_updated = last_updated

        return summary

    async def audit_consent_trail(
        self,
        user_id: Optional[str] = None,
        purpose: Optional[ConsentPurpose] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[ConsentAuditEntry]:
        """Get consent audit trail with optional filters.

        Args:
            user_id: Filter by user ID.
            purpose: Filter by purpose.
            start_date: Filter by start date.
            end_date: Filter by end date.

        Returns:
            List of matching audit entries.
        """
        entries = self._audit_trail

        if user_id:
            entries = [e for e in entries if e.user_id == user_id]

        if purpose:
            entries = [e for e in entries if e.purpose == purpose]

        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]

        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]

        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        return entries

    async def bulk_record_consent(
        self,
        user_id: str,
        purposes: List[ConsentPurpose],
        source: str = "web_form",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> List[ConsentRecord]:
        """Record consent for multiple purposes at once.

        Args:
            user_id: The user granting consent.
            purposes: List of purposes to consent to.
            source: How consent was obtained.
            ip_address: IP address of the request.
            user_agent: Browser/client user agent.

        Returns:
            List of created consent records.
        """
        records: List[ConsentRecord] = []

        for purpose in purposes:
            record = await self.record_consent(
                user_id=user_id,
                purpose=purpose,
                source=source,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            records.append(record)

        return records

    async def bulk_revoke_consent(
        self,
        user_id: str,
        purposes: Optional[List[ConsentPurpose]] = None,
        reason: Optional[str] = None,
    ) -> int:
        """Revoke consent for multiple purposes (or all purposes).

        Args:
            user_id: The user revoking consent.
            purposes: List of purposes to revoke (None = all).
            reason: Optional reason for revocation.

        Returns:
            Number of consents revoked.
        """
        if purposes is None:
            purposes = list(ConsentPurpose)

        revoked_count = 0

        for purpose in purposes:
            if await self.revoke_consent(user_id, purpose, reason):
                revoked_count += 1

        return revoked_count

    async def check_processing_allowed(
        self,
        user_id: str,
        purpose: ConsentPurpose,
    ) -> bool:
        """Quick check if processing is allowed for a purpose.

        This is a simplified check for use in request processing.
        Use get_consent_status for full details.

        Args:
            user_id: The user ID.
            purpose: The processing purpose.

        Returns:
            True if processing is allowed.
        """
        status = await self.get_consent_status(user_id, purpose)
        return status.can_process

    async def get_consented_users(
        self,
        purpose: ConsentPurpose,
    ) -> List[str]:
        """Get all users who have consented to a purpose.

        Useful for bulk operations like marketing campaigns.

        Args:
            purpose: The consent purpose.

        Returns:
            List of user IDs with active consent.
        """
        consented_users: List[str] = []

        for user_id, user_records in self._records.items():
            record = user_records.get(purpose)
            if record and record.is_active:
                consented_users.append(user_id)

        return consented_users

    async def export_consent_records(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """Export all consent records for a user (DSAR support).

        Args:
            user_id: The user ID.

        Returns:
            Dictionary of consent data suitable for export.
        """
        user_records = self._records.get(user_id, {})

        export = {
            "user_id": user_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "consents": [],
        }

        for purpose, record in user_records.items():
            export["consents"].append({
                "purpose": purpose.value,
                "granted_at": record.granted_at.isoformat(),
                "revoked_at": record.revoked_at.isoformat() if record.revoked_at else None,
                "is_active": record.is_active,
                "source": record.source,
                "consent_version": record.consent_version,
            })

        return export

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_record(
        self,
        user_id: str,
        purpose: ConsentPurpose,
    ) -> Optional[ConsentRecord]:
        """Get consent record for user and purpose."""
        user_records = self._records.get(user_id, {})
        return user_records.get(purpose)

    def _store_record(self, record: ConsentRecord) -> None:
        """Store a consent record."""
        if record.user_id not in self._records:
            self._records[record.user_id] = {}
        self._records[record.user_id][record.purpose] = record

    async def _propagate_revocation(
        self,
        user_id: str,
        purpose: ConsentPurpose,
    ) -> None:
        """Propagate consent revocation to downstream systems.

        In production, this would trigger:
        - Marketing: Unsubscribe from email lists
        - Analytics: Stop data collection
        - Third party: Notify data sharing partners
        """
        logger.info(
            "Propagating consent revocation: user=%s, purpose=%s",
            user_id,
            purpose.value,
        )

        if purpose == ConsentPurpose.MARKETING:
            await self._unsubscribe_marketing(user_id)
        elif purpose == ConsentPurpose.ANALYTICS:
            await self._disable_analytics(user_id)
        elif purpose == ConsentPurpose.THIRD_PARTY_SHARING:
            await self._notify_third_parties(user_id)

    async def _unsubscribe_marketing(self, user_id: str) -> None:
        """Unsubscribe user from marketing communications."""
        logger.debug("Unsubscribing user %s from marketing", user_id)
        # In production, call marketing service

    async def _disable_analytics(self, user_id: str) -> None:
        """Disable analytics collection for user."""
        logger.debug("Disabling analytics for user %s", user_id)
        # In production, update analytics settings

    async def _notify_third_parties(self, user_id: str) -> None:
        """Notify third parties of consent revocation."""
        logger.debug("Notifying third parties of revocation for user %s", user_id)
        # In production, call third party APIs


__all__ = [
    "ConsentManager",
    "ConsentAuditEntry",
    "ConsentStatusResult",
    "ConsentSummary",
]
