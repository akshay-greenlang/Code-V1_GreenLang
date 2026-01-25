# -*- coding: utf-8 -*-
"""
Opt-out handler for GDPR, CCPA, and CAN-SPAM compliance.

Processes opt-out requests, manages suppression lists, and enforces grace periods.
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import hashlib

from ..models import ConsentRecord, ConsentStatus
from ..exceptions import SupplierNotFoundError
from .registry import ConsentRegistry
from .jurisdictions import JurisdictionManager
from greenlang.determinism import DeterministicClock


logger = logging.getLogger(__name__)


class OptOutHandler:
    """
    Handles supplier opt-out requests with regulatory compliance.

    Features:
    - Immediate opt-out processing
    - Grace period enforcement (GDPR: 1 day, CCPA: 15 days, CAN-SPAM: 10 days)
    - Suppression list management
    - Audit trail for opt-outs
    """

    def __init__(self, consent_registry: ConsentRegistry):
        """
        Initialize opt-out handler.

        Args:
            consent_registry: Consent registry instance
        """
        self.registry = consent_registry
        self.jurisdiction_manager = JurisdictionManager()
        self.suppression_list: Dict[str, datetime] = {}
        logger.info("OptOutHandler initialized")

    def process_opt_out(
        self,
        supplier_id: str,
        reason: Optional[str] = None,
        source: str = "email_unsubscribe"
    ) -> ConsentRecord:
        """
        Process supplier opt-out request.

        Args:
            supplier_id: Supplier identifier
            reason: Reason for opt-out
            source: Source of opt-out (email_unsubscribe, portal, manual)

        Returns:
            Updated consent record

        Raises:
            SupplierNotFoundError: If supplier not registered
        """
        logger.info(
            f"Processing opt-out for supplier {supplier_id}. "
            f"Source: {source}, Reason: {reason or 'Not provided'}"
        )

        # Revoke consent in registry
        record = self.registry.revoke_consent(supplier_id, reason)

        # Add to suppression list
        self.add_to_suppression_list(supplier_id)

        # Log opt-out event
        self._log_opt_out_event(supplier_id, reason, source)

        return record

    def process_opt_out_by_email(
        self,
        email_address: str,
        reason: Optional[str] = None
    ) -> List[ConsentRecord]:
        """
        Process opt-out by email address (handles multiple supplier IDs).

        Args:
            email_address: Email address to opt out
            reason: Reason for opt-out

        Returns:
            List of updated consent records
        """
        # Find all supplier IDs with this email
        matching_suppliers = [
            supplier_id for supplier_id, record in self.registry.records.items()
            if record.email_address.lower() == email_address.lower()
        ]

        if not matching_suppliers:
            logger.warning(f"No suppliers found with email {email_address}")
            return []

        logger.info(
            f"Processing opt-out for {len(matching_suppliers)} suppliers "
            f"with email {email_address}"
        )

        updated_records = []
        for supplier_id in matching_suppliers:
            record = self.process_opt_out(supplier_id, reason, "email_unsubscribe")
            updated_records.append(record)

        return updated_records

    def add_to_suppression_list(self, supplier_id: str):
        """
        Add supplier to suppression list.

        Args:
            supplier_id: Supplier identifier
        """
        self.suppression_list[supplier_id] = DeterministicClock.utcnow()
        logger.debug(f"Added supplier {supplier_id} to suppression list")

    def remove_from_suppression_list(self, supplier_id: str):
        """
        Remove supplier from suppression list (e.g., if they re-consent).

        Args:
            supplier_id: Supplier identifier
        """
        if supplier_id in self.suppression_list:
            del self.suppression_list[supplier_id]
            logger.debug(f"Removed supplier {supplier_id} from suppression list")

    def is_suppressed(self, supplier_id: str) -> bool:
        """
        Check if supplier is on suppression list.

        Args:
            supplier_id: Supplier identifier

        Returns:
            True if suppressed, False otherwise
        """
        return supplier_id in self.suppression_list

    def check_grace_period(self, supplier_id: str) -> bool:
        """
        Check if supplier is within opt-out grace period.

        Args:
            supplier_id: Supplier identifier

        Returns:
            True if within grace period, False otherwise
        """
        record = self.registry.get_record(supplier_id)
        if not record or not record.opt_out_date:
            return False

        # Get jurisdiction-specific grace period
        rules = self.jurisdiction_manager.get_rules(record.country)
        grace_period = rules.opt_out_grace_period()

        # Check if still within grace period
        grace_end = record.opt_out_date + grace_period
        return DeterministicClock.utcnow() < grace_end

    def get_suppression_list(self) -> List[str]:
        """
        Get list of suppressed supplier IDs.

        Returns:
            List of supplier IDs
        """
        return list(self.suppression_list.keys())

    def generate_unsubscribe_token(
        self,
        supplier_id: str,
        campaign_id: str
    ) -> str:
        """
        Generate secure unsubscribe token for email links.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier

        Returns:
            Unsubscribe token
        """
        # Create token from supplier_id + campaign_id + timestamp
        timestamp = DeterministicClock.utcnow().isoformat()
        data = f"{supplier_id}:{campaign_id}:{timestamp}"
        token = hashlib.sha256(data.encode()).hexdigest()

        logger.debug(f"Generated unsubscribe token for supplier {supplier_id}")

        return token

    def validate_unsubscribe_token(
        self,
        token: str,
        supplier_id: str,
        campaign_id: str
    ) -> bool:
        """
        Validate unsubscribe token.

        Args:
            token: Unsubscribe token
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier

        Returns:
            True if valid, False otherwise
        """
        # In production, implement proper token validation with expiry
        # For now, simple validation
        return len(token) == 64  # SHA256 hash length

    def generate_unsubscribe_url(
        self,
        supplier_id: str,
        campaign_id: str,
        base_url: str = "https://portal.company.com"
    ) -> str:
        """
        Generate unsubscribe URL with token.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            base_url: Base portal URL

        Returns:
            Unsubscribe URL
        """
        token = self.generate_unsubscribe_token(supplier_id, campaign_id)
        url = f"{base_url}/unsubscribe?token={token}&supplier={supplier_id}&campaign={campaign_id}"

        logger.debug(f"Generated unsubscribe URL for supplier {supplier_id}")

        return url

    def _log_opt_out_event(
        self,
        supplier_id: str,
        reason: Optional[str],
        source: str
    ):
        """
        Log opt-out event for audit trail.

        Args:
            supplier_id: Supplier identifier
            reason: Reason for opt-out
            source: Source of opt-out
        """
        record = self.registry.get_record(supplier_id)
        if not record:
            return

        jurisdiction = self.jurisdiction_manager.get_jurisdiction(record.country)

        log_entry = {
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "supplier_id": supplier_id,
            "email": record.email_address,
            "jurisdiction": jurisdiction.value,
            "reason": reason,
            "source": source,
            "opt_out_date": record.opt_out_date.isoformat() if record.opt_out_date else None
        }

        logger.info(f"Opt-out event logged: {log_entry}")

    def get_opt_out_statistics(self) -> Dict:
        """
        Get opt-out statistics for reporting.

        Returns:
            Dictionary of opt-out statistics
        """
        opted_out_suppliers = self.registry.get_opted_out_suppliers()

        # Analyze opt-out reasons
        reasons = {}
        for supplier_id in opted_out_suppliers:
            record = self.registry.get_record(supplier_id)
            if record and record.opt_out_reason:
                reasons[record.opt_out_reason] = reasons.get(record.opt_out_reason, 0) + 1

        # Analyze by jurisdiction
        jurisdictions = {}
        for supplier_id in opted_out_suppliers:
            record = self.registry.get_record(supplier_id)
            if record:
                jurisdiction = self.jurisdiction_manager.get_jurisdiction(record.country)
                jurisdictions[jurisdiction.value] = jurisdictions.get(jurisdiction.value, 0) + 1

        return {
            "total_opted_out": len(opted_out_suppliers),
            "suppression_list_size": len(self.suppression_list),
            "opt_out_reasons": reasons,
            "opt_outs_by_jurisdiction": jurisdictions,
        }
