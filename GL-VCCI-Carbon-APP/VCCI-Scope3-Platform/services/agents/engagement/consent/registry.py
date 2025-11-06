"""
Consent registry for GDPR, CCPA, and CAN-SPAM compliance.

Manages supplier consent records, opt-ins/opt-outs, and data retention.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..models import ConsentRecord, ConsentStatus, LawfulBasis
from ..exceptions import (
    ConsentNotGrantedError,
    OptOutViolationError,
    SupplierNotFoundError
)
from .jurisdictions import JurisdictionManager


logger = logging.getLogger(__name__)


class ConsentRegistry:
    """
    GDPR/CCPA/CAN-SPAM compliant consent management system.

    Features:
    - Consent record management
    - Jurisdiction-specific rules enforcement
    - Opt-out tracking and enforcement
    - Data retention compliance
    - Audit logging
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize consent registry.

        Args:
            storage_path: Path to consent records storage (JSON file)
        """
        self.storage_path = storage_path or "data/consent_registry.json"
        self.records: Dict[str, ConsentRecord] = {}
        self.jurisdiction_manager = JurisdictionManager()
        self._load_records()
        logger.info("ConsentRegistry initialized")

    def _load_records(self):
        """Load consent records from storage."""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for supplier_id, record_data in data.items():
                        # Convert datetime strings back to datetime objects
                        if record_data.get('consent_date'):
                            record_data['consent_date'] = datetime.fromisoformat(
                                record_data['consent_date']
                            )
                        if record_data.get('opt_out_date'):
                            record_data['opt_out_date'] = datetime.fromisoformat(
                                record_data['opt_out_date']
                            )
                        if record_data.get('last_contacted'):
                            record_data['last_contacted'] = datetime.fromisoformat(
                                record_data['last_contacted']
                            )

                        self.records[supplier_id] = ConsentRecord(**record_data)
                logger.info(f"Loaded {len(self.records)} consent records")
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
                self.records = {}
        else:
            logger.info("No existing consent records found")
            path.parent.mkdir(parents=True, exist_ok=True)

    def _save_records(self):
        """Save consent records to storage."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert records to JSON-serializable format
            data = {}
            for supplier_id, record in self.records.items():
                record_dict = record.model_dump()
                # Convert datetime objects to ISO strings
                if record_dict.get('consent_date'):
                    record_dict['consent_date'] = record_dict['consent_date'].isoformat()
                if record_dict.get('opt_out_date'):
                    record_dict['opt_out_date'] = record_dict['opt_out_date'].isoformat()
                if record_dict.get('last_contacted'):
                    record_dict['last_contacted'] = record_dict['last_contacted'].isoformat()
                data[supplier_id] = record_dict

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.records)} consent records")
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")

    def register_supplier(
        self,
        supplier_id: str,
        email_address: str,
        country: str,
        lawful_basis: LawfulBasis = LawfulBasis.LEGITIMATE_INTEREST,
        auto_opt_in: bool = False
    ) -> ConsentRecord:
        """
        Register new supplier in consent registry.

        Args:
            supplier_id: Unique supplier identifier
            email_address: Contact email
            country: Supplier country (ISO 3166-1)
            lawful_basis: GDPR lawful basis
            auto_opt_in: Auto opt-in (for existing business relationships)

        Returns:
            Created consent record
        """
        # Determine initial status based on jurisdiction
        jurisdiction = self.jurisdiction_manager.get_jurisdiction(country)
        rules = self.jurisdiction_manager.get_rules(country)

        if rules.requires_opt_in() and not auto_opt_in:
            initial_status = ConsentStatus.PENDING
            consent_date = None
        else:
            # Opt-out model or auto opt-in
            initial_status = ConsentStatus.OPTED_IN
            consent_date = datetime.utcnow()

        record = ConsentRecord(
            supplier_id=supplier_id,
            email_address=email_address,
            consent_status=initial_status,
            lawful_basis=lawful_basis,
            country=country,
            consent_date=consent_date,
            metadata={
                "jurisdiction": jurisdiction.value,
                "registered_at": datetime.utcnow().isoformat()
            }
        )

        self.records[supplier_id] = record
        self._save_records()

        logger.info(
            f"Registered supplier {supplier_id} with status {initial_status.value} "
            f"under {jurisdiction.value}"
        )

        return record

    def grant_consent(self, supplier_id: str) -> ConsentRecord:
        """
        Grant consent for supplier (opt-in).

        Args:
            supplier_id: Supplier identifier

        Returns:
            Updated consent record

        Raises:
            SupplierNotFoundError: If supplier not registered
        """
        if supplier_id not in self.records:
            raise SupplierNotFoundError(supplier_id)

        record = self.records[supplier_id]
        record.consent_status = ConsentStatus.OPTED_IN
        record.consent_date = datetime.utcnow()
        record.opt_out_date = None
        record.opt_out_reason = None

        self._save_records()

        logger.info(f"Consent granted for supplier {supplier_id}")

        return record

    def revoke_consent(
        self,
        supplier_id: str,
        reason: Optional[str] = None
    ) -> ConsentRecord:
        """
        Revoke consent for supplier (opt-out).

        Args:
            supplier_id: Supplier identifier
            reason: Reason for opt-out

        Returns:
            Updated consent record

        Raises:
            SupplierNotFoundError: If supplier not registered
        """
        if supplier_id not in self.records:
            raise SupplierNotFoundError(supplier_id)

        record = self.records[supplier_id]
        record.consent_status = ConsentStatus.OPTED_OUT
        record.opt_out_date = datetime.utcnow()
        record.opt_out_reason = reason

        self._save_records()

        logger.warning(
            f"Consent revoked for supplier {supplier_id}. Reason: {reason or 'Not provided'}"
        )

        return record

    def check_consent(self, supplier_id: str) -> bool:
        """
        Check if supplier has valid consent for contact.

        Args:
            supplier_id: Supplier identifier

        Returns:
            True if contact is allowed, False otherwise

        Raises:
            SupplierNotFoundError: If supplier not registered
        """
        if supplier_id not in self.records:
            raise SupplierNotFoundError(supplier_id)

        record = self.records[supplier_id]

        # Check jurisdiction-specific rules
        can_contact = self.jurisdiction_manager.can_contact(
            country_code=record.country,
            consent_status=record.consent_status,
            lawful_basis=record.lawful_basis,
            opt_out_date=record.opt_out_date
        )

        return can_contact

    def enforce_consent(self, supplier_id: str) -> ConsentRecord:
        """
        Enforce consent check (raises exception if not granted).

        Args:
            supplier_id: Supplier identifier

        Returns:
            Consent record if valid

        Raises:
            SupplierNotFoundError: If supplier not registered
            ConsentNotGrantedError: If consent not granted
            OptOutViolationError: If supplier opted out
        """
        if supplier_id not in self.records:
            raise SupplierNotFoundError(supplier_id)

        record = self.records[supplier_id]

        # Check if opted out
        if record.consent_status == ConsentStatus.OPTED_OUT:
            raise OptOutViolationError(
                supplier_id,
                record.opt_out_date.isoformat() if record.opt_out_date else "unknown"
            )

        # Check jurisdiction rules
        if not self.check_consent(supplier_id):
            jurisdiction = self.jurisdiction_manager.get_jurisdiction(record.country)
            raise ConsentNotGrantedError(supplier_id, jurisdiction.value)

        return record

    def record_contact(self, supplier_id: str):
        """
        Record that supplier was contacted.

        Args:
            supplier_id: Supplier identifier
        """
        if supplier_id in self.records:
            self.records[supplier_id].last_contacted = datetime.utcnow()
            self._save_records()
            logger.debug(f"Recorded contact for supplier {supplier_id}")

    def get_record(self, supplier_id: str) -> Optional[ConsentRecord]:
        """
        Get consent record for supplier.

        Args:
            supplier_id: Supplier identifier

        Returns:
            Consent record or None if not found
        """
        return self.records.get(supplier_id)

    def get_opted_out_suppliers(self) -> List[str]:
        """
        Get list of opted-out supplier IDs.

        Returns:
            List of supplier IDs
        """
        return [
            sid for sid, record in self.records.items()
            if record.consent_status == ConsentStatus.OPTED_OUT
        ]

    def get_pending_consent_suppliers(self) -> List[str]:
        """
        Get list of suppliers with pending consent.

        Returns:
            List of supplier IDs
        """
        return [
            sid for sid, record in self.records.items()
            if record.consent_status == ConsentStatus.PENDING
        ]

    def cleanup_expired_records(self, retention_days: int = 730):
        """
        Remove expired records per GDPR Article 17 (right to erasure).

        Args:
            retention_days: Data retention period (default: 730 days / 2 years)

        Returns:
            Number of records removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        expired_suppliers = []

        for supplier_id, record in self.records.items():
            # Check if record has expired
            last_activity = record.last_contacted or record.consent_date
            if last_activity and last_activity < cutoff_date:
                # If opted out, respect retention period
                if record.consent_status == ConsentStatus.OPTED_OUT:
                    if record.opt_out_date and record.opt_out_date < cutoff_date:
                        expired_suppliers.append(supplier_id)

        # Remove expired records
        for supplier_id in expired_suppliers:
            del self.records[supplier_id]
            logger.info(f"Removed expired record for supplier {supplier_id}")

        if expired_suppliers:
            self._save_records()

        logger.info(f"Cleaned up {len(expired_suppliers)} expired records")

        return len(expired_suppliers)

    def export_records(self, supplier_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Export consent records (for GDPR data portability).

        Args:
            supplier_ids: Optional list of supplier IDs to export

        Returns:
            List of consent record dictionaries
        """
        if supplier_ids:
            records_to_export = [
                self.records[sid] for sid in supplier_ids
                if sid in self.records
            ]
        else:
            records_to_export = list(self.records.values())

        return [record.model_dump() for record in records_to_export]

    def get_statistics(self) -> Dict[str, int]:
        """
        Get consent registry statistics.

        Returns:
            Dictionary of statistics
        """
        total = len(self.records)
        opted_in = sum(
            1 for r in self.records.values()
            if r.consent_status == ConsentStatus.OPTED_IN
        )
        opted_out = sum(
            1 for r in self.records.values()
            if r.consent_status == ConsentStatus.OPTED_OUT
        )
        pending = sum(
            1 for r in self.records.values()
            if r.consent_status == ConsentStatus.PENDING
        )

        # Count by jurisdiction
        gdpr_count = sum(
            1 for r in self.records.values()
            if r.country in ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
                             'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
                             'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']
        )
        ccpa_count = sum(
            1 for r in self.records.values()
            if r.country == 'US-CA'
        )
        can_spam_count = total - gdpr_count - ccpa_count

        return {
            "total_records": total,
            "opted_in": opted_in,
            "opted_out": opted_out,
            "pending": pending,
            "gdpr_records": gdpr_count,
            "ccpa_records": ccpa_count,
            "can_spam_records": can_spam_count,
        }
