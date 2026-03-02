# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal - Supplier Registry Engine

Thread-safe singleton engine for supplier registration, installation
management, EORI validation, verification lifecycle, and importer linking.

All state mutations are protected by a threading.RLock.  Provenance hashes
(SHA-256) are computed for every write operation and stored on the record
for downstream audit trail verification.

Reference:
  - EU CBAM Regulation 2023/956, Art. 10 (authorised CBAM declarant)
  - EU Implementing Regulation 2023/1773, Art. 4-7 (registration)

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from supplier_portal.models import (
    CBAMSector,
    EORI_PATTERN,
    Installation,
    SupplierProfile,
    SupplierSearchResult,
    SupplierStatus,
    VerificationOutcome,
    VerificationRecord,
    VerificationStatus,
    VALID_ISO_COUNTRY_CODES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class RegistryError(Exception):
    """Base exception for registry operations."""


class DuplicateSupplierError(RegistryError):
    """Raised when a supplier with the same EORI already exists."""


class SupplierNotFoundError(RegistryError):
    """Raised when a requested supplier does not exist."""


class InstallationNotFoundError(RegistryError):
    """Raised when a requested installation does not exist."""


class InvalidEORIError(RegistryError):
    """Raised when an EORI number fails validation."""


class SupplierSuspendedError(RegistryError):
    """Raised when operating on a suspended supplier."""


class LinkageError(RegistryError):
    """Raised when an importer-supplier link operation fails."""


# ============================================================================
# JSON ENCODER FOR DECIMALS
# ============================================================================


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that converts Decimal to string for hashing."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return super().default(obj)


# ============================================================================
# SUPPLIER REGISTRY ENGINE
# ============================================================================


class SupplierRegistryEngine:
    """
    Thread-safe engine for CBAM supplier registration and lifecycle management.

    Provides methods for:
      - Registering and updating suppliers
      - EORI format validation with ISO country code check
      - Installation registration and management
      - Verification status tracking and expiry checks
      - Supplier-importer data sharing links
      - Multi-criteria supplier search
      - Supplier statistics and provenance tracking

    Thread Safety:
      All public methods acquire self._lock (RLock) before mutating state.
      Read-only methods also acquire the lock for snapshot consistency.

    Example:
        >>> engine = SupplierRegistryEngine()
        >>> profile = SupplierProfile(
        ...     company_name="Steel Co.", country="CN",
        ...     address="123 Factory Rd", contact_email="info@steelco.cn",
        ... )
        >>> registered = engine.register_supplier(profile)
        >>> assert registered.status == SupplierStatus.REGISTERED
    """

    _instance: Optional["SupplierRegistryEngine"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> "SupplierRegistryEngine":
        """Singleton pattern: return existing instance if available."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize registry stores and lock."""
        if self._initialized:
            return
        self._lock = threading.RLock()
        self._suppliers: Dict[str, SupplierProfile] = {}
        self._installations: Dict[str, Installation] = {}
        self._verifications: Dict[str, List[VerificationRecord]] = {}
        self._importer_links: Dict[str, Dict[str, _ImporterLink]] = {}
        self._audit_trail: List[Dict[str, Any]] = []
        self._initialized = True
        logger.info("SupplierRegistryEngine initialized")

    # ------------------------------------------------------------------
    # Supplier Registration
    # ------------------------------------------------------------------

    def register_supplier(self, profile: SupplierProfile) -> SupplierProfile:
        """
        Register a new supplier in the CBAM Supplier Portal.

        Validates the EORI number format (if provided), checks for
        duplicate registrations, assigns a unique supplier_id, and sets
        the initial status to REGISTERED.

        Args:
            profile: Supplier profile data to register.

        Returns:
            The registered SupplierProfile with assigned supplier_id.

        Raises:
            InvalidEORIError: If the EORI number is malformed.
            DuplicateSupplierError: If a supplier with the same EORI exists.
        """
        start_time = datetime.now(timezone.utc)

        with self._lock:
            # Validate EORI if provided
            if profile.eori_number:
                if not self.validate_eori(profile.eori_number):
                    raise InvalidEORIError(
                        f"Invalid EORI format: {profile.eori_number}"
                    )
                # Check for duplicate EORI
                if self._find_supplier_by_eori(profile.eori_number) is not None:
                    raise DuplicateSupplierError(
                        f"Supplier with EORI '{profile.eori_number}' already exists"
                    )

            # Assign new supplier_id
            supplier_id = self._generate_id("SUP")
            profile = profile.model_copy(
                update={
                    "supplier_id": supplier_id,
                    "status": SupplierStatus.REGISTERED,
                    "registration_date": datetime.now(timezone.utc),
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "provenance_hash": self._compute_provenance_hash(
                        profile.model_dump(mode="json")
                    ),
                }
            )

            self._suppliers[supplier_id] = profile
            self._importer_links[supplier_id] = {}

            self._record_audit(
                action="supplier_registered",
                resource_type="supplier",
                resource_id=supplier_id,
                details={
                    "company_name": profile.company_name,
                    "country": profile.country,
                    "eori": profile.eori_number,
                },
            )

            duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            logger.info(
                "Supplier registered: %s (%s) in %.1f ms",
                supplier_id,
                profile.company_name,
                duration_ms,
            )

        return profile

    def update_supplier(
        self, supplier_id: str, updates: Dict[str, Any]
    ) -> SupplierProfile:
        """
        Update an existing supplier's profile fields.

        Args:
            supplier_id: Unique supplier identifier.
            updates: Dictionary of field names to new values.

        Returns:
            The updated SupplierProfile.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
            InvalidEORIError: If updating EORI to an invalid format.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)

            # Validate EORI if being updated
            if "eori_number" in updates and updates["eori_number"] is not None:
                if not self.validate_eori(updates["eori_number"]):
                    raise InvalidEORIError(
                        f"Invalid EORI format: {updates['eori_number']}"
                    )
                existing = self._find_supplier_by_eori(updates["eori_number"])
                if existing is not None and existing.supplier_id != supplier_id:
                    raise DuplicateSupplierError(
                        f"EORI '{updates['eori_number']}' already in use"
                    )

            # Track changes for audit
            changes: Dict[str, Tuple[Any, Any]] = {}
            allowed_fields = {
                "company_name", "eori_number", "country", "address",
                "contact_email", "cbam_sectors", "tax_id", "certifications",
                "is_active",
            }

            update_dict: Dict[str, Any] = {}
            for field, new_value in updates.items():
                if field in allowed_fields:
                    old_value = getattr(profile, field, None)
                    if old_value != new_value:
                        changes[field] = (old_value, new_value)
                        update_dict[field] = new_value

            update_dict["updated_at"] = datetime.now(timezone.utc)
            profile = profile.model_copy(update=update_dict)
            profile = profile.model_copy(
                update={
                    "provenance_hash": self._compute_provenance_hash(
                        profile.model_dump(mode="json")
                    )
                }
            )
            self._suppliers[supplier_id] = profile

            if changes:
                self._record_audit(
                    action="supplier_updated",
                    resource_type="supplier",
                    resource_id=supplier_id,
                    details={
                        "changed_fields": list(changes.keys()),
                        "changes": {
                            k: {"old": str(v[0]), "new": str(v[1])}
                            for k, v in changes.items()
                        },
                    },
                )
                logger.info(
                    "Supplier updated: %s (fields: %s)",
                    supplier_id,
                    list(changes.keys()),
                )

        return profile

    def get_supplier(self, supplier_id: str) -> SupplierProfile:
        """
        Retrieve a supplier profile by ID.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            The SupplierProfile.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            return self._get_supplier_or_raise(supplier_id)

    def search_suppliers(
        self,
        country: Optional[str] = None,
        sector: Optional[CBAMSector] = None,
        name_query: Optional[str] = None,
        verified_only: bool = False,
    ) -> List[SupplierSearchResult]:
        """
        Search for suppliers with optional filtering criteria.

        Args:
            country: ISO 3166-1 alpha-2 country code filter.
            sector: CBAM sector filter.
            name_query: Case-insensitive substring match on company name.
            verified_only: If True, only return verified suppliers.

        Returns:
            List of SupplierSearchResult matching the criteria.
        """
        with self._lock:
            results: List[SupplierSearchResult] = []

            for profile in self._suppliers.values():
                if not profile.is_active:
                    continue

                # Apply country filter
                if country and profile.country.upper() != country.upper():
                    continue

                # Apply sector filter
                if sector and profile.cbam_sectors:
                    if sector not in profile.cbam_sectors:
                        continue
                elif sector and not profile.cbam_sectors:
                    continue

                # Apply name filter
                if name_query:
                    if name_query.lower() not in profile.company_name.lower():
                        continue

                # Apply verification filter
                if verified_only and profile.status != SupplierStatus.VERIFIED:
                    continue

                # Count installations
                inst_count = sum(
                    1
                    for inst in self._installations.values()
                    if inst.supplier_id == profile.supplier_id and inst.is_active
                )

                results.append(
                    SupplierSearchResult(
                        supplier_id=profile.supplier_id,
                        company_name=profile.company_name,
                        country=profile.country,
                        sectors=profile.cbam_sectors or [],
                        verification_status=self._derive_verification_status(
                            profile
                        ),
                        installations_count=inst_count,
                        last_submission_date=None,
                    )
                )

            logger.info(
                "Supplier search: %d results (country=%s, sector=%s, query=%s)",
                len(results),
                country,
                sector,
                name_query,
            )

        return results

    # ------------------------------------------------------------------
    # EORI Validation
    # ------------------------------------------------------------------

    def validate_eori(self, eori: str) -> bool:
        """
        Validate an EORI number format.

        The EORI must consist of a 2-letter ISO 3166-1 alpha-2 country code
        followed by up to 15 alphanumeric characters.

        Args:
            eori: The EORI number to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not eori:
            return False

        eori = eori.upper().strip()

        # Check overall pattern
        if not EORI_PATTERN.match(eori):
            return False

        # Extract and validate country code
        country_code = eori[:2]
        if country_code not in VALID_ISO_COUNTRY_CODES:
            logger.warning(
                "EORI validation failed: unknown country code '%s' in '%s'",
                country_code,
                eori,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Installation Management
    # ------------------------------------------------------------------

    def register_installation(
        self, supplier_id: str, installation: Installation
    ) -> Installation:
        """
        Register a new installation for an existing supplier.

        Args:
            supplier_id: ID of the parent supplier.
            installation: Installation data to register.

        Returns:
            The registered Installation with assigned installation_id.

        Raises:
            SupplierNotFoundError: If the parent supplier does not exist.
            SupplierSuspendedError: If the supplier is suspended.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)

            if profile.status == SupplierStatus.SUSPENDED:
                raise SupplierSuspendedError(
                    f"Cannot add installation: supplier '{supplier_id}' is suspended"
                )

            installation_id = self._generate_id("INST")
            installation = installation.model_copy(
                update={
                    "installation_id": installation_id,
                    "supplier_id": supplier_id,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

            self._installations[installation_id] = installation

            # Update supplier's installation list
            updated_installations = list(profile.installations) + [installation]
            profile = profile.model_copy(
                update={
                    "installations": updated_installations,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._suppliers[supplier_id] = profile

            self._record_audit(
                action="installation_registered",
                resource_type="installation",
                resource_id=installation_id,
                details={
                    "supplier_id": supplier_id,
                    "name": installation.name,
                    "country": installation.country,
                    "type": installation.installation_type.value,
                },
            )

            logger.info(
                "Installation registered: %s (%s) for supplier %s",
                installation_id,
                installation.name,
                supplier_id,
            )

        return installation

    def update_installation(
        self, installation_id: str, updates: Dict[str, Any]
    ) -> Installation:
        """
        Update an existing installation's fields.

        Args:
            installation_id: Unique installation identifier.
            updates: Dictionary of field names to new values.

        Returns:
            The updated Installation.

        Raises:
            InstallationNotFoundError: If the installation does not exist.
        """
        with self._lock:
            installation = self._get_installation_or_raise(installation_id)

            allowed_fields = {
                "name", "address", "country", "installation_type",
                "cbam_sectors", "production_processes", "capacity_mt_per_year",
                "cn_codes", "energy_source", "monitoring_methodology",
                "is_active",
            }

            update_dict: Dict[str, Any] = {}
            for field, new_value in updates.items():
                if field in allowed_fields:
                    update_dict[field] = new_value

            update_dict["updated_at"] = datetime.now(timezone.utc)
            installation = installation.model_copy(update=update_dict)
            self._installations[installation_id] = installation

            # Update in parent supplier's list
            self._sync_installation_in_supplier(installation)

            self._record_audit(
                action="installation_updated",
                resource_type="installation",
                resource_id=installation_id,
                details={
                    "supplier_id": installation.supplier_id,
                    "updated_fields": list(update_dict.keys()),
                },
            )

            logger.info(
                "Installation updated: %s (fields: %s)",
                installation_id,
                list(update_dict.keys()),
            )

        return installation

    def get_installations(self, supplier_id: str) -> List[Installation]:
        """
        Retrieve all active installations for a supplier.

        Args:
            supplier_id: Parent supplier identifier.

        Returns:
            List of active installations.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            self._get_supplier_or_raise(supplier_id)
            return [
                inst
                for inst in self._installations.values()
                if inst.supplier_id == supplier_id and inst.is_active
            ]

    def get_installation(self, installation_id: str) -> Installation:
        """
        Retrieve a single installation by ID.

        Args:
            installation_id: Unique installation identifier.

        Returns:
            The Installation.

        Raises:
            InstallationNotFoundError: If not found.
        """
        with self._lock:
            return self._get_installation_or_raise(installation_id)

    # ------------------------------------------------------------------
    # Verification Management
    # ------------------------------------------------------------------

    def update_verification_status(
        self, supplier_id: str, verification: VerificationRecord
    ) -> None:
        """
        Update a supplier's verification status based on a verification record.

        Applies the verification outcome to the installation and the parent
        supplier. Updates status to VERIFIED (on pass), PENDING_VERIFICATION
        (on conditional), or leaves as-is (on fail).

        Args:
            supplier_id: Supplier identifier.
            verification: The verification record to apply.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
            InstallationNotFoundError: If the installation does not exist.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)
            installation = self._get_installation_or_raise(
                verification.installation_id
            )

            # Store verification record
            inst_id = verification.installation_id
            if inst_id not in self._verifications:
                self._verifications[inst_id] = []
            self._verifications[inst_id].append(verification)

            # Update installation verification status
            new_inst_status = self._map_outcome_to_verification_status(
                verification.outcome
            )
            installation = installation.model_copy(
                update={
                    "verification_status": new_inst_status,
                    "verified_until": verification.next_visit_date,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._installations[inst_id] = installation
            self._sync_installation_in_supplier(installation)

            # Update supplier status based on installation outcomes
            new_supplier_status = self._derive_supplier_status_from_installations(
                supplier_id
            )
            profile = profile.model_copy(
                update={
                    "status": new_supplier_status,
                    "verification_expiry": verification.next_visit_date,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._suppliers[supplier_id] = profile

            self._record_audit(
                action="verification_updated",
                resource_type="installation",
                resource_id=inst_id,
                details={
                    "supplier_id": supplier_id,
                    "outcome": verification.outcome.value,
                    "verifier": verification.verifier_name,
                    "new_status": new_inst_status.value,
                },
            )

            logger.info(
                "Verification updated for installation %s: %s (supplier %s -> %s)",
                inst_id,
                verification.outcome.value,
                supplier_id,
                new_supplier_status.value,
            )

    def check_verification_expiry(self, supplier_id: str) -> Dict[str, Any]:
        """
        Check if a supplier's verification is current.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with keys:
              - is_current: bool
              - days_remaining: int or None
              - expiry_date: date or None
              - installations: list of per-installation status dicts

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)
            today = date.today()

            installation_status: List[Dict[str, Any]] = []
            any_current = False

            for inst in self._installations.values():
                if inst.supplier_id != supplier_id or not inst.is_active:
                    continue

                is_current = False
                days_remaining: Optional[int] = None

                if inst.verified_until:
                    is_current = inst.verified_until > today
                    days_remaining = (inst.verified_until - today).days

                if is_current:
                    any_current = True

                installation_status.append({
                    "installation_id": inst.installation_id,
                    "name": inst.name,
                    "verification_status": inst.verification_status.value,
                    "verified_until": (
                        inst.verified_until.isoformat() if inst.verified_until else None
                    ),
                    "is_current": is_current,
                    "days_remaining": days_remaining,
                })

            # Supplier-level expiry
            supplier_current = False
            supplier_days_remaining: Optional[int] = None
            if profile.verification_expiry:
                supplier_current = profile.verification_expiry > today
                supplier_days_remaining = (
                    profile.verification_expiry - today
                ).days

            # Auto-expire if needed
            if not supplier_current and profile.status == SupplierStatus.VERIFIED:
                profile = profile.model_copy(
                    update={
                        "status": SupplierStatus.EXPIRED,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                self._suppliers[supplier_id] = profile
                logger.warning(
                    "Supplier %s verification expired, status set to EXPIRED",
                    supplier_id,
                )

            return {
                "is_current": supplier_current or any_current,
                "days_remaining": supplier_days_remaining,
                "expiry_date": (
                    profile.verification_expiry.isoformat()
                    if profile.verification_expiry
                    else None
                ),
                "supplier_status": profile.status.value,
                "installations": installation_status,
            }

    # ------------------------------------------------------------------
    # Importer Linking
    # ------------------------------------------------------------------

    def link_supplier_to_importer(
        self,
        supplier_id: str,
        importer_id: str,
        authorization_token: str,
    ) -> bool:
        """
        Establish a data sharing link between a supplier and an EU importer.

        Args:
            supplier_id: Supplier identifier.
            importer_id: EU importer identifier.
            authorization_token: Token authorizing the link.

        Returns:
            True if the link was created or already exists.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
            SupplierSuspendedError: If the supplier is suspended.
            LinkageError: If the authorization token is missing.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)

            if profile.status == SupplierStatus.SUSPENDED:
                raise SupplierSuspendedError(
                    f"Cannot link: supplier '{supplier_id}' is suspended"
                )

            if not authorization_token or len(authorization_token.strip()) < 8:
                raise LinkageError(
                    "authorization_token must be at least 8 characters"
                )

            if supplier_id not in self._importer_links:
                self._importer_links[supplier_id] = {}

            # Check if link already exists
            if importer_id in self._importer_links[supplier_id]:
                logger.info(
                    "Importer link already exists: %s -> %s",
                    supplier_id,
                    importer_id,
                )
                return True

            link = _ImporterLink(
                importer_id=importer_id,
                authorization_token_hash=hashlib.sha256(
                    authorization_token.encode("utf-8")
                ).hexdigest(),
                linked_at=datetime.now(timezone.utc),
                is_active=True,
            )
            self._importer_links[supplier_id][importer_id] = link

            # Update supplier's linked importers list
            linked = list(profile.linked_importers)
            if importer_id not in linked:
                linked.append(importer_id)
            profile = profile.model_copy(
                update={
                    "linked_importers": linked,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._suppliers[supplier_id] = profile

            self._record_audit(
                action="importer_linked",
                resource_type="supplier",
                resource_id=supplier_id,
                details={"importer_id": importer_id},
            )

            logger.info(
                "Importer linked: %s -> %s",
                supplier_id,
                importer_id,
            )

        return True

    def unlink_supplier_from_importer(
        self, supplier_id: str, importer_id: str
    ) -> bool:
        """
        Revoke the data sharing link between a supplier and an importer.

        Args:
            supplier_id: Supplier identifier.
            importer_id: EU importer identifier.

        Returns:
            True if the link was revoked, False if it did not exist.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)

            links = self._importer_links.get(supplier_id, {})
            if importer_id not in links:
                return False

            links[importer_id].is_active = False
            del links[importer_id]

            linked = [
                imp for imp in profile.linked_importers if imp != importer_id
            ]
            profile = profile.model_copy(
                update={
                    "linked_importers": linked,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._suppliers[supplier_id] = profile

            self._record_audit(
                action="importer_unlinked",
                resource_type="supplier",
                resource_id=supplier_id,
                details={"importer_id": importer_id},
            )

            logger.info(
                "Importer unlinked: %s -> %s",
                supplier_id,
                importer_id,
            )

        return True

    def get_linked_importers(self, supplier_id: str) -> List[str]:
        """
        Get list of importer IDs authorized to access this supplier's data.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            List of importer IDs.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)
            return list(profile.linked_importers)

    def is_importer_linked(self, supplier_id: str, importer_id: str) -> bool:
        """
        Check if a specific importer is linked to a supplier.

        Args:
            supplier_id: Supplier identifier.
            importer_id: Importer identifier.

        Returns:
            True if the importer has an active link.
        """
        with self._lock:
            links = self._importer_links.get(supplier_id, {})
            link = links.get(importer_id)
            return link is not None and link.is_active

    # ------------------------------------------------------------------
    # Statistics & Provenance
    # ------------------------------------------------------------------

    def get_supplier_stats(self, supplier_id: str) -> Dict[str, Any]:
        """
        Compute registration statistics for a supplier.

        Returns a summary including installation count, verification status,
        linked importer count, and audit trail size.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary of statistics.

        Raises:
            SupplierNotFoundError: If the supplier does not exist.
        """
        with self._lock:
            profile = self._get_supplier_or_raise(supplier_id)

            installations = [
                inst
                for inst in self._installations.values()
                if inst.supplier_id == supplier_id and inst.is_active
            ]

            verification_records: List[VerificationRecord] = []
            for inst in installations:
                records = self._verifications.get(inst.installation_id, [])
                verification_records.extend(records)

            audit_count = sum(
                1
                for entry in self._audit_trail
                if entry.get("resource_id") == supplier_id
                or entry.get("details", {}).get("supplier_id") == supplier_id
            )

            return {
                "supplier_id": supplier_id,
                "company_name": profile.company_name,
                "status": profile.status.value,
                "country": profile.country,
                "installations_count": len(installations),
                "installations_verified": sum(
                    1
                    for inst in installations
                    if inst.verification_status == VerificationStatus.VERIFIED
                ),
                "linked_importers_count": len(profile.linked_importers),
                "verification_records_count": len(verification_records),
                "audit_trail_entries": audit_count,
                "registration_date": profile.registration_date.isoformat(),
                "verification_expiry": (
                    profile.verification_expiry.isoformat()
                    if profile.verification_expiry
                    else None
                ),
                "provenance_hash": profile.provenance_hash,
            }

    def get_all_suppliers(self) -> List[SupplierProfile]:
        """
        Retrieve all active supplier profiles.

        Returns:
            List of all active SupplierProfile objects.
        """
        with self._lock:
            return [
                p for p in self._suppliers.values() if p.is_active
            ]

    def get_audit_trail(
        self,
        supplier_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit trail entries, optionally filtered by supplier.

        Args:
            supplier_id: Filter to a specific supplier (optional).
            limit: Maximum number of entries to return.

        Returns:
            List of audit trail dictionaries.
        """
        with self._lock:
            entries = self._audit_trail

            if supplier_id:
                entries = [
                    e
                    for e in entries
                    if e.get("resource_id") == supplier_id
                    or e.get("details", {}).get("supplier_id") == supplier_id
                ]

            # Return newest first
            return list(reversed(entries[-limit:]))

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all in-memory stores. For testing only."""
        with self._lock:
            self._suppliers.clear()
            self._installations.clear()
            self._verifications.clear()
            self._importer_links.clear()
            self._audit_trail.clear()
            logger.warning("SupplierRegistryEngine stores reset")

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_supplier_or_raise(self, supplier_id: str) -> SupplierProfile:
        """Retrieve supplier or raise SupplierNotFoundError."""
        profile = self._suppliers.get(supplier_id)
        if profile is None:
            raise SupplierNotFoundError(
                f"Supplier '{supplier_id}' not found"
            )
        return profile

    def _get_installation_or_raise(
        self, installation_id: str
    ) -> Installation:
        """Retrieve installation or raise InstallationNotFoundError."""
        installation = self._installations.get(installation_id)
        if installation is None:
            raise InstallationNotFoundError(
                f"Installation '{installation_id}' not found"
            )
        return installation

    def _find_supplier_by_eori(self, eori: str) -> Optional[SupplierProfile]:
        """Find a supplier by EORI number."""
        eori_upper = eori.upper()
        for profile in self._suppliers.values():
            if (
                profile.eori_number
                and profile.eori_number.upper() == eori_upper
                and profile.is_active
            ):
                return profile
        return None

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with the given prefix."""
        short = uuid.uuid4().hex[:12].upper()
        return f"{prefix}-{short}"

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash for the given data."""
        serialized = json.dumps(data, sort_keys=True, cls=_DecimalEncoder)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _record_audit(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an entry to the in-memory audit trail."""
        entry = {
            "audit_id": self._generate_id("AUD"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "provenance_hash": self._compute_provenance_hash(
                f"{action}:{resource_id}:{datetime.now(timezone.utc).isoformat()}"
            ),
        }
        self._audit_trail.append(entry)

    def _derive_verification_status(
        self, profile: SupplierProfile
    ) -> VerificationStatus:
        """Derive aggregate verification status for a supplier."""
        if profile.status == SupplierStatus.VERIFIED:
            return VerificationStatus.VERIFIED
        elif profile.status == SupplierStatus.PENDING_VERIFICATION:
            return VerificationStatus.PENDING
        elif profile.status == SupplierStatus.EXPIRED:
            return VerificationStatus.EXPIRED
        elif profile.status == SupplierStatus.SUSPENDED:
            return VerificationStatus.REJECTED
        return VerificationStatus.UNVERIFIED

    def _map_outcome_to_verification_status(
        self, outcome: VerificationOutcome
    ) -> VerificationStatus:
        """Map a verification outcome to a VerificationStatus."""
        mapping = {
            VerificationOutcome.PASS: VerificationStatus.VERIFIED,
            VerificationOutcome.FAIL: VerificationStatus.REJECTED,
            VerificationOutcome.CONDITIONAL: VerificationStatus.PENDING,
        }
        return mapping.get(outcome, VerificationStatus.UNVERIFIED)

    def _derive_supplier_status_from_installations(
        self, supplier_id: str
    ) -> SupplierStatus:
        """
        Derive the supplier status from the aggregate of installation
        verification statuses.
        """
        installations = [
            inst
            for inst in self._installations.values()
            if inst.supplier_id == supplier_id and inst.is_active
        ]

        if not installations:
            return SupplierStatus.REGISTERED

        all_verified = all(
            inst.verification_status == VerificationStatus.VERIFIED
            for inst in installations
        )
        any_verified = any(
            inst.verification_status == VerificationStatus.VERIFIED
            for inst in installations
        )
        any_pending = any(
            inst.verification_status == VerificationStatus.PENDING
            for inst in installations
        )

        if all_verified:
            return SupplierStatus.VERIFIED
        elif any_verified or any_pending:
            return SupplierStatus.PENDING_VERIFICATION
        return SupplierStatus.REGISTERED

    def _sync_installation_in_supplier(
        self, installation: Installation
    ) -> None:
        """Sync an updated installation back into the parent supplier's list."""
        profile = self._suppliers.get(installation.supplier_id)
        if profile is None:
            return

        updated = [
            installation if inst.installation_id == installation.installation_id else inst
            for inst in profile.installations
        ]

        # Add if not found in list
        if not any(
            inst.installation_id == installation.installation_id
            for inst in updated
        ):
            updated.append(installation)

        profile = profile.model_copy(update={"installations": updated})
        self._suppliers[installation.supplier_id] = profile


# ============================================================================
# INTERNAL DATA CLASSES
# ============================================================================


class _ImporterLink:
    """Internal data structure for tracking importer-supplier links."""

    __slots__ = (
        "importer_id",
        "authorization_token_hash",
        "linked_at",
        "is_active",
    )

    def __init__(
        self,
        importer_id: str,
        authorization_token_hash: str,
        linked_at: datetime,
        is_active: bool = True,
    ) -> None:
        self.importer_id = importer_id
        self.authorization_token_hash = authorization_token_hash
        self.linked_at = linked_at
        self.is_active = is_active
