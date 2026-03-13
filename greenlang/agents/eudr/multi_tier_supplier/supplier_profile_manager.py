# -*- coding: utf-8 -*-
"""
Supplier Profile Manager - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Engine 2 of 8: Creates, updates, searches, and manages comprehensive
supplier profiles with EUDR compliance metadata. Implements profile
completeness scoring per PRD Appendix D, profile versioning for full
audit trails, duplicate profile merging, and batch import capabilities.

Profile Data Categories (per EUDR Article 9):
    - Legal identity: Legal name, registration ID, tax ID, DUNS
    - Location: GPS coordinates, address, country (ISO 3166), admin region
    - Commodity: EUDR commodity types, annual volumes, processing capacity
    - Certification: Type, ID, validity dates (FSC, RSPO, UTZ, RA)
    - Compliance: DDS reference, deforestation-free status
    - Contact: Primary contact, compliance contact

Completeness Scoring Weights (Appendix D):
    - Legal identity: 25%
    - Location: 20%
    - Commodity: 15%
    - Certification: 15%
    - Compliance: 15%
    - Contact: 10%

Zero-Hallucination Principle:
    All scoring uses deterministic weighted formulas. Profile
    validation uses explicit rule-based checks. No LLM calls
    in any computation path.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Engine version string.
ENGINE_VERSION: str = "1.0.0"

#: Prometheus metric prefix.
METRIC_PREFIX: str = "gl_eudr_mst_"

#: Default batch size for batch operations.
DEFAULT_BATCH_SIZE: int = 1000

#: Maximum profile versions retained (EUDR requires 5 years).
MAX_PROFILE_VERSIONS: int = 100

#: Profile completeness weights per PRD Appendix D.
COMPLETENESS_WEIGHTS: Dict[str, float] = {
    "legal_identity": 0.25,
    "location": 0.20,
    "commodity": 0.15,
    "certification": 0.15,
    "compliance": 0.15,
    "contact": 0.10,
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProfileStatus(str, Enum):
    """Status of a supplier profile."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_VERIFICATION = "pending_verification"
    DEACTIVATED = "deactivated"
    ARCHIVED = "archived"


class SupplierType(str, Enum):
    """Classification of supplier entity type."""

    TRADER = "trader"
    PROCESSOR = "processor"
    REFINERY = "refinery"
    MILL = "mill"
    AGGREGATOR = "aggregator"
    COOPERATIVE = "cooperative"
    FARMER = "farmer"
    PLANTATION = "plantation"
    RANCH = "ranch"
    SAWMILL = "sawmill"
    EXPORTER = "exporter"
    IMPORTER = "importer"
    DEALER = "dealer"
    COLLECTOR = "collector"
    OTHER = "other"


class CertificationType(str, Enum):
    """EUDR-relevant certification types."""

    FSC = "fsc"
    RSPO = "rspo"
    UTZ = "utz"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    ORGANIC = "organic"
    FAIR_TRADE = "fair_trade"
    ISO_14001 = "iso_14001"
    ISCC = "iscc"
    PEFC = "pefc"
    OTHER = "other"


class EUDRCommodity(str, Enum):
    """EUDR regulated commodities (7 commodities per Article 1)."""

    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


class ChangeType(str, Enum):
    """Type of profile change for audit tracking."""

    CREATE = "create"
    UPDATE = "update"
    MERGE = "merge"
    DEACTIVATE = "deactivate"
    REACTIVATE = "reactivate"
    VERSION = "version"


# ---------------------------------------------------------------------------
# Data Classes (local, independent of models.py)
# ---------------------------------------------------------------------------


@dataclass
class CertificationRecord:
    """A supplier certification record.

    Attributes:
        certification_id: Unique certification record ID.
        certification_type: Type of certification.
        certificate_number: Certificate reference number.
        issuing_body: Name of the certification body.
        valid_from: Validity start date (ISO 8601).
        valid_until: Validity end date (ISO 8601).
        scope: Certification scope description.
        commodities_covered: List of commodities covered.
        is_active: Whether the certification is currently active.
        metadata: Additional certification metadata.
    """

    certification_id: str = ""
    certification_type: str = CertificationType.OTHER.value
    certificate_number: str = ""
    issuing_body: str = ""
    valid_from: str = ""
    valid_until: str = ""
    scope: str = ""
    commodities_covered: List[str] = field(default_factory=list)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.certification_id:
            self.certification_id = str(uuid.uuid4())


@dataclass
class ContactInfo:
    """Contact information for a supplier.

    Attributes:
        contact_id: Unique contact ID.
        contact_type: Type (primary, compliance, logistics, etc.).
        name: Contact person name.
        email: Email address.
        phone: Phone number.
        role: Role/title of the contact.
    """

    contact_id: str = ""
    contact_type: str = "primary"
    name: str = ""
    email: str = ""
    phone: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.contact_id:
            self.contact_id = str(uuid.uuid4())


@dataclass
class SupplierProfile:
    """Comprehensive supplier profile with EUDR metadata.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Legal entity name.
        trade_name: Trade/business name (if different from legal).
        registration_id: Legal entity registration number.
        tax_id: Tax identification number.
        duns_number: Dun & Bradstreet DUNS number.
        supplier_type: Classification of the supplier entity.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Administrative region or state.
        gps_latitude: GPS latitude of primary location.
        gps_longitude: GPS longitude of primary location.
        address: Physical address string.
        commodity_types: EUDR commodities handled.
        annual_volume_tonnes: Annual volume in tonnes.
        processing_capacity_tonnes: Processing capacity in tonnes/year.
        upstream_supplier_count: Known upstream supplier count.
        certifications: List of certification records.
        dds_references: List of linked DDS IDs.
        deforestation_free_status: Deforestation-free verification status.
        contacts: List of contact records.
        profile_status: Current profile status.
        tier_level: Supplier tier level relative to operator.
        risk_score: Overall risk score (0-100).
        completeness_score: Profile completeness (0-100).
        version: Profile version number.
        created_at: Creation timestamp (ISO 8601).
        updated_at: Last update timestamp (ISO 8601).
        deactivated_at: Deactivation timestamp if applicable.
        deactivation_reason: Reason for deactivation.
        metadata: Additional key-value metadata.
    """

    supplier_id: str = ""
    legal_name: str = ""
    trade_name: str = ""
    registration_id: str = ""
    tax_id: str = ""
    duns_number: str = ""
    supplier_type: str = SupplierType.OTHER.value
    country_code: str = ""
    region: str = ""
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    address: str = ""
    commodity_types: List[str] = field(default_factory=list)
    annual_volume_tonnes: Optional[float] = None
    processing_capacity_tonnes: Optional[float] = None
    upstream_supplier_count: Optional[int] = None
    certifications: List[CertificationRecord] = field(
        default_factory=list
    )
    dds_references: List[str] = field(default_factory=list)
    deforestation_free_status: str = "unverified"
    contacts: List[ContactInfo] = field(default_factory=list)
    profile_status: str = ProfileStatus.PENDING_VERIFICATION.value
    tier_level: Optional[int] = None
    risk_score: Optional[float] = None
    completeness_score: float = 0.0
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    deactivated_at: str = ""
    deactivation_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamps if not provided."""
        if not self.supplier_id:
            self.supplier_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class ProfileVersion:
    """A versioned snapshot of a supplier profile.

    Attributes:
        version_id: Unique version identifier.
        supplier_id: ID of the supplier this version belongs to.
        version_number: Sequential version number.
        snapshot: Deep copy of the profile at this version.
        change_type: Type of change that created this version.
        change_reason: Human-readable reason for the change.
        changed_fields: List of fields that changed.
        changed_by: Actor who made the change.
        created_at: Timestamp when this version was created.
        provenance_hash: SHA-256 hash of the version data.
    """

    version_id: str = ""
    supplier_id: str = ""
    version_number: int = 0
    snapshot: Dict[str, Any] = field(default_factory=dict)
    change_type: str = ChangeType.CREATE.value
    change_reason: str = ""
    changed_fields: List[str] = field(default_factory=list)
    changed_by: str = "system"
    created_at: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.version_id:
            self.version_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SearchCriteria:
    """Criteria for searching supplier profiles.

    Attributes:
        name: Supplier name (partial match).
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity filter.
        supplier_type: Supplier type filter.
        tier_level: Tier level filter.
        profile_status: Profile status filter.
        min_completeness: Minimum completeness score.
        max_risk_score: Maximum risk score.
        has_certification: Whether supplier must have certifications.
        certification_type: Specific certification type filter.
        has_gps: Whether supplier must have GPS coordinates.
        has_dds: Whether supplier must have DDS references.
        limit: Maximum results to return.
        offset: Offset for pagination.
    """

    name: Optional[str] = None
    country_code: Optional[str] = None
    commodity: Optional[str] = None
    supplier_type: Optional[str] = None
    tier_level: Optional[int] = None
    profile_status: Optional[str] = None
    min_completeness: Optional[float] = None
    max_risk_score: Optional[float] = None
    has_certification: Optional[bool] = None
    certification_type: Optional[str] = None
    has_gps: Optional[bool] = None
    has_dds: Optional[bool] = None
    limit: int = 100
    offset: int = 0


@dataclass
class ProfileChangeLog:
    """A record of a change made to a supplier profile.

    Attributes:
        change_id: Unique change identifier.
        supplier_id: ID of the affected supplier.
        change_type: Type of change.
        changed_fields: Fields that were changed.
        old_values: Previous values of changed fields.
        new_values: New values of changed fields.
        change_reason: Reason for the change.
        changed_by: Actor who made the change.
        timestamp: When the change was made.
        provenance_hash: SHA-256 hash.
    """

    change_id: str = ""
    supplier_id: str = ""
    change_type: str = ChangeType.UPDATE.value
    changed_fields: List[str] = field(default_factory=list)
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    change_reason: str = ""
    changed_by: str = "system"
    timestamp: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.change_id:
            self.change_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ProfileOperationResult:
    """Result of a profile management operation.

    Attributes:
        success: Whether the operation succeeded.
        supplier_id: ID of the affected supplier.
        operation: Operation type (create, update, etc.).
        profile: The resulting profile (if successful).
        completeness_score: Computed completeness score.
        change_log: Change log entry (if applicable).
        version: Profile version created (if applicable).
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        errors: List of errors encountered.
        warnings: List of warnings generated.
        timestamp: Result generation timestamp.
    """

    success: bool = True
    supplier_id: str = ""
    operation: str = ""
    profile: Optional[SupplierProfile] = None
    completeness_score: float = 0.0
    change_log: Optional[ProfileChangeLog] = None
    version: Optional[ProfileVersion] = None
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class BatchProfileResult:
    """Result of a batch profile operation.

    Attributes:
        batch_id: Unique batch identifier.
        total_input: Total profiles in the input.
        total_created: Successfully created profiles.
        total_failed: Failed profile creations.
        results: Individual operation results.
        processing_time_ms: Total processing duration.
        provenance_hash: SHA-256 provenance hash.
        errors: Batch-level errors.
        timestamp: Result generation timestamp.
    """

    batch_id: str = ""
    total_input: int = 0
    total_created: int = 0
    total_failed: int = 0
    results: List[ProfileOperationResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _compute_provenance_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash for any serializable data.

    Args:
        data: Data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    try:
        if hasattr(data, "__dict__"):
            serialized = json.dumps(
                data.__dict__, sort_keys=True, default=str
            )
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _profile_to_dict(profile: SupplierProfile) -> Dict[str, Any]:
    """Convert a SupplierProfile to a dictionary for serialization.

    Args:
        profile: SupplierProfile instance.

    Returns:
        Dictionary representation.
    """
    result = {}
    for fld_name in profile.__dataclass_fields__:
        value = getattr(profile, fld_name)
        if isinstance(value, list):
            serialized_list = []
            for item in value:
                if hasattr(item, "__dict__"):
                    serialized_list.append(
                        {k: v for k, v in item.__dict__.items()}
                    )
                else:
                    serialized_list.append(item)
            result[fld_name] = serialized_list
        else:
            result[fld_name] = value
    return result


# ---------------------------------------------------------------------------
# SupplierProfileManager
# ---------------------------------------------------------------------------


class SupplierProfileManager:
    """Engine 2: Manages supplier profiles with EUDR compliance metadata.

    Provides full CRUD operations for supplier profiles with change
    tracking, profile versioning, completeness scoring, duplicate
    merging, and batch import capabilities. All operations produce
    SHA-256 provenance hashes for audit trails.

    Attributes:
        _profiles: In-memory store of supplier profiles (keyed by ID).
        _versions: In-memory store of profile versions (keyed by supplier ID).
        _change_logs: In-memory store of change logs.
        _profile_count: Running count for metrics.
        _operation_count: Running operation count for metrics.

    Example:
        >>> manager = SupplierProfileManager()
        >>> result = manager.create_profile({
        ...     "legal_name": "Ghana Cocoa Coop",
        ...     "country_code": "GH",
        ...     "commodity_types": ["cocoa"],
        ... })
        >>> assert result.success
        >>> assert result.completeness_score > 0
    """

    def __init__(self) -> None:
        """Initialize SupplierProfileManager."""
        self._profiles: Dict[str, SupplierProfile] = {}
        self._versions: Dict[str, List[ProfileVersion]] = {}
        self._change_logs: List[ProfileChangeLog] = []
        self._profile_count: int = 0
        self._operation_count: int = 0

        logger.info(
            "SupplierProfileManager initialized: version=%s",
            ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_profile(
        self,
        supplier_data: Dict[str, Any],
        changed_by: str = "system",
    ) -> ProfileOperationResult:
        """Create a new supplier profile with validation.

        Validates all input fields, creates the profile, computes
        the completeness score, creates an initial version snapshot,
        and generates a provenance hash.

        Args:
            supplier_data: Dictionary with supplier profile fields.
            changed_by: Actor creating the profile.

        Returns:
            ProfileOperationResult with the created profile.
        """
        start_time = time.monotonic()
        errors: List[str] = []
        warnings: List[str] = []

        logger.info(
            "Creating supplier profile: name=%s, country=%s",
            supplier_data.get("legal_name", "unknown"),
            supplier_data.get("country_code", "unknown"),
        )

        # Validate required fields
        legal_name = str(
            supplier_data.get("legal_name", "")
        ).strip()
        if not legal_name:
            errors.append("legal_name is required")

        if errors:
            return self._build_operation_result(
                success=False,
                operation="create",
                errors=errors,
                start_time=start_time,
            )

        # Build profile from data
        profile = self._build_profile_from_dict(supplier_data)

        # Validate profile
        validation_errors = self._validate_profile(profile)
        if validation_errors:
            warnings.extend(validation_errors)

        # Calculate completeness
        profile.completeness_score = self.calculate_completeness(
            profile
        )

        # Store profile
        self._profiles[profile.supplier_id] = profile
        self._profile_count += 1
        self._operation_count += 1

        # Create initial version
        version = self.version_profile(
            profile,
            change_type=ChangeType.CREATE.value,
            change_reason="Initial profile creation",
            changed_by=changed_by,
        )

        # Create change log
        change_log = ProfileChangeLog(
            supplier_id=profile.supplier_id,
            change_type=ChangeType.CREATE.value,
            changed_fields=list(supplier_data.keys()),
            new_values=supplier_data,
            change_reason="Profile created",
            changed_by=changed_by,
        )
        change_log.provenance_hash = _compute_provenance_hash(
            change_log
        )
        self._change_logs.append(change_log)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Profile created: supplier_id=%s, name=%s, "
            "completeness=%.1f, version=%d, duration_ms=%.2f",
            profile.supplier_id,
            profile.legal_name,
            profile.completeness_score,
            profile.version,
            elapsed_ms,
        )

        return ProfileOperationResult(
            success=True,
            supplier_id=profile.supplier_id,
            operation="create",
            profile=profile,
            completeness_score=profile.completeness_score,
            change_log=change_log,
            version=version,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(profile),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_profile(
        self,
        supplier_id: str,
        updates: Dict[str, Any],
        change_reason: str = "",
        changed_by: str = "system",
    ) -> ProfileOperationResult:
        """Update an existing supplier profile with change tracking.

        Records old values, applies updates, recalculates completeness,
        creates a new version snapshot, and logs the change.

        Args:
            supplier_id: ID of the supplier to update.
            updates: Dictionary of fields to update.
            change_reason: Human-readable reason for the change.
            changed_by: Actor making the change.

        Returns:
            ProfileOperationResult with the updated profile.
        """
        start_time = time.monotonic()

        logger.info(
            "Updating profile: supplier_id=%s, fields=%s",
            supplier_id,
            list(updates.keys()),
        )

        profile = self._profiles.get(supplier_id)
        if profile is None:
            return self._build_operation_result(
                success=False,
                supplier_id=supplier_id,
                operation="update",
                errors=[f"Profile not found: {supplier_id}"],
                start_time=start_time,
            )

        if profile.profile_status == ProfileStatus.DEACTIVATED.value:
            return self._build_operation_result(
                success=False,
                supplier_id=supplier_id,
                operation="update",
                errors=[
                    f"Cannot update deactivated profile: {supplier_id}"
                ],
                start_time=start_time,
            )

        # Record old values for changed fields
        old_values: Dict[str, Any] = {}
        changed_fields: List[str] = []

        for field_name, new_value in updates.items():
            if hasattr(profile, field_name):
                old_value = getattr(profile, field_name)
                if old_value != new_value:
                    old_values[field_name] = old_value
                    changed_fields.append(field_name)
                    setattr(profile, field_name, new_value)

        if not changed_fields:
            return self._build_operation_result(
                success=True,
                supplier_id=supplier_id,
                operation="update",
                profile=profile,
                warnings=["No fields changed"],
                start_time=start_time,
            )

        # Update timestamp and version
        profile.updated_at = _utcnow_iso()
        profile.version += 1

        # Recalculate completeness
        profile.completeness_score = self.calculate_completeness(
            profile
        )

        # Create version snapshot
        version = self.version_profile(
            profile,
            change_type=ChangeType.UPDATE.value,
            change_reason=change_reason or "Profile updated",
            changed_fields=changed_fields,
            changed_by=changed_by,
        )

        # Create change log
        change_log = ProfileChangeLog(
            supplier_id=supplier_id,
            change_type=ChangeType.UPDATE.value,
            changed_fields=changed_fields,
            old_values=old_values,
            new_values={k: updates[k] for k in changed_fields},
            change_reason=change_reason or "Profile updated",
            changed_by=changed_by,
        )
        change_log.provenance_hash = _compute_provenance_hash(
            change_log
        )
        self._change_logs.append(change_log)
        self._operation_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Profile updated: supplier_id=%s, changed=%s, "
            "completeness=%.1f, version=%d, duration_ms=%.2f",
            supplier_id,
            changed_fields,
            profile.completeness_score,
            profile.version,
            elapsed_ms,
        )

        return ProfileOperationResult(
            success=True,
            supplier_id=supplier_id,
            operation="update",
            profile=profile,
            completeness_score=profile.completeness_score,
            change_log=change_log,
            version=version,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(profile),
        )

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get_profile(
        self, supplier_id: str
    ) -> Optional[SupplierProfile]:
        """Retrieve a supplier profile by ID.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            SupplierProfile if found, None otherwise.
        """
        profile = self._profiles.get(supplier_id)
        if profile is None:
            logger.debug(
                "Profile not found: supplier_id=%s", supplier_id
            )
        return profile

    # ------------------------------------------------------------------
    # Deactivate
    # ------------------------------------------------------------------

    def deactivate_profile(
        self,
        supplier_id: str,
        reason: str,
        changed_by: str = "system",
    ) -> ProfileOperationResult:
        """Soft-deactivate a supplier profile.

        Sets the profile status to DEACTIVATED, records the reason
        and timestamp, creates a version snapshot, and logs the change.
        The profile data is retained for EUDR Article 14 audit
        requirements (5-year retention).

        Args:
            supplier_id: ID of the supplier to deactivate.
            reason: Reason for deactivation.
            changed_by: Actor performing the deactivation.

        Returns:
            ProfileOperationResult with the deactivated profile.
        """
        start_time = time.monotonic()

        logger.info(
            "Deactivating profile: supplier_id=%s, reason=%s",
            supplier_id,
            reason,
        )

        profile = self._profiles.get(supplier_id)
        if profile is None:
            return self._build_operation_result(
                success=False,
                supplier_id=supplier_id,
                operation="deactivate",
                errors=[f"Profile not found: {supplier_id}"],
                start_time=start_time,
            )

        if profile.profile_status == ProfileStatus.DEACTIVATED.value:
            return self._build_operation_result(
                success=False,
                supplier_id=supplier_id,
                operation="deactivate",
                errors=[f"Profile already deactivated: {supplier_id}"],
                start_time=start_time,
            )

        # Record old status
        old_status = profile.profile_status
        now = _utcnow_iso()

        profile.profile_status = ProfileStatus.DEACTIVATED.value
        profile.deactivated_at = now
        profile.deactivation_reason = reason
        profile.updated_at = now
        profile.version += 1

        # Create version snapshot
        version = self.version_profile(
            profile,
            change_type=ChangeType.DEACTIVATE.value,
            change_reason=reason,
            changed_fields=[
                "profile_status",
                "deactivated_at",
                "deactivation_reason",
            ],
            changed_by=changed_by,
        )

        # Create change log
        change_log = ProfileChangeLog(
            supplier_id=supplier_id,
            change_type=ChangeType.DEACTIVATE.value,
            changed_fields=[
                "profile_status",
                "deactivated_at",
                "deactivation_reason",
            ],
            old_values={"profile_status": old_status},
            new_values={
                "profile_status": ProfileStatus.DEACTIVATED.value,
                "deactivated_at": now,
                "deactivation_reason": reason,
            },
            change_reason=reason,
            changed_by=changed_by,
        )
        change_log.provenance_hash = _compute_provenance_hash(
            change_log
        )
        self._change_logs.append(change_log)
        self._operation_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Profile deactivated: supplier_id=%s, "
            "old_status=%s, reason=%s, duration_ms=%.2f",
            supplier_id,
            old_status,
            reason,
            elapsed_ms,
        )

        return ProfileOperationResult(
            success=True,
            supplier_id=supplier_id,
            operation="deactivate",
            profile=profile,
            completeness_score=profile.completeness_score,
            change_log=change_log,
            version=version,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(profile),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_profiles(
        self, criteria: SearchCriteria
    ) -> List[SupplierProfile]:
        """Search supplier profiles by multiple criteria.

        Applies all specified criteria as AND filters. Supports
        partial name matching, country, commodity, tier level,
        status, completeness, risk score, and certification filters.

        Args:
            criteria: Search criteria to apply.

        Returns:
            List of matching SupplierProfile instances.
        """
        start_time = time.monotonic()

        logger.info(
            "Searching profiles: name=%s, country=%s, "
            "commodity=%s, status=%s, limit=%d",
            criteria.name,
            criteria.country_code,
            criteria.commodity,
            criteria.profile_status,
            criteria.limit,
        )

        results: List[SupplierProfile] = []

        for profile in self._profiles.values():
            if self._matches_criteria(profile, criteria):
                results.append(profile)

        # Apply pagination
        total_matches = len(results)
        results = results[
            criteria.offset:criteria.offset + criteria.limit
        ]

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Search completed: total_matches=%d, returned=%d, "
            "duration_ms=%.2f",
            total_matches,
            len(results),
            elapsed_ms,
        )

        return results

    def _matches_criteria(
        self, profile: SupplierProfile, criteria: SearchCriteria
    ) -> bool:
        """Check if a profile matches all search criteria.

        Args:
            profile: Profile to check.
            criteria: Criteria to match against.

        Returns:
            True if all specified criteria are satisfied.
        """
        if criteria.name is not None:
            name_lower = criteria.name.lower()
            if (
                name_lower not in profile.legal_name.lower()
                and name_lower not in profile.trade_name.lower()
            ):
                return False

        if criteria.country_code is not None:
            if (
                profile.country_code.upper()
                != criteria.country_code.upper()
            ):
                return False

        if criteria.commodity is not None:
            if criteria.commodity not in profile.commodity_types:
                return False

        if criteria.supplier_type is not None:
            if profile.supplier_type != criteria.supplier_type:
                return False

        if criteria.tier_level is not None:
            if profile.tier_level != criteria.tier_level:
                return False

        if criteria.profile_status is not None:
            if profile.profile_status != criteria.profile_status:
                return False

        if criteria.min_completeness is not None:
            if profile.completeness_score < criteria.min_completeness:
                return False

        if criteria.max_risk_score is not None:
            if (
                profile.risk_score is not None
                and profile.risk_score > criteria.max_risk_score
            ):
                return False

        if criteria.has_certification is True:
            if not profile.certifications:
                return False

        if criteria.certification_type is not None:
            has_cert = any(
                c.certification_type == criteria.certification_type
                for c in profile.certifications
            )
            if not has_cert:
                return False

        if criteria.has_gps is True:
            if (
                profile.gps_latitude is None
                or profile.gps_longitude is None
            ):
                return False

        if criteria.has_dds is True:
            if not profile.dds_references:
                return False

        return True

    # ------------------------------------------------------------------
    # Completeness scoring
    # ------------------------------------------------------------------

    def calculate_completeness(
        self, profile: SupplierProfile
    ) -> float:
        """Score profile completeness 0-100 with weighted field categories.

        Uses the weight categories from PRD Appendix D:
        - Legal identity (25%): legal_name, registration_id, country_code
        - Location (20%): GPS coordinates, address, admin region
        - Commodity (15%): commodity_types, volumes, capacity
        - Certification (15%): cert type, ID, validity
        - Compliance (15%): DDS reference, deforestation status
        - Contact (10%): primary contact, compliance contact

        Args:
            profile: SupplierProfile to score.

        Returns:
            Completeness score between 0.0 and 100.0.
        """
        category_scores: Dict[str, float] = {}

        # Legal identity (25%)
        legal_fields = [
            bool(profile.legal_name),
            bool(profile.registration_id),
            bool(profile.country_code),
            bool(profile.tax_id),
            bool(profile.duns_number),
        ]
        category_scores["legal_identity"] = (
            sum(legal_fields) / len(legal_fields)
        )

        # Location (20%)
        location_fields = [
            profile.gps_latitude is not None,
            profile.gps_longitude is not None,
            bool(profile.address),
            bool(profile.region),
        ]
        category_scores["location"] = (
            sum(location_fields) / len(location_fields)
        )

        # Commodity (15%)
        commodity_fields = [
            bool(profile.commodity_types),
            profile.annual_volume_tonnes is not None,
            profile.processing_capacity_tonnes is not None,
            profile.upstream_supplier_count is not None,
        ]
        category_scores["commodity"] = (
            sum(commodity_fields) / len(commodity_fields)
        )

        # Certification (15%)
        cert_score = 0.0
        if profile.certifications:
            cert_fields = [
                any(c.certification_type for c in profile.certifications),
                any(c.certificate_number for c in profile.certifications),
                any(c.valid_from for c in profile.certifications),
                any(c.valid_until for c in profile.certifications),
            ]
            cert_score = sum(cert_fields) / len(cert_fields)
        category_scores["certification"] = cert_score

        # Compliance (15%)
        compliance_fields = [
            bool(profile.dds_references),
            profile.deforestation_free_status != "unverified",
            profile.risk_score is not None,
        ]
        category_scores["compliance"] = (
            sum(compliance_fields) / len(compliance_fields)
        )

        # Contact (10%)
        contact_score = 0.0
        if profile.contacts:
            has_primary = any(
                c.contact_type == "primary" for c in profile.contacts
            )
            has_compliance = any(
                c.contact_type == "compliance"
                for c in profile.contacts
            )
            contact_fields = [
                has_primary,
                has_compliance,
                any(c.email for c in profile.contacts),
                any(c.phone for c in profile.contacts),
            ]
            contact_score = sum(contact_fields) / len(contact_fields)
        category_scores["contact"] = contact_score

        # Calculate weighted total
        total_score = 0.0
        for category, weight in COMPLETENESS_WEIGHTS.items():
            cat_score = category_scores.get(category, 0.0)
            total_score += cat_score * weight * 100.0

        final_score = max(0.0, min(100.0, total_score))

        logger.debug(
            "Completeness scored: supplier_id=%s, "
            "legal=%.2f, location=%.2f, commodity=%.2f, "
            "certification=%.2f, compliance=%.2f, "
            "contact=%.2f, total=%.1f",
            profile.supplier_id,
            category_scores["legal_identity"],
            category_scores["location"],
            category_scores["commodity"],
            category_scores["certification"],
            category_scores["compliance"],
            category_scores["contact"],
            final_score,
        )

        return round(final_score, 1)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_profiles(
        self,
        primary_id: str,
        duplicate_id: str,
        changed_by: str = "system",
    ) -> ProfileOperationResult:
        """Merge a duplicate profile into a primary profile.

        Copies non-empty fields from the duplicate into the primary
        where the primary has empty/missing values. The duplicate
        profile is then deactivated with a merge reference.

        Args:
            primary_id: ID of the canonical (surviving) profile.
            duplicate_id: ID of the duplicate (to be deactivated).
            changed_by: Actor performing the merge.

        Returns:
            ProfileOperationResult with the merged primary profile.
        """
        start_time = time.monotonic()

        logger.info(
            "Merging profiles: primary=%s, duplicate=%s",
            primary_id,
            duplicate_id,
        )

        primary = self._profiles.get(primary_id)
        duplicate = self._profiles.get(duplicate_id)

        if primary is None:
            return self._build_operation_result(
                success=False,
                supplier_id=primary_id,
                operation="merge",
                errors=[f"Primary profile not found: {primary_id}"],
                start_time=start_time,
            )

        if duplicate is None:
            return self._build_operation_result(
                success=False,
                supplier_id=primary_id,
                operation="merge",
                errors=[
                    f"Duplicate profile not found: {duplicate_id}"
                ],
                start_time=start_time,
            )

        if primary_id == duplicate_id:
            return self._build_operation_result(
                success=False,
                supplier_id=primary_id,
                operation="merge",
                errors=["Cannot merge a profile with itself"],
                start_time=start_time,
            )

        # Merge non-empty fields from duplicate into primary
        merged_fields: List[str] = []

        merge_candidates = [
            "trade_name", "registration_id", "tax_id",
            "duns_number", "region", "address",
        ]
        for fld in merge_candidates:
            primary_val = getattr(primary, fld, "")
            duplicate_val = getattr(duplicate, fld, "")
            if not primary_val and duplicate_val:
                setattr(primary, fld, duplicate_val)
                merged_fields.append(fld)

        # Merge GPS if missing
        if primary.gps_latitude is None and duplicate.gps_latitude is not None:
            primary.gps_latitude = duplicate.gps_latitude
            merged_fields.append("gps_latitude")
        if primary.gps_longitude is None and duplicate.gps_longitude is not None:
            primary.gps_longitude = duplicate.gps_longitude
            merged_fields.append("gps_longitude")

        # Merge commodity types (union)
        if duplicate.commodity_types:
            original_commodities = set(primary.commodity_types)
            for commodity in duplicate.commodity_types:
                if commodity not in original_commodities:
                    primary.commodity_types.append(commodity)
                    if "commodity_types" not in merged_fields:
                        merged_fields.append("commodity_types")

        # Merge certifications (append non-duplicates)
        if duplicate.certifications:
            existing_cert_numbers = {
                c.certificate_number for c in primary.certifications
            }
            for cert in duplicate.certifications:
                if cert.certificate_number not in existing_cert_numbers:
                    primary.certifications.append(cert)
                    if "certifications" not in merged_fields:
                        merged_fields.append("certifications")

        # Merge contacts (append non-duplicates)
        if duplicate.contacts:
            existing_emails = {
                c.email for c in primary.contacts if c.email
            }
            for contact in duplicate.contacts:
                if contact.email and contact.email not in existing_emails:
                    primary.contacts.append(contact)
                    if "contacts" not in merged_fields:
                        merged_fields.append("contacts")

        # Merge DDS references (union)
        if duplicate.dds_references:
            existing_dds = set(primary.dds_references)
            for dds in duplicate.dds_references:
                if dds not in existing_dds:
                    primary.dds_references.append(dds)
                    if "dds_references" not in merged_fields:
                        merged_fields.append("dds_references")

        # Merge volumes (take larger if available)
        if (
            duplicate.annual_volume_tonnes is not None
            and (
                primary.annual_volume_tonnes is None
                or duplicate.annual_volume_tonnes
                > primary.annual_volume_tonnes
            )
        ):
            primary.annual_volume_tonnes = (
                duplicate.annual_volume_tonnes
            )
            merged_fields.append("annual_volume_tonnes")

        # Update primary metadata
        primary.updated_at = _utcnow_iso()
        primary.version += 1
        primary.completeness_score = self.calculate_completeness(
            primary
        )
        primary.metadata["merged_from"] = duplicate_id

        # Version the merged profile
        version = self.version_profile(
            primary,
            change_type=ChangeType.MERGE.value,
            change_reason=f"Merged from duplicate {duplicate_id}",
            changed_fields=merged_fields,
            changed_by=changed_by,
        )

        # Deactivate duplicate
        self.deactivate_profile(
            duplicate_id,
            reason=f"Merged into primary profile {primary_id}",
            changed_by=changed_by,
        )

        # Log merge
        change_log = ProfileChangeLog(
            supplier_id=primary_id,
            change_type=ChangeType.MERGE.value,
            changed_fields=merged_fields,
            new_values={"merged_from": duplicate_id},
            change_reason=f"Merged from {duplicate_id}",
            changed_by=changed_by,
        )
        change_log.provenance_hash = _compute_provenance_hash(
            change_log
        )
        self._change_logs.append(change_log)
        self._operation_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Profiles merged: primary=%s, duplicate=%s, "
            "merged_fields=%s, completeness=%.1f, "
            "duration_ms=%.2f",
            primary_id,
            duplicate_id,
            merged_fields,
            primary.completeness_score,
            elapsed_ms,
        )

        return ProfileOperationResult(
            success=True,
            supplier_id=primary_id,
            operation="merge",
            profile=primary,
            completeness_score=primary.completeness_score,
            change_log=change_log,
            version=version,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(primary),
        )

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def version_profile(
        self,
        profile: SupplierProfile,
        change_type: str = ChangeType.VERSION.value,
        change_reason: str = "",
        changed_fields: Optional[List[str]] = None,
        changed_by: str = "system",
    ) -> ProfileVersion:
        """Create a versioned snapshot of a profile for audit.

        Creates an immutable snapshot of the profile at the current
        point in time, with a SHA-256 provenance hash for integrity
        verification.

        Args:
            profile: Profile to snapshot.
            change_type: Type of change triggering the version.
            change_reason: Human-readable reason.
            changed_fields: List of fields that changed.
            changed_by: Actor who made the change.

        Returns:
            ProfileVersion with the snapshot and provenance hash.
        """
        supplier_id = profile.supplier_id

        # Get current version count
        if supplier_id not in self._versions:
            self._versions[supplier_id] = []
        version_number = len(self._versions[supplier_id]) + 1

        # Create snapshot
        snapshot = _profile_to_dict(profile)

        # Create version
        version = ProfileVersion(
            supplier_id=supplier_id,
            version_number=version_number,
            snapshot=snapshot,
            change_type=change_type,
            change_reason=change_reason,
            changed_fields=changed_fields or [],
            changed_by=changed_by,
        )
        version.provenance_hash = _compute_provenance_hash(snapshot)

        # Store version (respecting max retention)
        versions = self._versions[supplier_id]
        versions.append(version)
        if len(versions) > MAX_PROFILE_VERSIONS:
            self._versions[supplier_id] = versions[
                -MAX_PROFILE_VERSIONS:
            ]

        logger.debug(
            "Profile versioned: supplier_id=%s, version=%d, "
            "change_type=%s",
            supplier_id,
            version_number,
            change_type,
        )

        return version

    # ------------------------------------------------------------------
    # Batch create
    # ------------------------------------------------------------------

    def batch_create(
        self,
        profiles: List[Dict[str, Any]],
        changed_by: str = "system",
    ) -> BatchProfileResult:
        """Batch import of supplier profiles.

        Creates multiple profiles in sequence, tracking successes
        and failures independently.

        Args:
            profiles: List of supplier data dictionaries.
            changed_by: Actor performing the batch import.

        Returns:
            BatchProfileResult with individual results.
        """
        start_time = time.monotonic()
        batch_id = str(uuid.uuid4())

        logger.info(
            "Starting batch profile create: batch_id=%s, "
            "count=%d",
            batch_id,
            len(profiles),
        )

        results: List[ProfileOperationResult] = []
        total_created = 0
        total_failed = 0
        batch_errors: List[str] = []

        for idx, profile_data in enumerate(profiles):
            try:
                result = self.create_profile(
                    profile_data, changed_by=changed_by
                )
                results.append(result)
                if result.success:
                    total_created += 1
                else:
                    total_failed += 1
                    batch_errors.extend(result.errors)
            except Exception as exc:
                total_failed += 1
                error_msg = (
                    f"Batch item {idx} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                batch_errors.append(error_msg)
                logger.warning(error_msg)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        provenance_data = {
            "batch_id": batch_id,
            "total_input": len(profiles),
            "total_created": total_created,
            "total_failed": total_failed,
            "timestamp": _utcnow_iso(),
        }

        batch_result = BatchProfileResult(
            batch_id=batch_id,
            total_input=len(profiles),
            total_created=total_created,
            total_failed=total_failed,
            results=results,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(provenance_data),
            errors=batch_errors,
        )

        logger.info(
            "Batch profile create completed: batch_id=%s, "
            "created=%d, failed=%d, duration_ms=%.2f",
            batch_id,
            total_created,
            total_failed,
            elapsed_ms,
        )

        return batch_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profile_from_dict(
        self, data: Dict[str, Any]
    ) -> SupplierProfile:
        """Build a SupplierProfile from a dictionary.

        Args:
            data: Dictionary with profile fields.

        Returns:
            SupplierProfile instance.
        """
        # Handle certifications
        certs: List[CertificationRecord] = []
        for cert_data in data.get("certifications", []):
            if isinstance(cert_data, dict):
                certs.append(CertificationRecord(
                    certification_type=str(
                        cert_data.get("certification_type", "other")
                    ),
                    certificate_number=str(
                        cert_data.get("certificate_number", "")
                    ),
                    issuing_body=str(
                        cert_data.get("issuing_body", "")
                    ),
                    valid_from=str(cert_data.get("valid_from", "")),
                    valid_until=str(cert_data.get("valid_until", "")),
                    scope=str(cert_data.get("scope", "")),
                    commodities_covered=cert_data.get(
                        "commodities_covered", []
                    ),
                ))
            elif isinstance(cert_data, CertificationRecord):
                certs.append(cert_data)

        # Handle contacts
        contacts: List[ContactInfo] = []
        for contact_data in data.get("contacts", []):
            if isinstance(contact_data, dict):
                contacts.append(ContactInfo(
                    contact_type=str(
                        contact_data.get("contact_type", "primary")
                    ),
                    name=str(contact_data.get("name", "")),
                    email=str(contact_data.get("email", "")),
                    phone=str(contact_data.get("phone", "")),
                    role=str(contact_data.get("role", "")),
                ))
            elif isinstance(contact_data, ContactInfo):
                contacts.append(contact_data)

        return SupplierProfile(
            supplier_id=str(
                data.get("supplier_id", "")
            ) or str(uuid.uuid4()),
            legal_name=str(data.get("legal_name", "")).strip(),
            trade_name=str(data.get("trade_name", "")).strip(),
            registration_id=str(
                data.get("registration_id", "")
            ).strip(),
            tax_id=str(data.get("tax_id", "")).strip(),
            duns_number=str(data.get("duns_number", "")).strip(),
            supplier_type=str(
                data.get("supplier_type", SupplierType.OTHER.value)
            ),
            country_code=str(
                data.get("country_code", "")
            ).strip().upper(),
            region=str(data.get("region", "")).strip(),
            gps_latitude=self._safe_float(data.get("gps_latitude")),
            gps_longitude=self._safe_float(data.get("gps_longitude")),
            address=str(data.get("address", "")).strip(),
            commodity_types=data.get("commodity_types", []),
            annual_volume_tonnes=self._safe_float(
                data.get("annual_volume_tonnes")
            ),
            processing_capacity_tonnes=self._safe_float(
                data.get("processing_capacity_tonnes")
            ),
            upstream_supplier_count=self._safe_int(
                data.get("upstream_supplier_count")
            ),
            certifications=certs,
            dds_references=data.get("dds_references", []),
            deforestation_free_status=str(
                data.get("deforestation_free_status", "unverified")
            ),
            contacts=contacts,
            profile_status=str(
                data.get(
                    "profile_status",
                    ProfileStatus.PENDING_VERIFICATION.value,
                )
            ),
            tier_level=self._safe_int(data.get("tier_level")),
            risk_score=self._safe_float(data.get("risk_score")),
            metadata=data.get("metadata", {}),
        )

    def _validate_profile(
        self, profile: SupplierProfile
    ) -> List[str]:
        """Validate a supplier profile for data quality issues.

        Args:
            profile: Profile to validate.

        Returns:
            List of validation warning messages.
        """
        warnings: List[str] = []

        if not profile.legal_name:
            warnings.append("Missing legal_name")

        if not profile.country_code:
            warnings.append("Missing country_code")
        elif len(profile.country_code) != 2:
            warnings.append(
                f"Invalid country_code length: {profile.country_code}"
            )

        if (
            profile.gps_latitude is not None
            and not -90.0 <= profile.gps_latitude <= 90.0
        ):
            warnings.append(
                f"GPS latitude out of range: {profile.gps_latitude}"
            )

        if (
            profile.gps_longitude is not None
            and not -180.0 <= profile.gps_longitude <= 180.0
        ):
            warnings.append(
                f"GPS longitude out of range: {profile.gps_longitude}"
            )

        if not profile.commodity_types:
            warnings.append("No commodity types specified")

        return warnings

    def _build_operation_result(
        self,
        success: bool,
        operation: str,
        start_time: float,
        supplier_id: str = "",
        profile: Optional[SupplierProfile] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> ProfileOperationResult:
        """Build a ProfileOperationResult.

        Args:
            success: Whether the operation succeeded.
            operation: Operation type.
            start_time: Monotonic start time.
            supplier_id: Supplier ID.
            profile: Profile if available.
            errors: Errors list.
            warnings: Warnings list.

        Returns:
            ProfileOperationResult instance.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return ProfileOperationResult(
            success=success,
            supplier_id=supplier_id,
            operation=operation,
            profile=profile,
            completeness_score=(
                profile.completeness_score if profile else 0.0
            ),
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(
                {"operation": operation, "supplier_id": supplier_id}
            ),
            errors=errors or [],
            warnings=warnings or [],
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert a value to float.

        Args:
            value: Value to convert.

        Returns:
            Float value or None if conversion fails.
        """
        if value is None:
            return None
        try:
            result = float(value)
            if result != result or result == float("inf"):
                return None
            return result
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert a value to int.

        Args:
            value: Value to convert.

        Returns:
            Integer value or None if conversion fails.
        """
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def total_profiles(self) -> int:
        """Return total number of profiles stored.

        Returns:
            Count of profiles in the store.
        """
        return len(self._profiles)

    @property
    def active_profiles(self) -> int:
        """Return count of active profiles.

        Returns:
            Count of profiles with non-deactivated status.
        """
        return sum(
            1
            for p in self._profiles.values()
            if p.profile_status != ProfileStatus.DEACTIVATED.value
        )

    @property
    def total_operations(self) -> int:
        """Return total operation count.

        Returns:
            Running operation count.
        """
        return self._operation_count

    def get_change_history(
        self, supplier_id: str
    ) -> List[ProfileChangeLog]:
        """Get the change history for a supplier.

        Args:
            supplier_id: Supplier ID to query.

        Returns:
            List of ProfileChangeLog entries for the supplier.
        """
        return [
            cl
            for cl in self._change_logs
            if cl.supplier_id == supplier_id
        ]

    def get_versions(
        self, supplier_id: str
    ) -> List[ProfileVersion]:
        """Get all profile versions for a supplier.

        Args:
            supplier_id: Supplier ID to query.

        Returns:
            List of ProfileVersion entries.
        """
        return self._versions.get(supplier_id, [])


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "SupplierProfileManager",
    # Enums
    "ProfileStatus",
    "SupplierType",
    "CertificationType",
    "EUDRCommodity",
    "ChangeType",
    # Data classes
    "SupplierProfile",
    "CertificationRecord",
    "ContactInfo",
    "ProfileVersion",
    "SearchCriteria",
    "ProfileChangeLog",
    "ProfileOperationResult",
    "BatchProfileResult",
    # Constants
    "ENGINE_VERSION",
    "METRIC_PREFIX",
    "DEFAULT_BATCH_SIZE",
    "MAX_PROFILE_VERSIONS",
    "COMPLETENESS_WEIGHTS",
]
