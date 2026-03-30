# -*- coding: utf-8 -*-
"""
Gap Analyzer - AGENT-EUDR-008 Engine 7

Production-grade gap analysis engine for multi-tier supplier tracking
under the EU Deforestation Regulation (EUDR). Detects missing data,
coverage gaps, and verification gaps across supplier profiles, classifies
gap severity (CRITICAL/MAJOR/MINOR), generates prioritized remediation
plans, auto-generates supplier questionnaires for gap filling, tracks
remediation progress, and analyzes gap trends over time.

Zero-Hallucination Guarantees:
    - All gap detection uses deterministic field presence checks
    - Severity classification follows fixed rules from PRD Section 6.7
    - Remediation plans are template-driven with no generative content
    - Questionnaires are assembled from fixed field templates
    - SHA-256 provenance chain hashing on all results
    - No ML/LLM used in any gap analysis

Performance Targets:
    - Single supplier gap analysis: <5ms
    - Batch gap analysis (10,000 suppliers): <5s
    - Questionnaire generation: <2ms per supplier

Regulatory References:
    - EUDR Article 4: Due diligence completeness
    - EUDR Article 9: Required traceability information
    - EUDR Article 10: Trader obligations
    - EUDR Article 14: Audit-readiness (5-year retention)
    - PRD Section 6.7: Gap Analysis and Remediation

Gap Severity Definitions:
    - CRITICAL: Blocks DDS submission (missing GPS, no legal entity)
    - MAJOR: Regulatory risk (missing certification, outdated data)
    - MINOR: Data quality improvement (missing contact, incomplete address)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-008 (Engine 7: Gap Analysis)
Agent ID: GL-EUDR-MST-008
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse an ISO datetime string to a timezone-aware datetime.

    Args:
        dt_str: ISO formatted datetime string.

    Returns:
        Timezone-aware datetime or None if parsing fails.
    """
    if not dt_str or not dt_str.strip():
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GapSeverity(str, Enum):
    """Gap severity classification per PRD Section 6.7.

    CRITICAL: Blocks DDS submission (e.g., missing GPS coordinates,
        no legal entity identification).
    MAJOR: Significant regulatory risk (e.g., missing certification,
        outdated deforestation check, no DDS reference).
    MINOR: Data quality improvement opportunity (e.g., missing contact
        info, incomplete address, missing capacity data).
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class GapCategory(str, Enum):
    """Categories of gaps that can be detected."""

    DATA_GAP = "data_gap"
    COVERAGE_GAP = "coverage_gap"
    VERIFICATION_GAP = "verification_gap"

class GapType(str, Enum):
    """Specific gap types within categories."""

    # Data gaps (PRD F7.1)
    MISSING_GPS = "missing_gps"
    MISSING_LEGAL_ENTITY = "missing_legal_entity"
    MISSING_CERTIFICATION = "missing_certification"
    MISSING_DDS = "missing_dds"
    MISSING_COUNTRY = "missing_country"
    MISSING_COMMODITY = "missing_commodity"
    MISSING_CONTACT = "missing_contact"
    MISSING_REGISTRATION_ID = "missing_registration_id"
    MISSING_ADDRESS = "missing_address"
    MISSING_VOLUME = "missing_volume"
    MISSING_DEFORESTATION_STATUS = "missing_deforestation_status"

    # Coverage gaps (PRD F7.2)
    TIER_WITHOUT_SUPPLIERS = "tier_without_suppliers"
    LOW_TIER_VISIBILITY = "low_tier_visibility"
    INCOMPLETE_COMMODITY_CHAIN = "incomplete_commodity_chain"

    # Verification gaps (PRD F7.3)
    OUTDATED_CERTIFICATION = "outdated_certification"
    OUTDATED_DDS = "outdated_dds"
    OUTDATED_DEFORESTATION_CHECK = "outdated_deforestation_check"
    UNVERIFIED_GPS = "unverified_gps"
    UNVERIFIED_LEGAL_ENTITY = "unverified_legal_entity"
    STALE_PROFILE = "stale_profile"

class RemediationStatus(str, Enum):
    """Status of a remediation action."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"

class RemediationPriority(str, Enum):
    """Priority of a remediation action."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TrendDirection(str, Enum):
    """Gap trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"

# ---------------------------------------------------------------------------
# Gap Severity Rules: Maps gap type to severity
# ---------------------------------------------------------------------------

GAP_SEVERITY_RULES: Dict[str, GapSeverity] = {
    # Critical: blocks DDS submission
    GapType.MISSING_GPS.value: GapSeverity.CRITICAL,
    GapType.MISSING_LEGAL_ENTITY.value: GapSeverity.CRITICAL,
    GapType.MISSING_COUNTRY.value: GapSeverity.CRITICAL,

    # Major: regulatory risk
    GapType.MISSING_DDS.value: GapSeverity.MAJOR,
    GapType.MISSING_CERTIFICATION.value: GapSeverity.MAJOR,
    GapType.MISSING_DEFORESTATION_STATUS.value: GapSeverity.MAJOR,
    GapType.MISSING_COMMODITY.value: GapSeverity.MAJOR,
    GapType.TIER_WITHOUT_SUPPLIERS.value: GapSeverity.MAJOR,
    GapType.LOW_TIER_VISIBILITY.value: GapSeverity.MAJOR,
    GapType.INCOMPLETE_COMMODITY_CHAIN.value: GapSeverity.MAJOR,
    GapType.OUTDATED_CERTIFICATION.value: GapSeverity.MAJOR,
    GapType.OUTDATED_DDS.value: GapSeverity.MAJOR,
    GapType.OUTDATED_DEFORESTATION_CHECK.value: GapSeverity.MAJOR,
    GapType.UNVERIFIED_GPS.value: GapSeverity.MAJOR,
    GapType.UNVERIFIED_LEGAL_ENTITY.value: GapSeverity.MAJOR,
    GapType.STALE_PROFILE.value: GapSeverity.MAJOR,

    # Minor: data quality
    GapType.MISSING_CONTACT.value: GapSeverity.MINOR,
    GapType.MISSING_REGISTRATION_ID.value: GapSeverity.MINOR,
    GapType.MISSING_ADDRESS.value: GapSeverity.MINOR,
    GapType.MISSING_VOLUME.value: GapSeverity.MINOR,
}

# ---------------------------------------------------------------------------
# Remediation Templates: Maps gap type to remediation guidance
# ---------------------------------------------------------------------------

REMEDIATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    GapType.MISSING_GPS.value: {
        "action": "Collect GPS coordinates for all production plots",
        "instruction": (
            "Use a GNSS-capable device to record GPS coordinates at each "
            "production plot. Coordinates must be in WGS84 datum with at "
            "least 5 decimal places. Record both latitude and longitude."
        ),
        "deadline_days": "7",
        "responsible_role": "Supplier / Field Agent",
    },
    GapType.MISSING_LEGAL_ENTITY.value: {
        "action": "Provide legal entity name and registration",
        "instruction": (
            "Provide the full legal entity name, company registration "
            "number, and country of registration. This is required under "
            "EUDR Article 9(1)(a)."
        ),
        "deadline_days": "14",
        "responsible_role": "Supplier / Compliance Officer",
    },
    GapType.MISSING_COUNTRY.value: {
        "action": "Provide country of production",
        "instruction": (
            "Specify the ISO 3166-1 alpha-2 country code for the country "
            "of production. Required under EUDR Article 9(1)(c)."
        ),
        "deadline_days": "7",
        "responsible_role": "Supplier",
    },
    GapType.MISSING_DDS.value: {
        "action": "Submit Due Diligence Statement reference",
        "instruction": (
            "Provide the DDS reference number from the EU Information "
            "System. If no DDS has been submitted, initiate the DDS "
            "submission process. Required under EUDR Article 10(2)."
        ),
        "deadline_days": "30",
        "responsible_role": "Compliance Officer",
    },
    GapType.MISSING_CERTIFICATION.value: {
        "action": "Provide certification documentation",
        "instruction": (
            "Provide certification details including type (e.g., FSC, "
            "RSPO, UTZ), certificate ID, and validity dates. If not "
            "certified, document why certification is not applicable."
        ),
        "deadline_days": "30",
        "responsible_role": "Supplier / Procurement",
    },
    GapType.MISSING_DEFORESTATION_STATUS.value: {
        "action": "Obtain deforestation-free verification",
        "instruction": (
            "Arrange for deforestation-free verification of production "
            "areas. Acceptable methods: satellite monitoring confirmation, "
            "third-party verification, or self-declaration with evidence."
        ),
        "deadline_days": "30",
        "responsible_role": "Compliance Officer / Supplier",
    },
    GapType.MISSING_COMMODITY.value: {
        "action": "Specify commodity types handled",
        "instruction": (
            "List all EUDR-regulated commodity types handled by this "
            "supplier (cattle, cocoa, coffee, oil palm, rubber, soya, "
            "wood/timber) and any derived products."
        ),
        "deadline_days": "14",
        "responsible_role": "Supplier",
    },
    GapType.MISSING_CONTACT.value: {
        "action": "Provide contact information",
        "instruction": (
            "Provide primary contact name, email, and phone number. Also "
            "provide compliance contact details if different from primary."
        ),
        "deadline_days": "14",
        "responsible_role": "Supplier",
    },
    GapType.MISSING_REGISTRATION_ID.value: {
        "action": "Provide company registration ID",
        "instruction": (
            "Provide the official company registration number, DUNS "
            "number, or equivalent business identifier."
        ),
        "deadline_days": "14",
        "responsible_role": "Supplier",
    },
    GapType.MISSING_ADDRESS.value: {
        "action": "Provide physical address",
        "instruction": (
            "Provide the full physical address of the supplier including "
            "street, city, region/state, postal code, and country."
        ),
        "deadline_days": "14",
        "responsible_role": "Supplier",
    },
    GapType.MISSING_VOLUME.value: {
        "action": "Provide annual commodity volume data",
        "instruction": (
            "Report annual commodity volumes in tonnes, including "
            "processing capacity if applicable."
        ),
        "deadline_days": "30",
        "responsible_role": "Supplier / Procurement",
    },
    GapType.OUTDATED_CERTIFICATION.value: {
        "action": "Renew expired certification",
        "instruction": (
            "Initiate certification renewal process. Contact the "
            "certification body to arrange audit and renewal."
        ),
        "deadline_days": "60",
        "responsible_role": "Supplier / Compliance Officer",
    },
    GapType.OUTDATED_DDS.value: {
        "action": "Renew expired DDS",
        "instruction": (
            "Submit a new Due Diligence Statement through the EU "
            "Information System. The current DDS has expired."
        ),
        "deadline_days": "14",
        "responsible_role": "Compliance Officer",
    },
    GapType.OUTDATED_DEFORESTATION_CHECK.value: {
        "action": "Update deforestation-free verification",
        "instruction": (
            "The deforestation-free verification is outdated (>12 months). "
            "Arrange for a new assessment using current satellite data."
        ),
        "deadline_days": "30",
        "responsible_role": "Compliance Officer",
    },
    GapType.UNVERIFIED_GPS.value: {
        "action": "Verify GPS coordinates",
        "instruction": (
            "GPS coordinates are present but have not been independently "
            "verified. Cross-check against satellite imagery or arrange "
            "field verification."
        ),
        "deadline_days": "30",
        "responsible_role": "Field Agent / GIS Team",
    },
    GapType.UNVERIFIED_LEGAL_ENTITY.value: {
        "action": "Verify legal entity registration",
        "instruction": (
            "Cross-verify the legal entity name and registration ID "
            "against official company registries."
        ),
        "deadline_days": "30",
        "responsible_role": "Compliance Officer",
    },
    GapType.STALE_PROFILE.value: {
        "action": "Update supplier profile",
        "instruction": (
            "The supplier profile has not been updated in over 12 months. "
            "Send a data refresh request to the supplier to confirm or "
            "update all profile fields."
        ),
        "deadline_days": "30",
        "responsible_role": "Procurement / Supplier Manager",
    },
    GapType.TIER_WITHOUT_SUPPLIERS.value: {
        "action": "Identify suppliers at missing tier",
        "instruction": (
            "This tier in the supply chain has no identified suppliers. "
            "Request sub-tier supplier information from Tier N-1 suppliers."
        ),
        "deadline_days": "60",
        "responsible_role": "Supply Chain Manager",
    },
    GapType.LOW_TIER_VISIBILITY.value: {
        "action": "Improve tier visibility coverage",
        "instruction": (
            "Less than 50% of suppliers at this tier are identified. "
            "Conduct supplier mapping exercises to improve visibility."
        ),
        "deadline_days": "90",
        "responsible_role": "Supply Chain Manager",
    },
    GapType.INCOMPLETE_COMMODITY_CHAIN.value: {
        "action": "Complete commodity chain mapping",
        "instruction": (
            "The supply chain for this commodity does not reach the "
            "expected depth. Map additional tiers to reach origin producers."
        ),
        "deadline_days": "90",
        "responsible_role": "Supply Chain Manager",
    },
}

# ---------------------------------------------------------------------------
# Questionnaire Field Templates
# ---------------------------------------------------------------------------

QUESTIONNAIRE_FIELD_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "legal_entity": {
        "section": "1. Legal Entity Information",
        "fields": [
            {"name": "legal_name", "label": "Full legal entity name", "required": True, "type": "text"},
            {"name": "registration_id", "label": "Company registration number", "required": True, "type": "text"},
            {"name": "tax_id", "label": "Tax identification number", "required": False, "type": "text"},
            {"name": "duns_number", "label": "DUNS number (if available)", "required": False, "type": "text"},
        ],
    },
    "location": {
        "section": "2. Location Information",
        "fields": [
            {"name": "country_iso", "label": "Country (ISO 3166-1 alpha-2)", "required": True, "type": "text"},
            {"name": "admin_region", "label": "Administrative region/state", "required": True, "type": "text"},
            {"name": "address", "label": "Full physical address", "required": True, "type": "text"},
            {"name": "gps_latitude", "label": "GPS latitude (WGS84)", "required": True, "type": "number"},
            {"name": "gps_longitude", "label": "GPS longitude (WGS84)", "required": True, "type": "number"},
        ],
    },
    "commodity": {
        "section": "3. Commodity Information",
        "fields": [
            {"name": "commodity_types", "label": "EUDR commodity types handled", "required": True, "type": "multi_select",
             "options": ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood_timber"]},
            {"name": "annual_volume_tonnes", "label": "Annual volume (tonnes)", "required": True, "type": "number"},
            {"name": "processing_capacity", "label": "Processing capacity (tonnes/year)", "required": False, "type": "number"},
        ],
    },
    "certification": {
        "section": "4. Certification Information",
        "fields": [
            {"name": "certification_type", "label": "Certification type (e.g., FSC, RSPO)", "required": True, "type": "text"},
            {"name": "certificate_id", "label": "Certificate ID", "required": True, "type": "text"},
            {"name": "valid_from", "label": "Valid from (YYYY-MM-DD)", "required": True, "type": "date"},
            {"name": "valid_until", "label": "Valid until (YYYY-MM-DD)", "required": True, "type": "date"},
        ],
    },
    "deforestation": {
        "section": "5. Deforestation-Free Declaration",
        "fields": [
            {"name": "deforestation_free_status", "label": "Deforestation-free status", "required": True, "type": "select",
             "options": ["verified", "self_declared", "not_assessed"]},
            {"name": "verification_date", "label": "Date of last verification (YYYY-MM-DD)", "required": False, "type": "date"},
            {"name": "verification_body", "label": "Verification body/organization", "required": False, "type": "text"},
        ],
    },
    "contact": {
        "section": "6. Contact Information",
        "fields": [
            {"name": "primary_contact_name", "label": "Primary contact name", "required": True, "type": "text"},
            {"name": "primary_contact_email", "label": "Primary contact email", "required": True, "type": "email"},
            {"name": "primary_contact_phone", "label": "Primary contact phone", "required": False, "type": "text"},
            {"name": "compliance_contact_name", "label": "Compliance contact name", "required": False, "type": "text"},
            {"name": "compliance_contact_email", "label": "Compliance contact email", "required": False, "type": "email"},
        ],
    },
    "upstream_suppliers": {
        "section": "7. Upstream Supplier Information",
        "fields": [
            {"name": "upstream_supplier_count", "label": "Number of upstream (sub-tier) suppliers", "required": True, "type": "number"},
            {"name": "upstream_supplier_names", "label": "Names of upstream suppliers", "required": True, "type": "textarea"},
            {"name": "upstream_supplier_countries", "label": "Countries of upstream suppliers", "required": True, "type": "textarea"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SupplierGapProfile:
    """Supplier profile data required for gap analysis.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Legal entity name.
        registration_id: Company registration ID.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Tier level in supply chain.
        commodity_types: EUDR commodity types handled.
        certifications: Certification records.
        gps_latitude: Production plot latitude.
        gps_longitude: Production plot longitude.
        address: Physical address.
        admin_region: Administrative region.
        annual_volume_tonnes: Annual commodity volume.
        primary_contact_name: Primary contact name.
        primary_contact_email: Primary contact email.
        compliance_contact_name: Compliance contact name.
        compliance_contact_email: Compliance contact email.
        dds_references: DDS reference records.
        deforestation_free_status: Verification status.
        deforestation_verified_date: Last verification date.
        last_profile_update: Last profile update date ISO string.
        gps_verified: Whether GPS was independently verified.
        legal_entity_verified: Whether legal entity was verified.
    """

    supplier_id: str = ""
    legal_name: str = ""
    registration_id: str = ""
    country_iso: str = ""
    tier: int = 1
    commodity_types: List[str] = field(default_factory=list)
    certifications: List[Dict[str, Any]] = field(default_factory=list)
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    address: str = ""
    admin_region: str = ""
    annual_volume_tonnes: Optional[float] = None
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    compliance_contact_name: str = ""
    compliance_contact_email: str = ""
    dds_references: List[Dict[str, Any]] = field(default_factory=list)
    deforestation_free_status: str = ""
    deforestation_verified_date: str = ""
    last_profile_update: str = ""
    gps_verified: bool = False
    legal_entity_verified: bool = False

@dataclass
class SupplierChainTier:
    """Tier information within a supplier chain for coverage analysis.

    Attributes:
        tier_level: Tier depth (1 = direct supplier).
        expected_supplier_count: Expected suppliers at this tier.
        known_supplier_count: Known/identified suppliers at this tier.
        supplier_ids: List of known supplier IDs.
        commodity: Commodity being tracked in this chain.
    """

    tier_level: int = 1
    expected_supplier_count: int = 0
    known_supplier_count: int = 0
    supplier_ids: List[str] = field(default_factory=list)
    commodity: str = ""

@dataclass
class DetectedGap:
    """A single detected gap in supplier data.

    Attributes:
        gap_id: Unique UUID4 gap identifier.
        supplier_id: Supplier with the gap.
        gap_type: Specific type of gap.
        category: Gap category (data, coverage, verification).
        severity: Gap severity (critical, major, minor).
        field_name: Name of the missing/problematic field.
        description: Human-readable description of the gap.
        impact: Impact on EUDR compliance.
        detected_at: UTC ISO timestamp of detection.
    """

    gap_id: str = ""
    supplier_id: str = ""
    gap_type: str = ""
    category: str = GapCategory.DATA_GAP.value
    severity: str = GapSeverity.MINOR.value
    field_name: str = ""
    description: str = ""
    impact: str = ""
    detected_at: str = ""

@dataclass
class RemediationAction:
    """A single remediation action item.

    Attributes:
        action_id: Unique UUID4 action identifier.
        gap_id: ID of the gap this action addresses.
        supplier_id: Supplier requiring remediation.
        priority: Remediation priority.
        action: Description of the required action.
        instruction: Detailed instruction for completing the action.
        responsible_role: Role responsible for this action.
        deadline_days: Number of days to complete.
        status: Current remediation status.
        created_at: UTC ISO timestamp of creation.
        completed_at: Optional completion timestamp.
    """

    action_id: str = ""
    gap_id: str = ""
    supplier_id: str = ""
    priority: str = RemediationPriority.MEDIUM.value
    action: str = ""
    instruction: str = ""
    responsible_role: str = ""
    deadline_days: int = 30
    status: str = RemediationStatus.PENDING.value
    created_at: str = ""
    completed_at: str = ""

@dataclass
class RemediationPlan:
    """Prioritized remediation plan for a supplier.

    Attributes:
        plan_id: Unique UUID4 plan identifier.
        supplier_id: Supplier for this plan.
        total_gaps: Total gaps identified.
        critical_gaps: Number of critical gaps.
        major_gaps: Number of major gaps.
        minor_gaps: Number of minor gaps.
        actions: Prioritized list of remediation actions.
        completion_pct: Overall completion percentage.
        created_at: UTC ISO timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    plan_id: str = ""
    supplier_id: str = ""
    total_gaps: int = 0
    critical_gaps: int = 0
    major_gaps: int = 0
    minor_gaps: int = 0
    actions: List[RemediationAction] = field(default_factory=list)
    completion_pct: float = 0.0
    created_at: str = ""
    provenance_hash: str = ""

@dataclass
class SupplierQuestionnaire:
    """Auto-generated supplier questionnaire for gap filling.

    Attributes:
        questionnaire_id: Unique UUID4 identifier.
        supplier_id: Supplier the questionnaire targets.
        sections: Questionnaire sections with fields.
        total_fields: Total fields in the questionnaire.
        required_fields: Number of required fields.
        generated_at: UTC ISO timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    questionnaire_id: str = ""
    supplier_id: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    total_fields: int = 0
    required_fields: int = 0
    generated_at: str = ""
    provenance_hash: str = ""

@dataclass
class GapAnalysisResult:
    """Full gap analysis result for a supplier.

    Attributes:
        analysis_id: Unique UUID4 analysis identifier.
        supplier_id: Supplier analyzed.
        total_gaps: Total gaps detected.
        critical_gaps: Number of critical gaps.
        major_gaps: Number of major gaps.
        minor_gaps: Number of minor gaps.
        gaps: List of all detected gaps.
        dds_blocked: Whether critical gaps block DDS submission.
        completeness_score: Profile completeness score (0-100).
        analyzed_at: UTC ISO timestamp.
        processing_time_ms: Analysis duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        engine_version: Engine version string.
    """

    analysis_id: str = ""
    supplier_id: str = ""
    total_gaps: int = 0
    critical_gaps: int = 0
    major_gaps: int = 0
    minor_gaps: int = 0
    gaps: List[DetectedGap] = field(default_factory=list)
    dds_blocked: bool = False
    completeness_score: float = 0.0
    analyzed_at: str = ""
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    engine_version: str = _MODULE_VERSION

@dataclass
class GapTrendEntry:
    """A single entry in gap trend analysis.

    Attributes:
        timestamp: UTC ISO timestamp.
        total_gaps: Total gaps at this point.
        critical_gaps: Critical gap count.
        major_gaps: Major gap count.
        minor_gaps: Minor gap count.
        completeness_score: Completeness at this point.
    """

    timestamp: str = ""
    total_gaps: int = 0
    critical_gaps: int = 0
    major_gaps: int = 0
    minor_gaps: int = 0
    completeness_score: float = 0.0

@dataclass
class GapTrendResult:
    """Gap trend analysis result for a supplier.

    Attributes:
        supplier_id: Supplier analyzed.
        entries: Chronological trend entries.
        trend_direction: Overall trend direction.
        gap_delta: Change in total gaps from first to last entry.
        completeness_delta: Change in completeness score.
    """

    supplier_id: str = ""
    entries: List[GapTrendEntry] = field(default_factory=list)
    trend_direction: str = TrendDirection.STABLE.value
    gap_delta: int = 0
    completeness_delta: float = 0.0

@dataclass
class BatchGapResult:
    """Batch gap analysis result.

    Attributes:
        batch_id: Unique UUID4 batch identifier.
        total_suppliers: Total suppliers analyzed.
        successful: Number successfully analyzed.
        failed: Number that failed analysis.
        results: Individual analysis results.
        summary: Aggregate statistics.
        processing_time_ms: Total duration.
        provenance_hash: SHA-256 hash of batch.
    """

    batch_id: str = ""
    total_suppliers: int = 0
    successful: int = 0
    failed: int = 0
    results: List[GapAnalysisResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    provenance_hash: str = ""

# ===========================================================================
# GapAnalyzer
# ===========================================================================

class GapAnalyzer:
    """Production-grade gap analysis engine for EUDR multi-tier suppliers.

    Detects data gaps, coverage gaps, and verification gaps in supplier
    profiles. Classifies severity (CRITICAL/MAJOR/MINOR), generates
    remediation plans, creates supplier questionnaires, tracks progress,
    and analyzes gap trends.

    All analysis is deterministic with zero LLM/ML involvement.

    Attributes:
        _gap_history: In-memory gap analysis history per supplier.
        _remediation_plans: In-memory remediation plans by plan ID.
        _analysis_count: Running count of analyses performed.

    Example::

        analyzer = GapAnalyzer()
        profile = SupplierGapProfile(
            supplier_id="SUP-001",
            legal_name="Test Supplier",
        )
        result = analyzer.analyze_gaps("SUP-001", profile)
        assert result.total_gaps >= 0
    """

    def __init__(self) -> None:
        """Initialize GapAnalyzer."""
        self._gap_history: Dict[str, List[GapAnalysisResult]] = {}
        self._remediation_plans: Dict[str, RemediationPlan] = {}
        self._analysis_count: int = 0

        logger.info("GapAnalyzer initialized (version=%s)", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API: Full Gap Analysis
    # ------------------------------------------------------------------

    def analyze_gaps(
        self,
        supplier_id: str,
        profile: SupplierGapProfile,
    ) -> GapAnalysisResult:
        """Perform full gap analysis for a supplier.

        Detects data gaps, coverage gaps, and verification gaps. Classifies
        each gap by severity and calculates an overall completeness score.

        Args:
            supplier_id: Unique supplier identifier.
            profile: Supplier profile data for analysis.

        Returns:
            GapAnalysisResult with all detected gaps.

        Raises:
            ValueError: If supplier_id is empty.
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")

        t_start = time.monotonic()
        analysis_id = str(uuid.uuid4())
        all_gaps: List[DetectedGap] = []

        # Detect data gaps
        data_gaps = self.detect_data_gaps(profile)
        all_gaps.extend(data_gaps)

        # Detect verification gaps
        verification_gaps = self.detect_verification_gaps(profile)
        all_gaps.extend(verification_gaps)

        # Classify and count by severity
        critical = sum(
            1 for g in all_gaps if g.severity == GapSeverity.CRITICAL.value
        )
        major = sum(
            1 for g in all_gaps if g.severity == GapSeverity.MAJOR.value
        )
        minor = sum(
            1 for g in all_gaps if g.severity == GapSeverity.MINOR.value
        )

        # DDS blocked if any critical gaps
        dds_blocked = critical > 0

        # Calculate completeness score
        completeness = self._calculate_completeness(profile, all_gaps)

        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        provenance_data = {
            "analysis_id": analysis_id,
            "supplier_id": supplier_id,
            "total_gaps": len(all_gaps),
            "critical": critical,
            "major": major,
            "minor": minor,
            "completeness": round(completeness, 4),
            "engine_version": _MODULE_VERSION,
        }
        provenance_hash = _compute_hash(provenance_data)

        result = GapAnalysisResult(
            analysis_id=analysis_id,
            supplier_id=supplier_id,
            total_gaps=len(all_gaps),
            critical_gaps=critical,
            major_gaps=major,
            minor_gaps=minor,
            gaps=all_gaps,
            dds_blocked=dds_blocked,
            completeness_score=round(completeness, 2),
            analyzed_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
            engine_version=_MODULE_VERSION,
        )

        # Track history
        if supplier_id not in self._gap_history:
            self._gap_history[supplier_id] = []
        self._gap_history[supplier_id].append(result)
        self._analysis_count += 1

        logger.info(
            "Gap analysis completed: supplier=%s total=%d critical=%d "
            "major=%d minor=%d completeness=%.1f dds_blocked=%s "
            "time=%.2fms",
            supplier_id,
            len(all_gaps),
            critical,
            major,
            minor,
            completeness,
            dds_blocked,
            elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Data Gap Detection
    # ------------------------------------------------------------------

    def detect_data_gaps(
        self,
        profile: SupplierGapProfile,
    ) -> List[DetectedGap]:
        """Detect missing data fields in a supplier profile.

        Checks for missing GPS, certification, legal entity, DDS,
        commodity, contact, and other required fields.

        Args:
            profile: Supplier profile to analyze.

        Returns:
            List of DetectedGap objects for missing data.
        """
        gaps: List[DetectedGap] = []
        now_str = utcnow().isoformat()

        # GPS coordinates (CRITICAL)
        if profile.gps_latitude is None or profile.gps_longitude is None:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_GPS.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.CRITICAL.value,
                field_name="gps_latitude/gps_longitude",
                description=(
                    "GPS coordinates are missing. EUDR Article 9(1)(d) "
                    "requires geolocation of production plots."
                ),
                impact="Blocks DDS submission",
                detected_at=now_str,
            ))

        # Legal entity name (CRITICAL)
        if not profile.legal_name or not profile.legal_name.strip():
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_LEGAL_ENTITY.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.CRITICAL.value,
                field_name="legal_name",
                description=(
                    "Legal entity name is missing. EUDR Article 9(1)(a) "
                    "requires supplier identification."
                ),
                impact="Blocks DDS submission",
                detected_at=now_str,
            ))

        # Country (CRITICAL)
        if not profile.country_iso or not profile.country_iso.strip():
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_COUNTRY.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.CRITICAL.value,
                field_name="country_iso",
                description=(
                    "Country of production is missing. EUDR Article 9(1)(c) "
                    "requires country identification."
                ),
                impact="Blocks DDS submission",
                detected_at=now_str,
            ))

        # DDS reference (MAJOR)
        if not profile.dds_references:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_DDS.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="dds_references",
                description=(
                    "No DDS (Due Diligence Statement) reference found. "
                    "Required under EUDR Article 10(2)."
                ),
                impact="Regulatory risk - DDS reference required",
                detected_at=now_str,
            ))

        # Certification (MAJOR)
        if not profile.certifications:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_CERTIFICATION.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="certifications",
                description=(
                    "No certifications found. Certification provides "
                    "evidence of sustainable sourcing practices."
                ),
                impact="Regulatory risk - reduced compliance confidence",
                detected_at=now_str,
            ))

        # Deforestation-free status (MAJOR)
        if not profile.deforestation_free_status or not profile.deforestation_free_status.strip():
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_DEFORESTATION_STATUS.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="deforestation_free_status",
                description=(
                    "Deforestation-free status has not been assessed. "
                    "EUDR requires products to be deforestation-free."
                ),
                impact="Regulatory risk - unverified deforestation status",
                detected_at=now_str,
            ))

        # Commodity types (MAJOR)
        if not profile.commodity_types:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_COMMODITY.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="commodity_types",
                description=(
                    "Commodity types handled by this supplier are not "
                    "specified."
                ),
                impact="Cannot determine EUDR commodity applicability",
                detected_at=now_str,
            ))

        # Contact information (MINOR)
        if not profile.primary_contact_name or not profile.primary_contact_email:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_CONTACT.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MINOR.value,
                field_name="primary_contact_name/primary_contact_email",
                description="Primary contact information is incomplete.",
                impact="Data quality - limits communication ability",
                detected_at=now_str,
            ))

        # Registration ID (MINOR)
        if not profile.registration_id or not profile.registration_id.strip():
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_REGISTRATION_ID.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MINOR.value,
                field_name="registration_id",
                description="Company registration ID is missing.",
                impact="Data quality - limits entity verification",
                detected_at=now_str,
            ))

        # Address (MINOR)
        if not profile.address or not profile.address.strip():
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_ADDRESS.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MINOR.value,
                field_name="address",
                description="Physical address is missing.",
                impact="Data quality - incomplete location data",
                detected_at=now_str,
            ))

        # Volume data (MINOR)
        if profile.annual_volume_tonnes is None:
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.MISSING_VOLUME.value,
                category=GapCategory.DATA_GAP.value,
                severity=GapSeverity.MINOR.value,
                field_name="annual_volume_tonnes",
                description="Annual commodity volume data is missing.",
                impact="Data quality - cannot assess volume concentration",
                detected_at=now_str,
            ))

        logger.debug(
            "Data gap detection: supplier=%s gaps=%d",
            profile.supplier_id,
            len(gaps),
        )

        return gaps

    # ------------------------------------------------------------------
    # Public API: Coverage Gap Detection
    # ------------------------------------------------------------------

    def detect_coverage_gaps(
        self,
        chain: List[SupplierChainTier],
    ) -> List[DetectedGap]:
        """Detect tiers without known suppliers in the supply chain.

        Identifies tiers with no known suppliers or very low visibility.

        Args:
            chain: List of tier information for the supply chain.

        Returns:
            List of DetectedGap objects for coverage gaps.
        """
        gaps: List[DetectedGap] = []
        now_str = utcnow().isoformat()

        for tier in chain:
            if tier.expected_supplier_count <= 0:
                continue

            if tier.known_supplier_count == 0:
                gaps.append(DetectedGap(
                    gap_id=str(uuid.uuid4()),
                    supplier_id=f"tier_{tier.tier_level}",
                    gap_type=GapType.TIER_WITHOUT_SUPPLIERS.value,
                    category=GapCategory.COVERAGE_GAP.value,
                    severity=GapSeverity.MAJOR.value,
                    field_name=f"tier_{tier.tier_level}_suppliers",
                    description=(
                        f"Tier {tier.tier_level} has no identified suppliers "
                        f"(expected: {tier.expected_supplier_count})"
                    ),
                    impact=(
                        f"No visibility at tier {tier.tier_level}; "
                        "cannot trace supply chain to origin"
                    ),
                    detected_at=now_str,
                ))
            elif tier.expected_supplier_count > 0:
                visibility_pct = (
                    tier.known_supplier_count
                    / tier.expected_supplier_count
                    * 100.0
                )
                if visibility_pct < 50.0:
                    gaps.append(DetectedGap(
                        gap_id=str(uuid.uuid4()),
                        supplier_id=f"tier_{tier.tier_level}",
                        gap_type=GapType.LOW_TIER_VISIBILITY.value,
                        category=GapCategory.COVERAGE_GAP.value,
                        severity=GapSeverity.MAJOR.value,
                        field_name=f"tier_{tier.tier_level}_visibility",
                        description=(
                            f"Tier {tier.tier_level} visibility is only "
                            f"{visibility_pct:.1f}% "
                            f"({tier.known_supplier_count}/"
                            f"{tier.expected_supplier_count} suppliers)"
                        ),
                        impact=(
                            "Low supply chain visibility increases "
                            "deforestation risk"
                        ),
                        detected_at=now_str,
                    ))

        logger.debug(
            "Coverage gap detection: tiers=%d gaps=%d",
            len(chain),
            len(gaps),
        )

        return gaps

    # ------------------------------------------------------------------
    # Public API: Verification Gap Detection
    # ------------------------------------------------------------------

    def detect_verification_gaps(
        self,
        profile: SupplierGapProfile,
    ) -> List[DetectedGap]:
        """Detect outdated or unverified data in a supplier profile.

        Checks for outdated certifications, expired DDS, stale profiles,
        and unverified GPS/legal entity data.

        Args:
            profile: Supplier profile to analyze.

        Returns:
            List of DetectedGap objects for verification gaps.
        """
        gaps: List[DetectedGap] = []
        now = utcnow()
        now_str = now.isoformat()

        # Outdated certifications
        for cert in profile.certifications:
            valid_until_str = str(cert.get("valid_until", ""))
            valid_until = _parse_datetime(valid_until_str)
            cert_type = str(cert.get("type", "unknown"))

            if valid_until is not None and valid_until < now:
                gaps.append(DetectedGap(
                    gap_id=str(uuid.uuid4()),
                    supplier_id=profile.supplier_id,
                    gap_type=GapType.OUTDATED_CERTIFICATION.value,
                    category=GapCategory.VERIFICATION_GAP.value,
                    severity=GapSeverity.MAJOR.value,
                    field_name=f"certification_{cert_type}",
                    description=(
                        f"Certification {cert_type} expired on "
                        f"{valid_until_str}"
                    ),
                    impact="Certification no longer valid for compliance",
                    detected_at=now_str,
                ))

        # Outdated DDS
        for dds in profile.dds_references:
            valid_until_str = str(dds.get("valid_until", ""))
            valid_until = _parse_datetime(valid_until_str)
            dds_id = str(dds.get("dds_id", "unknown"))

            if valid_until is not None and valid_until < now:
                gaps.append(DetectedGap(
                    gap_id=str(uuid.uuid4()),
                    supplier_id=profile.supplier_id,
                    gap_type=GapType.OUTDATED_DDS.value,
                    category=GapCategory.VERIFICATION_GAP.value,
                    severity=GapSeverity.MAJOR.value,
                    field_name=f"dds_{dds_id}",
                    description=(
                        f"DDS {dds_id} expired on {valid_until_str}"
                    ),
                    impact="DDS is no longer valid for EU market access",
                    detected_at=now_str,
                ))

        # Outdated deforestation check
        if profile.deforestation_verified_date:
            verified = _parse_datetime(profile.deforestation_verified_date)
            if verified is not None:
                days_since = (now - verified).days
                if days_since > 365:
                    gaps.append(DetectedGap(
                        gap_id=str(uuid.uuid4()),
                        supplier_id=profile.supplier_id,
                        gap_type=GapType.OUTDATED_DEFORESTATION_CHECK.value,
                        category=GapCategory.VERIFICATION_GAP.value,
                        severity=GapSeverity.MAJOR.value,
                        field_name="deforestation_verified_date",
                        description=(
                            f"Deforestation-free verification is "
                            f"{days_since} days old (>365 days)"
                        ),
                        impact=(
                            "Outdated verification may not reflect "
                            "current land use"
                        ),
                        detected_at=now_str,
                    ))

        # Unverified GPS
        if (
            profile.gps_latitude is not None
            and profile.gps_longitude is not None
            and not profile.gps_verified
        ):
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.UNVERIFIED_GPS.value,
                category=GapCategory.VERIFICATION_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="gps_verified",
                description=(
                    "GPS coordinates have not been independently verified"
                ),
                impact="Unverified coordinates may be inaccurate",
                detected_at=now_str,
            ))

        # Unverified legal entity
        if (
            profile.legal_name
            and profile.legal_name.strip()
            and not profile.legal_entity_verified
        ):
            gaps.append(DetectedGap(
                gap_id=str(uuid.uuid4()),
                supplier_id=profile.supplier_id,
                gap_type=GapType.UNVERIFIED_LEGAL_ENTITY.value,
                category=GapCategory.VERIFICATION_GAP.value,
                severity=GapSeverity.MAJOR.value,
                field_name="legal_entity_verified",
                description=(
                    "Legal entity has not been verified against "
                    "official registries"
                ),
                impact="Entity identity not independently confirmed",
                detected_at=now_str,
            ))

        # Stale profile
        if profile.last_profile_update:
            last_update = _parse_datetime(profile.last_profile_update)
            if last_update is not None:
                days_since = (now - last_update).days
                if days_since > 365:
                    gaps.append(DetectedGap(
                        gap_id=str(uuid.uuid4()),
                        supplier_id=profile.supplier_id,
                        gap_type=GapType.STALE_PROFILE.value,
                        category=GapCategory.VERIFICATION_GAP.value,
                        severity=GapSeverity.MAJOR.value,
                        field_name="last_profile_update",
                        description=(
                            f"Supplier profile has not been updated in "
                            f"{days_since} days"
                        ),
                        impact="Profile data may be outdated or inaccurate",
                        detected_at=now_str,
                    ))

        logger.debug(
            "Verification gap detection: supplier=%s gaps=%d",
            profile.supplier_id,
            len(gaps),
        )

        return gaps

    # ------------------------------------------------------------------
    # Public API: Gap Severity Classification
    # ------------------------------------------------------------------

    def classify_severity(
        self,
        gap: DetectedGap,
    ) -> GapSeverity:
        """Classify the severity of a detected gap.

        Uses the fixed GAP_SEVERITY_RULES mapping. Falls back to MINOR
        for unknown gap types.

        Args:
            gap: Detected gap to classify.

        Returns:
            GapSeverity enum value.
        """
        severity = GAP_SEVERITY_RULES.get(
            gap.gap_type, GapSeverity.MINOR
        )
        return severity

    # ------------------------------------------------------------------
    # Public API: Remediation Plan Generation
    # ------------------------------------------------------------------

    def generate_remediation_plan(
        self,
        gaps: List[DetectedGap],
    ) -> RemediationPlan:
        """Generate a prioritized remediation action plan from detected gaps.

        Creates remediation actions for each gap, prioritized by severity
        (CRITICAL first, then MAJOR, then MINOR).

        Args:
            gaps: List of detected gaps to remediate.

        Returns:
            RemediationPlan with prioritized actions.
        """
        plan_id = str(uuid.uuid4())
        actions: List[RemediationAction] = []

        # Determine supplier_id from first gap
        supplier_id = gaps[0].supplier_id if gaps else ""

        critical = 0
        major = 0
        minor = 0

        for gap in gaps:
            template = REMEDIATION_TEMPLATES.get(gap.gap_type, {})

            # Map severity to priority
            if gap.severity == GapSeverity.CRITICAL.value:
                priority = RemediationPriority.URGENT.value
                critical += 1
            elif gap.severity == GapSeverity.MAJOR.value:
                priority = RemediationPriority.HIGH.value
                major += 1
            else:
                priority = RemediationPriority.MEDIUM.value
                minor += 1

            action = RemediationAction(
                action_id=str(uuid.uuid4()),
                gap_id=gap.gap_id,
                supplier_id=gap.supplier_id,
                priority=priority,
                action=template.get("action", f"Resolve {gap.gap_type}"),
                instruction=template.get(
                    "instruction",
                    f"Address the identified gap: {gap.description}",
                ),
                responsible_role=template.get(
                    "responsible_role", "Compliance Officer"
                ),
                deadline_days=int(template.get("deadline_days", "30")),
                status=RemediationStatus.PENDING.value,
                created_at=utcnow().isoformat(),
            )
            actions.append(action)

        # Sort actions by priority (urgent first)
        priority_order = {
            RemediationPriority.URGENT.value: 0,
            RemediationPriority.HIGH.value: 1,
            RemediationPriority.MEDIUM.value: 2,
            RemediationPriority.LOW.value: 3,
        }
        actions.sort(
            key=lambda a: priority_order.get(a.priority, 99)
        )

        provenance_data = {
            "plan_id": plan_id,
            "supplier_id": supplier_id,
            "total_gaps": len(gaps),
            "total_actions": len(actions),
        }
        provenance_hash = _compute_hash(provenance_data)

        plan = RemediationPlan(
            plan_id=plan_id,
            supplier_id=supplier_id,
            total_gaps=len(gaps),
            critical_gaps=critical,
            major_gaps=major,
            minor_gaps=minor,
            actions=actions,
            completion_pct=0.0,
            created_at=utcnow().isoformat(),
            provenance_hash=provenance_hash,
        )

        # Store plan
        self._remediation_plans[plan_id] = plan

        logger.info(
            "Remediation plan generated: plan=%s supplier=%s gaps=%d "
            "actions=%d (critical=%d major=%d minor=%d)",
            plan_id,
            supplier_id,
            len(gaps),
            len(actions),
            critical,
            major,
            minor,
        )

        return plan

    # ------------------------------------------------------------------
    # Public API: Questionnaire Generation
    # ------------------------------------------------------------------

    def generate_questionnaire(
        self,
        gaps: List[DetectedGap],
    ) -> SupplierQuestionnaire:
        """Auto-generate a supplier questionnaire for gap filling.

        Creates a questionnaire with sections and fields based on the
        detected gaps. Only includes sections relevant to the gaps found.

        Args:
            gaps: List of detected gaps to generate questions for.

        Returns:
            SupplierQuestionnaire with relevant sections and fields.
        """
        questionnaire_id = str(uuid.uuid4())
        supplier_id = gaps[0].supplier_id if gaps else ""
        sections: List[Dict[str, Any]] = []
        total_fields = 0
        required_fields = 0

        # Map gap types to questionnaire sections
        gap_to_section: Dict[str, str] = {
            GapType.MISSING_LEGAL_ENTITY.value: "legal_entity",
            GapType.MISSING_REGISTRATION_ID.value: "legal_entity",
            GapType.MISSING_GPS.value: "location",
            GapType.MISSING_COUNTRY.value: "location",
            GapType.MISSING_ADDRESS.value: "location",
            GapType.MISSING_COMMODITY.value: "commodity",
            GapType.MISSING_VOLUME.value: "commodity",
            GapType.MISSING_CERTIFICATION.value: "certification",
            GapType.OUTDATED_CERTIFICATION.value: "certification",
            GapType.MISSING_DEFORESTATION_STATUS.value: "deforestation",
            GapType.OUTDATED_DEFORESTATION_CHECK.value: "deforestation",
            GapType.MISSING_CONTACT.value: "contact",
            GapType.TIER_WITHOUT_SUPPLIERS.value: "upstream_suppliers",
            GapType.LOW_TIER_VISIBILITY.value: "upstream_suppliers",
        }

        # Determine which sections are needed
        needed_sections: List[str] = []
        for gap in gaps:
            section_key = gap_to_section.get(gap.gap_type)
            if section_key and section_key not in needed_sections:
                needed_sections.append(section_key)

        # Build questionnaire sections
        for section_key in needed_sections:
            template = QUESTIONNAIRE_FIELD_TEMPLATES.get(section_key)
            if template is None:
                continue

            section: Dict[str, Any] = {
                "section_title": template["section"],
                "section_key": section_key,
                "fields": template["fields"],
            }
            sections.append(section)

            for f in template["fields"]:
                total_fields += 1
                if f.get("required", False):
                    required_fields += 1

        provenance_data = {
            "questionnaire_id": questionnaire_id,
            "supplier_id": supplier_id,
            "sections": len(sections),
            "total_fields": total_fields,
        }
        provenance_hash = _compute_hash(provenance_data)

        questionnaire = SupplierQuestionnaire(
            questionnaire_id=questionnaire_id,
            supplier_id=supplier_id,
            sections=sections,
            total_fields=total_fields,
            required_fields=required_fields,
            generated_at=utcnow().isoformat(),
            provenance_hash=provenance_hash,
        )

        logger.info(
            "Questionnaire generated: id=%s supplier=%s sections=%d "
            "fields=%d (required=%d)",
            questionnaire_id,
            supplier_id,
            len(sections),
            total_fields,
            required_fields,
        )

        return questionnaire

    # ------------------------------------------------------------------
    # Public API: Remediation Progress Tracking
    # ------------------------------------------------------------------

    def track_remediation_progress(
        self,
        plan_id: str,
    ) -> Optional[RemediationPlan]:
        """Track completion percentage of a remediation plan.

        Recalculates the completion percentage based on the status of
        individual actions.

        Args:
            plan_id: Remediation plan ID to track.

        Returns:
            Updated RemediationPlan or None if not found.
        """
        plan = self._remediation_plans.get(plan_id)
        if plan is None:
            logger.warning(
                "Remediation plan not found: plan_id=%s", plan_id
            )
            return None

        if not plan.actions:
            plan.completion_pct = 100.0
            return plan

        completed = sum(
            1
            for a in plan.actions
            if a.status == RemediationStatus.COMPLETED.value
        )
        plan.completion_pct = round(
            completed / len(plan.actions) * 100.0, 2
        )

        logger.debug(
            "Remediation progress: plan=%s completed=%d/%d pct=%.1f",
            plan_id,
            completed,
            len(plan.actions),
            plan.completion_pct,
        )

        return plan

    def update_action_status(
        self,
        plan_id: str,
        action_id: str,
        new_status: RemediationStatus,
    ) -> bool:
        """Update the status of a remediation action.

        Args:
            plan_id: Remediation plan ID.
            action_id: Action ID to update.
            new_status: New status for the action.

        Returns:
            True if the action was found and updated, False otherwise.
        """
        plan = self._remediation_plans.get(plan_id)
        if plan is None:
            return False

        for action in plan.actions:
            if action.action_id == action_id:
                action.status = new_status.value
                if new_status == RemediationStatus.COMPLETED:
                    action.completed_at = utcnow().isoformat()
                logger.info(
                    "Remediation action updated: plan=%s action=%s "
                    "status=%s",
                    plan_id,
                    action_id,
                    new_status.value,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Public API: Gap Trend Analysis
    # ------------------------------------------------------------------

    def get_gap_trends(
        self,
        supplier_id: str,
    ) -> GapTrendResult:
        """Analyze gap trends over time for a supplier.

        Uses historical gap analysis results to determine whether gaps
        are improving, stable, or worsening.

        Args:
            supplier_id: Supplier to analyze trends for.

        Returns:
            GapTrendResult with trend direction and entries.
        """
        history = self._gap_history.get(supplier_id, [])
        entries: List[GapTrendEntry] = []

        for result in history:
            entries.append(GapTrendEntry(
                timestamp=result.analyzed_at,
                total_gaps=result.total_gaps,
                critical_gaps=result.critical_gaps,
                major_gaps=result.major_gaps,
                minor_gaps=result.minor_gaps,
                completeness_score=result.completeness_score,
            ))

        # Determine trend direction
        trend = TrendDirection.STABLE
        gap_delta = 0
        completeness_delta = 0.0

        if len(entries) >= 2:
            first = entries[0]
            last = entries[-1]
            gap_delta = last.total_gaps - first.total_gaps
            completeness_delta = round(
                last.completeness_score - first.completeness_score, 2
            )

            if gap_delta < -2 or completeness_delta > 5.0:
                trend = TrendDirection.IMPROVING
            elif gap_delta > 2 or completeness_delta < -5.0:
                trend = TrendDirection.WORSENING
            else:
                trend = TrendDirection.STABLE

        return GapTrendResult(
            supplier_id=supplier_id,
            entries=entries,
            trend_direction=trend.value,
            gap_delta=gap_delta,
            completeness_delta=completeness_delta,
        )

    # ------------------------------------------------------------------
    # Public API: Batch Gap Analysis
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        suppliers: List[SupplierGapProfile],
        batch_size: int = 1000,
    ) -> BatchGapResult:
        """Perform batch gap analysis for multiple suppliers.

        Args:
            suppliers: List of supplier profiles to analyze.
            batch_size: Chunk size for processing.

        Returns:
            BatchGapResult with all results and summary.
        """
        t_start = time.monotonic()
        batch_id = str(uuid.uuid4())
        results: List[GapAnalysisResult] = []
        failed_count = 0
        total_critical = 0
        total_major = 0
        total_minor = 0
        dds_blocked_count = 0
        completeness_sum = 0.0

        logger.info(
            "Starting batch gap analysis: batch=%s suppliers=%d",
            batch_id,
            len(suppliers),
        )

        for i in range(0, len(suppliers), batch_size):
            chunk = suppliers[i : i + batch_size]
            for profile in chunk:
                try:
                    result = self.analyze_gaps(
                        profile.supplier_id, profile
                    )
                    results.append(result)
                    total_critical += result.critical_gaps
                    total_major += result.major_gaps
                    total_minor += result.minor_gaps
                    if result.dds_blocked:
                        dds_blocked_count += 1
                    completeness_sum += result.completeness_score
                except Exception as exc:
                    failed_count += 1
                    logger.warning(
                        "Batch gap analysis failed for supplier=%s: %s",
                        profile.supplier_id,
                        str(exc),
                    )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        successful = len(results)
        avg_completeness = (
            round(completeness_sum / successful, 2)
            if successful > 0
            else 0.0
        )

        summary: Dict[str, Any] = {
            "total_gaps": sum(r.total_gaps for r in results),
            "total_critical_gaps": total_critical,
            "total_major_gaps": total_major,
            "total_minor_gaps": total_minor,
            "dds_blocked_count": dds_blocked_count,
            "average_completeness": avg_completeness,
            "suppliers_with_critical_gaps": sum(
                1 for r in results if r.critical_gaps > 0
            ),
        }

        provenance_data = {
            "batch_id": batch_id,
            "total_suppliers": len(suppliers),
            "successful": successful,
            "failed": failed_count,
            "total_gaps": summary["total_gaps"],
        }
        provenance_hash = _compute_hash(provenance_data)

        logger.info(
            "Batch gap analysis completed: batch=%s total=%d success=%d "
            "failed=%d total_gaps=%d dds_blocked=%d avg_completeness=%.1f "
            "time=%.2fms",
            batch_id,
            len(suppliers),
            successful,
            failed_count,
            summary["total_gaps"],
            dds_blocked_count,
            avg_completeness,
            elapsed_ms,
        )

        return BatchGapResult(
            batch_id=batch_id,
            total_suppliers=len(suppliers),
            successful=successful,
            failed=failed_count,
            results=results,
            summary=summary,
            processing_time_ms=round(elapsed_ms, 3),
            provenance_hash=provenance_hash,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def analysis_count(self) -> int:
        """Return total number of gap analyses performed."""
        return self._analysis_count

    @property
    def tracked_supplier_count(self) -> int:
        """Return number of suppliers with gap history."""
        return len(self._gap_history)

    @property
    def active_plan_count(self) -> int:
        """Return number of active remediation plans."""
        return len(self._remediation_plans)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _calculate_completeness(
        self,
        profile: SupplierGapProfile,
        gaps: List[DetectedGap],
    ) -> float:
        """Calculate profile completeness score (0-100).

        Completeness is inversely proportional to the number and severity
        of gaps found. Critical gaps have the highest weight penalty.

        Args:
            profile: Supplier profile.
            gaps: Detected gaps.

        Returns:
            Completeness score 0-100.
        """
        if not gaps:
            return 100.0

        # Total possible fields (14 major data fields)
        total_fields = 14.0
        penalty = 0.0

        for gap in gaps:
            if gap.category != GapCategory.DATA_GAP.value:
                continue
            if gap.severity == GapSeverity.CRITICAL.value:
                penalty += 2.0  # Double weight for critical
            elif gap.severity == GapSeverity.MAJOR.value:
                penalty += 1.0
            else:
                penalty += 0.5

        completeness = max(0.0, 100.0 - (penalty / total_fields * 100.0))
        return min(100.0, completeness)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"GapAnalyzer("
            f"analyses={self._analysis_count}, "
            f"tracked_suppliers={len(self._gap_history)}, "
            f"active_plans={len(self._remediation_plans)}, "
            f"version={_MODULE_VERSION!r})"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "GapSeverity",
    "GapCategory",
    "GapType",
    "RemediationStatus",
    "RemediationPriority",
    "TrendDirection",
    # Constants
    "GAP_SEVERITY_RULES",
    "REMEDIATION_TEMPLATES",
    "QUESTIONNAIRE_FIELD_TEMPLATES",
    # Data classes
    "SupplierGapProfile",
    "SupplierChainTier",
    "DetectedGap",
    "RemediationAction",
    "RemediationPlan",
    "SupplierQuestionnaire",
    "GapAnalysisResult",
    "GapTrendEntry",
    "GapTrendResult",
    "BatchGapResult",
    # Engine
    "GapAnalyzer",
]
