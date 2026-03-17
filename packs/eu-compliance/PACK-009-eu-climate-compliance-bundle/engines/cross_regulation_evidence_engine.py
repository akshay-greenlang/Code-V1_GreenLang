# -*- coding: utf-8 -*-
"""
CrossRegulationEvidenceEngine - PACK-009 EU Climate Compliance Bundle Engine 8

Manages a unified evidence repository spanning all 4 EU regulations (CSRD,
CBAM, EU Taxonomy, EUDR). Tracks evidence items, maps them to regulatory
requirements, identifies reusable evidence across regulations, detects
coverage gaps, monitors expiring evidence, and calculates cost savings
from evidence reuse.

Capabilities:
    1. Register evidence items with SHA-256 file hashes
    2. Map evidence to regulatory requirements across regulations
    3. Find reusable evidence (single audit report satisfies multiple regs)
    4. Check evidence completeness per regulation and requirement
    5. Identify gaps in evidence coverage
    6. Track expiring evidence and alert on upcoming expirations
    7. Calculate reuse savings (audit cost reduction)
    8. Compute deterministic SHA-256 provenance hashes

Evidence Reuse Principle:
    A single audit report, policy document, or data extract can satisfy
    requirements in CSRD, EU Taxonomy, CBAM, and EUDR simultaneously.
    The engine tracks these cross-regulation mappings to avoid redundant
    evidence collection and reduce compliance costs.

Zero-Hallucination:
    - Coverage percentages from deterministic field-presence counts
    - Savings calculations from configurable cost-per-evidence tables
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on all results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-009 EU Climate Compliance Bundle
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    elif isinstance(data, list):
        serializable = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in data
        ]
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _parse_date(value: Any) -> Optional[date]:
    """Parse a date from a string or date/datetime object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RegulationType(str, Enum):
    """Supported EU regulations."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class EvidenceType(str, Enum):
    """Categories of evidence artefacts."""
    AUDIT_REPORT = "AUDIT_REPORT"
    POLICY_DOCUMENT = "POLICY_DOCUMENT"
    DATA_EXTRACT = "DATA_EXTRACT"
    CERTIFICATE = "CERTIFICATE"
    VERIFICATION_STATEMENT = "VERIFICATION_STATEMENT"
    BOARD_MINUTES = "BOARD_MINUTES"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    DUE_DILIGENCE_REPORT = "DUE_DILIGENCE_REPORT"
    SUPPLIER_DECLARATION = "SUPPLIER_DECLARATION"
    GEOLOCATION_DATA = "GEOLOCATION_DATA"
    SATELLITE_IMAGERY = "SATELLITE_IMAGERY"
    EMISSION_CALCULATION = "EMISSION_CALCULATION"
    FINANCIAL_STATEMENT = "FINANCIAL_STATEMENT"
    TRAINING_RECORD = "TRAINING_RECORD"
    STAKEHOLDER_ENGAGEMENT = "STAKEHOLDER_ENGAGEMENT"
    METHODOLOGY_DOCUMENT = "METHODOLOGY_DOCUMENT"
    EXTERNAL_ASSURANCE = "EXTERNAL_ASSURANCE"
    INTERNAL_CONTROL = "INTERNAL_CONTROL"


class EvidenceStatus(str, Enum):
    """Lifecycle status of an evidence item."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    UNDER_REVIEW = "UNDER_REVIEW"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


# ---------------------------------------------------------------------------
# Reference Data - Evidence Requirements
# ---------------------------------------------------------------------------


EVIDENCE_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "CSRD": {
        "CSRD-GOV-1": ["POLICY_DOCUMENT", "BOARD_MINUTES"],
        "CSRD-GOV-2": ["POLICY_DOCUMENT", "BOARD_MINUTES", "TRAINING_RECORD"],
        "CSRD-GOV-3": ["POLICY_DOCUMENT", "STAKEHOLDER_ENGAGEMENT"],
        "CSRD-SBM-1": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT"],
        "CSRD-SBM-2": ["DATA_EXTRACT", "RISK_ASSESSMENT"],
        "CSRD-SBM-3": ["STAKEHOLDER_ENGAGEMENT", "DATA_EXTRACT"],
        "CSRD-IRO-1": ["RISK_ASSESSMENT", "METHODOLOGY_DOCUMENT"],
        "CSRD-IRO-2": ["RISK_ASSESSMENT", "DATA_EXTRACT"],
        "CSRD-E1-1": ["POLICY_DOCUMENT", "EMISSION_CALCULATION"],
        "CSRD-E1-2": ["EMISSION_CALCULATION", "DATA_EXTRACT"],
        "CSRD-E1-3": ["EMISSION_CALCULATION", "VERIFICATION_STATEMENT"],
        "CSRD-E1-4": ["EMISSION_CALCULATION", "METHODOLOGY_DOCUMENT"],
        "CSRD-E1-5": ["DATA_EXTRACT", "FINANCIAL_STATEMENT"],
        "CSRD-E1-6": ["EMISSION_CALCULATION", "DATA_EXTRACT", "EXTERNAL_ASSURANCE"],
        "CSRD-E1-7": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT"],
        "CSRD-E1-8": ["INTERNAL_CONTROL", "DATA_EXTRACT"],
        "CSRD-E1-9": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "CSRD-E4-1": ["RISK_ASSESSMENT", "DATA_EXTRACT"],
        "CSRD-E4-2": ["DATA_EXTRACT", "SATELLITE_IMAGERY"],
        "CSRD-S1-1": ["POLICY_DOCUMENT", "DUE_DILIGENCE_REPORT"],
    },
    "CBAM": {
        "CBAM-DEC-Q1": ["DATA_EXTRACT", "EMISSION_CALCULATION", "SUPPLIER_DECLARATION"],
        "CBAM-DEC-Q2": ["DATA_EXTRACT", "EMISSION_CALCULATION", "SUPPLIER_DECLARATION"],
        "CBAM-DEC-Q3": ["DATA_EXTRACT", "EMISSION_CALCULATION", "SUPPLIER_DECLARATION"],
        "CBAM-DEC-Q4": ["DATA_EXTRACT", "EMISSION_CALCULATION", "SUPPLIER_DECLARATION"],
        "CBAM-VER-01": ["VERIFICATION_STATEMENT", "AUDIT_REPORT"],
        "CBAM-VER-02": ["VERIFICATION_STATEMENT", "METHODOLOGY_DOCUMENT"],
        "CBAM-SUP-01": ["SUPPLIER_DECLARATION", "CERTIFICATE"],
        "CBAM-SUP-02": ["SUPPLIER_DECLARATION", "DATA_EXTRACT"],
        "CBAM-CERT-01": ["CERTIFICATE", "FINANCIAL_STATEMENT"],
        "CBAM-CERT-02": ["CERTIFICATE", "DATA_EXTRACT"],
        "CBAM-PREC-01": ["DATA_EXTRACT", "EMISSION_CALCULATION", "METHODOLOGY_DOCUMENT"],
        "CBAM-PREC-02": ["SUPPLIER_DECLARATION", "EMISSION_CALCULATION"],
        "CBAM-CP-01": ["CERTIFICATE", "FINANCIAL_STATEMENT"],
        "CBAM-NCA-01": ["AUDIT_REPORT", "INTERNAL_CONTROL", "DATA_EXTRACT"],
        "CBAM-NCA-02": ["DATA_EXTRACT", "VERIFICATION_STATEMENT"],
        "CBAM-FIN-01": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "CBAM-FIN-02": ["CERTIFICATE", "FINANCIAL_STATEMENT"],
        "CBAM-REP-01": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT"],
        "CBAM-REP-02": ["DATA_EXTRACT", "EMISSION_CALCULATION"],
        "CBAM-CN-01": ["DATA_EXTRACT", "CERTIFICATE"],
    },
    "EU_TAXONOMY": {
        "TAX-ELG-01": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT"],
        "TAX-ELG-02": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "TAX-ALN-01": ["DATA_EXTRACT", "VERIFICATION_STATEMENT"],
        "TAX-ALN-02": ["DATA_EXTRACT", "EMISSION_CALCULATION"],
        "TAX-SC-01": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT", "EMISSION_CALCULATION"],
        "TAX-SC-02": ["DATA_EXTRACT", "CERTIFICATE"],
        "TAX-DNSH-01": ["RISK_ASSESSMENT", "DATA_EXTRACT"],
        "TAX-DNSH-02": ["DATA_EXTRACT", "POLICY_DOCUMENT"],
        "TAX-DNSH-03": ["DATA_EXTRACT", "RISK_ASSESSMENT"],
        "TAX-DNSH-04": ["DATA_EXTRACT", "POLICY_DOCUMENT"],
        "TAX-DNSH-05": ["RISK_ASSESSMENT", "DATA_EXTRACT"],
        "TAX-DNSH-06": ["DATA_EXTRACT", "RISK_ASSESSMENT"],
        "TAX-MS-01": ["POLICY_DOCUMENT", "DUE_DILIGENCE_REPORT"],
        "TAX-MS-02": ["POLICY_DOCUMENT", "TRAINING_RECORD"],
        "TAX-MS-03": ["POLICY_DOCUMENT", "INTERNAL_CONTROL"],
        "TAX-MS-04": ["POLICY_DOCUMENT", "DUE_DILIGENCE_REPORT"],
        "TAX-KPI-01": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "TAX-KPI-02": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "TAX-KPI-03": ["FINANCIAL_STATEMENT", "DATA_EXTRACT"],
        "TAX-DIS-01": ["FINANCIAL_STATEMENT", "EXTERNAL_ASSURANCE"],
    },
    "EUDR": {
        "EUDR-COM-01": ["DATA_EXTRACT", "SUPPLIER_DECLARATION"],
        "EUDR-COM-02": ["SUPPLIER_DECLARATION", "CERTIFICATE"],
        "EUDR-GEO-01": ["GEOLOCATION_DATA", "SATELLITE_IMAGERY"],
        "EUDR-GEO-02": ["GEOLOCATION_DATA", "DATA_EXTRACT"],
        "EUDR-RA-01": ["RISK_ASSESSMENT", "DATA_EXTRACT"],
        "EUDR-RA-02": ["RISK_ASSESSMENT", "SATELLITE_IMAGERY"],
        "EUDR-RA-03": ["RISK_ASSESSMENT", "SUPPLIER_DECLARATION"],
        "EUDR-DD-01": ["DUE_DILIGENCE_REPORT", "DATA_EXTRACT"],
        "EUDR-DD-02": ["DUE_DILIGENCE_REPORT", "RISK_ASSESSMENT"],
        "EUDR-DD-03": ["DUE_DILIGENCE_REPORT", "SUPPLIER_DECLARATION"],
        "EUDR-SAT-01": ["SATELLITE_IMAGERY", "DATA_EXTRACT"],
        "EUDR-SAT-02": ["SATELLITE_IMAGERY", "METHODOLOGY_DOCUMENT"],
        "EUDR-TRACE-01": ["DATA_EXTRACT", "SUPPLIER_DECLARATION", "GEOLOCATION_DATA"],
        "EUDR-TRACE-02": ["SUPPLIER_DECLARATION", "CERTIFICATE"],
        "EUDR-MIT-01": ["POLICY_DOCUMENT", "DUE_DILIGENCE_REPORT"],
        "EUDR-MIT-02": ["SUPPLIER_DECLARATION", "DATA_EXTRACT"],
        "EUDR-NCA-01": ["AUDIT_REPORT", "DATA_EXTRACT", "DUE_DILIGENCE_REPORT"],
        "EUDR-NCA-02": ["DATA_EXTRACT", "INTERNAL_CONTROL"],
        "EUDR-LEG-01": ["DATA_EXTRACT", "CERTIFICATE"],
        "EUDR-INF-01": ["DATA_EXTRACT", "METHODOLOGY_DOCUMENT"],
    },
}


EVIDENCE_REUSE_MAP: Dict[str, Dict[str, List[str]]] = {
    "AUDIT_REPORT": {
        "description": "Third-party audit report on sustainability controls",
        "satisfies": [
            "CSRD-E1-6", "CSRD-IRO-1",
            "CBAM-VER-01", "CBAM-NCA-01",
            "TAX-DIS-01",
            "EUDR-NCA-01",
        ],
    },
    "POLICY_DOCUMENT:sustainability": {
        "description": "Overarching sustainability/ESG policy",
        "satisfies": [
            "CSRD-GOV-1", "CSRD-GOV-2", "CSRD-GOV-3", "CSRD-E1-1",
            "TAX-DNSH-02", "TAX-DNSH-04", "TAX-MS-01", "TAX-MS-02", "TAX-MS-03",
            "EUDR-MIT-01",
        ],
    },
    "POLICY_DOCUMENT:human_rights": {
        "description": "Human rights and due diligence policy",
        "satisfies": [
            "CSRD-S1-1",
            "TAX-MS-01", "TAX-MS-04",
            "EUDR-MIT-01",
        ],
    },
    "DUE_DILIGENCE_REPORT": {
        "description": "Due diligence report covering supply chain risks",
        "satisfies": [
            "CSRD-S1-1",
            "TAX-MS-01", "TAX-MS-04",
            "EUDR-DD-01", "EUDR-DD-02", "EUDR-DD-03", "EUDR-MIT-01", "EUDR-NCA-01",
        ],
    },
    "EMISSION_CALCULATION:ghg_inventory": {
        "description": "Organisation-wide GHG emissions inventory",
        "satisfies": [
            "CSRD-E1-2", "CSRD-E1-3", "CSRD-E1-4", "CSRD-E1-6",
            "CBAM-DEC-Q1", "CBAM-DEC-Q2", "CBAM-DEC-Q3", "CBAM-DEC-Q4",
            "CBAM-PREC-01", "CBAM-PREC-02", "CBAM-REP-02",
            "TAX-ALN-02", "TAX-SC-01",
        ],
    },
    "FINANCIAL_STATEMENT:annual": {
        "description": "Annual financial statement with sustainability disclosures",
        "satisfies": [
            "CSRD-E1-5", "CSRD-E1-9",
            "CBAM-FIN-01", "CBAM-FIN-02", "CBAM-CERT-01",
            "TAX-ELG-02", "TAX-KPI-01", "TAX-KPI-02", "TAX-KPI-03", "TAX-DIS-01",
        ],
    },
    "RISK_ASSESSMENT:climate": {
        "description": "Climate-related risk assessment (physical and transition)",
        "satisfies": [
            "CSRD-SBM-2", "CSRD-IRO-1", "CSRD-IRO-2",
            "TAX-DNSH-01", "TAX-DNSH-03", "TAX-DNSH-05", "TAX-DNSH-06",
            "EUDR-RA-01", "EUDR-RA-02", "EUDR-DD-02",
        ],
    },
    "SUPPLIER_DECLARATION:emissions": {
        "description": "Supplier-specific emissions declaration",
        "satisfies": [
            "CBAM-DEC-Q1", "CBAM-DEC-Q2", "CBAM-DEC-Q3", "CBAM-DEC-Q4",
            "CBAM-SUP-01", "CBAM-SUP-02", "CBAM-PREC-02",
            "EUDR-COM-01", "EUDR-RA-03", "EUDR-DD-03", "EUDR-TRACE-02", "EUDR-MIT-02",
        ],
    },
    "VERIFICATION_STATEMENT:emissions": {
        "description": "Third-party verification of emissions data",
        "satisfies": [
            "CSRD-E1-3",
            "CBAM-VER-01", "CBAM-VER-02", "CBAM-NCA-02",
            "TAX-ALN-01",
        ],
    },
    "INTERNAL_CONTROL:compliance": {
        "description": "Internal controls framework for regulatory compliance",
        "satisfies": [
            "CSRD-E1-8",
            "CBAM-NCA-01",
            "TAX-MS-03",
            "EUDR-NCA-02",
        ],
    },
    "METHODOLOGY_DOCUMENT:ghg": {
        "description": "GHG accounting methodology document",
        "satisfies": [
            "CSRD-E1-4", "CSRD-E1-7",
            "CBAM-PREC-01", "CBAM-REP-01",
            "TAX-ELG-01", "TAX-SC-01",
        ],
    },
    "SATELLITE_IMAGERY:deforestation": {
        "description": "Satellite imagery for deforestation monitoring",
        "satisfies": [
            "CSRD-E4-2",
            "EUDR-GEO-01", "EUDR-RA-02", "EUDR-SAT-01", "EUDR-SAT-02",
        ],
    },
    "GEOLOCATION_DATA:supply_chain": {
        "description": "Geolocation data for supply chain traceability",
        "satisfies": [
            "EUDR-GEO-01", "EUDR-GEO-02", "EUDR-TRACE-01",
        ],
    },
}


# Average cost per evidence collection (EUR) for savings calculation
_EVIDENCE_COST_TABLE: Dict[str, float] = {
    "AUDIT_REPORT": 15000.0,
    "POLICY_DOCUMENT": 2000.0,
    "DATA_EXTRACT": 500.0,
    "CERTIFICATE": 1000.0,
    "VERIFICATION_STATEMENT": 8000.0,
    "BOARD_MINUTES": 200.0,
    "RISK_ASSESSMENT": 5000.0,
    "DUE_DILIGENCE_REPORT": 10000.0,
    "SUPPLIER_DECLARATION": 300.0,
    "GEOLOCATION_DATA": 1500.0,
    "SATELLITE_IMAGERY": 3000.0,
    "EMISSION_CALCULATION": 4000.0,
    "FINANCIAL_STATEMENT": 1000.0,
    "TRAINING_RECORD": 500.0,
    "STAKEHOLDER_ENGAGEMENT": 2000.0,
    "METHODOLOGY_DOCUMENT": 3000.0,
    "EXTERNAL_ASSURANCE": 12000.0,
    "INTERNAL_CONTROL": 2500.0,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EvidenceConfig(BaseModel):
    """Configuration for the CrossRegulationEvidenceEngine."""

    require_hash: bool = Field(
        default=True,
        description="Require SHA-256 file hash on every evidence item",
    )
    max_evidence_age_days: int = Field(
        default=365, ge=30, le=1825,
        description="Maximum age of evidence in days before it is considered stale",
    )
    enable_reuse_tracking: bool = Field(
        default=True,
        description="Enable cross-regulation evidence reuse tracking",
    )
    expiry_warning_days: int = Field(
        default=60, ge=7, le=180,
        description="Days before expiry to trigger a warning",
    )
    cost_per_evidence: Dict[str, float] = Field(
        default_factory=lambda: dict(_EVIDENCE_COST_TABLE),
        description="Cost table for savings calculation (evidence_type -> EUR)",
    )


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A single evidence artefact in the repository."""

    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence identifier")
    title: str = Field(..., description="Descriptive title of the evidence")
    evidence_type: str = Field(..., description="Type of evidence (EvidenceType enum value)")
    evidence_subtype: str = Field(default="", description="Optional subtype for reuse mapping")
    file_hash: str = Field(default="", description="SHA-256 hash of the evidence file")
    regulations: List[str] = Field(
        default_factory=list, description="Regulations this evidence applies to"
    )
    requirements_satisfied: List[str] = Field(
        default_factory=list, description="Requirement IDs satisfied by this evidence"
    )
    upload_date: str = Field(default="", description="ISO-8601 upload date")
    expiry_date: str = Field(default="", description="ISO-8601 expiry date (empty = no expiry)")
    status: str = Field(default="ACTIVE", description="Evidence status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EvidenceMapping(BaseModel):
    """Mapping of an evidence item to a specific regulatory requirement."""

    mapping_id: str = Field(default_factory=_new_uuid, description="Mapping identifier")
    evidence_id: str = Field(..., description="Evidence item identifier")
    regulation: str = Field(..., description="Regulation identifier")
    requirement_id: str = Field(..., description="Requirement identifier")
    coverage_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Percentage of the requirement covered by this evidence",
    )


class EvidenceGap(BaseModel):
    """A missing evidence item for a requirement."""

    regulation: str = Field(..., description="Regulation identifier")
    requirement_id: str = Field(..., description="Requirement identifier")
    required_types: List[str] = Field(
        default_factory=list, description="Evidence types needed"
    )
    missing_types: List[str] = Field(
        default_factory=list, description="Evidence types not yet provided"
    )
    coverage_pct: float = Field(default=0.0, description="Current coverage percentage")


class ExpiringEvidence(BaseModel):
    """An evidence item approaching its expiry date."""

    evidence_id: str = Field(..., description="Evidence identifier")
    title: str = Field(default="", description="Evidence title")
    expiry_date: str = Field(default="", description="ISO-8601 expiry date")
    days_until_expiry: int = Field(default=0, description="Days remaining until expiry")
    regulations_affected: List[str] = Field(
        default_factory=list, description="Regulations affected by expiry"
    )
    requirements_affected: List[str] = Field(
        default_factory=list, description="Requirements affected by expiry"
    )


class ReuseSavings(BaseModel):
    """Calculated savings from evidence reuse across regulations."""

    total_evidence_items: int = Field(default=0, description="Total evidence items registered")
    reused_items: int = Field(default=0, description="Items reused across 2+ regulations")
    unique_requirements_satisfied: int = Field(default=0, description="Total requirements satisfied")
    without_reuse_cost_eur: float = Field(default=0.0, description="Cost if each requirement collected separately")
    with_reuse_cost_eur: float = Field(default=0.0, description="Cost with evidence reuse")
    savings_eur: float = Field(default=0.0, description="Total savings in EUR")
    savings_pct: float = Field(default=0.0, description="Savings as percentage")


class CoverageMatrix(BaseModel):
    """Coverage matrix showing evidence coverage per regulation and requirement."""

    regulation: str = Field(..., description="Regulation identifier")
    total_requirements: int = Field(default=0, description="Total requirements for this regulation")
    covered_requirements: int = Field(default=0, description="Requirements with sufficient evidence")
    coverage_pct: float = Field(default=0.0, description="Coverage percentage")
    requirement_details: Dict[str, float] = Field(
        default_factory=dict, description="Requirement ID -> coverage percentage"
    )


class EvidenceResult(BaseModel):
    """Complete result from the CrossRegulationEvidenceEngine."""

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    items: List[EvidenceItem] = Field(default_factory=list, description="All registered evidence items")
    mappings: List[EvidenceMapping] = Field(
        default_factory=list, description="Evidence-to-requirement mappings"
    )
    coverage_matrix: List[CoverageMatrix] = Field(
        default_factory=list, description="Per-regulation coverage matrix"
    )
    reuse_count: int = Field(default=0, description="Number of evidence items reused across regulations")
    gaps: List[EvidenceGap] = Field(default_factory=list, description="Evidence coverage gaps")
    expiring: List[ExpiringEvidence] = Field(
        default_factory=list, description="Evidence items approaching expiry"
    )
    savings: Optional[ReuseSavings] = Field(default=None, description="Reuse savings calculation")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CrossRegulationEvidenceEngine:
    """
    Unified evidence repository engine spanning all 4 EU regulations.

    Manages evidence lifecycle, maps items to requirements, tracks reuse,
    identifies gaps, and calculates compliance cost savings. All calculations
    are deterministic - coverage is field-presence counting, savings use a
    fixed cost table.

    Attributes:
        config: Engine configuration.
        _items: Internal evidence item store.
        _mappings: Internal evidence-to-requirement mapping store.

    Example:
        >>> config = EvidenceConfig()
        >>> engine = CrossRegulationEvidenceEngine(config)
        >>> item = EvidenceItem(title="GHG Audit 2025", evidence_type="AUDIT_REPORT",
        ...                     regulations=["CSRD", "CBAM"])
        >>> engine.register_evidence(item)
        >>> result = engine.check_completeness()
        >>> print(result.reuse_count)
    """

    def __init__(self, config: Optional[EvidenceConfig] = None) -> None:
        """Initialize the CrossRegulationEvidenceEngine.

        Args:
            config: Engine configuration. Uses defaults when *None*.
        """
        self.config = config or EvidenceConfig()
        self._items: List[EvidenceItem] = []
        self._mappings: List[EvidenceMapping] = []
        logger.info("CrossRegulationEvidenceEngine v%s initialised", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_evidence(self, item: EvidenceItem) -> EvidenceItem:
        """Register a new evidence item in the repository.

        If require_hash is True and file_hash is empty, a provenance hash
        is generated from the item content. The item is then auto-mapped
        to requirements based on its type and regulations.

        Args:
            item: The evidence item to register.

        Returns:
            The registered item with provenance hash and mappings populated.
        """
        if not item.upload_date:
            item.upload_date = _utcnow().date().isoformat()

        if self.config.require_hash and not item.file_hash:
            item.file_hash = self.compute_hash(item.title + item.evidence_type + item.upload_date)

        item.provenance_hash = _compute_hash(item)
        self._items.append(item)

        # Auto-map to requirements
        new_mappings = self.map_to_requirements(item)
        self._mappings.extend(new_mappings)

        # Update requirements_satisfied on the item
        item.requirements_satisfied = [m.requirement_id for m in new_mappings]

        logger.info(
            "Registered evidence '%s' (type=%s, regulations=%s, mapped=%d requirements)",
            item.title, item.evidence_type, item.regulations, len(new_mappings),
        )
        return item

    def map_to_requirements(self, item: EvidenceItem) -> List[EvidenceMapping]:
        """Map an evidence item to regulatory requirements it can satisfy.

        Matching is based on evidence_type against EVIDENCE_REQUIREMENTS.

        Args:
            item: The evidence item to map.

        Returns:
            List of EvidenceMapping objects.
        """
        mappings: List[EvidenceMapping] = []
        target_regs = item.regulations if item.regulations else [r.value for r in RegulationType]

        for reg in target_regs:
            req_dict = EVIDENCE_REQUIREMENTS.get(reg, {})
            for req_id, required_types in req_dict.items():
                if item.evidence_type in required_types:
                    # Calculate coverage: this evidence covers 1/N of the requirement types
                    coverage = _safe_div(1.0, len(required_types)) * 100.0
                    mappings.append(EvidenceMapping(
                        evidence_id=item.evidence_id,
                        regulation=reg,
                        requirement_id=req_id,
                        coverage_pct=round(coverage, 2),
                    ))

        return mappings

    def find_reusable_evidence(self) -> Dict[str, List[str]]:
        """Find evidence items that are reused across multiple regulations.

        Returns:
            Dict mapping evidence_id to list of regulations it serves.
        """
        reuse: Dict[str, Set[str]] = {}
        for mapping in self._mappings:
            reuse.setdefault(mapping.evidence_id, set()).add(mapping.regulation)

        return {
            eid: sorted(regs)
            for eid, regs in reuse.items()
            if len(regs) >= 2
        }

    def check_completeness(self) -> EvidenceResult:
        """Check evidence completeness across all regulations and requirements.

        Returns:
            EvidenceResult with coverage matrix, gaps, expiring items, and savings.
        """
        start = _utcnow()

        coverage_matrix = self._build_coverage_matrix()
        gaps = self.identify_gaps()
        expiring = self.get_expiring_evidence()
        reusable = self.find_reusable_evidence()
        reuse_count = len(reusable)
        savings = self.calculate_reuse_savings() if self.config.enable_reuse_tracking else None

        elapsed_ms = (_utcnow() - start).total_seconds() * 1000

        result = EvidenceResult(
            items=list(self._items),
            mappings=list(self._mappings),
            coverage_matrix=coverage_matrix,
            reuse_count=reuse_count,
            gaps=gaps,
            expiring=expiring,
            savings=savings,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        logger.info(
            "Completeness check: items=%d, reused=%d, gaps=%d, expiring=%d",
            len(self._items), reuse_count, len(gaps), len(expiring),
        )
        return result

    def identify_gaps(self) -> List[EvidenceGap]:
        """Identify requirements with insufficient evidence coverage.

        Returns:
            List of EvidenceGap objects for requirements not fully covered.
        """
        gaps: List[EvidenceGap] = []

        for reg, req_dict in EVIDENCE_REQUIREMENTS.items():
            for req_id, required_types in req_dict.items():
                # Find all mappings for this requirement
                req_mappings = [
                    m for m in self._mappings
                    if m.regulation == reg and m.requirement_id == req_id
                ]

                # Determine which evidence types have been provided
                provided_types: Set[str] = set()
                for m in req_mappings:
                    for item in self._items:
                        if item.evidence_id == m.evidence_id and item.status == EvidenceStatus.ACTIVE.value:
                            provided_types.add(item.evidence_type)

                missing = [t for t in required_types if t not in provided_types]
                coverage = _safe_div(
                    len(required_types) - len(missing), len(required_types)
                ) * 100.0

                if coverage < 100.0:
                    gaps.append(EvidenceGap(
                        regulation=reg,
                        requirement_id=req_id,
                        required_types=required_types,
                        missing_types=missing,
                        coverage_pct=round(coverage, 2),
                    ))

        return gaps

    def get_expiring_evidence(self) -> List[ExpiringEvidence]:
        """Get evidence items approaching expiry.

        Returns:
            List of ExpiringEvidence objects within the warning window.
        """
        today = _utcnow().date()
        warning_cutoff = today + timedelta(days=self.config.expiry_warning_days)
        expiring: List[ExpiringEvidence] = []

        for item in self._items:
            if item.status != EvidenceStatus.ACTIVE.value:
                continue
            if not item.expiry_date:
                continue

            exp_date = _parse_date(item.expiry_date)
            if exp_date is None:
                continue

            if exp_date <= warning_cutoff:
                days_left = (exp_date - today).days

                # Find affected regulations and requirements
                affected_regs: Set[str] = set()
                affected_reqs: Set[str] = set()
                for m in self._mappings:
                    if m.evidence_id == item.evidence_id:
                        affected_regs.add(m.regulation)
                        affected_reqs.add(m.requirement_id)

                expiring.append(ExpiringEvidence(
                    evidence_id=item.evidence_id,
                    title=item.title,
                    expiry_date=item.expiry_date,
                    days_until_expiry=max(days_left, 0),
                    regulations_affected=sorted(affected_regs),
                    requirements_affected=sorted(affected_reqs),
                ))

        # Sort by days until expiry (most urgent first)
        expiring.sort(key=lambda e: e.days_until_expiry)
        return expiring

    def calculate_reuse_savings(self) -> ReuseSavings:
        """Calculate cost savings from evidence reuse across regulations.

        Without reuse, each regulation would need its own copy of every
        evidence type. With reuse, a single item covers multiple regulations.

        Returns:
            ReuseSavings with cost comparison and percentage saved.
        """
        reusable = self.find_reusable_evidence()

        # Total unique requirements satisfied
        all_req_ids: Set[str] = set()
        for m in self._mappings:
            all_req_ids.add(f"{m.regulation}:{m.requirement_id}")

        # Cost without reuse: each regulation-requirement pair collects independently
        without_reuse = 0.0
        for reg, req_dict in EVIDENCE_REQUIREMENTS.items():
            for req_id, required_types in req_dict.items():
                for etype in required_types:
                    cost = self.config.cost_per_evidence.get(etype, 1000.0)
                    without_reuse += cost

        # Cost with reuse: each unique evidence item is collected once
        with_reuse = 0.0
        seen_items: Set[str] = set()
        for item in self._items:
            if item.evidence_id not in seen_items:
                cost = self.config.cost_per_evidence.get(item.evidence_type, 1000.0)
                with_reuse += cost
                seen_items.add(item.evidence_id)

        # If no items registered, estimate based on unique types needed
        if not self._items:
            needed_types: Set[str] = set()
            for req_dict in EVIDENCE_REQUIREMENTS.values():
                for required_types in req_dict.values():
                    for etype in required_types:
                        needed_types.add(etype)
            for etype in needed_types:
                with_reuse += self.config.cost_per_evidence.get(etype, 1000.0)

        savings = max(without_reuse - with_reuse, 0.0)
        savings_pct = _safe_div(savings, without_reuse) * 100.0

        return ReuseSavings(
            total_evidence_items=len(self._items),
            reused_items=len(reusable),
            unique_requirements_satisfied=len(all_req_ids),
            without_reuse_cost_eur=round(without_reuse, 2),
            with_reuse_cost_eur=round(with_reuse, 2),
            savings_eur=round(savings, 2),
            savings_pct=round(savings_pct, 2),
        )

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of a string (for file hashing).

        Args:
            content: The content string to hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_coverage_matrix(self) -> List[CoverageMatrix]:
        """Build per-regulation coverage matrix."""
        matrices: List[CoverageMatrix] = []

        for reg, req_dict in EVIDENCE_REQUIREMENTS.items():
            total = len(req_dict)
            covered = 0
            req_details: Dict[str, float] = {}

            for req_id, required_types in req_dict.items():
                # Find mappings for this requirement
                req_mappings = [
                    m for m in self._mappings
                    if m.regulation == reg and m.requirement_id == req_id
                ]

                # Get provided evidence types from active items
                provided_types: Set[str] = set()
                for m in req_mappings:
                    for item in self._items:
                        if item.evidence_id == m.evidence_id and item.status == EvidenceStatus.ACTIVE.value:
                            provided_types.add(item.evidence_type)

                # Coverage for this requirement
                type_coverage = _safe_div(
                    len(provided_types.intersection(set(required_types))),
                    len(required_types),
                ) * 100.0
                req_details[req_id] = round(type_coverage, 2)

                if type_coverage >= 100.0:
                    covered += 1

            coverage_pct = _safe_div(covered, total) * 100.0

            matrices.append(CoverageMatrix(
                regulation=reg,
                total_requirements=total,
                covered_requirements=covered,
                coverage_pct=round(coverage_pct, 2),
                requirement_details=req_details,
            ))

        return matrices
