# -*- coding: utf-8 -*-
"""
EEDComplianceBridge - EU Energy Efficiency Directive Compliance for EnMS
=========================================================================

This module provides integration with the EU Energy Efficiency Directive
(2023/1791 - recast EED) requirements. Organizations certified to
ISO 50001 are exempt from Article 8 mandatory energy audits, and this
bridge tracks and documents that exemption status.

EED Article Mapping:
    - Article 8: Energy audit exemption for ISO 50001 certified orgs
    - Article 9: Metering requirements cross-referenced with EnMS metering
    - Article 11: Energy management system requirements alignment
    - Article 21: Energy performance contracting support

Key Features:
    - Check EED Article 8 energy audit exemption status
    - Map ISO 50001 clauses to EED requirements
    - Generate exemption evidence documentation
    - Track EED compliance deadlines and transposition dates
    - SHA-256 provenance on all compliance assessments

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EEDArticle(str, Enum):
    """EU Energy Efficiency Directive articles relevant to ISO 50001."""

    ARTICLE_8 = "article_8"     # Energy audits
    ARTICLE_9 = "article_9"     # Metering
    ARTICLE_11 = "article_11"   # Energy management systems
    ARTICLE_21 = "article_21"   # Energy performance contracting
    ARTICLE_25 = "article_25"   # Heating and cooling assessment
    ARTICLE_26 = "article_26"   # District heating efficiency


class ExemptionStatus(str, Enum):
    """EED exemption status for ISO 50001 certified organizations."""

    EXEMPT = "exempt"
    CONDITIONALLY_EXEMPT = "conditionally_exempt"
    NOT_EXEMPT = "not_exempt"
    EXPIRED = "expired"
    PENDING_REVIEW = "pending_review"


class ComplianceLevel(str, Enum):
    """Compliance level for EED requirements."""

    FULL = "full"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EEDComplianceConfig(BaseModel):
    """Configuration for the EED Compliance Bridge."""

    pack_id: str = Field(default="PACK-034")
    enable_provenance: bool = Field(default=True)
    member_state: str = Field(default="DE", description="EU member state code")
    eed_version: str = Field(default="2023/1791", description="EED directive version")
    organization_type: str = Field(default="non_sme", description="sme or non_sme")
    annual_consumption_kwh: float = Field(
        default=0.0, ge=0.0, description="Annual primary energy consumption"
    )
    employee_count: int = Field(default=0, ge=0)


class EEDComplianceResult(BaseModel):
    """Result of an EED compliance assessment."""

    result_id: str = Field(default_factory=_new_uuid)
    exemption_status: ExemptionStatus = Field(default=ExemptionStatus.NOT_EXEMPT)
    article: EEDArticle = Field(default=EEDArticle.ARTICLE_8)
    requirements_met: List[Dict[str, Any]] = Field(default_factory=list)
    requirements_not_met: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_mapping: Dict[str, Any] = Field(default_factory=dict)
    overall_compliance: ComplianceLevel = Field(default=ComplianceLevel.NON_COMPLIANT)
    conditions: List[str] = Field(default_factory=list)
    next_review_date: Optional[str] = Field(None)
    member_state: str = Field(default="")
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# ISO 50001 to EED Mapping
# ---------------------------------------------------------------------------

ISO50001_TO_EED_MAPPING: Dict[str, Dict[str, Any]] = {
    "4.1": {"eed_article": "article_11", "eed_requirement": "Context of the organization", "eed_section": "11.1"},
    "4.4": {"eed_article": "article_11", "eed_requirement": "Energy management system scope", "eed_section": "11.2"},
    "5.1": {"eed_article": "article_11", "eed_requirement": "Leadership and commitment", "eed_section": "11.3"},
    "5.2": {"eed_article": "article_11", "eed_requirement": "Energy policy", "eed_section": "11.4"},
    "6.3": {"eed_article": "article_8", "eed_requirement": "Energy review (substitute for energy audit)", "eed_section": "8.4"},
    "6.6": {"eed_article": "article_11", "eed_requirement": "Planning for collection of energy data", "eed_section": "11.5"},
    "8.1": {"eed_article": "article_11", "eed_requirement": "Operational planning and control", "eed_section": "11.6"},
    "9.1": {"eed_article": "article_9", "eed_requirement": "Monitoring, measurement, analysis", "eed_section": "9.2"},
    "9.3": {"eed_article": "article_11", "eed_requirement": "Management review", "eed_section": "11.7"},
    "10.2": {"eed_article": "article_11", "eed_requirement": "Continual improvement", "eed_section": "11.8"},
}

# EED Article 8 exemption conditions
ARTICLE_8_EXEMPTION_CONDITIONS: List[str] = [
    "Organization holds valid ISO 50001 certification",
    "EnMS scope covers the organization's energy use",
    "Energy review meets Article 8(4) requirements",
    "EnMS includes systematic energy data collection",
    "Certification is from an accredited body",
    "Certificate is within validity period (3 years)",
]


# ---------------------------------------------------------------------------
# EEDComplianceBridge
# ---------------------------------------------------------------------------


class EEDComplianceBridge:
    """EU Energy Efficiency Directive compliance bridge for EnMS.

    Tracks EED compliance status, manages Article 8 energy audit exemptions
    for ISO 50001 certified organizations, and maps ISO 50001 clauses to
    EED requirements.

    Attributes:
        config: Compliance configuration.
        _assessments: Historical compliance assessments.

    Example:
        >>> bridge = EEDComplianceBridge()
        >>> result = bridge.check_eed_exemption({"certified": True, "scope_complete": True})
        >>> print(f"Exemption: {result.exemption_status.value}")
    """

    def __init__(self, config: Optional[EEDComplianceConfig] = None) -> None:
        """Initialize the EED Compliance Bridge.

        Args:
            config: Compliance configuration. Uses defaults if None.
        """
        self.config = config or EEDComplianceConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._assessments: Dict[str, EEDComplianceResult] = {}
        self.logger.info(
            "EEDComplianceBridge initialized: member_state=%s, eed=%s",
            self.config.member_state, self.config.eed_version,
        )

    def check_eed_exemption(self, enms_data: Dict[str, Any]) -> EEDComplianceResult:
        """Check EED Article 8 energy audit exemption status.

        Assesses whether the organization qualifies for the Article 8
        energy audit exemption based on ISO 50001 certification status.

        Args:
            enms_data: Dict with 'certified', 'scope_complete',
                       'certificate_valid_until', 'accredited_body'.

        Returns:
            EEDComplianceResult with exemption determination.
        """
        start = time.monotonic()

        is_certified = enms_data.get("certified", False)
        scope_complete = enms_data.get("scope_complete", False)
        certificate_valid = enms_data.get("certificate_valid_until", "")
        accredited = enms_data.get("accredited_body", False)

        requirements_met: List[Dict[str, Any]] = []
        requirements_not_met: List[Dict[str, Any]] = []

        # Assess each exemption condition
        conditions_check = [
            ("valid_certification", is_certified, "ISO 50001 certification"),
            ("scope_coverage", scope_complete, "EnMS scope covers energy use"),
            ("accredited_body", accredited, "Accredited certification body"),
        ]

        for cond_id, met, desc in conditions_check:
            entry = {"condition_id": cond_id, "description": desc, "met": met}
            if met:
                requirements_met.append(entry)
            else:
                requirements_not_met.append(entry)

        # Determine exemption status
        all_met = len(requirements_not_met) == 0 and is_certified
        if all_met:
            exemption_status = ExemptionStatus.EXEMPT
            compliance = ComplianceLevel.FULL
        elif is_certified and not scope_complete:
            exemption_status = ExemptionStatus.CONDITIONALLY_EXEMPT
            compliance = ComplianceLevel.PARTIAL
        else:
            exemption_status = ExemptionStatus.NOT_EXEMPT
            compliance = ComplianceLevel.NON_COMPLIANT

        result = EEDComplianceResult(
            exemption_status=exemption_status,
            article=EEDArticle.ARTICLE_8,
            requirements_met=requirements_met,
            requirements_not_met=requirements_not_met,
            overall_compliance=compliance,
            conditions=ARTICLE_8_EXEMPTION_CONDITIONS,
            next_review_date=certificate_valid or "Not set",
            member_state=self.config.member_state,
            message=f"EED Article 8 exemption: {exemption_status.value}",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self._assessments[result.result_id] = result
        self.logger.info(
            "EED exemption check: status=%s, met=%d, not_met=%d",
            exemption_status.value, len(requirements_met), len(requirements_not_met),
        )
        return result

    def map_iso50001_to_eed(
        self, compliance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map ISO 50001 clause compliance to EED requirements.

        Args:
            compliance_data: Dict mapping ISO 50001 clauses to compliance status.

        Returns:
            Dict with EED requirement mapping and compliance gaps.
        """
        start = time.monotonic()

        mapping_results: List[Dict[str, Any]] = []
        for clause, eed_info in ISO50001_TO_EED_MAPPING.items():
            clause_status = compliance_data.get(clause, "not_assessed")
            mapping_results.append({
                "iso50001_clause": clause,
                "eed_article": eed_info["eed_article"],
                "eed_requirement": eed_info["eed_requirement"],
                "eed_section": eed_info["eed_section"],
                "iso50001_status": clause_status,
                "eed_compliant": clause_status in ("conforming", "compliant"),
            })

        compliant_count = sum(1 for m in mapping_results if m["eed_compliant"])

        result = {
            "mapping_id": _new_uuid(),
            "total_mappings": len(mapping_results),
            "compliant": compliant_count,
            "non_compliant": len(mapping_results) - compliant_count,
            "compliance_pct": round(
                compliant_count / len(mapping_results) * 100 if mapping_results else 0, 1
            ),
            "mappings": mapping_results,
            "duration_ms": round((time.monotonic() - start) * 1000, 1),
        }

        if self.config.enable_provenance:
            result["provenance_hash"] = _compute_hash(result)
        return result

    def generate_exemption_evidence(self, enms_id: str) -> Dict[str, Any]:
        """Generate evidence documentation for EED exemption claims.

        Args:
            enms_id: EnMS instance identifier.

        Returns:
            Dict with evidence package for regulatory submission.
        """
        self.logger.info("Generating exemption evidence: enms_id=%s", enms_id)

        evidence = {
            "evidence_id": _new_uuid(),
            "enms_id": enms_id,
            "generated_at": _utcnow().isoformat(),
            "eed_directive": self.config.eed_version,
            "member_state": self.config.member_state,
            "evidence_items": [
                {"type": "certificate", "description": "ISO 50001 certificate copy", "status": "required"},
                {"type": "scope_statement", "description": "EnMS scope and boundaries", "status": "required"},
                {"type": "energy_review", "description": "Clause 6.3 energy review summary", "status": "required"},
                {"type": "enpi_report", "description": "EnPI performance report", "status": "required"},
                {"type": "management_review", "description": "Management review minutes", "status": "recommended"},
                {"type": "audit_report", "description": "Latest internal audit report", "status": "recommended"},
            ],
            "submission_guidelines": {
                "submit_to": "National energy agency",
                "format": "PDF package",
                "deadline": "Annual reporting deadline",
            },
        }

        if self.config.enable_provenance:
            evidence["provenance_hash"] = _compute_hash(evidence)
        return evidence

    def track_eed_deadlines(self) -> List[Dict[str, Any]]:
        """Track EED compliance deadlines and transposition dates.

        Returns:
            List of deadline entries with dates and descriptions.
        """
        deadlines = [
            {"deadline": "2024-10-11", "description": "EED 2023/1791 member state transposition deadline", "article": "all", "status": "passed"},
            {"deadline": "2025-10-11", "description": "First energy audit cycle under recast EED", "article": "article_8", "status": "active"},
            {"deadline": "2026-12-31", "description": "EnMS implementation for large enterprises", "article": "article_11", "status": "upcoming"},
            {"deadline": "2027-01-01", "description": "Enhanced metering requirements effective", "article": "article_9", "status": "upcoming"},
            {"deadline": "2030-01-01", "description": "EU energy efficiency target milestone", "article": "all", "status": "future"},
        ]
        return deadlines
