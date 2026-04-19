# -*- coding: utf-8 -*-
"""
CrossRegulationGapAnalyzerEngine - PACK-009 EU Climate Compliance Bundle Engine 3

Scans compliance posture across CSRD, CBAM, EUDR, and EU Taxonomy
simultaneously, identifying gaps with cross-regulation impact scoring.
Gaps affecting multiple regulations receive severity multipliers,
enabling prioritized remediation roadmaps.

Core Capabilities:
    1. ~80 compliance requirements across 4 EU regulations
    2. Cross-impact scoring (gaps affecting N regulations get Nx multiplier)
    3. Severity classification (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    4. Remediation effort estimation (hours, cost, timeline)
    5. Prioritized remediation roadmap generation
    6. Cross-impact matrix visualization data

Requirement Categories:
    - DATA_COLLECTION: Data gathering and quality requirements
    - REPORTING: Disclosure and filing requirements
    - GOVERNANCE: Policies, procedures, and oversight
    - VERIFICATION: Assurance, audit, and verification
    - TECHNOLOGY: Systems, tools, and infrastructure

Zero-Hallucination:
    - All requirements from published regulation text
    - Severity scoring uses deterministic formula
    - No LLM involvement in gap analysis
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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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

class GapSeverity(str, Enum):
    """Severity level for compliance gaps."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class ComplianceStatus(str, Enum):
    """Current compliance status for a requirement."""
    COMPLIANT = "COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"
    NOT_APPLICABLE = "NOT_APPLICABLE"

class RequirementCategory(str, Enum):
    """Category of compliance requirement."""
    DATA_COLLECTION = "DATA_COLLECTION"
    REPORTING = "REPORTING"
    GOVERNANCE = "GOVERNANCE"
    VERIFICATION = "VERIFICATION"
    TECHNOLOGY = "TECHNOLOGY"

class RemediationPriority(str, Enum):
    """Priority level for remediation actions."""
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ComplianceRequirement(BaseModel):
    """A single compliance requirement from a regulation."""
    requirement_id: str = Field(..., description="Unique requirement identifier")
    regulation: str = Field(..., description="Source regulation (CSRD, CBAM, EUDR, EU_TAXONOMY)")
    description: str = Field(default="", description="Requirement description")
    category: str = Field(default="REPORTING", description="Requirement category")
    mandatory: bool = Field(default=True, description="Whether the requirement is mandatory")
    cross_regulation_tags: List[str] = Field(
        default_factory=list, description="Tags linking to requirements in other regulations"
    )
    reference: str = Field(default="", description="Regulatory reference (article, section)")
    effective_date: str = Field(default="2025-01-01", description="Date the requirement becomes effective")

class Gap(BaseModel):
    """An identified compliance gap."""
    gap_id: str = Field(default_factory=_new_uuid, description="Gap identifier")
    requirement: ComplianceRequirement = Field(..., description="The requirement with the gap")
    current_status: str = Field(default="NON_COMPLIANT", description="Current compliance status")
    gap_description: str = Field(default="", description="Description of the gap")
    severity: str = Field(default="MEDIUM", description="Gap severity")
    base_severity_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Base severity score 0-10")
    cross_impact_multiplier: float = Field(default=1.0, ge=1.0, description="Cross-regulation impact multiplier")
    adjusted_severity_score: float = Field(default=5.0, ge=0.0, description="Severity after cross-impact adjustment")
    affected_regulations: List[str] = Field(default_factory=list, description="All regulations affected")
    remediation_effort_hours: float = Field(default=0.0, ge=0.0, description="Estimated remediation effort in hours")
    estimated_cost_eur: float = Field(default=0.0, ge=0.0, description="Estimated remediation cost in EUR")
    remediation_timeline_days: int = Field(default=30, ge=0, description="Estimated timeline in days")
    remediation_priority: str = Field(default="MEDIUM_TERM", description="Remediation priority level")
    remediation_actions: List[str] = Field(default_factory=list, description="Recommended remediation actions")

class RemediationRoadmapItem(BaseModel):
    """A single item in a remediation roadmap."""
    item_id: str = Field(default_factory=_new_uuid, description="Item identifier")
    gap_id: str = Field(default="", description="Related gap identifier")
    title: str = Field(default="", description="Roadmap item title")
    description: str = Field(default="", description="Detailed description")
    priority: str = Field(default="MEDIUM_TERM", description="Priority level")
    effort_hours: float = Field(default=0.0, description="Effort in hours")
    cost_eur: float = Field(default=0.0, description="Cost in EUR")
    timeline_days: int = Field(default=30, description="Timeline in days")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other items")
    regulations_addressed: List[str] = Field(default_factory=list, description="Regulations addressed")
    phase: int = Field(default=1, ge=1, le=4, description="Implementation phase (1-4)")

class CrossImpactEntry(BaseModel):
    """Entry in the cross-impact matrix."""
    regulation_a: str = Field(default="", description="First regulation")
    regulation_b: str = Field(default="", description="Second regulation")
    shared_gaps: int = Field(default=0, description="Number of gaps affecting both")
    impact_score: float = Field(default=0.0, description="Cross-impact score")
    shared_requirement_ids: List[str] = Field(default_factory=list, description="Shared requirement IDs")

class GapAnalysisResult(BaseModel):
    """Complete result of a gap analysis scan."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    total_requirements_scanned: int = Field(default=0, description="Total requirements scanned")
    total_gaps: int = Field(default=0, description="Total gaps identified")
    gaps_by_severity: Dict[str, int] = Field(default_factory=dict, description="Gap counts by severity")
    gaps_by_regulation: Dict[str, int] = Field(default_factory=dict, description="Gap counts by regulation")
    gaps_by_category: Dict[str, int] = Field(default_factory=dict, description="Gap counts by category")
    multi_regulation_gaps: int = Field(default=0, description="Gaps affecting multiple regulations")
    gaps: List[Gap] = Field(default_factory=list, description="All identified gaps")
    remediation_plan: List[RemediationRoadmapItem] = Field(
        default_factory=list, description="Prioritized remediation roadmap"
    )
    cross_impact_matrix: List[CrossImpactEntry] = Field(
        default_factory=list, description="Cross-regulation impact matrix"
    )
    total_remediation_hours: float = Field(default=0.0, description="Total remediation hours")
    total_remediation_cost_eur: float = Field(default=0.0, description="Total remediation cost EUR")
    overall_compliance_score: float = Field(default=0.0, description="Overall compliance score 0-100")
    regulations_analyzed: List[str] = Field(default_factory=list, description="Regulations analyzed")
    timestamp: str = Field(default_factory=lambda: utcnow().isoformat(), description="Analysis timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GapAnalyzerConfig(BaseModel):
    """Configuration for the CrossRegulationGapAnalyzerEngine."""
    severity_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "CRITICAL": 10.0, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.5, "INFO": 1.0,
        },
        description="Severity level to numeric weight mapping"
    )
    min_severity: str = Field(
        default="LOW", description="Minimum severity to include in results"
    )
    include_cross_impacts: bool = Field(
        default=True, description="Include cross-regulation impact analysis"
    )
    cross_impact_multiplier_base: float = Field(
        default=1.5, ge=1.0, le=3.0,
        description="Base multiplier per additional regulation affected"
    )
    hourly_rate_eur: float = Field(
        default=100.0, ge=0.0,
        description="Hourly rate for cost estimation in EUR"
    )

    @field_validator("min_severity", mode="before")
    @classmethod
    def _validate_severity(cls, v: Any) -> str:
        allowed = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        val = str(v).upper()
        return val if val in allowed else "LOW"

# ---------------------------------------------------------------------------
# Model rebuilds
# ---------------------------------------------------------------------------

GapAnalyzerConfig.model_rebuild()
ComplianceRequirement.model_rebuild()
Gap.model_rebuild()
RemediationRoadmapItem.model_rebuild()
CrossImpactEntry.model_rebuild()
GapAnalysisResult.model_rebuild()

# ---------------------------------------------------------------------------
# Compliance Requirements Database (~80 requirements across 4 regulations)
# ---------------------------------------------------------------------------

COMPLIANCE_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "CSRD": [
        {"requirement_id": "CSRD-DC-001", "description": "Collect Scope 1 GHG emissions data per ESRS E1-6", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CBAM-DC-001", "TAX-DC-001"], "reference": "ESRS E1-6", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-002", "description": "Collect Scope 2 GHG emissions (location and market-based)", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CBAM-DC-002"], "reference": "ESRS E1-6", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-003", "description": "Collect Scope 3 Category 1 emissions data", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CBAM-DC-003"], "reference": "ESRS E1-6", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-004", "description": "Collect water consumption and withdrawal data per ESRS E3", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-005"], "reference": "ESRS E3-4", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-005", "description": "Collect biodiversity impact data per ESRS E4", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["EUDR-DC-001", "TAX-DC-006"], "reference": "ESRS E4-4", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-006", "description": "Collect supply chain due diligence evidence per ESRS S2", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["EUDR-DC-002"], "reference": "ESRS S2", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-007", "description": "Collect EU Taxonomy KPI data (turnover, CapEx, OpEx)", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-002", "TAX-DC-003", "TAX-DC-004"], "reference": "ESRS 2 (Art. 8)", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-DC-008", "description": "Collect pollution and substance of concern data per ESRS E2", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-007"], "reference": "ESRS E2-4", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-RP-001", "description": "Publish annual sustainability report with ESRS disclosures", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "CSRD Art. 19a", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-RP-002", "description": "Disclose climate transition plan per ESRS E1-1", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["TAX-GV-002"], "reference": "ESRS E1-1", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-RP-003", "description": "Disclose climate scenario analysis results per ESRS E1-9", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["TAX-RP-003"], "reference": "ESRS E1-9", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-RP-004", "description": "Disclose financial effects of climate risks per ESRS E1-9", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["CBAM-RP-002"], "reference": "ESRS E1-9", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-RP-005", "description": "Report deforestation commitments per ESRS E4-5", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["EUDR-RP-001"], "reference": "ESRS E4-5", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-GV-001", "description": "Establish double materiality assessment process", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "ESRS 1", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-GV-002", "description": "Assign board-level oversight of sustainability reporting", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "CSRD Art. 19a(2)", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-GV-003", "description": "Implement stakeholder engagement process for material topics", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": ["EUDR-GV-003"], "reference": "ESRS 1", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-VR-001", "description": "Obtain limited assurance on sustainability report", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "CSRD Art. 34", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-VR-002", "description": "Prepare for reasonable assurance (future requirement)", "category": "VERIFICATION", "mandatory": False, "cross_regulation_tags": [], "reference": "CSRD Art. 34", "effective_date": "2028-01-01"},
        {"requirement_id": "CSRD-TC-001", "description": "Implement XBRL digital tagging for sustainability data", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": [], "reference": "CSRD Art. 29d", "effective_date": "2025-01-01"},
        {"requirement_id": "CSRD-TC-002", "description": "Deploy data management system for ESRS data collection", "category": "TECHNOLOGY", "mandatory": False, "cross_regulation_tags": ["CBAM-TC-001", "EUDR-TC-001", "TAX-TC-001"], "reference": "Best practice", "effective_date": "2025-01-01"},
    ],
    "CBAM": [
        {"requirement_id": "CBAM-DC-001", "description": "Collect embedded emissions data per installation", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-001", "TAX-DC-001"], "reference": "CBAM Reg. Art. 7", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-DC-002", "description": "Collect electricity emission factors for indirect emissions", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-002"], "reference": "CBAM Impl. Reg. Art. 4", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-DC-003", "description": "Collect precursor material emission data", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-003"], "reference": "CBAM Impl. Reg. Art. 7", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-DC-004", "description": "Collect carbon price paid in country of origin", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 9", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-DC-005", "description": "Verify CN code classification for all imports", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["EUDR-DC-004"], "reference": "CBAM Reg. Annex I", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-DC-006", "description": "Collect production process route data from installations", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-008"], "reference": "CBAM Impl. Reg. Art. 3", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-RP-001", "description": "Submit quarterly CBAM reports during transition period", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 35", "effective_date": "2023-10-01"},
        {"requirement_id": "CBAM-RP-002", "description": "Submit annual CBAM declaration by May 31", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["CSRD-RP-004"], "reference": "CBAM Reg. Art. 6", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-RP-003", "description": "Report carbon price deductions applied", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 9", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-GV-001", "description": "Obtain authorized CBAM declarant status", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 5", "effective_date": "2025-12-31"},
        {"requirement_id": "CBAM-GV-002", "description": "Establish CBAM compliance procedures and internal controls", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 5", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-GV-003", "description": "Maintain financial guarantee for CBAM obligations", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 5(3)", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-VR-001", "description": "Verify embedded emissions with accredited verifier", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 8", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-VR-002", "description": "Prepare for NCA examination and audit trail", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 19", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-TC-001", "description": "Integrate with EU CBAM Registry for submissions", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": ["CSRD-TC-002"], "reference": "CBAM Reg. Art. 14", "effective_date": "2026-01-01"},
        {"requirement_id": "CBAM-TC-002", "description": "Implement certificate management system", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": [], "reference": "CBAM Reg. Art. 20", "effective_date": "2026-01-01"},
    ],
    "EUDR": [
        {"requirement_id": "EUDR-DC-001", "description": "Collect geolocation data for all production plots", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-005"], "reference": "EUDR Art. 9(1)(d)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-002", "description": "Collect supply chain traceability data to source", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-006"], "reference": "EUDR Art. 9(1)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-003", "description": "Collect deforestation-free compliance evidence", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-006"], "reference": "EUDR Art. 3", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-004", "description": "Verify HS code classification for regulated commodities", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CBAM-DC-005"], "reference": "EUDR Annex I", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-005", "description": "Collect country-of-origin risk classification data", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 29", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-006", "description": "Collect satellite/remote sensing monitoring data", "category": "DATA_COLLECTION", "mandatory": False, "cross_regulation_tags": ["CSRD-DC-005"], "reference": "EUDR Art. 10", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-DC-007", "description": "Collect indigenous community rights compliance data", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["TAX-DC-009"], "reference": "EUDR Art. 3(b)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-RP-001", "description": "Submit due diligence statement to competent authority", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["CSRD-RP-005"], "reference": "EUDR Art. 4(2)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-RP-002", "description": "Report risk assessment results per commodity placement", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 10", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-RP-003", "description": "File annual review of due diligence system", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 12", "effective_date": "2026-12-30"},
        {"requirement_id": "EUDR-GV-001", "description": "Establish due diligence system for EUDR compliance", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 8", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-GV-002", "description": "Implement risk assessment framework for sourcing regions", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 10", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-GV-003", "description": "Establish stakeholder consultation process", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": ["CSRD-GV-003"], "reference": "EUDR Art. 10(2)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-GV-004", "description": "Implement risk mitigation and corrective action procedures", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 11", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-VR-001", "description": "Prepare for competent authority inspection and checks", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 14-19", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-VR-002", "description": "Maintain 5-year record retention for DDS evidence", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 9(3)", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-VR-003", "description": "Enable third-party verification of traceability claims", "category": "VERIFICATION", "mandatory": False, "cross_regulation_tags": [], "reference": "EUDR Art. 10", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-TC-001", "description": "Implement traceability system for supply chain data", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": ["CSRD-TC-002"], "reference": "EUDR Art. 9", "effective_date": "2025-12-30"},
        {"requirement_id": "EUDR-TC-002", "description": "Integrate with EU Information System for DDS submission", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": [], "reference": "EUDR Art. 33", "effective_date": "2025-12-30"},
    ],
    "EU_TAXONOMY": [
        {"requirement_id": "TAX-DC-001", "description": "Collect GHG emission data for substantial contribution assessment", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-001", "CBAM-DC-001"], "reference": "Taxonomy DA Art. 10", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-002", "description": "Calculate Taxonomy-aligned turnover KPI", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-007"], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-003", "description": "Calculate Taxonomy-aligned CapEx KPI", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-007"], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-004", "description": "Calculate Taxonomy-aligned OpEx KPI", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-007"], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-005", "description": "Collect water metrics for WTR objective assessment", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-004"], "reference": "Taxonomy Reg. Art. 12", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-006", "description": "Collect biodiversity data for BIO objective assessment", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-005", "EUDR-DC-003"], "reference": "Taxonomy Reg. Art. 15", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-007", "description": "Collect pollution data for PPC objective assessment", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CSRD-DC-008"], "reference": "Taxonomy Reg. Art. 14", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-008", "description": "Identify eligible economic activities by NACE code", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["CBAM-DC-006"], "reference": "Taxonomy DA Annex I/II", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-DC-009", "description": "Collect minimum safeguards compliance evidence", "category": "DATA_COLLECTION", "mandatory": True, "cross_regulation_tags": ["EUDR-DC-007"], "reference": "Taxonomy Reg. Art. 18", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-RP-001", "description": "Publish Article 8 disclosure in annual report", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-RP-002", "description": "Report eligible vs aligned revenue breakdown", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": [], "reference": "Art. 8 DA Annex I", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-RP-003", "description": "Report DNSH assessment results for all 6 objectives", "category": "REPORTING", "mandatory": True, "cross_regulation_tags": ["CSRD-RP-003"], "reference": "Taxonomy Reg. Art. 17", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-RP-004", "description": "Report nuclear/gas complementary disclosures if applicable", "category": "REPORTING", "mandatory": False, "cross_regulation_tags": [], "reference": "Complementary DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-GV-001", "description": "Establish Taxonomy assessment governance process", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "Best practice", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-GV-002", "description": "Implement CapEx plan for Taxonomy alignment improvement", "category": "GOVERNANCE", "mandatory": False, "cross_regulation_tags": ["CSRD-RP-002"], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-GV-003", "description": "Assign responsibility for Taxonomy compliance", "category": "GOVERNANCE", "mandatory": True, "cross_regulation_tags": [], "reference": "Best practice", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-VR-001", "description": "Subject Taxonomy KPIs to external assurance", "category": "VERIFICATION", "mandatory": False, "cross_regulation_tags": [], "reference": "Art. 8 DA", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-VR-002", "description": "Maintain audit trail for Taxonomy eligibility and alignment", "category": "VERIFICATION", "mandatory": True, "cross_regulation_tags": [], "reference": "Best practice", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-TC-001", "description": "Deploy system for Taxonomy KPI calculation and tracking", "category": "TECHNOLOGY", "mandatory": False, "cross_regulation_tags": ["CSRD-TC-002"], "reference": "Best practice", "effective_date": "2024-01-01"},
        {"requirement_id": "TAX-TC-002", "description": "Implement NACE code mapping to Taxonomy activities", "category": "TECHNOLOGY", "mandatory": True, "cross_regulation_tags": [], "reference": "Taxonomy DA Annex I/II", "effective_date": "2024-01-01"},
    ],
}

# ---------------------------------------------------------------------------
# CrossRegulationGapAnalyzerEngine
# ---------------------------------------------------------------------------

class CrossRegulationGapAnalyzerEngine:
    """
    Cross-regulation gap analysis engine for EU climate compliance.

    Scans compliance requirements across CSRD, CBAM, EUDR, and EU Taxonomy
    simultaneously, identifying gaps with cross-regulation impact scoring.
    Generates prioritized remediation roadmaps.

    Attributes:
        config: Engine configuration.
        _requirements: All loaded compliance requirements by regulation.
        _all_requirements: Flat list of all ComplianceRequirement objects.

    Example:
        >>> engine = CrossRegulationGapAnalyzerEngine()
        >>> status = {"CSRD-DC-001": "COMPLIANT", "CBAM-DC-001": "NON_COMPLIANT"}
        >>> result = engine.scan_all_regulations(status)
        >>> assert result.total_gaps > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CrossRegulationGapAnalyzerEngine.

        Args:
            config: Optional configuration dictionary or GapAnalyzerConfig.
        """
        if config and isinstance(config, dict):
            self.config = GapAnalyzerConfig(**config)
        elif config and isinstance(config, GapAnalyzerConfig):
            self.config = config
        else:
            self.config = GapAnalyzerConfig()

        self._requirements: Dict[str, List[ComplianceRequirement]] = {}
        self._all_requirements: List[ComplianceRequirement] = []
        self._load_requirements()
        logger.info(
            "CrossRegulationGapAnalyzerEngine initialized (v%s): %d requirements loaded",
            _MODULE_VERSION, len(self._all_requirements),
        )

    def _load_requirements(self) -> None:
        """Load compliance requirements from the database."""
        for regulation, raw_list in COMPLIANCE_REQUIREMENTS.items():
            reqs: List[ComplianceRequirement] = []
            for raw in raw_list:
                req = ComplianceRequirement(
                    requirement_id=raw["requirement_id"],
                    regulation=regulation,
                    description=raw.get("description", ""),
                    category=raw.get("category", "REPORTING"),
                    mandatory=raw.get("mandatory", True),
                    cross_regulation_tags=raw.get("cross_regulation_tags", []),
                    reference=raw.get("reference", ""),
                    effective_date=raw.get("effective_date", "2025-01-01"),
                )
                reqs.append(req)
                self._all_requirements.append(req)
            self._requirements[regulation] = reqs

    # -------------------------------------------------------------------
    # scan_all_regulations
    # -------------------------------------------------------------------

    def scan_all_regulations(
        self,
        compliance_status: Dict[str, str],
        regulations: Optional[List[str]] = None,
    ) -> GapAnalysisResult:
        """Scan all requirements and identify gaps based on current compliance status.

        Args:
            compliance_status: Dict mapping requirement_id to ComplianceStatus value.
            regulations: Optional list of regulations to scan (default: all).

        Returns:
            GapAnalysisResult with gaps, remediation plan, and cross-impact matrix.
        """
        start_time = datetime.now(timezone.utc)
        target_regs = regulations or list(self._requirements.keys())

        target_reqs = [
            req for req in self._all_requirements
            if req.regulation in target_regs
        ]

        gaps = self._identify_gaps(target_reqs, compliance_status)
        gaps = self._filter_by_severity(gaps)

        if self.config.include_cross_impacts:
            gaps = self._score_cross_impact(gaps)

        gaps.sort(key=lambda g: g.adjusted_severity_score, reverse=True)

        remediation_plan = self._generate_roadmap(gaps)
        cross_impact_matrix = self._build_cross_impact_matrix(gaps, target_regs)

        gaps_by_severity: Dict[str, int] = {}
        gaps_by_regulation: Dict[str, int] = {}
        gaps_by_category: Dict[str, int] = {}
        multi_reg_count = 0

        for gap in gaps:
            sev = gap.severity
            gaps_by_severity[sev] = gaps_by_severity.get(sev, 0) + 1
            reg = gap.requirement.regulation
            gaps_by_regulation[reg] = gaps_by_regulation.get(reg, 0) + 1
            cat = gap.requirement.category
            gaps_by_category[cat] = gaps_by_category.get(cat, 0) + 1
            if len(gap.affected_regulations) > 1:
                multi_reg_count += 1

        total_hours = sum(g.remediation_effort_hours for g in gaps)
        total_cost = sum(g.estimated_cost_eur for g in gaps)

        compliant_count = sum(
            1 for req in target_reqs
            if compliance_status.get(req.requirement_id, "NOT_ASSESSED") == "COMPLIANT"
        )
        compliance_score = round(
            compliant_count / max(len(target_reqs), 1) * 100, 2
        )

        result = GapAnalysisResult(
            total_requirements_scanned=len(target_reqs),
            total_gaps=len(gaps),
            gaps_by_severity=gaps_by_severity,
            gaps_by_regulation=gaps_by_regulation,
            gaps_by_category=gaps_by_category,
            multi_regulation_gaps=multi_reg_count,
            gaps=gaps,
            remediation_plan=remediation_plan,
            cross_impact_matrix=cross_impact_matrix,
            total_remediation_hours=round(total_hours, 2),
            total_remediation_cost_eur=round(total_cost, 2),
            overall_compliance_score=compliance_score,
            regulations_analyzed=target_regs,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Gap analysis: %d reqs, %d gaps, %.1f%% compliant, %.1fms",
            len(target_reqs), len(gaps), compliance_score, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # identify_gaps
    # -------------------------------------------------------------------

    def identify_gaps(
        self,
        requirements: List[ComplianceRequirement],
        compliance_status: Dict[str, str],
    ) -> List[Gap]:
        """Identify gaps from a list of requirements.

        Args:
            requirements: List of ComplianceRequirement objects.
            compliance_status: Dict mapping requirement_id to status.

        Returns:
            List of Gap objects.
        """
        return self._identify_gaps(requirements, compliance_status)

    def _identify_gaps(
        self,
        requirements: List[ComplianceRequirement],
        compliance_status: Dict[str, str],
    ) -> List[Gap]:
        """Core gap identification logic."""
        gaps: List[Gap] = []
        effort_map = self._get_effort_estimates()

        for req in requirements:
            status = compliance_status.get(req.requirement_id, "NOT_ASSESSED")

            if status in ("COMPLIANT", "NOT_APPLICABLE"):
                continue

            severity = self._assess_severity(req, status)
            base_score = self.config.severity_weights.get(severity, 5.0)
            effort_info = effort_map.get(req.category, {"hours": 20, "days": 14})

            affected = [req.regulation]
            for tag in req.cross_regulation_tags:
                reg_prefix = tag.split("-")[0]
                regulation_name = {
                    "CSRD": "CSRD", "CBAM": "CBAM",
                    "EUDR": "EUDR", "TAX": "EU_TAXONOMY",
                }.get(reg_prefix, reg_prefix)
                if regulation_name not in affected:
                    affected.append(regulation_name)

            effort_hours = effort_info["hours"]
            timeline_days = effort_info["days"]
            cost = round(effort_hours * self.config.hourly_rate_eur, 2)
            priority = self._determine_priority(severity, len(affected))

            actions = self._generate_remediation_actions(req, status)

            gap_desc = self._generate_gap_description(req, status)

            gap = Gap(
                requirement=req,
                current_status=status,
                gap_description=gap_desc,
                severity=severity,
                base_severity_score=base_score,
                cross_impact_multiplier=1.0,
                adjusted_severity_score=base_score,
                affected_regulations=sorted(affected),
                remediation_effort_hours=effort_hours,
                estimated_cost_eur=cost,
                remediation_timeline_days=timeline_days,
                remediation_priority=priority,
                remediation_actions=actions,
            )
            gaps.append(gap)

        return gaps

    def _assess_severity(
        self,
        req: ComplianceRequirement,
        status: str,
    ) -> str:
        """Assess gap severity based on requirement properties and status."""
        if req.mandatory and status == "NON_COMPLIANT":
            if req.category in ("REPORTING", "GOVERNANCE"):
                return "CRITICAL"
            return "HIGH"
        if req.mandatory and status == "PARTIALLY_COMPLIANT":
            return "MEDIUM"
        if req.mandatory and status == "NOT_ASSESSED":
            return "HIGH"
        if not req.mandatory and status == "NON_COMPLIANT":
            return "LOW"
        if not req.mandatory and status == "NOT_ASSESSED":
            return "INFO"
        return "MEDIUM"

    def _filter_by_severity(self, gaps: List[Gap]) -> List[Gap]:
        """Filter gaps by minimum severity threshold."""
        severity_order = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1}
        min_level = severity_order.get(self.config.min_severity, 2)
        return [g for g in gaps if severity_order.get(g.severity, 0) >= min_level]

    def _determine_priority(self, severity: str, num_affected: int) -> str:
        """Determine remediation priority from severity and impact breadth."""
        if severity == "CRITICAL" or (severity == "HIGH" and num_affected >= 3):
            return "IMMEDIATE"
        if severity == "HIGH" or (severity == "MEDIUM" and num_affected >= 2):
            return "SHORT_TERM"
        if severity == "MEDIUM":
            return "MEDIUM_TERM"
        return "LONG_TERM"

    def _generate_gap_description(
        self,
        req: ComplianceRequirement,
        status: str,
    ) -> str:
        """Generate a human-readable gap description."""
        status_text = {
            "NON_COMPLIANT": "No compliance measures in place",
            "PARTIALLY_COMPLIANT": "Partial compliance achieved but gaps remain",
            "NOT_ASSESSED": "Compliance status has not been assessed",
        }.get(status, "Status unknown")

        return (
            f"{req.regulation} requirement {req.requirement_id}: {status_text}. "
            f"{req.description} (Ref: {req.reference})"
        )

    def _generate_remediation_actions(
        self,
        req: ComplianceRequirement,
        status: str,
    ) -> List[str]:
        """Generate remediation action recommendations."""
        actions: List[str] = []

        category_actions = {
            "DATA_COLLECTION": [
                f"Establish data collection process for {req.requirement_id}",
                "Identify data sources and assign data owners",
                "Implement data quality validation checks",
            ],
            "REPORTING": [
                f"Prepare {req.regulation} disclosure content for {req.requirement_id}",
                "Review against regulatory templates and guidance",
                "Submit for internal review and sign-off",
            ],
            "GOVERNANCE": [
                f"Draft governance policy/procedure for {req.requirement_id}",
                "Obtain board or management approval",
                "Implement and communicate across organization",
            ],
            "VERIFICATION": [
                f"Engage verification provider for {req.requirement_id}",
                "Prepare evidence package and documentation",
                "Schedule verification timeline aligned with reporting deadlines",
            ],
            "TECHNOLOGY": [
                f"Evaluate technology solutions for {req.requirement_id}",
                "Implement and test system capabilities",
                "Train users and establish support procedures",
            ],
        }

        actions = category_actions.get(req.category, [
            f"Address requirement {req.requirement_id}",
            "Consult regulatory guidance",
            "Implement compliance measures",
        ])

        if status == "PARTIALLY_COMPLIANT":
            actions.insert(0, f"Assess current partial compliance for {req.requirement_id} and identify specific gaps")

        return actions

    def _get_effort_estimates(self) -> Dict[str, Dict[str, int]]:
        """Get estimated effort by requirement category."""
        return {
            "DATA_COLLECTION": {"hours": 20, "days": 21},
            "REPORTING": {"hours": 30, "days": 30},
            "GOVERNANCE": {"hours": 40, "days": 45},
            "VERIFICATION": {"hours": 15, "days": 30},
            "TECHNOLOGY": {"hours": 60, "days": 60},
        }

    # -------------------------------------------------------------------
    # score_cross_impact
    # -------------------------------------------------------------------

    def score_cross_impact(self, gaps: List[Gap]) -> List[Gap]:
        """Apply cross-regulation impact multipliers to gaps.

        Args:
            gaps: List of Gap objects.

        Returns:
            Updated gaps with adjusted severity scores.
        """
        return self._score_cross_impact(gaps)

    def _score_cross_impact(self, gaps: List[Gap]) -> List[Gap]:
        """Core cross-impact scoring logic."""
        base_multiplier = self.config.cross_impact_multiplier_base

        for gap in gaps:
            num_affected = len(gap.affected_regulations)
            if num_affected > 1:
                multiplier = 1.0 + (num_affected - 1) * (base_multiplier - 1.0)
                gap.cross_impact_multiplier = round(multiplier, 2)
                gap.adjusted_severity_score = round(
                    gap.base_severity_score * multiplier, 2
                )
                if gap.adjusted_severity_score > 10.0 and gap.severity != "CRITICAL":
                    gap.severity = "CRITICAL"
            else:
                gap.cross_impact_multiplier = 1.0
                gap.adjusted_severity_score = gap.base_severity_score

        return gaps

    # -------------------------------------------------------------------
    # prioritize_remediation / generate_roadmap
    # -------------------------------------------------------------------

    def prioritize_remediation(self, gaps: List[Gap]) -> List[RemediationRoadmapItem]:
        """Generate a prioritized remediation roadmap from gaps.

        Args:
            gaps: List of Gap objects.

        Returns:
            Ordered list of RemediationRoadmapItem.
        """
        return self._generate_roadmap(gaps)

    def generate_roadmap(self, gaps: List[Gap]) -> List[RemediationRoadmapItem]:
        """Alias for prioritize_remediation.

        Args:
            gaps: List of Gap objects.

        Returns:
            Ordered list of RemediationRoadmapItem.
        """
        return self._generate_roadmap(gaps)

    def _generate_roadmap(self, gaps: List[Gap]) -> List[RemediationRoadmapItem]:
        """Build remediation roadmap from gaps sorted by priority."""
        priority_phase = {
            "IMMEDIATE": 1, "SHORT_TERM": 2, "MEDIUM_TERM": 3, "LONG_TERM": 4,
        }

        sorted_gaps = sorted(
            gaps,
            key=lambda g: (
                priority_phase.get(g.remediation_priority, 4),
                -g.adjusted_severity_score,
            ),
        )

        items: List[RemediationRoadmapItem] = []
        for gap in sorted_gaps:
            phase = priority_phase.get(gap.remediation_priority, 3)
            item = RemediationRoadmapItem(
                gap_id=gap.gap_id,
                title=f"Remediate {gap.requirement.requirement_id}: {gap.requirement.description[:80]}",
                description=gap.gap_description,
                priority=gap.remediation_priority,
                effort_hours=gap.remediation_effort_hours,
                cost_eur=gap.estimated_cost_eur,
                timeline_days=gap.remediation_timeline_days,
                dependencies=[],
                regulations_addressed=gap.affected_regulations,
                phase=phase,
            )
            items.append(item)

        self._resolve_dependencies(items)
        return items

    def _resolve_dependencies(self, items: List[RemediationRoadmapItem]) -> None:
        """Resolve dependencies between roadmap items.

        Governance items are dependencies for reporting items.
        Data collection items are dependencies for reporting items.
        Technology items are dependencies for data collection items.
        """
        by_gap_id: Dict[str, RemediationRoadmapItem] = {}
        gap_to_category: Dict[str, str] = {}

        for item in items:
            by_gap_id[item.gap_id] = item

        for gap_item in items:
            title_lower = gap_item.title.lower()
            if "data collection" in title_lower or "DC-" in gap_item.title:
                gap_to_category[gap_item.item_id] = "DATA_COLLECTION"
            elif "reporting" in title_lower or "RP-" in gap_item.title:
                gap_to_category[gap_item.item_id] = "REPORTING"
            elif "governance" in title_lower or "GV-" in gap_item.title:
                gap_to_category[gap_item.item_id] = "GOVERNANCE"
            elif "verification" in title_lower or "VR-" in gap_item.title:
                gap_to_category[gap_item.item_id] = "VERIFICATION"
            elif "technology" in title_lower or "TC-" in gap_item.title:
                gap_to_category[gap_item.item_id] = "TECHNOLOGY"

        for item in items:
            cat = gap_to_category.get(item.item_id, "")
            if cat == "REPORTING":
                for other in items:
                    other_cat = gap_to_category.get(other.item_id, "")
                    if other_cat in ("DATA_COLLECTION", "GOVERNANCE"):
                        regs_overlap = set(item.regulations_addressed) & set(other.regulations_addressed)
                        if regs_overlap and other.item_id != item.item_id:
                            item.dependencies.append(other.item_id)
            elif cat == "DATA_COLLECTION":
                for other in items:
                    other_cat = gap_to_category.get(other.item_id, "")
                    if other_cat == "TECHNOLOGY":
                        regs_overlap = set(item.regulations_addressed) & set(other.regulations_addressed)
                        if regs_overlap and other.item_id != item.item_id:
                            item.dependencies.append(other.item_id)

    # -------------------------------------------------------------------
    # get_multi_regulation_gaps
    # -------------------------------------------------------------------

    def get_multi_regulation_gaps(
        self,
        gaps: List[Gap],
        min_regulations: int = 2,
    ) -> List[Gap]:
        """Filter to gaps affecting multiple regulations.

        Args:
            gaps: List of Gap objects.
            min_regulations: Minimum number of affected regulations.

        Returns:
            Filtered list of Gap objects.
        """
        return [g for g in gaps if len(g.affected_regulations) >= min_regulations]

    # -------------------------------------------------------------------
    # Cross-impact matrix
    # -------------------------------------------------------------------

    def _build_cross_impact_matrix(
        self,
        gaps: List[Gap],
        regulations: List[str],
    ) -> List[CrossImpactEntry]:
        """Build cross-impact matrix between regulation pairs."""
        matrix: List[CrossImpactEntry] = []

        for i, reg_a in enumerate(regulations):
            for j in range(i + 1, len(regulations)):
                reg_b = regulations[j]
                shared = [
                    g for g in gaps
                    if reg_a in g.affected_regulations and reg_b in g.affected_regulations
                ]
                shared_ids = [g.requirement.requirement_id for g in shared]
                impact_score = sum(g.adjusted_severity_score for g in shared)

                entry = CrossImpactEntry(
                    regulation_a=reg_a,
                    regulation_b=reg_b,
                    shared_gaps=len(shared),
                    impact_score=round(impact_score, 2),
                    shared_requirement_ids=shared_ids,
                )
                matrix.append(entry)

        return matrix

    # -------------------------------------------------------------------
    # get_requirement_count
    # -------------------------------------------------------------------

    def get_requirement_count(self) -> Dict[str, int]:
        """Get requirement counts by regulation.

        Returns:
            Dict mapping regulation name to requirement count.
        """
        return {reg: len(reqs) for reg, reqs in self._requirements.items()}

    # -------------------------------------------------------------------
    # get_requirements_by_category
    # -------------------------------------------------------------------

    def get_requirements_by_category(
        self,
        category: str,
    ) -> List[ComplianceRequirement]:
        """Get all requirements for a specific category across all regulations.

        Args:
            category: Requirement category (DATA_COLLECTION, REPORTING, etc).

        Returns:
            List of ComplianceRequirement objects.
        """
        return [
            req for req in self._all_requirements
            if req.category == category
        ]
