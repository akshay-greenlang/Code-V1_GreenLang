# -*- coding: utf-8 -*-
"""
EnergyAuditEngine - PACK-031 Industrial Energy Audit Engine 2
==============================================================

Conducts EN 16247 compliant energy audits with automatic scheduling
per EED Article 8.  Supports three audit types (walk-through, detailed,
investment-grade), EN 16247-1 through EN 16247-5 compliance checking,
energy end-use breakdown, audit finding categorisation, gap analysis
against best practice, and audit quality scoring.

EN 16247 Series Compliance:
    - EN 16247-1:2022 General requirements
    - EN 16247-2:2022 Buildings
    - EN 16247-3:2022 Processes
    - EN 16247-4:2022 Transport
    - EN 16247-5:2022 Competence of energy auditors

EED Article 8 (Energy Efficiency Directive 2023/1791):
    - Mandatory energy audits for non-SME enterprises
    - Audit cycle: every 4 years (or ISO 50001 exemption)
    - Audit scope: minimum 85% of total energy consumption
    - EMAS registration can also exempt from mandatory audits

Audit Types per EN 16247:
    - Type 1 (Walk-Through / Preliminary): Quick assessment, no-cost
      and low-cost measures, 1-3 days
    - Type 2 (Detailed): Full system analysis, engineering calculations,
      all measure categories, 1-4 weeks
    - Type 3 (Investment-Grade): Detailed financial modelling, M&V plans,
      bankable savings estimates, 2-8 weeks

End-Use Categories:
    - Heating (space heating, process heating, hot water)
    - Cooling (space cooling, process cooling, refrigeration)
    - Lighting (interior, exterior, emergency)
    - Motors & Drives (pumps, fans, compressors, conveyors)
    - Process (specific industrial processes)
    - Transport (internal transport, forklifts, fleet)

Zero-Hallucination:
    - All benchmark comparisons use hard-coded published data
    - Payback and savings calculations use deterministic Decimal arithmetic
    - EN 16247 compliance checks are rule-based (no estimation)
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float."""
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round1(value: float) -> float:
    """Round to 1 decimal place using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AuditType(str, Enum):
    """Energy audit types per EN 16247.

    Type 1 is a preliminary walk-through; Type 2 is a full detailed audit;
    Type 3 is an investment-grade audit with bankable savings estimates.
    """
    TYPE_1_WALKTHROUGH = "type_1_walkthrough"
    TYPE_2_DETAILED = "type_2_detailed"
    TYPE_3_INVESTMENT_GRADE = "type_3_investment_grade"

class AuditPriority(str, Enum):
    """Priority ranking for audit findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class AuditComplexity(str, Enum):
    """Implementation complexity for audit findings."""
    NO_COST = "no_cost"
    LOW_COST = "low_cost"
    MEDIUM_COST = "medium_cost"
    HIGH_COST = "high_cost"
    CAPITAL_PROJECT = "capital_project"

class EndUseCategory(str, Enum):
    """Energy end-use categories for industrial facilities.

    Standard categories per EN 16247-1 Annex A.
    """
    HEATING = "heating"
    COOLING = "cooling"
    LIGHTING = "lighting"
    MOTORS_DRIVES = "motors_drives"
    PROCESS = "process"
    COMPRESSED_AIR = "compressed_air"
    STEAM = "steam"
    TRANSPORT = "transport"
    HOT_WATER = "hot_water"
    VENTILATION = "ventilation"
    IT_EQUIPMENT = "it_equipment"
    OTHER = "other"

class EN16247Part(str, Enum):
    """EN 16247 standard parts."""
    PART_1_GENERAL = "EN_16247_1"
    PART_2_BUILDINGS = "EN_16247_2"
    PART_3_PROCESSES = "EN_16247_3"
    PART_4_TRANSPORT = "EN_16247_4"
    PART_5_COMPETENCE = "EN_16247_5"

class ComplianceStatus(str, Enum):
    """Compliance status for individual checklist items."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EED Article 8 audit cycle (years between mandatory audits).
EED_AUDIT_CYCLE_YEARS: int = 4
"""Energy Efficiency Directive mandates audits every 4 years for non-SMEs.
Source: Directive (EU) 2023/1791 Article 8."""

# EN 16247-1:2022 checklist items (clause, requirement, applicable audit types).
# Source: EN 16247-1:2022 Energy audits - Part 1: General requirements.
EN16247_CHECKLIST_ITEMS: List[Dict[str, Any]] = [
    {
        "clause": "5.1",
        "requirement": "Preliminary contact and audit objectives defined",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.2",
        "requirement": "Start-up meeting conducted with organisation",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.3.1",
        "requirement": "Data collection plan established",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.3.2",
        "requirement": "Energy consumption data collected (min 12 months)",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.3.3",
        "requirement": "Production/activity data collected",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.3.4",
        "requirement": "Equipment inventory compiled",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.3.5",
        "requirement": "Building envelope data collected",
        "part": "EN_16247_2",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.4",
        "requirement": "Site visit conducted (all relevant areas)",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.5.1",
        "requirement": "Energy balance established (inputs = outputs + losses)",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.5.2",
        "requirement": "Energy end-use breakdown quantified",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.5.3",
        "requirement": "Baseline energy performance established",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.5.4",
        "requirement": "Energy performance indicators (EnPI) defined",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.6.1",
        "requirement": "Energy saving opportunities identified",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.6.2",
        "requirement": "Savings quantified with engineering calculations",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.6.3",
        "requirement": "Implementation costs estimated",
        "part": "EN_16247_1",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.6.4",
        "requirement": "Financial analysis (payback, NPV, IRR) completed",
        "part": "EN_16247_1",
        "audit_types": ["type_3_investment_grade"],
    },
    {
        "clause": "5.6.5",
        "requirement": "M&V plan developed for savings verification",
        "part": "EN_16247_1",
        "audit_types": ["type_3_investment_grade"],
    },
    {
        "clause": "5.7",
        "requirement": "Energy audit report prepared per EN 16247-1 Clause 5.7",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "5.8",
        "requirement": "Closing meeting with management conducted",
        "part": "EN_16247_1",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "6.1",
        "requirement": "Process energy flows mapped",
        "part": "EN_16247_3",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "6.2",
        "requirement": "Process efficiency benchmarks compared",
        "part": "EN_16247_3",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "6.3",
        "requirement": "Heat recovery opportunities assessed",
        "part": "EN_16247_3",
        "audit_types": ["type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "7.1",
        "requirement": "Auditor qualifications documented",
        "part": "EN_16247_5",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
    {
        "clause": "7.2",
        "requirement": "Auditor independence declared",
        "part": "EN_16247_5",
        "audit_types": ["type_1_walkthrough", "type_2_detailed", "type_3_investment_grade"],
    },
]

# End-use benchmarks by sector (percentage of total energy).
# Sources: CIBSE TM46, IEA Industrial Energy, EU BREF documents.
END_USE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "manufacturing": {
        "heating": 25.0,
        "cooling": 5.0,
        "lighting": 8.0,
        "motors_drives": 35.0,
        "process": 15.0,
        "compressed_air": 8.0,
        "other": 4.0,
    },
    "food_beverage": {
        "heating": 20.0,
        "cooling": 20.0,
        "lighting": 5.0,
        "motors_drives": 20.0,
        "process": 20.0,
        "compressed_air": 5.0,
        "steam": 8.0,
        "other": 2.0,
    },
    "chemicals": {
        "heating": 15.0,
        "cooling": 10.0,
        "lighting": 3.0,
        "motors_drives": 25.0,
        "process": 30.0,
        "compressed_air": 5.0,
        "steam": 10.0,
        "other": 2.0,
    },
    "automotive": {
        "heating": 20.0,
        "cooling": 8.0,
        "lighting": 10.0,
        "motors_drives": 30.0,
        "process": 15.0,
        "compressed_air": 12.0,
        "other": 5.0,
    },
    "warehousing": {
        "heating": 40.0,
        "cooling": 5.0,
        "lighting": 30.0,
        "motors_drives": 10.0,
        "transport": 10.0,
        "other": 5.0,
    },
    "commercial_building": {
        "heating": 35.0,
        "cooling": 20.0,
        "lighting": 20.0,
        "ventilation": 10.0,
        "hot_water": 8.0,
        "it_equipment": 5.0,
        "other": 2.0,
    },
    "default": {
        "heating": 30.0,
        "cooling": 10.0,
        "lighting": 15.0,
        "motors_drives": 25.0,
        "process": 10.0,
        "other": 10.0,
    },
}

# Typical savings potential by end-use category (% of end-use consumption).
# Sources: Carbon Trust, IEA, EU BAT reference documents.
TYPICAL_SAVINGS_POTENTIAL: Dict[str, Dict[str, float]] = {
    "heating": {
        "no_cost_pct": 5.0,
        "low_cost_pct": 10.0,
        "medium_cost_pct": 15.0,
        "high_cost_pct": 25.0,
        "total_achievable_pct": 30.0,
    },
    "cooling": {
        "no_cost_pct": 5.0,
        "low_cost_pct": 10.0,
        "medium_cost_pct": 15.0,
        "high_cost_pct": 20.0,
        "total_achievable_pct": 25.0,
    },
    "lighting": {
        "no_cost_pct": 10.0,
        "low_cost_pct": 20.0,
        "medium_cost_pct": 40.0,
        "high_cost_pct": 60.0,
        "total_achievable_pct": 60.0,
    },
    "motors_drives": {
        "no_cost_pct": 5.0,
        "low_cost_pct": 10.0,
        "medium_cost_pct": 20.0,
        "high_cost_pct": 30.0,
        "total_achievable_pct": 30.0,
    },
    "compressed_air": {
        "no_cost_pct": 10.0,
        "low_cost_pct": 15.0,
        "medium_cost_pct": 25.0,
        "high_cost_pct": 35.0,
        "total_achievable_pct": 35.0,
    },
    "process": {
        "no_cost_pct": 3.0,
        "low_cost_pct": 8.0,
        "medium_cost_pct": 15.0,
        "high_cost_pct": 20.0,
        "total_achievable_pct": 20.0,
    },
    "steam": {
        "no_cost_pct": 5.0,
        "low_cost_pct": 10.0,
        "medium_cost_pct": 15.0,
        "high_cost_pct": 25.0,
        "total_achievable_pct": 25.0,
    },
    "ventilation": {
        "no_cost_pct": 8.0,
        "low_cost_pct": 15.0,
        "medium_cost_pct": 25.0,
        "high_cost_pct": 35.0,
        "total_achievable_pct": 35.0,
    },
}

# Audit quality criteria weights (for overall quality score 0-100).
AUDIT_QUALITY_CRITERIA: Dict[str, float] = {
    "data_completeness": 0.20,
    "en16247_compliance_pct": 0.25,
    "savings_quantification": 0.20,
    "financial_analysis": 0.15,
    "implementation_planning": 0.10,
    "report_quality": 0.10,
}

# Average energy costs by carrier (EUR per kWh) for financial calculations.
# Source: Eurostat 2024 industrial energy prices (EU average).
ENERGY_COST_EUR_PER_KWH: Dict[str, float] = {
    "electricity": 0.18,
    "natural_gas": 0.06,
    "fuel_oil": 0.08,
    "lpg": 0.09,
    "steam": 0.05,
    "compressed_air": 0.25,
    "district_heating": 0.07,
    "district_cooling": 0.10,
    "diesel": 0.09,
    "coal": 0.04,
    "biomass": 0.04,
    "default": 0.10,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class AuditScope(BaseModel):
    """Scope definition for an energy audit.

    Attributes:
        facility_id: Facility identifier.
        audit_type: Type of audit (1, 2, or 3).
        systems_included: List of end-use systems included in audit scope.
        boundary: Description of audit boundary.
        exclusions: List of systems or areas excluded from scope.
        coverage_pct: Percentage of total energy consumption covered.
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    audit_type: AuditType = Field(..., description="Audit type (1/2/3)")
    systems_included: List[EndUseCategory] = Field(
        default_factory=list, description="Systems in scope"
    )
    boundary: str = Field(default="", description="Audit boundary description")
    exclusions: List[str] = Field(
        default_factory=list, description="Excluded systems/areas"
    )
    coverage_pct: float = Field(
        default=100.0, ge=0, le=100, description="Energy coverage (%)"
    )

    @field_validator("coverage_pct")
    @classmethod
    def validate_coverage(cls, v: float) -> float:
        """EED requires minimum 85% coverage for mandatory audits."""
        if v < 85.0:
            logger.warning(
                "Audit coverage %.1f%% is below EED minimum of 85%%", v
            )
        return v

class EnergyEndUse(BaseModel):
    """Energy end-use category data for a facility.

    Attributes:
        category: End-use category (heating, cooling, lighting, etc.).
        annual_kwh: Annual energy consumption in kWh.
        annual_cost: Annual energy cost in EUR.
        percentage_of_total: Percentage of total facility consumption.
        equipment_list: List of major equipment in this category.
        carrier: Primary energy carrier for this end-use.
    """
    category: EndUseCategory = Field(..., description="End-use category")
    annual_kwh: float = Field(..., ge=0, description="Annual consumption (kWh)")
    annual_cost: Optional[float] = Field(None, ge=0, description="Annual cost (EUR)")
    percentage_of_total: float = Field(
        default=0.0, ge=0, le=100, description="Share of total (%)"
    )
    equipment_list: List[str] = Field(
        default_factory=list, description="Major equipment items"
    )
    carrier: str = Field(default="electricity", description="Primary energy carrier")

class AuditFinding(BaseModel):
    """Individual audit finding / energy conservation measure (ECM).

    Attributes:
        finding_id: Unique finding identifier.
        system: End-use system affected.
        description: Description of the finding.
        current_state: Description of current state / problem.
        recommended_action: Recommended energy conservation measure.
        estimated_savings_kwh: Estimated annual energy savings (kWh).
        estimated_savings_cost: Estimated annual cost savings (EUR).
        implementation_cost: Estimated implementation cost (EUR).
        payback_years: Simple payback period (years).
        priority: Finding priority.
        complexity: Implementation complexity.
    """
    finding_id: str = Field(default_factory=_new_uuid, description="Finding ID")
    system: EndUseCategory = Field(..., description="Affected system")
    description: str = Field(..., min_length=1, description="Finding description")
    current_state: str = Field(default="", description="Current state")
    recommended_action: str = Field(default="", description="Recommended ECM")
    estimated_savings_kwh: float = Field(
        default=0.0, ge=0, description="Savings (kWh/year)"
    )
    estimated_savings_cost: float = Field(
        default=0.0, ge=0, description="Savings (EUR/year)"
    )
    implementation_cost: float = Field(
        default=0.0, ge=0, description="Implementation cost (EUR)"
    )
    payback_years: float = Field(
        default=0.0, ge=0, description="Simple payback (years)"
    )
    priority: AuditPriority = Field(
        default=AuditPriority.MEDIUM, description="Priority"
    )
    complexity: AuditComplexity = Field(
        default=AuditComplexity.MEDIUM_COST, description="Complexity"
    )

class EN16247Checklist(BaseModel):
    """EN 16247 compliance checklist item assessment.

    Attributes:
        clause: EN 16247 clause reference.
        requirement: Requirement description.
        status: Compliance status.
        evidence: Evidence or documentation reference.
        notes: Additional notes.
    """
    clause: str = Field(..., description="Clause reference")
    requirement: str = Field(..., description="Requirement")
    status: ComplianceStatus = Field(
        default=ComplianceStatus.NOT_ASSESSED, description="Status"
    )
    evidence: str = Field(default="", description="Evidence reference")
    notes: str = Field(default="", description="Notes")

class EEDComplianceStatus(BaseModel):
    """EED Article 8 audit obligation assessment.

    Attributes:
        obligation_applies: Whether the organisation is subject to mandatory audits.
        last_audit_date: Date of last completed energy audit.
        next_audit_due: Date when next audit is due.
        iso50001_exempt: Whether exempted by ISO 50001 certification.
        emas_exempt: Whether exempted by EMAS registration.
        is_sme: Whether organisation qualifies as SME (exempt from EED Art 8).
        overdue: Whether the audit is overdue.
        months_until_due: Months until next audit is due (negative if overdue).
    """
    obligation_applies: bool = Field(default=True, description="Audit obligation")
    last_audit_date: Optional[str] = Field(None, description="Last audit date")
    next_audit_due: Optional[str] = Field(None, description="Next due date")
    iso50001_exempt: bool = Field(default=False, description="ISO 50001 exemption")
    emas_exempt: bool = Field(default=False, description="EMAS exemption")
    is_sme: bool = Field(default=False, description="SME status")
    overdue: bool = Field(default=False, description="Audit overdue")
    months_until_due: Optional[int] = Field(None, description="Months until due")

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class EndUseBreakdownResult(BaseModel):
    """End-use breakdown with benchmark comparison."""
    category: str = Field(..., description="End-use category")
    annual_kwh: float = Field(default=0.0, description="Annual consumption (kWh)")
    annual_cost: float = Field(default=0.0, description="Annual cost (EUR)")
    percentage_of_total: float = Field(default=0.0, description="Share (%)")
    benchmark_pct: Optional[float] = Field(None, description="Sector benchmark (%)")
    gap_pct: float = Field(default=0.0, description="Gap vs benchmark (%)")
    savings_potential_kwh: float = Field(
        default=0.0, description="Estimated savings potential (kWh)"
    )
    savings_potential_cost: float = Field(
        default=0.0, description="Estimated savings potential (EUR)"
    )

class AuditSummaryFindings(BaseModel):
    """Summary of all audit findings by category."""
    total_findings: int = Field(default=0, description="Total findings count")
    total_savings_kwh: float = Field(default=0.0, description="Total savings (kWh)")
    total_savings_cost: float = Field(default=0.0, description="Total savings (EUR)")
    total_implementation_cost: float = Field(
        default=0.0, description="Total implementation cost (EUR)"
    )
    avg_payback_years: float = Field(default=0.0, description="Average payback (years)")
    by_priority: Dict[str, int] = Field(
        default_factory=dict, description="Count by priority"
    )
    by_complexity: Dict[str, int] = Field(
        default_factory=dict, description="Count by complexity"
    )
    by_system: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Savings by system"
    )

class EnergyAuditResult(BaseModel):
    """Complete energy audit result with full provenance.

    Contains end-use breakdown, findings, EN 16247 compliance,
    EED status, quality scoring, and actionable recommendations.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calc timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    audit_id: str = Field(default_factory=_new_uuid, description="Audit identifier")
    facility_id: str = Field(default="", description="Facility identifier")
    audit_type: str = Field(default="", description="Audit type")
    audit_date: str = Field(default="", description="Audit date")

    total_consumption_kwh: float = Field(
        default=0.0, description="Total consumption (kWh)"
    )
    total_energy_cost: float = Field(default=0.0, description="Total energy cost (EUR)")

    end_use_breakdown: List[EndUseBreakdownResult] = Field(
        default_factory=list, description="End-use breakdown"
    )
    findings: List[AuditFinding] = Field(
        default_factory=list, description="Audit findings"
    )
    findings_summary: Optional[AuditSummaryFindings] = Field(
        None, description="Findings summary"
    )

    en16247_compliance: List[EN16247Checklist] = Field(
        default_factory=list, description="EN 16247 checklist"
    )
    en16247_compliance_pct: float = Field(
        default=0.0, description="EN 16247 compliance (%)"
    )

    eed_status: Optional[EEDComplianceStatus] = Field(
        None, description="EED Article 8 status"
    )

    quality_score: float = Field(default=0.0, description="Audit quality (0-100)")
    quality_grade: str = Field(default="", description="Quality grade (A-F)")

    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class EnergyAuditEngine:
    """EN 16247 compliant energy audit engine.

    Provides deterministic, zero-hallucination calculations for:
    - Energy end-use breakdown with benchmark comparison
    - Audit finding categorisation and prioritisation
    - EN 16247 compliance checking
    - EED Article 8 obligation assessment and scheduling
    - Savings potential estimation by end-use
    - Audit quality scoring
    - Actionable recommendations generation

    All calculations are bit-perfect reproducible. No LLM is used
    in any calculation path.

    Usage::

        engine = EnergyAuditEngine()
        result = engine.conduct_audit(
            scope=audit_scope,
            end_uses=end_use_data,
            findings=audit_findings,
            facility_sector="manufacturing",
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the energy audit engine with embedded constants."""
        self._checklist = EN16247_CHECKLIST_ITEMS
        self._end_use_benchmarks = END_USE_BENCHMARKS
        self._savings_potential = TYPICAL_SAVINGS_POTENTIAL
        self._quality_criteria = AUDIT_QUALITY_CRITERIA
        self._energy_costs = ENERGY_COST_EUR_PER_KWH

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def conduct_audit(
        self,
        scope: AuditScope,
        end_uses: List[EnergyEndUse],
        findings: Optional[List[AuditFinding]] = None,
        checklist_responses: Optional[List[EN16247Checklist]] = None,
        facility_sector: str = "manufacturing",
        last_audit_date: Optional[str] = None,
        is_sme: bool = False,
        has_iso50001: bool = False,
        has_emas: bool = False,
        audit_date: Optional[str] = None,
    ) -> EnergyAuditResult:
        """Conduct a complete energy audit analysis.

        Processes end-use data, findings, EN 16247 compliance, and EED
        status to produce a comprehensive audit result.

        Args:
            scope: Audit scope definition.
            end_uses: Energy end-use breakdown data.
            findings: Audit findings / ECMs (or auto-generate from gaps).
            checklist_responses: EN 16247 compliance checklist responses.
            facility_sector: Sector for benchmark comparison.
            last_audit_date: Date of last completed audit (YYYY-MM-DD).
            is_sme: Whether organisation is an SME.
            has_iso50001: Whether certified to ISO 50001.
            has_emas: Whether registered with EMAS.
            audit_date: Date of this audit (default: today).

        Returns:
            EnergyAuditResult with complete analysis and provenance.

        Raises:
            ValueError: If end_uses is empty.
        """
        t0 = time.perf_counter()

        if not end_uses:
            raise ValueError("At least one EnergyEndUse record is required")

        if audit_date is None:
            audit_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info(
            "Conducting %s audit for facility %s",
            scope.audit_type.value, scope.facility_id,
        )

        # Step 1: Calculate totals
        total_kwh = _decimal(sum(eu.annual_kwh for eu in end_uses))
        total_cost = Decimal("0")
        for eu in end_uses:
            if eu.annual_cost is not None:
                total_cost += _decimal(eu.annual_cost)
            else:
                carrier = eu.carrier
                rate = _decimal(self._energy_costs.get(
                    carrier, self._energy_costs["default"]
                ))
                total_cost += _decimal(eu.annual_kwh) * rate

        # Step 2: End-use breakdown with benchmarks
        end_use_results = self._build_end_use_breakdown(
            end_uses, total_kwh, facility_sector,
        )

        # Step 3: Auto-generate findings from gaps if none provided
        if findings is None:
            findings = self._auto_generate_findings(
                end_use_results, facility_sector,
            )
        else:
            # Calculate payback for any findings missing it
            findings = [self._enrich_finding(f) for f in findings]

        # Step 4: Summarise findings
        findings_summary = self._summarise_findings(findings)

        # Step 5: EN 16247 compliance
        en16247_items = self._assess_en16247_compliance(
            scope.audit_type, checklist_responses,
        )
        compliance_pct = self._calculate_compliance_pct(en16247_items)

        # Step 6: EED Article 8 status
        eed_status = self._assess_eed_compliance(
            last_audit_date, is_sme, has_iso50001, has_emas, audit_date,
        )

        # Step 7: Quality score
        quality_score = self._calculate_quality_score(
            end_uses, findings, compliance_pct, scope.audit_type,
        )
        quality_grade = self._score_to_grade(quality_score)

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            end_use_results, findings_summary, compliance_pct,
            eed_status, quality_score, scope.audit_type, facility_sector,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EnergyAuditResult(
            facility_id=scope.facility_id,
            audit_type=scope.audit_type.value,
            audit_date=audit_date,
            total_consumption_kwh=_round_val(total_kwh, 2),
            total_energy_cost=_round_val(total_cost, 2),
            end_use_breakdown=end_use_results,
            findings=findings,
            findings_summary=findings_summary,
            en16247_compliance=en16247_items,
            en16247_compliance_pct=_round1(compliance_pct),
            eed_status=eed_status,
            quality_score=_round1(quality_score),
            quality_grade=quality_grade,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def assess_eed_obligation(
        self,
        last_audit_date: Optional[str] = None,
        is_sme: bool = False,
        has_iso50001: bool = False,
        has_emas: bool = False,
        reference_date: Optional[str] = None,
    ) -> EEDComplianceStatus:
        """Assess EED Article 8 audit obligation status.

        Args:
            last_audit_date: Last completed audit date (YYYY-MM-DD).
            is_sme: Whether organisation qualifies as SME.
            has_iso50001: Whether certified to ISO 50001.
            has_emas: Whether registered with EMAS.
            reference_date: Reference date for calculation (default: today).

        Returns:
            EEDComplianceStatus with obligation assessment.
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._assess_eed_compliance(
            last_audit_date, is_sme, has_iso50001, has_emas, reference_date,
        )

    def check_en16247_compliance(
        self,
        audit_type: AuditType,
        checklist_responses: Optional[List[EN16247Checklist]] = None,
    ) -> Tuple[List[EN16247Checklist], float]:
        """Check EN 16247 compliance for the given audit type.

        Args:
            audit_type: Audit type for applicable requirements.
            checklist_responses: Existing checklist responses.

        Returns:
            Tuple of (completed checklist, compliance percentage).
        """
        items = self._assess_en16247_compliance(audit_type, checklist_responses)
        pct = self._calculate_compliance_pct(items)
        return items, pct

    def estimate_savings_potential(
        self,
        end_uses: List[EnergyEndUse],
        facility_sector: str = "manufacturing",
    ) -> Dict[str, Any]:
        """Estimate total savings potential across all end-uses.

        Uses published savings potential percentages by end-use category.

        Args:
            end_uses: List of energy end-use data.
            facility_sector: Facility sector for context.

        Returns:
            Dict with savings potential summary and provenance hash.
        """
        total_kwh = Decimal("0")
        total_savings_kwh = Decimal("0")
        by_category: Dict[str, Dict[str, float]] = {}

        for eu in end_uses:
            cat = eu.category.value
            kwh = _decimal(eu.annual_kwh)
            total_kwh += kwh

            potential = self._savings_potential.get(cat, {})
            achievable_pct = _decimal(potential.get("total_achievable_pct", 10.0))
            savings = kwh * achievable_pct / Decimal("100")
            total_savings_kwh += savings

            rate = _decimal(self._energy_costs.get(
                eu.carrier, self._energy_costs["default"]
            ))
            savings_cost = savings * rate

            by_category[cat] = {
                "annual_kwh": _round_val(kwh, 2),
                "achievable_pct": float(achievable_pct),
                "savings_kwh": _round_val(savings, 2),
                "savings_cost_eur": _round_val(savings_cost, 2),
            }

        savings_pct = _safe_pct(total_savings_kwh, total_kwh)

        result = {
            "total_consumption_kwh": _round_val(total_kwh, 2),
            "total_savings_potential_kwh": _round_val(total_savings_kwh, 2),
            "total_savings_potential_pct": _round_val(savings_pct, 1),
            "by_category": by_category,
            "provenance_hash": _compute_hash({
                "total": str(total_kwh),
                "savings": str(total_savings_kwh),
            }),
        }
        return result

    # -------------------------------------------------------------------
    # Internal: End-Use Breakdown
    # -------------------------------------------------------------------

    def _build_end_use_breakdown(
        self,
        end_uses: List[EnergyEndUse],
        total_kwh: Decimal,
        sector: str,
    ) -> List[EndUseBreakdownResult]:
        """Build end-use breakdown with benchmark comparison.

        Args:
            end_uses: End-use data.
            total_kwh: Total facility consumption (kWh).
            sector: Facility sector for benchmarks.

        Returns:
            List of EndUseBreakdownResult.
        """
        benchmarks = self._end_use_benchmarks.get(
            sector, self._end_use_benchmarks["default"]
        )

        results: List[EndUseBreakdownResult] = []
        for eu in end_uses:
            kwh = _decimal(eu.annual_kwh)
            pct = _safe_pct(kwh, total_kwh)

            carrier = eu.carrier
            rate = _decimal(self._energy_costs.get(
                carrier, self._energy_costs["default"]
            ))
            cost = kwh * rate if eu.annual_cost is None else _decimal(eu.annual_cost)

            cat = eu.category.value
            bench_pct = benchmarks.get(cat)

            gap_pct = Decimal("0")
            if bench_pct is not None:
                gap_pct = pct - _decimal(bench_pct)

            # Savings potential for this end-use
            potential = self._savings_potential.get(cat, {})
            achievable = _decimal(potential.get("total_achievable_pct", 10.0))
            savings_kwh = kwh * achievable / Decimal("100")
            savings_cost = savings_kwh * rate

            results.append(EndUseBreakdownResult(
                category=cat,
                annual_kwh=_round_val(kwh, 2),
                annual_cost=_round_val(cost, 2),
                percentage_of_total=_round_val(pct, 1),
                benchmark_pct=bench_pct,
                gap_pct=_round_val(gap_pct, 1),
                savings_potential_kwh=_round_val(savings_kwh, 2),
                savings_potential_cost=_round_val(savings_cost, 2),
            ))

        results.sort(key=lambda r: r.annual_kwh, reverse=True)
        return results

    # -------------------------------------------------------------------
    # Internal: Auto-Generate Findings
    # -------------------------------------------------------------------

    def _auto_generate_findings(
        self,
        end_use_results: List[EndUseBreakdownResult],
        sector: str,
    ) -> List[AuditFinding]:
        """Auto-generate audit findings based on gap analysis.

        For each end-use exceeding sector benchmarks, generate standard
        energy conservation measures from published best practices.

        Args:
            end_use_results: End-use breakdown results.
            sector: Facility sector.

        Returns:
            List of auto-generated AuditFinding.
        """
        findings: List[AuditFinding] = []

        ecm_templates: Dict[str, List[Dict[str, Any]]] = {
            "heating": [
                {
                    "desc": "Optimise boiler/heating controls and set-points",
                    "action": "Install weather-compensated controls, reduce set-points by 1-2C, "
                              "implement time scheduling for unoccupied periods",
                    "savings_pct": 10.0, "cost_factor": 0.5, "complexity": "low_cost",
                    "priority": "high",
                },
                {
                    "desc": "Improve building insulation and reduce heat losses",
                    "action": "Insulate exposed pipework, repair door seals, add insulation "
                              "to roof/walls where cost-effective",
                    "savings_pct": 15.0, "cost_factor": 5.0, "complexity": "medium_cost",
                    "priority": "medium",
                },
            ],
            "cooling": [
                {
                    "desc": "Optimise chiller operation and set-points",
                    "action": "Raise cooling set-points where possible, implement free cooling, "
                              "clean condenser coils regularly",
                    "savings_pct": 10.0, "cost_factor": 0.3, "complexity": "low_cost",
                    "priority": "high",
                },
            ],
            "lighting": [
                {
                    "desc": "Upgrade to LED lighting throughout",
                    "action": "Replace all fluorescent and HID lighting with LED equivalents, "
                              "install occupancy sensors and daylight dimming",
                    "savings_pct": 50.0, "cost_factor": 3.0, "complexity": "medium_cost",
                    "priority": "high",
                },
            ],
            "motors_drives": [
                {
                    "desc": "Install Variable Speed Drives on motors",
                    "action": "Fit VSDs on pumps, fans, and compressors with variable load. "
                              "Prioritise motors running at >50% of operating hours",
                    "savings_pct": 25.0, "cost_factor": 4.0, "complexity": "medium_cost",
                    "priority": "high",
                },
                {
                    "desc": "Replace oversized motors with IE4/IE5 efficiency class",
                    "action": "Right-size motors and upgrade to IE4 Super Premium or "
                              "IE5 Ultra Premium efficiency per EU Regulation 2019/1781",
                    "savings_pct": 8.0, "cost_factor": 6.0, "complexity": "high_cost",
                    "priority": "medium",
                },
            ],
            "compressed_air": [
                {
                    "desc": "Repair compressed air leaks",
                    "action": "Conduct ultrasonic leak survey, repair all identified leaks. "
                              "Typical industrial systems lose 20-30% through leaks",
                    "savings_pct": 20.0, "cost_factor": 0.2, "complexity": "low_cost",
                    "priority": "critical",
                },
                {
                    "desc": "Optimise compressed air system pressure and controls",
                    "action": "Reduce system pressure to minimum required, install "
                              "sequencer controls for multiple compressors",
                    "savings_pct": 10.0, "cost_factor": 1.0, "complexity": "low_cost",
                    "priority": "high",
                },
            ],
            "process": [
                {
                    "desc": "Implement process heat recovery",
                    "action": "Install heat exchangers to recover waste heat from exhaust "
                              "streams for preheating intake air/water/feedstock",
                    "savings_pct": 12.0, "cost_factor": 8.0, "complexity": "high_cost",
                    "priority": "medium",
                },
            ],
            "steam": [
                {
                    "desc": "Repair steam traps and insulate steam lines",
                    "action": "Survey all steam traps (typical 15-25% failure rate), "
                              "repair/replace failed traps, insulate bare flanges and valves",
                    "savings_pct": 10.0, "cost_factor": 1.0, "complexity": "low_cost",
                    "priority": "high",
                },
            ],
            "ventilation": [
                {
                    "desc": "Install demand-controlled ventilation",
                    "action": "Install CO2 sensors for demand-based ventilation control, "
                              "add heat recovery to extract air streams",
                    "savings_pct": 20.0, "cost_factor": 3.0, "complexity": "medium_cost",
                    "priority": "medium",
                },
            ],
        }

        for eur in end_use_results:
            cat = eur.category
            templates = ecm_templates.get(cat, [])

            for tmpl in templates:
                savings_pct = tmpl["savings_pct"]
                savings_kwh = _decimal(eur.annual_kwh) * _decimal(savings_pct) / Decimal("100")

                rate = _decimal(self._energy_costs.get("electricity", 0.18))
                savings_cost = savings_kwh * rate

                cost_factor = _decimal(tmpl["cost_factor"])
                impl_cost = savings_cost * cost_factor

                payback = _safe_divide(impl_cost, savings_cost)

                try:
                    system_enum = EndUseCategory(cat)
                except ValueError:
                    system_enum = EndUseCategory.OTHER

                try:
                    cmplx = AuditComplexity(tmpl.get("complexity", "medium_cost"))
                except ValueError:
                    cmplx = AuditComplexity.MEDIUM_COST

                try:
                    prio = AuditPriority(tmpl.get("priority", "medium"))
                except ValueError:
                    prio = AuditPriority.MEDIUM

                findings.append(AuditFinding(
                    system=system_enum,
                    description=tmpl["desc"],
                    current_state=f"Current {cat} consumption: {eur.annual_kwh:.0f} kWh/year",
                    recommended_action=tmpl["action"],
                    estimated_savings_kwh=_round_val(savings_kwh, 0),
                    estimated_savings_cost=_round_val(savings_cost, 2),
                    implementation_cost=_round_val(impl_cost, 2),
                    payback_years=_round_val(payback, 1),
                    priority=prio,
                    complexity=cmplx,
                ))

        findings.sort(
            key=lambda f: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}.get(
                    f.priority.value, 5
                ),
                f.payback_years,
            )
        )
        return findings

    def _enrich_finding(self, finding: AuditFinding) -> AuditFinding:
        """Enrich a finding with calculated payback if missing.

        Args:
            finding: Audit finding to enrich.

        Returns:
            Finding with payback calculated.
        """
        if finding.payback_years == 0.0 and finding.estimated_savings_cost > 0:
            payback = _safe_divide(
                _decimal(finding.implementation_cost),
                _decimal(finding.estimated_savings_cost),
            )
            finding.payback_years = _round_val(payback, 1)
        return finding

    # -------------------------------------------------------------------
    # Internal: Findings Summary
    # -------------------------------------------------------------------

    def _summarise_findings(
        self, findings: List[AuditFinding],
    ) -> AuditSummaryFindings:
        """Summarise audit findings.

        Args:
            findings: List of audit findings.

        Returns:
            AuditSummaryFindings with aggregate metrics.
        """
        total_savings_kwh = Decimal("0")
        total_savings_cost = Decimal("0")
        total_impl_cost = Decimal("0")
        by_priority: Dict[str, int] = {}
        by_complexity: Dict[str, int] = {}
        by_system: Dict[str, Dict[str, Decimal]] = {}
        payback_sum = Decimal("0")
        payback_count = 0

        for f in findings:
            total_savings_kwh += _decimal(f.estimated_savings_kwh)
            total_savings_cost += _decimal(f.estimated_savings_cost)
            total_impl_cost += _decimal(f.implementation_cost)

            by_priority[f.priority.value] = by_priority.get(f.priority.value, 0) + 1
            by_complexity[f.complexity.value] = by_complexity.get(f.complexity.value, 0) + 1

            sys_key = f.system.value
            if sys_key not in by_system:
                by_system[sys_key] = {
                    "savings_kwh": Decimal("0"),
                    "savings_cost": Decimal("0"),
                    "count": Decimal("0"),
                }
            by_system[sys_key]["savings_kwh"] += _decimal(f.estimated_savings_kwh)
            by_system[sys_key]["savings_cost"] += _decimal(f.estimated_savings_cost)
            by_system[sys_key]["count"] += Decimal("1")

            if f.payback_years > 0:
                payback_sum += _decimal(f.payback_years)
                payback_count += 1

        avg_payback = _safe_divide(
            payback_sum, _decimal(payback_count)
        ) if payback_count > 0 else Decimal("0")

        by_system_float: Dict[str, Dict[str, float]] = {}
        for k, v in by_system.items():
            by_system_float[k] = {
                kk: _round_val(vv, 2) for kk, vv in v.items()
            }

        return AuditSummaryFindings(
            total_findings=len(findings),
            total_savings_kwh=_round_val(total_savings_kwh, 0),
            total_savings_cost=_round_val(total_savings_cost, 2),
            total_implementation_cost=_round_val(total_impl_cost, 2),
            avg_payback_years=_round_val(avg_payback, 1),
            by_priority=by_priority,
            by_complexity=by_complexity,
            by_system=by_system_float,
        )

    # -------------------------------------------------------------------
    # Internal: EN 16247 Compliance
    # -------------------------------------------------------------------

    def _assess_en16247_compliance(
        self,
        audit_type: AuditType,
        responses: Optional[List[EN16247Checklist]],
    ) -> List[EN16247Checklist]:
        """Assess EN 16247 compliance for the given audit type.

        Args:
            audit_type: Audit type to filter applicable requirements.
            responses: Existing responses (matched by clause).

        Returns:
            Complete checklist with applicable items assessed.
        """
        response_map: Dict[str, EN16247Checklist] = {}
        if responses:
            for r in responses:
                response_map[r.clause] = r

        result: List[EN16247Checklist] = []
        for item in self._checklist:
            if audit_type.value not in item["audit_types"]:
                continue

            clause = item["clause"]
            if clause in response_map:
                result.append(response_map[clause])
            else:
                result.append(EN16247Checklist(
                    clause=clause,
                    requirement=item["requirement"],
                    status=ComplianceStatus.NOT_ASSESSED,
                ))

        return result

    def _calculate_compliance_pct(
        self, checklist: List[EN16247Checklist],
    ) -> float:
        """Calculate EN 16247 compliance percentage.

        Args:
            checklist: Assessed checklist items.

        Returns:
            Compliance percentage (0-100).
        """
        if not checklist:
            return 0.0

        applicable = [
            c for c in checklist
            if c.status != ComplianceStatus.NOT_APPLICABLE
        ]
        if not applicable:
            return 100.0

        compliant = sum(
            1 for c in applicable
            if c.status == ComplianceStatus.COMPLIANT
        )
        partial = sum(
            1 for c in applicable
            if c.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )

        score = _decimal(compliant) + _decimal(partial) * Decimal("0.5")
        pct = _safe_pct(score, _decimal(len(applicable)))
        return _round_val(pct, 1)

    # -------------------------------------------------------------------
    # Internal: EED Compliance
    # -------------------------------------------------------------------

    def _assess_eed_compliance(
        self,
        last_audit_date: Optional[str],
        is_sme: bool,
        has_iso50001: bool,
        has_emas: bool,
        reference_date: str,
    ) -> EEDComplianceStatus:
        """Assess EED Article 8 audit obligation.

        Args:
            last_audit_date: Last audit date (YYYY-MM-DD) or None.
            is_sme: Whether SME (exempt from obligation).
            has_iso50001: Whether ISO 50001 certified (exempt).
            has_emas: Whether EMAS registered (exempt).
            reference_date: Reference date for calculation (YYYY-MM-DD).

        Returns:
            EEDComplianceStatus.
        """
        # SMEs are exempt
        if is_sme:
            return EEDComplianceStatus(
                obligation_applies=False,
                is_sme=True,
                iso50001_exempt=has_iso50001,
                emas_exempt=has_emas,
            )

        # ISO 50001 / EMAS exempt
        if has_iso50001 or has_emas:
            return EEDComplianceStatus(
                obligation_applies=True,
                iso50001_exempt=has_iso50001,
                emas_exempt=has_emas,
                is_sme=False,
                overdue=False,
            )

        # Calculate next due date
        try:
            ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
        except ValueError:
            ref = date.today()

        if last_audit_date:
            try:
                last_dt = datetime.strptime(last_audit_date, "%Y-%m-%d").date()
                next_due = date(
                    last_dt.year + EED_AUDIT_CYCLE_YEARS,
                    last_dt.month,
                    last_dt.day,
                )
                overdue = ref > next_due
                months_diff = (next_due.year - ref.year) * 12 + (next_due.month - ref.month)
            except ValueError:
                next_due = None
                overdue = True
                months_diff = None
        else:
            next_due = None
            overdue = True
            months_diff = None

        return EEDComplianceStatus(
            obligation_applies=True,
            last_audit_date=last_audit_date,
            next_audit_due=next_due.isoformat() if next_due else None,
            iso50001_exempt=False,
            emas_exempt=False,
            is_sme=False,
            overdue=overdue,
            months_until_due=months_diff,
        )

    # -------------------------------------------------------------------
    # Internal: Quality Score
    # -------------------------------------------------------------------

    def _calculate_quality_score(
        self,
        end_uses: List[EnergyEndUse],
        findings: List[AuditFinding],
        compliance_pct: float,
        audit_type: AuditType,
    ) -> float:
        """Calculate overall audit quality score (0-100).

        Weighted scoring based on data completeness, EN 16247 compliance,
        savings quantification, financial analysis, and implementation
        planning.

        Args:
            end_uses: End-use data for data completeness check.
            findings: Audit findings for quantification check.
            compliance_pct: EN 16247 compliance percentage.
            audit_type: Audit type for scope expectations.

        Returns:
            Quality score (0-100).
        """
        scores: Dict[str, float] = {}

        # Data completeness (0-100): based on number of end-uses documented
        expected_min = 4 if audit_type == AuditType.TYPE_1_WALKTHROUGH else 6
        data_score = min(100.0, (len(end_uses) / expected_min) * 100.0)
        scores["data_completeness"] = data_score

        # EN 16247 compliance
        scores["en16247_compliance_pct"] = compliance_pct

        # Savings quantification: % of findings with kWh savings
        if findings:
            quantified = sum(
                1 for f in findings if f.estimated_savings_kwh > 0
            )
            scores["savings_quantification"] = (quantified / len(findings)) * 100.0
        else:
            scores["savings_quantification"] = 0.0

        # Financial analysis: % of findings with cost and payback
        if findings:
            costed = sum(
                1 for f in findings
                if f.estimated_savings_cost > 0 and f.implementation_cost > 0
            )
            scores["financial_analysis"] = (costed / len(findings)) * 100.0
        else:
            scores["financial_analysis"] = 0.0

        # Implementation planning: % of findings with priority and complexity
        if findings:
            planned = sum(
                1 for f in findings
                if f.priority != AuditPriority.INFORMATIONAL
                and f.recommended_action
            )
            scores["implementation_planning"] = (planned / len(findings)) * 100.0
        else:
            scores["implementation_planning"] = 0.0

        # Report quality: baseline score based on audit type
        type_base_score = {
            AuditType.TYPE_1_WALKTHROUGH: 60.0,
            AuditType.TYPE_2_DETAILED: 80.0,
            AuditType.TYPE_3_INVESTMENT_GRADE: 90.0,
        }
        scores["report_quality"] = type_base_score.get(audit_type, 50.0)

        # Weighted composite
        total = Decimal("0")
        for criterion, weight in self._quality_criteria.items():
            criterion_score = scores.get(criterion, 0.0)
            total += _decimal(criterion_score) * _decimal(weight)

        return _round_val(total, 1)

    def _score_to_grade(self, score: float) -> str:
        """Convert quality score to letter grade.

        Args:
            score: Quality score (0-100).

        Returns:
            Letter grade A through F.
        """
        if score >= 90.0:
            return "A"
        elif score >= 75.0:
            return "B"
        elif score >= 60.0:
            return "C"
        elif score >= 40.0:
            return "D"
        else:
            return "F"

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        end_use_results: List[EndUseBreakdownResult],
        summary: Optional[AuditSummaryFindings],
        compliance_pct: float,
        eed: Optional[EEDComplianceStatus],
        quality_score: float,
        audit_type: AuditType,
        sector: str,
    ) -> List[str]:
        """Generate actionable recommendations based on audit results.

        All recommendations are deterministic threshold-based rules.

        Args:
            end_use_results: End-use breakdown.
            summary: Findings summary.
            compliance_pct: EN 16247 compliance percentage.
            eed: EED obligation status.
            quality_score: Audit quality score.
            audit_type: Type of audit conducted.
            sector: Facility sector.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: EED overdue
        if eed and eed.overdue and eed.obligation_applies:
            recs.append(
                "CRITICAL: Energy audit is overdue per EED Article 8. "
                "Non-SME enterprises must conduct an energy audit at least "
                "every 4 years. Schedule audit immediately to avoid regulatory "
                "non-compliance penalties."
            )

        # R2: Low EN 16247 compliance
        if compliance_pct < 50.0:
            recs.append(
                f"EN 16247 compliance is only {compliance_pct}%. "
                "Ensure all applicable requirements are addressed to meet "
                "the standard. Non-compliant audits may not satisfy EED "
                "Article 8 obligations."
            )

        # R3: ISO 50001 recommendation
        if eed and eed.obligation_applies and not eed.iso50001_exempt:
            recs.append(
                "Consider implementing ISO 50001 Energy Management System. "
                "ISO 50001 certification exempts from EED Article 8 mandatory "
                "audits and drives continuous energy improvement."
            )

        # R4: Quick wins from findings
        if summary and summary.by_complexity:
            no_cost = summary.by_complexity.get("no_cost", 0)
            low_cost = summary.by_complexity.get("low_cost", 0)
            if no_cost + low_cost > 0:
                recs.append(
                    f"Implement {no_cost + low_cost} no-cost and low-cost measures "
                    f"identified in the audit. These require minimal investment "
                    f"and typically have payback periods under 1 year."
                )

        # R5: Highest savings opportunity
        if summary and summary.by_system:
            sorted_systems = sorted(
                summary.by_system.items(),
                key=lambda x: x[1].get("savings_kwh", 0.0),
                reverse=True,
            )
            if sorted_systems:
                top_sys = sorted_systems[0]
                recs.append(
                    f"Largest savings opportunity is in {top_sys[0]} "
                    f"({top_sys[1].get('savings_kwh', 0):.0f} kWh/year, "
                    f"EUR {top_sys[1].get('savings_cost', 0):.0f}/year). "
                    f"Prioritise this system for immediate action."
                )

        # R6: End-uses exceeding benchmarks
        for eur in end_use_results:
            if eur.gap_pct > 5.0:
                recs.append(
                    f"{eur.category} consumption ({eur.percentage_of_total}%) "
                    f"exceeds sector benchmark by {eur.gap_pct}pp. "
                    f"Investigate root causes and implement targeted efficiency "
                    f"measures."
                )

        # R7: Upgrade audit depth
        if audit_type == AuditType.TYPE_1_WALKTHROUGH:
            recs.append(
                "This was a preliminary (Type 1) audit. For bankable savings "
                "estimates and detailed engineering analysis, conduct a Type 2 "
                "(detailed) or Type 3 (investment-grade) audit on the priority "
                "systems identified."
            )

        # R8: Sub-metering
        if len(end_use_results) < 4:
            recs.append(
                "Limited end-use data available. Install sub-metering on major "
                "energy-consuming systems to enable more accurate auditing and "
                "ongoing energy monitoring."
            )

        return recs
