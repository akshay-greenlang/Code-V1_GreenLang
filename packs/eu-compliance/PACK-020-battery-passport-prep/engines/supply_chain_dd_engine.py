# -*- coding: utf-8 -*-
"""
SupplyChainDDEngine - PACK-020 Battery Passport Engine 5
=========================================================

Assesses supply chain due diligence for critical raw materials
used in battery manufacturing per Article 48 of the EU Battery
Regulation (2023/1542).

Article 48 of the EU Battery Regulation requires economic operators
placing batteries on the market to establish and implement a supply
chain due diligence policy consistent with the internationally
recognised due diligence standards.  Specifically, the regulation
references the OECD Due Diligence Guidance for Responsible Supply
Chains of Minerals from Conflict-Affected and High-Risk Areas and
its five-step framework.

Critical Raw Materials Covered:
    - Cobalt:  Key cathode material, major supply from DRC
    - Lithium: Primary charge carrier, sourced from AU/CL/AR/CN
    - Nickel:  Cathode material, sourced from ID/PH/RU/FI/AU
    - Natural Graphite:  Anode material, major supply from CN/MZ
    - Manganese:  Cathode additive, sourced from ZA/GA/AU/CN

OECD Five-Step Framework:
    Step 1:  Establish strong company management systems
    Step 2:  Identify and assess risk in the supply chain
    Step 3:  Design and implement a strategy to respond to risks
    Step 4:  Carry out independent third-party audit
    Step 5:  Report on supply chain due diligence

Regulatory References:
    - EU Regulation 2023/1542 (EU Battery Regulation), Article 48
    - OECD Due Diligence Guidance (Annex I minerals)
    - Regulation (EU) 2017/821 (Conflict Minerals Regulation)
    - UN Guiding Principles on Business and Human Rights
    - ILO Conventions on Forced Labour and Child Labour

Zero-Hallucination:
    - Risk scoring uses deterministic weighted rubrics
    - Compliance rates are ratio-based arithmetic
    - Country risk uses static lookup tables
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-020 Battery Passport Prep Pack
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CriticalRawMaterial(str, Enum):
    """Critical raw materials for battery manufacturing per Art 48.

    These are the key minerals and metals whose supply chains
    must be subject to due diligence under the EU Battery
    Regulation.
    """
    COBALT = "cobalt"
    LITHIUM = "lithium"
    NICKEL = "nickel"
    NATURAL_GRAPHITE = "natural_graphite"
    MANGANESE = "manganese"


class DueDiligenceRisk(str, Enum):
    """Due diligence risk levels for supply chain assessment.

    Risk levels are assigned based on country risk, supplier
    tier, audit status, and OECD compliance.
    """
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class OECDStep(str, Enum):
    """OECD Due Diligence five-step framework.

    The five steps of the OECD Due Diligence Guidance for
    Responsible Supply Chains of Minerals from Conflict-Affected
    and High-Risk Areas.
    """
    STEP_1 = "step_1_management_systems"
    STEP_2 = "step_2_risk_identification"
    STEP_3 = "step_3_risk_response"
    STEP_4 = "step_4_third_party_audit"
    STEP_5 = "step_5_reporting"


class SupplierTier(str, Enum):
    """Supplier tier in the battery supply chain.

    Tier 1 is a direct supplier to the battery manufacturer;
    higher tiers are further upstream towards raw material
    extraction.
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# High-risk countries for each critical raw material.
# Based on known conflict-affected and high-risk areas (CAHRAs)
# and countries with governance or environmental concerns.
HIGH_RISK_COUNTRIES: Dict[str, List[str]] = {
    CriticalRawMaterial.COBALT.value: [
        "CD",  # Democratic Republic of the Congo
        "MM",  # Myanmar
        "CN",  # China (artisanal processing concerns)
        "ZM",  # Zambia
        "CG",  # Republic of the Congo
        "MG",  # Madagascar
        "PH",  # Philippines (artisanal mining)
    ],
    CriticalRawMaterial.LITHIUM.value: [
        "CL",  # Chile (water scarcity concerns)
        "AU",  # Australia (Aboriginal land rights)
        "AR",  # Argentina (indigenous rights, water)
        "BO",  # Bolivia
        "ZW",  # Zimbabwe
        "CN",  # China (environmental governance)
    ],
    CriticalRawMaterial.NICKEL.value: [
        "ID",  # Indonesia (deforestation, labour)
        "PH",  # Philippines (environmental impact)
        "RU",  # Russia (sanctions risk)
        "MM",  # Myanmar
        "CN",  # China
        "GT",  # Guatemala
    ],
    CriticalRawMaterial.NATURAL_GRAPHITE.value: [
        "CN",  # China (dominant supplier, governance)
        "MZ",  # Mozambique (conflict)
        "MG",  # Madagascar
        "TZ",  # Tanzania
        "BR",  # Brazil (deforestation adjacent)
    ],
    CriticalRawMaterial.MANGANESE.value: [
        "ZA",  # South Africa (labour concerns)
        "GA",  # Gabon (governance)
        "GH",  # Ghana
        "CI",  # Cote d'Ivoire
        "MM",  # Myanmar
        "CN",  # China
    ],
}

# Elevated risk countries (medium risk) by material.
ELEVATED_RISK_COUNTRIES: Dict[str, List[str]] = {
    CriticalRawMaterial.COBALT.value: ["AU", "CA", "CU", "MA"],
    CriticalRawMaterial.LITHIUM.value: ["US", "PT", "BR", "MX"],
    CriticalRawMaterial.NICKEL.value: ["AU", "CA", "FI", "NC"],
    CriticalRawMaterial.NATURAL_GRAPHITE.value: [
        "IN", "CA", "KR", "NO",
    ],
    CriticalRawMaterial.MANGANESE.value: ["AU", "BR", "IN", "UA"],
}

# Country governance scores (0-100, higher is better governance).
# Based on composite of World Bank WGI, TI CPI, and EITI participation.
COUNTRY_GOVERNANCE_SCORES: Dict[str, int] = {
    "CD": 8, "MM": 12, "CN": 42, "ZM": 35, "CG": 18, "MG": 25,
    "CL": 68, "AU": 85, "AR": 45, "BO": 30, "ZW": 15,
    "ID": 40, "PH": 38, "RU": 25, "GT": 28,
    "MZ": 22, "TZ": 32, "BR": 50,
    "ZA": 52, "GA": 28, "GH": 50, "CI": 35,
    "FI": 92, "DE": 88, "SE": 90, "NO": 91, "CA": 87,
    "US": 80, "JP": 82, "KR": 75, "PT": 72,
    "NC": 70, "IN": 45, "UA": 30, "MA": 40, "CU": 22,
    "MX": 42, "BE": 85,
}

# OECD step descriptions for compliance reporting.
OECD_STEP_DESCRIPTIONS: Dict[str, str] = {
    OECDStep.STEP_1.value: (
        "Establish strong company management systems including a supply "
        "chain due diligence policy, internal management structures, and "
        "a system of controls and transparency over the supply chain"
    ),
    OECDStep.STEP_2.value: (
        "Identify and assess risk in the supply chain by mapping the "
        "supply chain, identifying high-risk areas, and assessing "
        "conditions against standards"
    ),
    OECDStep.STEP_3.value: (
        "Design and implement a strategy to respond to identified risks "
        "through risk mitigation plans, monitoring, and engagement with "
        "suppliers"
    ),
    OECDStep.STEP_4.value: (
        "Carry out independent third-party audit of supply chain due "
        "diligence practices at identified points in the supply chain"
    ),
    OECDStep.STEP_5.value: (
        "Report annually on supply chain due diligence policies, "
        "identified risks, risk mitigation measures, and audit findings"
    ),
}

# Risk score weights for supplier risk assessment.
RISK_WEIGHTS: Dict[str, float] = {
    "country_risk": 0.30,
    "tier_risk": 0.15,
    "oecd_compliance": 0.25,
    "audit_status": 0.20,
    "material_criticality": 0.10,
}

# Material criticality scores (0-100, higher means more critical).
MATERIAL_CRITICALITY: Dict[str, int] = {
    CriticalRawMaterial.COBALT.value: 95,
    CriticalRawMaterial.LITHIUM.value: 90,
    CriticalRawMaterial.NICKEL.value: 80,
    CriticalRawMaterial.NATURAL_GRAPHITE.value: 75,
    CriticalRawMaterial.MANGANESE.value: 60,
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class OECDStepAssessment(BaseModel):
    """Assessment of a single OECD due diligence step for a supplier.

    Records whether the supplier meets the requirements of each
    step in the OECD five-step framework.
    """
    step: str = Field(
        ...,
        description="OECD step identifier",
    )
    step_name: str = Field(
        default="",
        description="Human-readable step name",
    )
    description: str = Field(
        default="",
        description="Description of the step requirements",
    )
    compliant: bool = Field(
        default=False,
        description="Whether the supplier is compliant with this step",
    )
    evidence: str = Field(
        default="",
        description="Evidence supporting the compliance determination",
        max_length=2000,
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Identified gaps in compliance for this step",
    )


class SupplierAssessment(BaseModel):
    """Assessment of a single supplier in the battery supply chain.

    Captures the due diligence profile of a supplier including
    material supplied, geographic risk, tier, and compliance
    with OECD due diligence standards.
    """
    supplier_id: str = Field(
        ...,
        description="Unique supplier identifier",
    )
    name: str = Field(
        ...,
        description="Supplier name",
        min_length=1,
        max_length=500,
    )
    material: CriticalRawMaterial = Field(
        ...,
        description="Critical raw material supplied",
    )
    country: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code of supplier",
        min_length=2,
        max_length=2,
    )
    tier: SupplierTier = Field(
        ...,
        description="Supplier tier in the supply chain",
    )
    risk_level: DueDiligenceRisk = Field(
        default=DueDiligenceRisk.MEDIUM,
        description="Assessed risk level for this supplier",
    )
    oecd_compliant: bool = Field(
        default=False,
        description="Whether supplier is compliant with OECD 5-step framework",
    )
    third_party_audited: bool = Field(
        default=False,
        description="Whether supplier has undergone third-party audit",
    )
    risk_score: float = Field(
        default=0.0,
        description="Numeric risk score (0-100, higher is riskier)",
        ge=0.0,
        le=100.0,
    )
    country_governance_score: int = Field(
        default=50,
        description="Country governance score (0-100, higher is better)",
        ge=0,
        le=100,
    )
    oecd_step_assessments: List[OECDStepAssessment] = Field(
        default_factory=list,
        description="Assessment of each OECD step for this supplier",
    )
    mitigation_actions: List[str] = Field(
        default_factory=list,
        description="Required mitigation actions for this supplier",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

    @field_validator("country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alpha-2."""
        return v.upper().strip()


class RiskSummary(BaseModel):
    """Summary of risk distribution across all assessed suppliers.

    Provides counts and percentages for each risk level.
    """
    very_high_count: int = Field(
        default=0,
        description="Number of VERY_HIGH risk suppliers",
    )
    high_count: int = Field(
        default=0,
        description="Number of HIGH risk suppliers",
    )
    medium_count: int = Field(
        default=0,
        description="Number of MEDIUM risk suppliers",
    )
    low_count: int = Field(
        default=0,
        description="Number of LOW risk suppliers",
    )
    negligible_count: int = Field(
        default=0,
        description="Number of NEGLIGIBLE risk suppliers",
    )
    very_high_pct: float = Field(
        default=0.0,
        description="Percentage of VERY_HIGH risk suppliers",
    )
    high_pct: float = Field(
        default=0.0,
        description="Percentage of HIGH risk suppliers",
    )
    medium_pct: float = Field(
        default=0.0,
        description="Percentage of MEDIUM risk suppliers",
    )
    low_pct: float = Field(
        default=0.0,
        description="Percentage of LOW risk suppliers",
    )
    negligible_pct: float = Field(
        default=0.0,
        description="Percentage of NEGLIGIBLE risk suppliers",
    )
    materials_at_risk: List[str] = Field(
        default_factory=list,
        description="Materials with HIGH or VERY_HIGH risk suppliers",
    )
    countries_at_risk: List[str] = Field(
        default_factory=list,
        description="Countries with HIGH or VERY_HIGH risk suppliers",
    )


class DDResult(BaseModel):
    """Result of a complete supply chain due diligence assessment.

    Contains the full inventory of assessed suppliers, risk
    summaries, compliance rates, and actionable recommendations
    per Article 48 of the EU Battery Regulation.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC)",
    )
    suppliers_assessed: int = Field(
        default=0,
        description="Total number of suppliers assessed",
    )
    high_risk_count: int = Field(
        default=0,
        description="Number of HIGH or VERY_HIGH risk suppliers",
    )
    oecd_compliance_rate: float = Field(
        default=0.0,
        description="Percentage of suppliers compliant with OECD 5-step (0-100)",
    )
    audit_coverage_rate: float = Field(
        default=0.0,
        description="Percentage of suppliers with third-party audit (0-100)",
    )
    risk_summary: RiskSummary = Field(
        default_factory=RiskSummary,
        description="Summary of risk distribution across all suppliers",
    )
    mitigation_required: bool = Field(
        default=False,
        description="Whether mitigation actions are required",
    )
    supplier_assessments: List[SupplierAssessment] = Field(
        default_factory=list,
        description="Individual supplier assessment results",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for risk mitigation",
    )
    materials_assessed: List[str] = Field(
        default_factory=list,
        description="List of materials covered in this assessment",
    )
    overall_risk_level: DueDiligenceRisk = Field(
        default=DueDiligenceRisk.MEDIUM,
        description="Overall supply chain risk level",
    )
    average_risk_score: float = Field(
        default=0.0,
        description="Average risk score across all suppliers (0-100)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the entire result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SupplyChainDDEngine:
    """Supply chain due diligence engine per EU Battery Regulation Art 48.

    Provides deterministic, zero-hallucination assessment of:
    - Supplier risk scoring based on country, tier, audit, OECD compliance
    - OECD five-step framework compliance checks
    - Audit coverage calculation
    - High-risk supplier identification and flagging
    - Risk-based mitigation recommendations
    - Country governance scoring

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = SupplyChainDDEngine()
        suppliers = [
            SupplierAssessment(
                supplier_id="SUP-001",
                name="CobaltMine Corp",
                material=CriticalRawMaterial.COBALT,
                country="CD",
                tier=SupplierTier.TIER_3,
                oecd_compliant=False,
                third_party_audited=False,
            ),
        ]
        result = engine.assess_supply_chain(suppliers)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise SupplyChainDDEngine."""
        self._assessments: List[SupplierAssessment] = []
        logger.info(
            "SupplyChainDDEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Full Supply Chain Assessment                                         #
    # ------------------------------------------------------------------ #

    def assess_supply_chain(
        self,
        suppliers: List[SupplierAssessment],
    ) -> DDResult:
        """Perform a full supply chain due diligence assessment.

        Iterates through all suppliers, calculates risk scores,
        checks OECD compliance, and generates an overall assessment
        with recommendations per Article 48.

        Args:
            suppliers: List of SupplierAssessment objects to assess.

        Returns:
            DDResult with complete assessment including risk summary,
            compliance rates, and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Assessing supply chain due diligence for %d suppliers",
            len(suppliers),
        )

        assessed_suppliers: List[SupplierAssessment] = []

        for supplier in suppliers:
            assessed = self.assess_supplier_risk(supplier)
            assessed = self.check_oecd_compliance(assessed)
            assessed.provenance_hash = _compute_hash(assessed)
            assessed_suppliers.append(assessed)

        # Calculate aggregate metrics
        total = len(assessed_suppliers)
        high_risk_suppliers = self.identify_high_risk(assessed_suppliers)
        high_risk_count = len(high_risk_suppliers)

        oecd_rate = self._calculate_oecd_compliance_rate(assessed_suppliers)
        audit_rate = self.calculate_audit_coverage(assessed_suppliers)

        risk_summary = self._build_risk_summary(assessed_suppliers)
        materials_assessed = list(set(
            s.material.value for s in assessed_suppliers
        ))

        avg_risk = self._calculate_average_risk_score(assessed_suppliers)
        overall_risk = self._determine_overall_risk(avg_risk)

        mitigation_required = high_risk_count > 0
        recommendations = self._generate_recommendations(
            assessed_suppliers, risk_summary, oecd_rate, audit_rate
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = DDResult(
            suppliers_assessed=total,
            high_risk_count=high_risk_count,
            oecd_compliance_rate=oecd_rate,
            audit_coverage_rate=audit_rate,
            risk_summary=risk_summary,
            mitigation_required=mitigation_required,
            supplier_assessments=assessed_suppliers,
            recommendations=recommendations,
            materials_assessed=sorted(materials_assessed),
            overall_risk_level=overall_risk,
            average_risk_score=avg_risk,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)
        self._assessments = assessed_suppliers

        logger.info(
            "Supply chain DD complete: %d suppliers, %d high-risk, "
            "OECD=%.1f%%, audit=%.1f%% in %.3f ms",
            total,
            high_risk_count,
            oecd_rate,
            audit_rate,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Individual Supplier Risk Assessment                                  #
    # ------------------------------------------------------------------ #

    def assess_supplier_risk(
        self, supplier: SupplierAssessment
    ) -> SupplierAssessment:
        """Assess the risk level of a single supplier.

        Calculates a composite risk score based on:
        - Country risk (30% weight): governance score and high-risk lists
        - Tier risk (15% weight): upstream depth of the supplier
        - OECD compliance (25% weight): adherence to 5-step framework
        - Audit status (20% weight): third-party audit completion
        - Material criticality (10% weight): supply concentration risk

        Args:
            supplier: SupplierAssessment to evaluate.

        Returns:
            Updated SupplierAssessment with risk_score and risk_level.
        """
        t0 = time.perf_counter()

        # Country risk component (0-100, higher is riskier)
        country_risk = self._calculate_country_risk(
            supplier.country, supplier.material.value
        )

        # Tier risk component (0-100)
        tier_risk = self._calculate_tier_risk(supplier.tier)

        # OECD compliance component (0-100, higher is riskier if not compliant)
        oecd_risk = 0.0 if supplier.oecd_compliant else 80.0

        # Audit status component (0-100, higher is riskier if not audited)
        audit_risk = 0.0 if supplier.third_party_audited else 70.0

        # Material criticality component (0-100)
        material_risk = float(
            MATERIAL_CRITICALITY.get(supplier.material.value, 50)
        )

        # Weighted composite score
        risk_score = (
            country_risk * RISK_WEIGHTS["country_risk"]
            + tier_risk * RISK_WEIGHTS["tier_risk"]
            + oecd_risk * RISK_WEIGHTS["oecd_compliance"]
            + audit_risk * RISK_WEIGHTS["audit_status"]
            + material_risk * RISK_WEIGHTS["material_criticality"]
        )

        risk_score = _round2(risk_score)
        risk_level = self._score_to_risk_level(risk_score)

        # Look up governance score
        governance = COUNTRY_GOVERNANCE_SCORES.get(
            supplier.country.upper(), 50
        )

        # Determine required mitigations
        mitigations = self._determine_mitigations(
            supplier, risk_level, country_risk, oecd_risk, audit_risk
        )

        supplier.risk_score = risk_score
        supplier.risk_level = risk_level
        supplier.country_governance_score = governance
        supplier.mitigation_actions = mitigations

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
        logger.debug(
            "Assessed supplier %s: score=%.2f, level=%s in %.3f ms",
            supplier.supplier_id,
            risk_score,
            risk_level.value,
            elapsed_ms,
        )
        return supplier

    # ------------------------------------------------------------------ #
    # OECD Five-Step Compliance Check                                      #
    # ------------------------------------------------------------------ #

    def check_oecd_compliance(
        self, supplier: SupplierAssessment
    ) -> SupplierAssessment:
        """Check supplier compliance with the OECD five-step framework.

        Evaluates each of the five OECD steps and generates a
        per-step assessment with identified gaps.

        Args:
            supplier: SupplierAssessment to evaluate.

        Returns:
            Updated SupplierAssessment with oecd_step_assessments populated.
        """
        step_assessments: List[OECDStepAssessment] = []

        # Step 1: Management Systems
        step1 = self._assess_oecd_step_1(supplier)
        step_assessments.append(step1)

        # Step 2: Risk Identification
        step2 = self._assess_oecd_step_2(supplier)
        step_assessments.append(step2)

        # Step 3: Risk Response
        step3 = self._assess_oecd_step_3(supplier)
        step_assessments.append(step3)

        # Step 4: Third-Party Audit
        step4 = self._assess_oecd_step_4(supplier)
        step_assessments.append(step4)

        # Step 5: Reporting
        step5 = self._assess_oecd_step_5(supplier)
        step_assessments.append(step5)

        supplier.oecd_step_assessments = step_assessments

        # Overall OECD compliance: all 5 steps must be compliant
        all_compliant = all(s.compliant for s in step_assessments)
        supplier.oecd_compliant = all_compliant

        logger.debug(
            "OECD check for %s: %d/5 steps compliant, overall=%s",
            supplier.supplier_id,
            sum(1 for s in step_assessments if s.compliant),
            all_compliant,
        )
        return supplier

    # ------------------------------------------------------------------ #
    # Audit Coverage Calculation                                           #
    # ------------------------------------------------------------------ #

    def calculate_audit_coverage(
        self, suppliers: List[SupplierAssessment]
    ) -> float:
        """Calculate the percentage of suppliers with third-party audit.

        Args:
            suppliers: List of SupplierAssessment objects.

        Returns:
            Audit coverage percentage (0-100).
        """
        if not suppliers:
            return 0.0

        audited_count = sum(
            1 for s in suppliers if s.third_party_audited
        )
        rate = _safe_divide(
            float(audited_count), float(len(suppliers)), 0.0
        ) * 100.0
        return _round2(rate)

    def calculate_audit_coverage_by_material(
        self, suppliers: List[SupplierAssessment]
    ) -> Dict[str, float]:
        """Calculate audit coverage broken down by material.

        Args:
            suppliers: List of SupplierAssessment objects.

        Returns:
            Dict mapping material name to audit coverage percentage.
        """
        material_groups: Dict[str, List[SupplierAssessment]] = {}
        for s in suppliers:
            key = s.material.value
            if key not in material_groups:
                material_groups[key] = []
            material_groups[key].append(s)

        coverage: Dict[str, float] = {}
        for material, group in sorted(material_groups.items()):
            coverage[material] = self.calculate_audit_coverage(group)

        return coverage

    def calculate_audit_coverage_by_tier(
        self, suppliers: List[SupplierAssessment]
    ) -> Dict[str, float]:
        """Calculate audit coverage broken down by supplier tier.

        Args:
            suppliers: List of SupplierAssessment objects.

        Returns:
            Dict mapping tier name to audit coverage percentage.
        """
        tier_groups: Dict[str, List[SupplierAssessment]] = {}
        for s in suppliers:
            key = s.tier.value
            if key not in tier_groups:
                tier_groups[key] = []
            tier_groups[key].append(s)

        coverage: Dict[str, float] = {}
        for tier, group in sorted(tier_groups.items()):
            coverage[tier] = self.calculate_audit_coverage(group)

        return coverage

    # ------------------------------------------------------------------ #
    # High-Risk Identification                                             #
    # ------------------------------------------------------------------ #

    def identify_high_risk(
        self, suppliers: List[SupplierAssessment]
    ) -> List[SupplierAssessment]:
        """Identify suppliers with HIGH or VERY_HIGH risk levels.

        Flags suppliers that require immediate mitigation action
        or enhanced due diligence per Article 48.

        Args:
            suppliers: List of SupplierAssessment objects.

        Returns:
            List of high-risk SupplierAssessment objects.
        """
        high_risk = [
            s for s in suppliers
            if s.risk_level in (
                DueDiligenceRisk.HIGH,
                DueDiligenceRisk.VERY_HIGH,
            )
        ]

        if high_risk:
            logger.warning(
                "Identified %d high-risk suppliers out of %d total",
                len(high_risk),
                len(suppliers),
            )

        return high_risk

    def identify_high_risk_by_material(
        self, suppliers: List[SupplierAssessment]
    ) -> Dict[str, List[SupplierAssessment]]:
        """Identify high-risk suppliers grouped by material.

        Args:
            suppliers: List of SupplierAssessment objects.

        Returns:
            Dict mapping material name to list of high-risk suppliers.
        """
        high_risk = self.identify_high_risk(suppliers)
        grouped: Dict[str, List[SupplierAssessment]] = {}

        for s in high_risk:
            key = s.material.value
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(s)

        return grouped

    # ------------------------------------------------------------------ #
    # Supplier Summary and Reporting                                       #
    # ------------------------------------------------------------------ #

    def get_supplier_summary(
        self, supplier: SupplierAssessment
    ) -> Dict[str, Any]:
        """Return a structured summary of a single supplier assessment.

        Args:
            supplier: SupplierAssessment to summarise.

        Returns:
            Dict with supplier assessment details.
        """
        oecd_steps_compliant = sum(
            1 for s in supplier.oecd_step_assessments if s.compliant
        )
        oecd_total = len(supplier.oecd_step_assessments)

        return {
            "supplier_id": supplier.supplier_id,
            "name": supplier.name,
            "material": supplier.material.value,
            "country": supplier.country,
            "tier": supplier.tier.value,
            "risk_level": supplier.risk_level.value,
            "risk_score": supplier.risk_score,
            "country_governance_score": supplier.country_governance_score,
            "oecd_compliant": supplier.oecd_compliant,
            "oecd_steps_compliant": oecd_steps_compliant,
            "oecd_steps_total": oecd_total,
            "third_party_audited": supplier.third_party_audited,
            "mitigation_actions_count": len(supplier.mitigation_actions),
            "mitigation_actions": supplier.mitigation_actions,
            "provenance_hash": supplier.provenance_hash,
        }

    def get_material_risk_profile(
        self, suppliers: List[SupplierAssessment]
    ) -> Dict[str, Any]:
        """Get a risk profile grouped by critical raw material.

        Args:
            suppliers: List of assessed suppliers.

        Returns:
            Dict with per-material risk statistics.
        """
        profile: Dict[str, Any] = {}

        material_groups: Dict[str, List[SupplierAssessment]] = {}
        for s in suppliers:
            key = s.material.value
            if key not in material_groups:
                material_groups[key] = []
            material_groups[key].append(s)

        for material, group in sorted(material_groups.items()):
            high_risk = [
                s for s in group
                if s.risk_level in (
                    DueDiligenceRisk.HIGH, DueDiligenceRisk.VERY_HIGH
                )
            ]
            avg_score = _round2(
                _safe_divide(
                    sum(s.risk_score for s in group),
                    float(len(group)),
                    0.0,
                )
            )
            oecd_rate = _round2(
                _safe_divide(
                    sum(1 for s in group if s.oecd_compliant),
                    float(len(group)),
                    0.0,
                ) * 100.0
            )

            profile[material] = {
                "supplier_count": len(group),
                "high_risk_count": len(high_risk),
                "average_risk_score": avg_score,
                "oecd_compliance_rate": oecd_rate,
                "countries": sorted(set(s.country for s in group)),
                "criticality_score": MATERIAL_CRITICALITY.get(material, 50),
            }

        profile["provenance_hash"] = _compute_hash(profile)
        return profile

    def get_country_risk_assessment(
        self, country: str, material: str
    ) -> Dict[str, Any]:
        """Get the risk assessment for a specific country/material pair.

        Args:
            country: ISO 3166-1 alpha-2 country code.
            material: Critical raw material name.

        Returns:
            Dict with country risk details.
        """
        country = country.upper().strip()
        governance = COUNTRY_GOVERNANCE_SCORES.get(country, 50)
        is_high_risk = country in HIGH_RISK_COUNTRIES.get(material, [])
        is_elevated = country in ELEVATED_RISK_COUNTRIES.get(material, [])

        if is_high_risk:
            category = "HIGH_RISK"
        elif is_elevated:
            category = "ELEVATED_RISK"
        else:
            category = "STANDARD"

        country_risk_score = self._calculate_country_risk(country, material)

        return {
            "country": country,
            "material": material,
            "governance_score": governance,
            "risk_category": category,
            "risk_score": _round2(country_risk_score),
            "is_high_risk": is_high_risk,
            "is_elevated_risk": is_elevated,
            "is_conflict_affected": country in ["CD", "MM", "CG"],
            "provenance_hash": _compute_hash({
                "country": country,
                "material": material,
                "governance": governance,
                "risk_score": country_risk_score,
            }),
        }

    def get_high_risk_countries(
        self, material: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Return the high-risk country lists.

        Args:
            material: Optional material to filter for.

        Returns:
            Dict mapping material to high-risk country codes.
        """
        if material:
            countries = HIGH_RISK_COUNTRIES.get(material, [])
            return {material: countries}
        return dict(HIGH_RISK_COUNTRIES)

    def clear_assessments(self) -> None:
        """Clear all stored supplier assessments."""
        self._assessments.clear()
        logger.info("SupplyChainDDEngine assessments cleared")

    # ------------------------------------------------------------------ #
    # Private: Country Risk Calculation                                    #
    # ------------------------------------------------------------------ #

    def _calculate_country_risk(
        self, country: str, material: str
    ) -> float:
        """Calculate country risk score for a given material.

        The score is derived from governance data and high-risk
        country lists.  Higher score means higher risk.

        Args:
            country: ISO 3166-1 alpha-2 code.
            material: Critical raw material.

        Returns:
            Country risk score (0-100).
        """
        governance = COUNTRY_GOVERNANCE_SCORES.get(country.upper(), 50)

        # Invert governance so that low governance = high risk
        base_risk = float(100 - governance)

        # High-risk country penalty
        if country.upper() in HIGH_RISK_COUNTRIES.get(material, []):
            base_risk = min(100.0, base_risk + 25.0)

        # Elevated risk penalty (smaller)
        elif country.upper() in ELEVATED_RISK_COUNTRIES.get(material, []):
            base_risk = min(100.0, base_risk + 10.0)

        # Conflict-affected area extreme penalty
        if country.upper() in ("CD", "MM", "CG"):
            base_risk = min(100.0, base_risk + 10.0)

        return _round2(min(100.0, max(0.0, base_risk)))

    # ------------------------------------------------------------------ #
    # Private: Tier Risk Calculation                                       #
    # ------------------------------------------------------------------ #

    def _calculate_tier_risk(self, tier: SupplierTier) -> float:
        """Calculate tier risk score.

        Higher tiers (further upstream) carry more risk because
        visibility and control decrease with distance.

        Args:
            tier: SupplierTier enum value.

        Returns:
            Tier risk score (0-100).
        """
        tier_scores: Dict[str, float] = {
            SupplierTier.TIER_1.value: 20.0,
            SupplierTier.TIER_2.value: 45.0,
            SupplierTier.TIER_3.value: 70.0,
            SupplierTier.TIER_4.value: 90.0,
        }
        return tier_scores.get(tier.value, 50.0)

    # ------------------------------------------------------------------ #
    # Private: Risk Score to Level Mapping                                 #
    # ------------------------------------------------------------------ #

    def _score_to_risk_level(self, score: float) -> DueDiligenceRisk:
        """Map a numeric risk score to a DueDiligenceRisk level.

        Thresholds:
            >= 80: VERY_HIGH
            >= 60: HIGH
            >= 40: MEDIUM
            >= 20: LOW
            < 20: NEGLIGIBLE

        Args:
            score: Risk score (0-100).

        Returns:
            DueDiligenceRisk enum value.
        """
        if score >= 80.0:
            return DueDiligenceRisk.VERY_HIGH
        if score >= 60.0:
            return DueDiligenceRisk.HIGH
        if score >= 40.0:
            return DueDiligenceRisk.MEDIUM
        if score >= 20.0:
            return DueDiligenceRisk.LOW
        return DueDiligenceRisk.NEGLIGIBLE

    # ------------------------------------------------------------------ #
    # Private: OECD Step Assessments                                       #
    # ------------------------------------------------------------------ #

    def _assess_oecd_step_1(
        self, supplier: SupplierAssessment
    ) -> OECDStepAssessment:
        """Assess OECD Step 1: Management Systems.

        A supplier is considered compliant with Step 1 if they have
        demonstrated management commitment through either OECD self-
        declaration or third-party audit evidence.

        Args:
            supplier: Supplier to assess.

        Returns:
            OECDStepAssessment for Step 1.
        """
        # Step 1 requires a due diligence policy and management structures.
        # Proxy: supplier self-declares OECD compliance or has documentation.
        compliant = supplier.oecd_compliant
        gaps: List[str] = []

        if not compliant:
            gaps.append("No supply chain due diligence policy documented")
            gaps.append("Management commitment to responsible sourcing not evidenced")

        return OECDStepAssessment(
            step=OECDStep.STEP_1.value,
            step_name="Management Systems",
            description=OECD_STEP_DESCRIPTIONS[OECDStep.STEP_1.value],
            compliant=compliant,
            evidence=(
                "Supplier has documented due diligence policy and management systems"
                if compliant else ""
            ),
            gaps=gaps,
        )

    def _assess_oecd_step_2(
        self, supplier: SupplierAssessment
    ) -> OECDStepAssessment:
        """Assess OECD Step 2: Risk Identification.

        Compliance requires the supplier to have mapped their supply
        chain and identified risks.  High-risk country presence
        without mapping is a gap.

        Args:
            supplier: Supplier to assess.

        Returns:
            OECDStepAssessment for Step 2.
        """
        material = supplier.material.value
        is_high_risk_country = (
            supplier.country.upper()
            in HIGH_RISK_COUNTRIES.get(material, [])
        )

        # Compliant if OECD-compliant AND not in high-risk without mitigation
        compliant = supplier.oecd_compliant
        gaps: List[str] = []

        if is_high_risk_country and not supplier.oecd_compliant:
            compliant = False
            gaps.append(
                f"Supplier in high-risk country ({supplier.country}) "
                f"for {material} without documented risk assessment"
            )

        if not supplier.oecd_compliant:
            gaps.append("Supply chain mapping not evidenced")
            gaps.append("Risk identification and assessment not documented")

        return OECDStepAssessment(
            step=OECDStep.STEP_2.value,
            step_name="Risk Identification",
            description=OECD_STEP_DESCRIPTIONS[OECDStep.STEP_2.value],
            compliant=compliant,
            evidence=(
                "Supply chain risks identified and assessed"
                if compliant else ""
            ),
            gaps=gaps,
        )

    def _assess_oecd_step_3(
        self, supplier: SupplierAssessment
    ) -> OECDStepAssessment:
        """Assess OECD Step 3: Risk Response Strategy.

        Compliance requires documented risk mitigation plans.

        Args:
            supplier: Supplier to assess.

        Returns:
            OECDStepAssessment for Step 3.
        """
        compliant = supplier.oecd_compliant
        gaps: List[str] = []

        if not compliant:
            gaps.append("No risk mitigation strategy documented")
            gaps.append("Engagement plan with upstream suppliers not evidenced")

        material = supplier.material.value
        is_high_risk_country = (
            supplier.country.upper()
            in HIGH_RISK_COUNTRIES.get(material, [])
        )

        if is_high_risk_country and not compliant:
            gaps.append(
                "Enhanced due diligence measures required for high-risk "
                "country but not implemented"
            )

        return OECDStepAssessment(
            step=OECDStep.STEP_3.value,
            step_name="Risk Response",
            description=OECD_STEP_DESCRIPTIONS[OECDStep.STEP_3.value],
            compliant=compliant,
            evidence=(
                "Risk response strategy implemented and documented"
                if compliant else ""
            ),
            gaps=gaps,
        )

    def _assess_oecd_step_4(
        self, supplier: SupplierAssessment
    ) -> OECDStepAssessment:
        """Assess OECD Step 4: Third-Party Audit.

        Compliance requires completion of an independent third-party
        audit of supply chain due diligence practices.

        Args:
            supplier: Supplier to assess.

        Returns:
            OECDStepAssessment for Step 4.
        """
        compliant = supplier.third_party_audited
        gaps: List[str] = []

        if not compliant:
            gaps.append("No independent third-party audit completed")
            gaps.append(
                "Supplier should engage a recognised audit body "
                "(e.g., RMI, LBMA, RJC)"
            )

        return OECDStepAssessment(
            step=OECDStep.STEP_4.value,
            step_name="Third-Party Audit",
            description=OECD_STEP_DESCRIPTIONS[OECDStep.STEP_4.value],
            compliant=compliant,
            evidence=(
                "Independent third-party audit completed and documented"
                if compliant else ""
            ),
            gaps=gaps,
        )

    def _assess_oecd_step_5(
        self, supplier: SupplierAssessment
    ) -> OECDStepAssessment:
        """Assess OECD Step 5: Reporting.

        Compliance requires annual public reporting on due diligence.

        Args:
            supplier: Supplier to assess.

        Returns:
            OECDStepAssessment for Step 5.
        """
        # Reporting compliance requires both OECD compliance and audit
        compliant = supplier.oecd_compliant and supplier.third_party_audited
        gaps: List[str] = []

        if not supplier.oecd_compliant:
            gaps.append("Annual due diligence report not published")

        if not supplier.third_party_audited:
            gaps.append("Audit findings not available for public reporting")

        return OECDStepAssessment(
            step=OECDStep.STEP_5.value,
            step_name="Reporting",
            description=OECD_STEP_DESCRIPTIONS[OECDStep.STEP_5.value],
            compliant=compliant,
            evidence=(
                "Annual due diligence report published with audit findings"
                if compliant else ""
            ),
            gaps=gaps,
        )

    # ------------------------------------------------------------------ #
    # Private: Risk Summary                                                #
    # ------------------------------------------------------------------ #

    def _build_risk_summary(
        self, suppliers: List[SupplierAssessment]
    ) -> RiskSummary:
        """Build a risk distribution summary across all suppliers.

        Args:
            suppliers: List of assessed suppliers.

        Returns:
            RiskSummary with counts and percentages.
        """
        total = len(suppliers)
        if total == 0:
            return RiskSummary()

        counts: Dict[str, int] = {
            DueDiligenceRisk.VERY_HIGH.value: 0,
            DueDiligenceRisk.HIGH.value: 0,
            DueDiligenceRisk.MEDIUM.value: 0,
            DueDiligenceRisk.LOW.value: 0,
            DueDiligenceRisk.NEGLIGIBLE.value: 0,
        }

        for s in suppliers:
            counts[s.risk_level.value] = counts.get(
                s.risk_level.value, 0
            ) + 1

        # Identify materials and countries at risk
        materials_at_risk = sorted(set(
            s.material.value for s in suppliers
            if s.risk_level in (
                DueDiligenceRisk.HIGH, DueDiligenceRisk.VERY_HIGH
            )
        ))
        countries_at_risk = sorted(set(
            s.country for s in suppliers
            if s.risk_level in (
                DueDiligenceRisk.HIGH, DueDiligenceRisk.VERY_HIGH
            )
        ))

        return RiskSummary(
            very_high_count=counts[DueDiligenceRisk.VERY_HIGH.value],
            high_count=counts[DueDiligenceRisk.HIGH.value],
            medium_count=counts[DueDiligenceRisk.MEDIUM.value],
            low_count=counts[DueDiligenceRisk.LOW.value],
            negligible_count=counts[DueDiligenceRisk.NEGLIGIBLE.value],
            very_high_pct=_round2(
                _safe_divide(
                    float(counts[DueDiligenceRisk.VERY_HIGH.value]),
                    float(total),
                ) * 100.0
            ),
            high_pct=_round2(
                _safe_divide(
                    float(counts[DueDiligenceRisk.HIGH.value]),
                    float(total),
                ) * 100.0
            ),
            medium_pct=_round2(
                _safe_divide(
                    float(counts[DueDiligenceRisk.MEDIUM.value]),
                    float(total),
                ) * 100.0
            ),
            low_pct=_round2(
                _safe_divide(
                    float(counts[DueDiligenceRisk.LOW.value]),
                    float(total),
                ) * 100.0
            ),
            negligible_pct=_round2(
                _safe_divide(
                    float(counts[DueDiligenceRisk.NEGLIGIBLE.value]),
                    float(total),
                ) * 100.0
            ),
            materials_at_risk=materials_at_risk,
            countries_at_risk=countries_at_risk,
        )

    # ------------------------------------------------------------------ #
    # Private: OECD Compliance Rate                                        #
    # ------------------------------------------------------------------ #

    def _calculate_oecd_compliance_rate(
        self, suppliers: List[SupplierAssessment]
    ) -> float:
        """Calculate percentage of suppliers compliant with OECD 5-step.

        Args:
            suppliers: List of assessed suppliers.

        Returns:
            OECD compliance rate (0-100).
        """
        if not suppliers:
            return 0.0

        compliant_count = sum(
            1 for s in suppliers if s.oecd_compliant
        )
        rate = _safe_divide(
            float(compliant_count), float(len(suppliers)), 0.0
        ) * 100.0
        return _round2(rate)

    # ------------------------------------------------------------------ #
    # Private: Average Risk Score                                          #
    # ------------------------------------------------------------------ #

    def _calculate_average_risk_score(
        self, suppliers: List[SupplierAssessment]
    ) -> float:
        """Calculate the average risk score across all suppliers.

        Args:
            suppliers: List of assessed suppliers.

        Returns:
            Average risk score (0-100).
        """
        if not suppliers:
            return 0.0

        total_score = sum(s.risk_score for s in suppliers)
        return _round2(
            _safe_divide(total_score, float(len(suppliers)), 0.0)
        )

    # ------------------------------------------------------------------ #
    # Private: Overall Risk Determination                                  #
    # ------------------------------------------------------------------ #

    def _determine_overall_risk(
        self, average_score: float
    ) -> DueDiligenceRisk:
        """Determine overall supply chain risk level from average score.

        Args:
            average_score: Average risk score across all suppliers.

        Returns:
            DueDiligenceRisk enum value for the overall supply chain.
        """
        return self._score_to_risk_level(average_score)

    # ------------------------------------------------------------------ #
    # Private: Mitigation Actions                                          #
    # ------------------------------------------------------------------ #

    def _determine_mitigations(
        self,
        supplier: SupplierAssessment,
        risk_level: DueDiligenceRisk,
        country_risk: float,
        oecd_risk: float,
        audit_risk: float,
    ) -> List[str]:
        """Determine required mitigation actions for a supplier.

        Args:
            supplier: Supplier under assessment.
            risk_level: Assessed risk level.
            country_risk: Country risk component score.
            oecd_risk: OECD compliance risk component score.
            audit_risk: Audit status risk component score.

        Returns:
            List of required mitigation action descriptions.
        """
        actions: List[str] = []

        if risk_level == DueDiligenceRisk.VERY_HIGH:
            actions.append(
                "URGENT: Initiate immediate enhanced due diligence for "
                "this supplier per Art 48 requirements"
            )
            actions.append(
                "Consider temporary suspension of sourcing pending "
                "risk mitigation or alternative supplier qualification"
            )

        if risk_level in (
            DueDiligenceRisk.HIGH, DueDiligenceRisk.VERY_HIGH
        ):
            actions.append(
                "Conduct on-site assessment or commission independent "
                "third-party audit within 90 days"
            )

        if country_risk >= 70.0:
            actions.append(
                f"Country risk is elevated for {supplier.country} - "
                "implement enhanced monitoring of this supply chain node"
            )

        if oecd_risk > 0:
            actions.append(
                "Require supplier to implement OECD five-step due "
                "diligence framework and provide evidence of compliance"
            )

        if audit_risk > 0:
            actions.append(
                "Schedule third-party audit of supplier's sourcing "
                "practices by a recognised audit body (RMI, LBMA, etc.)"
            )

        if supplier.tier in (SupplierTier.TIER_3, SupplierTier.TIER_4):
            actions.append(
                f"Increase visibility into {supplier.tier.value} supply "
                "chain through direct engagement or traceability programme"
            )

        material = supplier.material.value
        is_high_risk_country = (
            supplier.country.upper()
            in HIGH_RISK_COUNTRIES.get(material, [])
        )
        if is_high_risk_country:
            actions.append(
                f"Source of {material} from high-risk country "
                f"({supplier.country}) requires conflict mineral "
                "due diligence per Regulation (EU) 2017/821"
            )

        return actions

    # ------------------------------------------------------------------ #
    # Private: Recommendations                                             #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        suppliers: List[SupplierAssessment],
        risk_summary: RiskSummary,
        oecd_rate: float,
        audit_rate: float,
    ) -> List[str]:
        """Generate actionable recommendations based on assessment.

        Args:
            suppliers: List of assessed suppliers.
            risk_summary: Risk distribution summary.
            oecd_rate: OECD compliance rate.
            audit_rate: Audit coverage rate.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # High/very-high risk recommendations
        total_high = (
            risk_summary.very_high_count + risk_summary.high_count
        )
        if total_high > 0:
            recommendations.append(
                f"{total_high} supplier(s) rated HIGH or VERY_HIGH risk. "
                "Prioritise enhanced due diligence and risk mitigation "
                "for these suppliers before placing batteries on the EU market."
            )

        # OECD compliance gap
        if oecd_rate < 100.0:
            non_compliant = sum(
                1 for s in suppliers if not s.oecd_compliant
            )
            recommendations.append(
                f"OECD compliance rate is {oecd_rate}%. "
                f"{non_compliant} supplier(s) are not compliant with the "
                "OECD five-step framework. Engage with these suppliers "
                "to establish management systems and risk assessments."
            )

        if oecd_rate < 50.0:
            recommendations.append(
                "CRITICAL: Less than 50% OECD compliance. This may "
                "constitute a barrier to EU market access under Art 48. "
                "Immediate action required."
            )

        # Audit coverage gap
        if audit_rate < 100.0:
            non_audited = sum(
                1 for s in suppliers if not s.third_party_audited
            )
            recommendations.append(
                f"Audit coverage is {audit_rate}%. "
                f"{non_audited} supplier(s) have not undergone third-party "
                "audit. Plan audit programme covering all material suppliers."
            )

        if audit_rate < 50.0:
            recommendations.append(
                "CRITICAL: Less than 50% audit coverage. Prioritise "
                "auditing high-risk and Tier 3/4 suppliers first."
            )

        # Material-specific recommendations
        if risk_summary.materials_at_risk:
            materials_str = ", ".join(risk_summary.materials_at_risk)
            recommendations.append(
                f"Materials with elevated risk: {materials_str}. "
                "Consider diversifying sourcing or establishing "
                "certified responsible sourcing programmes."
            )

        # Country-specific recommendations
        if risk_summary.countries_at_risk:
            countries_str = ", ".join(risk_summary.countries_at_risk)
            recommendations.append(
                f"High-risk sourcing countries: {countries_str}. "
                "Enhanced due diligence and traceability measures "
                "are required for these supply chain nodes."
            )

        # Tier-specific recommendations
        deep_tier_count = sum(
            1 for s in suppliers
            if s.tier in (SupplierTier.TIER_3, SupplierTier.TIER_4)
        )
        if deep_tier_count > 0:
            recommendations.append(
                f"{deep_tier_count} supplier(s) are at Tier 3 or Tier 4. "
                "Implement supply chain traceability programmes to "
                "increase visibility into upstream sourcing."
            )

        # Positive feedback
        if total_high == 0 and oecd_rate == 100.0 and audit_rate == 100.0:
            recommendations.append(
                "All suppliers are compliant with OECD due diligence "
                "requirements and have been independently audited. "
                "Supply chain due diligence obligations under Art 48 "
                "are met."
            )

        return recommendations
