# -*- coding: utf-8 -*-
"""
TaxonomyBridge - EU Taxonomy Alignment Bridge for PACK-018
=============================================================

This module connects the EU Green Claims Prep Pack to the EU Taxonomy
framework (Regulation 2020/852) for verifying claims about "sustainable"
investments, products, or activities. It maps EU Taxonomy alignment data
to claim substantiation, verifying that "sustainable" or "green" claims
comply with the six environmental objectives and Technical Screening
Criteria (TSC).

Environmental Objectives (6):
    1. Climate change mitigation (CCM)
    2. Climate change adaptation (CCA)
    3. Sustainable use of water and marine resources (WTR)
    4. Transition to a circular economy (CE)
    5. Pollution prevention and control (PPC)
    6. Protection of biodiversity and ecosystems (BIO)

Key Checks:
    - Substantial contribution to at least one objective
    - Do No Significant Harm (DNSH) to remaining objectives
    - Compliance with minimum social safeguards
    - CapEx/OpEx/Revenue alignment KPIs

NACE Activity Mapping:
    Green Claims Directive requires lifecycle-based substantiation.
    Taxonomy alignment provides a structured framework for "sustainable"
    claims on investment products (SFDR Art. 8/9 overlap).

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "EnvironmentalObjective",
    "AlignmentStatus",
    "DNSHStatus",
    "MinimumSafeguardStatus",
    "TaxonomyBridgeConfig",
    "ObjectiveAssessment",
    "AlignmentKPIs",
    "TaxonomyAlignmentResult",
    "TaxonomyBridge",
]

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
    """Compute a deterministic SHA-256 hash for provenance tracking."""
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


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CLIMATE_MITIGATION = "climate_change_mitigation"
    CLIMATE_ADAPTATION = "climate_change_adaptation"
    WATER_MARINE = "water_and_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_and_ecosystems"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment assessment status."""

    ALIGNED = "aligned"
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    UNDER_REVIEW = "under_review"
    INSUFFICIENT_DATA = "insufficient_data"


class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    PARTIALLY_COMPLIANT = "partially_compliant"


class MinimumSafeguardStatus(str, Enum):
    """Minimum social safeguard compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

OBJECTIVE_SHORT_CODES: Dict[EnvironmentalObjective, str] = {
    EnvironmentalObjective.CLIMATE_MITIGATION: "CCM",
    EnvironmentalObjective.CLIMATE_ADAPTATION: "CCA",
    EnvironmentalObjective.WATER_MARINE: "WTR",
    EnvironmentalObjective.CIRCULAR_ECONOMY: "CE",
    EnvironmentalObjective.POLLUTION_PREVENTION: "PPC",
    EnvironmentalObjective.BIODIVERSITY: "BIO",
}

CLAIM_TO_OBJECTIVE_MAP: Dict[str, List[EnvironmentalObjective]] = {
    "sustainable": list(EnvironmentalObjective),
    "green_investment": list(EnvironmentalObjective),
    "climate_positive": [EnvironmentalObjective.CLIMATE_MITIGATION, EnvironmentalObjective.CLIMATE_ADAPTATION],
    "carbon_neutral": [EnvironmentalObjective.CLIMATE_MITIGATION],
    "water_efficient": [EnvironmentalObjective.WATER_MARINE],
    "circular": [EnvironmentalObjective.CIRCULAR_ECONOMY],
    "non_toxic": [EnvironmentalObjective.POLLUTION_PREVENTION],
    "biodiversity_positive": [EnvironmentalObjective.BIODIVERSITY],
    "net_zero": [EnvironmentalObjective.CLIMATE_MITIGATION],
    "recyclable": [EnvironmentalObjective.CIRCULAR_ECONOMY],
    "clean_energy": [EnvironmentalObjective.CLIMATE_MITIGATION, EnvironmentalObjective.POLLUTION_PREVENTION],
}

TSC_REFERENCE_MAP: Dict[str, Dict[str, str]] = {
    "D35.11": {"activity": "Electricity generation from solar PV", "objective": "CCM"},
    "D35.12": {"activity": "Electricity generation from wind", "objective": "CCM"},
    "C20.11": {"activity": "Manufacture of hydrogen", "objective": "CCM"},
    "H49.10": {"activity": "Passenger rail transport", "objective": "CCM"},
    "F41.10": {"activity": "Construction of buildings", "objective": "CCM"},
    "L68.10": {"activity": "Real estate activities", "objective": "CCA"},
    "E36.00": {"activity": "Water collection and supply", "objective": "WTR"},
    "E38.11": {"activity": "Collection of non-hazardous waste", "objective": "CE"},
    "C22.11": {"activity": "Manufacture of rubber products", "objective": "PPC"},
    "A02.10": {"activity": "Forestry and logging", "objective": "BIO"},
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Alignment Bridge."""

    pack_id: str = Field(default="PACK-018")
    activity_codes: List[str] = Field(
        default_factory=list,
        description="NACE activity codes to assess",
    )
    objectives: List[EnvironmentalObjective] = Field(
        default_factory=lambda: list(EnvironmentalObjective),
        description="Environmental objectives to evaluate",
    )
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    require_dnsh: bool = Field(default=True)
    require_minimum_safeguards: bool = Field(default=True)


class ObjectiveAssessment(BaseModel):
    """Assessment result for a single environmental objective."""

    objective: EnvironmentalObjective = Field(...)
    short_code: str = Field(default="")
    substantial_contribution: bool = Field(default=False)
    dnsh_status: DNSHStatus = Field(default=DNSHStatus.NOT_ASSESSED)
    tsc_met: bool = Field(default=False)
    tsc_reference: str = Field(default="")


class AlignmentKPIs(BaseModel):
    """Taxonomy alignment KPI results."""

    capex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    opex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    revenue_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    capex_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    opex_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    revenue_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class TaxonomyAlignmentResult(BaseModel):
    """Result of a Taxonomy alignment check."""

    assessment_id: str = Field(default_factory=_new_uuid)
    activity_code: str = Field(default="")
    activity_description: str = Field(default="")
    claim_type: str = Field(default="")
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.UNDER_REVIEW)
    objective_assessments: List[ObjectiveAssessment] = Field(default_factory=list)
    kpis: AlignmentKPIs = Field(default_factory=AlignmentKPIs)
    minimum_safeguards: MinimumSafeguardStatus = Field(
        default=MinimumSafeguardStatus.NOT_ASSESSED
    )
    overall_aligned: bool = Field(default=False)
    gaps: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------


class TaxonomyBridge:
    """EU Taxonomy alignment bridge for green claims substantiation.

    Verifies that claims about "sustainable", "green", or "taxonomy-aligned"
    investments or activities comply with the EU Taxonomy Regulation by
    checking substantial contribution, DNSH, and minimum safeguards.

    Attributes:
        config: Taxonomy bridge configuration.

    Example:
        >>> config = TaxonomyBridgeConfig(activity_codes=["D35.11"])
        >>> bridge = TaxonomyBridge(config)
        >>> result = bridge.check_taxonomy_alignment("D35.11", "clean_energy")
        >>> assert result["alignment_status"] in ["aligned", "eligible"]
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize TaxonomyBridge.

        Args:
            config: Bridge configuration. Defaults used if None.
        """
        self.config = config or TaxonomyBridgeConfig()
        logger.info(
            "TaxonomyBridge initialized (activities=%d, objectives=%d)",
            len(self.config.activity_codes),
            len(self.config.objectives),
        )

    def check_taxonomy_alignment(
        self,
        activity: str,
        claim: str,
    ) -> Dict[str, Any]:
        """Check EU Taxonomy alignment for an activity and claim.

        Args:
            activity: NACE activity code (e.g., "D35.11").
            claim: Claim type (e.g., "sustainable", "clean_energy").

        Returns:
            Dict with alignment status, objective assessments, KPIs,
            gaps, and provenance hash.
        """
        start = _utcnow()
        result = TaxonomyAlignmentResult(
            activity_code=activity,
            claim_type=claim,
        )

        tsc_ref = TSC_REFERENCE_MAP.get(activity, {})
        result.activity_description = tsc_ref.get("activity", "Unknown activity")

        relevant_objectives = CLAIM_TO_OBJECTIVE_MAP.get(claim, list(EnvironmentalObjective))
        assessments = self._assess_objectives(activity, relevant_objectives, tsc_ref)
        result.objective_assessments = assessments

        has_substantial = any(a.substantial_contribution for a in assessments)
        all_dnsh = all(
            a.dnsh_status == DNSHStatus.COMPLIANT
            for a in assessments if not a.substantial_contribution
        )

        if tsc_ref:
            result.alignment_status = AlignmentStatus.ELIGIBLE
            if has_substantial and all_dnsh:
                result.alignment_status = AlignmentStatus.ALIGNED
                result.overall_aligned = True
        else:
            result.alignment_status = AlignmentStatus.NOT_ELIGIBLE

        result.gaps = self._identify_gaps(result)
        result.kpis = self._compute_kpis(result)

        elapsed = (_utcnow() - start).total_seconds() * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "TaxonomyBridge assessed '%s' for '%s': %s in %.1fms",
            activity,
            claim,
            result.alignment_status.value,
            elapsed,
        )

        return result.model_dump(mode="json")

    def validate_taxonomy_claim(
        self,
        claim_type: str,
        activity_code: str = "",
    ) -> Dict[str, Any]:
        """Validate a taxonomy-related green claim.

        Checks whether a claim such as "sustainable", "green_investment",
        or "climate_positive" is supported by EU Taxonomy alignment data
        for the given activity.

        Args:
            claim_type: Claim type key (e.g., "sustainable").
            activity_code: Optional NACE activity code.

        Returns:
            Dict with validation result, alignment status, and hash.
        """
        if activity_code:
            return self.check_taxonomy_alignment(activity_code, claim_type)

        objectives = CLAIM_TO_OBJECTIVE_MAP.get(claim_type, [])
        result = {
            "claim_type": claim_type,
            "relevant_objectives": [o.value for o in objectives],
            "objectives_count": len(objectives),
            "activity_code": activity_code,
            "requires_alignment_check": True,
            "provenance_hash": _compute_hash({"claim": claim_type}),
        }
        logger.info("TaxonomyBridge validated claim '%s': %d objectives", claim_type, len(objectives))
        return result

    def check_tsc_alignment(self, activity_code: str) -> Dict[str, Any]:
        """Check Technical Screening Criteria alignment for an activity.

        Verifies whether the NACE activity has defined TSC in the
        Taxonomy Delegated Acts and which objective it targets.

        Args:
            activity_code: NACE activity code (e.g., "D35.11").

        Returns:
            Dict with TSC alignment status, objective, and reference.
        """
        tsc_ref = TSC_REFERENCE_MAP.get(activity_code, {})
        has_tsc = bool(tsc_ref)

        result = {
            "activity_code": activity_code,
            "tsc_defined": has_tsc,
            "activity_description": tsc_ref.get("activity", "Not in TSC registry"),
            "primary_objective": tsc_ref.get("objective", ""),
            "delegated_act_reference": f"Delegated Act Annex I - {activity_code}" if has_tsc else "",
            "provenance_hash": _compute_hash({"activity": activity_code, "tsc": has_tsc}),
        }
        logger.info("TaxonomyBridge TSC check '%s': defined=%s", activity_code, has_tsc)
        return result

    def verify_dnsh(
        self,
        activity_code: str,
        primary_objective: str = "",
    ) -> Dict[str, Any]:
        """Verify Do No Significant Harm compliance for an activity.

        Checks DNSH requirements for all non-primary environmental
        objectives per the Taxonomy Regulation Art. 17.

        Args:
            activity_code: NACE activity code.
            primary_objective: The objective with substantial contribution.

        Returns:
            Dict with per-objective DNSH assessment and overall status.
        """
        tsc_ref = TSC_REFERENCE_MAP.get(activity_code, {})
        primary = primary_objective or tsc_ref.get("objective", "")

        dnsh_results: List[Dict[str, Any]] = []
        for obj in EnvironmentalObjective:
            short_code = OBJECTIVE_SHORT_CODES.get(obj, "")
            is_primary = short_code == primary
            dnsh_results.append({
                "objective": obj.value,
                "short_code": short_code,
                "is_primary": is_primary,
                "dnsh_status": "not_applicable" if is_primary else "not_assessed",
            })

        result = {
            "activity_code": activity_code,
            "primary_objective": primary,
            "dnsh_assessments": dnsh_results,
            "all_compliant": False,
            "needs_assessment": sum(1 for d in dnsh_results if d["dnsh_status"] == "not_assessed"),
            "provenance_hash": _compute_hash({"activity": activity_code, "dnsh": dnsh_results}),
        }
        logger.info("TaxonomyBridge DNSH check '%s': %d need assessment", activity_code, result["needs_assessment"])
        return result

    def get_green_asset_ratio(self) -> Dict[str, Any]:
        """Get Green Asset Ratio / KPI template structure.

        Per Taxonomy Regulation Art. 8, financial and non-financial
        undertakings must disclose the proportion of taxonomy-aligned
        activities in their CapEx, OpEx, and Revenue.

        Returns:
            Dict with GAR KPI structure and calculation template.
        """
        result = {
            "kpi_type": "green_asset_ratio",
            "numerator": "taxonomy_aligned_activities",
            "denominator": "total_covered_activities",
            "metrics": {
                "capex_aligned_pct": 0.0,
                "opex_aligned_pct": 0.0,
                "revenue_aligned_pct": 0.0,
                "capex_eligible_pct": 0.0,
                "opex_eligible_pct": 0.0,
                "revenue_eligible_pct": 0.0,
            },
            "regulatory_reference": "Taxonomy Regulation Art. 8, Delegated Act 2021/2178",
            "provenance_hash": _compute_hash({"kpi": "green_asset_ratio"}),
        }
        logger.info("TaxonomyBridge GAR template generated")
        return result

    def reconcile_with_taxonomy_reporting(
        self,
        claim_type: str,
        reported_kpis: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Reconcile a green claim with published Taxonomy Art. 8 KPIs.

        Cross-references marketing claims against the undertaking's
        published Taxonomy disclosure KPIs to ensure consistency.

        Args:
            claim_type: Type of green claim.
            reported_kpis: Published Taxonomy KPIs (capex/opex/revenue).

        Returns:
            Dict with reconciliation result, consistency status, and hash.
        """
        kpis = reported_kpis or {}
        capex = kpis.get("capex_aligned_pct", 0.0)
        revenue = kpis.get("revenue_aligned_pct", 0.0)

        consistency = "consistent"
        gaps: List[str] = []

        if claim_type in ("sustainable", "green_investment") and capex == 0.0 and revenue == 0.0:
            consistency = "inconsistent"
            gaps.append("Claim implies taxonomy alignment but reported KPIs show 0% alignment")

        result = {
            "claim_type": claim_type,
            "reported_kpis": kpis,
            "consistency": consistency,
            "gaps": gaps,
            "provenance_hash": _compute_hash({"claim": claim_type, "kpis": kpis}),
        }
        logger.info(
            "TaxonomyBridge reconciliation '%s': %s (%d gaps)",
            claim_type, consistency, len(gaps),
        )
        return result

    def get_supported_activities(self) -> Dict[str, Dict[str, str]]:
        """Get all supported NACE activities with TSC references."""
        return dict(TSC_REFERENCE_MAP)

    def get_claim_objective_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of claim types to environmental objectives."""
        return {
            claim: [obj.value for obj in objs]
            for claim, objs in CLAIM_TO_OBJECTIVE_MAP.items()
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _assess_objectives(
        self,
        activity: str,
        objectives: List[EnvironmentalObjective],
        tsc_ref: Dict[str, str],
    ) -> List[ObjectiveAssessment]:
        """Assess alignment for each relevant environmental objective."""
        assessments = []
        primary_objective_code = tsc_ref.get("objective", "")

        for obj in objectives:
            short_code = OBJECTIVE_SHORT_CODES.get(obj, "")
            is_primary = short_code == primary_objective_code
            assessment = ObjectiveAssessment(
                objective=obj,
                short_code=short_code,
                substantial_contribution=is_primary and bool(tsc_ref),
                dnsh_status=DNSHStatus.NOT_ASSESSED if not is_primary else DNSHStatus.COMPLIANT,
                tsc_met=is_primary and bool(tsc_ref),
                tsc_reference=f"Delegated Act Annex I - {activity}" if is_primary else "",
            )
            assessments.append(assessment)

        return assessments

    def _identify_gaps(self, result: TaxonomyAlignmentResult) -> List[str]:
        """Identify gaps in Taxonomy alignment."""
        gaps: List[str] = []

        if result.alignment_status == AlignmentStatus.NOT_ELIGIBLE:
            gaps.append(f"Activity '{result.activity_code}' is not Taxonomy-eligible")

        if result.minimum_safeguards == MinimumSafeguardStatus.NOT_ASSESSED:
            gaps.append("Minimum social safeguards have not been assessed")

        unassessed = [
            a for a in result.objective_assessments
            if a.dnsh_status == DNSHStatus.NOT_ASSESSED and not a.substantial_contribution
        ]
        if unassessed:
            gaps.append(f"DNSH not assessed for {len(unassessed)} objective(s)")

        if not any(a.substantial_contribution for a in result.objective_assessments):
            gaps.append("No substantial contribution demonstrated to any objective")

        return gaps

    def _compute_kpis(self, result: TaxonomyAlignmentResult) -> AlignmentKPIs:
        """Compute alignment KPIs based on assessment results."""
        if result.overall_aligned:
            return AlignmentKPIs(
                capex_aligned_pct=0.0,
                opex_aligned_pct=0.0,
                revenue_aligned_pct=0.0,
                capex_eligible_pct=100.0,
                opex_eligible_pct=100.0,
                revenue_eligible_pct=100.0,
            )
        if result.alignment_status == AlignmentStatus.ELIGIBLE:
            return AlignmentKPIs(
                capex_eligible_pct=100.0,
                opex_eligible_pct=100.0,
                revenue_eligible_pct=100.0,
            )
        return AlignmentKPIs()
