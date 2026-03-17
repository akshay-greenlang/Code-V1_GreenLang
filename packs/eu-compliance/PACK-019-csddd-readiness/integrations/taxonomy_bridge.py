# -*- coding: utf-8 -*-
"""
TaxonomyBridge - EU Taxonomy DNSH / CSDDD Environmental Impact Bridge for PACK-019
=====================================================================================

This module validates DNSH (Do No Significant Harm) criteria alignment with CSDDD
environmental adverse impacts. It maps EU Taxonomy environmental objectives to
CSDDD due diligence obligations, assesses Taxonomy eligibility and alignment for
economic activities, and cross-references DNSH criteria with CSDDD Art 6 impact
identification requirements.

Legal References:
    - Regulation (EU) 2020/852 (EU Taxonomy Regulation)
    - Delegated Regulation (EU) 2021/2139 (Climate Delegated Act)
    - Directive (EU) 2024/1760 (CSDDD), Art 6, 8 - Environmental impacts
    - CSDDD Annex Part II - Environmental adverse impacts catalogue

EU Taxonomy Environmental Objectives:
    1. Climate change mitigation
    2. Climate change adaptation
    3. Sustainable use of water and marine resources
    4. Transition to a circular economy
    5. Pollution prevention and control
    6. Protection of biodiversity and ecosystems

CSDDD-Taxonomy Intersection:
    - DNSH criteria violations may constitute CSDDD adverse environmental impacts
    - Taxonomy-aligned activities generally satisfy CSDDD Art 8 prevention
    - Minimum safeguards (UNGP, OECD, ILO) overlap with CSDDD human rights DD

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-019 CSDDD Readiness Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CLIMATE_MITIGATION = "climate_change_mitigation"
    CLIMATE_ADAPTATION = "climate_change_adaptation"
    WATER_MARINE = "water_and_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_and_ecosystems"


class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment status."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ASSESSED = "not_assessed"
    PARTIALLY_MET = "partially_met"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment assessment status."""

    ELIGIBLE = "eligible"
    ALIGNED = "aligned"
    NOT_ELIGIBLE = "not_eligible"
    UNDER_REVIEW = "under_review"


class CSDDDEnvironmentalImpact(str, Enum):
    """CSDDD Annex Part II environmental adverse impact categories."""

    GREENHOUSE_GAS = "greenhouse_gas_emissions"
    AIR_POLLUTION = "air_pollution"
    WATER_POLLUTION = "water_pollution"
    SOIL_POLLUTION = "soil_pollution"
    DEFORESTATION = "deforestation"
    BIODIVERSITY_LOSS = "biodiversity_loss"
    OVEREXPLOITATION_RESOURCES = "overexploitation_of_resources"
    WASTE_GENERATION = "waste_generation"
    ECOSYSTEM_DEGRADATION = "ecosystem_degradation"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Bridge."""

    pack_id: str = Field(default="PACK-019")
    taxonomy_version: str = Field(default="2024")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    include_minimum_safeguards: bool = Field(default=True)


class EconomicActivity(BaseModel):
    """An economic activity assessed for Taxonomy alignment."""

    activity_id: str = Field(default_factory=_new_uuid)
    nace_code: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    substantial_contribution_objective: EnvironmentalObjective = Field(
        default=EnvironmentalObjective.CLIMATE_MITIGATION
    )
    is_eligible: bool = Field(default=False)
    is_aligned: bool = Field(default=False)
    dnsh_results: Dict[str, DNSHStatus] = Field(default_factory=dict)
    minimum_safeguards_met: bool = Field(default=False)
    capex_eur: float = Field(default=0.0, ge=0.0)
    opex_eur: float = Field(default=0.0, ge=0.0)
    revenue_eur: float = Field(default=0.0, ge=0.0)


class DNSHAssessment(BaseModel):
    """DNSH assessment result for a single activity against all objectives."""

    activity_id: str = Field(default="")
    activity_name: str = Field(default="")
    sc_objective: EnvironmentalObjective = Field(
        default=EnvironmentalObjective.CLIMATE_MITIGATION
    )
    dnsh_results: Dict[str, DNSHStatus] = Field(default_factory=dict)
    all_passed: bool = Field(default=False)
    failed_objectives: List[str] = Field(default_factory=list)
    csddd_impacts: List[CSDDDEnvironmentalImpact] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EnvironmentalImpactMapping(BaseModel):
    """Mapping from CSDDD environmental impact to Taxonomy DNSH criteria."""

    impact: CSDDDEnvironmentalImpact = Field(
        default=CSDDDEnvironmentalImpact.GREENHOUSE_GAS
    )
    taxonomy_objectives: List[EnvironmentalObjective] = Field(default_factory=list)
    dnsh_relevance: str = Field(default="")
    csddd_articles: List[str] = Field(default_factory=list)


class TaxonomyAlignmentResult(BaseModel):
    """Result of Taxonomy alignment assessment."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    activities_assessed: int = Field(default=0)
    eligible_count: int = Field(default=0)
    aligned_count: int = Field(default=0)
    eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    dnsh_failures: int = Field(default=0)
    minimum_safeguards_failures: int = Field(default=0)
    csddd_relevant_impacts: List[CSDDDEnvironmentalImpact] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# CSDDD Impact to Taxonomy Objective Mapping
# ---------------------------------------------------------------------------

IMPACT_TO_OBJECTIVE_MAP: Dict[CSDDDEnvironmentalImpact, List[EnvironmentalObjective]] = {
    CSDDDEnvironmentalImpact.GREENHOUSE_GAS: [
        EnvironmentalObjective.CLIMATE_MITIGATION,
        EnvironmentalObjective.CLIMATE_ADAPTATION,
    ],
    CSDDDEnvironmentalImpact.AIR_POLLUTION: [
        EnvironmentalObjective.POLLUTION_PREVENTION,
    ],
    CSDDDEnvironmentalImpact.WATER_POLLUTION: [
        EnvironmentalObjective.WATER_MARINE,
        EnvironmentalObjective.POLLUTION_PREVENTION,
    ],
    CSDDDEnvironmentalImpact.SOIL_POLLUTION: [
        EnvironmentalObjective.POLLUTION_PREVENTION,
    ],
    CSDDDEnvironmentalImpact.DEFORESTATION: [
        EnvironmentalObjective.BIODIVERSITY,
        EnvironmentalObjective.CLIMATE_MITIGATION,
    ],
    CSDDDEnvironmentalImpact.BIODIVERSITY_LOSS: [
        EnvironmentalObjective.BIODIVERSITY,
    ],
    CSDDDEnvironmentalImpact.OVEREXPLOITATION_RESOURCES: [
        EnvironmentalObjective.CIRCULAR_ECONOMY,
        EnvironmentalObjective.WATER_MARINE,
    ],
    CSDDDEnvironmentalImpact.WASTE_GENERATION: [
        EnvironmentalObjective.CIRCULAR_ECONOMY,
        EnvironmentalObjective.POLLUTION_PREVENTION,
    ],
    CSDDDEnvironmentalImpact.ECOSYSTEM_DEGRADATION: [
        EnvironmentalObjective.BIODIVERSITY,
        EnvironmentalObjective.WATER_MARINE,
    ],
}

DNSH_FAILURE_TO_CSDDD_IMPACT: Dict[str, CSDDDEnvironmentalImpact] = {
    EnvironmentalObjective.CLIMATE_MITIGATION.value: CSDDDEnvironmentalImpact.GREENHOUSE_GAS,
    EnvironmentalObjective.CLIMATE_ADAPTATION.value: CSDDDEnvironmentalImpact.GREENHOUSE_GAS,
    EnvironmentalObjective.WATER_MARINE.value: CSDDDEnvironmentalImpact.WATER_POLLUTION,
    EnvironmentalObjective.CIRCULAR_ECONOMY.value: CSDDDEnvironmentalImpact.WASTE_GENERATION,
    EnvironmentalObjective.POLLUTION_PREVENTION.value: CSDDDEnvironmentalImpact.AIR_POLLUTION,
    EnvironmentalObjective.BIODIVERSITY.value: CSDDDEnvironmentalImpact.BIODIVERSITY_LOSS,
}


# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------


class TaxonomyBridge:
    """EU Taxonomy DNSH / CSDDD environmental impact bridge for PACK-019.

    Maps Taxonomy environmental objectives to CSDDD due diligence obligations,
    assesses DNSH criteria alignment, and identifies CSDDD-relevant adverse
    impacts from Taxonomy assessment data. All calculations are deterministic
    (zero-hallucination).

    Attributes:
        config: Bridge configuration.
        _activities: Cached economic activities.

    Example:
        >>> bridge = TaxonomyBridge(TaxonomyBridgeConfig())
        >>> activities = [{"nace_code": "C20.1", "name": "Chemicals", ...}]
        >>> result = bridge.get_taxonomy_alignment(activities)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize TaxonomyBridge."""
        self.config = config or TaxonomyBridgeConfig()
        self._activities: List[EconomicActivity] = []
        logger.info(
            "TaxonomyBridge initialized (version=%s, year=%d)",
            self.config.taxonomy_version,
            self.config.reporting_year,
        )

    def get_taxonomy_alignment(
        self,
        activities: List[Dict[str, Any]],
    ) -> TaxonomyAlignmentResult:
        """Assess Taxonomy alignment for a set of economic activities.

        Args:
            activities: List of activity dicts with keys:
                nace_code, name, is_eligible, is_aligned, dnsh_results,
                minimum_safeguards_met, capex_eur, opex_eur, revenue_eur.

        Returns:
            TaxonomyAlignmentResult with alignment statistics and CSDDD impacts.
        """
        result = TaxonomyAlignmentResult()

        try:
            parsed: List[EconomicActivity] = []
            csddd_impacts: set = set()

            for act_data in activities:
                sc_obj = EnvironmentalObjective(
                    act_data.get(
                        "substantial_contribution_objective",
                        EnvironmentalObjective.CLIMATE_MITIGATION.value,
                    )
                )

                dnsh_raw = act_data.get("dnsh_results", {})
                dnsh_results = {
                    k: DNSHStatus(v) for k, v in dnsh_raw.items()
                }

                activity = EconomicActivity(
                    activity_id=act_data.get("activity_id", _new_uuid()),
                    nace_code=act_data.get("nace_code", ""),
                    name=act_data.get("name", ""),
                    substantial_contribution_objective=sc_obj,
                    is_eligible=act_data.get("is_eligible", False),
                    is_aligned=act_data.get("is_aligned", False),
                    dnsh_results=dnsh_results,
                    minimum_safeguards_met=act_data.get(
                        "minimum_safeguards_met", False
                    ),
                    capex_eur=act_data.get("capex_eur", 0.0),
                    opex_eur=act_data.get("opex_eur", 0.0),
                    revenue_eur=act_data.get("revenue_eur", 0.0),
                )
                parsed.append(activity)

                # Identify CSDDD impacts from DNSH failures
                for obj_name, status in dnsh_results.items():
                    if status == DNSHStatus.FAILED:
                        impact = DNSH_FAILURE_TO_CSDDD_IMPACT.get(obj_name)
                        if impact:
                            csddd_impacts.add(impact)

            self._activities = parsed

            eligible = sum(1 for a in parsed if a.is_eligible)
            aligned = sum(1 for a in parsed if a.is_aligned)
            dnsh_failures = sum(
                1 for a in parsed
                if any(s == DNSHStatus.FAILED for s in a.dnsh_results.values())
            )
            ms_failures = sum(
                1 for a in parsed
                if not a.minimum_safeguards_met and a.is_eligible
            )

            result.activities_assessed = len(parsed)
            result.eligible_count = eligible
            result.aligned_count = aligned
            result.eligible_pct = (
                round(eligible / len(parsed) * 100, 1) if parsed else 0.0
            )
            result.aligned_pct = (
                round(aligned / len(parsed) * 100, 1) if parsed else 0.0
            )
            result.dnsh_failures = dnsh_failures
            result.minimum_safeguards_failures = ms_failures
            result.csddd_relevant_impacts = list(csddd_impacts)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            logger.info(
                "Taxonomy alignment: %d assessed, %d eligible, %d aligned, "
                "%d DNSH failures",
                len(parsed),
                eligible,
                aligned,
                dnsh_failures,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Taxonomy alignment failed: %s", str(exc))

        return result

    def check_dnsh_criteria(
        self,
        activity: Dict[str, Any],
    ) -> DNSHAssessment:
        """Check DNSH criteria for a single economic activity.

        Args:
            activity: Activity dict with dnsh_results, nace_code, name.

        Returns:
            DNSHAssessment with per-objective results and CSDDD impacts.
        """
        sc_obj = EnvironmentalObjective(
            activity.get(
                "substantial_contribution_objective",
                EnvironmentalObjective.CLIMATE_MITIGATION.value,
            )
        )

        dnsh_raw = activity.get("dnsh_results", {})
        dnsh_results: Dict[str, DNSHStatus] = {}
        failed_objectives: List[str] = []
        csddd_impacts: List[CSDDDEnvironmentalImpact] = []

        for obj in EnvironmentalObjective:
            if obj == sc_obj:
                continue  # SC objective excluded from DNSH

            status = DNSHStatus(dnsh_raw.get(obj.value, "not_assessed"))
            dnsh_results[obj.value] = status

            if status == DNSHStatus.FAILED:
                failed_objectives.append(obj.value)
                impact = DNSH_FAILURE_TO_CSDDD_IMPACT.get(obj.value)
                if impact:
                    csddd_impacts.append(impact)

        all_passed = len(failed_objectives) == 0 and all(
            s in (DNSHStatus.PASSED, DNSHStatus.NOT_ASSESSED)
            for s in dnsh_results.values()
        )

        assessment = DNSHAssessment(
            activity_id=activity.get("activity_id", ""),
            activity_name=activity.get("name", ""),
            sc_objective=sc_obj,
            dnsh_results=dnsh_results,
            all_passed=all_passed,
            failed_objectives=failed_objectives,
            csddd_impacts=csddd_impacts,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "DNSH check for %s: %s (%d failures)",
            activity.get("name", "unknown"),
            "passed" if all_passed else "failed",
            len(failed_objectives),
        )
        return assessment

    def map_environmental_impacts_to_dnsh(
        self,
        impacts: List[Dict[str, Any]],
    ) -> List[EnvironmentalImpactMapping]:
        """Map CSDDD environmental impacts to Taxonomy DNSH criteria.

        Args:
            impacts: List of impact dicts with keys: category (CSDDDEnvironmentalImpact).

        Returns:
            List of EnvironmentalImpactMapping objects.
        """
        mappings: List[EnvironmentalImpactMapping] = []

        for imp_data in impacts:
            try:
                impact = CSDDDEnvironmentalImpact(imp_data.get("category", ""))
            except ValueError:
                continue

            objectives = IMPACT_TO_OBJECTIVE_MAP.get(impact, [])

            mappings.append(EnvironmentalImpactMapping(
                impact=impact,
                taxonomy_objectives=objectives,
                dnsh_relevance=(
                    f"DNSH criteria for {', '.join(o.value for o in objectives)} "
                    f"are relevant to {impact.value}"
                ),
                csddd_articles=["Art_6", "Art_8"],
            ))

        logger.info("Mapped %d environmental impacts to DNSH criteria", len(mappings))
        return mappings

    def assess_taxonomy_eligibility(
        self,
        company_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess overall Taxonomy eligibility for company activities.

        Args:
            company_data: Dict with keys: activities (list), total_revenue_eur,
                          total_capex_eur, total_opex_eur.

        Returns:
            Dict with eligibility and alignment KPIs.
        """
        activities = company_data.get("activities", [])
        total_revenue = company_data.get("total_revenue_eur", 0.0)
        total_capex = company_data.get("total_capex_eur", 0.0)
        total_opex = company_data.get("total_opex_eur", 0.0)

        eligible_revenue = sum(
            a.get("revenue_eur", 0.0) for a in activities if a.get("is_eligible")
        )
        aligned_revenue = sum(
            a.get("revenue_eur", 0.0) for a in activities if a.get("is_aligned")
        )
        eligible_capex = sum(
            a.get("capex_eur", 0.0) for a in activities if a.get("is_eligible")
        )
        aligned_capex = sum(
            a.get("capex_eur", 0.0) for a in activities if a.get("is_aligned")
        )
        eligible_opex = sum(
            a.get("opex_eur", 0.0) for a in activities if a.get("is_eligible")
        )
        aligned_opex = sum(
            a.get("opex_eur", 0.0) for a in activities if a.get("is_aligned")
        )

        result = {
            "activities_assessed": len(activities),
            "revenue_eligible_pct": (
                round(eligible_revenue / total_revenue * 100, 1)
                if total_revenue > 0 else 0.0
            ),
            "revenue_aligned_pct": (
                round(aligned_revenue / total_revenue * 100, 1)
                if total_revenue > 0 else 0.0
            ),
            "capex_eligible_pct": (
                round(eligible_capex / total_capex * 100, 1)
                if total_capex > 0 else 0.0
            ),
            "capex_aligned_pct": (
                round(aligned_capex / total_capex * 100, 1)
                if total_capex > 0 else 0.0
            ),
            "opex_eligible_pct": (
                round(eligible_opex / total_opex * 100, 1)
                if total_opex > 0 else 0.0
            ),
            "opex_aligned_pct": (
                round(aligned_opex / total_opex * 100, 1)
                if total_opex > 0 else 0.0
            ),
            "csddd_relevance": (
                "Taxonomy-aligned activities demonstrate CSDDD Art 8 "
                "environmental prevention measures; DNSH failures may indicate "
                "CSDDD adverse environmental impacts requiring Art 6 identification"
            ),
            "provenance_hash": _compute_hash(activities),
        }

        logger.info(
            "Taxonomy eligibility: revenue=%.1f%% eligible, %.1f%% aligned",
            result["revenue_eligible_pct"],
            result["revenue_aligned_pct"],
        )
        return result
