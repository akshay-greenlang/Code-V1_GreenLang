# -*- coding: utf-8 -*-
"""
TaxonomyBridge - EU Taxonomy Alignment Bridge for PACK-017
=============================================================

Connects PACK-017 to the EU Taxonomy alignment framework for mapping
ESRS disclosures to the six EU Taxonomy environmental objectives,
calculating Taxonomy-eligible and Taxonomy-aligned CapEx/OpEx/Revenue
KPIs, and cross-referencing E1 transition plans with Taxonomy technical
screening criteria (TSC).

Methods:
    - map_to_taxonomy()        -- Map ESRS disclosures to Taxonomy objectives
    - calculate_alignment()    -- Calculate alignment KPIs (CapEx/OpEx/Revenue)
    - get_tsc_mapping()        -- Get technical screening criteria mapping
    - assess_dnsh()            -- Assess Do No Significant Harm criteria
    - get_objective_summary()  -- Get summary by environmental objective

Environmental Objectives:
    1. Climate change mitigation
    2. Climate change adaptation
    3. Sustainable use and protection of water and marine resources
    4. Transition to a circular economy
    5. Pollution prevention and control
    6. Protection and restoration of biodiversity and ecosystems

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-017 ESRS Full Coverage Pack
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


class AlignmentStatus(str, Enum):
    """Taxonomy alignment assessment status."""

    ELIGIBLE = "eligible"
    ALIGNED = "aligned"
    NOT_ELIGIBLE = "not_eligible"
    UNDER_REVIEW = "under_review"


class DNSHStatus(str, Enum):
    """Do No Significant Harm assessment status."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_ASSESSED = "not_assessed"
    PARTIALLY_MET = "partially_met"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Bridge."""

    pack_id: str = Field(default="PACK-017")
    taxonomy_version: str = Field(default="2024")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    include_nuclear_gas: bool = Field(
        default=True, description="Include Complementary Delegated Act for nuclear/gas"
    )
    nace_codes: List[str] = Field(default_factory=list)


class TaxonomyActivity(BaseModel):
    """EU Taxonomy economic activity."""

    activity_id: str = Field(default="")
    nace_code: str = Field(default="")
    name: str = Field(default="")
    substantial_contribution_objective: EnvironmentalObjective = Field(
        default=EnvironmentalObjective.CLIMATE_MITIGATION
    )
    capex_eur: float = Field(default=0.0, ge=0.0)
    opex_eur: float = Field(default=0.0, ge=0.0)
    revenue_eur: float = Field(default=0.0, ge=0.0)
    is_eligible: bool = Field(default=False)
    is_aligned: bool = Field(default=False)
    dnsh_status: Dict[str, DNSHStatus] = Field(default_factory=dict)
    minimum_safeguards_met: bool = Field(default=False)


class AlignmentKPIs(BaseModel):
    """Taxonomy alignment KPI results."""

    total_capex_eur: float = Field(default=0.0, ge=0.0)
    eligible_capex_eur: float = Field(default=0.0, ge=0.0)
    aligned_capex_eur: float = Field(default=0.0, ge=0.0)
    capex_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    capex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_opex_eur: float = Field(default=0.0, ge=0.0)
    eligible_opex_eur: float = Field(default=0.0, ge=0.0)
    aligned_opex_eur: float = Field(default=0.0, ge=0.0)
    opex_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    opex_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_revenue_eur: float = Field(default=0.0, ge=0.0)
    eligible_revenue_eur: float = Field(default=0.0, ge=0.0)
    aligned_revenue_eur: float = Field(default=0.0, ge=0.0)
    revenue_eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    revenue_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class BridgeResult(BaseModel):
    """Result from a bridge operation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# ESRS to Taxonomy Objective Mapping
# ---------------------------------------------------------------------------

ESRS_TO_TAXONOMY_MAP: Dict[str, List[EnvironmentalObjective]] = {
    "ESRS E1": [EnvironmentalObjective.CLIMATE_MITIGATION, EnvironmentalObjective.CLIMATE_ADAPTATION],
    "ESRS E2": [EnvironmentalObjective.POLLUTION_PREVENTION],
    "ESRS E3": [EnvironmentalObjective.WATER_MARINE],
    "ESRS E4": [EnvironmentalObjective.BIODIVERSITY],
    "ESRS E5": [EnvironmentalObjective.CIRCULAR_ECONOMY],
}

# TSC reference IDs per objective
TSC_REFERENCE_MAP: Dict[str, List[str]] = {
    "climate_change_mitigation": [
        "CCM-1.1", "CCM-1.2", "CCM-1.3", "CCM-1.4",
        "CCM-2.1", "CCM-3.1", "CCM-3.2", "CCM-3.3",
        "CCM-4.1", "CCM-4.2", "CCM-4.3",
        "CCM-5.1", "CCM-5.2", "CCM-5.3",
        "CCM-6.1", "CCM-6.2", "CCM-7.1", "CCM-7.2",
    ],
    "climate_change_adaptation": [
        "CCA-1.1", "CCA-1.2", "CCA-2.1", "CCA-3.1",
        "CCA-4.1", "CCA-5.1", "CCA-6.1",
    ],
    "water_and_marine_resources": [
        "WTR-1.1", "WTR-1.2", "WTR-2.1", "WTR-2.2",
    ],
    "circular_economy": [
        "CE-1.1", "CE-1.2", "CE-2.1", "CE-2.2",
        "CE-3.1", "CE-3.2", "CE-4.1",
    ],
    "pollution_prevention": [
        "PPC-1.1", "PPC-1.2", "PPC-2.1", "PPC-3.1",
    ],
    "biodiversity_and_ecosystems": [
        "BIO-1.1", "BIO-1.2", "BIO-2.1", "BIO-3.1",
    ],
}


# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------


class TaxonomyBridge:
    """EU Taxonomy alignment bridge for PACK-017.

    Maps ESRS disclosures to the six EU Taxonomy environmental objectives,
    calculates Taxonomy KPIs, assesses DNSH criteria, and cross-references
    E1 transition plans with technical screening criteria.

    Attributes:
        config: Bridge configuration.
        _activities: Cached Taxonomy activities.
        _alignment_cache: Cached alignment KPIs.

    Example:
        >>> bridge = TaxonomyBridge(TaxonomyBridgeConfig(reporting_year=2025))
        >>> result = bridge.map_to_taxonomy(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize TaxonomyBridge."""
        self.config = config or TaxonomyBridgeConfig()
        self._activities: List[TaxonomyActivity] = []
        self._alignment_cache: Optional[AlignmentKPIs] = None
        logger.info(
            "TaxonomyBridge initialized (version=%s, year=%d)",
            self.config.taxonomy_version,
            self.config.reporting_year,
        )

    def map_to_taxonomy(self, context: Dict[str, Any]) -> BridgeResult:
        """Map ESRS disclosures to EU Taxonomy environmental objectives.

        Args:
            context: Pipeline context with ESRS disclosure data.

        Returns:
            BridgeResult with mapping status.
        """
        result = BridgeResult(started_at=_utcnow())

        try:
            activities = context.get("taxonomy_activities", [])
            parsed: List[TaxonomyActivity] = []

            for act_data in activities:
                act = TaxonomyActivity(
                    activity_id=act_data.get("activity_id", _new_uuid()),
                    nace_code=act_data.get("nace_code", ""),
                    name=act_data.get("name", ""),
                    capex_eur=act_data.get("capex_eur", 0.0),
                    opex_eur=act_data.get("opex_eur", 0.0),
                    revenue_eur=act_data.get("revenue_eur", 0.0),
                    is_eligible=act_data.get("is_eligible", False),
                    is_aligned=act_data.get("is_aligned", False),
                    minimum_safeguards_met=act_data.get("minimum_safeguards_met", False),
                )
                parsed.append(act)

            self._activities = parsed

            # Map ESRS standards to objectives
            esrs_mappings: Dict[str, List[str]] = {}
            for std, objectives in ESRS_TO_TAXONOMY_MAP.items():
                esrs_mappings[std] = [o.value for o in objectives]
            context["taxonomy_esrs_mapping"] = esrs_mappings

            result.records_processed = len(parsed)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "activities": len(parsed),
                    "mappings": len(esrs_mappings),
                })

            logger.info(
                "Mapped %d activities to Taxonomy objectives",
                len(parsed),
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Taxonomy mapping failed: %s", str(exc))

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
        return result

    def calculate_alignment(self, context: Dict[str, Any]) -> AlignmentKPIs:
        """Calculate Taxonomy-eligible and Taxonomy-aligned KPIs.

        Uses deterministic arithmetic (zero-hallucination). No LLM calls.

        Args:
            context: Pipeline context with financial data.

        Returns:
            AlignmentKPIs with CapEx/OpEx/Revenue percentages.
        """
        financials = context.get("taxonomy_financials", {})

        total_capex = financials.get("total_capex_eur", 0.0)
        total_opex = financials.get("total_opex_eur", 0.0)
        total_revenue = financials.get("total_revenue_eur", 0.0)

        eligible_capex = sum(a.capex_eur for a in self._activities if a.is_eligible)
        aligned_capex = sum(a.capex_eur for a in self._activities if a.is_aligned)
        eligible_opex = sum(a.opex_eur for a in self._activities if a.is_eligible)
        aligned_opex = sum(a.opex_eur for a in self._activities if a.is_aligned)
        eligible_revenue = sum(a.revenue_eur for a in self._activities if a.is_eligible)
        aligned_revenue = sum(a.revenue_eur for a in self._activities if a.is_aligned)

        kpis = AlignmentKPIs(
            total_capex_eur=total_capex,
            eligible_capex_eur=eligible_capex,
            aligned_capex_eur=aligned_capex,
            capex_eligible_pct=round(eligible_capex / total_capex * 100, 1) if total_capex > 0 else 0.0,
            capex_aligned_pct=round(aligned_capex / total_capex * 100, 1) if total_capex > 0 else 0.0,
            total_opex_eur=total_opex,
            eligible_opex_eur=eligible_opex,
            aligned_opex_eur=aligned_opex,
            opex_eligible_pct=round(eligible_opex / total_opex * 100, 1) if total_opex > 0 else 0.0,
            opex_aligned_pct=round(aligned_opex / total_opex * 100, 1) if total_opex > 0 else 0.0,
            total_revenue_eur=total_revenue,
            eligible_revenue_eur=eligible_revenue,
            aligned_revenue_eur=aligned_revenue,
            revenue_eligible_pct=round(eligible_revenue / total_revenue * 100, 1) if total_revenue > 0 else 0.0,
            revenue_aligned_pct=round(aligned_revenue / total_revenue * 100, 1) if total_revenue > 0 else 0.0,
        )

        self._alignment_cache = kpis

        logger.info(
            "Alignment KPIs: CapEx=%.1f%%, OpEx=%.1f%%, Revenue=%.1f%% aligned",
            kpis.capex_aligned_pct,
            kpis.opex_aligned_pct,
            kpis.revenue_aligned_pct,
        )
        return kpis

    def get_tsc_mapping(
        self,
        objective: Optional[EnvironmentalObjective] = None,
    ) -> Dict[str, List[str]]:
        """Get technical screening criteria mapping.

        Args:
            objective: Optional objective filter.

        Returns:
            Dict mapping objective names to TSC reference IDs.
        """
        if objective:
            refs = TSC_REFERENCE_MAP.get(objective.value, [])
            return {objective.value: refs}
        return dict(TSC_REFERENCE_MAP)

    def assess_dnsh(
        self,
        activity: TaxonomyActivity,
    ) -> Dict[str, DNSHStatus]:
        """Assess Do No Significant Harm criteria for an activity.

        Args:
            activity: Taxonomy activity to assess.

        Returns:
            Dict mapping each objective to DNSH status.
        """
        dnsh_results: Dict[str, DNSHStatus] = {}
        sc_objective = activity.substantial_contribution_objective

        for obj in EnvironmentalObjective:
            if obj == sc_objective:
                continue  # SC objective is excluded from DNSH
            dnsh_results[obj.value] = activity.dnsh_status.get(
                obj.value, DNSHStatus.NOT_ASSESSED
            )

        logger.info(
            "DNSH assessed for activity %s: %d objectives checked",
            activity.activity_id,
            len(dnsh_results),
        )
        return dnsh_results

    def get_objective_summary(self) -> Dict[str, Any]:
        """Get summary of Taxonomy alignment by environmental objective.

        Returns:
            Dict with per-objective activity counts and alignment data.
        """
        summary: Dict[str, Any] = {}
        for obj in EnvironmentalObjective:
            acts = [
                a for a in self._activities
                if a.substantial_contribution_objective == obj
            ]
            summary[obj.value] = {
                "total_activities": len(acts),
                "eligible_activities": sum(1 for a in acts if a.is_eligible),
                "aligned_activities": sum(1 for a in acts if a.is_aligned),
                "total_capex_eur": sum(a.capex_eur for a in acts),
                "aligned_capex_eur": sum(a.capex_eur for a in acts if a.is_aligned),
                "tsc_count": len(TSC_REFERENCE_MAP.get(obj.value, [])),
                "esrs_standards": [
                    std for std, objs in ESRS_TO_TAXONOMY_MAP.items()
                    if obj in objs
                ],
            }
        return summary

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "taxonomy_version": self.config.taxonomy_version,
            "reporting_year": self.config.reporting_year,
            "activities_loaded": len(self._activities),
            "has_alignment_kpis": self._alignment_cache is not None,
            "objectives_count": len(EnvironmentalObjective),
        }
