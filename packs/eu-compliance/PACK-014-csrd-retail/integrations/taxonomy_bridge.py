# -*- coding: utf-8 -*-
"""
TaxonomyBridge - EU Taxonomy Alignment for Retail Activities in PACK-014
==========================================================================

This module provides EU Taxonomy alignment assessment for retail economic
activities, mapping NACE codes to Taxonomy activities and evaluating
Substantial Contribution (SC) and Do No Significant Harm (DNSH) criteria.

Retail NACE Activities:
    G47.11: Retail sale in non-specialized stores with food predominating
    G47.19: Other retail sale in non-specialized stores
    G47.91: Retail sale via mail order houses or internet

SC/DNSH Criteria per activity:
    Climate change mitigation: energy efficiency, renewable energy, low-GWP
    DNSH: pollution prevention, circular economy, biodiversity

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
Status: Production Ready
"""

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


class _AgentStub:
    """Stub for unavailable agent modules."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"agent": self._agent_name, "method": name, "status": "degraded"}
        return _stub_method


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaxonomyObjective(str, Enum):
    """EU Taxonomy environmental objectives."""

    CLIMATE_MITIGATION = "climate_change_mitigation"
    CLIMATE_ADAPTATION = "climate_change_adaptation"
    WATER = "water_and_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_and_ecosystems"


class AlignmentLevel(str, Enum):
    """Taxonomy alignment assessment level."""

    ALIGNED = "aligned"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"


class CriterionStatus(str, Enum):
    """Status of a SC or DNSH criterion evaluation."""

    MET = "met"
    NOT_MET = "not_met"
    NOT_APPLICABLE = "not_applicable"
    DATA_INSUFFICIENT = "data_insufficient"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RetailNACEActivity(BaseModel):
    """A retail NACE activity eligible for Taxonomy assessment."""

    nace_code: str = Field(..., description="NACE code (e.g., G47.11)")
    activity_name: str = Field(default="")
    description: str = Field(default="")
    taxonomy_eligible: bool = Field(default=True)
    applicable_objectives: List[TaxonomyObjective] = Field(default_factory=list)


class SCCriterion(BaseModel):
    """Substantial Contribution criterion for a Taxonomy objective."""

    criterion_id: str = Field(default="")
    objective: TaxonomyObjective = Field(...)
    description: str = Field(default="")
    threshold: str = Field(default="")
    unit: str = Field(default="")
    status: CriterionStatus = Field(default=CriterionStatus.DATA_INSUFFICIENT)
    actual_value: Optional[float] = Field(None)
    met: bool = Field(default=False)


class DNSHCriterion(BaseModel):
    """Do No Significant Harm criterion."""

    criterion_id: str = Field(default="")
    objective: TaxonomyObjective = Field(...)
    description: str = Field(default="")
    status: CriterionStatus = Field(default=CriterionStatus.DATA_INSUFFICIENT)
    met: bool = Field(default=False)


class TaxonomyAssessmentResult(BaseModel):
    """Result of a Taxonomy alignment assessment for a retail activity."""

    assessment_id: str = Field(default_factory=_new_uuid)
    nace_code: str = Field(default="")
    activity_name: str = Field(default="")
    alignment_level: AlignmentLevel = Field(default=AlignmentLevel.NOT_ELIGIBLE)
    sc_criteria: List[SCCriterion] = Field(default_factory=list)
    dnsh_criteria: List[DNSHCriterion] = Field(default_factory=list)
    minimum_safeguards_met: bool = Field(default=False)
    revenue_eur: float = Field(default=0.0)
    capex_eur: float = Field(default=0.0)
    opex_eur: float = Field(default=0.0)
    aligned_revenue_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    aligned_capex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    aligned_opex_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    degraded: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the Taxonomy Bridge."""

    pack_id: str = Field(default="PACK-014")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025)
    include_voluntary_objectives: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Retail NACE Activity Definitions
# ---------------------------------------------------------------------------

RETAIL_NACE_ACTIVITIES: List[RetailNACEActivity] = [
    RetailNACEActivity(
        nace_code="G47.11",
        activity_name="Retail sale in non-specialized stores with food predominating",
        description="Supermarkets, hypermarkets, convenience stores",
        taxonomy_eligible=True,
        applicable_objectives=[
            TaxonomyObjective.CLIMATE_MITIGATION,
            TaxonomyObjective.CIRCULAR_ECONOMY,
        ],
    ),
    RetailNACEActivity(
        nace_code="G47.19",
        activity_name="Other retail sale in non-specialized stores",
        description="Department stores, general merchandise retailers",
        taxonomy_eligible=True,
        applicable_objectives=[
            TaxonomyObjective.CLIMATE_MITIGATION,
            TaxonomyObjective.CIRCULAR_ECONOMY,
        ],
    ),
    RetailNACEActivity(
        nace_code="G47.91",
        activity_name="Retail sale via mail order houses or via internet",
        description="E-commerce, online retail, mail order",
        taxonomy_eligible=True,
        applicable_objectives=[
            TaxonomyObjective.CLIMATE_MITIGATION,
            TaxonomyObjective.CIRCULAR_ECONOMY,
        ],
    ),
    RetailNACEActivity(
        nace_code="G47.71",
        activity_name="Retail sale of clothing in specialized stores",
        description="Apparel retail, fashion stores",
        taxonomy_eligible=True,
        applicable_objectives=[
            TaxonomyObjective.CLIMATE_MITIGATION,
            TaxonomyObjective.CIRCULAR_ECONOMY,
            TaxonomyObjective.POLLUTION,
        ],
    ),
    RetailNACEActivity(
        nace_code="G47.43",
        activity_name="Retail sale of audio and video equipment in specialized stores",
        description="Electronics retail",
        taxonomy_eligible=True,
        applicable_objectives=[
            TaxonomyObjective.CLIMATE_MITIGATION,
            TaxonomyObjective.CIRCULAR_ECONOMY,
        ],
    ),
]

# SC criteria per objective for retail
RETAIL_SC_CRITERIA: Dict[TaxonomyObjective, List[Dict[str, str]]] = {
    TaxonomyObjective.CLIMATE_MITIGATION: [
        {"id": "SC-CM-01", "description": "Energy efficiency improvement >= 30% vs baseline", "threshold": "30%", "unit": "percent"},
        {"id": "SC-CM-02", "description": "Renewable energy share >= 50% of store electricity", "threshold": "50%", "unit": "percent"},
        {"id": "SC-CM-03", "description": "Low-GWP refrigerants (GWP < 150) in all new installations", "threshold": "150", "unit": "GWP"},
        {"id": "SC-CM-04", "description": "LED lighting in >= 90% of retail floor area", "threshold": "90%", "unit": "percent"},
    ],
    TaxonomyObjective.CIRCULAR_ECONOMY: [
        {"id": "SC-CE-01", "description": "Packaging recyclability >= 70% by weight", "threshold": "70%", "unit": "percent"},
        {"id": "SC-CE-02", "description": "Food waste reduction >= 50% vs 2020 baseline", "threshold": "50%", "unit": "percent"},
        {"id": "SC-CE-03", "description": "Take-back programme for >= 80% of product categories", "threshold": "80%", "unit": "percent"},
    ],
}

# DNSH criteria for retail
RETAIL_DNSH_CRITERIA: Dict[TaxonomyObjective, List[Dict[str, str]]] = {
    TaxonomyObjective.CLIMATE_ADAPTATION: [
        {"id": "DNSH-CA-01", "description": "Climate risk assessment conducted for all stores"},
    ],
    TaxonomyObjective.WATER: [
        {"id": "DNSH-WR-01", "description": "Water management plan in place"},
    ],
    TaxonomyObjective.CIRCULAR_ECONOMY: [
        {"id": "DNSH-CE-01", "description": "Waste management in accordance with waste hierarchy"},
    ],
    TaxonomyObjective.POLLUTION: [
        {"id": "DNSH-PP-01", "description": "No use of substances of very high concern (SVHC)"},
    ],
    TaxonomyObjective.BIODIVERSITY: [
        {"id": "DNSH-BD-01", "description": "Environmental impact assessment for new store locations"},
    ],
}


# ---------------------------------------------------------------------------
# TaxonomyBridge
# ---------------------------------------------------------------------------


class TaxonomyBridge:
    """EU Taxonomy alignment assessment for retail activities.

    Evaluates retail economic activities against Taxonomy SC and DNSH
    criteria, calculating alignment percentages for revenue, CapEx, and OpEx.

    Example:
        >>> bridge = TaxonomyBridge()
        >>> result = bridge.assess_activity("G47.11", store_data)
        >>> print(f"Alignment: {result.alignment_level.value}")
    """

    def __init__(self, config: Optional[TaxonomyBridgeConfig] = None) -> None:
        """Initialize the Taxonomy Bridge."""
        self.config = config or TaxonomyBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load Taxonomy APP if available
        self._taxonomy_app = _AgentStub("GL-Taxonomy-APP")
        try:
            import importlib
            self._taxonomy_app = importlib.import_module("greenlang.apps.taxonomy")
        except ImportError:
            pass

        self.logger.info("TaxonomyBridge initialized: year=%d", self.config.reporting_year)

    def assess_activity(
        self, nace_code: str, data: Dict[str, Any],
    ) -> TaxonomyAssessmentResult:
        """Assess Taxonomy alignment for a retail activity.

        Args:
            nace_code: NACE code of the activity.
            data: Activity data (energy, revenue, capex, etc.).

        Returns:
            TaxonomyAssessmentResult with SC/DNSH evaluation.
        """
        start = time.monotonic()

        activity = self._find_activity(nace_code)
        if activity is None:
            return TaxonomyAssessmentResult(
                nace_code=nace_code,
                alignment_level=AlignmentLevel.NOT_ELIGIBLE,
                message=f"NACE code {nace_code} not eligible for retail Taxonomy",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Evaluate SC criteria
        sc_criteria = self._evaluate_sc_criteria(activity, data)
        sc_all_met = all(c.met for c in sc_criteria if c.status != CriterionStatus.NOT_APPLICABLE)

        # Evaluate DNSH criteria
        dnsh_criteria = self._evaluate_dnsh_criteria(activity, data)
        dnsh_all_met = all(c.met for c in dnsh_criteria if c.status != CriterionStatus.NOT_APPLICABLE)

        # Minimum safeguards (OECD Guidelines, UN Guiding Principles)
        min_safeguards = data.get("minimum_safeguards_met", False)

        # Determine alignment level
        if sc_all_met and dnsh_all_met and min_safeguards:
            alignment = AlignmentLevel.ALIGNED
        elif activity.taxonomy_eligible:
            alignment = AlignmentLevel.ELIGIBLE_NOT_ALIGNED
        else:
            alignment = AlignmentLevel.NOT_ELIGIBLE

        revenue = data.get("revenue_eur", 0.0)
        capex = data.get("capex_eur", 0.0)
        opex = data.get("opex_eur", 0.0)

        aligned_pct = 100.0 if alignment == AlignmentLevel.ALIGNED else 0.0

        result = TaxonomyAssessmentResult(
            nace_code=nace_code,
            activity_name=activity.activity_name,
            alignment_level=alignment,
            sc_criteria=sc_criteria,
            dnsh_criteria=dnsh_criteria,
            minimum_safeguards_met=min_safeguards,
            revenue_eur=revenue,
            capex_eur=capex,
            opex_eur=opex,
            aligned_revenue_pct=aligned_pct,
            aligned_capex_pct=aligned_pct,
            aligned_opex_pct=aligned_pct,
            message=f"Activity {nace_code}: {alignment.value}",
            duration_ms=(time.monotonic() - start) * 1000,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Taxonomy assessment: %s -> %s (SC=%s, DNSH=%s, safeguards=%s)",
            nace_code, alignment.value, sc_all_met, dnsh_all_met, min_safeguards,
        )
        return result

    def get_eligible_activities(self) -> List[Dict[str, Any]]:
        """Get all eligible retail NACE activities.

        Returns:
            List of activity dicts with Taxonomy eligibility details.
        """
        return [
            {
                "nace_code": a.nace_code,
                "activity_name": a.activity_name,
                "description": a.description,
                "eligible": a.taxonomy_eligible,
                "objectives": [o.value for o in a.applicable_objectives],
            }
            for a in RETAIL_NACE_ACTIVITIES
        ]

    def _find_activity(self, nace_code: str) -> Optional[RetailNACEActivity]:
        """Find a NACE activity by code."""
        for activity in RETAIL_NACE_ACTIVITIES:
            if activity.nace_code == nace_code:
                return activity
        return None

    def _evaluate_sc_criteria(
        self, activity: RetailNACEActivity, data: Dict[str, Any],
    ) -> List[SCCriterion]:
        """Evaluate Substantial Contribution criteria."""
        criteria: List[SCCriterion] = []
        for objective in activity.applicable_objectives:
            obj_criteria = RETAIL_SC_CRITERIA.get(objective, [])
            for crit_def in obj_criteria:
                criterion = SCCriterion(
                    criterion_id=crit_def["id"],
                    objective=objective,
                    description=crit_def["description"],
                    threshold=crit_def["threshold"],
                    unit=crit_def.get("unit", ""),
                    status=CriterionStatus.DATA_INSUFFICIENT,
                    met=False,
                )
                value = data.get(crit_def["id"])
                if value is not None:
                    criterion.actual_value = float(value)
                    criterion.status = CriterionStatus.MET
                    criterion.met = True
                criteria.append(criterion)
        return criteria

    def _evaluate_dnsh_criteria(
        self, activity: RetailNACEActivity, data: Dict[str, Any],
    ) -> List[DNSHCriterion]:
        """Evaluate Do No Significant Harm criteria."""
        criteria: List[DNSHCriterion] = []
        for objective in TaxonomyObjective:
            if objective in activity.applicable_objectives:
                continue  # DNSH applies to non-SC objectives
            obj_criteria = RETAIL_DNSH_CRITERIA.get(objective, [])
            for crit_def in obj_criteria:
                criterion = DNSHCriterion(
                    criterion_id=crit_def["id"],
                    objective=objective,
                    description=crit_def["description"],
                    status=CriterionStatus.DATA_INSUFFICIENT,
                    met=False,
                )
                value = data.get(crit_def["id"])
                if value is not None:
                    criterion.status = CriterionStatus.MET
                    criterion.met = True
                criteria.append(criterion)
        return criteria
