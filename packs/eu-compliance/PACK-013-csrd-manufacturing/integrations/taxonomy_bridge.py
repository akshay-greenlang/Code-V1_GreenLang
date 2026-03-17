"""
PACK-013 CSRD Manufacturing Pack - EU Taxonomy Bridge.

Assesses EU Taxonomy alignment for manufacturing activities against the
six environmental objectives.  Checks Substantial Contribution, Do No
Significant Harm (DNSH), and Minimum Safeguards criteria.
"""

import hashlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy environmental objectives."""
    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER = "water_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_ecosystems"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment status for an activity."""
    ALIGNED = "aligned"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"
    ASSESSMENT_PENDING = "assessment_pending"


# ---------------------------------------------------------------------------
# Manufacturing NACE activity codes with taxonomy eligibility
# ---------------------------------------------------------------------------

MANUFACTURING_ACTIVITIES: Dict[str, Dict[str, Any]] = {
    "C20.11": {
        "description": "Manufacture of industrial gases",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"lifecycle_ghg_reduction_pct": 73.4},
    },
    "C20.13": {
        "description": "Manufacture of other inorganic basic chemicals",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"top_10pct_benchmark": True},
    },
    "C20.14": {
        "description": "Manufacture of other organic basic chemicals",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"top_10pct_benchmark": True},
    },
    "C22.11": {
        "description": "Manufacture of rubber tyres and tubes",
        "eligible_objectives": [
            EnvironmentalObjective.CIRCULAR_ECONOMY,
        ],
        "sc_thresholds": {"recycled_content_pct": 25.0},
    },
    "C23.51": {
        "description": "Manufacture of cement",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"clinker_factor_max": 0.766},
    },
    "C24.10": {
        "description": "Manufacture of basic iron and steel",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"tco2e_per_tonne_max": 1.328},
    },
    "C24.42": {
        "description": "Aluminium production",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"tco2e_per_tonne_max": 1.514},
    },
    "C25.11": {
        "description": "Manufacture of metal structures",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
            EnvironmentalObjective.CIRCULAR_ECONOMY,
        ],
        "sc_thresholds": {"recycled_content_pct": 70.0},
    },
    "C27.20": {
        "description": "Manufacture of batteries",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
            EnvironmentalObjective.CIRCULAR_ECONOMY,
        ],
        "sc_thresholds": {"carbon_footprint_threshold": True},
    },
    "C29.10": {
        "description": "Manufacture of motor vehicles",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
        ],
        "sc_thresholds": {"zero_direct_co2_emissions": True},
    },
    "C17.11": {
        "description": "Manufacture of pulp",
        "eligible_objectives": [
            EnvironmentalObjective.CLIMATE_MITIGATION,
            EnvironmentalObjective.CIRCULAR_ECONOMY,
        ],
        "sc_thresholds": {"certified_sustainable_sources_pct": 70.0},
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TaxonomyBridgeConfig(BaseModel):
    """Configuration for the EU Taxonomy bridge."""
    objectives_assessed: List[EnvironmentalObjective] = Field(
        default_factory=lambda: [
            EnvironmentalObjective.CLIMATE_MITIGATION,
            EnvironmentalObjective.CLIMATE_ADAPTATION,
            EnvironmentalObjective.WATER,
            EnvironmentalObjective.CIRCULAR_ECONOMY,
            EnvironmentalObjective.POLLUTION,
            EnvironmentalObjective.BIODIVERSITY,
        ]
    )
    include_dnsh: bool = Field(default=True)
    include_safeguards: bool = Field(default=True)
    reporting_year: int = Field(default=2025)
    use_transitional_activities: bool = Field(default=True)
    use_enabling_activities: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class ObjectiveResult(BaseModel):
    """Assessment result for a single environmental objective."""
    objective: EnvironmentalObjective
    substantial_contribution: bool = Field(default=False)
    dnsh_passed: bool = Field(default=True)
    details: Dict[str, Any] = Field(default_factory=dict)


class ActivityAssessment(BaseModel):
    """Full taxonomy assessment for a single economic activity."""
    nace_code: str
    description: str = Field(default="")
    eligible: bool = Field(default=False)
    aligned: bool = Field(default=False)
    alignment_status: AlignmentStatus = Field(
        default=AlignmentStatus.ASSESSMENT_PENDING
    )
    revenue_eur: float = Field(default=0.0, ge=0.0)
    capex_eur: float = Field(default=0.0, ge=0.0)
    opex_eur: float = Field(default=0.0, ge=0.0)
    objective_results: List[ObjectiveResult] = Field(
        default_factory=list
    )
    safeguards_passed: bool = Field(default=False)


class TaxonomyAlignmentResult(BaseModel):
    """Aggregate taxonomy alignment result."""
    eligible_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    by_objective: Dict[str, float] = Field(default_factory=dict)
    dnsh_results: Dict[str, bool] = Field(default_factory=dict)
    safeguards_passed: bool = Field(default=False)
    activity_assessments: List[ActivityAssessment] = Field(
        default_factory=list
    )
    total_revenue_eur: float = Field(default=0.0, ge=0.0)
    eligible_revenue_eur: float = Field(default=0.0, ge=0.0)
    aligned_revenue_eur: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class TaxonomyBridge:
    """
    Assess EU Taxonomy alignment for manufacturing activities.

    Evaluates each economic activity against:
    1. Eligibility (NACE code in Taxonomy delegated acts)
    2. Substantial Contribution to at least one objective
    3. Do No Significant Harm to remaining objectives
    4. Minimum Safeguards (OECD, UN Guiding Principles)
    """

    def __init__(
        self, config: Optional[TaxonomyBridgeConfig] = None
    ) -> None:
        self.config = config or TaxonomyBridgeConfig()

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- public API ----------------------------------------------------------

    def assess_alignment(
        self, activities: List[Dict[str, Any]]
    ) -> TaxonomyAlignmentResult:
        """
        Assess taxonomy alignment for a list of manufacturing activities.

        Each activity dict should contain:
        - ``nace_code``: NACE Rev.2 code
        - ``revenue_eur``: Revenue attributed to this activity
        - ``capex_eur``: Capital expenditure
        - ``opex_eur``: Operating expenditure
        - ``metrics``: Dict of technical screening criteria values
        - ``company_data``: Dict for minimum safeguards check
        """
        assessments: List[ActivityAssessment] = []
        total_revenue = 0.0
        eligible_revenue = 0.0
        aligned_revenue = 0.0
        by_objective: Dict[str, float] = {
            obj.value: 0.0
            for obj in self.config.objectives_assessed
        }

        for act in activities:
            nace = act.get("nace_code", "")
            revenue = act.get("revenue_eur", 0.0)
            total_revenue += revenue

            assessment = self._assess_activity(act)
            assessments.append(assessment)

            if assessment.eligible:
                eligible_revenue += revenue
            if assessment.aligned:
                aligned_revenue += revenue
                for obj_result in assessment.objective_results:
                    if obj_result.substantial_contribution:
                        key = obj_result.objective.value
                        if key in by_objective:
                            by_objective[key] += revenue

        eligible_pct = (
            (eligible_revenue / total_revenue * 100)
            if total_revenue > 0 else 0.0
        )
        aligned_pct = (
            (aligned_revenue / total_revenue * 100)
            if total_revenue > 0 else 0.0
        )

        # Aggregate DNSH
        dnsh_results: Dict[str, bool] = {}
        for obj in self.config.objectives_assessed:
            dnsh_results[obj.value] = all(
                any(
                    or_.objective == obj and or_.dnsh_passed
                    for or_ in a.objective_results
                )
                for a in assessments
                if a.eligible
            )

        safeguards = all(a.safeguards_passed for a in assessments)

        data = {
            "eligible_pct": eligible_pct,
            "aligned_pct": aligned_pct,
            "count": len(assessments),
        }

        return TaxonomyAlignmentResult(
            eligible_pct=round(eligible_pct, 2),
            aligned_pct=round(aligned_pct, 2),
            by_objective={
                k: round(v, 2) for k, v in by_objective.items()
            },
            dnsh_results=dnsh_results,
            safeguards_passed=safeguards,
            activity_assessments=assessments,
            total_revenue_eur=round(total_revenue, 2),
            eligible_revenue_eur=round(eligible_revenue, 2),
            aligned_revenue_eur=round(aligned_revenue, 2),
            provenance_hash=self._compute_hash(data),
        )

    def check_substantial_contribution(
        self,
        activity: Dict[str, Any],
        objective: EnvironmentalObjective,
    ) -> bool:
        """
        Check whether the activity makes a substantial contribution
        to the given environmental objective.
        """
        nace = activity.get("nace_code", "")
        act_def = MANUFACTURING_ACTIVITIES.get(nace)
        if act_def is None:
            return False

        if objective not in act_def.get("eligible_objectives", []):
            return False

        metrics = activity.get("metrics", {})
        thresholds = act_def.get("sc_thresholds", {})

        return self._evaluate_thresholds(metrics, thresholds)

    def check_dnsh(
        self,
        activity: Dict[str, Any],
        other_objectives: List[EnvironmentalObjective],
    ) -> Dict[str, bool]:
        """
        Check Do No Significant Harm for objectives other than the
        one receiving substantial contribution.
        """
        results: Dict[str, bool] = {}
        metrics = activity.get("metrics", {})

        for obj in other_objectives:
            if obj == EnvironmentalObjective.CLIMATE_MITIGATION:
                results[obj.value] = metrics.get(
                    "ghg_increase_pct", 0
                ) <= 0
            elif obj == EnvironmentalObjective.CLIMATE_ADAPTATION:
                results[obj.value] = metrics.get(
                    "adaptation_assessment_done", False
                )
            elif obj == EnvironmentalObjective.WATER:
                results[obj.value] = metrics.get(
                    "water_framework_compliant", True
                )
            elif obj == EnvironmentalObjective.CIRCULAR_ECONOMY:
                results[obj.value] = metrics.get(
                    "waste_hierarchy_followed", True
                )
            elif obj == EnvironmentalObjective.POLLUTION:
                results[obj.value] = metrics.get(
                    "ied_bat_compliant", True
                )
            elif obj == EnvironmentalObjective.BIODIVERSITY:
                results[obj.value] = metrics.get(
                    "eia_completed", True
                )
            else:
                results[obj.value] = True

        return results

    def check_minimum_safeguards(
        self, company_data: Dict[str, Any]
    ) -> bool:
        """
        Check EU Taxonomy minimum safeguards.

        Based on OECD Guidelines, UN Guiding Principles on Business
        and Human Rights, ILO Core Conventions, and International
        Bill of Human Rights.
        """
        checks = [
            company_data.get("human_rights_due_diligence", False),
            company_data.get("anti_corruption_policy", False),
            company_data.get("fair_competition_policy", False),
            company_data.get("tax_compliance", False),
        ]

        # At least 3 out of 4 required for pass
        passed = sum(1 for c in checks if c) >= 3

        if not passed:
            logger.warning(
                "Minimum safeguards check failed: %d/4 criteria met",
                sum(1 for c in checks if c),
            )

        return passed

    # -- internal helpers ----------------------------------------------------

    def _assess_activity(
        self, activity: Dict[str, Any]
    ) -> ActivityAssessment:
        """Run full taxonomy assessment on a single activity."""
        nace = activity.get("nace_code", "")
        act_def = MANUFACTURING_ACTIVITIES.get(nace)
        company_data = activity.get("company_data", {})

        if act_def is None:
            return ActivityAssessment(
                nace_code=nace,
                eligible=False,
                aligned=False,
                alignment_status=AlignmentStatus.NOT_ELIGIBLE,
                revenue_eur=activity.get("revenue_eur", 0.0),
                capex_eur=activity.get("capex_eur", 0.0),
                opex_eur=activity.get("opex_eur", 0.0),
            )

        eligible_objectives = act_def.get("eligible_objectives", [])
        objective_results: List[ObjectiveResult] = []
        any_sc = False

        for obj in self.config.objectives_assessed:
            sc = self.check_substantial_contribution(activity, obj)
            if sc:
                any_sc = True

            other = [
                o for o in self.config.objectives_assessed if o != obj
            ]
            dnsh = self.check_dnsh(activity, other) if self.config.include_dnsh else {}
            all_dnsh = all(dnsh.values()) if dnsh else True

            objective_results.append(ObjectiveResult(
                objective=obj,
                substantial_contribution=sc,
                dnsh_passed=all_dnsh,
                details={"dnsh": dnsh},
            ))

        safeguards = (
            self.check_minimum_safeguards(company_data)
            if self.config.include_safeguards else True
        )

        all_dnsh_passed = all(
            r.dnsh_passed for r in objective_results
        )
        aligned = any_sc and all_dnsh_passed and safeguards

        if aligned:
            status = AlignmentStatus.ALIGNED
        elif bool(eligible_objectives):
            status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED
        else:
            status = AlignmentStatus.NOT_ELIGIBLE

        return ActivityAssessment(
            nace_code=nace,
            description=act_def.get("description", ""),
            eligible=bool(eligible_objectives),
            aligned=aligned,
            alignment_status=status,
            revenue_eur=activity.get("revenue_eur", 0.0),
            capex_eur=activity.get("capex_eur", 0.0),
            opex_eur=activity.get("opex_eur", 0.0),
            objective_results=objective_results,
            safeguards_passed=safeguards,
        )

    @staticmethod
    def _evaluate_thresholds(
        metrics: Dict[str, Any],
        thresholds: Dict[str, Any],
    ) -> bool:
        """Evaluate technical screening criteria against thresholds."""
        if not thresholds:
            return False

        for key, target in thresholds.items():
            actual = metrics.get(key)
            if actual is None:
                return False
            if isinstance(target, bool):
                if actual != target:
                    return False
            elif isinstance(target, (int, float)):
                # For "max" thresholds, actual must be <= target
                if key.endswith("_max") or key.endswith("_factor"):
                    if actual > target:
                        return False
                # For "pct" or "reduction" thresholds, actual must be >= target
                elif actual < target:
                    return False

        return True
