"""
Substantial Contribution Engine -- TSC Threshold Evaluation

Implements the second step of the EU Taxonomy alignment pipeline: evaluating
whether an eligible economic activity makes a *substantial contribution* to
at least one of the six environmental objectives by meeting the Technical
Screening Criteria (TSC) thresholds defined in the Climate Delegated Act
(EU 2021/2139) and the Environmental Delegated Act (EU 2023/2486).

Key capabilities:
  - Quantitative threshold evaluation per activity per objective
  - Enabling / transitional activity classification
  - Batch assessment for multiple activities
  - Evidence recording and audit trail
  - Activity SC profile retrieval (all objectives and thresholds)
  - Assessment history tracking

All numeric calculations are deterministic (zero-hallucination).
Thresholds are sourced from the Delegated Acts, not from LLM inference.

Reference:
    - Regulation (EU) 2020/852, Articles 10-15
    - Climate Delegated Act (EU) 2021/2139 Annex I (CCM) + Annex II (CCA)
    - Environmental Delegated Act (EU) 2023/2486 Annexes I-IV
    - Complementary Delegated Act (EU) 2022/1214 (Nuclear & Gas)

Example:
    >>> engine = SubstantialContributionEngine(config)
    >>> result = engine.assess_substantial_contribution(
    ...     "org-1", "4.1", "climate_mitigation",
    ...     {"lifecycle_ghg_gco2e_kwh": 25.0})
    >>> result.sc_met
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    ActivityType,
    AlignmentStatus,
    EnvironmentalObjective,
    TaxonomyAppConfig,
    ENVIRONMENTAL_OBJECTIVES,
    TAXONOMY_ACTIVITIES,
)
from .models import (
    EconomicActivity,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TSC Thresholds Registry (Deterministic -- sourced from Delegated Acts)
# ---------------------------------------------------------------------------

_TSC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Electricity generation (solar PV, wind, hydro, ocean, geothermal, etc.)
    "4.1": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
        },
    },
    "4.3": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
        },
    },
    "4.5": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh or power density > 5 W/m2",
        },
    },
    "4.29": {
        "climate_mitigation": {
            "metric": "direct_ghg_gco2e_kwh",
            "threshold": Decimal("270"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Direct emissions <270 gCO2e/kWh, lifecycle <550, 20yr avg <100",
            "additional_thresholds": {
                "lifecycle_ghg_gco2e_kwh": Decimal("550"),
                "lifecycle_20yr_avg_gco2e_kwh": Decimal("100"),
            },
        },
    },
    # Cement
    "3.6": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("0.498"),
            "operator": "lt",
            "unit": "tCO2e/t grey cement clinker",
            "description": "Specific GHG emissions below 0.498 tCO2e/t clinker",
        },
    },
    "3.7": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("0.722"),
            "operator": "lt",
            "unit": "tCO2e/t clinker",
            "description": "Specific GHG emissions below 0.722 tCO2e/t clinker",
        },
    },
    # Aluminium
    "3.8": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("1.484"),
            "operator": "lt",
            "unit": "tCO2e/t aluminium",
            "description": "Direct GHG emissions not exceeding 1.484 tCO2e/t aluminium",
        },
    },
    # Iron and Steel
    "3.9": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("1.331"),
            "operator": "lt",
            "unit": "tCO2e/t steel",
            "description": (
                "BF-BOF: <1.331 tCO2e/t hot metal; EAF: <0.266 tCO2e/t steel"
            ),
            "route_thresholds": {
                "bf_bof": Decimal("1.331"),
                "eaf": Decimal("0.266"),
            },
        },
    },
    # Hydrogen
    "3.10": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_tco2e_th2",
            "threshold": Decimal("3.0"),
            "operator": "lt",
            "unit": "tCO2e/tH2",
            "description": "Life-cycle GHG emissions below 3 tCO2e per tonne hydrogen",
        },
    },
    # Carbon black
    "3.11": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("1.141"),
            "operator": "lt",
            "unit": "tCO2e/t carbon black",
            "description": "GHG emissions below 1.141 tCO2e per tonne carbon black",
        },
    },
    # Soda ash
    "3.12": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("0.789"),
            "operator": "lt",
            "unit": "tCO2e/t soda ash",
            "description": "GHG emissions below 0.789 tCO2e per tonne soda ash",
        },
    },
    # Chlorine
    "3.13": {
        "climate_mitigation": {
            "metric": "specific_electricity_mwh_t",
            "threshold": Decimal("2.45"),
            "operator": "lt",
            "unit": "MWh/t chlorine",
            "description": "Electricity consumption below 2.45 MWh per tonne chlorine",
        },
    },
    # Ammonia
    "3.15": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("1.96"),
            "operator": "lt",
            "unit": "tCO2e/t ammonia",
            "description": "GHG emissions below 1.96 tCO2e per tonne ammonia",
        },
    },
    # Nitric acid
    "3.16": {
        "climate_mitigation": {
            "metric": "specific_ghg_tco2e_t",
            "threshold": Decimal("0.58"),
            "operator": "lt",
            "unit": "tCO2e/t nitric acid",
            "description": "GHG emissions below 0.58 tCO2e per tonne nitric acid",
        },
    },
    # Buildings -- new
    "7.1": {
        "climate_mitigation": {
            "metric": "pct_below_nzeb",
            "threshold": Decimal("10"),
            "operator": "gte",
            "unit": "% below NZEB",
            "description": "Primary energy demand at least 10% below NZEB threshold",
        },
    },
    # Buildings -- renovation
    "7.2": {
        "climate_mitigation": {
            "metric": "pct_energy_reduction",
            "threshold": Decimal("30"),
            "operator": "gte",
            "unit": "% primary energy reduction",
            "description": "At least 30% reduction in primary energy demand",
        },
    },
    # Passenger cars (zero tailpipe)
    "6.5": {
        "climate_mitigation": {
            "metric": "direct_tailpipe_gco2_km",
            "threshold": Decimal("0"),
            "operator": "lte",
            "unit": "gCO2/km",
            "description": "Zero direct (tailpipe) CO2 emissions",
        },
    },
    # Rail passenger
    "6.1": {
        "climate_mitigation": {
            "metric": "direct_co2_gco2_pkm",
            "threshold": Decimal("50"),
            "operator": "lt",
            "unit": "gCO2/pkm",
            "description": "Direct CO2 emissions below 50 gCO2 per passenger-km",
        },
    },
    # Sea and coastal freight
    "6.10": {
        "climate_mitigation": {
            "metric": "pct_below_eedi_reference",
            "threshold": Decimal("20"),
            "operator": "gte",
            "unit": "% below EEDI reference",
            "description": "At least 20% below EEDI reference line value",
        },
    },
    # Sea and coastal passenger
    "6.11": {
        "climate_mitigation": {
            "metric": "pct_below_eedi_reference",
            "threshold": Decimal("20"),
            "operator": "gte",
            "unit": "% below EEDI reference",
            "description": "At least 20% below EEDI reference line value",
        },
    },
    # Data centres
    "8.1": {
        "climate_mitigation": {
            "metric": "pue",
            "threshold": Decimal("1.5"),
            "operator": "lte",
            "unit": "PUE ratio",
            "description": "Power Usage Effectiveness (PUE) below 1.5 (existing) or 1.3 (new)",
        },
    },
    # Nuclear (Complementary DA)
    "4.26": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Lifecycle GHG emissions below 100 gCO2e/kWh",
        },
    },
    "4.27": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Lifecycle GHG emissions below 100 gCO2e/kWh; construction permit by 2045",
        },
    },
    "4.28": {
        "climate_mitigation": {
            "metric": "lifecycle_ghg_gco2e_kwh",
            "threshold": Decimal("100"),
            "operator": "lt",
            "unit": "gCO2e/kWh",
            "description": "Lifecycle GHG emissions below 100 gCO2e/kWh; authorized by 2040",
        },
    },
}

# Activity type classification (enabling/transitional/own_performance)
_ACTIVITY_TYPE_MAP: Dict[str, str] = {}
for _ac, _ad in TAXONOMY_ACTIVITIES.items():
    _ACTIVITY_TYPE_MAP[_ac] = _ad.get("activity_type", "own_performance")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TSCResult(BaseModel):
    """Result of a single threshold check."""

    metric: str = Field(..., description="Metric identifier")
    actual_value: float = Field(..., description="Reported actual value")
    threshold_value: float = Field(..., description="Regulatory threshold")
    operator: str = Field(default="lt", description="Comparison operator (lt, lte, gt, gte)")
    unit: str = Field(default="", description="Measurement unit")
    passed: bool = Field(default=False, description="Whether the threshold was met")
    description: str = Field(default="", description="Human-readable criterion text")


class ThresholdEvaluation(BaseModel):
    """Evaluation result for a specific metric against its threshold."""

    activity_code: str = Field(...)
    objective: str = Field(...)
    metric: str = Field(...)
    actual_value: float = Field(...)
    threshold_value: float = Field(...)
    operator: str = Field(default="lt")
    unit: str = Field(default="")
    passed: bool = Field(default=False)
    margin: float = Field(default=0.0, description="Distance from threshold (positive=pass)")
    description: str = Field(default="")
    provenance_hash: str = Field(default="")


class SCAssessmentResult(BaseModel):
    """Full substantial contribution assessment result for an activity."""

    assessment_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    activity_code: str = Field(...)
    objective: str = Field(...)
    sc_met: bool = Field(default=False, description="Whether SC criteria are met")
    activity_type: str = Field(default="own_performance")
    threshold_results: List[TSCResult] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")


class ActivitySCProfile(BaseModel):
    """Complete SC profile for an activity across all objectives."""

    activity_code: str = Field(...)
    activity_name: str = Field(default="")
    activity_type: str = Field(default="own_performance")
    objectives: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="TSC criteria per objective",
    )
    objective_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


class EnablingTransitionalResult(BaseModel):
    """Classification of an activity as own-performance, enabling, or transitional."""

    activity_code: str = Field(...)
    objective: str = Field(...)
    activity_type: str = Field(default="own_performance")
    is_enabling: bool = Field(default=False)
    is_transitional: bool = Field(default=False)
    classification_basis: str = Field(default="")
    constraints: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# SubstantialContributionEngine
# ---------------------------------------------------------------------------

class SubstantialContributionEngine:
    """
    Substantial Contribution Engine for EU Taxonomy TSC evaluation.

    Evaluates whether an economic activity meets the Technical Screening
    Criteria (TSC) for a given environmental objective.  Implements
    quantitative threshold checks, enabling/transitional classification,
    evidence recording, and batch assessment.

    Attributes:
        config: Application configuration.
        _assessments: In-memory store keyed by assessment_id.
        _evidence: In-memory evidence store keyed by assessment_id.
        _history: Assessment history keyed by (org_id, activity_code).

    Example:
        >>> engine = SubstantialContributionEngine(config)
        >>> result = engine.assess_substantial_contribution(
        ...     "org-1", "4.1", "climate_mitigation",
        ...     {"lifecycle_ghg_gco2e_kwh": 25.0})
        >>> result.sc_met
        True
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize SubstantialContributionEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or TaxonomyAppConfig()
        self._assessments: Dict[str, SCAssessmentResult] = {}
        self._evidence: Dict[str, List[Dict[str, Any]]] = {}
        self._history: Dict[str, List[SCAssessmentResult]] = {}
        logger.info("SubstantialContributionEngine initialized")

    # ------------------------------------------------------------------
    # Core Assessment
    # ------------------------------------------------------------------

    def assess_substantial_contribution(
        self,
        org_id: str,
        activity_code: str,
        objective: str,
        evidence_data: Dict[str, Any],
    ) -> SCAssessmentResult:
        """
        Assess whether an activity meets TSC for a given objective.

        Performs deterministic threshold checks against the TSC registry.
        Each metric in evidence_data is compared against its regulatory
        threshold.

        Args:
            org_id: Organization identifier.
            activity_code: Taxonomy activity code (e.g. '4.1').
            objective: Environmental objective (e.g. 'climate_mitigation').
            evidence_data: Dict mapping metric names to actual values.

        Returns:
            SCAssessmentResult with pass/fail and threshold details.

        Example:
            >>> result = engine.assess_substantial_contribution(
            ...     "org-1", "4.1", "climate_mitigation",
            ...     {"lifecycle_ghg_gco2e_kwh": 25.0})
        """
        start = datetime.utcnow()

        # Get TSC for this activity and objective
        activity_tsc = _TSC_REGISTRY.get(activity_code, {})
        objective_tsc = activity_tsc.get(objective)

        threshold_results: List[TSCResult] = []
        all_passed = True

        if objective_tsc is None:
            # No quantitative TSC defined; check if enabling activity
            act_type = _ACTIVITY_TYPE_MAP.get(activity_code, "own_performance")
            if act_type == "enabling":
                # Enabling activities: SC based on enabling others
                all_passed = True
                threshold_results.append(TSCResult(
                    metric="enabling_assessment",
                    actual_value=1.0,
                    threshold_value=1.0,
                    operator="gte",
                    unit="qualitative",
                    passed=True,
                    description="Enabling activity: SC assessed qualitatively",
                ))
            else:
                all_passed = False
                threshold_results.append(TSCResult(
                    metric="no_tsc_defined",
                    actual_value=0.0,
                    threshold_value=0.0,
                    operator="gte",
                    unit="n/a",
                    passed=False,
                    description=f"No TSC defined for {activity_code}/{objective}",
                ))
        else:
            # Evaluate primary threshold
            metric = objective_tsc["metric"]
            threshold = objective_tsc["threshold"]
            operator = objective_tsc.get("operator", "lt")
            unit = objective_tsc.get("unit", "")
            desc = objective_tsc.get("description", "")

            actual_value = evidence_data.get(metric)
            if actual_value is not None:
                passed = self._compare_value(
                    Decimal(str(actual_value)), threshold, operator,
                )
                threshold_results.append(TSCResult(
                    metric=metric,
                    actual_value=float(actual_value),
                    threshold_value=float(threshold),
                    operator=operator,
                    unit=unit,
                    passed=passed,
                    description=desc,
                ))
                if not passed:
                    all_passed = False
            else:
                # Metric not provided in evidence
                all_passed = False
                threshold_results.append(TSCResult(
                    metric=metric,
                    actual_value=0.0,
                    threshold_value=float(threshold),
                    operator=operator,
                    unit=unit,
                    passed=False,
                    description=f"Metric '{metric}' not provided in evidence data",
                ))

            # Check additional thresholds (e.g. gas electricity has multiple)
            additional = objective_tsc.get("additional_thresholds", {})
            for add_metric, add_threshold in additional.items():
                add_value = evidence_data.get(add_metric)
                if add_value is not None:
                    add_passed = self._compare_value(
                        Decimal(str(add_value)), add_threshold, "lt",
                    )
                    threshold_results.append(TSCResult(
                        metric=add_metric,
                        actual_value=float(add_value),
                        threshold_value=float(add_threshold),
                        operator="lt",
                        unit=unit,
                        passed=add_passed,
                        description=f"Additional threshold for {add_metric}",
                    ))
                    if not add_passed:
                        all_passed = False

            # Check route-specific thresholds (e.g. iron/steel BF-BOF vs EAF)
            route_thresholds = objective_tsc.get("route_thresholds", {})
            if route_thresholds:
                route = evidence_data.get("production_route", "")
                if route in route_thresholds:
                    route_threshold = route_thresholds[route]
                    route_value = evidence_data.get(metric)
                    if route_value is not None:
                        route_passed = self._compare_value(
                            Decimal(str(route_value)), route_threshold, operator,
                        )
                        threshold_results.append(TSCResult(
                            metric=f"{metric}_route_{route}",
                            actual_value=float(route_value),
                            threshold_value=float(route_threshold),
                            operator=operator,
                            unit=unit,
                            passed=route_passed,
                            description=f"Route-specific threshold ({route})",
                        ))
                        # Route-specific overrides general
                        if not route_passed:
                            all_passed = False

        act_type = _ACTIVITY_TYPE_MAP.get(activity_code, "own_performance")
        provenance = _sha256(
            f"sc_assess:{org_id}:{activity_code}:{objective}:{all_passed}"
        )

        result = SCAssessmentResult(
            org_id=org_id,
            activity_code=activity_code,
            objective=objective,
            sc_met=all_passed,
            activity_type=act_type,
            threshold_results=threshold_results,
            provenance_hash=provenance,
        )

        # Store assessment
        self._assessments[result.assessment_id] = result
        hist_key = f"{org_id}:{activity_code}"
        self._history.setdefault(hist_key, []).append(result)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "SC assessment for %s/%s/%s: sc_met=%s (%d checks) in %.1f ms",
            org_id, activity_code, objective, all_passed,
            len(threshold_results), elapsed_ms,
        )
        return result

    def evaluate_threshold(
        self,
        activity_code: str,
        objective: str,
        metric: str,
        actual_value: float,
    ) -> ThresholdEvaluation:
        """
        Evaluate a single metric against its TSC threshold.

        Args:
            activity_code: Taxonomy activity code.
            objective: Environmental objective.
            metric: Metric identifier to evaluate.
            actual_value: Reported actual value.

        Returns:
            ThresholdEvaluation with pass/fail and margin.
        """
        activity_tsc = _TSC_REGISTRY.get(activity_code, {})
        objective_tsc = activity_tsc.get(objective)

        if objective_tsc is None or objective_tsc.get("metric") != metric:
            # Check additional thresholds
            threshold_val = None
            if objective_tsc and metric in objective_tsc.get("additional_thresholds", {}):
                threshold_val = objective_tsc["additional_thresholds"][metric]
                operator = "lt"
                unit = objective_tsc.get("unit", "")
                desc = f"Additional threshold for {metric}"
            if threshold_val is None:
                provenance = _sha256(f"threshold_eval:{activity_code}:{metric}:not_found")
                return ThresholdEvaluation(
                    activity_code=activity_code,
                    objective=objective,
                    metric=metric,
                    actual_value=actual_value,
                    threshold_value=0.0,
                    passed=False,
                    margin=0.0,
                    description=f"No threshold defined for {metric} in {activity_code}/{objective}",
                    provenance_hash=provenance,
                )
        else:
            threshold_val = objective_tsc["threshold"]
            operator = objective_tsc.get("operator", "lt")
            unit = objective_tsc.get("unit", "")
            desc = objective_tsc.get("description", "")

        passed = self._compare_value(
            Decimal(str(actual_value)), threshold_val, operator,
        )
        margin = float(threshold_val) - actual_value
        if operator in ("gt", "gte"):
            margin = actual_value - float(threshold_val)

        provenance = _sha256(
            f"threshold_eval:{activity_code}:{metric}:{actual_value}:{passed}"
        )

        return ThresholdEvaluation(
            activity_code=activity_code,
            objective=objective,
            metric=metric,
            actual_value=actual_value,
            threshold_value=float(threshold_val),
            operator=operator,
            unit=unit,
            passed=passed,
            margin=round(margin, 6),
            description=desc,
            provenance_hash=provenance,
        )

    def batch_assess(
        self,
        org_id: str,
        activities_with_data: List[Dict[str, Any]],
    ) -> List[SCAssessmentResult]:
        """
        Batch assess multiple activities for substantial contribution.

        Args:
            org_id: Organization identifier.
            activities_with_data: List of dicts, each with keys:
                'activity_code', 'objective', 'evidence_data'.

        Returns:
            List of SCAssessmentResult for each activity.
        """
        start = datetime.utcnow()
        results: List[SCAssessmentResult] = []

        for item in activities_with_data:
            act_code = item.get("activity_code", "")
            objective = item.get("objective", "climate_mitigation")
            evidence = item.get("evidence_data", {})

            result = self.assess_substantial_contribution(
                org_id, act_code, objective, evidence,
            )
            results.append(result)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        passed_count = sum(1 for r in results if r.sc_met)
        logger.info(
            "Batch SC assessment for org %s: %d/%d passed in %.1f ms",
            org_id, passed_count, len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Activity SC Profile
    # ------------------------------------------------------------------

    def get_activity_sc_profile(self, activity_code: str) -> ActivitySCProfile:
        """
        Get the full SC profile for an activity across all objectives.

        Returns all TSC thresholds and criteria for the given activity
        code, organised by environmental objective.

        Args:
            activity_code: Taxonomy activity code.

        Returns:
            ActivitySCProfile with objectives and thresholds.
        """
        act_data = TAXONOMY_ACTIVITIES.get(activity_code, {})
        act_name = act_data.get("name", "Unknown")
        act_type = act_data.get("activity_type", "own_performance")

        objectives: Dict[str, Dict[str, Any]] = {}
        tsc_data = _TSC_REGISTRY.get(activity_code, {})

        for obj_key, obj_tsc in tsc_data.items():
            objectives[obj_key] = {
                "metric": obj_tsc.get("metric", ""),
                "threshold": float(obj_tsc.get("threshold", 0)),
                "operator": obj_tsc.get("operator", "lt"),
                "unit": obj_tsc.get("unit", ""),
                "description": obj_tsc.get("description", ""),
            }
            # Add route thresholds if present
            if "route_thresholds" in obj_tsc:
                objectives[obj_key]["route_thresholds"] = {
                    k: float(v) for k, v in obj_tsc["route_thresholds"].items()
                }
            # Add additional thresholds if present
            if "additional_thresholds" in obj_tsc:
                objectives[obj_key]["additional_thresholds"] = {
                    k: float(v) for k, v in obj_tsc["additional_thresholds"].items()
                }

        # Also add objective from activity catalogue even if no quantitative TSC
        act_objectives = act_data.get("objectives", [])
        act_objective = act_objectives[0] if act_objectives else ""
        if act_objective and act_objective not in objectives:
            objectives[act_objective] = {
                "metric": "qualitative",
                "threshold": 0,
                "operator": "qualitative",
                "unit": "qualitative",
                "description": f"Qualitative SC criteria for {act_objective}",
            }

        provenance = _sha256(f"sc_profile:{activity_code}:{len(objectives)}")

        return ActivitySCProfile(
            activity_code=activity_code,
            activity_name=act_name,
            activity_type=act_type,
            objectives=objectives,
            objective_count=len(objectives),
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Enabling / Transitional Classification
    # ------------------------------------------------------------------

    def classify_activity_type(
        self,
        activity_code: str,
        objective: str = "climate_mitigation",
    ) -> EnablingTransitionalResult:
        """
        Classify an activity as own-performance, enabling, or transitional.

        Enabling activities (Art. 16): directly enable other activities to make
        a substantial contribution. Transitional activities (Art. 10(2)):
        activities for which there are no technologically and economically
        feasible low-carbon alternatives.

        Args:
            activity_code: Taxonomy activity code.
            objective: Environmental objective context.

        Returns:
            EnablingTransitionalResult with classification.
        """
        act_data = TAXONOMY_ACTIVITIES.get(activity_code, {})
        act_type = act_data.get("activity_type", "own_performance")

        is_enabling = act_type == "enabling"
        is_transitional = act_type == "transitional"

        constraints: List[str] = []
        basis = ""

        if is_enabling:
            basis = (
                "Article 16 EU Taxonomy Regulation: Activity directly enables "
                "other activities to make a substantial contribution to one or "
                "more environmental objectives."
            )
            constraints = [
                "Must not lead to lock-in of carbon-intensive assets",
                "Must have substantial positive environmental impact based on lifecycle",
                "Revenue from enabling activity capped at enabling share only",
            ]
        elif is_transitional:
            basis = (
                "Article 10(2) EU Taxonomy Regulation: Activity for which there "
                "is no technologically and economically feasible low-carbon "
                "alternative, provided it supports the transition to a climate-"
                "neutral economy consistent with 1.5C pathway."
            )
            constraints = [
                "Must not hamper development of low-carbon alternatives",
                "Must not lead to lock-in of carbon-intensive assets",
                "Must have GHG performance substantially better than sector average",
                "Transitional classification is subject to periodic review",
            ]
        else:
            basis = (
                "Own-performance activity: Makes a direct substantial contribution "
                "to the environmental objective through its own performance metrics."
            )

        provenance = _sha256(f"classify:{activity_code}:{act_type}")

        return EnablingTransitionalResult(
            activity_code=activity_code,
            objective=objective,
            activity_type=act_type,
            is_enabling=is_enabling,
            is_transitional=is_transitional,
            classification_basis=basis,
            constraints=constraints,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # SC Criteria Retrieval
    # ------------------------------------------------------------------

    def get_sc_criteria(
        self,
        activity_code: str,
        objective: str,
    ) -> List[Dict[str, Any]]:
        """
        Get the TSC criteria for an activity and objective.

        Args:
            activity_code: Taxonomy activity code.
            objective: Environmental objective.

        Returns:
            List of TSC criterion dictionaries.
        """
        activity_tsc = _TSC_REGISTRY.get(activity_code, {})
        objective_tsc = activity_tsc.get(objective)

        if objective_tsc is None:
            return [{
                "metric": "none",
                "description": f"No quantitative TSC defined for {activity_code}/{objective}",
            }]

        criteria: List[Dict[str, Any]] = [
            {
                "metric": objective_tsc["metric"],
                "threshold": float(objective_tsc["threshold"]),
                "operator": objective_tsc.get("operator", "lt"),
                "unit": objective_tsc.get("unit", ""),
                "description": objective_tsc.get("description", ""),
            }
        ]

        # Add route thresholds
        for route, val in objective_tsc.get("route_thresholds", {}).items():
            criteria.append({
                "metric": f"{objective_tsc['metric']}_route_{route}",
                "threshold": float(val),
                "operator": objective_tsc.get("operator", "lt"),
                "unit": objective_tsc.get("unit", ""),
                "description": f"Route-specific threshold ({route})",
            })

        # Add additional thresholds
        for add_m, add_v in objective_tsc.get("additional_thresholds", {}).items():
            criteria.append({
                "metric": add_m,
                "threshold": float(add_v),
                "operator": "lt",
                "unit": objective_tsc.get("unit", ""),
                "description": f"Additional threshold for {add_m}",
            })

        return criteria

    def check_quantitative_threshold(
        self,
        activity_code: str,
        metric: str,
        value: float,
        unit: str = "",
    ) -> bool:
        """
        Quick boolean check of a single quantitative threshold.

        Args:
            activity_code: Taxonomy activity code.
            metric: Metric identifier.
            value: Actual reported value.
            unit: Optional unit for validation.

        Returns:
            True if the threshold is met, False otherwise.
        """
        for objective, tsc in _TSC_REGISTRY.get(activity_code, {}).items():
            if tsc.get("metric") == metric:
                return self._compare_value(
                    Decimal(str(value)), tsc["threshold"], tsc.get("operator", "lt"),
                )
            # Check in additional thresholds
            for add_m, add_v in tsc.get("additional_thresholds", {}).items():
                if add_m == metric:
                    return self._compare_value(
                        Decimal(str(value)), add_v, "lt",
                    )
        return False

    # ------------------------------------------------------------------
    # Evidence Recording
    # ------------------------------------------------------------------

    def record_evidence(
        self,
        org_id: str,
        assessment_id: str,
        evidence_type: str,
        description: str,
        document_ref: str = "",
    ) -> str:
        """
        Record evidence supporting an SC assessment.

        Args:
            org_id: Organization identifier.
            assessment_id: Assessment ID to attach evidence to.
            evidence_type: Type of evidence (measurement, report, certificate, etc.).
            description: Human-readable evidence description.
            document_ref: External document reference or URL.

        Returns:
            Evidence record ID.
        """
        evidence_id = _new_id()
        record = {
            "evidence_id": evidence_id,
            "org_id": org_id,
            "assessment_id": assessment_id,
            "evidence_type": evidence_type,
            "description": description,
            "document_ref": document_ref,
            "recorded_at": _now().isoformat(),
        }

        self._evidence.setdefault(assessment_id, []).append(record)

        # Update assessment with evidence ref
        if assessment_id in self._assessments:
            self._assessments[assessment_id].evidence_refs.append(evidence_id)

        logger.info(
            "Recorded evidence %s for assessment %s (type=%s)",
            evidence_id, assessment_id, evidence_type,
        )
        return evidence_id

    # ------------------------------------------------------------------
    # Assessment History
    # ------------------------------------------------------------------

    def get_assessment_history(
        self,
        org_id: str,
        activity_code: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve assessment history for an organization.

        Args:
            org_id: Organization identifier.
            activity_code: Optional activity code filter.

        Returns:
            List of assessment records.
        """
        results: List[Dict[str, Any]] = []

        for key, assessments in self._history.items():
            key_org, key_act = key.split(":", 1)
            if key_org != org_id:
                continue
            if activity_code and key_act != activity_code:
                continue

            for a in assessments:
                results.append({
                    "assessment_id": a.assessment_id,
                    "activity_code": a.activity_code,
                    "objective": a.objective,
                    "sc_met": a.sc_met,
                    "activity_type": a.activity_type,
                    "threshold_count": len(a.threshold_results),
                    "evidence_count": len(a.evidence_refs),
                    "assessed_at": a.assessed_at.isoformat(),
                })

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_sc_summary(self, org_id: str) -> Dict[str, Any]:
        """
        Get SC assessment summary statistics for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            Dictionary with SC summary statistics.
        """
        all_assessments: List[SCAssessmentResult] = []
        for key, assessments in self._history.items():
            if key.startswith(f"{org_id}:"):
                all_assessments.extend(assessments)

        total = len(all_assessments)
        passed = sum(1 for a in all_assessments if a.sc_met)
        failed = total - passed

        # Breakdown by objective
        obj_breakdown: Dict[str, Dict[str, int]] = {}
        for a in all_assessments:
            if a.objective not in obj_breakdown:
                obj_breakdown[a.objective] = {"passed": 0, "failed": 0}
            if a.sc_met:
                obj_breakdown[a.objective]["passed"] += 1
            else:
                obj_breakdown[a.objective]["failed"] += 1

        # Breakdown by activity type
        type_breakdown: Dict[str, int] = {}
        for a in all_assessments:
            type_breakdown[a.activity_type] = type_breakdown.get(a.activity_type, 0) + 1

        return {
            "org_id": org_id,
            "total_assessments": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "objective_breakdown": obj_breakdown,
            "type_breakdown": type_breakdown,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compare_value(
        self,
        actual: Decimal,
        threshold: Decimal,
        operator: str,
    ) -> bool:
        """
        Compare actual value against threshold using the specified operator.

        Deterministic comparison -- no LLM involved.

        Args:
            actual: Actual reported value.
            threshold: Regulatory threshold.
            operator: One of 'lt', 'lte', 'gt', 'gte', 'eq'.

        Returns:
            True if the comparison passes.
        """
        if operator == "lt":
            return actual < threshold
        elif operator == "lte":
            return actual <= threshold
        elif operator == "gt":
            return actual > threshold
        elif operator == "gte":
            return actual >= threshold
        elif operator == "eq":
            return actual == threshold
        else:
            logger.warning("Unknown operator '%s', defaulting to lt", operator)
            return actual < threshold
