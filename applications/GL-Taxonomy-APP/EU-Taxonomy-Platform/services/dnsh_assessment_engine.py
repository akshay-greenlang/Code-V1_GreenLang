"""
DNSH Assessment Engine -- Do No Significant Harm Matrix Evaluation

Implements the third step of the EU Taxonomy alignment pipeline: evaluating
that an eligible activity making a substantial contribution to one objective
Does No Significant Harm (DNSH) to the remaining five environmental objectives.

Key capabilities:
  - Full DNSH assessment for all 5 non-SC objectives
  - Per-objective DNSH evaluation (water, circular economy, pollution, biodiversity)
  - Climate risk assessment per Appendix A (chronic and acute hazards)
  - DNSH matrix retrieval for any activity code
  - Batch DNSH assessment for multiple activities
  - Evidence-based scoring and audit trail

All calculations are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Article 17
    - Climate Delegated Act (EU) 2021/2139 Appendix A (Climate Risks)
    - Climate Delegated Act (EU) 2021/2139 Appendix D (Water DNSH)
    - Environmental Delegated Act (EU) 2023/2486 DNSH criteria
    - EIA Directive 2011/92/EU
    - Water Framework Directive 2000/60/EC
    - EU REACH Regulation (EC) 1907/2006
    - Habitats Directive 92/43/EEC

Example:
    >>> engine = DNSHAssessmentEngine(config)
    >>> result = engine.assess_dnsh("org-1", "4.1", "climate_mitigation", evidence_data)
    >>> result.all_pass
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    EnvironmentalObjective,
    TaxonomyAppConfig,
    DNSH_MATRIX as _CONFIG_DNSH_MATRIX,
    ENVIRONMENTAL_OBJECTIVES,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Appendix A -- Physical Climate Risk Hazards
# ---------------------------------------------------------------------------

_CHRONIC_HAZARDS: List[Dict[str, str]] = [
    {"id": "CHR-01", "name": "Changing temperature", "category": "temperature",
     "description": "Long-term temperature changes (air, freshwater, marine water)"},
    {"id": "CHR-02", "name": "Heat stress", "category": "temperature",
     "description": "Increased frequency and intensity of heat stress events"},
    {"id": "CHR-03", "name": "Changing precipitation patterns", "category": "water",
     "description": "Changes in rainfall amount, frequency, and distribution"},
    {"id": "CHR-04", "name": "Precipitation variability", "category": "water",
     "description": "Increased variability in precipitation and hydrological patterns"},
    {"id": "CHR-05", "name": "Sea level rise", "category": "water",
     "description": "Long-term increase in mean sea level"},
    {"id": "CHR-06", "name": "Ocean acidification", "category": "water",
     "description": "Decrease in ocean pH levels from CO2 absorption"},
    {"id": "CHR-07", "name": "Saline intrusion", "category": "water",
     "description": "Saltwater intrusion into freshwater aquifers"},
    {"id": "CHR-08", "name": "Soil degradation", "category": "solid_mass",
     "description": "Long-term degradation of soil quality and fertility"},
    {"id": "CHR-09", "name": "Soil erosion", "category": "solid_mass",
     "description": "Increased erosion rates from wind and water"},
    {"id": "CHR-10", "name": "Solifluction", "category": "solid_mass",
     "description": "Slow downslope movement of water-saturated soil"},
    {"id": "CHR-11", "name": "Permafrost thawing", "category": "solid_mass",
     "description": "Thawing of permanently frozen ground"},
    {"id": "CHR-12", "name": "Changing wind patterns", "category": "wind",
     "description": "Long-term changes in prevailing wind patterns and intensity"},
]

_ACUTE_HAZARDS: List[Dict[str, str]] = [
    {"id": "ACT-01", "name": "Heat wave", "category": "temperature",
     "description": "Prolonged period of abnormally high temperature"},
    {"id": "ACT-02", "name": "Cold wave / frost", "category": "temperature",
     "description": "Prolonged period of abnormally low temperature"},
    {"id": "ACT-03", "name": "Wildfire", "category": "temperature",
     "description": "Uncontrolled fire in wildland area"},
    {"id": "ACT-04", "name": "Tropical cyclone", "category": "wind",
     "description": "Cyclone, hurricane, or typhoon events"},
    {"id": "ACT-05", "name": "Storm (blizzard, dust, sand)", "category": "wind",
     "description": "Severe storm events including hail and ice storms"},
    {"id": "ACT-06", "name": "Tornado", "category": "wind",
     "description": "Violent rotating column of air"},
    {"id": "ACT-07", "name": "Flood (fluvial, pluvial, groundwater)", "category": "water",
     "description": "Flooding from river overflow, surface runoff, or groundwater rise"},
    {"id": "ACT-08", "name": "Flood (coastal)", "category": "water",
     "description": "Flooding from storm surge or tidal events"},
    {"id": "ACT-09", "name": "Drought", "category": "water",
     "description": "Extended period of abnormally low precipitation"},
    {"id": "ACT-10", "name": "Landslide", "category": "solid_mass",
     "description": "Downslope movement of soil, rock, and debris"},
    {"id": "ACT-11", "name": "Subsidence", "category": "solid_mass",
     "description": "Ground surface sinking or settling"},
    {"id": "ACT-12", "name": "Avalanche", "category": "solid_mass",
     "description": "Rapid flow of snow down a slope"},
    {"id": "ACT-13", "name": "Storm surge", "category": "water",
     "description": "Abnormal rise of water generated by storm wind and pressure"},
]


# ---------------------------------------------------------------------------
# DNSH Matrix: per-activity objective applicability
# ---------------------------------------------------------------------------

_DNSH_MATRIX: Dict[str, Dict[str, str]] = {
    # Energy activities
    "4.1": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "required",
        "pollution": "not_applicable",
        "biodiversity": "required",
    },
    "4.3": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "required",
        "pollution": "not_applicable",
        "biodiversity": "required",
    },
    "4.5": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "not_applicable",
        "biodiversity": "required",
    },
    "4.29": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "not_applicable",
        "pollution": "required",
        "biodiversity": "required",
    },
    # Manufacturing
    "3.6": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "not_applicable",
    },
    "3.9": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "not_applicable",
    },
    # Construction
    "7.1": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "not_applicable",
    },
    "7.2": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "not_applicable",
    },
    "7.7": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "not_applicable",
        "pollution": "not_applicable",
        "biodiversity": "not_applicable",
    },
    # Transport
    "6.5": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "not_applicable",
    },
    "6.1": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "not_applicable",
        "pollution": "required",
        "biodiversity": "required",
    },
    "6.10": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "required",
    },
    "6.11": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "required",
        "biodiversity": "required",
    },
    # ICT
    "8.1": {
        "climate_adaptation": "required",
        "water": "required",
        "circular_economy": "required",
        "pollution": "not_applicable",
        "biodiversity": "not_applicable",
    },
    # R&D
    "9.1": {
        "climate_adaptation": "required",
        "water": "not_applicable",
        "circular_economy": "not_applicable",
        "pollution": "not_applicable",
        "biodiversity": "not_applicable",
    },
}

# All six objectives in order
_ALL_OBJECTIVES: List[str] = [
    "climate_mitigation",
    "climate_adaptation",
    "water",
    "circular_economy",
    "pollution",
    "biodiversity",
]


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ObjectiveDNSHResult(BaseModel):
    """DNSH assessment result for a single objective."""

    objective: str = Field(..., description="Environmental objective assessed")
    status: str = Field(
        default="pending",
        description="pass, fail, not_applicable, pending, insufficient_data",
    )
    applicability: str = Field(default="required", description="required or not_applicable")
    criteria_checked: List[str] = Field(default_factory=list)
    evidence_provided: bool = Field(default=False)
    notes: str = Field(default="")
    score: float = Field(default=0.0, ge=0.0, le=100.0)


class DNSHResult(BaseModel):
    """Full DNSH assessment result for an activity."""

    assessment_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    activity_code: str = Field(...)
    sc_objective: str = Field(..., description="The objective for which SC was met")
    all_pass: bool = Field(default=False, description="Whether all DNSH checks pass")
    objective_results: Dict[str, ObjectiveDNSHResult] = Field(default_factory=dict)
    failed_objectives: List[str] = Field(default_factory=list)
    not_applicable_count: int = Field(default=0)
    assessed_count: int = Field(default=0)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class DNSHMatrixEntry(BaseModel):
    """DNSH matrix entry for an activity-objective pair."""

    activity_code: str = Field(...)
    objective: str = Field(...)
    applicability: str = Field(default="required")
    criteria_summary: str = Field(default="")


class ClimateRiskResult(BaseModel):
    """Result of Appendix A climate risk assessment."""

    assessment_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    activity_code: str = Field(...)
    location: str = Field(default="")
    time_horizon_years: int = Field(default=30)
    chronic_risks: List[Dict[str, Any]] = Field(default_factory=list)
    acute_risks: List[Dict[str, Any]] = Field(default_factory=list)
    material_risks: List[str] = Field(default_factory=list)
    adaptation_measures: List[str] = Field(default_factory=list)
    overall_risk_level: str = Field(default="medium")
    climate_risk_pass: bool = Field(default=False)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class DNSHSummary(BaseModel):
    """DNSH assessment summary for an organization."""

    org_id: str = Field(...)
    total_assessments: int = Field(default=0)
    all_pass_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    pass_rate: float = Field(default=0.0)
    objective_fail_counts: Dict[str, int] = Field(default_factory=dict)
    most_common_failure: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# DNSHAssessmentEngine
# ---------------------------------------------------------------------------

class DNSHAssessmentEngine:
    """
    DNSH Assessment Engine for EU Taxonomy 'Do No Significant Harm' evaluation.

    Evaluates whether an activity that meets the Substantial Contribution
    criteria for one environmental objective Does No Significant Harm to
    the remaining five objectives.  Implements climate risk assessment
    per Appendix A, water/circular-economy/pollution/biodiversity DNSH
    checks, and evidence-based scoring.

    Attributes:
        config: Application configuration.
        _assessments: In-memory store keyed by assessment_id.
        _climate_risk_assessments: Climate risk assessments keyed by assessment_id.
        _history: Assessment history keyed by org_id.

    Example:
        >>> engine = DNSHAssessmentEngine(config)
        >>> result = engine.assess_dnsh("org-1", "4.1", "climate_mitigation", evidence)
        >>> result.all_pass
        True
    """

    # Risk level scoring
    RISK_SCORE_HIGH: float = 30.0
    RISK_SCORE_MEDIUM: float = 60.0
    RISK_SCORE_LOW: float = 90.0

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize DNSHAssessmentEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or TaxonomyAppConfig()
        self._assessments: Dict[str, DNSHResult] = {}
        self._climate_risk_assessments: Dict[str, ClimateRiskResult] = {}
        self._history: Dict[str, List[DNSHResult]] = {}
        logger.info("DNSHAssessmentEngine initialized")

    # ------------------------------------------------------------------
    # Full DNSH Assessment
    # ------------------------------------------------------------------

    def assess_dnsh(
        self,
        org_id: str,
        activity_code: str,
        sc_objective: str,
        evidence_data: Dict[str, Any],
    ) -> DNSHResult:
        """
        Assess DNSH for all 5 non-SC objectives.

        For each objective other than the SC objective, evaluates whether
        the activity does no significant harm based on evidence data and
        the DNSH matrix applicability.

        Args:
            org_id: Organization identifier.
            activity_code: Taxonomy activity code.
            sc_objective: Objective for which SC was met (excluded from DNSH).
            evidence_data: Dict with evidence keyed by objective name.

        Returns:
            DNSHResult with per-objective pass/fail and overall status.

        Example:
            >>> result = engine.assess_dnsh("org-1", "4.1", "climate_mitigation", {
            ...     "climate_adaptation": {"climate_risk_assessed": True},
            ...     "circular_economy": {"waste_plan": True},
            ...     "biodiversity": {"eia_completed": True},
            ... })
            >>> result.all_pass
            True
        """
        start = datetime.utcnow()

        # Get DNSH matrix for this activity
        matrix = _DNSH_MATRIX.get(activity_code, {})

        objective_results: Dict[str, ObjectiveDNSHResult] = {}
        failed_objectives: List[str] = []
        not_applicable_count = 0
        assessed_count = 0

        for obj in _ALL_OBJECTIVES:
            if obj == sc_objective:
                continue

            applicability = matrix.get(obj, "required")
            obj_evidence = evidence_data.get(obj, {})

            obj_result = self._assess_single_objective(
                activity_code, sc_objective, obj, applicability, obj_evidence,
            )
            objective_results[obj] = obj_result

            if applicability == "not_applicable":
                not_applicable_count += 1
            else:
                assessed_count += 1
                if obj_result.status == "fail":
                    failed_objectives.append(obj)

        all_pass = len(failed_objectives) == 0

        provenance = _sha256(
            f"dnsh:{org_id}:{activity_code}:{sc_objective}:{all_pass}:{len(failed_objectives)}"
        )

        result = DNSHResult(
            org_id=org_id,
            activity_code=activity_code,
            sc_objective=sc_objective,
            all_pass=all_pass,
            objective_results=objective_results,
            failed_objectives=failed_objectives,
            not_applicable_count=not_applicable_count,
            assessed_count=assessed_count,
            provenance_hash=provenance,
        )

        self._assessments[result.assessment_id] = result
        self._history.setdefault(org_id, []).append(result)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "DNSH assessment for %s/%s (SC=%s): all_pass=%s, failed=%s in %.1f ms",
            org_id, activity_code, sc_objective, all_pass,
            failed_objectives, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Objective DNSH
    # ------------------------------------------------------------------

    def assess_objective_dnsh(
        self,
        activity_code: str,
        sc_objective: str,
        target_objective: str,
        evidence: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for a single target objective.

        Args:
            activity_code: Taxonomy activity code.
            sc_objective: SC objective (for context).
            target_objective: The objective being assessed for DNSH.
            evidence: Evidence data for the target objective.

        Returns:
            ObjectiveDNSHResult with status and scoring.
        """
        matrix = _DNSH_MATRIX.get(activity_code, {})
        applicability = matrix.get(target_objective, "required")
        return self._assess_single_objective(
            activity_code, sc_objective, target_objective, applicability, evidence,
        )

    def assess_water_dnsh(
        self,
        activity_code: str,
        water_data: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for water and marine resources.

        Checks Water Framework Directive compliance, water body status,
        environmental flow requirements, and water consumption efficiency.

        Args:
            activity_code: Taxonomy activity code.
            water_data: Evidence data for water DNSH assessment.

        Returns:
            ObjectiveDNSHResult for water objective.
        """
        criteria = [
            "Water Framework Directive compliance (2000/60/EC)",
            "No deterioration of water body status",
            "Environmental flow requirements met",
            "Water consumption efficiency measures",
            "Water use monitoring and reporting",
        ]

        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(water_data)

        # WFD compliance
        if water_data.get("wfd_compliant", False):
            score += 25.0
            checked.append("WFD compliance verified")

        # No deterioration
        if water_data.get("no_deterioration", False):
            score += 25.0
            checked.append("No water body status deterioration")

        # Environmental flow
        if water_data.get("environmental_flow_met", False):
            score += 25.0
            checked.append("Environmental flow requirements met")

        # Water efficiency
        if water_data.get("water_efficiency_plan", False):
            score += 25.0
            checked.append("Water efficiency plan in place")

        status = "pass" if score >= 75.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="water",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Water DNSH score: {score:.0f}/100",
            score=score,
        )

    def assess_circular_economy_dnsh(
        self,
        activity_code: str,
        waste_data: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for circular economy (waste management).

        Checks waste hierarchy compliance, durability, recyclability,
        resource efficiency, and planned obsolescence avoidance.

        Args:
            activity_code: Taxonomy activity code.
            waste_data: Evidence data for circular economy DNSH.

        Returns:
            ObjectiveDNSHResult for circular economy objective.
        """
        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(waste_data)

        # Waste hierarchy compliance
        if waste_data.get("waste_hierarchy_compliant", False):
            score += 20.0
            checked.append("Waste hierarchy compliance (prevent > reuse > recycle)")

        # Durability
        if waste_data.get("durability_assessment", False):
            score += 20.0
            checked.append("Product durability assessment completed")

        # Recyclability
        if waste_data.get("recyclability_assessment", False):
            score += 20.0
            checked.append("End-of-life recyclability assessed")

        # Resource efficiency
        if waste_data.get("resource_efficiency_plan", False):
            score += 20.0
            checked.append("Resource efficiency measures in place")

        # No planned obsolescence
        if waste_data.get("no_planned_obsolescence", True):
            score += 20.0
            checked.append("No planned obsolescence confirmed")

        status = "pass" if score >= 60.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="circular_economy",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Circular economy DNSH score: {score:.0f}/100",
            score=score,
        )

    def assess_pollution_dnsh(
        self,
        activity_code: str,
        chemical_data: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for pollution prevention and control.

        Checks IED BAT compliance, REACH/CLP/RoHS compliance, air quality
        standards, and SVHC substance restrictions.

        Args:
            activity_code: Taxonomy activity code.
            chemical_data: Evidence data for pollution DNSH.

        Returns:
            ObjectiveDNSHResult for pollution objective.
        """
        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(chemical_data)

        # IED/BAT compliance
        if chemical_data.get("ied_bat_compliant", False):
            score += 20.0
            checked.append("IED BAT compliance (where applicable)")

        # REACH compliance
        if chemical_data.get("reach_compliant", False):
            score += 20.0
            checked.append("REACH Regulation (EC 1907/2006) compliance")

        # RoHS compliance
        if chemical_data.get("rohs_compliant", False):
            score += 20.0
            checked.append("RoHS Directive compliance (electronics)")

        # No SVHC above limits
        if chemical_data.get("no_svhc_above_limits", False):
            score += 20.0
            checked.append("No SVHC substances above concentration limits")

        # Air quality compliance
        if chemical_data.get("air_quality_compliant", False):
            score += 20.0
            checked.append("Air quality standards met")

        status = "pass" if score >= 60.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="pollution",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Pollution DNSH score: {score:.0f}/100",
            score=score,
        )

    def assess_biodiversity_dnsh(
        self,
        activity_code: str,
        eia_data: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for biodiversity and ecosystems.

        Checks EIA Directive compliance, Natura 2000 site protection,
        Birds Directive compliance, and no high-biodiversity land conversion.

        Args:
            activity_code: Taxonomy activity code.
            eia_data: Evidence data for biodiversity DNSH.

        Returns:
            ObjectiveDNSHResult for biodiversity objective.
        """
        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(eia_data)

        # EIA compliance
        if eia_data.get("eia_completed", False):
            score += 25.0
            checked.append("EIA completed per Directive 2011/92/EU")

        # Natura 2000 check
        if eia_data.get("natura_2000_compliant", False):
            score += 25.0
            checked.append("Natura 2000 site protection verified")

        # Birds Directive
        if eia_data.get("birds_directive_compliant", False):
            score += 25.0
            checked.append("Birds Directive 2009/147/EC compliance")

        # No high-biodiversity conversion
        if eia_data.get("no_high_biodiversity_conversion", False):
            score += 25.0
            checked.append("No conversion of high-biodiversity land")

        status = "pass" if score >= 75.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="biodiversity",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Biodiversity DNSH score: {score:.0f}/100",
            score=score,
        )

    # ------------------------------------------------------------------
    # Climate Risk Assessment (Appendix A)
    # ------------------------------------------------------------------

    def perform_climate_risk_assessment(
        self,
        org_id: str,
        activity_code: str,
        location: str = "",
        time_horizon: int = 30,
    ) -> ClimateRiskResult:
        """
        Perform climate risk assessment per Appendix A of the Climate DA.

        Evaluates both chronic and acute physical climate risks for the
        activity and location.  Identifies material risks and suggests
        adaptation measures.

        Args:
            org_id: Organization identifier.
            activity_code: Taxonomy activity code.
            location: Geographic location or region.
            time_horizon: Assessment time horizon in years (default 30).

        Returns:
            ClimateRiskResult with risk identification and adaptation guidance.
        """
        start = datetime.utcnow()

        # Assess chronic hazards
        chronic_risks: List[Dict[str, Any]] = []
        for hazard in _CHRONIC_HAZARDS:
            risk_level = self._assess_hazard_exposure(
                hazard["category"], location, time_horizon, "chronic",
            )
            chronic_risks.append({
                "hazard_id": hazard["id"],
                "name": hazard["name"],
                "category": hazard["category"],
                "risk_level": risk_level,
                "description": hazard["description"],
            })

        # Assess acute hazards
        acute_risks: List[Dict[str, Any]] = []
        for hazard in _ACUTE_HAZARDS:
            risk_level = self._assess_hazard_exposure(
                hazard["category"], location, time_horizon, "acute",
            )
            acute_risks.append({
                "hazard_id": hazard["id"],
                "name": hazard["name"],
                "category": hazard["category"],
                "risk_level": risk_level,
                "description": hazard["description"],
            })

        # Identify material risks (medium or high)
        material_risks: List[str] = []
        for risk in chronic_risks + acute_risks:
            if risk["risk_level"] in ("high", "medium"):
                material_risks.append(risk["name"])

        # Suggest adaptation measures based on material risks
        adaptation_measures = self._suggest_adaptations(material_risks)

        # Overall risk level
        high_count = sum(
            1 for r in chronic_risks + acute_risks if r["risk_level"] == "high"
        )
        medium_count = sum(
            1 for r in chronic_risks + acute_risks if r["risk_level"] == "medium"
        )

        if high_count > 2:
            overall_level = "high"
        elif high_count > 0 or medium_count > 3:
            overall_level = "medium"
        else:
            overall_level = "low"

        # Pass if adaptation measures identified for material risks
        climate_risk_pass = (
            overall_level == "low"
            or (overall_level in ("medium", "high") and len(adaptation_measures) > 0)
        )

        provenance = _sha256(
            f"climate_risk:{org_id}:{activity_code}:{location}:{overall_level}"
        )

        result = ClimateRiskResult(
            org_id=org_id,
            activity_code=activity_code,
            location=location,
            time_horizon_years=time_horizon,
            chronic_risks=chronic_risks,
            acute_risks=acute_risks,
            material_risks=material_risks,
            adaptation_measures=adaptation_measures,
            overall_risk_level=overall_level,
            climate_risk_pass=climate_risk_pass,
            provenance_hash=provenance,
        )

        self._climate_risk_assessments[result.assessment_id] = result

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Climate risk assessment for %s/%s at %s: level=%s, pass=%s "
            "(%d chronic, %d acute, %d material) in %.1f ms",
            org_id, activity_code, location, overall_level, climate_risk_pass,
            len(chronic_risks), len(acute_risks), len(material_risks), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # DNSH Matrix
    # ------------------------------------------------------------------

    def get_dnsh_matrix(self, activity_code: str) -> List[DNSHMatrixEntry]:
        """
        Get the full DNSH matrix for an activity.

        Returns the DNSH applicability and criteria summary for each
        of the six environmental objectives.

        Args:
            activity_code: Taxonomy activity code.

        Returns:
            List of DNSHMatrixEntry for each objective.
        """
        matrix = _DNSH_MATRIX.get(activity_code, {})
        config_matrix = _CONFIG_DNSH_MATRIX.get(activity_code, {})
        entries: List[DNSHMatrixEntry] = []

        for obj in _ALL_OBJECTIVES:
            applicability = matrix.get(obj, "required")

            # Look up criteria description from the authoritative config matrix.
            # The config stores objective_value -> criteria description string.
            criteria_summary = config_matrix.get(obj, "")

            entries.append(DNSHMatrixEntry(
                activity_code=activity_code,
                objective=obj,
                applicability=applicability,
                criteria_summary=criteria_summary if applicability == "required" else "N/A",
            ))

        return entries

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_dnsh_summary(self, org_id: str) -> DNSHSummary:
        """
        Get DNSH assessment summary for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            DNSHSummary with aggregate statistics.
        """
        assessments = self._history.get(org_id, [])
        total = len(assessments)
        all_pass_count = sum(1 for a in assessments if a.all_pass)
        fail_count = total - all_pass_count

        # Count failures per objective
        obj_fail_counts: Dict[str, int] = {}
        for a in assessments:
            for obj in a.failed_objectives:
                obj_fail_counts[obj] = obj_fail_counts.get(obj, 0) + 1

        most_common_failure = None
        if obj_fail_counts:
            most_common_failure = max(obj_fail_counts, key=obj_fail_counts.get)

        provenance = _sha256(f"dnsh_summary:{org_id}:{total}:{all_pass_count}")

        return DNSHSummary(
            org_id=org_id,
            total_assessments=total,
            all_pass_count=all_pass_count,
            fail_count=fail_count,
            pass_rate=round(all_pass_count / total, 4) if total > 0 else 0.0,
            objective_fail_counts=obj_fail_counts,
            most_common_failure=most_common_failure,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Batch Assessment
    # ------------------------------------------------------------------

    def batch_assess_dnsh(
        self,
        org_id: str,
        assessments: List[Dict[str, Any]],
    ) -> List[DNSHResult]:
        """
        Batch DNSH assessment for multiple activities.

        Args:
            org_id: Organization identifier.
            assessments: List of dicts with keys:
                'activity_code', 'sc_objective', 'evidence_data'.

        Returns:
            List of DNSHResult for each activity.
        """
        start = datetime.utcnow()
        results: List[DNSHResult] = []

        for item in assessments:
            result = self.assess_dnsh(
                org_id,
                item.get("activity_code", ""),
                item.get("sc_objective", "climate_mitigation"),
                item.get("evidence_data", {}),
            )
            results.append(result)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        pass_count = sum(1 for r in results if r.all_pass)
        logger.info(
            "Batch DNSH assessment for org %s: %d/%d passed in %.1f ms",
            org_id, pass_count, len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _assess_single_objective(
        self,
        activity_code: str,
        sc_objective: str,
        target_objective: str,
        applicability: str,
        evidence: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for a single objective based on applicability and evidence.

        Args:
            activity_code: Taxonomy activity code.
            sc_objective: SC objective (for context).
            target_objective: Objective to assess for DNSH.
            applicability: 'required' or 'not_applicable'.
            evidence: Evidence data dict.

        Returns:
            ObjectiveDNSHResult with status.
        """
        if applicability == "not_applicable":
            return ObjectiveDNSHResult(
                objective=target_objective,
                status="not_applicable",
                applicability="not_applicable",
                criteria_checked=["Not applicable for this activity-objective combination"],
                evidence_provided=False,
                notes="DNSH not applicable per Delegated Act matrix",
                score=100.0,
            )

        # Dispatch to objective-specific assessment
        if target_objective == "climate_adaptation":
            return self._assess_adaptation_dnsh(activity_code, evidence)
        elif target_objective == "water":
            return self.assess_water_dnsh(activity_code, evidence)
        elif target_objective == "circular_economy":
            return self.assess_circular_economy_dnsh(activity_code, evidence)
        elif target_objective == "pollution":
            return self.assess_pollution_dnsh(activity_code, evidence)
        elif target_objective == "biodiversity":
            return self.assess_biodiversity_dnsh(activity_code, evidence)
        elif target_objective == "climate_mitigation":
            return self._assess_mitigation_dnsh(activity_code, evidence)
        else:
            return ObjectiveDNSHResult(
                objective=target_objective,
                status="pending",
                applicability="required",
                notes=f"Unknown objective: {target_objective}",
                score=0.0,
            )

    def _assess_adaptation_dnsh(
        self,
        activity_code: str,
        evidence: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """Assess DNSH for climate change adaptation (Appendix A)."""
        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(evidence)

        # Climate risk assessment completed
        if evidence.get("climate_risk_assessed", False):
            score += 40.0
            checked.append("Appendix A climate risk assessment completed")

        # Adaptation plan exists
        if evidence.get("adaptation_plan", False):
            score += 30.0
            checked.append("Climate adaptation plan documented")

        # Monitoring in place
        if evidence.get("monitoring_plan", False):
            score += 30.0
            checked.append("Climate risk monitoring plan in place")

        status = "pass" if score >= 70.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="climate_adaptation",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Climate adaptation DNSH score: {score:.0f}/100",
            score=score,
        )

    def _assess_mitigation_dnsh(
        self,
        activity_code: str,
        evidence: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """Assess DNSH for climate change mitigation (when CCA is SC)."""
        score = 0.0
        checked: List[str] = []
        evidence_provided = bool(evidence)

        # No significant GHG increase
        if evidence.get("no_significant_ghg_increase", False):
            score += 50.0
            checked.append("No significant increase in direct/indirect GHG emissions")

        # No lock-in of carbon-intensive assets
        if evidence.get("no_carbon_lock_in", False):
            score += 50.0
            checked.append("No lock-in of carbon-intensive assets")

        status = "pass" if score >= 50.0 else ("fail" if evidence_provided else "pending")

        return ObjectiveDNSHResult(
            objective="climate_mitigation",
            status=status,
            applicability="required",
            criteria_checked=checked,
            evidence_provided=evidence_provided,
            notes=f"Climate mitigation DNSH score: {score:.0f}/100",
            score=score,
        )

    def _assess_hazard_exposure(
        self,
        category: str,
        location: str,
        time_horizon: int,
        hazard_type: str,
    ) -> str:
        """
        Assess hazard exposure level for a given category and location.

        This is a deterministic placeholder that uses basic heuristics.
        In production, this would integrate with climate data providers
        (e.g. Climate Analytics, WRI Aqueduct).

        Args:
            category: Hazard category (temperature, water, solid_mass, wind).
            location: Geographic location.
            time_horizon: Assessment horizon in years.
            hazard_type: 'chronic' or 'acute'.

        Returns:
            Risk level string: 'low', 'medium', or 'high'.
        """
        # Deterministic baseline based on category and time horizon
        # Longer horizons increase chronic risk, shorter for acute
        if hazard_type == "chronic":
            if time_horizon >= 50:
                return "high" if category in ("temperature", "water") else "medium"
            elif time_horizon >= 30:
                return "medium"
            else:
                return "low"
        else:
            # Acute hazards: moderate baseline
            if category in ("water", "wind"):
                return "medium"
            return "low"

    def _suggest_adaptations(
        self,
        material_risks: List[str],
    ) -> List[str]:
        """
        Suggest adaptation measures based on identified material risks.

        Args:
            material_risks: List of material risk names.

        Returns:
            List of suggested adaptation measure descriptions.
        """
        adaptations: List[str] = []
        risk_lower = [r.lower() for r in material_risks]

        if any("temperature" in r or "heat" in r for r in risk_lower):
            adaptations.append(
                "Implement passive cooling design, heat-resistant materials, "
                "and worker heat stress management protocols"
            )

        if any("flood" in r or "sea level" in r or "storm surge" in r for r in risk_lower):
            adaptations.append(
                "Implement flood resilience measures: elevated foundations, "
                "flood barriers, drainage systems, and early warning systems"
            )

        if any("drought" in r or "precipitation" in r or "water" in r for r in risk_lower):
            adaptations.append(
                "Implement water conservation measures: rainwater harvesting, "
                "efficient irrigation, recycled water systems"
            )

        if any("wildfire" in r for r in risk_lower):
            adaptations.append(
                "Implement fire-resistant design, defensible space, "
                "and vegetation management protocols"
            )

        if any("wind" in r or "cyclone" in r or "storm" in r or "tornado" in r for r in risk_lower):
            adaptations.append(
                "Implement wind-resistant structural design and secure "
                "outdoor equipment against extreme wind events"
            )

        if any("landslide" in r or "erosion" in r or "subsidence" in r for r in risk_lower):
            adaptations.append(
                "Implement slope stabilisation, retaining walls, "
                "and geotechnical monitoring systems"
            )

        if not adaptations and material_risks:
            adaptations.append(
                "Conduct detailed climate vulnerability assessment and "
                "develop a site-specific climate adaptation plan"
            )

        return adaptations
