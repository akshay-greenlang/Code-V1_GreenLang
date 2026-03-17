# -*- coding: utf-8 -*-
"""
CivilLiabilityEngine - PACK-019 CSDDD Civil Liability Engine
==============================================================

Assesses civil liability exposure per Article 29 of the EU Corporate
Sustainability Due Diligence Directive (CSDDD / CS3D).

Article 29 establishes a civil liability regime under which companies
may be held liable for damage caused to natural or legal persons by
the failure to comply with their due diligence obligations under
Articles 7 and 8 (prevention and mitigation of adverse impacts).

The engine evaluates:
    - Liability triggers (failure to prevent, mitigate, remediate)
    - Defence positions available to the company
    - Financial exposure estimation
    - Insurance adequacy assessment
    - Compliance gap analysis

CSDDD Article 29 Requirements:
    - Art 29(1): Member States shall ensure that a company is liable
      for damage caused to a natural or legal person, provided that
      the company intentionally or negligently failed to comply with
      Articles 7 and 8
    - Art 29(2): Where the company is liable, the injured party
      shall be entitled to full compensation
    - Art 29(3): The company shall not be liable if the damage was
      caused only by its business partners in the chain of activities
    - Art 29(4): Joint and several liability where multiple companies
      caused the damage
    - Art 29(5): Five-year limitation period

Key Defences:
    - Compliance with due diligence obligations (Art 7-8)
    - Damage caused only by business partner (Art 29(3))
    - Contractual assurances obtained and verified
    - Force majeure
    - Limitation period expired (5 years per Art 29(5))

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Article 29: Civil liability
    - Articles 7-8: Prevention and mitigation obligations
    - Rome II Regulation (EC) 864/2007 (applicable law)
    - Brussels I Regulation (EU) 1215/2012 (jurisdiction)
    - National tort law implementations

Zero-Hallucination:
    - Exposure scores computed from deterministic criteria counts
    - Financial exposure uses multiplicative factors on damage estimates
    - Defence strength uses weighted boolean scoring
    - Insurance adequacy computed from coverage vs exposure ratios
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-019 CSDDD Readiness
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


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


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


def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )


def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LiabilityTrigger(str, Enum):
    """Events that may trigger civil liability under Art 29 CSDDD.

    Each trigger corresponds to a specific failure in the company's
    due diligence obligations under Articles 7 and 8 of the CSDDD.
    """
    FAILURE_TO_PREVENT = "failure_to_prevent"
    FAILURE_TO_MITIGATE = "failure_to_mitigate"
    FAILURE_TO_REMEDIATE = "failure_to_remediate"
    INADEQUATE_DUE_DILIGENCE = "inadequate_due_diligence"
    CONTRACTUAL_BREACH = "contractual_breach"


class DefencePosition(str, Enum):
    """Available defence positions under Art 29 CSDDD.

    Defences that a company may invoke to reduce or eliminate
    liability for adverse impacts.
    """
    FULL_COMPLIANCE = "full_compliance"
    PARTIAL_COMPLIANCE = "partial_compliance"
    REASONABLE_EFFORTS = "reasonable_efforts"
    FORCE_MAJEURE = "force_majeure"
    LIMITATION_EXPIRED = "limitation_expired"


class ExposureLevel(str, Enum):
    """Civil liability exposure level classification.

    Indicates the severity of potential financial and reputational
    exposure from liability claims under Art 29 CSDDD.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ImpactSeverity(str, Enum):
    """Severity of the adverse impact in a liability scenario.

    Used to weight the exposure estimation based on the nature
    and scale of the harm caused.
    """
    CATASTROPHIC = "catastrophic"
    SEVERE = "severe"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"


class ImpactDomain(str, Enum):
    """Domain of the adverse impact.

    Classifies the nature of the adverse impact for risk
    categorisation and defence evaluation.
    """
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENT = "environment"
    LABOUR_RIGHTS = "labour_rights"
    HEALTH_AND_SAFETY = "health_and_safety"
    COMMUNITY_RIGHTS = "community_rights"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Art 29(5): limitation period in years
LIMITATION_PERIOD_YEARS: int = 5

# Exposure multipliers by severity (applied to base damage estimate)
SEVERITY_MULTIPLIERS: Dict[str, Decimal] = {
    ImpactSeverity.CATASTROPHIC.value: Decimal("3.0"),
    ImpactSeverity.SEVERE.value: Decimal("2.0"),
    ImpactSeverity.SIGNIFICANT.value: Decimal("1.5"),
    ImpactSeverity.MODERATE.value: Decimal("1.0"),
    ImpactSeverity.MINOR.value: Decimal("0.5"),
}

# Defence strength weights (how much each defence reduces exposure)
DEFENCE_REDUCTION_FACTORS: Dict[str, Decimal] = {
    DefencePosition.FULL_COMPLIANCE.value: Decimal("0.90"),
    DefencePosition.PARTIAL_COMPLIANCE.value: Decimal("0.50"),
    DefencePosition.REASONABLE_EFFORTS.value: Decimal("0.30"),
    DefencePosition.FORCE_MAJEURE.value: Decimal("0.80"),
    DefencePosition.LIMITATION_EXPIRED.value: Decimal("1.00"),
}

# Due diligence element weights for compliance assessment
DD_ELEMENT_WEIGHTS: Dict[str, Decimal] = {
    "risk_identification": Decimal("0.15"),
    "prevention_measures": Decimal("0.20"),
    "mitigation_actions": Decimal("0.20"),
    "contractual_assurances": Decimal("0.15"),
    "verification_measures": Decimal("0.15"),
    "stakeholder_consultation": Decimal("0.15"),
}

# Exposure level thresholds (EUR)
EXPOSURE_THRESHOLDS: Dict[str, Decimal] = {
    ExposureLevel.CRITICAL.value: Decimal("10000000"),      # > 10M EUR
    ExposureLevel.HIGH.value: Decimal("1000000"),            # > 1M EUR
    ExposureLevel.MEDIUM.value: Decimal("100000"),           # > 100K EUR
    ExposureLevel.LOW.value: Decimal("10000"),               # > 10K EUR
    ExposureLevel.NEGLIGIBLE.value: Decimal("0"),            # <= 10K EUR
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class LiabilityScenario(BaseModel):
    """A single civil liability scenario for assessment under Art 29.

    Represents a potential or actual liability situation where
    the company may face claims for damage caused by adverse
    human rights or environmental impacts.
    """
    scenario_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this liability scenario",
    )
    company_name: str = Field(
        default="",
        description="Name of the company being assessed",
        max_length=500,
    )
    adverse_impact: str = Field(
        ...,
        description="Description of the adverse impact",
        max_length=5000,
    )
    impact_domain: ImpactDomain = Field(
        default=ImpactDomain.HUMAN_RIGHTS,
        description="Domain of the adverse impact",
    )
    impact_severity: ImpactSeverity = Field(
        default=ImpactSeverity.MODERATE,
        description="Severity of the adverse impact",
    )
    due_diligence_performed: bool = Field(
        default=False,
        description="Whether due diligence was performed for this risk",
    )
    risk_identified: bool = Field(
        default=False,
        description="Whether the risk was identified through DD processes",
    )
    prevention_measures_taken: bool = Field(
        default=False,
        description="Whether prevention measures were implemented per Art 7",
    )
    prevention_measures_description: str = Field(
        default="",
        description="Description of prevention measures taken",
        max_length=5000,
    )
    mitigation_actions_taken: bool = Field(
        default=False,
        description="Whether mitigation actions were taken per Art 8",
    )
    mitigation_actions_description: str = Field(
        default="",
        description="Description of mitigation actions taken",
        max_length=5000,
    )
    contractual_assurances_obtained: bool = Field(
        default=False,
        description="Whether contractual assurances were obtained from "
                    "business partners",
    )
    verification_measures_applied: bool = Field(
        default=False,
        description="Whether verification measures were applied to "
                    "check compliance",
    )
    stakeholders_consulted: bool = Field(
        default=False,
        description="Whether affected stakeholders were consulted",
    )
    caused_by_business_partner_only: bool = Field(
        default=False,
        description="Whether the damage was caused only by a business "
                    "partner (Art 29(3) defence)",
    )
    remediation_provided: bool = Field(
        default=False,
        description="Whether remediation has been provided to affected parties",
    )
    remediation_adequate: bool = Field(
        default=False,
        description="Whether the remediation is considered adequate",
    )
    jurisdiction: str = Field(
        default="",
        description="Jurisdiction where the claim may be brought",
        max_length=200,
    )
    limitation_period_years: int = Field(
        default=LIMITATION_PERIOD_YEARS,
        description="Applicable limitation period in years",
        ge=1,
    )
    years_since_impact: int = Field(
        default=0,
        description="Years elapsed since the adverse impact occurred",
        ge=0,
    )
    damage_estimate_eur: Decimal = Field(
        default=Decimal("0"),
        description="Estimated damage amount in EUR",
        ge=Decimal("0"),
    )
    number_of_affected_persons: int = Field(
        default=0,
        description="Number of persons affected by the adverse impact",
        ge=0,
    )
    joint_liability_parties: int = Field(
        default=0,
        description="Number of other parties jointly liable (Art 29(4))",
        ge=0,
    )
    insurance_coverage_eur: Decimal = Field(
        default=Decimal("0"),
        description="Available insurance coverage for this scenario (EUR)",
        ge=Decimal("0"),
    )
    is_class_action_risk: bool = Field(
        default=False,
        description="Whether there is a risk of class action litigation",
    )
    prior_warnings_received: bool = Field(
        default=False,
        description="Whether prior warnings about the risk were received",
    )
    regulatory_enforcement_active: bool = Field(
        default=False,
        description="Whether regulatory enforcement is active for this issue",
    )


class LiabilityAssessment(BaseModel):
    """Assessment result for a single liability scenario."""
    scenario_id: str = Field(
        default="", description="Scenario identifier"
    )
    triggers: List[str] = Field(
        default_factory=list,
        description="Identified liability triggers",
    )
    exposure_level: str = Field(
        default=ExposureLevel.NEGLIGIBLE.value,
        description="Exposure level classification",
    )
    defence_positions: List[str] = Field(
        default_factory=list,
        description="Available defence positions",
    )
    defence_strength_score: Decimal = Field(
        default=Decimal("0"),
        description="Defence strength score (0-100)",
    )
    dd_compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Due diligence compliance score (0-100)",
    )
    mitigating_factors: List[str] = Field(
        default_factory=list,
        description="Factors that mitigate liability",
    )
    aggravating_factors: List[str] = Field(
        default_factory=list,
        description="Factors that aggravate liability",
    )
    estimated_exposure_eur: Decimal = Field(
        default=Decimal("0"),
        description="Estimated financial exposure (EUR)",
    )
    residual_exposure_eur: Decimal = Field(
        default=Decimal("0"),
        description="Residual exposure after defences (EUR)",
    )
    limitation_expired: bool = Field(
        default=False,
        description="Whether the limitation period has expired",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for this scenario",
    )


class InsuranceAdequacy(BaseModel):
    """Insurance adequacy assessment result."""
    total_exposure_eur: Decimal = Field(
        default=Decimal("0"), description="Total exposure (EUR)"
    )
    total_insurance_eur: Decimal = Field(
        default=Decimal("0"), description="Total insurance coverage (EUR)"
    )
    coverage_ratio_pct: Decimal = Field(
        default=Decimal("0"), description="Coverage ratio (%)"
    )
    gap_eur: Decimal = Field(
        default=Decimal("0"), description="Insurance gap (EUR)"
    )
    is_adequate: bool = Field(
        default=False, description="Whether coverage is adequate"
    )
    assessment: str = Field(
        default="", description="Assessment summary"
    )


class CivilLiabilityResult(BaseModel):
    """Complete civil liability assessment result per Art 29 CSDDD.

    Aggregates all scenario assessments, total exposure analysis,
    defence evaluations, and insurance adequacy into a single
    result with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version used"
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp of assessment (UTC)"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    scenarios_count: int = Field(
        default=0, description="Number of scenarios assessed"
    )
    scenario_assessments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-scenario assessment results",
    )
    total_exposure_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total estimated financial exposure (EUR)",
    )
    total_residual_exposure_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total residual exposure after defences (EUR)",
    )
    highest_risk_scenario: Dict[str, Any] = Field(
        default_factory=dict,
        description="The scenario with highest exposure",
    )
    exposure_by_domain: Dict[str, str] = Field(
        default_factory=dict,
        description="Exposure by impact domain (EUR)",
    )
    exposure_by_severity: Dict[str, str] = Field(
        default_factory=dict,
        description="Exposure by impact severity (EUR)",
    )
    overall_exposure_level: str = Field(
        default=ExposureLevel.NEGLIGIBLE.value,
        description="Overall exposure level",
    )
    defence_strength_score: Decimal = Field(
        default=Decimal("0"),
        description="Average defence strength score (0-100)",
    )
    dd_compliance_score: Decimal = Field(
        default=Decimal("0"),
        description="Average DD compliance score (0-100)",
    )
    insurance_adequacy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Insurance adequacy assessment",
    )
    triggers_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each liability trigger across scenarios",
    )
    compliance_gaps: List[str] = Field(
        default_factory=list,
        description="Identified compliance gaps",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for reducing exposure",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and assessment steps",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CivilLiabilityEngine:
    """CSDDD Article 29 civil liability assessment engine.

    Provides deterministic, zero-hallucination assessments for
    civil liability exposure under Art 29 CSDDD:

    - Liability trigger identification (Art 7-8 failures)
    - Defence position evaluation (compliance, business partner)
    - Financial exposure estimation (severity-weighted)
    - Insurance adequacy assessment (coverage vs exposure)
    - Due diligence compliance scoring

    All calculations use Decimal arithmetic for reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = CivilLiabilityEngine()
        result = engine.assess_liability(
            scenarios=[LiabilityScenario(...)],
        )
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Main Assessment Method                                               #
    # ------------------------------------------------------------------ #

    def assess_liability(
        self,
        scenarios: List[LiabilityScenario],
        entity_name: str = "",
        reporting_year: int = 0,
    ) -> CivilLiabilityResult:
        """Perform a complete civil liability assessment.

        Orchestrates evaluation of all scenarios, calculates total
        exposure, evaluates defences, and assesses insurance adequacy.

        Args:
            scenarios: List of LiabilityScenario instances.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.

        Returns:
            CivilLiabilityResult with complete assessment and provenance.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing civil liability: %d scenarios, entity=%s, year=%d",
            len(scenarios), entity_name, reporting_year,
        )

        # Step 1: Assess each scenario
        scenario_assessments: List[Dict[str, Any]] = []
        for scenario in scenarios:
            assessment = self.assess_scenario(scenario)
            scenario_assessments.append(assessment)

        # Step 2: Calculate total exposure
        total_exposure = sum(
            _decimal(a.get("estimated_exposure_eur", 0))
            for a in scenario_assessments
        )
        total_residual = sum(
            _decimal(a.get("residual_exposure_eur", 0))
            for a in scenario_assessments
        )

        # Step 3: Find highest risk scenario
        highest_risk = {}
        if scenario_assessments:
            highest_risk = max(
                scenario_assessments,
                key=lambda a: _decimal(
                    a.get("estimated_exposure_eur", 0)
                ),
            )

        # Step 4: Exposure by domain
        exposure_by_domain = self._aggregate_by_domain(
            scenarios, scenario_assessments
        )

        # Step 5: Exposure by severity
        exposure_by_severity = self._aggregate_by_severity(
            scenarios, scenario_assessments
        )

        # Step 6: Overall exposure level
        overall_exposure_level = self._classify_exposure(
            total_residual
        )

        # Step 7: Average defence strength
        avg_defence = Decimal("0")
        if scenario_assessments:
            defence_scores = [
                _decimal(a.get("defence_strength_score", 0))
                for a in scenario_assessments
            ]
            avg_defence = _round_val(
                sum(defence_scores) / _decimal(len(defence_scores)),
                1,
            )

        # Step 8: Average DD compliance
        avg_dd = Decimal("0")
        if scenario_assessments:
            dd_scores = [
                _decimal(a.get("dd_compliance_score", 0))
                for a in scenario_assessments
            ]
            avg_dd = _round_val(
                sum(dd_scores) / _decimal(len(dd_scores)), 1
            )

        # Step 9: Insurance adequacy
        total_insurance = sum(
            s.insurance_coverage_eur for s in scenarios
        )
        insurance_result = self.assess_insurance_adequacy(
            total_residual, total_insurance
        )

        # Step 10: Triggers summary
        triggers_summary = self._summarise_triggers(
            scenario_assessments
        )

        # Step 11: Compliance gaps
        compliance_gaps = self._identify_compliance_gaps(
            scenarios, scenario_assessments, insurance_result,
        )

        # Step 12: Recommendations
        recommendations = self._generate_recommendations(
            scenarios, scenario_assessments, insurance_result,
            avg_defence, avg_dd, overall_exposure_level,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CivilLiabilityResult(
            entity_name=entity_name,
            reporting_year=reporting_year,
            scenarios_count=len(scenarios),
            scenario_assessments=scenario_assessments,
            total_exposure_eur=_round_val(total_exposure, 2),
            total_residual_exposure_eur=_round_val(total_residual, 2),
            highest_risk_scenario=highest_risk,
            exposure_by_domain=exposure_by_domain,
            exposure_by_severity=exposure_by_severity,
            overall_exposure_level=overall_exposure_level,
            defence_strength_score=avg_defence,
            dd_compliance_score=avg_dd,
            insurance_adequacy=insurance_result,
            triggers_summary=triggers_summary,
            compliance_gaps=compliance_gaps,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Civil liability assessed: total_exposure=EUR %.2f, "
            "residual=EUR %.2f, level=%s, defence=%.1f%%, "
            "scenarios=%d, hash=%s",
            float(total_exposure), float(total_residual),
            overall_exposure_level, float(avg_defence),
            len(scenarios), result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Single Scenario Assessment                                           #
    # ------------------------------------------------------------------ #

    def assess_scenario(
        self, scenario: LiabilityScenario
    ) -> Dict[str, Any]:
        """Assess a single liability scenario under Art 29 CSDDD.

        Identifies triggers, evaluates defences, estimates exposure,
        and generates scenario-specific recommendations.

        Args:
            scenario: LiabilityScenario to assess.

        Returns:
            Dict with complete scenario assessment.
        """
        # Identify triggers
        triggers = self.identify_triggers(scenario)

        # Check limitation period
        limitation_expired = (
            scenario.years_since_impact
            >= scenario.limitation_period_years
        )

        # Evaluate defences
        defence_result = self.evaluate_defences(scenario)
        defence_positions = defence_result["positions"]
        defence_strength = _decimal(
            defence_result["defence_strength_score"]
        )

        # DD compliance score
        dd_compliance = self._calculate_dd_compliance(scenario)

        # Estimate exposure
        exposure_result = self.estimate_exposure(scenario)
        estimated_exposure = _decimal(
            exposure_result["estimated_exposure_eur"]
        )

        # Residual exposure after defences
        # Higher defence strength = lower residual exposure
        defence_reduction = _safe_divide(
            defence_strength, Decimal("100")
        )
        residual_exposure = _round_val(
            estimated_exposure * (Decimal("1") - defence_reduction),
            2,
        )
        if limitation_expired:
            residual_exposure = Decimal("0")

        # Exposure level
        exposure_level = self._classify_exposure(residual_exposure)

        # Mitigating and aggravating factors
        mitigating = self._identify_mitigating_factors(scenario)
        aggravating = self._identify_aggravating_factors(scenario)

        # Recommendations
        recs = self._scenario_recommendations(
            scenario, triggers, defence_positions, exposure_level,
        )

        result = {
            "scenario_id": scenario.scenario_id,
            "adverse_impact": scenario.adverse_impact[:200],
            "impact_domain": scenario.impact_domain.value,
            "impact_severity": scenario.impact_severity.value,
            "triggers": triggers,
            "trigger_count": len(triggers),
            "exposure_level": exposure_level,
            "defence_positions": defence_positions,
            "defence_strength_score": defence_strength,
            "dd_compliance_score": dd_compliance,
            "mitigating_factors": mitigating,
            "aggravating_factors": aggravating,
            "estimated_exposure_eur": _round_val(estimated_exposure, 2),
            "residual_exposure_eur": residual_exposure,
            "limitation_expired": limitation_expired,
            "limitation_period_years": scenario.limitation_period_years,
            "years_since_impact": scenario.years_since_impact,
            "number_of_affected_persons": scenario.number_of_affected_persons,
            "joint_liability_parties": scenario.joint_liability_parties,
            "recommendations": recs,
        }

        logger.info(
            "Scenario %s: triggers=%d, exposure=%s (EUR %.2f), "
            "defence=%.1f%%, residual=EUR %.2f",
            scenario.scenario_id, len(triggers), exposure_level,
            float(estimated_exposure), float(defence_strength),
            float(residual_exposure),
        )

        return result

    # ------------------------------------------------------------------ #
    # Trigger Identification                                               #
    # ------------------------------------------------------------------ #

    def identify_triggers(
        self, scenario: LiabilityScenario
    ) -> List[str]:
        """Identify liability triggers for a scenario under Art 29.

        Evaluates which of the five liability triggers are present
        based on the scenario's due diligence and response posture.

        Args:
            scenario: LiabilityScenario to evaluate.

        Returns:
            List of LiabilityTrigger values that are active.
        """
        triggers: List[str] = []

        # Failure to prevent (Art 7)
        if not scenario.prevention_measures_taken:
            triggers.append(LiabilityTrigger.FAILURE_TO_PREVENT.value)

        # Failure to mitigate (Art 8)
        if not scenario.mitigation_actions_taken:
            triggers.append(LiabilityTrigger.FAILURE_TO_MITIGATE.value)

        # Failure to remediate
        if not scenario.remediation_provided:
            triggers.append(
                LiabilityTrigger.FAILURE_TO_REMEDIATE.value
            )

        # Inadequate due diligence
        if not scenario.due_diligence_performed:
            triggers.append(
                LiabilityTrigger.INADEQUATE_DUE_DILIGENCE.value
            )

        # Contractual breach
        if (
            scenario.contractual_assurances_obtained
            and not scenario.verification_measures_applied
        ):
            triggers.append(
                LiabilityTrigger.CONTRACTUAL_BREACH.value
            )

        return triggers

    # ------------------------------------------------------------------ #
    # Defence Evaluation                                                   #
    # ------------------------------------------------------------------ #

    def evaluate_defences(
        self, scenario: LiabilityScenario
    ) -> Dict[str, Any]:
        """Evaluate available defence positions for a scenario.

        Determines which defences under Art 29 CSDDD are available
        and calculates a defence strength score.

        Args:
            scenario: LiabilityScenario to evaluate.

        Returns:
            Dict with available positions, strength score, and details.
        """
        positions: List[str] = []
        strength_factors: List[Decimal] = []

        # Full compliance defence
        full_compliance = (
            scenario.due_diligence_performed
            and scenario.prevention_measures_taken
            and scenario.mitigation_actions_taken
            and scenario.contractual_assurances_obtained
            and scenario.verification_measures_applied
            and scenario.stakeholders_consulted
        )
        if full_compliance:
            positions.append(DefencePosition.FULL_COMPLIANCE.value)
            strength_factors.append(
                DEFENCE_REDUCTION_FACTORS[
                    DefencePosition.FULL_COMPLIANCE.value
                ]
            )

        # Partial compliance
        dd_elements_met = sum([
            scenario.due_diligence_performed,
            scenario.prevention_measures_taken,
            scenario.mitigation_actions_taken,
            scenario.contractual_assurances_obtained,
            scenario.verification_measures_applied,
            scenario.stakeholders_consulted,
        ])
        if not full_compliance and dd_elements_met >= 3:
            positions.append(
                DefencePosition.PARTIAL_COMPLIANCE.value
            )
            strength_factors.append(
                DEFENCE_REDUCTION_FACTORS[
                    DefencePosition.PARTIAL_COMPLIANCE.value
                ]
            )
        elif not full_compliance and dd_elements_met >= 1:
            positions.append(
                DefencePosition.REASONABLE_EFFORTS.value
            )
            strength_factors.append(
                DEFENCE_REDUCTION_FACTORS[
                    DefencePosition.REASONABLE_EFFORTS.value
                ]
            )

        # Force majeure (if damage caused only by business partner)
        if scenario.caused_by_business_partner_only:
            positions.append(DefencePosition.FORCE_MAJEURE.value)
            strength_factors.append(
                DEFENCE_REDUCTION_FACTORS[
                    DefencePosition.FORCE_MAJEURE.value
                ]
            )

        # Limitation period
        if (
            scenario.years_since_impact
            >= scenario.limitation_period_years
        ):
            positions.append(
                DefencePosition.LIMITATION_EXPIRED.value
            )
            strength_factors.append(
                DEFENCE_REDUCTION_FACTORS[
                    DefencePosition.LIMITATION_EXPIRED.value
                ]
            )

        # Defence strength score: max of all available factor reductions
        # expressed as percentage
        defence_strength = Decimal("0")
        if strength_factors:
            defence_strength = _round_val(
                max(strength_factors) * Decimal("100"), 1
            )

        result = {
            "positions": positions,
            "positions_count": len(positions),
            "defence_strength_score": defence_strength,
            "dd_elements_met": dd_elements_met,
            "dd_elements_total": 6,
            "full_compliance": full_compliance,
            "business_partner_only": (
                scenario.caused_by_business_partner_only
            ),
            "limitation_expired": (
                scenario.years_since_impact
                >= scenario.limitation_period_years
            ),
        }

        return result

    # ------------------------------------------------------------------ #
    # Exposure Estimation                                                  #
    # ------------------------------------------------------------------ #

    def estimate_exposure(
        self, scenario: LiabilityScenario
    ) -> Dict[str, Any]:
        """Estimate financial exposure for a single liability scenario.

        Applies severity multipliers to the base damage estimate
        and adjusts for class action risk, affected persons, and
        joint liability.

        Args:
            scenario: LiabilityScenario to estimate.

        Returns:
            Dict with exposure breakdown.
        """
        base_damage = scenario.damage_estimate_eur

        # Apply severity multiplier
        severity_multiplier = SEVERITY_MULTIPLIERS.get(
            scenario.impact_severity.value, Decimal("1.0")
        )
        adjusted_damage = base_damage * severity_multiplier

        # Class action risk premium (additional 50% exposure)
        class_action_premium = Decimal("0")
        if scenario.is_class_action_risk:
            class_action_premium = _round_val(
                adjusted_damage * Decimal("0.50"), 2
            )
            adjusted_damage += class_action_premium

        # Joint liability adjustment (divide by number of parties)
        joint_liability_adjustment = Decimal("0")
        company_share = adjusted_damage
        if scenario.joint_liability_parties > 0:
            total_parties = scenario.joint_liability_parties + 1
            company_share = _round_val(
                adjusted_damage / _decimal(total_parties), 2
            )
            joint_liability_adjustment = adjusted_damage - company_share

        # Regulatory enforcement premium (additional 25%)
        regulatory_premium = Decimal("0")
        if scenario.regulatory_enforcement_active:
            regulatory_premium = _round_val(
                company_share * Decimal("0.25"), 2
            )
            company_share += regulatory_premium

        estimated_exposure = _round_val(company_share, 2)

        result = {
            "base_damage_eur": str(_round_val(base_damage, 2)),
            "severity_multiplier": str(severity_multiplier),
            "severity_adjusted_eur": str(
                _round_val(base_damage * severity_multiplier, 2)
            ),
            "class_action_premium_eur": str(class_action_premium),
            "joint_liability_adjustment_eur": str(
                _round_val(joint_liability_adjustment, 2)
            ),
            "regulatory_premium_eur": str(regulatory_premium),
            "estimated_exposure_eur": estimated_exposure,
            "is_class_action_risk": scenario.is_class_action_risk,
            "joint_liability_parties": scenario.joint_liability_parties,
        }

        return result

    # ------------------------------------------------------------------ #
    # Insurance Adequacy                                                   #
    # ------------------------------------------------------------------ #

    def assess_insurance_adequacy(
        self,
        total_exposure: Decimal,
        insurance_coverage: Decimal,
    ) -> Dict[str, Any]:
        """Assess whether insurance coverage is adequate for the exposure.

        Compares total insurance coverage to total residual exposure
        and classifies the adequacy level.

        Args:
            total_exposure: Total residual exposure (EUR).
            insurance_coverage: Total insurance coverage (EUR).

        Returns:
            Dict with insurance adequacy assessment.
        """
        gap = total_exposure - insurance_coverage
        if gap < Decimal("0"):
            gap = Decimal("0")

        coverage_ratio = Decimal("0")
        if total_exposure > Decimal("0"):
            coverage_ratio = _round_val(
                insurance_coverage / total_exposure * Decimal("100"),
                1,
            )

        is_adequate = insurance_coverage >= total_exposure

        if coverage_ratio >= Decimal("100"):
            assessment = "fully_covered"
        elif coverage_ratio >= Decimal("75"):
            assessment = "substantially_covered"
        elif coverage_ratio >= Decimal("50"):
            assessment = "partially_covered"
        elif coverage_ratio > Decimal("0"):
            assessment = "significantly_underinsured"
        else:
            assessment = "uninsured"

        result = {
            "total_exposure_eur": str(_round_val(total_exposure, 2)),
            "total_insurance_eur": str(
                _round_val(insurance_coverage, 2)
            ),
            "coverage_ratio_pct": coverage_ratio,
            "gap_eur": str(_round_val(gap, 2)),
            "is_adequate": is_adequate,
            "assessment": assessment,
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Insurance adequacy: coverage=%.1f%%, gap=EUR %.2f, "
            "assessment=%s",
            float(coverage_ratio), float(gap), assessment,
        )

        return result

    # ------------------------------------------------------------------ #
    # DD Compliance Score                                                   #
    # ------------------------------------------------------------------ #

    def _calculate_dd_compliance(
        self, scenario: LiabilityScenario
    ) -> Decimal:
        """Calculate due diligence compliance score for a scenario.

        Weighted scoring based on which DD elements were performed.

        Args:
            scenario: LiabilityScenario to score.

        Returns:
            DD compliance score as Decimal (0-100).
        """
        elements = {
            "risk_identification": scenario.risk_identified,
            "prevention_measures": scenario.prevention_measures_taken,
            "mitigation_actions": scenario.mitigation_actions_taken,
            "contractual_assurances": (
                scenario.contractual_assurances_obtained
            ),
            "verification_measures": (
                scenario.verification_measures_applied
            ),
            "stakeholder_consultation": scenario.stakeholders_consulted,
        }

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for element, performed in elements.items():
            weight = DD_ELEMENT_WEIGHTS.get(element, Decimal("0"))
            total_weight += weight
            if performed:
                weighted_sum += weight

        score = _round_val(
            _safe_divide(weighted_sum, total_weight) * Decimal("100"),
            1,
        )

        return score

    # ------------------------------------------------------------------ #
    # Exposure Classification                                              #
    # ------------------------------------------------------------------ #

    def _classify_exposure(self, exposure_eur: Decimal) -> str:
        """Classify exposure level based on financial thresholds.

        Args:
            exposure_eur: Financial exposure amount (EUR).

        Returns:
            ExposureLevel value string.
        """
        if exposure_eur >= EXPOSURE_THRESHOLDS[
            ExposureLevel.CRITICAL.value
        ]:
            return ExposureLevel.CRITICAL.value
        elif exposure_eur >= EXPOSURE_THRESHOLDS[
            ExposureLevel.HIGH.value
        ]:
            return ExposureLevel.HIGH.value
        elif exposure_eur >= EXPOSURE_THRESHOLDS[
            ExposureLevel.MEDIUM.value
        ]:
            return ExposureLevel.MEDIUM.value
        elif exposure_eur >= EXPOSURE_THRESHOLDS[
            ExposureLevel.LOW.value
        ]:
            return ExposureLevel.LOW.value
        else:
            return ExposureLevel.NEGLIGIBLE.value

    # ------------------------------------------------------------------ #
    # Factor Analysis                                                      #
    # ------------------------------------------------------------------ #

    def _identify_mitigating_factors(
        self, scenario: LiabilityScenario
    ) -> List[str]:
        """Identify mitigating factors for a liability scenario.

        Args:
            scenario: LiabilityScenario to analyse.

        Returns:
            List of mitigating factor descriptions.
        """
        factors: List[str] = []

        if scenario.due_diligence_performed:
            factors.append(
                "Due diligence was performed for this risk area."
            )
        if scenario.prevention_measures_taken:
            factors.append(
                "Prevention measures were implemented per Art 7."
            )
        if scenario.mitigation_actions_taken:
            factors.append(
                "Mitigation actions were taken per Art 8."
            )
        if scenario.contractual_assurances_obtained:
            factors.append(
                "Contractual assurances were obtained from "
                "business partners."
            )
        if scenario.verification_measures_applied:
            factors.append(
                "Verification measures were applied to check "
                "compliance."
            )
        if scenario.stakeholders_consulted:
            factors.append(
                "Affected stakeholders were consulted."
            )
        if scenario.remediation_provided:
            factors.append("Remediation was provided to affected parties.")
        if scenario.remediation_adequate:
            factors.append(
                "Remediation is considered adequate."
            )
        if scenario.caused_by_business_partner_only:
            factors.append(
                "Damage was caused only by a business partner "
                "(Art 29(3) defence)."
            )
        if scenario.joint_liability_parties > 0:
            factors.append(
                f"Joint liability with {scenario.joint_liability_parties} "
                f"other parties reduces individual exposure."
            )

        return factors

    def _identify_aggravating_factors(
        self, scenario: LiabilityScenario
    ) -> List[str]:
        """Identify aggravating factors for a liability scenario.

        Args:
            scenario: LiabilityScenario to analyse.

        Returns:
            List of aggravating factor descriptions.
        """
        factors: List[str] = []

        if not scenario.due_diligence_performed:
            factors.append(
                "No due diligence was performed for this risk."
            )
        if not scenario.prevention_measures_taken:
            factors.append(
                "No prevention measures were implemented."
            )
        if scenario.prior_warnings_received:
            factors.append(
                "Prior warnings about the risk were received "
                "but not adequately addressed."
            )
        if scenario.is_class_action_risk:
            factors.append(
                "Risk of class action litigation increases "
                "potential exposure."
            )
        if scenario.regulatory_enforcement_active:
            factors.append(
                "Active regulatory enforcement increases "
                "reputational and financial risk."
            )
        if (
            scenario.impact_severity
            in (ImpactSeverity.CATASTROPHIC, ImpactSeverity.SEVERE)
        ):
            factors.append(
                f"Impact severity is {scenario.impact_severity.value}, "
                f"which attracts higher damages."
            )
        if scenario.number_of_affected_persons > 100:
            factors.append(
                f"Large number of affected persons "
                f"({scenario.number_of_affected_persons}) increases "
                f"aggregate exposure."
            )
        if not scenario.remediation_provided:
            factors.append(
                "No remediation has been provided to affected parties."
            )
        if scenario.risk_identified and not scenario.prevention_measures_taken:
            factors.append(
                "Risk was identified but no prevention measures were "
                "taken, suggesting negligent inaction."
            )

        return factors

    # ------------------------------------------------------------------ #
    # Aggregation Helpers                                                  #
    # ------------------------------------------------------------------ #

    def _aggregate_by_domain(
        self,
        scenarios: List[LiabilityScenario],
        assessments: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Aggregate exposure by impact domain.

        Args:
            scenarios: Original scenarios.
            assessments: Scenario assessments.

        Returns:
            Dict mapping domain names to total exposure (EUR string).
        """
        domain_totals: Dict[str, Decimal] = {}
        for scenario, assessment in zip(scenarios, assessments):
            domain = scenario.impact_domain.value
            exposure = _decimal(
                assessment.get("estimated_exposure_eur", 0)
            )
            domain_totals[domain] = (
                domain_totals.get(domain, Decimal("0")) + exposure
            )

        return {
            k: str(_round_val(v, 2))
            for k, v in sorted(domain_totals.items())
        }

    def _aggregate_by_severity(
        self,
        scenarios: List[LiabilityScenario],
        assessments: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Aggregate exposure by impact severity.

        Args:
            scenarios: Original scenarios.
            assessments: Scenario assessments.

        Returns:
            Dict mapping severity levels to total exposure (EUR string).
        """
        severity_totals: Dict[str, Decimal] = {}
        for scenario, assessment in zip(scenarios, assessments):
            severity = scenario.impact_severity.value
            exposure = _decimal(
                assessment.get("estimated_exposure_eur", 0)
            )
            severity_totals[severity] = (
                severity_totals.get(severity, Decimal("0")) + exposure
            )

        return {
            k: str(_round_val(v, 2))
            for k, v in sorted(severity_totals.items())
        }

    def _summarise_triggers(
        self, assessments: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Summarise liability trigger counts across all scenarios.

        Args:
            assessments: Scenario assessment results.

        Returns:
            Dict mapping trigger names to their occurrence count.
        """
        trigger_counts: Dict[str, int] = {}
        for assessment in assessments:
            for trigger in assessment.get("triggers", []):
                trigger_counts[trigger] = (
                    trigger_counts.get(trigger, 0) + 1
                )
        return trigger_counts

    # ------------------------------------------------------------------ #
    # Scenario Recommendations                                             #
    # ------------------------------------------------------------------ #

    def _scenario_recommendations(
        self,
        scenario: LiabilityScenario,
        triggers: List[str],
        defence_positions: List[str],
        exposure_level: str,
    ) -> List[str]:
        """Generate recommendations for a single scenario.

        Args:
            scenario: LiabilityScenario.
            triggers: Identified triggers.
            defence_positions: Available defences.
            exposure_level: Classified exposure level.

        Returns:
            List of recommendations.
        """
        recs: List[str] = []

        if LiabilityTrigger.FAILURE_TO_PREVENT.value in triggers:
            recs.append(
                "Implement prevention measures per Art 7 CSDDD to "
                "address the identified adverse impact."
            )

        if LiabilityTrigger.FAILURE_TO_MITIGATE.value in triggers:
            recs.append(
                "Take mitigation actions per Art 8 CSDDD to bring "
                "the adverse impact to an end or minimise its extent."
            )

        if LiabilityTrigger.FAILURE_TO_REMEDIATE.value in triggers:
            recs.append(
                "Provide remediation to affected parties, including "
                "financial compensation and restoration measures."
            )

        if LiabilityTrigger.INADEQUATE_DUE_DILIGENCE.value in triggers:
            recs.append(
                "Strengthen due diligence processes to identify, "
                "assess, and address adverse impacts proactively."
            )

        if not scenario.contractual_assurances_obtained:
            recs.append(
                "Obtain contractual assurances from business partners "
                "requiring compliance with CSDDD standards."
            )

        if not scenario.verification_measures_applied:
            recs.append(
                "Apply verification measures (audits, assessments) "
                "to confirm business partner compliance."
            )

        if not scenario.stakeholders_consulted:
            recs.append(
                "Consult affected stakeholders as part of the "
                "due diligence process per Art 11."
            )

        if (
            exposure_level in (
                ExposureLevel.CRITICAL.value,
                ExposureLevel.HIGH.value,
            )
            and scenario.insurance_coverage_eur < scenario.damage_estimate_eur
        ):
            recs.append(
                "Review and increase insurance coverage to match "
                "the estimated exposure for this scenario."
            )

        # Cap per-scenario recommendations
        if len(recs) > 8:
            recs = recs[:8]

        return recs

    # ------------------------------------------------------------------ #
    # Compliance Gap Identification                                        #
    # ------------------------------------------------------------------ #

    def _identify_compliance_gaps(
        self,
        scenarios: List[LiabilityScenario],
        assessments: List[Dict[str, Any]],
        insurance_result: Dict[str, Any],
    ) -> List[str]:
        """Identify compliance gaps across all scenarios.

        Args:
            scenarios: All liability scenarios.
            assessments: Scenario assessment results.
            insurance_result: Insurance adequacy result.

        Returns:
            List of compliance gap descriptions.
        """
        gaps: List[str] = []

        # Count DD failures
        no_dd_count = sum(
            1 for s in scenarios if not s.due_diligence_performed
        )
        if no_dd_count > 0:
            gaps.append(
                f"Art 7-8: Due diligence not performed for "
                f"{no_dd_count} scenario(s), exposing the company "
                f"to liability under Art 29."
            )

        # Prevention gaps
        no_prevention = sum(
            1 for s in scenarios if not s.prevention_measures_taken
        )
        if no_prevention > 0:
            gaps.append(
                f"Art 7: Prevention measures not taken for "
                f"{no_prevention} scenario(s)."
            )

        # Mitigation gaps
        no_mitigation = sum(
            1 for s in scenarios if not s.mitigation_actions_taken
        )
        if no_mitigation > 0:
            gaps.append(
                f"Art 8: Mitigation actions not taken for "
                f"{no_mitigation} scenario(s)."
            )

        # Remediation gaps
        no_remediation = sum(
            1 for s in scenarios
            if not s.remediation_provided
            and s.damage_estimate_eur > Decimal("0")
        )
        if no_remediation > 0:
            gaps.append(
                f"Remediation not provided for {no_remediation} "
                f"scenario(s) with estimated damages."
            )

        # Stakeholder consultation gaps
        no_consultation = sum(
            1 for s in scenarios if not s.stakeholders_consulted
        )
        if no_consultation > 0:
            gaps.append(
                f"Art 11: Stakeholders not consulted for "
                f"{no_consultation} scenario(s)."
            )

        # Insurance gap
        if not insurance_result.get("is_adequate", True):
            gap_eur = insurance_result.get("gap_eur", "0")
            gaps.append(
                f"Insurance coverage is inadequate. Gap of "
                f"EUR {gap_eur}."
            )

        # High/critical exposure without full compliance
        critical_without_defence = sum(
            1 for a in assessments
            if a.get("exposure_level") in (
                ExposureLevel.CRITICAL.value,
                ExposureLevel.HIGH.value,
            )
            and DefencePosition.FULL_COMPLIANCE.value
            not in a.get("defence_positions", [])
        )
        if critical_without_defence > 0:
            gaps.append(
                f"{critical_without_defence} high/critical exposure "
                f"scenario(s) lack full compliance defence."
            )

        return gaps

    # ------------------------------------------------------------------ #
    # Overall Recommendations                                              #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        scenarios: List[LiabilityScenario],
        assessments: List[Dict[str, Any]],
        insurance_result: Dict[str, Any],
        avg_defence: Decimal,
        avg_dd: Decimal,
        overall_exposure: str,
    ) -> List[str]:
        """Generate overall recommendations for reducing liability exposure.

        Args:
            scenarios: All liability scenarios.
            assessments: Scenario assessment results.
            insurance_result: Insurance adequacy result.
            avg_defence: Average defence strength score.
            avg_dd: Average DD compliance score.
            overall_exposure: Overall exposure level.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: List[str] = []

        # DD compliance recommendations
        if avg_dd < Decimal("50"):
            recommendations.append(
                "Due diligence compliance is below 50%. Urgently "
                "strengthen DD processes across all risk areas to "
                "build Art 29 defence positions."
            )
        elif avg_dd < Decimal("80"):
            recommendations.append(
                "Due diligence compliance is moderate. Close gaps "
                "in risk identification, prevention, and verification "
                "to strengthen defence positions."
            )

        # Defence strength recommendations
        if avg_defence < Decimal("50"):
            recommendations.append(
                "Defence strength is low. Focus on achieving full "
                "compliance with Art 7-8 obligations including "
                "contractual assurances and verification."
            )

        # Exposure-level recommendations
        if overall_exposure == ExposureLevel.CRITICAL.value:
            recommendations.append(
                "CRITICAL exposure level. Immediate action required: "
                "implement prevention/mitigation measures, provide "
                "remediation, and review insurance coverage."
            )
        elif overall_exposure == ExposureLevel.HIGH.value:
            recommendations.append(
                "HIGH exposure level. Priority action needed to "
                "strengthen DD compliance and reduce residual risk."
            )

        # Insurance recommendations
        if not insurance_result.get("is_adequate", True):
            recommendations.append(
                "Insurance coverage is inadequate for the estimated "
                "exposure. Increase coverage or reduce exposure "
                "through stronger DD compliance."
            )

        # Remediation recommendations
        unremediating = [
            s for s in scenarios
            if not s.remediation_provided
            and s.damage_estimate_eur > Decimal("0")
        ]
        if unremediating:
            recommendations.append(
                f"Provide remediation for {len(unremediating)} "
                f"scenario(s) with estimated damages to reduce "
                f"liability exposure and demonstrate good faith."
            )

        # Contractual assurances
        no_contracts = sum(
            1 for s in scenarios
            if not s.contractual_assurances_obtained
        )
        if no_contracts > 0:
            recommendations.append(
                "Obtain contractual assurances from business partners "
                "for all identified risk scenarios to support "
                "Art 29(3) defence."
            )

        # Verification
        no_verification = sum(
            1 for s in scenarios
            if not s.verification_measures_applied
        )
        if no_verification > 0:
            recommendations.append(
                "Apply verification measures (audits, monitoring) "
                "for all risk scenarios to demonstrate active "
                "compliance oversight."
            )

        # Scenario-level recommendations (top priorities)
        for assessment in assessments:
            if assessment.get("exposure_level") in (
                ExposureLevel.CRITICAL.value,
                ExposureLevel.HIGH.value,
            ):
                for rec in assessment.get("recommendations", [])[:2]:
                    if rec not in recommendations:
                        recommendations.append(rec)

        # Cap at 15
        if len(recommendations) > 15:
            recommendations = recommendations[:15]

        return recommendations

    # ------------------------------------------------------------------ #
    # Period Comparison                                                     #
    # ------------------------------------------------------------------ #

    def compare_periods(
        self,
        current: CivilLiabilityResult,
        previous: CivilLiabilityResult,
    ) -> Dict[str, Any]:
        """Compare liability exposure across two reporting periods.

        Args:
            current: Current period result.
            previous: Previous period result.

        Returns:
            Dict with period-over-period changes and provenance.
        """
        comparison = {
            "current_period": current.reporting_year,
            "previous_period": previous.reporting_year,
            "scenarios_change": (
                current.scenarios_count - previous.scenarios_count
            ),
            "total_exposure_change_eur": str(_round_val(
                current.total_exposure_eur - previous.total_exposure_eur,
                2,
            )),
            "residual_exposure_change_eur": str(_round_val(
                current.total_residual_exposure_eur
                - previous.total_residual_exposure_eur,
                2,
            )),
            "defence_strength_change_pp": _round_val(
                current.defence_strength_score
                - previous.defence_strength_score,
                1,
            ),
            "dd_compliance_change_pp": _round_val(
                current.dd_compliance_score
                - previous.dd_compliance_score,
                1,
            ),
            "exposure_level_change": {
                "from": previous.overall_exposure_level,
                "to": current.overall_exposure_level,
            },
            "direction": (
                "improving"
                if current.total_residual_exposure_eur
                < previous.total_residual_exposure_eur
                else (
                    "stable"
                    if current.total_residual_exposure_eur
                    == previous.total_residual_exposure_eur
                    else "deteriorating"
                )
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison
