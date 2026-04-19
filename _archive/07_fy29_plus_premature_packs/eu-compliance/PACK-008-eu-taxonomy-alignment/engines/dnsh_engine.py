"""
DNSH Assessment Engine - PACK-008 EU Taxonomy Alignment

This module evaluates Do No Significant Harm (DNSH) criteria for EU Taxonomy
alignment.  For each non-SC environmental objective, the engine checks whether
the economic activity causes significant harm according to the criteria
defined in the Climate and Environmental Delegated Acts.

Example:
    >>> engine = DNSHAssessmentEngine()
    >>> result = engine.assess_dnsh(
    ...     activity_id="CCM-4.1",
    ...     sc_objective=EnvironmentalObjective.CCM,
    ...     data={"climate_risk_assessment_completed": True, "water_recycling_pct": 85.0},
    ... )
    >>> print(f"DNSH pass: {result.overall_pass}, objectives failed: {result.failed_objectives}")
"""

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""
    CCM = "CCM"
    CCA = "CCA"
    WTR = "WTR"
    CE = "CE"
    PPC = "PPC"
    BIO = "BIO"


OBJECTIVE_NAMES: Dict[str, str] = {
    "CCM": "Climate Change Mitigation",
    "CCA": "Climate Change Adaptation",
    "WTR": "Sustainable Use of Water and Marine Resources",
    "CE": "Transition to a Circular Economy",
    "PPC": "Pollution Prevention and Control",
    "BIO": "Protection and Restoration of Biodiversity and Ecosystems",
}


class DNSHStatus(str, Enum):
    """Status of DNSH check for a single objective."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DNSHCriterion(BaseModel):
    """A single DNSH criterion for one objective."""

    criterion_id: str = Field(..., description="Unique criterion identifier")
    dnsh_objective: EnvironmentalObjective = Field(
        ..., description="Objective this DNSH criterion protects"
    )
    metric_key: str = Field(..., description="Key expected in input data")
    description: str = Field(..., description="Human-readable description")
    check_type: str = Field(
        ..., description="BOOLEAN, THRESHOLD_MIN, THRESHOLD_MAX, QUALITATIVE"
    )
    threshold_value: Optional[float] = Field(
        None, description="Numeric threshold (if applicable)"
    )
    unit: str = Field(default="", description="Unit of measurement")
    is_mandatory: bool = Field(default=True, description="Whether criterion is mandatory")
    delegated_act_ref: str = Field(
        default="EU 2021/2139", description="Source Delegated Act"
    )


class ObjectiveDNSHResult(BaseModel):
    """DNSH result for a single environmental objective."""

    objective: EnvironmentalObjective = Field(
        ..., description="Environmental objective assessed"
    )
    objective_name: str = Field(..., description="Full objective name")
    status: DNSHStatus = Field(..., description="DNSH assessment status")
    criteria_total: int = Field(..., ge=0, description="Total criteria checked")
    criteria_passed: int = Field(..., ge=0, description="Criteria passed")
    criteria_failed: int = Field(..., ge=0, description="Criteria failed")
    criteria_no_data: int = Field(..., ge=0, description="Criteria with no data")
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-criterion detail records"
    )
    evidence_required: List[str] = Field(
        default_factory=list,
        description="Evidence documents needed for this objective"
    )


class DNSHResult(BaseModel):
    """Complete DNSH assessment result for an activity."""

    activity_id: str = Field(..., description="Taxonomy activity ID")
    sc_objective: EnvironmentalObjective = Field(
        ..., description="Substantial Contribution objective (excluded from DNSH)"
    )
    overall_pass: bool = Field(
        ..., description="True only if DNSH is passed for ALL non-SC objectives"
    )
    objectives_assessed: int = Field(
        ..., ge=0, description="Number of non-SC objectives assessed"
    )
    objectives_passed: int = Field(..., ge=0, description="Objectives that passed DNSH")
    objectives_failed: int = Field(..., ge=0, description="Objectives that failed DNSH")
    objectives_no_data: int = Field(
        ..., ge=0, description="Objectives with insufficient data"
    )
    failed_objectives: List[str] = Field(
        default_factory=list,
        description="Names of objectives that failed"
    )
    objective_results: Dict[str, ObjectiveDNSHResult] = Field(
        default_factory=dict,
        description="Per-objective DNSH results"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment timestamp"
    )


# ---------------------------------------------------------------------------
# DNSH Criteria Matrix
# ---------------------------------------------------------------------------

# Keys: (sc_objective, dnsh_objective) -> list of criteria
# This matrix defines what criteria apply when an activity contributes to
# sc_objective and must not harm dnsh_objective.

def _build_dnsh_matrix() -> Dict[str, Dict[str, List[DNSHCriterion]]]:
    """
    Build the DNSH criteria matrix.

    Outer key = activity_id (or '*' for generic).
    Inner key = dnsh_objective value.
    Value = list of DNSHCriterion.
    """
    matrix: Dict[str, Dict[str, List[DNSHCriterion]]] = {}

    # ================================================================
    # Generic DNSH criteria applicable to most activities when SC = CCM
    # ================================================================
    ccm_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCA.value: [
            DNSHCriterion(
                criterion_id="DNSH-CCA-01",
                dnsh_objective=EnvironmentalObjective.CCA,
                metric_key="climate_risk_assessment_completed",
                description="Climate risk and vulnerability assessment performed (Appendix A, Section II)",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-CCA-02",
                dnsh_objective=EnvironmentalObjective.CCA,
                metric_key="adaptation_solutions_implemented",
                description="Adaptation solutions implemented to reduce material physical climate risks",
                check_type="BOOLEAN",
            ),
        ],
        EnvironmentalObjective.WTR.value: [
            DNSHCriterion(
                criterion_id="DNSH-WTR-01",
                dnsh_objective=EnvironmentalObjective.WTR,
                metric_key="water_framework_directive_compliance",
                description="Activity complies with Water Framework Directive (2000/60/EC)",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-WTR-02",
                dnsh_objective=EnvironmentalObjective.WTR,
                metric_key="water_recycling_pct",
                description="Water recycling rate meets sector best practice (>= 50%)",
                check_type="THRESHOLD_MIN",
                threshold_value=50.0,
                unit="%",
                is_mandatory=False,
            ),
        ],
        EnvironmentalObjective.CE.value: [
            DNSHCriterion(
                criterion_id="DNSH-CE-01",
                dnsh_objective=EnvironmentalObjective.CE,
                metric_key="waste_hierarchy_compliance",
                description="Waste management follows waste hierarchy (prevention, reuse, recycle)",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-CE-02",
                dnsh_objective=EnvironmentalObjective.CE,
                metric_key="recyclable_design_assessment",
                description="Products designed for durability, recyclability, and repairability",
                check_type="BOOLEAN",
                is_mandatory=False,
            ),
        ],
        EnvironmentalObjective.PPC.value: [
            DNSHCriterion(
                criterion_id="DNSH-PPC-01",
                dnsh_objective=EnvironmentalObjective.PPC,
                metric_key="reach_compliance",
                description="Activity does not involve manufacture/use of REACH restricted substances",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-PPC-02",
                dnsh_objective=EnvironmentalObjective.PPC,
                metric_key="air_emissions_within_limits",
                description="Air pollutant emissions within IED BAT-AEL or national limits",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-PPC-03",
                dnsh_objective=EnvironmentalObjective.PPC,
                metric_key="water_pollutant_discharge_within_limits",
                description="Water pollutant discharges within IED BAT-AEL or national limits",
                check_type="BOOLEAN",
                is_mandatory=False,
            ),
        ],
        EnvironmentalObjective.BIO.value: [
            DNSHCriterion(
                criterion_id="DNSH-BIO-01",
                dnsh_objective=EnvironmentalObjective.BIO,
                metric_key="eia_completed",
                description="Environmental Impact Assessment (EIA) or screening completed",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-BIO-02",
                dnsh_objective=EnvironmentalObjective.BIO,
                metric_key="natura2000_no_impact",
                description="No negative impact on Natura 2000 sites or protected areas",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-BIO-03",
                dnsh_objective=EnvironmentalObjective.BIO,
                metric_key="biodiversity_management_plan",
                description="Biodiversity management plan in place for operations near sensitive areas",
                check_type="BOOLEAN",
                is_mandatory=False,
            ),
        ],
    }

    # Generic set for activities with SC = CCM
    matrix["*_CCM"] = ccm_generic

    # ================================================================
    # Generic DNSH criteria when SC = CCA
    # ================================================================
    cca_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCM.value: [
            DNSHCriterion(
                criterion_id="DNSH-CCM-01",
                dnsh_objective=EnvironmentalObjective.CCM,
                metric_key="ghg_emissions_not_increased",
                description="Activity does not lead to increased GHG emissions",
                check_type="BOOLEAN",
            ),
            DNSHCriterion(
                criterion_id="DNSH-CCM-02",
                dnsh_objective=EnvironmentalObjective.CCM,
                metric_key="fossil_fuel_lock_in_avoided",
                description="No lock-in of fossil-fuel-intensive assets",
                check_type="BOOLEAN",
            ),
        ],
        EnvironmentalObjective.WTR.value: ccm_generic[EnvironmentalObjective.WTR.value],
        EnvironmentalObjective.CE.value: ccm_generic[EnvironmentalObjective.CE.value],
        EnvironmentalObjective.PPC.value: ccm_generic[EnvironmentalObjective.PPC.value],
        EnvironmentalObjective.BIO.value: ccm_generic[EnvironmentalObjective.BIO.value],
    }
    matrix["*_CCA"] = cca_generic

    # ================================================================
    # Generic DNSH criteria when SC = WTR
    # ================================================================
    wtr_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCM.value: cca_generic[EnvironmentalObjective.CCM.value],
        EnvironmentalObjective.CCA.value: ccm_generic[EnvironmentalObjective.CCA.value],
        EnvironmentalObjective.CE.value: ccm_generic[EnvironmentalObjective.CE.value],
        EnvironmentalObjective.PPC.value: ccm_generic[EnvironmentalObjective.PPC.value],
        EnvironmentalObjective.BIO.value: ccm_generic[EnvironmentalObjective.BIO.value],
    }
    matrix["*_WTR"] = wtr_generic

    # ================================================================
    # Generic DNSH criteria when SC = CE
    # ================================================================
    ce_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCM.value: cca_generic[EnvironmentalObjective.CCM.value],
        EnvironmentalObjective.CCA.value: ccm_generic[EnvironmentalObjective.CCA.value],
        EnvironmentalObjective.WTR.value: ccm_generic[EnvironmentalObjective.WTR.value],
        EnvironmentalObjective.PPC.value: ccm_generic[EnvironmentalObjective.PPC.value],
        EnvironmentalObjective.BIO.value: ccm_generic[EnvironmentalObjective.BIO.value],
    }
    matrix["*_CE"] = ce_generic

    # ================================================================
    # Generic DNSH criteria when SC = PPC
    # ================================================================
    ppc_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCM.value: cca_generic[EnvironmentalObjective.CCM.value],
        EnvironmentalObjective.CCA.value: ccm_generic[EnvironmentalObjective.CCA.value],
        EnvironmentalObjective.WTR.value: ccm_generic[EnvironmentalObjective.WTR.value],
        EnvironmentalObjective.CE.value: ccm_generic[EnvironmentalObjective.CE.value],
        EnvironmentalObjective.BIO.value: ccm_generic[EnvironmentalObjective.BIO.value],
    }
    matrix["*_PPC"] = ppc_generic

    # ================================================================
    # Generic DNSH criteria when SC = BIO
    # ================================================================
    bio_generic: Dict[str, List[DNSHCriterion]] = {
        EnvironmentalObjective.CCM.value: cca_generic[EnvironmentalObjective.CCM.value],
        EnvironmentalObjective.CCA.value: ccm_generic[EnvironmentalObjective.CCA.value],
        EnvironmentalObjective.WTR.value: ccm_generic[EnvironmentalObjective.WTR.value],
        EnvironmentalObjective.CE.value: ccm_generic[EnvironmentalObjective.CE.value],
        EnvironmentalObjective.PPC.value: ccm_generic[EnvironmentalObjective.PPC.value],
    }
    matrix["*_BIO"] = bio_generic

    return matrix


DNSH_MATRIX = _build_dnsh_matrix()


# Evidence lookup per DNSH objective
DNSH_EVIDENCE: Dict[str, List[str]] = {
    EnvironmentalObjective.CCM.value: [
        "GHG emissions trend analysis",
        "Fossil fuel phase-out plan",
    ],
    EnvironmentalObjective.CCA.value: [
        "Climate risk and vulnerability assessment report (ISO 14090 or equivalent)",
        "Adaptation solutions implementation record",
    ],
    EnvironmentalObjective.WTR.value: [
        "Water Framework Directive compliance certificate",
        "Water use and discharge monitoring data",
    ],
    EnvironmentalObjective.CE.value: [
        "Waste management plan",
        "Product lifecycle assessment (recyclability/durability)",
    ],
    EnvironmentalObjective.PPC.value: [
        "REACH compliance declaration",
        "Air and water emissions monitoring reports",
        "IED permit or equivalent",
    ],
    EnvironmentalObjective.BIO.value: [
        "Environmental Impact Assessment (EIA) report",
        "Natura 2000 screening report",
        "Biodiversity management plan",
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DNSHAssessmentEngine:
    """
    DNSH Assessment Engine for PACK-008 EU Taxonomy Alignment.

    This engine evaluates the Do No Significant Harm condition across the
    five non-SC environmental objectives using a 6x6 criteria matrix.  It
    follows GreenLang's zero-hallucination principle by applying only
    deterministic boolean and threshold checks defined in the Delegated Acts.

    Attributes:
        matrix: DNSH criteria matrix
        evidence: Evidence requirements per objective

    Example:
        >>> engine = DNSHAssessmentEngine()
        >>> result = engine.assess_dnsh(
        ...     "CCM-4.1", EnvironmentalObjective.CCM,
        ...     {"climate_risk_assessment_completed": True},
        ... )
        >>> assert result.overall_pass or result.objectives_no_data > 0
    """

    def __init__(self) -> None:
        """Initialize the DNSH Assessment Engine."""
        self.matrix = DNSH_MATRIX
        self.evidence = DNSH_EVIDENCE

        logger.info(
            "Initialized DNSHAssessmentEngine with %d matrix entries",
            len(self.matrix),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_dnsh(
        self,
        activity_id: str,
        sc_objective: EnvironmentalObjective,
        data: Dict[str, Any],
    ) -> DNSHResult:
        """
        Perform full DNSH assessment for an activity.

        Args:
            activity_id: Taxonomy activity ID (e.g. "CCM-4.1").
            sc_objective: The objective the activity substantially contributes to.
                          This objective is excluded from DNSH checks.
            data: Dictionary of metric/boolean values for DNSH evaluation.

        Returns:
            DNSHResult with per-objective and overall pass/fail.

        Raises:
            ValueError: If activity_id is empty.
        """
        if not activity_id:
            raise ValueError("activity_id is required")

        start = datetime.utcnow()
        activity_id = activity_id.strip().upper()

        logger.info(
            "Assessing DNSH for activity=%s sc_objective=%s with %d data points",
            activity_id, sc_objective.value, len(data),
        )

        # Determine which objectives to check (all except SC objective)
        dnsh_objectives = [
            obj for obj in EnvironmentalObjective if obj != sc_objective
        ]

        objective_results: Dict[str, ObjectiveDNSHResult] = {}
        passed_count = 0
        failed_count = 0
        no_data_count = 0
        failed_objectives: List[str] = []

        for obj in dnsh_objectives:
            obj_result = self._assess_single_objective(
                activity_id, sc_objective, obj, data
            )
            objective_results[obj.value] = obj_result

            if obj_result.status == DNSHStatus.PASS:
                passed_count += 1
            elif obj_result.status == DNSHStatus.FAIL:
                failed_count += 1
                failed_objectives.append(obj.value)
            elif obj_result.status == DNSHStatus.INSUFFICIENT_DATA:
                no_data_count += 1

        overall_pass = (failed_count == 0 and no_data_count == 0)

        provenance_hash = self._hash(
            f"{activity_id}|{sc_objective.value}|{passed_count}|{failed_count}"
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = DNSHResult(
            activity_id=activity_id,
            sc_objective=sc_objective,
            overall_pass=overall_pass,
            objectives_assessed=len(dnsh_objectives),
            objectives_passed=passed_count,
            objectives_failed=failed_count,
            objectives_no_data=no_data_count,
            failed_objectives=failed_objectives,
            objective_results=objective_results,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "DNSH assessment for %s (SC=%s): overall=%s passed=%d failed=%d "
            "no_data=%d in %.1fms",
            activity_id, sc_objective.value,
            "PASS" if overall_pass else "FAIL",
            passed_count, failed_count, no_data_count, elapsed_ms,
        )

        return result

    def get_dnsh_criteria(
        self,
        activity_id: str,
        sc_objective: EnvironmentalObjective,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return DNSH criteria applicable to an activity given its SC objective.

        Args:
            activity_id: Taxonomy activity ID.
            sc_objective: Substantial Contribution objective.

        Returns:
            Dict mapping DNSH objective value to list of criterion dicts.
        """
        dnsh_objectives = [
            obj for obj in EnvironmentalObjective if obj != sc_objective
        ]

        result: Dict[str, List[Dict[str, Any]]] = {}

        for obj in dnsh_objectives:
            criteria = self._resolve_criteria(activity_id, sc_objective, obj)
            result[obj.value] = [c.dict() for c in criteria]

        return result

    def get_dnsh_evidence(
        self,
        objective: EnvironmentalObjective,
    ) -> List[str]:
        """
        Return evidence requirements for a specific DNSH objective.

        Args:
            objective: Environmental objective.

        Returns:
            List of required evidence document descriptions.
        """
        return self.evidence.get(objective.value, [])

    def batch_assess(
        self,
        assessments: List[Dict[str, Any]],
    ) -> List[DNSHResult]:
        """
        Batch DNSH assessment for multiple activities.

        Each dict must contain ``activity_id``, ``sc_objective`` (str), ``data`` (dict).

        Args:
            assessments: List of assessment request dicts.

        Returns:
            List of DNSHResult, one per input.
        """
        results: List[DNSHResult] = []
        for item in assessments:
            activity_id = item["activity_id"]
            sc_objective = EnvironmentalObjective(item["sc_objective"])
            data = item.get("data", {})
            results.append(self.assess_dnsh(activity_id, sc_objective, data))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_single_objective(
        self,
        activity_id: str,
        sc_objective: EnvironmentalObjective,
        dnsh_objective: EnvironmentalObjective,
        data: Dict[str, Any],
    ) -> ObjectiveDNSHResult:
        """
        Assess DNSH for one environmental objective.

        Args:
            activity_id: Taxonomy activity ID.
            sc_objective: SC objective.
            dnsh_objective: Objective being checked for harm.
            data: Input data.

        Returns:
            ObjectiveDNSHResult for the single objective.
        """
        criteria = self._resolve_criteria(activity_id, sc_objective, dnsh_objective)

        if not criteria:
            return ObjectiveDNSHResult(
                objective=dnsh_objective,
                objective_name=OBJECTIVE_NAMES.get(dnsh_objective.value, dnsh_objective.value),
                status=DNSHStatus.NOT_APPLICABLE,
                criteria_total=0,
                criteria_passed=0,
                criteria_failed=0,
                criteria_no_data=0,
                evidence_required=self.evidence.get(dnsh_objective.value, []),
            )

        details: List[Dict[str, Any]] = []
        passed = 0
        failed = 0
        no_data = 0

        for criterion in criteria:
            value = data.get(criterion.metric_key)
            detail = self._check_criterion(criterion, value)
            details.append(detail)

            if detail["status"] == "PASS":
                passed += 1
            elif detail["status"] == "FAIL":
                failed += 1
            else:
                no_data += 1

        # Mandatory failure = overall objective fail
        mandatory_failed = any(
            d["status"] == "FAIL" and d["is_mandatory"]
            for d in details
        )

        if mandatory_failed or failed > 0:
            status = DNSHStatus.FAIL
        elif no_data > 0:
            status = DNSHStatus.INSUFFICIENT_DATA
        else:
            status = DNSHStatus.PASS

        return ObjectiveDNSHResult(
            objective=dnsh_objective,
            objective_name=OBJECTIVE_NAMES.get(dnsh_objective.value, dnsh_objective.value),
            status=status,
            criteria_total=len(criteria),
            criteria_passed=passed,
            criteria_failed=failed,
            criteria_no_data=no_data,
            details=details,
            evidence_required=self.evidence.get(dnsh_objective.value, []),
        )

    def _resolve_criteria(
        self,
        activity_id: str,
        sc_objective: EnvironmentalObjective,
        dnsh_objective: EnvironmentalObjective,
    ) -> List[DNSHCriterion]:
        """
        Resolve which DNSH criteria apply for a given activity and objective pair.

        Checks activity-specific overrides first, then falls back to the
        generic matrix keyed by SC objective.
        """
        # Try activity-specific entry
        activity_key = f"{activity_id}_{sc_objective.value}"
        criteria_map = self.matrix.get(activity_key, {})
        criteria = criteria_map.get(dnsh_objective.value, [])
        if criteria:
            return criteria

        # Fall back to generic for this SC objective
        generic_key = f"*_{sc_objective.value}"
        criteria_map = self.matrix.get(generic_key, {})
        return criteria_map.get(dnsh_objective.value, [])

    @staticmethod
    def _check_criterion(
        criterion: DNSHCriterion,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Check a single DNSH criterion against a provided value.

        Returns a detail dict with status, actual value, and description.
        """
        detail: Dict[str, Any] = {
            "criterion_id": criterion.criterion_id,
            "dnsh_objective": criterion.dnsh_objective.value,
            "metric_key": criterion.metric_key,
            "description": criterion.description,
            "is_mandatory": criterion.is_mandatory,
            "check_type": criterion.check_type,
            "threshold_value": criterion.threshold_value,
            "actual_value": value,
        }

        if value is None:
            detail["status"] = "NO_DATA"
            return detail

        if criterion.check_type == "BOOLEAN":
            is_pass = bool(value)
        elif criterion.check_type == "THRESHOLD_MIN":
            is_pass = float(value) >= (criterion.threshold_value or 0.0)
        elif criterion.check_type == "THRESHOLD_MAX":
            is_pass = float(value) <= (criterion.threshold_value or float("inf"))
        elif criterion.check_type == "QUALITATIVE":
            is_pass = bool(value)
        else:
            is_pass = bool(value)

        detail["status"] = "PASS" if is_pass else "FAIL"
        return detail

    @staticmethod
    def _hash(data: str) -> str:
        """Return SHA-256 hex digest."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
