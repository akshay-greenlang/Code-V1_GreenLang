"""
Substantial Contribution Engine - PACK-008 EU Taxonomy Alignment

This module evaluates whether economic activities make a Substantial Contribution
(SC) to one of the six EU Taxonomy environmental objectives by checking quantitative
and qualitative Technical Screening Criteria (TSC) defined in the Delegated Acts.

Example:
    >>> engine = SubstantialContributionEngine()
    >>> result = engine.evaluate_sc(
    ...     activity_id="CCM-4.1",
    ...     objective=EnvironmentalObjective.CCM,
    ...     metrics={"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
    ... )
    >>> print(f"SC met: {result.is_met}, score: {result.score}")
"""

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

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


class SCStatus(str, Enum):
    """Substantial Contribution evaluation status."""
    MET = "MET"
    NOT_MET = "NOT_MET"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ThresholdType(str, Enum):
    """Type of Technical Screening Criteria threshold."""
    LESS_THAN = "LESS_THAN"
    LESS_THAN_OR_EQUAL = "LESS_THAN_OR_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_THAN_OR_EQUAL = "GREATER_THAN_OR_EQUAL"
    BETWEEN = "BETWEEN"
    QUALITATIVE = "QUALITATIVE"


class ActivityClassification(str, Enum):
    """Activity classification under the Taxonomy Regulation."""
    STANDARD = "STANDARD"
    ENABLING = "ENABLING"
    TRANSITIONAL = "TRANSITIONAL"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TSCThreshold(BaseModel):
    """A single Technical Screening Criteria threshold."""

    criterion_id: str = Field(..., description="Unique criterion identifier")
    metric_key: str = Field(..., description="Metric key expected in input data")
    description: str = Field(..., description="Human-readable criterion description")
    threshold_type: ThresholdType = Field(..., description="Comparison operator")
    threshold_value: Optional[float] = Field(
        None, description="Numeric threshold value"
    )
    threshold_upper: Optional[float] = Field(
        None, description="Upper bound for BETWEEN checks"
    )
    unit: str = Field(default="", description="Measurement unit")
    is_mandatory: bool = Field(default=True, description="Whether criterion is mandatory")
    delegated_act: str = Field(
        default="EU 2021/2139", description="Source Delegated Act reference"
    )


class ThresholdResult(BaseModel):
    """Result of checking a single TSC threshold."""

    criterion_id: str = Field(..., description="Criterion identifier")
    metric_key: str = Field(..., description="Metric key checked")
    description: str = Field(default="", description="Criterion description")
    threshold_value: Optional[float] = Field(None, description="Required threshold")
    actual_value: Optional[float] = Field(None, description="Actual reported value")
    is_met: bool = Field(..., description="Whether criterion is satisfied")
    has_data: bool = Field(default=True, description="Whether data was available")
    gap: Optional[float] = Field(
        None, description="Distance to threshold (negative = below, positive = above)"
    )
    unit: str = Field(default="", description="Unit of measurement")


class SCResult(BaseModel):
    """Result of evaluating Substantial Contribution for an activity-objective pair."""

    activity_id: str = Field(..., description="Taxonomy activity ID")
    objective: EnvironmentalObjective = Field(
        ..., description="Environmental objective evaluated"
    )
    status: SCStatus = Field(..., description="Substantial Contribution status")
    is_met: bool = Field(..., description="Whether SC is met overall")
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="SC score (fraction of criteria met)"
    )
    criteria_total: int = Field(..., ge=0, description="Total criteria evaluated")
    criteria_met: int = Field(..., ge=0, description="Criteria met")
    criteria_not_met: int = Field(..., ge=0, description="Criteria not met")
    criteria_no_data: int = Field(..., ge=0, description="Criteria with missing data")
    threshold_results: List[ThresholdResult] = Field(
        default_factory=list, description="Per-criterion results"
    )
    classification: ActivityClassification = Field(
        default=ActivityClassification.STANDARD,
        description="Activity classification"
    )
    evidence_requirements: List[str] = Field(
        default_factory=list,
        description="Evidence documents needed for compliance"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Evaluation timestamp"
    )


# ---------------------------------------------------------------------------
# TSC Threshold Database
# ---------------------------------------------------------------------------

TSC_THRESHOLDS: Dict[str, Dict[str, List[TSCThreshold]]] = {
    # ---- Electricity generation (CCM-4.x) ----
    "CCM-4.1": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-4.1-01",
                metric_key="lifecycle_ghg_emissions_gco2e_kwh",
                description="Life-cycle GHG emissions < 100 gCO2e/kWh",
                threshold_type=ThresholdType.LESS_THAN,
                threshold_value=100.0,
                unit="gCO2e/kWh",
            ),
        ],
    },
    "CCM-4.3": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-4.3-01",
                metric_key="lifecycle_ghg_emissions_gco2e_kwh",
                description="Life-cycle GHG emissions < 100 gCO2e/kWh",
                threshold_type=ThresholdType.LESS_THAN,
                threshold_value=100.0,
                unit="gCO2e/kWh",
            ),
        ],
    },
    "CCM-4.5": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-4.5-01",
                metric_key="lifecycle_ghg_emissions_gco2e_kwh",
                description="Life-cycle GHG emissions < 100 gCO2e/kWh",
                threshold_type=ThresholdType.LESS_THAN,
                threshold_value=100.0,
                unit="gCO2e/kWh",
            ),
            TSCThreshold(
                criterion_id="CCM-4.5-02",
                metric_key="power_density_w_per_m2",
                description="Power density > 5 W/m2",
                threshold_type=ThresholdType.GREATER_THAN,
                threshold_value=5.0,
                unit="W/m2",
            ),
        ],
    },
    # ---- Cement (CCM-3.7) ----
    "CCM-3.7": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-3.7-01",
                metric_key="specific_ghg_emissions_tco2e_per_t_product",
                description="Specific GHG emissions from clinker <= 0.722 tCO2e/t clinker (grey) or <= 0.469 (alternative binder)",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=0.722,
                unit="tCO2e/t clinker",
            ),
        ],
    },
    # ---- Iron & Steel (CCM-3.9) ----
    "CCM-3.9": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-3.9-01",
                metric_key="specific_ghg_emissions_tco2e_per_t_product",
                description="GHG emissions <= 1.331 tCO2e/t crude steel (integrated) or <= 0.209 (EAF)",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=1.331,
                unit="tCO2e/t crude steel",
            ),
        ],
    },
    # ---- Aluminium (CCM-3.8) ----
    "CCM-3.8": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-3.8-01",
                metric_key="specific_ghg_emissions_tco2e_per_t_product",
                description="GHG emissions <= 1.484 tCO2e/t aluminium",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=1.484,
                unit="tCO2e/t aluminium",
            ),
        ],
    },
    # ---- Buildings (CCM-7.1, CCM-7.2, CCM-7.7) ----
    "CCM-7.1": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-7.1-01",
                metric_key="primary_energy_demand_kwh_per_m2",
                description="Primary energy demand at least 10% below NZEB threshold",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=90.0,
                unit="kWh/m2/yr (relative to national NZEB)",
                is_mandatory=True,
            ),
            TSCThreshold(
                criterion_id="CCM-7.1-02",
                metric_key="airtightness_test_completed",
                description="Airtightness and thermal integrity testing completed",
                threshold_type=ThresholdType.QUALITATIVE,
                unit="boolean",
            ),
            TSCThreshold(
                criterion_id="CCM-7.1-03",
                metric_key="gwp_lifecycle_assessment_completed",
                description="Life-cycle GWP calculated and disclosed for buildings > 5000 m2",
                threshold_type=ThresholdType.QUALITATIVE,
                unit="boolean",
                is_mandatory=False,
            ),
        ],
    },
    "CCM-7.2": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-7.2-01",
                metric_key="primary_energy_reduction_pct",
                description="Major renovation achieves >= 30% primary energy demand reduction",
                threshold_type=ThresholdType.GREATER_THAN_OR_EQUAL,
                threshold_value=30.0,
                unit="%",
            ),
        ],
    },
    "CCM-7.7": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-7.7-01",
                metric_key="epc_rating",
                description="Energy Performance Certificate class A or top 15% of national stock",
                threshold_type=ThresholdType.QUALITATIVE,
                unit="EPC class",
            ),
        ],
    },
    # ---- Transport (CCM-6.3, CCM-6.5, CCM-6.6) ----
    "CCM-6.3": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-6.3-01",
                metric_key="direct_co2_emissions_gco2_per_km",
                description="Direct CO2 emissions = 0 g/km (zero tailpipe) until 2025, then lifecycle < 50g",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=0.0,
                unit="gCO2/km (tailpipe)",
            ),
        ],
    },
    "CCM-6.5": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-6.5-01",
                metric_key="direct_co2_emissions_gco2_per_km",
                description="Direct CO2 emissions = 0 gCO2/km for personal vehicles",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=0.0,
                unit="gCO2/km",
            ),
        ],
    },
    "CCM-6.6": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-6.6-01",
                metric_key="direct_co2_emissions_gco2_per_tkm",
                description="Direct CO2 emissions = 0 gCO2/tkm for heavy-duty vehicles, or qualify as zero-emission heavy-duty vehicles",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=0.0,
                unit="gCO2/tkm",
            ),
        ],
    },
    # ---- Data centres (CCM-8.1) ----
    "CCM-8.1": {
        EnvironmentalObjective.CCM.value: [
            TSCThreshold(
                criterion_id="CCM-8.1-01",
                metric_key="pue_power_usage_effectiveness",
                description="Power Usage Effectiveness (PUE) <= 1.3 (for existing), 1.2 (new large DC)",
                threshold_type=ThresholdType.LESS_THAN_OR_EQUAL,
                threshold_value=1.3,
                unit="PUE ratio",
            ),
            TSCThreshold(
                criterion_id="CCM-8.1-02",
                metric_key="eu_code_of_conduct_compliance",
                description="Implements practices in EU Code of Conduct for Data Centres",
                threshold_type=ThresholdType.QUALITATIVE,
                unit="boolean",
            ),
        ],
    },
}

# Activity classification reference
ACTIVITY_CLASSIFICATIONS: Dict[str, ActivityClassification] = {
    "CCM-3.3": ActivityClassification.ENABLING,   # Low-carbon transport manufacture
    "CCM-3.4": ActivityClassification.ENABLING,   # Batteries
    "CCM-3.7": ActivityClassification.TRANSITIONAL,  # Cement
    "CCM-3.8": ActivityClassification.TRANSITIONAL,  # Aluminium
    "CCM-3.9": ActivityClassification.TRANSITIONAL,  # Iron & steel
    "CCM-4.3": ActivityClassification.ENABLING,   # Wind power
    "CCM-4.9": ActivityClassification.ENABLING,   # Electricity T&D
    "CCM-6.6": ActivityClassification.TRANSITIONAL,  # Road freight
    "CCM-7.3": ActivityClassification.ENABLING,   # Energy efficiency equipment
    "CCM-8.2": ActivityClassification.ENABLING,   # Data-driven solutions
}

# Evidence requirements per activity
EVIDENCE_REQUIREMENTS: Dict[str, List[str]] = {
    "CCM-4.1": [
        "Life-cycle GHG emission assessment (ISO 14067 or PEF)",
        "Power purchase agreement or generation meter data",
    ],
    "CCM-3.7": [
        "EU ETS verified emissions report",
        "Clinker-to-cement ratio documentation",
        "Best Available Techniques Reference Document (BREF) compliance",
    ],
    "CCM-3.9": [
        "EU ETS verified emissions report",
        "Production route documentation (BF-BOF, EAF, DRI-EAF)",
    ],
    "CCM-7.1": [
        "Energy Performance Certificate (EPC)",
        "Airtightness test report",
        "Building specifications and design documentation",
    ],
    "CCM-7.2": [
        "Pre-renovation energy audit",
        "Post-renovation EPC",
        "Primary energy demand comparison report",
    ],
    "CCM-6.5": [
        "Vehicle type-approval certificate",
        "WLTP CO2 emission test results",
    ],
    "CCM-8.1": [
        "PUE measurement report (EN 50600-4-2)",
        "EU Code of Conduct self-declaration or third-party assessment",
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SubstantialContributionEngine:
    """
    Substantial Contribution Engine for PACK-008 EU Taxonomy Alignment.

    This engine evaluates whether an economic activity makes a Substantial
    Contribution to one of the six environmental objectives by checking the
    Technical Screening Criteria defined in the EU Taxonomy Delegated Acts.

    It follows GreenLang's zero-hallucination principle by using only
    deterministic threshold comparisons -- no LLM inference in the
    calculation path.

    Attributes:
        thresholds: TSC threshold database
        classifications: Activity classification lookup
        evidence_map: Evidence requirements lookup

    Example:
        >>> engine = SubstantialContributionEngine()
        >>> result = engine.evaluate_sc(
        ...     "CCM-4.1", EnvironmentalObjective.CCM,
        ...     {"lifecycle_ghg_emissions_gco2e_kwh": 15.0},
        ... )
        >>> assert result.is_met is True
    """

    def __init__(self) -> None:
        """Initialize the Substantial Contribution Engine."""
        self.thresholds = TSC_THRESHOLDS
        self.classifications = ACTIVITY_CLASSIFICATIONS
        self.evidence_map = EVIDENCE_REQUIREMENTS

        logger.info(
            "Initialized SubstantialContributionEngine with %d activity TSC sets",
            len(self.thresholds),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_sc(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
        metrics: Dict[str, Any],
    ) -> SCResult:
        """
        Evaluate Substantial Contribution for an activity-objective pair.

        Args:
            activity_id: Taxonomy activity ID (e.g. "CCM-4.1").
            objective: Environmental objective to evaluate against.
            metrics: Dictionary of metric values keyed by ``metric_key``.

        Returns:
            SCResult with per-criterion threshold results and overall status.

        Raises:
            ValueError: If activity_id is empty.
        """
        if not activity_id:
            raise ValueError("activity_id is required")

        start = datetime.utcnow()
        activity_id = activity_id.strip().upper()

        logger.info(
            "Evaluating SC for activity=%s objective=%s with %d metrics",
            activity_id, objective.value, len(metrics),
        )

        # Look up TSC thresholds
        activity_thresholds = self.thresholds.get(activity_id, {})
        criteria = activity_thresholds.get(objective.value, [])

        if not criteria:
            logger.warning(
                "No TSC thresholds found for activity=%s objective=%s",
                activity_id, objective.value,
            )
            return self._build_result(
                activity_id, objective, [], metrics,
                status=SCStatus.INSUFFICIENT_DATA,
            )

        # Evaluate each criterion
        threshold_results: List[ThresholdResult] = []
        for criterion in criteria:
            result = self.check_single_threshold(criterion, metrics)
            threshold_results.append(result)

        # Determine overall status
        status, is_met, score, met_ct, not_met_ct, no_data_ct = self._aggregate_results(
            threshold_results, criteria
        )

        # Build final result
        classification = self.classifications.get(
            activity_id, ActivityClassification.STANDARD
        )
        evidence = self.evidence_map.get(activity_id, [])

        provenance_hash = self._hash(
            f"{activity_id}|{objective.value}|{score}|{met_ct}|{not_met_ct}"
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = SCResult(
            activity_id=activity_id,
            objective=objective,
            status=status,
            is_met=is_met,
            score=score,
            criteria_total=len(criteria),
            criteria_met=met_ct,
            criteria_not_met=not_met_ct,
            criteria_no_data=no_data_ct,
            threshold_results=threshold_results,
            classification=classification,
            evidence_requirements=evidence,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "SC evaluation for %s/%s: status=%s score=%.2f (%d/%d met) in %.1fms",
            activity_id, objective.value, status.value, score,
            met_ct, len(criteria), elapsed_ms,
        )

        return result

    def check_thresholds(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
        data: Dict[str, Any],
    ) -> List[ThresholdResult]:
        """
        Check all TSC thresholds for an activity-objective pair.

        This is a lower-level method that returns only the per-criterion
        results without the aggregated SCResult.

        Args:
            activity_id: Taxonomy activity ID.
            objective: Environmental objective.
            data: Metric values.

        Returns:
            List of ThresholdResult for each criterion.
        """
        activity_id = activity_id.strip().upper()
        activity_thresholds = self.thresholds.get(activity_id, {})
        criteria = activity_thresholds.get(objective.value, [])

        return [self.check_single_threshold(c, data) for c in criteria]

    def check_single_threshold(
        self,
        criterion: TSCThreshold,
        metrics: Dict[str, Any],
    ) -> ThresholdResult:
        """
        Check a single TSC threshold against provided metrics.

        Args:
            criterion: The TSCThreshold to check.
            metrics: Dictionary of metric values.

        Returns:
            ThresholdResult with pass/fail and gap.
        """
        actual = metrics.get(criterion.metric_key)

        if actual is None:
            return ThresholdResult(
                criterion_id=criterion.criterion_id,
                metric_key=criterion.metric_key,
                description=criterion.description,
                threshold_value=criterion.threshold_value,
                actual_value=None,
                is_met=False,
                has_data=False,
                gap=None,
                unit=criterion.unit,
            )

        actual_float = float(actual)
        is_met = self._compare(
            actual_float,
            criterion.threshold_type,
            criterion.threshold_value,
            criterion.threshold_upper,
        )

        gap = None
        if criterion.threshold_value is not None:
            gap = actual_float - criterion.threshold_value

        return ThresholdResult(
            criterion_id=criterion.criterion_id,
            metric_key=criterion.metric_key,
            description=criterion.description,
            threshold_value=criterion.threshold_value,
            actual_value=actual_float,
            is_met=is_met,
            has_data=True,
            gap=gap,
            unit=criterion.unit,
        )

    def get_classification(
        self,
        activity_id: str,
    ) -> ActivityClassification:
        """
        Return the classification for an activity (standard / enabling / transitional).

        Args:
            activity_id: Taxonomy activity ID.

        Returns:
            ActivityClassification enum.
        """
        return self.classifications.get(
            activity_id.strip().upper(),
            ActivityClassification.STANDARD,
        )

    def get_evidence_requirements(
        self,
        activity_id: str,
    ) -> List[str]:
        """
        Return evidence documentation required for SC compliance.

        Args:
            activity_id: Taxonomy activity ID.

        Returns:
            List of required evidence document descriptions.
        """
        return self.evidence_map.get(activity_id.strip().upper(), [])

    def batch_evaluate(
        self,
        evaluations: List[Dict[str, Any]],
    ) -> List[SCResult]:
        """
        Evaluate SC for a batch of activity-objective-metrics triples.

        Each dict must contain ``activity_id``, ``objective`` (str), and ``metrics``.

        Args:
            evaluations: List of evaluation request dicts.

        Returns:
            List of SCResult, one per input.
        """
        results: List[SCResult] = []
        for item in evaluations:
            activity_id = item["activity_id"]
            objective = EnvironmentalObjective(item["objective"])
            metrics = item.get("metrics", {})
            results.append(self.evaluate_sc(activity_id, objective, metrics))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compare(
        actual: float,
        threshold_type: ThresholdType,
        value: Optional[float],
        upper: Optional[float],
    ) -> bool:
        """Perform a threshold comparison."""
        if threshold_type == ThresholdType.QUALITATIVE:
            # Qualitative criteria treated as met if value is truthy
            return bool(actual)

        if value is None:
            return False

        if threshold_type == ThresholdType.LESS_THAN:
            return actual < value
        elif threshold_type == ThresholdType.LESS_THAN_OR_EQUAL:
            return actual <= value
        elif threshold_type == ThresholdType.GREATER_THAN:
            return actual > value
        elif threshold_type == ThresholdType.GREATER_THAN_OR_EQUAL:
            return actual >= value
        elif threshold_type == ThresholdType.BETWEEN:
            return value <= actual <= (upper or value)

        return False

    def _aggregate_results(
        self,
        results: List[ThresholdResult],
        criteria: List[TSCThreshold],
    ) -> tuple:
        """
        Aggregate per-criterion results into overall SC status.

        Returns:
            (status, is_met, score, met_count, not_met_count, no_data_count)
        """
        met_ct = sum(1 for r in results if r.is_met)
        no_data_ct = sum(1 for r in results if not r.has_data)
        not_met_ct = len(results) - met_ct - no_data_ct

        # Mandatory criteria that are not met
        mandatory_not_met = 0
        for criterion, result in zip(criteria, results):
            if criterion.is_mandatory and not result.is_met and result.has_data:
                mandatory_not_met += 1

        total = len(results)
        score = met_ct / total if total > 0 else 0.0

        if no_data_ct == total:
            status = SCStatus.INSUFFICIENT_DATA
            is_met = False
        elif mandatory_not_met > 0:
            status = SCStatus.NOT_MET
            is_met = False
        elif met_ct == total:
            status = SCStatus.MET
            is_met = True
        elif met_ct > 0:
            status = SCStatus.PARTIAL
            is_met = False
        else:
            status = SCStatus.NOT_MET
            is_met = False

        return status, is_met, score, met_ct, not_met_ct, no_data_ct

    def _build_result(
        self,
        activity_id: str,
        objective: EnvironmentalObjective,
        threshold_results: List[ThresholdResult],
        metrics: Dict[str, Any],
        status: SCStatus = SCStatus.INSUFFICIENT_DATA,
    ) -> SCResult:
        """Build an SCResult for edge cases (e.g. no criteria found)."""
        classification = self.classifications.get(
            activity_id, ActivityClassification.STANDARD
        )
        evidence = self.evidence_map.get(activity_id, [])
        provenance_hash = self._hash(
            f"{activity_id}|{objective.value}|{status.value}"
        )

        return SCResult(
            activity_id=activity_id,
            objective=objective,
            status=status,
            is_met=False,
            score=0.0,
            criteria_total=0,
            criteria_met=0,
            criteria_not_met=0,
            criteria_no_data=0,
            threshold_results=threshold_results,
            classification=classification,
            evidence_requirements=evidence,
            provenance_hash=provenance_hash,
        )

    @staticmethod
    def _hash(data: str) -> str:
        """Return SHA-256 hex digest."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
