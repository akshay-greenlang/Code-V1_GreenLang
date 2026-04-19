"""
GL-007: EU Taxonomy Agent

This module implements the EU Taxonomy Alignment Agent for classifying
economic activities and calculating taxonomy-aligned KPIs per
EU Regulation 2020/852.

The agent supports:
- Activity eligibility assessment
- Technical Screening Criteria (TSC) evaluation
- Do No Significant Harm (DNSH) assessment
- Minimum Safeguards verification
- Taxonomy KPI calculation (Revenue, CapEx, OpEx)

Example:
    >>> agent = EUTaxonomyAgent()
    >>> result = agent.run(TaxonomyInput(
    ...     nace_code="D35.11",
    ...     activity_description="Electricity generation from solar PV",
    ...     revenue_eur=10000000,
    ...     environmental_data={"ghg_intensity": 0}
    ... ))
    >>> print(f"Taxonomy aligned: {result.data.is_aligned}")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy environmental objectives."""

    CLIMATE_MITIGATION = "climate_change_mitigation"
    CLIMATE_ADAPTATION = "climate_change_adaptation"
    WATER = "water_marine_resources"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity_ecosystems"


class AlignmentStatus(str, Enum):
    """Taxonomy alignment status."""

    ALIGNED = "aligned"
    ELIGIBLE_NOT_ALIGNED = "eligible_not_aligned"
    NOT_ELIGIBLE = "not_eligible"
    ASSESSMENT_REQUIRED = "assessment_required"


class DNSHStatus(str, Enum):
    """Do No Significant Harm status."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    ASSESSMENT_REQUIRED = "assessment_required"


class MinimumSafeguardsStatus(str, Enum):
    """Minimum safeguards compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"


class TaxonomyInput(BaseModel):
    """
    Input model for EU Taxonomy Agent.

    Attributes:
        nace_code: NACE Rev. 2 code
        activity_description: Description of economic activity
        revenue_eur: Revenue from activity in EUR
        capex_eur: Capital expenditure in EUR
        opex_eur: Operating expenditure in EUR
        environmental_data: Environmental performance data
        dnsh_data: DNSH assessment data
        safeguards_data: Minimum safeguards data
    """

    nace_code: str = Field(..., description="NACE Rev. 2 code")
    activity_description: str = Field(..., description="Activity description")

    # Financial KPIs
    revenue_eur: float = Field(0, ge=0, description="Revenue from activity EUR")
    capex_eur: float = Field(0, ge=0, description="CapEx EUR")
    opex_eur: float = Field(0, ge=0, description="OpEx EUR")
    total_revenue_eur: float = Field(0, ge=0, description="Total company revenue EUR")
    total_capex_eur: float = Field(0, ge=0, description="Total company CapEx EUR")
    total_opex_eur: float = Field(0, ge=0, description="Total company OpEx EUR")

    # Environmental data for TSC
    environmental_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental metrics for TSC evaluation"
    )

    # DNSH data
    dnsh_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="DNSH assessment data"
    )

    # Minimum safeguards
    safeguards_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimum safeguards compliance data"
    )

    # Target objective
    primary_objective: EnvironmentalObjective = Field(
        EnvironmentalObjective.CLIMATE_MITIGATION,
        description="Primary environmental objective"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TSCResult(BaseModel):
    """Technical Screening Criteria evaluation result."""

    objective: str
    criteria_met: bool
    criteria_details: Dict[str, Any]
    threshold_value: Optional[float]
    actual_value: Optional[float]


class DNSHResult(BaseModel):
    """DNSH assessment result for one objective."""

    objective: str
    status: DNSHStatus
    criteria_checked: List[str]
    issues: List[str]


class TaxonomyOutput(BaseModel):
    """
    Output model for EU Taxonomy Agent.

    Includes eligibility, alignment, and KPI results.
    """

    nace_code: str = Field(..., description="NACE code assessed")
    taxonomy_activity_code: Optional[str] = Field(None, description="Taxonomy activity code")
    activity_name: Optional[str] = Field(None, description="Taxonomy activity name")

    # Eligibility and alignment
    is_eligible: bool = Field(..., description="Activity is taxonomy-eligible")
    is_aligned: bool = Field(..., description="Activity is taxonomy-aligned")
    alignment_status: str = Field(..., description="Alignment status")

    # Substantial contribution
    primary_objective: str = Field(..., description="Primary environmental objective")
    substantial_contribution: bool = Field(..., description="Meets SC criteria")
    tsc_results: List[Dict[str, Any]] = Field(..., description="TSC evaluation results")

    # DNSH
    dnsh_pass: bool = Field(..., description="All DNSH criteria met")
    dnsh_results: List[Dict[str, Any]] = Field(..., description="DNSH results by objective")

    # Minimum safeguards
    minimum_safeguards_compliant: bool = Field(..., description="Minimum safeguards met")
    safeguards_details: Dict[str, Any] = Field(..., description="Safeguards assessment")

    # KPIs
    revenue_aligned_eur: float = Field(..., description="Aligned revenue EUR")
    revenue_aligned_pct: float = Field(..., description="Aligned revenue %")
    capex_aligned_eur: float = Field(..., description="Aligned CapEx EUR")
    capex_aligned_pct: float = Field(..., description="Aligned CapEx %")
    opex_aligned_eur: float = Field(..., description="Aligned OpEx EUR")
    opex_aligned_pct: float = Field(..., description="Aligned OpEx %")

    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class TaxonomyActivity(BaseModel):
    """EU Taxonomy activity definition."""

    code: str
    name: str
    nace_codes: List[str]
    objective: EnvironmentalObjective
    tsc_thresholds: Dict[str, Any]


class EUTaxonomyAgent:
    """
    GL-007: EU Taxonomy Agent.

    This agent evaluates economic activities against the EU Taxonomy
    using zero-hallucination deterministic assessments:
    - Eligibility: NACE code in taxonomy delegated acts
    - Substantial Contribution: Meets TSC thresholds
    - DNSH: No significant harm to other objectives
    - Minimum Safeguards: Human rights, anti-corruption, etc.

    Aligned with:
    - EU Taxonomy Regulation (EU) 2020/852
    - Climate Delegated Act (EU) 2021/2139
    - Environmental Delegated Act (EU) 2023/2486

    Attributes:
        activities: Database of taxonomy activities
        tsc_thresholds: Technical screening criteria thresholds

    Example:
        >>> agent = EUTaxonomyAgent()
        >>> result = agent.run(TaxonomyInput(
        ...     nace_code="D35.11",
        ...     activity_description="Solar PV generation",
        ...     revenue_eur=10000000
        ... ))
        >>> assert result.is_eligible
    """

    AGENT_ID = "regulatory/eu_taxonomy_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "EU Taxonomy alignment calculator"

    # Taxonomy activities (Climate Delegated Act)
    TAXONOMY_ACTIVITIES: Dict[str, TaxonomyActivity] = {
        "4.1": TaxonomyActivity(
            code="4.1",
            name="Electricity generation using solar photovoltaic technology",
            nace_codes=["D35.11", "F42.22"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={"lifecycle_ghg_gco2e_kwh": 100},
        ),
        "4.2": TaxonomyActivity(
            code="4.2",
            name="Electricity generation using concentrated solar power (CSP)",
            nace_codes=["D35.11", "F42.22"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={"lifecycle_ghg_gco2e_kwh": 100},
        ),
        "4.3": TaxonomyActivity(
            code="4.3",
            name="Electricity generation from wind power",
            nace_codes=["D35.11", "F42.22"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={"lifecycle_ghg_gco2e_kwh": 100},
        ),
        "4.5": TaxonomyActivity(
            code="4.5",
            name="Electricity generation from hydropower",
            nace_codes=["D35.11", "F42.22"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "lifecycle_ghg_gco2e_kwh": 100,
                "power_density_w_m2": 5,
            },
        ),
        "4.9": TaxonomyActivity(
            code="4.9",
            name="Transmission and distribution of electricity",
            nace_codes=["D35.12", "D35.13"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={},  # System criteria
        ),
        "4.10": TaxonomyActivity(
            code="4.10",
            name="Storage of electricity",
            nace_codes=["D35.11", "C27.20"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={},  # Dedicated storage
        ),
        "6.5": TaxonomyActivity(
            code="6.5",
            name="Transport by motorbikes, passenger cars and light commercial vehicles",
            nace_codes=["H49.32", "H49.39", "N77.11"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "direct_co2_gkm": 0,  # Zero emissions
                "co2_gkm_wltp": 50,  # Or <50g until 2025
            },
        ),
        "7.1": TaxonomyActivity(
            code="7.1",
            name="Construction of new buildings",
            nace_codes=["F41.1", "F41.2"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "primary_energy_demand_pct_nzeb": 10,
                "air_tightness": True,
            },
        ),
        "7.2": TaxonomyActivity(
            code="7.2",
            name="Renovation of existing buildings",
            nace_codes=["F41", "F43"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "primary_energy_reduction_pct": 30,
            },
        ),
        "7.7": TaxonomyActivity(
            code="7.7",
            name="Acquisition and ownership of buildings",
            nace_codes=["L68"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "epc_rating": "A",
                "primary_energy_demand_kwh_m2": 100,
            },
        ),
        "3.1": TaxonomyActivity(
            code="3.1",
            name="Manufacture of renewable energy technologies",
            nace_codes=["C25", "C27", "C28"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={},  # Substantial contribution by nature
        ),
        "3.3": TaxonomyActivity(
            code="3.3",
            name="Manufacture of low carbon technologies for transport",
            nace_codes=["C29.10"],
            objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            tsc_thresholds={
                "vehicle_co2_gkm": 50,
            },
        ),
    }

    # NACE to Taxonomy activity mapping
    NACE_TO_ACTIVITIES: Dict[str, List[str]] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EU Taxonomy Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        # Build NACE to activity mapping
        self._build_nace_mapping()

        logger.info(f"EUTaxonomyAgent initialized (version {self.VERSION})")

    def _build_nace_mapping(self) -> None:
        """Build NACE code to taxonomy activities mapping."""
        for code, activity in self.TAXONOMY_ACTIVITIES.items():
            for nace in activity.nace_codes:
                if nace not in self.NACE_TO_ACTIVITIES:
                    self.NACE_TO_ACTIVITIES[nace] = []
                self.NACE_TO_ACTIVITIES[nace].append(code)

    def run(self, input_data: TaxonomyInput) -> TaxonomyOutput:
        """
        Execute the EU Taxonomy alignment assessment.

        ZERO-HALLUCINATION assessment:
        - Eligibility: NACE code in delegated acts
        - SC: actual_value <= threshold_value
        - DNSH: all objectives checked
        - Alignment: eligible AND SC AND DNSH AND safeguards

        Args:
            input_data: Validated taxonomy input data

        Returns:
            Assessment result with alignment status and KPIs
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Assessing EU Taxonomy alignment: NACE={input_data.nace_code}, "
            f"objective={input_data.primary_objective}"
        )

        try:
            # Step 1: Check eligibility
            is_eligible, activity = self._check_eligibility(input_data.nace_code)

            self._track_step("eligibility_check", {
                "nace_code": input_data.nace_code,
                "is_eligible": is_eligible,
                "activity_code": activity.code if activity else None,
                "activity_name": activity.name if activity else None,
            })

            # Step 2: Evaluate substantial contribution (TSC)
            tsc_pass, tsc_results = self._evaluate_tsc(
                activity,
                input_data.environmental_data,
            ) if activity else (False, [])

            self._track_step("tsc_evaluation", {
                "substantial_contribution": tsc_pass,
                "criteria_checked": len(tsc_results),
            })

            # Step 3: Evaluate DNSH
            dnsh_pass, dnsh_results = self._evaluate_dnsh(
                activity,
                input_data.dnsh_data,
            ) if activity else (False, [])

            self._track_step("dnsh_evaluation", {
                "dnsh_pass": dnsh_pass,
                "objectives_checked": len(dnsh_results),
            })

            # Step 4: Evaluate minimum safeguards
            safeguards_compliant, safeguards_details = self._evaluate_safeguards(
                input_data.safeguards_data
            )

            self._track_step("safeguards_evaluation", {
                "compliant": safeguards_compliant,
            })

            # Step 5: ZERO-HALLUCINATION CALCULATION
            # Aligned = eligible AND SC AND DNSH AND safeguards
            is_aligned = is_eligible and tsc_pass and dnsh_pass and safeguards_compliant

            # Determine alignment status
            if is_aligned:
                alignment_status = AlignmentStatus.ALIGNED
            elif is_eligible:
                alignment_status = AlignmentStatus.ELIGIBLE_NOT_ALIGNED
            else:
                alignment_status = AlignmentStatus.NOT_ELIGIBLE

            self._track_step("alignment_calculation", {
                "formula": "aligned = eligible AND sc AND dnsh AND safeguards",
                "eligible": is_eligible,
                "sc": tsc_pass,
                "dnsh": dnsh_pass,
                "safeguards": safeguards_compliant,
                "aligned": is_aligned,
            })

            # Step 6: Calculate KPIs
            revenue_aligned = input_data.revenue_eur if is_aligned else 0
            capex_aligned = input_data.capex_eur if is_aligned else 0
            opex_aligned = input_data.opex_eur if is_aligned else 0

            revenue_pct = (revenue_aligned / input_data.total_revenue_eur * 100) if input_data.total_revenue_eur > 0 else 0
            capex_pct = (capex_aligned / input_data.total_capex_eur * 100) if input_data.total_capex_eur > 0 else 0
            opex_pct = (opex_aligned / input_data.total_opex_eur * 100) if input_data.total_opex_eur > 0 else 0

            self._track_step("kpi_calculation", {
                "revenue_aligned_eur": revenue_aligned,
                "revenue_aligned_pct": revenue_pct,
                "capex_aligned_eur": capex_aligned,
                "capex_aligned_pct": capex_pct,
                "opex_aligned_eur": opex_aligned,
                "opex_aligned_pct": opex_pct,
            })

            # Step 7: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 8: Create output
            output = TaxonomyOutput(
                nace_code=input_data.nace_code,
                taxonomy_activity_code=activity.code if activity else None,
                activity_name=activity.name if activity else None,
                is_eligible=is_eligible,
                is_aligned=is_aligned,
                alignment_status=alignment_status.value,
                primary_objective=input_data.primary_objective.value,
                substantial_contribution=tsc_pass,
                tsc_results=[r.dict() for r in tsc_results] if tsc_results else [],
                dnsh_pass=dnsh_pass,
                dnsh_results=[r.dict() for r in dnsh_results] if dnsh_results else [],
                minimum_safeguards_compliant=safeguards_compliant,
                safeguards_details=safeguards_details,
                revenue_aligned_eur=revenue_aligned,
                revenue_aligned_pct=round(revenue_pct, 2),
                capex_aligned_eur=capex_aligned,
                capex_aligned_pct=round(capex_pct, 2),
                opex_aligned_eur=opex_aligned,
                opex_aligned_pct=round(opex_pct, 2),
                provenance_hash=provenance_hash,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"EU Taxonomy assessment complete: aligned={is_aligned}, "
                f"status={alignment_status.value} "
                f"(duration: {duration_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"EU Taxonomy assessment failed: {str(e)}", exc_info=True)
            raise

    def _check_eligibility(
        self,
        nace_code: str,
    ) -> Tuple[bool, Optional[TaxonomyActivity]]:
        """
        Check if NACE code is taxonomy-eligible.

        ZERO-HALLUCINATION: Deterministic lookup in delegated acts.
        """
        # Try exact match
        if nace_code in self.NACE_TO_ACTIVITIES:
            activity_code = self.NACE_TO_ACTIVITIES[nace_code][0]
            return True, self.TAXONOMY_ACTIVITIES[activity_code]

        # Try prefix match (e.g., F41 matches F41.1)
        for nace_prefix, activities in self.NACE_TO_ACTIVITIES.items():
            if nace_code.startswith(nace_prefix) or nace_prefix.startswith(nace_code):
                activity_code = activities[0]
                return True, self.TAXONOMY_ACTIVITIES[activity_code]

        return False, None

    def _evaluate_tsc(
        self,
        activity: Optional[TaxonomyActivity],
        environmental_data: Dict[str, Any],
    ) -> Tuple[bool, List[TSCResult]]:
        """
        Evaluate Technical Screening Criteria.

        ZERO-HALLUCINATION: actual_value <= threshold_value
        """
        results: List[TSCResult] = []

        if not activity or not activity.tsc_thresholds:
            # No specific thresholds = substantial contribution by nature
            results.append(TSCResult(
                objective=activity.objective.value if activity else "",
                criteria_met=True,
                criteria_details={"note": "Substantial contribution by nature of activity"},
                threshold_value=None,
                actual_value=None,
            ))
            return True, results

        all_met = True
        for criterion, threshold in activity.tsc_thresholds.items():
            actual = environmental_data.get(criterion)

            if isinstance(threshold, bool):
                criteria_met = actual == threshold if actual is not None else False
            elif isinstance(threshold, (int, float)):
                criteria_met = actual <= threshold if actual is not None else False
            else:
                criteria_met = actual == threshold if actual is not None else False

            if not criteria_met:
                all_met = False

            results.append(TSCResult(
                objective=activity.objective.value,
                criteria_met=criteria_met,
                criteria_details={"criterion": criterion, "required": threshold},
                threshold_value=threshold if isinstance(threshold, (int, float)) else None,
                actual_value=actual if isinstance(actual, (int, float)) else None,
            ))

        return all_met, results

    def _evaluate_dnsh(
        self,
        activity: Optional[TaxonomyActivity],
        dnsh_data: Dict[str, Any],
    ) -> Tuple[bool, List[DNSHResult]]:
        """
        Evaluate Do No Significant Harm criteria.

        All 5 other objectives must be checked (excluding primary).
        """
        results: List[DNSHResult] = []

        if not activity:
            return False, results

        # DNSH must be assessed for all objectives except primary
        objectives_to_check = [
            obj for obj in EnvironmentalObjective
            if obj != activity.objective
        ]

        all_pass = True
        for objective in objectives_to_check:
            objective_key = objective.value
            dnsh_status = DNSHStatus.ASSESSMENT_REQUIRED
            issues: List[str] = []
            criteria: List[str] = []

            if objective_key in dnsh_data:
                data = dnsh_data[objective_key]
                if data.get("compliant", False):
                    dnsh_status = DNSHStatus.PASS
                    criteria = data.get("criteria_checked", [])
                else:
                    dnsh_status = DNSHStatus.FAIL
                    issues = data.get("issues", ["DNSH criteria not met"])
                    all_pass = False
            else:
                all_pass = False
                issues = ["DNSH assessment not provided"]

            results.append(DNSHResult(
                objective=objective_key,
                status=dnsh_status,
                criteria_checked=criteria,
                issues=issues,
            ))

        return all_pass, results

    def _evaluate_safeguards(
        self,
        safeguards_data: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate minimum safeguards compliance.

        Covers: OECD Guidelines, UN Guiding Principles, ILO Conventions
        """
        required_safeguards = [
            "oecd_guidelines",
            "un_guiding_principles",
            "ilo_conventions",
            "international_bill_rights",
        ]

        details: Dict[str, Any] = {}
        all_compliant = True

        for safeguard in required_safeguards:
            is_compliant = safeguards_data.get(safeguard, False)
            details[safeguard] = is_compliant
            if not is_compliant:
                all_compliant = False

        return all_compliant, details

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_taxonomy_activities(self) -> List[Dict[str, Any]]:
        """Get list of taxonomy activities."""
        return [
            {
                "code": a.code,
                "name": a.name,
                "objective": a.objective.value,
                "nace_codes": a.nace_codes,
            }
            for a in self.TAXONOMY_ACTIVITIES.values()
        ]

    def get_environmental_objectives(self) -> List[str]:
        """Get list of environmental objectives."""
        return [obj.value for obj in EnvironmentalObjective]


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "regulatory/eu_taxonomy_v1",
    "name": "EU Taxonomy Agent",
    "version": "1.0.0",
    "summary": "EU Taxonomy alignment calculator",
    "tags": ["eu-taxonomy", "sustainable-finance", "tsc", "dnsh"],
    "owners": ["regulatory-team"],
    "compute": {
        "entrypoint": "python://agents.gl_007_eu_taxonomy.agent:EUTaxonomyAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://eu/taxonomy-delegated-acts/2024"},
    ],
    "provenance": {
        "regulation_version": "EU 2020/852",
        "delegated_act_version": "EU 2021/2139",
        "enable_audit": True,
    },
}
