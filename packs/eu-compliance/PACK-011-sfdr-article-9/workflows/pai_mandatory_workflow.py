# -*- coding: utf-8 -*-
"""
PAI Mandatory Workflow
================================================

Four-phase workflow for mandatory Principal Adverse Impact (PAI) indicator
assessment under SFDR Article 9. Orchestrates data sourcing, PAI calculation,
integration assessment, and action planning into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 9 products must consider and disclose all mandatory PAI indicators
      as defined in Annex I, Table 1 of the RTS.
    - 14 mandatory PAI indicators cover climate and environmental impacts
      (indicators 1-6) plus social and governance impacts (indicators 7-14).
    - For each PAI, the product must disclose: metric, impact, actions taken,
      actions planned, and targets set.
    - PAI data must be sourced from reliable providers with quality assessment.
    - Integration of PAI considerations into the investment process must be
      documented with concrete actions and timelines.

    Mandatory PAI Indicators (Table 1):
    1. GHG emissions (Scope 1, 2, 3, total)
    2. Carbon footprint
    3. GHG intensity of investee companies
    4. Exposure to fossil fuel sector
    5. Non-renewable energy share (consumption and production)
    6. Energy consumption intensity per high-impact climate sector
    7. Activities negatively affecting biodiversity-sensitive areas
    8. Emissions to water
    9. Hazardous waste and radioactive waste ratio
    10. Violations of UNGC principles and OECD guidelines
    11. Lack of UNGC/OECD compliance processes and mechanisms
    12. Unadjusted gender pay gap
    13. Board gender diversity
    14. Exposure to controversial weapons

Phases:
    1. DataSourcing - Source PAI data from providers and assess quality
    2. PAICalculation - Calculate all 14 mandatory PAI indicators
    3. IntegrationAssessment - Assess PAI integration into investment process
    4. ActionPlanning - Define actions, targets, and timelines for PAI mitigation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITIES
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class PAICategory(str, Enum):
    """PAI indicator category."""
    CLIMATE_ENVIRONMENTAL = "CLIMATE_ENVIRONMENTAL"
    SOCIAL_GOVERNANCE = "SOCIAL_GOVERNANCE"


class PAIDataQuality(str, Enum):
    """PAI data quality classification."""
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    PROXY = "PROXY"
    UNAVAILABLE = "UNAVAILABLE"


class ActionPriority(str, Enum):
    """Action plan priority level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=_utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# DATA MODELS - PAI MANDATORY
# =============================================================================


# Canonical list of 14 mandatory PAI indicators (Annex I, Table 1)
MANDATORY_PAI_INDICATORS = [
    {
        "id": "PAI-01", "number": 1,
        "name": "GHG Emissions",
        "metric": "Scope 1, 2, 3 and total GHG emissions",
        "unit": "tCO2e",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-02", "number": 2,
        "name": "Carbon Footprint",
        "metric": "Carbon footprint per EUR million invested",
        "unit": "tCO2e/EUR million",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-03", "number": 3,
        "name": "GHG Intensity",
        "metric": "Weighted average GHG intensity of investee companies",
        "unit": "tCO2e/EUR million revenue",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-04", "number": 4,
        "name": "Fossil Fuel Exposure",
        "metric": "Share of investments in fossil fuel sector",
        "unit": "percentage",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-05", "number": 5,
        "name": "Non-Renewable Energy",
        "metric": "Share of non-renewable energy consumption and production",
        "unit": "percentage",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-06", "number": 6,
        "name": "Energy Consumption Intensity",
        "metric": "Energy consumption intensity per high-impact climate sector",
        "unit": "GWh/EUR million revenue",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-07", "number": 7,
        "name": "Biodiversity Impact",
        "metric": "Activities negatively affecting biodiversity-sensitive areas",
        "unit": "percentage",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-08", "number": 8,
        "name": "Water Emissions",
        "metric": "Emissions to water",
        "unit": "tonnes",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-09", "number": 9,
        "name": "Hazardous Waste",
        "metric": "Hazardous waste and radioactive waste ratio",
        "unit": "tonnes",
        "category": PAICategory.CLIMATE_ENVIRONMENTAL.value,
    },
    {
        "id": "PAI-10", "number": 10,
        "name": "UNGC/OECD Violations",
        "metric": "Violations of UNGC principles and OECD guidelines",
        "unit": "percentage",
        "category": PAICategory.SOCIAL_GOVERNANCE.value,
    },
    {
        "id": "PAI-11", "number": 11,
        "name": "UNGC/OECD Compliance Mechanisms",
        "metric": "Lack of compliance processes and mechanisms",
        "unit": "percentage",
        "category": PAICategory.SOCIAL_GOVERNANCE.value,
    },
    {
        "id": "PAI-12", "number": 12,
        "name": "Gender Pay Gap",
        "metric": "Average unadjusted gender pay gap",
        "unit": "percentage",
        "category": PAICategory.SOCIAL_GOVERNANCE.value,
    },
    {
        "id": "PAI-13", "number": 13,
        "name": "Board Gender Diversity",
        "metric": "Average ratio of female to male board members",
        "unit": "percentage",
        "category": PAICategory.SOCIAL_GOVERNANCE.value,
    },
    {
        "id": "PAI-14", "number": 14,
        "name": "Controversial Weapons",
        "metric": "Exposure to controversial weapons",
        "unit": "percentage",
        "category": PAICategory.SOCIAL_GOVERNANCE.value,
    },
]


class PAIMandatoryInput(BaseModel):
    """Input configuration for the mandatory PAI workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    reporting_date: str = Field(
        ..., description="Reporting date YYYY-MM-DD"
    )
    reporting_period_start: str = Field(
        ..., description="Period start date YYYY-MM-DD"
    )
    reporting_period_end: str = Field(
        ..., description="Period end date YYYY-MM-DD"
    )
    data_providers: List[str] = Field(
        default_factory=list,
        description="PAI data provider identifiers"
    )
    portfolio_holdings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Portfolio holdings with PAI data"
    )
    portfolio_value_eur: float = Field(
        default=0.0, ge=0.0,
        description="Total portfolio value in EUR"
    )
    prior_period_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Prior period PAI values for comparison"
    )
    engagement_activities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Engagement activities related to PAI"
    )
    existing_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Existing PAI mitigation actions"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date", "reporting_period_start", "reporting_period_end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD format")
        return v


class PAIMandatoryResult(WorkflowResult):
    """Complete result from the mandatory PAI workflow."""
    product_name: str = Field(default="")
    reporting_period: str = Field(default="")
    total_indicators: int = Field(default=14)
    indicators_calculated: int = Field(default=0)
    indicators_with_data: int = Field(default=0)
    data_coverage_pct: float = Field(default=0.0)
    climate_indicators_count: int = Field(default=0)
    social_indicators_count: int = Field(default=0)
    integration_score: float = Field(default=0.0)
    actions_planned: int = Field(default=0)
    high_priority_actions: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class DataSourcingPhase:
    """
    Phase 1: Data Sourcing.

    Sources PAI data from configured data providers, validates data
    quality, identifies gaps, and assigns quality classifications
    for each of the 14 mandatory indicators.
    """

    PHASE_NAME = "data_sourcing"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute data sourcing phase.

        Args:
            context: Workflow context with portfolio and provider config.

        Returns:
            PhaseResult with sourced PAI data and quality assessment.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("portfolio_holdings", [])
            data_providers = config.get("data_providers", [])

            outputs["product_name"] = config.get("product_name", "")
            outputs["data_providers"] = data_providers
            outputs["holdings_count"] = len(holdings)

            # Source data for each mandatory PAI indicator
            pai_data: List[Dict[str, Any]] = []
            indicators_with_data = 0

            for indicator in MANDATORY_PAI_INDICATORS:
                pai_id = indicator["id"]
                pai_name = indicator["name"]

                # Collect holding-level data for this indicator
                holding_values: List[Dict[str, Any]] = []
                data_available_count = 0

                for holding in holdings:
                    holding_id = holding.get("holding_id", "")
                    holding_name = holding.get("name", "")
                    weight = holding.get("weight_pct", 0.0)
                    pai_values = holding.get("pai_data", {})
                    value = pai_values.get(pai_id)

                    if value is not None:
                        quality = PAIDataQuality.REPORTED.value
                        data_available_count += 1
                    elif holding.get("estimated_pai", {}).get(pai_id):
                        value = holding["estimated_pai"][pai_id]
                        quality = PAIDataQuality.ESTIMATED.value
                        data_available_count += 1
                    else:
                        quality = PAIDataQuality.UNAVAILABLE.value

                    holding_values.append({
                        "holding_id": holding_id,
                        "holding_name": holding_name,
                        "weight_pct": weight,
                        "value": value,
                        "data_quality": quality,
                    })

                coverage_pct = (
                    data_available_count / len(holdings) * 100.0
                    if holdings else 0.0
                )

                if data_available_count > 0:
                    indicators_with_data += 1

                pai_data.append({
                    "indicator": indicator,
                    "holding_values": holding_values,
                    "data_available_count": data_available_count,
                    "coverage_pct": round(coverage_pct, 1),
                })

                if coverage_pct < 50.0:
                    warnings.append(
                        f"PAI indicator '{pai_name}' has low data "
                        f"coverage: {coverage_pct:.1f}%"
                    )

            # Overall data quality assessment
            total_coverage = (
                indicators_with_data / len(MANDATORY_PAI_INDICATORS) * 100.0
            )

            outputs["pai_data"] = pai_data
            outputs["indicators_with_data"] = indicators_with_data
            outputs["total_indicators"] = len(MANDATORY_PAI_INDICATORS)
            outputs["overall_coverage_pct"] = round(total_coverage, 1)
            outputs["quality_summary"] = {
                "reported_count": sum(
                    1 for p in pai_data
                    if any(
                        hv["data_quality"] == PAIDataQuality.REPORTED.value
                        for hv in p["holding_values"]
                    )
                ),
                "estimated_count": sum(
                    1 for p in pai_data
                    if any(
                        hv["data_quality"] == PAIDataQuality.ESTIMATED.value
                        for hv in p["holding_values"]
                    )
                ),
            }

            status = PhaseStatus.COMPLETED
            records = len(pai_data)

        except Exception as exc:
            logger.error(
                "DataSourcing failed: %s", exc, exc_info=True
            )
            errors.append(f"Data sourcing failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class PAICalculationPhase:
    """
    Phase 2: PAI Calculation.

    Calculates all 14 mandatory PAI indicators using deterministic
    formulas (weighted averages, sums, or portfolio-level aggregations).
    Compares with prior period values where available.
    """

    PHASE_NAME = "pai_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute PAI calculation phase.

        Args:
            context: Workflow context with sourced PAI data.

        Returns:
            PhaseResult with calculated PAI values.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            data_output = context.get_phase_output("data_sourcing")
            pai_data = data_output.get("pai_data", [])
            prior_values = config.get("prior_period_values", {})
            portfolio_value = config.get("portfolio_value_eur", 0.0)

            calculated_indicators: List[Dict[str, Any]] = []
            climate_count = 0
            social_count = 0

            for pai_entry in pai_data:
                indicator = pai_entry.get("indicator", {})
                pai_id = indicator.get("id", "")
                pai_name = indicator.get("name", "")
                pai_unit = indicator.get("unit", "")
                pai_category = indicator.get("category", "")
                holding_values = pai_entry.get("holding_values", [])

                # Filter holdings with available data
                available = [
                    hv for hv in holding_values
                    if hv["value"] is not None
                ]

                if not available:
                    calculated_indicators.append({
                        "pai_id": pai_id,
                        "name": pai_name,
                        "unit": pai_unit,
                        "category": pai_category,
                        "calculated_value": None,
                        "prior_value": prior_values.get(pai_id),
                        "change_pct": None,
                        "data_points": 0,
                        "calculation_method": "N/A",
                    })
                    continue

                # Deterministic calculation based on unit type
                if pai_unit == "percentage":
                    # Weighted average for percentage-based indicators
                    total_weight = sum(
                        hv["weight_pct"] for hv in available
                    )
                    if total_weight > 0:
                        calc_value = sum(
                            hv["value"] * hv["weight_pct"]
                            for hv in available
                        ) / total_weight
                    else:
                        calc_value = sum(
                            hv["value"] for hv in available
                        ) / len(available)
                    calc_method = "weighted_average"

                elif pai_unit in ("tCO2e", "tonnes"):
                    # Sum for absolute values, then normalize by weight
                    calc_value = sum(
                        hv["value"] * hv["weight_pct"] / 100.0
                        for hv in available
                    )
                    calc_method = "weighted_sum"

                elif "EUR million" in pai_unit:
                    # Intensity: weighted average
                    total_weight = sum(
                        hv["weight_pct"] for hv in available
                    )
                    if total_weight > 0:
                        calc_value = sum(
                            hv["value"] * hv["weight_pct"]
                            for hv in available
                        ) / total_weight
                    else:
                        calc_value = 0.0
                    calc_method = "weighted_average_intensity"

                else:
                    # Default: weighted average
                    total_weight = sum(
                        hv["weight_pct"] for hv in available
                    )
                    if total_weight > 0:
                        calc_value = sum(
                            hv["value"] * hv["weight_pct"]
                            for hv in available
                        ) / total_weight
                    else:
                        calc_value = 0.0
                    calc_method = "weighted_average"

                calc_value = round(calc_value, 4)

                # Prior period comparison
                prior_val = prior_values.get(pai_id)
                change_pct = None
                if prior_val is not None and prior_val != 0:
                    change_pct = round(
                        (calc_value - prior_val) / abs(prior_val) * 100.0,
                        2,
                    )

                if pai_category == PAICategory.CLIMATE_ENVIRONMENTAL.value:
                    climate_count += 1
                else:
                    social_count += 1

                calculated_indicators.append({
                    "pai_id": pai_id,
                    "name": pai_name,
                    "unit": pai_unit,
                    "category": pai_category,
                    "calculated_value": calc_value,
                    "prior_value": prior_val,
                    "change_pct": change_pct,
                    "data_points": len(available),
                    "calculation_method": calc_method,
                })

            indicators_calculated = sum(
                1 for i in calculated_indicators
                if i["calculated_value"] is not None
            )

            outputs["calculated_indicators"] = calculated_indicators
            outputs["indicators_calculated"] = indicators_calculated
            outputs["climate_indicators_count"] = climate_count
            outputs["social_indicators_count"] = social_count
            outputs["portfolio_value_eur"] = portfolio_value

            if indicators_calculated < len(MANDATORY_PAI_INDICATORS):
                missing = len(MANDATORY_PAI_INDICATORS) - indicators_calculated
                warnings.append(
                    f"{missing} mandatory PAI indicator(s) could not "
                    f"be calculated due to insufficient data"
                )

            status = PhaseStatus.COMPLETED
            records = len(calculated_indicators)

        except Exception as exc:
            logger.error(
                "PAICalculation failed: %s", exc, exc_info=True
            )
            errors.append(f"PAI calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class IntegrationAssessmentPhase:
    """
    Phase 3: Integration Assessment.

    Assesses how PAI considerations are integrated into the investment
    decision-making process, including screening, engagement, voting,
    and exclusion mechanisms.
    """

    PHASE_NAME = "integration_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute integration assessment phase.

        Args:
            context: Workflow context with calculated PAI values.

        Returns:
            PhaseResult with integration assessment results.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            calc_output = context.get_phase_output("pai_calculation")
            calculated = calc_output.get("calculated_indicators", [])
            engagement_activities = config.get(
                "engagement_activities", []
            )
            existing_actions = config.get("existing_actions", [])

            # Assess integration mechanisms
            integration_checks = []

            # Check 1: Pre-investment screening
            has_screening = any(
                a.get("type") == "screening" for a in existing_actions
            )
            integration_checks.append({
                "mechanism": "pre_investment_screening",
                "implemented": has_screening,
                "description": (
                    "PAI indicators are considered during "
                    "pre-investment screening"
                    if has_screening
                    else "No PAI screening process documented"
                ),
            })

            # Check 2: Ongoing monitoring
            has_monitoring = any(
                a.get("type") == "monitoring" for a in existing_actions
            )
            integration_checks.append({
                "mechanism": "ongoing_monitoring",
                "implemented": has_monitoring,
                "description": (
                    "PAI indicators are monitored on an ongoing basis"
                    if has_monitoring
                    else "No ongoing PAI monitoring documented"
                ),
            })

            # Check 3: Engagement
            has_engagement = len(engagement_activities) > 0
            integration_checks.append({
                "mechanism": "engagement",
                "implemented": has_engagement,
                "description": (
                    f"{len(engagement_activities)} engagement "
                    f"activity(ies) documented"
                    if has_engagement
                    else "No engagement activities documented"
                ),
                "activities_count": len(engagement_activities),
            })

            # Check 4: Exclusion policy
            has_exclusion = any(
                a.get("type") == "exclusion" for a in existing_actions
            )
            integration_checks.append({
                "mechanism": "exclusion_policy",
                "implemented": has_exclusion,
                "description": (
                    "PAI-based exclusion criteria are applied"
                    if has_exclusion
                    else "No PAI-based exclusion policy documented"
                ),
            })

            # Check 5: Proxy voting
            has_voting = any(
                a.get("type") == "voting" for a in existing_actions
            )
            integration_checks.append({
                "mechanism": "proxy_voting",
                "implemented": has_voting,
                "description": (
                    "PAI considerations integrated in proxy voting"
                    if has_voting
                    else "No PAI-aligned voting policy documented"
                ),
            })

            # Calculate integration score
            implemented_count = sum(
                1 for c in integration_checks if c["implemented"]
            )
            total_mechanisms = len(integration_checks)
            integration_score = round(
                implemented_count / total_mechanisms * 100.0
                if total_mechanisms > 0 else 0.0, 1
            )

            # Identify high-impact PAIs needing attention
            high_impact_pais = []
            for indicator in calculated:
                change = indicator.get("change_pct")
                if change is not None and change > 10.0:
                    high_impact_pais.append({
                        "pai_id": indicator["pai_id"],
                        "name": indicator["name"],
                        "change_pct": change,
                        "direction": "worsening",
                    })
                elif change is not None and change < -10.0:
                    high_impact_pais.append({
                        "pai_id": indicator["pai_id"],
                        "name": indicator["name"],
                        "change_pct": change,
                        "direction": "improving",
                    })

            outputs["integration_checks"] = integration_checks
            outputs["integration_score"] = integration_score
            outputs["mechanisms_implemented"] = implemented_count
            outputs["total_mechanisms"] = total_mechanisms
            outputs["engagement_activities_count"] = len(
                engagement_activities
            )
            outputs["existing_actions_count"] = len(existing_actions)
            outputs["high_impact_pais"] = high_impact_pais

            if integration_score < 60.0:
                warnings.append(
                    f"PAI integration score is low: {integration_score:.1f}%. "
                    f"Article 9 products should demonstrate strong "
                    f"PAI integration."
                )

            status = PhaseStatus.COMPLETED
            records = len(integration_checks)

        except Exception as exc:
            logger.error(
                "IntegrationAssessment failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Integration assessment failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


class ActionPlanningPhase:
    """
    Phase 4: Action Planning.

    Defines concrete actions, targets, and timelines for PAI mitigation
    and improvement. Prioritizes actions based on impact severity and
    regulatory requirements.
    """

    PHASE_NAME = "action_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute action planning phase.

        Args:
            context: Workflow context with PAI calculations and
                integration assessment.

        Returns:
            PhaseResult with action plans and targets.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            calc_output = context.get_phase_output("pai_calculation")
            integration_output = context.get_phase_output(
                "integration_assessment"
            )
            calculated = calc_output.get("calculated_indicators", [])
            high_impact = integration_output.get(
                "high_impact_pais", []
            )
            integration_checks = integration_output.get(
                "integration_checks", []
            )

            action_plans: List[Dict[str, Any]] = []

            # Generate actions for worsening PAIs
            for pai in high_impact:
                if pai.get("direction") == "worsening":
                    priority = ActionPriority.HIGH.value
                    action_plans.append({
                        "action_id": str(uuid.uuid4()),
                        "pai_id": pai["pai_id"],
                        "pai_name": pai["name"],
                        "priority": priority,
                        "action_type": "mitigation",
                        "description": (
                            f"Address worsening trend in "
                            f"'{pai['name']}' "
                            f"(change: {pai['change_pct']:+.1f}%)"
                        ),
                        "target": (
                            f"Reduce {pai['name']} by "
                            f"{abs(pai['change_pct']):.1f}% "
                            f"within next reporting period"
                        ),
                        "timeline": "12 months",
                        "responsible": "Investment team",
                        "status": "PLANNED",
                    })

            # Generate actions for unimplemented integration mechanisms
            for check in integration_checks:
                if not check.get("implemented"):
                    mechanism = check.get("mechanism", "")
                    action_plans.append({
                        "action_id": str(uuid.uuid4()),
                        "pai_id": "ALL",
                        "pai_name": f"Integration: {mechanism}",
                        "priority": ActionPriority.MEDIUM.value,
                        "action_type": "process_improvement",
                        "description": (
                            f"Implement {mechanism.replace('_', ' ')} "
                            f"mechanism for PAI integration"
                        ),
                        "target": (
                            f"Establish documented {mechanism} process"
                        ),
                        "timeline": "6 months",
                        "responsible": "Compliance team",
                        "status": "PLANNED",
                    })

            # Generate data quality improvement actions
            for indicator in calculated:
                if indicator.get("calculated_value") is None:
                    action_plans.append({
                        "action_id": str(uuid.uuid4()),
                        "pai_id": indicator["pai_id"],
                        "pai_name": indicator["name"],
                        "priority": ActionPriority.HIGH.value,
                        "action_type": "data_improvement",
                        "description": (
                            f"Improve data availability for mandatory "
                            f"PAI indicator '{indicator['name']}'"
                        ),
                        "target": (
                            f"Achieve minimum 70% data coverage for "
                            f"{indicator['name']}"
                        ),
                        "timeline": "6 months",
                        "responsible": "Data team",
                        "status": "PLANNED",
                    })

            # Sort by priority
            priority_order = {
                ActionPriority.CRITICAL.value: 0,
                ActionPriority.HIGH.value: 1,
                ActionPriority.MEDIUM.value: 2,
                ActionPriority.LOW.value: 3,
            }
            action_plans.sort(
                key=lambda a: priority_order.get(
                    a.get("priority", "LOW"), 99
                )
            )

            high_priority_count = sum(
                1 for a in action_plans
                if a.get("priority") in (
                    ActionPriority.HIGH.value,
                    ActionPriority.CRITICAL.value,
                )
            )

            outputs["action_plans"] = action_plans
            outputs["actions_planned"] = len(action_plans)
            outputs["high_priority_actions"] = high_priority_count
            outputs["action_types"] = {
                "mitigation": sum(
                    1 for a in action_plans
                    if a["action_type"] == "mitigation"
                ),
                "process_improvement": sum(
                    1 for a in action_plans
                    if a["action_type"] == "process_improvement"
                ),
                "data_improvement": sum(
                    1 for a in action_plans
                    if a["action_type"] == "data_improvement"
                ),
            }
            outputs["generated_at"] = _utcnow().isoformat()

            if high_priority_count > 0:
                warnings.append(
                    f"{high_priority_count} high/critical priority "
                    f"action(s) require attention"
                )

            status = PhaseStatus.COMPLETED
            records = len(action_plans)

        except Exception as exc:
            logger.error(
                "ActionPlanning failed: %s", exc, exc_info=True
            )
            errors.append(f"Action planning failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = _utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class PAIMandatoryWorkflow:
    """
    Four-phase mandatory PAI indicator workflow for Article 9.

    Orchestrates the complete PAI assessment pipeline from data sourcing
    through calculation, integration assessment, and action planning.
    Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = PAIMandatoryWorkflow()
        >>> input_data = PAIMandatoryInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     reporting_date="2026-01-01",
        ...     reporting_period_start="2025-01-01",
        ...     reporting_period_end="2025-12-31",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "pai_mandatory"

    PHASE_ORDER = [
        "data_sourcing",
        "pai_calculation",
        "integration_assessment",
        "action_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the mandatory PAI workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_sourcing": DataSourcingPhase(),
            "pai_calculation": PAICalculationPhase(),
            "integration_assessment": IntegrationAssessmentPhase(),
            "action_planning": ActionPlanningPhase(),
        }

    async def run(
        self, input_data: PAIMandatoryInput
    ) -> PAIMandatoryResult:
        """
        Execute the complete 4-phase mandatory PAI workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            PAIMandatoryResult with per-phase details and summary.
        """
        started_at = _utcnow()
        logger.info(
            "Starting mandatory PAI workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                logger.info(
                    "Phase '%s' already completed, skipping",
                    phase_name,
                )
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "data_sourcing":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting",
                            phase_name,
                        )
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=_utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
            )

        completed_at = _utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Mandatory PAI workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return PAIMandatoryResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            reporting_period=summary.get("reporting_period", ""),
            total_indicators=14,
            indicators_calculated=summary.get(
                "indicators_calculated", 0
            ),
            indicators_with_data=summary.get(
                "indicators_with_data", 0
            ),
            data_coverage_pct=summary.get("data_coverage_pct", 0.0),
            climate_indicators_count=summary.get(
                "climate_indicators_count", 0
            ),
            social_indicators_count=summary.get(
                "social_indicators_count", 0
            ),
            integration_score=summary.get("integration_score", 0.0),
            actions_planned=summary.get("actions_planned", 0),
            high_priority_actions=summary.get(
                "high_priority_actions", 0
            ),
        )

    def _build_config(
        self, input_data: PAIMandatoryInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return input_data.model_dump()

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        data_out = context.get_phase_output("data_sourcing")
        calc_out = context.get_phase_output("pai_calculation")
        integration_out = context.get_phase_output(
            "integration_assessment"
        )
        action_out = context.get_phase_output("action_planning")

        return {
            "product_name": data_out.get("product_name", ""),
            "reporting_period": (
                f"{context.config.get('reporting_period_start', '')} to "
                f"{context.config.get('reporting_period_end', '')}"
            ),
            "indicators_calculated": calc_out.get(
                "indicators_calculated", 0
            ),
            "indicators_with_data": data_out.get(
                "indicators_with_data", 0
            ),
            "data_coverage_pct": data_out.get(
                "overall_coverage_pct", 0.0
            ),
            "climate_indicators_count": calc_out.get(
                "climate_indicators_count", 0
            ),
            "social_indicators_count": calc_out.get(
                "social_indicators_count", 0
            ),
            "integration_score": integration_out.get(
                "integration_score", 0.0
            ),
            "actions_planned": action_out.get(
                "actions_planned", 0
            ),
            "high_priority_actions": action_out.get(
                "high_priority_actions", 0
            ),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
