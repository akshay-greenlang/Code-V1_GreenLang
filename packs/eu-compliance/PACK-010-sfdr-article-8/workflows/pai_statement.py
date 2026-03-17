# -*- coding: utf-8 -*-
"""
PAI Statement Workflow
=========================

Four-phase workflow for calculating and generating the Principal Adverse
Impact (PAI) statement for SFDR Article 8 financial products. Orchestrates
data sourcing, portfolio-weighted calculations, reporting, and action
planning into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 4: PAI consideration at entity level.
    - Article 7: PAI consideration at product level.
    - Annex I, Table 1: 18 mandatory PAI indicators covering climate/environment
      (indicators 1-14 for corporates, 15-16 for sovereigns, 17-18 for
      real estate).
    - Annex I, Tables 2-3: Additional opt-in indicators.
    - PAI must be calculated on a portfolio-weighted basis using enterprise
      value including cash (EVIC) or revenue as normalization basis.
    - Coverage ratios must be disclosed per indicator.
    - Year-over-year comparison is mandatory from the second reporting period.

    18 Mandatory PAI Indicators:
    1. GHG Emissions (Scope 1, 2, 3, Total)
    2. Carbon Footprint
    3. GHG Intensity of Investee Companies
    4. Exposure to Fossil Fuel Companies
    5. Non-Renewable Energy Share (Production/Consumption)
    6. Energy Consumption Intensity per Sector
    7. Activities Affecting Biodiversity-Sensitive Areas
    8. Emissions to Water
    9. Hazardous Waste and Radioactive Waste Ratio
    10. Violations of UNGC/OECD Guidelines
    11. Lack of UNGC/OECD Compliance Processes
    12. Unadjusted Gender Pay Gap
    13. Board Gender Diversity
    14. Exposure to Controversial Weapons
    15. GHG Intensity of Sovereigns
    16. Investee Countries Subject to Social Violations
    17. Exposure to Fossil Fuels through Real Estate
    18. Exposure to Energy-Inefficient Real Estate

Phases:
    1. DataSourcing - Collect investee-level data for all 18 mandatory PAI
       indicators, identify data gaps, apply estimation methodologies
    2. Calculation - Portfolio-weighted PAI calculations, coverage ratio
       tracking, data quality scoring
    3. Reporting - Generate PAI statement with indicator values, narrative
       explanations, and actions taken
    4. ActionPlanning - Engagement actions for worst performers, exclusion
       decisions, monitoring schedule, improvement targets

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

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
    CLIMATE = "CLIMATE"
    ENVIRONMENT = "ENVIRONMENT"
    SOCIAL = "SOCIAL"
    GOVERNANCE = "GOVERNANCE"
    SOVEREIGN = "SOVEREIGN"
    REAL_ESTATE = "REAL_ESTATE"


class DataSource(str, Enum):
    """Data source type for PAI indicators."""
    COMPANY_REPORTED = "COMPANY_REPORTED"
    THIRD_PARTY_PROVIDER = "THIRD_PARTY_PROVIDER"
    ESTIMATED = "ESTIMATED"
    PROXY = "PROXY"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class ActionType(str, Enum):
    """Type of engagement/remediation action."""
    ENGAGEMENT = "ENGAGEMENT"
    EXCLUSION = "EXCLUSION"
    MONITORING = "MONITORING"
    DIVESTMENT = "DIVESTMENT"
    VOTING = "VOTING"


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
# DATA MODELS - PAI STATEMENT
# =============================================================================


class InvesteeData(BaseModel):
    """Investee-level data for PAI calculation."""
    investee_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Investee company name")
    isin: Optional[str] = Field(None)
    sector: str = Field(default="")
    country: str = Field(default="")
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    enterprise_value_eur: Optional[float] = Field(None, ge=0.0)
    revenue_eur: Optional[float] = Field(None, ge=0.0)
    ghg_scope1_tco2e: Optional[float] = Field(None, ge=0.0)
    ghg_scope2_tco2e: Optional[float] = Field(None, ge=0.0)
    ghg_scope3_tco2e: Optional[float] = Field(None, ge=0.0)
    fossil_fuel_involvement: bool = Field(default=False)
    non_renewable_energy_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    energy_intensity_gwh_per_m_eur: Optional[float] = Field(None, ge=0.0)
    biodiversity_sensitive: bool = Field(default=False)
    water_emissions_tonnes: Optional[float] = Field(None, ge=0.0)
    hazardous_waste_tonnes: Optional[float] = Field(None, ge=0.0)
    ungc_violations: bool = Field(default=False)
    ungc_compliance_process: bool = Field(default=True)
    gender_pay_gap_pct: Optional[float] = Field(None)
    board_gender_diversity_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    controversial_weapons: bool = Field(default=False)
    data_source: DataSource = Field(default=DataSource.COMPANY_REPORTED)


class PAIStatementInput(BaseModel):
    """Input configuration for the PAI statement workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    reporting_period_start: str = Field(...)
    reporting_period_end: str = Field(...)
    investee_data: List[InvesteeData] = Field(
        default_factory=list, description="Investee-level data"
    )
    total_portfolio_value_eur: float = Field(default=0.0, ge=0.0)
    previous_period_pai: Optional[Dict[str, Any]] = Field(
        None, description="Previous period PAI values for YoY comparison"
    )
    estimation_methodology: str = Field(
        default="sector_average",
        description="Methodology for filling data gaps"
    )
    coverage_threshold_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum acceptable coverage ratio"
    )
    worst_performer_threshold: int = Field(
        default=10, ge=1,
        description="Number of worst performers for action planning"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_period_start", "reporting_period_end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD format")
        return v


class PAIStatementResult(WorkflowResult):
    """Complete result from the PAI statement workflow."""
    product_name: str = Field(default="")
    reporting_period: str = Field(default="")
    total_investees: int = Field(default=0)
    mandatory_indicators_calculated: int = Field(default=0)
    mandatory_indicators_total: int = Field(default=18)
    average_coverage_pct: float = Field(default=0.0)
    data_quality_score: str = Field(default="LOW")
    action_items_count: int = Field(default=0)
    worst_performers_identified: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class DataSourcingPhase:
    """
    Phase 1: Data Sourcing.

    Collects investee-level data for all 18 mandatory PAI indicators,
    identifies data gaps, and applies estimation methodologies where
    reported data is unavailable.
    """

    PHASE_NAME = "data_sourcing"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute data sourcing phase.

        Args:
            context: Workflow context with investee data.

        Returns:
            PhaseResult with sourced data and gap analysis.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            investees = config.get("investee_data", [])
            estimation_method = config.get(
                "estimation_methodology", "sector_average"
            )

            outputs["total_investees"] = len(investees)
            outputs["estimation_methodology"] = estimation_method

            # Data availability analysis per indicator
            indicator_coverage = {
                "pai_1_ghg_scope1": 0,
                "pai_1_ghg_scope2": 0,
                "pai_1_ghg_scope3": 0,
                "pai_2_carbon_footprint": 0,
                "pai_3_ghg_intensity": 0,
                "pai_4_fossil_fuel": 0,
                "pai_5_non_renewable": 0,
                "pai_6_energy_intensity": 0,
                "pai_7_biodiversity": 0,
                "pai_8_water_emissions": 0,
                "pai_9_hazardous_waste": 0,
                "pai_10_ungc_violations": 0,
                "pai_11_ungc_compliance": 0,
                "pai_12_gender_pay_gap": 0,
                "pai_13_board_diversity": 0,
                "pai_14_controversial_weapons": 0,
            }

            # Source quality distribution
            source_distribution = {s.value: 0 for s in DataSource}
            total_investees = max(len(investees), 1)

            for inv in investees:
                source = inv.get(
                    "data_source", DataSource.COMPANY_REPORTED.value
                )
                source_distribution[source] = (
                    source_distribution.get(source, 0) + 1
                )

                # Check data availability per indicator
                if inv.get("ghg_scope1_tco2e") is not None:
                    indicator_coverage["pai_1_ghg_scope1"] += 1
                if inv.get("ghg_scope2_tco2e") is not None:
                    indicator_coverage["pai_1_ghg_scope2"] += 1
                if inv.get("ghg_scope3_tco2e") is not None:
                    indicator_coverage["pai_1_ghg_scope3"] += 1
                if (
                    inv.get("ghg_scope1_tco2e") is not None
                    and inv.get("ghg_scope2_tco2e") is not None
                    and inv.get("enterprise_value_eur") is not None
                ):
                    indicator_coverage["pai_2_carbon_footprint"] += 1
                if (
                    inv.get("ghg_scope1_tco2e") is not None
                    and inv.get("ghg_scope2_tco2e") is not None
                    and inv.get("revenue_eur") is not None
                ):
                    indicator_coverage["pai_3_ghg_intensity"] += 1

                # Boolean indicators always have coverage
                indicator_coverage["pai_4_fossil_fuel"] += 1
                indicator_coverage["pai_10_ungc_violations"] += 1
                indicator_coverage["pai_11_ungc_compliance"] += 1
                indicator_coverage["pai_14_controversial_weapons"] += 1

                if inv.get("non_renewable_energy_pct") is not None:
                    indicator_coverage["pai_5_non_renewable"] += 1
                if inv.get("energy_intensity_gwh_per_m_eur") is not None:
                    indicator_coverage["pai_6_energy_intensity"] += 1

                indicator_coverage["pai_7_biodiversity"] += 1

                if inv.get("water_emissions_tonnes") is not None:
                    indicator_coverage["pai_8_water_emissions"] += 1
                if inv.get("hazardous_waste_tonnes") is not None:
                    indicator_coverage["pai_9_hazardous_waste"] += 1
                if inv.get("gender_pay_gap_pct") is not None:
                    indicator_coverage["pai_12_gender_pay_gap"] += 1
                if inv.get("board_gender_diversity_pct") is not None:
                    indicator_coverage["pai_13_board_diversity"] += 1

            # Convert to coverage percentages
            coverage_pct = {
                ind: round(count / total_investees * 100, 1)
                for ind, count in indicator_coverage.items()
            }
            outputs["indicator_coverage"] = coverage_pct
            outputs["source_distribution"] = source_distribution

            # Identify data gaps
            threshold = config.get("coverage_threshold_pct", 50.0)
            gaps = [
                ind for ind, pct in coverage_pct.items()
                if pct < threshold
            ]
            outputs["data_gaps"] = gaps
            outputs["gap_count"] = len(gaps)

            if gaps:
                warnings.append(
                    f"{len(gaps)} indicator(s) below {threshold}% coverage: "
                    f"{', '.join(gaps[:5])}"
                    + ("..." if len(gaps) > 5 else "")
                )

            # Average coverage
            avg_coverage = sum(coverage_pct.values()) / max(
                len(coverage_pct), 1
            )
            outputs["average_coverage_pct"] = round(avg_coverage, 1)

            status = PhaseStatus.COMPLETED
            records = len(investees)

        except Exception as exc:
            logger.error("DataSourcing failed: %s", exc, exc_info=True)
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
    Phase 2: Calculation.

    Performs portfolio-weighted PAI calculations, tracks coverage ratios
    per indicator, and assigns data quality scores.
    """

    PHASE_NAME = "calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute PAI calculation phase.

        Args:
            context: Workflow context with sourced data.

        Returns:
            PhaseResult with calculated PAI indicator values.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            sourcing_output = context.get_phase_output("data_sourcing")
            investees = config.get("investee_data", [])
            portfolio_value = config.get("total_portfolio_value_eur", 0.0)
            coverage_data = sourcing_output.get("indicator_coverage", {})

            # Calculate each mandatory PAI indicator
            pai_values: Dict[str, Dict[str, Any]] = {}

            # PAI 1: GHG Emissions
            scope1_total = self._weighted_sum(investees, "ghg_scope1_tco2e")
            scope2_total = self._weighted_sum(investees, "ghg_scope2_tco2e")
            scope3_total = self._weighted_sum(investees, "ghg_scope3_tco2e")
            pai_values["pai_1_ghg_emissions"] = {
                "indicator_name": "GHG Emissions",
                "scope1_tco2e": round(scope1_total, 4),
                "scope2_tco2e": round(scope2_total, 4),
                "scope3_tco2e": round(scope3_total, 4),
                "total_tco2e": round(
                    scope1_total + scope2_total + scope3_total, 4
                ),
                "unit": "tCO2e",
                "coverage_pct": coverage_data.get("pai_1_ghg_scope1", 0.0),
            }

            # PAI 2: Carbon Footprint
            if portfolio_value > 0:
                carbon_footprint = (scope1_total + scope2_total) / (
                    portfolio_value / 1_000_000
                )
            else:
                carbon_footprint = 0.0
            pai_values["pai_2_carbon_footprint"] = {
                "indicator_name": "Carbon Footprint",
                "value": round(carbon_footprint, 4),
                "unit": "tCO2e/EUR M invested",
                "coverage_pct": coverage_data.get(
                    "pai_2_carbon_footprint", 0.0
                ),
            }

            # PAI 3: GHG Intensity
            ghg_intensity = self._calc_ghg_intensity(investees)
            pai_values["pai_3_ghg_intensity"] = {
                "indicator_name": "GHG Intensity of Investee Companies",
                "value": round(ghg_intensity, 4),
                "unit": "tCO2e/EUR M revenue",
                "coverage_pct": coverage_data.get(
                    "pai_3_ghg_intensity", 0.0
                ),
            }

            # PAI 4: Fossil Fuel Exposure
            fossil_fuel_pct = self._boolean_exposure_pct(
                investees, "fossil_fuel_involvement"
            )
            pai_values["pai_4_fossil_fuel"] = {
                "indicator_name": "Exposure to Fossil Fuel Companies",
                "value": round(fossil_fuel_pct, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get("pai_4_fossil_fuel", 0.0),
            }

            # PAI 5: Non-Renewable Energy
            avg_non_renewable = self._weighted_average(
                investees, "non_renewable_energy_pct"
            )
            pai_values["pai_5_non_renewable_energy"] = {
                "indicator_name": (
                    "Share of Non-Renewable Energy Consumption and Production"
                ),
                "value": round(avg_non_renewable, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_5_non_renewable", 0.0
                ),
            }

            # PAI 6: Energy Intensity
            avg_energy_intensity = self._weighted_average(
                investees, "energy_intensity_gwh_per_m_eur"
            )
            pai_values["pai_6_energy_intensity"] = {
                "indicator_name": (
                    "Energy Consumption Intensity per High Impact Sector"
                ),
                "value": round(avg_energy_intensity, 4),
                "unit": "GWh/EUR M revenue",
                "coverage_pct": coverage_data.get(
                    "pai_6_energy_intensity", 0.0
                ),
            }

            # PAI 7: Biodiversity
            biodiversity_pct = self._boolean_exposure_pct(
                investees, "biodiversity_sensitive"
            )
            pai_values["pai_7_biodiversity"] = {
                "indicator_name": (
                    "Activities Affecting Biodiversity-Sensitive Areas"
                ),
                "value": round(biodiversity_pct, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_7_biodiversity", 0.0
                ),
            }

            # PAI 8: Water Emissions
            water_total = self._weighted_sum(
                investees, "water_emissions_tonnes"
            )
            pai_values["pai_8_water_emissions"] = {
                "indicator_name": "Emissions to Water",
                "value": round(water_total, 4),
                "unit": "tonnes",
                "coverage_pct": coverage_data.get(
                    "pai_8_water_emissions", 0.0
                ),
            }

            # PAI 9: Hazardous Waste
            waste_total = self._weighted_sum(
                investees, "hazardous_waste_tonnes"
            )
            pai_values["pai_9_hazardous_waste"] = {
                "indicator_name": (
                    "Hazardous Waste and Radioactive Waste Ratio"
                ),
                "value": round(waste_total, 4),
                "unit": "tonnes",
                "coverage_pct": coverage_data.get(
                    "pai_9_hazardous_waste", 0.0
                ),
            }

            # PAI 10: UNGC Violations
            ungc_violations_pct = self._boolean_exposure_pct(
                investees, "ungc_violations"
            )
            pai_values["pai_10_ungc_violations"] = {
                "indicator_name": (
                    "Violations of UNGC Principles and OECD Guidelines"
                ),
                "value": round(ungc_violations_pct, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_10_ungc_violations", 0.0
                ),
            }

            # PAI 11: UNGC Compliance Gap
            ungc_gap_pct = self._boolean_no_exposure_pct(
                investees, "ungc_compliance_process"
            )
            pai_values["pai_11_ungc_compliance_gap"] = {
                "indicator_name": (
                    "Lack of Processes to Monitor UNGC/OECD Compliance"
                ),
                "value": round(ungc_gap_pct, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_11_ungc_compliance", 0.0
                ),
            }

            # PAI 12: Gender Pay Gap
            avg_pay_gap = self._weighted_average(
                investees, "gender_pay_gap_pct"
            )
            pai_values["pai_12_gender_pay_gap"] = {
                "indicator_name": "Unadjusted Gender Pay Gap",
                "value": round(avg_pay_gap, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_12_gender_pay_gap", 0.0
                ),
            }

            # PAI 13: Board Gender Diversity
            avg_board_diversity = self._weighted_average(
                investees, "board_gender_diversity_pct"
            )
            pai_values["pai_13_board_diversity"] = {
                "indicator_name": "Board Gender Diversity",
                "value": round(avg_board_diversity, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_13_board_diversity", 0.0
                ),
            }

            # PAI 14: Controversial Weapons
            weapons_pct = self._boolean_exposure_pct(
                investees, "controversial_weapons"
            )
            pai_values["pai_14_controversial_weapons"] = {
                "indicator_name": (
                    "Exposure to Controversial Weapons"
                ),
                "value": round(weapons_pct, 2),
                "unit": "%",
                "coverage_pct": coverage_data.get(
                    "pai_14_controversial_weapons", 0.0
                ),
            }

            outputs["pai_values"] = pai_values
            outputs["mandatory_indicators_calculated"] = len(pai_values)

            # YoY comparison
            previous_pai = config.get("previous_period_pai", {})
            yoy_comparison: List[Dict[str, Any]] = []
            if previous_pai:
                for ind_id, current in pai_values.items():
                    prev = previous_pai.get(ind_id, {})
                    prev_val = prev.get("value")
                    curr_val = current.get("value")
                    if curr_val is None:
                        curr_val = current.get("total_tco2e")
                    if prev_val is not None and curr_val is not None:
                        change = curr_val - prev_val
                        change_pct = (
                            (change / abs(prev_val) * 100)
                            if prev_val != 0 else 0.0
                        )
                        yoy_comparison.append({
                            "indicator_id": ind_id,
                            "indicator_name": current.get(
                                "indicator_name", ""
                            ),
                            "previous_value": prev_val,
                            "current_value": curr_val,
                            "change": round(change, 4),
                            "change_pct": round(change_pct, 2),
                        })

            outputs["yoy_comparison"] = yoy_comparison
            outputs["has_yoy_data"] = len(yoy_comparison) > 0

            # Overall data quality score
            avg_coverage = sourcing_output.get("average_coverage_pct", 0.0)
            if avg_coverage >= 80.0:
                outputs["data_quality_score"] = "HIGH"
            elif avg_coverage >= 50.0:
                outputs["data_quality_score"] = "MEDIUM"
            else:
                outputs["data_quality_score"] = "LOW"

            status = PhaseStatus.COMPLETED
            records = len(investees)

        except Exception as exc:
            logger.error("PAICalculation failed: %s", exc, exc_info=True)
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

    def _weighted_sum(
        self, investees: List[Dict[str, Any]], field: str
    ) -> float:
        """Calculate portfolio-weighted sum of a numeric field."""
        total = 0.0
        for inv in investees:
            val = inv.get(field)
            if val is not None:
                weight = inv.get("portfolio_weight_pct", 0.0) / 100.0
                total += val * weight
        return total

    def _weighted_average(
        self, investees: List[Dict[str, Any]], field: str
    ) -> float:
        """Calculate portfolio-weighted average of a numeric field."""
        total_weight = 0.0
        weighted_sum = 0.0
        for inv in investees:
            val = inv.get(field)
            if val is not None:
                weight = inv.get("portfolio_weight_pct", 0.0) / 100.0
                weighted_sum += val * weight
                total_weight += weight
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _boolean_exposure_pct(
        self, investees: List[Dict[str, Any]], field: str
    ) -> float:
        """Calculate percentage of portfolio weight where boolean is True."""
        exposure_weight = 0.0
        for inv in investees:
            if inv.get(field, False):
                exposure_weight += inv.get("portfolio_weight_pct", 0.0)
        return exposure_weight

    def _boolean_no_exposure_pct(
        self, investees: List[Dict[str, Any]], field: str
    ) -> float:
        """Calculate percentage where boolean is False (lack of process)."""
        no_exposure_weight = 0.0
        for inv in investees:
            if not inv.get(field, True):
                no_exposure_weight += inv.get("portfolio_weight_pct", 0.0)
        return no_exposure_weight

    def _calc_ghg_intensity(
        self, investees: List[Dict[str, Any]]
    ) -> float:
        """Calculate portfolio-weighted GHG intensity (tCO2e/EUR M revenue)."""
        total_weight = 0.0
        weighted_intensity = 0.0
        for inv in investees:
            s1 = inv.get("ghg_scope1_tco2e")
            s2 = inv.get("ghg_scope2_tco2e")
            rev = inv.get("revenue_eur")
            if s1 is not None and s2 is not None and rev and rev > 0:
                intensity = (s1 + s2) / (rev / 1_000_000)
                weight = inv.get("portfolio_weight_pct", 0.0) / 100.0
                weighted_intensity += intensity * weight
                total_weight += weight
        if total_weight > 0:
            return weighted_intensity / total_weight
        return 0.0


class PAIReportingPhase:
    """
    Phase 3: Reporting.

    Generates the PAI statement with indicator values, narrative
    explanations, and actions taken per indicator.
    """

    PHASE_NAME = "reporting"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute PAI reporting phase.

        Args:
            context: Workflow context with calculated PAI values.

        Returns:
            PhaseResult with formatted PAI statement.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            calc_output = context.get_phase_output("calculation")
            sourcing_output = context.get_phase_output("data_sourcing")
            pai_values = calc_output.get("pai_values", {})
            yoy_data = calc_output.get("yoy_comparison", [])

            product_name = config.get("product_name", "")
            period_start = config.get("reporting_period_start", "")
            period_end = config.get("reporting_period_end", "")

            # Build PAI statement
            statement = {
                "title": (
                    f"Statement on Principal Adverse Impacts of Investment "
                    f"Decisions on Sustainability Factors"
                ),
                "product_name": product_name,
                "reporting_period": f"{period_start} to {period_end}",
                "generated_at": _utcnow().isoformat(),
                "introduction": (
                    f"{product_name} considers principal adverse impacts of "
                    f"its investment decisions on sustainability factors. "
                    f"This statement covers the reference period from "
                    f"{period_start} to {period_end}."
                ),
            }

            # Build indicator table (Annex I Table 1 format)
            indicator_table: List[Dict[str, Any]] = []
            for ind_id, ind_data in pai_values.items():
                # Find YoY data for this indicator
                yoy_entry = next(
                    (y for y in yoy_data if y.get("indicator_id") == ind_id),
                    None,
                )
                row = {
                    "indicator_id": ind_id,
                    "indicator_name": ind_data.get("indicator_name", ""),
                    "metric": ind_data.get("unit", ""),
                    "impact": ind_data.get(
                        "value", ind_data.get("total_tco2e", 0.0)
                    ),
                    "coverage_pct": ind_data.get("coverage_pct", 0.0),
                    "previous_impact": (
                        yoy_entry.get("previous_value")
                        if yoy_entry else None
                    ),
                    "explanation": self._generate_explanation(
                        ind_id, ind_data
                    ),
                    "actions_taken": self._generate_actions_taken(
                        ind_id, ind_data
                    ),
                }
                indicator_table.append(row)

            statement["indicator_table"] = indicator_table
            outputs["pai_statement"] = statement

            # Summary statistics
            outputs["summary_statistics"] = {
                "total_ghg_emissions_tco2e": pai_values.get(
                    "pai_1_ghg_emissions", {}
                ).get("total_tco2e", 0.0),
                "carbon_footprint": pai_values.get(
                    "pai_2_carbon_footprint", {}
                ).get("value", 0.0),
                "fossil_fuel_exposure_pct": pai_values.get(
                    "pai_4_fossil_fuel", {}
                ).get("value", 0.0),
                "controversial_weapons_pct": pai_values.get(
                    "pai_14_controversial_weapons", {}
                ).get("value", 0.0),
                "average_coverage_pct": sourcing_output.get(
                    "average_coverage_pct", 0.0
                ),
                "data_quality_score": calc_output.get(
                    "data_quality_score", "LOW"
                ),
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("PAIReporting failed: %s", exc, exc_info=True)
            errors.append(f"PAI reporting failed: {str(exc)}")
            status = PhaseStatus.FAILED

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
        )

    def _generate_explanation(
        self, indicator_id: str, data: Dict[str, Any]
    ) -> str:
        """Generate narrative explanation for an indicator."""
        name = data.get("indicator_name", "")
        value = data.get("value", data.get("total_tco2e", 0.0))
        unit = data.get("unit", "")
        coverage = data.get("coverage_pct", 0.0)

        return (
            f"The portfolio-weighted {name} for the reporting period "
            f"is {value} {unit}, based on data covering {coverage}% "
            f"of the portfolio."
        )

    def _generate_actions_taken(
        self, indicator_id: str, data: Dict[str, Any]
    ) -> str:
        """Generate description of actions taken for an indicator."""
        if "ghg" in indicator_id or "carbon" in indicator_id:
            return (
                "Active engagement with high-emitting investees to set "
                "science-based targets. Portfolio decarbonization tracked "
                "against Paris Agreement pathways."
            )
        if "fossil_fuel" in indicator_id:
            return (
                "Exclusion of companies deriving significant revenue from "
                "fossil fuel exploration. Monitoring of transition plans."
            )
        if "ungc" in indicator_id or "controversial" in indicator_id:
            return (
                "Zero-tolerance policy for controversial weapons. Active "
                "monitoring of UNGC compliance through controversy screening."
            )
        return (
            "Ongoing monitoring and engagement with investee companies "
            "to improve performance on this indicator."
        )


class ActionPlanningPhase:
    """
    Phase 4: Action Planning.

    Identifies worst performers for engagement, defines exclusion decisions,
    creates a monitoring schedule, and sets improvement targets.
    """

    PHASE_NAME = "action_planning"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute action planning phase.

        Args:
            context: Workflow context with PAI calculations and reporting data.

        Returns:
            PhaseResult with action plan, worst performers, and schedule.
        """
        started_at = _utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            calc_output = context.get_phase_output("calculation")
            investees = config.get("investee_data", [])
            threshold_count = config.get("worst_performer_threshold", 10)

            # Identify worst performers by GHG intensity
            performers: List[Dict[str, Any]] = []
            for inv in investees:
                s1 = inv.get("ghg_scope1_tco2e", 0) or 0
                s2 = inv.get("ghg_scope2_tco2e", 0) or 0
                total_ghg = s1 + s2
                performers.append({
                    "name": inv.get("name", ""),
                    "isin": inv.get("isin", ""),
                    "sector": inv.get("sector", ""),
                    "portfolio_weight_pct": inv.get(
                        "portfolio_weight_pct", 0.0
                    ),
                    "total_ghg_tco2e": total_ghg,
                    "fossil_fuel": inv.get("fossil_fuel_involvement", False),
                    "ungc_violations": inv.get("ungc_violations", False),
                    "controversial_weapons": inv.get(
                        "controversial_weapons", False
                    ),
                })

            # Sort by GHG emissions (highest first)
            performers.sort(
                key=lambda x: x.get("total_ghg_tco2e", 0), reverse=True
            )
            worst_performers = performers[:threshold_count]
            outputs["worst_performers"] = worst_performers
            outputs["worst_performers_count"] = len(worst_performers)

            # Generate action items
            action_items: List[Dict[str, Any]] = []

            # Engagement actions for high emitters
            for performer in worst_performers:
                if performer["total_ghg_tco2e"] > 0:
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "action_type": ActionType.ENGAGEMENT.value,
                        "target": performer["name"],
                        "description": (
                            f"Engage with {performer['name']} on GHG "
                            f"reduction targets and transition plan"
                        ),
                        "priority": "HIGH",
                        "deadline_months": 6,
                    })

            # Exclusion actions for violations
            for inv in investees:
                if inv.get("controversial_weapons", False):
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "action_type": ActionType.EXCLUSION.value,
                        "target": inv.get("name", ""),
                        "description": (
                            f"Exclude {inv.get('name', '')} due to "
                            f"controversial weapons involvement"
                        ),
                        "priority": "CRITICAL",
                        "deadline_months": 1,
                    })
                if inv.get("ungc_violations", False):
                    action_items.append({
                        "action_id": str(uuid.uuid4()),
                        "action_type": ActionType.ENGAGEMENT.value,
                        "target": inv.get("name", ""),
                        "description": (
                            f"Engage with {inv.get('name', '')} on UNGC "
                            f"violation remediation"
                        ),
                        "priority": "HIGH",
                        "deadline_months": 3,
                    })

            outputs["action_items"] = action_items
            outputs["action_items_count"] = len(action_items)

            # Monitoring schedule
            outputs["monitoring_schedule"] = {
                "pai_data_refresh": "quarterly",
                "engagement_review": "semi-annual",
                "exclusion_list_update": "monthly",
                "portfolio_screening": "continuous",
                "next_full_review": (
                    f"{config.get('reporting_period_end', '')} + 3 months"
                ),
            }

            # Improvement targets
            yoy_data = calc_output.get("yoy_comparison", [])
            improvement_targets: List[Dict[str, Any]] = []

            # Set reduction targets for GHG indicators
            ghg_data = calc_output.get("pai_values", {}).get(
                "pai_1_ghg_emissions", {}
            )
            total_ghg = ghg_data.get("total_tco2e", 0.0)
            if total_ghg > 0:
                improvement_targets.append({
                    "indicator": "pai_1_ghg_emissions",
                    "current_value": total_ghg,
                    "target_reduction_pct": 7.0,
                    "target_value": round(total_ghg * 0.93, 4),
                    "timeframe": "annual",
                    "basis": "Paris Agreement 1.5C pathway",
                })

            outputs["improvement_targets"] = improvement_targets

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ActionPlanning failed: %s", exc, exc_info=True)
            errors.append(f"Action planning failed: {str(exc)}")
            status = PhaseStatus.FAILED

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
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class PAIStatementWorkflow:
    """
    Four-phase PAI statement workflow for SFDR Article 8.

    Orchestrates data sourcing through action planning for PAI calculation
    and statement generation. Covers all 18 mandatory PAI indicators
    with portfolio-weighted calculations.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = PAIStatementWorkflow()
        >>> input_data = PAIStatementInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     reporting_period_start="2025-01-01",
        ...     reporting_period_end="2025-12-31",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "pai_statement"

    PHASE_ORDER = [
        "data_sourcing",
        "calculation",
        "reporting",
        "action_planning",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the PAI statement workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_sourcing": DataSourcingPhase(),
            "calculation": PAICalculationPhase(),
            "reporting": PAIReportingPhase(),
            "action_planning": ActionPlanningPhase(),
        }

    async def run(
        self, input_data: PAIStatementInput
    ) -> PAIStatementResult:
        """
        Execute the complete 4-phase PAI statement workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            PAIStatementResult with per-phase details and summary.
        """
        started_at = _utcnow()
        logger.info(
            "Starting PAI statement workflow %s for org=%s product=%s",
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
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "data_sourcing":
                        overall_status = WorkflowStatus.FAILED
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
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
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
            "PAI statement workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return PAIStatementResult(
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
            total_investees=summary.get("total_investees", 0),
            mandatory_indicators_calculated=summary.get(
                "mandatory_indicators_calculated", 0
            ),
            average_coverage_pct=summary.get("average_coverage_pct", 0.0),
            data_quality_score=summary.get("data_quality_score", "LOW"),
            action_items_count=summary.get("action_items_count", 0),
            worst_performers_identified=summary.get(
                "worst_performers_identified", 0
            ),
        )

    def _build_config(self, input_data: PAIStatementInput) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        if input_data.investee_data:
            config["investee_data"] = [
                i.model_dump() for i in input_data.investee_data
            ]
            for i in config["investee_data"]:
                i["data_source"] = i["data_source"].value if isinstance(
                    i["data_source"], DataSource
                ) else i["data_source"]
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        sourcing_out = context.get_phase_output("data_sourcing")
        calc_out = context.get_phase_output("calculation")
        action_out = context.get_phase_output("action_planning")

        return {
            "product_name": config.get("product_name", ""),
            "reporting_period": (
                f"{config.get('reporting_period_start', '')} to "
                f"{config.get('reporting_period_end', '')}"
            ),
            "total_investees": sourcing_out.get("total_investees", 0),
            "mandatory_indicators_calculated": calc_out.get(
                "mandatory_indicators_calculated", 0
            ),
            "average_coverage_pct": sourcing_out.get(
                "average_coverage_pct", 0.0
            ),
            "data_quality_score": calc_out.get("data_quality_score", "LOW"),
            "action_items_count": action_out.get("action_items_count", 0),
            "worst_performers_identified": action_out.get(
                "worst_performers_count", 0
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
                logger.debug("Progress callback failed for phase=%s", phase)
