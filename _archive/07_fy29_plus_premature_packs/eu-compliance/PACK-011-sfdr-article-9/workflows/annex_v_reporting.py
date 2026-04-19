# -*- coding: utf-8 -*-
"""
Annex V Periodic Reporting Workflow
=======================================

Five-phase workflow for generating SFDR Annex V periodic disclosures for
Article 9 financial products. Orchestrates data collection, sustainable
objective attainment assessment, PAI calculation, template generation, and
filing package assembly into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 11: Periodic disclosure requirements for Article 9 products.
    - Annex V: Template for Article 9 periodic disclosures.
    - Must demonstrate that the sustainable investment objective was met
      during the reporting period.
    - 100% sustainable investment verification with actual portfolio data.
    - Comparison of actual sustainable objective attainment vs pre-contractual
      commitments is mandatory.
    - EU Climate Benchmark performance comparison required if designated.
    - All applicable PAI indicators must be reported with YoY data.
    - Top 15 investments by value must be disclosed.

Phases:
    1. DataCollection - Gather portfolio holdings, emissions/ESG data,
       benchmark performance at reporting date
    2. ObjectiveAttainmentAssessment - Measure actual sustainable objective
       attainment, verify 100% sustainable allocation, compare vs commitments
    3. PAICalculation - Calculate all mandatory PAI indicators with
       year-over-year comparison and coverage assessment
    4. TemplateGeneration - Generate Annex V template with actual figures,
       top 15 investments, benchmark comparison
    5. FilingPackage - Assemble filing-ready package with evidence and
       compliance certificate

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

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

class AttainmentStatus(str, Enum):
    """Sustainable objective attainment status."""
    MET = "MET"
    PARTIALLY_MET = "PARTIALLY_MET"
    NOT_MET = "NOT_MET"
    NOT_APPLICABLE = "NOT_APPLICABLE"

class DataQualityTier(str, Enum):
    """Data quality tier classification."""
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    PROXY = "PROXY"
    UNAVAILABLE = "UNAVAILABLE"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
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
# DATA MODELS - ANNEX V REPORTING
# =============================================================================

class PortfolioHolding(BaseModel):
    """A single holding in the portfolio at reporting date."""
    holding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer_name: str = Field(..., description="Issuer/investee name")
    isin: Optional[str] = Field(None, description="ISIN of security")
    sector: str = Field(default="", description="NACE sector")
    country: str = Field(default="", description="Country ISO code")
    market_value_eur: float = Field(default=0.0, ge=0.0)
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    esg_rating: Optional[float] = Field(None, ge=0.0, le=100.0)
    is_sustainable_investment: bool = Field(default=True)
    taxonomy_aligned_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    ghg_scope1_tco2e: Optional[float] = Field(None, ge=0.0)
    ghg_scope2_tco2e: Optional[float] = Field(None, ge=0.0)
    ghg_scope3_tco2e: Optional[float] = Field(None, ge=0.0)
    data_quality: DataQualityTier = Field(default=DataQualityTier.REPORTED)
    dnsh_compliant: bool = Field(default=True)
    good_governance_verified: bool = Field(default=True)

class PAIIndicatorData(BaseModel):
    """Principal Adverse Impact indicator input data."""
    indicator_id: str = Field(..., description="PAI indicator ID")
    indicator_name: str = Field(..., description="Indicator name")
    category: str = Field(default="climate")
    current_period_value: Optional[float] = Field(None)
    previous_period_value: Optional[float] = Field(None)
    unit: str = Field(default="")
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_quality: DataQualityTier = Field(default=DataQualityTier.REPORTED)

class AnnexVReportingInput(BaseModel):
    """Input configuration for the Annex V periodic reporting workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None)
    reporting_period_start: str = Field(
        ..., description="Period start YYYY-MM-DD"
    )
    reporting_period_end: str = Field(
        ..., description="Period end YYYY-MM-DD"
    )
    portfolio_holdings: List[PortfolioHolding] = Field(
        default_factory=list
    )
    pai_indicators: List[PAIIndicatorData] = Field(
        default_factory=list
    )
    total_portfolio_value_eur: float = Field(default=0.0, ge=0.0)
    sustainable_investment_commitment_pct: float = Field(
        default=100.0, ge=0.0, le=100.0
    )
    taxonomy_alignment_commitment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    benchmark_type: Optional[str] = Field(None)
    benchmark_name: Optional[str] = Field(None)
    benchmark_performance: Optional[Dict[str, Any]] = Field(None)
    previous_period_data: Optional[Dict[str, Any]] = Field(None)
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

class AnnexVReportingResult(WorkflowResult):
    """Complete result from the Annex V reporting workflow."""
    product_name: str = Field(default="")
    reporting_period: str = Field(default="")
    total_holdings: int = Field(default=0)
    portfolio_value_eur: float = Field(default=0.0)
    actual_sustainable_investment_pct: float = Field(default=0.0)
    actual_taxonomy_aligned_pct: float = Field(default=0.0)
    objective_met: bool = Field(default=False)
    dnsh_compliant_pct: float = Field(default=0.0)
    pai_indicators_reported: int = Field(default=0)
    average_pai_coverage_pct: float = Field(default=0.0)
    filing_package_id: str = Field(default="")

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class DataCollectionPhase:
    """
    Phase 1: Data Collection.

    Gathers portfolio holdings, emissions/ESG data, and benchmark
    performance at reporting date for Article 9 periodic disclosure.
    """

    PHASE_NAME = "data_collection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute data collection phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            holdings = config.get("portfolio_holdings", [])
            pai_indicators = config.get("pai_indicators", [])
            total_value = config.get("total_portfolio_value_eur", 0.0)

            outputs["total_holdings"] = len(holdings)
            outputs["total_portfolio_value_eur"] = total_value

            # Data quality assessment
            quality_counts = {t.value: 0 for t in DataQualityTier}
            for h in holdings:
                tier = h.get("data_quality", DataQualityTier.REPORTED.value)
                if tier in quality_counts:
                    quality_counts[tier] += 1

            outputs["data_quality_distribution"] = quality_counts
            total_h = max(len(holdings), 1)
            reported_pct = quality_counts.get("REPORTED", 0) / total_h * 100
            outputs["reported_data_pct"] = round(reported_pct, 1)

            if reported_pct < 70.0:
                warnings.append(
                    f"Only {reported_pct:.1f}% reported data. Article 9 "
                    f"products require high data quality."
                )

            # Sustainable investment classification
            sustainable_holdings = [
                h for h in holdings
                if h.get("is_sustainable_investment", True)
            ]
            non_sustainable_holdings = [
                h for h in holdings
                if not h.get("is_sustainable_investment", True)
            ]
            outputs["sustainable_holdings_count"] = len(sustainable_holdings)
            outputs["non_sustainable_holdings_count"] = len(
                non_sustainable_holdings
            )

            # DNSH compliance check
            dnsh_compliant = [
                h for h in holdings
                if h.get("dnsh_compliant", True)
            ]
            outputs["dnsh_compliant_count"] = len(dnsh_compliant)
            outputs["dnsh_compliant_pct"] = round(
                len(dnsh_compliant) / total_h * 100, 1
            )

            # Good governance check
            governance_verified = [
                h for h in holdings
                if h.get("good_governance_verified", True)
            ]
            outputs["governance_verified_count"] = len(governance_verified)

            # Sector breakdown
            sector_allocation: Dict[str, float] = {}
            for h in holdings:
                sector = h.get("sector", "Other")
                weight = h.get("portfolio_weight_pct", 0.0)
                sector_allocation[sector] = (
                    sector_allocation.get(sector, 0.0) + weight
                )
            outputs["sector_allocation"] = sector_allocation

            # Geographic breakdown
            geo_allocation: Dict[str, float] = {}
            for h in holdings:
                country = h.get("country", "Other")
                weight = h.get("portfolio_weight_pct", 0.0)
                geo_allocation[country] = (
                    geo_allocation.get(country, 0.0) + weight
                )
            outputs["geographic_allocation"] = geo_allocation

            # Top 15 holdings
            sorted_holdings = sorted(
                holdings,
                key=lambda x: x.get("portfolio_weight_pct", 0.0),
                reverse=True,
            )
            top_15 = sorted_holdings[:15]
            outputs["top_15_investments"] = [
                {
                    "issuer_name": h.get("issuer_name", ""),
                    "isin": h.get("isin", ""),
                    "sector": h.get("sector", ""),
                    "country": h.get("country", ""),
                    "portfolio_weight_pct": h.get(
                        "portfolio_weight_pct", 0.0
                    ),
                    "market_value_eur": h.get("market_value_eur", 0.0),
                    "is_sustainable": h.get(
                        "is_sustainable_investment", True
                    ),
                }
                for h in top_15
            ]

            # PAI data completeness
            outputs["pai_indicators_count"] = len(pai_indicators)
            pai_with_data = sum(
                1 for p in pai_indicators
                if p.get("current_period_value") is not None
            )
            outputs["pai_indicators_with_data"] = pai_with_data

            # Emissions data
            total_scope1 = 0.0
            total_scope2 = 0.0
            total_scope3 = 0.0
            emissions_coverage = 0

            for h in holdings:
                weight = h.get("portfolio_weight_pct", 0.0) / 100.0
                s1 = h.get("ghg_scope1_tco2e")
                if s1 is not None:
                    total_scope1 += s1 * weight
                    emissions_coverage += 1
                s2 = h.get("ghg_scope2_tco2e")
                if s2 is not None:
                    total_scope2 += s2 * weight
                s3 = h.get("ghg_scope3_tco2e")
                if s3 is not None:
                    total_scope3 += s3 * weight

            outputs["emissions_data"] = {
                "scope1_tco2e": round(total_scope1, 4),
                "scope2_tco2e": round(total_scope2, 4),
                "scope3_tco2e": round(total_scope3, 4),
                "total_tco2e": round(
                    total_scope1 + total_scope2 + total_scope3, 4
                ),
                "coverage_count": emissions_coverage,
                "coverage_pct": round(
                    emissions_coverage / total_h * 100, 1
                ),
            }

            status = PhaseStatus.COMPLETED
            records = len(holdings)

        except Exception as exc:
            logger.error("DataCollection failed: %s", exc, exc_info=True)
            errors.append(f"Data collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

class ObjectiveAttainmentAssessmentPhase:
    """
    Phase 2: Objective Attainment Assessment.

    Measures actual sustainable objective attainment, verifies 100%
    sustainable allocation, and compares actuals vs commitments.
    """

    PHASE_NAME = "objective_attainment_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute objective attainment assessment phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            data_output = context.get_phase_output("data_collection")
            holdings = config.get("portfolio_holdings", [])
            si_commitment = config.get(
                "sustainable_investment_commitment_pct", 100.0
            )
            tax_commitment = config.get(
                "taxonomy_alignment_commitment_pct", 0.0
            )

            # Actual sustainable investment proportion
            si_holdings = [
                h for h in holdings
                if h.get("is_sustainable_investment", True)
            ]
            si_weight = sum(
                h.get("portfolio_weight_pct", 0.0) for h in si_holdings
            )
            outputs["actual_sustainable_investment_pct"] = round(
                si_weight, 2
            )

            # Actual taxonomy alignment
            taxonomy_weight = sum(
                h.get("portfolio_weight_pct", 0.0)
                * h.get("taxonomy_aligned_pct", 0.0) / 100.0
                for h in holdings
            )
            outputs["actual_taxonomy_aligned_pct"] = round(
                taxonomy_weight, 2
            )

            # DNSH compliance verification
            dnsh_pct = data_output.get("dnsh_compliant_pct", 0.0)
            outputs["dnsh_compliant_pct"] = dnsh_pct

            dnsh_non_compliant = [
                h for h in holdings
                if not h.get("dnsh_compliant", True)
                and h.get("is_sustainable_investment", True)
            ]
            if dnsh_non_compliant:
                warnings.append(
                    f"{len(dnsh_non_compliant)} sustainable holding(s) "
                    f"failed DNSH assessment"
                )

            # Objective attainment determination
            objective_met = (
                si_weight >= si_commitment * 0.99
                and taxonomy_weight >= tax_commitment * 0.99
                and dnsh_pct >= 95.0
            )
            outputs["objective_met"] = objective_met

            if objective_met:
                outputs["objective_attainment"] = AttainmentStatus.MET.value
            elif si_weight >= si_commitment * 0.90:
                outputs["objective_attainment"] = (
                    AttainmentStatus.PARTIALLY_MET.value
                )
            else:
                outputs["objective_attainment"] = (
                    AttainmentStatus.NOT_MET.value
                )

            # Commitment comparison
            outputs["comparison_table"] = {
                "sustainable_investment": {
                    "committed_pct": si_commitment,
                    "actual_pct": round(si_weight, 2),
                    "variance_pct": round(si_weight - si_commitment, 2),
                    "met": si_weight >= si_commitment * 0.99,
                },
                "taxonomy_alignment": {
                    "committed_pct": tax_commitment,
                    "actual_pct": round(taxonomy_weight, 2),
                    "variance_pct": round(
                        taxonomy_weight - tax_commitment, 2
                    ),
                    "met": taxonomy_weight >= tax_commitment * 0.99,
                },
                "dnsh_compliance": {
                    "required_pct": 100.0,
                    "actual_pct": dnsh_pct,
                    "met": dnsh_pct >= 95.0,
                },
            }

            if si_weight < si_commitment:
                warnings.append(
                    f"Sustainable investment ({si_weight:.1f}%) below "
                    f"commitment ({si_commitment:.1f}%)"
                )
            if taxonomy_weight < tax_commitment:
                warnings.append(
                    f"Taxonomy alignment ({taxonomy_weight:.1f}%) below "
                    f"commitment ({tax_commitment:.1f}%)"
                )

            # Benchmark comparison
            benchmark_perf = config.get("benchmark_performance")
            if benchmark_perf:
                outputs["benchmark_comparison"] = {
                    "applicable": True,
                    "benchmark_name": config.get("benchmark_name", ""),
                    "benchmark_type": config.get("benchmark_type", ""),
                    "performance_data": benchmark_perf,
                }
            else:
                outputs["benchmark_comparison"] = {
                    "applicable": False,
                    "description": "No benchmark designated.",
                }

            # Average ESG rating
            rated = [
                h for h in holdings if h.get("esg_rating") is not None
            ]
            if rated:
                avg_esg = sum(
                    h.get("esg_rating", 0) for h in rated
                ) / len(rated)
                outputs["average_esg_rating"] = round(avg_esg, 2)
            else:
                outputs["average_esg_rating"] = None

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "ObjectiveAttainment failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Objective attainment assessment failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED

        completed_at = utcnow()
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

class PAICalculationPhase:
    """
    Phase 3: PAI Calculation.

    Calculates all mandatory PAI indicators with year-over-year
    comparison and coverage assessment.
    """

    PHASE_NAME = "pai_calculation"

    MANDATORY_PAI_INDICATORS = [
        "pai_1_ghg_emissions",
        "pai_2_carbon_footprint",
        "pai_3_ghg_intensity",
        "pai_4_fossil_fuel_exposure",
        "pai_5_non_renewable_energy",
        "pai_6_energy_intensity",
        "pai_7_biodiversity",
        "pai_8_water_emissions",
        "pai_9_hazardous_waste",
        "pai_10_ungc_violations",
        "pai_11_ungc_compliance_gap",
        "pai_12_gender_pay_gap",
        "pai_13_board_gender_diversity",
        "pai_14_controversial_weapons",
        "pai_15_ghg_intensity_sovereigns",
        "pai_16_investee_countries_social",
        "pai_17_real_estate_fossil_fuel",
        "pai_18_real_estate_energy_inefficient",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute PAI calculation phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            pai_inputs = config.get("pai_indicators", [])

            pai_results: List[Dict[str, Any]] = []
            total_coverage = 0.0
            indicators_with_data = 0

            for indicator in pai_inputs:
                indicator_id = indicator.get("indicator_id", "")
                current_val = indicator.get("current_period_value")
                previous_val = indicator.get("previous_period_value")
                coverage = indicator.get("coverage_pct", 0.0)

                yoy_change = None
                yoy_change_pct = None
                if current_val is not None and previous_val is not None:
                    yoy_change = round(current_val - previous_val, 4)
                    if previous_val != 0:
                        yoy_change_pct = round(
                            (current_val - previous_val)
                            / abs(previous_val) * 100, 2
                        )

                pai_results.append({
                    "indicator_id": indicator_id,
                    "indicator_name": indicator.get(
                        "indicator_name", ""
                    ),
                    "category": indicator.get("category", "climate"),
                    "current_period_value": current_val,
                    "previous_period_value": previous_val,
                    "yoy_change": yoy_change,
                    "yoy_change_pct": yoy_change_pct,
                    "unit": indicator.get("unit", ""),
                    "coverage_pct": coverage,
                    "data_quality": indicator.get(
                        "data_quality", "REPORTED"
                    ),
                    "has_data": current_val is not None,
                })

                if current_val is not None:
                    indicators_with_data += 1
                    total_coverage += coverage

            outputs["pai_results"] = pai_results
            outputs["pai_indicators_reported"] = indicators_with_data
            outputs["pai_indicators_total"] = len(pai_inputs)

            avg_coverage = (
                total_coverage / max(indicators_with_data, 1)
            )
            outputs["average_coverage_pct"] = round(avg_coverage, 1)

            # Check mandatory indicators
            reported_ids = {
                p.get("indicator_id") for p in pai_inputs
                if p.get("current_period_value") is not None
            }
            missing_mandatory = [
                ind for ind in self.MANDATORY_PAI_INDICATORS
                if ind not in reported_ids
            ]
            outputs["missing_mandatory_indicators"] = missing_mandatory

            if missing_mandatory:
                warnings.append(
                    f"{len(missing_mandatory)} mandatory PAI indicator(s) "
                    f"missing: {', '.join(missing_mandatory[:5])}"
                    + ("..." if len(missing_mandatory) > 5 else "")
                )

            # Data quality score
            if indicators_with_data > 0 and avg_coverage >= 80.0:
                outputs["data_quality_score"] = "HIGH"
            elif indicators_with_data > 0 and avg_coverage >= 50.0:
                outputs["data_quality_score"] = "MEDIUM"
            else:
                outputs["data_quality_score"] = "LOW"

            status = PhaseStatus.COMPLETED
            records = len(pai_inputs)

        except Exception as exc:
            logger.error("PAICalculation failed: %s", exc, exc_info=True)
            errors.append(f"PAI calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
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

class TemplateGenerationPhase:
    """
    Phase 4: Template Generation.

    Generates the Annex V template with actual figures, top 15
    investments, benchmark comparison, and DNSH compliance.
    """

    PHASE_NAME = "template_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute template generation phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            data_output = context.get_phase_output("data_collection")
            attainment_output = context.get_phase_output(
                "objective_attainment_assessment"
            )
            pai_output = context.get_phase_output("pai_calculation")

            product_name = config.get("product_name", "")
            period_start = config.get("reporting_period_start", "")
            period_end = config.get("reporting_period_end", "")

            template = {
                "header": {
                    "product_name": product_name,
                    "product_isin": config.get("product_isin", ""),
                    "reporting_period": (
                        f"{period_start} to {period_end}"
                    ),
                    "classification": "ARTICLE_9",
                    "generated_at": utcnow().isoformat(),
                },
                "sustainable_objective_attainment": {
                    "description": (
                        f"This section details how {product_name} met "
                        f"its sustainable investment objective during "
                        f"the reporting period."
                    ),
                    "objective_met": attainment_output.get(
                        "objective_met", False
                    ),
                    "attainment_status": attainment_output.get(
                        "objective_attainment", "NOT_MET"
                    ),
                    "comparison_table": attainment_output.get(
                        "comparison_table", {}
                    ),
                },
                "top_investments": {
                    "top_15": data_output.get(
                        "top_15_investments", []
                    ),
                    "total_holdings": data_output.get(
                        "total_holdings", 0
                    ),
                    "total_value_eur": data_output.get(
                        "total_portfolio_value_eur", 0.0
                    ),
                },
                "proportion_of_investments": {
                    "sustainable_investment_actual_pct": attainment_output.get(
                        "actual_sustainable_investment_pct", 0.0
                    ),
                    "taxonomy_aligned_actual_pct": attainment_output.get(
                        "actual_taxonomy_aligned_pct", 0.0
                    ),
                    "dnsh_compliant_pct": attainment_output.get(
                        "dnsh_compliant_pct", 0.0
                    ),
                },
                "sector_allocation": data_output.get(
                    "sector_allocation", {}
                ),
                "geographic_allocation": data_output.get(
                    "geographic_allocation", {}
                ),
                "pai_statement": {
                    "indicators": pai_output.get("pai_results", []),
                    "indicators_reported": pai_output.get(
                        "pai_indicators_reported", 0
                    ),
                    "average_coverage_pct": pai_output.get(
                        "average_coverage_pct", 0.0
                    ),
                    "data_quality_score": pai_output.get(
                        "data_quality_score", "LOW"
                    ),
                },
                "emissions_summary": data_output.get(
                    "emissions_data", {}
                ),
                "benchmark_comparison": attainment_output.get(
                    "benchmark_comparison", {}
                ),
                "data_quality": {
                    "distribution": data_output.get(
                        "data_quality_distribution", {}
                    ),
                    "reported_data_pct": data_output.get(
                        "reported_data_pct", 0.0
                    ),
                },
            }

            outputs["annex_v_template"] = template
            outputs["template_version"] = "1.0"
            outputs["template_format"] = "structured_json"
            outputs["generated_at"] = utcnow().isoformat()

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "TemplateGeneration failed: %s", exc, exc_info=True
            )
            errors.append(f"Template generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
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

class FilingPackagePhase:
    """
    Phase 5: Filing Package.

    Assembles filing-ready package with Annex V template, evidence,
    and compliance certificate.
    """

    PHASE_NAME = "filing_package"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """Execute filing package assembly phase."""
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            template_output = context.get_phase_output(
                "template_generation"
            )
            attainment_output = context.get_phase_output(
                "objective_attainment_assessment"
            )
            pai_output = context.get_phase_output("pai_calculation")
            data_output = context.get_phase_output("data_collection")

            package_id = str(uuid.uuid4())
            outputs["filing_package_id"] = package_id

            outputs["package_manifest"] = {
                "package_id": package_id,
                "created_at": utcnow().isoformat(),
                "product_name": config.get("product_name", ""),
                "reporting_period": (
                    f"{config.get('reporting_period_start', '')} to "
                    f"{config.get('reporting_period_end', '')}"
                ),
                "documents": [
                    {
                        "document_type": "annex_v_template",
                        "format": "structured_json",
                        "status": "complete",
                    },
                    {
                        "document_type": "pai_statement",
                        "format": "structured_json",
                        "status": "complete",
                    },
                    {
                        "document_type": "dnsh_verification",
                        "format": "structured_json",
                        "status": "complete",
                    },
                    {
                        "document_type": "holdings_snapshot",
                        "format": "structured_json",
                        "status": "complete",
                        "record_count": data_output.get(
                            "total_holdings", 0
                        ),
                    },
                    {
                        "document_type": "compliance_certificate",
                        "format": "structured_json",
                        "status": "complete",
                    },
                ],
            }

            # Compliance certificate
            objective_met = attainment_output.get(
                "objective_met", False
            )
            pai_reported = pai_output.get(
                "pai_indicators_reported", 0
            )
            missing_mandatory = pai_output.get(
                "missing_mandatory_indicators", []
            )
            dnsh_pct = attainment_output.get("dnsh_compliant_pct", 0.0)

            is_compliant = (
                objective_met
                and len(missing_mandatory) == 0
                and data_output.get("total_holdings", 0) > 0
                and dnsh_pct >= 95.0
            )

            compliance_certificate = {
                "certificate_id": str(uuid.uuid4()),
                "issued_at": utcnow().isoformat(),
                "product_name": config.get("product_name", ""),
                "is_compliant": is_compliant,
                "checks": {
                    "objective_met": objective_met,
                    "dnsh_compliance_pct": dnsh_pct,
                    "mandatory_pai_complete": len(
                        missing_mandatory
                    ) == 0,
                    "missing_pai_indicators": missing_mandatory,
                    "pai_indicators_reported": pai_reported,
                    "holdings_data_present": (
                        data_output.get("total_holdings", 0) > 0
                    ),
                    "reported_data_pct": data_output.get(
                        "reported_data_pct", 0.0
                    ),
                },
                "recommendations": [],
            }

            if not is_compliant:
                if not objective_met:
                    compliance_certificate["recommendations"].append(
                        "Sustainable investment objective not fully met. "
                        "Review portfolio alignment."
                    )
                if missing_mandatory:
                    compliance_certificate["recommendations"].append(
                        f"{len(missing_mandatory)} mandatory PAI "
                        f"indicator(s) lack data."
                    )
                if dnsh_pct < 95.0:
                    compliance_certificate["recommendations"].append(
                        f"DNSH compliance at {dnsh_pct:.1f}%, below "
                        f"95% threshold."
                    )

            outputs["compliance_certificate"] = compliance_certificate

            outputs["evidence_index"] = {
                "portfolio_snapshot_hash": _hash_data(
                    data_output.get("top_15_investments", [])
                ),
                "pai_data_hash": _hash_data(
                    pai_output.get("pai_results", [])
                ),
                "attainment_data_hash": _hash_data(
                    attainment_output.get("comparison_table", {})
                ),
                "template_hash": _hash_data(
                    template_output.get("annex_v_template", {})
                ),
            }

            outputs["filing_status"] = (
                "ready_for_filing" if is_compliant
                else "requires_review"
            )

            if not is_compliant:
                warnings.append(
                    "Filing package has compliance issues. Review "
                    "compliance certificate."
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "FilingPackage failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Filing package assembly failed: {str(exc)}"
            )
            status = PhaseStatus.FAILED

        completed_at = utcnow()
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

class AnnexVReportingWorkflow:
    """
    Five-phase Annex V periodic reporting workflow for Article 9.

    Orchestrates data collection through filing package assembly for
    Annex V periodic disclosures. Supports checkpoint/resume and
    phase skipping.

    Example:
        >>> wf = AnnexVReportingWorkflow()
        >>> input_data = AnnexVReportingInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     reporting_period_start="2025-01-01",
        ...     reporting_period_end="2025-12-31",
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "annex_v_reporting"

    PHASE_ORDER = [
        "data_collection",
        "objective_attainment_assessment",
        "pai_calculation",
        "template_generation",
        "filing_package",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """Initialize the Annex V reporting workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "data_collection": DataCollectionPhase(),
            "objective_attainment_assessment": ObjectiveAttainmentAssessmentPhase(),
            "pai_calculation": PAICalculationPhase(),
            "template_generation": TemplateGenerationPhase(),
            "filing_package": FilingPackagePhase(),
        }

    async def run(
        self, input_data: AnnexVReportingInput
    ) -> AnnexVReportingResult:
        """Execute the complete 5-phase Annex V reporting workflow."""
        started_at = utcnow()
        logger.info(
            "Starting Annex V reporting workflow %s for org=%s product=%s",
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
                    if phase_name == "data_collection":
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
                    started_at=utcnow(),
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

        completed_at = utcnow()
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
            "Annex V reporting workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return AnnexVReportingResult(
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
            total_holdings=summary.get("total_holdings", 0),
            portfolio_value_eur=summary.get(
                "portfolio_value_eur", 0.0
            ),
            actual_sustainable_investment_pct=summary.get(
                "actual_sustainable_investment_pct", 0.0
            ),
            actual_taxonomy_aligned_pct=summary.get(
                "actual_taxonomy_aligned_pct", 0.0
            ),
            objective_met=summary.get("objective_met", False),
            dnsh_compliant_pct=summary.get(
                "dnsh_compliant_pct", 0.0
            ),
            pai_indicators_reported=summary.get(
                "pai_indicators_reported", 0
            ),
            average_pai_coverage_pct=summary.get(
                "average_pai_coverage_pct", 0.0
            ),
            filing_package_id=summary.get("filing_package_id", ""),
        )

    def _build_config(
        self, input_data: AnnexVReportingInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        if input_data.portfolio_holdings:
            config["portfolio_holdings"] = [
                h.model_dump() for h in input_data.portfolio_holdings
            ]
            for h in config["portfolio_holdings"]:
                h["data_quality"] = (
                    h["data_quality"].value
                    if isinstance(h["data_quality"], DataQualityTier)
                    else h["data_quality"]
                )
        if input_data.pai_indicators:
            config["pai_indicators"] = [
                p.model_dump() for p in input_data.pai_indicators
            ]
            for p in config["pai_indicators"]:
                p["data_quality"] = (
                    p["data_quality"].value
                    if isinstance(p["data_quality"], DataQualityTier)
                    else p["data_quality"]
                )
        return config

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        config = context.config
        data_out = context.get_phase_output("data_collection")
        attainment_out = context.get_phase_output(
            "objective_attainment_assessment"
        )
        pai_out = context.get_phase_output("pai_calculation")
        filing_out = context.get_phase_output("filing_package")

        return {
            "product_name": config.get("product_name", ""),
            "reporting_period": (
                f"{config.get('reporting_period_start', '')} to "
                f"{config.get('reporting_period_end', '')}"
            ),
            "total_holdings": data_out.get("total_holdings", 0),
            "portfolio_value_eur": data_out.get(
                "total_portfolio_value_eur", 0.0
            ),
            "actual_sustainable_investment_pct": attainment_out.get(
                "actual_sustainable_investment_pct", 0.0
            ),
            "actual_taxonomy_aligned_pct": attainment_out.get(
                "actual_taxonomy_aligned_pct", 0.0
            ),
            "objective_met": attainment_out.get(
                "objective_met", False
            ),
            "dnsh_compliant_pct": attainment_out.get(
                "dnsh_compliant_pct", 0.0
            ),
            "pai_indicators_reported": pai_out.get(
                "pai_indicators_reported", 0
            ),
            "average_pai_coverage_pct": pai_out.get(
                "average_coverage_pct", 0.0
            ),
            "filing_package_id": filing_out.get(
                "filing_package_id", ""
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
