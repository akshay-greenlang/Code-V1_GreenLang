# -*- coding: utf-8 -*-
"""
BenchmarkAlignmentEngine - PACK-011 SFDR Article 9 Engine 5
=============================================================

EU Climate Benchmark alignment verification for SFDR Article 9(3) products.
Article 9(3) products that designate a reference benchmark must align with
either a Climate Transition Benchmark (CTB) or a Paris-Aligned Benchmark (PAB)
as defined in EU Regulation 2019/2089 (Low Carbon Benchmarks Regulation).

Key Requirements:
    CTB (Climate Transition Benchmark):
        - At least 30% lower carbon intensity vs investable universe
        - Exclusion of controversial weapons manufacturers
        - 7% year-on-year decarbonization trajectory

    PAB (Paris-Aligned Benchmark):
        - At least 50% lower carbon intensity vs investable universe
        - Fossil fuel exclusions:
            * >= 1% revenue from coal/oil/gas exploration
            * >= 10% revenue from refining/processing
            * >= 50% revenue from distribution/transportation
            * > 100g CO2/kWh power generation
        - Controversial weapons exclusion
        - 7% year-on-year decarbonization trajectory

Formulas:
    Intensity Reduction = 1 - (portfolio_intensity / benchmark_intensity)
    Decarbonization Rate = 1 - (current_year_intensity / prior_year_intensity)
    Projected Intensity(year) = base_intensity * (1 - 0.07) ^ (year - base_year)
    Tracking Error = sqrt(sum((r_p - r_b)^2) / (n - 1))

Zero-Hallucination:
    - All calculations use deterministic Python arithmetic
    - Exclusion checks are boolean rule-based evaluations
    - Decarbonization trajectories use fixed 7% compounding
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage or 0.0 on zero denominator.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BenchmarkType(str, Enum):
    """EU Climate Benchmark classification."""
    CTB = "ctb"   # Climate Transition Benchmark
    PAB = "pab"   # Paris-Aligned Benchmark
    CUSTOM = "custom"  # Custom climate benchmark

class ComplianceStatus(str, Enum):
    """Benchmark compliance assessment status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"

class ExclusionCategory(str, Enum):
    """Categories of benchmark exclusions."""
    CONTROVERSIAL_WEAPONS = "controversial_weapons"
    COAL_EXPLORATION = "coal_exploration"
    OIL_GAS_EXPLORATION = "oil_gas_exploration"
    FOSSIL_REFINING = "fossil_refining"
    FOSSIL_DISTRIBUTION = "fossil_distribution"
    HIGH_CARBON_POWER = "high_carbon_power"
    TOBACCO = "tobacco"
    UNGC_VIOLATIONS = "ungc_violations"

# ---------------------------------------------------------------------------
# PAB / CTB Exclusion Thresholds
# ---------------------------------------------------------------------------

PAB_EXCLUSION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    ExclusionCategory.CONTROVERSIAL_WEAPONS.value: {
        "field": "controversial_weapons",
        "threshold": 0.0,
        "operator": "bool_true",
        "description": "Involved in controversial weapons (anti-personnel mines, "
                       "cluster munitions, chemical/biological weapons)",
    },
    ExclusionCategory.COAL_EXPLORATION.value: {
        "field": "coal_exploration_revenue_pct",
        "threshold": 1.0,
        "operator": "gte",
        "description": ">=1% revenue from coal, oil, or gas exploration",
    },
    ExclusionCategory.OIL_GAS_EXPLORATION.value: {
        "field": "oil_gas_exploration_revenue_pct",
        "threshold": 1.0,
        "operator": "gte",
        "description": ">=1% revenue from oil/gas exploration",
    },
    ExclusionCategory.FOSSIL_REFINING.value: {
        "field": "fossil_refining_revenue_pct",
        "threshold": 10.0,
        "operator": "gte",
        "description": ">=10% revenue from refining/processing fossil fuels",
    },
    ExclusionCategory.FOSSIL_DISTRIBUTION.value: {
        "field": "fossil_distribution_revenue_pct",
        "threshold": 50.0,
        "operator": "gte",
        "description": ">=50% revenue from distribution/transportation of fossil fuels",
    },
    ExclusionCategory.HIGH_CARBON_POWER.value: {
        "field": "power_generation_carbon_intensity",
        "threshold": 100.0,
        "operator": "gt",
        "description": ">100g CO2/kWh power generation",
    },
}

CTB_EXCLUSION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    ExclusionCategory.CONTROVERSIAL_WEAPONS.value: {
        "field": "controversial_weapons",
        "threshold": 0.0,
        "operator": "bool_true",
        "description": "Involved in controversial weapons",
    },
}

# Annual decarbonization rate per EU Low Carbon Benchmarks Regulation
ANNUAL_DECARBONIZATION_RATE: float = 0.07  # 7% per year

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class HoldingBenchmarkData(BaseModel):
    """Benchmark-relevant data for a single portfolio holding.

    Contains carbon intensity, exclusion screening data, and financial
    details needed for benchmark alignment assessment.
    """
    holding_id: str = Field(default_factory=_new_uuid, description="Unique holding identifier")
    company_name: str = Field(default="", description="Investee company name")
    isin: str = Field(default="", description="Security ISIN")
    sector: str = Field(default="", description="NACE/GICS sector code")
    country: str = Field(default="", description="Country of domicile (ISO 3166)")
    # Carbon intensity
    carbon_intensity: float = Field(
        default=0.0, description="Carbon intensity (tCO2e/EUR M revenue)"
    )
    scope1: float = Field(default=0.0, description="Scope 1 emissions (tCO2e)")
    scope2: float = Field(default=0.0, description="Scope 2 emissions (tCO2e)")
    scope3: float = Field(default=0.0, description="Scope 3 emissions (tCO2e)")
    revenue_eur: float = Field(default=0.0, description="Annual revenue (EUR)")
    # Portfolio position
    holding_value: float = Field(default=0.0, description="Portfolio holding value (EUR)")
    weight_pct: float = Field(default=0.0, description="Portfolio weight (%)")
    # Exclusion screening fields
    controversial_weapons: bool = Field(
        default=False, description="Involved in controversial weapons"
    )
    coal_exploration_revenue_pct: float = Field(
        default=0.0, description="% revenue from coal/oil/gas exploration"
    )
    oil_gas_exploration_revenue_pct: float = Field(
        default=0.0, description="% revenue from oil/gas exploration"
    )
    fossil_refining_revenue_pct: float = Field(
        default=0.0, description="% revenue from fossil fuel refining/processing"
    )
    fossil_distribution_revenue_pct: float = Field(
        default=0.0, description="% revenue from fossil fuel distribution"
    )
    power_generation_carbon_intensity: float = Field(
        default=0.0, description="Power generation carbon intensity (gCO2/kWh)"
    )
    tobacco_revenue_pct: float = Field(
        default=0.0, description="% revenue from tobacco"
    )
    has_ungc_violations: bool = Field(
        default=False, description="Has UNGC/OECD principle violations"
    )
    # Historical intensity for trajectory
    prior_year_carbon_intensity: float = Field(
        default=0.0, description="Prior year carbon intensity (tCO2e/EUR M revenue)"
    )
    reporting_year: int = Field(default=2025, description="Year of data")

    @model_validator(mode="after")
    def _compute_intensity(self) -> "HoldingBenchmarkData":
        """Auto-compute carbon intensity if not provided."""
        if self.carbon_intensity <= 0.0 and self.revenue_eur > 0.0:
            total_emissions = self.scope1 + self.scope2 + self.scope3
            revenue_m = self.revenue_eur / 1_000_000.0
            if revenue_m > 0.0:
                self.carbon_intensity = total_emissions / revenue_m
        return self

class ExclusionViolation(BaseModel):
    """Details of a single exclusion rule violation."""
    violation_id: str = Field(default_factory=_new_uuid, description="Unique violation identifier")
    holding_id: str = Field(description="ID of the violating holding")
    company_name: str = Field(default="", description="Company name")
    category: ExclusionCategory = Field(description="Exclusion category violated")
    rule_description: str = Field(default="", description="Description of the exclusion rule")
    actual_value: float = Field(default=0.0, description="Actual value triggering violation")
    threshold_value: float = Field(default=0.0, description="Threshold that was breached")
    weight_pct: float = Field(default=0.0, description="Portfolio weight of violating holding (%)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class TrajectoryDataPoint(BaseModel):
    """A single data point on the decarbonization trajectory."""
    year: int = Field(description="Calendar year")
    target_intensity: float = Field(description="Target carbon intensity for this year")
    actual_intensity: Optional[float] = Field(
        default=None, description="Actual carbon intensity (if available)"
    )
    on_track: bool = Field(default=True, description="Whether actual is at or below target")
    gap_pct: float = Field(
        default=0.0, description="Gap between actual and target as percentage"
    )

class TrackingErrorResult(BaseModel):
    """Tracking error calculation between portfolio and benchmark."""
    tracking_error_id: str = Field(
        default_factory=_new_uuid, description="Unique identifier"
    )
    tracking_error_annualized: float = Field(
        default=0.0, description="Annualized tracking error"
    )
    tracking_error_monthly: float = Field(
        default=0.0, description="Monthly tracking error"
    )
    observation_count: int = Field(default=0, description="Number of return observations")
    mean_excess_return: float = Field(
        default=0.0, description="Mean excess return vs benchmark"
    )
    information_ratio: float = Field(
        default=0.0, description="Information ratio (excess return / tracking error)"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class MethodologyDisclosure(BaseModel):
    """Methodology disclosure details for benchmark alignment."""
    benchmark_type: BenchmarkType = Field(description="Type of benchmark used")
    benchmark_name: str = Field(default="", description="Name of the benchmark index")
    benchmark_provider: str = Field(default="", description="Benchmark provider/administrator")
    intensity_reduction_target_pct: float = Field(
        default=0.0, description="Required intensity reduction (%)"
    )
    decarbonization_rate_pct: float = Field(
        default=7.0, description="Annual decarbonization rate (%)"
    )
    base_year: int = Field(default=2019, description="Decarbonization base year")
    exclusions_applied: List[str] = Field(
        default_factory=list, description="List of exclusion categories applied"
    )
    scope_coverage: str = Field(
        default="scope_1_2", description="Emission scope coverage"
    )
    methodology_notes: str = Field(
        default="", description="Additional methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class CTBComplianceResult(BaseModel):
    """Climate Transition Benchmark compliance assessment result."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    status: ComplianceStatus = Field(description="Overall CTB compliance status")
    intensity_reduction_pct: float = Field(
        default=0.0, description="Actual intensity reduction vs universe (%)"
    )
    intensity_reduction_required_pct: float = Field(
        default=30.0, description="Required intensity reduction (30% for CTB)"
    )
    intensity_reduction_met: bool = Field(
        default=False, description="Whether 30% reduction requirement is met"
    )
    portfolio_intensity: float = Field(
        default=0.0, description="Portfolio carbon intensity (tCO2e/EUR M)"
    )
    benchmark_intensity: float = Field(
        default=0.0, description="Benchmark/universe carbon intensity (tCO2e/EUR M)"
    )
    controversial_weapons_exclusion_met: bool = Field(
        default=False, description="Whether controversial weapons exclusion is satisfied"
    )
    exclusion_violations: List[ExclusionViolation] = Field(
        default_factory=list, description="List of exclusion violations"
    )
    decarbonization_rate_pct: float = Field(
        default=0.0, description="Actual year-over-year decarbonization rate (%)"
    )
    decarbonization_target_met: bool = Field(
        default=False, description="Whether 7% annual decarbonization target is met"
    )
    holdings_screened: int = Field(default=0, description="Number of holdings screened")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class PABComplianceResult(BaseModel):
    """Paris-Aligned Benchmark compliance assessment result."""
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    status: ComplianceStatus = Field(description="Overall PAB compliance status")
    intensity_reduction_pct: float = Field(
        default=0.0, description="Actual intensity reduction vs universe (%)"
    )
    intensity_reduction_required_pct: float = Field(
        default=50.0, description="Required intensity reduction (50% for PAB)"
    )
    intensity_reduction_met: bool = Field(
        default=False, description="Whether 50% reduction requirement is met"
    )
    portfolio_intensity: float = Field(
        default=0.0, description="Portfolio carbon intensity (tCO2e/EUR M)"
    )
    benchmark_intensity: float = Field(
        default=0.0, description="Benchmark/universe carbon intensity (tCO2e/EUR M)"
    )
    fossil_fuel_exclusions_met: bool = Field(
        default=False, description="Whether all fossil fuel exclusions are satisfied"
    )
    controversial_weapons_exclusion_met: bool = Field(
        default=False, description="Whether controversial weapons exclusion is satisfied"
    )
    exclusion_violations: List[ExclusionViolation] = Field(
        default_factory=list, description="List of all exclusion violations"
    )
    fossil_exploration_violations: int = Field(
        default=0, description="Count of fossil exploration violations (>=1%)"
    )
    fossil_refining_violations: int = Field(
        default=0, description="Count of fossil refining violations (>=10%)"
    )
    fossil_distribution_violations: int = Field(
        default=0, description="Count of fossil distribution violations (>=50%)"
    )
    high_carbon_power_violations: int = Field(
        default=0, description="Count of high-carbon power violations (>100gCO2/kWh)"
    )
    decarbonization_rate_pct: float = Field(
        default=0.0, description="Actual year-over-year decarbonization rate (%)"
    )
    decarbonization_target_met: bool = Field(
        default=False, description="Whether 7% annual decarbonization target is met"
    )
    holdings_screened: int = Field(default=0, description="Number of holdings screened")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class BenchmarkResult(BaseModel):
    """Comprehensive benchmark alignment result.

    Combines CTB/PAB compliance, trajectory, tracking error,
    and methodology disclosure into a single result.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result identifier")
    benchmark_type: BenchmarkType = Field(description="Type of benchmark assessed")
    overall_status: ComplianceStatus = Field(description="Overall compliance status")
    ctb_result: Optional[CTBComplianceResult] = Field(
        default=None, description="CTB compliance result"
    )
    pab_result: Optional[PABComplianceResult] = Field(
        default=None, description="PAB compliance result"
    )
    trajectory: List[TrajectoryDataPoint] = Field(
        default_factory=list, description="Decarbonization trajectory data points"
    )
    tracking_error: Optional[TrackingErrorResult] = Field(
        default=None, description="Tracking error result"
    )
    methodology: MethodologyDisclosure = Field(
        default_factory=lambda: MethodologyDisclosure(benchmark_type=BenchmarkType.PAB),
        description="Methodology disclosure"
    )
    total_holdings: int = Field(default=0, description="Total holdings assessed")
    exclusion_violation_count: int = Field(
        default=0, description="Total exclusion violations found"
    )
    portfolio_waci: float = Field(
        default=0.0, description="Portfolio WACI (tCO2e/EUR M revenue)"
    )
    generated_at: datetime = Field(default_factory=utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration for the BenchmarkAlignmentEngine.

    Controls benchmark type, thresholds, and trajectory parameters.
    """
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.PAB, description="Target benchmark type"
    )
    benchmark_name: str = Field(
        default="", description="Name of the reference benchmark index"
    )
    benchmark_provider: str = Field(
        default="", description="Benchmark index provider"
    )
    benchmark_universe_intensity: float = Field(
        default=0.0,
        description="Carbon intensity of the investable universe (tCO2e/EUR M revenue)"
    )
    base_year: int = Field(
        default=2019, description="Decarbonization trajectory base year"
    )
    base_year_intensity: float = Field(
        default=0.0, description="Carbon intensity in the base year"
    )
    current_year: int = Field(
        default=2025, description="Current reporting year"
    )
    projection_end_year: int = Field(
        default=2050, description="End year for trajectory projection"
    )
    annual_decarbonization_rate: float = Field(
        default=ANNUAL_DECARBONIZATION_RATE,
        description="Annual decarbonization rate (default 7%)"
    )
    ctb_intensity_reduction_pct: float = Field(
        default=30.0, description="CTB minimum intensity reduction (%)"
    )
    pab_intensity_reduction_pct: float = Field(
        default=50.0, description="PAB minimum intensity reduction (%)"
    )
    scope_coverage: str = Field(
        default="scope_1_2", description="Emission scope coverage"
    )
    include_additional_exclusions: bool = Field(
        default=False, description="Include tobacco and UNGC violation exclusions"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

BenchmarkConfig.model_rebuild()
HoldingBenchmarkData.model_rebuild()
ExclusionViolation.model_rebuild()
TrajectoryDataPoint.model_rebuild()
TrackingErrorResult.model_rebuild()
MethodologyDisclosure.model_rebuild()
CTBComplianceResult.model_rebuild()
PABComplianceResult.model_rebuild()
BenchmarkResult.model_rebuild()

# ---------------------------------------------------------------------------
# BenchmarkAlignmentEngine
# ---------------------------------------------------------------------------

class BenchmarkAlignmentEngine:
    """
    EU Climate Benchmark alignment engine for SFDR Article 9(3) products.

    Verifies portfolio alignment with Climate Transition Benchmarks (CTB)
    or Paris-Aligned Benchmarks (PAB) as required by Article 9(3).
    Implements exclusion screening, intensity reduction checks, decarbonization
    trajectory projection, and tracking error calculation.

    Attributes:
        config: Engine configuration parameters.
        _holdings: Stored holding benchmark data.
        _total_portfolio_value: Calculated total portfolio value.

    Example:
        >>> engine = BenchmarkAlignmentEngine({"benchmark_type": "pab"})
        >>> holdings = [HoldingBenchmarkData(
        ...     company_name="Corp A",
        ...     carbon_intensity=45.0,
        ...     holding_value=10_000_000,
        ...     weight_pct=5.0
        ... )]
        >>> result = engine.assess_alignment(holdings, universe_intensity=120.0)
        >>> print(f"Status: {result.overall_status.value}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BenchmarkAlignmentEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = BenchmarkConfig(**config)
        elif config and isinstance(config, BenchmarkConfig):
            self.config = config
        else:
            self.config = BenchmarkConfig()

        self._holdings: List[HoldingBenchmarkData] = []
        self._total_portfolio_value: float = 0.0

        logger.info(
            "BenchmarkAlignmentEngine initialized (version=%s, type=%s)",
            _MODULE_VERSION,
            self.config.benchmark_type.value,
        )

    # ------------------------------------------------------------------
    # Full Alignment Assessment
    # ------------------------------------------------------------------

    def assess_alignment(
        self,
        holdings: List[HoldingBenchmarkData],
        universe_intensity: Optional[float] = None,
    ) -> BenchmarkResult:
        """Perform comprehensive benchmark alignment assessment.

        Runs exclusion screening, intensity reduction check, and
        decarbonization trajectory analysis based on the configured
        benchmark type (CTB or PAB).

        Args:
            holdings: List of holding benchmark data.
            universe_intensity: Carbon intensity of the investable universe.
                Overrides config if provided.

        Returns:
            BenchmarkResult with complete alignment assessment.
        """
        start = utcnow()
        self._holdings = holdings
        self._total_portfolio_value = sum(h.holding_value for h in holdings)
        self._ensure_weights(holdings)

        u_intensity = universe_intensity or self.config.benchmark_universe_intensity
        btype = self.config.benchmark_type

        ctb_result: Optional[CTBComplianceResult] = None
        pab_result: Optional[PABComplianceResult] = None

        if btype in (BenchmarkType.CTB, BenchmarkType.CUSTOM):
            ctb_result = self.assess_ctb_compliance(holdings, u_intensity)
        if btype in (BenchmarkType.PAB, BenchmarkType.CUSTOM):
            pab_result = self.assess_pab_compliance(holdings, u_intensity)

        trajectory = self.project_trajectory(holdings)
        portfolio_waci = self._calculate_portfolio_waci(holdings)

        # Determine overall status
        if btype == BenchmarkType.CTB and ctb_result:
            overall = ctb_result.status
            violation_count = len(ctb_result.exclusion_violations)
        elif btype == BenchmarkType.PAB and pab_result:
            overall = pab_result.status
            violation_count = len(pab_result.exclusion_violations)
        elif btype == BenchmarkType.CUSTOM:
            pab_ok = pab_result.status == ComplianceStatus.COMPLIANT if pab_result else False
            ctb_ok = ctb_result.status == ComplianceStatus.COMPLIANT if ctb_result else False
            if pab_ok:
                overall = ComplianceStatus.COMPLIANT
            elif ctb_ok:
                overall = ComplianceStatus.PARTIAL
            else:
                overall = ComplianceStatus.NON_COMPLIANT
            violation_count = 0
            if pab_result:
                violation_count += len(pab_result.exclusion_violations)
            if ctb_result:
                violation_count += len(ctb_result.exclusion_violations)
        else:
            overall = ComplianceStatus.INSUFFICIENT_DATA
            violation_count = 0

        methodology = self._build_methodology_disclosure(btype, u_intensity)

        result = BenchmarkResult(
            benchmark_type=btype,
            overall_status=overall,
            ctb_result=ctb_result,
            pab_result=pab_result,
            trajectory=trajectory,
            methodology=methodology,
            total_holdings=len(holdings),
            exclusion_violation_count=violation_count,
            portfolio_waci=round(portfolio_waci, 4),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Benchmark alignment assessed: type=%s, status=%s, violations=%d, "
            "WACI=%.2f in %dms",
            btype.value,
            overall.value,
            violation_count,
            portfolio_waci,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # CTB Compliance
    # ------------------------------------------------------------------

    def assess_ctb_compliance(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
        universe_intensity: Optional[float] = None,
    ) -> CTBComplianceResult:
        """Assess Climate Transition Benchmark (CTB) compliance.

        CTB requirements:
          1. Carbon intensity at least 30% lower than investable universe
          2. Exclusion of controversial weapons manufacturers
          3. 7% annual decarbonization trajectory

        Args:
            holdings: Optional list (uses stored if not provided).
            universe_intensity: Carbon intensity of the investable universe.

        Returns:
            CTBComplianceResult with detailed assessment.
        """
        start = utcnow()
        if holdings is not None:
            self._holdings = holdings
            self._total_portfolio_value = sum(h.holding_value for h in holdings)
            self._ensure_weights(holdings)
        holdings_list = self._holdings
        u_intensity = universe_intensity or self.config.benchmark_universe_intensity

        # Step 1: Calculate portfolio carbon intensity (WACI)
        portfolio_intensity = self._calculate_portfolio_waci(holdings_list)

        # Step 2: Check intensity reduction
        if u_intensity > 0.0:
            reduction_pct = (1.0 - (portfolio_intensity / u_intensity)) * 100.0
        else:
            reduction_pct = 0.0
        required_reduction = self.config.ctb_intensity_reduction_pct
        intensity_met = reduction_pct >= required_reduction

        # Step 3: Screen for exclusions (CTB = controversial weapons only)
        violations = self._screen_exclusions(holdings_list, CTB_EXCLUSION_THRESHOLDS)
        weapons_met = not any(
            v.category == ExclusionCategory.CONTROVERSIAL_WEAPONS for v in violations
        )

        # Step 4: Check decarbonization rate
        decarb_rate = self._calculate_decarbonization_rate(holdings_list)
        decarb_target_pct = self.config.annual_decarbonization_rate * 100.0
        decarb_met = decarb_rate >= decarb_target_pct

        # Determine overall status
        if intensity_met and weapons_met and decarb_met:
            status = ComplianceStatus.COMPLIANT
        elif intensity_met and weapons_met:
            status = ComplianceStatus.PARTIAL
        elif u_intensity <= 0.0:
            status = ComplianceStatus.INSUFFICIENT_DATA
        else:
            status = ComplianceStatus.NON_COMPLIANT

        result = CTBComplianceResult(
            status=status,
            intensity_reduction_pct=round(reduction_pct, 2),
            intensity_reduction_required_pct=required_reduction,
            intensity_reduction_met=intensity_met,
            portfolio_intensity=round(portfolio_intensity, 4),
            benchmark_intensity=round(u_intensity, 4),
            controversial_weapons_exclusion_met=weapons_met,
            exclusion_violations=violations,
            decarbonization_rate_pct=round(decarb_rate, 2),
            decarbonization_target_met=decarb_met,
            holdings_screened=len(holdings_list),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "CTB compliance: status=%s, reduction=%.1f%% (req %.0f%%), "
            "weapons_ok=%s, decarb=%.1f%% in %dms",
            status.value,
            reduction_pct,
            required_reduction,
            weapons_met,
            decarb_rate,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # PAB Compliance
    # ------------------------------------------------------------------

    def assess_pab_compliance(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
        universe_intensity: Optional[float] = None,
    ) -> PABComplianceResult:
        """Assess Paris-Aligned Benchmark (PAB) compliance.

        PAB requirements:
          1. Carbon intensity at least 50% lower than investable universe
          2. Fossil fuel exclusions (exploration >=1%, refining >=10%,
             distribution >=50%, power >100gCO2/kWh)
          3. Controversial weapons exclusion
          4. 7% annual decarbonization trajectory

        Args:
            holdings: Optional list (uses stored if not provided).
            universe_intensity: Carbon intensity of the investable universe.

        Returns:
            PABComplianceResult with detailed assessment.
        """
        start = utcnow()
        if holdings is not None:
            self._holdings = holdings
            self._total_portfolio_value = sum(h.holding_value for h in holdings)
            self._ensure_weights(holdings)
        holdings_list = self._holdings
        u_intensity = universe_intensity or self.config.benchmark_universe_intensity

        # Step 1: Calculate portfolio carbon intensity
        portfolio_intensity = self._calculate_portfolio_waci(holdings_list)

        # Step 2: Check intensity reduction (50% for PAB)
        if u_intensity > 0.0:
            reduction_pct = (1.0 - (portfolio_intensity / u_intensity)) * 100.0
        else:
            reduction_pct = 0.0
        required_reduction = self.config.pab_intensity_reduction_pct
        intensity_met = reduction_pct >= required_reduction

        # Step 3: Screen for all PAB exclusions
        violations = self._screen_exclusions(holdings_list, PAB_EXCLUSION_THRESHOLDS)

        # Categorize violations
        weapons_violations = [
            v for v in violations
            if v.category == ExclusionCategory.CONTROVERSIAL_WEAPONS
        ]
        exploration_violations = [
            v for v in violations
            if v.category in (
                ExclusionCategory.COAL_EXPLORATION,
                ExclusionCategory.OIL_GAS_EXPLORATION,
            )
        ]
        refining_violations = [
            v for v in violations
            if v.category == ExclusionCategory.FOSSIL_REFINING
        ]
        distribution_violations = [
            v for v in violations
            if v.category == ExclusionCategory.FOSSIL_DISTRIBUTION
        ]
        power_violations = [
            v for v in violations
            if v.category == ExclusionCategory.HIGH_CARBON_POWER
        ]

        weapons_met = len(weapons_violations) == 0
        fossil_met = (
            len(exploration_violations) == 0
            and len(refining_violations) == 0
            and len(distribution_violations) == 0
            and len(power_violations) == 0
        )

        # Step 4: Check decarbonization rate
        decarb_rate = self._calculate_decarbonization_rate(holdings_list)
        decarb_target_pct = self.config.annual_decarbonization_rate * 100.0
        decarb_met = decarb_rate >= decarb_target_pct

        # Determine overall status
        all_exclusions_met = weapons_met and fossil_met
        if intensity_met and all_exclusions_met and decarb_met:
            status = ComplianceStatus.COMPLIANT
        elif intensity_met and all_exclusions_met:
            status = ComplianceStatus.PARTIAL
        elif u_intensity <= 0.0:
            status = ComplianceStatus.INSUFFICIENT_DATA
        else:
            status = ComplianceStatus.NON_COMPLIANT

        result = PABComplianceResult(
            status=status,
            intensity_reduction_pct=round(reduction_pct, 2),
            intensity_reduction_required_pct=required_reduction,
            intensity_reduction_met=intensity_met,
            portfolio_intensity=round(portfolio_intensity, 4),
            benchmark_intensity=round(u_intensity, 4),
            fossil_fuel_exclusions_met=fossil_met,
            controversial_weapons_exclusion_met=weapons_met,
            exclusion_violations=violations,
            fossil_exploration_violations=len(exploration_violations),
            fossil_refining_violations=len(refining_violations),
            fossil_distribution_violations=len(distribution_violations),
            high_carbon_power_violations=len(power_violations),
            decarbonization_rate_pct=round(decarb_rate, 2),
            decarbonization_target_met=decarb_met,
            holdings_screened=len(holdings_list),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "PAB compliance: status=%s, reduction=%.1f%% (req %.0f%%), "
            "fossil_ok=%s, weapons_ok=%s, decarb=%.1f%% in %dms",
            status.value,
            reduction_pct,
            required_reduction,
            fossil_met,
            weapons_met,
            decarb_rate,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Decarbonization Trajectory
    # ------------------------------------------------------------------

    def project_trajectory(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
        base_intensity: Optional[float] = None,
        current_intensity: Optional[float] = None,
    ) -> List[TrajectoryDataPoint]:
        """Project the decarbonization trajectory from base year to 2050.

        Generates yearly target intensities using the 7% annual
        decarbonization rate, and compares against actuals where available.

        Formula:
            target(year) = base_intensity * (1 - rate) ^ (year - base_year)

        Args:
            holdings: Optional list (uses stored if not provided).
            base_intensity: Override base year intensity.
            current_intensity: Override current year intensity.

        Returns:
            List of TrajectoryDataPoint from base year to projection end.
        """
        start = utcnow()
        if holdings is not None:
            self._holdings = holdings
            self._ensure_weights(holdings)

        b_intensity = base_intensity or self.config.base_year_intensity
        if b_intensity <= 0.0 and self._holdings:
            b_intensity = self._calculate_portfolio_waci(self._holdings) * 1.5

        c_intensity = current_intensity
        if c_intensity is None and self._holdings:
            c_intensity = self._calculate_portfolio_waci(self._holdings)

        rate = self.config.annual_decarbonization_rate
        base_year = self.config.base_year
        end_year = self.config.projection_end_year
        current_year = self.config.current_year

        trajectory: List[TrajectoryDataPoint] = []

        for year in range(base_year, end_year + 1):
            years_elapsed = year - base_year
            target = b_intensity * ((1.0 - rate) ** years_elapsed)

            actual: Optional[float] = None
            on_track = True
            gap = 0.0

            if year == current_year and c_intensity is not None:
                actual = c_intensity
                on_track = actual <= target
                if target > 0.0:
                    gap = ((actual - target) / target) * 100.0

            point = TrajectoryDataPoint(
                year=year,
                target_intensity=round(target, 4),
                actual_intensity=round(actual, 4) if actual is not None else None,
                on_track=on_track,
                gap_pct=round(gap, 2),
            )
            trajectory.append(point)

        logger.info(
            "Trajectory projected: %d years (%d-%d), base=%.2f, rate=%.0f%% in %dms",
            len(trajectory),
            base_year,
            end_year,
            b_intensity,
            rate * 100,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return trajectory

    # ------------------------------------------------------------------
    # Tracking Error
    # ------------------------------------------------------------------

    def calculate_tracking_error(
        self,
        portfolio_returns: List[float],
        benchmark_returns: List[float],
    ) -> TrackingErrorResult:
        """Calculate tracking error between portfolio and benchmark returns.

        Tracking Error = sqrt(sum((r_p - r_b)^2) / (n - 1))

        Args:
            portfolio_returns: List of periodic portfolio returns.
            benchmark_returns: List of periodic benchmark returns.

        Returns:
            TrackingErrorResult with annualized and monthly tracking error.

        Raises:
            ValueError: If return lists have different lengths or fewer than 2 items.
        """
        start = utcnow()

        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError(
                f"Return lists must have equal length: "
                f"portfolio={len(portfolio_returns)}, benchmark={len(benchmark_returns)}"
            )
        if len(portfolio_returns) < 2:
            raise ValueError("At least 2 return observations are required")

        n = len(portfolio_returns)
        excess_returns = [
            p - b for p, b in zip(portfolio_returns, benchmark_returns)
        ]
        mean_excess = sum(excess_returns) / n

        # Tracking error (standard deviation of excess returns)
        sum_sq_diff = sum((er - mean_excess) ** 2 for er in excess_returns)
        te_periodic = math.sqrt(sum_sq_diff / (n - 1))

        # Annualize assuming monthly data (sqrt(12))
        te_annualized = te_periodic * math.sqrt(12)
        info_ratio = _safe_divide(mean_excess, te_periodic)

        result = TrackingErrorResult(
            tracking_error_annualized=round(te_annualized, 6),
            tracking_error_monthly=round(te_periodic, 6),
            observation_count=n,
            mean_excess_return=round(mean_excess, 6),
            information_ratio=round(info_ratio, 6),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Tracking error: annualized=%.4f, monthly=%.4f, n=%d, IR=%.4f in %dms",
            te_annualized,
            te_periodic,
            n,
            info_ratio,
            int((utcnow() - start).total_seconds() * 1000),
        )
        return result

    # ------------------------------------------------------------------
    # Exclusion Screening
    # ------------------------------------------------------------------

    def screen_exclusions(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
        benchmark_type: Optional[BenchmarkType] = None,
    ) -> List[ExclusionViolation]:
        """Screen holdings against benchmark exclusion rules.

        Args:
            holdings: Optional list (uses stored if not provided).
            benchmark_type: Override benchmark type for threshold selection.

        Returns:
            List of ExclusionViolation for all violating holdings.
        """
        if holdings is not None:
            self._holdings = holdings
        btype = benchmark_type or self.config.benchmark_type

        if btype == BenchmarkType.PAB:
            thresholds = PAB_EXCLUSION_THRESHOLDS
        else:
            thresholds = CTB_EXCLUSION_THRESHOLDS

        violations = self._screen_exclusions(self._holdings, thresholds)

        # Include additional exclusions if configured
        if self.config.include_additional_exclusions:
            additional = self._screen_additional_exclusions(self._holdings)
            violations.extend(additional)

        return violations

    # ------------------------------------------------------------------
    # Methodology Disclosure
    # ------------------------------------------------------------------

    def get_methodology_disclosure(
        self,
        universe_intensity: Optional[float] = None,
    ) -> MethodologyDisclosure:
        """Generate methodology disclosure for regulatory reporting.

        Args:
            universe_intensity: Carbon intensity of the investable universe.

        Returns:
            MethodologyDisclosure with all required disclosure fields.
        """
        u_intensity = universe_intensity or self.config.benchmark_universe_intensity
        return self._build_methodology_disclosure(self.config.benchmark_type, u_intensity)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _calculate_portfolio_waci(
        self, holdings: List[HoldingBenchmarkData]
    ) -> float:
        """Calculate weighted average carbon intensity of the portfolio.

        WACI = SUM(weight_i * carbon_intensity_i)

        Args:
            holdings: List of holdings with carbon intensity data.

        Returns:
            Portfolio WACI in tCO2e/EUR M revenue.
        """
        waci = 0.0
        for h in holdings:
            if h.carbon_intensity > 0.0 and h.weight_pct > 0.0:
                waci += (h.weight_pct / 100.0) * h.carbon_intensity
        return waci

    def _calculate_decarbonization_rate(
        self, holdings: List[HoldingBenchmarkData]
    ) -> float:
        """Calculate year-over-year decarbonization rate.

        Rate = (1 - current_intensity / prior_intensity) * 100

        Args:
            holdings: Holdings with current and prior year intensity data.

        Returns:
            Decarbonization rate as percentage (e.g. 7.0 for 7%).
        """
        current_waci = 0.0
        prior_waci = 0.0
        covered_weight = 0.0

        for h in holdings:
            if h.carbon_intensity > 0.0 and h.prior_year_carbon_intensity > 0.0:
                weight = h.weight_pct / 100.0
                current_waci += weight * h.carbon_intensity
                prior_waci += weight * h.prior_year_carbon_intensity
                covered_weight += weight

        if prior_waci <= 0.0 or covered_weight <= 0.0:
            return 0.0

        rate = (1.0 - (current_waci / prior_waci)) * 100.0
        return max(rate, 0.0)

    def _screen_exclusions(
        self,
        holdings: List[HoldingBenchmarkData],
        thresholds: Dict[str, Dict[str, Any]],
    ) -> List[ExclusionViolation]:
        """Screen holdings against a set of exclusion thresholds.

        Args:
            holdings: Holdings to screen.
            thresholds: Exclusion threshold definitions.

        Returns:
            List of ExclusionViolation objects.
        """
        violations: List[ExclusionViolation] = []

        for h in holdings:
            for cat_name, rule in thresholds.items():
                field_name = rule["field"]
                threshold = rule["threshold"]
                operator = rule["operator"]
                description = rule["description"]

                value = getattr(h, field_name, None)
                if value is None:
                    continue

                is_violation = False
                if operator == "bool_true" and value is True:
                    is_violation = True
                elif operator == "gte" and isinstance(value, (int, float)) and value >= threshold:
                    is_violation = True
                elif operator == "gt" and isinstance(value, (int, float)) and value > threshold:
                    is_violation = True

                if is_violation:
                    violation = ExclusionViolation(
                        holding_id=h.holding_id,
                        company_name=h.company_name,
                        category=ExclusionCategory(cat_name),
                        rule_description=description,
                        actual_value=float(value) if isinstance(value, (int, float)) else 1.0,
                        threshold_value=threshold,
                        weight_pct=h.weight_pct,
                    )
                    violation.provenance_hash = _compute_hash(violation)
                    violations.append(violation)

        return violations

    def _screen_additional_exclusions(
        self,
        holdings: List[HoldingBenchmarkData],
    ) -> List[ExclusionViolation]:
        """Screen for additional optional exclusions (tobacco, UNGC).

        Args:
            holdings: Holdings to screen.

        Returns:
            List of ExclusionViolation objects for additional exclusions.
        """
        violations: List[ExclusionViolation] = []

        for h in holdings:
            # Tobacco exclusion
            if h.tobacco_revenue_pct > 0.0:
                v = ExclusionViolation(
                    holding_id=h.holding_id,
                    company_name=h.company_name,
                    category=ExclusionCategory.TOBACCO,
                    rule_description="Revenue from tobacco products",
                    actual_value=h.tobacco_revenue_pct,
                    threshold_value=0.0,
                    weight_pct=h.weight_pct,
                )
                v.provenance_hash = _compute_hash(v)
                violations.append(v)

            # UNGC violations
            if h.has_ungc_violations:
                v = ExclusionViolation(
                    holding_id=h.holding_id,
                    company_name=h.company_name,
                    category=ExclusionCategory.UNGC_VIOLATIONS,
                    rule_description="UN Global Compact / OECD Guidelines violations",
                    actual_value=1.0,
                    threshold_value=0.0,
                    weight_pct=h.weight_pct,
                )
                v.provenance_hash = _compute_hash(v)
                violations.append(v)

        return violations

    def _ensure_weights(self, holdings: List[HoldingBenchmarkData]) -> None:
        """Ensure portfolio weights are populated from holding values.

        Args:
            holdings: List of holdings to update.
        """
        total_value = sum(h.holding_value for h in holdings)
        if total_value <= 0:
            return

        for h in holdings:
            if h.weight_pct <= 0.0 and h.holding_value > 0:
                h.weight_pct = (h.holding_value / total_value) * 100.0

    def _build_methodology_disclosure(
        self,
        btype: BenchmarkType,
        universe_intensity: float,
    ) -> MethodologyDisclosure:
        """Build methodology disclosure for regulatory reporting.

        Args:
            btype: Benchmark type.
            universe_intensity: Investable universe carbon intensity.

        Returns:
            MethodologyDisclosure with all disclosure fields populated.
        """
        if btype == BenchmarkType.PAB:
            reduction_target = self.config.pab_intensity_reduction_pct
            exclusions = [
                ExclusionCategory.CONTROVERSIAL_WEAPONS.value,
                ExclusionCategory.COAL_EXPLORATION.value,
                ExclusionCategory.OIL_GAS_EXPLORATION.value,
                ExclusionCategory.FOSSIL_REFINING.value,
                ExclusionCategory.FOSSIL_DISTRIBUTION.value,
                ExclusionCategory.HIGH_CARBON_POWER.value,
            ]
        elif btype == BenchmarkType.CTB:
            reduction_target = self.config.ctb_intensity_reduction_pct
            exclusions = [ExclusionCategory.CONTROVERSIAL_WEAPONS.value]
        else:
            reduction_target = self.config.pab_intensity_reduction_pct
            exclusions = list(PAB_EXCLUSION_THRESHOLDS.keys())

        if self.config.include_additional_exclusions:
            exclusions.extend([
                ExclusionCategory.TOBACCO.value,
                ExclusionCategory.UNGC_VIOLATIONS.value,
            ])

        methodology = MethodologyDisclosure(
            benchmark_type=btype,
            benchmark_name=self.config.benchmark_name,
            benchmark_provider=self.config.benchmark_provider,
            intensity_reduction_target_pct=reduction_target,
            decarbonization_rate_pct=self.config.annual_decarbonization_rate * 100.0,
            base_year=self.config.base_year,
            exclusions_applied=exclusions,
            scope_coverage=self.config.scope_coverage,
            methodology_notes=(
                f"Benchmark universe intensity: {universe_intensity:.2f} tCO2e/EUR M. "
                f"Decarbonization trajectory uses {self.config.annual_decarbonization_rate * 100:.0f}% "
                f"annual reduction from {self.config.base_year} base year per "
                f"EU Regulation 2019/2089."
            ),
        )
        methodology.provenance_hash = _compute_hash(methodology)
        return methodology

    # ------------------------------------------------------------------
    # Summary and Reporting
    # ------------------------------------------------------------------

    def get_exclusion_summary(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
        benchmark_type: Optional[BenchmarkType] = None,
    ) -> Dict[str, Any]:
        """Generate a summary of exclusion screening results.

        Args:
            holdings: Optional list of holdings.
            benchmark_type: Override benchmark type.

        Returns:
            Dictionary with exclusion summary by category.
        """
        violations = self.screen_exclusions(holdings, benchmark_type)

        summary: Dict[str, Any] = {
            "total_violations": len(violations),
            "total_weight_pct": round(sum(v.weight_pct for v in violations), 4),
            "by_category": {},
            "violating_holdings": [],
        }

        cat_counts: Dict[str, int] = defaultdict(int)
        cat_weights: Dict[str, float] = defaultdict(float)
        seen_holdings: Dict[str, Dict[str, Any]] = {}

        for v in violations:
            cat_counts[v.category.value] += 1
            cat_weights[v.category.value] += v.weight_pct

            if v.holding_id not in seen_holdings:
                seen_holdings[v.holding_id] = {
                    "holding_id": v.holding_id,
                    "company_name": v.company_name,
                    "weight_pct": round(v.weight_pct, 4),
                    "violations": [],
                }
            seen_holdings[v.holding_id]["violations"].append({
                "category": v.category.value,
                "actual_value": v.actual_value,
                "threshold": v.threshold_value,
            })

        for cat_val, count in cat_counts.items():
            summary["by_category"][cat_val] = {
                "count": count,
                "weight_pct": round(cat_weights[cat_val], 4),
            }

        summary["violating_holdings"] = list(seen_holdings.values())
        return summary

    def get_trajectory_summary(
        self,
        holdings: Optional[List[HoldingBenchmarkData]] = None,
    ) -> Dict[str, Any]:
        """Generate a summary of the decarbonization trajectory.

        Args:
            holdings: Optional list of holdings.

        Returns:
            Dictionary with trajectory summary including key milestones.
        """
        trajectory = self.project_trajectory(holdings)

        milestones = {}
        for point in trajectory:
            if point.year in (2025, 2030, 2035, 2040, 2045, 2050):
                milestones[str(point.year)] = {
                    "target_intensity": point.target_intensity,
                    "actual_intensity": point.actual_intensity,
                    "on_track": point.on_track,
                    "gap_pct": point.gap_pct,
                }

        current_point = None
        for point in trajectory:
            if point.actual_intensity is not None:
                current_point = point
                break

        return {
            "base_year": self.config.base_year,
            "end_year": self.config.projection_end_year,
            "annual_rate_pct": self.config.annual_decarbonization_rate * 100.0,
            "total_data_points": len(trajectory),
            "milestones": milestones,
            "current_status": {
                "year": current_point.year if current_point else None,
                "on_track": current_point.on_track if current_point else None,
                "gap_pct": current_point.gap_pct if current_point else None,
            },
        }
