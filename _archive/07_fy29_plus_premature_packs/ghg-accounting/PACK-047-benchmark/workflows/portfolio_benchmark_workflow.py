# -*- coding: utf-8 -*-
"""
Portfolio Benchmark Workflow
====================================

5-phase workflow for financed emissions portfolio benchmarking within
PACK-047 GHG Emissions Benchmark Pack.

Phases:
    1. HoldingsLoading            -- Load portfolio holdings with ownership
                                     shares, asset classes, and sector
                                     classifications; validate completeness
                                     and ownership percentage totals.
    2. PCAFQuality                -- Score each holding's emissions data
                                     quality using PCAF Global Standard
                                     levels (1-5), compute portfolio-weighted
                                     data quality score, and flag holdings
                                     requiring quality improvement.
    3. WeightedAggregation        -- Aggregate financed emissions by asset
                                     class using attribution factors per
                                     PCAF methodology (equity share for
                                     listed equity, loan-to-value for
                                     mortgages, outstanding amount for
                                     corporate loans).
    4. WACICalculation            -- Compute Weighted Average Carbon
                                     Intensity (WACI), portfolio carbon
                                     footprint (tCO2e/EUR M invested), and
                                     carbon intensity (tCO2e/EUR M revenue)
                                     as required by TCFD, SFDR, and ESRS.
    5. IndexComparison            -- Compare portfolio metrics against
                                     benchmark index (e.g. MSCI World,
                                     S&P 500, Paris-Aligned Benchmark),
                                     compute relative performance, and
                                     identify high-impact holdings.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    PCAF Global Standard (2022) - Financed emissions methodology
    TCFD Recommendations (2017) - Portfolio carbon metrics
    SFDR PAI Indicators (2023) - Carbon footprint and GHG intensity
    ESRS E1 (2024) - Financed emissions disclosures
    EU Paris-Aligned Benchmark Regulation (2020/1818)
    EU Climate Transition Benchmark Regulation (2020/1818)
    NZAOA Target Setting Protocol (2024)
    CDP Financial Services (2026) - Portfolio emissions

Schedule: Quarterly for listed portfolios, annually for illiquid
Estimated duration: 2-3 weeks

Author: GreenLang Team
Version: 47.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class PortfolioPhase(str, Enum):
    """Portfolio benchmark workflow phases."""

    HOLDINGS_LOADING = "holdings_loading"
    PCAF_QUALITY = "pcaf_quality"
    WEIGHTED_AGGREGATION = "weighted_aggregation"
    WACI_CALCULATION = "waci_calculation"
    INDEX_COMPARISON = "index_comparison"

class AssetClass(str, Enum):
    """Financial asset class."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    CORPORATE_LOANS = "corporate_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    SOVEREIGN_BONDS = "sovereign_bonds"
    OTHER = "other"

class PCAFLevel(str, Enum):
    """PCAF data quality level (1=best, 5=worst)."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    LEVEL_5 = "level_5"

class BenchmarkIndex(str, Enum):
    """Benchmark index for portfolio comparison."""

    MSCI_WORLD = "msci_world"
    MSCI_ACWI = "msci_acwi"
    SP_500 = "sp_500"
    STOXX_600 = "stoxx_600"
    PARIS_ALIGNED = "paris_aligned"
    CLIMATE_TRANSITION = "climate_transition"
    CUSTOM = "custom"

# =============================================================================
# PCAF QUALITY SCORE MAPPING (Zero-Hallucination Reference Data)
# =============================================================================

PCAF_QUALITY_SCORES: Dict[str, float] = {
    "level_1": 1.0,
    "level_2": 2.0,
    "level_3": 3.0,
    "level_4": 4.0,
    "level_5": 5.0,
}

ASSET_CLASS_ATTRIBUTION: Dict[str, str] = {
    "listed_equity": "evic",
    "corporate_bonds": "evic",
    "corporate_loans": "outstanding_balance",
    "project_finance": "project_value",
    "commercial_real_estate": "property_value",
    "mortgages": "loan_to_value",
    "sovereign_bonds": "gdp_share",
    "other": "outstanding_balance",
}

# Reference index WACI values (tCO2e / USD M revenue)
INDEX_WACI_REFERENCE: Dict[str, float] = {
    "msci_world": 148.0,
    "msci_acwi": 165.0,
    "sp_500": 132.0,
    "stoxx_600": 120.0,
    "paris_aligned": 85.0,
    "climate_transition": 105.0,
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class PortfolioHolding(BaseModel):
    """A single portfolio holding."""

    holding_id: str = Field(default_factory=lambda: f"h-{_new_uuid()[:8]}")
    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    asset_class: AssetClass = Field(default=AssetClass.LISTED_EQUITY)
    sector: str = Field(default="")
    portfolio_weight_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    invested_amount_usd_m: float = Field(default=0.0, ge=0.0)
    evic_usd_m: float = Field(
        default=0.0, ge=0.0,
        description="Enterprise Value Including Cash",
    )
    revenue_usd_m: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality: PCAFLevel = Field(default=PCAFLevel.LEVEL_3)
    outstanding_balance_usd_m: float = Field(default=0.0, ge=0.0)
    property_value_usd_m: float = Field(default=0.0, ge=0.0)

class PCAFQualityResult(BaseModel):
    """PCAF quality scoring result for a holding."""

    holding_id: str = Field(...)
    entity_name: str = Field(default="")
    pcaf_level: PCAFLevel = Field(...)
    quality_score: float = Field(default=0.0, ge=1.0, le=5.0)
    needs_improvement: bool = Field(default=False)
    improvement_note: str = Field(default="")
    provenance_hash: str = Field(default="")

class AssetClassAggregation(BaseModel):
    """Aggregated financed emissions for an asset class."""

    asset_class: AssetClass = Field(...)
    holdings_count: int = Field(default=0, ge=0)
    total_invested_usd_m: float = Field(default=0.0, ge=0.0)
    financed_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    attribution_method: str = Field(default="")
    avg_pcaf_quality: float = Field(default=0.0, ge=0.0, le=5.0)
    provenance_hash: str = Field(default="")

class PortfolioMetric(BaseModel):
    """A computed portfolio carbon metric."""

    metric_name: str = Field(...)
    value: float = Field(default=0.0)
    unit: str = Field(default="")
    description: str = Field(default="")
    provenance_hash: str = Field(default="")

class IndexComparisonResult(BaseModel):
    """Comparison of portfolio metric against benchmark index."""

    metric_name: str = Field(...)
    portfolio_value: float = Field(default=0.0)
    index_value: float = Field(default=0.0)
    index_name: str = Field(default="")
    absolute_difference: float = Field(default=0.0)
    relative_difference_pct: float = Field(default=0.0)
    outperforms: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class PortfolioBenchmarkInput(BaseModel):
    """Input data model for PortfolioBenchmarkWorkflow."""

    organization_id: str = Field(..., min_length=1)
    portfolio_name: str = Field(default="Main Portfolio")
    reporting_period: str = Field(default="2024")
    holdings: List[PortfolioHolding] = Field(
        ..., min_length=1, description="Portfolio holdings",
    )
    benchmark_indices: List[BenchmarkIndex] = Field(
        default_factory=lambda: [
            BenchmarkIndex.MSCI_WORLD,
            BenchmarkIndex.PARIS_ALIGNED,
        ],
    )
    custom_index_waci: Optional[float] = Field(
        default=None,
        description="Custom benchmark WACI for comparison",
    )
    include_scope3: bool = Field(default=False)
    pcaf_improvement_threshold: PCAFLevel = Field(
        default=PCAFLevel.LEVEL_3,
        description="Holdings below this level flagged for improvement",
    )
    total_portfolio_value_usd_m: float = Field(
        default=0.0, ge=0.0,
        description="Total portfolio AUM in USD millions",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class PortfolioBenchmarkResult(BaseModel):
    """Complete result from portfolio benchmark workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="portfolio_benchmark")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    pcaf_quality_results: List[PCAFQualityResult] = Field(default_factory=list)
    asset_class_aggregations: List[AssetClassAggregation] = Field(default_factory=list)
    portfolio_metrics: List[PortfolioMetric] = Field(default_factory=list)
    index_comparisons: List[IndexComparisonResult] = Field(default_factory=list)
    total_financed_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    portfolio_waci: float = Field(default=0.0, ge=0.0)
    portfolio_carbon_footprint: float = Field(default=0.0, ge=0.0)
    weighted_pcaf_quality: float = Field(default=0.0, ge=0.0, le=5.0)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class PortfolioBenchmarkWorkflow:
    """
    5-phase workflow for financed emissions portfolio benchmarking.

    Loads portfolio holdings, scores PCAF data quality, aggregates financed
    emissions by asset class, computes WACI and carbon footprint, and
    compares against benchmark indices.

    Zero-hallucination: all attribution uses PCAF deterministic formulas
    (ownership share * company emissions); WACI uses standard weighted
    average; no LLM calls in calculation path; SHA-256 provenance on
    every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _holdings: Loaded portfolio holdings.
        _pcaf_results: PCAF quality scoring results.
        _aggregations: Asset class aggregations.
        _metrics: Portfolio carbon metrics.
        _comparisons: Index comparison results.

    Example:
        >>> wf = PortfolioBenchmarkWorkflow()
        >>> holding = PortfolioHolding(entity_id="comp-001", scope1_tco2e=5000)
        >>> inp = PortfolioBenchmarkInput(
        ...     organization_id="bank-001",
        ...     holdings=[holding],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[PortfolioPhase] = [
        PortfolioPhase.HOLDINGS_LOADING,
        PortfolioPhase.PCAF_QUALITY,
        PortfolioPhase.WEIGHTED_AGGREGATION,
        PortfolioPhase.WACI_CALCULATION,
        PortfolioPhase.INDEX_COMPARISON,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PortfolioBenchmarkWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._holdings: List[PortfolioHolding] = []
        self._pcaf_results: List[PCAFQualityResult] = []
        self._aggregations: List[AssetClassAggregation] = []
        self._metrics: List[PortfolioMetric] = []
        self._comparisons: List[IndexComparisonResult] = []
        self._total_financed: float = 0.0
        self._waci: float = 0.0
        self._carbon_footprint: float = 0.0
        self._weighted_quality: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PortfolioBenchmarkResult:
        """
        Execute the 5-phase portfolio benchmark workflow.

        Args:
            input_data: Portfolio holdings and benchmark configuration.

        Returns:
            PortfolioBenchmarkResult with WACI, carbon footprint, and comparisons.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting portfolio benchmark %s org=%s holdings=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.holdings),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_holdings_loading,
            self._phase_2_pcaf_quality,
            self._phase_3_weighted_aggregation,
            self._phase_4_waci_calculation,
            self._phase_5_index_comparison,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Portfolio benchmark failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = PortfolioBenchmarkResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            pcaf_quality_results=self._pcaf_results,
            asset_class_aggregations=self._aggregations,
            portfolio_metrics=self._metrics,
            index_comparisons=self._comparisons,
            total_financed_emissions_tco2e=self._total_financed,
            portfolio_waci=self._waci,
            portfolio_carbon_footprint=self._carbon_footprint,
            weighted_pcaf_quality=self._weighted_quality,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Portfolio benchmark %s completed in %.2fs status=%s waci=%.2f",
            self.workflow_id, elapsed, overall_status.value, self._waci,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Holdings Loading
    # -------------------------------------------------------------------------

    async def _phase_1_holdings_loading(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PhaseResult:
        """Load portfolio holdings and validate completeness."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._holdings = list(input_data.holdings)

        # Validate weight totals
        total_weight = sum(h.portfolio_weight_pct for h in self._holdings)
        if abs(total_weight - 100.0) > 1.0:
            warnings.append(
                f"Portfolio weights sum to {total_weight:.2f}%, not 100%"
            )

        # Validate EVIC availability for equity/bond holdings
        missing_evic = 0
        for h in self._holdings:
            if h.asset_class in (AssetClass.LISTED_EQUITY, AssetClass.CORPORATE_BONDS):
                if h.evic_usd_m <= 0:
                    missing_evic += 1

        if missing_evic > 0:
            warnings.append(f"{missing_evic} equity/bond holdings missing EVIC data")

        # Asset class breakdown
        ac_counts: Dict[str, int] = {}
        for h in self._holdings:
            ac = h.asset_class.value
            ac_counts[ac] = ac_counts.get(ac, 0) + 1

        outputs["holdings_loaded"] = len(self._holdings)
        outputs["total_weight_pct"] = round(total_weight, 2)
        outputs["asset_class_breakdown"] = ac_counts
        outputs["missing_evic"] = missing_evic
        outputs["total_invested_usd_m"] = round(
            sum(h.invested_amount_usd_m for h in self._holdings), 2,
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 HoldingsLoading: %d holdings, weight=%.1f%%",
            len(self._holdings), total_weight,
        )
        return PhaseResult(
            phase_name="holdings_loading", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: PCAF Quality Scoring
    # -------------------------------------------------------------------------

    async def _phase_2_pcaf_quality(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PhaseResult:
        """Score each holding's emissions data quality using PCAF levels."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._pcaf_results = []
        threshold_num = PCAF_QUALITY_SCORES.get(
            input_data.pcaf_improvement_threshold.value, 3.0,
        )

        total_weighted_quality = Decimal("0")
        total_weight = Decimal("0")

        for h in self._holdings:
            quality_num = PCAF_QUALITY_SCORES.get(h.data_quality.value, 5.0)
            needs_improvement = quality_num > threshold_num

            improvement_note = ""
            if needs_improvement:
                if quality_num >= 5.0:
                    improvement_note = "Engage investee for reported emissions data"
                elif quality_num >= 4.0:
                    improvement_note = "Request sector-specific emissions data"
                else:
                    improvement_note = "Improve data source granularity"

            pcaf_data = {
                "holding": h.holding_id, "level": h.data_quality.value,
                "score": quality_num,
            }
            self._pcaf_results.append(PCAFQualityResult(
                holding_id=h.holding_id,
                entity_name=h.entity_name,
                pcaf_level=h.data_quality,
                quality_score=quality_num,
                needs_improvement=needs_improvement,
                improvement_note=improvement_note,
                provenance_hash=_compute_hash(pcaf_data),
            ))

            weight = Decimal(str(h.portfolio_weight_pct))
            total_weighted_quality += Decimal(str(quality_num)) * weight
            total_weight += weight

        if total_weight > 0:
            self._weighted_quality = float(
                (total_weighted_quality / total_weight).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            )
        else:
            self._weighted_quality = 5.0

        needs_count = sum(1 for r in self._pcaf_results if r.needs_improvement)

        outputs["holdings_scored"] = len(self._pcaf_results)
        outputs["weighted_pcaf_quality"] = self._weighted_quality
        outputs["needs_improvement_count"] = needs_count
        outputs["level_distribution"] = {}
        for level in PCAFLevel:
            count = sum(1 for r in self._pcaf_results if r.pcaf_level == level)
            outputs["level_distribution"][level.value] = count

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 PCAFQuality: weighted=%.2f, %d need improvement",
            self._weighted_quality, needs_count,
        )
        return PhaseResult(
            phase_name="pcaf_quality", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Weighted Aggregation
    # -------------------------------------------------------------------------

    async def _phase_3_weighted_aggregation(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PhaseResult:
        """Aggregate financed emissions by asset class using PCAF attribution."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._aggregations = []
        ac_groups: Dict[AssetClass, List[PortfolioHolding]] = {}
        for h in self._holdings:
            ac_groups.setdefault(h.asset_class, []).append(h)

        total_financed = Decimal("0")

        for asset_class, holdings in sorted(ac_groups.items(), key=lambda x: x[0].value):
            ac_financed = Decimal("0")
            total_invested = Decimal("0")
            quality_sum = Decimal("0")

            for h in holdings:
                company_emissions = h.scope1_tco2e + h.scope2_tco2e
                if input_data.include_scope3:
                    company_emissions += h.scope3_tco2e

                # PCAF attribution factor
                attribution = self._compute_attribution(h)
                financed = Decimal(str(company_emissions)) * Decimal(str(attribution))
                ac_financed += financed
                total_invested += Decimal(str(h.invested_amount_usd_m))
                quality_sum += Decimal(str(
                    PCAF_QUALITY_SCORES.get(h.data_quality.value, 5.0)
                ))

            avg_quality = float(
                (quality_sum / Decimal(str(max(len(holdings), 1)))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            )

            agg_data = {
                "asset_class": asset_class.value,
                "financed": float(ac_financed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
                "count": len(holdings),
            }
            self._aggregations.append(AssetClassAggregation(
                asset_class=asset_class,
                holdings_count=len(holdings),
                total_invested_usd_m=float(
                    total_invested.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ),
                financed_emissions_tco2e=float(
                    ac_financed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ),
                attribution_method=ASSET_CLASS_ATTRIBUTION.get(
                    asset_class.value, "outstanding_balance",
                ),
                avg_pcaf_quality=avg_quality,
                provenance_hash=_compute_hash(agg_data),
            ))

            total_financed += ac_financed

        self._total_financed = float(
            total_financed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        outputs["asset_classes_aggregated"] = len(self._aggregations)
        outputs["total_financed_emissions_tco2e"] = self._total_financed
        for agg in self._aggregations:
            outputs[f"financed_{agg.asset_class.value}"] = agg.financed_emissions_tco2e

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 WeightedAggregation: %d asset classes, total=%.2f tCO2e",
            len(self._aggregations), self._total_financed,
        )
        return PhaseResult(
            phase_name="weighted_aggregation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: WACI Calculation
    # -------------------------------------------------------------------------

    async def _phase_4_waci_calculation(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PhaseResult:
        """Compute WACI, carbon footprint, and carbon intensity."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._metrics = []

        # WACI: sum of (portfolio_weight * company_emissions / company_revenue)
        waci_sum = Decimal("0")
        waci_holdings = 0
        for h in self._holdings:
            if h.revenue_usd_m > 0 and h.portfolio_weight_pct > 0:
                company_emissions = h.scope1_tco2e + h.scope2_tco2e
                if input_data.include_scope3:
                    company_emissions += h.scope3_tco2e
                company_intensity = Decimal(str(company_emissions)) / Decimal(str(h.revenue_usd_m))
                weight_frac = Decimal(str(h.portfolio_weight_pct)) / Decimal("100")
                waci_sum += weight_frac * company_intensity
                waci_holdings += 1

        self._waci = float(
            waci_sum.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        # Carbon footprint: total financed emissions / portfolio value
        portfolio_value = input_data.total_portfolio_value_usd_m
        if portfolio_value <= 0:
            portfolio_value = sum(h.invested_amount_usd_m for h in self._holdings)

        if portfolio_value > 0:
            self._carbon_footprint = float(
                (Decimal(str(self._total_financed)) / Decimal(str(portfolio_value))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            )
        else:
            self._carbon_footprint = 0.0
            warnings.append("Portfolio value is zero; carbon footprint undefined")

        # Carbon intensity: total financed emissions / total portfolio revenue
        total_revenue = Decimal(str(sum(h.revenue_usd_m for h in self._holdings)))
        carbon_intensity = 0.0
        if total_revenue > 0:
            carbon_intensity = float(
                (Decimal(str(self._total_financed)) / total_revenue).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            )

        # Record metrics
        metrics_data = [
            ("waci", self._waci, "tCO2e/USD M revenue",
             "Weighted Average Carbon Intensity"),
            ("carbon_footprint", self._carbon_footprint, "tCO2e/USD M invested",
             "Portfolio Carbon Footprint"),
            ("carbon_intensity", carbon_intensity, "tCO2e/USD M portfolio revenue",
             "Portfolio Carbon Intensity"),
            ("total_financed_emissions", self._total_financed, "tCO2e",
             "Total Financed Emissions"),
        ]

        for name, value, unit, desc in metrics_data:
            m_data = {"metric": name, "value": value}
            self._metrics.append(PortfolioMetric(
                metric_name=name,
                value=round(value, 4),
                unit=unit,
                description=desc,
                provenance_hash=_compute_hash(m_data),
            ))

        outputs["waci"] = self._waci
        outputs["carbon_footprint"] = self._carbon_footprint
        outputs["carbon_intensity"] = carbon_intensity
        outputs["total_financed_emissions"] = self._total_financed
        outputs["waci_holdings_count"] = waci_holdings
        outputs["portfolio_value_usd_m"] = portfolio_value

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 WACICalculation: waci=%.2f footprint=%.2f",
            self._waci, self._carbon_footprint,
        )
        return PhaseResult(
            phase_name="waci_calculation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Index Comparison
    # -------------------------------------------------------------------------

    async def _phase_5_index_comparison(
        self, input_data: PortfolioBenchmarkInput,
    ) -> PhaseResult:
        """Compare portfolio metrics against benchmark indices."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._comparisons = []

        for index in input_data.benchmark_indices:
            index_waci = INDEX_WACI_REFERENCE.get(index.value)
            if index == BenchmarkIndex.CUSTOM:
                index_waci = input_data.custom_index_waci

            if index_waci is None or index_waci <= 0:
                warnings.append(f"No WACI reference for index {index.value}")
                continue

            abs_diff = self._waci - index_waci
            rel_diff = (abs_diff / max(index_waci, 1e-12)) * 100.0
            outperforms = self._waci < index_waci

            comp_data = {
                "index": index.value, "portfolio": self._waci,
                "index_waci": index_waci, "diff": abs_diff,
            }
            self._comparisons.append(IndexComparisonResult(
                metric_name="waci",
                portfolio_value=round(self._waci, 4),
                index_value=round(index_waci, 4),
                index_name=index.value,
                absolute_difference=round(abs_diff, 4),
                relative_difference_pct=round(rel_diff, 4),
                outperforms=outperforms,
                provenance_hash=_compute_hash(comp_data),
            ))

        outputs["indices_compared"] = len(self._comparisons)
        outputs["outperforming"] = sum(
            1 for c in self._comparisons if c.outperforms
        )
        for comp in self._comparisons:
            outputs[f"vs_{comp.index_name}"] = {
                "difference": comp.absolute_difference,
                "outperforms": comp.outperforms,
            }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 IndexComparison: %d indices, %d outperforming",
            len(self._comparisons),
            sum(1 for c in self._comparisons if c.outperforms),
        )
        return PhaseResult(
            phase_name="index_comparison", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: PortfolioBenchmarkInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio

                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Calculation Helpers
    # -------------------------------------------------------------------------

    def _compute_attribution(self, h: PortfolioHolding) -> float:
        """Compute PCAF attribution factor for a holding."""
        if h.asset_class in (AssetClass.LISTED_EQUITY, AssetClass.CORPORATE_BONDS):
            if h.evic_usd_m > 0:
                return min(h.invested_amount_usd_m / h.evic_usd_m, 1.0)
            return 0.0
        elif h.asset_class == AssetClass.CORPORATE_LOANS:
            if h.outstanding_balance_usd_m > 0 and h.evic_usd_m > 0:
                return min(h.outstanding_balance_usd_m / h.evic_usd_m, 1.0)
            return 0.0
        elif h.asset_class == AssetClass.COMMERCIAL_REAL_ESTATE:
            if h.property_value_usd_m > 0:
                return min(h.invested_amount_usd_m / h.property_value_usd_m, 1.0)
            return 0.0
        elif h.asset_class == AssetClass.MORTGAGES:
            if h.property_value_usd_m > 0:
                return min(h.outstanding_balance_usd_m / h.property_value_usd_m, 1.0)
            return 0.0
        else:
            if h.evic_usd_m > 0:
                return min(h.invested_amount_usd_m / h.evic_usd_m, 1.0)
            return h.portfolio_weight_pct / 100.0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._holdings = []
        self._pcaf_results = []
        self._aggregations = []
        self._metrics = []
        self._comparisons = []
        self._total_financed = 0.0
        self._waci = 0.0
        self._carbon_footprint = 0.0
        self._weighted_quality = 0.0

    def _compute_provenance(self, result: PortfolioBenchmarkResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.portfolio_waci}|{result.total_financed_emissions_tco2e}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
